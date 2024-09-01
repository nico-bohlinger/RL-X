import os
import shutil
import json
import logging
import time
from collections import deque
import tree
import numpy as np
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.espo.flax.general_properties import GeneralProperties
from rl_x.algorithms.espo.flax.policy import get_policy
from rl_x.algorithms.espo.flax.critic import get_critic
from rl_x.algorithms.espo.flax.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class ESPO:
    def __init__(self, config, env, run_path, writer) -> None:
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.max_epochs = config.algorithm.max_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.max_ratio_delta = config.algorithm.max_ratio_delta
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size

        if self.evaluation_frequency % (self.nr_steps * self.nr_envs) != 0 and self.evaluation_frequency != -1:
            raise ValueError("Evaluation frequency must be a multiple of the number of steps and environments.")

        if config.algorithm.delta_calc_operator == "mean":
            self.delta_calc_operator = jnp.mean
        elif config.algorithm.delta_calc_operator == "median":
            self.delta_calc_operator = jnp.median
        else:
            raise ValueError("Unknown delta_calc_operator")

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.os_shape = env.single_observation_space.shape
        self.as_shape = env.single_action_space.shape
        
        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            # This gets jit compiled so we don't know the current number of epochs already run
            fraction = 1.0 - (count // self.nr_epochs) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([env.single_observation_space.sample()])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(learning_rate=learning_rate),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    
    def train(self):
        @jax.jit
        def get_action_and_value(policy_state: TrainState, critic_state: TrainState, state: np.ndarray , key: jax.random.PRNGKey):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            action_std = jnp.exp(action_logstd)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            value = self.critic.apply(critic_state.params, state)
            processed_action = self.get_processed_action(action)
            return processed_action, action, value.reshape(-1), log_prob.sum(1), key
        

        @jax.jit
        def calculate_gae_advantages(critic_state: TrainState, next_states: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, values: np.ndarray):
            def compute_advantages(carry, t):
                prev_advantage = carry[0]
                advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                return (advantage,), advantage

            next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
            delta = rewards + self.gamma * next_values * (1.0 - terminations) - values
            init_advantages = delta[-1]
            _, advantages = jax.lax.scan(compute_advantages, (init_advantages,), jnp.arange(self.nr_steps - 2, -1, -1))
            advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
            returns = advantages + values
            return advantages, returns
        

        def update(policy_state: TrainState, critic_state: TrainState,
                   states: np.ndarray, actions: np.ndarray, advantages: np.ndarray, returns: np.ndarray, values: np.ndarray, log_probs: np.ndarray,
                   key: jax.random.PRNGKey):
            for epoch in range(self.max_epochs):
                policy_state, critic_state, ratio_delta, metrics, key = update_epoch(policy_state, critic_state, states, actions, advantages, returns, log_probs, key)

                if ratio_delta > self.max_ratio_delta:
                    break
            
            metrics = {key: jnp.mean(metrics[key]) for key in metrics}
            metrics["optim/nr_epochs"] = epoch + 1
            metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            metrics["v_value/explained_variance"] = 1 - jnp.var(returns - values) / (jnp.var(returns) + 1e-8)
            metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))
            
            return policy_state, critic_state, epoch, metrics, key


        @jax.jit
        def update_epoch(policy_state: TrainState, critic_state: TrainState,
                         states: np.ndarray, actions: np.ndarray, advantages: np.ndarray, returns: np.ndarray, log_probs: np.ndarray,
                         key: jax.random.PRNGKey):
            def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                # Policy loss
                action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                action_std = jnp.exp(action_logstd)
                new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                new_log_prob = new_log_prob.sum(1)
                entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                
                logratio = new_log_prob - log_prob_b
                ratio = jnp.exp(logratio)
                ratio_delta = self.delta_calc_operator(jnp.abs(ratio - 1))
                approx_kl_div = (ratio - 1) - logratio

                pg_loss = -advantage_b * ratio
                
                entropy_loss = entropy.sum(1)

                # Critic loss
                new_value = self.critic.apply(critic_params, state_b)
                critic_loss = 0.5 * (new_value - return_b) ** 2

                # Combine losses
                loss = pg_loss - self.entropy_coef * entropy_loss + self.critic_coef * critic_loss

                # Create metrics
                metrics = {
                    "loss/policy_gradient_loss": pg_loss,
                    "loss/critic_loss": critic_loss,
                    "loss/entropy_loss": entropy_loss,
                    "policy_ratio/ratio_delta": ratio_delta,
                    "policy_ratio/approx_kl": approx_kl_div,
                }

                return loss, (metrics, ratio_delta)
            

            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape((-1,) + self.as_shape)
            batch_advantages = advantages.reshape(-1)
            batch_returns = returns.reshape(-1)
            batch_log_probs = log_probs.reshape(-1)

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

            key, subkey = jax.random.split(key)
            minibatch_indices = jax.random.choice(subkey, self.batch_size, shape=(self.minibatch_size,), replace=False)

            minibatch_advantages = batch_advantages[minibatch_indices]
            minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

            (loss, (metrics, ratio_delta)), (policy_gradients, critic_gradients) = grad_loss_fn(
                policy_state.params,
                critic_state.params,
                batch_states[minibatch_indices],
                batch_actions[minibatch_indices],
                batch_log_probs[minibatch_indices],
                batch_returns[minibatch_indices],
                minibatch_advantages
            )

            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            critic_state = critic_state.apply_gradients(grads=critic_gradients)

            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

            return policy_state, critic_state, ratio_delta, metrics, key
        

        @jax.jit
        def get_deterministic_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)
        

        self.set_train_mode()

        batch = Batch(
            states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            next_states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=np.zeros((self.nr_steps, self.nr_envs) + self.as_shape),
            rewards=np.zeros((self.nr_steps, self.nr_envs)),
            values=np.zeros((self.nr_steps, self.nr_envs)),
            terminations=np.zeros((self.nr_steps, self.nr_envs)),
            log_probs=np.zeros((self.nr_steps, self.nr_envs)),
            advantages=np.zeros((self.nr_steps, self.nr_envs)),
            returns=np.zeros((self.nr_steps, self.nr_envs)),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}


            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                processed_action, action, value, log_prob, self.key = get_action_and_value(self.policy_state, self.critic_state, state, self.key)
                next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = self.env.get_final_observation_at_index(info, i)
                        saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                for key, info_value in self.env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                batch.states[step] = state
                batch.next_states[step] = actual_next_state
                batch.actions[step] = action
                batch.rewards[step] = reward
                batch.values[step] = value
                batch.terminations[step] = terminated
                batch.log_probs[step] = log_prob
                state = next_state
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating advantages and returns
            batch.advantages, batch.returns = calculate_gae_advantages(self.critic_state, batch.next_states, batch.rewards, batch.terminations, batch.values)
            
            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            self.policy_state, self.critic_state, epoch, optimization_metrics, self.key = update(
                self.policy_state, self.critic_state,
                batch.states, batch.actions, batch.advantages, batch.returns, batch.values, batch.log_probs,
                self.key
            )
            optimization_metrics = {key: value.item() for key, value in optimization_metrics.items()}
            nr_updates += epoch + 1

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time
            
            
            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                self.set_eval_mode()
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    processed_action = get_deterministic_action(self.policy_state, state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics["eval/episode_return"].append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                            evaluation_metrics["eval/episode_length"].append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
                state, _ = self.env.reset()
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics["time/evaluating_time"] = evaluating_end_time - optimizing_end_time


            # Saving
            # Also only save when there were finished episodes this update
            if self.save_model and dones_this_rollout > 0:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time

            time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time))


            # Logging
            self.start_logging(global_step)

            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes

            rollout_info_metrics = {}
            env_info_metrics = {}
            if step_info_collection:
                info_names = list(step_info_collection.keys())
                for info_name in info_names:
                    metric_group = "rollout" if info_name in ["episode_return", "episode_length"] else "env_info"
                    metric_dict = rollout_info_metrics if metric_group == "rollout" else env_info_metrics
                    mean_value = np.mean(step_info_collection[info_name])
                    if mean_value == mean_value:  # Check if mean_value is NaN
                        metric_dict[f"{metric_group}/{info_name}"] = mean_value
            
            combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

            self.end_logging()


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │", flush=False)


    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐", flush=False)
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")


    def save(self):
        checkpoint = {
            "policy": self.policy_state,
            "critic": self.critic_state
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        with open(f"{self.save_path}/tmp/config_algorithm.json", "w") as f:
            json.dump(self.config.algorithm.to_dict(), f)
        shutil.make_archive(f"{self.save_path}/{self.best_model_file_name}", "zip", f"{self.save_path}/tmp")
        os.rename(f"{self.save_path}/{self.best_model_file_name}.zip", f"{self.save_path}/{self.best_model_file_name}")
        shutil.rmtree(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)


    def load(config, env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params:
                config.algorithm[key] = value
        model = ESPO(config, env, run_path, writer)

        target = {
            "policy": model.policy_state,
            "critic": model.critic_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.policy_state = checkpoint["policy"]
        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while not done:
                processed_action = get_action(self.policy_state, state)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")

            
    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
