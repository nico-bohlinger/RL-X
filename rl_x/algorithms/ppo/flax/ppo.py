import os
import random
import logging
import pickle
import time
from collections import deque
import tree
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import wandb

from rl_x.algorithms.ppo.flax.policy import get_policy
from rl_x.algorithms.ppo.flax.critic import get_critic
from rl_x.algorithms.ppo.flax.storage import Storage

rlx_logger = logging.getLogger("rl_x")


class PPO:
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
        self.nr_epochs = config.algorithm.nr_epochs
        self.minibatch_size = config.algorithm.minibatch_size
        self.gamma = config.algorithm.gamma
        self.gae_lambda = config.algorithm.gae_lambda
        self.clip_range = config.algorithm.clip_range
        self.entropy_coef = config.algorithm.entropy_coef
        self.critic_coef = config.algorithm.critic_coef
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.std_dev = config.algorithm.std_dev
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.nr_minibatches = self.batch_size // self.minibatch_size

        if config.algorithm.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.os_shape = env.observation_space.shape
        self.as_shape = env.action_space.shape
        
        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction

        learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([env.observation_space.sample()])

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

    
    def train(self):
        @jax.jit
        def get_action_and_value(policy_state: TrainState, critic_state: TrainState, state: np.ndarray, storage: Storage, step: int, key: jax.random.PRNGKey):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            action_std = jnp.exp(action_logstd)
            key, subkey = jax.random.split(key)
            action = action_mean + action_std * jax.random.normal(subkey, shape=action_mean.shape)
            log_prob = -0.5 * ((action - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
            value = self.critic.apply(critic_state.params, state)
            storage = storage.replace(
                states=storage.states.at[step].set(state),
                actions=storage.actions.at[step].set(action),
                log_probs=storage.log_probs.at[step].set(log_prob.sum(1)),
                values=storage.values.at[step].set(value.reshape(-1))
            )
            processed_action = self.get_processed_action(action)
            return processed_action, storage, key
        

        @jax.jit
        def calculate_gae_advantages(critic_state: TrainState, next_states: jnp.array, rewards: np.ndarray, terminations: np.ndarray, storage: Storage):
            def compute_advantages(carry, t):
                prev_advantage, delta, terminations = carry
                advantage = delta[t] + self.gamma * self.gae_lambda * (1 - terminations[t]) * prev_advantage
                return (advantage, delta, terminations), advantage

            next_values = self.critic.apply(critic_state.params, next_states).squeeze(-1)
            delta = rewards + self.gamma * next_values * (1.0 - terminations) - storage.values
            init_advantages = delta[-1]
            _, advantages = jax.lax.scan(compute_advantages, (init_advantages, delta, terminations), jnp.arange(self.nr_steps - 2, -1, -1))
            advantages = jnp.concatenate([advantages[::-1], jnp.array([init_advantages])])
            storage = storage.replace(
                advantages=advantages,
                returns=advantages + storage.values
            )
            return storage
        

        @jax.jit
        def update(policy_state: TrainState, critic_state: TrainState, storage: Storage, key: jax.random.PRNGKey):
            def loss_fn(policy_params, critic_params, state_b, action_b, log_prob_b, return_b, advantage_b):
                # Policy loss
                action_mean, action_logstd = self.policy.apply(policy_params, state_b)
                action_std = jnp.exp(action_logstd)
                new_log_prob = -0.5 * ((action_b - action_mean) / action_std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - action_logstd
                new_log_prob = new_log_prob.sum(1)
                entropy = action_logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)
                
                logratio = new_log_prob - log_prob_b
                ratio = jnp.exp(logratio)
                approx_kl_div = (ratio - 1) - logratio
                clip_fraction = jnp.float32((jnp.abs(ratio - 1) > self.clip_range))

                pg_loss1 = -advantage_b * ratio
                pg_loss2 = -advantage_b * jnp.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)
                pg_loss = jnp.maximum(pg_loss1, pg_loss2)
                
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
                    "policy_ratio/approx_kl": approx_kl_div,
                    "policy_ratio/clip_fraction": clip_fraction,
                }

                return loss, (metrics)
            

            batch_states = storage.states.reshape((-1,) + self.os_shape)
            batch_actions = storage.actions.reshape((-1,) + self.as_shape)
            batch_advantages = storage.advantages.reshape(-1)
            batch_returns = storage.returns.reshape(-1)
            batch_log_probs = storage.log_probs.reshape(-1)

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1), has_aux=True)

            key, subkey = jax.random.split(key)
            batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
            batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
            batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

            def minibatch_update(carry, minibatch_indices):
                policy_state, critic_state, batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages = carry

                minibatch_advantages = batch_advantages[minibatch_indices]
                minibatch_advantages = (minibatch_advantages - jnp.mean(minibatch_advantages)) / (jnp.std(minibatch_advantages) + 1e-8)

                (loss, (metrics)), (policy_gradients, critic_gradients) = grad_loss_fn(
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

                carry = (policy_state, critic_state, batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages)

                return carry, (metrics)
            
            init_carry = (policy_state, critic_state, batch_states, batch_actions, batch_log_probs, batch_returns, batch_advantages)
            carry, (metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
            policy_state, critic_state, _, _, _, _, _ = carry

            # Calculate mean metrics
            mean_metrics = {key: jnp.mean(metrics[key]) for key in metrics}
            mean_metrics["lr/learning_rate"] = policy_state.opt_state[1].hyperparams["learning_rate"]
            mean_metrics["v_value/explained_variance"] = 1 - jnp.var(storage.returns - storage.values) / (jnp.var(storage.returns) + 1e-8)
            mean_metrics["policy/std_dev"] = jnp.mean(jnp.exp(policy_state.params["params"]["policy_logstd"]))

            return policy_state, critic_state, mean_metrics, key
        

        self.set_train_mode()

        storage = Storage(
            states=jnp.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=jnp.zeros((self.nr_steps, self.nr_envs) + self.as_shape),
            log_probs=jnp.zeros((self.nr_steps, self.nr_envs)),
            values=jnp.zeros((self.nr_steps, self.nr_envs)),
            advantages=jnp.zeros((self.nr_steps, self.nr_envs)),
            returns=jnp.zeros((self.nr_steps, self.nr_envs)),
        )
        rewards = np.zeros((self.nr_steps, self.nr_envs))
        terminations = np.zeros((self.nr_steps, self.nr_envs))
        next_states = np.zeros((self.nr_steps, self.nr_envs) + self.os_shape)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        episode_info_buffer = deque(maxlen=100 * self.nr_envs)
        time_metrics_buffer = deque(maxlen=1)
        optimization_metrics_buffer = deque(maxlen=self.nr_epochs * self.nr_minibatches)

        state = self.env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}


            # Acting
            for step in range(self.nr_steps):
                processed_action, storage, self.key = get_action_and_value(self.policy_state, self.critic_state, state, storage, step, self.key)
                next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        maybe_final_observation = self.env.get_final_observation(info, i)
                        if maybe_final_observation is not None:
                            actual_next_state[i] = maybe_final_observation
                        nr_episodes += 1

                next_states[step] = actual_next_state
                rewards[step] = reward
                terminations[step] = terminated
                state = next_state
                global_step += self.nr_envs

                episode_infos = self.env.get_episode_infos(info)
                episode_info_buffer.extend(episode_infos)
            saving_return_buffer.extend([ep_info["r"] for ep_info in episode_info_buffer if "r" in ep_info])
            
            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating advantages and returns
            storage = calculate_gae_advantages(self.critic_state, next_states, rewards, terminations, storage)
            
            calc_adv_return_end_time = time.time()
            time_metrics["time/calc_adv_and_return_time"] = calc_adv_return_end_time - acting_end_time


            # Optimizing
            self.policy_state, self.critic_state, metrics, self.key = update(self.policy_state, self.critic_state, storage, self.key)
            optimization_metrics_buffer.append(metrics)
            nr_updates += self.nr_epochs * self.nr_minibatches

            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_adv_return_end_time
            

            # Saving
            # Only save when the total return buffer (over multiple updates) isn't empty
            # Also only save when the episode info buffer isn't empty -> there were finished episodes this update
            if self.save_model and saving_return_buffer and episode_info_buffer:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics["time/saving_time"] = saving_end_time - optimizing_end_time

            time_metrics["time/fps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - start_time))

            time_metrics_buffer.append(time_metrics)


            # Logging
            self.start_logging(global_step)

            steps_metrics["steps/nr_env_steps"] = global_step
            steps_metrics["steps/nr_updates"] = nr_updates
            steps_metrics["steps/nr_episodes"] = nr_episodes

            if len(episode_info_buffer) > 0:
                self.log("rollout/episode_reward", np.mean([ep_info["r"] for ep_info in episode_info_buffer if "r" in ep_info]), global_step)
                self.log("rollout/episode_length", np.mean([ep_info["l"] for ep_info in episode_info_buffer if "l" in ep_info]), global_step)
                names = list(episode_info_buffer[0].keys())
                for name in names:
                    if name != "r" and name != "l" and name != "t":
                        self.log(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info]), global_step)
            mean_time_metrics = {key: np.mean([metrics[key] for metrics in time_metrics_buffer]) for key in time_metrics_buffer[0].keys()}
            mean_optimization_metrics = {key: np.mean([metrics[key] for metrics in optimization_metrics_buffer]) for key in optimization_metrics_buffer[0].keys()}
            combined_metrics = {**steps_metrics, **mean_time_metrics, **mean_optimization_metrics}
            for key, value in combined_metrics.items():
                self.log(f"{key}", value, global_step)

            episode_info_buffer.clear()
            time_metrics_buffer.clear()
            optimization_metrics_buffer.clear()

            self.end_logging()


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │")


    def start_logging(self, step):
        if self.track_console:
            rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐")
        else:
            rlx_logger.info(f"Step: {step}")


    def end_logging(self):
        if self.track_console:
            rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

    
    def save(self):
        jax_file_path = self.save_path + "/model_best_jax_0"
        config_file_path = self.save_path + "/model_best_config_0"

        checkpoints.save_checkpoint(
            ckpt_dir=self.save_path,
            target={"policy": self.policy_state, "critic": self.critic_state},
            step=0,
            prefix="model_best_jax_",
            overwrite=True
        )

        with open(config_file_path, "wb") as file:
            pickle.dump({"config_algorithm": self.config.algorithm}, file, pickle.HIGHEST_PROTOCOL)

        if self.track_wandb:
            wandb.save(jax_file_path, base_path=os.path.dirname(jax_file_path))
            wandb.save(config_file_path, base_path=os.path.dirname(config_file_path))
    

    def load(config, env, run_path, writer):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_name = splitted_path[-1]
        splitted_checkpoint_name = checkpoint_name.split("_")

        config_file_name = "_".join(splitted_checkpoint_name[:-2]) + "_config_" + splitted_checkpoint_name[-1]
        with open(f"{checkpoint_dir}/{config_file_name}", "rb") as file:
            config.algorithm = pickle.load(file)["config_algorithm"]
        model = PPO(config, env, run_path, writer)

        jax_file_name = "_".join(splitted_checkpoint_name[:-1]) + "_"
        step = int(splitted_checkpoint_name[-1])
        restored_train_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target={"policy": model.policy_state, "critic": model.critic_state},
            step=step,
            prefix=jax_file_name
        )
        model.policy_state = restored_train_state["policy"]
        model.critic_state = restored_train_state["critic"]

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray):
            action_mean, action_logstd = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(action_mean)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            state = self.env.reset()
            while not done:
                processed_action = get_action(self.policy_state, state)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
                done = terminated | truncated
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")
    
            
    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...
