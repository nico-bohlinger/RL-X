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
import flax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.pqn.flax.general_properties import GeneralProperties
from rl_x.algorithms.pqn.flax.critic import get_critic
from rl_x.algorithms.pqn.flax.batch import Batch

rlx_logger = logging.getLogger("rl_x")


class PQN:
    def __init__(self, config, train_env, eval_env, run_path, writer):
        self.config = config
        self.train_env = train_env
        self.eval_env = eval_env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(run_path, "models")
        self.track_console = config.runner.track_console
        self.track_tb = config.runner.track_tb
        self.track_wandb = config.runner.track_wandb
        self.seed = config.environment.seed
        self.total_timesteps = config.algorithm.total_timesteps
        self.nr_envs = config.environment.nr_envs
        self.nr_epochs = config.algorithm.nr_epochs
        self.nr_minibatches = config.algorithm.nr_minibatches
        self.learning_rate = config.algorithm.learning_rate
        self.anneal_learning_rate = config.algorithm.anneal_learning_rate
        self.nr_steps = config.algorithm.nr_steps
        self.gamma = config.algorithm.gamma
        self.q_lambda = config.algorithm.q_lambda
        self.epsilon_start = config.algorithm.epsilon_start
        self.epsilon_end = config.algorithm.epsilon_end
        self.max_grad_norm = config.algorithm.max_grad_norm
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes
        self.batch_size = config.environment.nr_envs * config.algorithm.nr_steps
        self.nr_updates = config.algorithm.total_timesteps // self.batch_size
        self.minibatch_size = self.batch_size // self.nr_minibatches
        self.epsilon_decay_steps = self.total_timesteps * config.algorithm.epsilon_decay_fraction

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, critic_key = jax.random.split(self.key)

        self.os_shape = self.train_env.single_observation_space.shape
        self.nr_available_actions = self.train_env.get_single_action_logit_size()

        self.critic = get_critic(config, self.train_env)

        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(count):
            fraction = 1.0 - (count // (self.nr_minibatches * self.nr_epochs)) / self.nr_updates
            return self.learning_rate * fraction
        
        self.critic_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.train_env.single_observation_space.sample()])

        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state),
            tx=optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.inject_hyperparams(optax.radam)(learning_rate=self.critic_learning_rate),
            )
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "best.model"
            self.best_model_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
    
    def train(self):
        @jax.jit
        def get_action(critic_state: TrainState, state: np.ndarray, epsilon: float, key: jax.random.PRNGKey):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            random_action = jax.random.randint(subkey1, (self.nr_envs,), 0, self.nr_available_actions)
            greedy_action = jnp.argmax(self.critic.apply(critic_state.params, state), axis=-1)
            action = jnp.where(
                jax.random.uniform(subkey2, (self.nr_envs,)) < epsilon,
                random_action,
                greedy_action,
            )

            return action, key
        

        @jax.jit
        def calculate_q_targets(critic_state: TrainState, next_states: np.ndarray, rewards: np.ndarray, terminations: np.ndarray):
            next_values = next_values = jax.vmap(self.critic.apply, in_axes=(None, 0))(critic_state.params, next_states).max(axis=-1)
            last_target = rewards[-1] + self.gamma * next_values[-1] * (1.0 - terminations[-1])

            def compute_q_targets(carry_target, inputs):
                reward_t, termination_t, next_q_t = inputs
                mixed_bootstrap = self.q_lambda * carry_target + (1 - self.q_lambda) * next_q_t
                q_target = reward_t + self.gamma * mixed_bootstrap * (1.0 - termination_t)
                return q_target, q_target
            
            _, q_targets = jax.lax.scan(
                compute_q_targets,
                last_target,
                (rewards, terminations, next_values),
                reverse=True
            )
            return q_targets


        @jax.jit
        def update(critic_state: TrainState, states: np.ndarray, actions: np.ndarray, q_targets: np.ndarray, key: jax.random.PRNGKey):
            def loss_fn(critic_params: flax.core.FrozenDict, state: np.ndarray, action: np.ndarray, q_target: np.ndarray):
                q = jnp.squeeze(self.critic.apply(critic_params, jnp.expand_dims(state, 0)))[action]
                q_loss = 0.5 * (q - q_target) ** 2

                # Create metrics
                metrics = {
                    "loss/q_loss": q_loss,
                    "q_value/q_value": q,
                }

                return q_loss, (metrics)
            

            batch_states = states.reshape((-1,) + self.os_shape)
            batch_actions = actions.reshape(-1)
            batch_q_targets = q_targets.reshape(-1)
            
            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0,), has_aux=True)

            key, subkey = jax.random.split(key)
            batch_indices = jnp.tile(jnp.arange(self.batch_size), (self.nr_epochs, 1))
            batch_indices = jax.random.permutation(subkey, batch_indices, axis=1, independent=True)
            batch_indices = batch_indices.reshape((self.nr_epochs * self.nr_minibatches, self.minibatch_size))

            def minibatch_update(carry, minibatch_indices):
                critic_state = carry[0]

                (loss, (metrics)), (critic_gradients,) = grad_loss_fn(
                    critic_state.params,
                    batch_states[minibatch_indices],
                    batch_actions[minibatch_indices],
                    batch_q_targets[minibatch_indices],
                )

                critic_state = critic_state.apply_gradients(grads=critic_gradients)

                metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

                carry = (critic_state,)

                return carry, (metrics)
            
            init_carry = (critic_state,)
            carry, (metrics) = jax.lax.scan(minibatch_update, init_carry, batch_indices)
            critic_state = carry[0]

            # Calculate mean metrics
            mean_metrics = {key: jnp.mean(metrics[key]) for key in metrics}
            mean_metrics["lr/learning_rate"] = critic_state.opt_state[1].hyperparams["learning_rate"]

            return critic_state, mean_metrics, key
        

        @jax.jit
        def get_greedy_action(critic_state: TrainState, state: np.ndarray):
            action = jnp.argmax(self.critic.apply(critic_state.params, state), axis=-1)
            return action


        self.set_train_mode()

        batch = Batch(
            states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            next_states=np.zeros((self.nr_steps, self.nr_envs) + self.os_shape),
            actions=np.zeros((self.nr_steps, self.nr_envs), dtype=np.int32),
            rewards=np.zeros((self.nr_steps, self.nr_envs)),
            terminations=np.zeros((self.nr_steps, self.nr_envs), dtype=bool),
            q_targets=np.zeros((self.nr_steps, self.nr_envs)),
        )

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.train_env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        prev_saving_end_time = None
        logging_time_prev = None
        
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}
            if logging_time_prev:
                time_metrics["time/logging_time_prev"] = logging_time_prev


            # Acting
            dones_this_rollout = 0
            step_info_collection = {}
            for step in range(self.nr_steps):
                epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * min(1.0, global_step / self.epsilon_decay_steps)
                action, self.key = get_action(self.critic_state, state, epsilon, self.key)

                next_state, reward, terminated, truncated, info = self.train_env.step(jax.device_get(action))
                done = terminated | truncated
                actual_next_state = next_state.copy()
                for i, single_done in enumerate(done):
                    if single_done:
                        actual_next_state[i] = self.train_env.get_final_observation_at_index(info, i)
                        saving_return_buffer.append(self.train_env.get_final_info_value_at_index(info, "episode_return", i))
                        dones_this_rollout += 1
                for key, info_value in self.train_env.get_logging_info_dict(info).items():
                    step_info_collection.setdefault(key, []).extend(info_value)

                batch.states[step] = state
                batch.next_states[step] = actual_next_state
                batch.actions[step] = action
                batch.rewards[step] = reward
                batch.terminations[step] = terminated
                state = next_state
                global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # Calculating Q targets
            batch.q_targets = calculate_q_targets(self.critic_state, batch.next_states, batch.rewards, batch.terminations)
            
            calc_q_target_end_time = time.time()
            time_metrics["time/calc_q_target_time"] = calc_q_target_end_time - acting_end_time


            # Optimizing
            self.critic_state, optimization_metrics, self.key = update(self.critic_state, batch.states, batch.actions, batch.q_targets, self.key)
            optimization_metrics = {key: value.item() for key, value in optimization_metrics.items()}
            optimization_metrics["epsilon/epsilon"] = epsilon
            nr_updates += self.nr_epochs * self.nr_minibatches
            
            optimizing_end_time = time.time()
            time_metrics["time/optimizing_time"] = optimizing_end_time - calc_q_target_end_time


            # Evaluating
            evaluation_metrics = {}
            if global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1:
                self.set_eval_mode()
                eval_state, _ = self.eval_env.reset()
                eval_nr_episodes = 0
                evaluation_metrics = {"eval/episode_return": [], "eval/episode_length": []}
                while True:
                    eval_action = get_greedy_action(self.critic_state, eval_state)
                    eval_state, eval_reward, eval_terminated, eval_truncated, eval_info = self.eval_env.step(jax.device_get(eval_action))
                    eval_done = eval_terminated | eval_truncated
                    for i, single_done in enumerate(eval_done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics["eval/episode_return"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_return", i))
                            evaluation_metrics["eval/episode_length"].append(self.eval_env.get_final_info_value_at_index(eval_info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics.items()}
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
            if prev_saving_end_time:
                time_metrics["time/sps"] = int((self.nr_steps * self.nr_envs) / (saving_end_time - prev_saving_end_time))
            prev_saving_end_time = saving_end_time
            time_metrics["time/saving_time"] = saving_end_time - evaluating_end_time


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

            logging_end_time = time.time()
            logging_time_prev = logging_end_time - saving_end_time


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


    def load(config, train_env, eval_env, run_path, writer, explicitly_set_algorithm_params):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = os.path.abspath("/".join(splitted_path[:-1]))
        checkpoint_file_name = splitted_path[-1]
        shutil.unpack_archive(f"{checkpoint_dir}/{checkpoint_file_name}", f"{checkpoint_dir}/tmp", "zip")
        checkpoint_dir = f"{checkpoint_dir}/tmp"

        loaded_algorithm_config = json.load(open(f"{checkpoint_dir}/config_algorithm.json", "r"))
        for key, value in loaded_algorithm_config.items():
            if f"algorithm.{key}" not in explicitly_set_algorithm_params and key in config.algorithm:
                config.algorithm[key] = value
        model = PQN(config, train_env, eval_env, run_path, writer)

        target = {
            "critic": model.critic_state
        }
        restore_args = orbax_utils.restore_args_from_target(target)
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = checkpointer.restore(checkpoint_dir, item=target, restore_args=restore_args)

        model.critic_state = checkpoint["critic"]

        shutil.rmtree(checkpoint_dir)

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(critic_state: TrainState, state: np.ndarray):
            action = jnp.argmax(self.critic.apply(critic_state.params, state), axis=-1)
            return action
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.eval_env.reset()
            while not done:
                action = get_action(self.critic_state, state)
                state, reward, terminated, truncated, info = self.eval_env.step(jax.device_get(action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
