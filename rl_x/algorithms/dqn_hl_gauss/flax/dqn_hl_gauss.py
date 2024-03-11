import os
import logging
import pickle
import time
from collections import deque
import tree
import numpy as np
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import orbax.checkpoint
import optax
import wandb

from rl_x.algorithms.dqn_hl_gauss.flax.general_properties import GeneralProperties
from rl_x.algorithms.dqn_hl_gauss.flax.default_config import get_config
from rl_x.algorithms.dqn_hl_gauss.flax.critic import get_critic
from rl_x.algorithms.dqn_hl_gauss.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.dqn_hl_gauss.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class DQN_HL_Gauss:
    def __init__(self, config, env, run_path, writer):
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
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.gamma = config.algorithm.gamma
        self.nr_bins = config.algorithm.nr_bins
        self.sigma_to_final_sigma_ratio = config.algorithm.sigma_to_final_sigma_ratio
        self.v_min = config.algorithm.v_min
        self.v_max = config.algorithm.v_max
        self.epsilon_start = config.algorithm.epsilon_start
        self.epsilon_end = config.algorithm.epsilon_end
        self.epsilon_decay_steps = config.algorithm.epsilon_decay_steps
        self.update_frequency = config.algorithm.update_frequency
        self.target_update_frequency = config.algorithm.target_update_frequency
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        self.bin_width = (self.v_max - self.v_min) / self.nr_bins
        self.sigma = self.bin_width * self.sigma_to_final_sigma_ratio
        self.support = jnp.linspace(self.v_min, self.v_max, self.nr_bins + 1, dtype=jnp.float32)
        self.centers = (self.support[:-1] + self.support[1:]) / 2

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, critic_key = jax.random.split(self.key)

        self.nr_available_actions = env.get_single_action_logit_size()

        self.critic = get_critic(config, env)

        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction
        
        self.critic_learning_rate = linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.env.single_observation_space.sample()])

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state),
            target_params=self.critic.init(critic_key, state),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.critic_learning_rate)
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
            self.best_model_file_name = "model_best_jax"
            best_model_check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=self.best_model_file_name)
            self.best_model_checkpointer = orbax.checkpoint.Checkpointer(best_model_check_point_handler)
        
    
    def train(self):
        @jax.jit
        def get_action(critic_state: TrainState, state: np.ndarray, epsilon: float, key: jax.random.PRNGKey):
            key, subkey1, subkey2 = jax.random.split(key, 3)

            random_action = jax.random.randint(subkey1, (self.nr_envs,), 0, self.nr_available_actions)

            q_logits = self.critic.apply(critic_state.params, state)
            q_logits_exp = jnp.exp(q_logits - jnp.max(q_logits, axis=-1, keepdims=True))
            q_probs = q_logits_exp / jnp.sum(q_logits_exp, axis=-1, keepdims=True)
            q = jnp.sum(q_probs * self.centers, axis=-1)
            greedy_action = jnp.argmax(q, axis=-1)

            action = jnp.where(
                jax.random.uniform(subkey2) < epsilon,
                random_action,
                greedy_action,
            )

            return action, key


        @jax.jit
        def update(
                critic_state: RLTrainState, current_step: int,
                states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray
            ):
            def loss_fn(critic_params: flax.core.FrozenDict,
                        state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray
                ):
                next_q_target_logits = self.critic.apply(critic_state.target_params, next_state)
                next_q_target_logits_exp = jnp.exp(next_q_target_logits - jnp.max(next_q_target_logits, axis=-1, keepdims=True))
                next_q_target_probs = next_q_target_logits_exp / jnp.sum(next_q_target_logits_exp, axis=-1, keepdims=True)
                next_q_target = jnp.sum(next_q_target_probs * self.centers, axis=-1)

                y = reward + self.gamma * (1 - terminated) * jnp.max(next_q_target)
                y = jnp.clip(y, self.v_min, self.v_max)

                cdf_evals = jax.scipy.special.erf((self.support - y) / (jnp.sqrt(2) * self.sigma))
                z = cdf_evals[-1] - cdf_evals[0]
                target_probs = cdf_evals[1:] - cdf_evals[:-1]
                target_probs /= z

                q_logits = jnp.squeeze(self.critic.apply(critic_params, state))[action]
                q_logits_exp = jnp.exp(q_logits - jnp.max(q_logits))
                q_probs = q_logits_exp / jnp.sum(q_logits_exp)
                q = jnp.sum(q_probs * self.centers)

                q_loss = -jnp.sum(target_probs * jnp.log(q_probs + 1e-8))

                # Create metrics
                metrics = {
                    "loss/q_loss": q_loss,
                    "q_value/q_value": q,
                }

                return q_loss, (metrics)
            

            states = jnp.expand_dims(states, axis=1)
            next_states = jnp.expand_dims(next_states, axis=1)
            
            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0,), has_aux=True)

            (loss, (metrics)), (critic_gradients,) = grad_loss_fn(
                critic_state.params,
                states, next_states, actions, rewards, terminations)

            critic_state = critic_state.apply_gradients(grads=critic_gradients)

            # Update targets
            critic_state = critic_state.replace(target_params=optax.periodic_update(critic_state.params, critic_state.target_params, current_step, self.target_update_frequency))

            metrics["lr/learning_rate"] = critic_state.opt_state.hyperparams["learning_rate"]
            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

            return critic_state, metrics
        

        @jax.jit
        def get_greedy_action(critic_state: TrainState, state: np.ndarray):
            q_logits = self.critic.apply(critic_state.params, state)
            q_logits_exp = jnp.exp(q_logits - jnp.max(q_logits, axis=-1, keepdims=True))
            q_probs = q_logits_exp / jnp.sum(q_logits_exp, axis=-1, keepdims=True)
            q = jnp.sum(q_probs * self.centers, axis=-1)
            action = jnp.argmax(q, axis=-1)

            return action


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.single_observation_space.shape, self.env.single_action_space.shape, self.rng)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.env.reset()
        global_step = 0
        nr_updates = 0
        nr_episodes = 0
        time_metrics_collection = {}
        step_info_collection = {}
        optimization_metrics_collection = {}
        evaluation_metrics_collection = {}
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            dones_this_rollout = 0
            if global_step < self.learning_starts:
                action = np.array([self.env.single_action_space.sample() for _ in range(self.nr_envs)])
            else:
                epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * min(1.0, (global_step - self.learning_starts) / self.epsilon_decay_steps)
                action, self.key = get_action(self.critic_state, state, epsilon, self.key)
                optimization_metrics_collection.setdefault("epsilon/epsilon", []).append(epsilon)
            
            next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    actual_next_state[i] = self.env.get_final_observation_at_index(info, i)
                    saving_return_buffer.append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                    dones_this_rollout += 1
            for key, info_value in self.env.get_logging_info_dict(info).items():
                step_info_collection.setdefault(key, []).extend(info_value)
            
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start and global_step % self.update_frequency == 0
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing - Prepare batches
            if should_optimize:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-function
            if should_optimize:
                self.critic_state, optimization_metrics = update(self.critic_state, global_step, batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations)
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_updates += 1
            
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                while True:
                    action = get_greedy_action(self.critic_state, state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(action))
                    done = terminated | truncated
                    for i, single_done in enumerate(done):
                        if single_done:
                            eval_nr_episodes += 1
                            evaluation_metrics_collection.setdefault("eval/episode_return", []).append(self.env.get_final_info_value_at_index(info, "episode_return", i))
                            evaluation_metrics_collection.setdefault("eval/episode_length", []).append(self.env.get_final_info_value_at_index(info, "episode_length", i))
                            if eval_nr_episodes == self.evaluation_episodes:
                                break
                    if eval_nr_episodes == self.evaluation_episodes:
                        break
                state, _ = self.env.reset()
                self.set_train_mode()
            
            evaluating_end_time = time.time()
            time_metrics_collection.setdefault("time/evaluating_time", []).append(evaluating_end_time - optimizing_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics_collection.setdefault("time/saving_time", []).append(saving_end_time - evaluating_end_time)

            time_metrics_collection.setdefault("time/sps", []).append(self.nr_envs / (saving_end_time - start_time))


            # Logging
            if should_log:
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
                
                time_metrics = {key: np.mean(value) for key, value in time_metrics_collection.items()}
                optimization_metrics = {key: np.mean(value) for key, value in optimization_metrics_collection.items()}
                evaluation_metrics = {key: np.mean(value) for key, value in evaluation_metrics_collection.items()}
                combined_metrics = {**rollout_info_metrics, **evaluation_metrics, **env_info_metrics, **steps_metrics, **time_metrics, **optimization_metrics}
                for key, value in combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                time_metrics_collection = {}
                step_info_collection = {}
                optimization_metrics_collection = {}
                evaluation_metrics_collection = {}

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
            "config_algorithm": self.config.algorithm.to_dict(),
            "critic": self.critic_state         
        }
        save_args = orbax_utils.save_args_from_target(checkpoint)
        self.best_model_checkpointer.save(f"{self.save_path}/tmp", checkpoint, save_args=save_args)
        os.rename(f"{self.save_path}/tmp/{self.best_model_file_name}", f"{self.save_path}/{self.best_model_file_name}")
        os.remove(f"{self.save_path}/tmp/_METADATA")
        os.rmdir(f"{self.save_path}/tmp")

        if self.track_wandb:
            wandb.save(f"{self.save_path}/{self.best_model_file_name}", base_path=self.save_path)


    def load(config, env, run_path, writer):
        splitted_path = config.runner.load_model.split("/")
        checkpoint_dir = "/".join(splitted_path[:-1])
        checkpoint_file_name = splitted_path[-1]

        check_point_handler = orbax.checkpoint.PyTreeCheckpointHandler(aggregate_filename=checkpoint_file_name)
        checkpointer = orbax.checkpoint.Checkpointer(check_point_handler)

        loaded_algorithm_config = checkpointer.restore(checkpoint_dir)["config_algorithm"]
        default_algorithm_config = get_config(config.algorithm.name)
        for key, value in loaded_algorithm_config.items():
            if config.algorithm[key] == default_algorithm_config[key]:
                config.algorithm[key] = value
        model = DQN_HL_Gauss(config, env, run_path, writer)

        target = {
            "config_algorithm": config.algorithm.to_dict(),
            "critic": model.critic_state
        }
        checkpoint = checkpointer.restore(checkpoint_dir, item=target)

        model.critic_state = checkpoint["critic"]

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(critic_state: TrainState, state: np.ndarray):
            q_logits = self.critic.apply(critic_state.params, state)
            q_logits_exp = jnp.exp(q_logits - jnp.max(q_logits, axis=-1, keepdims=True))
            q_probs = q_logits_exp / jnp.sum(q_logits_exp, axis=-1, keepdims=True)
            q = jnp.sum(q_probs * self.centers, axis=-1)
            action = jnp.argmax(q, axis=-1)
            
            return action
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state, _ = self.env.reset()
            while not done:
                action = get_action(self.critic_state, state)
                state, reward, terminated, truncated, info = self.env.step(jax.device_get(action))
                done = terminated | truncated
                episode_return += reward
            rlx_logger.info(f"Episode {i + 1} - Return: {episode_return}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...


    def general_properties():
        return GeneralProperties
