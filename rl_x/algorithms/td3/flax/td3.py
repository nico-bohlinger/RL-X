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
from flax.training import checkpoints
import optax
import wandb

from rl_x.algorithms.td3.flax.general_properties import GeneralProperties
from rl_x.algorithms.td3.flax.default_config import get_config
from rl_x.algorithms.td3.flax.policy import get_policy
from rl_x.algorithms.td3.flax.critic import get_critic
from rl_x.algorithms.td3.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.td3.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class TD3:
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
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.epsilon = config.algorithm.epsilon
        self.smoothing_epsilon = config.algorithm.smoothing_epsilon
        self.smoothing_clip_value = config.algorithm.smoothing_clip_value
        self.policy_delay = config.algorithm.policy_delay
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.logging_frequency = config.algorithm.logging_frequency
        self.evaluation_frequency = config.algorithm.evaluation_frequency
        self.evaluation_episodes = config.algorithm.evaluation_episodes

        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        self.rng = np.random.default_rng(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key = jax.random.split(self.key, 3)

        self.env_as_low = env.single_action_space.low
        self.env_as_high = env.single_action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)

        def linear_schedule_critic(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / total_steps)
            return self.learning_rate * fraction

        def linear_schedule_policy(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - ((step * self.policy_delay) / total_steps)
            return self.learning_rate * fraction
        
        self.q_learning_rate = linear_schedule_critic if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = linear_schedule_policy if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.env.single_observation_space.sample()])
        action = jnp.array([self.env.single_action_space.sample()])

        self.policy_state = RLTrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            target_params=self.policy.init(policy_key, state),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate)
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, action),
            target_params=self.critic.init(critic_key, state, action),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.q_learning_rate)
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
        
    
    def train(self):
        @jax.jit
        def get_action(policy_state: RLTrainState, state: np.ndarray, key: jax.random.PRNGKey):
            key, subkey = jax.random.split(key)
            mean_action = self.policy.apply(policy_state.params, state)
            action = mean_action + self.epsilon * jax.random.normal(subkey, mean_action.shape)
            action = jnp.clip(action, -1.0, 1.0)
            return action, key


        @jax.jit
        def update_critic(
                policy_state: RLTrainState, critic_state: RLTrainState,
                states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, key: jax.random.PRNGKey
            ):
            def loss_fn(critic_params: flax.core.FrozenDict,
                        critic_target_params: flax.core.FrozenDict, policy_target_params: flax.core.FrozenDict,
                        state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray,
                        key1: jax.random.PRNGKey
                ):
                # Critic loss
                next_action = stop_gradient(self.policy.apply(policy_target_params, next_state))
                smoothing_noise = jax.random.normal(key1, next_action.shape) * self.smoothing_epsilon
                smoothing_noise = jnp.clip(smoothing_noise, -self.smoothing_clip_value, self.smoothing_clip_value)
                next_action = jnp.clip(next_action + smoothing_noise, -1.0, 1.0)

                next_q_target = stop_gradient(self.critic.apply(critic_target_params, next_state, next_action))
                min_next_q_target = jnp.min(next_q_target)
                y = reward + self.gamma * (1 - terminated) * min_next_q_target
                q = self.critic.apply(critic_params, state, action)
                q_loss = (q - y) ** 2

                # Create metrics
                metrics = {
                    "loss/q_loss": q_loss,
                }

                return q_loss, (metrics)
            

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, 0, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0,), has_aux=True)

            keys = jax.random.split(key, self.batch_size + 1)
            key, keys1 = keys[0], keys[1:]

            (loss, (metrics)), (critic_gradients,) = grad_loss_fn(
                critic_state.params,
                critic_state.target_params, policy_state.target_params,
                states, next_states, actions, rewards, terminations, keys1)

            critic_state = critic_state.apply_gradients(grads=critic_gradients)

            metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)

            return policy_state, critic_state, metrics, key
        

        @jax.jit
        def update_policy_and_targets(
                policy_state: RLTrainState, critic_state: RLTrainState, states: np.ndarray
            ):
            def loss_fn(policy_params: flax.core.FrozenDict, critic_params: flax.core.FrozenDict, state: np.ndarray):
                # Policy loss
                current_action = self.policy.apply(policy_params, state)
                q = self.critic.apply(stop_gradient(critic_params), state, current_action)
                min_q = jnp.min(q)
                policy_loss = -min_q

                # Create metrics
                metrics = {
                    "loss/policy_loss": policy_loss,
                    "q_value/q_value": min_q,
                }

                return policy_loss, (metrics)
            

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))
            grad_loss_fn = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0,), has_aux=True)

            (loss, (metrics)), (policy_gradients,) = grad_loss_fn(policy_state.params, critic_state.params, states)

            policy_state = policy_state.apply_gradients(grads=policy_gradients)

            # Update targets
            critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))
            policy_state = policy_state.replace(target_params=optax.incremental_update(policy_state.params, policy_state.target_params, self.tau))

            metrics["lr/learning_rate"] = policy_state.opt_state.hyperparams["learning_rate"]
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)

            return policy_state, critic_state, metrics
        

        @jax.jit
        def get_deterministic_action(policy_state: RLTrainState, state: np.ndarray):
            mean_action = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(mean_action)


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.single_observation_space.shape, self.env.single_action_space.shape, self.rng)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)

        state, _ = self.env.reset()
        global_step = 0
        nr_critic_updates = 0
        nr_policy_updates = 0
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
                processed_action = np.array([self.env.single_action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                action, self.key = get_action(self.policy_state, state, self.key)
                processed_action = self.get_processed_action(action)
            
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
            
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs
            nr_episodes += dones_this_rollout

            acting_end_time = time.time()
            time_metrics_collection.setdefault("time/acting_time", []).append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize_critic = should_learning_start
            should_optimize_policy = should_learning_start and global_step % self.policy_delay == 0
            should_evaluate = global_step % self.evaluation_frequency == 0 and self.evaluation_frequency != -1
            should_try_to_save = should_learning_start and self.save_model and dones_this_rollout > 0
            should_log = global_step % self.logging_frequency == 0


            # Optimizing - Prepare batches
            if should_optimize_critic:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-functions
            if should_optimize_critic:
                self.policy_state, self.critic_state, optimization_metrics, self.key = update_critic(self.policy_state, self.critic_state, batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations, self.key)
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_critic_updates += 1


            # Optimizing - Policy and target networks
            if should_optimize_policy:
                self.policy_state, self.critic_state, optimization_metrics = update_policy_and_targets(self.policy_state, self.critic_state, batch_states)
                for key, value in optimization_metrics.items():
                    optimization_metrics_collection.setdefault(key, []).append(value)
                nr_policy_updates += 1
            
            optimizing_end_time = time.time()
            time_metrics_collection.setdefault("time/optimizing_time", []).append(optimizing_end_time - acting_end_time)


            # Evaluating
            if should_evaluate:
                self.set_eval_mode()
                state, _ = self.env.reset()
                eval_nr_episodes = 0
                while True:
                    processed_action = get_deterministic_action(self.policy_state, state)
                    state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
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
                steps_metrics["steps/nr_critic_updates"] = nr_critic_updates
                steps_metrics["steps/nr_policy_updates"] = nr_policy_updates
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
            loaded_algorithm_config = pickle.load(file)["config_algorithm"]
        default_algorithm_config = get_config(config.algorithm.name)
        for key, value in loaded_algorithm_config.items():
            if config.algorithm[key] == default_algorithm_config[key]:
                config.algorithm[key] = value
        model = TD3(config, env, run_path, writer)

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
        def get_action(policy_state: RLTrainState, state: np.ndarray):
            mean_action = self.policy.apply(policy_state.params, state)
            return self.get_processed_action(mean_action)
        
        self.set_eval_mode()
        for i in range(episodes):
            done = False
            episode_return = 0
            state = self.env.reset()
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
