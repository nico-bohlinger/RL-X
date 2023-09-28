import os
import logging
import pickle
import random
import time
from collections import deque
import tree
import numpy as np
import jax
from jax.lax import stop_gradient
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import wandb

from rl_x.algorithms.redq.flax.default_config import get_config
from rl_x.algorithms.redq.flax.policy import get_policy
from rl_x.algorithms.redq.flax.critic import get_critic
from rl_x.algorithms.redq.flax.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.redq.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.redq.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class REDQ():
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
        self.buffer_size = config.algorithm.buffer_size
        self.learning_starts = config.algorithm.learning_starts
        self.batch_size = config.algorithm.batch_size
        self.tau = config.algorithm.tau
        self.gamma = config.algorithm.gamma
        self.ensemble_size = config.algorithm.ensemble_size
        self.in_target_minimization_size = config.algorithm.in_target_minimization_size
        self.q_update_steps = config.algorithm.q_update_steps
        self.target_entropy = config.algorithm.target_entropy
        self.logging_freq = config.algorithm.logging_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units

        if config.algorithm.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        rlx_logger.info(f"Using device: {jax.default_backend()}")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, entropy_coefficient_key = jax.random.split(self.key, 4)

        self.env_as_low = env.action_space.low
        self.env_as_high = env.action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.critic = get_critic(config, env)
        
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(env.get_single_action_space_shape()).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.entropy_coefficient = EntropyCoefficient(1.0)

        self.policy.apply = jax.jit(self.policy.apply)
        self.critic.apply = jax.jit(self.critic.apply)
        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)

        def q_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps * self.q_update_steps))
            return self.learning_rate * fraction
    
        def policy_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps * self.policy_update_steps))
            return self.learning_rate * fraction

        def entropy_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps * self.entropy_update_steps))
            return self.learning_rate * fraction
        
        self.q_learning_rate = q_linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.policy_learning_rate = policy_linear_schedule if self.anneal_learning_rate else self.learning_rate
        self.entropy_learning_rate = entropy_linear_schedule if self.anneal_learning_rate else self.learning_rate

        state = jnp.array([self.env.observation_space.sample()])
        action = jnp.array([self.env.action_space.sample()])

        self.policy_state = TrainState.create(
            apply_fn=self.policy.apply,
            params=self.policy.init(policy_key, state),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.policy_learning_rate)
        )

        self.critic_state = RLTrainState.create(
            apply_fn=self.critic.apply,
            params=self.critic.init(critic_key, state, action),
            target_params=self.critic.init(critic_key, state, action),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.q_learning_rate)
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.inject_hyperparams(optax.adam)(learning_rate=self.entropy_learning_rate)
        )

        if self.save_model:
            os.makedirs(self.save_path)
            self.best_mean_return = -np.inf
        
    
    def train(self):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray, key: jax.random.PRNGKey):
            dist = self.policy.apply(policy_state.params, state)
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            return action, key


        @jax.jit
        def update(
                policy_state: TrainState, critic_state: RLTrainState, entropy_coefficient_state: TrainState,
                states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, terminations: np.ndarray, key: jax.random.PRNGKey
            ):
            def critic_loss_fn(critic_params: flax.core.FrozenDict, m_target_params,
                        state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.ndarray, terminated: np.ndarray,
                        key1: jax.random.PRNGKey
                ):
                # Critic loss
                dist = self.policy.apply(policy_state.params, next_state)
                next_action = dist.sample(seed=key1)
                next_log_prob = dist.log_prob(next_action)

                alpha = self.entropy_coefficient.apply(entropy_coefficient_state.params)

                next_q_target = self.critic.apply(m_target_params, next_state, next_action)
                min_next_q_target = jnp.min(next_q_target)
                y = reward + self.gamma * (1 - terminated) * (min_next_q_target - alpha * next_log_prob)

                q = self.critic.apply(critic_params, state, action)

                q_loss = (q - y) ** 2

                # Create metrics
                metrics = {
                    "loss/q_loss": q_loss,
                }

                return q_loss, (metrics)


            def policy_entropy_loss_fn(policy_params: flax.core.FrozenDict, entropy_coefficient_params: flax.core.FrozenDict,
                        state: np.ndarray, key1: jax.random.PRNGKey
                ):
                # Policy loss
                alpha_with_grad = self.entropy_coefficient.apply(entropy_coefficient_params)
                alpha = stop_gradient(alpha_with_grad)

                dist = self.policy.apply(policy_params, state)
                current_action = dist.sample(seed=key1)
                current_log_prob = dist.log_prob(current_action)
                entropy = stop_gradient(-current_log_prob)

                q = self.critic.apply(critic_state.params, state, current_action)
                min_q = jnp.min(q)

                policy_loss = alpha * current_log_prob - min_q

                # Entropy loss
                entropy_loss = alpha_with_grad * (entropy - self.target_entropy)

                # Combine losses
                loss = policy_loss + entropy_loss

                # Create metrics
                metrics = {
                    "loss/policy_loss": policy_loss,
                    "loss/entropy_loss": entropy_loss,
                    "entropy/entropy": entropy,
                    "entropy/alpha": alpha,
                    "q_value/q_value": min_q,
                }

                return loss, (metrics)

            vmap_critic_loss_fn = jax.vmap(critic_loss_fn, in_axes=(None, None, 0, 0, 0, 0, 0, 0), out_axes=0)
            vmap_policy_entropy_loss_fn = jax.vmap(policy_entropy_loss_fn, in_axes=(None, None, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_critic_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_critic_loss_fn(*a, **k))
            mean_vmapped_policy_entropy_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_policy_entropy_loss_fn(*a, **k))
            grad_critic_loss_fn = jax.value_and_grad(mean_vmapped_critic_loss_fn, has_aux=True)
            grad_policy_entropy_loss_fn = jax.value_and_grad(mean_vmapped_policy_entropy_loss_fn, argnums=(0, 1), has_aux=True)
            
            # Update critic
            # Tested jax.lax.scan for the loop: Improves initial compilation time but slows down fps by 10 percent
            metrics_list = []
            for i in range(self.q_update_steps):
                keys = jax.random.split(key, self.batch_size + 2)
                key, sample_key, keys1 = keys[0], keys[1], keys[2:]

                all_indices = jnp.arange(self.ensemble_size)
                m_indices = jax.random.choice(key=sample_key, a=all_indices, shape=(self.in_target_minimization_size,), replace=False)
                m_target_params = jax.tree_map(lambda x: x[m_indices], critic_state.target_params)

                (loss, (metrics)), (critic_gradients) = grad_critic_loss_fn(critic_state.params, m_target_params, states[i], next_states[i], actions[i], rewards[i], terminations[i], keys1)
                
                critic_state = critic_state.apply_gradients(grads=critic_gradients)
                critic_state = critic_state.replace(target_params=optax.incremental_update(critic_state.params, critic_state.target_params, self.tau))
                
                metrics["gradients/critic_grad_norm"] = optax.global_norm(critic_gradients)
                metrics_list.append(metrics)

            critic_metrics = {key: jnp.mean(jnp.array([metrics[key] for metrics in metrics_list])) for key in metrics_list[0].keys()}

            # Update policy and entropy coefficient
            keys = jax.random.split(key, self.batch_size + 1)
            key, keys1 = keys[0], keys[1:]

            (loss, (policy_entropy_metrics)), (policy_gradients, entropy_gradients) = grad_policy_entropy_loss_fn(policy_state.params, entropy_coefficient_state.params, states[0], keys1)

            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)

            # Complete metrics
            metrics = {**critic_metrics, **policy_entropy_metrics}
            metrics["lr/learning_rate"] = policy_state.opt_state.hyperparams["learning_rate"]
            metrics["gradients/policy_grad_norm"] = optax.global_norm(policy_gradients)
            metrics["gradients/entropy_grad_norm"] = optax.global_norm(entropy_gradients)

            return policy_state, critic_state, entropy_coefficient_state, metrics, key


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.observation_space.shape, self.env.action_space.shape)

        saving_return_buffer = deque(maxlen=100 * self.nr_envs)
        episode_info_buffer = deque(maxlen=self.logging_freq)
        step_info_buffer = deque(maxlen=self.logging_freq)
        time_metrics_buffer = deque(maxlen=self.logging_freq)
        optimization_metrics_buffer = deque(maxlen=self.logging_freq)

        state = self.env.reset()

        global_step = 0
        nr_policy_updates = 0
        nr_q_updates = 0
        nr_episodes = 0
        steps_metrics = {}
        while global_step < self.total_timesteps:
            start_time = time.time()
            time_metrics = {}


            # Acting
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                action, self.key = get_action(self.policy_state, state, self.key)
                processed_action = self.get_processed_action(action)
            
            next_state, reward, terminated, truncated, info = self.env.step(jax.device_get(processed_action))
            done = terminated | truncated
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    maybe_final_observation = self.env.get_final_observation(info, i)
                    if maybe_final_observation is not None:
                        actual_next_state[i] = maybe_final_observation
                    nr_episodes += 1
            
            replay_buffer.add(state, actual_next_state, action, reward, terminated)

            state = next_state
            global_step += self.nr_envs

            episode_infos = self.env.get_episode_infos(info)
            step_infos = self.env.get_step_infos(info)
            episode_info_buffer.extend(episode_infos)
            step_info_buffer.extend(step_infos)
            saving_return_buffer.extend([ep_info["r"] for ep_info in episode_infos if "r" in ep_info])

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_try_to_save = should_learning_start and self.save_model and episode_infos
            should_log = global_step % self.logging_freq == 0


            # Optimizing - Prepare batches
            if should_optimize:
                nr_batches_needed = self.q_update_steps
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations = replay_buffer.sample(self.batch_size, nr_batches_needed)

            
            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                self.policy_state, self.critic_state, self.entropy_coefficient_state, optimization_metrics, self.key = update(self.policy_state, self.critic_state, self.entropy_coefficient_state, batch_states, batch_next_states, batch_actions, batch_rewards, batch_terminations, self.key)
                optimization_metrics_buffer.append(optimization_metrics)
                nr_policy_updates += 1
                nr_q_updates += self.q_update_steps
            
            optimize_end_time = time.time()
            time_metrics["time/optimize_time"] = optimize_end_time - acting_end_time


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            time_metrics["time/saving_time"] = saving_end_time - optimize_end_time

            time_metrics["time/fps"] = self.nr_envs / (saving_end_time - start_time)

            time_metrics_buffer.append(time_metrics)


            # Logging
            if should_log:
                self.start_logging(global_step)

                steps_metrics["steps/nr_env_steps"] = global_step
                steps_metrics["steps/nr_policy_updates"] = nr_policy_updates
                steps_metrics["steps/nr_q_updates"] = nr_q_updates
                steps_metrics["steps/nr_episodes"] = nr_episodes

                rollout_info_metrics = {}
                env_info_metrics = {}
                if len(episode_info_buffer) > 0:
                    rollout_info_metrics["rollout/episode_reward"] = np.mean([ep_info["r"] for ep_info in episode_info_buffer if "r" in ep_info])
                    rollout_info_metrics["rollout/episode_length"] = np.mean([ep_info["l"] for ep_info in episode_info_buffer if "l" in ep_info])
                    names = list(episode_info_buffer[0].keys())
                    for name in names:
                        if name != "r" and name != "l" and name != "t":
                            env_info_metrics[f"env_info/{name}"] = np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info])
                if len(step_info_buffer) > 0:
                    names = list(step_info_buffer[0].keys())
                    for name in names:
                        env_info_metrics[f"env_info/{name}"] = np.mean([info[name] for info in step_info_buffer if name in info])
                
                mean_time_metrics = {key: np.mean([metrics[key] for metrics in time_metrics_buffer]) for key in time_metrics_buffer[0].keys()}
                mean_optimization_metrics = {} if not should_learning_start else {key: np.mean([metrics[key] for metrics in optimization_metrics_buffer]) for key in optimization_metrics_buffer[0].keys()}
                combined_metrics = {**rollout_info_metrics, **env_info_metrics, **steps_metrics, **mean_time_metrics, **mean_optimization_metrics}
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
            target={"policy": self.policy_state, "critic": self.critic_state, "entropy_coefficient": self.entropy_coefficient_state},
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
        model = REDQ(config, env, run_path, writer)

        jax_file_name = "_".join(splitted_checkpoint_name[:-1]) + "_"
        step = int(splitted_checkpoint_name[-1])
        restored_train_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target={"policy": model.policy_state, "critic": model.critic_state, "entropy_coefficient": model.entropy_coefficient_state},
            step=step,
            prefix=jax_file_name
        )
        model.policy_state = restored_train_state["policy"]
        model.critic_state = restored_train_state["critic"]
        model.entropy_coefficient_state = restored_train_state["entropy_coefficient"]

        return model
    

    def test(self, episodes):
        @jax.jit
        def get_action(policy_state: TrainState, state: np.ndarray):
            dist = self.policy.apply(policy_state.params, state)
            action = dist.mode()
            return self.get_processed_action(action)
        
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
