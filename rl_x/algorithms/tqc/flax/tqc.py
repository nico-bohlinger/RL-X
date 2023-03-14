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

from rl_x.algorithms.tqc.flax.policy import get_policy
from rl_x.algorithms.tqc.flax.critic import get_critic
from rl_x.algorithms.tqc.flax.entropy_coefficient import EntropyCoefficient
from rl_x.algorithms.tqc.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.tqc.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class TQC:
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
        self.ensemble_size = config.algorithm.ensemble_size
        self.nr_atoms_per_net = config.algorithm.nr_atoms_per_net
        self.nr_dropped_atoms_per_net = config.algorithm.nr_dropped_atoms_per_net
        self.huber_kappa = config.algorithm.huber_kappa
        self.target_entropy = config.algorithm.target_entropy
        self.logging_freq = config.algorithm.logging_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.nr_total_atoms = self.nr_atoms_per_net * self.ensemble_size
        self.nr_target_atoms = self.nr_total_atoms - (self.nr_dropped_atoms_per_net * self.ensemble_size)

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
        self.vector_critic = get_critic(config, env)
        
        if self.target_entropy == "auto":
            self.target_entropy = -np.prod(env.get_single_action_space_shape()).item()
        else:
            self.target_entropy = float(self.target_entropy)
        self.entropy_coefficient = EntropyCoefficient(1.0)

        self.policy.apply = jax.jit(self.policy.apply)
        self.vector_critic.apply = jax.jit(self.vector_critic.apply)
        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)

        def q_linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / (total_steps * self.q_update_steps))
            return self.learning_rate * fraction
    
        def policy_linear_schedule(step):
            total_steps = self.total_timesteps
            fraction = 1.0 - (step / (total_steps * self.policy_update_steps))
            return self.learning_rate * fraction

        def entropy_linear_schedule(step):
            total_steps = self.total_timesteps
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

        self.vector_critic_state = RLTrainState.create(
            apply_fn=self.vector_critic.apply,
            params=self.vector_critic.init(critic_key, state, action),
            target_params=self.vector_critic.init(critic_key, state, action),
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
                policy_state: TrainState, vector_critic_state: RLTrainState, entropy_coefficient_state: TrainState,
                states: np.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, key: jax.random.PRNGKey
            ):
            def loss_fn(policy_params: flax.core.FrozenDict, vector_critic_params: flax.core.FrozenDict, vector_critic_target_params: flax.core.FrozenDict, entropy_coefficient_params: flax.core.FrozenDict,
                        state: np.ndarray, next_state: np.ndarray, action: np.ndarray, reward: np.ndarray, done: np.ndarray,
                        key1: jax.random.PRNGKey, key2: jax.random.PRNGKey
                ):
                # Update critic
                dist = stop_gradient(self.policy.apply(policy_params, next_state))
                next_action = dist.sample(seed=key1)
                next_log_prob = dist.log_prob(next_action)

                alpha_with_grad = self.entropy_coefficient.apply(entropy_coefficient_params)
                alpha = stop_gradient(alpha_with_grad)

                next_q_target_atoms = stop_gradient(self.vector_critic.apply(vector_critic_target_params, next_state, next_action))
                next_q_target_atoms = jnp.sort(next_q_target_atoms.reshape(self.nr_total_atoms))[:self.nr_target_atoms]

                y = reward + self.gamma * (1 - done) * (next_q_target_atoms - alpha * next_log_prob)
                y = jnp.expand_dims(y, axis=0)  # (1, nr_target_atoms)

                q_atoms = self.vector_critic.apply(vector_critic_params, state, action)
                q_atoms = jnp.expand_dims(q_atoms.reshape(self.nr_total_atoms), axis=1)  # (nr_total_atoms, 1)
                
                cumulative_prob = (jnp.arange(self.nr_total_atoms, dtype=jnp.float32) + 0.5) / self.nr_total_atoms
                cumulative_prob = jnp.expand_dims(cumulative_prob, axis=(0, -1))  # (1, nr_total_atoms, 1)

                delta_i_j = y - q_atoms
                abs_delta_i_j = jnp.abs(delta_i_j)
                huber_loss = jnp.where(abs_delta_i_j <= self.huber_kappa, 0.5 * delta_i_j ** 2, self.huber_kappa * (abs_delta_i_j - 0.5 * self.huber_kappa))
                q_loss = jnp.mean(jnp.abs(cumulative_prob - (delta_i_j < 0).astype(jnp.float32)) * huber_loss / self.huber_kappa)


                # Update policy
                dist = self.policy.apply(policy_params, state)
                current_action = dist.sample(seed=key2)
                current_log_prob = dist.log_prob(current_action)
                entropy = stop_gradient(-current_log_prob)

                q_atoms = self.vector_critic.apply(stop_gradient(vector_critic_params), state, current_action)
                mean_q_atoms = jnp.mean(q_atoms)

                policy_loss = alpha * current_log_prob - mean_q_atoms


                # Update entropy coefficient
                entropy_loss = alpha_with_grad * (entropy - self.target_entropy)


                # Combine losses
                loss = q_loss + policy_loss + entropy_loss


                # Create metrics
                metrics = {
                    "train/q_loss": q_loss,
                    "train/policy_loss": policy_loss,
                    "train/entropy_loss": entropy_loss,
                    "train/entropy": entropy,
                    "train/alpha": alpha
                }

                return loss, (metrics)
            

            keys = jax.random.split(key, (self.batch_size * 2) + 1)
            key, keys1, keys2 = keys[0], keys[1::2], keys[2::2]

            vmap_loss_fn = jax.vmap(loss_fn, in_axes=(None, None, None, None, 0, 0, 0, 0, 0, 0, 0), out_axes=0)
            safe_mean = lambda x: jnp.mean(x) if x is not None else x
            mean_vmapped_loss_fn = lambda *a, **k: tree.map_structure(safe_mean, vmap_loss_fn(*a, **k))

            (loss, (metrics)), (policy_gradients, vector_critic_gradients, entropy_gradients) = jax.value_and_grad(mean_vmapped_loss_fn, argnums=(0, 1, 3), has_aux=True)(
                policy_state.params, vector_critic_state.params, vector_critic_state.target_params, entropy_coefficient_state.params,
                states, next_states, actions, rewards, dones, keys1, keys2)

            policy_state = policy_state.apply_gradients(grads=policy_gradients)
            vector_critic_state = vector_critic_state.apply_gradients(grads=vector_critic_gradients)
            entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=entropy_gradients)

            # Update targets
            vector_critic_state = vector_critic_state.replace(target_params=optax.incremental_update(vector_critic_state.params, vector_critic_state.target_params, self.tau))

            return policy_state, vector_critic_state, entropy_coefficient_state, metrics, key


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.observation_space.shape, self.env.action_space.shape)

        saving_return_buffer = deque(maxlen=100)
        episode_info_buffer = deque(maxlen=self.logging_freq)
        time_metrics_buffer = deque(maxlen=self.logging_freq)
        loss_metrics_buffer = deque(maxlen=self.logging_freq)

        state = self.env.reset()

        global_step = 0
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
            
            next_state, reward, done, info = self.env.step(jax.device_get(processed_action))
            actual_next_state = next_state.copy()
            for i, single_done in enumerate(done):
                if single_done:
                    maybe_terminal_observation = self.env.get_terminal_observation(info, i)
                    if maybe_terminal_observation is not None:
                        actual_next_state[i] = maybe_terminal_observation
            
            replay_buffer.add(state, actual_next_state, action, reward, done)

            state = next_state
            global_step += self.nr_envs

            episode_infos = self.env.get_episode_infos(info)
            episode_info_buffer.extend(episode_infos)
            saving_return_buffer.extend([ep_info["r"] for ep_info in episode_infos])

            acting_end_time = time.time()
            time_metrics["time/acting_time"] = acting_end_time - start_time


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_try_to_save = should_learning_start and self.save_model and episode_infos
            should_log = global_step % self.logging_freq == 0


            # Optimizing - Prepare batches
            if should_optimize:
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(self.batch_size)


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                self.policy_state, self.vector_critic_state, self.entropy_coefficient_state, loss_metrics, self.key = update(self.policy_state, self.vector_critic_state, self.entropy_coefficient_state, batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, self.key)
                loss_metrics_buffer.append(loss_metrics)
            
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

                if len(episode_info_buffer) > 0:
                    self.log("rollout/ep_rew_mean", np.mean([ep_info["r"] for ep_info in episode_info_buffer]), global_step)
                    self.log("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in episode_info_buffer]), global_step)
                    names = list(episode_info_buffer[0].keys())
                    for name in names:
                        if name != "r" and name != "l" and name != "t":
                            self.log(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info.keys()]), global_step)
                mean_time_metrics = {key: np.mean([metrics[key] for metrics in time_metrics_buffer]) for key in time_metrics_buffer[0].keys()}
                mean_lr_metrics = {"train/learning_rate": self.policy_state.opt_state.hyperparams["learning_rate"].item()}
                mean_loss_metrics = {} if not should_learning_start else {key: np.mean([metrics[key] for metrics in loss_metrics_buffer]) for key in loss_metrics_buffer[0].keys()}
                mean_combined_metrics = {**mean_time_metrics, **mean_lr_metrics, **mean_loss_metrics}
                for key, value in mean_combined_metrics.items():
                    self.log(f"{key}", value, global_step)

                episode_info_buffer.clear()
                time_metrics_buffer.clear()
                loss_metrics_buffer.clear()

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
            self.log_console("global_step", step)
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
            target={"policy": self.policy_state, "vector_critic": self.vector_critic_state, "entropy_coefficient": self.entropy_coefficient_state},
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
        model = TQC(config, env, run_path, writer)

        jax_file_name = "_".join(splitted_checkpoint_name[:-1]) + "_"
        step = int(splitted_checkpoint_name[-1])
        restored_train_state = checkpoints.restore_checkpoint(
            ckpt_dir=checkpoint_dir,
            target={"policy": model.policy_state, "vector_critic": model.vector_critic_state, "entropy_coefficient": model.entropy_coefficient_state},
            step=step,
            prefix=jax_file_name
        )
        model.policy_state = restored_train_state["policy"]
        model.vector_critic_state = restored_train_state["vector_critic"]
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
                state, reward, done, info = self.env.step(jax.device_get(processed_action))
            return_val = self.env.get_episode_infos(info)[0]["r"]
            rlx_logger.info(f"Episode {i + 1} - Return: {return_val}")
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...
