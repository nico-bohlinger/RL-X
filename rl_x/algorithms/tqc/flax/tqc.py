import os
import logging
import pickle
import random
import time
from collections import deque
import numpy as np
import jax
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


class TQC():
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
        self.nr_envs = config.algorithm.nr_envs
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
        self.q_update_steps = config.algorithm.q_update_steps
        self.policy_update_steps = config.algorithm.policy_update_steps
        self.entropy_update_steps = config.algorithm.entropy_update_steps
        self.target_entropy = config.algorithm.target_entropy
        self.log_freq = config.algorithm.log_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units
        self.nr_total_atoms = self.nr_atoms_per_net * self.ensemble_size
        self.nr_target_atoms = self.nr_total_atoms - (self.nr_dropped_atoms_per_net * self.ensemble_size)
        self.max_nr_batches_needed = max(self.q_update_steps, self.policy_update_steps, self.entropy_update_steps)

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
            tx=optax.adam(learning_rate=self.policy_learning_rate)
        )

        self.vector_critic_state = RLTrainState.create(
            apply_fn=self.vector_critic.apply,
            params=self.vector_critic.init(critic_key, state, action),
            target_params=self.vector_critic.init(critic_key, state, action),
            tx=optax.adam(learning_rate=self.q_learning_rate)
        )

        self.entropy_coefficient_state = TrainState.create(
            apply_fn=self.entropy_coefficient.apply,
            params=self.entropy_coefficient.init(entropy_coefficient_key),
            tx=optax.adam(learning_rate=self.entropy_learning_rate)
        )

        self.policy.apply = jax.jit(self.policy.apply)
        self.vector_critic.apply = jax.jit(self.vector_critic.apply)
        self.entropy_coefficient.apply = jax.jit(self.entropy_coefficient.apply)

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
                states: jnp.ndarray, next_states: jnp.ndarray, actions: jnp.ndarray, rewards: jnp.ndarray, dones: jnp.ndarray, key: jax.random.PRNGKey
            ):
            # Update critic
            q_losses = []

            for i in range(self.q_update_steps):
                key, action_sample_key = jax.random.split(key)

                dist = self.policy.apply(policy_state.params, next_states[i])
                next_actions = dist.sample(seed=action_sample_key)
                next_log_probs = dist.log_prob(next_actions)

                alpha = self.entropy_coefficient.apply(entropy_coefficient_state.params)

                next_q_target_atoms = self.vector_critic.apply(vector_critic_state.target_params, next_states[i], next_actions)
                next_q_target_atoms = jnp.transpose(next_q_target_atoms, (1, 0, 2)).reshape(self.batch_size, self.nr_total_atoms)  # (batch_size, nr_total_atoms)
                next_q_target_atoms = jnp.sort(next_q_target_atoms)
                next_q_target_atoms = next_q_target_atoms[:, :self.nr_target_atoms]

                y = rewards[i].reshape(-1, 1) + self.gamma * (1 - dones[i].reshape(-1, 1)) * (next_q_target_atoms - alpha * next_log_probs.reshape(-1, 1))
                y = jnp.expand_dims(y, axis=1)  # (batch_size, 1, nr_target_atoms)

                def huber_loss_fn(vector_critic_params: flax.core.FrozenDict):
                    q_atoms = self.vector_critic.apply(vector_critic_params, states[i], actions[i])
                    q_atoms = jnp.transpose(q_atoms, (1, 0, 2)).reshape(self.batch_size, -1)
                    q_atoms = jnp.expand_dims(q_atoms, axis=2)  # (batch_size, nr_total_atoms, 1)
                    
                    cumulative_prob = (jnp.arange(self.nr_total_atoms, dtype=jnp.float32) + 0.5) / self.nr_total_atoms
                    cumulative_prob = jnp.expand_dims(cumulative_prob, axis=(0, -1))  # (1, nr_total_atoms, 1)

                    delta_i_j = y - q_atoms
                    abs_delta_i_j = jnp.abs(delta_i_j)
                    huber_loss = jnp.where(abs_delta_i_j <= self.huber_kappa, 0.5 * delta_i_j ** 2, self.huber_kappa * (abs_delta_i_j - 0.5 * self.huber_kappa))
                    loss = jnp.abs(cumulative_prob - (delta_i_j < 0).astype(jnp.float32)) * huber_loss / self.huber_kappa

                    return loss.mean()
                
                q_loss, grads = jax.value_and_grad(huber_loss_fn, has_aux=False)(vector_critic_state.params)
                vector_critic_state = vector_critic_state.apply_gradients(grads=grads)
                q_losses.append(q_loss)

                # Update targets
                vector_critic_state = vector_critic_state.replace(target_params=optax.incremental_update(vector_critic_state.params, vector_critic_state.target_params, self.tau))


            # Update policy
            policy_losses = []
            entropies = []
            alphas = []

            for i in range(self.policy_update_steps):
                key, subkey = jax.random.split(key)

                def loss_fn(policy_params: flax.core.FrozenDict):
                    dist = self.policy.apply(policy_params, states[i])
                    current_actions = dist.sample(seed=subkey)
                    current_log_probs = dist.log_prob(current_actions).reshape(-1, 1)

                    q_atoms = self.vector_critic.apply(vector_critic_state.params, states[i], current_actions)
                    q_atoms = jnp.transpose(q_atoms, (1, 0, 2)).reshape(self.batch_size, self.nr_total_atoms)
                    mean_q_atoms = jnp.mean(q_atoms, axis=1)

                    alpha = self.entropy_coefficient.apply(entropy_coefficient_state.params)

                    policy_loss = (alpha * current_log_probs - mean_q_atoms).mean()
                    return policy_loss, (-current_log_probs.mean(), alpha)
                
                (policy_loss, (entropy, alpha)), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_state.params)
                policy_state = policy_state.apply_gradients(grads=grads)
                policy_losses.append(policy_loss)
                entropies.append(entropy)
                alphas.append(alpha)


            # Update entropy coefficient
            entropy_losses = []

            for i in range(self.entropy_update_steps):
                def loss_fn(entropy_coefficient_params: flax.core.FrozenDict):
                    alpha = self.entropy_coefficient.apply(entropy_coefficient_params)
                    return alpha * (entropies[i] - self.target_entropy)
            
                entropy_loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(entropy_coefficient_state.params)
                entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=grads)
                entropy_losses.append(entropy_loss)


            return jnp.array(q_losses), jnp.array(policy_losses), jnp.array(entropies), jnp.array(alphas), jnp.array(entropy_losses), policy_state, vector_critic_state, entropy_coefficient_state, key


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.observation_space.shape, self.env.action_space.shape)

        saving_return_buffer = deque(maxlen=100)
        episode_info_buffer = deque(maxlen=self.log_freq)
        acting_time_buffer = deque(maxlen=self.log_freq)
        optimize_time_buffer = deque(maxlen=self.log_freq)
        saving_time_buffer = deque(maxlen=self.log_freq)
        fps_buffer = deque(maxlen=self.log_freq)
        q_loss_buffer = deque(maxlen=self.log_freq)
        policy_loss_buffer = deque(maxlen=self.log_freq)
        entropy_loss_buffer = deque(maxlen=self.log_freq)
        entropy_buffer = deque(maxlen=self.log_freq)
        alpha_buffer = deque(maxlen=self.log_freq)

        state = self.env.reset()

        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()


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
            acting_time_buffer.append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_optimize = should_learning_start
            should_try_to_save = self.learning_starts and self.save_model and episode_infos
            should_log = global_step % self.log_freq == 0


            # Optimizing - Prepare batches
            if should_optimize:
                max_nr_batches_needed = max(self.q_update_steps, self.policy_update_steps, self.entropy_update_steps)
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(self.batch_size, max_nr_batches_needed)


            # Optimizing - Q-functions, policy and entropy coefficient
            if should_optimize:
                q_loss, policy_loss, entropies, alphas, entropy_loss, self.policy_state, self.vector_critic_state, self.entropy_coefficient_state, self.key = update(self.policy_state, self.vector_critic_state, self.entropy_coefficient_state, batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, self.key)
                q_loss_buffer.extend(q_loss)
                policy_loss_buffer.extend(policy_loss)
                entropy_loss_buffer.extend(entropy_loss)
                entropy_buffer.extend(entropies)
                alpha_buffer.extend(alphas)
            
            optimize_end_time = time.time()
            optimize_time_buffer.append(optimize_end_time - acting_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            saving_time_buffer.append(saving_end_time - optimize_end_time)

            fps_buffer.append(self.nr_envs / (saving_end_time - start_time))


            # Logging                
            if should_log:
                if self.track_console:
                    rlx_logger.info("┌" + "─" * 31 + "┬" + "─" * 16 + "┐")
                    self.log_console("global_step", global_step)
                else:
                    rlx_logger.info(f"Step: {global_step}")

                if len(episode_info_buffer) > 0:
                    self.log("rollout/ep_rew_mean", np.mean([ep_info["r"] for ep_info in episode_info_buffer]), global_step)
                    self.log("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in episode_info_buffer]), global_step)
                    names = list(episode_info_buffer[0].keys())
                    for name in names:
                        if name != "r" and name != "l" and name != "t":
                            self.log(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info.keys()]), global_step)
                self.log("time/fps", np.mean(fps_buffer), global_step)
                self.log("time/acting_time", np.mean(acting_time_buffer), global_step)
                self.log("time/optimize_time", np.mean(optimize_time_buffer), global_step)
                self.log("time/saving_time", np.mean(saving_time_buffer), global_step)
                self.log("train/learning_rate", self.policy_learning_rate if isinstance(self.policy_learning_rate, float) else self.policy_learning_rate(global_step), global_step)
                self.log("train/q_loss", self.get_buffer_mean(q_loss_buffer), global_step)
                self.log("train/policy_loss", self.get_buffer_mean(policy_loss_buffer), global_step)
                self.log("train/entropy_loss", self.get_buffer_mean(entropy_loss_buffer), global_step)
                self.log("train/entropy", self.get_buffer_mean(entropy_buffer), global_step)
                self.log("train/alpha", self.get_buffer_mean(alpha_buffer), global_step)

                if self.track_console:
                    rlx_logger.info("└" + "─" * 31 + "┴" + "─" * 16 + "┘")

                episode_info_buffer.clear()
                acting_time_buffer.clear()
                optimize_time_buffer.clear()
                saving_time_buffer.clear()
                fps_buffer.clear()
                q_loss_buffer.clear()
                policy_loss_buffer.clear()
                entropy_loss_buffer.clear()
                entropy_buffer.clear()
                alpha_buffer.clear()


    def get_buffer_mean(self, buffer):
        if len(buffer) > 0:
            return np.mean(buffer)
        else:
            return 0.0


    def log(self, name, value, step):
        if self.track_tb:
            self.writer.add_scalar(name, value, step)
        if self.track_console:
            self.log_console(name, value)
    

    def log_console(self, name, value):
        value = np.format_float_positional(value, trim="-")
        rlx_logger.info(f"│ {name.ljust(30)}│ {str(value).ljust(14)[:14]} │")
        

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
