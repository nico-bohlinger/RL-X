import os
import logging
import random
import time
from collections import deque
import numpy as np
import jax
import jax.numpy as jnp
import flax
from flax.training.train_state import TrainState
import optax

from rl_x.algorithms.sac.flax.policy import get_policy
from rl_x.algorithms.sac.flax.critic import get_critic
from rl_x.algorithms.sac.flax.entropy_coefficient import EntropyCoefficient, ConstantEntropyCoefficient
from rl_x.algorithms.sac.flax.replay_buffer import ReplayBuffer
from rl_x.algorithms.sac.flax.rl_train_state import RLTrainState

rlx_logger = logging.getLogger("rl_x")


class SAC():
    def __init__(self, config, env, writer) -> None:
        self.config = config
        self.env = env
        self.writer = writer

        self.save_model = config.runner.save_model
        self.save_path = os.path.join(config.runner.run_path, "models")
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
        self.q_update_freq = config.algorithm.q_update_freq
        self.q_update_steps = config.algorithm.q_update_steps
        self.q_target_update_freq = config.algorithm.q_target_update_freq
        self.policy_update_freq = config.algorithm.policy_update_freq
        self.policy_update_steps = config.algorithm.policy_update_steps
        self.entropy_update_freq = config.algorithm.entropy_update_freq
        self.entropy_update_steps = config.algorithm.entropy_update_steps
        self.entropy_coef = config.algorithm.entropy_coef
        self.target_entropy = config.algorithm.target_entropy
        self.log_freq = config.algorithm.log_freq
        self.nr_hidden_units = config.algorithm.nr_hidden_units

        if config.algorithm.device == "cpu":
            jax.config.update("jax_platform_name", "cpu")
        
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.key = jax.random.PRNGKey(self.seed)
        self.key, policy_key, critic_key, entropy_coefficient_key = jax.random.split(self.key, 4)

        self.env_as_low = env.action_space.low
        self.env_as_high = env.action_space.high

        self.policy, self.get_processed_action = get_policy(config, env)
        self.vector_critic = get_critic(config, env)
        
        if self.entropy_coef == "auto":
            if self.target_entropy == "auto":
                self.target_entropy = -np.prod(env.get_single_action_space_shape()).item()
            else:
                self.target_entropy = float(self.target_entropy)
            self.entropy_coefficient = EntropyCoefficient(1.0)
        else:
            self.entropy_coefficient = ConstantEntropyCoefficient(self.entropy_coef)

        def q_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps / self.q_update_freq * self.q_update_steps))
            return self.learning_rate * fraction
    
        def policy_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps / self.policy_update_freq * self.policy_update_steps))
            return self.learning_rate * fraction

        def entropy_linear_schedule(count):
            step = count - self.learning_starts
            total_steps = self.total_timesteps - self.learning_starts
            fraction = 1.0 - (step / (total_steps / self.entropy_update_freq * self.entropy_update_steps))
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
        def get_action(state: np.ndarray, key: jax.random.PRNGKey):
            dist = self.policy.apply(self.policy_state.params, state)
            key, subkey = jax.random.split(key)
            action = dist.sample(seed=subkey)
            return action, key
        

        @jax.jit
        def convert_batches(batches):
            batch_states = jnp.array([batch[0] for batch in batches])
            batch_next_states = jnp.array([batch[1] for batch in batches])
            batch_actions = jnp.array([batch[2] for batch in batches])
            batch_rewards = jnp.array([batch[3] for batch in batches])
            batch_dones = jnp.array([batch[4] for batch in batches])
            return batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones


        @jax.jit
        def update_critics(states: jnp.ndarray, next_states: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, key: jax.random.PRNGKey):
            q_losses = []
            vector_critic_state = self.vector_critic_state

            for i in range(self.q_update_steps):
                dist = self.policy.apply(self.policy_state.params, next_states[i])
                key, subkey = jax.random.split(key)
                next_actions = dist.sample(seed=subkey)
                next_log_probs = dist.log_prob(next_actions)

                alpha = self.entropy_coefficient.apply(self.entropy_coefficient_state.params)

                next_q_targets = self.vector_critic.apply(vector_critic_state.target_params, next_states[i], next_actions)
                min_next_q_target = jnp.min(next_q_targets, axis=0).reshape(-1, 1)
                y = rewards[i].reshape(-1, 1) + self.gamma * (1 - dones[i].reshape(-1, 1)) * (min_next_q_target - alpha * next_log_probs.reshape(-1, 1))

                def loss_fn(vector_critic_params: flax.core.FrozenDict):
                    qs = self.vector_critic.apply(vector_critic_params, states[i], actions[i])
                    return ((qs - y) ** 2).mean()
                
                q_loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(vector_critic_state.params)
                vector_critic_state = vector_critic_state.apply_gradients(grads=grads)
                q_losses.append(q_loss)

            return jnp.array(q_losses), vector_critic_state, key


        @jax.jit
        def update_critic_targets():
            return self.vector_critic_state.replace(target_params=optax.incremental_update(self.vector_critic_state.params, self.vector_critic_state.target_params, self.tau))


        @jax.jit
        def update_policy(states: jnp.ndarray, key: jax.random.PRNGKey):
            policy_losses = []
            entropies = []
            policy_state = self.policy_state

            for i in range(self.policy_update_steps):
                key, subkey = jax.random.split(key)

                def loss_fn(policy_params: flax.core.FrozenDict):
                    dist = self.policy.apply(policy_params, states[i])
                    current_actions = dist.sample(seed=subkey)
                    current_log_probs = dist.log_prob(current_actions).reshape(-1, 1)

                    qs = self.vector_critic.apply(self.vector_critic_state.params, states[i], current_actions)
                    min_q = jnp.min(qs, axis=0).reshape(-1, 1)

                    alpha = self.entropy_coefficient.apply(self.entropy_coefficient_state.params)

                    policy_loss = (alpha * current_log_probs - min_q).mean()
                    return policy_loss, -current_log_probs.mean()
                
                (policy_loss, entropy), grads = jax.value_and_grad(loss_fn, has_aux=True)(policy_state.params)
                policy_state = policy_state.apply_gradients(grads=grads)
                policy_losses.append(policy_loss)
                entropies.append(entropy)

            return jnp.array(policy_losses), jnp.array(entropies), policy_state, key
        

        @jax.jit
        def get_additional_entropies(states: jnp.ndarray, key: jax.random.PRNGKey):
            entropies = []

            for i in range(self.entropy_update_freq - self.policy_update_freq):
                dist = self.policy.apply(self.policy_state.params, states[i + self.policy_update_freq])
                key, subkey = jax.random.split(key)
                current_actions = dist.sample(seed=subkey)
                current_log_probs = dist.log_prob(current_actions)
                entropies.append(-current_log_probs.mean())
            
            return jnp.array(entropies), key


        @jax.jit
        def update_entropy_coefficient(entropies: jnp.ndarray, key: jax.random.PRNGKey):
            entropy_losses = []
            alphas = []
            entropy_coefficient_state = self.entropy_coefficient_state

            for i in range(self.entropy_update_steps):
                def loss_fn(entropy_coefficient_params: flax.core.FrozenDict):
                    alpha = self.entropy_coefficient.apply(entropy_coefficient_params)
                    return alpha * (entropies[i] - self.target_entropy), alpha
            
                (entropy_loss, alpha), grads = jax.value_and_grad(loss_fn, has_aux=True)(entropy_coefficient_state.params)
                entropy_coefficient_state = entropy_coefficient_state.apply_gradients(grads=grads)
                entropy_losses.append(entropy_loss)
                alphas.append(alpha)

            return jnp.array(entropy_losses), jnp.array(alphas), entropy_coefficient_state, key


        self.set_train_mode()

        replay_buffer = ReplayBuffer(int(self.buffer_size), self.nr_envs, self.env.observation_space.shape, self.env.action_space.shape)

        saving_return_buffer = deque(maxlen=100)
        episode_info_buffer = deque(maxlen=100)
        acting_time_buffer = deque(maxlen=100)
        q_update_time_buffer = deque(maxlen=100)
        q_target_update_time_buffer = deque(maxlen=100)
        policy_update_time_buffer = deque(maxlen=100)
        entropy_update_time_buffer = deque(maxlen=100)
        saving_time_buffer = deque(maxlen=100)
        fps_buffer = deque(maxlen=100)
        q_loss_buffer = deque(maxlen=100)
        policy_loss_buffer = deque(maxlen=100)
        entropy_loss_buffer = deque(maxlen=100)
        entropy_buffer = deque(maxlen=100)
        alpha_buffer = deque(maxlen=100)

        state = self.env.reset()

        global_step = 0
        while global_step < self.total_timesteps:
            start_time = time.time()


            # Acting
            if global_step < self.learning_starts:
                processed_action = np.array([self.env.action_space.sample() for _ in range(self.nr_envs)])
                action = (processed_action - self.env_as_low) / (self.env_as_high - self.env_as_low) * 2.0 - 1.0
            else:
                action, self.key = get_action(jnp.array(state), self.key)
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
            if len(episode_infos) > 0:
                ep_info_returns = [ep_info["r"] for ep_info in episode_infos]
                saving_return_buffer.extend(ep_info_returns)

            acting_end_time = time.time()
            acting_time_buffer.append(acting_end_time - start_time)


            # What to do in this step after acting
            should_learning_start = global_step > self.learning_starts
            should_update_q = should_learning_start and global_step % self.q_update_freq == 0
            should_update_q_target = should_learning_start and global_step % self.q_target_update_freq == 0
            should_update_policy = should_learning_start and global_step % self.policy_update_freq == 0
            should_update_entropy = should_learning_start and self.entropy_coef == "auto" and global_step % self.entropy_update_freq == 0
            should_try_to_save = self.learning_starts and self.save_model and episode_infos
            should_log = global_step % self.log_freq == 0


            # Optimizing - Prepare batches
            if should_update_q or should_update_policy or should_update_entropy:
                max_nr_batches_needed = max(should_update_q * self.q_update_freq, should_update_policy * self.policy_update_freq, should_update_entropy * self.entropy_update_freq)
                batches = [(replay_buffer.sample(self.batch_size)) for _ in range(max_nr_batches_needed)]
                batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = convert_batches(batches)

            
            # Optimizing - Q-functions
            if should_update_q:
                q_losses, self.vector_critic_state, self.key = update_critics(batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones, self.key)
                q_loss_buffer.extend(jax.device_get(q_losses))

            q_update_end_time = time.time()
            q_update_time_buffer.append(q_update_end_time - acting_end_time)


            # Optimizing - Q-function targets
            if should_update_q_target:
                self.vector_critic_state = update_critic_targets()

            q_target_update_end_time = time.time()
            q_target_update_time_buffer.append(q_target_update_end_time - q_update_end_time)


            # Optimizing - Policy
            if should_update_policy:
                policy_losses, batch_entropies, self.policy_state, self.key = update_policy(batch_states, self.key)
                policy_loss_buffer.append(jax.device_get(policy_losses))

            policy_update_end_time = time.time()
            policy_update_time_buffer.append(policy_update_end_time - q_target_update_end_time)


            # Optimizing - Entropy
            if should_update_entropy:
                if self.entropy_update_freq > self.policy_update_freq:
                    additional_entropies, self.key = get_additional_entropies(batch_states, self.key)
                    batch_entropies = jnp.concatenate([batch_entropies, additional_entropies])

                entropy_losses, alphas, self.entropy_coefficient_state, self.key = update_entropy_coefficient(batch_entropies, self.key)
                entropy_loss_buffer.append(jax.device_get(entropy_losses))
                alpha_buffer.append(jax.device_get(alphas))

            entropy_update_end_time = time.time()
            entropy_update_time_buffer.append(entropy_update_end_time - policy_update_end_time)


            # Saving
            if should_try_to_save:
                mean_return = np.mean(saving_return_buffer)
                if mean_return > self.best_mean_return:
                    self.best_mean_return = mean_return
                    self.save()
            
            saving_end_time = time.time()
            saving_time_buffer.append(saving_end_time - entropy_update_end_time)

            fps_buffer.append(self.nr_envs / (saving_end_time - start_time))


            # Logging                
            if should_log:
                if self.track_tb:
                    if len(episode_info_buffer) > 0:
                        self.writer.add_scalar("rollout/ep_rew_mean", np.mean(ep_info_returns), global_step)
                        self.writer.add_scalar("rollout/ep_len_mean", np.mean([ep_info["l"] for ep_info in episode_info_buffer]), global_step)
                        names = list(episode_info_buffer[0].keys())
                        for name in names:
                            if name != "r" and name != "l" and name != "t":
                                self.writer.add_scalar(f"env_info/{name}", np.mean([ep_info[name] for ep_info in episode_info_buffer if name in ep_info.keys()]), global_step)
                    self.writer.add_scalar("time/fps", np.mean(fps_buffer), global_step)
                    self.writer.add_scalar("time/acting_time", np.mean(acting_time_buffer), global_step)
                    self.writer.add_scalar("time/q_update_time", np.mean(q_update_time_buffer), global_step)
                    self.writer.add_scalar("time/q_target_update_time", np.mean(q_target_update_time_buffer), global_step)
                    self.writer.add_scalar("time/policy_update_time", np.mean(policy_update_time_buffer), global_step)
                    self.writer.add_scalar("time/entropy_update_time", np.mean(entropy_update_time_buffer), global_step)
                    self.writer.add_scalar("time/saving_time", np.mean(saving_time_buffer), global_step)
                    self.writer.add_scalar("train/learning_rate", self.policy_learning_rate if isinstance(self.policy_learning_rate, float) else self.policy_learning_rate(global_step), global_step)
                    self.writer.add_scalar("train/q_loss", self.get_buffer_mean(q_loss_buffer), global_step)
                    self.writer.add_scalar("train/policy_loss", self.get_buffer_mean(policy_loss_buffer), global_step)
                    self.writer.add_scalar("train/entropy_loss", self.get_buffer_mean(entropy_loss_buffer), global_step)
                    self.writer.add_scalar("train/entropy", self.get_buffer_mean(entropy_buffer), global_step)
                    self.writer.add_scalar("train/alpha", self.get_buffer_mean(alpha_buffer), global_step)


                episode_info_buffer.clear()
                acting_time_buffer.clear()
                q_update_time_buffer.clear()
                q_target_update_time_buffer.clear()
                policy_update_time_buffer.clear()
                entropy_update_time_buffer.clear()
                saving_time_buffer.clear()
                fps_buffer.clear()
                q_loss_buffer.clear()
                policy_loss_buffer.clear()
                entropy_loss_buffer.clear()
                entropy_buffer.clear()
                alpha_buffer.clear()

                rlx_logger.info(f"Step: {global_step}")


    def get_buffer_mean(self, buffer):
        if len(buffer) > 0:
            return np.mean(buffer)
        else:
            return 0.0


    def test(self):
        self.set_eval_mode()
    

    def set_train_mode(self):
        ...


    def set_eval_mode(self):
        ...
