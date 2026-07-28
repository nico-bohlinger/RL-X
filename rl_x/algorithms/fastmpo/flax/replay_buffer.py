import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size_per_env, nr_envs, os_shape, as_shape, n_steps, gamma, rng):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = buffer_size_per_env
        self.nr_envs = nr_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.rng = rng
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs) + as_shape, dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.truncations = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0


    def add(self, states, next_states, actions, rewards, dones, truncations):
        self.states[self.pos] = states
        self.next_states[self.pos] = next_states
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones
        self.truncations[self.pos] = truncations
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, nr_samples):
        if self.n_steps == 1:
            idx_t = self.rng.integers(0, self.size, size=nr_samples)
            idx_e = self.rng.integers(0, self.nr_envs, size=nr_samples)
            states = self.states[idx_t, idx_e].reshape((nr_samples,) + self.os_shape)
            next_states = self.next_states[idx_t, idx_e].reshape((nr_samples,) + self.os_shape)
            actions = self.actions[idx_t, idx_e].reshape((nr_samples,) + self.as_shape)
            rewards = self.rewards[idx_t, idx_e].reshape(nr_samples)
            dones = self.dones[idx_t, idx_e].reshape(nr_samples)
            truncations = self.truncations[idx_t, idx_e].reshape(nr_samples)
            return states, next_states, actions, rewards, dones, truncations, np.ones_like(dones)

        if self.size >= self.capacity:
            truncations_for_sampling = self.truncations.copy()
            last_idx = (self.pos - 1) % self.capacity
            truncations_for_sampling[last_idx] = np.where(self.dones[last_idx] > 0.0, truncations_for_sampling[last_idx], np.ones_like(truncations_for_sampling[last_idx]))
            max_start = self.capacity
        else:
            truncations_for_sampling = self.truncations
            max_start = max(1, self.size - self.n_steps + 1)

        idx_t = self.rng.integers(0, max_start, size=nr_samples)
        idx_e = self.rng.integers(0, self.nr_envs, size=nr_samples)
        states = self.states[idx_t, idx_e]
        actions = self.actions[idx_t, idx_e]
        steps = np.arange(self.n_steps, dtype=np.int64)
        all_t = (idx_t[:, None] + steps[None]) % self.capacity
        env_indices = np.broadcast_to(idx_e[:, None], all_t.shape)
        all_rewards = self.rewards[all_t, env_indices]
        all_dones = self.dones[all_t, env_indices]
        all_truncations = truncations_for_sampling[all_t, env_indices]
        done_masks = np.cumprod(1.0 - np.concatenate([np.zeros((nr_samples, 1), dtype=all_dones.dtype), all_dones[:, :-1]], axis=1), axis=1)
        effective_n_steps = np.sum(done_masks, axis=1)
        rewards = np.sum(all_rewards * done_masks * (self.gamma ** np.arange(self.n_steps, dtype=np.float32))[None], axis=1)
        first_done = np.argmax((all_dones > 0.0).astype(np.int32), axis=1)
        first_truncation = np.argmax((all_truncations > 0.0).astype(np.int32), axis=1)
        first_done = np.where(np.sum(all_dones > 0.0, axis=1) == 0, self.n_steps - 1, first_done)
        first_truncation = np.where(np.sum(all_truncations > 0.0, axis=1) == 0, self.n_steps - 1, first_truncation)
        final_t = all_t[np.arange(nr_samples), np.minimum(first_done, first_truncation)]
        return states, self.next_states[final_t, idx_e], actions, rewards, self.dones[final_t, idx_e], truncations_for_sampling[final_t, idx_e], effective_n_steps
