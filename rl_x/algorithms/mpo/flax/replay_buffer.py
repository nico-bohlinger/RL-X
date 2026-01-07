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
            effective_n_steps = np.ones_like(dones)
            return states, next_states, actions, rewards, dones, truncations, effective_n_steps
        

        if self.size >= self.capacity:
            truncations_for_sampling = self.truncations.copy()
            last_idx = (self.pos - 1) % self.capacity
            last_trunc_row = truncations_for_sampling[last_idx]
            last_done_row = self.dones[last_idx]
            patched_last_trunc_row = np.where(last_done_row > 0.0, last_trunc_row, np.ones_like(last_trunc_row))
            truncations_for_sampling[last_idx] = patched_last_trunc_row
            max_start = self.capacity
        else:
            truncations_for_sampling = self.truncations
            max_start = max(1, self.size - self.n_steps + 1)
        
        idx_t = self.rng.integers(0, max_start, size=nr_samples)
        idx_e = self.rng.integers(0, self.nr_envs, size=nr_samples)

        states = self.states[idx_t, idx_e]
        actions = self.actions[idx_t, idx_e]

        steps = np.arange(self.n_steps, dtype=np.int64)
        all_t = (idx_t[:, None] + steps[None, :]) % self.capacity
        env_indices = np.broadcast_to(idx_e[:, None], all_t.shape)
        all_rewards = self.rewards[all_t, env_indices]
        all_dones = self.dones[all_t, env_indices]
        all_truncations = truncations_for_sampling[all_t, env_indices]

        zeros_first = np.zeros((nr_samples, 1), dtype=all_dones.dtype)
        all_dones_shifted = np.concatenate([zeros_first, all_dones[:, :-1]], axis=1)
        done_masks = np.cumprod(1.0 - all_dones_shifted, axis=1)
        effective_n_steps = np.sum(done_masks, axis=1)
        discounts = self.gamma ** np.arange(self.n_steps, dtype=np.float32)
        rewards = np.sum(all_rewards * done_masks * discounts[None, :], axis=1)

        all_dones_int = (all_dones > 0.0).astype(np.int32)
        all_trunc_int = (all_truncations > 0.0).astype(np.int32)
        first_done = np.argmax(all_dones_int, axis=1)
        first_trunc = np.argmax(all_trunc_int, axis=1)
        no_dones = np.sum(all_dones_int, axis=1) == 0
        no_truncs = np.sum(all_trunc_int, axis=1) == 0
        first_done = np.where(no_dones, self.n_steps - 1, first_done)
        first_trunc = np.where(no_truncs, self.n_steps - 1, first_trunc)
        final_offset = np.minimum(first_done, first_trunc)
        final_t = all_t[np.arange(nr_samples), final_offset]
        next_states = self.next_states[final_t, idx_e]
        dones = self.dones[final_t, idx_e]
        truncations = truncations_for_sampling[final_t, idx_e]

        return states, next_states, actions, rewards, dones, truncations, effective_n_steps
