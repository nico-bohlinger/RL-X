import numpy as np


class ReplayBuffer:
    def __init__(self, buffer_size, nr_envs, os_shape, action_dim, n_steps, gamma):
        self.capacity = int(buffer_size // nr_envs)
        if self.capacity < n_steps:
            raise ValueError("The replay buffer must hold at least n_steps transitions per environment.")
        self.nr_envs = nr_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.next_states = np.zeros((self.capacity, nr_envs) + os_shape, dtype=np.float32)
        self.actions = np.zeros((self.capacity, nr_envs, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.dones = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.truncations = np.zeros((self.capacity, nr_envs), dtype=np.float32)
        self.pos = 0
        self.size = 0


    def add(self, state, next_state, action, reward, done, truncated):
        self.states[self.pos] = state
        self.next_states[self.pos] = next_state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.truncations[self.pos] = truncated
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, rng, batch_size):
        if self.n_steps == 1:
            idx_t = rng.integers(0, self.size, size=batch_size)
            idx_e = rng.integers(0, self.nr_envs, size=batch_size)
            return (
                self.states[idx_t, idx_e],
                self.next_states[idx_t, idx_e],
                self.actions[idx_t, idx_e],
                self.rewards[idx_t, idx_e],
                self.dones[idx_t, idx_e],
                self.truncations[idx_t, idx_e],
                np.ones((batch_size,), dtype=np.float32),
            )

        max_start = self.capacity if self.size >= self.capacity else self.size - self.n_steps + 1
        idx_t = rng.integers(0, max_start, size=batch_size)
        idx_e = rng.integers(0, self.nr_envs, size=batch_size)

        steps = np.arange(self.n_steps)
        all_indices = (idx_t[:, None] + steps) % self.capacity
        env_indices = np.broadcast_to(idx_e[:, None], all_indices.shape)
        all_rewards = self.rewards[all_indices, env_indices]
        all_dones = self.dones[all_indices, env_indices]
        all_truncs = self.truncations[all_indices, env_indices]
        if self.size >= self.capacity:
            last_idx = (self.pos - 1) % self.capacity
            all_truncs = np.where((all_indices == last_idx) & (all_dones <= 0.0), 1.0, all_truncs)

        all_episode_ends = np.maximum(all_dones, all_truncs)
        zeros_first = np.zeros((batch_size, 1), dtype=all_episode_ends.dtype)
        shifted = np.concatenate([zeros_first, all_episode_ends[:, :-1]], axis=-1)
        done_masks = np.cumprod(1 - shifted, axis=-1)
        effective_n_steps = np.sum(done_masks, axis=-1).astype(np.float32)
        discounts = self.gamma ** np.arange(self.n_steps)
        reward = np.sum(all_rewards * done_masks * discounts, axis=-1)

        all_dones_int = (all_dones > 0.0).astype(np.int32)
        all_trunc_int = (all_truncs > 0.0).astype(np.int32)
        first_done = np.where(np.sum(all_dones_int, -1) == 0, self.n_steps - 1, np.argmax(all_dones_int, -1))
        first_trunc = np.where(np.sum(all_trunc_int, -1) == 0, self.n_steps - 1, np.argmax(all_trunc_int, -1))
        final_offset = np.minimum(first_done, first_trunc)
        final_t = np.take_along_axis(all_indices, final_offset[:, None], axis=-1).squeeze(-1)
        next_state = self.next_states[final_t, idx_e]
        done = self.dones[final_t, idx_e]
        truncated = self.truncations[final_t, idx_e]
        if self.size >= self.capacity:
            truncated = np.where((final_t == last_idx) & (done <= 0.0), 1.0, truncated)

        state = self.states[idx_t, idx_e]
        action = self.actions[idx_t, idx_e]
        return state, next_state, action, reward, done, truncated, effective_n_steps


    def can_sample(self):
        return self.size >= self.n_steps
