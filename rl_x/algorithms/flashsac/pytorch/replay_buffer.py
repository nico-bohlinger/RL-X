import torch


class ReplayBuffer:
    def __init__(self, buffer_size, nr_envs, os_shape, action_dim, n_steps, gamma, device):
        self.capacity = int(buffer_size // nr_envs)
        if self.capacity < n_steps:
            raise ValueError("The replay buffer must hold at least n_steps transitions per environment.")
        self.nr_envs = nr_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device
        self.states = torch.zeros((self.capacity, nr_envs) + os_shape, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, nr_envs) + os_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, nr_envs, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, nr_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.capacity, nr_envs), dtype=torch.float32, device=device)
        self.truncations = torch.zeros((self.capacity, nr_envs), dtype=torch.float32, device=device)
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


    def sample(self, batch_size):
        if self.n_steps == 1:
            idx_t = torch.randint(0, self.size, (batch_size,), device=self.device)
            idx_e = torch.randint(0, self.nr_envs, (batch_size,), device=self.device)
            return (
                self.states[idx_t, idx_e],
                self.next_states[idx_t, idx_e],
                self.actions[idx_t, idx_e],
                self.rewards[idx_t, idx_e],
                self.dones[idx_t, idx_e],
                self.truncations[idx_t, idx_e],
                torch.ones((batch_size,), dtype=torch.float32, device=self.device),
            )

        max_start = self.capacity if self.size >= self.capacity else self.size - self.n_steps + 1
        idx_t = torch.randint(0, max_start, (batch_size,), device=self.device)
        idx_e = torch.randint(0, self.nr_envs, (batch_size,), device=self.device)

        steps = torch.arange(self.n_steps, device=self.device)
        all_indices = (idx_t.unsqueeze(-1) + steps) % self.capacity
        env_indices = idx_e.unsqueeze(-1).expand_as(all_indices)
        all_rewards = self.rewards[all_indices, env_indices]
        all_dones = self.dones[all_indices, env_indices]
        all_truncs = self.truncations[all_indices, env_indices]
        if self.size >= self.capacity:
            last_idx = (self.pos - 1) % self.capacity
            all_truncs = torch.where((all_indices == last_idx) & (all_dones <= 0.0), torch.ones_like(all_truncs), all_truncs)

        all_episode_ends = torch.maximum(all_dones, all_truncs)
        zeros_first = torch.zeros((batch_size, 1), dtype=all_episode_ends.dtype, device=self.device)
        shifted = torch.cat([zeros_first, all_episode_ends[:, :-1]], dim=-1)
        done_masks = torch.cumprod(1 - shifted, dim=-1)
        effective_n_steps = done_masks.sum(-1)
        discounts = self.gamma ** torch.arange(self.n_steps, device=self.device).float()
        reward = (all_rewards * done_masks * discounts).sum(-1)

        all_dones_int = (all_dones > 0.0).int()
        all_trunc_int = (all_truncs > 0.0).int()
        first_done = torch.where(all_dones_int.sum(-1) == 0, torch.tensor(self.n_steps - 1, device=self.device), all_dones_int.argmax(-1))
        first_trunc = torch.where(all_trunc_int.sum(-1) == 0, torch.tensor(self.n_steps - 1, device=self.device), all_trunc_int.argmax(-1))
        final_offset = torch.minimum(first_done, first_trunc)
        final_t = torch.gather(all_indices, -1, final_offset.unsqueeze(-1)).squeeze(-1)
        next_state = self.next_states[final_t, idx_e]
        done = self.dones[final_t, idx_e]
        truncated = self.truncations[final_t, idx_e]
        if self.size >= self.capacity:
            truncated = torch.where((final_t == last_idx) & (done <= 0.0), torch.ones_like(truncated), truncated)

        state = self.states[idx_t, idx_e]
        action = self.actions[idx_t, idx_e]
        return state, next_state, action, reward, done, truncated, effective_n_steps


    def can_sample(self):
        return self.size >= self.n_steps
