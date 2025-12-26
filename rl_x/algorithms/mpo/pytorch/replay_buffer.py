import torch


class ReplayBuffer:
    def __init__(self, buffer_size_per_env, nr_envs, os_shape, as_shape, n_steps, gamma, device):
        self.os_shape = os_shape
        self.as_shape = as_shape
        self.capacity = buffer_size_per_env
        self.nr_envs = nr_envs
        self.n_steps = n_steps
        self.gamma = gamma
        self.device = device
        self.states = torch.zeros((self.capacity, nr_envs) + os_shape, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((self.capacity, nr_envs) + os_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.capacity, nr_envs) + as_shape, dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, nr_envs), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.capacity, nr_envs), dtype=torch.float32, device=device)
        self.truncations = torch.zeros((self.capacity, nr_envs), dtype=torch.float32, device=device)
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
            idx_t = torch.randint(0, self.size, (nr_samples,), device=self.device)
            idx_e = torch.randint(0, self.nr_envs, (nr_samples,), device=self.device)
            states = self.states[idx_t, idx_e].reshape((nr_samples,) + self.os_shape)
            next_states = self.next_states[idx_t, idx_e].reshape((nr_samples,) + self.os_shape)
            actions = self.actions[idx_t, idx_e].reshape((nr_samples,) + self.as_shape)
            rewards = self.rewards[idx_t, idx_e].reshape(nr_samples)
            dones = self.dones[idx_t, idx_e].reshape(nr_samples)
            truncations = self.truncations[idx_t, idx_e].reshape(nr_samples)
            effective_n_steps = torch.ones_like(dones)
            return states, next_states, actions, rewards, dones, truncations, effective_n_steps
        

        if self.size >= self.capacity:
            truncations_for_sampling = self.truncations.clone()
            last_idx = (self.pos - 1) % self.capacity
            last_trunc_row = truncations_for_sampling[last_idx]
            last_done_row = self.dones[last_idx]
            patched_last_trunc_row = torch.where(last_done_row > 0.0, last_trunc_row, torch.ones_like(last_trunc_row))
            truncations_for_sampling[last_idx] = patched_last_trunc_row
            max_start = self.capacity
        else:
            truncations_for_sampling = self.truncations
            max_start = max(1, self.size - self.n_steps + 1)
        
        idx_t = torch.randint(0, max_start, (nr_samples,), device=self.device)
        idx_e = torch.randint(0, self.nr_envs, (nr_samples,), device=self.device)

        states = self.states[idx_t, idx_e]
        actions = self.actions[idx_t, idx_e]

        steps = torch.arange(self.n_steps, dtype=torch.int, device=self.device)
        all_t = (idx_t.unsqueeze(1) + steps.unsqueeze(0)) % self.capacity
        env_indices = idx_e.unsqueeze(1).expand_as(all_t)
        all_rewards = self.rewards[all_t, env_indices]
        all_dones = self.dones[all_t, env_indices]
        all_truncations = truncations_for_sampling[all_t, env_indices]

        zeros_first = torch.zeros((nr_samples, 1), dtype=all_dones.dtype, device=self.device)
        all_dones_shifted = torch.cat([zeros_first, all_dones[:, :-1]], dim=1)
        done_masks = torch.cumprod(1.0 - all_dones_shifted, dim=1)
        effective_n_steps = torch.sum(done_masks, dim=1)
        discounts = self.gamma ** torch.arange(self.n_steps, dtype=torch.float32, device=self.device)
        rewards = torch.sum(all_rewards * done_masks * discounts.view(1, -1), dim=1)

        all_dones_int = (all_dones > 0.0).int()
        all_trunc_int = (all_truncations > 0.0).int()
        first_done = torch.argmax(all_dones_int, dim=1)
        first_trunc = torch.argmax(all_trunc_int, dim=1)
        no_dones = torch.sum(all_dones_int, dim=1) == 0
        no_truncs = torch.sum(all_trunc_int, dim=1) == 0
        last_step = torch.full_like(first_done, self.n_steps - 1)
        first_done = torch.where(no_dones, last_step, first_done)
        first_trunc = torch.where(no_truncs, last_step, first_trunc)
        final_offset = torch.minimum(first_done, first_trunc)
        batch_idx = torch.arange(nr_samples, dtype=torch.long, device=self.device)
        final_t = all_t[batch_idx, final_offset]
        next_states = self.next_states[final_t, idx_e]
        dones = self.dones[final_t, idx_e]
        truncations = truncations_for_sampling[final_t, idx_e]

        return states, next_states, actions, rewards, dones, truncations, effective_n_steps
