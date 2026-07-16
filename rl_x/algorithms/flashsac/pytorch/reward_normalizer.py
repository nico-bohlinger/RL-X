import torch


class RewardNormalizer:
    def __init__(self, nr_envs, gamma, normalized_g_max, device, epsilon=1e-8):
        self.gamma = gamma
        self.G_max = normalized_g_max
        self.epsilon = epsilon
        self.device = device
        self.G_r = torch.zeros((nr_envs,), dtype=torch.float32, device=device)
        self.G_r_max = torch.zeros((), dtype=torch.float32, device=device)
        self.rms_mean = torch.zeros((), dtype=torch.float32, device=device)
        self.rms_var = torch.ones((), dtype=torch.float32, device=device)
        self.rms_count = torch.zeros((), dtype=torch.float32, device=device)


    @torch.no_grad()
    def update(self, reward, terminated, truncated):
        done = (terminated | truncated).float()
        self.G_r = self.gamma * (1.0 - done) * self.G_r + reward
        self.G_r_max = torch.maximum(self.G_r_max, torch.max(torch.abs(self.G_r)))

        sample_mean = self.G_r.mean()
        sample_var = self.G_r.var(unbiased=False)
        sample_count = torch.tensor(float(self.G_r.shape[0]), device=self.device)
        delta = sample_mean - self.rms_mean
        total = self.rms_count + sample_count
        ratio = sample_count / total
        new_mean = self.rms_mean + delta * ratio
        m_a = self.rms_var * (self.rms_count + 1e-4)
        m_b = sample_var * sample_count
        m_2 = m_a + m_b + delta.square() * self.rms_count * ratio
        self.rms_mean = new_mean
        self.rms_var = m_2 / total
        self.rms_count = total


    def normalize(self, reward):
        var_denom = torch.sqrt(self.rms_var + self.epsilon)
        min_denom = self.G_r_max / self.G_max
        denom = torch.maximum(var_denom, min_denom)
        return reward / denom


    def state_dict(self):
        return {
            "G_r": self.G_r,
            "G_r_max": self.G_r_max,
            "rms_mean": self.rms_mean,
            "rms_var": self.rms_var,
            "rms_count": self.rms_count,
        }


    def load_state_dict(self, state_dict):
        self.G_r = state_dict["G_r"].to(self.device)
        self.G_r_max = state_dict["G_r_max"].to(self.device)
        self.rms_mean = state_dict["rms_mean"].to(self.device)
        self.rms_var = state_dict["rms_var"].to(self.device)
        self.rms_count = state_dict["rms_count"].to(self.device)
