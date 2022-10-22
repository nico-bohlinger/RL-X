import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_x.algorithms.sac.torchscript.q_network import get_q_network

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.get_observation_space_type()

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(config, env, device)
    else:
        raise ValueError(f"Unsupported observation_space_type: {observation_space_type}")


class Critic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.gamma = config.algorithm.gamma

        self.q1 = torch.jit.script(get_q_network(config, env).to(device))
        self.q2 = torch.jit.script(get_q_network(config, env).to(device))
        self.q1_target = torch.jit.script(get_q_network(config, env).to(device))
        self.q2_target = torch.jit.script(get_q_network(config, env).to(device))
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
    
        
    @torch.jit.export
    def loss(self, states, next_states, actions, next_actions, next_log_probs, rewards, dones, alpha):
        with torch.no_grad():
            next_q1_target = self.q1_target(next_states, next_actions)
            next_q2_target = self.q2_target(next_states, next_actions)
            min_next_q_target = torch.minimum(next_q1_target, next_q2_target)
            y = rewards.reshape(-1, 1) + self.gamma * (1 - dones.reshape(-1, 1)) * (min_next_q_target - alpha.detach() * next_log_probs)

        q1 = self.q1(states, actions)
        q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        q_loss = (q1_loss + q2_loss) / 2

        return q_loss
    