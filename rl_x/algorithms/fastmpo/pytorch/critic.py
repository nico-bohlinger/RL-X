import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.algorithms.fastmpo.pytorch.q_network import get_q_network


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Critic(config, env, device)


class Critic(nn.Module):
    def __init__(self, config, env, device):
        super().__init__()
        self.q1 = get_q_network(config, env, device)
        self.q2 = get_q_network(config, env, device)
        self.q1_target = get_q_network(config, env, device)
        self.q2_target = get_q_network(config, env, device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
