import numpy as np
import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.IMAGES:
        return Critic(env.single_observation_space.shape, env.get_single_action_logit_size(), config.algorithm.nr_hidden_units).to(device)


class Critic(nn.Module):
    def __init__(self, observation_shape, nr_available_actions, nr_hidden_units):
        super().__init__()
        height, width = observation_shape[1:]
        height, width = (height - 8) // 4 + 1, (width - 8) // 4 + 1
        height, width = (height - 4) // 2 + 1, (width - 4) // 2 + 1
        height, width = height - 2, width - 2

        self.conv1 = nn.Conv2d(observation_shape[0], 32, kernel_size=8, stride=4)
        self.norm1 = nn.LayerNorm(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.norm2 = nn.LayerNorm(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.norm3 = nn.LayerNorm(64)
        self.linear = nn.Linear(64 * height * width, nr_hidden_units)
        self.norm4 = nn.LayerNorm(nr_hidden_units)
        self.output = nn.Linear(nr_hidden_units, nr_available_actions)

        for layer in [self.conv1, self.conv2, self.conv3, self.linear]:
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
            nn.init.zeros_(layer.bias)
        nn.init.normal_(self.output.weight, std=np.sqrt(1 / nr_hidden_units))
        nn.init.zeros_(self.output.bias)


    def forward(self, x):
        x = x / 255.0
        x = torch.relu(self.norm1(self.conv1(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        x = torch.relu(self.norm2(self.conv2(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        x = torch.relu(self.norm3(self.conv3(x).permute(0, 2, 3, 1))).permute(0, 3, 1, 2)
        x = x.reshape(x.shape[0], -1)
        x = torch.relu(self.norm4(self.linear(x)))
        return self.output(x)
