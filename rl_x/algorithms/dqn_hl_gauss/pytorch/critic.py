import torch
import torch.nn as nn

from rl_x.environments.observation_space_type import ObservationSpaceType


def get_critic(config, env, device):
    observation_space_type = env.general_properties.observation_space_type

    if observation_space_type == ObservationSpaceType.IMAGES:
        return Critic(env.single_observation_space.shape[0], env.get_single_action_logit_size(), config.algorithm.nr_bins, config.algorithm.nr_hidden_units).to(device)


class Critic(nn.Module):
    def __init__(self, nr_input_channels, nr_available_actions, nr_bins, nr_hidden_units):
        super().__init__()
        self.nr_available_actions = nr_available_actions
        self.nr_bins = nr_bins
        self.network = nn.Sequential(
            nn.Conv2d(nr_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.LazyLinear(nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_available_actions * nr_bins),
        )


    def forward(self, x):
        return self.network(x / 255.0).reshape(x.shape[0], self.nr_available_actions, self.nr_bins)
