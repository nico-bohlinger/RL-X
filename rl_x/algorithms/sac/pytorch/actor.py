import numpy as np
import torch.nn as nn

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


def get_actor(config, env):
    action_space_type = env.get_action_space_type()
    observation_space_type = env.get_observation_space_type()

    if action_space_type == ActionSpaceType.CONTINUOUS and observation_space_type == ObservationSpaceType.FLAT_VALUES:
        return Actor(env, config.algorithm.nr_hidden_units)
    else:
        raise ValueError(f"Unsupported action_space_type: {action_space_type} and observation_space_type: {observation_space_type} combination")
    

class Actor(nn.Module):
    def __init__(self, env, nr_hidden_units):
        super().__init__()
        single_os_shape = env.observation_space.shape
        single_as_shape = env.get_single_action_space_shape()

        self.torso = nn.Sequential(
            nn.Linear(np.prod(single_os_shape), nr_hidden_units),
            nn.ReLU(),
            nn.Linear(nr_hidden_units, nr_hidden_units),
            nn.ReLU(),
        )
        self.mean = nn.Linear(nr_hidden_units, single_as_shape)
        self.logstd = nn.Linear(nr_hidden_units, single_as_shape)