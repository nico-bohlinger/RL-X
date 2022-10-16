import gym
import numpy as np

from rl_x.environments.action_space_type import ActionSpaceType


class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)


    def get_episode_infos(self, info):
        episode_infos = []
        for single_info in info:
            maybe_episode_info = single_info.get("episode")
            if maybe_episode_info is not None:
                episode_infos.append(maybe_episode_info)
        return episode_infos
    

    def get_terminal_observation(self, info, id):
        return info[id]["terminal_observation"]

    
    def get_action_space_type(self):
        return ActionSpaceType.CONTINUOUS


    def get_single_action_space_shape(self):
        return np.prod(self.action_space.shape)
