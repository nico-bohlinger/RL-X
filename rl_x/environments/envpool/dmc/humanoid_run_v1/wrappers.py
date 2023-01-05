import gymnasium as gym
import numpy as np

from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_count = 0
        self.episode_returns = None
        self.episode_lengths = None
    
    def extract_obervation(self, timestep):
        joint_angles = timestep.observation.joint_angles
        head_height = timestep.observation.head_height.reshape(-1, 1)
        extremities = timestep.observation.extremities
        torso_vertical = timestep.observation.torso_vertical
        com_velocity = timestep.observation.com_velocity
        position = timestep.observation.position
        velocity = timestep.observation.velocity
        return np.concatenate([joint_angles, head_height, extremities, torso_vertical, com_velocity, position, velocity], axis=1)

    def reset(self):
        timestep = self.env.reset()
        observations = self.extract_obervation(timestep)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        timestep = super(RecordEpisodeStatistics, self).step(action)
        observations = self.extract_obervation(timestep)
        rewards = timestep.reward
        dones = timestep.step_type == 2  # 2 is StepType.LAST
        infos = {}

        self.episode_returns += rewards
        self.episode_lengths += 1
        infos["episode"] = [None] * self.num_envs
        infos["terminal_observation"] = [None] * self.num_envs
        for i in range(len(dones)):
            if dones[i]:
                episode_return = self.episode_returns[i]
                episode_length = self.episode_lengths[i]
                episode_info = {
                    "r": episode_return,
                    "l": episode_length
                }
                infos["episode"][i] = episode_info
                self.episode_count += 1
                self.episode_returns[i] = 0
                self.episode_lengths[i] = 0
                infos["terminal_observation"][i] = np.array(observations[i])
                timestep_ = self.env.reset(np.array([i]))
                observations[i] = self.extract_obervation(timestep_)
        return (observations, rewards, dones, infos)
    

class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)


    def reset(self):
        return self.env.reset()


    def get_episode_infos(self, info):
        episode_infos = []
        for maybe_episode_info in info["episode"]:
            if maybe_episode_info is not None:
                episode_infos.append(maybe_episode_info)
        return episode_infos
    

    def get_terminal_observation(self, info, id):
        return info["terminal_observation"][id]
    

    def get_action_space_type(self):
        return ActionSpaceType.CONTINUOUS

    
    def get_single_action_space_shape(self):
        return self.action_space.shape
    

    def get_observation_space_type(self):
        return ObservationSpaceType.FLAT_VALUES
