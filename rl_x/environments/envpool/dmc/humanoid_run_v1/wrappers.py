import gymnasium as gym
import numpy as np
    

class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)
        self.nr_envs = len(env.all_env_ids)
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


    def reset(self, **kwargs):
        timestep = self.env.reset()
        observations = self.extract_obervation(timestep)
        self.episode_returns = np.zeros(self.nr_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.nr_envs, dtype=np.int32)
        return observations, {}


    def step(self, action):
        timestep = super(RLXInfo, self).step(action)
        observations = self.extract_obervation(timestep)
        rewards = timestep.reward
        terminations = timestep.step_type == 2  # 2 is StepType.LAST
        truncations = np.zeros_like(terminations)  # dmc envs don't have truncated episodes
        dones = terminations | truncations
        infos = {}

        self.episode_returns += rewards
        self.episode_lengths += 1
        if any(dones):
            infos["final_info"] = [{} for _ in range(self.nr_envs)]
            infos["final_observation"] = np.zeros_like(observations)
            infos["done"] = dones
            for i in range(len(dones)):
                if dones[i]:
                    infos["final_info"][i]["episode_return"] = np.array([self.episode_returns[i]])
                    infos["final_info"][i]["episode_length"] = np.array([self.episode_lengths[i]])
                    infos["final_observation"][i] = observations[i].copy()
                    self.episode_returns[i] = 0
                    self.episode_lengths[i] = 0
                    timestep_ = self.env.reset(np.array([i]))
                    observations[i] = self.extract_obervation(timestep_)

        return (observations, rewards, terminations, truncations, infos)


    def get_logging_info_dict(self, info):
        all_keys = list(info.keys())
        keys_to_remove = ["final_observation", "final_info", "done", "env_id", "players", "elapsed_step"]

        logging_info = {key: info[key].tolist() for key in all_keys if key not in keys_to_remove}
        if "final_info" in info:
            for done, final_info in zip(info["done"], info["final_info"]):
                if done:   
                    for key, info_value in final_info.items():
                        if key not in keys_to_remove:
                            logging_info.setdefault(key, []).append(info_value)

        return logging_info


    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]
    

    def get_final_info_value_at_index(self, info, key, index):
        return info["final_info"][index][key]
