import gymnasium as gym
import torch


class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)
    

    def get_logging_info_dict(self, info):
        keys_to_remove = ["final_observation", "_final_observation", "final_info", "_final_info", "elapsed_steps", "_elapsed_steps", "reconfigure", "fail", "episode"]

        if "final_info" in info:
            for key in info["final_info"].keys():
                if key not in keys_to_remove:
                    info[key] = torch.where(info["_final_info"].unsqueeze(1), info["final_info"][key], info[key])
            info["episode_return"] = info["final_info"]["episode"]["return"]
            info["episode_length"] = info["final_info"]["episode"]["episode_len"]

        logging_info = {
            key: info[key].tolist() for key in list(info.keys()) if key not in keys_to_remove
        }

        return logging_info


class RenderWrapper(gym.Wrapper):
    def __init__(self, env):
        super(RenderWrapper, self).__init__(env)


    def step(self, action):
        data = super(RenderWrapper, self).step(action)
        self.env.render()
        return data
