from gymnasium import Env
from gymnasium.spaces import Box
import numpy as np

from rl_x.environments.custom_interface.prototype.connection import Connection


class CustomEnvironment(Env):
    def __init__(self, ip, port):
        self.connection = Connection(port)
        action_count, observation_count = self.connection.start(ip)
        self.action_space = Box(low=-1, high=1, shape=(action_count,), dtype=np.float32)
        self.observation_space = Box(low=-1, high=1, shape=(observation_count,), dtype=np.float32)

    def step(self, action):
        self.connection.send(action)
        reaction = self.connection.recv()
        info = {}
        if "extraValueNames" in reaction:
            for extra_value_name, extra_value in zip(reaction["extraValueNames"], reaction["extraValues"]):
                info[extra_value_name] = extra_value
        return reaction["observation"], reaction["reward"], reaction["terminated"], reaction["truncated"], info

    def reset(self, seed=None):
        return self.connection.recv()["observation"], {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
