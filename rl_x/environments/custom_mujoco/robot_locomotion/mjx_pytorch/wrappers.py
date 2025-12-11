import gymnasium as gym
import numpy as np
import jax
from jax.dlpack import from_dlpack
import torch.utils.dlpack as tpack

from rl_x.environments.custom_mujoco.robot_locomotion.mjx_pytorch.box_space import BoxSpace


class RLXInfo(gym.Wrapper):
    def __init__(self, env, nr_envs, seed):
        super(RLXInfo, self).__init__(env)
        self.mjx_env = env
        self.nr_envs = nr_envs
        self.seed = seed

        self.single_action_space = BoxSpace(low=np.asarray(env.single_action_space.low), high=np.asarray(env.single_action_space.high), shape=env.single_action_space.shape, dtype=np.float32, center=np.asarray(env.single_action_space.center), scale=np.asarray(env.single_action_space.scale))
        self.single_observation_space = BoxSpace(low=env.single_observation_space.low, high=env.single_observation_space.high, shape=env.single_observation_space.shape, dtype=np.float32)
        
        self.policy_observation_indices = np.asarray(env.policy_observation_indices, dtype=np.int64)
        self.critic_observation_indices = np.asarray(env.critic_observation_indices, dtype=np.int64)
        
        self.key = jax.random.PRNGKey(self.seed)
        self.step_fn = jax.jit(self.mjx_env.step)
        self.reset_fn = jax.jit(self.mjx_env.reset, static_argnums=(1,))
        self.env_state = None


    def reset(self):
        self.key, reset_key = jax.random.split(self.key)
        self.reset_keys = jax.random.split(reset_key, self.nr_envs)
        self.env_state = self.reset_fn(self.reset_keys, False)
        observation = _jax_to_torch(self.env_state.next_observation)
        return observation, {}
    

    def step(self, action):
        action = _torch_to_jax(action)
        self.env_state = self.step_fn(self.env_state, action)
        reward = _jax_to_torch(self.env_state.reward)
        observation = _jax_to_torch(self.env_state.next_observation)
        terminated = _jax_to_torch(self.env_state.terminated)
        truncated = _jax_to_torch(self.env_state.truncated)
        info = self.env_state.info
        return observation, reward, terminated, truncated, info
    

    def get_logging_info_dict(self, info):
        return info
    

    def close(self):
        self.mjx_env.close()


def _jax_to_torch(tensor):
    tensor = tpack.from_dlpack(tensor)
    return tensor


def _torch_to_jax(tensor):
    tensor = from_dlpack(tensor)
    return tensor
