import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp


class RLXInfo(gym.Wrapper):
    def __init__(self, env):
        super(RLXInfo, self).__init__(env)
        self.single_action_space = self.action_space
        self.single_observation_space = self.observation_space
    

    def get_logging_info_dict(self, info):
        all_keys = list(info.keys())

        keys_to_remove = ["final_observation", "final_info", "done", "episode_return", "episode_length"]
        logging_info = {key: info[key].tolist() for key in all_keys if key not in keys_to_remove}

        for key in info["final_info"]:
            if key not in logging_info:
                logging_info[key] = []
            logging_info[key].extend(info["final_info"][key][info["done"]])

        return logging_info
    

    def get_final_observation_at_index(self, info, index):
        return info["final_observation"][index]
    

    def get_final_info_value_at_index(self, info, key, index):
        return info["final_info"][key][index]


class GymWrapper(gym.Wrapper):
    def __init__(self, env, seed=0, nr_envs=1):
        self.env = env
        self.seed = seed
        self.nr_envs = nr_envs
        self.reset_fn = jax.jit(jax.vmap(self.env.reset))
        self.step_fn = jax.jit(jax.vmap(self.env.step))
        self.key = jax.random.PRNGKey(seed)
        action_bounds = self.env.model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=jnp.float32, seed=seed)
        self.observation_space = gym.spaces.Box(low=-jnp.inf, high=jnp.inf, shape=(34,), dtype=jnp.float32, seed=seed)


    def reset(self, **kwargs):
        keys = jax.random.split(self.key, self.nr_envs+1)
        self.key, env_keys = keys[0], keys[1:]

        self.state = self.reset_fn(env_keys)
        observation = np.asarray(self.state.observation)
        info = self.get_gym_info()

        return observation, info
    

    def step(self, action):
        self.state = self.step_fn(self.state, action)
        observation = np.asarray(self.state.observation)
        reward = np.asarray(self.state.reward)
        terminated = np.asarray(self.state.terminated)
        truncated = np.asarray(self.state.truncated)
        info = self.get_gym_info()

        return observation, reward, terminated, truncated, info
    

    def get_gym_info(self):
        info_keys_to_remove = ["key",]
        info = {key: self.state.info[key] for key in self.state.info if key not in info_keys_to_remove}

        return info
    

    def close(self):
        pass
