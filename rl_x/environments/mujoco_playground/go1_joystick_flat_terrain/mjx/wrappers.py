from functools import partial
import gymnasium as gym
import jax
import jax.numpy as jnp

from rl_x.environments.mujoco_playground.go1_joystick_flat_terrain.mjx.box_space import BoxSpace
from rl_x.environments.mujoco_playground.go1_joystick_flat_terrain.mjx.wrapper_state import WrapperState


class RLXInfo(gym.Wrapper):
    def __init__(self, env, nr_envs):
        super(RLXInfo, self).__init__(env)
        self.nr_envs = nr_envs

        lower_joint_limit, upper_joint_limit = env.env.unwrapped.mj_model.jnt_range[1:].T
        nominal_joint_positions = env.env.unwrapped._default_pose
        action_scale_factor = env.env.unwrapped._config.action_scale
        self.single_action_space = BoxSpace(low=lower_joint_limit, high=upper_joint_limit, shape=(env.action_size,), dtype=jnp.float32, center=nominal_joint_positions, scale=action_scale_factor)
        self.single_observation_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=env.observation_size["privileged_state"], dtype=jnp.float32)

        # This works because the policy observations are contained at the start of the privileged observations
        self.policy_observation_indices = jnp.arange(env.observation_size["state"][0])
        self.critic_observation_indices = jnp.arange(env.observation_size["privileged_state"][0])


    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        env_state = self.env.reset(key)
        info = {
            "rollout/episode_return": jnp.zeros(self.nr_envs),
            "rollout/episode_length": jnp.zeros(self.nr_envs),
            **env_state.metrics
        }
        info_episode_store = {
            "episode_return": jnp.zeros(self.nr_envs),
            "episode_length": jnp.zeros(self.nr_envs),
        }
        wrapper_state = WrapperState(
            env_state=env_state,
            next_observation=env_state.obs["privileged_state"], 
            actual_next_observation=env_state.obs["privileged_state"],
            reward=env_state.reward,
            terminated=jnp.zeros(self.nr_envs, dtype=jnp.bool),
            truncated=jnp.zeros(self.nr_envs, dtype=jnp.bool),
            info=info,
            info_episode_store=info_episode_store
        )
        return wrapper_state
    

    @partial(jax.jit, static_argnums=(0,))
    def step(self, wrapper_state, action):
        action = jnp.clip(action, -1, 1)
        env_state = self.env.step(wrapper_state.env_state, action)
        done = env_state.done.astype(jnp.bool)
        episode_return = wrapper_state.info_episode_store["episode_return"] + env_state.reward
        episode_length = wrapper_state.info_episode_store["episode_length"] + 1.0
        info = {
            "rollout/episode_return": jnp.where(done, episode_return, wrapper_state.info["rollout/episode_return"]),
            "rollout/episode_length": jnp.where(done, episode_length, wrapper_state.info["rollout/episode_length"]),
            **env_state.metrics
        }
        info_episode_store = {
            "episode_return": jnp.where(done, jnp.zeros(self.nr_envs), episode_return),
            "episode_length": jnp.where(done, jnp.zeros(self.nr_envs), episode_length),
        }
        truncated = env_state.info["truncation"].astype(jnp.bool)
        terminated = done & (~truncated)
        wrapper_state = WrapperState(
            env_state=env_state, 
            next_observation=env_state.obs["privileged_state"], 
            actual_next_observation=env_state.obs["privileged_state"],
            reward=env_state.reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store=info_episode_store
        )
        return wrapper_state
    

    def render(self, wrapper_state):
        self.env.render(wrapper_state.env_state)
        return wrapper_state
    

    def close(self):
        pass
