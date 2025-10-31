from copy import deepcopy
from pathlib import Path
from functools import partial
import mujoco
from mujoco import mjx
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.ant.mjx.state import State
from rl_x.environments.custom_mujoco.ant.mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.ant.mjx.viewer import MujocoViewer


class Ant:
    def __init__(self, render, horizon=1000):
        self.horizon = horizon
        
        xml_path = (Path(__file__).resolve().parent / "data" / "ant.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        self.nr_intermediate_steps = 1

        self.initial_qpos = jnp.array(self.mj_model.keyframe("home").qpos)
        self.initial_qvel = jnp.array(self.mj_model.keyframe("home").qvel)

        self.target_local_x_velocity = 2.0
        self.target_local_y_velocity = 0.0

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.single_action_space = BoxSpace(low=action_low, high=action_high, shape=(8,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(34,), dtype=jnp.float32)

        self.viewer = None
        if render:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)
            c_model = deepcopy(self.mj_model)
            c_data = mujoco.MjData(c_model)
            mujoco.mj_step(c_model, c_data, 1)
            self.light_xdir = c_data.light_xdir
            self.light_xpos = c_data.light_xpos
            del c_model, c_data

    
    def render(self, state):
        env_id = 0
        data = mjx.get_data(self.mj_model, state.data)[env_id]

        data.light_xdir = self.light_xdir
        data.light_xpos = self.light_xpos

        self.viewer.render(data)

        return state
    


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "rollout/episode_return": reward,
            "rollout/episode_length": 0,
            "env_info/reward_xy_vel_cmd": 0.0,
            "env_info/xy_vel_diff_norm": 0.0,
        }
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        state = State(data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, key)

        return self._reset(state)


    @partial(jax.vmap, in_axes=(None, 0))
    @partial(jax.jit, static_argnums=(0,))
    def _vmap_reset(self, state):
        return self._reset(state)


    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        data = self.mjx_data
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel)

        next_observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        new_state = state.replace(
            data=data,
            next_observation=next_observation, actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated, truncated=truncated,
            info_episode_store=info_episode_store
        )

        return new_state


    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        return self._step(state, action)


    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action):
        data, _ = jax.lax.scan(
            f=lambda data, _: (mjx.step(self.mjx_model, data.replace(ctrl=action)), None),
            init=state.data,
            xs=(),
            length=self.nr_intermediate_steps
        )

        state.info_episode_store["episode_length"] += 1

        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data)
        terminated = (data.qpos[2] < 0.2) | (data.qpos[2] > 1.0)
        truncated = state.info_episode_store["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info_episode_store["episode_return"] += reward
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_length"], state.info["rollout/episode_length"])

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
            return start_state
        def when_not_done(_):
            return state.replace(data=data, next_observation=next_observation, actual_next_observation=next_observation, reward=reward, terminated=terminated, truncated=truncated)
        state = jax.lax.cond(done, when_done, when_not_done, None)

        return state


    def get_observation(self, data):
        global_height = jnp.array([data.qpos[2]])
        joint_positions = data.qpos[7:]
        joint_velocities = data.qvel[6:]
        local_angular_velocities = data.qvel[3:6]

        base_orientation = [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]  # scipy quaternion format: [x, y, z, w]
        inverted_rotation = Rotation.from_quat(base_orientation).inv()
        global_linear_velocities = data.qvel[:3]
        local_linear_velocities = inverted_rotation.apply(global_linear_velocities)
        projected_gravity_vector = inverted_rotation.apply(jnp.array([0.0, 0.0, -1.0]))

        observation = jnp.concatenate([
            global_height,
            joint_positions, joint_velocities,
            local_linear_velocities, local_angular_velocities,
            projected_gravity_vector,
            data.ctrl
        ])

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -100.0, 100.0)

        return observation
    
    
    def get_reward(self, data):
        base_orientation = [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]
        inverted_rotation = Rotation.from_quat(base_orientation).inv()
        current_global_linear_velocity = data.qvel[:3]
        current_local_linear_velocity = inverted_rotation.apply(current_global_linear_velocity)
        target_local_linear_velocity_xy = jnp.array([self.target_local_x_velocity, self.target_local_y_velocity])
        xy_velocity_difference_norm =  jnp.sum(jnp.square(target_local_linear_velocity_xy - current_local_linear_velocity[:2]))
        tracking_xy_velocity_command_reward = jnp.exp(-xy_velocity_difference_norm / 0.25)

        reward = tracking_xy_velocity_command_reward

        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = jnp.clip(reward, -10.0, 10.0)

        info = {
            "env_info/reward_xy_vel_cmd": tracking_xy_velocity_command_reward,
            "env_info/xy_vel_diff_norm": xy_velocity_difference_norm,
        }

        return reward, info


    def close(self):
        if self.viewer:
            self.viewer.close()
