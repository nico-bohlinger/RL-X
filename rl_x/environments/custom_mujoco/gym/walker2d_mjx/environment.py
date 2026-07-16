from copy import deepcopy
from pathlib import Path
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from rl_x.environments.custom_mujoco.gym.walker2d_mjx.state import State
from rl_x.environments.custom_mujoco.gym.walker2d_mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.gym.walker2d_mjx.viewer import MujocoViewer


class Walker2D:
    def __init__(self, render, horizon=1000):
        self.horizon = horizon

        xml_path = (Path(__file__).resolve().parent / "data" / "walker2d_v5.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        self.nr_intermediate_steps = 1

        initial_qpos = [0.0, 1.25, 0.0] + [0.0] * (self.mjx_model.nq - 3)
        self.initial_qpos = jnp.array(initial_qpos)
        self.initial_qvel = jnp.zeros(self.mjx_model.nv)

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.single_action_space = BoxSpace(low=action_low, high=action_high, shape=(self.mjx_model.nu,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self.mjx_model.nq + self.mjx_model.nv - 1,),
            dtype=jnp.float32,
        )

        self.forward_reward_weight = 1.0
        self.healthy_z_range: Tuple[float, float] = (0.8, 2.0)
        self.healthy_angle_range = (-1.0, 1.0)
        self.terminate_when_unhealthy = True
        self.ctrl_cost_weight: float = 1e-3
        self.healthy_reward = 1.0
        self.reset_noise_scale = 5e-3

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

    @partial(jax.vmap, in_axes=(None, 0, None))
    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, eval_mode):
        data = self.mjx_data

        next_observation = jnp.zeros(self.single_observation_space.shape, dtype=jnp.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "rollout/episode_return": reward,
            "rollout/episode_length": 0,
            "env_info/local_vel_x": 0.0,
            "env_info/is_healthy": 1.0,
            "env_info/ctrl_cost": 0.0,
        }
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        state = State(data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        key, qpos_key, qvel_key = jax.random.split(state.key, 3)
        qpos = self.initial_qpos + jax.random.uniform(qpos_key, (self.mjx_model.nq,), minval=-self.reset_noise_scale, maxval=self.reset_noise_scale)
        qvel = self.initial_qvel + jax.random.uniform(qvel_key, (self.mjx_model.nv,), minval=-self.reset_noise_scale, maxval=self.reset_noise_scale)

        data = self.mjx_data
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jnp.zeros(self.mjx_model.nu))
        data = mjx.forward(self.mjx_model, data)

        next_observation = self.get_observation(data)
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info_episode_store=info_episode_store,
            key=key,
        )

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
            length=self.nr_intermediate_steps,
        )

        state.info_episode_store["episode_length"] += 1

        next_observation = self.get_observation(data)
        reward, r_info = self.get_reward(data)
        terminated = r_info["env_info/is_healthy"] < 0.5
        truncated = state.info_episode_store["episode_length"] >= self.horizon
        done = terminated | truncated

        state.info.update(r_info)
        state.info_episode_store["episode_return"] += reward
        state.info["rollout/episode_return"] = jnp.where(done, state.info_episode_store["episode_return"], state.info["rollout/episode_return"])
        state.info["rollout/episode_length"] = jnp.where(done, state.info_episode_store["episode_length"], state.info["rollout/episode_length"])

        def when_done(_):
            start_state = self._reset(state)
            start_state = start_state.replace(
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )
            return start_state

        def when_not_done(_):
            return state.replace(
                data=data,
                next_observation=next_observation,
                actual_next_observation=next_observation,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
            )

        return jax.lax.cond(done, when_done, when_not_done, None)

    def get_observation(self, data):
        torso_height = jnp.array([data.qpos[1]])
        torso_pitch = jnp.array([data.qpos[2]])
        joint_positions = data.qpos[3:]

        torso_vel_x = jnp.clip(jnp.array([data.qvel[0]]), -10, 10)
        torso_vel_z = jnp.clip(jnp.array([data.qvel[1]]), -10, 10)
        torso_ang_vel = jnp.clip(jnp.array([data.qvel[2]]), -10, 10)
        joint_velocities = jnp.clip(data.qvel[3:], -10, 10)

        observation = jnp.nan_to_num(jnp.concatenate([
            torso_height,
            torso_pitch,
            joint_positions,
            torso_vel_x,
            torso_vel_z,
            torso_ang_vel,
            joint_velocities,
        ]))
        return observation

    def get_reward(self, data):
        torso_height = data.qpos[1]
        torso_pitch = data.qpos[2]
        local_lin_vel = data.qvel[0]

        forward_reward = self.forward_reward_weight * local_lin_vel

        min_z, max_z = self.healthy_z_range
        min_angle, max_angle = self.healthy_angle_range
        is_healthy = jnp.clip(
            jnp.nan_to_num(((torso_height > min_z) & (torso_height < max_z) & (torso_pitch > min_angle) & (torso_pitch < max_angle)).astype("float32")),
            a_min=0.0,
            a_max=1.0,
        )
        healthy_reward = jax.lax.cond(
            self.terminate_when_unhealthy,
            lambda _: self.healthy_reward,
            lambda _: self.healthy_reward * is_healthy,
            operand=None,
        )

        ctrl_cost = self.ctrl_cost_weight * jnp.sum(jnp.square(data.ctrl))
        reward = jnp.nan_to_num(jnp.clip(forward_reward, max=1e4) + healthy_reward - ctrl_cost)

        info = {
            "env_info/local_vel_x": local_lin_vel,
            "env_info/is_healthy": is_healthy,
            "env_info/ctrl_cost": ctrl_cost,
        }
        return reward, info

    def close(self):
        if self.viewer:
            self.viewer.close()
