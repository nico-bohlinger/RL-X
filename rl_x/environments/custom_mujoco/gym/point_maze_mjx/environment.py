from copy import deepcopy
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from rl_x.environments.custom_mujoco.gym.point_maze_mjx.state import State
from rl_x.environments.custom_mujoco.gym.point_maze_mjx.box_space import BoxSpace
from rl_x.environments.custom_mujoco.gym.point_maze_mjx.viewer import MujocoViewer


class PointMaze:
    def __init__(self, render, horizon=100, reward_style="dense", flipped=False, success_radius=0.1):
        self.horizon = horizon
        self.reward_style = reward_style
        self.success_radius = success_radius

        if flipped:
            xml_path = (Path(__file__).resolve().parent / "data" / "point_maze_flipped.xml").as_posix()
        else:
            xml_path = (Path(__file__).resolve().parent / "data" / "point_maze.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.make_data(self.mjx_model)

        self.nr_intermediate_steps = 1

        self.initial_qpos = jnp.zeros(self.mjx_model.nq, dtype=jnp.float32)
        self.initial_qvel = jnp.zeros(self.mjx_model.nv, dtype=jnp.float32)

        self.particle_body_id = self.mj_model.body("particle").id
        self.target_body_id = self.mj_model.body("target").id

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.single_action_space = BoxSpace(low=action_low, high=action_high, shape=(self.mjx_model.nu,), dtype=jnp.float32)
        self.single_observation_space = BoxSpace(low=-jnp.inf, high=jnp.inf, shape=(4,), dtype=jnp.float32)

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
            "env_info/target_x": 0.0,
            "env_info/target_y": 0.0,
            "env_info/is_success": 0.0,
            "env_info/reward_dist": 0.0,
            "env_info/reward_ctrl": 0.0,
        }
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        state = State(data, next_observation, next_observation, reward, terminated, truncated, info, info_episode_store, key)
        return self._reset(state)

    @partial(jax.jit, static_argnums=(0,))
    def _reset(self, state):
        data = self.mjx_data
        data = data.replace(qpos=self.initial_qpos, qvel=self.initial_qvel, ctrl=jnp.zeros(self.mjx_model.nu))
        data = mjx.forward(self.mjx_model, data)

        next_observation = self.get_observation(data)
        target_pos = self.get_target_position(data)
        reward = 0.0
        terminated = False
        truncated = False
        info_episode_store = {
            "episode_return": reward,
            "episode_length": 0,
        }

        info = dict(state.info)
        info["env_info/target_x"] = target_pos[0]
        info["env_info/target_y"] = target_pos[1]
        info["env_info/is_success"] = 0.0
        info["env_info/reward_dist"] = 0.0
        info["env_info/reward_ctrl"] = 0.0

        return state.replace(
            data=data,
            next_observation=next_observation,
            actual_next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store=info_episode_store,
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
        reward, r_info = self.get_reward(data, action)
        terminated = r_info["env_info/is_success"] > 0.5
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

    def get_target_position(self, data):
        return data.xpos[self.target_body_id][:2]

    def get_observation(self, data):
        particle_pos = data.xpos[self.particle_body_id][:2]
        target_pos = self.get_target_position(data)
        observation = jnp.concatenate([particle_pos, target_pos])
        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return observation

    def get_reward(self, data, action):
        particle_pos = data.xpos[self.particle_body_id][:2]
        target_pos = self.get_target_position(data)
        diff = particle_pos - target_pos
        dist = jnp.linalg.norm(diff)

        reward_dist = -dist
        reward_ctrl = -jnp.sum(jnp.square(action))
        is_success = (dist <= self.success_radius).astype(jnp.float32)

        if self.reward_style == "sparse":
            reward = jnp.where(is_success > 0.5, 1.0, 0.0)
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        reward = jnp.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        info = {
            "env_info/target_x": target_pos[0],
            "env_info/target_y": target_pos[1],
            "env_info/is_success": is_success,
            "env_info/reward_dist": reward_dist,
            "env_info/reward_ctrl": reward_ctrl,
        }
        return reward, info

    def close(self):
        if self.viewer:
            self.viewer.close()
