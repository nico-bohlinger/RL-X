from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from rl_x.environments.custom_mujoco.gym.point_maze_mujoco.viewer import MujocoViewer


class PointMaze(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render, horizon=100, reward_style="dense", flipped=False, success_radius=0.1):
        self.horizon = horizon
        self.reward_style = reward_style
        self.success_radius = success_radius

        if flipped:
            xml_path = (Path(__file__).resolve().parent.parent / "point_maze_mjx" / "data" / "point_maze_flipped.xml").as_posix()
        else:
            xml_path = (Path(__file__).resolve().parent.parent / "point_maze_mjx" / "data" / "point_maze.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)

        self.nr_intermediate_steps = 1

        self.initial_qpos = np.zeros(self.mj_model.nq, dtype=np.float32)
        self.initial_qvel = np.zeros(self.mj_model.nv, dtype=np.float32)

        self.particle_body_id = self.mj_model.body("particle").id
        self.target_body_id = self.mj_model.body("target").id

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.mj_model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)

        self.render_mode = "human" if render else None
        self.viewer = None
        self.renderer = None
        self.dt = self.mj_model.opt.timestep * self.nr_intermediate_steps


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = self.initial_qpos
        self.mj_data.qvel[:] = self.initial_qvel
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        self.episode_return = 0.0
        self.episode_length = 0
        if self.render_mode == "human":
            self.render()
        return self.get_observation(self.mj_data).astype(np.float32), {}


    def step(self, action):
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data, nstep=self.nr_intermediate_steps)
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        observation = self.get_observation(self.mj_data).astype(np.float32)
        reward, info = self.get_reward(self.mj_data, action)
        terminated = info['env_info/is_success'] > 0.5
        self.episode_length += 1
        self.episode_return += reward
        truncated = self.episode_length >= self.horizon
        info = {key.removeprefix("env_info/"): float(value) for key, value in info.items()}
        if terminated or truncated:
            info["episode_return"] = float(self.episode_return)
            info["episode_length"] = self.episode_length
        if self.render_mode == "human":
            self.render()
        return observation, float(reward), bool(terminated), truncated, info


    def get_target_position(self, data):
        return data.xpos[self.target_body_id][:2]


    def get_observation(self, data):
        particle_pos = data.xpos[self.particle_body_id][:2]
        target_pos = self.get_target_position(data)
        observation = np.concatenate([particle_pos, target_pos])
        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        return observation


    def get_reward(self, data, action):
        particle_pos = data.xpos[self.particle_body_id][:2]
        target_pos = self.get_target_position(data)
        diff = particle_pos - target_pos
        dist = np.linalg.norm(diff)

        reward_dist = -dist
        reward_ctrl = -np.sum(np.square(action))
        is_success = (dist <= self.success_radius).astype(np.float32)

        if self.reward_style == "sparse":
            reward = np.where(is_success > 0.5, 1.0, 0.0)
        else:
            reward = reward_dist + 0.001 * reward_ctrl

        reward = np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        info = {
            "env_info/target_x": target_pos[0],
            "env_info/target_y": target_pos[1],
            "env_info/is_success": is_success,
            "env_info/reward_dist": reward_dist,
            "env_info/reward_ctrl": reward_ctrl,
        }
        return reward, info


    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.mj_model)
            self.renderer.update_scene(self.mj_data)
            return self.renderer.render()
        if self.viewer is None:
            self.viewer = MujocoViewer(self.mj_model, self.dt)
        self.viewer.render(self.mj_data)


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
