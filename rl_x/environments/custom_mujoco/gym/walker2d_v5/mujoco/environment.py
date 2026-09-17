from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from rl_x.environments.custom_mujoco.gym.walker2d_v5.mujoco.viewer import MujocoViewer


class Walker2D(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render, horizon=1000):
        self.horizon = horizon

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "walker2d_v5.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)

        self.nr_intermediate_steps = 4

        initial_qpos = [0.0, 1.25, 0.0] + [0.0] * (self.mj_model.nq - 3)
        self.initial_qpos = np.array(initial_qpos)
        self.initial_qvel = np.zeros(self.mj_model.nv)

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.mj_model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.mj_model.nq + self.mj_model.nv - 1,),
            dtype=np.float32,
        )

        self.forward_reward_weight = 1.0
        self.healthy_z_range = (0.8, 2.0)
        self.healthy_angle_range = (-1.0, 1.0)
        self.terminate_when_unhealthy = True
        self.ctrl_cost_weight = 1e-3
        self.healthy_reward = 1.0
        self.reset_noise_scale = 5e-3
        self.dt = self.mj_model.opt.timestep * self.nr_intermediate_steps

        self.render_mode = "human" if render else None
        self.viewer = None
        self.renderer = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = self.initial_qpos
        self.mj_data.qvel[:] = self.initial_qvel
        self.mj_data.qpos[:] += self.np_random.uniform(-self.reset_noise_scale, self.reset_noise_scale, self.mj_model.nq)
        self.mj_data.qvel[:] += self.np_random.uniform(-self.reset_noise_scale, self.reset_noise_scale, self.mj_model.nv)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        self.episode_return = 0.0
        self.episode_length = 0
        if self.render_mode == "human":
            self.render()
        return self.get_observation(self.mj_data).astype(np.float32), {}


    def step(self, action):
        position_before = self.mj_data.qpos[0]
        self.mj_data.ctrl[:] = action
        mujoco.mj_step(self.mj_model, self.mj_data, nstep=self.nr_intermediate_steps)
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        observation = self.get_observation(self.mj_data).astype(np.float32)
        reward, info = self.get_reward(self.mj_data, position_before)
        terminated = info['env_info/is_healthy'] < 0.5
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


    def get_observation(self, data):
        torso_height = np.array([data.qpos[1]])
        torso_pitch = np.array([data.qpos[2]])
        joint_positions = data.qpos[3:]

        torso_vel_x = np.clip(np.array([data.qvel[0]]), -10, 10)
        torso_vel_z = np.clip(np.array([data.qvel[1]]), -10, 10)
        torso_ang_vel = np.clip(np.array([data.qvel[2]]), -10, 10)
        joint_velocities = np.clip(data.qvel[3:], -10, 10)

        observation = np.nan_to_num(np.concatenate([
            torso_height,
            torso_pitch,
            joint_positions,
            torso_vel_x,
            torso_vel_z,
            torso_ang_vel,
            joint_velocities,
        ]))
        return observation


    def get_reward(self, data, x_position_before):
        torso_height = data.qpos[1]
        torso_pitch = data.qpos[2]
        local_lin_vel = (data.qpos[0] - x_position_before) / self.dt

        forward_reward = self.forward_reward_weight * local_lin_vel

        min_z, max_z = self.healthy_z_range
        min_angle, max_angle = self.healthy_angle_range
        is_healthy = np.clip(
            np.nan_to_num(((torso_height > min_z) & (torso_height < max_z) & (torso_pitch > min_angle) & (torso_pitch < max_angle)).astype("float32")),
            a_min=0.0,
            a_max=1.0,
        )
        healthy_reward = self.healthy_reward * is_healthy

        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(data.ctrl))
        reward = np.nan_to_num(np.clip(forward_reward, a_min=None, a_max=1e4) + healthy_reward - ctrl_cost)

        info = {
            "env_info/local_vel_x": local_lin_vel,
            "env_info/is_healthy": is_healthy,
            "env_info/ctrl_cost": ctrl_cost,
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
