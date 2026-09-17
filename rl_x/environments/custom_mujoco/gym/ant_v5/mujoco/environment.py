from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.gym.ant_v5.mujoco.viewer import MujocoViewer


class Ant(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render, horizon=1000):
        self.horizon = horizon

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "ant.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)

        self.nr_intermediate_steps = 5

        initial_height = 0.75
        initial_rotation_quaternion = [1.0, 0.0, 0.0, 0.0]
        initial_joint_angles = [0.0, 0.0] * 4
        self.initial_qpos = np.array([0.0, 0.0, initial_height, *initial_rotation_quaternion, *initial_joint_angles])
        self.initial_qvel = np.zeros(self.mj_model.nv)

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.mj_model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.mj_model.nq - 2) + self.mj_model.nv + (self.mj_model.nbody - 1) * 6,),
            dtype=np.float32,
        )

        self.forward_reward_weight = 1.0
        self.healthy_z_range = (0.2, 1.0)
        self.terminate_when_unhealthy = True
        self.ctrl_cost_weight = 0.5
        self.healthy_reward = 1.0
        self.contact_force_range = (-1.0, 1.0)
        self.contact_cost_weight = 5e-4
        self.reset_noise_scale = 0.1
        self.dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
        self.main_body_id = self.mj_model.body("torso").id

        self.render_mode = "human" if render else None
        self.viewer = None
        self.renderer = None


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[:] = self.initial_qpos
        self.mj_data.qvel[:] = self.initial_qvel
        self.mj_data.qpos[:] += self.np_random.uniform(-self.reset_noise_scale, self.reset_noise_scale, self.mj_model.nq)
        self.mj_data.qvel[:] += self.np_random.normal(size=self.mj_model.nv) * self.reset_noise_scale
        mujoco.mj_forward(self.mj_model, self.mj_data)
        mujoco.mj_rnePostConstraint(self.mj_model, self.mj_data)
        self.episode_return = 0.0
        self.episode_length = 0
        if self.render_mode == "human":
            self.render()
        return self.get_observation(self.mj_data).astype(np.float32), {}


    def step(self, action):
        position_before = self.mj_data.xpos[self.main_body_id][:2].copy()
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
        position = data.qpos[2:]
        velocity = data.qvel[:]
        raw_contact_forces = data.cfrc_ext
        min_value, max_value = self.contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        contact_force = contact_forces[1:].flatten()

        observation = np.nan_to_num(np.concatenate([
            position,
            velocity,
            contact_force,
        ]))
        return observation


    def get_reward(self, data, xy_position_before):
        torso_height = data.qpos[2]
        base_orientation = [data.qpos[4], data.qpos[5], data.qpos[6], data.qpos[3]]
        inverted_rotation = Rotation.from_quat(base_orientation).inv()
        current_global_linear_velocity = (data.xpos[self.main_body_id][:2] - xy_position_before) / self.dt
        current_local_linear_velocity = inverted_rotation.apply(data.qvel[:3])[0]
        forward_reward = self.forward_reward_weight * current_global_linear_velocity[0]

        min_z, max_z = self.healthy_z_range
        state = np.concatenate([data.qpos, data.qvel])
        is_healthy = np.clip(
            np.nan_to_num((np.all(np.isfinite(state)) & (torso_height >= min_z) & (torso_height <= max_z)).astype("float32")),
            a_min=0.0,
            a_max=1.0,
        )
        healthy_reward = self.healthy_reward * is_healthy

        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(data.ctrl))

        raw_contact_forces = data.cfrc_ext
        min_value, max_value = self.contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        contact_cost = self.contact_cost_weight * np.sum(np.square(contact_forces))

        reward = np.nan_to_num(np.clip(forward_reward, a_min=None, a_max=1e4) + healthy_reward - ctrl_cost - contact_cost)

        info = {
            "env_info/global_vel_x": current_global_linear_velocity[0],
            "env_info/local_vel_x": current_local_linear_velocity,
            "env_info/is_healthy": is_healthy,
            "env_info/ctrl_cost": ctrl_cost,
            "env_info/contact_cost": contact_cost,
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
