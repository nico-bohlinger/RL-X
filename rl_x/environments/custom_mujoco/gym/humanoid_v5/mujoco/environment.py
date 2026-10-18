from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from rl_x.environments.custom_mujoco.gym.humanoid_v5.mujoco.viewer import MujocoViewer


class Humanoid(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render, horizon=1000):
        self.horizon = horizon

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "humanoid.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)

        self.nr_intermediate_steps = 5

        initial_qpos = [0.0, 0.0, 1.4, 1.0] + [0.0] * (self.mj_model.nq - 4)

        self.initial_qpos = np.array(initial_qpos)
        self.initial_qvel = np.zeros(self.mj_model.nv)
        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.mj_model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                (self.mj_model.nq - 2)
                + self.mj_model.nv
                + (self.mj_model.nbody - 1) * 10
                + (self.mj_model.nbody - 1) * 6
                + (self.mj_model.nv - 6)
                + (self.mj_model.nbody - 1) * 6,
            ),
            dtype=np.float32,
        )

        self.forward_reward_weight = 1.25
        self.healthy_z_range = (1.0, 2.0)
        self.terminate_when_unhealthy = True
        self.ctrl_cost_weight = 0.1
        self.contact_cost_weight = 5e-7
        self.contact_cost_range = (-np.inf, 10.0)
        self.healthy_reward = 5.0
        self.reset_noise_scale = 1e-2
        self.dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
        self.body_mass = np.array(self.mj_model.body_mass)
        self.total_mass = self.mj_model.body_mass.sum()

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
        position_before = self.get_mass_center(self.mj_data).copy()
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
        position = data.qpos[2:] # exclude x and y coordinates of the torso
        velocity = data.qvel[:]
        com_inertia = data.cinert[1:].flatten()
        com_velocity = data.cvel[1:].flatten()
        actuator_forces = data.qfrc_actuator[6:].flatten()
        external_contact_forces = data.cfrc_ext[1:].flatten()

        observation = np.nan_to_num(np.concatenate([
            position,
            velocity,
            com_inertia,
            com_velocity,
            actuator_forces,
            external_contact_forces,
        ]))

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)

        return observation


    def get_mass_center(self, data):
        return np.einsum("b,bj->j", self.body_mass, data.xipos)[:2] / self.total_mass


    def get_reward(self, data, xy_position_before):
        """
        Rewards forward motion - control cost
        """
        torso_height = data.qpos[2]
        current_global_linear_velocity = (self.get_mass_center(data) - xy_position_before) / self.dt
        forward_reward = self.forward_reward_weight * current_global_linear_velocity[0]

        min_z, max_z = self.healthy_z_range
        is_healthy = np.clip(np.nan_to_num(((torso_height > min_z) & (torso_height < max_z)).astype('float32')), a_min=0.0, a_max=1.0)
        healthy_reward = self.healthy_reward * is_healthy

        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(data.ctrl))

        contact_forces = data.cfrc_ext
        contact_cost = self.contact_cost_weight * np.sum(np.square(contact_forces))
        min_cost, max_cost = self.contact_cost_range
        contact_cost = np.clip(contact_cost, a_min=min_cost, a_max=max_cost)

        reward = np.nan_to_num(np.clip(forward_reward, a_min=None, a_max=1e4) + healthy_reward - ctrl_cost - contact_cost)


        info = {
            "env_info/global_vel_x": current_global_linear_velocity[0],
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
