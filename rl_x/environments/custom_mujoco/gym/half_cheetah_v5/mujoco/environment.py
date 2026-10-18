from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np

from rl_x.environments.custom_mujoco.gym.half_cheetah_v5.mujoco.viewer import MujocoViewer


class HalfCheetah(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render, horizon=1000):
        self.horizon = horizon

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "half_cheetah.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        self.mj_data = mujoco.MjData(self.mj_model)

        self.nr_intermediate_steps = 5

        self.initial_qpos = np.zeros(self.mj_model.nq)
        self.initial_qvel = np.zeros(self.mj_model.nv)

        action_bounds = self.mj_model.actuator_ctrlrange
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, shape=(self.mj_model.nu,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=((self.mj_model.nq - 1) + self.mj_model.nv,),
            dtype=np.float32,
        )

        self.forward_reward_weight = 1.0
        self.ctrl_cost_weight = 0.1
        self.reset_noise_scale = 0.1
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
        self.mj_data.qvel[:] += self.np_random.normal(size=self.mj_model.nv) * self.reset_noise_scale
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
        terminated = False
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
        position = data.qpos[1:]
        velocity = data.qvel[:]
        observation = np.nan_to_num(np.concatenate([
            position,
            velocity,
        ]))
        return observation


    def get_reward(self, data, x_position_before):
        local_lin_vel = (data.qpos[0] - x_position_before) / self.dt
        forward_reward = self.forward_reward_weight * local_lin_vel
        ctrl_cost = self.ctrl_cost_weight * np.sum(np.square(data.ctrl))
        reward = np.nan_to_num(np.clip(forward_reward, a_min=None, a_max=1e4) - ctrl_cost)

        info = {
            "env_info/local_vel_x": local_lin_vel,
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
