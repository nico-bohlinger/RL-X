from pathlib import Path
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import gymnasium as gym

from rl_x.environments.custom_mujoco.ant.viewer import MujocoViewer


class Ant(gym.Env):
    def __init__(self, horizon=1000, render=False):
        self.horizon = horizon

        xml_path = (Path(__file__).resolve().parent / "data" / "ant.xml").as_posix()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.nr_substeps = 1
        self.nr_intermediate_steps = 1
        self.dt = self.model.opt.timestep * self.nr_substeps * self.nr_intermediate_steps

        self.viewer = None if not render else MujocoViewer(self.model, self.dt)

        action_bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        action_low, action_high = action_bounds.T
        self.action_space = gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)

        self.target_local_x_velocity = 1.0
        self.target_local_y_velocity = 0.0


    def reset(self, seed=None):
        self.episode_step = 0
        
        qpos = np.zeros(self.model.nq)
        qpos[2] = 0.75  # z position
        rotation_angles = np.array([0.0, 0.0, 0.0])  # yaw, pitch, roll
        rotation_quaternion = Rotation.from_euler("xyz", rotation_angles).as_quat()
        qpos[3:7] = rotation_quaternion

        qvel = np.zeros(self.model.nv)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

        if self.viewer:
            self.viewer.render(self.data)

        return self.get_observation(), {}


    def step(self, action):
        for _ in range(self.nr_intermediate_steps):
            self.data.ctrl = action
            mujoco.mj_step(self.model, self.data, self.nr_substeps)

        if self.viewer:
            self.viewer.render(self.data)
        
        self.episode_step += 1

        next_state = self.get_observation()
        reward, r_info = self.get_reward()
        terminated = False
        truncated = self.episode_step >= self.horizon
        info = {**r_info}

        return next_state, reward, terminated, truncated, info


    def get_observation(self):
        joint_positions = self.data.qpos[7:]
        joint_velocities = self.data.qvel[6:]
        local_angular_velocities = self.data.qvel[3:6]
        projected_gravity_vector = np.matmul(self.data.body("torso").xmat.reshape(3, 3).T, np.array([0.0, 0.0, -1.0]))
        observation = np.concatenate([joint_positions, joint_velocities, local_angular_velocities, projected_gravity_vector])
        return observation


    def get_reward(self):
        rotation_quaternion = self.data.qpos[3:7]
        yaw_angle = Rotation.from_quat(rotation_quaternion).as_euler("xyz")[0]
        target_global_x_velocity = self.target_local_x_velocity * np.cos(yaw_angle) - self.target_local_y_velocity * np.sin(yaw_angle)
        target_global_y_velocity = self.target_local_x_velocity * np.sin(yaw_angle) + self.target_local_y_velocity * np.cos(yaw_angle)
        target_global_xy_velocity = np.array([target_global_x_velocity, target_global_y_velocity])
        current_global_xy_velocity = self.data.qvel[:2]
        xy_velocity_difference_norm = np.sum(np.square(target_global_xy_velocity - current_global_xy_velocity))
        tracking_xy_velocity_command_reward = np.exp(-xy_velocity_difference_norm / 0.25)

        reward = tracking_xy_velocity_command_reward

        info = {
            "reward/track_xy_vel_cmd": tracking_xy_velocity_command_reward,
            "env_info/xy_vel_diff_norm": xy_velocity_difference_norm,
        }

        return reward, info


    def close(self):
        if self.viewer:
            self.viewer.close()
