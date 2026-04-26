from __future__ import annotations
import gc
from copy import deepcopy
from pathlib import Path
import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjwarp

from rl_x.environments.custom_mujoco.ant.warp_torch.box_space import BoxSpace
from rl_x.environments.custom_mujoco.ant.warp_torch.viewer import MujocoViewer


class Ant:
    def __init__(self, env_config):
        self.should_render = env_config.render
        self.horizon = env_config.horizon
        self.action_scaling_factor = env_config.action_scaling_factor
        self.nr_envs = env_config.nr_envs

        torch_device = "cuda" if env_config.device == "gpu" and torch.cuda.is_available() else "cpu"
        self.device = torch.device(torch_device)
        self.wp_device = wp.get_device("cuda" if torch_device == "cuda" else "cpu")

        xml_path = (Path(__file__).resolve().parent.parent / "data" / "ant.xml").as_posix()
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        with wp.ScopedDevice(self.wp_device):
            self.wp_model = mjwarp.put_model(self.mj_model)
            self.wp_data = mjwarp.put_data(self.mj_model, self.mj_data, nworld=self.nr_envs)

        self.qpos = wp.to_torch(self.wp_data.qpos)
        self.qvel = wp.to_torch(self.wp_data.qvel)
        self.ctrl = wp.to_torch(self.wp_data.ctrl)

        self.nr_intermediate_steps = 4

        kf_qpos = np.asarray(self.mj_model.keyframe("home").qpos, dtype=np.float32)
        kf_qvel = np.asarray(self.mj_model.keyframe("home").qvel, dtype=np.float32)
        self.initial_qpos = torch.from_numpy(kf_qpos).to(self.device)
        self.initial_qvel = torch.from_numpy(kf_qvel).to(self.device)

        self.target_local_xy_velocity = torch.tensor([2.0, 0.0], dtype=torch.float32, device=self.device)
        self.gravity_world = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=self.device)

        action_space_size = self.mj_model.nu
        lower_joint_limit, upper_joint_limit = self.mj_model.jnt_range.T
        self.nominal_joint_positions = self.initial_qpos[7:]
        low = torch.from_numpy(np.asarray(lower_joint_limit[1:], dtype=np.float32)).to(self.device)
        high = torch.from_numpy(np.asarray(upper_joint_limit[1:], dtype=np.float32)).to(self.device)
        self.single_action_space = BoxSpace(
            low=low, high=high, shape=(action_space_size,), dtype=torch.float32,
            center=self.nominal_joint_positions, scale=self.action_scaling_factor, device=self.device,
        )
        self.single_observation_space = BoxSpace(
            low=-float("inf"), high=float("inf"), shape=(34,), dtype=torch.float32, device=self.device,
        )

        self.episode_step = torch.zeros(self.nr_envs, dtype=torch.int32, device=self.device)
        self.episode_return = torch.zeros(self.nr_envs, dtype=torch.float32, device=self.device)
        self.last_episode_return = torch.zeros(self.nr_envs, dtype=torch.float32, device=self.device)
        self.last_episode_length = torch.zeros(self.nr_envs, dtype=torch.float32, device=self.device)

        self.step_graph = None
        self.forward_graph = None

        self.reset_all_inplace()

        # Try to capture CUDA graphs
        # Requires a CUDA device otherwise we fall back to direct kernel launches
        self.use_cuda_graph = False
        if self.wp_device.is_cuda:
            try:
                driver_ver = wp.context.runtime.driver_version
                has_mempool = wp.is_mempool_enabled(self.wp_device)
                if driver_ver is not None and has_mempool and driver_ver >= (12, 4):
                    self.use_cuda_graph = True
            except Exception:
                pass
        if self.use_cuda_graph:
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                with wp.ScopedDevice(self.wp_device):
                    with wp.ScopedCapture() as cap:
                        for _ in range(self.nr_intermediate_steps):
                            mjwarp.step(self.wp_model, self.wp_data)
                    self.step_graph = cap.graph
                    with wp.ScopedCapture() as cap:
                        mjwarp.forward(self.wp_model, self.wp_data)
                    self.forward_graph = cap.graph
            finally:
                if gc_was_enabled:
                    gc.enable()

        self.viewer = None
        self.render_data = None
        if self.should_render:
            dt = self.mj_model.opt.timestep * self.nr_intermediate_steps
            self.viewer = MujocoViewer(self.mj_model, dt)
            c_model = deepcopy(self.mj_model)
            self.render_data = mujoco.MjData(c_model)
            mujoco.mj_step(c_model, self.render_data, 1)
            self.render_model = c_model


    def render(self) -> None:
        rd = self.render_data
        rd.qpos[:] = self.qpos[0].detach().cpu().numpy()
        rd.qvel[:] = self.qvel[0].detach().cpu().numpy()
        rd.ctrl[:] = self.ctrl[0].detach().cpu().numpy()
        mujoco.mj_forward(self.render_model, rd)
        self.viewer.render(rd)


    def reset_all_inplace(self) -> None:
        self.qpos[:] = self.initial_qpos
        self.qvel[:] = self.initial_qvel
        self.ctrl.zero_()
        self.episode_step.zero_()
        self.episode_return.zero_()
        self.last_episode_return.zero_()
        self.last_episode_length.zero_()

        with wp.ScopedDevice(self.wp_device):
            if self.forward_graph is not None:
                wp.capture_launch(self.forward_graph)
            else:
                mjwarp.forward(self.wp_model, self.wp_data)


    def reset(self, seed=None):
        self.reset_all_inplace()
        if self.viewer is not None:
            self.render()
        return self.get_observation(), {}


    def step(self, action: torch.Tensor):
        self.ctrl[:] = self.nominal_joint_positions + action * self.action_scaling_factor

        with wp.ScopedDevice(self.wp_device):
            if self.step_graph is not None:
                wp.capture_launch(self.step_graph)
            else:
                for _ in range(self.nr_intermediate_steps):
                    mjwarp.step(self.wp_model, self.wp_data)

        if self.viewer is not None:
            self.render()

        self.episode_step += 1

        next_state = self.get_observation()
        reward, info = self.get_reward()
        height = self.qpos[:, 2]
        terminated = (height < 0.2) | (height > 1.0)
        truncated = self.episode_step >= self.horizon
        done = terminated | truncated

        self.episode_return += reward
        self.last_episode_return = torch.where(done, self.episode_return, self.last_episode_return)
        self.last_episode_length = torch.where(done, self.episode_step.float(), self.last_episode_length)
        info["rollout/episode_return"] = self.last_episode_return
        info["rollout/episode_length"] = self.last_episode_length

        # Auto-reset done envs
        if bool(done.any()):
            done_idx = done.nonzero(as_tuple=True)[0]
            self.qpos[done_idx] = self.initial_qpos
            self.qvel[done_idx] = self.initial_qvel
            self.ctrl[done_idx] = 0.0
            self.episode_step[done_idx] = 0
            self.episode_return[done_idx] = 0.0
            with wp.ScopedDevice(self.wp_device):
                if self.forward_graph is not None:
                    wp.capture_launch(self.forward_graph)
                else:
                    mjwarp.forward(self.wp_model, self.wp_data)
            next_state_after = self.get_observation()
            next_state = torch.where(done.unsqueeze(-1), next_state_after, next_state)

        return next_state, reward, terminated, truncated, info
    

    def get_reward(self) -> tuple[torch.Tensor, dict]:
        base_quat = self.qpos[:, 3:7]
        current_global_linear_velocity = self.qvel[:, 0:3]
        current_local_linear_velocity = quat_rotate_inverse(base_quat, current_global_linear_velocity)
        xy_velocity_difference_norm = torch.sum(torch.square(self.target_local_xy_velocity - current_local_linear_velocity[:, :2]), dim=-1)
        tracking_xy_velocity_command_reward = torch.exp(-xy_velocity_difference_norm / 0.25)

        reward = tracking_xy_velocity_command_reward

        reward = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
        reward = torch.clamp(reward, -10.0, 10.0)

        info = {
            "reward_xy_vel_cmd": tracking_xy_velocity_command_reward,
            "xy_vel_diff_norm": xy_velocity_difference_norm,
        }

        return reward, info


    def get_observation(self) -> torch.Tensor:
        qpos = self.qpos
        qvel = self.qvel
        ctrl = self.ctrl

        global_height = qpos[:, 2:3]
        joint_positions = qpos[:, 7:] - self.nominal_joint_positions
        joint_velocities = qvel[:, 6:]
        local_angular_velocities = qvel[:, 3:6]

        base_quat = qpos[:, 3:7]  # MuJoCo: [w, x, y, z]
        global_linear_velocities = qvel[:, 0:3]
        local_linear_velocities = quat_rotate_inverse(base_quat, global_linear_velocities)
        gravity_world = self.gravity_world.expand_as(global_linear_velocities)
        projected_gravity_vector = quat_rotate_inverse(base_quat, gravity_world)

        observation = torch.cat([
            global_height,
            joint_positions, joint_velocities,
            local_linear_velocities, local_angular_velocities,
            projected_gravity_vector,
            ctrl,
        ], dim=-1)

        observation = torch.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = torch.clamp(observation, -100.0, 100.0)
        return observation


    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()



def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) `v` by the inverse of unit quaternion(s) `q`.

    Quaternion convention: [w, x, y, z] (MuJoCo).
    Shapes: q (..., 4), v (..., 3) -> (..., 3).
    """
    qw = q[..., 0:1]
    qvec = q[..., 1:4]
    a = v * (2.0 * qw * qw - 1.0)
    b = torch.cross(qvec, v, dim=-1) * (qw * 2.0)
    c = qvec * (qvec * v).sum(-1, keepdim=True) * 2.0
    return a - b + c
