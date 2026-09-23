from copy import deepcopy
from pathlib import Path
import gymnasium as gym
import mujoco
from dm_control import mjcf
import numpy as np
from scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.motion_library import MotionLibrary
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.control_modules.handler import get_control_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.reward_modules.handler import get_reward_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.termination_modules.handler import get_termination_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.initial_state_modules.handler import get_initial_state_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.mujoco_model_modules.handler import get_mujoco_model_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.actuator_modules.handler import get_actuator_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.action_delay_modules.handler import get_action_delay_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.perturbation_modules.handler import get_perturbation_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.domain_randomization.observation_noise_modules.handler import get_observation_noise_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mujoco.sampling_modules.handler import get_sampling_module


class MotionTrackingEnv(gym.Env):
    def __init__(self, robot_config, runner_mode, seed, render, env_config, nr_envs, eval_mode=False):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs

        self.np_rng = np.random.default_rng(seed)

        self.body_angular_velocity_slice = slice(None, 3)
        self.body_linear_velocity_slice = slice(3, None)
        self.quaternion_wxyz_to_xyzw_indices = [1, 2, 3, 0]
        self.quaternion_xyzw_to_wxyz_indices = [3, 0, 1, 2]

        xml_path = (self.robot_config["directory_path"] / "data" / "plane.xml").as_posix()
        xml_handle = mjcf.from_path(xml_path)
        self.has_object = env_config["dataset"] == "omomo"
        if self.has_object:
            object_body = xml_handle.worldbody.add("body", name="tracked_object", pos=env_config["object"]["pos"], quat=env_config["object"]["quat"])
            object_body.add("freejoint", name="object_freejoint")
            object_geom = object_body.add(
                "geom",
                name="object_collision",
                type=env_config["object"]["type"],
                mass=env_config["object"]["mass"],
                contype=env_config["object"]["contype"],
                conaffinity=env_config["object"]["conaffinity"],
                friction=env_config["object"]["friction"],
                rgba=env_config["object"]["rgba"],
                solref=env_config["object"]["solref"],
            )
            if env_config["object"]["type"] == "mesh":
                mesh_path = Path(env_config["object"]["mesh_path"]).expanduser().resolve(strict=True)
                object_geom.mesh = xml_handle.asset.add("mesh", name="tracking_object_mesh", file=str(mesh_path))
            else:
                object_geom.size = env_config["object"]["half_size"][:2] if env_config["object"]["type"] == "cylinder" else env_config["object"]["half_size"]
                object_geom.pos = env_config["object"]["center"]

            for keyframe in xml_handle.find_all("key"):
                keyframe.qpos = np.concatenate((keyframe.qpos, env_config["object"]["pos"], env_config["object"]["quat"]))

        self.initial_mj_model = mujoco.MjModel.from_xml_string(xml=xml_handle.to_xml_string(), assets=xml_handle.get_assets())
        self.dt = self.initial_mj_model.opt.timestep * env_config["control_decimation"]
        self.horizon = max(1, round(env_config["episode_length_seconds"] / self.dt))
        self.trunk_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, robot_config["base_body_name"])
        self.base_joint_id = self.initial_mj_model.body_jntadr[self.trunk_body_id]
        self.has_free_base = self.base_joint_id >= 0 and self.initial_mj_model.jnt_type[self.base_joint_id] == mujoco.mjtJoint.mjJNT_FREE
        self.base_qpos_dim = 7 if self.has_free_base else 0
        self.base_qvel_dim = 6 if self.has_free_base else 0
        self.base_qpos_adr = self.initial_mj_model.jnt_qposadr[self.base_joint_id]
        base_qvel_adr = self.initial_mj_model.jnt_dofadr[self.base_joint_id]
        self.root_linear_qvel_slice = slice(base_qvel_adr, base_qvel_adr + 3)
        self.root_angular_qvel_slice = slice(base_qvel_adr + 3, base_qvel_adr + self.base_qvel_dim)
        self.root_position_qpos_slice = slice(self.base_qpos_adr, self.base_qpos_adr + 3)
        self.root_quaternion_qpos_slice = slice(self.base_qpos_adr + 3, self.base_qpos_adr + self.base_qpos_dim)

        actuator_joint_ids = self.initial_mj_model.actuator_trnid[:, 0]
        self.actuator_joint_names = [mujoco.mj_id2name(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in actuator_joint_ids]
        self.actuator_joint_mask_joints = actuator_joint_ids
        self.actuator_joint_mask_qpos = self.initial_mj_model.jnt_qposadr[actuator_joint_ids]
        self.actuator_joint_mask_qvel = self.initial_mj_model.jnt_dofadr[actuator_joint_ids]
        self.nr_actuators = self.initial_mj_model.nu
        self.nr_actuator_joints = len(actuator_joint_ids)

        if self.has_object:
            self.object_joint_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
            object_qpos_adr = self.initial_mj_model.jnt_qposadr[self.object_joint_id]
            object_qvel_adr = self.initial_mj_model.jnt_dofadr[self.object_joint_id]
            self.object_position_qpos_slice = slice(object_qpos_adr, object_qpos_adr + 3)
            self.object_quaternion_qpos_slice = slice(object_qpos_adr + 3, object_qpos_adr + self.base_qpos_dim)
            self.object_qvel_slice = slice(object_qvel_adr, object_qvel_adr + self.base_qvel_dim)
            self.object_linear_qvel_slice = slice(object_qvel_adr, object_qvel_adr + 3)
            self.object_angular_qvel_slice = slice(object_qvel_adr + 3, object_qvel_adr + self.base_qvel_dim)
            self.object_body_id = self.initial_mj_model.jnt_bodyid[self.object_joint_id]
            self.object_geom_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_GEOM, "object_collision")

        self.internal_state = {
            "mj_model": deepcopy(self.initial_mj_model),
            "data": mujoco.MjData(self.initial_mj_model),
            "in_eval_mode": eval_mode,
            "info_episode_store": {"episode_step": 0},
        }

        self.motion_library = MotionLibrary(env_config, self.internal_state["mj_model"])
        self.body_ids = self.motion_library.body_ids
        self.body_roots = self.initial_mj_model.body_rootid[self.motion_library.body_ids]
        self.anchor_body_index = self.motion_library.anchor_index
        self.actuator_joint_nominal_positions = np.asarray(self.initial_mj_model.keyframe("home").qpos[self.actuator_joint_mask_qpos], dtype=np.float32)
        joint_range = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints]
        self.lower = joint_range[:, 0]
        self.upper = joint_range[:, 1]
        self.internal_state["last_action"] = np.zeros(self.nr_actuators, dtype=np.float32)

        self.control_module = get_control_module(env_config["control"]["module"], self)
        self.torso_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, robot_config["torso_body_name"])
        self.robot_geom_ids = np.flatnonzero(self.initial_mj_model.body_rootid[self.initial_mj_model.geom_bodyid] == self.trunk_body_id)

        self.reward_module = get_reward_module(env_config["reward"]["module"], self)
        self.reward_module.init()
        self.termination_module = get_termination_module(env_config["termination"]["module"], self)

        self.action_space = gym.spaces.Box(-env_config["action_clip"], env_config["action_clip"], (self.nr_actuators,), np.float32)
        self.observation_space = self.get_observation_space()

        self.viewer = None
        self.renderer = None

        self.initial_state_module = get_initial_state_module(env_config["domain_randomization"]["initial_state"]["module"], self)
        self.mujoco_model_module = get_mujoco_model_module(env_config["domain_randomization"]["mujoco_model"]["module"], self)
        self.actuator_module = get_actuator_module(env_config["domain_randomization"]["actuator"]["module"], self)
        self.action_delay_module = get_action_delay_module(env_config["domain_randomization"]["action_delay"]["module"], self)
        self.perturbation_module = get_perturbation_module(env_config["domain_randomization"]["perturbation"]["module"], self)
        self.observation_noise_module = get_observation_noise_module(env_config["domain_randomization"]["observation_noise"]["module"], self)

        self.mujoco_model_sampling_module = get_sampling_module(env_config["domain_randomization"]["mujoco_model"]["sampling"]["module"], self, **env_config["domain_randomization"]["mujoco_model"]["sampling"].get(env_config["domain_randomization"]["mujoco_model"]["sampling"]["module"], {}))
        self.actuator_bias_sampling_module = get_sampling_module(env_config["domain_randomization"]["actuator"]["bias_sampling"]["module"], self, **env_config["domain_randomization"]["actuator"]["bias_sampling"].get(env_config["domain_randomization"]["actuator"]["bias_sampling"]["module"], {}))
        self.actuator_gain_sampling_module = get_sampling_module(env_config["domain_randomization"]["actuator"]["gain_sampling"]["module"], self, **env_config["domain_randomization"]["actuator"]["gain_sampling"].get(env_config["domain_randomization"]["actuator"]["gain_sampling"]["module"], {}))
        self.action_delay_sampling_module = get_sampling_module(env_config["domain_randomization"]["action_delay"]["sampling"]["module"], self, **env_config["domain_randomization"]["action_delay"]["sampling"].get(env_config["domain_randomization"]["action_delay"]["sampling"]["module"], {}))
        self.perturbation_sampling_module = get_sampling_module(env_config["domain_randomization"]["perturbation"]["sampling"]["module"], self, **env_config["domain_randomization"]["perturbation"]["sampling"].get(env_config["domain_randomization"]["perturbation"]["sampling"]["module"], {}))
        self.nr_sampling_bins = int(self.motion_library.total_frames * self.dt) + 1
        self.internal_state["failures"] = np.zeros(self.nr_sampling_bins, dtype=np.float32)
        self.actuator_module.init()
        self.action_delay_module.init()
        self.handle_domain_randomization(is_initial=True)


    def sample_starts(self):
        probabilities = self.internal_state["failures"] + self.env_config["adaptive_uniform_ratio"] / len(self.internal_state["failures"])
        if not self.env_config["adaptive_sampling"] or self.internal_state["in_eval_mode"]:
            probabilities = np.ones_like(probabilities)
        probabilities = probabilities.astype(np.float64) / probabilities.sum(dtype=np.float64)
        bin_id = self.np_rng.choice(len(self.internal_state["failures"]), p=probabilities)
        frame = min(int((bin_id + self.np_rng.random()) / len(self.internal_state["failures"]) * self.motion_library.total_frames), self.motion_library.total_frames - 1)
        clip = np.searchsorted(self.motion_library.starts + self.motion_library.lengths, frame, side="right")
        time = min((frame - self.motion_library.starts[clip]) / self.motion_library.fps[clip], max(0.0, self.motion_library.durations[clip] - self.dt))
        if self.internal_state["in_eval_mode"] or not self.env_config["random_start"]:
            time = 0.0
        return clip, time


    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_rng = np.random.default_rng(seed)

        self.internal_state["clip"], self.internal_state["time"] = self.sample_starts()
        if options:
            self.internal_state["clip"] = options.get("clip", self.internal_state["clip"])
            self.internal_state["time"] = options.get("time", self.internal_state["time"])

        reference = self.motion_library.sample(self.internal_state["clip"], np.asarray(self.internal_state["time"]))
        qpos, qvel = self.initial_state_module.sample(reference["qpos"], reference["qvel"])
        mujoco.mj_resetData(self.internal_state["mj_model"], self.internal_state["data"])
        self.internal_state["data"].qpos[:], self.internal_state["data"].qvel[:] = qpos, qvel
        self.internal_state["data"].ctrl[:] = 0.0
        self.internal_state["last_action"] = np.zeros(self.nr_actuators, dtype=np.float32)
        self.internal_state["info_episode_store"]["episode_step"] = 0
        self.action_delay_module.setup()
        self.handle_domain_randomization(is_episode_start=True)
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        self.reward_module.setup()

        self.internal_state.update(self.get_tracking_state(self.internal_state["data"], reference))
        observation = self.get_observation(self.internal_state["last_action"])
        self.reward_module.reward_and_info(self.internal_state["last_action"])
        self.termination_module.should_terminate()

        return observation, self.internal_state["info"]


    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        delayed_action = self.action_delay_module.delay_action(action)
        target_joint_positions = self.control_module.process_action(delayed_action)
        for _ in range(self.env_config["control_decimation"]):
            self.internal_state["data"].ctrl[:] = self.control_module.control(target_joint_positions)
            mujoco.mj_step(self.internal_state["mj_model"], self.internal_state["data"])
            self.reward_module.step()
        mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        self.internal_state["info_episode_store"]["episode_step"] += 1

        reference = self.motion_library.sample(self.internal_state["clip"], np.asarray(self.internal_state["time"]))
        self.internal_state.update(self.get_tracking_state(self.internal_state["data"], reference))
        info = {}
        self.internal_state["info"] = info
        observation = self.get_observation(action, add_noise=False)
        reward = self.reward_module.reward_and_info(action)
        terminated = self.termination_module.should_terminate()
        info["reward/total"] = reward
        truncated = self.internal_state["info_episode_store"]["episode_step"] >= self.horizon
        self.internal_state["last_action"] = action

        current_frame = self.motion_library.starts[self.internal_state["clip"]] + int(self.internal_state["time"] * self.motion_library.fps[self.internal_state["clip"]])
        current_bin = min(int(current_frame * len(self.internal_state["failures"]) / self.motion_library.total_frames), len(self.internal_state["failures"]) - 1)
        self.internal_state["time"] += self.dt
        resample = self.internal_state["time"] >= self.motion_library.durations[self.internal_state["clip"]] and not (terminated or truncated)
        if resample:
            self.internal_state["clip"], self.internal_state["time"] = self.sample_starts()
            reference = self.motion_library.sample(self.internal_state["clip"], np.asarray(self.internal_state["time"]))
            self.internal_state["data"].qpos[:], self.internal_state["data"].qvel[:] = self.initial_state_module.sample(reference["qpos"], reference["qvel"])
            self.internal_state["data"].qacc_warmstart[:] = 0.0

        if self.handle_domain_randomization(done=terminated or truncated) or resample:
            mujoco.mj_forward(self.internal_state["mj_model"], self.internal_state["data"])
        if not (terminated or truncated):
            reference = self.motion_library.sample(self.internal_state["clip"], np.asarray(self.internal_state["time"]))
            self.internal_state.update(self.get_tracking_state(self.internal_state["data"], reference))
            observation = self.get_observation(action)
        if not self.internal_state["in_eval_mode"]:
            self.internal_state["failures"] *= 1 - self.env_config["adaptive_alpha"]
            self.internal_state["failures"][current_bin] += self.env_config["adaptive_alpha"] * terminated

        if self.should_render:
            self.render()

        return observation, float(reward), bool(terminated), truncated, info


    def get_tracking_state(self, data, reference):
        body_positions = data.xpos[..., self.body_ids, :]
        body_quaternions = data.xquat[..., self.body_ids, :]
        valid_body_quaternions = np.all(np.isfinite(body_quaternions), axis=-1) & (np.linalg.norm(body_quaternions, axis=-1) > 0)
        body_quaternions = np.where(valid_body_quaternions[..., None], body_quaternions, [1.0, 0.0, 0.0, 0.0])
        body_com_velocities = data.cvel[..., self.body_ids, :]
        body_velocities = np.concatenate((body_com_velocities[..., self.body_angular_velocity_slice], body_com_velocities[..., self.body_linear_velocity_slice] + np.cross(body_com_velocities[..., self.body_angular_velocity_slice], body_positions - data.subtree_com[..., self.body_roots, :])), axis=-1)
        body_rotation = Rotation.from_quat(body_quaternions[..., self.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))

        anchor_position = body_positions[..., self.anchor_body_index, :]
        anchor_rotation = Rotation.from_quat(body_quaternions[..., self.anchor_body_index, :][..., self.quaternion_wxyz_to_xyzw_indices])
        reference_anchor_position = reference["body_positions"][..., self.anchor_body_index, :]
        reference_anchor_rotation = Rotation.from_quat(reference["body_quaternions"][..., self.anchor_body_index, :][..., self.quaternion_wxyz_to_xyzw_indices])
        anchor_rotation_inverse = anchor_rotation.inv()
        body_anchor_rotation_inverse = Rotation.from_quat(np.repeat(anchor_rotation_inverse.as_quat().reshape(-1, 4), len(self.body_ids), axis=0))

        relative_rotation = (anchor_rotation * reference_anchor_rotation.inv()).as_matrix()
        yaw = np.arctan2(relative_rotation[..., 1, 0], relative_rotation[..., 0, 0])
        yaw_rotation = Rotation.from_rotvec(np.stack((np.zeros_like(yaw), np.zeros_like(yaw), yaw), axis=-1))
        body_yaw_rotation = Rotation.from_quat(np.repeat(yaw_rotation.as_quat().reshape(-1, 4), len(self.body_ids), axis=0))
        reference_body_rotation = Rotation.from_quat(reference["body_quaternions"][..., self.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
        target_origin = np.concatenate((anchor_position[..., :2], reference_anchor_position[..., 2:3]), axis=-1)
        aligned_body_positions = target_origin[..., None, :] + body_yaw_rotation.apply((reference["body_positions"] - reference_anchor_position[..., None, :]).reshape(-1, 3)).reshape(body_positions.shape)

        internal_state = {
            "reference": {
                **reference,
                "anchor_position": reference_anchor_position,
                "anchor_rotation": reference_anchor_rotation,
                "aligned_body_positions": aligned_body_positions,
                "aligned_body_rotation": body_yaw_rotation * reference_body_rotation,
            },
            "info": {},
            "body_positions": body_positions,
            "body_rotation": body_rotation,
            "body_velocities": body_velocities,
            "anchor_position": anchor_position,
            "anchor_rotation": anchor_rotation,
            "anchor_rotation_inverse": anchor_rotation_inverse,
            "body_anchor_rotation_inverse": body_anchor_rotation_inverse,
            "projected_gravity": anchor_rotation_inverse.apply([0.0, 0.0, -1.0]),
        }
        if self.motion_library.has_object:
            object_quat = data.qpos[..., self.object_quaternion_qpos_slice][..., self.quaternion_wxyz_to_xyzw_indices]
            valid_object_quat = np.all(np.isfinite(object_quat), axis=-1) & (np.linalg.norm(object_quat, axis=-1) > 0)
            object_quat = np.where(valid_object_quat[..., None], object_quat, [0.0, 0.0, 0.0, 1.0])
            internal_state["object_rotation"] = Rotation.from_quat(object_quat)
            internal_state["reference"]["object_rotation"] = Rotation.from_quat(reference["qpos"][..., self.object_quaternion_qpos_slice][..., self.quaternion_wxyz_to_xyzw_indices])

        return internal_state


    def get_observation(self, action, add_noise=True):
        root_quat = self.internal_state["data"].qpos[..., self.root_quaternion_qpos_slice]
        valid_root = np.all(np.isfinite(root_quat), axis=-1) & (np.linalg.norm(root_quat, axis=-1) > 0)
        root_quat = np.where(valid_root[..., None], root_quat, [1.0, 0.0, 0.0, 0.0])
        root_inverse = Rotation.from_quat(root_quat[..., self.quaternion_wxyz_to_xyzw_indices]).inv()

        policy_previous_actions = action
        policy_base_angular_velocity = self.internal_state["data"].qvel[..., self.root_angular_qvel_slice]
        policy_joint_velocities = self.internal_state["data"].qvel[..., self.actuator_joint_mask_qvel]
        policy_reference_joint_positions = self.internal_state["reference"]["qpos"][..., self.actuator_joint_mask_qpos]
        policy_reference_joint_velocities = self.internal_state["reference"]["qvel"][..., self.actuator_joint_mask_qvel]

        critic_reference_anchor_position = self.internal_state["anchor_rotation_inverse"].apply(self.internal_state["reference"]["anchor_position"] - self.internal_state["anchor_position"])
        reference_anchor_matrix = (self.internal_state["anchor_rotation_inverse"] * self.internal_state["reference"]["anchor_rotation"]).as_matrix()
        policy_reference_anchor_orientation = reference_anchor_matrix[..., :2].reshape(self.internal_state["data"].qpos.shape[:-1] + (6,))
        critic_base_linear_velocity = root_inverse.apply(self.internal_state["data"].qvel[..., self.root_linear_qvel_slice])
        policy_joint_positions = self.internal_state["data"].qpos[..., self.actuator_joint_mask_qpos] - self.actuator_joint_nominal_positions

        critic_object_position = np.empty(self.internal_state["data"].qpos.shape[:-1] + (0,))
        critic_object_orientation = np.empty(self.internal_state["data"].qpos.shape[:-1] + (0,))
        critic_object_velocity = np.empty(self.internal_state["data"].qpos.shape[:-1] + (0,))
        if self.motion_library.has_object:
            critic_object_position = self.internal_state["anchor_rotation_inverse"].apply(self.internal_state["data"].qpos[..., self.object_position_qpos_slice] - self.internal_state["anchor_position"])
            object_matrix = (self.internal_state["anchor_rotation_inverse"] * self.internal_state["object_rotation"]).as_matrix()
            critic_object_orientation = object_matrix[..., :2].reshape(self.internal_state["data"].qpos.shape[:-1] + (6,))
            # Match Holosoma's obj_lin_vel_b frame-transform convention.
            critic_object_velocity = self.internal_state["anchor_rotation_inverse"].apply(self.internal_state["data"].qvel[..., self.object_linear_qvel_slice] - self.internal_state["anchor_position"])

        relative_pos = self.internal_state["body_anchor_rotation_inverse"].apply((self.internal_state["body_positions"] - self.internal_state["anchor_position"][..., None, :]).reshape(-1, 3))
        relative_matrix = (self.internal_state["body_anchor_rotation_inverse"] * self.internal_state["body_rotation"]).as_matrix()
        critic_body_positions = relative_pos.reshape(self.internal_state["data"].qpos.shape[:-1] + (-1,))
        critic_body_orientations = relative_matrix[..., :2].reshape(self.internal_state["data"].qpos.shape[:-1] + (-1,))

        critic_previous_actions = action
        critic_base_angular_velocity = policy_base_angular_velocity
        critic_joint_positions = policy_joint_positions
        critic_joint_velocities = policy_joint_velocities
        critic_reference_joint_positions = policy_reference_joint_positions
        critic_reference_joint_velocities = policy_reference_joint_velocities
        critic_reference_anchor_orientation = policy_reference_anchor_orientation

        observation = np.concatenate([
            policy_previous_actions,
            policy_base_angular_velocity,
            policy_joint_positions,
            policy_joint_velocities,
            policy_reference_joint_positions,
            policy_reference_joint_velocities,
            policy_reference_anchor_orientation,
            critic_previous_actions,
            critic_base_angular_velocity,
            critic_base_linear_velocity,
            critic_joint_positions,
            critic_joint_velocities,
            critic_reference_joint_positions,
            critic_reference_joint_velocities,
            critic_reference_anchor_orientation,
            critic_reference_anchor_position,
            critic_object_velocity,
            critic_object_orientation,
            critic_object_position,
            critic_body_orientations,
            critic_body_positions,
        ], axis=-1).astype(np.float32)

        if add_noise:
            observation = self.observation_noise_module.apply(observation)

        observation[..., self.joint_positions_normalization_idx] = observation[..., self.joint_positions_normalization_idx] / self.env_config["observation"]["joint_position_range"]
        observation[..., self.reference_joint_positions_normalization_idx] = (observation[..., self.reference_joint_positions_normalization_idx] - self.actuator_joint_nominal_positions) / self.env_config["observation"]["joint_position_range"]
        observation[..., self.joint_velocities_normalization_idx] = observation[..., self.joint_velocities_normalization_idx] / self.env_config["observation"]["joint_velocity_range"]
        observation[..., self.previous_actions_normalization_idx] = observation[..., self.previous_actions_normalization_idx] / self.env_config["observation"]["action_range"]
        observation[..., self.angular_velocities_normalization_idx] = np.clip(observation[..., self.angular_velocities_normalization_idx] / self.env_config["observation"]["angular_velocity_range"], -1.0, 1.0)
        observation[..., self.linear_velocities_normalization_idx] = np.clip(observation[..., self.linear_velocities_normalization_idx] / self.env_config["observation"]["linear_velocity_range"], -1.0, 1.0)
        observation[..., self.relative_positions_normalization_idx] = observation[..., self.relative_positions_normalization_idx] / self.env_config["observation"]["relative_position_range"]
        observation[..., self.orientations_normalization_idx] = np.clip(observation[..., self.orientations_normalization_idx], -1.0, 1.0)

        observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = np.clip(observation, -self.env_config["observation"]["clip"], self.env_config["observation"]["clip"])

        return observation


    def handle_domain_randomization(self, is_episode_start=False, is_initial=False, done=False):
        should_randomize_model = self.mujoco_model_sampling_module.setup(is_initial) if is_initial or is_episode_start else self.mujoco_model_sampling_module.step()
        should_randomize_bias = self.actuator_bias_sampling_module.setup(is_initial) if is_initial or is_episode_start else self.actuator_bias_sampling_module.step()

        if should_randomize_model:
            self.mujoco_model_module.sample()
        if should_randomize_bias:
            self.actuator_module.sample_bias()

        should_randomize_gains = self.actuator_gain_sampling_module.setup(is_initial) if is_initial or is_episode_start else self.actuator_gain_sampling_module.step()
        should_randomize_delay = self.action_delay_sampling_module.setup(is_initial) if is_initial or is_episode_start else self.action_delay_sampling_module.step()
        if should_randomize_gains:
            self.actuator_module.sample_gains()
        if should_randomize_delay:
            self.action_delay_module.sample()

        if is_initial:
            return

        should_perturb = self.perturbation_sampling_module.setup() if is_episode_start else self.perturbation_sampling_module.step()
        self.internal_state["data"].qvel[:] = self.perturbation_module.apply(self.internal_state["data"].qvel, self.internal_state["data"].qpos, should_perturb and not done)

        return should_randomize_model or (should_perturb and not done)


    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_previous_actions_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuators)
        current_observation_idx += self.nr_actuators
        self.base_angular_velocity_obs_idx = np.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.joint_positions_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.reference_joint_positions_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.reference_joint_velocities_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.reference_anchor_orientation_obs_idx = np.arange(current_observation_idx, current_observation_idx + 6)
        current_observation_idx += 6

        self.actor_observation_size = current_observation_idx
        self.policy_observation_indices = np.arange(current_observation_idx)

        self.critic_joint_previous_actions_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuators)
        current_observation_idx += self.nr_actuators
        self.critic_base_angular_velocity_obs_idx = np.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.critic_base_linear_velocity_obs_idx = np.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.critic_joint_positions_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_joint_velocities_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_reference_joint_positions_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_reference_joint_velocities_obs_idx = np.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_reference_anchor_orientation_obs_idx = np.arange(current_observation_idx, current_observation_idx + 6)
        current_observation_idx += 6
        self.critic_reference_anchor_position_obs_idx = np.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.critic_object_velocity_obs_idx = np.arange(current_observation_idx, current_observation_idx + (3 if self.has_object else 0))
        current_observation_idx += (3 if self.has_object else 0)
        self.critic_object_orientation_obs_idx = np.arange(current_observation_idx, current_observation_idx + (6 if self.has_object else 0))
        current_observation_idx += (6 if self.has_object else 0)
        self.critic_object_position_obs_idx = np.arange(current_observation_idx, current_observation_idx + (3 if self.has_object else 0))
        current_observation_idx += (3 if self.has_object else 0)
        self.critic_body_orientations_obs_idx = np.arange(current_observation_idx, current_observation_idx + 6 * len(self.body_ids))
        current_observation_idx += 6 * len(self.body_ids)
        self.critic_body_positions_obs_idx = np.arange(current_observation_idx, current_observation_idx + 3 * len(self.body_ids))
        current_observation_idx += 3 * len(self.body_ids)

        self.critic_observation_indices = np.arange(self.actor_observation_size, current_observation_idx)

        self.joint_positions_normalization_idx = np.concatenate((self.joint_positions_obs_idx, self.critic_joint_positions_obs_idx))
        self.reference_joint_positions_normalization_idx = np.stack((self.reference_joint_positions_obs_idx, self.critic_reference_joint_positions_obs_idx))
        self.joint_velocities_normalization_idx = np.concatenate((self.joint_velocities_obs_idx, self.critic_joint_velocities_obs_idx, self.reference_joint_velocities_obs_idx, self.critic_reference_joint_velocities_obs_idx))
        self.previous_actions_normalization_idx = np.concatenate((self.joint_previous_actions_obs_idx, self.critic_joint_previous_actions_obs_idx))
        self.angular_velocities_normalization_idx = np.concatenate((self.base_angular_velocity_obs_idx, self.critic_base_angular_velocity_obs_idx))
        self.linear_velocities_normalization_idx = np.concatenate((self.critic_base_linear_velocity_obs_idx, self.critic_object_velocity_obs_idx))
        self.relative_positions_normalization_idx = np.concatenate((self.critic_reference_anchor_position_obs_idx, self.critic_object_position_obs_idx, self.critic_body_positions_obs_idx))
        self.orientations_normalization_idx = np.concatenate((self.reference_anchor_orientation_obs_idx, self.critic_reference_anchor_orientation_obs_idx, self.critic_object_orientation_obs_idx, self.critic_body_orientations_obs_idx))

        return gym.spaces.Box(-self.env_config["observation"]["clip"], self.env_config["observation"]["clip"], (current_observation_idx,), np.float32)


    def render(self):
        if self.viewer is None:
            self.viewer = MujocoViewer(self.internal_state["mj_model"], self.dt)
        self.viewer.render(self.internal_state["data"], self.internal_state["reference"]["qpos"])


    def render_rgb(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.internal_state["mj_model"], height=480, width=640)
        self.renderer.update_scene(self.internal_state["data"], camera="track")
        return self.renderer.render()


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            self.renderer.close()
