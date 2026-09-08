from pathlib import Path
from functools import partial
import numpy as np
import mujoco
from mujoco import mjx
from warp import JaxCallableGraphMode
from dm_control import mjcf
from jax.scipy.spatial.transform import Rotation
import jax
import jax.numpy as jnp

from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.state import State
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.box_space import BoxSpace
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.motion_library import MotionLibrary
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.viewer import MujocoViewer
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.control_modules.handler import get_control_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.reward_modules.handler import get_reward_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.termination_modules.handler import get_termination_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.initial_state_modules.handler import get_initial_state_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.mujoco_model_modules.handler import get_mujoco_model_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.actuator_modules.handler import get_actuator_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.action_delay_modules.handler import get_action_delay_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.perturbation_modules.handler import get_perturbation_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.domain_randomization.observation_noise_modules.handler import get_observation_noise_module
from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.sampling_modules.handler import get_sampling_module


WARP_GRAPH_MODES = {
    "jax": JaxCallableGraphMode.JAX,
    "warp": JaxCallableGraphMode.WARP,
    "warp_staged": JaxCallableGraphMode.WARP_STAGED,
    "warp_staged_ex": JaxCallableGraphMode.WARP_STAGED_EX,
}


class MotionTrackingEnv:
    def __init__(self, robot_config, runner_mode, render, env_config, nr_envs):
        self.robot_config = robot_config
        self.runner_mode = runner_mode
        self.should_render = render
        self.env_config = env_config
        self.nr_envs = nr_envs

        self.device = jax.devices(env_config["device"])[0]
        self.graph_mode = "jax" if self.device.platform == "cpu" else env_config["graph_mode"]

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

        self.motion_library = MotionLibrary(env_config, self.initial_mj_model)
        self.body_ids = jnp.asarray(self.motion_library.body_ids)
        self.body_roots = jnp.asarray(self.initial_mj_model.body_rootid[self.motion_library.body_ids])
        self.anchor_body_index = self.motion_library.anchor_index
        self.actuator_joint_nominal_positions = jnp.asarray(self.initial_mj_model.keyframe("home").qpos[self.actuator_joint_mask_qpos], dtype=jnp.float32)
        joint_range = self.initial_mj_model.jnt_range[self.actuator_joint_mask_joints]
        self.lower = jnp.asarray(joint_range[:, 0])
        self.upper = jnp.asarray(joint_range[:, 1])

        self.control_module = get_control_module(env_config["control"]["module"], self)
        self.torso_body_id = mujoco.mj_name2id(self.initial_mj_model, mujoco.mjtObj.mjOBJ_BODY, robot_config["torso_body_name"])
        self.robot_geom_ids = np.flatnonzero(self.initial_mj_model.body_rootid[self.initial_mj_model.geom_bodyid] == self.trunk_body_id)

        self.reward_module = get_reward_module(env_config["reward"]["module"], self)
        self.termination_module = get_termination_module(env_config["termination"]["module"], self)

        self.initial_mjx_model = mjx.put_model(self.initial_mj_model, impl="warp", device=self.device, graph_mode=WARP_GRAPH_MODES[self.graph_mode])
        self.mjx_data = mjx.make_data(self.initial_mj_model, impl="warp", device=self.device, naconmax=self.nr_envs * env_config["naconmax_per_env"], njmax=env_config["njmax"])
        self.single_action_space = BoxSpace(np.full(self.nr_actuators, -env_config["action_clip"], np.float32), np.full(self.nr_actuators, env_config["action_clip"], np.float32), (self.nr_actuators,), np.float32)
        self.single_observation_space = self.get_observation_space()
        self.viewer = None

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


    def sample_starts(self, key, failures, eval_mode):
        key_bin, key_time, key_state = jax.random.split(key, 3)
        probabilities = failures + self.env_config["adaptive_uniform_ratio"] / len(failures)
        probabilities = jnp.where((not self.env_config["adaptive_sampling"]) | eval_mode, jnp.ones_like(failures), probabilities)
        probabilities = probabilities / jnp.sum(probabilities)
        bins = jax.random.choice(key_bin, len(failures), (self.nr_envs,), p=probabilities)
        frame = ((bins + jax.random.uniform(key_time, (self.nr_envs,))) / len(failures) * self.motion_library.total_frames).astype(jnp.int32)
        frame = jnp.minimum(frame, self.motion_library.total_frames - 1)
        clips = jnp.searchsorted(self.motion_library.starts + self.motion_library.lengths, frame, side="right")
        clips = jnp.where(eval_mode, jnp.arange(self.nr_envs) % len(self.motion_library.files), clips)
        time = jnp.minimum((frame - self.motion_library.starts[clips]) / self.motion_library.fps[clips], jnp.maximum(0.0, self.motion_library.durations[clips] - self.dt))
        time = jnp.where(eval_mode | (not self.env_config["random_start"]), 0.0, time)
        reference = self.motion_library.sample(clips, time)
        qpos, qvel = self.initial_state_module.sample(key_state, reference["qpos"], reference["qvel"], eval_mode)
        return clips, time, qpos, qvel


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, keys, eval_mode):
        key, sample_key, model_key, bias_key, gains_key, delay_key, _, noise_key = jax.random.split(keys[0], 8)
        failures = jnp.zeros(self.nr_sampling_bins, jnp.float32)
        clips, time, qpos, qvel = self.sample_starts(sample_key, failures, eval_mode)
        data = jax.vmap(lambda q, v: self.mjx_data.replace(qpos=q, qvel=v))(qpos, qvel)
        actions = jnp.zeros((self.nr_envs, self.nr_actuators), jnp.float32)
        internal_state = {"in_eval_mode": eval_mode}
        self.actuator_module.init(internal_state)
        self.action_delay_module.init(internal_state)
        self.reward_module.init(internal_state)
        randomization_keys = {"model": model_key, "bias": bias_key, "gains": gains_key, "delay": delay_key}
        data, model, internal_state = self.handle_domain_randomization(data, self.initial_mjx_model, internal_state, randomization_keys, eval_mode, is_initial=True)
        data = jax.vmap(mjx.forward, in_axes=(self.mujoco_model_module.model_axes, 0))(model, data)
        reference = self.motion_library.sample(clips, time)
        internal_state.update(self.get_tracking_state(data, reference))
        observation = self.get_observation(data, internal_state, actions, noise_key, eval_mode)
        internal_state["last_action"] = actions
        info = {}
        self.reward_module.reward_and_info(data, internal_state, actions, info)
        self.termination_module.should_terminate(data, internal_state, info)

        info.update({
            "rollout/episode_return": jnp.zeros(self.nr_envs, jnp.float32),
            "rollout/episode_length": jnp.zeros(self.nr_envs, jnp.float32),
        })
        internal_state.update({
            "clip": clips,
            "time": time,
            "failures": failures,
            "in_eval_mode": eval_mode,
            "eval_done": jnp.zeros(self.nr_envs, bool),
        })
        info_episode_store = {
            "episode_step": jnp.zeros(self.nr_envs, jnp.int32),
            "episode_return": jnp.zeros(self.nr_envs, jnp.float32),
        }

        state = State(
            mjx_model=model,
            data=data,
            next_observation=observation,
            actual_next_observation=observation,
            reward=jnp.zeros(self.nr_envs, jnp.float32),
            terminated=jnp.zeros(self.nr_envs, bool),
            truncated=jnp.zeros(self.nr_envs, bool),
            info=info,
            info_episode_store=info_episode_store,
            internal_state=internal_state,
            key=key,
        )

        return state


    @partial(jax.jit, static_argnums=(0,))
    def step(self, state, action):
        key, sample_key, noise_key, gains_key, delay_key, push_key, perturbation_sampling_key = jax.random.split(state.key, 7)
        internal_state = dict(state.internal_state)
        delayed_action = self.action_delay_module.delay_action(action, internal_state)
        target_joint_positions = self.control_module.process_action(delayed_action, internal_state)

        def physics(model, data, target_joint_positions, proportional_gain_factors, derivative_gain_factors, contact_forces):
            def substep(_, carry):
                data, forces = carry
                data = mjx.step(model, data.replace(ctrl=self.control_module.control(data, target_joint_positions, proportional_gain_factors, derivative_gain_factors)))
                reward_state = {"contact_forces": forces}
                self.reward_module.step(data, reward_state)
                return data, reward_state["contact_forces"]

            data, contact_forces = jax.lax.fori_loop(0, self.env_config["control_decimation"], substep, (data, contact_forces))
            return mjx.forward(model, data), contact_forces

        data, contact_forces = jax.vmap(physics, in_axes=(self.mujoco_model_module.model_axes, 0, 0, 0, 0, 0))(state.mjx_model, state.data, target_joint_positions, internal_state["proportional_gain_factors"], internal_state["derivative_gain_factors"], state.internal_state["contact_forces"])
        reference = self.motion_library.sample(state.internal_state["clip"], state.internal_state["time"])
        internal_state.update(self.get_tracking_state(data, reference))
        observation = self.get_observation(data, internal_state, action, noise_key, state.internal_state["in_eval_mode"])
        internal_state["contact_forces"] = contact_forces
        info = {}
        reward = self.reward_module.reward_and_info(data, internal_state, action, info)
        terminated = self.termination_module.should_terminate(data, internal_state, info)
        info["reward/total"] = reward
        observation = jnp.nan_to_num(observation)

        step = state.info_episode_store["episode_step"] + 1
        time = state.internal_state["time"] + self.dt
        truncated = step >= self.horizon
        done = terminated | truncated
        resample = done | (time >= self.motion_library.durations[state.internal_state["clip"]])
        returns = state.info_episode_store["episode_return"] + reward
        record_episode = done & (~state.internal_state["in_eval_mode"] | ~state.internal_state["eval_done"])
        info["rollout/episode_return"] = jnp.where(record_episode, returns, state.info["rollout/episode_return"])
        info["rollout/episode_length"] = jnp.where(record_episode, step, state.info["rollout/episode_length"])

        current_frames = self.motion_library.starts[state.internal_state["clip"]] + (state.internal_state["time"] * self.motion_library.fps[state.internal_state["clip"]]).astype(jnp.int32)
        current_bins = jnp.minimum((current_frames * len(state.internal_state["failures"]) / self.motion_library.total_frames).astype(jnp.int32), len(state.internal_state["failures"]) - 1)
        failed = jnp.bincount(current_bins, weights=terminated.astype(jnp.float32), length=len(state.internal_state["failures"]))
        clips, new_time, qpos, qvel = self.sample_starts(sample_key, state.internal_state["failures"], state.internal_state["in_eval_mode"])
        failures = jnp.where(state.internal_state["in_eval_mode"], state.internal_state["failures"], (1 - self.env_config["adaptive_alpha"]) * state.internal_state["failures"] + self.env_config["adaptive_alpha"] * failed)
        next_clip = jnp.where(resample, clips, state.internal_state["clip"])
        next_time = jnp.where(resample, new_time, time)

        next_data = data.replace(qpos=jnp.where(resample[:, None], qpos, data.qpos), qvel=jnp.where(resample[:, None], qvel, data.qvel),
                                 ctrl=jnp.where(done[:, None], 0.0, data.ctrl), qacc_warmstart=jnp.where(resample[:, None], 0.0, data.qacc_warmstart), time=jnp.where(done, 0.0, data.time))
        model_key, bias_key = jax.random.split(jax.random.fold_in(state.key, 1))
        randomization_keys = {"model": model_key, "bias": bias_key, "gains": gains_key, "delay": delay_key, "push": push_key, "perturbation_sampling": perturbation_sampling_key}
        next_data, next_model, internal_state = self.handle_domain_randomization(next_data, state.mjx_model, internal_state, randomization_keys, state.internal_state["in_eval_mode"], is_episode_start=done)
        next_data = jax.vmap(mjx.forward, in_axes=(self.mujoco_model_module.model_axes, 0))(next_model, next_data)
        next_action = jnp.where(done[:, None], 0.0, action)
        self.action_delay_module.setup(internal_state, done)
        next_reference = self.motion_library.sample(next_clip, next_time)
        internal_state.update(self.get_tracking_state(next_data, next_reference))
        next_observation = jnp.nan_to_num(self.get_observation(next_data, internal_state, next_action, noise_key, state.internal_state["in_eval_mode"]))

        internal_state.update({
            "clip": next_clip,
            "time": next_time,
            "failures": failures,
            "last_action": next_action,
            "eval_done": state.internal_state["eval_done"] | (state.internal_state["in_eval_mode"] & done),
        })
        info_episode_store = {
            "episode_step": jnp.where(done, 0, step),
            "episode_return": jnp.where(done, 0.0, returns),
        }
        self.reward_module.setup(internal_state, done)

        state = state.replace(
            mjx_model=next_model,
            data=next_data,
            next_observation=next_observation,
            actual_next_observation=jnp.where(done[:, None], observation, next_observation),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            info_episode_store=info_episode_store,
            internal_state=internal_state,
            key=key,
        )

        return state


    def get_tracking_state(self, data, reference):
        body_positions = data.xpos[..., self.body_ids, :]
        body_quaternions = data.xquat[..., self.body_ids, :]
        valid_body_quaternions = jnp.all(jnp.isfinite(body_quaternions), axis=-1) & (jnp.linalg.norm(body_quaternions, axis=-1) > 0)
        body_quaternions = jnp.where(valid_body_quaternions[..., None], body_quaternions, jnp.asarray([1.0, 0.0, 0.0, 0.0]))
        body_com_velocities = data.cvel[..., self.body_ids, :]
        body_velocities = jnp.concatenate((body_com_velocities[..., self.body_angular_velocity_slice], body_com_velocities[..., self.body_linear_velocity_slice] + jnp.cross(body_com_velocities[..., self.body_angular_velocity_slice], body_positions - data.subtree_com[..., self.body_roots, :])), axis=-1)
        body_rotation = Rotation.from_quat(body_quaternions[..., self.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))

        anchor_position = body_positions[..., self.anchor_body_index, :]
        anchor_rotation = Rotation.from_quat(body_quaternions[..., self.anchor_body_index, :][..., self.quaternion_wxyz_to_xyzw_indices])
        reference_anchor_position = reference["body_positions"][..., self.anchor_body_index, :]
        reference_anchor_rotation = Rotation.from_quat(reference["body_quaternions"][..., self.anchor_body_index, :][..., self.quaternion_wxyz_to_xyzw_indices])
        anchor_rotation_inverse = anchor_rotation.inv()
        body_anchor_rotation_inverse = Rotation.from_quat(jnp.repeat(anchor_rotation_inverse.as_quat().reshape(-1, 4), len(self.body_ids), axis=0))

        relative_rotation = (anchor_rotation * reference_anchor_rotation.inv()).as_matrix()
        yaw = jnp.arctan2(relative_rotation[..., 1, 0], relative_rotation[..., 0, 0])
        yaw_rotation = Rotation.from_rotvec(jnp.stack((jnp.zeros_like(yaw), jnp.zeros_like(yaw), yaw), axis=-1))
        body_yaw_rotation = Rotation.from_quat(jnp.repeat(yaw_rotation.as_quat().reshape(-1, 4), len(self.body_ids), axis=0))
        reference_body_rotation = Rotation.from_quat(reference["body_quaternions"][..., self.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
        target_origin = jnp.concatenate((anchor_position[..., :2], reference_anchor_position[..., 2:3]), axis=-1)
        aligned_body_positions = target_origin[..., None, :] + body_yaw_rotation.apply((reference["body_positions"] - reference_anchor_position[..., None, :]).reshape(-1, 3)).reshape(body_positions.shape)

        internal_state = {
            "reference": {
                **reference,
                "anchor_position": reference_anchor_position,
                "anchor_rotation": reference_anchor_rotation,
                "aligned_body_positions": aligned_body_positions,
                "aligned_body_rotation": body_yaw_rotation * reference_body_rotation,
            },
            "body_positions": body_positions,
            "body_rotation": body_rotation,
            "body_velocities": body_velocities,
            "anchor_position": anchor_position,
            "anchor_rotation": anchor_rotation,
            "anchor_rotation_inverse": anchor_rotation_inverse,
            "body_anchor_rotation_inverse": body_anchor_rotation_inverse,
            "projected_gravity": anchor_rotation_inverse.apply(jnp.asarray([0.0, 0.0, -1.0])),
        }
        if self.motion_library.has_object:
            object_quat = data.qpos[..., self.object_quaternion_qpos_slice][..., self.quaternion_wxyz_to_xyzw_indices]
            valid_object_quat = jnp.all(jnp.isfinite(object_quat), axis=-1) & (jnp.linalg.norm(object_quat, axis=-1) > 0)
            object_quat = jnp.where(valid_object_quat[..., None], object_quat, jnp.asarray([0.0, 0.0, 0.0, 1.0]))
            internal_state["object_rotation"] = Rotation.from_quat(object_quat)
            internal_state["reference"]["object_rotation"] = Rotation.from_quat(reference["qpos"][..., self.object_quaternion_qpos_slice][..., self.quaternion_wxyz_to_xyzw_indices])

        return internal_state


    def get_observation(self, data, internal_state, action, key, eval_mode):
        root_quat = data.qpos[..., self.root_quaternion_qpos_slice]
        valid_root = jnp.all(jnp.isfinite(root_quat), axis=-1) & (jnp.linalg.norm(root_quat, axis=-1) > 0)
        root_quat = jnp.where(valid_root[..., None], root_quat, jnp.asarray([1.0, 0.0, 0.0, 0.0]))
        root_inverse = Rotation.from_quat(root_quat[..., self.quaternion_wxyz_to_xyzw_indices]).inv()

        policy_previous_actions = action
        policy_base_angular_velocity = data.qvel[..., self.root_angular_qvel_slice]
        policy_joint_velocities = data.qvel[..., self.actuator_joint_mask_qvel]
        policy_reference_joint_positions = internal_state["reference"]["qpos"][..., self.actuator_joint_mask_qpos]
        policy_reference_joint_velocities = internal_state["reference"]["qvel"][..., self.actuator_joint_mask_qvel]

        critic_reference_anchor_position = internal_state["anchor_rotation_inverse"].apply(internal_state["reference"]["anchor_position"] - internal_state["anchor_position"])
        reference_anchor_matrix = (internal_state["anchor_rotation_inverse"] * internal_state["reference"]["anchor_rotation"]).as_matrix()
        policy_reference_anchor_orientation = reference_anchor_matrix[..., :2].reshape(data.qpos.shape[:-1] + (6,))
        critic_base_linear_velocity = root_inverse.apply(data.qvel[..., self.root_linear_qvel_slice])
        policy_joint_positions = data.qpos[..., self.actuator_joint_mask_qpos] - self.actuator_joint_nominal_positions

        critic_object_position = jnp.empty(data.qpos.shape[:-1] + (0,))
        critic_object_orientation = jnp.empty(data.qpos.shape[:-1] + (0,))
        critic_object_velocity = jnp.empty(data.qpos.shape[:-1] + (0,))
        if self.motion_library.has_object:
            critic_object_position = internal_state["anchor_rotation_inverse"].apply(data.qpos[..., self.object_position_qpos_slice] - internal_state["anchor_position"])
            object_matrix = (internal_state["anchor_rotation_inverse"] * internal_state["object_rotation"]).as_matrix()
            critic_object_orientation = object_matrix[..., :2].reshape(data.qpos.shape[:-1] + (6,))
            # Match Holosoma's obj_lin_vel_b frame-transform convention.
            critic_object_velocity = internal_state["anchor_rotation_inverse"].apply(data.qvel[..., self.object_linear_qvel_slice] - internal_state["anchor_position"])

        relative_pos = internal_state["body_anchor_rotation_inverse"].apply((internal_state["body_positions"] - internal_state["anchor_position"][..., None, :]).reshape(-1, 3))
        relative_matrix = (internal_state["body_anchor_rotation_inverse"] * internal_state["body_rotation"]).as_matrix()
        critic_body_positions = relative_pos.reshape(data.qpos.shape[:-1] + (-1,))
        critic_body_orientations = relative_matrix[..., :2].reshape(data.qpos.shape[:-1] + (-1,))

        critic_previous_actions = action
        critic_base_angular_velocity = policy_base_angular_velocity
        critic_joint_positions = policy_joint_positions
        critic_joint_velocities = policy_joint_velocities
        critic_reference_joint_positions = policy_reference_joint_positions
        critic_reference_joint_velocities = policy_reference_joint_velocities
        critic_reference_anchor_orientation = policy_reference_anchor_orientation

        observation = jnp.concatenate([
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
        ], axis=-1).astype(jnp.float32)

        observation = self.observation_noise_module.apply(key, observation, eval_mode)

        observation = observation.at[..., self.joint_positions_normalization_idx].set(observation[..., self.joint_positions_normalization_idx] / self.env_config["observation"]["joint_position_range"])
        observation = observation.at[..., self.reference_joint_positions_normalization_idx].set((observation[..., self.reference_joint_positions_normalization_idx] - self.actuator_joint_nominal_positions) / self.env_config["observation"]["joint_position_range"])
        observation = observation.at[..., self.joint_velocities_normalization_idx].set(observation[..., self.joint_velocities_normalization_idx] / self.env_config["observation"]["joint_velocity_range"])
        observation = observation.at[..., self.previous_actions_normalization_idx].set(observation[..., self.previous_actions_normalization_idx] / self.env_config["observation"]["action_range"])
        observation = observation.at[..., self.angular_velocities_normalization_idx].set(jnp.clip(observation[..., self.angular_velocities_normalization_idx] / self.env_config["observation"]["angular_velocity_range"], -1.0, 1.0))
        observation = observation.at[..., self.linear_velocities_normalization_idx].set(jnp.clip(observation[..., self.linear_velocities_normalization_idx] / self.env_config["observation"]["linear_velocity_range"], -1.0, 1.0))
        observation = observation.at[..., self.relative_positions_normalization_idx].set(observation[..., self.relative_positions_normalization_idx] / self.env_config["observation"]["relative_position_range"])
        observation = observation.at[..., self.orientations_normalization_idx].set(jnp.clip(observation[..., self.orientations_normalization_idx], -1.0, 1.0))

        observation = jnp.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        observation = jnp.clip(observation, -self.env_config["observation"]["clip"], self.env_config["observation"]["clip"])

        return observation


    def handle_domain_randomization(self, data, model, internal_state, keys, eval_mode, is_episode_start=False, is_initial=False):
        if is_initial:
            model = self.mujoco_model_module.sample(keys["model"], eval_mode)
            self.actuator_module.sample_bias(internal_state, jnp.ones(self.nr_envs, bool), keys["bias"])
            self.actuator_module.sample_gains(internal_state, jnp.ones(self.nr_envs, bool), keys["gains"])
            self.action_delay_module.sample(internal_state, jnp.ones(self.nr_envs, bool), keys["delay"])
            return data, model, internal_state

        should_randomize_model = jnp.where(is_episode_start, self.mujoco_model_sampling_module.setup(), self.mujoco_model_sampling_module.step(jax.random.fold_in(keys["model"], 0)))
        should_randomize_bias = jnp.where(is_episode_start, self.actuator_bias_sampling_module.setup(), self.actuator_bias_sampling_module.step(jax.random.fold_in(keys["bias"], 0)))
        should_randomize_gains = jnp.where(is_episode_start, self.actuator_gain_sampling_module.setup(), self.actuator_gain_sampling_module.step(jax.random.fold_in(keys["gains"], 0)))
        should_randomize_delay = jnp.where(is_episode_start, self.action_delay_sampling_module.setup(), self.action_delay_sampling_module.step(jax.random.fold_in(keys["delay"], 0)))

        sampled_model = self.mujoco_model_module.sample(keys["model"], eval_mode)
        model = jax.tree.map(
            lambda current, sampled, axis: current if axis is None else jnp.where(should_randomize_model.reshape((-1,) + (1,) * (current.ndim - 1)), sampled, current),
            model, sampled_model, self.mujoco_model_module.model_axes, is_leaf=lambda value: value is None,
        )
        self.actuator_module.sample_bias(internal_state, should_randomize_bias, keys["bias"])
        self.actuator_module.sample_gains(internal_state, should_randomize_gains, keys["gains"])
        self.action_delay_module.sample(internal_state, should_randomize_delay, keys["delay"])

        should_perturb = jnp.where(is_episode_start, self.perturbation_sampling_module.setup(), self.perturbation_sampling_module.step(keys["perturbation_sampling"]))
        data = data.replace(qvel=self.perturbation_module.apply(keys["push"], data.qvel, data.qpos, should_perturb, eval_mode))

        return data, model, internal_state


    def get_observation_space(self):
        current_observation_idx = 0

        self.joint_previous_actions_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuators)
        current_observation_idx += self.nr_actuators
        self.base_angular_velocity_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.joint_positions_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.joint_velocities_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.reference_joint_positions_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.reference_joint_velocities_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.reference_anchor_orientation_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 6)
        current_observation_idx += 6

        self.actor_observation_size = current_observation_idx
        self.policy_observation_indices = jnp.arange(current_observation_idx)

        self.critic_joint_previous_actions_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuators)
        current_observation_idx += self.nr_actuators
        self.critic_base_angular_velocity_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.critic_base_linear_velocity_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.critic_joint_positions_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_joint_velocities_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_reference_joint_positions_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_reference_joint_velocities_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + self.nr_actuator_joints)
        current_observation_idx += self.nr_actuator_joints
        self.critic_reference_anchor_orientation_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 6)
        current_observation_idx += 6
        self.critic_reference_anchor_position_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 3)
        current_observation_idx += 3
        self.critic_object_velocity_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + (3 if self.has_object else 0))
        current_observation_idx += (3 if self.has_object else 0)
        self.critic_object_orientation_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + (6 if self.has_object else 0))
        current_observation_idx += (6 if self.has_object else 0)
        self.critic_object_position_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + (3 if self.has_object else 0))
        current_observation_idx += (3 if self.has_object else 0)
        self.critic_body_orientations_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 6 * len(self.body_ids))
        current_observation_idx += 6 * len(self.body_ids)
        self.critic_body_positions_obs_idx = jnp.arange(current_observation_idx, current_observation_idx + 3 * len(self.body_ids))
        current_observation_idx += 3 * len(self.body_ids)

        self.critic_observation_indices = jnp.arange(self.actor_observation_size, current_observation_idx)

        self.joint_positions_normalization_idx = jnp.concatenate((self.joint_positions_obs_idx, self.critic_joint_positions_obs_idx))
        self.reference_joint_positions_normalization_idx = jnp.stack((self.reference_joint_positions_obs_idx, self.critic_reference_joint_positions_obs_idx))
        self.joint_velocities_normalization_idx = jnp.concatenate((self.joint_velocities_obs_idx, self.critic_joint_velocities_obs_idx, self.reference_joint_velocities_obs_idx, self.critic_reference_joint_velocities_obs_idx))
        self.previous_actions_normalization_idx = jnp.concatenate((self.joint_previous_actions_obs_idx, self.critic_joint_previous_actions_obs_idx))
        self.angular_velocities_normalization_idx = jnp.concatenate((self.base_angular_velocity_obs_idx, self.critic_base_angular_velocity_obs_idx))
        self.linear_velocities_normalization_idx = jnp.concatenate((self.critic_base_linear_velocity_obs_idx, self.critic_object_velocity_obs_idx))
        self.relative_positions_normalization_idx = jnp.concatenate((self.critic_reference_anchor_position_obs_idx, self.critic_object_position_obs_idx, self.critic_body_positions_obs_idx))
        self.orientations_normalization_idx = jnp.concatenate((self.reference_anchor_orientation_obs_idx, self.critic_reference_anchor_orientation_obs_idx, self.critic_object_orientation_obs_idx, self.critic_body_orientations_obs_idx))

        return BoxSpace(-self.env_config["observation"]["clip"], self.env_config["observation"]["clip"], (current_observation_idx,), np.float32)


    def render(self, state):
        if self.viewer is None:
            self.render_data = mujoco.MjData(self.initial_mj_model)
            self.viewer = MujocoViewer(self.initial_mj_model, self.dt)
        self.render_data.qpos[:] = np.asarray(state.data.qpos[0])
        self.render_data.qvel[:] = np.asarray(state.data.qvel[0])
        self.render_data.ctrl[:] = np.asarray(state.data.ctrl[0])
        self.render_data.time = float(state.data.time[0])
        mujoco.mj_forward(self.initial_mj_model, self.render_data)
        self.viewer.render(self.render_data, np.asarray(state.internal_state["reference"]["qpos"][0]))
        return state


    def close(self):
        if self.viewer is not None:
            self.viewer.close()
