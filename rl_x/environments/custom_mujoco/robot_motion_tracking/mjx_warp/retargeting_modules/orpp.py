from functools import partial
import hashlib
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from rl_x.environments.custom_mujoco.robot_motion_tracking.mjx_warp.retargeting_modules.kinematics import Kinematics


class OmniRetarget:
    def __init__(self, env):
        self.env = env
        self.mode = env.env_config["retargeting"]["orpp"]["mode"]
        self.interaction_bodies = env.env_config["retargeting"]["orpp"]["interaction_bodies"]
        self.seed_offset = env.env_config["retargeting"]["orpp"]["seed_offset"]
        self.object_scale_range = env.env_config["retargeting"]["orpp"]["object_scale_range"]
        self.robot_scale_range = env.env_config["retargeting"]["orpp"]["robot_scale_range"]
        self.tile_size = env.env_config["retargeting"]["orpp"]["tile_size"]
        self.joint_envelope = env.env_config["retargeting"]["orpp"]["joint_envelope"]
        self.coordinate_envelope = env.env_config["retargeting"]["orpp"]["coordinate_envelope"]
        self.root_envelope = env.env_config["retargeting"]["orpp"]["root_envelope"]
        self.ground_envelope = env.env_config["retargeting"]["orpp"]["ground_envelope"]
        self.interaction_envelope = env.env_config["retargeting"]["orpp"]["interaction_envelope"]
        self.laplacian_weight = env.env_config["retargeting"]["orpp"]["laplacian_weight"]
        self.regularizer = env.env_config["retargeting"]["orpp"]["regularizer"]
        if not env.has_object or (env.env_config["object"]["type"] != "box" and self.mode != "robot_only"):
            raise ValueError("Scanned-mesh size variation needs per-world mesh support before object modes can train")
        if not jax.config.jax_enable_x64 or jax.default_backend() != "gpu":
            raise RuntimeError("Online OR++ requires x64 CUDA and has no numerical fallback")
        self.kinematics = Kinematics(env)
        self.interaction_body_ids = jnp.asarray([env.initial_mj_model.body(name).id for name in self.interaction_bodies])
        self.robot_qpos_indices = jnp.asarray(env.motion_library.robot_qpos_indices)
        self.free_coordinates = jnp.asarray([index for index in env.motion_library.robot_qpos_indices if index not in range(env.root_quaternion_qpos_slice.start, env.root_quaternion_qpos_slice.stop)])
        self.joint_indices = jnp.asarray(env.actuator_joint_mask_qpos)
        self.lower = jnp.asarray(env.initial_mj_model.jnt_range[env.actuator_joint_mask_joints, 0])
        self.upper = jnp.asarray(env.initial_mj_model.jnt_range[env.actuator_joint_mask_joints, 1])
        key_object, key_robot = jax.random.split(jax.random.PRNGKey(env.env_config["seed"] + self.seed_offset))
        self.object_scale = jax.random.uniform(key_object, (env.nr_envs, 3), dtype=jnp.float64, minval=self.object_scale_range[0], maxval=self.object_scale_range[1])
        self.robot_scale = jax.random.uniform(key_robot, (env.nr_envs, 3), dtype=jnp.float64, minval=self.robot_scale_range[0], maxval=self.robot_scale_range[1])
        if self.mode == "object_only":
            self.robot_scale = jnp.ones_like(self.robot_scale)
        if self.mode == "robot_only":
            self.object_scale = jnp.ones_like(self.object_scale)
        self.object_scale = self.object_scale.at[0].set(1.0)
        self.robot_scale = self.robot_scale.at[0].set(1.0)
        self.records = []
        self.nominal_qpos = []
        self.caches = []
        for path in env.motion_library.files:
            with np.load(path, allow_pickle=False) as source:
                self.nominal_qpos.append(jnp.asarray(source["qpos"], dtype=jnp.float64))
                self.caches.append({name: jnp.asarray(source[name], dtype=jnp.float64) for name in ("robot_laplacian", "object_laplacian", "source_target", "object_points")})
        self.reference_arrays = {name: [] for name in env.motion_library.arrays}
        for clip in range(len(env.motion_library.files)):
            started = time.perf_counter()
            nominal_ground = jax.vmap(self.kinematics.ground, in_axes=(0, None))(self.nominal_qpos[clip], jnp.ones(3))
            identity_gradient = jax.vmap(jax.grad(self.objective), in_axes=(0, None, None, 0, 0, 0, None))(
                self.nominal_qpos[clip], jnp.ones(3), jnp.ones(3), self.caches[clip]["robot_laplacian"], self.caches[clip]["object_laplacian"], self.caches[clip]["source_target"], self.caches[clip]["object_points"])
            references = {name: [] for name in env.motion_library.arrays}
            metrics = []
            for start in range(0, env.nr_envs, self.tile_size):
                count = min(self.tile_size, env.nr_envs - start)
                object_scale = jnp.pad(self.object_scale[start:start + count], ((0, self.tile_size - count), (0, 0)), constant_values=1.0)
                robot_scale = jnp.pad(self.robot_scale[start:start + count], ((0, self.tile_size - count), (0, 0)), constant_values=1.0)
                qpos, quality = self.correct_tile(self.nominal_qpos[clip], nominal_ground, identity_gradient, object_scale, robot_scale, self.caches[clip])
                jax.block_until_ready(qpos)
                reference = self.make_reference(qpos, robot_scale, env.motion_library.fps[clip])
                for name in references:
                    identity = jnp.all(object_scale[:count] == 1.0, axis=1) & jnp.all(robot_scale[:count] == 1.0, axis=1)
                    nominal_reference = env.motion_library.arrays[name][env.motion_library.starts[clip]:env.motion_library.starts[clip] + env.motion_library.lengths[clip]]
                    references[name].append(jnp.where(identity.reshape((-1,) + (1,) * (reference[name].ndim - 1)), nominal_reference[None], reference[name][:count].astype(jnp.float32)))
                metrics.append(np.asarray(quality[:count]))
            metrics = np.concatenate(metrics)
            record = {"clip": str(env.motion_library.files[clip]), "frames": len(self.nominal_qpos[clip]), "creation_seconds": time.perf_counter() - started,
                      "finite": bool(np.isfinite(metrics).all()), "max_joint_change": float(metrics[:, 0].max()), "max_coordinate_change": float(metrics[:, 1].max()),
                      "max_root_change": float(metrics[:, 2].max()), "max_additional_ground": float(metrics[:, 3].max()), "max_interaction_change": float(metrics[:, 4].max()),
                      "max_box_violation": float(metrics[:, 5].max()), "identity_error": float(metrics[0, 6])}
            record["passed"] = record["finite"] and record["max_joint_change"] <= self.joint_envelope and record["max_coordinate_change"] <= self.coordinate_envelope and record["max_root_change"] <= self.root_envelope and record["max_additional_ground"] <= self.ground_envelope and record["max_interaction_change"] <= self.interaction_envelope and record["max_box_violation"] <= 0.0 and record["identity_error"] == 0.0
            self.records.append(record)
            if not record["passed"]:
                raise RuntimeError(f"Online OR++ qualification failed before training: {record}")
            for name in references:
                self.reference_arrays[name].append(jnp.concatenate(references[name]))
        self.reference_arrays = {name: jnp.concatenate(arrays, axis=1) for name, arrays in self.reference_arrays.items()}
        self.metadata_sha256 = hashlib.sha256(np.asarray(jnp.concatenate((self.object_scale, self.robot_scale), axis=1)).tobytes()).hexdigest()
        env.motion_library.variant_arrays = self.reference_arrays


    def residual(self, qpos, object_scale, robot_scale, robot_laplacian, object_laplacian, target, points):
        positions, _ = self.kinematics.transforms(qpos, robot_scale)
        object_rotation = Rotation.from_quat(qpos[self.env.object_quaternion_qpos_slice][self.env.quaternion_wxyz_to_xyzw_indices])
        local_positions = object_rotation.inv().apply(positions[self.interaction_body_ids] - qpos[self.env.object_position_qpos_slice])
        return robot_laplacian @ local_positions + object_laplacian @ (points * object_scale) - target


    def objective(self, qpos, object_scale, robot_scale, robot_laplacian, object_laplacian, target, points):
        return 0.5 * self.laplacian_weight * jnp.sum(self.residual(qpos, object_scale, robot_scale, robot_laplacian, object_laplacian, target, points) ** 2)


    @partial(jax.jit, static_argnums=(0,))
    def correct_tile(self, nominal, baseline_ground, identity_gradient, object_scales, robot_scales, cache):
        def frame(qpos, ground, nominal_gradient, object_scale, robot_scale, robot_laplacian, object_laplacian, target):
            residual = self.residual(qpos, object_scale, robot_scale, robot_laplacian, object_laplacian, target, cache["object_points"])
            jacobian = jax.jacrev(self.residual)(qpos, object_scale, robot_scale, robot_laplacian, object_laplacian, target, cache["object_points"])[..., self.free_coordinates].reshape(-1, len(self.free_coordinates))
            hessian = self.laplacian_weight * jacobian.T @ jacobian + self.regularizer * jnp.eye(len(self.free_coordinates))
            gradient = self.laplacian_weight * jacobian.T @ residual.reshape(-1) - nominal_gradient[self.free_coordinates]
            factor = jnp.linalg.cholesky(hessian)
            forward = jax.lax.linalg.triangular_solve(factor, -gradient[:, None], left_side=True, lower=True)
            direction = jax.lax.linalg.triangular_solve(factor, forward, left_side=True, lower=True, transpose_a=True)[:, 0]
            candidate = qpos.at[self.free_coordinates].add(direction)
            joint_margin = 128.0 * jnp.finfo(jnp.float32).eps * jnp.maximum(1.0, jnp.maximum(jnp.abs(qpos[self.joint_indices]), jnp.maximum(jnp.abs(self.lower), jnp.abs(self.upper))))
            candidate = candidate.at[self.joint_indices].set(jnp.clip(candidate[self.joint_indices], jnp.maximum(self.lower, qpos[self.joint_indices] - self.joint_envelope + joint_margin), jnp.minimum(self.upper, qpos[self.joint_indices] + self.joint_envelope - joint_margin)))
            candidate_ground = self.kinematics.ground(candidate, robot_scale)
            ground_margin = 128.0 * jnp.finfo(jnp.float32).eps * jnp.maximum(1.0, jnp.maximum(jnp.max(jnp.abs(ground)), jnp.max(jnp.abs(candidate_ground))))
            candidate = candidate.at[self.env.root_position_qpos_slice.start + 2].add(jnp.maximum(0.0, jnp.max(ground - self.ground_envelope + ground_margin - candidate_ground)))
            identity = jnp.all(object_scale == 1.0) & jnp.all(robot_scale == 1.0)
            candidate = jnp.where(identity, qpos, candidate)
            delta = candidate - qpos
            final = self.objective(candidate, object_scale, robot_scale, robot_laplacian, object_laplacian, target, cache["object_points"])
            initial = self.laplacian_weight * jnp.sum(residual ** 2)
            quality = jnp.stack((jnp.max(jnp.abs(delta[self.joint_indices])), jnp.max(jnp.abs(delta[self.robot_qpos_indices])), jnp.linalg.norm(delta[self.env.root_position_qpos_slice]),
                                 jnp.maximum(0.0, jnp.max(ground - self.kinematics.ground(candidate, robot_scale))), 2.0 * final + self.regularizer * jnp.sum(delta ** 2) - initial,
                                 jnp.maximum(jnp.max(self.lower - candidate[self.joint_indices]), jnp.max(candidate[self.joint_indices] - self.upper)), jnp.where(identity, jnp.max(jnp.abs(delta)), 0.0)))
            return candidate, quality

        def lane(object_scale, robot_scale):
            qpos, quality = jax.vmap(frame, in_axes=(0, 0, 0, None, None, 0, 0, 0))(nominal, baseline_ground, identity_gradient, object_scale, robot_scale, cache["robot_laplacian"], cache["object_laplacian"], cache["source_target"])
            return qpos, jnp.max(quality, axis=0)

        return jax.vmap(lane)(object_scales, robot_scales)


    @partial(jax.jit, static_argnums=(0,))
    def make_reference(self, qpos, scales, fps):
        positions, quaternions = jax.vmap(jax.vmap(self.kinematics.transforms, in_axes=(0, None)))(qpos, scales)
        left = jnp.maximum(jnp.arange(qpos.shape[1]) - 1, 0)
        right = jnp.minimum(jnp.arange(qpos.shape[1]) + 1, qpos.shape[1] - 1)
        elapsed = (right - left) / fps
        qvel = jnp.zeros((qpos.shape[0], qpos.shape[1], self.env.initial_mj_model.nv), dtype=qpos.dtype)
        qvel = qvel.at[..., self.env.actuator_joint_mask_qvel].set((qpos[:, right][..., self.joint_indices] - qpos[:, left][..., self.joint_indices]) / elapsed[None, :, None])
        for position_slice, quaternion_slice, velocity_slice in ((self.env.root_position_qpos_slice, self.env.root_quaternion_qpos_slice, slice(self.env.root_linear_qvel_slice.start, self.env.root_angular_qvel_slice.stop)), (self.env.object_position_qpos_slice, self.env.object_quaternion_qpos_slice, self.env.object_qvel_slice)):
            qvel = qvel.at[..., velocity_slice.start:velocity_slice.start + 3].set((qpos[:, right, position_slice] - qpos[:, left, position_slice]) / elapsed[None, :, None])
            before = Rotation.from_quat(qpos[:, left, quaternion_slice][..., self.env.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
            after = Rotation.from_quat(qpos[:, right, quaternion_slice][..., self.env.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
            angular = (before.inv() * after).as_rotvec().reshape(qpos.shape[0], qpos.shape[1], 3) / elapsed[None, :, None]
            qvel = qvel.at[..., velocity_slice.start + 3:velocity_slice.stop].set(angular)
        body_positions = positions[..., self.env.body_ids, :]
        body_quaternions = quaternions[..., self.env.body_ids, :]
        before = Rotation.from_quat(body_quaternions[:, left][..., self.env.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
        after = Rotation.from_quat(body_quaternions[:, right][..., self.env.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
        angular = (after * before.inv()).as_rotvec().reshape(*body_positions.shape) / elapsed[None, :, None, None]
        linear = (body_positions[:, right] - body_positions[:, left]) / elapsed[None, :, None, None]
        return {"qpos": qpos, "qvel": qvel, "body_positions": body_positions, "body_quaternions": body_quaternions, "body_velocities": jnp.concatenate((angular, linear), axis=-1)}


    def randomize_model(self, model):
        robot_scale = self.robot_scale.astype(jnp.float32)
        object_scale = self.object_scale.astype(jnp.float32)
        mean_scale = jnp.mean(robot_scale, axis=1)
        body_scale = jnp.where(jnp.asarray(self.kinematics.robot_bodies)[None, :, None], robot_scale[:, None], 1.0)
        geom_scale = body_scale[:, self.env.initial_mj_model.geom_bodyid]
        joint_scale = body_scale[:, self.env.initial_mj_model.jnt_bodyid]
        size_scale = jnp.mean(geom_scale, axis=2, keepdims=True)
        geom_size = self.env.initial_mjx_model.geom_size[None] * size_scale
        geom_pos = self.env.initial_mjx_model.geom_pos[None] * geom_scale
        geom_rbound = self.env.initial_mjx_model.geom_rbound[None] * size_scale[..., 0]
        if self.env.env_config["object"]["type"] == "box":
            geom_size = geom_size.at[:, self.env.object_geom_id].set(self.env.initial_mjx_model.geom_size[self.env.object_geom_id] * object_scale)
            geom_pos = geom_pos.at[:, self.env.object_geom_id].set(self.env.initial_mjx_model.geom_pos[self.env.object_geom_id] * object_scale)
            geom_rbound = geom_rbound.at[:, self.env.object_geom_id].set(jnp.linalg.norm(geom_size[:, self.env.object_geom_id], axis=1))
        mass_scale = jnp.where(jnp.asarray(self.kinematics.robot_bodies)[None], mean_scale[:, None] ** 3, 1.0)
        mass = model.body_mass * mass_scale
        second_moments = jnp.maximum(0.5 * jnp.sum(model.body_inertia, axis=2, keepdims=True) - model.body_inertia, 0.0) * mass_scale[..., None] * body_scale ** 2
        inertia = jnp.sum(second_moments, axis=2, keepdims=True) - second_moments
        if self.env.env_config["object"]["type"] == "box":
            object_mass = model.body_mass[:, self.env.object_body_id] * jnp.prod(object_scale, axis=1)
            object_size = geom_size[:, self.env.object_geom_id]
            object_inertia = object_mass[:, None] / 3.0 * (jnp.sum(object_size ** 2, axis=1, keepdims=True) - object_size ** 2)
            mass = mass.at[:, self.env.object_body_id].set(object_mass)
            inertia = inertia.at[:, self.env.object_body_id].set(object_inertia)
        else:
            inertia = inertia.at[:, self.env.object_body_id].set(model.body_inertia[:, self.env.object_body_id])
        body_ipos = model.body_ipos * body_scale
        body_ipos = body_ipos.at[:, self.env.object_body_id].set(self.env.initial_mjx_model.body_ipos[self.env.object_body_id] * object_scale)
        dof_scale = jnp.ones((self.env.nr_envs, self.env.initial_mj_model.nv), dtype=jnp.float32).at[:, self.env.actuator_joint_mask_qvel].set(mean_scale[:, None])
        return model.replace(body_pos=self.env.initial_mjx_model.body_pos[None] * body_scale, jnt_pos=self.env.initial_mjx_model.jnt_pos[None] * joint_scale,
                             geom_pos=geom_pos, geom_size=geom_size, geom_rbound=geom_rbound,
                             site_pos=self.env.initial_mjx_model.site_pos[None] * body_scale[:, self.env.initial_mj_model.site_bodyid],
                             actuator_forcerange=self.env.initial_mjx_model.actuator_forcerange[None] * mean_scale[:, None, None],
                             dof_damping=self.env.initial_mjx_model.dof_damping[None] * dof_scale, dof_armature=self.env.initial_mjx_model.dof_armature[None] * dof_scale, dof_frictionloss=self.env.initial_mjx_model.dof_frictionloss[None] * dof_scale,
                             body_ipos=body_ipos, body_mass=mass, body_inertia=inertia, body_subtreemass=mass @ self.env.mujoco_model_module.descendants.T)
