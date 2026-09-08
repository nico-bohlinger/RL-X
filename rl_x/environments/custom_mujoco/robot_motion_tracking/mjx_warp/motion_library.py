from pathlib import Path
import glob
import hashlib
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as Rotation_NP
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


class MotionLibrary:
    def __init__(self, env_config, mj_model):
        self.trunk_body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self.base_joint_id = mj_model.body_jntadr[self.trunk_body_id]
        self.has_free_base = self.base_joint_id >= 0 and mj_model.jnt_type[self.base_joint_id] == mujoco.mjtJoint.mjJNT_FREE
        self.base_qpos_dim = 7 if self.has_free_base else 0
        self.base_qvel_dim = 6 if self.has_free_base else 0
        self.base_qpos_adr = mj_model.jnt_qposadr[self.base_joint_id]
        base_qvel_adr = mj_model.jnt_dofadr[self.base_joint_id]
        self.root_linear_qvel_slice = slice(base_qvel_adr, base_qvel_adr + 3)
        self.root_angular_qvel_slice = slice(base_qvel_adr + 3, base_qvel_adr + self.base_qvel_dim)
        self.root_position_qpos_slice = slice(self.base_qpos_adr, self.base_qpos_adr + 3)
        self.root_quaternion_qpos_slice = slice(self.base_qpos_adr + 3, self.base_qpos_adr + self.base_qpos_dim)

        actuator_joint_ids = mj_model.actuator_trnid[:, 0]
        self.actuator_joint_names = [mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) for joint_id in actuator_joint_ids]
        self.actuator_joint_mask_joints = actuator_joint_ids
        self.actuator_joint_mask_qpos = mj_model.jnt_qposadr[actuator_joint_ids]
        self.actuator_joint_mask_qvel = mj_model.jnt_dofadr[actuator_joint_ids]
        self.nr_actuators = mj_model.nu
        self.nr_actuator_joints = len(actuator_joint_ids)

        self.has_object = env_config["dataset"] == "omomo"
        if self.has_object:
            self.object_joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint")
            object_qpos_adr = mj_model.jnt_qposadr[self.object_joint_id]
            object_qvel_adr = mj_model.jnt_dofadr[self.object_joint_id]
            self.object_position_qpos_slice = slice(object_qpos_adr, object_qpos_adr + 3)
            self.object_quaternion_qpos_slice = slice(object_qpos_adr + 3, object_qpos_adr + self.base_qpos_dim)
            self.object_qvel_slice = slice(object_qvel_adr, object_qvel_adr + self.base_qvel_dim)

        self.quaternion_wxyz_to_xyzw_indices = [1, 2, 3, 0]
        self.quaternion_xyzw_to_wxyz_indices = [3, 0, 1, 2]

        files = sorted(set(path for pattern in env_config["motion_files"].split(",") if pattern.strip()
                           for path in glob.glob(str(Path(pattern.strip()).expanduser()))))
        if not files:
            raise ValueError("motion_files must name existing retargeted NPZ files; no synthetic fallback or online retargeting")

        source_joint_qpos_indices = np.sort(self.actuator_joint_mask_qpos)
        self.robot_qpos_indices = np.arange(mj_model.nq)
        if self.has_object:
            self.robot_qpos_indices = np.concatenate((self.robot_qpos_indices[:self.object_position_qpos_slice.start], self.robot_qpos_indices[self.object_quaternion_qpos_slice.stop:]))
        self.nr_robot_qpos = len(self.robot_qpos_indices)
        body_names = env_config["tracked_bodies"]
        self.body_ids = np.array([mj_model.body(name).id for name in body_names])
        self.anchor_index = body_names.index(env_config["anchor_body"])
        self.hashes = {}
        starts = []
        lengths = []
        fps_list = []
        arrays = []

        offset = 0
        for file in files:
            self.hashes[file] = hashlib.sha256(Path(file).read_bytes()).hexdigest()
            with np.load(file, allow_pickle=False) as source:
                fps = float(np.squeeze(source["fps"] if "fps" in source else env_config["motion_fps"]))
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"{file}: invalid fps")
                if "qpos" in source:
                    qpos = source["qpos"].astype(np.float64)
                elif "joint_pos" in source and source["joint_pos"].ndim == 2 and source["joint_pos"].shape[1] == self.nr_robot_qpos:
                    qpos = np.broadcast_to(mj_model.qpos0, (len(source["joint_pos"]), mj_model.nq)).copy()
                    qpos[:, self.robot_qpos_indices] = source["joint_pos"]
                    if self.has_object:
                        qpos[:, self.object_position_qpos_slice] = source["object_pos_w"]
                        qpos[:, self.object_quaternion_qpos_slice] = source["object_quat_w"]
                elif "joint_pos" in source and "body_pos_w" in source and "body_quat_w" in source:
                    if "body_names" not in source:
                        raise ValueError(f"{file}: maximal-coordinate input requires body_names (no inferred body ordering)")
                    src_bodies = source["body_names"].astype(str).tolist()
                    root_name = "trunk" if "trunk" in src_bodies else "pelvis"
                    root_id = src_bodies.index(root_name)
                    qpos = np.broadcast_to(mj_model.qpos0, (len(source["joint_pos"]), mj_model.nq)).copy()
                    qpos[:, self.root_position_qpos_slice] = source["body_pos_w"][:, root_id]
                    qpos[:, self.root_quaternion_qpos_slice] = source["body_quat_w"][:, root_id]
                    qpos[:, source_joint_qpos_indices] = source["joint_pos"]
                    if self.has_object:
                        qpos[:, self.object_position_qpos_slice] = source["object_pos_w"]
                        qpos[:, self.object_quaternion_qpos_slice] = source["object_quat_w"]
                else:
                    raise ValueError(f"{file}: expected native qpos or named maximal-coordinate reference arrays")

                if qpos.ndim != 2 or qpos.shape[1] != mj_model.nq or len(qpos) < 2 or not np.isfinite(qpos).all():
                    raise ValueError(f"{file}: expected finite [T>=2,{mj_model.nq}] qpos")
                if "joint_names" in source:
                    order = source["joint_names"].astype(str).tolist()
                    if len(order) != self.nr_actuator_joints or len(set(order)) != self.nr_actuator_joints or set(order) != set(self.actuator_joint_names):
                        raise ValueError(f"{file}: joint_names must exactly match G1 joints")
                    qpos[:, self.actuator_joint_mask_qpos] = qpos[:, source_joint_qpos_indices[[order.index(name) for name in self.actuator_joint_names]]]
                elif not env_config["native_joint_order"]:
                    raise ValueError(f"{file}: supply joint_names or explicitly enable native_joint_order for native MuJoCo G1 order")

                quaternion_order = source["quaternion_order"].item() if "quaternion_order" in source else env_config["quaternion_order"]
                if quaternion_order not in ("wxyz", "xyzw"):
                    raise ValueError(f"{file}: quaternion_order must be wxyz or xyzw")
                stored_fields = ("joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_names", "joint_names")
                stored_reference = {key: source[key].copy() for key in stored_fields} if all(key in source for key in stored_fields) else None
                stored_object_velocity = source["object_lin_vel_w"].copy() if self.has_object and "object_lin_vel_w" in source else None

                for start in ([self.root_quaternion_qpos_slice.start, self.object_quaternion_qpos_slice.start] if self.has_object else [self.root_quaternion_qpos_slice.start]):
                    if quaternion_order == "xyzw":
                        qpos[:, start:start + 4] = qpos[:, start + np.array(self.quaternion_xyzw_to_wxyz_indices)]
                    norms = np.linalg.norm(qpos[:, start:start + 4], axis=-1)
                    if np.max(np.abs(norms - 1)) > 1e-3:
                        raise ValueError(f"{file}: invalid quaternion norm")
                    qpos[:, start:start + 4] /= norms[:, None]

            if stored_reference is not None:
                source_bodies = stored_reference["body_names"].astype(str).tolist()
                source_bodies = ["trunk" if name == "pelvis" else name for name in source_bodies]
                source_joints = stored_reference["joint_names"].astype(str).tolist()
                if len(set(source_bodies)) != len(source_bodies) or not all(name in source_bodies for name in body_names):
                    raise ValueError(f"{file}: reference body joint_names must be unique and contain all tracked bodies")
                body_indices = [source_bodies.index(name) for name in body_names]
                root_index = source_bodies.index("trunk")
                joint_indices = np.asarray([source_joints.index(name) for name in self.actuator_joint_names])
                pos = np.asarray(stored_reference["body_pos_w"][:, body_indices], dtype=np.float64)
                quat = np.asarray(stored_reference["body_quat_w"][:, body_indices], dtype=np.float64)
                root_quat = np.asarray(stored_reference["body_quat_w"][:, root_index], dtype=np.float64)
                if quaternion_order == "xyzw":
                    quat = quat[..., self.quaternion_xyzw_to_wxyz_indices]
                    root_quat = root_quat[..., self.quaternion_xyzw_to_wxyz_indices]
                if pos.shape != (len(qpos), len(body_names), 3) or quat.shape != (len(qpos), len(body_names), 4):
                    raise ValueError(f"{file}: invalid stored body reference shape")
                if np.max(np.abs(np.linalg.norm(quat, axis=-1) - 1.0)) > 1e-3:
                    raise ValueError(f"{file}: invalid stored body quaternion norm")
                qpos[:, self.root_position_qpos_slice] = stored_reference["body_pos_w"][:, root_index]
                qpos[:, self.root_quaternion_qpos_slice] = root_quat
                qvel = np.zeros((len(qpos), mj_model.nv), dtype=np.float64)
                joint_vel = stored_reference["joint_vel"]
                source_velocity_offset = joint_vel.shape[1] - len(source_joints)
                if source_velocity_offset not in (0, self.base_qvel_dim):
                    raise ValueError(f"{file}: invalid stored joint velocity width")
                qvel[:, self.actuator_joint_mask_qvel] = joint_vel[:, source_velocity_offset + joint_indices]
                qvel[:, self.root_linear_qvel_slice] = stored_reference["body_lin_vel_w"][:, root_index]
                root_rotation = Rotation_NP.from_quat(root_quat[:, self.quaternion_wxyz_to_xyzw_indices])
                qvel[:, self.root_angular_qvel_slice] = root_rotation.inv().apply(stored_reference["body_ang_vel_w"][:, root_index])
                if self.has_object:
                    if stored_object_velocity is None or stored_object_velocity.shape != (len(qpos), 3):
                        raise ValueError(f"{file}: stored object reference requires object_lin_vel_w")
                    qvel[:, self.object_qvel_slice.start:self.object_qvel_slice.start + 3] = stored_object_velocity
                velocity = np.concatenate((stored_reference["body_ang_vel_w"][:, body_indices], stored_reference["body_lin_vel_w"][:, body_indices]), axis=-1)
            else:
                qvel = np.empty((len(qpos), mj_model.nv), dtype=np.float64)
                for frame in range(len(qpos)):
                    left = max(0, frame - 1)
                    right = min(len(qpos) - 1, frame + 1)
                    mujoco.mj_differentiatePos(mj_model, qvel[frame], (right - left) / fps, qpos[left], qpos[right])

                data = mujoco.MjData(mj_model)
                pos = []
                quat = []
                velocity = []
                for q, v in zip(qpos, qvel):
                    data.qpos[:], data.qvel[:] = q, v
                    mujoco.mj_forward(mj_model, data)
                    pos.append(data.xpos[self.body_ids].copy())
                    quat.append(data.xquat[self.body_ids].copy())
                    body_velocity = np.empty((len(self.body_ids), 6))
                    for index, body in enumerate(self.body_ids):
                        mujoco.mj_objectVelocity(mj_model, data, mujoco.mjtObj.mjOBJ_XBODY, body, body_velocity[index], 0)
                    velocity.append(body_velocity)

            reference = dict(qpos=qpos, qvel=qvel, body_positions=np.asarray(pos), body_quaternions=np.asarray(quat), body_velocities=np.asarray(velocity))
            if not all(np.isfinite(value).all() for value in reference.values()):
                raise ValueError(f"{file}: nonfinite reference data")
            arrays.append(reference)
            starts.append(offset)
            lengths.append(len(qpos))
            fps_list.append(fps)
            offset += len(qpos)

        self.total_frames = offset
        self.files = files
        self.starts = jnp.asarray(starts)
        self.lengths = jnp.asarray(lengths)
        self.fps = jnp.asarray(fps_list, dtype=jnp.float32)
        self.durations = (self.lengths - 1) / self.fps
        self.arrays = {key: jnp.asarray(np.concatenate([a[key] for a in arrays]), dtype=jnp.float32) for key in arrays[0]}


    def sample(self, clip, time):
        frame = jnp.clip(time * self.fps[clip], 0, self.lengths[clip] - 1)
        frame_index = frame.astype(jnp.int32)
        left = self.starts[clip] + frame_index
        right = self.starts[clip] + jnp.minimum(frame_index + 1, self.lengths[clip] - 1)
        alpha = frame - frame_index

        result = {}
        for name, array in self.arrays.items():
            left_value = array[left]
            right_value = array[right]
            weight = jnp.reshape(alpha, alpha.shape + (1,) * (left_value.ndim - alpha.ndim))
            result[name] = left_value + weight * (right_value - left_value)

        quaternion_fields = [("body_quaternions", 0, 4), ("qpos", self.root_quaternion_qpos_slice.start, self.root_quaternion_qpos_slice.stop)]
        if self.has_object:
            quaternion_fields.append(("qpos", self.object_quaternion_qpos_slice.start, self.object_quaternion_qpos_slice.stop))
        for name, start, stop in quaternion_fields:
            a, b = self.arrays[name][left][..., start:stop], self.arrays[name][right][..., start:stop]
            left_rotation = Rotation.from_quat(a[..., self.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
            right_rotation = Rotation.from_quat(b[..., self.quaternion_wxyz_to_xyzw_indices].reshape(-1, 4))
            weight = jnp.reshape(alpha, alpha.shape + (1,) * (a.ndim - 1 - alpha.ndim))
            weight = jnp.broadcast_to(weight, a.shape[:-1]).reshape(-1, 1)
            rotation = left_rotation * Rotation.from_rotvec((left_rotation.inv() * right_rotation).as_rotvec() * weight)
            quaternion = rotation.as_quat()[..., self.quaternion_xyzw_to_wxyz_indices].reshape(a.shape)
            result[name] = result[name].at[..., start:stop].set(quaternion)

        return result
