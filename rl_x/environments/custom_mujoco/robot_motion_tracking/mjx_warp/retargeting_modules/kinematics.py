from functools import partial
import mujoco
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation


class Kinematics:
    def __init__(self, env):
        self.env = env
        self.parent_ids = tuple(env.initial_mj_model.body_parentid)
        self.joint_ids = tuple(env.initial_mj_model.body_jntadr)
        self.joint_counts = tuple(env.initial_mj_model.body_jntnum)
        self.joint_types = tuple(env.initial_mj_model.jnt_type)
        self.qpos_addresses = tuple(env.initial_mj_model.jnt_qposadr)
        self.body_pos = jnp.asarray(env.initial_mj_model.body_pos)
        self.body_quat = jnp.asarray(env.initial_mj_model.body_quat)
        self.joint_pos = jnp.asarray(env.initial_mj_model.jnt_pos)
        self.joint_axis = jnp.asarray(env.initial_mj_model.jnt_axis)
        self.robot_bodies = tuple(env.initial_mj_model.body_rootid == env.trunk_body_id)
        self.ground_ids = np.array([geom for geom in env.robot_geom_ids if env.initial_mj_model.geom_contype[geom] or env.initial_mj_model.geom_conaffinity[geom]])
        self.ground_types = tuple(env.initial_mj_model.geom_type[self.ground_ids])
        if not all(kind in (mujoco.mjtGeom.mjGEOM_SPHERE, mujoco.mjtGeom.mjGEOM_CAPSULE) for kind in self.ground_types):
            raise ValueError("Online tracking ground geometry requires the named G1 sphere and capsule collision model")
        self.ground_bodies = jnp.asarray(env.initial_mj_model.geom_bodyid[self.ground_ids])
        self.geom_pos = jnp.asarray(env.initial_mj_model.geom_pos[self.ground_ids])
        self.geom_quat = jnp.asarray(env.initial_mj_model.geom_quat[self.ground_ids])
        self.geom_size = jnp.asarray(env.initial_mj_model.geom_size[self.ground_ids])
        self.capsule = jnp.asarray(env.initial_mj_model.geom_type[self.ground_ids] == mujoco.mjtGeom.mjGEOM_CAPSULE)


    def transforms(self, qpos, scale):
        positions = [jnp.zeros(3, qpos.dtype)]
        rotations = [Rotation.from_quat(jnp.array([0.0, 0.0, 0.0, 1.0], dtype=qpos.dtype))]
        for body in range(1, len(self.parent_ids)):
            joint = self.joint_ids[body]
            if self.joint_counts[body] and self.joint_types[joint] == mujoco.mjtJoint.mjJNT_FREE:
                address = self.qpos_addresses[joint]
                positions.append(qpos[address:address + 3])
                rotations.append(Rotation.from_quat(qpos[address + 3:address + 7][self.env.quaternion_wxyz_to_xyzw_indices]))
            else:
                local_pos = self.body_pos[body] * (scale if self.robot_bodies[body] else 1.0)
                rest = Rotation.from_quat(self.body_quat[body][self.env.quaternion_wxyz_to_xyzw_indices])
                local_rotation = rest
                if self.joint_counts[body]:
                    local_rotation = rest * Rotation.from_rotvec(self.joint_axis[joint] * qpos[self.qpos_addresses[joint]])
                    pivot = self.joint_pos[joint] * scale
                    local_pos = local_pos + rest.apply(pivot) - local_rotation.apply(pivot)
                positions.append(positions[self.parent_ids[body]] + rotations[self.parent_ids[body]].apply(local_pos))
                rotations.append(rotations[self.parent_ids[body]] * local_rotation)
        return jnp.stack(positions), jnp.stack([rotation.as_quat()[self.env.quaternion_xyzw_to_wxyz_indices] for rotation in rotations])


    @partial(jax.jit, static_argnums=(0,))
    def ground(self, qpos, scale):
        positions, quaternions = self.transforms(qpos, scale)
        body_rotation = Rotation.from_quat(quaternions[self.ground_bodies][:, self.env.quaternion_wxyz_to_xyzw_indices])
        centers = positions[self.ground_bodies] + body_rotation.apply(self.geom_pos * scale)
        geom_rotation = body_rotation * Rotation.from_quat(self.geom_quat[:, self.env.quaternion_wxyz_to_xyzw_indices])
        axis_z = geom_rotation.as_matrix()[:, 2, 2]
        return centers[:, 2] - jnp.mean(scale) * (self.geom_size[:, 0] + jnp.where(self.capsule, self.geom_size[:, 1] * jnp.abs(axis_z), 0.0))
