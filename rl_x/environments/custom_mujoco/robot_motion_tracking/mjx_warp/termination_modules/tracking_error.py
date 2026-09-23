import jax.numpy as jnp


class TrackingErrorTermination:
    def __init__(self, env):
        self.env = env
        self.body_indices = [env.env_config["tracked_bodies"].index(name) for name in env.env_config["termination"]["tracking_error"]["tracked_body_names"]]

        self.anchor_height_threshold = env.env_config["termination"]["tracking_error"]["anchor_height_threshold"]
        self.anchor_orientation_threshold = env.env_config["termination"]["tracking_error"]["anchor_orientation_threshold"]
        self.body_height_threshold = env.env_config["termination"]["tracking_error"]["body_height_threshold"]
        self.object_position_threshold = env.env_config["termination"]["tracking_error"]["object_position_threshold"]
        self.object_orientation_threshold = env.env_config["termination"]["tracking_error"]["object_orientation_threshold"]


    def should_terminate(self, data, internal_state, info):
        anchor_height_difference = jnp.abs(internal_state["anchor_position"][..., 2] - internal_state["reference"]["anchor_position"][..., 2])
        anchor_height_failure = anchor_height_difference > self.anchor_height_threshold

        reference_gravity = internal_state["reference"]["anchor_rotation"].inv().apply(jnp.asarray([0.0, 0.0, -1.0]))
        anchor_orientation_difference = jnp.abs(internal_state["projected_gravity"][..., 2] - reference_gravity[..., 2])
        anchor_orientation_failure = anchor_orientation_difference > self.anchor_orientation_threshold

        body_height_difference = jnp.abs(internal_state["body_positions"][..., 2] - internal_state["reference"]["aligned_body_positions"][..., 2])
        body_height_failure = jnp.any(body_height_difference[..., self.body_indices] > self.body_height_threshold, axis=-1)

        terminated = anchor_height_failure | anchor_orientation_failure | body_height_failure
        if self.env.motion_library.has_object:
            object_position_error_squared = jnp.sum(jnp.square(internal_state["reference"]["qpos"][..., self.env.object_position_qpos_slice] - data.qpos[..., self.env.object_position_qpos_slice]), axis=-1)
            object_position_failure = object_position_error_squared > self.object_position_threshold**2
            object_orientation_error_squared = jnp.square((internal_state["reference"]["object_rotation"] * internal_state["object_rotation"].inv()).magnitude())
            object_orientation_failure = object_orientation_error_squared > self.object_orientation_threshold**2
            terminated |= object_position_failure | object_orientation_failure

        info["termination/anchor_height"] = anchor_height_failure
        info["termination/anchor_orientation"] = anchor_orientation_failure
        info["termination/body_height"] = body_height_failure
        if self.env.motion_library.has_object:
            info["termination/object_position"] = object_position_failure
            info["termination/object_orientation"] = object_orientation_failure

        return terminated
