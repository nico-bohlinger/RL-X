import jax
import jax.numpy as jnp


class DefaultDRJointDropout:
    def __init__(self, env):
        self.env = env

        self.dropout_open_chance = env.env_config["domain_randomization"]["joint_dropout"]["dropout_open_chance"]
        self.dropout_lock_chance = env.env_config["domain_randomization"]["joint_dropout"]["dropout_lock_chance"]


    def init(self, internal_state):
        internal_state["joint_dropout_open_mask"] = jnp.ones((self.env.nr_envs, self.env.nr_actuator_joints), dtype=bool)
        internal_state["joint_dropout_lock_mask"] = jnp.ones((self.env.nr_envs, self.env.nr_actuator_joints), dtype=bool)


    def sample(self, internal_state, mjx_model, should_randomize, key):
        nr_envs = self.env.nr_envs
        nr_actuator_joints = self.env.nr_actuator_joints
        open_key, lock_key = jax.random.split(key)
        curriculum_coeff = internal_state["env_curriculum_coeff"][:, None]

        internal_state["joint_dropout_open_mask"] = jnp.where(should_randomize[:, None],
            jax.random.binomial(open_key, n=1, p=(1 - curriculum_coeff * self.dropout_open_chance), shape=(nr_envs, nr_actuator_joints)) == 1.0,
            internal_state["joint_dropout_open_mask"]
        )

        internal_state["joint_dropout_lock_mask"] = jnp.where(should_randomize[:, None],
            jax.random.binomial(lock_key, n=1, p=(1 - curriculum_coeff * self.dropout_lock_chance), shape=(nr_envs, nr_actuator_joints)) == 1.0,
            internal_state["joint_dropout_lock_mask"]
        )

        internal_state["joint_dropout_mask"] = internal_state["joint_dropout_open_mask"] | internal_state["joint_dropout_lock_mask"]

        actuator_gainprm = internal_state["partial_actuator_gainprm_without_dropout"] * internal_state["joint_dropout_open_mask"]
        actuator_biasprm = internal_state["partial_actuator_biasprm_without_dropout"] * internal_state["joint_dropout_open_mask"][:, :, None]

        model_actuator_gainprm = mjx_model.actuator_gainprm
        model_actuator_biasprm = mjx_model.actuator_biasprm
        model_jnt_range = mjx_model.jnt_range
        if model_actuator_gainprm.ndim == 2:
            model_actuator_gainprm = jnp.broadcast_to(model_actuator_gainprm, (nr_envs,) + model_actuator_gainprm.shape)
        if model_actuator_biasprm.ndim == 2:
            model_actuator_biasprm = jnp.broadcast_to(model_actuator_biasprm, (nr_envs,) + model_actuator_biasprm.shape)
        if model_jnt_range.ndim == 2:
            model_jnt_range = jnp.broadcast_to(model_jnt_range, (nr_envs,) + model_jnt_range.shape)

        actuators_gainprm = model_actuator_gainprm.at[:, :, 0].set(actuator_gainprm)
        actuators_biasprm = model_actuator_biasprm.at[:, :, 1:3].set(actuator_biasprm)

        locked_actuator_joint_ranges_min = internal_state["actuator_joint_nominal_positions"] - 0.001
        locked_actuator_joint_ranges_max = internal_state["actuator_joint_nominal_positions"] + 0.001
        locked_actuator_joint_ranges = jnp.stack([locked_actuator_joint_ranges_min, locked_actuator_joint_ranges_max], axis=-1)
        actuator_joint_ranges = jnp.where(
            internal_state["joint_dropout_lock_mask"][:, :, None],
            internal_state["seen_joint_ranges"][:, self.env.actuator_joint_mask_joints - 1],
            locked_actuator_joint_ranges
        )
        jnt_range = model_jnt_range.at[:, self.env.actuator_joint_mask_joints].set(actuator_joint_ranges)

        mjx_model = mjx_model.tree_replace(
            {
                "actuator_gainprm": actuators_gainprm,
                "actuator_biasprm": actuators_biasprm,
                "jnt_range": jnt_range,
            }
        )

        return mjx_model
