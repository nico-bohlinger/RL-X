import jax
import jax.numpy as jnp


class DefaultDRUnseenRobotFunction:
    def __init__(self, env):
        self.env = env

        self.mass_inertia_factor = env.env_config["domain_randomization"]["unseen_robot"]["mass_inertia_factor"]
        self.com_factor = env.env_config["domain_randomization"]["unseen_robot"]["com_factor"]
        self.body_position_factor = env.env_config["domain_randomization"]["unseen_robot"]["body_position_factor"]
        self.joint_damping_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_damping_factor"]
        self.joint_armature_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_armature_factor"]
        self.joint_stiffness_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_stiffness_factor"]
        self.joint_friction_loss_factor = env.env_config["domain_randomization"]["unseen_robot"]["joint_friction_loss_factor"]
        self.p_gain_factor = env.env_config["domain_randomization"]["unseen_robot"]["p_gain_factor"]
        self.d_gain_factor = env.env_config["domain_randomization"]["unseen_robot"]["d_gain_factor"]
        self.position_offset = env.env_config["domain_randomization"]["unseen_robot"]["position_offset"]

        self.nr_bodies = self.env.initial_mjx_model.body_mass[1:].shape[0]
        self.nr_joint_dampings = self.env.initial_mjx_model.dof_damping[6:].shape[0]
        self.nr_joint_armatures = self.env.initial_mjx_model.dof_armature[6:].shape[0]
        self.nr_joint_stiffnesses = self.env.initial_mjx_model.jnt_stiffness[1:].shape[0]
        self.nr_joint_frictionlosses = self.env.initial_mjx_model.dof_frictionloss[6:].shape[0]


    def init(self, internal_state):
        internal_state["mass_inertia_noise_factors"] = jnp.ones(self.nr_bodies)
        internal_state["com_noise_factors"] = jnp.ones((self.nr_bodies, 3))
        internal_state["body_position_noise_factors"] = jnp.ones((self.nr_bodies, 3))
        internal_state["joint_damping_noise_factors"] = jnp.ones(self.nr_joint_dampings)
        internal_state["joint_armature_noise_factors"] = jnp.ones(self.nr_joint_armatures)
        internal_state["joint_stiffness_noise_factors"] = jnp.ones(self.nr_joint_stiffnesses)
        internal_state["joint_friction_loss_noise_factors"] = jnp.ones(self.nr_joint_frictionlosses)
        internal_state["p_gain_noise_factors"] = jnp.ones(self.env.nr_actuator_joints)
        internal_state["d_gain_noise_factors"] = jnp.ones(self.env.nr_actuator_joints)
        internal_state["position_offsets"] = jnp.zeros(self.env.nr_actuator_joints)


    def sample(self, internal_state, should_randomize, key):
        keys = jax.random.split(key, 10)

        internal_state["mass_inertia_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[0], minval=-self.mass_inertia_factor, maxval=self.mass_inertia_factor, shape=(self.nr_bodies,)), internal_state["mass_inertia_noise_factors"])
        internal_state["com_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[1], minval=-self.com_factor, maxval=self.com_factor, shape=(self.nr_bodies, 3)), internal_state["com_noise_factors"])
        internal_state["body_position_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[2], minval=-self.body_position_factor, maxval=self.body_position_factor, shape=(self.nr_bodies, 3)), internal_state["body_position_noise_factors"])
        internal_state["joint_damping_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[3], minval=-self.joint_damping_factor, maxval=self.joint_damping_factor, shape=(self.nr_joint_dampings,)), internal_state["joint_damping_noise_factors"])
        internal_state["joint_armature_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[4], minval=-self.joint_armature_factor, maxval=self.joint_armature_factor, shape=(self.nr_joint_armatures,)), internal_state["joint_armature_noise_factors"])
        internal_state["joint_stiffness_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[5], minval=-self.joint_stiffness_factor, maxval=self.joint_stiffness_factor, shape=(self.nr_joint_stiffnesses,)), internal_state["joint_stiffness_noise_factors"])
        internal_state["joint_friction_loss_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[6], minval=-self.joint_friction_loss_factor, maxval=self.joint_friction_loss_factor, shape=(self.nr_joint_frictionlosses,)), internal_state["joint_friction_loss_noise_factors"])
        internal_state["p_gain_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[7], minval=-self.p_gain_factor, maxval=self.p_gain_factor, shape=(self.env.nr_actuator_joints,)), internal_state["p_gain_noise_factors"])
        internal_state["d_gain_noise_factors"] = jnp.where(should_randomize, 1 + internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[8], minval=-self.d_gain_factor, maxval=self.d_gain_factor, shape=(self.env.nr_actuator_joints,)), internal_state["d_gain_noise_factors"])
        internal_state["position_offsets"] = jnp.where(should_randomize, internal_state["env_curriculum_coeff"] * jax.random.uniform(keys[9], minval=-self.position_offset, maxval=self.position_offset, shape=(self.env.nr_actuator_joints,)), internal_state["position_offsets"])
