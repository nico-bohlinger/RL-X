import jax
import jax.numpy as jnp


class DefaultDRMuJoCoModel:
    def __init__(self, env):
        self.env = env

        self.friction_tangential_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["friction_tangential_min"]
        self.friction_tangential_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["friction_tangential_max"]
        self.friction_torsional_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["friction_torsional_min"]
        self.friction_torsional_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["friction_torsional_max"]
        self.friction_rolling_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["friction_rolling_min"]
        self.friction_rolling_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["friction_rolling_max"]
        self.damping_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["damping_min"]
        self.damping_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["damping_max"]
        self.stiffness_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["stiffness_min"]
        self.stiffness_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["stiffness_max"]
        self.gravity_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["gravity_min"]
        self.gravity_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["gravity_max"]
        self.add_trunk_mass_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["add_trunk_mass_min"]
        self.add_trunk_mass_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["add_trunk_mass_max"]
        self.add_com_displacement_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["add_com_displacement_min"]
        self.add_com_displacement_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["add_com_displacement_max"]
        self.foot_size_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["foot_size_min"]
        self.foot_size_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["foot_size_max"]
        self.joint_damping_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_damping_min"]
        self.joint_damping_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_damping_max"]
        self.joint_armature_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_armature_min"]
        self.joint_armature_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_armature_max"]
        self.joint_stiffness_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_stiffness_min"]
        self.joint_stiffness_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_stiffness_max"]
        self.joint_friction_loss_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_friction_loss_min"]
        self.joint_friction_loss_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["mujoco_model"]["joint_friction_loss_max"]

        self.default_trunk_mass = self.env.initial_mjx_model.body_mass[1]
        self.default_trunk_inertia = self.env.initial_mjx_model.body_inertia[1]
        self.default_trunk_com = self.env.initial_mjx_model.body_ipos[1]
        self.default_joint_damping = self.env.initial_mjx_model.dof_damping[6:]
        self.default_joint_armature = self.env.initial_mjx_model.dof_armature[6:]
        self.default_joint_stiffness = self.env.initial_mjx_model.jnt_stiffness[1:]
        self.default_joint_frictionloss = self.env.initial_mjx_model.dof_frictionloss[6:]


    def sample(self, mjx_model, should_randomize, key):
        keys = jax.random.split(key, 11)

        interpolation = jax.random.uniform(keys[0])
        sampled_friction_tangential = self.friction_tangential_min + (self.friction_tangential_max - self.friction_tangential_min) * interpolation
        sampled_friction_torsional = self.friction_torsional_min + (self.friction_torsional_max - self.friction_torsional_min) * interpolation
        sampled_friction_rolling = self.friction_rolling_min + (self.friction_rolling_max - self.friction_rolling_min) * interpolation
        geom_friction = mjx_model.geom_friction.at[self.env.foot_geom_indices].set(jnp.array([sampled_friction_tangential, sampled_friction_torsional, sampled_friction_rolling]))

        interpolation = jax.random.uniform(keys[1])
        sampled_damping = self.damping_min + (self.damping_max - self.damping_min) * interpolation
        sampled_stiffness = self.stiffness_min + (self.stiffness_max - self.stiffness_min) * interpolation
        geom_solref = mjx_model.geom_solref.at[:, 0].set(-sampled_stiffness)
        geom_solref = geom_solref.at[:, 1].set(-sampled_damping)

        sampled_gravity = jax.random.uniform(keys[2], minval=self.gravity_min, maxval=self.gravity_max)
        opt_gravity = mjx_model.opt.gravity.at[2].set(-sampled_gravity)

        trunk_mass = self.default_trunk_mass + jax.random.uniform(keys[3], minval=self.add_trunk_mass_min, maxval=self.add_trunk_mass_max)
        body_mass = mjx_model.body_mass.at[1].set(trunk_mass)
        trunk_inertia = self.default_trunk_inertia * (trunk_mass / self.default_trunk_mass)
        body_inertia = mjx_model.body_inertia.at[1].set(trunk_inertia)

        trunk_com = self.default_trunk_com + jax.random.uniform(keys[4], minval=self.add_com_displacement_min, maxval=self.add_com_displacement_max)
        body_ipos = mjx_model.body_ipos.at[1].set(trunk_com)

        foot_size = jax.random.uniform(keys[5], minval=self.foot_size_min, maxval=self.foot_size_max)
        geom_size = mjx_model.geom_size.at[self.env.foot_geom_indices, 0].set(foot_size)

        random_joint_attributes = jax.random.uniform(keys[6]) < 0.5

        joint_damping = jax.random.uniform(keys[7], minval=self.joint_damping_min, maxval=self.joint_damping_max, shape=self.default_joint_damping.shape)
        dof_damping = mjx_model.dof_damping.at[6:].set(joint_damping)
        dof_damping = jnp.where(random_joint_attributes, dof_damping, self.env.initial_mjx_model.dof_damping)

        joint_armature = jax.random.uniform(keys[8], minval=self.joint_armature_min, maxval=self.joint_armature_max, shape=self.default_joint_armature.shape)
        dof_armature = mjx_model.dof_armature.at[6:].set(joint_armature)
        dof_armature = jnp.where(random_joint_attributes, dof_armature, self.env.initial_mjx_model.dof_armature)

        joint_stiffness = jax.random.uniform(keys[9], minval=self.joint_stiffness_min, maxval=self.joint_stiffness_max, shape=self.default_joint_stiffness.shape)
        jnt_stiffness = mjx_model.jnt_stiffness.at[1:].set(joint_stiffness)
        jnt_stiffness = jnp.where(random_joint_attributes, jnt_stiffness, self.env.initial_mjx_model.jnt_stiffness)

        joint_frictionloss = jax.random.uniform(keys[10], minval=self.joint_friction_loss_min, maxval=self.joint_friction_loss_max, shape=self.default_joint_frictionloss.shape)
        dof_frictionloss = mjx_model.dof_frictionloss.at[6:].set(joint_frictionloss)
        dof_frictionloss = jnp.where(random_joint_attributes, dof_frictionloss, self.env.initial_mjx_model.dof_frictionloss)

        new_mjx_model = mjx_model.tree_replace(
            {
                "geom_friction": geom_friction,
                "geom_solref": geom_solref,
                "opt.gravity": opt_gravity,
                "body_mass": body_mass,
                "body_inertia": body_inertia,
                "body_ipos": body_ipos,
                "geom_size": geom_size,
                "dof_damping": dof_damping,
                "dof_armature": dof_armature,
                "jnt_stiffness": jnt_stiffness,
                "dof_frictionloss": dof_frictionloss
            }
        )
        mjx_model = jax.lax.cond(should_randomize, lambda x: new_mjx_model, lambda x: mjx_model, None)

        return mjx_model
