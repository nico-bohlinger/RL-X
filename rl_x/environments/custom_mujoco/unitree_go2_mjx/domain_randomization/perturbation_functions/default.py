import jax
import jax.numpy as jnp


class DefaultDRPerturbation:
    def __init__(self, env):
        self.env = env

        self.push_velocity_x_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["push_velocity_x_min"]
        self.push_velocity_x_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["push_velocity_x_max"]
        self.push_velocity_y_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["push_velocity_y_min"]
        self.push_velocity_y_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["push_velocity_y_max"]
        self.push_velocity_z_min = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["push_velocity_z_min"]
        self.push_velocity_z_max = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["push_velocity_z_max"]
        self.add_chance = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["add_chance"]
        self.additive_multiplier = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["perturbation"]["additive_multiplier"]

        self.vel_min = jnp.array([self.push_velocity_x_min, self.push_velocity_y_min, self.push_velocity_z_min])
        self.vel_max = jnp.array([self.push_velocity_x_max, self.push_velocity_y_max, self.push_velocity_z_max])


    def sample(self, data, should_randomize, key):
        perturbation_key, add_key = jax.random.split(key, 2)

        linear_velocity_perturbation = jax.random.uniform(perturbation_key, (3,), minval=self.vel_min, maxval=self.vel_max)
        linear_velocity_perturbation = jnp.where(jax.random.uniform(add_key) < self.add_chance, data.qvel[0:3] + linear_velocity_perturbation * self.additive_multiplier, linear_velocity_perturbation)
        linear_velocity = jnp.where(should_randomize, linear_velocity_perturbation, data.qvel[0:3])
        data = data.replace(qvel=jnp.concatenate([linear_velocity, data.qvel[3:]]))
        
        return data
