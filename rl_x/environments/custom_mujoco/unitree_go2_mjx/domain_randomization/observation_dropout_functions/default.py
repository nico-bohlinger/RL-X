import jax
import jax.numpy as jnp


class DefaultDRObservationDropout:
    def __init__(self, env):
        self.env = env
        
        self.dynamic_dropout_chance = env.robot_config["locomotion_envs"]["default"]["domain_randomization"]["observation_dropout"]["dynamic_chance"]

        self.dynamic_observation_indices = self.env.joint_observation_indices + self.env.feet_observation_indices
        self.nr_dynamic_observations = len(self.dynamic_observation_indices)


    def modify_observation(self, observation, key):
        keys = jax.random.split(key, self.nr_dynamic_observations)

        for i, idx in enumerate(self.dynamic_observation_indices):
            observation = observation.at[idx].set(jnp.where(jax.random.uniform(keys[i]) < self.dynamic_dropout_chance, 0.0, observation[idx]))

        return observation
