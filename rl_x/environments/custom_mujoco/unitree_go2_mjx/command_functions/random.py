import jax
import jax.numpy as jnp


class RandomCommands:
    def __init__(self, env,
                 min_x_velocity=-1.0, max_x_velocity=1.0,
                 min_y_velocity=-1.0, max_y_velocity=1.0,
                 min_yaw_velocity=-1.0, max_yaw_velocity=1.0,
                 zero_clip=False, zero_clip_threshold=0.1):
        self.env = env
        self.min_x_velocity = min_x_velocity
        self.max_x_velocity = max_x_velocity
        self.min_y_velocity = min_y_velocity
        self.max_y_velocity = max_y_velocity
        self.min_yaw_velocity = min_yaw_velocity
        self.max_yaw_velocity = max_yaw_velocity
        self.zero_clip = zero_clip
        self.zero_clip_threshold = zero_clip_threshold

        self.min_velocities = jnp.array([self.min_x_velocity, self.min_y_velocity, self.min_yaw_velocity])
        self.max_velocities = jnp.array([self.max_x_velocity, self.max_y_velocity, self.max_yaw_velocity])


    def init(self, internal_state):
        return


    def get_next_command(self, data, internal_state, should_sample_commands, subkey):
        goal_velocities = jax.random.uniform(subkey, (3,), minval=self.min_velocities, maxval=self.max_velocities)

        if self.zero_clip:
            goal_velocities = jnp.where(jnp.abs(goal_velocities) < self.zero_clip_threshold, 0.0, goal_velocities)

        internal_state["goal_velocities"] = jnp.where(should_sample_commands, goal_velocities, internal_state["goal_velocities"])
