import jax.numpy as jnp


class BelowGroundAndPowerTermination:
    def __init__(self, env, power_history_length=3):
        self.env = env

        self.power_limit_watt = self.env.robot_config["power_limit_watt"]
        self.power_history_length = power_history_length if self.power_limit_watt else 0


    def setup(self, internal_state):
        internal_state["power_history"] = jnp.zeros(self.power_history_length)


    def should_terminate(self, data, internal_state, torques):
        below_ground = data.qpos[2] < 0.0

        if self.power_limit_watt:
            power = jnp.sum(abs(torques) * abs(data.qvel[6:]))
            internal_state["power_history"] = jnp.roll(internal_state["power_history"], shift=-1)
            internal_state["power_history"] = internal_state["power_history"].at[-1].set(power)
            power_limit_reached = jnp.all(internal_state["power_history"] > self.power_limit_watt)
        else:
            power_limit_reached = False

        return below_ground | power_limit_reached
