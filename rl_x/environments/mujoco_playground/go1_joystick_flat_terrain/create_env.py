import jax
from mujoco_playground import registry

from rl_x.environments.gym.mujoco.humanoid_v4.general_properties import GeneralProperties


def create_env(config):
    mbp_env_config = registry.get_default_config(config.environment.type)
    env = registry.load(config.environment.type, config=mbp_env_config)

    env.reset = jax.jit(jax.vmap(env.reset))
    env.step = jax.jit(jax.vmap(env.step))

    key = jax.random.PRNGKey(1)
    key, reset_key = jax.random.split(key, 2)
    nr_envs = 2
    reset_key = jax.random.split(reset_key, nr_envs)
    env_state = env.reset(reset_key)
    key, action_key = jax.random.split(key, 2)
    env_state = env.step(env_state, jax.random.uniform(action_key, (nr_envs, 12), minval=-1.0, maxval=1.0))
    print("loaded")
    exit()

    env.general_properties = GeneralProperties

    return env
