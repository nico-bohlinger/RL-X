from rl_x.environments.environment_manager import register_environment
from rl_x.environments.envpool.cart_pole_v1.create_env import create_env
from rl_x.environments.envpool.cart_pole_v1.default_config import get_config


ENVPOOL_CART_POLE_V1 = "Envpool CartPole-v1"
register_environment(ENVPOOL_CART_POLE_V1, get_config, create_env)
