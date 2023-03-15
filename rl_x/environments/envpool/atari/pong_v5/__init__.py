from rl_x.environments.environment_manager import register_environment
from rl_x.environments.envpool.atari.pong_v5.create_env import create_env
from rl_x.environments.envpool.atari.pong_v5.default_config import get_config


environment_name = __file__.split("rl_x/environments/")[1].split("/__init__.py")[0].replace("/", ".")

ENVPOOL_ATARI_PONG_V5 = environment_name
register_environment(ENVPOOL_ATARI_PONG_V5, get_config, create_env)
