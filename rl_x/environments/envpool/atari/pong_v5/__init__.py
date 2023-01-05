from rl_x.environments.environment_manager import register_environment
from rl_x.environments.envpool.atari.pong_v5.create_env import create_env
from rl_x.environments.envpool.atari.pong_v5.default_config import get_config


ENVPOOL_ATARI_PONG_V5 = "Envpool Atari Pong-v5"
register_environment(ENVPOOL_ATARI_PONG_V5, get_config, create_env)
