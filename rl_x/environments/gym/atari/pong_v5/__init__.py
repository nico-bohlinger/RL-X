from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from rl_x.environments.gym.atari.pong_v5.create_env import create_env
from rl_x.environments.gym.atari.pong_v5.default_config import get_config
from rl_x.environments.gym.atari.pong_v5.general_properties import GeneralProperties


GYM_ATARI_PONG_V5 = extract_environment_name_from_file(__file__)
register_environment(GYM_ATARI_PONG_V5, get_config, create_env, GeneralProperties)
