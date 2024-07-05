from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from rl_x.environments.custom_mujoco.brax.create_env import create_env
from rl_x.environments.custom_mujoco.brax.default_config import get_config
from rl_x.environments.custom_mujoco.brax.general_properties import GeneralProperties


BRAX = extract_environment_name_from_file(__file__)
register_environment(BRAX, get_config, create_env, GeneralProperties)
