from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from rl_x.environments.custom_mujoco.ant.create_env import create_env
from rl_x.environments.custom_mujoco.ant.default_config import get_config
from rl_x.environments.custom_mujoco.ant.general_properties import GeneralProperties


CUSTOM_MUJOCO_ANT = extract_environment_name_from_file(__file__)
register_environment(CUSTOM_MUJOCO_ANT, get_config, create_env, GeneralProperties)
