from rl_x.environments.environment_manager import extract_environment_name_from_file, register_environment
from rl_x.environments.envpool.dmc.humanoid_run_v1.create_env import create_env
from rl_x.environments.envpool.dmc.humanoid_run_v1.default_config import get_config
from rl_x.environments.envpool.dmc.humanoid_run_v1.general_properties import GeneralProperties


ENVPOOL_DMC_HUMANOID_RUN_V1 = extract_environment_name_from_file(__file__)
register_environment(ENVPOOL_DMC_HUMANOID_RUN_V1, get_config, create_env, GeneralProperties)
