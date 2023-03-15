from rl_x.environments.environment_manager import register_environment
from rl_x.environments.envpool.dmc.humanoid_run_v1.create_env import create_env
from rl_x.environments.envpool.dmc.humanoid_run_v1.default_config import get_config


environment_name = __file__.split("rl_x/environments/")[1].split("/__init__.py")[0].replace("/", ".")

ENVPOOL_DMC_HUMANOID_RUN_V1 = environment_name
register_environment(ENVPOOL_DMC_HUMANOID_RUN_V1, get_config, create_env)
