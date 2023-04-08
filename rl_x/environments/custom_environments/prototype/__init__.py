from rl_x.environments.environment_manager import register_environment
from rl_x.environments.custom_environments.prototype.create_env import create_env
from rl_x.environments.custom_environments.prototype.default_config import get_config


environment_name = __file__.split("rl_x/environments/")[1].split("/__init__.py")[0].replace("/", ".")

CUSTOM_ENVIRONMENT_PROTOTYPE = environment_name
register_environment(CUSTOM_ENVIRONMENT_PROTOTYPE, get_config, create_env)
