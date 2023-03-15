from rl_x.environments.environment_manager import register_environment
from rl_x.environments.gym.mujoco.humanoid_v4.create_env import create_env
from rl_x.environments.gym.mujoco.humanoid_v4.default_config import get_config


environment_name = __file__.split("rl_x/environments/")[1].split("/__init__.py")[0].replace("/", ".")

GYM_MUJOCO_HUMANOID_V4 = environment_name
register_environment(GYM_MUJOCO_HUMANOID_V4, get_config, create_env)
