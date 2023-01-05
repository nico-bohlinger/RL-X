from rl_x.environments.environment_manager import register_environment
from rl_x.environments.envpool.mujoco.humanoid_v4.create_env import create_env
from rl_x.environments.envpool.mujoco.humanoid_v4.default_config import get_config


ENVPOOL_MUJOCO_HUMANOID_V4 = "EnvPool Mujoco Humanoid-v4"
register_environment(ENVPOOL_MUJOCO_HUMANOID_V4, get_config, create_env)
