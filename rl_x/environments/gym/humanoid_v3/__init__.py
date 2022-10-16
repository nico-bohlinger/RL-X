from rl_x.environments.environment_manager import register_environment
from rl_x.environments.gym.humanoid_v3.create_env import create_env
from rl_x.environments.gym.humanoid_v3.default_config import get_config


GYM_HUMANOID_V3 = "Gym Humanoid-v3"
register_environment(GYM_HUMANOID_V3, get_config, create_env)
