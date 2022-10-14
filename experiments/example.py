from rl_x.algorithms.algorithm_manager import Algorithm
from rl_x.environments.environment_manager import Environment
from rl_x.runner import Runner


ALGORITHM = Algorithm.SAC_FLAX
ENVIRONMENT = Environment.ENVPOOL_HUMANOID_V4

runner = Runner(ALGORITHM, ENVIRONMENT)
runner.run()
