from rl_x.algorithms.algorithm_manager import Algorithm
from rl_x.environments.environment_manager import Environment
from rl_x.runner.runner_mode import RunnerMode
from rl_x.runner.runner import Runner


ALGORITHM = Algorithm.PPO_PYTORCH
ENVIRONMENT = Environment.ENVPOOL_HUMANOID_V4
RUNNER_MODE = RunnerMode.RUN_EXPERIMENT


if __name__ == "__main__":
    runner = Runner(ALGORITHM, ENVIRONMENT)
    runner.run(RUNNER_MODE)
