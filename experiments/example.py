from rl_x.runner.runner_mode import RunnerMode
from rl_x.runner.runner import Runner

# Algorithms
## PPO
from rl_x.algorithms.ppo.pytorch import PPO_PYTORCH
from rl_x.algorithms.ppo.torchscript import PPO_TORCHSCRIPT
from rl_x.algorithms.ppo.flax import PPO_FLAX
## ESPO
from rl_x.algorithms.espo.pytorch import ESPO_PYTORCH
from rl_x.algorithms.espo.torchscript import ESPO_TORCHSCRIPT
from rl_x.algorithms.espo.flax import ESPO_FLAX
## SAC
from rl_x.algorithms.sac.pytorch import SAC_PYTORCH
from rl_x.algorithms.sac.torchscript import SAC_TORCHSCRIPT
from rl_x.algorithms.sac.flax import SAC_FLAX
## REDQ
from rl_x.algorithms.redq.flax import REDQ_FLAX
## DroQ
from rl_x.algorithms.droq.flax import DROQ_FLAX
## TQC
from rl_x.algorithms.tqc.flax import TQC_FLAX
## TQC+DroQ
from rl_x.algorithms.tqc_droq.flax import TQC_DROQ_FLAX

# Environments
from rl_x.environments.envpool.humanoid_v4 import ENVPOOL_HUMANOID_V4
from rl_x.environments.envpool.cart_pole_v1 import ENVPOOL_CART_POLE_V1
from rl_x.environments.envpool.pong_v5 import ENVPOOL_PONG_V5
from rl_x.environments.gym.humanoid_v3 import GYM_HUMANOID_V3


ALGORITHM = PPO_PYTORCH
ENVIRONMENT = ENVPOOL_HUMANOID_V4
RUNNER_MODE = RunnerMode.RUN_EXPERIMENT


if __name__ == "__main__":
    runner = Runner(ALGORITHM, ENVIRONMENT)
    runner.run(RUNNER_MODE)
