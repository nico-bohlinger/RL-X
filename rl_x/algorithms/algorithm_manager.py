from rl_x.algorithms.algorithm import Algorithm
from rl_x.environments.environment import Environment

from rl_x.algorithms.sac.flax.sac import SAC as sac_flax_model_class
from rl_x.algorithms.sac.flax.default_config import get_config as sac_flax_get_config
from rl_x.algorithms.sac.pytorch.sac import SAC as sac_pytorch_model_class
from rl_x.algorithms.sac.pytorch.default_config import get_config as sac_pytorch_get_config
from rl_x.algorithms.ppo.pytorch.ppo import PPO as ppo_pytorch_model_class
from rl_x.algorithms.ppo.pytorch.default_config import get_config as ppo_pytorch_get_config


class AlgorithmManager:
    def get_default_config(algorithm: Algorithm, environment: Environment):
        if algorithm == Algorithm.SAC_FLAX:
            get_config = sac_flax_get_config
        elif algorithm == Algorithm.SAC_PYTORCH:
            get_config = sac_pytorch_get_config
        elif algorithm == Algorithm.PPO_PYTORCH:
            get_config = ppo_pytorch_get_config
        else:
            raise NotImplementedError
            
        return get_config(algorithm, environment)
    

    def get_model_class(algorithm: Algorithm):
        if algorithm == Algorithm.SAC_FLAX:
            return sac_flax_model_class
        elif algorithm == Algorithm.SAC_PYTORCH:
            return sac_pytorch_model_class
        elif algorithm == Algorithm.PPO_PYTORCH:
            return ppo_pytorch_model_class
        else:
            raise NotImplementedError
