from rl_x.environments.custom_mujoco.ant.warp_torch.environment import Ant
from rl_x.environments.custom_mujoco.ant.warp_torch.general_properties import GeneralProperties
from rl_x.environments.custom_mujoco.ant.warp_torch.wrappers import RLXInfo


def create_train_and_eval_env(config):
    train_env = RLXInfo(Ant(config.environment))
    train_env.general_properties = GeneralProperties

    if config.environment.copy_train_env_for_eval:
        return train_env, train_env

    eval_env = RLXInfo(Ant(config.environment))
    eval_env.general_properties = GeneralProperties

    return train_env, eval_env
