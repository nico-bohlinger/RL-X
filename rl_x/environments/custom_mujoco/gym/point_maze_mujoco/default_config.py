from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 1
    config.render = False
    config.device = "cpu"
    config.copy_train_env_for_eval = True

    config.horizon = 100
    config.reward_style = "dense"
    config.flipped = False
    config.success_radius = 0.1

    return config
