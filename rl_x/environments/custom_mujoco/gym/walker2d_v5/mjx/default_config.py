from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 4096
    config.render = False
    config.device = "gpu"
    config.copy_train_env_for_eval = True

    return config
