from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.type = "Pong-v5"
    config.seed = 1
    config.nr_envs = 1

    return config
