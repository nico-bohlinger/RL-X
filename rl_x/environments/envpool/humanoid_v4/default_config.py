from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.environment_name = environment_name

    config.seed = 1
    config.nr_envs = 1
    config.dones_at_timeout = False

    return config
