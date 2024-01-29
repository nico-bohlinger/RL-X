from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 1
    config.async_skip_percentage = 0.0
    config.ip = "localhost"
    config.port = 5555

    return config
