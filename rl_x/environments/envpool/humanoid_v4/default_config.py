from ml_collections import config_dict


def get_config(algorithm, environment):
    config = config_dict.ConfigDict()

    config.environment_name = environment.name

    return config
