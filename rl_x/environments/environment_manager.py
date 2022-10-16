from rl_x.environments.environment import Environment


_environments = {}


def register_environment(name, get_default_config, create_env):
    _environments[name] = Environment(name, get_default_config, create_env)


def get_environment_config(environment_name):
    return _environments[environment_name].get_default_config(environment_name)


def get_environment_create_env(environment_name):
    return _environments[environment_name].create_env
