from os import sep as slash
from rl_x.environments.environment import Environment


_environments = {}


def extract_environment_name_from_file(file_name):
    return file_name.split(f"environments{slash}")[1].split(f"{slash}__init__.py")[0].replace(slash, ".")


def register_environment(name, get_default_config, create_env, general_properties):
    _environments[name] = Environment(name, get_default_config, create_env, general_properties)


def get_environment_config(environment_name):
    return _environments[environment_name].get_default_config(environment_name)


def get_environment_create_env(environment_name):
    return _environments[environment_name].create_env


def get_environment_general_properties(environment_name):
    return _environments[environment_name].general_properties
