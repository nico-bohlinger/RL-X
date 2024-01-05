from os import sep as slash
from rl_x.algorithms.algorithm import Algorithm


_algorithms = {}


def extract_algorithm_name_from_file(file_name):
    return file_name.split(f"algorithms{slash}")[1].split(f"{slash}__init__.py")[0].replace(slash, ".")


def register_algorithm(name, get_default_config, get_model_class, general_properties):
    _algorithms[name] = Algorithm(name, get_default_config, get_model_class, general_properties)


def get_algorithm_config(algorithm_name):
    return _algorithms[algorithm_name].get_default_config(algorithm_name)


def get_algorithm_model_class(algorithm_name):
    return _algorithms[algorithm_name].get_model_class


def get_algorithm_general_properties(algorithm_name):
    return _algorithms[algorithm_name].general_properties
