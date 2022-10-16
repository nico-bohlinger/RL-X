from rl_x.algorithms.algorithm import Algorithm


_algorithms = {}


def register_algorithm(name, get_default_config, get_model_class):
    _algorithms[name] = Algorithm(name, get_default_config, get_model_class)


def get_algorithm_config(algorithm_name):
    return _algorithms[algorithm_name].get_default_config(algorithm_name)


def get_algorithm_model_class(algorithm_name):
    return _algorithms[algorithm_name].get_model_class
