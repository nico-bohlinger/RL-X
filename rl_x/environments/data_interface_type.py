from enum import Enum


class DataInterfaceType(Enum):
    LIST = 0
    NUMPY = 1
    TORCH = 2
    JAX = 3
