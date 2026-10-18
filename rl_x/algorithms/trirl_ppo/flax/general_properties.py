from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.environments.data_interface_type import DataInterfaceType
from rl_x.algorithms.deep_learning_framework_type import DeepLearningFrameworkType


class GeneralProperties:
    observation_space_types = [ObservationSpaceType.FLAT_VALUES]
    action_space_types = [ActionSpaceType.CONTINUOUS]
    data_interface_types = [DataInterfaceType.NUMPY]

    deep_learning_framework_type = DeepLearningFrameworkType.JAX
