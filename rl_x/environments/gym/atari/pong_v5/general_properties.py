from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.environments.data_interface_type import DataInterfaceType


class GeneralProperties:
    observation_space_type = ObservationSpaceType.IMAGES
    action_space_type = ActionSpaceType.DISCRETE
    data_interface_type = DataInterfaceType.NUMPY
