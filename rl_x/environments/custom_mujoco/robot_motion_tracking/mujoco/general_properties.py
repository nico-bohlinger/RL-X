from rl_x.environments.action_space_type import ActionSpaceType
from rl_x.environments.observation_space_type import ObservationSpaceType
from rl_x.environments.data_interface_type import DataInterfaceType
from rl_x.environments.simulation_type import SimulationType


class GeneralProperties:
    observation_space_type = ObservationSpaceType.FLAT_VALUES
    action_space_type = ActionSpaceType.CONTINUOUS
    data_interface_type = DataInterfaceType.NUMPY

    simulation_type = SimulationType.DEFAULT
