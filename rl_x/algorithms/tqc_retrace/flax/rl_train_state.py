import flax
from flax.training.train_state import TrainState


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict
