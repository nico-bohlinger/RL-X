import flax
from flax.training.train_state import TrainState


class PolicyTrainState(TrainState):
    target_params: flax.core.FrozenDict
    env_params: flax.core.FrozenDict


class CriticTrainState(TrainState):
    target_params: flax.core.FrozenDict
