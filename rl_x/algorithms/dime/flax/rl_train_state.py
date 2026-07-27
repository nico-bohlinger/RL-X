from flax.training.train_state import TrainState


class RLTrainState(TrainState):
    batch_stats: dict
