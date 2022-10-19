import flax


@flax.struct.dataclass
class AgentParams:
    actor_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict
