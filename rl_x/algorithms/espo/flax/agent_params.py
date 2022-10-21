import flax


@flax.struct.dataclass
class AgentParams:
    policy_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict
