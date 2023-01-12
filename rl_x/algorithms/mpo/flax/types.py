import jax
import jax.numpy as jnp
import flax
from flax import struct
import optax


class AgentParams(struct.PyTreeNode):
    policy_params: flax.core.FrozenDict
    critic_params: flax.core.FrozenDict


class DualParams(struct.PyTreeNode):
    log_temperature: jnp.ndarray
    log_alpha_mean: jnp.ndarray
    log_alpha_stddev: jnp.ndarray
    log_penalty_temperature: jnp.ndarray


class TrainingState(struct.PyTreeNode):
    agent_params: AgentParams
    agent_target_params: AgentParams
    dual_params: DualParams
    agent_optimizer_state: optax.OptState
    dual_optimizer_state: optax.OptState
    steps: int
