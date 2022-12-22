import jax.numpy as jnp
import flax


@flax.struct.dataclass
class Storage:
    states: jnp.array
    actions: jnp.array
    log_probs: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
