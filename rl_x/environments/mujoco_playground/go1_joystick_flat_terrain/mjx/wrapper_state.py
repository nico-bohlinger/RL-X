from typing import Any, Dict
import jax
from flax import struct


@struct.dataclass
class WrapperState:
    env_state: object
    next_observation: jax.Array
    actual_next_observation: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: Dict[str, Any]
    info_episode_store: Dict[str, Any]
