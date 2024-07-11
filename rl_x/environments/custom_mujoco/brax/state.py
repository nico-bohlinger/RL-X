from typing import Any, Dict
import jax
from flax import struct
from mujoco import mjx


@struct.dataclass
class State:
    data: mjx.Data
    next_observation: jax.Array
    actual_next_observation: jax.Array
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = struct.field(default_factory=dict)
