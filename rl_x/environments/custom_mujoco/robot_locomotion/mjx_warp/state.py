from typing import Any, Dict
import jax
from flax import struct
from mujoco import mjx


@struct.dataclass
class State:
    mjx_model: mjx.Model
    data: mjx.Data
    next_observation: jax.Array
    actual_next_observation: jax.Array
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]
    info_episode_store: Dict[str, Any]
    internal_state: Dict[str, Any]
    key: jax.random.PRNGKey
