import numpy as np
import jax
import jax.numpy as jnp

def prepare_expert_data(data_path, cutoff=1):
    dataset = dict()
    expert_files = np.load(data_path)

    def _flatten_feature_array(x):
        x = np.asarray(x)
        if x.ndim <= 2:
            return x
        return x.reshape(-1, x.shape[-1])

    def _flatten_scalar_array(x):
        return np.asarray(x).reshape(-1)

    states = _flatten_feature_array(expert_files["states"])
    actions = _flatten_feature_array(expert_files["actions"])

    cutoff = int(states.shape[0] / cutoff)

    dataset["states"] = states[:cutoff]
    dataset["actions"] = actions[:cutoff]

    try:
        dataset["next_actions"] = _flatten_feature_array(expert_files["next_actions"])[:cutoff]
        dataset["next_next_states"] = _flatten_feature_array(expert_files["next_next_states"])[:cutoff]
    except KeyError:
        print("Did not find next action or next next state.")

    try:
        dataset["next_states"] = _flatten_feature_array(expert_files["next_states"])[:cutoff]
        dataset["absorbing"] = _flatten_scalar_array(expert_files["absorbing"])[:cutoff]
    except KeyError as e:
        print("Warning Dataset: %s" % e)

    try:
        dataset["episode_returns"] = _flatten_scalar_array(expert_files["episode_returns"])[:cutoff]
        return dataset
    except KeyError:
        print("Warning Dataset: No episode returns. Falling back to step-based reward.")

    try:
        dataset["rewards"] = _flatten_scalar_array(expert_files["rewards"])[:cutoff]
        return dataset
    except KeyError:
        raise KeyError("The dataset has neither an episode nor a step-based reward!")



def expert_data_spec(num_samples, state_dim, action_dim):
    """
    Dummy spec for use with full jittting
    """
    return {
        "states": jax.ShapeDtypeStruct(
            shape=(num_samples, state_dim),
            dtype=jnp.float32,
        ),
        "actions": jax.ShapeDtypeStruct(
            shape=(num_samples, action_dim),
            dtype=jnp.float32,
        ),
        "next_states": jax.ShapeDtypeStruct(
            shape=(num_samples, state_dim),
            dtype=jnp.float32,
        ),
        "absorbing": jax.ShapeDtypeStruct(
            shape=(num_samples,),
            dtype=jnp.float32,
        ),
        "rewards": jax.ShapeDtypeStruct(
            shape=(num_samples,),
            dtype=jnp.float32,
        ),
    }