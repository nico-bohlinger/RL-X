import numpy as np


def prepare_expert_data(data_path, cutoff=1):
    def flatten_feature_array(value):
        value = np.asarray(value, dtype=np.float32)
        return value if value.ndim <= 2 else value.reshape(-1, value.shape[-1])

    with np.load(data_path) as expert_files:
        states = flatten_feature_array(expert_files["states"])
        cutoff = int(states.shape[0] / cutoff)
        return {
            "states": states[:cutoff],
            "actions": flatten_feature_array(expert_files["actions"])[:cutoff],
            "next_states": flatten_feature_array(expert_files["next_states"])[:cutoff],
            "absorbing": np.asarray(expert_files["absorbing"], dtype=np.float32).reshape(-1)[:cutoff],
        }
