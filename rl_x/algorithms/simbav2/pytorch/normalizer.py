import numpy as np


def update_normalizer(state, batch):
    batch_mean = np.mean(batch, axis=0)
    batch_var = np.var(batch, axis=0)
    batch_count = np.float32(batch.shape[0])
    delta = batch_mean - state["mean"]
    total = state["count"] + batch_count
    return {
        "mean": state["mean"] + delta * batch_count / total,
        "var": (state["var"] * state["count"] + batch_var * batch_count + np.square(delta) * state["count"] * batch_count / total) / total,
        "count": total,
    }


def update_reward_normalizer(state, reward, terminated, truncated, gamma):
    done = np.logical_or(terminated, truncated).astype(np.float32)
    new_G_r = gamma * (1.0 - done) * state["G_r"] + reward
    sample_mean = np.mean(new_G_r)
    sample_var = np.var(new_G_r)
    sample_count = np.float32(new_G_r.shape[0])
    delta = sample_mean - state["rms_mean"]
    total_count = state["rms_count"] + sample_count
    ratio = sample_count / total_count
    return {
        "G_r": new_G_r,
        "G_r_max": np.maximum(state["G_r_max"], np.max(np.abs(new_G_r))),
        "rms_mean": state["rms_mean"] + delta * ratio,
        "rms_var": (state["rms_var"] * state["rms_count"] + sample_var * sample_count + np.square(delta) * state["rms_count"] * ratio) / total_count,
        "rms_count": total_count,
    }
