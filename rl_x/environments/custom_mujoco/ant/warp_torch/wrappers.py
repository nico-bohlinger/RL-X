import torch


class RLXInfo:
    """Lightweight wrapper around the warp Ant env that exposes the helper
    methods expected by the PPO loop (logging aggregation and per-env final
    value access). Mirrors the role of the gymnasium-based `RLXInfo` wrapper
    used by the plain MuJoCo version.
    """

    def __init__(self, env):
        self.env = env


    def get_logging_info_dict(self, info):
        out: dict = {}
        done_mask = info.get("dones")
        skip_keys = {"dones", "episode_return", "episode_length"}
        for key, value in info.items():
            if key in skip_keys:
                continue
            if not torch.is_tensor(value):
                continue
            v = value
            if key.startswith("rollout/") and done_mask is not None:
                if not bool(done_mask.any()):
                    continue
                v = v[done_mask]
            out[key] = v.detach().cpu().tolist()
        return out


    def get_final_observation_at_index(self, info, index):
        # Auto-reset already returns the new episode's first observation in
        # `next_state` for done envs, so this should never be needed.
        raise NotImplementedError(
            "Final observations are not exposed by the torch warp env; "
            "`next_state` is already the post-reset observation for done envs."
        )


    def get_final_info_value_at_index(self, info, key, index):
        value = info.get(key)
        if value is None:
            raise KeyError(f"info has no key {key!r}")
        item = value[index]
        return item.item() if torch.is_tensor(item) else item


    def __getattr__(self, name):
        return getattr(self.env, name)
