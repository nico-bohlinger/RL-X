from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 4096
    config.render = False
    config.render_callback_type = "debug_callback"
    config.device = "gpu"
    # CUDA graph capture mode: "jax", "warp", "warp_staged", "warp_staged_ex"
    # "jax" avoids Warp's own CUDA graph capture entirely, using JAX/XLA as the outer JIT layer instead
    # The Warp/CUDA conditional graph nodes (used inside the solver) are only supported from driver 12.4+.
    # On driver 12.4+, "warp" is generally the fastest option.
    config.graph_mode = "warp"
    config.naconmax = 8 * 4096
    config.njmax = 64
    config.horizon = 1000
    config.action_scaling_factor = 0.3
    config.copy_train_env_for_eval = True

    return config
