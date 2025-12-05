from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.seed = 1
    config.nr_envs = 4096
    config.render = False
    config.copy_train_env_for_eval = True
    config.disable_fabric = False
    config.livestream = -1
    config.enable_cameras = False
    config.xr = False
    config.device = "gpu"
    config.cpu = False
    config.verbose = False
    config.info = False
    config.experience = ""
    config.rendering_mode = None
    config.kit_args = ""
    config.anim_recording_enabled = False
    config.anim_recording_start_time = 0
    config.anim_recording_stop_time = 10

    return config
