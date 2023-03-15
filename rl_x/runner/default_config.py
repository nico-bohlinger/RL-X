from ml_collections import config_dict
import time


def get_config(runner_mode):
    config = config_dict.ConfigDict()

    config.mode = runner_mode

    config.track_console = False
    config.track_tb = False
    config.track_wandb = False
    config.wandb_entity = "placeholder"
    config.project_name = "placeholder"
    config.exp_name = "placeholder"
    config.run_name = f"{int(time.time())}"
    config.notes = "placeholder"

    config.save_model = False
    config.load_model = ""

    config.nr_test_episodes = 10  # if runner mode = test

    return config
