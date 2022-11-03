from ml_collections import config_dict
import time


def get_config():
    config = config_dict.ConfigDict()

    config.mode = "train"  # train, test
    config.test_episodes = 10

    config.save_model = False
    config.load_model = ""

    config.track_console = False
    config.track_tb = False
    config.track_wandb = False
    config.wandb_entity = "placeholder"
    config.project_name = "placeholder"
    config.exp_name = "placeholder"
    config.run_name = f"{int(time.time())}"
    config.notes = "placeholder"

    return config
