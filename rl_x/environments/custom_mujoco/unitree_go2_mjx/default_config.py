from ml_collections import config_dict


def get_config(environment_name):
    config = config_dict.ConfigDict()

    config.name = environment_name

    config.nr_envs = 4096

    config.seed = 1
    config.render = False
    config.device = "gpu"
    config.robot = "unitree_go2"                                  
    config.control_type = "pd"                                                      # "pd"
    config.command_type = "random"                                                  # "random"
    config.command_sampling_type = "step_probability"                               # "step_probability", "every_step", "none"
    config.initial_state_type = "random"                                            # "random", "default"
    config.reward_type = "default"                                                  # "default"
    config.termination_type = "below_ground_and_power"                              # "below_ground_and_power"
    config.domain_randomization_sampling_type = "step_probability"                  # "step_probability", "every_step", "none"
    config.domain_randomization_action_delay_type = "default"                       # "default", "none"
    config.domain_randomization_mujoco_model_type = "default"                       # "default", "none"
    config.domain_randomization_control_type = "default"                            # "default", "none"
    config.domain_randomization_perturbation_sampling_type = "step_probability"     # "step_probability", "every_step", "none"
    config.domain_randomization_perturbation_type = "default"                       # "default", "none"
    config.observation_noise_type = "default"                                       # "default", "none"
    config.observation_dropout_type = "default"                                     # "default", "none"
    config.terrain_type = "plane"                                                   # "plane"
    config.add_goal_arrow = False
    config.timestep = 0.005
    config.episode_length_in_seconds = 20

    return config
