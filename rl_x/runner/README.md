# Runner


## Important files

**default_config.py**
- Defines the get_config() function for the runner
- Those configs can be later overwritten via the command line
- They are available under the config.runner namespace
    - Example: ```--runner.track_console=True```
    - See: ```experiments/start_experiment.sh```

**runner_mode.py**
- Contains all possible runner modes
    - TRAIN: Runs the training loop
    - TEST: Runs the testing loop, with ```--runner.nr_test_episodes``` many episodes
    - SHOW_CONFIG: Prints out all available configs of the given algorithm, environment and runner combination
- Can be set via the command line
    - Example: ```--runner.mode=train```
    - See: ```experiments/start_experiment.sh```

**runner.py**
- Parses all configs to run an experiment
- Sets the default algorithm, environment and runner mode
    - Default algorithm: ```ppo.pytorch```
    - Default environment: ```gym.mujoco.humanoid_v4```
    - Default runner mode: ```train```


## Usage for experiments
See ```experiments/experiment.py``` for an example of how to run an experiment by importing the runner.

See ```experiments/start_experiment.sh``` for an example of how to run an experiment and setting all configs via the command line.

When a new algorithm or environment is added and registered outside of RL-X, the package needs to be added to the implementation_package_names list when creating the Runner object in the experiment script. See the README files in ```rl_x/algorithms``` and ```rl_x/environments``` for more details.
