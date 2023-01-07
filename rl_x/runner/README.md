# Runner


## Important files

**default_config.py**
- Defines the get_config() function for the runner
- Those configs can be later overwritten via the command line
- They are available under the config.runner namespace
    - Example: ```--config.runner.track_console=True```
    - See: ```experiments/experiment.sh```

**runner_mode.py**
- Contains all possible runner modes
    - TRAIN: Runs the training loop
    - TEST: Runs the testing loop, with ```--config.runner.nr_test_episodes``` many episodes
    - SHOW_CONFIG: Prints out all available configs of the given algorithm, environment and runner combination

**runner.py**
- Takes algorithm and environment and parses all configs to run an experiment


## Usage for experiments
See ```experiments/experiment.py``` for an example of how to run an experiment by importing the algorithm, environment and runner and setting the runner mode.