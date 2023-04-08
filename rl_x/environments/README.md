# Environments


## Important files
**environment_manager.py**
- Enables the registration of environments
- Similar to gym.envs.registration

**environment.py**
- Base class for all environments

**action_space_type.py**
- Defines all possible action space types

**observation_space_type.py**
- Defines all possible observation space types

**vec_env.py**
- Provides a wrapper to vectorize environments
- Copied from Stable Baselines3
- Not needed when using EnvPool

**\<environment name\>**
- Contains either subdirectories or directly the implementation of a concrete environment
- See below for more details


## Standard environment structure
```
gym
└───mujoco
│   └───humanoid_v4
│   │   │   __init__.py
│   │   │   create_env.py
│   │   │   default_config.py
│   │   │   wrappers.py
```

**\_\_init__.py**
- Registers the environment with the environment manager
- Needs to import create_env() and get_config() functions for the registration
- Creates a constant for the environment name which can be imported for running an experiment. Environment names use their directory structure as the name.
    - Example: ```--config.environment.name="gym.mujoco.humanoid_v4"```
    - See: ```experiments/start_experiment.py```

**create_env.py**
- Defines the create_env() function for the registration
- Uses wrappers from wrappers.py

**default_config.py**
- Defines the get_config() function for the registration
- Those configs can be later overwritten via the command line
- They are available under the config.environment namespace
    - Example: ```--config.environment.nr_envs=10```
    - See: ```experiments/start_experiment.sh```

**wrappers.py**
- Wrappers around the environment
- Defines action space and observation space type of the environment
- Handles things like handling of terminal observations, episode infos, action space shape, etc.
- Makes sure every environment provides the same data in the same format for the algorithm


## Adding new environments
Fast testing of a new environment with an already provided framework (Gymnasium or EnvPool) can be done by changing the environment name in the create_env.py file or even make it a variable and add it to the config file.

To add a completely new environment or permanently a new one from an already provided framework, create a new directory with the same structure as outlined above.

For concrete implementations see the provided environments, e.g. ```rl_x/environments/gym/mujoco``` or ```rl_x/environments/envpool/atari```.

A prototype for a custom environment interface with simple socket communication can be found in ```rl_x/environments/custom_environments```.


## Mix and match environments and algorithms
If the wrappers of an environment provide all data in the standard format, i.e. like the already implemented environments, then environments and algorithms can be freely combined.  

Keep in mind the algorithm is responsible for handling the action space and observation space type of the environment.  
For simplicity most of the implemented algorithms only support continuous action spaces and non-image observation spaces.  

> ✅ **Every environment can be used with the PPO PyTorch version.** The PyTorch implementation of PPO supports all currently used action and observation space types.
