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

**data_interface_type.py**
- Defines all possible data interface types, i.e. the data types of actions and observations

**simulation_type.py**
- Defines all possible simulation types; currently only default and JAX-based

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
│   │   │   general_properties.py
│   │   │   wrappers.py
```

**\_\_init__.py**
- Registers the environment with the environment manager
- Needs to import create_env(), get_config() and GeneralProperties for the registration
- Creates a constant for the environment name which can be imported for running an experiment. The name is extracted from the environment's directory structure.
    - Example: ```--environment.name=gym.mujoco.humanoid_v4```
    - See: ```experiments/start_experiment.py```

**create_env.py**
- Defines the create_env() function for the registration
- Uses wrappers from wrappers.py

**default_config.py**
- Defines the get_config() function for the registration
- Those configs can be later overwritten via the command line
- They are available under the config.environment namespace
    - Example: ```--environment.nr_envs=10```
    - See: ```experiments/start_experiment.sh```

**general_properties.py**
- Defines the GeneralProperties class for the registration
- Currently those properties are the action space type, observation space type and data interface type and the type of simulation framework used
- Properties are used to check if environment and algorithm are compatible and to set default settings of the Deep Learning framework

**wrappers.py**
- Wrappers around the environment
- Handles things like final observations, action and observation spaces, etc.
- Makes sure every environment provides the same data in the same format for the algorithms


## Adding new environments
Fast testing of a new environment with an already provided framework (Gymnasium or EnvPool) can be done by changing the environment name in the create_env.py file or even make it a variable and add it to the config file.

To add a completely new environment or permanently a new one from an already provided framework, create a new directory with the same structure as outlined above.

Environments can be added and registered outside of RL-X by keeping the same directory structure as RL-X, e.g. ```mypackage/environments/unitree_a1``` and then adding ```mypackage``` to the ```implementation_package_names``` list when creating the Runner object in the experiment script, e.g. ```Runner(implementation_package_names=["rl_x", "mypackage"])```.

For concrete implementations of provided frameworks, look into the ```rl_x/environments/gym``` and ```rl_x/environments/envpool``` directories.

An example for a custom mujoco environment can be found in ```rl_x/environments/custom_mujoco```.

A prototype for a custom environment interface with simple socket communication can be found in ```rl_x/environments/custom_interface```.


## Mix and match environments and algorithms
If the wrappers of an environment provide all data in the standard format, i.e. like the already implemented environments, then environments and algorithms can be freely combined.  

Keep in mind the algorithm has to be able to handle the action space, observation space and data interface type of the environment.  
For simplicity most of the implemented algorithms only support continuous actions and non-image observations in the form of a numpy arrays.

> ✅ **Every environment can be used with the PPO PyTorch version.** The PyTorch implementation of PPO supports all currently used action space, observation space and data interface types.
