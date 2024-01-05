# Algorithms


## Important files

**algorithm_manager.py**
- Enables the registration of algorithms
- Similar to gym.envs.registration but for algorithms

**algorithm.py**
- Base class for all algorithms

**deep_learning_framework_type.py**
- Defines all possible Deep Learning framework types

**\<algorithm name\>**
- Contains either subdirectories or directly the implementation of a concrete algorithm
- It is recommended to use subdirectories for each framework, e.g. PyTorch, Flax, etc.
- See below for more details


## Standard algorithm structure
```
ppo
└───pytorch
│   │   __init__.py
│   │   default_config.py
│   │   general_properties.py
│   │   ppo.py
│   │   <helper files>
```

**\_\_init__.py**
- Registers the algorithm with the algorithm manager
- Needs to import get_config() and GeneralProperties for the registration
- Creates a constant for the algorithm name which can be specified for running an experiment. The name is extracted from the algorithm's directory structure.
    - Example: ```--algorithm.name=ppo.pytorch```
    - See: ```experiments/start_experiment.sh```

**default_config.py**
- Defines the get_config() function for the registration
- Those configs can be later overwritten via the command line
- They are available under the config.algorithm namespace
    - Example: ```--algorithm.nr_steps=512```
    - See: ```experiments/start_experiment.sh```

**general_properties.py**
- Defines the GeneralProperties class for the registration
- Currently those properties are the action space types, observation space types and data interface types the algorithm can handle and the type of Deep Learning framework used
- Properties are used to check if environment and algorithm are compatible and to set default settings of the Deep Learning framework

**\<algorithm name\>.py**
- Defines the algorithm class for the registration
- The class needs to implement train(), test() and load() functions, which are called by the runner
- A function for save() is optional and only called by the algorithm itself but encouraged, so load() can be properly used
- The function general_properties() is currently not used but is recommended to be implemented

**\<helper files\>**
- Helper files can be used to prevent cluttering the main algorithm file
- E.g. ```policy.py``` and ```critic.py``` in actor-critic algorithms to define the neural networks or ```replay_buffer.py``` for off-policy algorithms


## Adding new algorithms
To add a new algorithm, create a new directory with the algorithm name, a subdirectory for the framework and add the files described above.

Algorithms can be added and registered outside of RL-X by keeping the same directory structure as RL-X, e.g. ```mypackage/algorithms/ppo/tensorflow``` and then adding ```mypackage``` to the ```implementation_package_names``` list when creating the Runner object in the experiment script, e.g. ```Runner(implementation_package_names=["rl_x", "mypackage"])```.

For concrete implementations see the provided algorithms, e.g. ```rl_x/algorithm/ppo/pytorch``` or ```rl_x/algorithm/sac/flax```.


## Mix and match algorithms and environments
If the algorithm provides all training and testing functions in the standard format, i.e. like the already implemented algorithms, then algorithms and environments can be freely combined.  

Keep in mind environments have different action space, observation space and data interface types, which need to be handled by the algorithm.  
The handling of the action and observation space types should be done by the classes / files that create the needed neural networks, e.g. ```policy.py``` and ```critic.py``` in an actor-critic algorithm.  
The handling of the data interface types should be done by the algorithm class itself and revolves around the step() function or more precisely the action and observation variables.  
For simplicity most of the implemented algorithms only support continuous actions and non-image observations in the form of a numpy arrays. 

> ✅ **Every environment can be used with the PPO PyTorch version.** The PyTorch implementation of PPO supports all currently used action space, observation space and data interface types.
