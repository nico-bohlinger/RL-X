# Algorithms


## Important files

**algorithm_manager.py**
- Enables the registration of algorithms
- Similar to gym.envs.registration but for algorithms

**algorithm.py**
- Base class for all algorithms

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
│   │   ppo.py
│   │   <helper files>
```

**\_\_init__.py**
- Registers the algorithm with the algorithm manager
- Needs to import get_config() function for the registration
- Creates a constant for the algorithm name which can be specified for running an experiment. Algorithm names use their directory structure as the name.
    - Example: ```--config.algorithm.name="ppo.pytorch"```
    - See: ```experiments/start_experiment.sh```

**default_config.py**
- Defines the get_config() function for the registration
- Those configs can be later overwritten via the command line
- They are available under the config.algorithm namespace
    - Example: ```--config.algorithm.nr_steps=512```
    - See: ```experiments/start_experiment.sh```

**\<algorithm name\>.py**
- Defines the algorithm class for the registration
- The class needs to implement train(), test() and load() functions, which are called by the runner
- A function for save() is optional and only called by the algorithm itself but encouraged, so load() can be properly used

**\<helper files\>**
- Helper files can be used to prevent cluttering the main algorithm file
- E.g. ```policy.py``` and ```critic.py``` in actor-critic algorithms to define the neural networks or ```replay_buffer.py``` for off-policy algorithms


## Adding new algorithms
To add a new algorithm, create a new directory with the algorithm name, a subdirectory for the framework and add the files described above.

For concrete implementations see the provided algorithms, e.g. ```rl_x/algorithm/ppo/pytorch``` or ```rl_x/algorithm/sac/flax```.


## Mix and match algorithms and environments
If the algorithm provides all training and testing functions in the standard format, i.e. like the already implemented algorithms, then algorithms and environments can be freely combined.  

Keep in mind environments have different action and observation space types, which need to be handled by the algorithm.  
This handling should be done by the classes / files that create the needed neural networks, e.g. ```policy.py``` and ```critic.py``` in an actor-critic algorithm. If an action and observation space type combination is not supported, an adequate error should be raised.  
For simplicity most of the implemented algorithms only support continuous action spaces and non-image observation spaces.  

> ✅ **Every environment can be used with the PPO PyTorch version.** The PyTorch implementation of PPO supports all currently used action and observation space types.
