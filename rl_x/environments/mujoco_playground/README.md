# MuJoCo Playground

Contains the environments from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

Keep in mind the PyTorch wrapper of MuJoCo Playground clips the actions to be between -1 and 1, the RL-X wrapper for the MJX version does the same.

The reference implementation contains the following environments:
| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| G1 Joystick Flat Terrain MJX | Flat value | Continuous | JAX |
| G1 Joystick Flat Terrain PyTorch | Flat value | Continuous | Torch |

For testing other MuJoCo Playground environments, the environment name can simply be changed with the ```type``` config, e.g. ```--environment.mujoco_playground.g1_joystick_flat_terrain.type="Go1JoystickRoughTerrain"```
Or for proper usage, create a new directory for the environment.