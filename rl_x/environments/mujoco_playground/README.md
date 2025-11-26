# MuJoCo Playground

Contains the environments from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

Keep in mind the PyTorch wrapper of MuJoCo Playground clips the actions to be between -1 and 1, while the pure MJX version does not clip the actions.

The reference implementation contains the following environments:
| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Go1 Joystick Flat Terrain MJX | Flat value | Continuous | JAX |
| Go1 Joystick Flat Terrain PyTorch | Flat value | Continuous | Torch |

For testing other MuJoCo Playground environments, the environment name can simply be changed with the ```type``` config, e.g. ```--environment.mujoco_playground.go1_joystick_flat_terrain.type="G1JoystickRoughTerrain"```
Or for proper usage, create a new directory for the environment.