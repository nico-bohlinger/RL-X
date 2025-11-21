# MuJoCo Playground

Contains the environments from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).

The reference implementation contains the following environments:
| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Go1 Joystick Flat Terrain | Flat value | Continuous | JAX |

For testing other MuJoCo Playground environments, the environment name can simply be changed with the ```type``` config, e.g. ```--environment.mujoco_playground.go1_joystick_flat_terrain.type="G1JoystickRoughTerrain"```
Or for proper usage, create a new directory for the environment.