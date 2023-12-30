# Gymnasium

Contains the environments from [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (former OpenAI Gym).

The reference implementation contains the following environments:
| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Atari Pong-v5 | Image | Discrete | Numpy |
| Classic CartPole-v1 | Flat value | Discrete | Numpy |
| MuJoCo Humanoid-v4 | Flat value | Continuous | Numpy |

For testing other Gymnasium environments, the environment name can simply be changed in the create_env.py file.
Or for proper usage, create a new directory for the environment.