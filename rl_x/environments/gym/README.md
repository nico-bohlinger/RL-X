# Gymnasium

Contains the environments from [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium) (former OpenAI Gym).

For the DeepMind Control Suite environments, [Farama-Foundation/shimmy](https://github.com/Farama-Foundation/Shimmy) is used to convert the API to the Gymnasium format.

The reference implementation contains the following environments:
| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Atari Pong-v5 | Image | Discrete | Numpy |
| Classic CartPole-v1 | Flat value | Discrete | Numpy |
| DeepMind Control Suite | HumanoidRun-v1 | Flat value | Continuous | Numpy |
| MuJoCo Humanoid-v4 | Flat value | Continuous | Numpy |

For testing other Gymnasium environments, the environment name can simply be changed with the ```type``` config, e.g. ```--environment.gym.atari.pong_v5.type="Breakout-v5"```
Or for proper usage, create a new directory for the environment.