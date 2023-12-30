# EnvPool

Contains the environments from [EnvPool](https://github.com/sail-sg/envpool).

EnvPool is a collection of environments, including Atari, MuJoCo, DeepMind Control Suite, and more.  
The environments provided by EnvPool are inherently vectorized and enable faster training than with the original implementations.  
EnvPool also provides XLA support, which means the environment's step() function can be jitted with JAX. But this is not currently used in RL-X.

The reference implementation contains the following environments:
| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Atari Pong-v5 | Image | Discrete | Numpy |
| Classic CartPole-v1 | Flat value | Discrete | Numpy |
| DeepMind Control Suite | HumanoidRun-v1 | Flat value | Continuous | Numpy |
| MuJoCo Humanoid-v4 | Flat value | Continuous | Numpy |

For testing other EnvPool environments, the environment name can simply be changed in the create_env.py file.
Or for proper usage, create a new directory for the environment.
