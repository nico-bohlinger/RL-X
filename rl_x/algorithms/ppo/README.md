# Proximal Policy Optimization

Contains the implementation of [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Based on the PPO-Clip version: Clipping the ratio of the new and old policy

**Supported frameworks**
- PyTorch, JAX (Flax)

**Supported observation space, action space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| PyTorch | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources

- Paper: [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)

- Release blog post by OpenAI: [here](https://openai.com/blog/openai-baselines-ppo/)

- Spinning Up documentation: [here](https://spinningup.openai.com/en/latest/algorithms/ppo.html)

- Blog post on implementation details by Costa Huang: [here](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)

- Blog post by Lilian Weng: [here](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/#ppo)

- Repositories:
    - Stable Baselines3: [here](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py)
    - Repo from Ilya Kostrikov: [here](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py)
    - CleanRL: [here](https://github.com/vwxyzjn/cleanrl/tree/master/cleanrl)
    - TorchScript + layer-normalized GRU: [here](https://gist.github.com/7thStringofZhef/67cb7b4cb17baec4fab339b3b9deb2f1)