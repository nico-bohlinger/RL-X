# Maximum a Posteriori Policy Optimization

Contains the implementation of [Maximum a Posteriori Policy Optimization (MPO)](https://arxiv.org/pdf/1806.06920).

On how the algorithms works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Heavily based on the implementation in [Acme](https://github.com/deepmind/acme)
    - Per-dimension KL constraint
    - Out-of-bound action penalization
- Distributional TQC critics  

**Notes**
- Achieves great performance in general but suffers from sudden loss spikes that can cause the performance to collapse

**Supported frameworks**
- JAX (Flax)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |


## Resources
- Paper: [Maximum a Posteriori Policy Optimization (Abdolmaleki et al., 2018)](https://arxiv.org/pdf/1806.06920)

- Related papers:
    - [Relative Entropy Regularized Policy Iteration (Abdolmaleki et al., 2018)](https://arxiv.org/pdf/1812.02256)
    - [Revisiting Gaussian mixture critics in off-policy reinforcement learning: a sample-based approach (Shahriari et al., 2022)](https://arxiv.org/pdf/2204.10256)
    - [A Distributional View on Multi-Objective Policy Optimization (Abdolmaleki et al., 2020)](https://arxiv.org/pdf/2005.07513)

- Repositories:
    - Acme: [here](https://github.com/deepmind/acme/tree/master/acme/agents/jax/mpo)
    - jax-rl: [here](https://github.com/henry-prior/jax-rl/blob/master/jax_rl/MPO.py)
