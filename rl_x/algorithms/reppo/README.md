# REPPO

Contains the implementation of [REPPO](https://arxiv.org/pdf/2507.11019).

On how the algorithm works, refer to the [Resources](#resources) section.


## RL-X implementation

**Implementation Details**
- Uses fixed, configurable values for `gamma`, `v_min`, and `v_max`. The official implementation uses `gamma = 1 - 10 / T` for environment horizon `T` and derives the HL-Gauss range from the environment reward range
- Rescales normalized policy actions inside the algorithm instead of requiring an action-rescaling environment wrapper

**Supported frameworks**
- JAX (Flax)
- JAX (Flax, full JIT)
- PyTorch

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| JAX (Flax, full JIT) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |


## Resources
- Paper: [Relative Entropy Pathwise Policy Optimization (Voelcker, Brunnbauer et al., 2026)](https://arxiv.org/pdf/2507.11019)

- Repositories:
    - Source code of the paper: [here](https://github.com/cvoelcker/reppo)
