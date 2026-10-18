# Noise-conditioned Energy-based Annealed Rewards

Contains the implementation of [Noise-conditioned Energy-based Annealed Rewards (NEAR)](https://arxiv.org/abs/2501.14856) with PPO for policy optimization.


## RL-X implementation

**Dataset**
- Hosted on HuggingFace https://huggingface.co/datasets/anishdiwan/trirl_dataset.
```bash
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
curl -sSfL https://hf.co/git-xet/install.sh | sh

# Place in the top level directory .../RL-X/
git clone https://huggingface.co/datasets/anishdiwan/trirl_dataset
```

**Implementation Details**
- Allows using both state-action and state-next_state rewards
- Allows using [ncsnv1](https://arxiv.org/abs/1907.05600) and [ncsnv2](https://arxiv.org/abs/2006.09011)
- Based on the PPO-Clip version: Clipping the ratio of the new and old policy
- The hyperparameters and network architecture for the ```flax_full_jit``` version are tuned for strong performance on many parallel environments for mujoco benchmark environments

**Supported frameworks**

- JAX (Flax), with standard and fully jitted versions
- PyTorch

**Supported observation space, action space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ❌ | ❌ |
| PyTorch | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ | ❌ |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |

## Backend selection

The standard Flax and PyTorch versions preserve the full-JIT implementation's networks, objectives, defaults, and reward handling. Standard Flax collects rollouts through the NumPy environment interface and compiles the optimization step; PyTorch supports NumPy and Torch environments. TRIRL's PyTorch versions use chunked `torch.func.vmap` to evaluate discriminator histories.

Use the [native MuJoCo benchmarks](../../environments/custom_mujoco/gym/README.md) for rendering without compiling MJX physics. A full-JIT Flax checkpoint can be loaded with the matching standard Flax algorithm in `test` mode, with a single MuJoCo environment and without the expert dataset. PyTorch uses its own checkpoint format. For training, ensure `nr_envs * nr_steps >= minibatch_size`; the original defaults target many parallel environments.

## Resources

- Paper: [Noise-conditioned Energy-based Annealed Rewards (NEAR) (Diwan et al., 2025)](https://arxiv.org/abs/2501.14856)
