# Adversarial Motion Priors

Contains the implementation of [Adversarial Motion Priors (AMP)](https://arxiv.org/abs/2104.02180) with PPO for policy optimization.


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
- Includes an option to handle absorbing states appropriately by fiting the discriminator on samples in absorbing states, adding an indicator variable in the discriminator input to indicate the absorbing state, and use the absorbing state value during advantage estimation. Enabled by default through `algorithm.handle_absorbing_states`
- Includes gradient penalty through `algorithm.gp_lambda` and an option to compose with the true environment reward through `algorithm.env_reward_frac` (default: 0.0).
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

- Paper: [AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control (Peng et al., 2021)](https://arxiv.org/abs/2104.02180)
