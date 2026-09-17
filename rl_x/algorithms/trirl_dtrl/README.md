# Trust Region Inverse Reinforcement Learning + Differentiable Trust Region Projection Layers

Contains the implementation of [Trust Region Inverse Reinforcement Learning (TRIRL)](https://arxiv.org/pdf/2605.11020) with PPO+Differentiable Trust Region Projection Layers for policy optimization.

On how the algorithms works, refer to the [Resources](#resources) section.

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
- This implements the max-eta version of TRIRL.
- Based on the PPO-Clip version: Clipping the ratio of the new and old policy
- The hyperparameters and network architecture for the ```flax_full_jit``` version are tuned for strong performance on many parallel environments for mujoco benchmark environments
- The reward correction evaluates the discriminator buffer in chunks, so that only ```chunk_size``` discriminators are held on the device at a time. The default of 10 is chosen to fit on a small GPU. It is purely a memory-for-speed trade-off and does not change the computed reward, so on a GPU with more memory a larger ```chunk_size``` can be used to run the correction in fewer passes.

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

- Paper: [Trust Region Inverse Reinforcement Learning: Explicit Dual Ascent using Local Policy Updates (Diwan et al., 2026)](https://arxiv.org/pdf/2605.11020)

- Repositories:
    - Official Codebase: [here](https://github.com/anishhdiwan/trust-region-irl)
## Checkpoint and entropy semantics

These ports intentionally retain the PR's current behavior: the actor entropy term has an effective coefficient of `entropy_coef ** 2`, while the absorbing-state tail uses `entropy_coef`. With `reward_fn_approximator=True`, checkpoints restore the fitted network but omit discriminator/eta histories; correction targets restart from the new history rather than being initialized from that network. This is approximate continuation, not an exact resume. The full-buffer mode retains those histories, but neither mode restores simulator state or the random stream.
