# Adversarial Motion Priors

Contains the implementation of [Adversarial Inverse Reinforcement Learning (AIRL)](https://arxiv.org/abs/1710.11248) with PPO for policy optimization.


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
- JAX (Flax)

**Supported observation space, action space and data interface types**
| Version | Flat value obs | Image obs | Contiuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax) full JIT | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |