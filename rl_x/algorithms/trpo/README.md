# Trust Region Policy Optimization

Contains the implementation of [Trust Region Policy Optimization (TRPO)](https://proceedings.mlr.press/v37/schulman15.html).

TRPO maximizes the importance-weighted policy surrogate subject to an empirical mean-KL constraint. It obtains a natural-gradient direction with damped Fisher-vector products and conjugate gradient, scales that direction to the KL boundary, and uses backtracking line search to accept only a surrogate-improving update inside the trust region.


## RL-X implementation

**Implementation details**
- Uses the Gaussian likelihood-ratio surrogate and analytic old-to-new diagonal-Gaussian KL
- Computes Fisher-vector products with automatic differentiation and conjugate gradient without constructing the Fisher matrix
- Enforces the hard mean-KL constraint with the reference backtracking acceptance criteria
- Optimizes the value function separately with Adam over shuffled rollout minibatches so critic gradients cannot bypass the policy trust region
- Uses GAE, normalized rollout advantages, and the RL-X fully JIT-compiled policy, critic, environment, logging, evaluation, and checkpoint paths
- Supports reference-style policy-batch subsampling while using the complete rollout by default
- Keeps fully JIT-compiled robot environments in their native action domain by default; `action_clipping_and_rescaling=True` enables conventional bounded-environment processing

The port was audited against the original paper, OpenAI Spinning Up revision `038665d62d569055401d91856abb287263096178`, and SB3-Contrib revision `075bd5be8d43848b0f0cd3bc8a32f6892d0d58fb`.

**Supported frameworks**
- JAX (Flax, fully JIT-compiled)

**Supported action space, observation space and data interface types**
| Version | Flat value obs | Image obs | Continuous actions | Discrete actions | List interface | Numpy interface | Torch interface | JAX interface |
| :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: | :-----------: |
| JAX (Flax, fully JIT-compiled) | ✅ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ✅ |


## Resources

- Paper: [Trust Region Policy Optimization](https://proceedings.mlr.press/v37/schulman15.html)
- Maintained reference implementation: [Stable-Baselines3 Contrib TRPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/tree/master/sb3_contrib/trpo)
- Reference derivation and implementation: [OpenAI Spinning Up TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html)
