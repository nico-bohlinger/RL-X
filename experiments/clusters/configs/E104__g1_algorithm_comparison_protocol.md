# E104 G1 algorithm comparison

The four jobs compare the standalone SPO, FPO, DPPO, and DIME reference-method ports on one packed four-GPU node.
They use the unchanged Unitree G1 MJX locomotion task, 4096 environments, seed 0, no evaluation, and no model
saving. The only outcome is unsmoothed `env_curriculum/coefficient`.

Report both:

- `env_curriculum/coefficient` versus matched `global_step` gates;
- `env_curriculum/coefficient` versus elapsed W&B `_runtime`, with compilation identified separately from training
  throughput.

These are standalone reference-method ports adapted to RL-X's massively parallel G1 setting, not byte-for-byte
executions of the authors' repositories. The environment and outcome protocol are shared, while each method keeps
its characteristic policy and objective. The actual production optimization budgets are:

- SPO: 128-step rollouts, 524,288 rollout samples, 10 epochs, 32,768-sample minibatches, 16 minibatches per epoch,
  and 160 optimizer updates per rollout; policy and value networks use 512/256/128 hidden units.
- FPO: 128-step rollouts, 524,288 rollout samples, 8 epochs, 2,048-sample minibatches, 256 minibatches per epoch,
  2,048 optimizer updates per rollout, 10 flow steps, and 8 flow samples per environment action; the policy uses
  32/32/32/32 hidden units and the value network uses five 256-unit layers.
- DPPO: 128-step rollouts, 524,288 rollout samples, 8 epochs, 2,048-sample minibatches, 256 minibatches per epoch,
  2,048 optimizer updates per rollout, and 8 denoising steps; both policy and value networks use 256/256/256 hidden
  units.
- DIME: 8,192-sample replay minibatches, replay capacity 1,024 transitions per environment, 2 critic updates per
  vector-environment step, one delayed actor/temperature update every 3 critic updates, 16 diffusion steps, a
  256/256/256 score network, and twin 2,048/2,048 distributional critics with 101 atoms.

DIME's replay size, warmup, and batching are a massive-parallel adaptation of its reference implementation; the
reference algorithm's diffusion-control objective, delayed actor, entropy temperature, and distributional twin
critic are retained. FPO and DPPO likewise retain their reference policy parameterizations but use the explicit
RL-X budgets above. Report every budget alongside wall-clock comparisons. A method is not called faster or better
from a transient peak, an unmatched step, or evaluation return.

The production run set is tagged `g1-algorithm-comparison-dashboard` and shown alongside the fixed PPO, FastSAC,
FastTD3, FlashSAC, RePPO histories and the mature FastMPO histories `6aufiuih` and `edo7pc2n`.
