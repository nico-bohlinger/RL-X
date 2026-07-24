# G1 reference-corrected algorithm comparison

The first SPO, FPO, DPPO, and DIME G1 jobs are implementation-failure histories, not algorithm results. They are retained for provenance and excluded from algorithm conclusions. E111 replaces them with independently implemented, reference-faithful RL-X algorithms.

All production jobs use the unchanged G1 MJX locomotion environment, 4096 environments, seed 0, `algorithm.evaluation_active=False`, and `runner.save_model=False`. The sole outcome is unsmoothed `env_curriculum/coefficient` against matched `global_step`. `_runtime`, steps per second, compile overhead, wall-clock time to each curriculum level, update counts, batch geometry, and accelerator type are reported as efficiency context.

## Fidelity boundaries

- SPO follows the authors' MuJoCo objective, observation and discounted-return reward normalization, action clipping, seven-layer Tanh policy, clipped value loss, joint gradient clipping, 256-step rollout, 10 epochs, and four minibatches.
- FPO follows the authors' G1 locomotion configuration: 64 flow steps, 32 CFM samples, reference actor and critic widths, ASPO loss, CFM clamps, action perturbation, AdamW, 24-step rollout, 32 epochs, and four minibatches.
- DPPO follows the reference from-scratch locomotion method: 10-step cosine DDPM, stochastic posterior transitions, complete denoising paths, transition log probabilities, denoising-dependent discount and clipping, reward and observation normalization, residual networks, 10 epochs, and 20 minibatches.
- DIME follows the reference off-policy method: PISGRAD score network, 16-step Euler-Maruyama path, adjoint running/stochastic/terminal costs, learned timestep and per-action friction, delayed actor updates, automatic entropy coefficient, twin categorical CrossQ critics with Batch Renormalization, and samplewise update-to-data ratio 2.

The G1 rollout horizon is an environment-scale adaptation where a reference has no G1 configuration. Such runs are comparable within this campaign but must not be described as exact reproductions of a paper table.

## Decision gates

Compare curriculum at 25M, 50M, 75M, 100M, 125M, 150M, 200M, 300M, 400M, 500M, 600M, and later matched steps. Record peak, current, drawdown, 10M slope, runtime, throughput, and wall-clock time to curriculum thresholds. Caught exceptions, nonfinite updates, silent stalls, or scheduler completion with an internal traceback are failures.
