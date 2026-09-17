# MuJoCo imitation-learning benchmarks

These environments use the PR's XMLs, reset distributions, observations, rewards, and termination conditions. Native MuJoCo provides a NumPy interface for standard Flax and PyTorch algorithms. MJX provides a JAX interface for fully jitted algorithms. Floating-point arithmetic and physics solvers can still produce different trajectories across backends.

| Task | Native MuJoCo identifier | MJX identifier |
| --- | --- | --- |
| Ant | `custom_mujoco.gym.ant_v5.mujoco` | `custom_mujoco.gym.ant_v5.mjx` |
| Half Cheetah | `custom_mujoco.gym.half_cheetah_v5.mujoco` | `custom_mujoco.gym.half_cheetah_v5.mjx` |
| Hopper | `custom_mujoco.gym.hopper_v5.mujoco` | `custom_mujoco.gym.hopper_v5.mjx` |
| Walker2D | `custom_mujoco.gym.walker2d_v5.mujoco` | `custom_mujoco.gym.walker2d_v5.mjx` |
| Humanoid | `custom_mujoco.gym.humanoid_v5.mujoco` | `custom_mujoco.gym.humanoid_v5.mjx` |
| Point Maze | `custom_mujoco.gym.point_maze_mujoco` | `custom_mujoco.gym.point_maze_mjx` |

Point Maze is a custom fixed-reset task, not Gymnasium-Robotics PointMaze. Native MuJoCo exposes `horizon`, `reward_style`, `flipped`, and `success_radius`. All native variants support human rendering; the underlying single environment also supports `render_mode="rgb_array"` for offscreen frames. Vector environments autoreset and retain terminal observations and episode statistics.

## Render a full-JIT Flax checkpoint without MJX compilation

From `experiments/`, select the matching algorithm's `flax` version and the matching MuJoCo environment:

```bash
python experiment.py \
  --runner.mode=test \
  --runner.load_model=/absolute/path/to/latest.model \
  --runner.nr_test_episodes=5 \
  --algorithm.name=trirl_ppo.flax \
  --algorithm.device=cpu \
  --environment.name=custom_mujoco.gym.ant_v5.mujoco \
  --environment.nr_envs=1 \
  --environment.render=True
```

The test path loads only the policy from Flax checkpoints and does not require the expert dataset. For a checkpoint trained in PyTorch, select the same algorithm's `pytorch` version instead. Cross-framework checkpoint conversion is not provided.

For native-MuJoCo training, increase `algorithm.nr_steps` or decrease `algorithm.minibatch_size` to fit the smaller environment count. Defaults in the algorithms deliberately retain the full-JIT hyperparameters. Native environments default to one instance; rendering many instances opens many viewers.
