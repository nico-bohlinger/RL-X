# Custom MuJoCo Environments

Contains three examples for custom MuJoCo environments: a minimal Ant tracking task, a more complete Robot Locomotion setup, and a RoboCup Soccer locomotion variant.

The examples can be used as a template for other custom MuJoCo environments. They contain:
- All necessary handling of the MuJoCo physics engine directly via its Python bindings to form a stand-alone environment class
- Implementation of a GLFW viewer for rendering (non-MJX version)

The ant example is a simple environment using the Ant robot to track a given velocity command.
It is implemented in four versions: one using normal MuJoCo, one using MJX with the standard JAX backend, one using MJX with the Warp backend, and one using MuJoCo Warp with a PyTorch interface.

The robot locomotion example is a more complex environment and contains everything that is needed to train a quadruped (Unitree Go2) or humanoid (Unitree G1) robot to walk and afterwards deploy the learned policy on the real robot.
It is implemented in three versions: one using normal MuJoCo, one using MJX with the standard JAX backend, and one using MJX with a PyTorch interface.

The RoboCup soccer example builds on top of the robot locomotion environment and is tailored for training bipedal humanoids (e.g. the Booster T1) for the [MuJoCo-based RoboCup Soccer Simulation Server (RCSSServerMJ)](https://gitlab.com/robocup-sim/rcssservermj).
It is implemented in two versions: one using normal MuJoCo and one using MJX with the standard JAX backend.

The environment versions with the MJX suffix use the new MuJoCo XLA (MJX) backend that enables running the environment on a GPU (similar to Isaac Gym / Sim / Lab).
A modern NVIDIA GPU can easily handle 4000 of those environments in parallel.
This gives a significant speedup compared to using normal MuJoCo.  
MJX-based environments break the typical Gym interface and can currently only be used with ```flax_full_jit``` algorithm implementations (e.g. ```ppo.flax_full_jit```).

The environment versions with the Warp Torch suffix use [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp).
It runs all environments in parallel through NVIDIA Warp kernels and exposes the simulation state as zero-copy PyTorch tensors via `wp.to_torch`, so observations, rewards, resets and `ctrl` writes happen entirely on the GPU.
On CUDA devices the inner step / forward pipeline is captured into a CUDA Graph for minimal kernel-launch overhead; on CPU it falls back to plain `mjwarp.step`.
Similar to the MJX version, the Warp-based environment can run thousands of parallel environments on a modern NVIDIA GPU, giving a significant speedup compared to normal MuJoCo.
The environment is meant to be used with PyTorch algorithms (e.g. ```ppo.pytorch```).

The environment versions with the MJX Warp suffix use MJX with the MuJoCo Warp physics engine as backend (`impl='warp'`).
It runs all environments natively in JAX/XLA without an outer `vmap`, using the Warp backend for GPU-accelerated physics.
It is designed to be used with JAX algorithms (e.g. ```ppo.flax_full_jit```).

More specifically, the example uses the Ant robot and defines as the task to track a given velocity command.

| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| Ant MuJoCo | Flat value | Continuous | Numpy |
| Ant MJX | Flat value | Continuous | JAX |
| Ant MJX Warp | Flat value | Continuous | JAX |
| Ant Warp Torch | Flat value | Continuous | Torch |
| Robot Locomotion MuJoCo | Flat value | Continuous | Numpy |
| Robot Locomotion MJX | Flat value | Continuous | JAX |
| Robot Locomotion MJX PyTorch | Flat value | Continuous | Torch |
| RoboCup Soccer Locomotion MuJoCo | Flat value | Continuous | Numpy |
| RoboCup Soccer Locomotion MJX | Flat value | Continuous | JAX |