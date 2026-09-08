# Custom MuJoCo Environments

Contains four examples for custom MuJoCo environments. These cover Ant velocity tracking, Robot Locomotion, Motion Tracking and RoboCup Soccer.

The examples can be used as templates for other custom MuJoCo environments. They handle the physics engine directly through its Python bindings in stand-alone environment classes.

The [Ant](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/ant) example uses the Ant robot to track a given velocity command.
It is implemented with normal MuJoCo, MJX, MJX Warp and Warp Torch.

The [Robot Locomotion](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_locomotion) example contains the setup to train a quadruped (Unitree Go2) or humanoid (Unitree G1) to walk and deploy the learned policy on the real robot.
It is implemented with normal MuJoCo, MJX, MJX Warp and MJX with a Torch interface.

The [Robot Motion Tracking](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_motion_tracking) example implements BeyondMimic-style tracking with the Unitree G1. It uses already-retargeted LAFAN motions for robot-only tracking and OMOMO motions for robot and object interaction.
It is implemented with normal MuJoCo and MJX Warp.

The [RoboCup Soccer](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robocup_soccer) example builds on Robot Locomotion to train bipedal humanoids such as the Booster T1 for the [MuJoCo-based RoboCup Soccer Simulation Server (RCSSServerMJ)](https://gitlab.com/robocup-sim/rcssservermj).
It is implemented with normal MuJoCo and MJX.

The normal MuJoCo versions use a Numpy interface and can be used with standard algorithms such as ```ppo.flax``` or ```ppo.pytorch```.
The versions with the MJX suffix use MuJoCo XLA (MJX) to run environments in parallel on a GPU. They use a batched JAX interface rather than the typical Gym interface and require ```flax_full_jit``` algorithms such as ```ppo.flax_full_jit```.
GPU simulation supports thousands of parallel environments and can be substantially faster than normal MuJoCo. Capacity and throughput depend on the task and hardware.

The versions with the Warp Torch suffix use [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) and run environments in parallel through NVIDIA Warp kernels.
Simulation state is exposed as zero-copy Torch tensors through ```wp.to_torch```, so observations, rewards, resets and ```ctrl``` writes stay on the GPU during training.
On supported CUDA devices, physics steps and forward passes are captured in CUDA graphs to reduce kernel-launch overhead. Without graph capture, including on CPU, the environment calls ```mjwarp.step``` and ```mjwarp.forward``` directly.
These versions use Torch algorithms such as ```ppo.pytorch```.

The versions with the MJX Warp suffix use MuJoCo Warp through MJX with ```impl='warp'```.
They retain a JAX/XLA interface and use ```flax_full_jit``` algorithms such as ```ppo.flax_full_jit```.

The Robot Locomotion MJX Torch version exposes MJX physics through a Torch interface for algorithms such as ```fastsac.pytorch```. It is less optimized than the JAX versions.
See the [environment interface guide](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/README.md#mix-and-match-environments-and-algorithms) for algorithm compatibility.

| Version | Observation space | Action space | Data interface |
| ----------- | ----------- | ----------- | ----------- |
| [Ant MuJoCo](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/ant/mujoco) | Flat value | Continuous | Numpy |
| [Ant MJX](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/ant/mjx) | Flat value | Continuous | JAX |
| [Ant MJX Warp](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/ant/mjx_warp) | Flat value | Continuous | JAX |
| [Ant Warp Torch](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/ant/warp_torch) | Flat value | Continuous | Torch |
| [Robot Locomotion MuJoCo](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_locomotion/mujoco) | Flat value | Continuous | Numpy |
| [Robot Locomotion MJX](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_locomotion/mjx) | Flat value | Continuous | JAX |
| [Robot Locomotion MJX Warp](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_locomotion/mjx_warp) | Flat value | Continuous | JAX |
| [Robot Locomotion MJX Torch](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_locomotion/mjx_torch) | Flat value | Continuous | Torch |
| [Robot Motion Tracking MuJoCo](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_motion_tracking/mujoco) | Flat value | Continuous | Numpy |
| [Robot Motion Tracking MJX Warp](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robot_motion_tracking/mjx_warp) | Flat value | Continuous | JAX |
| [RoboCup Soccer Locomotion MuJoCo](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robocup_soccer/locomotion/mujoco) | Flat value | Continuous | Numpy |
| [RoboCup Soccer Locomotion MJX](https://github.com/nico-bohlinger/RL-X/tree/master/rl_x/environments/custom_mujoco/robocup_soccer/locomotion/mjx) | Flat value | Continuous | JAX |
