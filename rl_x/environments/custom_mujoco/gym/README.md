# MuJoCo Benchmark Environments

These environments implement the Gymnasium v5 tasks Ant, Half Cheetah, Hopper, Walker2D and Humanoid.
They also include Point Maze, a custom task where a particle has to reach a target position.
All environments use flat observations and continuous actions.
They can be used with reinforcement learning and imitation learning algorithms.


## Environment versions

Each environment is implemented with normal MuJoCo and MuJoCo XLA (MJX).
The two versions use the same models, reset distributions, observations, rewards and termination conditions.
Small numerical differences between the physics engines can still lead to different trajectories.

The normal MuJoCo versions use a Numpy interface and work with standard Flax and PyTorch algorithms such as ```ppo.flax``` and ```ppo.pytorch```.
They are useful for testing, rendering and debugging a policy without waiting for MJX compilation.
The MJX versions run environments in parallel on a GPU and use a JAX interface.
Use them with ```flax_full_jit``` algorithms such as ```ppo.flax_full_jit``` for large-scale training.

Set ```environment.name``` to one of the names in the table.

| Environment | MuJoCo | MJX |
| ----------- | ----------- | ----------- |
| Ant | ```custom_mujoco.gym.ant_v5.mujoco``` | ```custom_mujoco.gym.ant_v5.mjx``` |
| Half Cheetah | ```custom_mujoco.gym.half_cheetah_v5.mujoco``` | ```custom_mujoco.gym.half_cheetah_v5.mjx``` |
| Hopper | ```custom_mujoco.gym.hopper_v5.mujoco``` | ```custom_mujoco.gym.hopper_v5.mjx``` |
| Walker2D | ```custom_mujoco.gym.walker2d_v5.mujoco``` | ```custom_mujoco.gym.walker2d_v5.mjx``` |
| Humanoid | ```custom_mujoco.gym.humanoid_v5.mujoco``` | ```custom_mujoco.gym.humanoid_v5.mjx``` |
| Point Maze | ```custom_mujoco.gym.point_maze_mujoco``` | ```custom_mujoco.gym.point_maze_mjx``` |


## Training

Run experiments from the ```experiments``` folder in the repository.
The normal MuJoCo versions use one environment by default.
The MJX versions use 4096 parallel environments for the locomotion tasks and 64 for Point Maze.
Change ```environment.nr_envs``` to adjust the number of environments to your hardware.

PPO and the imitation learning algorithms collect ```environment.nr_envs * algorithm.nr_steps``` samples before each update.
Choose an ```algorithm.minibatch_size``` that divides this number evenly.
When using fewer environments, increase ```algorithm.nr_steps``` or decrease ```algorithm.minibatch_size``` so that each update still has enough samples.
For example, when training Point Maze with the default ```ppo.flax_full_jit``` configuration, set ```--algorithm.minibatch_size=512``` because its default minibatch size is larger than the collected batch.

For imitation learning, also set ```algorithm.data_path``` to an expert dataset for the selected task.
The algorithm READMEs contain the dataset instructions.


## Testing a trained model

The GAIL, AMP, AIRL, NEAR, TRIRL+PPO and TRIRL+DTRL implementations can load a model trained with ```flax_full_jit``` into the matching ```flax``` version.
Use the normal MuJoCo version of the training environment to render the policy without compiling MJX physics.
These Flax implementations load only the policy in test mode, so the expert dataset is not needed.

The following example renders a TRIRL+PPO policy trained on Ant.
Run the command from the ```experiments``` folder and replace the model path with the path to your saved model.

```bash
python experiment.py \
    --algorithm.name=trirl_ppo.flax \
    --algorithm.device=cpu \
    --environment.name=custom_mujoco.gym.ant_v5.mujoco \
    --environment.nr_envs=1 \
    --environment.render=True \
    --runner.mode=test \
    --runner.load_model="/absolute/path/to/latest.model" \
    --runner.nr_test_episodes=5
```

For another algorithm or task, change ```algorithm.name``` and ```environment.name``` to match the training run.
For a model trained with PyTorch, use the same algorithm's ```pytorch``` version.
Flax and PyTorch model files cannot be loaded interchangeably.
Keep ```environment.nr_envs=1``` when rendering to avoid opening multiple viewers.


## Point Maze

Point Maze is a custom environment and does not reproduce the Gymnasium-Robotics task with the same name.
It starts the particle from a fixed position and returns the particle and target positions as a flat observation.
The default reward penalizes the distance to the target and the size of the actions.
An episode ends when the particle reaches the target or after 100 steps.

The normal MuJoCo version exposes additional settings in its default config.
Set ```environment.reward_style=sparse``` to give a reward of one when the target is reached and zero otherwise.
Set ```environment.flipped=True``` to use the alternative maze layout.
Use ```environment.success_radius``` to change how close the particle must get to the target.
Use ```environment.horizon``` to change the maximum number of steps per episode.
The MJX version uses the defaults in its environment class and does not expose these settings through the command line.
