# Robot Locomotion MuJoCo Environments

Contains MuJoCo and MJX environments for a robot locomotion task with the Unitree Go2 quadruped and Unitree G1 humanoid robots.
The resulting policies can be directly transferred to the real robots.
Example deployment code for the Go2 can be found [here](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/custom_mujoco/robot_locomotion/deployment/unitree_go2/).

The MuJoCo version is a copy of the MJX version but uses the standard MuJoCo physics engine, which makes it easier and quicker to load, visualize and debug policies trained in MJX.
Use the MJX version for large-scale training on a GPU.

The learning environment is written in a way that makes it robot agnostic, so it can be used with different base robots (Go2, G1, etc.). The robots are defined in the ```rl_x/environments/custom_mujoco/robot_locomotion/robots/``` folder.
The code structure and design decisions are based on a multi-embodiment learning [project](https://github.com/nico-bohlinger/one_policy_to_run_them_all).

## Testing a trained model
0. If you trained with ppo.flax_full_jit and now want to test with ppo.flax (i.e. trained in MJX and test / visualize in MuJoCo), change the policy and critic architectures to match the full_jit version.
1. Create test.sh file in the experiments folder (all .sh files besides slurm_experiment.sh are ignored by git)
```bash
cd rl_x/experiments
touch test.sh
```
2. Add the following content to the test.sh file
```bash
python experiment.py \
    --algorithm.name=ppo.flax \
    --environment.name=custom_mujoco.robot_locomotion.mujoco \
    --environment.mode=test \
    --environment.render=True \
    --runner.mode=test \
    --runner.load_model="latest.model"
```
3. Run the test.sh script
```bash
bash test.sh
```
#### Controlling the robot
Either create commands.txt file
```bash
cd rl_x/experiments
touch commands.txt
```
And add the following content to the commands.txt file. Where the values are target x, y and yaw velocities
```bash
1.0
0.0
0.0
```

Or connect a **Xbox 360** controller and control the target x,y velocity with the left joystick and the yaw velocity with the right joystick.


## Miscellaneous notes
- **Evaluation mode:** The environment supports an ```eval_mode``` flag that sets the curriculum to the highest difficulty. The MuJoCo version or rather the non-flax_full_jit algorithms do not pass this flag during evaluation by default (for compatibility reason with the standard gym environments), so if you want to run in eval mode you need to modify the algorithm code to set the eval mode before and after evaluation rollouts.
- **Training with MuJoCo:** While the MuJoCo environment can be used for training, it is not optimized for throughput like the MJX version. The ppo.flax_full_jit algorithm uses optimized hyperparameters for the larger number of parallel environments in MJX, it also uses better performing policy and critic architectures. If you want to train with MuJoCo, consider using the ppo.flax version and adjust hyperparameters accordingly. Importantly, ```config.action_clipping_and_rescaling``` needs to be set to ```False``` as the environment already enforces torque limits and the actual action space is defined as (-inf, inf).