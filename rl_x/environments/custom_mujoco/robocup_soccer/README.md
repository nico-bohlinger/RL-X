# RoboCup Soccer MuJoCo Environments

Contains MuJoCo and MJX environments for the official [MuJoCo-based RoboCup Soccer Simulation Server (RCSSServerMJ)](https://gitlab.com/robocup-sim/rcssservermj).

How a trained locomotion policy looks like in RCSSServerMJ:

![locomotion](game.gif)
The full video sequence can be found [here](https://youtu.be/f6MvmqiiU6Q).


The locomotion environment is a straight copy of the ```rl_x/environments/custom_mujoco/robot_locomotion``` environment.
For information on the design decisions and general usage refer to its [README file](https://github.com/nico-bohlinger/RL-X/blob/master/rl_x/environments/custom_mujoco/robot_locomotion/README.md).
The differences are:
- Sinusoidal gait phase manager, that adds a reward term and observations, to encourage the policy to learn a smooth gait pattern.
- Slightly modified foot-related reward terms to improve the gait.
- Reduced domain randomization ranges, observation noise and perturbations.
This means the trained policy might not be directly sim-to-real transferable anymore but this can simply be achieved by increasing the respective parameters again (see the ```robot_locomotion``` environment for reference).
This is done to help training that is focused on the simulation competition.
- The action delay is fixed to 1 step (20 ms) to match the RCSSServerMJ delay.
This is more delay than expected on the real robot.
If sim-to-real transfer is desired, bring back less and also varying action delay (see the ```robot_locomotion``` environment for reference)
- The ```robots``` folder only contains the Booster T1 with the XML and assets as specified by the RCSSServerMJ codebase.
Because the codebase is written in a way to easily add more robots (see the ```robot_locomotion``` environment for reference), it is possible to add different bipedal robots by just adding their config, XML, assets etc. to the ```robots``` folder.


## Training
The training is designed and tuned for the MJX version of the environment and the ```ppo_gru.flax_full_jit``` algorithm, so it is recommended to use this combination for training.
The following training configuration was used to train the model shown in the videos above.
1. Create train.sh file in the experiments folder (all .sh files besides slurm_experiment.sh are ignored by git)
```bash
cd rl_x/experiments
touch train.sh
```
2. Add the following content to the train.sh file
```bash
python experiment.py \
    --algorithm.name=ppo_gru.flax_full_jit \
    --algorithm.total_timesteps=10019143680 \
    --algorithm.gae_lambda=0.7 \
    --algorithm.entropy_coef=0.002 \
    --algorithm.max_grad_norm=3.0 \
    --environment.name=custom_mujoco.robocup_soccer.locomotion.mjx \
    --environment.seed=0 \
    --runner.track_console=False \
    --runner.track_tb=False \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity=<your_wandb_name> \
    --runner.project_name="robocup_soccer_locomotion" \
    --runner.exp_name="some_descriptive_experiment_name" \
    --runner.notes="your notes about the experiment"
```
3. Run the train.sh script on a machine with jax[cuda12] installed and a compatible GPU
```bash
bash train.sh
```

## Testing
### In RL-X
0. If you trained with ```ppo_gru.flax_full_jit``` and now want to test and visualize with ```ppo_gru.flax```, change the policy and critic architectures to match the full_jit version, i.e. simply copy the policy and critic architecture definitions from ```ppo_gru.flax_full_jit``` to the ```ppo_gru.flax``` files.
1. Copy or download the saved model from your training run to the ```rl_x/experiments``` folder.
2. Create test.sh file in the experiments folder (all .sh files besides slurm_experiment.sh are ignored by git)
```bash
cd rl_x/experiments
touch test.sh
```
3. Add the following content to the test.sh file
```bash
python experiment.py \
    --algorithm.name=ppo_gru.flax \
    --environment.name=custom_mujoco.robocup_soccer.locomotion.mujoco \
    --environment.render=True \
    --runner.mode=test \
    --runner.load_model="latest.model"
```
4. Run the test.sh script
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


### In RCSSServerMJ
1. Copy the trained model to the ```rcssserver_deployment``` folder.
2. Run the ```convert.py``` script to convert the model to PyTorch-based weights
```bash
cd rcssserver_deployment
python convert.py --model latest.model --output locomotion_nn.pth --meta-output locomotion_nn_meta.json --obs-dim 82 --action-dim 23
```
3. A locomotion policy trained exactly with the above training configuration is already part of the RCSSServerMJ codebase as an example [here](https://gitlab.com/robocup-sim/rcssservermj/-/tree/master/example/nn_client?ref_type=heads).
You can simply copy / overwrite the converted policy and meta files in the ```nn_client``` folder.
For your own RoboCup team, you can follow the code in ```nn_client.py``` to see how to load the policy and use it for inference in RCSSServerMJ.


## Miscellaneous notes
- **Training with default PPO:** While the default PPO implementation ```ppo.flax_full_jit``` is totally useable for training, it performs worse due to the significant action delay in the robocup soccer environments.
- **Training with MuJoCo:** While the MuJoCo environment can be used for training, it is not optimized for throughput like the MJX version. The ```ppo_gru.flax_full_jit``` algorithm uses optimized hyperparameters for the larger number of parallel environments in MJX, it also uses better performing policy and critic architectures. If you want to train with MuJoCo, consider using the ```ppo_gru.flax``` version and adjust hyperparameters accordingly. Importantly, ```config.action_clipping_and_rescaling``` needs to be set to ```False``` as the environment already enforces torque limits and the actual action space is defined as (-inf, inf).
- **Policy learns to stand still:** In general, locomotion policies can learn to stand still instead of following their commanded target velocities when the penalty terms or other rewards are too strong.
In this specific codebase the ```feet_phase``` term adds a strong positive reward to follow a nice gait pattern, which can be exploited by standing still.
The reward coefficients were tuned to prevent this behavior, but it can still happen when the environment is modified or when using different hyperparameters.
An easy fix is to connect the curriculum in the code, which scales all reward coefficients besides the target velocity tracking reward, with the current xy_tracking_error + episode length instead of the episode return.
Simply track the average xy_tracking_error in an episode and redefine the success criterion of the curriculum based on if it is below a certain threshold (like 0.25) and if the episode length is above a certain threshold (like 500).