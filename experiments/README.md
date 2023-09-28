# Experiments


## Important files

**experiment.py**
- Imports the runner and runs the experiment

**start_experiment.sh**
- Works as a template for how to setup an experiment with the command line
- Runs the experiment in a background process and saves the output to ```log/out_and_err.txt```
- Can be used to run or even schedule multiple experiments with different configs

**stop_experiment.sh**
- Stops all running experiments


## Examples

**Use all default configs**
```
python experiment.py
```

**Overwrite configs by setting the command line arguments**
```
python experiment.py --algorithm.gamma=0.9
```

**Change algorithm and environment**
```
python experiment.py --algorithm.name="sac.flax" --environment.name="envpool.mujoco.humanoid_v4"
```

**Structure and run experiments with a bash script**
```
bash start_experiment.sh
```

## Experiment tracking
- Set ```--runner.track_console=True``` to get detailed console output of the training run
- Set ```--runner.track_tb=True``` to track the training run with Tensorboard

**Weights and Biases**
- Set ```--runner.track_tb=True --runner.track_wandb=True```
- Set ```--runner.wandb_entity=<your_wandb_entity>``` to your Weights and Biases account name
- Uses ```--runner.project_name=<your_project_name>``` as the Weights and Biases project name
- Uses ```--runner.exp_name=<your_exp_name>``` as the Weights and Biases group name in the given project
- Uses ```--runner.notes=<your_exp_notes>``` as the notes for the training run

## Saving & Loading models
- Set ```--runner.save_model=True``` to save a trained model
- Set ```--runner.load_model=<path_to_model>``` to load a saved model
- Saved models are stored in ```runs/<project_name>/<exp_name>/<run_name>/models```
    - The ```runs``` directory is created in the directory where the training script is executed
    - The ```<project_name>``` is set with ```--runner.project_name=<your_project_name>```
    - The ```<exp_name>``` is set with ```--runner.exp_name=<your_exp_name>```
    - The ```<run_name>``` is a timestamp of the training run or can be set manually with ```--runner.run_name=<your_run_name>```