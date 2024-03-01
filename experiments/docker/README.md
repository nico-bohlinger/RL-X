# Docker

This directory contains the Dockerfile and all other necessary scripts to run experiments in a Docker container.

The container setup is designed to pull the latest version of the code from the remote repository, checkout the desired commit, apply the desired patch and start the experiment. In that way many different experiments can be started in parallel without interfering with each others code. This also means that the container is not designed to be used for code development.

The installation of RL-X in the Dockerfile assumes that the machine has an NVIDIA GPU. If that is not the case, adjust the installation according to the general [detailed installation guide](https://nico-bohlinger.github.io/RL-X/#detailed-installation-guide). 

## Example
Initial setup:
- Build the image: ```bash build.sh```
- Create file ```env.config``` and put in the following content: WANDB_API_KEY=\<your_wandb_api_key\>

For running experiments:
1. Create diff for the experiment (```experiment.sh```) and optionally store it in ```/diffs```
2. Edit the commit hash and diff path in slurm_experiment.sh
3. Run the experiment: ```bash run.sh``` or ```sbatch slurm_experiment.sh```

## Important files and directories
**diffs**
- Can be used as storage directory for the diffs that will be applied via ```git apply```
- Will be mounted into the container if a diff is specified for ```run.sh```

**build.sh**
- Runs the docker build command to create the image

**Dockerfile**
- Defines the image

**entrypoint.sh**
- Handles fetching and pulling the latest changes, checkout the commit hash, applying the diff and starting the experiment by running the ```experiment.sh```

**experiment.sh**
- Defines the experiment and its parameters
- This file is most likely part of the applied diff

**run.sh**
- Runs the docker run command to start the container
- With ```bash run.sh -c <commit> -d <diff> -h <headless>``` the container will be started with the given commit, diff and head mode
- If ```-c``` is not specified, the latest commit will be used
- If ```-d``` is not specified, no diff will be applied
- If ```-h``` is not specified, the container will be started in headless mode, i.e. ```-h true```
- The folder the diff is located in will be mounted into the container

**slurm_experiment.sh**
- Wrapper around ```run.sh```
- Pass commit and diff like in ```run.sh```, i.e. ```sbatch slurm_experiment.sh -c <commit> -d <diff>```
- Will always run the container in not-headless mode