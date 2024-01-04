#!/bin/bash

#SBATCH --job-name=dummy_job
#SBATCH --output=log/output_%j.txt
#SBATCH --error=log/error_%j.txt
#SBATCH --partition=amd2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00

eval "$(/home/bohlinger/miniconda3/bin/conda shell.bash hook)"
conda activate rlx3114

python experiment.py \
    --algorithm.name="ppo.flax" \
    --algorithm.total_timesteps=1000000 \
    --algorithm.nr_steps=20000 \
    --algorithm.minibatch_size=2000 \
    --algorithm.nr_epochs=5 \
    --algorithm.evaluation_frequency=-1 \
    --algorithm.device="cpu" \
    --environment.name="custom_mujoco.ant" \
    --environment.nr_envs=4 \
    --runner.track_console=True \
