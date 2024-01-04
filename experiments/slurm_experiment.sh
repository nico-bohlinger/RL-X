#!/bin/bash

#SBATCH --job-name=rlx_experiment
#SBATCH --output=log/out_and_err.txt
#SBATCH --error=log/out_and_err.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

eval "$(/home/bohlinger/miniconda3/bin/conda shell.bash hook)"
conda activate rlx

python experiment.py \
    --algorithm.name="ppo.pytorch" \
    --algorithm.total_timesteps=10000 \
    --environment.name="gym.mujoco.humanoid_v4" \
    --environment.nr_envs=1 \
    --environment.seed=0 \
    --runner.mode="train" \
    --runner.track_console=False \
    --runner.track_tb=True \
    --runner.track_wandb=True \
    --runner.save_model=True \
    --runner.wandb_entity="placeholder" \
    --runner.project_name="placeholder" \
    --runner.exp_name="placeholder" \
    --runner.notes="placeholder"
