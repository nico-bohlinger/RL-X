#!/bin/bash
{
python experiment.py \
    --config.algorithm.name="ppo.pytorch" \
    --config.algorithm.total_timesteps=10000 \
    --config.environment.name="gym.mujoco.humanoid_v4" \
    --config.environment.nr_envs=1 \
    --config.environment.seed=0 \
    --config.runner.mode="train" \
    --config.runner.track_console=False \
    --config.runner.track_tb=True \
    --config.runner.track_wandb=True \
    --config.runner.save_model=True \
    --config.runner.wandb_entity="placeholder" \
    --config.runner.project_name="placeholder" \
    --config.runner.exp_name="placeholder" \
    --config.runner.notes="placeholder" \
    >log/out_and_err.txt 2>&1 &
pid=$!

echo "Experiment started"
wait $pid
echo "Experiment finished"
} &
