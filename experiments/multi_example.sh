#!/bin/bash
python example.py \
    --config.runner.track_tb=True \
    --config.runner.track_wandb=True \
    --config.runner.save_model=True \
    --config.runner.project_name="test" \
    --config.runner.exp_name="test" \
    --config.algorithm.total_timesteps=10000 \
    --config.algorithm.nr_envs=2 \
    --config.environment.seed=0

python example.py \
    --config.runner.track_tb=True \
    --config.runner.track_wandb=True \
    --config.runner.save_model=True \
    --config.runner.project_name="test" \
    --config.runner.exp_name="test" \
    --config.algorithm.total_timesteps=10000 \
    --config.algorithm.nr_envs=2 \
    --config.environment.seed=1

echo "finished"
