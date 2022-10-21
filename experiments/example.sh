#!/bin/bash
python example.py \
    --config.runner.track_tb=True \
    --config.runner.track_wandb=True \
    --config.runner.save_model=False \
    --config.runner.wandb_entity="nico-bohlinger" \
    --config.runner.project_name="mujoco_gym" \
    --config.runner.exp_name="humanoid_envpool_rlx_sac_pytorch" \
    --config.algorithm.total_timesteps=1000000000000 \
    --config.algorithm.nr_envs=1 \
    --config.algorithm.device="cuda" \
    --config.algorithm.learning_starts=256 \
    --config.algorithm.batch_size=256 \

echo "finished"
