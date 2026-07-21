#!/bin/bash
set -euo pipefail

: "${RLX_SOURCE:?Set RLX_SOURCE to an immutable RL-X source snapshot}"
: "${PYTHON_BIN:?Set PYTHON_BIN to the cluster Python executable}"
: "${RUN_ARGS_FILE:?Set RUN_ARGS_FILE to a one-argument-per-line config}"
: "${EXPECTED_RLX_COMMIT:?Set EXPECTED_RLX_COMMIT to the source commit}"
: "${RLX_RUN_DIR:?Set RLX_RUN_DIR to the job output directory}"

test "$(git -C "$RLX_SOURCE" rev-parse HEAD)" = "$EXPECTED_RLX_COMMIT"
test -z "$(git -C "$RLX_SOURCE" status --porcelain)"
test -f "$RUN_ARGS_FILE"

mapfile -t RUN_ARGS < "$RUN_ARGS_FILE"
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
export PYTHONPATH="$RLX_SOURCE${PYTHON_OVERLAY:+:$PYTHON_OVERLAY}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_DIR="$RLX_RUN_DIR/wandb"
mkdir -p "$JAX_CACHE_DIR" "$WANDB_DIR"

echo "RLX_G1_LAUNCH_OK cluster=${CLUSTER_NAME:-unknown} job=${SLURM_JOB_ID:-none} commit=$EXPECTED_RLX_COMMIT args=$RUN_ARGS_FILE"
"$PYTHON_BIN" -c 'import jax, rl_x; print("jax_version=", jax.__version__); print("jax_devices=", jax.devices()); print("rl_x_source=", rl_x.__file__)'

cd "$RLX_SOURCE/experiments"
"$PYTHON_BIN" experiment.py \
    "${RUN_ARGS[@]}" \
    --algorithm.evaluation_active=False \
    --environment.name=custom_mujoco.robot_locomotion.mjx \
    --environment.train_robot=unitree_g1 \
    --environment.seed=0 \
    --runner.mode=train \
    --runner.jax_cache_dir="$JAX_CACHE_DIR" \
    --runner.track_console=False \
    --runner.track_tb=False \
    --runner.track_wandb=True \
    --runner.save_model=False \
    --runner.wandb_entity=nico-bohlinger \
    --runner.project_name=custom_mujoco_robot_locomotion \
    --runner.run_name="${RUNNER_RUN_NAME:-${SLURM_JOB_ID:-manual}}"
