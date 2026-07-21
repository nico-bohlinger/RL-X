#!/bin/bash
#SBATCH --account=kisski-merl
#SBATCH --partition=grete:interactive
#SBATCH --qos=2h
#SBATCH --gpus=1g.20gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=rlx_g1_smoke
#SBATCH --output=/user/nico.bohlinger/u29390/rlx_g1_cluster/runs/kisski_g1_smoke_%j.log

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch kisski_g1_one_gpu.sbatch <commit> <run.args>" >&2
    exit 2
fi

export CLUSTER_NAME=KISSKI
export EXPECTED_RLX_COMMIT="$1"
export RUN_ARGS_FILE="$2"
export RLX_ROOT="$HOME/rlx_g1_cluster"
export RLX_SOURCE="$RLX_ROOT/software/RL-X/$EXPECTED_RLX_COMMIT"
export PYTHON_BIN="$HOME/loco_mjx_cluster/env/bin/python"
export JAX_CACHE_DIR="$RLX_ROOT/jax_cache/${SLURM_JOB_ID}_smoke"
export RLX_RUN_DIR="$RLX_ROOT/runs/${SLURM_JOB_ID}_smoke"
export RUNNER_RUN_NAME="kisski_smoke_${SLURM_JOB_ID}"

bash "$RLX_ROOT/scripts/run_g1_noeval.sh"
