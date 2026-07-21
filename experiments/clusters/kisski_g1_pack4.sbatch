#!/bin/bash
#SBATCH --account=kisski-merl
#SBATCH --partition=kisski
#SBATCH --gpus=A100:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=1-23:59:00
#SBATCH --job-name=rlx_g1_pack4
#SBATCH --output=/user/nico.bohlinger/u29390/rlx_g1_cluster/runs/kisski_pack4_%j.log

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "usage: sbatch kisski_g1_pack4.sbatch <commit> <gpu0.args> <gpu1.args> <gpu2.args> <gpu3.args>" >&2
    exit 2
fi

EXPECTED_RLX_COMMIT="$1"
shift
RLX_ROOT="$HOME/rlx_g1_cluster"
RLX_SOURCE="$RLX_ROOT/software/RL-X/$EXPECTED_RLX_COMMIT"
CONFIG_PATHS=("$@")
PIDS=()

for GPU_INDEX in 0 1 2 3; do
    RUN_ARGS_FILE="${CONFIG_PATHS[$GPU_INDEX]}"
    RUN_TAG="$(basename "$RUN_ARGS_FILE" .args)"
    (
        export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
        export CLUSTER_NAME=KISSKI
        export RLX_SOURCE
        export EXPECTED_RLX_COMMIT
        export RUN_ARGS_FILE
        export PYTHON_BIN="$HOME/loco_mjx_cluster/env/bin/python"
        export JAX_CACHE_DIR="$RLX_ROOT/jax_cache/${SLURM_JOB_ID}_${GPU_INDEX}_${RUN_TAG}"
        export RLX_RUN_DIR="$RLX_ROOT/runs/${SLURM_JOB_ID}_${GPU_INDEX}_${RUN_TAG}"
        export RUNNER_RUN_NAME="kisski_${SLURM_JOB_ID}_gpu${GPU_INDEX}"
        bash "$RLX_ROOT/scripts/run_g1_noeval.sh"
    ) > "$RLX_ROOT/runs/${RUN_TAG}_${SLURM_JOB_ID}.log" 2>&1 &
    PIDS+=("$!")
done

STATUS=0
for PID in "${PIDS[@]}"; do
    wait "$PID" || STATUS=1
done
exit "$STATUS"
