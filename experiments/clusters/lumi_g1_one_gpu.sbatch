#!/bin/bash
#SBATCH --account=project_465003217
#SBATCH --partition=small-g
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-23:59:00
#SBATCH --job-name=rlx_g1
#SBATCH --output=/project/project_465003217/rlx_g1_cluster/runs/lumi_g1_%j.log

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch lumi_g1_one_gpu.sbatch <commit> <run.args>" >&2
    exit 2
fi

module use /appl/local/csc/modulefiles
module load jax/0.4.38
export SING_FLAGS="${SING_FLAGS// -B \/usr\/lib64\/libjson-c.so.3:\/usr\/lib64\/libjson-c.so.3/}"
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_gpu_enable_command_buffer="

export CLUSTER_NAME=LUMI
export EXPECTED_RLX_COMMIT="$1"
export RUN_ARGS_FILE="$2"
export RLX_ROOT=/project/project_465003217/rlx_g1_cluster
export RLX_SOURCE="$RLX_ROOT/software/RL-X/$EXPECTED_RLX_COMMIT"
export PYTHON_BIN=/appl/local/csc/soft/ai/bin/python
export PYTHON_OVERLAY=/project/project_465003217/loco_mjx_cluster/python_overlay
export JAX_CACHE_DIR="${TMPDIR:-/tmp}/rlx_jax_cache_${SLURM_JOB_ID}"
export RLX_RUN_DIR="$RLX_ROOT/runs/${SLURM_JOB_ID}"
export RUNNER_RUN_NAME="lumi_${SLURM_JOB_ID}"

bash "$RLX_ROOT/scripts/run_g1_noeval.sh"
