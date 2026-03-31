#!/usr/bin/env bash
#
# Single-node GPU test run for LOBArena.
# Soft cluster policy: always keep jobs to one node to reduce compute burn.
#
# Usage:
#   sbatch slurm/sbatch_gpu_tests.sh
#   sbatch --export=ALL,CONDA_ENV=lobs5,PYTEST_ARGS="-q tests/test_batch_evaluation.py" slurm/sbatch_gpu_tests.sh
#
# Logs:
#   outputs/evaluations/slurm_logs/gpu_tests_<jobid>.{out,err}

#SBATCH --job-name=loba-gpu-tests
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=outputs/evaluations/slurm_logs/gpu_tests_%j.out
#SBATCH --error=outputs/evaluations/slurm_logs/gpu_tests_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
mkdir -p outputs/evaluations/slurm_logs

CONDA_ENV="${CONDA_ENV:-lobs5}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/${CONDA_ENV}/bin/python}"
PYTEST_ARGS="${PYTEST_ARGS:--v tests}"

export PYTHONPATH="${ROOT%/LOBArena}"
export JAX_PLATFORMS=gpu
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

echo "[LOBArena][gpu-tests] python=${PYTHON_BIN}"
echo "[LOBArena][gpu-tests] args=${PYTEST_ARGS}"
echo "[LOBArena][gpu-tests] nodes=${SLURM_JOB_NUM_NODES:-1} gpus=${SLURM_GPUS_ON_NODE:-1}"

"$PYTHON_BIN" -m pytest ${PYTEST_ARGS}

