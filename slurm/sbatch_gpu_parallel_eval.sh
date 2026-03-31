#!/usr/bin/env bash
#
# Single-node GPU parallel evaluation wrapper for LOBArena.
# Soft cluster policy: always keep jobs to one node to reduce compute burn.
#
# Usage:
#   sbatch slurm/sbatch_gpu_parallel_eval.sh
#   sbatch --export=ALL,DATA_DIR=/path/to/test,POLICY_MODE=random,RUN_NAME=my_gpu_eval slurm/sbatch_gpu_parallel_eval.sh
#
# Logs:
#   outputs/evaluations/slurm_logs/gpu_parallel_eval_<jobid>.{out,err}

#SBATCH --job-name=loba-gpu-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=outputs/evaluations/slurm_logs/gpu_parallel_eval_%j.out
#SBATCH --error=outputs/evaluations/slurm_logs/gpu_parallel_eval_%j.err

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$ROOT"
mkdir -p outputs/evaluations/slurm_logs

CONDA_ENV="${CONDA_ENV:-lobs5}"
PYTHON_BIN="${PYTHON_BIN:-$HOME/miniforge3/envs/${CONDA_ENV}/bin/python}"
DATA_DIR="${DATA_DIR:-/path/to/test_data}"
WORLD_MODEL="${WORLD_MODEL:-historical}"
POLICY_MODE="${POLICY_MODE:-random}"
RUN_NAME="${RUN_NAME:-slurm_gpu_multi_window_${SLURM_JOB_ID:-manual}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/evaluations}"
N_STEPS="${N_STEPS:-25}"
SAMPLE_INDEX="${SAMPLE_INDEX:-0}"
TEST_SPLIT="${TEST_SPLIT:-1.0}"
MULTI_WINDOW_WORKERS="${MULTI_WINDOW_WORKERS:-4}"
RISK_WEIGHTS="${RISK_WEIGHTS:-pnl=1.0,drawdown=0.5,risk=0.1,inventory=0.0}"

export PYTHONPATH="${ROOT%/LOBArena}"
export JAX_PLATFORMS=gpu
export JAX_PLATFORM_NAME=gpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.70
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-12}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"

echo "[LOBArena][gpu-parallel-eval] python=${PYTHON_BIN}"
echo "[LOBArena][gpu-parallel-eval] run_name=${RUN_NAME}"
echo "[LOBArena][gpu-parallel-eval] data_dir=${DATA_DIR}"
echo "[LOBArena][gpu-parallel-eval] nodes=${SLURM_JOB_NUM_NODES:-1} gpus=${SLURM_GPUS_ON_NODE:-1}"

"$PYTHON_BIN" scripts/evaluate_checkpoint.py \
  --world_model "$WORLD_MODEL" \
  --policy_mode "$POLICY_MODE" \
  --data_dir "$DATA_DIR" \
  --n_steps "$N_STEPS" \
  --sample_index "$SAMPLE_INDEX" \
  --test_split "$TEST_SPLIT" \
  --output_root "$OUTPUT_ROOT" \
  --run_name "$RUN_NAME" \
  --device gpu \
  --fast_startup \
  --multi_window \
  --multi_window_workers "$MULTI_WINDOW_WORKERS" \
  --risk_weights "$RISK_WEIGHTS"

