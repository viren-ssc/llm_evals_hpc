#!/bin/bash -l
#SBATCH --job-name=canadianness
#SBATCH --clusters=gpsc7
#SBATCH --partition=gpu_a100
#SBATCH --account=ssc_dsai__gpu_a100
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gres=gpu:4
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@120
#SBATCH --comment="image=registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu24.04"
set -euo pipefail
: "${PROJECT_ROOT:?Set PROJECT_ROOT to discovered project directory}"
: "${MODEL_ID:?Set MODEL_ID to gc_llm, apertus or llama}"
INFERENCE_PYTHON="${INFERENCE_PYTHON:-/gpfs/fs7/ssc/ssc_dsai/.cache/envs/vib000/canai-chandra-ocr2/bin/python}"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT/.runtime"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn OMP_NUM_THREADS=4
export VLLM_CACHE_ROOT="$PROJECT_ROOT/outputs/runs/cache-${SLURM_JOB_ID}"
export TMPDIR="/tmp/caneval-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$VLLM_CACHE_ROOT"
# Keep distributed engine traffic local to the allocated node.
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"
monitor_pid=''
child_pid=''
cleanup() {
  if [[ -n "$child_pid" ]]; then kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" || true; fi
  if [[ -n "$monitor_pid" ]]; then kill "$monitor_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu --format=csv -l 30 > "outputs/runs/gpu-${SLURM_JOB_ID}.csv" &
monitor_pid=$!
run_phase() {
  "$INFERENCE_PYTHON" -m llm_evals "$@" &
  child_pid=$!
  wait "$child_pid"
  child_pid=''
}
run_phase generate --model "$MODEL_ID" --resume
# Export uses system Python with its installed workbook library.
PYTHONPATH="$PROJECT_ROOT/src" /usr/bin/python3 -m llm_evals export --partial
