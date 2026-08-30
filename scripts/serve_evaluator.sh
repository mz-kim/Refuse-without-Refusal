#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
    echo "Usage: $0 <gpu_ids> [evaluator_model] [port]" >&2
    exit 1
fi

GPU_IDS=$1
EVALUATOR=${2:-meta-llama/Llama-3.3-70B-Instruct}
PORT=${3:-8000}
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}

CUDA_VISIBLE_DEVICES="${GPU_IDS}" vllm serve "${EVALUATOR}" \
    --dtype bfloat16 \
    --api-key EMPTY \
    --tensor-parallel-size "${NUM_GPUS}" \
    --max-model-len 15000 \
    --port "${PORT}"
