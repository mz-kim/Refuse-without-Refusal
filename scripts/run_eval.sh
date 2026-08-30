#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "Usage: $0 <gpu_ids> <model_path> [evaluator_base_url] [gpu_memory_utilization]" >&2
    exit 1
fi

GPU_IDS=$1
MODEL_PATH=$2
EVALUATOR_BASE_URL=${3:-http://localhost:8000/v1}
GPU_UTILIZATION=${4:-0.85}
IFS=',' read -ra GPU_ARRAY <<< "${GPU_IDS}"
NUM_GPUS=${#GPU_ARRAY[@]}

for BENCHMARK in advbench malicious_instruct oktest xstest; do
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m "eval.${BENCHMARK}.run_eval" \
        --model "${MODEL_PATH}" \
        --use_chat_format \
        --evaluator meta-llama/Llama-3.3-70B-Instruct \
        --evaluator_engine_backend vllm-openai \
        --evaluator_backend_base_url "${EVALUATOR_BASE_URL}" \
        --model_num_gpus "${NUM_GPUS}" \
        --gpu_memory_utilization "${GPU_UTILIZATION}"
done
