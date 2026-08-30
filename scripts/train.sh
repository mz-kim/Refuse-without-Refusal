#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 <gpu_ids> <model> <dataset_path>" >&2
    exit 1
fi

GPU_IDS=$1
MODEL=$2
DATASET_PATH=$3

CUDA_VISIBLE_DEVICES="${GPU_IDS}" python -m train.safety_ft \
    --model "${MODEL}" \
    --dataset_path "${DATASET_PATH}"
