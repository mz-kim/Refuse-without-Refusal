#!/usr/bin/env bash
set -euo pipefail

download_and_verify() {
    local url="$1"
    local output_path="$2"
    local expected_sha256="$3"
    local temporary_path="${output_path}.tmp"

    mkdir -p "$(dirname "${output_path}")"
    if ! wget --quiet --show-progress --output-document "${temporary_path}" "${url}"; then
        rm -f "${temporary_path}"
        echo "Download failed: ${url}" >&2
        return 1
    fi

    if ! printf '%s  %s\n' "${expected_sha256}" "${temporary_path}" | sha256sum --check --status; then
        rm -f "${temporary_path}"
        echo "Checksum verification failed: ${output_path}" >&2
        return 1
    fi

    mv "${temporary_path}" "${output_path}"
    echo "Prepared ${output_path}"
}

download_and_verify \
    "https://raw.githubusercontent.com/vinid/safety-tuned-llamas/36a4b8d5c2177ed165bf61f59b590161394f7f12/data/training/safety_only_data_Instructions.json" \
    "dataset/train/original/safety_only_data_Instructions.json" \
    "7dde08acf87f10cf74fd1b76f56538eae8e413ae8e6b09e3f7bcd97769058dbe"

python -m utils.generate_training_data \
    --num_examples 1024 \
    --seed 0 \
    --output_path dataset/train/alpaca_1024.jsonl

download_and_verify \
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv" \
    "dataset/eval/advbench/harmful_behaviors.csv" \
    "6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1"

download_and_verify \
    "https://raw.githubusercontent.com/InvokerStark/OverKill/9551a7ff7b2cb18d178e64fb5cc95e4684311b3b/data/OKTest.csv" \
    "dataset/eval/oktest/oktest.csv" \
    "1be3e8d3fe6b45a170653da63c8cf70dbf97508a48735e65db0713a44e158860"
