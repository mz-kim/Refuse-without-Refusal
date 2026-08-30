<div align="center">

<h1>Refuse without Refusal</h1>
<h3>A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models</h3>

<p><strong>EMNLP 2026 Main Conference</strong></p>
<p>Minji Kim · Hyounghun Kim</p>
<p>
  <a href="https://arxiv.org/abs/2609.04714">📄 Paper</a> ·
  <a href="#license">⚖️ License</a>
</p>
<p>
  <a href="https://arxiv.org/abs/2609.04714"><img src="https://img.shields.io/badge/arXiv-2609.04714-b31b1b.svg" alt="arXiv:2609.04714"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License: Apache 2.0"></a>
</p>

<img src="assets/rwr_overview.png" alt="RwR compares Statement-and-Rationale and Rationale-Only responses on harmful and benign queries" width="92%">

</div>

## Overview

Safety-tuned language models often refuse benign requests that merely resemble harmful ones. RwR studies how the internal structure of a safety response influences this false-refusal behavior while preserving refusal performance on genuinely harmful requests.

The experiments decompose a safety response into a concise refusal statement and a request-dependent rationale, then vary three structural properties:

<p align="center">
  <img src="assets/rwr_response_structure.png" alt="A safety response decomposed into a refusal statement and refusal rationale" width="88%">
</p>

- **Components:** Statement-Only, Rationale-Only, and Statement and Rationale
- **Position:** Beginning, Middle, and End
- **Rationale explicitness:** Generic and Request-Specific

The repository provides the prompts and utilities for constructing these response variants, preparing the utility–safety training mixtures, QLoRA fine-tuning, four safety evaluations, and the URIAL prompting study.

## Setup

The released pipeline targets Python 3.11 and CUDA-capable GPUs.

```bash
conda create -n rwr python=3.11
conda activate rwr
pip install -r requirements.txt
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

The pinned PyTorch, vLLM, bitsandbytes, and FlashAttention packages require a compatible CUDA installation. Install the PyTorch build matching the local CUDA runtime first if the pinned build is not suitable for your system.

Run the commands below from the repository root.

## Dataset construction

### Source preparation

Prepare the Safety-Tuned LLaMAs source responses, sample 1,024 utility examples from Alpaca-Cleaned, and fetch the AdvBench and OKTest evaluation inputs:

```bash
./scripts/prepare_data.sh
```

The script pins each upstream revision and verifies direct downloads using SHA-256 checksums. It creates the following local files:

```text
dataset/
|-- eval/
|   |-- advbench/harmful_behaviors.csv
|   `-- oktest/oktest.csv
`-- train/
    |-- alpaca_1024.jsonl
    `-- original/
        `-- safety_only_data_Instructions.json
```

### Safety-response variants

Response construction uses `meta-llama/Llama-3.3-70B-Instruct`. Start an OpenAI-compatible vLLM server in a separate terminal:

```bash
./scripts/serve_evaluator.sh 0,1,2,3 \
  meta-llama/Llama-3.3-70B-Instruct \
  8000
```

The generation prompts in `dataset/refine_util.py` correspond to Appendix L, Tables 36–41. First normalize the source responses into the Statement and Rationale format:

```bash
python -m dataset.refine_dataset \
  --target_dataset_path dataset/train/original/safety_only_data_Instructions.json \
  --mode dataset_refine \
  --model_backend_base_url http://localhost:8000/v1
```

This produces `dataset/train/refined/dataset_refine.jsonl`, which is also the Beginning condition in the position experiment. Responses unrelated to safety are returned as `None` by the generation prompt and removed during post-processing.

Generate the component variants from the normalized responses:

```bash
python -m dataset.refine_dataset \
  --target_dataset_path dataset/train/refined/dataset_refine.jsonl \
  --mode only_refusal \
  --model_backend_base_url http://localhost:8000/v1

python -m dataset.refine_dataset \
  --target_dataset_path dataset/train/refined/dataset_refine.jsonl \
  --mode only_rationale \
  --model_backend_base_url http://localhost:8000/v1
```

Generate the Middle and End position variants:

```bash
python -m dataset.refine_dataset \
  --target_dataset_path dataset/train/refined/dataset_refine.jsonl \
  --mode middle_position \
  --model_backend_base_url http://localhost:8000/v1

python -m dataset.refine_dataset \
  --target_dataset_path dataset/train/refined/dataset_refine.jsonl \
  --mode end_position \
  --model_backend_base_url http://localhost:8000/v1
```

Generate Generic and Request-Specific rationales from the Rationale-Only responses:

```bash
python -m dataset.refine_dataset \
  --target_dataset_path dataset/train/refined/only_rationale.jsonl \
  --mode only_generic_rationale \
  --model_backend_base_url http://localhost:8000/v1

python -m dataset.refine_dataset \
  --target_dataset_path dataset/train/refined/only_rationale.jsonl \
  --mode only_request_specific_rationale \
  --model_backend_base_url http://localhost:8000/v1
```

| Mode | Experimental condition |
| --- | --- |
| `dataset_refine` | Statement and Rationale / Beginning |
| `only_refusal` | Statement-Only |
| `only_rationale` | Rationale-Only |
| `middle_position` | Middle |
| `end_position` | End |
| `only_generic_rationale` | Generic rationale |
| `only_request_specific_rationale` | Request-Specific rationale |

## Training

### Training-data preparation

Each training set contains 1,024 utility examples and 256 safety examples. For example, construct the Rationale-Only mixture with:

```bash
python -m utils.generate_safety_mixture \
  --utility_dataset_path dataset/train/alpaca_1024.jsonl \
  --safety_dataset_path dataset/train/refined/only_rationale.jsonl \
  --num_utility_examples 1024 \
  --num_safety_examples 256 \
  --seed 0 \
  --output_path dataset/train/mixture/alpaca1024+only_rationale256.jsonl
```

Use the corresponding refined file as `--safety_dataset_path` to construct another experimental condition.

### QLoRA fine-tuning

The experiments fine-tune base pretrained models rather than instruction-tuned variants. Run QLoRA training with:

```bash
./scripts/train.sh 0,1,2,3 \
  meta-llama/Llama-3.1-8B \
  dataset/train/mixture/alpaca1024+only_rationale256.jsonl
```

The released configuration uses LoRA rank 64, alpha 16, dropout 0.1, 4-bit NF4 quantization, a constant learning rate of `1e-4`, 10 epochs, and a maximum sequence length of 2,048. Checkpoints and merged models are written under `outputs/finetuned_models/`.

The paper applies the same procedure to Llama-3.1-8B, Mistral-7B-v0.3, Gemma-2-9B, and Qwen2.5-7B.

## Evaluation

The evaluation suite uses AdvBench and MaliciousInstruct for harmful requests, and the safe split of XSTest and OKTest for pseudo-harmful requests. `meta-llama/Llama-3.3-70B-Instruct` serves as the automatic judge.

Start the judge server on GPUs separate from target-model inference:

```bash
./scripts/serve_evaluator.sh 4,5,6,7 \
  meta-llama/Llama-3.3-70B-Instruct \
  8000
```

Run all four benchmarks with:

```bash
./scripts/run_eval.sh 0 \
  outputs/finetuned_models/Llama-3.1-8B_qlora/alpaca1024+only_rationale256 \
  http://localhost:8000/v1
```

MaliciousInstruct and XSTest are loaded from pinned Hugging Face revisions at evaluation time. Evaluation records and metrics are saved below the local model directory, or under `outputs/remote_models/` when evaluating a Hugging Face model ID.

### URIAL prompting

The `prompts/` directory contains the Statement and Rationale, Statement-Only, and Rationale-Only demonstrations used in the prompting study. Pass a template to the XSTest or OKTest evaluator with `--icl_prompt`:

```bash
python -m eval.xstest.run_eval \
  --model meta-llama/Llama-3.1-8B \
  --model_num_gpus 1 \
  --icl_prompt prompts/URIAL_1k_v4_only_rationale.txt \
  --evaluator_backend_base_url http://localhost:8000/v1
```

## Citation

```bibtex
@misc{kim2026refuserefusalstructuralanalysis,
  title         = {Refuse without Refusal: A Structural Analysis of Safety-Tuning Responses for Reducing False Refusals in Language Models},
  author        = {Minji Kim and Hyounghun Kim},
  year          = {2026},
  eprint        = {2609.04714},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2609.04714}
}
```

## License

The original code, documentation, and configuration files in this repository
are licensed under the [Apache License 2.0](LICENSE). The prompt templates in
`prompts/` are adapted from [URIAL](https://github.com/Re-Align/URIAL), which is
also licensed under Apache 2.0; see [NOTICE](NOTICE) for attribution.

This repository does not redistribute the datasets or model weights used by
the paper. Files downloaded or generated by the preparation scripts, and any
models used with the pipeline, remain subject to their respective upstream
licenses and terms. In particular, Safety-Tuned LLaMAs identifies its code as
MIT-licensed and its data as CC BY-NC 4.0. The paper itself is a separate work
and is not covered by this repository's software license.

## Acknowledgements

This work builds on resources released by [Safety-Tuned LLaMAs](https://github.com/vinid/safety-tuned-llamas), [Alpaca-Cleaned](https://huggingface.co/datasets/yahma/alpaca-cleaned), [AdvBench](https://github.com/llm-attacks/llm-attacks), [MaliciousInstruct](https://huggingface.co/datasets/walledai/MaliciousInstruct), [XSTest](https://huggingface.co/datasets/walledai/XSTest), [OKTest](https://github.com/InvokerStark/OverKill), and [URIAL](https://github.com/Re-Align/URIAL).
