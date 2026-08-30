import argparse
import os
import random
import re

from datasets import Dataset

from llm import GenerationArgs, LLMInferenceEngine, UniversalGenParams
from .refine_util import CONTENT_PROMPTS, POSITION_MODES, POSITION_PROMPT


MODES = tuple(CONTENT_PROMPTS) + tuple(POSITION_MODES)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model used to rewrite the responses.",
    )
    parser.add_argument(
        "--model_engine_backend",
        choices=("vllm", "vllm-openai"),
        default="vllm-openai",
    )
    parser.add_argument("--model_backend_base_url", default=None)
    parser.add_argument("--model_num_gpus", type=int, default=4)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--target_dataset_path", required=True)
    parser.add_argument("--save_dir", default="dataset/train/refined")
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--mode", choices=MODES, default="dataset_refine")
    return parser.parse_args()


def format_prompt(prompt, example, **kwargs):
    response = example.get("response") or example.get("output")
    if response is None:
        raise ValueError("Each example must contain a 'response' or 'output' field")
    return prompt.format(
        user_request=example["instruction"],
        llm_response=response,
        **kwargs,
    )


def build_model_inputs(dataset, mode):
    if mode in POSITION_MODES:
        position = POSITION_MODES[mode]
        return [
            format_prompt(POSITION_PROMPT, example, position=position)
            for example in dataset
        ]

    prompt = CONTENT_PROMPTS[mode]
    return [format_prompt(prompt, example) for example in dataset]


def parse_dataset(dataset):
    def clean_response(text):
        return re.sub(
            r"^\s*\[\s*Response\s*\]\s*:\s*",
            "",
            text.strip(),
            flags=re.IGNORECASE,
        )

    def is_empty_response(text):
        normalized = text.lower().strip().strip('"\'`').rstrip(".")
        return normalized in {"none", "null"}

    cleaned_outputs = [clean_response(output) for output in dataset["refined_output"]]
    dataset = dataset.remove_columns("refined_output")
    dataset = dataset.add_column("refined_output", cleaned_outputs)
    keep_indices = [
        index
        for index, output in enumerate(dataset["refined_output"])
        if not is_empty_response(output)
    ]
    dataset = dataset.select(keep_indices)
    refined_outputs = dataset["refined_output"]

    columns_to_remove = [
        column
        for column in ("response", "output", "refined_output")
        if column in dataset.column_names
    ]
    dataset = dataset.remove_columns(columns_to_remove)
    dataset = dataset.add_column("response", refined_outputs)
    dataset = dataset.add_column("output", refined_outputs)
    return dataset


def main():
    args = parse_args()
    dataset = Dataset.from_json(args.target_dataset_path)

    if args.max_num_examples is not None:
        indices = list(range(len(dataset)))
        random.Random(0).shuffle(indices)
        dataset = dataset.select(sorted(indices[: args.max_num_examples]))

    model_inputs = build_model_inputs(dataset, args.mode)

    if args.model_engine_backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.model_num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
        if "gemma" in args.model.lower():
            backend_kwargs["max_num_seqs"] = 64
    else:
        backend_kwargs = {"base_url": args.model_backend_base_url}

    model = LLMInferenceEngine(
        args.model,
        backend=args.model_engine_backend,
        backend_kwargs=backend_kwargs,
    )
    generation_args = GenerationArgs(
        engine_input=model_inputs,
        gen_params=UniversalGenParams(
            n=1,
            max_new_tokens=2048,
            temperature=0,
        ),
        is_batch_input=True,
        apply_chat_template=True,
    )
    outputs = model.generate(generation_args)
    refined_outputs = [output.output_seqs[0] for output in outputs]
    model.shutdown()

    dataset = dataset.add_column("refined_output", refined_outputs)
    dataset = parse_dataset(dataset)

    os.makedirs(args.save_dir, exist_ok=True)
    output_path = os.path.join(args.save_dir, f"{args.mode}.jsonl")
    dataset.to_json(output_path, lines=True)
    print(f"Saved {len(dataset)} examples to {output_path}")


if __name__ == "__main__":
    main()
