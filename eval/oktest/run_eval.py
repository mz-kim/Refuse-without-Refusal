import argparse
import json
import os
import random

from datasets import Dataset

from eval.safety_eval_utils import (
    IMPLICIT_REFUSAL_JUDGE_PROMPT,
    compute_implicit_refusal_metrics,
    refusal_judge_output_parser,
    stop_remover,
)
from llm import GenerationArgs, LLMInferenceEngine, UniversalGenParams

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--model_engine_backend",
        choices=("vllm", "vllm-openai"),
        default="vllm",
    )
    parser.add_argument("--model_backend_base_url", default=None)
    parser.add_argument("--evaluator", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument(
        "--evaluator_engine_backend",
        choices=("vllm", "vllm-openai"),
        default="vllm-openai",
    )
    parser.add_argument("--evaluator_backend_base_url", default=None)
    parser.add_argument("--model_num_gpus", type=int, default=1)
    parser.add_argument("--evaluator_num_gpus", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--use_chat_format", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_num_examples", type=int, default=None)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--data_dir", default="dataset/eval/oktest/oktest.csv")
    parser.add_argument("--icl_prompt", default=None)
    args = parser.parse_args()
    if args.save_dir is None:
        args.save_dir = (
            os.path.join(args.model, "evals/oktest")
            if os.path.exists(args.model)
            else os.path.join(
                "outputs",
                "remote_models",
                args.model.split("/")[-1],
                "evals/oktest",
            )
        )
    return args

def load_and_reformat(data_dir):
    dataset = Dataset.from_csv(data_dir)
    dataset = dataset.map(
        lambda example: {"prompt": example["prompt"].rstrip()}
    )
    return dataset

def main():
    args = parse_args()

    if args.model_engine_backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.model_num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_num_seqs": 16,
        }
    elif args.model_engine_backend == "vllm-openai":
        backend_kwargs = {"base_url": args.model_backend_base_url}
    else:
        backend_kwargs = {}
    model = LLMInferenceEngine(args.model, backend=args.model_engine_backend, backend_kwargs=backend_kwargs)
    dataset = load_and_reformat(args.data_dir)

    if args.max_num_examples is not None:
        random.seed(0)
        random_ids = sorted(random.sample(range(len(dataset)), min(args.max_num_examples, len(dataset))))
        dataset = dataset.select(random_ids)

    prediction_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)

    instructions = dataset["prompt"]
    stop_strings = None
    if args.icl_prompt is not None:
        stop_strings = ["# Query:", "# Answer:", "# Instruction"]
        with open(args.icl_prompt, "r", encoding="utf-8") as file:
            prompt_template = file.read().strip()
        instructions = [
            prompt_template.format(instruction=instruction)
            for instruction in instructions
        ]
        args.use_chat_format = False
        prediction_gen_params = UniversalGenParams(
            n=1,
            max_new_tokens=2048,
            temperature=0,
            stop=stop_strings,
        )

    prediction_gen_args = GenerationArgs(
        engine_input=instructions,
        gen_params=prediction_gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
        apply_chat_template=args.use_chat_format,
    )

    model_outputs = model.generate(prediction_gen_args)
    model_outputs = [output.output_seqs[0] for output in model_outputs]
    if stop_strings is not None:
        model_outputs = [
            stop_remover(output, stop_strings).strip().rstrip("`")
            for output in model_outputs
        ]
    dataset = dataset.add_column(name="model_output", column=model_outputs)
    model.shutdown()

    if args.evaluator_engine_backend == "vllm":
        backend_kwargs = {
            "tensor_parallel_size": args.evaluator_num_gpus,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        }
    elif args.evaluator_engine_backend == "vllm-openai":
        backend_kwargs = {"base_url": args.evaluator_backend_base_url}
    else:
        backend_kwargs = {}
    evaluator = LLMInferenceEngine(
        args.evaluator,
        backend=args.evaluator_engine_backend,
        backend_kwargs=backend_kwargs,
    )
    eval_inputs = [
        IMPLICIT_REFUSAL_JUDGE_PROMPT.format(
            prompt=example["prompt"],
            model_output=example["model_output"],
        )
        for example in dataset
    ]
    eval_gen_params = UniversalGenParams(n=1, max_new_tokens=2048, temperature=0)
    eval_gen_args = GenerationArgs(
        engine_input=eval_inputs,
        gen_params=eval_gen_params,
        is_multi_turn_input=False,
        is_batch_input=True,
        apply_chat_template=True,
    )

    eval_outputs = evaluator.generate(eval_gen_args)
    eval_outputs = [output.output_seqs[0] for output in eval_outputs]
    evaluator.shutdown()
    eval_labels = [refusal_judge_output_parser(output) for output in eval_outputs]
    dataset = dataset.add_column(name="refusal_clf_output", column=eval_outputs)
    dataset = dataset.add_column(name="refusal_clf_label", column=eval_labels)
    dataset = dataset.add_column(name="refusal_clf", column=[args.evaluator] * len(dataset))

    metrics = compute_implicit_refusal_metrics(dataset)
    print(metrics)

    if args.max_num_examples is None:
        results_save_path = os.path.join(args.save_dir, f"results_oktest_evaluator_{args.evaluator.split('/')[-1]}.jsonl")
        metrics_save_path = os.path.join(args.save_dir, f"metrics_oktest_evaluator_{args.evaluator.split('/')[-1]}.json")
    else:
        results_save_path = os.path.join(args.save_dir, f"results_oktest_{args.max_num_examples}_evaluator_{args.evaluator.split('/')[-1]}.jsonl")
        metrics_save_path = os.path.join(args.save_dir, f"metrics_oktest_{args.max_num_examples}_evaluator_{args.evaluator.split('/')[-1]}.json")

    os.makedirs(args.save_dir, exist_ok=True)
    dataset.to_json(results_save_path, lines=True)
    with open(metrics_save_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

if __name__ == "__main__":
    main()
