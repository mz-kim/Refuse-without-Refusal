import argparse
import os
import random

from datasets import load_dataset

ALPACA_CLEANED_REVISION = "12567cabf869d7c92e573c7c783905fc160e9639"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_examples", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_proc", type=int, default=8)
    parser.add_argument("--output_path", default="dataset/train/alpaca_1024.jsonl")
    return parser.parse_args()


def reformat_alpaca(example):
    instruction = example["instruction"]
    if example["input"]:
        instruction = f"{instruction}\n\n{example['input']}"
    response = example["output"]
    return {
        "instruction": instruction,
        "input": example["input"],
        "response": response,
        "output": response,
    }


def select_examples(dataset, num_examples, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return dataset.select(indices[: min(num_examples, len(dataset))])


def main():
    args = parse_args()
    dataset = load_dataset(
        "yahma/alpaca-cleaned",
        split="train",
        revision=ALPACA_CLEANED_REVISION,
    )
    dataset = dataset.map(
        reformat_alpaca,
        remove_columns=dataset.column_names,
        num_proc=args.num_proc,
    )
    dataset = select_examples(dataset, args.num_examples, args.seed)

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    dataset.to_json(args.output_path, lines=True)
    print(f"Saved {len(dataset)} examples to {args.output_path}")


if __name__ == "__main__":
    main()
