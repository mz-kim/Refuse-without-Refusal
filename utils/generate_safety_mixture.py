import argparse
import os
import random

from datasets import Dataset, concatenate_datasets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utility_dataset_path", required=True)
    parser.add_argument("--safety_dataset_path", required=True)
    parser.add_argument("--num_utility_examples", type=int, default=1024)
    parser.add_argument("--num_safety_examples", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output_path",
        default="dataset/train/mixture/alpaca1024+safety256.jsonl",
    )
    return parser.parse_args()


def standardize(example):
    response = example.get("response") or example.get("output")
    if response is None:
        raise ValueError("Each example must contain a 'response' or 'output' field")
    return {
        "instruction": example["instruction"],
        "response": response,
    }


def load_standardized(path):
    dataset = Dataset.from_json(path)
    return dataset.map(standardize, remove_columns=dataset.column_names)


def sample(dataset, num_examples, seed):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return dataset.select(indices[: min(num_examples, len(dataset))])


def main():
    args = parse_args()
    utility = sample(
        load_standardized(args.utility_dataset_path),
        args.num_utility_examples,
        args.seed,
    )
    safety = sample(
        load_standardized(args.safety_dataset_path),
        args.num_safety_examples,
        args.seed,
    )
    mixture = concatenate_datasets([utility, safety])

    parent = os.path.dirname(args.output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    mixture.to_json(args.output_path, lines=True)
    print(
        f"Saved {len(mixture)} examples "
        f"({len(utility)} utility + {len(safety)} safety) to {args.output_path}"
    )


if __name__ == "__main__":
    main()
