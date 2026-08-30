import argparse
import os

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_adapter_path", required=True)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--attn_implementation", default="flash_attention_2")
    return parser.parse_args()

def merge(lora_adapter_path, attn_implementation="flash_attention_2", save_path=None):
    peft_config = PeftConfig.from_pretrained(lora_adapter_path)
    model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(lora_adapter_path)

    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
    model = PeftModel.from_pretrained(model, lora_adapter_path).to("cuda")
    model = model.merge_and_unload()

    if save_path is None:
        save_path = os.path.join(lora_adapter_path, "lora_merged")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Saved model to {save_path}")

def main():
    args = parse_args()
    merge(args.lora_adapter_path, args.attn_implementation, save_path=args.save_path)

if __name__ == '__main__':
    main()
