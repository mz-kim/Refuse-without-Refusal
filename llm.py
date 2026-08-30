import concurrent.futures
import gc
import time
from typing import List, Optional, Union

import torch
from openai import OpenAI, OpenAIError
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)


class UniversalGenParams:
    def __init__(
        self,
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = -1,
        seed: int = 0,
        stop: Optional[List[str]] = None,
    ):
        self.n = n
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.stop = stop

    def get_vllm_params(self):
        return SamplingParams(
            n=self.n,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            seed=self.seed,
            stop=self.stop,
        )

    def get_openai_params(self):
        return {
            "n": self.n,
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "stop": self.stop,
        }


class GenerationArgs:
    def __init__(
        self,
        engine_input: Union[str, List[str]],
        gen_params: Optional[UniversalGenParams] = None,
        is_multi_turn_input: bool = False,
        is_batch_input: bool = False,
        apply_chat_template: bool = True,
    ):
        self.engine_input = engine_input
        self.gen_params = gen_params or UniversalGenParams()
        self.is_multi_turn_input = is_multi_turn_input
        self.is_batch_input = is_batch_input
        self.apply_chat_template = apply_chat_template


class LLMInferenceOutput:
    def __init__(self, output_seqs, input_prompt=None, output_objects=None):
        self.output_seqs = output_seqs
        self.input_prompt = input_prompt
        self.output_objects = output_objects


class LLMInferenceEngine:
    def __init__(self, model_id, backend, backend_kwargs=None):
        self.model_id = model_id
        self.backend = backend
        self._is_shutdown = False
        self._load_model(dict(backend_kwargs or {}))

    def _load_model(self, backend_kwargs):
        if self.backend == "vllm":
            self.model = LLM(
                model=self.model_id,
                trust_remote_code=True,
                **backend_kwargs,
            )
            self.tokenizer = self.model.get_tokenizer()
        elif self.backend == "vllm-openai":
            base_url = backend_kwargs.pop("base_url", None) or "http://localhost:8000/v1"
            api_key = backend_kwargs.pop("api_key", "EMPTY")
            self.num_openai_workers = backend_kwargs.pop("num_workers", 64)
            self.model = OpenAI(base_url=base_url, api_key=api_key, timeout=None)
            self.tokenizer = None
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def generate(self, gen_args):
        if self.backend == "vllm":
            return self._vllm_generate(gen_args)
        return self._openai_generate(gen_args)

    def _as_batch(self, gen_args):
        if gen_args.is_batch_input:
            return list(gen_args.engine_input)
        return [gen_args.engine_input]

    def _construct_chat(self, prompt):
        if isinstance(prompt, list) and all(isinstance(item, dict) for item in prompt):
            return prompt
        return [{"role": "user", "content": prompt}]

    def _vllm_generate(self, gen_args):
        prompts = self._as_batch(gen_args)
        if gen_args.apply_chat_template:
            prompts = [
                self.tokenizer.apply_chat_template(
                    self._construct_chat(prompt),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]

        outputs = self.model.generate(prompts, gen_args.gen_params.get_vllm_params())
        return [
            LLMInferenceOutput(
                output_seqs=[sequence.text for sequence in sorted(output.outputs, key=lambda x: x.index)],
                input_prompt=output.prompt,
                output_objects=output,
            )
            for output in outputs
        ]

    def _openai_generate(self, gen_args):
        prompts = self._as_batch(gen_args)
        chats = [self._construct_chat(prompt) for prompt in prompts]
        params = gen_args.gen_params.get_openai_params()

        if not gen_args.is_batch_input:
            response = self._openai_inference(chats[0], params)
            return [self._parse_openai_output(response, chats[0])]

        if not chats:
            return []

        results = [None] * len(chats)
        max_workers = min(len(chats), self.num_openai_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._openai_inference, chat, params): index
                for index, chat in enumerate(chats)
            }
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="OpenAI inference",
            ):
                index = futures[future]
                results[index] = self._parse_openai_output(future.result(), chats[index])
        return results

    def _openai_inference(self, messages, params):
        last_error = None
        for attempt in range(10):
            try:
                return self.model.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    **params,
                )
            except OpenAIError as error:
                last_error = error
                if attempt < 9:
                    time.sleep(min(5 * (attempt + 1), 60))
        raise RuntimeError("OpenAI-compatible inference failed after 10 attempts") from last_error

    def _parse_openai_output(self, response, prompt):
        choices = sorted(response.choices, key=lambda choice: choice.index)
        return LLMInferenceOutput(
            output_seqs=[choice.message.content for choice in choices],
            input_prompt=prompt,
            output_objects=response,
        )

    def shutdown(self):
        if self._is_shutdown:
            return
        try:
            if self.backend == "vllm":
                destroy_model_parallel()
                destroy_distributed_environment()
                if hasattr(self.model, "llm_engine"):
                    del self.model.llm_engine.model_executor
        finally:
            if hasattr(self, "model"):
                del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self._is_shutdown = True

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
