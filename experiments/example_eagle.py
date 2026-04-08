import os
import sys
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    model_path = os.path.expanduser("/mnt/data/weights/Qwen2-7B-Instruct/")
    draft_path = os.path.expanduser("/mnt/data/weights/EAGLE-Qwen2-7B-Instruct/")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    llm = LLM(
        model_path,
        draft_model=draft_path,
        num_speculative_tokens=5,
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(temperature=0.001, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
