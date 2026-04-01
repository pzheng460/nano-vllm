"""Example: PanGu-72B inference with MTP speculative decoding.

PanGu-72B uses:
  - Sink attention (learnable sink KV prepended to each sequence)
  - MoE (Mixture of Experts) with shared + routed experts
  - Sandwich norm (extra pre/post MLP normalization)
  - Partial RoPE (only qk_rope_dim dimensions rotated)
  - 1 MTP layer with independent head/norm
  - Chat template required for proper generation (default slow thinking)
"""
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # Change to your PanGu-72B model path
    path = os.path.expanduser("/mnt/scratch/weights/PanGu-72B/")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    llm = LLM(
        path,
        enforce_eager=True,     # MoE dynamic routing is incompatible with CUDA graphs
        tensor_parallel_size=4, # 72B model needs multiple GPUs
        max_model_len=4096,
        gpu_memory_utilization=0.9,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=512)

    prompts = [
        "please introduce yourself",
        "list the first 10 prime numbers",
    ]

    # Apply chat template (required for PanGu to generate properly)
    formatted_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        formatted_prompts.append(text)

    outputs = llm.generate(formatted_prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt}")
        print(f"Completion: {output['text']}")


if __name__ == "__main__":
    main()
