"""Example: MiMo-7B-Base inference with SSD (Speculative Streaming Decoding).

SSD runs the MTP draft layers on a separate GPU asynchronously,
pre-computing a tree of speculations with fan-out for instant cache hits.
Uses penultimate (second-to-last) layer hidden states for early MTP prediction.

Requires 2 GPUs: GPU 0 for the target model, GPU 1 for the MTP draft.
"""
from nanovllm import LLM, SamplingParams


def main():
    path = "/root/huggingface/MiMo-7B-Base/"

    # Enable SSD: draft_async=True triggers async MTP draft on GPU 1
    # num_speculative_tokens: recursive MTP generates K tokens per step (K > 1)
    # async_fan_out: number of tree branches per position
    llm = LLM(
        path,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        draft_async=True,                # enable SSD
        draft_gpu=1,                     # draft on GPU 1
        num_speculative_tokens=3,        # recursive MTP: 3 tokens per step
        async_fan_out=3,                 # 3-way fan-out per position
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=128)
    prompts = [
        "Hello, MiMo!",
        "List all prime numbers within 50:",
        "What is the capital of France?",
        "Write a Python function to calculate fibonacci:",
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
