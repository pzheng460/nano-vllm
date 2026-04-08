"""Example: MiMo-7B-Base inference with MTP speculative decoding."""
from nanovllm import LLM, SamplingParams


def main():
    path = "/root/huggingface/MiMo-7B-Base/"

    # MTP is auto-detected from model config (num_nextn_predict_layers > 0)
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1, max_model_len=4096)

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
