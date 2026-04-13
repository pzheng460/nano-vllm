"""Profile sync MTP vs async SSD with torch.profiler.
Outputs Chrome trace JSON files for viewing in chrome://tracing or https://ui.perfetto.dev/
"""
import argparse
import os
import torch
from nanovllm import LLM, SamplingParams


def profile(name, llm, prompts, sp, output_dir):
    # Warmup
    llm.generate(prompts[:2], sp)

    trace_path = os.path.join(output_dir, f"{name}.json")
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        with_stack=True,
    ) as prof:
        llm.generate(prompts, sp)

    prof.export_chrome_trace(trace_path)
    print(f"Trace saved to: {trace_path}")

    # Also print top ops summary
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sync1", "async1", "async3",
                                            "eagle_sync", "eagle_async"], required=True)
    parser.add_argument("--model", default="/mnt/data/peizhen/MiMo-7B-RL")
    parser.add_argument("--draft-model", default="/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct")
    parser.add_argument("--K", type=int, default=3)
    parser.add_argument("--early-layers", type=int, default=2)
    parser.add_argument("--fan-out", type=int, default=3)
    parser.add_argument("--prompts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    prompts = [
        "A train travels at 60 mph for 3 hours. How far does it travel?",
    ] * args.prompts

    from nanovllm import SamplingParams

    if args.mode == "sync1":
        llm = LLM(args.model, enforce_eager=True, tensor_parallel_size=1, max_model_len=4096)
        profile("sync_mtp_k1", sp=sp, llm=llm, prompts=prompts, output_dir=args.output_dir)
    elif args.mode in ("async1", "async3"):
        k = args.K or (1 if args.mode == "async1" else 3)
        llm = LLM(
            args.model,
            enforce_eager=True,
            tensor_parallel_size=1,
            max_model_len=4096,
            draft_async=True,
            draft_gpu=1,
            num_speculative_tokens=k,
            async_fan_out=args.fan_out,
            ssd_early_layers=args.early_layers,
        )
        profile(f"async_ssd_k{k}", sp=sp, llm=llm, prompts=prompts, output_dir=args.output_dir)
    elif args.mode == "eagle_sync":
        eagle_model = "/mnt/data/peizhen/Qwen2-7B-Instruct"
        llm = LLM(eagle_model, draft_model=args.draft_model, enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   num_speculative_tokens=args.K)
        profile(f"eagle_sync_k{args.K}", sp=sp, llm=llm, prompts=prompts, output_dir=args.output_dir)
    elif args.mode == "eagle_async":
        eagle_model = "/mnt/data/peizhen/Qwen2-7B-Instruct"
        llm = LLM(eagle_model, draft_model=args.draft_model, enforce_eager=True,
                   tensor_parallel_size=1, max_model_len=4096,
                   num_speculative_tokens=args.K,
                   draft_async=True, draft_gpu=1,
                   async_fan_out=args.fan_out,
                   ssd_early_layers=args.early_layers,
                   ssd_tree_decode=True)
        profile(f"eagle_async_k{args.K}_el{args.early_layers}", sp=sp, llm=llm, prompts=prompts, output_dir=args.output_dir)
