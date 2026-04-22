"""vLLM EAGLE reference run on Spec-Bench prompts.

Feeds a Spec-Bench question.jsonl through vLLM with EAGLE spec decode, using
the same chat-template logic as our nano-vllm harness. Reports cumulative
per-position accept rate + mean accept length from vLLM's SpecDecodingStats.

Run with the vLLM venv (not nano-vllm's):
    /mnt/data/peizhen/vllm/.venv/bin/python experiments/bench_vllm_specbench.py \
        --model /mnt/data/peizhen/vicuna-7b-v1.3 \
        --draft /mnt/data/peizhen/EAGLE-Vicuna-7B-v1.3 \
        --input tests/data/specbench_120.jsonl \
        --output /tmp/specbench_align/vicuna_vllm.jsonl \
        --K 3 --max-new-tokens 512
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tests.harness.chat import build_turn_input_ids  # noqa: E402


def load_questions(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def collect_per_pos(llm, K):
    drafts = 0
    accepted = 0
    per_pos = [0] * K
    for m in llm.get_metrics():
        if m.name == "vllm:spec_decode_num_drafts":
            drafts += int(m.value)
        elif m.name == "vllm:spec_decode_num_accepted_tokens":
            accepted += int(m.value)
        elif m.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            for i, v in enumerate(m.values):
                if i < K:
                    per_pos[i] += int(v)
    return {"drafts": drafts, "accepted": accepted, "per_pos": per_pos}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--method", choices=("eagle", "eagle3"), default="eagle")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-model-len", type=int, default=2048)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    questions = load_questions(args.input)
    print(f"[vllm-ref] {len(questions)} questions from {args.input}", flush=True)

    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=max(args.max_model_len, 2048),
        gpu_memory_utilization=0.85,
        speculative_config={
            "method": args.method,
            "model": args.draft,
            "num_speculative_tokens": args.K,
        },
        disable_log_stats=False,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    answers = []
    t_start = time.perf_counter()

    for qi, q in enumerate(questions):
        prior_replies = []
        turns_text = []
        new_tokens_list = []
        wall_time_list = []

        for _ in range(len(q["turns"])):
            ids = build_turn_input_ids(tokenizer, q["turns"], prior_replies)
            t0 = time.perf_counter()
            out = llm.generate(
                [TokensPrompt(prompt_token_ids=ids)],
                sampling_params=sp,
                use_tqdm=False,
            )
            elapsed = time.perf_counter() - t0
            gen = out[0].outputs[0]
            prior_replies.append(gen.text)
            turns_text.append(gen.text)
            new_tokens_list.append(len(gen.token_ids))
            wall_time_list.append(elapsed)

        answers.append({
            "question_id": q["question_id"],
            "answer_id": uuid.uuid4().hex,
            "category": q.get("category", ""),
            "model_id": f"vllm-eagle-K{args.K}-{Path(args.model).name}",
            "choices": [{
                "index": 0,
                "turns": turns_text,
                "decoding_steps": new_tokens_list,
                "new_tokens": new_tokens_list,
                "wall_time": wall_time_list,
                "accept_lengths": [],
            }],
            "tstamp": time.time(),
        })
        if (qi + 1) % 20 == 0 or qi + 1 == len(questions):
            print(f"  [{qi+1}/{len(questions)}]", flush=True)

    elapsed_total = time.perf_counter() - t_start
    stats = collect_per_pos(llm, args.K)
    drafts = stats["drafts"]
    per_pos_counts = stats["per_pos"]
    per_pos_rates = [c / max(drafts, 1) for c in per_pos_counts]
    mean_accept = 1 + (stats["accepted"] / max(drafts, 1))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for a in answers:
            f.write(json.dumps(a) + "\n")
    with open(args.output + ".metrics.json", "w") as f:
        json.dump({
            "model": args.model, "draft": args.draft, "K": args.K,
            "num_questions": len(questions),
            "num_drafts": drafts,
            "num_accepted_tokens": stats["accepted"],
            "per_pos_counts": per_pos_counts,
            "per_pos_rate": per_pos_rates,
            "mean_accept_length": mean_accept,
            "elapsed_sec": elapsed_total,
        }, f, indent=2)

    print("\n=== vLLM EAGLE reference ===")
    print(f"model       : {args.model}")
    print(f"K           : {args.K}")
    print(f"questions   : {len(questions)}")
    print(f"n_drafts    : {drafts}")
    print(f"mean_accept : {1 + stats['accepted']/max(drafts,1):.3f}  (bonus included)")
    for i, r in enumerate(per_pos_rates):
        print(f"  pos{i}: {r*100:6.2f}%")
    print(f"elapsed     : {elapsed_total:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
