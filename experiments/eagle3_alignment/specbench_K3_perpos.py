"""Spec-Bench chain K=3 reference with per-question accept_lens saved."""
import argparse
import json
import sys
import time

sys.path.insert(0, "/mnt/data/peizhen/Spec-Bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    import torch
    from model.eagle3.ea_model import EaModel

    print(f"[K3] loading {args.draft} ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=args.model,
        ea_model_path=args.draft,
        total_token=4, depth=3, top_k=1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tok = model.get_tokenizer()

    with open(args.input) as f:
        qs = [json.loads(l) for l in f if l.strip()][: args.limit]
    print(f"[K3] {len(qs)} prompts", flush=True)

    all_lens = []
    t0 = time.perf_counter()
    for i, q in enumerate(qs := qs if False else qs):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": q["turns"][0]}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok([prompt], return_tensors="pt").input_ids.cuda()
        _, new_tok, step, al = model.eagenerate(
            ids, temperature=0.0, max_new_tokens=args.max_new, log=True,
        )
        all_lens.extend(int(x) for x in al)
        if (i + 1) % 20 == 0 or i + 1 == len(qs):
            mean = sum(all_lens) / max(len(all_lens), 1)
            print(f"  [{i+1}/{len(qs)}] running mean={mean:.3f}", flush=True)

    # Per-pos accept: pos_i_accept = #steps where accept_len > i+1
    # For K=3 chain, accept_len in [1,2,3,4]; bonus always counted.
    K = 3
    total_steps = len(all_lens)
    per_pos = []
    for i in range(K):
        count = sum(1 for l in all_lens if l > i + 1)
        per_pos.append(count)
    mean_accept = sum(all_lens) / max(total_steps, 1)

    from collections import Counter
    dist = Counter(all_lens)
    print(f"\n[K3] summary (Spec-Bench chain K=3 = depth=2 top_k=1 total=3):")
    print(f"  steps: {total_steps}")
    print(f"  mean_accept: {mean_accept:.4f}")
    print(f"  accept_len distribution:")
    for k in sorted(dist.keys()):
        print(f"    len={k}: {dist[k]} ({100*dist[k]/total_steps:.1f}%)")
    print(f"  per-pos accept rate (# steps where draft_i accepted):")
    for i, c in enumerate(per_pos):
        print(f"    pos{i}: {100*c/max(total_steps,1):.2f}% ({c}/{total_steps})")

    with open(args.output, "w") as f:
        json.dump({
            "total_steps": total_steps,
            "mean_accept_length": mean_accept,
            "per_pos_counts": per_pos,
            "per_pos_rate": [c/max(total_steps,1) for c in per_pos],
            "len_distribution": dict(sorted(dist.items())),
            "all_accept_lens": [int(x) for x in all_lens],
        }, f, indent=2)


if __name__ == "__main__":
    main()
