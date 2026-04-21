"""Compute per-position accept stats from a Spec-Bench-style model_answer jsonl.

Our runner stores `choices[0].accept_lengths` as a flat list of per-decode-step
accepted token counts (1..K+1; 1 = no drafts accepted, only bonus). We derive
the vLLM-comparable per-position accept rate:

    pos_i_accept_rate = (# steps with accept_len > i + 1) / total_steps

Usage:
    python experiments/specbench_accept_stats.py <model_answer.jsonl> [K=3]
"""
from __future__ import annotations

import json
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: specbench_accept_stats.py <model_answer.jsonl> [K=3]")
        return 2
    path = sys.argv[1]
    K = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    total_steps = 0
    total_accept = 0
    per_pos = [0] * K
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            for al in d["choices"][0].get("accept_lengths") or []:
                total_steps += 1
                ac = max(int(al) - 1, 0)
                total_accept += ac
                for i in range(K):
                    if i < ac:
                        per_pos[i] += 1

    import os
    print(f"file         : {os.path.basename(sys.argv[1])}")
    print(f"K            : {K}")
    print(f"decode steps : {total_steps}")
    print(f"accepted tok : {total_accept}")
    print(f"mean accept  : {1 + total_accept / max(total_steps, 1):.4f}")
    print("per-pos rate :")
    for i, c in enumerate(per_pos):
        r = c / max(total_steps, 1)
        print(f"  pos{i}: {r * 100:6.2f}%  ({c}/{total_steps})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
