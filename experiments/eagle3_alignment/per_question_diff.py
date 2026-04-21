"""Compare per-question mean_accept between nano-vllm and Spec-Bench.

Inputs:
- nano-vllm: model_answer jsonl (with flat accept_lengths per choice)
- Spec-Bench: per-question json from specbench_per_q.py

Outputs:
- Worst / best N prompts (ranked by nano - spec gap)
- Per-category gap breakdown
- Quarter-by-quarter mean_accept for the N worst prompts (checks whether
  the gap is stationary or accumulates over time)
"""
import argparse
import json
import statistics
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano-jsonl", required=True)
    ap.add_argument("--spec-json", required=True)
    ap.add_argument("--worst-n", type=int, default=10)
    ap.add_argument("--best-n", type=int, default=5)
    ap.add_argument("--quarters-for", type=int, default=5,
                    help="dump per-quarter mean_accept for the N worst prompts")
    args = ap.parse_args()

    sb = {d["qid"]: d for d in json.load(open(args.spec_json))}
    nano = {}
    with open(args.nano_jsonl) as f:
        for line in f:
            d = json.loads(line)
            qid = d["question_id"]
            acc = d["choices"][0].get("accept_lengths", [])
            nano[qid] = {
                "accept_lens": acc,
                "mean": sum(acc) / max(len(acc), 1),
                "steps": len(acc),
                "cat": d.get("category", ""),
            }

    rows = []
    for qid, n in nano.items():
        if qid not in sb:
            continue
        s = sb[qid]
        sm = sum(s["accept_lens"]) / max(s["steps"], 1)
        rows.append((qid, n["cat"], n["mean"], sm, n["mean"] - sm,
                     n["steps"], s["steps"]))
    rows.sort(key=lambda r: r[4])

    hdr_fmt = "{:>5s} {:<15s} {:>8s} {:>8s} {:>7s} {:>7s} {:>7s}"
    row_fmt = "{:>5d} {:<15s} {:>8.3f} {:>8.3f} {:>+7.3f} {:>7d} {:>7d}"
    print(hdr_fmt.format("qid", "cat", "nano", "spec", "gap",
                         "n_step", "s_step"))
    print("-" * 66)
    print(f"-- {args.worst_n} worst (nano << spec) --")
    for r in rows[: args.worst_n]:
        print(row_fmt.format(*r))
    print(f"-- {args.best_n} best --")
    for r in rows[-args.best_n:]:
        print(row_fmt.format(*r))

    mean_gap = sum(r[4] for r in rows) / len(rows)
    print(f"\nOverall mean gap (nano - spec)    : {mean_gap:+.3f}")
    print(f"Num with gap < -0.3               : "
          f"{sum(1 for r in rows if r[4] < -0.3):>3d}")
    print(f"Num within +-0.3                  : "
          f"{sum(1 for r in rows if abs(r[4]) <= 0.3):>3d}")
    print(f"Num with gap > +0.3               : "
          f"{sum(1 for r in rows if r[4] > 0.3):>3d}")

    by_cat = {}
    for r in rows:
        by_cat.setdefault(r[1], []).append(r[4])
    print("\nper-category mean gap:")
    for cat, lst in sorted(by_cat.items(), key=lambda kv: statistics.mean(kv[1])):
        print(f"  {cat:<15s}: n={len(lst):>3d}  mean_gap={statistics.mean(lst):+.3f}")

    if args.quarters_for > 0:
        print("\n== quarter-by-quarter breakdown of worst prompts ==")
        for r in rows[: args.quarters_for]:
            qid = r[0]
            # refetch accept_lens from nano file
            nano_al = None
            with open(args.nano_jsonl) as f:
                for line in f:
                    d = json.loads(line)
                    if d["question_id"] == qid:
                        nano_al = d["choices"][0].get("accept_lengths", [])
                        break
            spec_al = sb[qid]["accept_lens"]
            print(f"\nqid={qid} cat={r[1]} gap={r[4]:+.3f}")
            print(f"  nano (n={len(nano_al)}):")
            for i in range(4):
                chunk = nano_al[i * len(nano_al) // 4:(i + 1) * len(nano_al) // 4]
                m = sum(chunk) / max(len(chunk), 1)
                d = Counter(chunk)
                print(f"    Q{i+1}: mean={m:.3f}  dist={dict(sorted(d.items()))}")
            print(f"  spec (n={len(sb[qid]['accept_lens'])}):")
            for i in range(4):
                chunk = sb[qid]["accept_lens"][i * len(spec_al) // 4:(i + 1) * len(spec_al) // 4]
                m = sum(chunk) / max(len(chunk), 1)
                d = Counter(chunk)
                print(f"    Q{i+1}: mean={m:.3f}  dist={dict(sorted(d.items()))}")


if __name__ == "__main__":
    main()
