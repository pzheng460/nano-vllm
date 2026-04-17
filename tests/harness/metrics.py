"""Metric utilities aligned with Spec-Bench's evaluation/speed.py + equal.py.

Per-step `accept_lengths` convention — matches Spec-Bench exactly:

    accept_lengths[i] = tokens committed in decode step i
                      = draft_tokens_accepted + 1 bonus token
    range: 1..K+1, where K = num_speculative_tokens
    baseline: always 1 (no drafts)

Upstream `speed.py` computes:

    speeds = [sum(new_tokens)/sum(wall_time) for each question]
    tokens_per_second = np.mean(speeds)        # MACRO-average over questions
    #Mean accepted tokens = np.mean(flat accept_lengths across all questions)

We expose BOTH a Spec-Bench-exact `tokens_per_second_macro` and a pooled
`tokens_per_second` (= sum tokens / sum time across all questions). The macro
version is what you should compare against Spec-Bench leaderboard numbers;
the pooled version is less sensitive to per-prompt outliers.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


# --- Aggregation -------------------------------------------------------------

def _aggregate(items: Iterable[dict]) -> dict:
    total_new = 0
    total_wall = 0.0
    total_steps = 0
    accepts: list[int] = []
    per_question_tps: list[float] = []
    n = 0
    for a in items:
        choice = a["choices"][0]
        q_tokens = sum(choice["new_tokens"])
        q_wall = sum(choice["wall_time"])
        total_new += q_tokens
        total_wall += q_wall
        total_steps += sum(choice["decoding_steps"])
        accepts.extend(choice["accept_lengths"])
        if q_wall > 0:
            per_question_tps.append(q_tokens / q_wall)
        n += 1
    tps_pool = total_new / total_wall if total_wall > 0 else 0.0
    tps_macro = (
        sum(per_question_tps) / len(per_question_tps)
        if per_question_tps else 0.0
    )
    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    return {
        "tokens_per_second": tps_pool,              # pooled (sum/sum)
        "tokens_per_second_macro": tps_macro,       # Spec-Bench canonical
        "mean_accept_length": mean_accept,
        "total_new_tokens": total_new,
        "total_wall_time": total_wall,
        "total_decoding_steps": total_steps,
        "num_answers": n,
        "num_accept_steps": len(accepts),
    }


def compute_speed(answers: list[dict]) -> dict:
    """Aggregate answers overall and per-category."""
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for a in answers:
        by_cat[a["category"]].append(a)
    out = {"overall": _aggregate(answers)}
    out["by_category"] = {cat: _aggregate(lst) for cat, lst in by_cat.items()}
    return out


def speedup(spec: list[dict], baseline: list[dict]) -> dict:
    """Compute speedup = tps(spec) / tps(baseline) overall + per category.

    Records are matched on `question_id`; any question missing on either side
    is dropped silently.
    """
    base_by_id = {a["question_id"]: a for a in baseline}
    matched_spec: list[dict] = []
    matched_base: list[dict] = []
    for s in spec:
        b = base_by_id.get(s["question_id"])
        if b is not None:
            matched_spec.append(s)
            matched_base.append(b)

    s_stats = compute_speed(matched_spec)
    b_stats = compute_speed(matched_base)

    def _row(s: dict, b: dict) -> dict:
        s_tps = s["tokens_per_second_macro"]
        b_tps = b["tokens_per_second_macro"]
        s_pool = s["tokens_per_second"]
        b_pool = b["tokens_per_second"]
        return {
            # Spec-Bench-aligned (macro-average over questions)
            "spec_tps": s_tps,
            "base_tps": b_tps,
            "speedup": (s_tps / b_tps) if b_tps > 0 else 0.0,
            # Pooled (sum tokens / sum wall) — length-weighted
            "spec_tps_pooled": s_pool,
            "base_tps_pooled": b_pool,
            "speedup_pooled": (s_pool / b_pool) if b_pool > 0 else 0.0,
            # Common
            "mean_accept_length": s["mean_accept_length"],
            "num_answers": s["num_answers"],
        }

    out = {"overall": _row(s_stats["overall"], b_stats["overall"])}
    cats = sorted(set(s_stats["by_category"]) | set(b_stats["by_category"]))
    for cat in cats:
        s = s_stats["by_category"].get(cat)
        b = b_stats["by_category"].get(cat)
        if s and b:
            out[cat] = _row(s, b)
    return out


def specbench_exact_speedup(
    spec: list[dict],
    baseline: list[dict],
    tokenizer,
) -> dict:
    """Compute speedup using Spec-Bench's exact formula, including upstream's
    quirk of re-tokenizing the baseline's decoded turns rather than using the
    recorded `new_tokens`.

    Upstream speed.py (baseline path, lines 49-57):
        for datapoint in data_base:
            answer = datapoint["choices"][0]["turns"]
            tokens = sum(len(tokenizer(t).input_ids) - 1 for t in answer)
            times  = sum(datapoint["choices"][0]["wall_time"])
            speeds0.append(tokens/times)
        tps_baseline = mean(speeds0)

    This makes nano-vllm baseline numbers directly comparable to published
    Spec-Bench leaderboards.
    """
    speeds_spec: list[float] = []
    accepts_spec: list[int] = []
    for d in spec:
        ch = d["choices"][0]
        toks = sum(ch["new_tokens"])
        t = sum(ch["wall_time"])
        if t > 0:
            speeds_spec.append(toks / t)
        accepts_spec.extend(ch["accept_lengths"])

    speeds_base: list[float] = []
    base_by_id = {a["question_id"]: a for a in baseline}
    for qid in [d["question_id"] for d in spec]:
        b = base_by_id.get(qid)
        if b is None:
            continue
        ch = b["choices"][0]
        toks = sum(
            max(len(tokenizer(turn).input_ids) - 1, 0) for turn in ch["turns"]
        )
        t = sum(ch["wall_time"])
        if t > 0:
            speeds_base.append(toks / t)

    tps_spec = sum(speeds_spec) / len(speeds_spec) if speeds_spec else 0.0
    tps_base = sum(speeds_base) / len(speeds_base) if speeds_base else 0.0
    return {
        "tokens_per_second": tps_spec,
        "tokens_per_second_baseline": tps_base,
        "mean_accepted": (sum(accepts_spec) / len(accepts_spec)) if accepts_spec else 0.0,
        "speedup_ratio": (tps_spec / tps_base) if tps_base > 0 else 0.0,
        "num_answers": len(speeds_spec),
    }


def speedup_table(spec: list[dict], baseline: list[dict]) -> str:
    """Render a speedup table. `spec_tps`/`base_tps`/`speedup` are the
    Spec-Bench-aligned macro-average metric (mean of per-question tok/s);
    `*_pool` columns are the pooled variant (sum tokens / sum wall)."""
    rows = speedup(spec, baseline)
    header = (
        f"{'category':<14}{'spec_tps':>10}{'base_tps':>10}{'speedup':>9}"
        f"{'spec_pool':>11}{'base_pool':>11}{'sp_pool':>9}"
        f"{'mean_accept':>12}{'n':>6}"
    )
    lines = [header, "-" * len(header)]
    order = ["overall"] + sorted(k for k in rows if k != "overall")
    for cat in order:
        r = rows[cat]
        lines.append(
            f"{cat:<14}{r['spec_tps']:>10.2f}{r['base_tps']:>10.2f}"
            f"{r['speedup']:>9.3f}"
            f"{r['spec_tps_pooled']:>11.2f}{r['base_tps_pooled']:>11.2f}"
            f"{r['speedup_pooled']:>9.3f}"
            f"{r['mean_accept_length']:>12.3f}"
            f"{r['num_answers']:>6}"
        )
    return "\n".join(lines)


# --- Greedy equivalence ------------------------------------------------------

def equivalence(a: list[dict], b: list[dict], strict: bool = True) -> dict:
    """Check per-turn decoded text matches byte-for-byte between two answer
    sets. Returns match statistics and (optionally) a list of mismatches.

    `strict=False` tolerates one side being a prefix of the other (useful when
    max_new_tokens differs).
    """
    by_id_a = {x["question_id"]: x for x in a}
    by_id_b = {x["question_id"]: x for x in b}
    common = sorted(set(by_id_a) & set(by_id_b), key=lambda k: (isinstance(k, str), k))

    total = 0
    matched = 0
    divergences: list[dict] = []
    for qid in common:
        turns_a = by_id_a[qid]["choices"][0]["turns"]
        turns_b = by_id_b[qid]["choices"][0]["turns"]
        for i in range(min(len(turns_a), len(turns_b))):
            total += 1
            ta, tb = turns_a[i], turns_b[i]
            if ta == tb:
                matched += 1
                continue
            if not strict and (ta.startswith(tb) or tb.startswith(ta)):
                matched += 1
                continue
            divergences.append(_first_diff(qid, i, ta, tb))
    return {
        "num_turns": total,
        "matched_turns": matched,
        "match_ratio": matched / total if total else 1.0,
        "divergences": divergences,
    }


def _first_diff(qid, turn_idx: int, a: str, b: str) -> dict:
    n = min(len(a), len(b))
    pos = n
    for i in range(n):
        if a[i] != b[i]:
            pos = i
            break
    lo = max(0, pos - 20)
    hi = pos + 20
    return {
        "question_id": qid,
        "turn": turn_idx,
        "pos": pos,
        "a_snippet": a[lo:hi],
        "b_snippet": b[lo:hi],
        "len_a": len(a),
        "len_b": len(b),
    }


