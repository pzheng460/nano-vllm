"""Analyze profile_merged.json.gz — split kernels into target/draft, compute
overlap between target compute, NCCL, and draft compute.

Usage:
    .venv/bin/python experiments/pangu_lsd/analyze_trace.py \\
      [path/to/profile_merged.json.gz]   # default: ./profile_merged.json.gz
"""
import gzip
import sys
import collections

import ijson


def merge_iv(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def total_iv(intervals):
    return sum(e - s for s, e in intervals)


def intersect_total(a, b):
    events = []
    for s, e in a:
        events += [(s, +1, "a"), (e, -1, "a")]
    for s, e in b:
        events += [(s, +1, "b"), (e, -1, "b")]
    events.sort()
    inA = inB = 0
    last = None
    overlap = 0
    for ts, delta, who in events:
        if inA > 0 and inB > 0 and last is not None:
            overlap += ts - last
        if who == "a":
            inA += delta
        else:
            inB += delta
        last = ts
    return overlap


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "profile_merged.json.gz"
    src = gzip.open(path, "rb") if path.endswith(".gz") else open(path, "rb")

    target_compute = []
    target_nccl = []
    draft_compute = []
    draft_nccl = []
    with src as f:
        for ev in ijson.items(f, "traceEvents.item"):
            if ev.get("ph") != "X" or ev.get("cat") != "kernel":
                continue
            ts = ev.get("ts", 0)
            dur = ev.get("dur", 0)
            nm = ev.get("name", "")
            is_target = ev.get("pid") < 10000
            iv = (ts, ts + dur)
            if "nccl" in nm.lower():
                (target_nccl if is_target else draft_nccl).append(iv)
            else:
                (target_compute if is_target else draft_compute).append(iv)

    tc = merge_iv(target_compute)
    dc = merge_iv(draft_compute)
    tn = merge_iv(target_nccl)
    dn = merge_iv(draft_nccl)

    print(f"target compute: {total_iv(tc) / 1000:.2f}ms")
    print(f"target NCCL   : {total_iv(tn) / 1000:.2f}ms")
    print(f"draft  compute: {total_iv(dc) / 1000:.2f}ms")
    print(f"draft  NCCL   : {total_iv(dn) / 1000:.2f}ms (mostly idle recv wait)")
    print()
    cd = intersect_total(tc, dc)
    cn = intersect_total(tc, tn)
    dn_tc = intersect_total(dc, tn)
    print(f"target_compute ∩ draft_compute = {cd/1000:6.2f}ms "
          f"({100*cd/max(total_iv(dc), 1):.1f}% of draft compute hidden)")
    print(f"target_compute ∩ target_NCCL   = {cn/1000:6.2f}ms "
          f"({100*cn/max(total_iv(tn), 1):.1f}% of target NCCL hidden)")
    print(f"draft_compute  ∩ target_NCCL   = {dn_tc/1000:6.2f}ms "
          f"(when draft works during target NCCL wait)")


if __name__ == "__main__":
    main()
