"""
EAGLE-3 (Qwen2.5-7B-Instruct) acceptance rate calibration: vLLM vs nano-vllm.

Usage:
  # Step 1: Run vLLM baseline
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python tests/calibrate_eagle3.py vllm

  # Step 2: Run nano-vllm
  CUDA_VISIBLE_DEVICES=5 .venv/bin/python tests/calibrate_eagle3.py nano

  # Step 3: Compare results
  .venv/bin/python tests/calibrate_eagle3.py compare

  # Or run both sequentially on one GPU:
  CUDA_VISIBLE_DEVICES=4 .venv/bin/python tests/calibrate_eagle3.py both
"""
import sys
import os
import json
import time

TARGET = '/mnt/data/peizhen/Qwen2.5-7B-Instruct'
EAGLE3 = '/mnt/data/peizhen/EAGLE3-Qwen2.5-7B-Instruct'
K = 3
MAX_TOKENS = 256
RESULT_DIR = '/tmp/eagle3_calibration'

PROMPTS = [
    "Explain the mechanism of nuclear fission at the molecular level.",
    "A train travels at 60 mph for 3 hours, then 80 mph for 2 hours. What is the total distance?",
    "What is the difference between mitosis and meiosis? Explain with examples.",
    "Janet sells 16 pies. Each pie costs 8 dollars. How much money does she make?",
    "Describe the process of photosynthesis including light and dark reactions.",
    "A store has a 25% off sale. An item costs $80. What is the final price with 10% tax?",
    "Explain the second law of thermodynamics with a real-world example.",
    "Tom has 3 times as many books as Jerry. Together they have 48. How many each?",
    "What are the key differences between TCP and UDP in computer networking?",
    "A rectangular pool is 12m by 8m and 2m deep. How many liters of water to fill it?",
    "Explain how CRISPR-Cas9 works and its applications in medicine.",
    "Two trains leave a station at the same time traveling opposite directions at 60 and 80 km/h. When are they 350 km apart?",
    "Describe the structure and function of DNA double helix.",
    "If you invest $1000 at 5% compound interest annually, what is the amount after 10 years?",
    "Explain quantum entanglement in simple terms.",
    "A store has 120 apples. They sell 45 in the morning and 32 in the afternoon. How many are left?",
    "What is the difference between supervised and unsupervised machine learning?",
    "Calculate the derivative of f(x) = 3x^4 - 2x^3 + 5x - 7.",
    "Explain what causes ocean tides and the role of the Moon.",
    "How many 3-digit palindrome numbers exist? List the pattern.",
]


def run_vllm():
    """Run vLLM EAGLE-3 and collect Prometheus metrics."""
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"vLLM EAGLE-3 K={K}, {len(PROMPTS)} prompts, max_tokens={MAX_TOKENS}")
    print(f"{'='*60}")

    llm = LLM(
        model=TARGET,
        speculative_config={
            'method': 'eagle3',
            'model': EAGLE3,
            'num_speculative_tokens': K,
        },
        enforce_eager=True,
        max_model_len=4096,
        gpu_memory_utilization=0.85,
    )

    from vllm import SamplingParams as VSP
    sp = VSP(temperature=0.0, max_tokens=MAX_TOKENS)

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, sp)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    token_ids_list = [list(o.outputs[0].token_ids) for o in outputs]
    texts = [o.outputs[0].text for o in outputs]

    # Collect Prometheus metrics
    metrics = {}
    try:
        from prometheus_client import REGISTRY
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if 'spec_decode' in sample.name and sample.value != 0:
                    key = sample.name
                    if sample.labels:
                        key += str(sample.labels)
                    metrics[key] = sample.value
    except Exception as e:
        print(f"Warning: could not collect Prometheus metrics: {e}")

    # Parse per-position acceptance
    per_pos_accepted = {}
    for key, val in metrics.items():
        if 'per_position' in key or 'per_pos' in key:
            per_pos_accepted[key] = val

    num_drafts = metrics.get('vllm:spec_decode_num_drafts_total', 0)
    num_draft_tokens = metrics.get('vllm:spec_decode_num_draft_tokens_total', 0)
    num_accepted = metrics.get('vllm:spec_decode_num_accepted_tokens_total', 0)

    result = {
        'backend': 'vllm',
        'num_prompts': len(PROMPTS),
        'num_speculative_tokens': K,
        'max_tokens': MAX_TOKENS,
        'total_output_tokens': sum(len(ids) for ids in token_ids_list),
        'elapsed_seconds': 0,
        'tokens_per_second': 0,
        'num_drafts': num_drafts,
        'num_draft_tokens': num_drafts * K,
        'num_accepted_tokens': num_accepted,
        'total_acceptance_rate': num_accepted / (num_drafts * K) if num_drafts > 0 else 0,
        'mean_acceptance_length': 1 + num_accepted / num_drafts if num_drafts > 0 else 0,
        'raw_prometheus_metrics': metrics,
        'per_position_raw': per_pos_metrics_to_list(metrics, K),
        'token_ids': token_ids_list,
        'texts': texts[:5],  # save first 5 texts for reference
    }
    result['elapsed_seconds'] = elapsed
    result['tokens_per_second'] = result['total_output_tokens'] / elapsed

    print(f"\nvLLM Results:")
    print(f"  Output: {result['total_output_tokens']} tokens in {elapsed:.2f}s ({result['tokens_per_second']:.1f} tok/s)")
    print(f"  Drafts: {result['num_drafts']}")
    print(f"  Accepted: {result['num_accepted_tokens']}/{result['num_draft_tokens']}")
    if result['num_drafts'] > 0:
        print(f"  Acceptance rate: {result['total_acceptance_rate']:.1%}")
        print(f"  Mean acceptance length: {result['mean_acceptance_length']:.2f}")
    if result['per_position_raw']:
        print(f"  Per-position: {result['per_position_raw']}")
    print(f"  All Prometheus metrics: {json.dumps(metrics, indent=2)}")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return result


def per_pos_metrics_to_list(metrics, k):
    """Extract per-position acceptance from vLLM Prometheus metrics."""
    per_pos = []
    for pos in range(k):
        # Try different naming conventions
        found = False
        for key, val in metrics.items():
            if f"'position_in_proposal': '{pos}'" in key and 'accepted' in key:
                per_pos.append(val)
                found = True
                break
        if not found:
            per_pos.append(None)
    return per_pos


def run_nano():
    """Run nano-vllm EAGLE-3 and collect acceptance stats."""
    from nanovllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"nano-vllm EAGLE-3 K={K}, {len(PROMPTS)} prompts, max_tokens={MAX_TOKENS}")
    print(f"{'='*60}")

    llm = LLM(
        TARGET,
        draft_model=EAGLE3,
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=4096,
        num_speculative_tokens=K,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)

    t0 = time.perf_counter()
    outputs = llm.generate(PROMPTS, sp, use_tqdm=True)
    elapsed = time.perf_counter() - t0

    total_tokens = sum(len(o['token_ids']) for o in outputs)
    token_ids_list = [o['token_ids'] for o in outputs]
    texts = [o['text'] for o in outputs]

    result = {
        'backend': 'nano-vllm',
        'num_prompts': len(PROMPTS),
        'num_speculative_tokens': K,
        'max_tokens': MAX_TOKENS,
        'total_output_tokens': total_tokens,
        'elapsed_seconds': elapsed,
        'tokens_per_second': total_tokens / elapsed,
        'token_ids': token_ids_list,
        'texts': [t[:200] for t in texts],
    }

    print(f"\nnano-vllm Results:")
    print(f"  Output: {total_tokens} tokens in {elapsed:.2f}s ({total_tokens/elapsed:.1f} tok/s)")

    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    return result


def compare():
    """Load and compare vLLM vs nano-vllm results."""
    vllm_path = os.path.join(RESULT_DIR, 'vllm_result.json')
    nano_path = os.path.join(RESULT_DIR, 'nano_result.json')

    for p, name in [(vllm_path, 'vllm'), (nano_path, 'nano')]:
        if not os.path.exists(p):
            print(f"Missing {p}. Run: python tests/calibrate_eagle3.py {name}")
            return

    with open(vllm_path) as f:
        vr = json.load(f)
    with open(nano_path) as f:
        nr = json.load(f)

    print(f"\n{'='*70}")
    print("EAGLE-3 Calibration Report")
    print(f"{'='*70}")
    print(f"Model:       Qwen2.5-7B-Instruct")
    print(f"Draft:       EAGLE3-Qwen2.5-7B-Instruct")
    print(f"K:           {K}")
    print(f"Prompts:     {len(PROMPTS)}")
    print(f"Max tokens:  {MAX_TOKENS}")
    print(f"Temperature: 0.0 (greedy)")
    print()

    # --- Throughput ---
    print("Throughput:")
    print(f"  vLLM:      {vr.get('tokens_per_second', 0):.1f} tok/s ({vr.get('total_output_tokens', 0)} tokens)")
    print(f"  nano-vllm: {nr.get('tokens_per_second', 0):.1f} tok/s ({nr.get('total_output_tokens', 0)} tokens)")
    print()

    # --- Acceptance rate (vLLM only, nano-vllm prints to stdout) ---
    if vr.get('num_drafts', 0) > 0:
        print("vLLM Acceptance Stats:")
        print(f"  Total drafts:    {vr['num_drafts']}")
        print(f"  Draft tokens:    {vr['num_draft_tokens']}")
        print(f"  Accepted tokens: {vr['num_accepted_tokens']}")
        print(f"  Acceptance rate: {vr['total_acceptance_rate']:.1%}")
        print(f"  Mean accept len: {vr['mean_acceptance_length']:.2f}")
        print()

    # --- Phase 4: Greedy output comparison ---
    v_ids = vr.get('token_ids', [])
    n_ids = nr.get('token_ids', [])

    if v_ids and n_ids:
        print("Phase 4: Greedy Output Comparison (temperature=0)")
        print("-" * 60)
        n_match = 0
        n_total = min(len(v_ids), len(n_ids))
        mismatches = []
        for i in range(n_total):
            v, n = v_ids[i], n_ids[i]
            if v == n:
                n_match += 1
                print(f"  Prompt {i:2d}: MATCH ({len(v)} tokens)")
            else:
                min_len = min(len(v), len(n))
                first_diff = min_len
                for j in range(min_len):
                    if v[j] != n[j]:
                        first_diff = j
                        break
                n_same = first_diff
                print(f"  Prompt {i:2d}: MISMATCH at pos {first_diff} "
                      f"(vllm={v[first_diff] if first_diff < len(v) else 'EOS'}, "
                      f"nano={n[first_diff] if first_diff < len(n) else 'EOS'}) "
                      f"| same={n_same}/{min(len(v), len(n))}, "
                      f"vlen={len(v)}, nlen={len(n)}")
                mismatches.append((i, first_diff, len(v), len(n)))

        print(f"\n  Exact match: {n_match}/{n_total}")
        match_pct = n_match / n_total * 100 if n_total > 0 else 0
        print(f"  Match rate:  {match_pct:.1f}%")

        if mismatches:
            avg_prefix = sum(m[1] for m in mismatches) / len(mismatches)
            print(f"  Avg matching prefix: {avg_prefix:.1f} tokens")

        print()
        if match_pct == 100:
            print("  VERDICT: PASS — all outputs identical")
        elif match_pct >= 80:
            print("  VERDICT: PASS (soft) — most outputs match, minor precision diff")
        else:
            print("  VERDICT: FAIL — significant divergence detected")
            print("  → Run Phase 3 (inference flow verification) to isolate root cause")

    print(f"\n{'='*70}")


def per_pos_metrics_to_list(metrics, k):
    """Try to extract per-position acceptance from prometheus metrics."""
    result = []
    for pos in range(k):
        for key, val in metrics.items():
            if f"'position_in_proposal': '{pos}'" in key:
                result.append(val)
                break
        else:
            result.append(None)
    return result


def save_result(result, name):
    os.makedirs(RESULT_DIR, exist_ok=True)
    path = os.path.join(RESULT_DIR, f'{name}_result.json')
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {path}")


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'both'

    if mode == 'vllm':
        result = run_vllm()
        save_result(result, 'vllm')

    elif mode == 'nano':
        result = run_nano()
        save_result(result, 'nano')

    elif mode == 'both':
        vr = run_vllm()
        save_result(vr, 'vllm')
        nr = run_nano()
        save_result(nr, 'nano')
        compare()

    elif mode == 'compare':
        compare()

    else:
        print(f"Unknown mode: {mode}")
        print("Usage: python tests/calibrate_eagle3.py [vllm|nano|both|compare]")
