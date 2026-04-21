"""Diff nano-vllm and Spec-Bench EAGLE-3 captures to locate first divergent op.

Loads two .pt files (one per framework) and compares the draft layer output,
final norm output, and fc output per step. For each comparable step reports
max abs error, cosine similarity, argmax token id.
"""
import argparse
import torch
import torch.nn.functional as F


def pretty_diff(a, b, label, atol=1e-3):
    if a is None or b is None:
        return f"{label}: one is None"
    if a.shape != b.shape:
        return f"{label}: shape diff {tuple(a.shape)} vs {tuple(b.shape)}"
    diff = (a.float() - b.float())
    max_abs = diff.abs().max().item()
    mean_abs = diff.abs().mean().item()
    cos = F.cosine_similarity(a.float().flatten().unsqueeze(0),
                               b.float().flatten().unsqueeze(0)).item()
    flag = " ✅" if max_abs < atol else " ⚠️"
    return f"{label}: shape={tuple(a.shape)} max_abs={max_abs:.4e} mean_abs={mean_abs:.4e} cos={cos:.6f}{flag}"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nano", default="/tmp/specbench_align/capture_nano.pt")
    ap.add_argument("--spec", default="/tmp/specbench_align/capture_specbench.pt")
    args = ap.parse_args()

    nano = torch.load(args.nano, weights_only=False)
    spec = torch.load(args.spec, weights_only=False)

    print("=" * 70)
    print(f"NANO meta: {nano['meta']}")
    print(f"SPEC meta: {spec['meta']}")
    print("=" * 70)

    print(f"\nfc calls:   nano={len(nano['fc'])}    spec={len(spec['fc'])}")
    print(f"layer calls: nano={len(nano['layer'])}   spec={len(spec['midlayer'])}")
    print(f"norm calls:  nano={len(nano['norm'])}    spec={len(spec['norm_out'])}")

    # Try pairing:
    # Spec-Bench eagenerate: 1 fc (prefill) + chain = ...
    # For each round, draft runs for 4 steps: initial prefill + 3 chain
    # Let's print both call sequences for manual inspection
    def _in(x):  return x.get('in', x.get('input'))
    def _out(x): return x.get('out', x.get('output'))

    print("\n[nano fc call shapes]:")
    for i, x in enumerate(nano['fc']):
        print(f"  {i}: in={tuple(_in(x).shape)} out={tuple(_out(x).shape)}")
    print("\n[spec fc call shapes]:")
    for i, x in enumerate(spec['fc']):
        print(f"  {i}: in={tuple(_in(x).shape)} out={tuple(_out(x).shape)}")

    print("\n[nano layer outputs]:")
    for i, x in enumerate(nano['layer']):
        print(f"  {i}: hidden={tuple(x['hidden'].shape)}")
    print("\n[spec midlayer outputs]:")
    for i, x in enumerate(spec['midlayer']):
        print(f"  {i}: shape={tuple(x.shape)}")


if __name__ == "__main__":
    main()
