"""Diff nano vs vLLM MTP traces."""
import torch

NANO = "/tmp/pangu_nano_mtp_trace.pt"
VLLM = "/tmp/pangu_vllm_mtp_trace.pt"

nano = torch.load(NANO, map_location='cpu', weights_only=False)
vllm = torch.load(VLLM, map_location='cpu', weights_only=False)

print("=== NANO trace ===")
for k in sorted(nano.keys()):
    if not k.startswith('call') or not isinstance(nano[k], dict):
        continue
    rec = nano[k]
    pos = rec['positions'].tolist() if rec['positions'].numel() <= 12 else f"len={rec['positions'].numel()}"
    print(f"  {k}: pos={pos}")
    for kk in ('input_embeds', 'hidden_in', 'normed_out', 'prenorm_out'):
        if kk in rec:
            t = rec[kk]
            print(f"    {kk}: shape={tuple(t.shape)} mean|x|={t.float().abs().mean().item():.4f}")
print(f"  output_tokens: {nano.get('output_token_ids', '?')}")

print("\n=== VLLM trace ===")
for i, rec in enumerate(vllm):
    pos = rec['positions'].tolist() if rec['positions'].numel() <= 12 else f"len={rec['positions'].numel()}"
    print(f"  call{i}: pos={pos}" if False else f"  call{i}: pos={pos}")
    for kk in ('inputs_embeds', 'hidden_in', 'out'):
        if kk in rec and rec[kk] is not None:
            t = rec[kk]
            print(f"    {kk}: shape={tuple(t.shape)} mean|x|={t.float().abs().mean().item():.4f}")
    if 'input_ids' in rec and rec['input_ids'] is not None:
        print(f"    input_ids: {rec['input_ids'].tolist()[:10]}")

# Compare first decode call: nano call2 (pos [6]), vllm call1 (pos [6,7])
print("\n=== Hidden_in comparison at shared position 6 ===")
nano_call2 = nano['call2']
vllm_call1 = vllm[1]
n_pos = nano_call2['positions'].tolist()
v_pos = vllm_call1['positions'].tolist()
print(f"nano positions: {n_pos}")
print(f"vllm positions: {v_pos}")
# Find shared position
shared = [p for p in n_pos if p in v_pos]
if shared:
    p = shared[0]
    ni = n_pos.index(p)
    vi = v_pos.index(p)
    n_h = nano_call2['hidden_in'][ni]
    v_h = vllm_call1['hidden_in'][vi]
    print(f"position {p}: nano|hidden_in| mean={n_h.abs().mean().item():.4f}  vllm|hidden_in| mean={v_h.abs().mean().item():.4f}")
    print(f"  nano[0:5] = {n_h[:5].tolist()}")
    print(f"  vllm[0:5] = {v_h[:5].tolist()}")
    diff = (n_h - v_h).abs()
    print(f"  absdiff max={diff.max().item():.4f} mean={diff.mean().item():.4f}")

# Also compare input_embeds
print("\n=== input_embeds comparison at position 6 ===")
n_ie = nano['call2']['input_embeds'][n_pos.index(6)] if 6 in n_pos else None
v_ie = vllm_call1['inputs_embeds'][v_pos.index(6)] if 6 in v_pos else None
if n_ie is not None and v_ie is not None:
    diff = (n_ie - v_ie).abs()
    print(f"  nano|ie| mean={n_ie.abs().mean().item():.4f}  vllm|ie| mean={v_ie.abs().mean().item():.4f}")
    print(f"  absdiff max={diff.max().item():.4f} mean={diff.mean().item():.4f}")
    print(f"  nano[0:5] = {n_ie[:5].tolist()}")
    print(f"  vllm[0:5] = {v_ie[:5].tolist()}")
