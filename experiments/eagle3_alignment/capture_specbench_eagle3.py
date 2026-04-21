"""Capture Spec-Bench EAGLE-3 per-step intermediates.

Run EaModel3 on a fixed prompt with chain K=3 config (depth=3, top_k=1,
total_token=4). Hooks the draft layer to collect per-call intermediate tensors.
Saves a torch .pt with the captures.
"""
import argparse
import sys
import time

sys.path.insert(0, "/mnt/data/peizhen/Spec-Bench")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new", type=int, default=40)
    args = ap.parse_args()

    import torch
    from model.eagle3.ea_model import EaModel

    print(f"[spec-cap] loading ...", flush=True)
    model = EaModel.from_pretrained(
        base_model_path=args.model,
        ea_model_path=args.draft,
        total_token=4, depth=3, top_k=1,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tok = model.get_tokenizer()
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )
    input_ids = tok([prompt], return_tensors="pt").input_ids.cuda()
    print(f"[spec-cap] prompt_ids last 5: {input_ids[0, -5:].tolist()}", flush=True)

    caps = {
        "meta": {"framework": "spec-bench", "prompt": args.prompt,
                 "prompt_len": int(input_ids.shape[1])},
        "fc": [],           # each call's fc input/output
        "midlayer": [],     # each call's midlayer output (tuple[0])
        "norm_out": [],     # each call's layer-norm output (RMSNorm before lm_head)
    }

    ea = model.ea_layer

    def fc_hook(mod, inputs, output):
        caps["fc"].append({
            "input": inputs[0].detach().cpu().float(),
            "output": output.detach().cpu().float(),
        })

    def mid_hook(mod, inputs, output):
        out = output[0] if isinstance(output, tuple) else output
        caps["midlayer"].append(out.detach().cpu().float())

    def norm_hook(mod, inputs, output):
        caps["norm_out"] = caps.get("norm_out", [])
        caps["norm_out"].append(output.detach().cpu().float())

    h1 = ea_fc_hook = ea.fc.register_forward_hook(fc_hook)
    h2 = ea.midlayer.register_forward_hook(mid_hook)
    # ea.norm is RMSNorm applied before lm_head
    h3 = ea.norm.register_forward_hook(norm_hook)

    print(f"[spec-cap] running eagenerate (max_new={args.max_new if hasattr(args,'max_new') else 40}) ...")
    t0 = time.perf_counter()
    out_ids, new_tok, n_steps, al = model.eagenerate(
        input_ids, temperature=0.0, max_new_tokens=args.max_new, log=True,
    )
    wall = time.perf_counter() - t0

    h1.remove(); h2.remove(); h3.remove()

    caps["meta"] = {
        **caps["meta"],
        "new_tokens": int(new_tok),
        "n_steps": n_steps,
        "accept_lens": list(al),
        "generated_ids": out_ids[0].tolist(),
        "wall_s": wall,
    }
    print(f"[spec-cap] new_tok={int(new_tok)} steps={n_steps} "
          f"accept_lens={list(al)[:10]} wall={wall:.2f}s", flush=True)
    print(f"[spec-cap] captured: {len(caps['fc'])} fc, {len(caps['midlayer'])} midlayer, "
          f"{len(caps.get('norm_out', []))} norm")

    import torch
    torch.save(caps, args.output)
    print(f"[spec-cap] saved to {args.output}")


if __name__ == "__main__":
    import argparse
    import time
    main()
