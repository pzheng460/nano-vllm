"""Capture nano-vllm EAGLE-3 per-step intermediates.

Runs nano-vllm with Qwen2.5 + EAGLE3 on a fixed prompt, hooks Eagle3DecoderLayer
and the model.fc + model.norm to collect per-step tensors. Saves to torch .pt.
"""
import argparse
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--prompt", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    sys.path.insert(0, "/mnt/data/peizhen/nano-vllm")
    import torch
    from nanovllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    llm = LLM(
        args.model, draft_model=args.draft, num_speculative_tokens=args.k,
        enforce_eager=True, max_model_len=2048,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False, add_generation_prompt=True,
    )

    caps = {
        "meta": {"framework": "nano-vllm", "prompt": args.prompt, "K": args.k},
        "fc": [],            # combine_hidden_states calls: (in, out)
        "layer": [],         # Eagle3DecoderLayer forward output: (hidden, residual, combined)
        "norm": [],          # model.norm calls (before lm_head)
    }

    draft = llm.model_runner.draft_model
    layer0 = draft.model.layers[0]
    fc_mod = draft.model.fc
    norm_mod = draft.model.norm

    def fc_hook(mod, inputs, output):
        caps["fc"].append({
            "in": inputs[0].detach().cpu().float(),
            "out": output.detach().cpu().float(),
        })

    def layer_hook(mod, inputs, output):
        # output = (hidden, residual) — both shape [seq, H]
        h, r = output
        caps_entry = {
            "hidden": h.detach().cpu().float(),
            "residual": r.detach().cpu().float(),
            "combined": (h + r).detach().cpu().float(),
        }
        caps["layer"].append(caps_entry)

    def norm_hook(mod, inputs, output):
        out = output if not isinstance(output, tuple) else output[0]
        caps["norm"].append(out.detach().cpu().float())

    h1 = fc_mod.register_forward_hook(fc_hook)
    h2 = layer0.register_forward_hook(layer_hook)
    h3 = norm_mod.register_forward_hook(norm_hook)

    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new)
    print(f"[nano-cap] running generate ...")
    t0 = time.perf_counter()
    outs = llm.generate([prompt], sp)
    wall = time.perf_counter() - t0
    caps["meta"]["output_text"] = outs[0]["text"][:500]

    h1.remove(); h2.remove(); h3.remove()

    text = outs[0]["text"]
    gen_ids = llm.tokenizer(text, add_special_tokens=False).input_ids
    print(f"[nano-cap] generated first 30 ids: {gen_ids[:30]}")
    print(f"[nano-cap] decoded: {text[:200]!r}")
    print(f"[nano-cap] fc calls={len(caps['fc'])} layer calls={len(caps['layer'])} norm calls={len(caps['norm'])}")
    print(f"[nano-cap] wall={wall:.2f}s")

    caps["meta"]["generated_ids"] = gen_ids
    caps["meta"]["decoded"] = text
    caps["meta"]["wall_s"] = wall

    torch.save(caps, args.output)
    print(f"[nano-cap] saved -> {args.output}")


if __name__ == "__main__":
    import argparse
    main()
