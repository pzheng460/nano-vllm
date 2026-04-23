"""Dump nano-vllm PanGu MTP intermediate tensors for first few forward calls."""
import os

MODEL = "/mnt/data/weights/openPangu-R-72B-2512"
OUT = "/tmp/pangu_nano_mtp_trace.pt"
PROMPT = "The capital of France is"
MAX_TOK = 3


def main():
    import torch
    from nanovllm.models import pangu

    # Install trace hook on PanguMTPLayer.forward (before LLM instantiates it)
    trace = {}
    call_idx = [0]
    _orig = pangu.PanguMTPLayer.forward

    def traced(self, input_embeds, hidden_states, positions):
        out = _orig(self, input_embeds, hidden_states, positions)
        normed = out[0] if isinstance(out, tuple) else out
        prenorm = out[1] if isinstance(out, tuple) and len(out) > 1 else None
        idx = call_idx[0]
        if idx < 4:
            rec = {
                "input_embeds": input_embeds.detach().cpu().float().clone(),
                "hidden_in": hidden_states.detach().cpu().float().clone(),
                "positions": positions.detach().cpu().clone(),
                "normed_out": normed.detach().cpu().float().clone(),
            }
            if prenorm is not None:
                rec["prenorm_out"] = prenorm.detach().cpu().float().clone()
            trace[f"call{idx}"] = rec
            print(f"[trace] call{idx}: ie_shape={tuple(rec['input_embeds'].shape)} "
                  f"hin_shape={tuple(rec['hidden_in'].shape)} "
                  f"pos={positions.tolist() if positions.numel() <= 20 else '[:20]+...'}",
                  flush=True)
        call_idx[0] += 1
        return out

    # Bind variable inside traced
    def _install():
        pangu.PanguMTPLayer.forward = traced

    _install()

    from nanovllm import LLM, SamplingParams
    llm = LLM(
        MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=512,
        enforce_eager=True,
        num_speculative_tokens=1,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOK)
    out = llm.generate([PROMPT], sp)

    trace["output_token_ids"] = list(out[0]["token_ids"])
    trace["call_count"] = call_idx[0]
    torch.save(trace, OUT)
    print(f"Saved {len(trace)-2} MTP traces to {OUT}")
    print(f"output tokens: {trace['output_token_ids']}")


if __name__ == "__main__":
    main()
