"""Dump vLLM PanGu MTP intermediate tensors for first few calls.

Uses collective_rpc to install hook + dump from workers.
"""
import os


def _install_hook(self):
    """Worker-side: monkey-patch MTP forward to record first 4 calls.
    `self` is the worker object (vLLM passes it in collective_rpc)."""
    import torch
    # The MTP layer forward is defined on DeepSeekMultiTokenPredictorLayer;
    # OpenPanguMultiTokenPredictorLayer inherits from it.
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMultiTokenPredictorLayer

    if getattr(DeepSeekMultiTokenPredictorLayer, '_nano_traced', False):
        return "already_installed"

    _orig = DeepSeekMultiTokenPredictorLayer.forward
    DeepSeekMultiTokenPredictorLayer._trace_store = []
    DeepSeekMultiTokenPredictorLayer._trace_call_idx = 0

    def traced(self, input_ids, positions, previous_hidden_states, inputs_embeds=None, spec_step_index=0):
        out = _orig(self, input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_index)
        idx = DeepSeekMultiTokenPredictorLayer._trace_call_idx
        if idx < 4:
            rec = {
                'input_ids': input_ids.detach().cpu().clone() if input_ids is not None else None,
                'hidden_in': previous_hidden_states.detach().cpu().float().clone(),
                'inputs_embeds': inputs_embeds.detach().cpu().float().clone() if inputs_embeds is not None else None,
                'positions': positions.detach().cpu().clone(),
                'out': out.detach().cpu().float().clone(),
            }
            DeepSeekMultiTokenPredictorLayer._trace_store.append(rec)
            rank = int(os.environ.get('RANK', '-1'))
            pos_str = str(positions.tolist()) if positions.numel() <= 20 else f'shape={tuple(positions.shape)}'
            print(f"[vllm trace rank={rank}] call{idx}: hin={tuple(rec['hidden_in'].shape)} "
                  f"ie={tuple(rec['inputs_embeds'].shape) if rec['inputs_embeds'] is not None else None} pos={pos_str}",
                  flush=True)
        DeepSeekMultiTokenPredictorLayer._trace_store = DeepSeekMultiTokenPredictorLayer._trace_store
        DeepSeekMultiTokenPredictorLayer._trace_call_idx += 1
        return out

    DeepSeekMultiTokenPredictorLayer.forward = traced = traced
    DeepSeekMultiTokenPredictorLayer.forward = traced
    DeepSeekMultiTokenPredictorLayer._traced = True
    DeepSeekMultiTokenPredictorLayer._nano_traced = True
    return "installed"


def _dump_trace(self, path):
    import torch
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMultiTokenPredictorLayer
    rank = int(os.environ.get('RANK', '0'))
    trace = getattr(DeepSeekMultiTokenPredictorLayer, '_trace_store', [])
    if rank == 0:
        torch.save(trace, path)
        return f"saved rank0 {len(trace)} records to {path}"
    return f"rank{rank} skipped"


def main():
    from vllm import LLM, SamplingParams
    os.environ['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'

    llm = LLM(
        model="/mnt/data/weights/openPangu-R-72B-2512",
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=512,
        enforce_eager=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )

    if hasattr(llm, 'collective_rpc'):
        r = llm.collective_rpc(_install_hook)
        print("hook install:", r)

    sp = SamplingParams(temperature=0.0, max_tokens=3)
    out = llm.generate(["The capital of France is"], sp)
    print("output tokens:", list(out[0].outputs[0].token_ids))
    print("output text:", out[0].outputs[0].text)

    if hasattr(llm, 'collective_rpc'):
        r = llm.collective_rpc(_dump_trace, args=("/tmp/pangu_vllm_mtp_trace.pt",))
        print("dump:", r)


if __name__ == "__main__":
    main()
