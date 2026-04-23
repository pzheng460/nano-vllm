"""Load PanGu in nano-vllm and verify MTP weights were actually loaded from checkpoint."""
import sys

MODEL = "/mnt/data/weights/openPangu-R-72B-2512"


def main():
    from nanovllm import LLM
    from safetensors import safe_open
    import torch

    llm = LLM(
        MODEL,
        tensor_parallel_size=4,
        trust_remote_code=True,
        max_model_len=2048,
        enforce_eager=True,
        num_speculative_tokens=1,
    )

    # Broadcast check to rank 0 only (multi-proc would need collective_rpc)
    mr = llm.model_runner
    model = mr.model
    print(f"Rank: {mr.rank}/{mr.tp_size}")
    print(f"Model: {type(model).__name__}")
    print(f"MTP layers: {len(model.model.mtp_layers)}")
    mtp0 = model.model.mtp_layers[0]
    print(f"MTP[0] submodules: {[n for n, _ in mtp0.named_children()]}")
    # Print some weight stats
    for name, p in [
        ("mtp_layers.0.enorm.weight", mtp0.enorm.weight),
        ("mtp_layers.0.hnorm.weight", mtp0.hnorm.weight),
        ("mtp_layers.0.eh_proj.weight", mtp0.eh_proj.weight),
        ("mtp_layers.0.shared_head.norm.weight", mtp0.shared_head.norm.weight),
        ("mtp_layers.0.shared_head.head.weight", mtp0.shared_head.head.weight),
    ]:
        mean = p.float().abs().mean().item()
        nonzero = (p != 0).float().mean().item()
        print(f"  {name:55s} shape={tuple(p.shape)} |x|mean={mean:.6f} nonzero={nonzero:.4f}")

    # Compare to main model weights for sanity
    print("\nMain model refs:")
    for name, p in [
        ("model.embed_tokens", model.model.embed_tokens.weight),
        ("model.norm", model.model.norm.weight),
        ("lm_head", model.lm_head.weight),
    ]:
        print(f"  {name:55s} shape={tuple(p.shape)} |x|mean={p.float().abs().mean().item():.6f}")

    # Compare MTP head to main lm_head (check they match — both stored separately in ckpt but weights equal)
    main_head = model.lm_head.weight
    mtp_head = mtp0 = model.model.mtp_layers[0].shared_head.head.weight
    same_shape = main_head.shape == mtp_head.shape
    if same_shape:
        close = torch.allclose(main_head, mtp_head)
        max_diff = (main_head - mtp_head).abs().max().item() if close else None
        print(f"  main_head == MTP head: {close}, max_diff={max_diff}")

    # Compare MTP shared_head.norm to main model.norm (the checkpoint analysis showed these differ)
    main_norm = model.model.norm.weight
    mtp_norm = model.model.mtp_layers[0].shared_head.norm.weight
    close = torch.allclose(main_norm, mtp_norm)
    max_diff = (main_norm - mtp_norm).abs().max().item()
    print(f"  main_model_norm vs mtp_shared_head_norm: close={close} max_diff={max_diff:.4f}")


if __name__ == "__main__":
    main()
