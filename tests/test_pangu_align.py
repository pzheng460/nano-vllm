"""Test PanGu MTP alignment: compare baseline vs MTP output token-by-token."""
import gc
import torch
from nanovllm import LLM, SamplingParams


def main():
    prompts = [
        'What is the capital of France?',
        'Write a Python fibonacci function:',
        'Explain quantum computing in simple terms:',
        'List the planets in our solar system:',
        'How does photosynthesis work?',
    ] * 2
    sp = SamplingParams(temperature=0, max_tokens=100)
    MODEL = '/mnt/data/weights/openPangu-R-72B-2512'

    # --- Baseline (no MTP) ---
    print("=== Running baseline (no MTP) ===", flush=True)
    llm = LLM(MODEL, enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, disable_mtp=True)
    out = llm.generate(prompts, sp)
    base = [o['token_ids'] for o in out]
    del llm; gc.collect(); torch.cuda.empty_cache()
    from multiprocessing.shared_memory import SharedMemory
    try:
        s = SharedMemory('nanovllm'); s.close(); s.unlink()
    except Exception:
        pass

    # --- MTP K=1 ---
    print("=== Running MTP K=1 ===", flush=True)
    llm2 = LLM(MODEL, enforce_eager=True, tensor_parallel_size=4, max_model_len=4096, num_speculative_tokens=1)
    out2 = llm2.generate(prompts, sp)
    mtp = [o['token_ids'] for o in out2]

    # --- Compare ---
    print("\n=== Results ===")
    full = 0
    for i in range(len(base)):
        ml = 0
        for j in range(min(len(base[i]), len(mtp[i]))):
            if base[i][j] == mtp[i][j]:
                ml += 1
            else:
                break
        if ml == len(base[i]) == len(mtp[i]):
            full += 1
        else:
            print(f'  seq[{i}]: diverge@{ml}/{min(len(base[i]),len(mtp[i]))} '
                  f'base={base[i][ml] if ml<len(base[i]) else "END"} '
                  f'mtp={mtp[i][ml] if ml<len(mtp[i]) else "END"}')
    print(f'Full match: {full}/{len(base)}')
    print(f'Sample base[0][:15]: {base[0][:15]}')
    print(f'Sample mtp[0][:15]:  {mtp[0][:15]}')


if __name__ == '__main__':
    main()
