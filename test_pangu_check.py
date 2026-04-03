"""Quick check: PanGu MTP output vs vLLM ground truth."""
from nanovllm import LLM, SamplingParams

VLLM_GROUND_TRUTH = {
    0: [29314, 6870, 18497, 11288, 41141, 89352, 91029, 90341, 91518, 90251, 89793, 133304, 4154, 49, 43775],
    1: [34031, 39115, 93809, 87866, 89074, 35642, 5980, 18497, 14571, 35740, 38930, 34555, 89352, 93110, 25130],
}


def main():
    prompts = ['What is the capital of France?', 'Write a Python fibonacci function:']
    sp = SamplingParams(temperature=0, max_tokens=50)
    llm = LLM('/mnt/data/weights/openPangu-R-72B-2512', enforce_eager=True,
              tensor_parallel_size=4, max_model_len=4096, num_speculative_tokens=1)
    out = llm.generate(prompts, sp)

    for i, o in enumerate(out):
        tids = o['token_ids'][:15]
        gt = VLLM_GROUND_TRUTH[i]
        match = sum(1 for a, b in zip(tids, gt) if a == b)
        status = "PASS" if tids == gt else f"FAIL (match {match}/15)"
        print(f'[{i}] {status}')
        print(f'  nano: {tids}')
        print(f'  vllm: {gt}')
        print(f'  text: {o["text"][:120]}')


if __name__ == '__main__':
    main()
