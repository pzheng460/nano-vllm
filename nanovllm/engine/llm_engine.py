import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config, get_config, set_config, reset_config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        set_config(model, **kwargs)
        config = get_config()
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        eos = self.tokenizer.eos_token_id
        config.eos = eos[0] if isinstance(eos, list) else eos
        self.speculative = config.draft_model is not None or config.use_mtp
        self.num_speculative_tokens = config.num_speculative_tokens
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        draft_count = accept_count = 0
        if self.speculative and not is_prefill:
            all_token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess_speculative(seqs, all_token_ids)
            num_tokens = -sum(len(tids) for tids in all_token_ids)
            k = self.num_speculative_tokens
            draft_count = k * len(seqs)
            accept_count = sum(len(tids) - 1 for tids in all_token_ids)
        else:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, draft_count, accept_count

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        total_draft = total_accept = 0
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, draft_count, accept_count = self.step()
            total_draft += draft_count
            total_accept += accept_count
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                postfix = {
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                }
                if total_draft > 0:
                    postfix["Accept"] = f"{total_accept/total_draft:.1%}"
                pbar.set_postfix(postfix)
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        if total_draft > 0:
            print(f"Speculative decoding: {total_accept}/{total_draft} draft tokens accepted ({total_accept/total_draft:.1%}), "
                  f"avg {total_accept/max(total_draft//self.num_speculative_tokens, 1):.2f} tokens/step")
        return outputs
