import atexit
import os
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

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
        self.draft_async = config.draft_async
        # Set env var before spawning draft so it picks up profiling
        if config.profile and config.draft_async:
            os.environ["DRAFT_PROFILE_PATH"] = "draft_trace.json"
        ctx = mp.get_context("spawn")

        # Spawn TP workers (ranks 1..tp_size-1)
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        # Spawn draft process BEFORE rank 0 init (collective init_process_group)
        if config.draft_async:
            from nanovllm.engine._draft_entry import draft_entry
            init_q = ctx.Queue()
            self.draft_process = ctx.Process(
                target=draft_entry,
                args=(config, config.draft_rank, init_q),
            )
            self.draft_process.start()
            self.ps.append(self.draft_process)

        # Rank 0 joins (blocks in init_process_group until all ranks ready)
        self.model_runner = ModelRunner(config, 0, self.events)

        # Wait for draft to report its num_kvcache_blocks
        if config.draft_async:
            num_blocks = init_q.get(timeout=180)
            init_q.close()

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True, trust_remote_code=True)
        eos = self.tokenizer.eos_token_id
        config.eos = eos[0] if isinstance(eos, list) else eos
        self.speculative = config.draft_model is not None or config.use_mtp
        self.num_speculative_tokens = config.num_speculative_tokens
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        # Send exit to draft via NCCL
        if self.draft_async and self.model_runner.async_pg is not None:
            self.model_runner._send_cmd(2)  # exit
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()
        # Merge target + draft traces if profiling was enabled
        self._merge_traces()

    def _merge_traces(self):
        """Merge target + draft profiler traces into a single file."""
        target_path = "target_trace.json"
        draft_path = os.environ.get("DRAFT_PROFILE_PATH", "")
        if not os.path.exists(target_path):
            return
        import json, gzip
        with open(target_path) as f:
            target = json.load(f)
        # Tag target process names
        for ev in target.get("traceEvents", []):
            if ev.get("ph") == "M" and ev.get("name") == "process_name":
                ev["args"]["name"] = "Target_GPU0 " + str(ev["args"].get("name", ""))
        if draft_path and os.path.exists(draft_path):
            with open(draft_path) as f:
                draft = json.load(f)
            for ev in draft.get("traceEvents", []):
                pid = ev.get("pid")
                if isinstance(pid, int):
                    ev["pid"] = pid + 10000
                if ev.get("ph") == "M" and ev.get("name") == "process_name":
                    ev["args"]["name"] = "Draft_GPU1 " + str(ev["args"].get("name", ""))
            target["traceEvents"].extend(draft["traceEvents"])
            os.remove(draft_path)
            merged_path = "profile_merged.json.gz"
        else:
            merged_path = "profile_target.json.gz"
        with gzip.open(merged_path, "wt") as f:
            json.dump(target, f)
        os.remove(target_path)
        print(f"Profile trace saved to: {merged_path}")

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        draft_count = accept_count = 0
        per_pos_accept = [0] * self.num_speculative_tokens if self.speculative else []
        if self.speculative and not is_prefill:
            all_token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess_speculative(seqs, all_token_ids)
            num_tokens = -sum(len(tids) for tids in all_token_ids)
            k = self.num_speculative_tokens
            draft_count = k * len(seqs)
            accept_count = sum(len(tids) - 1 for tids in all_token_ids)
            for tids in all_token_ids:
                num_accepted = len(tids) - 1  # number of draft positions accepted
                for j in range(min(num_accepted, k)):
                    per_pos_accept[j] += 1
        else:
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids)
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        # Clean up finished sequences in draft
        if self.draft_async and self.model_runner.async_pg is not None:
            # Drain pending push from draft BEFORE cleanup (NCCL ordering)
            mr = self.model_runner
            # MTP path drains its push tree-cache inside _run_ssd_decode itself
            # (irecv kicked off right after cmd=5 isend, waited before return —
            # so the data transfer overlaps with target's accept work). Only
            # the eagle_async path still drains here.
            if (getattr(mr, '_push_pending', False) and not is_prefill
                    and not mr.use_mtp):
                import torch
                d = mr.device.device_name
                if not hasattr(mr, '_local_tree_cache'):
                    mr._local_tree_cache = {}
                sz_buf = torch.zeros(1, dtype=torch.int64, device=d)
                dist.recv(sz_buf, src=mr.draft_rank, group=mr.async_pg)
                buf_size = int(sz_buf[0].tolist())
                push_buf = torch.zeros(buf_size, dtype=torch.int64, device=d)
                dist.recv(push_buf, src=mr.draft_rank, group=mr.async_pg)
                idx = 0
                n_push = int(push_buf[idx].tolist()); idx += 1
                for _ in range(n_push):
                    sid = int(push_buf[idx].tolist()); n_e = int(push_buf[idx+1].tolist()); K_a = int(push_buf[idx+2].tolist()); idx += 3
                    if n_e > 0:
                        keys = push_buf[idx:idx+n_e*3].reshape(n_e, 3).clone(); idx += n_e*3
                        tokens = push_buf[idx:idx+n_e*K_a].reshape(n_e, K_a).clone(); idx += n_e*K_a
                        mr._local_tree_cache[sid] = (keys, tokens)
                mr._push_pending = False
            for seq in seqs:
                if seq.is_finished:
                    self.model_runner._send_cmd(3)  # cleanup
                    import torch
                    meta = torch.tensor([seq.seq_id], dtype=torch.int64, device=self.model_runner.device.device_name)
                    dist.send(meta, dst=self.model_runner.draft_rank, group=self.model_runner.async_pg)
                    ack = torch.zeros(1, dtype=torch.int64, device=self.model_runner.device.device_name)
                    dist.recv(ack, src=self.model_runner.draft_rank, group=self.model_runner.async_pg)
        # Drop per-seq caches for finished seqs (these hold .clone()'d GPU
        # tensors; without eviction they grow unbounded and were traced to
        # a mid-run CUDA illegal-access crash on Vicuna EAGLE-1 ~prompt 13).
        mr = getattr(self, 'model_runner', None)
        if mr is not None:
            finished_ids = [seq.seq_id for seq in seqs if seq.is_finished]
            for cache_attr in ('last_hidden', '_draft_prev_hidden', '_pending_draft'):
                cache = getattr(mr, cache_attr, None)
                if isinstance(cache, dict):
                    for sid in finished_ids:
                        cache.pop(sid, None)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens, draft_count, accept_count, per_pos_accept

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        config = get_config()
        prof = None
        if config.profile:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=True,
            )
            prof.__enter__()
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        # Reset cache hit counters
        if hasattr(self.model_runner, '_cache_hit'):
            self.model_runner._cache_hit = 0
            self.model_runner._cache_miss = 0
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        total_draft = total_accept = 0
        k = self.num_speculative_tokens
        total_per_pos = [0] * k
        total_per_pos_total = [0] * k
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens, draft_count, accept_count, per_pos_accept = self.step()
            total_draft += draft_count
            total_accept += accept_count
            if draft_count > 0:
                n_seqs = draft_count // k
                for j in range(k):
                    total_per_pos[j] += per_pos_accept[j]
                    total_per_pos_total[j] += n_seqs
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
                  f"avg {total_accept/max(total_draft//k, 1):.2f} tokens/step")
            per_pos_str = ", ".join(
                f"pos{j}: {total_per_pos[j]}/{total_per_pos_total[j]} ({total_per_pos[j]/max(total_per_pos_total[j],1):.1%})"
                for j in range(k)
            )
            print(f"Per-position acceptance: {per_pos_str}")
            if hasattr(self.model_runner, '_cache_hit'):
                ch, cm = self.model_runner._cache_hit, self.model_runner._cache_miss
                total_c = ch + cm
                print(f"Cache hit: {ch}/{total_c} ({ch/total_c:.1%})" if total_c > 0 else "Cache hit: N/A")
        if prof:
            prof.__exit__(None, None, None)
            target_path = "target_trace.json"
            prof.export_chrome_trace(target_path)
            print(f"Target trace saved to: {target_path}")
            # Merge target + draft traces if draft trace exists
            draft_path = os.environ.get("DRAFT_PROFILE_PATH", "")
            # Note: draft trace is written by draft process on exit,
            # merge must happen after draft exits (in LLM.cleanup or manually)
        return outputs
