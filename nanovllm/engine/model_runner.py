import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.eagle import EAGLEModel
from nanovllm.layers.sampler import Sampler
from nanovllm.layers.attention import Attention
from nanovllm.utils.context import set_context, get_context, reset_context, init_graph_params
from nanovllm.utils.loader import load_model
from nanovllm.utils.device import get_device_backend


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Get device backend (already initialized in nanovllm.__init__)
        self.device = get_device_backend()

        dist.init_process_group(self.device.get_dist_backend(), "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        self.device.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device.device_name)
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.speculative = config.draft_model is not None
        self.num_speculative_tokens = config.num_speculative_tokens
        if self.speculative:
            self.draft_model = EAGLEModel(
                hf_config,
                embed_tokens=self.model.model.embed_tokens,
                lm_head=self.model.lm_head,
            )
            load_model(self.draft_model, config.draft_model)
            self.last_hidden = {}  # seq_id -> hidden_state tensor
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if self.speculative:
            self.allocate_draft_kv_cache()
        if not self.enforce_eager:
            if self.device.is_cuda:
                self.capture_cudagraph()
            else:
                self.capture_aclgraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        self.device.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        self.device.empty_cache()
        self.device.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        self.device.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = self.device.mem_get_info()
        used = total - free
        peak = self.device.memory_stats()["allocated_bytes.all.peak"]
        current = self.device.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # Account for draft model KV cache (1 layer, full MHA)
        if self.speculative:
            num_attn_heads = hf_config.num_attention_heads // self.world_size
            draft_block_bytes = 2 * 1 * self.block_size * num_attn_heads * head_dim * hf_config.torch_dtype.itemsize
            block_bytes += draft_block_bytes
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def allocate_draft_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_attn_heads = hf_config.num_attention_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self.draft_kv_cache = torch.empty(
            2, 1, config.num_kvcache_blocks, self.block_size,
            num_attn_heads, head_dim,
        )
        for module in self.draft_model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.draft_kv_cache[0, 0]
                module.v_cache = self.draft_kv_cache[1, 0]

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True)
        block_tables = self.device.to_device(block_tables)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = self.device.to_device(torch.tensor(input_ids, dtype=torch.int64, pin_memory=True))
        positions = self.device.to_device(torch.tensor(positions, dtype=torch.int64, pin_memory=True))
        cu_seqlens_q = self.device.to_device(torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True))
        cu_seqlens_k = self.device.to_device(torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True))
        slot_mapping = self.device.to_device(torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True))
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        input_ids = self.device.to_device(torch.tensor(input_ids, dtype=torch.int64, pin_memory=True))
        positions = self.device.to_device(torch.tensor(positions, dtype=torch.int64, pin_memory=True))
        slot_mapping = self.device.to_device(torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True))
        block_tables = self.prepare_block_tables(seqs)
        if self.device.is_cuda:
            context_lens = self.device.to_device(torch.tensor(context_lens, dtype=torch.int32, pin_memory=True))
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        else:
            cu_seqlens_q = [(i + 1) for i in range(len(seqs))]
            set_context(False, cu_seqlens_q=cu_seqlens_q, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = self.device.to_device(torch.tensor(temperatures, dtype=torch.float32, pin_memory=True))
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager:
            return self.model.compute_logits(self.model(input_ids, positions))

        if self.device.is_cuda:
            # GPU: Use CUDA Graph
            if input_ids.size(0) > 512:
                return self.model.compute_logits(self.model(input_ids, positions))
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])
        else:
            # NPU: Use ACL Graph (similar to CUDA Graph)
            if input_ids.size(0) > 512:
                return self.model.compute_logits(self.model(input_ids, positions))
            bs = input_ids.size(0)
            context = get_context()
            graph_bs = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[graph_bs]
            graph_vars = self.graph_vars
            # Update tensor inputs (same as CUDA Graph)
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            # NPU-specific: Update List params (cu_seqlens_q, context_lens) via graph_task_update
            Attention.update_graph_params(graph_bs, context.cu_seqlens_q, context.context_lens)
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool):
        if self.speculative:
            if is_prefill:
                return self._run_prefill_with_hidden(seqs)
            else:
                return self._run_speculative_decode(seqs)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def _run_prefill_with_hidden(self, seqs: list[Sequence]) -> list[int]:
        """Prefill that also saves last hidden states for EAGLE draft."""
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        context = get_context()
        hidden = self.model(input_ids, positions)
        # Save last hidden state per sequence
        last_indices = context.cu_seqlens_q[1:] - 1
        for i, seq in enumerate(seqs):
            self.last_hidden[seq.seq_id] = hidden[last_indices[i]:last_indices[i]+1].clone()
        logits = self.model.compute_logits(hidden)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _prepare_verify(self, seqs: list[Sequence], all_draft_tokens: list[list[int]]):
        """Prepare prefill-like context for verification of speculative tokens."""
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        for seq, draft_tokens in zip(seqs, all_draft_tokens):
            # Verify tokens: [last_token, d_0, d_1, ..., d_{k-1}]
            verify_tokens = [seq.last_token] + draft_tokens
            num_verify = len(verify_tokens)
            seq_start_pos = len(seq) - 1  # position of last_token
            input_ids.extend(verify_tokens)
            positions.extend(list(range(seq_start_pos, seq_start_pos + num_verify)))
            cu_seqlens_q.append(cu_seqlens_q[-1] + num_verify)
            seqlen_k = len(seq) + len(draft_tokens)  # full context including draft
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(num_verify, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # Slot mapping for the verify tokens
            for j in range(num_verify):
                token_pos = len(seq) - 1 + j  # absolute position in sequence
                block_idx = token_pos // self.block_size
                block_offset = token_pos % self.block_size
                slot = seq.block_table[block_idx] * self.block_size + block_offset
                slot_mapping.append(slot)
        block_tables = self.prepare_block_tables(seqs)
        input_ids = self.device.to_device(torch.tensor(input_ids, dtype=torch.int64, pin_memory=True))
        positions = self.device.to_device(torch.tensor(positions, dtype=torch.int64, pin_memory=True))
        cu_seqlens_q = self.device.to_device(torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True))
        cu_seqlens_k = self.device.to_device(torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True))
        slot_mapping = self.device.to_device(torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True))
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def _run_speculative_decode(self, seqs: list[Sequence]) -> list[list[int]]:
        """Draft k tokens with EAGLE, then verify with target model."""
        k = self.num_speculative_tokens
        all_draft_tokens = []  # per-seq draft token lists

        # === Draft phase (per-seq, eager) ===
        for seq in seqs:
            target_hidden = self.last_hidden[seq.seq_id]
            draft_tokens = []
            cur_token_id = seq.last_token
            cur_pos = len(seq) - 1
            for step in range(k):
                input_id = self.device.to_device(
                    torch.tensor([cur_token_id], dtype=torch.int64, pin_memory=True))
                pos = self.device.to_device(
                    torch.tensor([cur_pos], dtype=torch.int64, pin_memory=True))
                # Slot mapping for draft KV cache
                block_idx = cur_pos // self.block_size
                block_offset = cur_pos % self.block_size
                slot = seq.block_table[block_idx] * self.block_size + block_offset
                slot_map = self.device.to_device(
                    torch.tensor([slot], dtype=torch.int32, pin_memory=True))
                block_table = self.device.to_device(
                    torch.tensor([seq.block_table], dtype=torch.int32, pin_memory=True))
                context_lens = self.device.to_device(
                    torch.tensor([cur_pos + 1], dtype=torch.int32, pin_memory=True))
                set_context(False, slot_mapping=slot_map, context_lens=context_lens, block_tables=block_table)
                draft_hidden = self.draft_model(input_id, pos, target_hidden)
                reset_context()
                # Get draft logits (use lm_head directly for single token)
                draft_logits = self.model.compute_logits_all(draft_hidden)
                if self.rank == 0:
                    d_token = draft_logits.argmax(dim=-1).item()
                else:
                    d_token = 0
                # Broadcast draft token in TP
                if self.world_size > 1:
                    d_tensor = torch.tensor([d_token], dtype=torch.int64, device=self.device.device_name)
                    dist.broadcast(d_tensor, 0)
                    d_token = d_tensor.item()
                draft_tokens.append(d_token)
                target_hidden = draft_hidden
                cur_token_id = d_token
                cur_pos += 1
            all_draft_tokens.append(draft_tokens)

        # === Verify phase (target model, prefill-like) ===
        input_ids, positions = self._prepare_verify(seqs, all_draft_tokens)
        hidden = self.model(input_ids, positions)
        target_logits = self.model.compute_logits_all(hidden)
        reset_context()

        # === Accept phase (greedy, rank 0 only) ===
        if self.rank == 0:
            all_accepted = []
            offset = 0
            for seq, draft_tokens in zip(seqs, all_draft_tokens):
                num_verify = len(draft_tokens) + 1
                seq_logits = target_logits[offset:offset + num_verify]
                target_predicted = seq_logits.argmax(dim=-1)  # [k+1]
                accepted = []
                for j in range(k):
                    if target_predicted[j].item() == draft_tokens[j]:
                        accepted.append(draft_tokens[j])
                    else:
                        accepted.append(target_predicted[j].item())
                        break
                else:
                    # All draft tokens accepted, add bonus token
                    accepted.append(target_predicted[k].item())
                # Save last hidden state at accepted position
                accepted_idx = offset + len(accepted) - 1
                self.last_hidden[seq.seq_id] = hidden[accepted_idx:accepted_idx+1].clone()
                all_accepted.append(accepted)
                offset += num_verify
            return all_accepted
        else:
            return None

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    @torch.inference_mode()
    def capture_aclgraph(self):
        """NPU ACL Graph capture (similar to capture_cudagraph)."""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size, dtype=hf_config.torch_dtype)
        # NPU-specific: List params for attention (cannot be captured by graph)
        context_lens = [1] * max_bs
        cu_seqlens_q = [(i + 1) for i in range(max_bs)]
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        init_graph_params(self.graph_bs)

        for bs in reversed(self.graph_bs):
            graph = torch.npu.NPUGraph()
            set_context(False, cu_seqlens_q=cu_seqlens_q[:bs], slot_mapping=slot_mapping[:bs],
                       context_lens=context_lens[:bs], block_tables=block_tables[:bs], capturing=False)
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            reset_context()
            set_context(False, cu_seqlens_q=cu_seqlens_q[:bs], slot_mapping=slot_mapping[:bs],
                       context_lens=context_lens[:bs], block_tables=block_tables[:bs], capturing=True)
            with torch.npu.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            self.device.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
            outputs=outputs,
        )
