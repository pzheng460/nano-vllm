import os
import pickle
import torch
import torch.nn.functional as F
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


def _get_model_cls(hf_config):
    model_type = getattr(hf_config, 'model_type', 'qwen3')
    if model_type == 'mimo':
        from nanovllm.models.mimo import MiMoForCausalLM
        return MiMoForCausalLM
    # PanGu model types
    pangu_types = {'PanguProMoE', 'PanguProMoEV2', 'PanguUltraMoE', 'PanguEmbedded', 'pangu'}
    if model_type in pangu_types or getattr(hf_config, 'param_sink_number', 0) > 0:
        from nanovllm.models.pangu import PanguForCausalLM
        return PanguForCausalLM
    if model_type == 'llama':
        from nanovllm.models.llama import LlamaForCausalLM
        return LlamaForCausalLM
    # Qwen2/Qwen3 (bias/QK-norm handled by config flags)
    return Qwen3ForCausalLM


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tp_size = config.tensor_parallel_size
        self.world_size = config.num_gpus  # includes draft rank if async
        self.rank = rank
        self.event = event

        self.device = get_device_backend()

        # Unified process group: all ranks (target TP + draft) join one world
        dist.init_process_group(self.device.get_dist_backend(), f"tcp://localhost:{os.environ.get('NCCL_PORT', '2333')}",
                                world_size=self.world_size, rank=rank)
        # Sub-groups IMMEDIATELY after init (collective, all ranks must participate)
        tp_ranks = list(range(self.tp_size))
        self.tp_group = dist.new_group(tp_ranks) if self.world_size > 1 else None
        self.async_pg = None
        if config.draft_async:
            self.async_pg = dist.new_group([0, config.draft_rank])
            self.draft_rank = config.draft_rank

        self.device.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device.device_name)
        model_cls = _get_model_cls(hf_config)
        self.model = model_cls(hf_config, tp_group=self.tp_group, tp_size=self.tp_size)
        load_model(self.model, config.model)
        self.use_mtp = config.use_mtp
        self.draft_async = config.draft_async
        self.speculative = config.draft_model is not None or self.use_mtp
        # MTP layers have their own hidden norm (MiMo: hidden_layernorm, PanGu: hnorm)
        # and always expect unnormed hidden from the target model's residual stream
        self._mtp_uses_unnormed = (
            self.use_mtp
            and hasattr(self.model.model, 'mtp_layers')
            and len(self.model.model.mtp_layers) > 0
        )
        # MTP uses its own shared_head (norm + head trained together, NOT shared with target lm_head)
        self.num_speculative_tokens = config.num_speculative_tokens
        if config.draft_model is not None:
            draft_hf = config.draft_hf_config
            is_eagle3 = draft_hf is not None and getattr(draft_hf, 'draft_vocab_size', None) is not None
            self._is_eagle3 = is_eagle3
            if is_eagle3:
                from nanovllm.models.eagle import Eagle3Model
                self.draft_model = Eagle3Model(
                    hf_config, draft_hf,
                    embed_tokens=self.model.model.embed_tokens,
                    tp_group=self.tp_group, tp_size=self.tp_size,
                )
            else:
                eagle_cfg = draft_hf if draft_hf is not None else hf_config
                self.draft_model = EAGLEModel(
                    eagle_cfg,
                    embed_tokens=self.model.model.embed_tokens,
                    lm_head=self.model.lm_head,
                    tp_group=self.tp_group, tp_size=self.tp_size,
                )
            load_model(self.draft_model, config.draft_model)
        if self.speculative:
            self.last_hidden = {}
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if config.draft_model is not None:
            self.allocate_draft_kv_cache()
        if self.use_mtp:
            self.allocate_mtp_kv_cache()
        if not self.enforce_eager:
            if self.device.is_cuda:
                self.capture_cudagraph()
            else:
                self.capture_aclgraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        # Pre-allocate NCCL buffers for SSD communication
        if self.draft_async and rank == 0:
            d = self.device.device_name
            max_seqs = config.max_num_seqs
            K = config.num_speculative_tokens
            self._cmd_buf = torch.zeros(2, dtype=torch.int64, device=d)
            # Spec phase pre-allocated buffers
            self._lookup_buf = torch.zeros(1 + max_seqs * 3, dtype=torch.int64, device=d)
            self._result_buf = torch.zeros(max_seqs * K, dtype=torch.int64, device=d)

        if self.tp_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier(group=self.tp_group)
            elif rank < self.tp_size:
                dist.barrier(group=self.tp_group)
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.tp_size > 1:
            self.shm.close()
            dist.barrier(group=self.tp_group)
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
        assert self.tp_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.tp_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.tp_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        self.device.empty_cache()
        self.device.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        # During warmup, skip SSD communication and EAGLE KV prefill
        saved_async = self.draft_async
        self.draft_async = False
        self._warmup = True
        self.run(seqs, True)
        self._warmup = False
        self.draft_async = saved_async
        self.device.empty_cache()

    def _get_kv_head_dim(self):
        """Get the effective head_dim for KV cache allocation.
        For PanGu with sink attention, head_dim = qk_rope_dim + qk_nope_dim."""
        hf = self.config.hf_config
        qk_rope_dim = getattr(hf, 'qk_rope_dim', None)
        qk_nope_dim = getattr(hf, 'qk_nope_dim', None)
        if qk_rope_dim is not None and qk_nope_dim is not None:
            return qk_rope_dim + qk_nope_dim
        return getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = self.device.mem_get_info()
        used = total - free
        peak = self.device.memory_stats()["allocated_bytes.all.peak"]
        current = self.device.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size
        head_dim = self._get_kv_head_dim()
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # Account for draft model KV cache (1 layer)
        if self.speculative and config.draft_model is not None:
            draft_hf = config.draft_hf_config
            if draft_hf is not None:
                num_attn_heads = getattr(draft_hf, 'num_key_value_heads', draft_hf.num_attention_heads) // self.tp_size
            else:
                num_attn_heads = hf_config.num_attention_heads // self.tp_size
            draft_block_bytes = 2 * 1 * self.block_size * num_attn_heads * head_dim * hf_config.torch_dtype.itemsize
            block_bytes += draft_block_bytes
        # Account for MTP KV cache (GQA, same kv_heads as main model)
        if self.use_mtp:
            num_mtp_layers = hf_config.num_nextn_predict_layers
            mtp_block_bytes = 2 * num_mtp_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
            block_bytes += mtp_block_bytes
        # Reserve blocks for sink KV
        num_sink_blocks = config.num_sink_blocks
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > num_sink_blocks
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        from nanovllm.models.pangu import PanguSinkAttention
        layer_id = 0
        for layer in self.model.model.layers:
            for module in layer.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.kv_cache[0, layer_id]
                    module.v_cache = self.kv_cache[1, layer_id]
                    layer_id += 1
            # After setting cache, populate sink KV on PanguSinkAttention
            if num_sink_blocks > 0 and hasattr(layer, 'self_attn') and isinstance(layer.self_attn, PanguSinkAttention):
                layer.self_attn.sink_block_ids = list(range(num_sink_blocks))
                layer.self_attn._populate_sink_to_cache(self.block_size)

    def allocate_draft_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # EAGLE3 uses GQA (num_kv_heads from draft config), EAGLE1 uses full MHA
        draft_hf = config.draft_hf_config
        if draft_hf is not None:
            num_attn_heads = getattr(draft_hf, 'num_key_value_heads', draft_hf.num_attention_heads) // self.tp_size
        else:
            num_attn_heads = hf_config.num_attention_heads // self.tp_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        self.draft_kv_cache = torch.empty(
            2, 1, config.num_kvcache_blocks, self.block_size,
            num_attn_heads, head_dim,
        )
        for module in self.draft_model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.draft_kv_cache[0, 0]
                module.v_cache = self.draft_kv_cache[1, 0]

    def allocate_mtp_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size
        head_dim = self._get_kv_head_dim()
        num_mtp_layers = hf_config.num_nextn_predict_layers
        num_sink_blocks = config.num_sink_blocks
        self.mtp_kv_cache = torch.empty(
            2, num_mtp_layers, config.num_kvcache_blocks, self.block_size,
            num_kv_heads, head_dim,
        )
        from nanovllm.models.pangu import PanguSinkAttention
        layer_id = 0
        for mtp_layer in self.model.model.mtp_layers:
            for module in mtp_layer.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.mtp_kv_cache[0, layer_id]
                    module.v_cache = self.mtp_kv_cache[1, layer_id]
                    layer_id += 1
            # Populate sink KV for MTP's decoder layer
            if num_sink_blocks > 0 and hasattr(mtp_layer, 'mtp_block'):
                attn = getattr(mtp_layer.mtp_block, 'self_attn', None)
                if isinstance(attn, PanguSinkAttention):
                    attn.sink_block_ids = list(range(num_sink_blocks))
                    attn._populate_sink_to_cache(self.block_size)

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
            # Prepend sink block IDs for prefix cache prefill
            if self.config.num_sink_blocks > 0:
                sink_ids = list(range(self.config.num_sink_blocks))
                bs = block_tables.size(0)
                sink_cols = torch.tensor([sink_ids] * bs, dtype=torch.int32, device=block_tables.device)
                block_tables = torch.cat([sink_cols, block_tables], dim=1)
                sink_ctx = self.config.sink_len
                cu_seqlens_k = [cu_seqlens_k[0]] + [c + sink_ctx * i for i, c in enumerate(cu_seqlens_k[1:], 1)]
                max_seqlen_k += sink_ctx
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
            context_lens.append(len(seq))  # No sink adjustment; handled inside attention
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

    def lm_head_logits(self, hidden_normed: torch.Tensor) -> torch.Tensor:
        """Compute logits from already-normed hidden states (avoids double-norm)."""
        return self.model.lm_head(hidden_normed)

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
                if self.draft_async:
                    return self._run_ssd_prefill_with_hidden(seqs)
                return self._run_prefill_with_hidden(seqs)
            elif self.draft_async:
                return self._run_ssd_decode(seqs)
            elif self.use_mtp:
                return self._run_mtp_decode(seqs)
            else:
                return self._run_speculative_decode(seqs)
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    def _get_eagle3_aux_layers(self):
        """Get auxiliary layer indices for EAGLE3: (2, N//2, N-3)."""
        N = self.config.hf_config.num_hidden_layers
        return (2, N // 2, N - 3)

    def _run_target_with_aux(self, input_ids, positions):
        """Run target model layer-by-layer, extracting aux hidden states for EAGLE3."""
        model_inner = self.model.model
        hidden_states = model_inner.embed_tokens(input_ids)
        residual = None
        aux_layers = self._get_eagle3_aux_layers()
        aux_hiddens = {}
        for i, layer in enumerate(model_inner.layers):
            hidden_states, residual = layer(positions, hidden_states, residual)
            if i in aux_layers:
                aux_hiddens[i] = (hidden_states + residual).detach().clone()
        hidden_states, _ = model_inner.norm(hidden_states, residual)
        # Concatenate aux hiddens in order
        aux_concat = torch.cat([aux_hiddens[l] for l in aux_layers], dim=-1)
        return hidden_states, aux_concat

    @torch.inference_mode()
    def _run_prefill_with_hidden(self, seqs: list[Sequence]) -> list[int]:
        """Prefill that also saves last hidden states for speculative draft."""
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        context = get_context()
        is_eagle3 = getattr(self, '_is_eagle3', False)
        if is_eagle3:
            model_out, aux_concat = self._run_target_with_aux(input_ids, positions)
        else:
            model_out = self.model(input_ids, positions)
        # MiMo returns unnormed residual; Qwen3 returns normed hidden
        if self._mtp_uses_unnormed:
            # MiMo: model returns unnormed, need to norm for logits
            hidden = model_out  # unnormed
            hidden_normed = self.model.model.norm(hidden)
        else:
            # Qwen3/Qwen2/PanGu: model returns normed
            hidden_normed = model_out
            hidden = model_out  # already normed (no unnormed available)
        last_indices = context.cu_seqlens_q[1:] - 1
        logits = self.lm_head_logits(hidden_normed)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        if self.use_mtp:
            # vLLM shifts input_ids for MTP: position i gets token_{i+1}'s embedding
            shifted_ids = input_ids.clone()
            shifted_ids[:-1] = input_ids[1:]
            # Last position of each seq gets the sampled next token
            if self.rank == 0:
                for i, idx in enumerate(last_indices.tolist()):
                    shifted_ids[idx] = token_ids[i]
            if self.tp_size > 1:
                dist.broadcast(shifted_ids, 0, group=self.tp_group)
            embeds = self.model.model.embed_tokens(shifted_ids)
            mtp_hidden = hidden if self._mtp_uses_unnormed else hidden_normed
            self.model.model.mtp_layers[0](embeds, mtp_hidden, positions)
        # EAGLE uses normed hidden (trained with model.model() output which is post-RMSNorm);
        # MiMo MTP uses unnormed (model returns it directly)
        save_hidden = hidden
        # Populate EAGLE KV cache during prefill so draft attention has valid context
        if hasattr(self, 'draft_model') and not getattr(self, '_warmup', False):
            ctx = get_context()
            set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                        ctx.max_seqlen_k, ctx.slot_mapping, None, None)
            if is_eagle3:
                self.draft_model(input_ids, positions, hidden_normed, aux_hiddens=aux_concat)
            else:
                self.draft_model(input_ids, positions, hidden_normed)
            set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                        ctx.max_seqlen_k, ctx.slot_mapping, ctx.context_lens, ctx.block_tables)
        for i, seq in enumerate(seqs):
            self.last_hidden[seq.seq_id] = save_hidden[last_indices[i]:last_indices[i]+1].clone()
            if is_eagle3:
                if not hasattr(self, '_last_aux_hidden'):
                    self._last_aux_hidden = {}
                self._last_aux_hidden[seq.seq_id] = aux_concat[last_indices[i]:last_indices[i]+1].clone()
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
        # Sink KV is handled inside PanguSinkAttention (no external adjustment needed)
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
                if getattr(self, '_is_eagle3', False):
                    # EAGLE3: aux_hiddens only for step 0 (from target); step 1+ uses None (fc uses tripled hidden)
                    aux_h = self._last_aux_hidden.get(seq.seq_id) if (step == 0 and hasattr(self, '_last_aux_hidden')) else None
                    draft_hidden = self.draft_model(input_id, pos, target_hidden, aux_hiddens=aux_h)
                else:
                    draft_hidden = self.draft_model(input_id, pos, target_hidden)
                reset_context()
                # Get draft logits
                if getattr(self, '_is_eagle3', False):
                    # EAGLE3: draft logits in reduced vocab → map to target vocab
                    draft_logits = self.draft_model.compute_logits(draft_hidden)
                    if self.rank == 0:
                        draft_id = draft_logits.argmax(dim=-1).item()
                        d_token = self.draft_model.d2t[draft_id].item()
                else:
                    draft_logits = self.model.compute_logits_all(draft_hidden)
                if self.rank == 0 and not getattr(self, '_is_eagle3', False):
                    d_token = draft_logits.argmax(dim=-1).item()
                elif self.rank != 0:
                    d_token = 0
                # Broadcast draft token in TP
                if self.tp_size > 1:
                    d_tensor = torch.tensor([d_token], dtype=torch.int64, device=self.device.device_name)
                    dist.broadcast(d_tensor, 0, group=self.tp_group)
                    d_token = d_tensor.item()
                draft_tokens.append(d_token)
                target_hidden = draft_hidden
                cur_token_id = d_token
                cur_pos += 1
            all_draft_tokens.append(draft_tokens)

        # === Verify phase (target model, prefill-like) ===
        input_ids, positions = self._prepare_verify(seqs, all_draft_tokens)
        is_eagle3 = getattr(self, '_is_eagle3', False)
        if is_eagle3:
            hidden, aux_concat = self._run_target_with_aux(input_ids, positions)
        else:
            hidden = self.model(input_ids, positions)  # normed for Qwen2/3, unnormed for MiMo
        # MiMo returns unnormed → norm for logits; Qwen2/3 returns normed → use directly
        if self._mtp_uses_unnormed:
            target_logits = self.model.compute_logits_all(self.model.model.norm(hidden))
        else:
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
                # Save normed hidden at accepted position (EAGLE trained with normed hidden)
                accepted_idx = offset + len(accepted) - 1
                self.last_hidden[seq.seq_id] = hidden[accepted_idx:accepted_idx+1].clone()
                if is_eagle3:
                    if not hasattr(self, '_last_aux_hidden'):
                        self._last_aux_hidden = {}
                    self._last_aux_hidden[seq.seq_id] = aux_concat[accepted_idx:accepted_idx+1].clone()
                all_accepted.append(accepted)
                offset += num_verify

            # === EAGLE first-pass: update EAGLE KV for accepted tokens (matches vLLM) ===
            if hasattr(self, 'draft_model'):
                d = self.device.device_name
                offset = 0
                for seq_idx, seq in enumerate(seqs):
                    num_accepted = len(all_accepted[seq_idx])
                    num_verify = len(all_draft_tokens[seq_idx]) + 1
                    seq_start_pos = len(seq) - 1
                    # Update EAGLE KV at accepted positions with real target hidden
                    # (skip last accepted — next round's first draft step overwrites it)
                    for j in range(num_accepted - 1):
                        acc_pos = seq_start_pos + j + 1
                        tok_id = all_accepted[seq_idx][j]
                        target_h = hidden[offset + j + 1:offset + j + 2]
                        inp = self.device.to_device(
                            torch.tensor([tok_id], dtype=torch.int64, pin_memory=True))
                        p = self.device.to_device(
                            torch.tensor([acc_pos], dtype=torch.int64, pin_memory=True))
                        bi = acc_pos // self.block_size
                        bo = acc_pos % self.block_size
                        sl = seq.block_table[bi] * self.block_size + bo
                        sm = self.device.to_device(
                            torch.tensor([sl], dtype=torch.int32, pin_memory=True))
                        bt = self.device.to_device(
                            torch.tensor([seq.block_table], dtype=torch.int32, pin_memory=True))
                        cl = self.device.to_device(
                            torch.tensor([acc_pos + 1], dtype=torch.int32, pin_memory=True))
                        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
                        self.draft_model(inp, p, target_h)
                        reset_context()
                    offset += num_verify

            return all_accepted
        else:
            return None

    @torch.inference_mode()
    def _run_mtp_decode(self, seqs: list[Sequence]) -> list[list[int]]:
        """Draft k tokens with MTP layers, then verify with target model.
        Batched: all seqs processed together per draft step.
        """
        k = self.num_speculative_tokens
        d = self.device.device_name
        n_seqs = len(seqs)
        all_draft_tokens = [[] for _ in seqs]
        mtp_layer = self.model.model.mtp_layers[0]
        embed_tokens = self.model.model.embed_tokens

        # === Draft phase (batched) ===
        block_tables = self.prepare_block_tables(seqs)
        cur_tokens = [seq.last_token for seq in seqs]
        cur_positions = [len(seq) - 1 for seq in seqs]
        cur_hiddens = torch.cat([self.last_hidden[seq.seq_id] for seq in seqs], dim=0)

        def _compute_mtp_logits(normed):
            if hasattr(mtp_layer, 'shared_head'):
                return F.linear(normed, mtp_layer.shared_head.head.weight)
            return F.linear(normed, self.model.lm_head.weight)

        for step in range(k):
            input_ids = self.device.to_device(torch.tensor(cur_tokens, dtype=torch.int64, pin_memory=True))
            positions = self.device.to_device(torch.tensor(cur_positions, dtype=torch.int64, pin_memory=True))
            slot_mapping = []
            context_lens = []
            for i, seq in enumerate(seqs):
                pos = cur_positions[i]
                bi = pos // self.block_size
                bo = pos % self.block_size
                slot_mapping.append(seq.block_table[bi] * self.block_size + bo)
                context_lens.append(pos + 1)
            slot_mapping = self.device.to_device(torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True))
            context_lens_t = self.device.to_device(torch.tensor(context_lens, dtype=torch.int32, pin_memory=True))
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens_t, block_tables=block_tables)
            token_embeds = embed_tokens(input_ids)
            mtp_normed, _ = mtp_layer(token_embeds, cur_hiddens, positions)
            reset_context()
            draft_logits = _compute_mtp_logits(mtp_normed)
            if self.tp_size > 1:
                all_l = [torch.empty_like(draft_logits) for _ in range(self.tp_size)] if self.rank == 0 else None
                dist.gather(draft_logits, all_l, 0, group=self.tp_group)
                draft_logits = torch.cat(all_l, -1) if self.rank == 0 else None
            if self.rank == 0:
                tokens = draft_logits.argmax(dim=-1).tolist()
            else:
                tokens = [0] * n_seqs
            if self.tp_size > 1:
                t = torch.tensor(tokens, dtype=torch.int64, device=d)
                dist.broadcast(t, 0, group=self.tp_group)
                tokens = t.tolist()
            for i in range(n_seqs):
                all_draft_tokens[i].append(tokens[i])
            cur_tokens = tokens
            cur_positions = [p + 1 for p in cur_positions]
            cur_hiddens = mtp_normed

        # === Verify ===
        verify_ids, verify_pos = self._prepare_verify(seqs, all_draft_tokens)
        hidden = self.model(verify_ids, verify_pos)
        hidden_normed = self.model.model.norm(hidden)
        target_logits = self.model.compute_logits_all(hidden_normed)
        reset_context()

        # === Accept ===
        if self.rank == 0:
            all_accepted = []
            offset = 0
            for seq, draft_tokens in zip(seqs, all_draft_tokens):
                num_verify = len(draft_tokens) + 1
                seq_logits = target_logits[offset:offset + num_verify]
                target_predicted = seq_logits.argmax(dim=-1)
                accepted = []
                for j in range(k):
                    if target_predicted[j].item() == draft_tokens[j]:
                        accepted.append(draft_tokens[j])
                    else:
                        accepted.append(target_predicted[j].item())
                        break
                else:
                    accepted.append(target_predicted[k].item())
                all_accepted.append(accepted)
                offset += num_verify
        else:
            all_accepted = [[] for _ in seqs]

        # === Broadcast accepted tokens (TP) & update last_hidden ===
        if self.tp_size > 1:
            n_acc = torch.tensor([len(a) for a in all_accepted], dtype=torch.int64, device=d)
            dist.broadcast(n_acc, 0, group=self.tp_group)
            if self.rank != 0:
                all_accepted = [[0] * int(n_acc[i].item()) for i in range(n_seqs)]
            for i in range(n_seqs):
                if len(all_accepted[i]) > 0:
                    acc_t = torch.tensor(all_accepted[i], dtype=torch.int64, device=d)
                    dist.broadcast(acc_t, 0, group=self.tp_group)
                    if self.rank != 0:
                        all_accepted[i] = acc_t.tolist()

        if self.rank == 0:
            offset = 0
            for seq_idx, seq in enumerate(seqs):
                num_accepted = len(all_accepted[seq_idx])
                accepted_idx = offset + num_accepted - 1 if num_accepted > 0 else offset
                self.last_hidden[seq.seq_id] = hidden[accepted_idx:accepted_idx + 1].clone()
                offset += len(all_draft_tokens[seq_idx]) + 1

        if self.rank == 0:
            return all_accepted
        return None

    def _send_cmd(self, cmd: int, aux: int = 0):
        """Send command to draft via NCCL. [cmd, aux] in one message."""
        if not hasattr(self, '_cmd_buf'):
            return
        self._cmd_buf[0] = cmd
        self._cmd_buf[1] = aux
        dist.send(self._cmd_buf, dst=self.draft_rank, group=self.async_pg)

    @torch.inference_mode()
    def _run_ssd_prefill_with_hidden(self, seqs: list[Sequence]) -> list[int]:
        """SSD prefill: run target model, send hidden to draft via NCCL."""
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        context = get_context()
        d = self.device.device_name

        hidden = self.model(input_ids, positions)
        last_indices = context.cu_seqlens_q[1:] - 1

        # Sample first — needed for shifted MTP token IDs
        logits = self.model.compute_logits(hidden)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None

        # MTP KV cache update with shifted token IDs (matching sync path)
        eagle_async = self.config.eagle_async
        if self.use_mtp:
            shifted_ids = input_ids.clone()
            shifted_ids[:-1] = input_ids[1:]
            if self.rank == 0:
                for i, idx in enumerate(last_indices.tolist()):
                    shifted_ids[idx] = token_ids[i]
            if self.tp_size > 1:
                dist.broadcast(shifted_ids, 0, group=self.tp_group)
            embeds = self.model.model.embed_tokens(shifted_ids)
            mtp_hidden = hidden if self._mtp_uses_unnormed else self.model.model.norm(hidden)
            self.model.model.mtp_layers[0](embeds, mtp_hidden, positions)

        # EAGLE async: skip local EAGLE KV prefill — draft GPU handles cache miss via JIT (cmd=0)

        for i, seq in enumerate(seqs):
            self.last_hidden[seq.seq_id] = hidden[last_indices[i]:last_indices[i]+1].clone()

            if self.rank == 0 and self.async_pg is not None:
                start = 0 if i == 0 else context.cu_seqlens_q[i].item()
                end = context.cu_seqlens_q[i + 1].item()
                n = end - start
                seq_positions = list(range(seq.num_cached_tokens, len(seq)))
                bt = seq.block_table

                # EAGLE: send original token IDs + normed hidden
                # MTP: send shifted token IDs + hidden
                send_ids = input_ids[start:end] if eagle_async else shifted_ids[start:end]

                self._send_cmd(1)
                meta = torch.tensor([seq.seq_id, n, len(bt)], dtype=torch.int64, device=d)
                dist.send(meta, dst=self.draft_rank, group=self.async_pg)
                dist.send(send_ids.contiguous(), dst=self.draft_rank, group=self.async_pg)
                dist.send(hidden[start:end].contiguous(), dst=self.draft_rank, group=self.async_pg)
                dist.send(torch.tensor(seq_positions, dtype=torch.int64, device=d),
                          dst=self.draft_rank, group=self.async_pg)
                dist.send(torch.tensor(bt, dtype=torch.int32, device=d),
                          dst=self.draft_rank, group=self.async_pg)
                ack = torch.zeros(1, dtype=torch.int64, device=d)
                dist.recv(ack, src=self.draft_rank, group=self.async_pg)

        reset_context()

        if self.rank == 0:
            for seq, tid in zip(seqs, token_ids):
                seq.recovery_token_id = tid
        return token_ids

    @torch.inference_mode()
    def _run_ssd_decode(self, seqs: list[Sequence]) -> list[list[int]]:
        """SSD decode: send spec request to draft via NCCL, verify with target."""
        k = self.num_speculative_tokens
        all_draft_tokens = []
        d = self.device.device_name

        # === Speculation phase ===
        # Check if draft has cached result (from last step's early_speculate).
        # On hit: NCCL to draft (fast). On miss: local MTP/EAGLE fallback.
        eagle_async = self.config.eagle_async
        if not eagle_async:
            mtp_layer = self.model.model.mtp_layers[0]
        if self.rank == 0 and self.async_pg is not None:
            if eagle_async:
                # Use locally pushed tree cache (received in llm_engine.step() after previous step)
                if not hasattr(self, '_local_tree_cache'):
                    self._local_tree_cache = {}
                    self._cache_hit = 0
                    self._cache_miss = 0
                for seq in seqs:
                    local_cache = self._local_tree_cache.get(seq.seq_id)
                    if local_cache is not None:
                        keys, tokens = local_cache
                        acc_len = seq.last_accepted_len
                        if tokens.shape[1] >= k:
                            # Tree mode: direct lookup returns all K tokens
                            req = torch.tensor([[seq.seq_id, acc_len, seq.last_token]],
                                                dtype=torch.int64, device=d)
                            match = torch.all(req == keys, dim=1)
                            if match.any():
                                idx = match.float().argmax().item()
                                all_draft_tokens.append(tokens[idx, :k].tolist())
                                self._cache_hit += 1
                                continue
                        else:
                            # Chain mode: K chained lookups
                            draft_tokens = []
                            cur_token = seq.last_token
                            hit = False
                            for step in range(k):
                                req = torch.tensor([[seq.seq_id, acc_len + step, cur_token]],
                                                    dtype=torch.int64, device=d)
                                match = torch.all(req == keys, dim=1)
                                if match.any():
                                    idx = match.float().argmax().item()
                                    cur_token = tokens[idx, 0].item()
                                    draft_tokens.append(cur_token)
                                    if step == 0:
                                        hit = True
                                else:
                                    draft_tokens.extend([0] * (k - step))
                                    break
                            if hit:
                                all_draft_tokens.append(draft_tokens)
                                self._cache_hit += 1
                                continue
                    # Cache miss: JIT on draft (only happens on first step or rare miss)
                    self._cache_miss += 1
                    self._send_cmd(0)
                    meta = torch.tensor([seq.seq_id, getattr(seq, 'last_accepted_len', 0),
                                          seq.last_token, len(seq), len(seq.block_table)],
                                         dtype=torch.int64, device=d)
                    dist.send(meta, dst=self.draft_rank, group=self.async_pg)
                    dist.send(self.last_hidden[seq.seq_id].squeeze(0).contiguous(),
                               dst=self.draft_rank, group=self.async_pg)
                    dist.send(torch.tensor(seq.block_table, dtype=torch.int32, device=d),
                               dst=self.draft_rank, group=self.async_pg)
                    jit_buf = torch.zeros(k, dtype=torch.int64, device=d)
                    dist.recv(jit_buf, src=self.draft_rank, group=self.async_pg)
                    all_draft_tokens.append(jit_buf.tolist())
            else:
                # MTP path: batched NCCL lookup — merged cmd+ns+lookup into 1 send
                if not hasattr(self, '_cache_hit'):
                    self._cache_hit = 0
                    self._cache_miss = 0
                n = len(seqs)
                self._send_cmd(6, n)
                # Pack lookup data into pre-allocated buffer
                buf = self._lookup_buf[:n * 3]
                for i, seq in enumerate(seqs):
                    buf[i*3] = seq.seq_id
                    buf[i*3 + 1] = seq.last_accepted_len
                    buf[i*3 + 2] = seq.last_token
                dist.send(buf, dst=self.draft_rank, group=self.async_pg)
                result_buf = self._result_buf[:n * k]
                result_buf.zero_()
                dist.recv(result_buf, src=self.draft_rank, group=self.async_pg)
                result_list = result_buf.tolist()
                for i in range(len(seqs)):
                    tokens = result_list[i*k:(i+1)*k]
                    if tokens[0] != 0:
                        all_draft_tokens.append(tokens)
                        self._cache_hit += 1
                    else:
                        # Cache miss: local MTP fallback
                        self._cache_miss += 1
                        seq = seqs[i]
                        embed_tokens = self.model.model.embed_tokens
                        target_hidden = self.last_hidden[seq.seq_id]
                        draft_tokens = []
                        cur_token_id = seq.last_token
                        cur_pos = len(seq) - 1
                        for step in range(k):
                            input_id = self.device.to_device(
                                torch.tensor([cur_token_id], dtype=torch.int64, pin_memory=True))
                            p = self.device.to_device(
                                torch.tensor([cur_pos], dtype=torch.int64, pin_memory=True))
                            bi = cur_pos // self.block_size
                            bo = cur_pos % self.block_size
                            slot = seq.block_table[bi] * self.block_size + bo
                            slot_map = self.device.to_device(
                                torch.tensor([slot], dtype=torch.int32, pin_memory=True))
                            bt = seq.block_table
                            if self.config.num_sink_blocks > 0:
                                bt = list(range(self.config.num_sink_blocks)) + bt
                            sink_ctx_ssd = self.config.num_sink_blocks * self.block_size
                            block_table_t = self.device.to_device(
                                torch.tensor([bt], dtype=torch.int32, pin_memory=True))
                            context_lens = self.device.to_device(
                                torch.tensor([cur_pos + 1 + sink_ctx_ssd], dtype=torch.int32, pin_memory=True))
                            set_context(False, slot_mapping=slot_map, context_lens=context_lens, block_tables=block_table_t)
                            token_embeds = embed_tokens(input_id)
                            mtp_normed, mtp_prenorm = mtp_layer(token_embeds, target_hidden, p)
                            reset_context()
                            draft_logits = self.model.compute_logits_all(mtp_normed)
                            d_token = draft_logits.argmax(dim=-1).item()
                            draft_tokens.append(d_token)
                            target_hidden = mtp_normed
                            cur_token_id = d_token
                            cur_pos += 1
                        all_draft_tokens.append(draft_tokens)

        # Broadcast draft tokens to TP workers
        if self.tp_size > 1:
            if self.rank != 0:
                all_draft_tokens = [[0] * k for _ in seqs]
            for i, seq in enumerate(seqs):
                for j in range(k):
                    d_tensor = torch.tensor([all_draft_tokens[i][j]], dtype=torch.int64, device=d)
                    dist.broadcast(d_tensor, 0, group=self.tp_group)
                    if self.rank != 0:
                        all_draft_tokens[i][j] = d_tensor.item()

        # === Verify phase (layer-by-layer for early hidden extraction) ===
        verify_ids, verify_pos = self._prepare_verify(seqs, all_draft_tokens)
        model_inner = self.model.model
        early_layer = len(model_inner.layers) - self.config.ssd_early_layers

        hidden_states = model_inner.embed_tokens(verify_ids)
        residual = None
        _async_send_handles = []
        for i, layer in enumerate(model_inner.layers):
            hidden_states, residual = layer(verify_pos, hidden_states, residual)
            if i == early_layer and self.rank == 0 and self.async_pg is not None:
                early_hidden_all = (hidden_states + residual).clone()
                # Non-blocking sends: overlap with remaining layers
                num_seqs_batch = len(seqs)
                self._cmd_buf[0] = 5
                self._cmd_buf[1] = num_seqs_batch
                meta_list = []
                bt_list = []
                pos_list = []
                for seq, dt in zip(seqs, all_draft_tokens):
                    nv = len(dt) + 1
                    bt = seq.block_table
                    seq_start_pos = len(seq) - 1
                    meta_list.extend([seq.seq_id, nv, len(seq) + len(dt), len(bt), seq_start_pos])
                    bt_list.extend(bt)
                    pos_list.extend(range(seq_start_pos, seq_start_pos + nv))
                meta_t = torch.tensor(meta_list, dtype=torch.int64, device=d)
                bt_t = torch.tensor(bt_list, dtype=torch.int32, device=d)
                pos_t = torch.tensor(pos_list, dtype=torch.int64, device=d)
                _async_send_handles = [
                    dist.isend(self._cmd_buf, dst=self.draft_rank, group=self.async_pg),
                    dist.isend(meta_t, dst=self.draft_rank, group=self.async_pg),
                    dist.isend(early_hidden_all, dst=self.draft_rank, group=self.async_pg),
                    dist.isend(bt_t, dst=self.draft_rank, group=self.async_pg),
                    dist.isend(pos_t, dst=self.draft_rank, group=self.async_pg),
                ]

        hidden_states, residual = model_inner.norm(hidden_states, residual)

        # Wait for async sends to complete before proceeding
        for h in _async_send_handles:
            h.wait()
        hidden = residual  # unnormed, for MTP last_hidden

        target_logits = self.model.compute_logits_all(hidden_states)
        reset_context()

        # === Accept phase ===
        if self.rank == 0:
            all_accepted = []
            offset = 0
            for seq, draft_tokens in zip(seqs, all_draft_tokens):
                num_verify = len(draft_tokens) + 1
                seq_logits = target_logits[offset:offset + num_verify]
                target_predicted = seq_logits.argmax(dim=-1)
                accepted = []
                for j in range(k):
                    if target_predicted[j].item() == draft_tokens[j]:
                        accepted.append(draft_tokens[j])
                    else:
                        accepted.append(target_predicted[j].item())
                        break
                else:
                    accepted.append(target_predicted[k].item())

                seq.last_accepted_len = len(accepted) - 1
                seq.recovery_token_id = accepted[-1]
                accepted_idx = offset + len(accepted) - 1
                # EAGLE needs normed hidden; MTP needs unnormed (residual)
                save_h = hidden_states if eagle_async else hidden
                self.last_hidden[seq.seq_id] = save_h[accepted_idx:accepted_idx+1].clone()
                all_accepted.append(accepted)
                offset += num_verify
        else:
            all_accepted = [[] for _ in seqs]

        # Skip MTP KV cache update in SSD mode — cache hit rate ~99% makes it unnecessary
        # (MTP KV only needed for cache miss local fallback, which is rare)

        # Mark push as pending — will be received at the beginning of next step
        if eagle_async and self.rank == 0 and self.async_pg is not None:
            self._push_pending = True

        if self.rank == 0:
            return all_accepted
        return None


    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size + config.num_sink_blocks
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
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size + config.num_sink_blocks
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
