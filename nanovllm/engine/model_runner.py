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


def _push_max_buf_size(config):
    """Worst-case push tree-cache buffer size in int64 elements.
    Both target and draft compute this identically — eliminates the size
    handshake and enables a fixed pre-allocated buffer.

    Layout: [num_seqs, (sid, n_e, K_a, keys[n_e*3], tokens[n_e*K_a]) per seq]
    """
    K = config.num_speculative_tokens
    F = config.async_fan_out
    nv = K + 1
    n_e_max = nv * F
    K_a_max = K if (config.latent_tree_decode and K > 1) else 1
    per_seq_max = 3 + n_e_max * 3 + n_e_max * K_a_max
    return 1 + config.max_num_seqs * per_seq_max


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
    if model_type == 'qwen2':
        from nanovllm.models.qwen2 import Qwen2ForCausalLM
        return Qwen2ForCausalLM
    # Qwen3 and other compatible architectures
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
        # Second TP group (same ranks, distinct NCCL communicator) for the
        # early-MTP side stream so its allreduces can run concurrently with
        # target's tail-layer allreduces. We mark it high-priority so the
        # underlying NCCL stream is created with high CUDA stream priority
        # (PG is_high_priority_stream=True). Combined with the env var
        # TORCH_NCCL_HIGH_PRIORITY=1, this gives the MTP collective stream
        # actual scheduling priority on-GPU and a separate ncclStream from
        # tp_group's internal stream.
        if self.world_size > 1 and getattr(config, 'mtp_early_layers', -1) < -1:
            try:
                from torch.distributed.distributed_c10d import ProcessGroupNCCL
                _opts = ProcessGroupNCCL.Options()
                _opts.is_high_priority_stream = True
                self.tp_group_mtp = dist.new_group(tp_ranks, pg_options=_opts)
            except Exception:
                self.tp_group_mtp = dist.new_group(tp_ranks)
        else:
            self.tp_group_mtp = None
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
                # EAGLE draft's config usually omits rope_scaling; inherit from target.
                # Also override if target has a real scaling type (e.g. 'llama3')
                # while draft's only carries the HF-synthesized default ({'rope_type':'default'}).
                tgt_scaling = getattr(hf_config, 'rope_scaling', None)
                drf_scaling = getattr(eagle_cfg, 'rope_scaling', None)
                tgt_type = (tgt_scaling or {}).get('rope_type', 'default')
                drf_type = (drf_scaling or {}).get('rope_type', 'default')
                if tgt_type != 'default' and drf_type == 'default':
                    eagle_cfg.rope_scaling = tgt_scaling
                self.draft_model = EAGLEModel(
                    eagle_cfg,
                    embed_tokens=self.model.model.embed_tokens,
                    lm_head=self.model.lm_head,
                    tp_group=self.tp_group, tp_size=self.tp_size,
                )
            load_model(self.draft_model, config.draft_model)
        if self.speculative:
            # Per-seq caches indexed by seq_id. llm_engine.step() evicts the
            # seq_id when the sequence finishes (see #6 — unbounded growth
            # caused a CUDA illegal-access crash on Vicuna EAGLE-1).
            #   last_hidden       — target's final hidden at last prefill pos
            #                       (MTP path; EAGLE no longer reads it)
            #   _draft_prev_hidden — draft's prefill output at last position;
            #                       fed as chain-step-0 `hidden_states`
            self.last_hidden = {}
            self._draft_prev_hidden = {}
            # _pending_draft: next step's MTP draft tokens (post-verify flow).
            # Populated at prefill end and after each decode step. Mirrors
            # vLLM's propose(): MTP runs on the target's fresh verify hidden
            # to produce the draft used in the NEXT verify window.
            self._pending_draft = {}
        self.sampler = Sampler()
        # If MTP early-layer parallel mode is on, rewire MTP submodules to use
        # tp_group_mtp so their allreduces share a separate NCCL communicator
        # from target's, allowing real overlap on the side stream.
        if (
            self.use_mtp
            and self.tp_group_mtp is not None
            and hasattr(self.model.model, 'mtp_layers')
        ):
            for mtp in self.model.model.mtp_layers:
                for sub in mtp.modules():
                    if hasattr(sub, 'tp_group') and sub.tp_group is self.tp_group:
                        sub.tp_group = self.tp_group_mtp
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

        # Pre-allocate NCCL buffers for Latent SD communication
        if self.draft_async and rank == 0:
            d = self.device.device_name
            max_seqs = config.max_num_seqs
            K = config.num_speculative_tokens
            self._cmd_buf = torch.zeros(4, dtype=torch.int64, device=d)
            # Spec phase pre-allocated buffers
            self._lookup_buf = torch.zeros(1 + max_seqs * 3, dtype=torch.int64, device=d)
            self._result_buf = torch.zeros(max_seqs * K, dtype=torch.int64, device=d)
            # Pre-allocated payload buf for early_speculate send
            max_nv = (K + 1) * max_seqs
            max_bt = (config.max_model_len // self.block_size + 2) * max_seqs
            hidden_mult = 3 if getattr(config, 'eagle3', False) else 1
            hidden_i64_size = max_nv * config.hf_config.hidden_size * hidden_mult * config.hf_config.torch_dtype.itemsize // 8
            self._payload_buf = torch.zeros(1 + max_seqs * 5 + max_bt + max_nv + hidden_i64_size, dtype=torch.int64, device=d)
            # Pre-allocated push tree-cache recv buf (skip size handshake).
            self._push_buf_max_size = _push_max_buf_size(config)
            self._push_buf_max = torch.zeros(self._push_buf_max_size, dtype=torch.int64, device=d)
            # Pre-allocated TP broadcast buf for draft tokens (per-step).
            # Layout: [len_seq0, t0..t_{k-1}, len_seq1, t0..t_{k-1}, ...]
            if self.tp_size > 1:
                self._draft_bcast_buf = torch.zeros(max_seqs * (1 + K), dtype=torch.int64, device=d)
        elif self.draft_async and self.tp_size > 1:
            # Other TP ranks also need the broadcast buf
            d = self.device.device_name
            K = config.num_speculative_tokens
            max_seqs = config.max_num_seqs
            self._draft_bcast_buf = torch.zeros(max_seqs * (1 + K), dtype=torch.int64, device=d)

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
        # During warmup, skip Latent SD communication and EAGLE KV prefill
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
                    return self._run_latent_prefill_with_hidden(seqs)
                return self._run_prefill_with_hidden(seqs)
            elif self.draft_async:
                return self._run_latent_decode(seqs)
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
        """Get auxiliary layer indices for EAGLE3.
        vLLM captures the INPUT to layer i, which equals the OUTPUT of layer i-1.
        So we extract after layers (1, N//2-1, N-4)."""
        N = self.config.hf_config.num_hidden_layers
        return (1, N // 2 - 1, N - 4)

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
        prenorm = hidden_states + residual  # prenorm for chain step hidden
        hidden_states, _ = model_inner.norm(hidden_states, residual)
        # Concatenate aux hiddens in order
        aux_concat = torch.cat([aux_hiddens[l] for l in aux_layers], dim=-1)
        return hidden_states, aux_concat, prenorm

    @torch.inference_mode()
    def _run_prefill_with_hidden(self, seqs: list[Sequence]) -> list[int]:
        """Prefill that also saves last hidden states for speculative draft."""
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        context = get_context()
        is_eagle3 = getattr(self, '_is_eagle3', False)
        if is_eagle3:
            model_out, aux_concat, eagle3_prenorm = self._run_target_with_aux(input_ids, positions)
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
            mtp_out = self.model.model.mtp_layers[0](embeds, mtp_hidden, positions)
            mtp_normed = mtp_out[0] if isinstance(mtp_out, tuple) else mtp_out
            # Capture first draft per seq for next step's verify (post-verify flow).
            if not getattr(self, '_warmup', False):
                self._stash_pending_draft(seqs, mtp_normed, last_indices)
        # EAGLE uses normed hidden (trained with model.model() output which is post-RMSNorm);
        # MiMo MTP uses unnormed (model returns it directly)
        save_hidden = hidden
        # Populate EAGLE KV cache during prefill so draft attention has valid context.
        # EAGLE training pairs (token_{i+1}, aux_i) → predict token_{i+2}; we must
        # shift input_ids by one (drop first token, append sampled bonus at last pos)
        # before feeding the draft, otherwise draft KV at positions 0..L-1 carries
        # off-by-one token embeddings and chain-step attention reads wrong context.
        if hasattr(self, 'draft_model') and not getattr(self, '_warmup', False):
            shifted_ids = input_ids.clone()
            shifted_ids[:-1] = input_ids[1:]
            if self.rank == 0:
                for i, idx in enumerate(last_indices.tolist()):
                    shifted_ids[idx] = token_ids[i]
            if self.tp_size > 1:
                dist.broadcast(shifted_ids, 0, group=self.tp_group)
            ctx = get_context()
            set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                        ctx.max_seqlen_k, ctx.slot_mapping, None, None)
            if is_eagle3:
                draft_prefill_out = self.draft_model(shifted_ids, positions, hidden_normed, aux_hiddens=aux_concat)
            else:
                draft_prefill_out = self.draft_model(shifted_ids, positions, hidden_normed)
            set_context(True, ctx.cu_seqlens_q, ctx.cu_seqlens_k, ctx.max_seqlen_q,
                        ctx.max_seqlen_k, ctx.slot_mapping, ctx.context_lens, ctx.block_tables)
        for i, seq in enumerate(seqs):
            self.last_hidden[seq.seq_id] = save_hidden[last_indices[i]:last_indices[i]+1].clone()
            if hasattr(self, 'draft_model') and not getattr(self, '_warmup', False):
                self._draft_prev_hidden[seq.seq_id] = draft_prefill_out[last_indices[i]:last_indices[i]+1].clone()
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

    def _sample_draft_tokens(self, draft_hidden, n_seqs, d, is_eagle3):
        """Sample next draft token from draft hidden (EAGLE-3 applies d2t)."""
        logits = self.draft_model.compute_logits(draft_hidden)
        if self.rank == 0:
            if is_eagle3:
                draft_ids = logits.argmax(dim=-1)
                tokens = self.draft_model.d2t[draft_ids].tolist()
            else:
                tokens = logits.argmax(dim=-1).tolist()
        else:
            tokens = [0] * n_seqs
        if self.tp_size > 1:
            t = torch.tensor(tokens, dtype=torch.int64, device=d)
            dist.broadcast(t, 0, group=self.tp_group)
            tokens = t.tolist()
        return tokens

    @torch.inference_mode()
    def _run_speculative_decode(self, seqs: list[Sequence]) -> list[list[int]]:
        """Draft k tokens with EAGLE, then verify with target model.

        Follows Spec-Bench / vLLM's EAGLE chain convention:
        - d_0 is sampled directly from the draft's prefill-last hidden via
          compute_logits — no extra forward (the prefill forward already wrote
          draft KV at position L-1 whose output predicts token[L+1] per the
          -1 shift the draft was trained on).
        - Chain runs K-1 forwards producing d_1..d_{K-1}. At position L+i the
          input pair is (d_i, draft_prev_hidden) — the draft's own previous
          output hidden stands in for aux[L+i] (which the target hasn't yet
          produced). Feeding target's stale aux[L-1] at position L (the old
          nano path) misaligned the draft vs its training shift.
        """
        k = self.num_speculative_tokens
        d = self.device.device_name
        n_seqs = len(seqs)
        is_eagle3 = getattr(self, '_is_eagle3', False)
        all_draft_tokens = [[] for _ in seqs]

        # === d_0 from draft prefill last hidden (no forward) ===
        draft_prev_hidden = torch.cat(
            [self._draft_prev_hidden[seq.seq_id] for seq in seqs], dim=0)
        d0_tokens = self._sample_draft_tokens(draft_prev_hidden, n_seqs, d, is_eagle3)
        for i in range(n_seqs):
            all_draft_tokens[i].append(d0_tokens[i])
        cur_tokens = d0_tokens
        cur_hiddens = draft_prev_hidden

        # === Chain K-1 forwards for d_1..d_{K-1} ===
        for step in range(k - 1):
            input_ids = self.device.to_device(torch.tensor(cur_tokens, dtype=torch.int64, pin_memory=True))
            positions = self.device.to_device(torch.tensor([len(seq) - 1 + step for seq in seqs], dtype=torch.int64, pin_memory=True))
            slots = []
            ctx_lens = []
            for i, seq in enumerate(seqs):
                p = len(seq) - 1 + step
                bi = p // self.block_size
                bo = p % self.block_size
                slots.append(seq.block_table[bi] * self.block_size + bo)
                ctx_lens.append(p + 1)
            slot_map = self.device.to_device(torch.tensor(slots, dtype=torch.int32, pin_memory=True))
            ctx_lens_t = self.device.to_device(torch.tensor(ctx_lens, dtype=torch.int32, pin_memory=True))
            set_context(False, slot_mapping=slot_map, context_lens=ctx_lens_t,
                        block_tables=self.prepare_block_tables(seqs))
            draft_hidden = self.draft_model(input_ids, positions, cur_hiddens)
            reset_context()
            tokens = self._sample_draft_tokens(draft_hidden, n_seqs, d, is_eagle3)
            for i in range(n_seqs):
                all_draft_tokens[i].append(tokens[i])
            cur_tokens = tokens
            cur_hiddens = draft_hidden

        # === Verify phase (target model, prefill-like) ===
        input_ids, positions = self._prepare_verify(seqs, all_draft_tokens)
        is_eagle3 = getattr(self, '_is_eagle3', False)
        if is_eagle3:
            hidden, aux_concat, eagle3_prenorm = self._run_target_with_aux(input_ids, positions)
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
                predicted = seq_logits.argmax(dim=-1).tolist()
                accepted = []
                for j in range(k):
                    if predicted[j] == draft_tokens[j]:
                        accepted.append(draft_tokens[j])
                    else:
                        accepted.append(predicted[j])
                        break
                else:
                    accepted.append(predicted[k])
                # Save hidden at accepted position
                accepted_idx = offset + len(accepted) - 1
                self.last_hidden[seq.seq_id] = hidden[accepted_idx:accepted_idx+1].clone()
                all_accepted.append(accepted)
                offset += num_verify

            # === EAGLE KV refresh after accept (Spec-Bench/vLLM convention) ===
            # Chain step wrote positions L..L+K-2 using (d_i, draft_prev_out)
            # pairs — draft's own hidden stood in for aux. After verify we
            # have real target aux[L..L+K], so replay draft at accepted
            # positions L..L+m-1 (m=num_accepted) using the -1 shift pair
            # (token[p+1] = accepted[j], aux[p] = aux[L+j]) — the same pair
            # prefill uses. The last replay's output hidden is the draft's
            # hidden at position L+m-1, which is exactly what next round's
            # chain step 0 needs as `draft_prev_hidden` (prev-slot draft
            # output) at its position L_new-1 = L+m-1.
            if hasattr(self, 'draft_model'):
                d = self.device.device_name
                offset = 0
                for seq_idx, seq in enumerate(seqs):
                    num_accepted = len(all_accepted[seq_idx])
                    num_verify = len(all_draft_tokens[seq_idx]) + 1
                    seq_start_pos = len(seq) - 1
                    last_draft_out = None
                    for j in range(num_accepted):
                        acc_pos = seq_start_pos + j
                        tok_id = all_accepted[seq_idx][j]
                        if is_eagle3:
                            target_h = self.draft_model.combine_hidden_states(
                                aux_concat[offset + j:offset + j + 1])
                        else:
                            target_h = hidden[offset + j:offset + j + 1]
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
                        last_draft_out = self.draft_model(inp, p, target_h)
                        reset_context()
                    if last_draft_out is not None:
                        self._draft_prev_hidden[seq.seq_id] = last_draft_out.clone()
                    offset += num_verify

            return all_accepted
        else:
            return None

    def _stash_pending_draft(self, seqs, mtp_normed, last_indices):
        """Compute argmax draft from mtp_normed at last_indices and cache per seq.
        Matches vLLM's propose(): MTP hidden → shared_head.head linear → argmax.
        Gathers logits across TP (the shared_head.head weight is col-parallel),
        broadcasts the chosen token to all ranks, then writes _pending_draft[seq_id].
        """
        d = self.device.device_name
        n_seqs = len(seqs)
        mtp_layer = self.model.model.mtp_layers[0]
        last_normed = mtp_normed[last_indices] if isinstance(last_indices, torch.Tensor) \
            else mtp_normed[torch.tensor(last_indices, dtype=torch.int64, device=d)]
        draft_logits = F.linear(last_normed, mtp_layer.shared_head.head.weight)
        if self.tp_size > 1:
            all_l = [torch.empty_like(draft_logits) for _ in range(self.tp_size)] if self.rank == 0 else None
            dist.gather(draft_logits, all_l, 0, group=self.tp_group)
            draft_logits = torch.cat(all_l, -1) if self.rank == 0 else None
        if self.rank == 0:
            drafts = draft_logits.argmax(dim=-1).tolist()
        else:
            drafts = [0] * n_seqs
        if self.tp_size > 1:
            t = torch.tensor(drafts, dtype=torch.int64, device=d)
            dist.broadcast(t, 0, group=self.tp_group)
            drafts = t.tolist()
        for i, seq in enumerate(seqs):
            self._pending_draft[seq.seq_id] = [drafts[i]]

    def _argmax_logits_tp(self, normed: torch.Tensor) -> list[int]:
        """Apply lm_head to (n_seqs, H), gather across TP, argmax → per-seq token.
        Returns list[int] of length n_seqs on every rank (broadcast result)."""
        n_seqs = normed.shape[0]
        d = self.device.device_name
        mtp_layer = self.model.model.mtp_layers[0]
        logits = F.linear(normed, mtp_layer.shared_head.head.weight)
        if self.tp_size > 1:
            all_l = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.rank == 0 else None
            dist.gather(logits, all_l, 0, group=self.tp_group)
            logits = torch.cat(all_l, -1) if self.rank == 0 else None
        if self.rank == 0:
            tokens = logits.argmax(dim=-1).tolist()
        else:
            tokens = [0] * n_seqs
        if self.tp_size > 1:
            t = torch.tensor(tokens, dtype=torch.int64, device=d)
            dist.broadcast(t, 0, group=self.tp_group)
            tokens = t.tolist()
        return tokens

    def _stash_pending_draft_chain(
        self,
        seqs: list[Sequence],
        mtp_normed_postverify: torch.Tensor,
        last_indices: torch.Tensor,
        n_acc_list: list[int],
        K: int,
    ) -> None:
        """Chain MTP K times to produce K drafts per seq, then stash."""
        d = self.device.device_name
        n_seqs = len(seqs)
        mtp_layer = self.model.model.mtp_layers[0]
        embed_tokens = self.model.model.embed_tokens

        # Step 0 hidden = post-verify MTP output at each seq's last accepted position.
        if isinstance(last_indices, torch.Tensor):
            cur_hidden = mtp_normed_postverify[last_indices]
        else:
            cur_hidden = mtp_normed_postverify[
                torch.tensor(last_indices, dtype=torch.int64, device=d)
            ]

        drafts_per_seq = [[] for _ in range(n_seqs)]
        for k in range(K):
            d_k = self._argmax_logits_tp(cur_hidden)
            for i in range(n_seqs):
                drafts_per_seq[i].append(d_k[i])
            if k == K - 1:
                break
            # Chain step k+1: MTP at position L+n_acc+k for each seq with input=embed(d_k)
            chain_positions_py = []
            chain_slot_py = []
            chain_cu_q = [0]
            chain_cu_k = [0]
            max_chain_k = 0
            for i in range(n_seqs):
                # next draft's absolute position is (len(seq) - 1 + n_acc + k + 1 - 1) = len(seq) + n_acc + k - 1
                # but we want MTP at position p such that its output predicts next token.
                # post-verify last MTP position = L + n_acc - 1 (where L = len(seq) - 1).
                # chain step (k=1) predicts at position p = L + n_acc + 0 (= len(seq) + n_acc - 1).
                pos = len(seqs[i]) - 1 + n_acc_list[i] + k
                chain_positions_py.append(pos)
                bi = pos // self.block_size
                bo = pos % self.block_size
                chain_slot_py.append(seqs[i].block_table[bi] * self.block_size + bo)
                chain_cu_q_inc = 1
                chain_cu_q_last = chain_cu_q[-1]
                chain_cu_q.append(chain_cu_q_last + chain_cu_q_inc)
                seqlen_k = pos  # context covers 0..pos-1
                chain_cu_k.append(chain_cu_k[-1] + seqlen_k)
                max_chain_k = max(max_chain_k, seqlen_k)
            chain_pos_t = self.device.to_device(
                torch.tensor(chain_positions_py, dtype=torch.int64, pin_memory=True))
            chain_input_ids_py = [drafts_per_seq[i][-1] for i in range(n_seqs)]
            chain_input_t = self.device.to_device(
                torch.tensor(chain_input_ids_py, dtype=torch.int64, pin_memory=True))
            chain_slot_t = self.device.to_device(
                torch.tensor(chain_slot_py, dtype=torch.int32, pin_memory=True))
            chain_cu_q_t = self.device.to_device(
                torch.tensor(chain_cu_q, dtype=torch.int32, pin_memory=True))
            chain_cu_k_t = self.device.to_device(
                torch.tensor(chain_cu_k, dtype=torch.int32, pin_memory=True))
            chain_block_tables = self.prepare_block_tables(seqs)
            set_context(True, chain_cu_q_t, chain_cu_k_t, 1, max_chain_k,
                        chain_slot_t, None, chain_block_tables)
            chain_embeds = embed_tokens(chain_input_t)
            chain_out = mtp_layer(chain_embeds, cur_hidden, chain_pos_t)
            chain_normed = chain_out[0] if isinstance(chain_out, tuple) else chain_out
            cur_hidden = chain_normed
            reset_context()

        for i, seq in enumerate(seqs):
            self._pending_draft[seq.seq_id] = drafts_per_seq[i]

    @torch.inference_mode()
    def _run_mtp_decode(self, seqs: list[Sequence]) -> list[list[int]]:
        """MTP decode, vLLM-aligned post-verify draft flow.

        Per-step:
          1. Use the draft cached in _pending_draft (populated by prefill or the
             previous decode step). No live draft forward here.
          2. Run target verify on [last_token, d_0, ..., d_{k-1}] per seq.
          3. Greedy-accept by comparing target argmax against drafts.
          4. Run MTP on accepted verify positions with shifted token IDs and
             the target's UNNORMED hidden. MTP output at each seq's last
             accepted position becomes the draft for NEXT step's verify.

        This matches vLLM's drafter.propose() call shape: MTP sees (target_hidden[p],
        embed(token[p+1])) and predicts token[p+2], using the freshly-committed
        sequence prefix. The prior nano flow drafted BEFORE verify using the stale
        prefill `last_hidden` → lost ~50pp accept rate vs vLLM on PanGu.
        """
        if getattr(self.config, 'mtp_early_layers', -1) < -1:
            return self._run_mtp_decode_early(seqs)
        k = self.num_speculative_tokens
        d = self.device.device_name
        n_seqs = len(seqs)
        mtp_layer = self.model.model.mtp_layers[0]
        embed_tokens = self.model.model.embed_tokens

        # === Draft tokens come from _pending_draft (seeded at prefill end) ===
        all_draft_tokens = [list(self._pending_draft[seq.seq_id]) for seq in seqs]

        # === Verify ===
        verify_ids, verify_pos = self._prepare_verify(seqs, all_draft_tokens)
        hidden = self.model(verify_ids, verify_pos)  # UNNORMED residual out
        hidden_normed = self.model.model.norm(hidden)
        target_logits = self.model.compute_logits_all(hidden_normed)
        reset_context()

        # === Accept (rank 0) ===
        if self.rank == 0:
            all_accepted = []
            offset = 0
            for seq, draft_tokens in zip(seqs, all_draft_tokens):
                num_verify = len(draft_tokens) + 1
                seq_logits = target_logits[offset:offset + num_verify]
                predicted = seq_logits.argmax(dim=-1).tolist()
                accepted = []
                nd = len(draft_tokens)
                for j in range(nd):
                    if predicted[j] == draft_tokens[j]:
                        accepted.append(draft_tokens[j])
                    else:
                        accepted.append(predicted[j])
                        break
                else:
                    accepted.append(predicted[nd])
                all_accepted.append(accepted)
                offset += num_verify
        else:
            all_accepted = [[] for _ in seqs]

        # === Broadcast accepted tokens (TP) ===
        if self.tp_size > 1:
            n_acc_t = torch.tensor(
                [len(a) for a in all_accepted] if self.rank == 0 else [0] * n_seqs,
                dtype=torch.int64, device=d)
            dist.broadcast(n_acc_t, 0, group=self.tp_group)
            n_acc_list = n_acc_t.tolist()
            if self.rank != 0:
                all_accepted = [[0] * n_acc_list[i] for i in range(n_seqs)]
            for i in range(n_seqs):
                if n_acc_list[i] > 0:
                    acc_t = torch.tensor(all_accepted[i], dtype=torch.int64, device=d)
                    dist.broadcast(acc_t, 0, group=self.tp_group)
                    if self.rank != 0:
                        all_accepted[i] = acc_t.tolist()
        else:
            n_acc_list = [len(a) for a in all_accepted]

        # === Post-verify MTP: generate next step's draft per seq ===
        # Ragged batch with `num_accepted_i` positions per seq.
        # At verify pos p (= L + j), input_id = accepted[j] (shifted by 1: tok at p+1).
        mtp_positions_py = []
        mtp_input_ids_py = []
        mtp_hidden_idx_py = []
        mtp_slot_py = []
        mtp_cu_q = [0]
        mtp_cu_k = [0]
        mtp_last_idx_py = []
        max_q = 0
        max_k = 0

        verify_offset = 0
        for i, seq in enumerate(seqs):
            num_verify = len(self._pending_draft[seq.seq_id]) + 1  # K+1
            n_acc = n_acc_list[i]
            L = len(seq) - 1  # position of last_token at verify time
            for j in range(n_acc):
                pos = L + j
                mtp_positions_py.append(pos)
                mtp_input_ids_py.append(all_accepted[i][j])  # shift: tok at pos+1
                mtp_hidden_idx_py.append(verify_offset + j)
                bi = pos // self.block_size
                bo = pos % self.block_size
                mtp_slot_py.append(seq.block_table[bi] * self.block_size + bo)
            mtp_cu_q.append(mtp_cu_q[-1] + n_acc)
            seqlen_k = L + n_acc  # context covers 0..L+n_acc-1
            mtp_cu_k.append(mtp_cu_k[-1] + seqlen_k)
            max_q = max(max_q, n_acc)
            max_k = max(max_k, seqlen_k)
            mtp_last_idx_py.append(mtp_cu_q[-1] - 1)
            verify_offset += num_verify

        mtp_positions = self.device.to_device(
            torch.tensor(mtp_positions_py, dtype=torch.int64, pin_memory=True))
        mtp_input_ids = self.device.to_device(
            torch.tensor(mtp_input_ids_py, dtype=torch.int64, pin_memory=True))
        mtp_hidden_idx = self.device.to_device(
            torch.tensor(mtp_hidden_idx_py, dtype=torch.int64, pin_memory=True))
        mtp_hidden_input = hidden.index_select(0, mtp_hidden_idx)
        mtp_slot = self.device.to_device(
            torch.tensor(mtp_slot_py, dtype=torch.int32, pin_memory=True))
        mtp_cu_q_t = self.device.to_device(
            torch.tensor(mtp_cu_q, dtype=torch.int32, pin_memory=True))
        mtp_cu_k_t = self.device.to_device(
            torch.tensor(mtp_cu_k, dtype=torch.int32, pin_memory=True))
        mtp_block_tables = self.prepare_block_tables(seqs)

        set_context(True, mtp_cu_q_t, mtp_cu_k_t, max_q, max_k,
                    mtp_slot, None, mtp_block_tables)
        mtp_embeds = embed_tokens(mtp_input_ids)
        mtp_out = mtp_layer(mtp_embeds, mtp_hidden_input, mtp_positions)
        mtp_normed = mtp_out[0] if isinstance(mtp_out, tuple) else mtp_out
        reset_context()

        last_idx_t = self.device.to_device(
            torch.tensor(mtp_last_idx_py, dtype=torch.int64, pin_memory=True))
        if k > 1:
            # Chain MTP K times to produce K drafts for next-step verify.
            self._stash_pending_draft_chain(seqs, mtp_normed, last_idx_t, n_acc_list, k)
        else:
            self._stash_pending_draft(seqs, mtp_normed, last_idx_t)

        # Update last_hidden (unused by new flow, kept for external readers) ---
        if self.rank == 0:
            offset = 0
            for seq_idx, seq in enumerate(seqs):
                n_acc = n_acc_list[seq_idx]
                num_verify = len(all_draft_tokens[seq_idx]) + 1
                accepted_idx = offset + n_acc - 1 if n_acc > 0 else offset
                self.last_hidden[seq.seq_id] = hidden[accepted_idx:accepted_idx + 1].clone()
                offset += num_verify

        return all_accepted if self.rank == 0 else None

    @torch.inference_mode()
    def _run_mtp_decode_early(self, seqs: list[Sequence]) -> list[list[int]]:
        """Early-hidden sync MTP variant (single-GPU parallelism).

        Captures pre-norm hidden at layer N + mtp_early_layers (e.g., -4 -> N-4),
        launches the post-verify MTP forward on a side CUDA stream so it overlaps
        with the target's remaining |K|-1 layers + norm + lm_head, then skips the
        usual post-verify MTP loop. Assumes K=1 and that d_0 is accepted (next
        draft = MTP(early_hidden_at_pos_L, embed(d_0))). Accept rate may drop
        when bonus is taken (next-step draft was conditioned on the wrong
        position) but MTP latency is hidden behind the tail target layers.
        """
        early_K = self.config.mtp_early_layers
        assert early_K < -1, f"_run_mtp_decode_early called with K={early_K} >= -1"
        k = self.num_speculative_tokens
        assert k == 1, f"early-MTP prototype only supports num_speculative_tokens=1, got {k}"

        d = self.device.device_name
        n_seqs = len(seqs)
        mtp_layer = self.model.model.mtp_layers[0]
        embed_tokens = self.model.model.embed_tokens
        model_inner = self.model.model
        n_layers = len(model_inner.layers)
        early_layer_idx = n_layers + self.config.mtp_early_layers
        assert 0 <= early_layer_idx < n_layers - 1, (
            f"mtp_early_layers={self.config.mtp_early_layers} maps to layer "
            f"{early_layer_idx}; need 0 <= idx < {n_layers - 1}"
        )

        # Side stream for MTP overlap (lazy init).
        if not hasattr(self, '_mtp_side_stream') or self._mtp_side_stream is None:
            self._mtp_side_stream = torch.cuda.Stream()
        side_stream = self._mtp_side_stream
        main_stream = torch.cuda.current_stream()

        # === Draft tokens come from _pending_draft (seeded at prefill end) ===
        all_draft_tokens = [list(self._pending_draft[seq.seq_id]) for seq in seqs]

        # === Verify input (sets target verify context) ===
        verify_ids, verify_pos = self._prepare_verify(seqs, all_draft_tokens)
        verify_ctx = get_context()
        v_args = (verify_ctx.is_prefill, verify_ctx.cu_seqlens_q, verify_ctx.cu_seqlens_k,
                  verify_ctx.max_seqlen_q, verify_ctx.max_seqlen_k,
                  verify_ctx.slot_mapping, verify_ctx.context_lens, verify_ctx.block_tables)

        # === Pre-build MTP context (one position per seq: position L, input=d_0) ===
        mtp_input_ids_py = []
        mtp_positions_py = []
        mtp_slot_py = []
        mtp_hidden_idx_py = []
        mtp_cu_q = [0]
        mtp_cu_k = [0]
        mtp_last_idx_py = []
        max_mtp_q = 1
        max_mtp_k = 0

        verify_offset = 0
        for i, seq in enumerate(seqs):
            nv = len(all_draft_tokens[i]) + 1  # K+1
            L = len(seq) - 1
            d_0 = all_draft_tokens[i][0]
            mtp_positions_py.append(L)
            mtp_input_ids_py.append(d_0)
            mtp_hidden_idx_py.append(verify_offset)  # position L = first verify slot of this seq
            bi = L // self.block_size
            bo = L % self.block_size
            mtp_slot_py.append(seq.block_table[bi] * self.block_size + bo)
            mtp_cu_q.append(mtp_cu_q[-1] + 1)
            mtp_cu_k.append(mtp_cu_k[-1] + len(seq))  # MTP attention context length
            mtp_last_idx_py.append(mtp_cu_q[-1] - 1)
            verify_offset += nv

        max_mtp_k = max(mtp_cu_k[i + 1] - mtp_cu_k[i] for i in range(n_seqs))

        # Pack the MTP-side meta tensors. These are allocated on main_stream
        # but consumed on side_stream → record so the caching allocator
        # waits for side_stream before recycling them.
        mtp_positions_t = self.device.to_device(
            torch.tensor(mtp_positions_py, dtype=torch.int64, pin_memory=True))
        mtp_input_ids_t = self.device.to_device(
            torch.tensor(mtp_input_ids_py, dtype=torch.int64, pin_memory=True))
        mtp_hidden_idx_t = self.device.to_device(
            torch.tensor(mtp_hidden_idx_py, dtype=torch.int64, pin_memory=True))
        mtp_slot_t = self.device.to_device(
            torch.tensor(mtp_slot_py, dtype=torch.int32, pin_memory=True))
        mtp_cu_q_t = self.device.to_device(
            torch.tensor(mtp_cu_q, dtype=torch.int32, pin_memory=True))
        mtp_cu_k_t = self.device.to_device(
            torch.tensor(mtp_cu_k, dtype=torch.int32, pin_memory=True))
        mtp_block_tables = self.prepare_block_tables(seqs)
        last_idx_t = self.device.to_device(
            torch.tensor(mtp_last_idx_py, dtype=torch.int64, pin_memory=True))
        for _t in (mtp_positions_t, mtp_input_ids_t, mtp_hidden_idx_t,
                   mtp_slot_t, mtp_cu_q_t, mtp_cu_k_t, mtp_block_tables):
            _t.record_stream(side_stream)

        # === Manual layer iteration with mid-loop MTP launch on side stream ===
        # Cache events on self to avoid per-step Event allocation overhead.
        if not hasattr(self, '_mtp_early_event') or self._mtp_early_event is None:
            self._mtp_early_event = torch.cuda.Event()
            self._mtp_done_event = torch.cuda.Event()
        early_event = self._mtp_early_event
        mtp_done_event = self._mtp_done_event
        mtp_normed_holder = []

        hidden_states = model_inner.embed_tokens(verify_ids)
        residual = None

        for i, layer in enumerate(model_inner.layers):
            hidden_states, residual = layer(verify_pos, hidden_states, residual)
            if i == early_layer_idx:
                # `hidden_states + residual` already produces a fresh tensor;
                # no clone needed. Tag it for side_stream so the allocator
                # doesn't recycle it while side_stream is still reading.
                early_hidden = hidden_states + residual
                early_hidden.record_stream(side_stream)
                early_event.record(main_stream)
                with torch.cuda.stream(side_stream):
                    side_stream.wait_event(early_event)
                    set_context(True, mtp_cu_q_t, mtp_cu_k_t, max_mtp_q, max_mtp_k,
                                mtp_slot_t, None, mtp_block_tables)
                    mtp_hidden_input = early_hidden.index_select(0, mtp_hidden_idx_t)
                    mtp_embeds = embed_tokens(mtp_input_ids_t)
                    mtp_out = mtp_layer(mtp_embeds, mtp_hidden_input, mtp_positions_t)
                    mtp_normed = mtp_out[0] if isinstance(mtp_out, tuple) else mtp_out
                    # mtp_normed is allocated on side_stream but consumed on
                    # main_stream by _stash_pending_draft → record_stream so
                    # the side_stream allocator does not recycle it.
                    mtp_normed.record_stream(main_stream)
                    mtp_normed_holder.append(mtp_normed)
                    mtp_done_event.record(side_stream)
                # Restore target verify context for the remaining layers on main stream.
                set_context(*v_args)

        hidden_states, _ = model_inner.norm(hidden_states, residual)
        target_logits = self.model.compute_logits_all(hidden_states)
        reset_context()

        # === Accept on rank 0 (mirrors sync MTP path) ===
        if self.rank == 0:
            all_accepted = []
            offset = 0
            for seq, draft_tokens in zip(seqs, all_draft_tokens):
                num_verify = len(draft_tokens) + 1
                seq_logits = target_logits[offset:offset + num_verify]
                predicted = seq_logits.argmax(dim=-1).tolist()
                accepted = []
                nd = len(draft_tokens)
                for j in range(nd):
                    if predicted[j] == draft_tokens[j]:
                        accepted.append(draft_tokens[j])
                    else:
                        accepted.append(predicted[j])
                        break
                else:
                    accepted.append(predicted[nd])
                all_accepted.append(accepted)
                offset += num_verify
        else:
            pass
        if self.rank != 0:
            all_accepted = [[] for _ in seqs]

        # === TP broadcast accepted tokens (same as sync path) ===
        if self.tp_size > 1:
            n_acc_t = torch.tensor(
                [len(a) for a in all_accepted] if self.rank == 0 else [0] * n_seqs,
                dtype=torch.int64, device=d)
            dist.broadcast(n_acc_t, 0, group=self.tp_group)
            n_acc_list = n_acc_t.tolist()
            if self.rank != 0:
                all_accepted = [[0] * n_acc_list[i] for i in range(n_seqs)]
            for i in range(n_seqs):
                if n_acc_list[i] > 0:
                    acc_t = torch.tensor(all_accepted[i], dtype=torch.int64, device=d)
                    dist.broadcast(acc_t, 0, group=self.tp_group)
                    if self.rank != 0:
                        all_accepted[i] = acc_t.tolist()
        else:
            n_acc_list = [len(a) for a in all_accepted]

        # === Wait for side-stream MTP, stash next draft ===
        main_stream.wait_event(mtp_done_event)
        if mtp_normed_holder:
            self._stash_pending_draft(seqs, mtp_normed_holder[0], last_idx_t)

        return all_accepted if self.rank == 0 else None

    def _send_cmd(self, cmd: int, aux: int = 0):
        """Send command to draft via NCCL. [cmd, aux, 0, 0] in one message."""
        if not hasattr(self, '_cmd_buf'):
            return
        self._cmd_buf[0] = cmd
        self._cmd_buf[1] = aux
        dist.send(self._cmd_buf, dst=self.draft_rank, group=self.async_pg)

    @torch.inference_mode()
    def _run_latent_prefill_with_hidden(self, seqs: list[Sequence]) -> list[int]:
        """Latent SD prefill: run target model, send hidden to draft via NCCL."""
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        context = get_context()
        d = self.device.device_name

        eagle3 = getattr(self.config, 'eagle3', False)
        if eagle3:
            hidden, aux_concat, _ = self._run_target_with_aux(input_ids, positions)
        else:
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
                if eagle3:
                    # EAGLE-3: send 3H aux concat instead of H hidden
                    dist.send(aux_concat[start:end].contiguous(), dst=self.draft_rank, group=self.async_pg)
                else:
                    dist.send(hidden[start:end].contiguous(), dst=self.draft_rank, group=self.async_pg)
                dist.send(torch.tensor(seq_positions, dtype=torch.int64, device=d),
                          dst=self.draft_rank, group=self.async_pg)
                dist.send(torch.tensor(bt, dtype=torch.int32, device=d),
                          dst=self.draft_rank, group=self.async_pg)
                ack = torch.zeros(1, dtype=torch.int64, device=d)
                dist.recv(ack, src=self.draft_rank, group=self.async_pg)

                # Receive prewarm tree-cache push from draft (MTP path only;
                # eagle async sends nothing during prefill).
                if self.use_mtp:
                    # Recv directly into pre-allocated max-size buf (no size handshake).
                    dist.recv(self._push_buf_max, src=self.draft_rank, group=self.async_pg)
                    if not hasattr(self, '_local_tree_cache'):
                        self._local_tree_cache = {}
                    push_cpu = self._push_buf_max.cpu().tolist()
                    pidx = 0
                    n_push = push_cpu[pidx]; pidx += 1
                    for _ in range(n_push):
                        sid = push_cpu[pidx]
                        n_e = push_cpu[pidx + 1]
                        K_a = push_cpu[pidx + 2]
                        pidx += 3
                        if n_e > 0:
                            keys_flat = push_cpu[pidx:pidx + n_e * 3]
                            pidx += n_e * 3
                            toks_flat = push_cpu[pidx:pidx + n_e * K_a]
                            pidx += n_e * K_a
                            self._local_tree_cache[sid] = (keys_flat, toks_flat, n_e, K_a)

        reset_context()

        if self.rank == 0:
            for seq, tid in zip(seqs, token_ids):
                seq.recovery_token_id = tid
        return token_ids

    @torch.inference_mode()
    def _run_latent_decode(self, seqs: list[Sequence]) -> list[list[int]]:
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
                # EAGLE async: same pure-CPU lookup pattern as MTP. Cache stored
                # as flat Python lists by llm_engine push drain (CPU side, after
                # .cpu().tolist() of the whole push buf). No GPU op per lookup.
                if not hasattr(self, '_local_tree_cache'):
                    self._local_tree_cache = {}
                    self._cache_hit = 0
                    self._cache_miss = 0
                for seq in seqs:
                    cache = self._local_tree_cache.get(seq.seq_id)
                    sid_t, acc_t, tok_t = seq.seq_id, seq.last_accepted_len, seq.last_token
                    found = None
                    if cache is not None:
                        keys_flat, toks_flat, n_e, K_a = cache
                        if K_a >= k:
                            # Tree mode: scan for (sid, acc, last_token) match
                            for j in range(n_e):
                                base = j * 3
                                if (keys_flat[base] == sid_t
                                        and keys_flat[base + 1] == acc_t
                                        and keys_flat[base + 2] == tok_t):
                                    tbase = j * K_a
                                    found = toks_flat[tbase:tbase + k]
                                    break
                        else:
                            # Chain mode: K chained lookups
                            chain = []
                            cur = tok_t
                            for step in range(k):
                                hit_step = False
                                target_acc = acc_t + step
                                for j in range(n_e):
                                    base = j * 3
                                    if (keys_flat[base] == sid_t
                                            and keys_flat[base + 1] == target_acc
                                            and keys_flat[base + 2] == cur):
                                        cur = toks_flat[j * K_a]
                                        chain.append(cur)
                                        hit_step = True
                                        break
                                if not hit_step:
                                    break
                            if chain and len(chain) == k:
                                found = chain
                            elif chain:
                                found = chain + [0] * (k - len(chain))
                    if found is not None:
                        all_draft_tokens.append(found)
                        self._cache_hit += 1
                    else:
                        self._cache_miss += 1
                        if not self.config.enable_fallback:
                            # Cache miss → baseline 1-token verify (no spec).
                            all_draft_tokens.append([])
                        else:
                            # Fallback spec: ~10ms NCCL roundtrip per miss seq.
                            self._send_cmd(0)
                            meta = torch.tensor(
                                [seq.seq_id, getattr(seq, 'last_accepted_len', 0),
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
                # MTP path: local tree cache lookup — pure CPU scan.
                # Cache is stored as flat Python lists (populated from push_cpu
                # at end of previous step). Each lookup compares a 3-tuple
                # against ≤32 cached entries — pure Python is faster than
                # dispatching a GPU kernel (~5-10μs fixed overhead per op).
                if not hasattr(self, '_local_tree_cache'):
                    self._local_tree_cache = {}
                if not hasattr(self, '_cache_hit'):
                    self._cache_hit = 0
                    self._cache_miss = 0
                miss_idxs = []
                for i, seq in enumerate(seqs):
                    cache = self._local_tree_cache.get(seq.seq_id)
                    sid_t, acc_t, tok_t = seq.seq_id, seq.last_accepted_len, seq.last_token
                    found = None
                    if cache is not None:
                        keys_flat, toks_flat, n_e, K_a = cache
                        if K_a >= k:
                            # Tree mode: scan for exact (sid, acc, last_token) match
                            for j in range(n_e):
                                base = j * 3
                                if (keys_flat[base] == sid_t
                                        and keys_flat[base + 1] == acc_t
                                        and keys_flat[base + 2] == tok_t):
                                    tbase = j * K_a
                                    found = toks_flat[tbase:tbase + k]
                                    break
                        else:
                            # Chain mode: K chained lookups
                            chain = []
                            cur = tok_t
                            for step in range(k):
                                hit_step = False
                                target_acc = acc_t + step
                                for j in range(n_e):
                                    base = j * 3
                                    if (keys_flat[base] == sid_t
                                            and keys_flat[base + 1] == target_acc
                                            and keys_flat[base + 2] == cur):
                                        cur = toks_flat[j * K_a]
                                        chain.append(cur)
                                        hit_step = True
                                        break
                                if not hit_step:
                                    break
                            if chain and len(chain) == k:
                                found = chain
                            elif chain:
                                # partial chain — pad with zeros so verify still
                                # has K draft positions; accept will reject the
                                # zeros after first mismatch
                                found = chain + [0] * (k - len(chain))
                    if found is not None:
                        all_draft_tokens.append(found)
                        self._cache_hit += 1
                    else:
                        all_draft_tokens.append(None)
                        miss_idxs.append(i)
                        self._cache_miss += 1
                # === Handle misses ===
                # Default (enable_fallback=False): no spec → baseline
                # 1-token verify (verify_ids = [last_token] for this seq).
                # Optional (enable_fallback=True): send cmd=0 to draft
                # for sync fallback spec — costs ~10ms NCCL roundtrip but
                # recovers K spec tokens per miss.
                use_fallback = self.config.enable_fallback
                for i in miss_idxs:
                    if not use_fallback:
                        all_draft_tokens[i] = []
                    else:
                        seq = seqs[i]
                        self._send_cmd(0)
                        meta = torch.tensor(
                            [seq.seq_id, getattr(seq, 'last_accepted_len', 0),
                             seq.last_token, len(seq), len(seq.block_table)],
                            dtype=torch.int64, device=d)
                        dist.send(meta, dst=self.draft_rank, group=self.async_pg)
                        dist.send(self.last_hidden[seq.seq_id].squeeze(0).contiguous(),
                                  dst=self.draft_rank, group=self.async_pg)
                        dist.send(torch.tensor(seq.block_table, dtype=torch.int32, device=d),
                                  dst=self.draft_rank, group=self.async_pg)
                        jit_buf = torch.zeros(k, dtype=torch.int64, device=d)
                        dist.recv(jit_buf, src=self.draft_rank, group=self.async_pg)
                        all_draft_tokens[i] = jit_buf.tolist()

        # Broadcast draft tokens to TP workers — single batched op.
        # Layout: per-seq [actual_len, t0, t1, ..., t_{k-1}] (zero-padded).
        # rank 0 fills, others recv into pre-allocated _draft_bcast_buf, single
        # .cpu().tolist() to settle all draft tokens at once.
        if self.tp_size > 1:
            n = len(seqs)
            slot_size = 1 + k
            buf = self._draft_bcast_buf[:n * slot_size]
            if self.rank == 0:
                flat = []
                for tokens in all_draft_tokens:
                    actual_len = len(tokens)
                    flat.append(actual_len)
                    flat.extend(tokens)
                    flat.extend([0] * (k - actual_len))
                buf.copy_(torch.tensor(flat, dtype=torch.int64))
            dist.broadcast(buf, 0, group=self.tp_group)
            if self.rank != 0:
                flat_cpu = buf.cpu().tolist()
                all_draft_tokens = []
                for i in range(n):
                    base = i * slot_size
                    actual_len = flat_cpu[base]
                    all_draft_tokens.append(flat_cpu[base + 1:base + 1 + actual_len])

        # === Verify phase (layer-by-layer for early hidden extraction) ===
        verify_ids, verify_pos = self._prepare_verify(seqs, all_draft_tokens)
        model_inner = self.model.model
        n_layers = len(model_inner.layers)
        eagle3 = getattr(self.config, 'eagle3', False) and self.config.eagle_async
        if eagle3:
            # EAGLE-3: extract at 3 layers, send after the last one (N-3)
            eagle3_layers = self._get_eagle3_aux_layers()
            early_layer = n_layers - 3
            send_after_norm = False
        else:
            # latent_early_layers semantic — K MUST be < 0:
            #   K=-1 → extract after the last layer (= 倒数第1, sync MTP equivalent)
            #   K=-3 → extract after the 3rd-from-last (= 倒数第3)
            # The hidden sent to draft is always pre-final-norm (unnormed
            # residual sum). Draft applies its own norm downstream.
            ssd_K = self.config.latent_early_layers
            assert ssd_K < 0, (
                f"latent_early_layers must be < 0 "
                f"(K=-1 → last layer, K=-N → 倒数第N); got {ssd_K}"
            )
            send_after_norm = False
            early_layer = n_layers + ssd_K   # K=-1 → N-1, K=-3 → N-3
            eagle3_layers = ()
        eagle3_hiddens = {}

        def _build_eagle_payload_and_send(hidden_to_send):
            """Pack meta + verify_ids + hidden into one buffer and isend (EAGLE async)."""
            packed_list = []
            for seq, dt in zip(seqs, all_draft_tokens):
                nv = len(dt) + 1
                bt = seq.block_table
                sp = len(seq) - 1
                packed_list.extend([seq.seq_id, nv, len(seq) + len(dt), len(bt), sp])
                packed_list.extend(bt)
                packed_list.extend(range(sp, sp + nv))
            meta_len = len(packed_list)
            meta_t = torch.tensor(packed_list, dtype=torch.int64, device=d)
            verify_ids_i64 = verify_ids.to(torch.int64)
            n_verify_tok = verify_ids_i64.shape[0]
            hidden_i64 = hidden_to_send.contiguous().view(-1).view(torch.int64)
            total_len = 1 + meta_len + n_verify_tok + hidden_i64.shape[0]
            payload = self._payload_buf[:total_len]
            payload[0] = meta_len
            payload[1:1 + meta_len] = meta_t
            payload[1 + meta_len:1 + meta_len + n_verify_tok] = verify_ids_i64
            payload[1 + meta_len + n_verify_tok:] = hidden_i64
            self._cmd_buf[0] = 5
            self._cmd_buf[1] = len(seqs)
            self._cmd_buf[2] = total_len
            self._cmd_buf[3] = meta_len
            return [
                dist.isend(self._cmd_buf, dst=self.draft_rank, group=self.async_pg),
                dist.isend(payload, dst=self.draft_rank, group=self.async_pg),
            ]

        hidden_states = model_inner.embed_tokens(verify_ids)
        residual = None
        _async_send_handles = []
        for i, layer in enumerate(model_inner.layers):
            hidden_states, residual = layer(verify_pos, hidden_states, residual)
            # Collect tri-layer hiddens for EAGLE-3
            if eagle3 and i in eagle3_layers:
                eagle3_hiddens[i] = (hidden_states + residual).clone()
            if i == early_layer and self.rank == 0 and self.async_pg is not None:
                if eagle3 and len(eagle3_hiddens) == 3:
                    # Concatenate 3 layer hiddens: [N, 3*H]
                    sorted_keys = sorted(eagle3_hiddens.keys())
                    early_hidden_all = torch.cat([eagle3_hiddens[k] for k in sorted_keys], dim=-1)
                else:
                    early_hidden_all = (hidden_states + residual).clone()
                num_seqs_batch = len(seqs)
                if self.config.eagle_async:
                    _async_send_handles = _build_eagle_payload_and_send(early_hidden_all)
                else:
                    # MTP: separate sends matching MTP draft recv protocol
                    meta_only = []
                    bt_list = []
                    pos_list = []
                    for seq, dt in zip(seqs, all_draft_tokens):
                        nv = len(dt) + 1
                        bt = seq.block_table
                        spos = len(seq) - 1
                        meta_only.extend([seq.seq_id, nv, len(seq) + len(dt), len(bt), spos])
                        bt_list.extend(bt)
                        pos_list.extend(range(spos, spos + nv))
                    self._cmd_buf[0] = 5
                    self._cmd_buf[1] = num_seqs_batch
                    _async_send_handles = [
                        dist.isend(self._cmd_buf, dst=self.draft_rank, group=self.async_pg),
                        dist.isend(torch.tensor(meta_only, dtype=torch.int64, device=d), dst=self.draft_rank, group=self.async_pg),
                        dist.isend(early_hidden_all, dst=self.draft_rank, group=self.async_pg),
                        dist.isend(torch.tensor(bt_list, dtype=torch.int32, device=d), dst=self.draft_rank, group=self.async_pg),
                        dist.isend(torch.tensor(pos_list, dtype=torch.int64, device=d), dst=self.draft_rank, group=self.async_pg),
                    ]

        hidden_states, residual = model_inner.norm(hidden_states, residual)

        # Wait for async sends to complete before proceeding
        for h in _async_send_handles:
            h.wait()
        hidden = residual  # unnormed, for MTP last_hidden

        # Pipelined recv into pre-allocated max-size buf (no size handshake).
        # Both sides know the max size from config; draft pads accordingly.
        push_irecv = None
        if (self.rank == 0 and self.async_pg is not None and self.use_mtp
                and len(_async_send_handles) > 0):
            push_irecv = dist.irecv(
                self._push_buf_max, src=self.draft_rank, group=self.async_pg)

        target_logits = self.model.compute_logits_all(hidden_states)
        reset_context()

        # === Accept phase ===
        if self.rank == 0:
            all_accepted = []
            offset = 0
            for seq, draft_tokens in zip(seqs, all_draft_tokens):
                num_verify = len(draft_tokens) + 1
                seq_logits = target_logits[offset:offset + num_verify]
                predicted = seq_logits.argmax(dim=-1).tolist()
                accepted = []
                # Iterate up to len(draft_tokens). When draft is [] (cache miss
                # → no-spec), this just picks predicted[0] as the only accepted
                # token (= baseline 1-token decode).
                k_local = len(draft_tokens)
                for j in range(k_local):
                    if predicted[j] == draft_tokens[j]:
                        accepted.append(draft_tokens[j])
                    else:
                        accepted.append(predicted[j])
                        break
                else:
                    accepted.append(predicted[k_local])

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

        # Skip MTP KV cache update in Latent SD mode — cache hit rate ~99% makes it unnecessary
        # (MTP KV only needed for cache miss local fallback, which is rare)

        # Drain push tree-cache (cmd=5 response). The size irecv was kicked off
        # right after the cmd=5 isends completed, so its NCCL transfer has been
        # running in parallel with target's logits + accept work above. Wait
        # the size, recv the variable-length payload synchronously, parse into
        # _local_tree_cache. Done BEFORE engine.step's cleanup_send so the p2p
        # FIFO is preserved (target=[isend cmd5, irecv sz, recv push, send
        # cmd3, recv ack] vs draft=[recv cmd5, send sz, send push, recv cmd3,
        # send ack]).
        if push_irecv is not None:
            push_irecv.wait()
            if not hasattr(self, '_local_tree_cache'):
                self._local_tree_cache = {}
            push_cpu = self._push_buf_max.cpu().tolist()
            pidx = 0
            n_push = push_cpu[pidx]; pidx += 1
            for _ in range(n_push):
                sid = push_cpu[pidx]
                n_e = push_cpu[pidx + 1]
                K_a = push_cpu[pidx + 2]
                pidx += 3
                if n_e > 0:
                    # Store cache as flat CPU lists. Lookup is a tight Python
                    # scan over 1-32 entries — much cheaper than dispatching a
                    # GPU kernel which carries ~5-10μs fixed overhead per op.
                    keys_flat = push_cpu[pidx:pidx + n_e * 3]
                    pidx += n_e * 3
                    toks_flat = push_cpu[pidx:pidx + n_e * K_a]
                    pidx += n_e * K_a
                    self._local_tree_cache[sid] = (keys_flat, toks_flat, n_e, K_a)
        # _push_pending no longer needed — we drained inside this function.

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
