"""MTP Draft Runner for SSD (Speculative Streaming Decoding).

Runs MTP layers on a separate GPU. Uses NCCL for target↔draft communication.
Implements: tree cache, glue decode, fork tokens, tree decode.
"""
import os
from glob import glob
import torch
import torch.nn.functional as F
import torch.distributed as dist
from safetensors import safe_open

from nanovllm.config import Config
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.models.mimo import MiMoMTPLayer
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import _load_weight
from nanovllm.utils.device import get_device_backend

_PANGU_TYPES = {'PanguProMoE', 'PanguProMoEV2', 'PanguUltraMoE', 'PanguEmbedded', 'pangu'}


def _is_pangu(hf_config):
    model_type = getattr(hf_config, 'model_type', '')
    return model_type in _PANGU_TYPES or getattr(hf_config, 'param_sink_number', 0) > 0


def _load_draft_model(model, path):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    safetensor_files = glob(os.path.join(path, "*.safetensors"))
    if safetensor_files:
        for file in safetensor_files:
            with safe_open(file, "pt", "cpu") as f:
                for weight_name in f.keys():
                    try:
                        _load_weight(model, packed_modules_mapping, weight_name, f.get_tensor(weight_name))
                    except (AttributeError, KeyError):
                        pass
    else:
        bin_files = glob(os.path.join(path, "pytorch_model*.bin"))
        for file in bin_files:
            state_dict = torch.load(file, map_location="cpu", weights_only=True)
            for weight_name, weight_tensor in state_dict.items():
                try:
                    _load_weight(model, packed_modules_mapping, weight_name, weight_tensor)
                except (AttributeError, KeyError):
                    pass
            del state_dict


class MiMoMTPDraftModel(torch.nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        # tp_group=None, tp_size=1 → no TP sharding in draft
        self.model = torch.nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, tp_size=1)
        self.model.mtp_layers = torch.nn.ModuleList([
            MiMoMTPLayer(config, tp_size=1)
            for _ in range(getattr(config, 'num_nextn_predict_layers', 1))
        ])
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, tp_size=1)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data


class PanguMTPDraftModel(torch.nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        from nanovllm.models.pangu import PanguMTPLayer, PanguModel
        self.model = torch.nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size, tp_size=1)
        self.model.mtp_layers = torch.nn.ModuleList([
            PanguMTPLayer(config, layer_idx=config.num_hidden_layers + i, tp_size=1)
            for i in range(getattr(config, 'num_nextn_predict_layers', 1))
        ])
        # Main model norm + lm_head (for early_speculate candidate computation)
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size, tp_size=1)
        if getattr(config, 'tie_word_embeddings', False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data


def _load_pangu_draft_model(model, path, hf_config):
    """Load Pangu draft model with MTP weight name rewriting and MoE expert handling."""
    import re
    from nanovllm.utils.loader import default_weight_loader

    num_hidden = hf_config.num_hidden_layers
    num_mtp = getattr(hf_config, 'num_nextn_predict_layers', 0)
    spec_layer_names = ["enorm", "hnorm", "eh_proj", "shared_head"]
    packed_modules_mapping = model.packed_modules_mapping
    params_dict = dict(model.named_parameters())

    safetensor_files = sorted(glob(os.path.join(path, "*.safetensors")))
    for file in safetensor_files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                name = weight_name
                # Only load MTP layers, embed_tokens, lm_head, model.norm
                is_mtp = False
                for mtp_idx in range(num_mtp):
                    layer_idx = num_hidden + mtp_idx
                    old_prefix = f"model.layers.{layer_idx}."
                    if not name.startswith(old_prefix):
                        continue
                    is_mtp = True
                    is_spec = any(w in name for w in spec_layer_names)
                    is_shared = "embed_tokens" in name
                    if is_shared:
                        name = name.replace(old_prefix, "model.")
                    elif is_spec:
                        name = name.replace(old_prefix, f"model.mtp_layers.{mtp_idx}.")
                    else:
                        name = name.replace(old_prefix, f"model.mtp_layers.{mtp_idx}.mtp_block.")
                    break
                # Also load embed_tokens, lm_head, model.norm
                if not is_mtp:
                    if not any(k in name for k in ["embed_tokens", "lm_head", "model.norm."]):
                        continue
                # e_score_correction_bias remapping
                if name.endswith("e_score_correction_bias") and "gate." not in name:
                    name = name.replace("e_score_correction_bias", "gate.e_score_correction_bias")
                # Handle MoE expert weights
                m = re.search(r'\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$', name)
                if m:
                    expert_id = int(m.group(1))
                    proj_type = m.group(2)
                    moe_prefix = name[:m.start()] + ".mlp."
                    if proj_type in ("gate_proj", "up_proj"):
                        param_name = moe_prefix + "w13_weight"
                        shard_id = 0 if proj_type == "gate_proj" else 1
                    else:
                        param_name = moe_prefix + "w2_weight"
                        shard_id = None
                    if param_name in params_dict:
                        param = params_dict[param_name]
                        loader = getattr(param, "weight_loader")
                        if shard_id is not None:
                            loader(param, f.get_tensor(weight_name), shard_id, expert_id=expert_id)
                        else:
                            loader(param, f.get_tensor(weight_name), expert_id=expert_id)
                    continue
                # Handle packed modules (gate_up_proj)
                loaded = False
                for ckpt_name, (param_name, shard_id) in packed_modules_mapping.items():
                    if f".{ckpt_name}." in name:
                        mapped = name.replace(ckpt_name, param_name)
                        if mapped in params_dict:
                            param = params_dict[mapped]
                            loader = getattr(param, "weight_loader")
                            loader(param, f.get_tensor(weight_name), shard_id)
                            loaded = True
                        break
                if loaded:
                    continue
                # Direct load
                if name in params_dict:
                    param = params_dict[name]
                    loader = getattr(param, "weight_loader", default_weight_loader)
                    loader(param, f.get_tensor(weight_name))


class MTPDraftRunner:

    def __init__(self, config: Config, rank: int, init_q=None):
        self.config = config
        hf_config = config.hf_config
        self.hf_config = hf_config
        self.block_size = config.kvcache_block_size
        self.K = config.num_speculative_tokens
        self.F = config.async_fan_out
        self.fan_out_list = config.fan_out_list
        self.mq_len = config.mq_len
        self.rank = rank

        self.device_backend = get_device_backend()
        self.device_backend.set_device(config.draft_gpu)
        self.device = f"cuda:{config.draft_gpu}"

        # Join unified world (same as target + TP workers)
        print(f"[Draft rank={rank}] calling init_process_group world={config.num_gpus}...", flush=True)
        dist.init_process_group("nccl", f"tcp://localhost:{os.environ.get('NCCL_PORT', '2333')}",
                                world_size=config.num_gpus, rank=rank)
        print(f"[Draft rank={rank}] init_process_group done, creating groups...", flush=True)
        # Must participate in new_group calls (collective)
        tp_ranks = list(range(config.tensor_parallel_size))
        self.tp_group = dist.new_group(tp_ranks)  # draft not in this group
        self.async_pg = dist.new_group([0, config.draft_rank])

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)

        self.is_pangu = _is_pangu(hf_config)
        if self.is_pangu:
            self.draft_model = PanguMTPDraftModel(hf_config)
            _load_pangu_draft_model(self.draft_model, config.model, hf_config)
        else:
            self.draft_model = MiMoMTPDraftModel(hf_config)
            _load_draft_model(self.draft_model, config.model)

        self.mtp_layer = self.draft_model.model.mtp_layers[0]
        self.embed_tokens = self.draft_model.model.embed_tokens
        self.lm_head = self.draft_model.lm_head

        # Warmup + allocate KV cache (same sizing as target)
        self._warmup_and_allocate_kv_cache()
        # Populate sink KV for Pangu MTP layers
        if self.is_pangu:
            self._populate_pangu_sink_kv()
        # Send num_kvcache_blocks to target via init_q
        if init_q is not None:
            init_q.put(config.num_kvcache_blocks)
            init_q.close()

        # CUDA graph capture for single-token MTP decode (skip for Pangu: MoE breaks graph)
        if not self.is_pangu:
            self._capture_mtp_graph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        self.tree_decode = config.ssd_tree_decode
        self._reset_tree_cache()
        self.last_hidden = {}
        # Pre-allocate recv buffers
        self._cmd_buf = torch.zeros(2, dtype=torch.int64, device=self.device)

        print(f"[MTPDraftRunner] Initialized on GPU {config.draft_gpu}, K={self.K}, F={self.F}, tree_decode={self.tree_decode}", flush=True)

    def _warmup_and_allocate_kv_cache(self):
        """Compute available memory and allocate MTP KV cache."""
        hf_config = self.hf_config
        config = self.config
        torch.cuda.empty_cache()

        num_kv_heads = hf_config.num_key_value_heads
        # Pangu uses qk_rope_dim + qk_nope_dim as head_dim for KV cache
        qk_rope = getattr(hf_config, 'qk_rope_dim', None)
        qk_nope = getattr(hf_config, 'qk_nope_dim', None)
        if qk_rope is not None and qk_nope is not None:
            head_dim = qk_rope + qk_nope
        else:
            head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        num_mtp_layers = hf_config.num_nextn_predict_layers
        # Compute num_blocks from available memory
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        block_bytes = 2 * num_mtp_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        # Draft only needs MTP KV cache — cap to max concurrent tokens, not fill GPU
        max_tokens = config.max_num_seqs * (config.max_model_len // self.block_size + 1)
        avail_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // max(block_bytes, 1)
        num_blocks = min(max(avail_blocks, 1), max_tokens)
        config.num_kvcache_blocks = num_blocks
        self.mtp_kv_cache = torch.empty(
            2, num_mtp_layers, num_blocks, self.block_size,
            num_kv_heads, head_dim, device=self.device, dtype=hf_config.torch_dtype,
        )
        layer_id = 0
        for mtp_layer in self.draft_model.model.mtp_layers:
            for module in mtp_layer.modules():
                if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                    module.k_cache = self.mtp_kv_cache[0, layer_id]
                    module.v_cache = self.mtp_kv_cache[1, layer_id]
                    layer_id += 1

    def _populate_pangu_sink_kv(self):
        """Finalize Pangu sink KV params (k_layernorm + v_padding) after weight loading."""
        from nanovllm.models.pangu import PanguSinkAttention
        for mtp_layer in self.draft_model.model.mtp_layers:
            if hasattr(mtp_layer, 'mtp_block'):
                attn = getattr(mtp_layer.mtp_block, 'self_attn', None)
                if isinstance(attn, PanguSinkAttention):
                    attn.post_weight_load()

    @torch.inference_mode()
    def _capture_mtp_graph(self):
        """Capture CUDA graph for single-token MTP decode (bs=1)."""
        hf = self.hf_config
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size + 10
        d = self.device

        # Pre-allocate static input/output buffers for graph replay
        g = {}
        g["input_id"] = torch.zeros(1, dtype=torch.int64, device=d)
        g["pos"] = torch.zeros(1, dtype=torch.int64, device=d)
        g["hidden"] = torch.zeros(1, hf.hidden_size, dtype=hf.torch_dtype, device=d)
        g["slot"] = torch.zeros(1, dtype=torch.int32, device=d)
        g["ctx_len"] = torch.zeros(1, dtype=torch.int32, device=d)
        g["bt"] = torch.zeros(1, max_num_blocks, dtype=torch.int32, device=d)
        g["out_normed"] = torch.zeros(1, hf.hidden_size, dtype=hf.torch_dtype, device=d)
        g["out_prenorm"] = torch.zeros(1, hf.hidden_size, dtype=hf.torch_dtype, device=d)

        # Warmup
        set_context(False, slot_mapping=g["slot"], context_lens=g["ctx_len"], block_tables=g["bt"])
        embeds = self.embed_tokens(g["input_id"])
        normed, prenorm = self.mtp_layer(embeds, g["hidden"], g["pos"])
        g["out_normed"].copy_(normed)
        g["out_prenorm"].copy_(prenorm)
        reset_context()

        # Capture graph
        self._mtp_graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=g["slot"], context_lens=g["ctx_len"], block_tables=g["bt"])
        with torch.cuda.graph(self._mtp_graph):
            embeds = self.embed_tokens(g["input_id"])
            normed, prenorm = self.mtp_layer(embeds, g["hidden"], g["pos"])
            g["out_normed"].copy_(normed)
            g["out_prenorm"].copy_(prenorm)
        reset_context()
        torch.cuda.synchronize()
        self._g = g

    def _reset_tree_cache(self, seq_id=None):
        if seq_id is None:
            self.tree_caches = {}  # seq_id -> (keys, tokens)
        elif seq_id in self.tree_caches:
            del self.tree_caches[seq_id]

    _hit = 0
    _miss = 0

    def compute_logits_all(self, hidden_states):
        return F.linear(hidden_states, self.lm_head.weight)

    def compute_mtp_logits(self, hidden_states):
        """Compute logits using MTP's own head (Pangu shared_head) or main lm_head (MiMo)."""
        if hasattr(self.mtp_layer, 'shared_head'):
            return F.linear(hidden_states, self.mtp_layer.shared_head.head.weight)
        return F.linear(hidden_states, self.lm_head.weight)

    @torch.inference_mode()
    def _mtp_step(self, token_id, hidden_state, cur_pos, block_table_t):
        """Single MTP forward step using CUDA graph. Returns (prenorm, logits, next_token)."""
        block_idx = cur_pos // self.block_size
        bt_len = block_table_t.shape[0]
        block_offset = cur_pos % self.block_size
        slot = block_table_t[min(block_idx, bt_len - 1)] * self.block_size + block_offset

        g = self._g
        g["input_id"][0] = token_id
        g["pos"][0] = cur_pos
        if hidden_state.dim() == 1:
            g["hidden"][0].copy_(hidden_state)
        else:
            g["hidden"].copy_(hidden_state)
        g["slot"][0] = slot
        g["ctx_len"][0] = cur_pos + 1
        g["bt"][0, :bt_len] = block_table_t

        set_context(False, slot_mapping=g["slot"], context_lens=g["ctx_len"], block_tables=g["bt"])
        self._mtp_graph.replay()
        reset_context()

        logits = self.compute_mtp_logits(g["out_normed"])
        next_token = logits.argmax(dim=-1).item()
        return g["out_prenorm"][0].clone(), logits.squeeze(0), next_token

    @torch.inference_mode()
    def jit_speculate(self, recovery_token, hidden_state, num_tokens, block_table_t):
        K = self.Kneng
        draft_tokens = []
        cur_token = recovery_token
        cur_pos = num_tokens - 1
        cur_hidden = hidden_state
        for _ in range(K):
            prenorm, logits, d_token = self._mtp_step(cur_token, cur_hidden, cur_pos, block_table_t)
            draft_tokens.append(d_token)
            cur_hidden = prenorm
            cur_token = d_token
            cur_pos += 1
        return draft_tokens

    @torch.inference_mode()
    def hit_cache_and_respond(self, seq_id, accepted_len, recovery_token, hidden_state, num_tokens, block_table_t):
        if seq_id in self.tree_caches:
            cache_keys, cache_tokens = self.tree_caches[seq_id]
            request_key = torch.tensor([[seq_id, accepted_len, recovery_token]], dtype=torch.int64, device=self.device)
            match = torch.all(request_key.unsqueeze(1) == cache_keys.unsqueeze(0), dim=2).squeeze(0)
            if match.any():
                idx = match.float().argmax().item()
                MTPDraftRunner._hit += 1
                return cache_tokens[idx].tolist(), True
        MTPDraftRunner._miss += 1
        tokens = self.jit_speculate(recovery_token, hidden_state, num_tokens, block_table_t)
        return tokens, False

    @torch.inference_mode()
    def build_tree(self, seq_id, recovery_token, draft_tokens, hidden_state, num_tokens, block_table_t):
        K, F = self.K, self.F
        # 1. Glue decode
        tokens = [recovery_token] + draft_tokens
        all_prenorms, all_logits = [], []
        cur_hidden = hidden_state.unsqueeze(0) if hidden_state.dim() == 1 else hidden_state
        cur_pos = num_tokens - 1
        for token in tokens:
            prenorm, logits, _ = self._mtp_step(token, cur_hidden, cur_pos, block_table_t)
            all_prenorms.append(prenorm)
            all_logits.append(logits)
            cur_hidden = prenorm.unsqueeze(0)
            cur_pos += 1
        glue_logits = torch.stack(all_logits)
        glue_prenorms = torch.stack(all_prenorms)

        # 2. Fork
        fork_logits = glue_logits.clone()
        returned_t = torch.tensor(tokens, dtype=torch.int64, device=self.device)
        for j in range(K):
            fork_logits[j, returned_t[j + 1]] = float('-inf')
        _, topk_idx = torch.topk(fork_logits, F, dim=-1)
        forked_tokens = topk_idx.reshape(-1)

        # 3. Tree decode
        MQ_LEN = self.mq_len
        spec_tokens = torch.zeros((MQ_LEN, K), dtype=torch.int64, device=self.device)
        fan_t = torch.tensor(self.fan_out_list, device=self.device)
        hidden_states = glue_prenorms.repeat_interleave(fan_t, dim=0)
        current_ids = forked_tokens
        bt = block_table_t
        max_bi = bt.shape[0] - 1
        base_pos = num_tokens - 1 + (K + 1)
        j_indices = torch.arange(K + 1, device=self.device, dtype=torch.int64).repeat_interleave(fan_t)

        for depth in range(K):
            positions = torch.arange(MQ_LEN, device=self.device, dtype=torch.int64) + base_pos + depth * MQ_LEN
            rope_positions = (num_tokens - 1) + j_indices + depth + 1
            bi = (positions // self.block_size).long().clamp(0, max_bi)
            bo = (positions % self.block_size).int()
            slot_mapping = (bt[bi] * self.block_size + bo).int()
            context_lens = (positions + 1).int()
            block_tables = bt.unsqueeze(0).expand(MQ_LEN, -1)

            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
            embeds = self.embed_tokens(current_ids)
            normed, prenorm = self.mtp_layer(embeds, hidden_states, rope_positions)
            reset_context()
            logits = self.compute_mtp_logits(normed)
            next_tokens = logits.argmax(dim=-1)
            spec_tokens[:, depth] = next_tokens
            hidden_states = prenorm
            current_ids = next_tokens

        # 4. Populate per-seq cache
        seq_ids = torch.full((MQ_LEN,), seq_id, dtype=torch.int64, device=self.device)
        j_flat = torch.arange(K + 1, device=self.device, dtype=torch.int64).repeat_interleave(fan_t)
        keys = torch.stack([seq_ids, j_flat, forked_tokens], dim=1)
        self.tree_caches[seq_id] = (keys, spec_tokens)

    @torch.inference_mode()
    def _build_simple_cache(self, seq_id, recovery_token, draft_tokens, hidden_state, num_tokens, block_table_t):
        """Simple cache: pre-compute K draft tokens for the most likely next recovery tokens.
        For each position j (0..K), predict the next token via MTP at that position,
        then speculate K tokens from it. No tree decode needed.
        """
        K = self.K
        # After JIT, we have the last prenorm from position num_tokens-1+K
        # The "all accepted" scenario: bonus = target's argmax at pos K
        # We don't know what target will choose, but we can pre-compute from
        # the MTP's own predictions at each glue position

        # Run one more MTP step from last draft token to get "bonus prediction"
        last_draft = draft_tokens[-1]
        cur_pos = num_tokens - 1 + K
        prenorm, logits, bonus_pred = self._mtp_step(last_draft, hidden_state, cur_pos, block_table_t)

        # Pre-compute K tokens from bonus_pred (the most likely next recovery)
        next_draft = self.jit_speculate(bonus_pred, prenorm, num_tokens + K, block_table_t)

        # Store in cache: key = (seq_id, K, bonus_pred) → next_draft tokens
        key = torch.tensor([[seq_id, K, bonus_pred]], dtype=torch.int64, device=self.device)
        tokens = torch.tensor([next_draft], dtype=torch.int64, device=self.device)
        self.tree_caches[seq_id] = (key, tokens)

    def handle_prefill(self):
        meta = torch.zeros(3, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)
        seq_id, n, bt_len = meta.tolist()
        n, bt_len = int(n), int(bt_len)

        token_ids = torch.zeros(n, dtype=torch.int64, device=self.device)
        hidden_states = torch.zeros(n, self.hf_config.hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)
        positions = torch.zeros(n, dtype=torch.int64, device=self.device)
        block_table = torch.zeros(bt_len, dtype=torch.int32, device=self.device)

        dist.recv(token_ids, src=0, group=self.async_pg)
        dist.recv(hidden_states, src=0, group=self.async_pg)
        dist.recv(positions, src=0, group=self.async_pg)
        dist.recv(block_table, src=0, group=self.async_pg)

        # Build slot mapping and run MTP prefill
        slot_mapping = []
        for pos in positions.tolist():
            bidx = int(pos) // self.block_size
            boff = int(pos) % self.block_size
            slot = int(block_table[bidx]) * self.block_size + boff
            slot_mapping.append(slot)

        total_seqlen = int(positions[-1].item()) + 1
        cu_q = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        cu_k = torch.tensor([0, total_seqlen], dtype=torch.int32, device=self.device)
        slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)

        set_context(True, cu_q, cu_k, n, total_seqlen, slot_t, None, None)
        embeds = self.embed_tokens(token_ids)
        normed, prenorm = self.mtp_layer(embeds, hidden_states, positions)
        reset_context()

        self.last_hidden[int(seq_id)] = prenorm[-1:].clone()
        # Send ack
        ack = torch.ones(1, dtype=torch.int64, device=self.device)
        dist.send(ack, dst=0, group=self.async_pg)

    def handle_speculate(self):
        meta = torch.zeros(5, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)
        seq_id, accepted_len, recovery_token, num_tokens, bt_len = meta.tolist()
        seq_id, accepted_len, recovery_token, num_tokens, bt_len = int(seq_id), int(accepted_len), int(recovery_token), int(num_tokens), int(bt_len)

        hidden_state = torch.zeros(self.hf_config.hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)
        dist.recv(hidden_state, src=0, group=self.async_pg)
        block_table_t = torch.zeros(bt_len, dtype=torch.int32, device=self.device)
        dist.recv(block_table_t, src=0, group=self.async_pg)

        self.last_hidden[seq_id] = hidden_state.unsqueeze(0)

        draft_tokens, cache_hit = self.hit_cache_and_respond(
            seq_id, accepted_len, recovery_token, hidden_state, num_tokens, block_table_t)

        # Send draft tokens via NCCL (no logits!)
        resp = torch.tensor(draft_tokens, dtype=torch.int64, device=self.device)
        dist.send(resp, dst=0, group=self.async_pg)
        # Cache is populated by early_speculate (cmd=5) during next verify

    def handle_early_speculate(self, num_seqs):
        """Receive batched early hidden for all seqs from target.
        For each seq: norm → lm_head → topF → embed → MTP → cache."""

        # Receive per-seq metadata: [sid, nv, ntok, btlen, spos] * num_seqs
        meta = torch.zeros(num_seqs * 5, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)

        meta_list = meta.tolist()
        seq_infos = []
        total_nv = 0
        total_bt = 0
        for i in range(num_seqs):
            b = i * 5
            sid, nv, ntok, btlen, spos = meta_list[b], meta_list[b+1], meta_list[b+2], meta_list[b+3], meta_list[b+4]
            seq_infos.append((sid, nv, ntok, btlen, spos))
            total_nv += nv
            total_bt += btlen

        # Receive batched data
        early_hidden_all = torch.zeros(total_nv, self.hf_config.hidden_size,
                                       dtype=self.hf_config.torch_dtype, device=self.device)
        dist.recv(early_hidden_all, src=0, group=self.async_pg)
        all_bt = torch.zeros(total_bt, dtype=torch.int32, device=self.device)
        dist.recv(all_bt, src=0, group=self.async_pg)
        all_pos = torch.zeros(total_nv, dtype=torch.int64, device=self.device)
        dist.recv(all_pos, src=0, group=self.async_pg)

        # === Batched processing: all seqs' candidates in one MTP forward ===
        fan = self.F
        d = self.device

        # 1. Batch norm + lm_head + topk for all seqs at once
        normed_all = self.draft_model.model.norm(early_hidden_all)
        early_logits_all = self.compute_logits_all(normed_all)
        _, cands_all = torch.topk(early_logits_all, fan, dim=-1)  # (total_nv, fan)

        # 2. Batch expand: all positions x fan candidates (fully vectorized)
        # cands_all: (total_nv, fan), early_hidden_all: (total_nv, hidden)
        batch_cands = cands_all.reshape(-1)                           # (total_nv * fan,)
        batch_hidden = early_hidden_all.repeat_interleave(fan, dim=0) # (total_nv * fan, hidden)
        batch_pos = all_pos.repeat_interleave(fan)                    # (total_nv * fan,)
        batch_embeds = self.embed_tokens(batch_cands)

        # Build per-seq metadata for tree cache keys
        all_flat_acc_lens = []
        all_flat_sids = []
        per_seq_meta = []
        bt_off = 0
        grand_total = 0
        all_slot_list = []
        all_ctx_lens = []
        all_bt_rows = []
        for sid, nv, ntok, btlen, spos in seq_infos:
            total = nv * fan
            block_table = all_bt[bt_off:bt_off+btlen]
            flat_acc_lens = torch.arange(nv, device=d, dtype=torch.int64).repeat_interleave(fan)
            flat_sids = torch.full((total,), sid, dtype=torch.int64, device=d)
            all_flat_acc_lens.append(flat_acc_lens)
            all_flat_sids.append(flat_sids)

            if self.tree_decode and self.K > 1:
                max_bi = block_table.shape[0] - 1
                base_vpos = spos + nv
                per_seq_meta.append((total, base_vpos, max_bi, block_table, grand_total))
            else:
                # Vectorized slot computation (no Python loop)
                flat_pos = batch_pos[grand_total:grand_total+total]
                bi = (flat_pos // self.block_size).long()
                bo = (flat_pos % self.block_size).int()
                all_slot_list.append((block_table[bi] * self.block_size + bo).int())
                all_ctx_lens.append((flat_pos + 1).to(torch.int32))
                all_bt_rows.append(block_table.unsqueeze(0).expand(total, -1))

            grand_total += total
            bt_off += btlen

        batch_acc_lens = torch.cat(all_flat_acc_lens)
        batch_sids = torch.cat(all_flat_sids)

        if self.tree_decode and self.K > 1:
            # 3a. Batched tree decode: K steps for ALL seqs' candidates at once
            all_draft = []
            current_hidden = batch_hidden
            current_ids = batch_cands
            for depth in range(self.K):
                all_slots = []
                all_ctxs = []
                all_ropes = []
                all_bts = []
                for total_s, base_vpos, max_bi, block_table, goffset in per_seq_meta:
                    vpos = torch.arange(total_s, device=d, dtype=torch.int64) + base_vpos + depth * total_s
                    all_ropes.append(batch_pos[goffset:goffset+total_s] + depth)
                    bi = (vpos // self.block_size).long().clamp(0, max_bi)
                    bo = (vpos % self.block_size).int()
                    all_slots.append((block_table[bi] * self.block_size + bo).int())
                    all_ctxs.append((vpos + 1).to(torch.int32))
                    all_bts.append(block_table.unsqueeze(0).expand(total_s, -1))
                b_slots = torch.cat(all_slots)
                b_ctxs = torch.cat(all_ctxs)
                b_ropes = torch.cat(all_ropes)
                max_btl = max(bt.shape[1] for bt in all_bts)
                b_bt = torch.cat([F.pad(bt, (0, max_btl - bt.shape[1])) for bt in all_bts]).contiguous()

                set_context(False, slot_mapping=b_slots, context_lens=b_ctxs, block_tables=b_bt)
                cur_embeds = self.embed_tokens(current_ids)
                normed_out, _ = self.mtp_layer(cur_embeds, current_hidden, b_ropes)
                reset_context()
                logits = self.compute_mtp_logits(normed_out)
                next_tokens = logits.argmax(dim=-1)
                all_draft.append(next_tokens)
                current_hidden = normed_out
                current_ids = next_tokens

            spec_tokens = torch.stack(all_draft, dim=1)
            offset = 0
            for sid, nv, ntok, btlen, spos in seq_infos:
                total = nv * fan
                keys = torch.stack([batch_sids[offset:offset+total],
                                     batch_acc_lens[offset:offset+total],
                                     batch_cands[offset:offset+total]], dim=1)
                self.tree_caches[sid] = (keys, spec_tokens[offset:offset+total])
                offset += total
        else:
            # 3b. Batched chain: single MTP step for ALL seqs' candidates
            batch_slots = torch.cat(all_slot_list)
            batch_ctxs = torch.cat(all_ctx_lens)
            max_bt_len = max(bt.shape[1] for bt in all_bt_rows)
            batch_bt = torch.cat([F.pad(bt, (0, max_bt_len - bt.shape[1])) for bt in all_bt_rows]).contiguous()

            set_context(False, slot_mapping=batch_slots, context_lens=batch_ctxs, block_tables=batch_bt)
            normed_out, _ = self.mtp_layer(batch_embeds, batch_hidden, batch_pos)
            reset_context()
            logits = self.compute_mtp_logits(normed_out)
            draft_tokens = logits.argmax(dim=-1)

            # Split back per seq
            offset = 0
            for sid, nv, ntok, btlen, spos in seq_infos:
                total = nv * fan
                keys = torch.stack([batch_sids[offset:offset+total],
                                     batch_acc_lens[offset:offset+total],
                                     batch_cands[offset:offset+total]], dim=1)
                self.tree_caches[sid] = (keys, draft_tokens[offset:offset+total].unsqueeze(1))
                offset += total

    def handle_cache_lookup(self):
        """Batched cache lookup: n_seqs from _cmd_buf[1], then receive lookup data."""
        n_seqs = int(self._cmd_buf[1].item())
        if not hasattr(self, '_lookup_recv_buf'):
            self._lookup_recv_buf = torch.zeros(512 * 3, dtype=torch.int64, device=self.device)
        lookup = self._lookup_recv_buf[:n_seqs * 3]
        if n_seqs > 0:
            dist.recv(lookup, src=0, group=self.async_pg)
        result = torch.zeros(n_seqs * self.K, dtype=torch.int64, device=self.device)
        lookup_list = lookup.tolist()
        for i in range(n_seqs):
            seq_id = int(lookup_list[i*3])
            accepted_len = int(lookup_list[i*3+1])
            recovery_token = int(lookup_list[i*3+2])
            hit = False
            if seq_id in self.tree_caches:
                cache_keys, cache_tokens = self.tree_caches[seq_id]
                if self.tree_decode:
                    rk = torch.tensor([[seq_id, accepted_len, recovery_token]],
                                       dtype=torch.int64, device=self.device)
                    match = torch.all(rk == cache_keys, dim=1)
                    if match.any():
                        mi = match.float().argmax().item()
                        k = min(self.K, cache_tokens.shape[1])
                        result[i*self.K:i*self.K+k] = cache_tokens[mi, :k]
                        hit = True
                else:
                    cur_token = recovery_token
                    for step in range(self.K):
                        rk = torch.tensor([[seq_id, accepted_len + step, cur_token]],
                                           dtype=torch.int64, device=self.device)
                        match = torch.all(rk == cache_keys, dim=1)
                        if match.any():
                            mi = match.float().argmax().item()
                            d_tok = cache_tokens[mi, 0].item()
                            result[i*self.K + step] = d_tok
                            cur_token = d_tok
                            if step == 0:
                                hit = True
                        else:
                            break
            if hit:
                MTPDraftRunner._hit += 1
            else:
                MTPDraftRunner._miss += 1
        dist.send(result, dst=0, group=self.async_pg)

    def handle_cache_update(self):
        """Receive pre-computed next draft token from target and store in cache."""
        meta = torch.zeros(3, dtype=torch.int64, device=self.device)  # [seq_id, accepted_len, recovery_token]
        dist.recv(meta, src=0, group=self.async_pg)
        m = meta.tolist()
        seq_id, acc_len, rec_tok = int(m[0]), int(m[1]), int(m[2])
        next_draft = torch.zeros(self.K, dtype=torch.int64, device=self.device)
        dist.recv(next_draft, src=0, group=self.async_pg)
        key = torch.tensor([[seq_id, acc_len, rec_tok]], dtype=torch.int64, device=self.device)
        self.tree_caches[seq_id] = (key, next_draft.unsqueeze(0))

    def handle_cleanup(self):
        meta = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)
        seq_id = int(meta[0])
        if seq_id in self.last_hidden:
            del self.last_hidden[seq_id]
        self._reset_tree_cache(seq_id)
        ack = torch.ones(1, dtype=torch.int64, device=self.device)
        dist.send(ack, dst=0, group=self.async_pg)

    def draft_loop(self):
        print("[MTPDraftRunner] Starting draft loop", flush=True)
        while True:
            dist.recv(self._cmd_buf, src=0, group=self.async_pg)
            cmd = self._cmd_buf[0].tolist()
            if cmd == 0:
                self.handle_speculate()
            elif cmd == 1:
                self.handle_prefill()
            elif cmd == 3:
                self.handle_cleanup()
            elif cmd == 4:  # cache_update from target
                self.handle_cache_update()
            elif cmd == 5:  # early_speculate: target sent [cmd, num_seqs] in _cmd_buf
                num_seqs = self._cmd_buf[1].tolist()
                self.handle_early_speculate(num_seqs)
            elif cmd == 6:  # cache_lookup: lightweight hit query
                self.handle_cache_lookup()
            elif cmd == 2:
                total = MTPDraftRunner._hit + MTPDraftRunner._miss
                rate = MTPDraftRunner._hit / total * 100 if total else 0
                print(f"[MTPDraftRunner] Exiting. Cache hit: {MTPDraftRunner._hit}/{total} ({rate:.1f}%)", flush=True)


class EAGLEDraftModel(torch.nn.Module):
    """EAGLE draft model for async SSD on separate GPU."""
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config):
        super().__init__()
        from nanovllm.models.eagle import EAGLEAttention, EAGLEDecoderLayer
        from nanovllm.layers.linear import ReplicatedLinear
        hidden_size = config.hidden_size
        # Use 'model' submodule for embed_tokens/norm to match target checkpoint names
        # (target checkpoint has "model.embed_tokens", "model.norm")
        self.model = torch.nn.Module()
        self.model.embed_tokens = VocabParallelEmbedding(config.vocab_size, hidden_size, tp_size=1)
        self.model.norm = RMSNorm(hidden_size, eps=config.rms_norm_eps)
        model_type = getattr(config, 'model_type', '')
        fc_bias = model_type in ('qwen2',)
        self.fc = ReplicatedLinear(hidden_size * 2, hidden_size, bias=fc_bias, tp_size=1)
        self.layers = torch.nn.ModuleList([EAGLEDecoderLayer(config, tp_size=1)])
        self.lm_head = ParallelLMHead(config.vocab_size, hidden_size, tp_size=1)

    def forward(self, input_ids, positions, target_hidden):
        token_embeds = self.model.embed_tokens(input_ids)
        hidden = self.fc(torch.cat([token_embeds, target_hidden], dim=-1))
        hidden = self.layers[0](positions, hidden)
        return hidden


class EAGLEDraftRunner:
    """Async EAGLE draft runner for SSD on separate GPU."""

    _hit = 0
    _miss = 0

    def __init__(self, config: Config, rank: int, init_q=None):
        self.config = config
        hf_config = config.hf_config
        self.hf_config = hf_config
        self.block_size = config.kvcache_block_size
        self.K = config.num_speculative_tokens
        self.F = config.async_fan_out
        self.rank = rank

        self.device_backend = get_device_backend()
        self.device_backend.set_device(config.draft_gpu)
        self.device = f"cuda:{config.draft_gpu}"

        print(f"[EAGLEDraft rank={rank}] calling init_process_group world={config.num_gpus}...", flush=True)
        dist.init_process_group("nccl", f"tcp://localhost:{os.environ.get('NCCL_PORT', '2333')}",
                                world_size=config.num_gpus, rank=rank)
        tp_ranks = list(range(config.tensor_parallel_size))
        self.tp_group = dist.new_group(tp_ranks)
        self.async_pg = dist.new_group([0, config.draft_rank])

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)

        # Load EAGLE model
        # Use draft_hf_config for EAGLE architecture (num_kv_heads=num_heads, etc)
        eagle_config = config.draft_hf_config if config.draft_hf_config is not None else hf_config
        self.draft_model = EAGLEDraftModel(eagle_config)
        # Load target model weights first (embed_tokens, lm_head, norm)
        _load_draft_model(self.draft_model, config.model)
        # Then load EAGLE-specific weights (fc, decoder layer) — overwrites layers.0.* with EAGLE's
        _load_draft_model(self.draft_model, config.draft_model)

        self.embed_tokens = self.draft_model.model.embed_tokens
        self.lm_head = self.draft_model.lm_head

        self._warmup_and_allocate_kv_cache(eagle_config)
        if init_q is not None:
            init_q.put(config.num_kvcache_blocks)
            init_q.close()

        self._capture_eagle_graph(eagle_config)

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        self.tree_decode = config.ssd_tree_decode
        self._reset_tree_cache()
        self.last_hidden = {}
        self._cmd_buf = torch.zeros(2, dtype=torch.int64, device=self.device)

        print(f"[EAGLEDraftRunner] Initialized on GPU {config.draft_gpu}, K={self.K}, F={self.F}", flush=True)
        print(f"[EAGLEDraftRunner] norm.weight norm={self.draft_model.model.norm.weight.data.norm().item():.4f}", flush=True)
        print(f"[EAGLEDraftRunner] lm_head.weight norm={self.lm_head.weight.data.norm().item():.4f}", flush=True)
        print(f"[EAGLEDraftRunner] fc.weight norm={self.draft_model.fc.weight.data.norm().item():.4f}", flush=True)

    def _warmup_and_allocate_kv_cache(self, eagle_config):
        hf_config = self.hf_config
        config = self.config
        torch.cuda.empty_cache()
        # EAGLE KV heads: Qwen2 uses full MHA, Llama uses GQA
        num_kv_heads = getattr(eagle_config, 'num_key_value_heads', eagle_config.num_attention_heads)
        head_dim = getattr(eagle_config, "head_dim", eagle_config.hidden_size // eagle_config.num_attention_heads)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        block_bytes = 2 * 1 * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        max_tokens = config.max_num_seqs * (config.max_model_len // self.block_size + 1)
        avail_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // max(block_bytes, 1)
        num_blocks = min(max(avail_blocks, 1), max_tokens)
        config.num_kvcache_blocks = num_blocks
        self.eagle_kv_cache = torch.empty(
            2, 1, num_blocks, self.block_size, num_kv_heads, head_dim,
            device=self.device, dtype=hf_config.torch_dtype,
        )
        for module in self.draft_model.layers[0].modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.eagle_kv_cache[0, 0]
                module.v_cache = self.eagle_kv_cache[1, 0]

    @torch.inference_mode()
    def _capture_eagle_graph(self, eagle_config):
        hf = self.hf_config
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size + 10
        d = self.device
        g = {}
        g["input_id"] = torch.zeros(1, dtype=torch.int64, device=d)
        g["pos"] = torch.zeros(1, dtype=torch.int64, device=d)
        g["hidden"] = torch.zeros(1, hf.hidden_size, dtype=hf.torch_dtype, device=d)
        g["slot"] = torch.zeros(1, dtype=torch.int32, device=d)
        g["ctx_len"] = torch.zeros(1, dtype=torch.int32, device=d)
        g["bt"] = torch.zeros(1, max_num_blocks, dtype=torch.int32, device=d)
        g["out"] = torch.zeros(1, hf.hidden_size, dtype=hf.torch_dtype, device=d)

        # Warmup
        set_context(False, slot_mapping=g["slot"], context_lens=g["ctx_len"], block_tables=g["bt"])
        out = self.draft_model(g["input_id"], g["pos"], g["hidden"])
        g["out"].copy_(out)
        reset_context()

        # Capture
        self._eagle_graph = torch.cuda.CUDAGraph()
        set_context(False, slot_mapping=g["slot"], context_lens=g["ctx_len"], block_tables=g["bt"])
        with torch.cuda.graph(self._eagle_graph):
            out = self.draft_model(g["input_id"], g["pos"], g["hidden"])
            g["out"].copy_(out)
        reset_context()
        torch.cuda.synchronize()
        self._g = g

    @torch.inference_mode()
    def _eagle_step(self, token_id, hidden_state, cur_pos, block_table_t):
        """Single EAGLE forward step using CUDA graph."""
        block_idx = cur_pos // self.block_size
        bt_len = block_table_t.shape[0]
        block_offset = cur_pos % self.block_size
        slot = block_table_t[min(block_idx, bt_len - 1)] * self.block_size + block_offset

        g = self._g
        g["input_id"][0] = token_id
        g["pos"][0] = cur_pos
        if hidden_state.dim() == 1:
            g["hidden"][0].copy_(hidden_state)
        else:
            g["hidden"].copy_(hidden_state)
        g["slot"][0] = slot
        g["ctx_len"][0] = cur_pos + 1
        g["bt"][0, :bt_len] = block_table_t

        set_context(False, slot_mapping=g["slot"], context_lens=g["ctx_len"], block_tables=g["bt"])
        self._eagle_graph.replay()
        reset_context()

        logits = F.linear(g["out"], self.lm_head.weight)
        next_token = logits.argmax(dim=-1).item()
        return g["out"][0].clone(), logits.squeeze(0), next_token

    def _reset_tree_cache(self, seq_id=None):
        if seq_id is None:
            self.tree_caches = {}
        elif seq_id in self.tree_caches:
            del self.tree_caches[seq_id]

    def jit_speculate(self, recovery_token, hidden_state, num_tokens, block_table_t):
        K = self.K
        draft_tokens = []
        cur_token = recovery_token
        cur_pos = num_tokens - 1
        cur_hidden = hidden_state
        for i in range(K):
            out, logits, d_token = self._eagle_step(cur_token, cur_hidden, cur_pos, block_table_t)
            draft_tokens.append(d_token)
            cur_hidden = out
            cur_token = d_token
            cur_pos += 1
        return draft_tokens

    def hit_cache_and_respond(self, seq_id, accepted_len, recovery_token, hidden_state, num_tokens, block_table_t):
        if seq_id in self.tree_caches:
            cache_keys, cache_tokens = self.tree_caches[seq_id]
            request_key = torch.tensor([[seq_id, accepted_len, recovery_token]], dtype=torch.int64, device=self.device)
            match = torch.all(request_key.unsqueeze(1) == cache_keys.unsqueeze(0), dim=2).squeeze(0)
            if match.any():
                idx = match.float().argmax().item()
                EAGLEDraftRunner._hit += 1
                return cache_tokens[idx].tolist(), True
        EAGLEDraftRunner._miss += 1
        tokens = self.jit_speculate(recovery_token, hidden_state, num_tokens, block_table_t)
        return tokens, False

    def handle_prefill(self):
        """Receive prefill data and populate EAGLE KV cache."""
        meta = torch.zeros(3, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)
        seq_id, n, bt_len = meta.tolist()
        n, bt_len = int(n), int(bt_len)

        token_ids = torch.zeros(n, dtype=torch.int64, device=self.device)
        hidden_states = torch.zeros(n, self.hf_config.hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)
        positions = torch.zeros(n, dtype=torch.int64, device=self.device)
        block_table = torch.zeros(bt_len, dtype=torch.int32, device=self.device)

        dist.recv(token_ids, src=0, group=self.async_pg)
        dist.recv(hidden_states, src=0, group=self.async_pg)
        dist.recv(positions, src=0, group=self.async_pg)
        dist.recv(block_table, src=0, group=self.async_pg)

        slot_mapping = []
        for pos in positions.tolist():
            bidx = int(pos) // self.block_size
            boff = int(pos) % self.block_size
            slot = int(block_table[bidx]) * self.block_size + boff
            slot_mapping.append(slot)

        total_seqlen = int(positions[-1].item()) + 1
        cu_q = torch.tensor([0, n], dtype=torch.int32, device=self.device)
        cu_k = torch.tensor([0, total_seqlen], dtype=torch.int32, device=self.device)
        slot_t = torch.tensor(slot_mapping, dtype=torch.int32, device=self.device)

        # Run EAGLE prefill (populates KV cache)
        set_context(True, cu_q, cu_k, n, total_seqlen, slot_t, None, None)
        eagle_out = self.draft_model(token_ids, positions, hidden_states)
        reset_context()

        self.last_hidden[int(seq_id)] = eagle_out[-1:].clone()
        ack = torch.ones(1, dtype=torch.int64, device=self.device)
        dist.send(ack, dst=0, group=self.async_pg)

    def handle_early_speculate(self, num_seqs=None):
        """Receive early hidden from target, build tree cache with EAGLE."""
        if num_seqs is None:
            ns_buf = torch.zeros(1, dtype=torch.int64, device=self.device)
            dist.recv(ns_buf, src=0, group=self.async_pg)
            num_seqs = int(ns_buf[0].item())

        meta = torch.zeros(num_seqs * 5, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)

        seq_infos = []
        total_nv = 0
        total_bt = 0
        for i in range(num_seqs):
            b = i * 5
            sid, nv, ntok, btlen, spos = meta_list[b], meta_list[b+1], meta_list[b+2], meta_list[b+3], meta_list[b+4]
            seq_infos.append((sid, nv, ntok, btlen, spos))
            total_nv += nv
            total_bt += btlen

        early_hidden_all = torch.zeros(total_nv, self.hf_config.hidden_size,
                                        dtype=self.hf_config.torch_dtype, device=self.device)
        dist.recv(early_hidden_all, src=0, group=self.async_pg)
        all_bt = torch.zeros(total_bt, dtype=torch.int32, device=self.device)
        dist.recv(all_bt, src=0, group=self.async_pg)
        all_pos = torch.zeros(total_nv, dtype=torch.int64, device=self.device)
        dist.recv(all_pos, src=0, group=self.async_pg)

        # === Batched processing: all seqs' candidates in one EAGLE forward ===
        fan = self.F
        d = self.device

        # 1. Batch norm + lm_head + topk
        normed_all = self.draft_model.model.norm(early_hidden_all)
        early_logits_all = F.linear(normed_all, self.lm_head.weight)
        _, cands_all = torch.topk(early_logits_all, fan, dim=-1)

        # 2. Build per-seq expanded tensors, then concatenate
        all_flat_cands = []
        all_flat_hidden = []
        all_flat_acc_lens = []
        all_flat_pos = []
        all_flat_sids = []
        all_slot_list = []
        all_ctx_lens = []
        all_bt_rows = []
        per_seq_meta = []

        h_off = 0
        bt_off = 0
        grand_total = 0
        for sid, nv, ntok, btlen, spos in seq_infos:
            cands = cands_all[h_off:h_off+nv]
            normed_h = normed_all[h_off:h_off+nv]
            block_table = all_bt[bt_off:bt_off+btlen]
            verify_pos = all_pos[h_off:h_off+nv]

            total = nv * fan
            all_flat_cands.append(cands.reshape(-1))
            all_flat_hidden.append(normed_h.repeat_interleave(fan, dim=0))
            all_flat_acc_lens.append(torch.arange(nv, device=d, dtype=torch.int64).repeat_interleave(fan))
            all_flat_pos.append(verify_pos.repeat_interleave(fan))
            all_flat_sids.append(torch.full((total,), sid, dtype=torch.int64, device=d))

            if self.tree_decode and self.K > 1:
                per_seq_meta.append((total, spos + nv, block_table.shape[0] - 1, block_table, grand_total))
            else:
                for p in verify_pos.repeat_interleave(fan).tolist():
                    p = int(p)
                    all_slot_list.append(int(block_table[p // self.block_size]) * self.block_size + p % self.block_size)
                all_ctx_lens.append((verify_pos.repeat_interleave(fan) + 1).to(torch.int32))
                all_bt_rows.append(block_table.unsqueeze(0).expand(total, -1))

            grand_total += total
            h_off += nv
            bt_off += btlen

        batch_cands = torch.cat(all_flat_cands)
        batch_hidden = torch.cat(all_flat_hidden)
        batch_acc_lens = torch.cat(all_flat_acc_lens)
        batch_pos = torch.cat(all_flat_pos)
        batch_sids = torch.cat(all_flat_sids)

        if self.tree_decode and self.K > 1:
            # 3a. Batched tree decode
            all_draft = []
            current_hidden = batch_hidden
            current_ids = batch_cands
            for depth in range(self.K):
                all_slots = []
                all_ctxs = []
                all_ropes = []
                all_bts = []
                for total_s, base_vpos, max_bi, block_table, offset in per_seq_meta:
                    vpos = torch.arange(total_s, device=d, dtype=torch.int64) + base_vpos + depth * total_s
                    all_ropes.append(batch_pos[offset:offset+total_s] + depth)
                    bi = (vpos // self.block_size).long().clamp(0, max_bi)
                    bo = (vpos % self.block_size).int()
                    all_slots.append((block_table[bi] * self.block_size + bo).int())
                    all_ctxs.append((vpos + 1).to(torch.int32))
                    all_bts.append(block_table.unsqueeze(0).expand(total_s, -1))
                b_slots = torch.cat(all_slots)
                b_ctxs = torch.cat(all_ctxs)
                b_ropes = torch.cat(all_ropes)
                max_btl = max(bt.shape[1] for bt in all_bts)
                b_bt = torch.cat([F.pad(bt, (0, max_btl - bt.shape[1])) for bt in all_bts]).contiguous()

                set_context(False, slot_mapping=b_slots, context_lens=b_ctxs, block_tables=b_bt)
                eagle_out = self.draft_model(current_ids, b_ropes, current_hidden)
                reset_context()
                logits = F.linear(eagle_out, self.lm_head.weight)
                next_tokens = logits.argmax(dim=-1)
                all_draft.append(next_tokens)
                current_hidden = eagle_out
                current_ids = next_tokens

            spec_tokens = torch.stack(all_draft, dim=1)
            offset = 0
            for sid, nv, ntok, btlen, spos in seq_infos:
                total = nv * fan
                keys = torch.stack([batch_sids[offset:offset+total], batch_acc_lens[offset:offset+total],
                                     batch_cands[offset:offset+total]], dim=1)
                self.tree_caches[sid] = (keys, spec_tokens[offset:offset+total])
                offset += total
        else:
            # 3b. Batched chain: single EAGLE step
            b_slots = torch.tensor(all_slot_list, dtype=torch.int32, device=d)
            b_ctxs = torch.cat(all_ctx_lens)
            max_btl = max(bt.shape[1] for bt in all_bt_rows)
            b_bt = torch.cat([F.pad(bt, (0, max_btl - bt.shape[1])) for bt in all_bt_rows]).contiguous()

            set_context(False, slot_mapping=b_slots, context_lens=b_ctxs, block_tables=b_bt)
            eagle_out = self.draft_model(batch_cands, batch_pos, batch_hidden)
            reset_context()
            logits = F.linear(eagle_out, self.lm_head.weight)
            draft_tokens = logits.argmax(dim=-1)

            offset = 0
            for sid, nv, ntok, btlen, spos in seq_infos:
                total = nv * fan
                keys = torch.stack([batch_sids[offset:offset+total], batch_acc_lens[offset:offset+total],
                                     batch_cands[offset:offset+total]], dim=1)
                self.tree_caches[sid] = (keys, draft_tokens[offset:offset+total].unsqueeze(1))
                offset += total

        # Push tree cache results to target in ONE NCCL send
        # Format: flat tensor [num_seqs, (sid, n_entries, K, keys_flat, tokens_flat) × num_seqs]
        parts = [torch.tensor([num_seqs], dtype=torch.int64, device=self.device)]
        for sid, nv, ntok, btlen, spos in seq_infos:
            if sid in self.tree_caches:
                keys, tokens = self.tree_caches[sid]
                n = keys.shape[0]
                K_a = tokens.shape[1] if tokens.dim() > 1 else 1
                parts.append(torch.tensor([sid, n, K_a], dtype=torch.int64, device=self.device))
                parts.append(keys.reshape(-1).to(torch.int64))
                parts.append(tokens.reshape(-1).to(torch.int64))
            else:
                parts.append(torch.tensor([sid, 0, 0], dtype=torch.int64, device=self.device))
        push_buf = torch.cat(parts)
        # Send size first, then data
        dist.send(torch.tensor([push_buf.shape[0]], dtype=torch.int64, device=self.device), dst=0, group=self.async_pg)
        dist.send(push_buf, dst=0, group=self.async_pg)

    def handle_cache_lookup(self):
        """Batched cache lookup: receive [n_seqs, sid0, al0, rt0, ...], return N*K tokens."""
        # First receive n_seqs
        ns_buf = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(ns_buf, src=0, group=self.async_pg)
        n_seqs = int(ns_buf[0].item())
        # Then receive lookup data
        lookup_data = torch.zeros(n_seqs * 3, dtype=torch.int64, device=self.device)
        if n_seqs > 0:
            dist.recv(lookup_data, src=0, group=self.async_pg)
        result = torch.zeros(n_seqs * self.K, dtype=torch.int64, device=self.device)

        for i in range(n_seqs):
            seq_id = int(lookup_data[i*3].item())
            accepted_len = int(lookup_data[i*3 + 1].item())
            recovery_token = int(lookup_data[i*3 + 2].item())

            hit = False
            if seq_id in self.tree_caches:
                cache_keys, cache_tokens = self.tree_caches[seq_id]
                if self.tree_decode:
                    request_key = torch.tensor([[seq_id, accepted_len, recovery_token]],
                                                dtype=torch.int64, device=self.device)
                    match = torch.all(request_key == cache_keys, dim=1)
                    if match.any():
                        idx = match.float().argmax().item()
                        k = min(self.K, cache_tokens.shape[1])
                        result[i*self.K:i*self.K+k] = cache_tokens[idx, :k]
                        hit = True
                else:
                    cur_token = recovery_token
                    for step in range(self.K):
                        request_key = torch.tensor([[seq_id, accepted_len + step, cur_token]],
                                                    dtype=torch.int64, device=self.device)
                        match = torch.all(request_key == cache_keys, dim=1)
                        if match.any():
                            idx = match.float().argmax().item()
                            d = cache_tokens[idx, 0].item()
                            result[i*self.K + step] = d
                            cur_token = d
                            if step == 0:
                                hit = True
                        else:
                            break
            if hit:
                EAGLEDraftRunner._hit += 1
            else:
                EAGLEDraftRunner._miss += 1

        dist.send(result, dst=0, group=self.async_pg)

    def handle_speculate(self):
        meta = torch.zeros(5, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)
        seq_id, accepted_len, recovery_token, num_tokens, bt_len = (int(x) for x in meta.tolist())
        hidden_state = torch.zeros(self.hf_config.hidden_size, dtype=self.hf_config.torch_dtype, device=self.device)
        dist.recv(hidden_state, src=0, group=self.async_pg)
        block_table_t = torch.zeros(bt_len, dtype=torch.int32, device=self.device)
        dist.recv(block_table_t, src=0, group=self.async_pg)
        self.last_hidden[seq_id] = hidden_state.unsqueeze(0)
        draft_tokens, _ = self.hit_cache_and_respond(seq_id, accepted_len, recovery_token, hidden_state, num_tokens, block_table_t)
        resp = torch.tensor(draft_tokens, dtype=torch.int64, device=self.device)
        dist.send(resp, dst=0, group=self.async_pg)

    def handle_cleanup(self):
        meta = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)
        seq_id = int(meta[0])
        if seq_id in self.last_hidden:
            del self.last_hidden[seq_id]
        self._reset_tree_cache(seq_id)
        ack = torch.ones(1, dtype=torch.int64, device=self.device)
        dist.send(ack, dst=0, group=self.async_pg)

    def draft_loop(self):
        print("[EAGLEDraftRunner] Starting draft loop", flush=True)
        while True:
            dist.recv(self._cmd_buf, src=0, group=self.async_pg)
            cmd = int(self._cmd_buf[0].tolist())
            if cmd == 0:
                self.handle_speculate()
            elif cmd == 1:
                self.handle_prefill()
            elif cmd == 3:
                self.handle_cleanup()
            elif cmd == 5:
                num_seqs = self._cmd_buf[1].tolist()
                self.handle_early_speculate(num_seqs)
            elif cmd == 6:
                self.handle_cache_lookup()
            elif cmd == 2:
                total = EAGLEDraftRunner._hit + EAGLEDraftRunner._miss
                rate = EAGLEDraftRunner._hit / total * 100 if total else 0
                print(f"[EAGLEDraftRunner] Exiting. Cache hit: {EAGLEDraftRunner._hit}/{total} ({rate:.1f}%)", flush=True)
                break


def launch_draft_runner(config, rank, init_q=None):
    import sys
    print(f"[Draft rank={rank}] launch_draft_runner starting", flush=True)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    try:
        from nanovllm.utils.device import DeviceBackend
        DeviceBackend.initialize()
        print(f"[Draft rank={rank}] DeviceBackend initialized, creating runner...", flush=True)
        if config.eagle_async:
            runner = EAGLEDraftRunner(config, rank, init_q=init_q)
        else:
            runner = MTPDraftRunner(config, rank, init_q=init_q)
        runner.draft_loop()
        dist.destroy_process_group()
    except Exception as e:
        print(f"[Draft rank={rank}] FATAL ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
