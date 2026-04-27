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

        # CUDA graph capture for single-token MTP decode.
        # Pangu was historically skipped because the MoE topk dispatch had
        # dynamic shapes; now that the BMM path uses fixed [N*top_k, ...]
        # gathers and the c003a62 sink-attention fix made attention
        # capturable, capture is safe for Pangu single-token decode too.
        try:
            self._capture_mtp_graph()
        except Exception as exc:
            print(f"[MTPDraftRunner] CUDA graph capture failed ({exc!r}); falling back to eager.", flush=True)
            self._g = None

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        self.tree_decode = config.ssd_tree_decode
        self._reset_tree_cache()
        self.last_hidden = {}
        # Pre-allocate recv buffers
        self._cmd_buf = torch.zeros(4, dtype=torch.int64, device=self.device)

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
        """Single MTP forward step. Uses the pre-captured CUDA graph when
        available (MiMo etc.); falls back to eager for models where the MoE
        path blocks graph capture (PanGu). Returns (prenorm, logits, next_token).
        """
        block_idx = cur_pos // self.block_size
        bt_len = block_table_t.shape[0]
        block_offset = cur_pos % self.block_size
        slot = block_table_t[min(block_idx, bt_len - 1)] * self.block_size + block_offset

        if getattr(self, "_g", None) is not None:
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

        # Eager fallback (PanGu / any model that skipped graph capture)
        dev = self.device
        input_id_t = torch.tensor([token_id], dtype=torch.int64, device=dev)
        pos_t = torch.tensor([cur_pos], dtype=torch.int64, device=dev)
        hidden = hidden_state.unsqueeze(0) if hidden_state.dim() == 1 else hidden_state
        slot_t = torch.tensor([int(slot.item()) if torch.is_tensor(slot) else int(slot)],
                              dtype=torch.int32, device=dev)
        ctx_len_t = torch.tensor([cur_pos + 1], dtype=torch.int32, device=dev)
        bt_t = block_table_t.to(torch.int32).reshape(1, -1)
        set_context(False, slot_mapping=slot_t, context_lens=ctx_len_t, block_tables=bt_t)
        embeds = self.embed_tokens(input_id_t)
        mtp_normed, mtp_prenorm = self.mtp_layer(embeds, hidden, pos_t)
        reset_context()
        logits = self.compute_mtp_logits(mtp_normed)
        d_token = logits.argmax(dim=-1).item()
        return mtp_prenorm[0].clone(), logits.squeeze(0), d_token

    @torch.inference_mode()
    def jit_speculate(self, recovery_token, hidden_state, num_tokens, block_table_t):
        K = self.K
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

        # Build per-seq metadata using Python lists to avoid per-seq GPU tensor creation
        acc_lens_list = []
        sids_list = []
        per_seq_meta = []
        bt_off = 0
        grand_total = 0
        all_slot_list = []
        all_ctx_lens = []
        all_bt_rows = []
        for sid, nv, ntok, btlen, spos in seq_infos:
            total = nv * fan
            block_table = all_bt[bt_off:bt_off+btlen]
            for j in range(nv):
                for _ in range(fan):
                    acc_lens_list.append(j)
                    sids_list.append(sid)

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

        batch_acc_lens = torch.tensor(acc_lens_list, dtype=torch.int64, device=d)
        batch_sids = torch.tensor(sids_list, dtype=torch.int64, device=d)

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

        # Push tree cache to target (same protocol as EAGLE)
        d = self.device
        parts = [torch.tensor([num_seqs], dtype=torch.int64, device=d)]
        for sid, nv, ntok, btlen, spos in seq_infos:
            if sid in self.tree_caches:
                keys, tokens = self.tree_caches[sid]
                n = keys.shape[0]
                K_a = tokens.shape[1] if tokens.dim() > 1 else 1
                parts.append(torch.tensor([sid, n, K_a], dtype=torch.int64, device=d))
                parts.append(keys.reshape(-1).to(torch.int64))
                parts.append(tokens.reshape(-1).to(torch.int64))
            else:
                parts.append(torch.tensor([sid, 0, 0], dtype=torch.int64, device=d))
        push_buf = torch.cat(parts)
        dist.send(torch.tensor([push_buf.shape[0]], dtype=torch.int64, device=d), dst=0, group=self.async_pg)
        dist.send(push_buf, dst=0, group=self.async_pg)

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
        profile_path = os.environ.get("DRAFT_PROFILE_PATH")
        prof = None
        if profile_path:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=True,
            )
            prof.__enter__()
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
                if prof:
                    prof.__exit__(None, None, None)
                    prof.export_chrome_trace(profile_path)
                    print(f"[MTPDraftRunner] Draft trace saved to: {profile_path}", flush=True)
                break


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
        # EAGLE-1 checkpoints vary on whether they ship an fc bias (Qwen2 and
        # Vicuna do, some Llama variants don't). Always allocate — when the
        # checkpoint has no bias, the zero-init makes the add a no-op.
        self.fc = ReplicatedLinear(hidden_size * 2, hidden_size, bias=True, tp_size=1)
        self.layers = torch.nn.ModuleList([EAGLEDecoderLayer(config, tp_size=1)])
        self.lm_head = ParallelLMHead(config.vocab_size, hidden_size, tp_size=1)

    def forward(self, input_ids, positions, target_hidden):
        token_embeds = self.model.embed_tokens(input_ids)
        hidden = self.fc(torch.cat([token_embeds, target_hidden], dim=-1))
        hidden = self.layers[0](positions, hidden)
        return hidden


# Eagle3DraftModel removed: now uses Eagle3Model from nanovllm.models.eagle



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
        eagle_config = config.draft_hf_config if config.draft_hf_config is not None else hf_config
        self.eagle3 = getattr(config, 'eagle3', False)
        if self.eagle3:
            from nanovllm.models.eagle import Eagle3Model
            self.draft_model = Eagle3Model(hf_config, eagle_config, tp_size=1)
            # load_weights handles name remapping and skips mismatched weights;
            # first call loads target embed_tokens, second loads EAGLE-3 weights + d2t
            self.draft_model.load_weights(config.model)
            self.draft_model.load_weights(config.draft_model)
            self.d2t = self.draft_model.d2t
        else:
            self.draft_model = EAGLEDraftModel(eagle_config)
            _load_draft_model(self.draft_model, config.model)
            _load_draft_model(self.draft_model, config.draft_model)

        self.embed_tokens = self.draft_model.model.embed_tokens
        self.lm_head = self.draft_model.lm_head

        self._warmup_and_allocate_kv_cache(eagle_config)
        if init_q is not None:
            init_q.put(config.num_kvcache_blocks)
            init_q.close()

        if not self.eagle3:
            self._capture_eagle_graph(eagle_config)
            self._capture_batched_eagle_graph()
        else:
            self._batched_eagle_graphs = {}
            self._eagle_graph = None
            self._g = None
            self._glue_graph = None
            self._glue_g = None

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        self.tree_decode = config.ssd_tree_decode
        self._reset_tree_cache()
        self.last_hidden = {}
        self._cmd_buf = torch.zeros(4, dtype=torch.int64, device=self.device)

        # Pre-allocate recv buffers to avoid per-call torch.zeros
        max_seqs = config.max_num_seqs
        max_nv = (self.K + 1) * max_seqs
        max_bt = (config.max_model_len // self.block_size + 2) * max_seqs
        d = self.device
        # Payload buf: header(1) + meta(5*seqs + bt + pos) + hidden_as_int64
        hidden_mult = 3 if self.eagle3 else 1
        hidden_i64_size = max_nv * hf_config.hidden_size * hidden_mult * hf_config.torch_dtype.itemsize // 8
        self._meta_buf = torch.zeros(1 + max_seqs * 5 + max_bt + max_nv + hidden_i64_size, dtype=torch.int64, device=d)
        self._bt_buf = torch.zeros(max_bt, dtype=torch.int32, device=d)
        self._pos_buf = torch.zeros(max_nv, dtype=torch.int64, device=d)
        # Precompute repeat index for fan-out expand (avoids repeat_interleave kernel)
        max_total = max_nv * self.F
        self._fan_idx = torch.arange(max_nv, device=d).repeat_interleave(self.F)[:max_total]
        # Tree decode constants for bs=1 fast path (no per-call torch.arange)
        nv1 = self.K + 1              # num verify positions for single seq
        ts1 = nv1 * self.F            # total entries = nv * fan
        K = self.K
        self._ts1 = ts1
        self._base_arange = torch.arange(ts1, device=d, dtype=torch.int64)       # [0..ts1-1]
        self._depth_offsets = torch.arange(K, device=d, dtype=torch.int64) * ts1  # [0, ts1, ...]
        self._depth_range = torch.arange(K, device=d, dtype=torch.int64)          # [0, 1, ..., K-1]
        # Pre-allocated [K, ts1] buffers for tree decode (filled in-place)
        self._vpos_buf = torch.zeros(K, ts1, dtype=torch.int64, device=d)
        self._ropes_buf = torch.zeros(K, ts1, dtype=torch.int64, device=d)
        self._slots_buf = torch.zeros(K, ts1, dtype=torch.int32, device=d)
        self._ctxs_buf = torch.zeros(K, ts1, dtype=torch.int32, device=d)
        # Pre-computed tree cache key template for bs=1: acc_lens = [0,0,0,1,1,1,...,nv-1,nv-1,nv-1]
        self._acc_lens_t = torch.arange(nv1, device=d, dtype=torch.int64).repeat_interleave(self.F)
        # Pre-allocated sid tensor for bs=1
        self._sid_buf = torch.zeros(ts1, dtype=torch.int64, device=d)

        print(f"[EAGLEDraftRunner] Initialized on GPU {config.draft_gpu}, K={self.K}, F={self.F}", flush=True)
        print(f"[EAGLEDraftRunner] norm.weight norm={self.draft_model.model.norm.weight.data.norm().item():.4f}", flush=True)
        print(f"[EAGLEDraftRunner] lm_head.weight norm={self.lm_head.weight.data.norm().item():.4f}", flush=True)
        fc_mod = self.draft_model.model.fc if self.eagle3 else self.draft_model.fc
        print(f"[EAGLEDraftRunner] fc.weight norm={fc_mod.weight.data.norm().item():.4f}", flush=True)

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
        # Both EAGLE-1 and EAGLE-3 now use model.layers[0]
        draft_layer = self.draft_model.model.layers[0] if self.eagle3 else self.draft_model.layers[0]
        for module in draft_layer.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.eagle_kv_cache[0, 0]
                module.v_cache = self.eagle_kv_cache[1, 0]

    def _capture_batched_eagle_graph(self):
        """Capture CUDA graph for batched EAGLE forward (used in tree decode)."""
        hf = self.hf_config
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size + 10
        d = self.device
        self._batched_eagle_graphs = {}
        pool = None

        # Capture for the most common batch size: nv * fan for bs=1
        nv = self.K + 1
        common_bs = nv * self.F
        for bs in [common_bs]:
            # Create static buffers OUTSIDE inference_mode so they are regular tensors
            g = {}
            g["ids"] = torch.zeros(bs, dtype=torch.int64, device=d)
            g["pos"] = torch.zeros(bs, dtype=torch.int64, device=d)
            g["hidden"] = torch.zeros(bs, hf.hidden_size, dtype=hf.torch_dtype, device=d)
            g["slot"] = torch.zeros(bs, dtype=torch.int32, device=d)
            g["ctx"] = torch.zeros(bs, dtype=torch.int32, device=d)
            g["bt"] = torch.zeros(bs, max_num_blocks, dtype=torch.int32, device=d)
            g["out"] = torch.zeros(bs, hf.hidden_size, dtype=hf.torch_dtype, device=d)
            g["tokens"] = torch.zeros(bs, dtype=torch.int64, device=d)

            # Warmup + Capture under inference_mode
            with torch.inference_mode():
                set_context(False, slot_mapping=g["slot"], context_lens=g["ctx"], block_tables=g["bt"])
                out = self.draft_model(g["ids"], g["pos"], g["hidden"])
                logits = F.linear(out, self.lm_head.weight)
                g["out"].copy_(out)
                g["tokens"].copy_(logits.argmax(dim=-1))
                reset_context()

                # Capture
                graph = torch.cuda.CUDAGraph()
                set_context(False, slot_mapping=g["slot"], context_lens=g["ctx"], block_tables=g["bt"])
                with torch.cuda.graph(graph, pool):
                    out = self.draft_model(g["ids"], g["pos"], g["hidden"])
                    logits = F.linear(out, self.lm_head.weight)
                    g["out"].copy_(out)
                    g["tokens"].copy_(logits.argmax(dim=-1))
                reset_context()
            if pool is None:
                pool = graph.pool()
            self._batched_eagle_graphs[bs] = (graph, g)
            torch.cuda.synchronize()

        # --- Glue forward graph (EAGLE-1, bs=K+1) ---
        # Captures norm + EAGLE glue forward + lm_head + top-F into one replay,
        # so the common bs=1 handle_early_speculate path avoids per-kernel Python
        # overhead. Outputs land in gg["normed"] / gg["cands"]; the per-depth
        # tree-decode graph above still runs K times after this.
        gg = {
            "ids":       torch.zeros(nv, dtype=torch.int64, device=d),
            "pos":       torch.zeros(nv, dtype=torch.int64, device=d),
            "early_raw": torch.zeros(nv, hf.hidden_size, dtype=hf.torch_dtype, device=d),
            "normed":    torch.zeros(nv, hf.hidden_size, dtype=hf.torch_dtype, device=d),
            "cands":     torch.zeros(nv, self.F, dtype=torch.int64, device=d),
            "slot":      torch.zeros(nv, dtype=torch.int32, device=d),
            "ctx":       torch.zeros(nv, dtype=torch.int32, device=d),
            "bt":        torch.zeros(nv, max_num_blocks, dtype=torch.int32, device=d),
        }

        def _glue_body():
            n = self.draft_model.model.norm(gg["early_raw"])
            self.draft_model(gg["ids"], gg["pos"], n)
            lg = F.linear(n, self.lm_head.weight)
            _, cds = torch.topk(lg, self.F, dim=-1)
            gg["normed"].copy_(n)
            gg["cands"].copy_(cds)

        with torch.inference_mode():
            set_context(False, slot_mapping=gg["slot"], context_lens=gg["ctx"], block_tables=gg["bt"])
            _glue_body()  # warmup
            reset_context()
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=gg["slot"], context_lens=gg["ctx"], block_tables=gg["bt"])
            with torch.cuda.graph(graph, pool):
                _glue_body()
            reset_context()
        self._glue_graph = graph
        self._glue_g = gg
        torch.cuda.synchronize()

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
        """Single EAGLE forward step (CUDA graph or eager for EAGLE-3)."""
        block_idx = cur_pos // self.block_size
        bt_len = block_table_t.shape[0]
        block_offset = cur_pos % self.block_size
        slot = block_table_t[min(block_idx, bt_len - 1)] * self.block_size + block_offset

        if self._g is not None:
            # CUDA graph path (EAGLE-1)
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
        else:
            # Eager path (EAGLE-3)
            inp = torch.tensor([token_id], dtype=torch.int64, device=self.device)
            pos = torch.tensor([cur_pos], dtype=torch.int64, device=self.device)
            sm = torch.tensor([slot], dtype=torch.int32, device=self.device)
            cl = torch.tensor([cur_pos + 1], dtype=torch.int32, device=self.device)
            bt = block_table_t.unsqueeze(0)
            h = hidden_state.unsqueeze(0) if hidden_state.dim() == 1 else hidden_state
            set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
            out = self.draft_model(inp, pos, h)
            reset_context()
            if self.eagle3:
                logits = self.draft_model.compute_logits(out)
                draft_tok = logits.argmax(dim=-1).item()
                next_token = self.d2t[draft_tok].item()
            else:
                logits = F.linear(out, self.lm_head.weight)
                next_token = logits.argmax(dim=-1).item()
            return out[0].clone(), logits.squeeze(0), next_token

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

        H = self.hf_config.hidden_size
        recv_cols = H * 3 if self.eagle3 else H  # EAGLE-3: 3H aux concat
        token_ids = torch.zeros(n, dtype=torch.int64, device=self.device)
        hidden_recv = torch.zeros(n, recv_cols, dtype=self.hf_config.torch_dtype, device=self.device)
        positions = torch.zeros(n, dtype=torch.int64, device=self.device)
        block_table = torch.zeros(bt_len, dtype=torch.int32, device=self.device)

        dist.recv(token_ids, src=0, group=self.async_pg)
        dist.recv(hidden_recv, src=0, group=self.async_pg)
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
        if self.eagle3:
            # hidden_recv is 3H aux concat for EAGLE-3
            eagle_out = self.draft_model(token_ids, positions, None, aux_hiddens=hidden_recv)
        else:
            eagle_out = self.draft_model(token_ids, positions, hidden_recv)
        reset_context()

        self.last_hidden[int(seq_id)] = eagle_out[-1:].clone()
        ack = torch.ones(1, dtype=torch.int64, device=self.device)
        dist.send(ack, dst=0, group=self.async_pg)

    def handle_early_speculate(self, num_seqs=None, packed_len=None, meta_len=None):
        """Receive early hidden from target, build tree cache with EAGLE.
        Uses preallocated buffers + merged payload to minimize NCCL ops."""
        if num_seqs is None:
            ns_buf = torch.zeros(1, dtype=torch.int64, device=self.device)
            dist.recv(ns_buf, src=0, group=self.async_pg)
            num_seqs = int(ns_buf[0].item())
            packed_len = None
            meta_len = None

        # Recv single payload: [meta_len_header, packed_meta..., hidden_as_int64...]
        payload_len = packed_len
        if payload_len is None:
            max_nv = (self.K + 1) * num_seqs
            hmult = 3 if self.eagle3 else 1
            hbytes = max_nv * self.hf_config.hidden_size * hmult * self.hf_config.torch_dtype.itemsize // 8
            payload_len = 1 + num_seqs * 5 + (self.config.max_model_len // self.block_size + 2) * num_seqs + max_nv + hbytes
        payload = self._meta_buf[:payload_len]
        dist.recv(payload, src=0, group=self.async_pg)

        # Split payload: [meta_len, meta..., verify_ids..., hidden_i64...]
        if meta_len is None:
            meta_len = int(payload[0].item())
        meta_part = payload[1:1 + meta_len]
        rest = payload[1 + meta_len:]  # verify_ids + hidden_i64

        if num_seqs == 1:
            # === Fast path: read header NOW while GPU idle after recv ===
            nv = self.K + 1
            btlen = meta_len - 5 - nv
            total_nv = nv
            total_bt = btlen
            hdr = meta_part[:5].tolist()  # free: GPU idle right after recv
            _sid = int(hdr[0])
            _spos = int(hdr[4])
            seq_infos = [(_sid, nv, int(hdr[2]), btlen, _spos)]
            # GPU slicing
            all_bt = meta_part[5:5 + btlen].to(torch.int32)
            all_pos = meta_part[5 + btlen:5 + btlen + nv]
        else:
            # Multi-seq: parse with tolist
            packed = meta_part.tolist()
            seq_infos = []
            total_nv = 0
            total_bt = 0
            off = 0
            bt_ranges = []
            pos_ranges = []
            for _ in range(num_seqs):
                sid, nv, ntok, btlen, spos = int(packed[off]), int(packed[off+1]), int(packed[off+2]), int(packed[off+3]), int(packed[off+4])
                off += 5
                seq_infos.append((sid, nv, ntok, btlen, spos))
                bt_ranges.append((off, btlen))
                off += btlen
                pos_ranges.append((off, nv))
                off += nv
                total_nv += nv
                total_bt += btlen
            all_bt = self._bt_buf[:total_bt]
            all_pos = self._pos_buf[:total_nv]
            bt_off = 0
            pos_off = 0
            for (bstart, blen), (pstart, plen) in zip(bt_ranges, pos_ranges):
                all_bt[bt_off:bt_off+blen] = meta_part[bstart:bstart+blen].to(torch.int32)
                all_pos[pos_off:pos_off+plen] = meta_part[pstart:pstart+plen]
                bt_off += blen
                pos_off += plen

        # Extract verify_ids and hidden from rest
        total_nv_val = total_nv
        verify_token_ids = rest[:total_nv_val].to(torch.int64)
        hidden_i64 = rest[total_nv_val:]
        H = self.hf_config.hidden_size
        hidden_cols = H * 3 if self.eagle3 else H  # EAGLE-3: 3*H, EAGLE-1: H
        n_i64 = total_nv_val * hidden_cols * self.hf_config.torch_dtype.itemsize // 8
        early_hidden_all = hidden_i64[:n_i64].view(self.hf_config.torch_dtype).reshape(total_nv_val, hidden_cols)

        # === Prepare slot/ctx/bt tensors for the glue forward ===
        blk = self.block_size
        if num_seqs == 1:
            _, nv, _, btlen, _ = seq_infos[0]
            pos_slice = all_pos[:nv]
            bi = (pos_slice // blk).long().clamp(max=btlen - 1)
            glue_slot_t = (all_bt[bi] * blk + (pos_slice % blk).to(torch.int32)).to(torch.int32)
            glue_ctx_t = (pos_slice + 1).to(torch.int32)
            glue_bt_t = all_bt.unsqueeze(0).expand(nv, -1).contiguous()
        else:
            bt_off = 0; h_off = 0
            glue_slots_all = []; glue_ctxs_all = []; glue_bts_all = []
            for sid, nv, ntok, btlen, spos in seq_infos:
                bt_seq = all_bt[bt_off:bt_off + btlen]
                pos_slice = all_pos[h_off:h_off + nv]
                bi = (pos_slice // blk).long().clamp(max=btlen - 1)
                glue_slots_all.append((bt_seq[bi] * blk + (pos_slice % blk).to(torch.int32)).to(torch.int32))
                glue_ctxs_all.append((pos_slice + 1).to(torch.int32))
                glue_bts_all.append(bt_seq.unsqueeze(0).expand(nv, -1))
                h_off += nv
                bt_off += btlen
            glue_slot_t = torch.cat(glue_slots_all)
            glue_ctx_t = torch.cat(glue_ctxs_all)
            max_bt_len = max(bt.shape[1] for bt in glue_bts_all)
            glue_bt_t = torch.cat(
                [F.pad(bt, (0, max_bt_len - bt.shape[1])) for bt in glue_bts_all]
            ).contiguous()
        pos_int64 = all_pos[:total_nv_val].to(torch.int64)

        # Glue forward: writes EAGLE KV at verify positions + extracts candidates.
        # Fast path (EAGLE-1, bs=1, ssd_early_layers >= 0): one CUDA graph replay.
        # Else: eager forward.
        use_glue_graph = (
            not self.eagle3
            and self._glue_graph is not None
            and num_seqs == 1
            and self.config.ssd_early_layers >= 0
        )
        if use_glue_graph:
            gg = self._glue_g
            gg["ids"].copy_(verify_token_ids)
            gg["pos"].copy_(pos_int64)
            gg["early_raw"].copy_(early_hidden_all)
            gg["slot"].copy_(glue_slot_t)
            gg["ctx"].copy_(glue_ctx_t)
            bt_len = glue_bt_t.shape[1]
            gg["bt"][:, :bt_len] = glue_bt_t
            if bt_len < gg["bt"].shape[1]:
                gg["bt"][:, bt_len:] = 0
            set_context(False, slot_mapping=gg["slot"], context_lens=gg["ctx"], block_tables=gg["bt"])
            self._glue_graph.replay()
            reset_context()
            normed_all_pre = gg["normed"]
        else:
            if self.eagle3:
                normed_all_pre = None
                glue_hidden_all = early_hidden_all
            elif self.config.ssd_early_layers < 0:
                normed_all_pre = early_hidden_all
                glue_hidden_all = early_hidden_all
            else:
                normed_all_pre = self.draft_model.model.norm(early_hidden_all)
                glue_hidden_all = normed_all_pre
            set_context(False, slot_mapping=glue_slot_t, context_lens=glue_ctx_t, block_tables=glue_bt_t)
            if self.eagle3:
                self.draft_model(verify_token_ids, pos_int64, None, aux_hiddens=glue_hidden_all)
            else:
                self.draft_model(verify_token_ids, pos_int64, glue_hidden_all)
            reset_context()

        # === Batched processing: all seqs' candidates in one EAGLE forward ===
        fan = self.F
        d = self.device

        # 1. Compute candidate tokens via topk (approximate: fc without decoder layer)
        if self.eagle3:
            fc_hidden = self.draft_model.combine_hidden_states(early_hidden_all)
            early_logits_all = self.draft_model.compute_logits(fc_hidden)
            _, cands_draft = torch.topk(early_logits_all, fan, dim=-1)
            cands_all = self.d2t[cands_draft]
            normed_all = fc_hidden
        else:
            # EAGLE-1: shared norm+lm_head (reuse normed_all_pre from glue stage)
            normed_all = normed_all_pre
            if use_glue_graph:
                # Candidates already computed inside the glue graph
                cands_all = self._glue_g["cands"]
            else:
                early_logits_all = F.linear(normed_all, self.lm_head.weight)
                _, cands_all = torch.topk(early_logits_all, fan, dim=-1)

        # 2. Expand via pre-computed fan index
        batch_cands = cands_all.reshape(-1)
        fan_idx = self._fan_idx[:total_nv * fan]
        batch_hidden = normed_all[fan_idx]
        batch_pos = all_pos[fan_idx]

        # Per-seq metadata for tree decode
        per_seq_meta = []
        grand_total = 0
        if num_seqs == 1:
            nv = self.K + 1
            total_s = nv * fan
            if self.tree_decode and self.K > 1:
                # base_vpos = spos + nv, keep as GPU tensor for vpos arithmetic
                per_seq_meta.append((total_s, _spos + nv, all_bt.shape[0] - 1, all_bt, 0))
            grand_total = total_s
        else:
            bt_off = 0
            for sid, nv, ntok, btlen, spos in seq_infos:
                block_table = all_bt[bt_off:bt_off+btlen]
                if self.tree_decode and self.K > 1:
                    per_seq_meta.append((nv * fan, spos + nv, block_table.shape[0] - 1, block_table, grand_total))
                grand_total += nv * fan
                bt_off += btlen

        if self.tree_decode and self.K > 1:
            # 3a. Batched tree decode — precompute block table for all depths
            # For single-seq (common case), avoid per-depth per-seq loop
            all_draft = []
            current_hidden = batch_hidden
            current_ids = batch_cands
            N = grand_total  # total entries across all seqs

            if num_seqs == 1:
                # Fast path: use pre-allocated buffers from __init__
                total_s, base_vpos, max_bi, block_table, _ = per_seq_meta[0]
                blk = self.block_size

                # Compute into pre-allocated [K, ts1] buffers
                all_vpos = self._vpos_buf
                all_vpos[:] = self._base_arange.unsqueeze(0) + base_vpos + self._depth_offsets.unsqueeze(1)
                all_ropes = self._ropes_buf
                all_ropes[:] = batch_pos.unsqueeze(0) + self._depth_range.unsqueeze(1)
                bi = (all_vpos // blk).clamp(max=max_bi)
                self._slots_buf[:] = (block_table[bi] * blk + all_vpos % blk).to(torch.int32)
                self._ctxs_buf[:] = (all_vpos + 1).to(torch.int32)
                all_slots = self._slots_buf
                all_ctxs = self._ctxs_buf

                # Block table: prepare once
                use_graph = total_s in self._batched_eagle_graphs
                if use_graph:
                    graph, g = self._batched_eagle_graphs[total_s]
                    bt_len = block_table.shape[0]
                    g["bt"][:, :bt_len] = block_table.unsqueeze(0).expand(total_s, -1)
                    if bt_len < g["bt"].shape[1]:
                        g["bt"][:, bt_len:] = 0
                else:
                    bt_expanded = block_table.unsqueeze(0).expand(total_s, -1)

                for depth in range(self.K):
                    if use_graph:
                        g["ids"].copy_(current_ids)
                        g["pos"].copy_(all_ropes[depth])
                        g["hidden"].copy_(current_hidden)
                        g["slot"].copy_(all_slots[depth])
                        g["ctx"].copy_(all_ctxs[depth])
                        set_context(False, slot_mapping=g["slot"], context_lens=g["ctx"], block_tables=g["bt"])
                        graph.replay()
                        reset_context()
                        all_draft.append(g["tokens"].clone())
                        current_hidden = g["out"]  # safe: copied to g["hidden"] before next replay
                        current_ids = g["tokens"]   # safe: copied to g["ids"] before next replay
                    else:
                        set_context(False, slot_mapping=all_slots[depth], context_lens=all_ctxs[depth], block_tables=bt_expanded)
                        if self.eagle3:
                            eagle_out = self.draft_model(current_ids, all_ropes[depth], current_hidden)
                        else:
                            eagle_out = self.draft_model(current_ids, all_ropes[depth], current_hidden)
                        reset_context()
                        if self.eagle3:
                            logits = self.draft_model.compute_logits(eagle_out)
                            draft_tokens = logits.argmax(dim=-1)
                            next_tokens = self.d2t[draft_tokens]
                        else:
                            logits = F.linear(eagle_out, self.lm_head.weight)
                            next_tokens = logits.argmax(dim=-1)
                        all_draft.append(next_tokens)
                        current_hidden = eagle_out
                        current_ids = next_tokens
            else:
                # Multi-seq path
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
                    if self.eagle3:
                        logits = self.draft_model.compute_logits(eagle_out)
                        draft_tokens = logits.argmax(dim=-1)
                        next_tokens = self.d2t[draft_tokens]
                    else:
                        logits = F.linear(eagle_out, self.lm_head.weight)
                        next_tokens = logits.argmax(dim=-1)
                    all_draft.append(next_tokens)
                    current_hidden = eagle_out
                    current_ids = next_tokens

            spec_tokens = torch.stack(all_draft, dim=1)
            if num_seqs == 1:
                # sid already read right after recv (no extra sync)
                total = grand_total
                sid = seq_infos[0][0]
                s = torch.full((total,), sid, dtype=torch.int64, device=d)
                keys = torch.stack([s, self._acc_lens_t[:total], batch_cands[:total]], dim=1)
                self.tree_caches[sid] = (keys, spec_tokens)
            else:
                offset = 0
                for sid, nv, ntok, btlen, spos in seq_infos:
                    total = nv * fan
                    s = torch.full((total,), sid, dtype=torch.int64, device=d)
                    a = torch.arange(nv, device=d, dtype=torch.int64).repeat_interleave(fan)
                    keys = torch.stack([s, a, batch_cands[offset:offset+total]], dim=1)
                    self.tree_caches[sid] = (keys, spec_tokens[offset:offset+total])
                    offset += total
        else:
            # 3b. Batched chain: single EAGLE step
            # Compute slots from batch_pos and block tables
            all_chain_slots = []
            all_chain_ctxs = []
            all_chain_bts = []
            h_off2 = 0; bt_off2 = 0
            for sid, nv, ntok, btlen, spos in seq_infos:
                block_table = all_bt[bt_off2:bt_off2+btlen]
                vp = all_pos[h_off2:h_off2+nv].repeat_interleave(fan)
                bi = (vp // self.block_size).long()
                bo = (vp % self.block_size).int()
                all_chain_slots.append((block_table[bi] * self.block_size + bo).int())
                all_chain_ctxs.append((vp + 1).to(torch.int32))
                all_chain_bts.append(block_table.unsqueeze(0).expand(nv * fan, -1))
                h_off2 += nv; bt_off2 += btlen
            b_slots = torch.cat(all_chain_slots)
            b_ctxs = torch.cat(all_chain_ctxs)
            max_btl = max(bt.shape[1] for bt in all_chain_bts)
            b_bt = torch.cat([F.pad(bt, (0, max_btl - bt.shape[1])) for bt in all_chain_bts]).contiguous()

            set_context(False, slot_mapping=b_slots, context_lens=b_ctxs, block_tables=b_bt)
            eagle_out = self.draft_model(batch_cands, batch_pos, batch_hidden)
            reset_context()
            if self.eagle3:
                logits = self.draft_model.compute_logits(eagle_out)
                draft_tokens = self.d2t[logits.argmax(dim=-1)]
            else:
                logits = F.linear(eagle_out, self.lm_head.weight)
                draft_tokens = logits.argmax(dim=-1)

            offset = 0
            for sid, nv, ntok, btlen, spos in seq_infos:
                total = nv * fan
                s = torch.full((total,), sid, dtype=torch.int64, device=d)
                a = torch.arange(nv, device=d, dtype=torch.int64).repeat_interleave(fan)
                keys = torch.stack([s, a, batch_cands[offset:offset+total]], dim=1)
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
        profile_path = os.environ.get("DRAFT_PROFILE_PATH")
        prof = None
        if profile_path:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, with_stack=True,
            )
            prof.__enter__()
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
                vals = self._cmd_buf[1:4].tolist()
                num_seqs, packed_len, meta_len = int(vals[0]), int(vals[1]), int(vals[2])
                self.handle_early_speculate(num_seqs, packed_len, meta_len)
            elif cmd == 6:
                self.handle_cache_lookup()
            elif cmd == 2:
                total = EAGLEDraftRunner._hit + EAGLEDraftRunner._miss
                rate = EAGLEDraftRunner._hit / total * 100 if total else 0
                print(f"[EAGLEDraftRunner] Exiting. Cache hit: {EAGLEDraftRunner._hit}/{total} ({rate:.1f}%)", flush=True)
                if prof:
                    prof.__exit__(None, None, None)
                    prof.export_chrome_trace(profile_path)
                    print(f"[EAGLEDraftRunner] Draft trace saved to: {profile_path}", flush=True)
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
