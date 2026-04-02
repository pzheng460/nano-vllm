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
        dist.init_process_group("nccl", "tcp://localhost:2333",
                                world_size=config.num_gpus, rank=rank)
        print(f"[Draft rank={rank}] init_process_group done, creating groups...", flush=True)
        # Must participate in new_group calls (collective)
        tp_ranks = list(range(config.tensor_parallel_size))
        self.tp_group = dist.new_group(tp_ranks)  # draft not in this group
        self.async_pg = dist.new_group([0, config.draft_rank])

        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(self.device)

        self.draft_model = MiMoMTPDraftModel(hf_config)
        _load_draft_model(self.draft_model, config.model)

        self.mtp_layer = self.draft_model.model.mtp_layers[0]
        self.embed_tokens = self.draft_model.model.embed_tokens
        self.lm_head = self.draft_model.lm_head

        # Warmup + allocate KV cache (same sizing as target)
        self._warmup_and_allocate_kv_cache()
        # Send num_kvcache_blocks to target via init_q
        if init_q is not None:
            init_q.put(config.num_kvcache_blocks)
            init_q.close()

        # CUDA graph capture for single-token MTP decode
        self._capture_mtp_graph()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        self._reset_tree_cache()
        self.last_hidden = {}
        # Pre-allocate recv buffers
        self._cmd_buf = torch.zeros(1, dtype=torch.int64, device=self.device)

        print(f"[MTPDraftRunner] Initialized on GPU {config.draft_gpu}, K={self.K}, F={self.F}", flush=True)

    def _warmup_and_allocate_kv_cache(self):
        """Compute available memory and allocate MTP KV cache."""
        hf_config = self.hf_config
        config = self.config
        torch.cuda.empty_cache()

        num_kv_heads = hf_config.num_key_value_heads
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

        logits = self.compute_logits_all(g["out_normed"])
        next_token = logits.argmax(dim=-1).item()
        return g["out_prenorm"][0].clone(), logits.squeeze(0), next_token

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
            logits = self.compute_logits_all(normed)
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

    def handle_early_speculate(self):
        """Receive batched early hidden for all seqs from target.
        For each seq: norm → lm_head → topF → embed → MTP → cache."""
        # Receive num_seqs
        ns_buf = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(ns_buf, src=0, group=self.async_pg)
        num_seqs = int(ns_buf[0].item())

        # Receive per-seq metadata: [sid, nv, ntok, btlen, spos] * num_seqs
        meta = torch.zeros(num_seqs * 5, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)

        seq_infos = []
        total_nv = 0
        total_bt = 0
        for i in range(num_seqs):
            b = i * 5
            sid, nv, ntok, btlen, spos = (int(meta[b+j].item()) for j in range(5))
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

        # Process each seq
        fan = self.F
        h_off = 0
        bt_off = 0
        for sid, nv, ntok, btlen, spos in seq_infos:
            early_hidden = early_hidden_all[h_off:h_off+nv]
            block_table = all_bt[bt_off:bt_off+btlen]
            verify_pos = all_pos[h_off:h_off+nv]
            h_off += nv
            bt_off += btlen

            normed_h = self.draft_model.model.norm(early_hidden)
            early_logits = self.compute_logits_all(normed_h)
            _, cands = torch.topk(early_logits, fan, dim=-1)

            total = nv * fan
            flat_cands = cands.reshape(-1)
            flat_embeds = self.embed_tokens(flat_cands)
            flat_hidden = early_hidden.repeat_interleave(fan, dim=0)
            flat_acc_lens = torch.arange(nv, device=self.device, dtype=torch.int64).repeat_interleave(fan)
            flat_pos = verify_pos.repeat_interleave(fan)

            slot_list = []
            for p in flat_pos.tolist():
                p = int(p)
                slot_list.append(int(block_table[p // self.block_size]) * self.block_size + p % self.block_size)
            slot_t = torch.tensor(slot_list, dtype=torch.int32, device=self.device)
            bt_exp = block_table.unsqueeze(0).expand(total, -1).contiguous()
            ctx_lens = (flat_pos + 1).to(torch.int32)

            set_context(False, slot_mapping=slot_t, context_lens=ctx_lens, block_tables=bt_exp)
            normed_out, _ = self.mtp_layer(flat_embeds, flat_hidden, flat_pos)
            reset_context()
            logits = self.compute_logits_all(normed_out)
            draft_tokens = logits.argmax(dim=-1)

            seq_ids = torch.full((total,), sid, dtype=torch.int64, device=self.device)
            keys = torch.stack([seq_ids, flat_acc_lens, flat_cands], dim=1)
            self.tree_caches[sid] = (keys, draft_tokens.unsqueeze(1))

    def handle_cache_lookup(self):
        """Chain lookup: receive (seq_id, accepted_len, recovery_token), do K chained lookups."""
        lookup = torch.zeros(3, dtype=torch.int64, device=self.device)
        dist.recv(lookup, src=0, group=self.async_pg)
        seq_id, accepted_len, recovery_token = int(lookup[0].item()), int(lookup[1].item()), int(lookup[2].item())

        result = torch.zeros(self.K, dtype=torch.int64, device=self.device)
        hit = False
        if seq_id in self.tree_caches:
            cache_keys, cache_tokens = self.tree_caches[seq_id]
            cur_token = recovery_token
            for step in range(self.K):
                request_key = torch.tensor([[seq_id, accepted_len + step, cur_token]],
                                           dtype=torch.int64, device=self.device)
                match = torch.all(request_key == cache_keys, dim=1)
                if match.any():
                    idx = match.float().argmax().item()
                    d = cache_tokens[idx, 0].item()
                    result[step] = d
                    cur_token = d
                    if step == 0:
                        hit = True
                else:
                    # Chain broken — fill remaining with 0
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
        seq_id, acc_len, rec_tok = int(meta[0].item()), int(meta[1].item()), int(meta[2].item())
        next_draft = torch.zeros(self.K, dtype=torch.int64, device=self.device)
        dist.recv(next_draft, src=0, group=self.async_pg)
        key = torch.tensor([[seq_id, acc_len, rec_tok]], dtype=torch.int64, device=self.device)
        self.tree_caches[seq_id] = (key, next_draft.unsqueeze(0))

    def handle_cleanup(self):
        meta = torch.zeros(1, dtype=torch.int64, device=self.device)
        dist.recv(meta, src=0, group=self.async_pg)
        seq_id = int(meta[0].item())
        if seq_id in self.last_hidden:
            del self.last_hidden[seq_id]
        self._reset_tree_cache(seq_id)
        ack = torch.ones(1, dtype=torch.int64, device=self.device)
        dist.send(ack, dst=0, group=self.async_pg)

    def draft_loop(self):
        print("[MTPDraftRunner] Starting draft loop", flush=True)
        while True:
            dist.recv(self._cmd_buf, src=0, group=self.async_pg)
            cmd = int(self._cmd_buf[0].item())
            if cmd == 0:
                self.handle_speculate()
            elif cmd == 1:
                self.handle_prefill()
            elif cmd == 3:
                self.handle_cleanup()
            elif cmd == 4:  # cache_update from target
                self.handle_cache_update()
            elif cmd == 5:  # early_speculate: target sent early hidden during verify
                self.handle_early_speculate()
            elif cmd == 6:  # cache_lookup: lightweight hit query
                self.handle_cache_lookup()
            elif cmd == 2:
                total = MTPDraftRunner._hit + MTPDraftRunner._miss
                rate = MTPDraftRunner._hit / total * 100 if total else 0
                print(f"[MTPDraftRunner] Exiting. Cache hit: {MTPDraftRunner._hit}/{total} ({rate:.1f}%)", flush=True)
                break


def launch_draft_runner(config, rank, init_q=None):
    import sys
    print(f"[Draft rank={rank}] launch_draft_runner starting", flush=True)
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    try:
        from nanovllm.utils.device import DeviceBackend
        DeviceBackend.initialize()
        print(f"[Draft rank={rank}] DeviceBackend initialized, creating runner...", flush=True)
        runner = MTPDraftRunner(config, rank, init_q=init_q)
        runner.draft_loop()
        dist.destroy_process_group()
    except Exception as e:
        print(f"[Draft rank={rank}] FATAL ERROR: {e}", flush=True)
        import traceback; traceback.print_exc()
        sys.exit(1)
