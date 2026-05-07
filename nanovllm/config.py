import os
from dataclasses import dataclass, fields
from typing import Optional
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    device_type: Optional[str] = None  # "cuda", "npu", or None (auto)
    draft_model: Optional[str] = None          # EAGLE checkpoint path
    num_speculative_tokens: int = 5            # number of speculative tokens per step
    # Latent Speculative Decoding options
    draft_async: bool = False                  # enable Latent SD async draft on separate GPU
    draft_gpu: int = -1                        # GPU for draft model (-1 = auto)
    async_fan_out: int = 3                     # fan-out for tree speculation
    latent_early_layers: int = -3                 # extract early hidden at layer N+K (K must be < 0).
                                               # -1 = last layer (sync-equiv), -3 = 倒数第3, etc.
    latent_tree_decode: bool = True               # tree decode: K draft steps per candidate (default on for correct multi-step speculation)
    enable_fallback: bool = False              # On cache miss, send cmd=0 to draft for sync fallback spec.
                                               # Off (default) → cache miss falls through to baseline 1-token verify (no spec).
                                               # On  → ~10ms NCCL roundtrip per miss to recover K spec tokens (rarely worth it).
    # Sync-MTP early-hidden parallel. -1 = current sync MTP (last layer, no
    # overlap). -X (X>=2) → extract pre-norm hidden at layer N+K, run MTP on
    # a side CUDA stream in parallel with the target's remaining |K|-1
    # layers, and skip post-verify MTP (next-step draft = early MTP output;
    # accept rate may drop, but MTP latency is hidden).
    mtp_early_layers: int = -1
    num_gpus: int = -1                         # total world size (auto-computed)
    draft_rank: int = -1                       # rank of draft process (auto-computed)
    disable_mtp: bool = False                  # force disable MTP speculative decoding
    profile: bool = False                      # enable torch.profiler tracing (saves to profile_dir)

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        self.use_mtp = (not self.disable_mtp) and getattr(self.hf_config, 'num_nextn_predict_layers', 0) > 0
        if self.use_mtp and self.draft_model is None and not self.draft_async:
            # Only auto-set K if user didn't override (default is 5)
            if self.num_speculative_tokens == 5:
                self.num_speculative_tokens = self.hf_config.num_nextn_predict_layers
        # PanGu sink attention config
        self.sink_len = getattr(self.hf_config, 'param_sink_number', 0) or 0
        # No dedicated sink blocks. Sink KV is embedded in _sink_k/_sink_v buffers
        # and prepended by PanguSinkAttention during attention computation.
        self.num_sink_blocks = 0
        if self.draft_model is not None:
            assert os.path.isdir(self.draft_model)
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model, trust_remote_code=True)
        else:
            self.draft_hf_config = None
        # Latent SD: compute world size and draft rank
        self.num_gpus = self.tensor_parallel_size + (1 if self.draft_async else 0)
        self.eagle_async = self.draft_async and self.draft_model is not None and not self.use_mtp
        # Detect EAGLE-3: has draft_vocab_size or architecture LlamaForCausalLMEagle3
        self.eagle3 = False
        if self.draft_hf_config is not None:
            archs = getattr(self.draft_hf_config, 'architectures', []) or []
            self.eagle3 = (
                getattr(self.draft_hf_config, 'draft_vocab_size', None) is not None
                or any('eagle3' in a.lower() for a in archs)
            )
        if self.draft_async:
            assert self.use_mtp or self.draft_model is not None, \
                "Latent SD async draft requires MTP model or EAGLE draft_model"
            self.draft_rank = self.tensor_parallel_size
            if self.draft_gpu == -1:
                self.draft_gpu = self.draft_rank
            # Compute fan-out list: async_fan_out at each of K+1 positions
            K = self.num_speculative_tokens
            F = self.async_fan_out
            self.fan_out_list = [F] * (K + 1)
            self.mq_len = sum(self.fan_out_list)
            # Total lookahead for block pre-allocation:
            # speculative tokens (K+1) + tree decode positions (K * MQ_LEN)
            self.latent_total_lookahead = K + 1 + K * self.mq_len

_current_config = None

def get_config() -> Config:
    global _current_config
    return _current_config

def set_config(model, **kwargs):
    global _current_config
    config_fields = {field.name for field in fields(Config)}
    config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
    _current_config = Config(model, **config_kwargs)

def reset_config():
    global _current_config
    _current_config = None