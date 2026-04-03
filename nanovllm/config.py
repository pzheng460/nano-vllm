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
    # SSD (Speculative Streaming Decoding) options
    draft_async: bool = False                  # enable SSD async draft on separate GPU
    draft_gpu: int = -1                        # GPU for draft model (-1 = auto)
    async_fan_out: int = 3                     # fan-out for tree speculation
    ssd_early_layers: int = 2                  # extract early hidden at layer N-X
    ssd_tree_decode: bool = False              # tree decode: K MTP steps per candidate (vs 1 + chain lookup)
    num_gpus: int = -1                         # total world size (auto-computed)
    draft_rank: int = -1                       # rank of draft process (auto-computed)
    disable_mtp: bool = False                  # force disable MTP speculative decoding

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
        # No dedicated sink blocks. Sink KV is embedded in each sequence's first block.
        # Slot offset = sink_len; position mapping handled by model_runner.
        self.num_sink_blocks = 0
        if self.draft_model is not None:
            assert os.path.isdir(self.draft_model)
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model, trust_remote_code=True)
        else:
            self.draft_hf_config = None
        # SSD: compute world size and draft rank
        self.num_gpus = self.tensor_parallel_size + (1 if self.draft_async else 0)
        if self.draft_async:
            assert self.use_mtp, "SSD async draft requires MTP model"
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
            self.ssd_total_lookahead = K + 1 + K * self.mq_len

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