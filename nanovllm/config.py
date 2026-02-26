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

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.draft_model is not None:
            assert os.path.isdir(self.draft_model)
            self.draft_hf_config = AutoConfig.from_pretrained(self.draft_model)
        else:
            self.draft_hf_config = None

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