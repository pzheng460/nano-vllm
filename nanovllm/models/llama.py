"""Llama model family (Llama 2/3/3.1).

Architecture identical to Qwen3 except:
- No QK normalization (Qwen3 has q_norm/k_norm)
- No QKV bias (Llama uses bias=False)
- Llama3 uses scaled RoPE (rope_type="llama3")
"""
from nanovllm.models.qwen3 import Qwen3ForCausalLM


# Llama shares the same architecture as Qwen3.
# Differences (QK norm, bias, RoPE scaling) are handled by config flags:
#   - model_type="llama" → qk_norm=False in Qwen3DecoderLayer
#   - attention_bias=False → no QKV bias
#   - rope_scaling.rope_type="llama3" → scaled frequencies
LlamaForCausalLM = Qwen3ForCausalLM
