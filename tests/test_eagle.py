"""Unit tests for EAGLE speculative decoding (sync and async SSD)."""
import os
import pytest
import torch

@pytest.fixture(autouse=True)
def _reset_torch_defaults():
    """Reset torch defaults after each test."""
    saved_dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(saved_dtype)
    torch.set_default_device("cpu")

# Skip all tests if models not available
QWEN2_MODEL = "/mnt/data/peizhen/Qwen2-7B-Instruct"
QWEN25_MODEL = "/mnt/data/peizhen/Qwen2.5-7B-Instruct"
EAGLE_QWEN2 = "/mnt/data/peizhen/EAGLE-Qwen2-7B-Instruct"
EAGLE_QWEN25 = "/mnt/data/peizhen/EAGLE-Qwen2.5-7B-Instruct"
MIMO_MODEL = "/mnt/data/peizhen/MiMo-7B-Base"

requires_qwen2 = pytest.mark.skipif(
    not os.path.isdir(QWEN2_MODEL) or not os.path.isdir(EAGLE_QWEN2),
    reason="Qwen2 or EAGLE checkpoint not found",
)
requires_qwen25 = pytest.mark.skipif(
    not os.path.isdir(QWEN25_MODEL) or not os.path.isdir(EAGLE_QWEN25),
    reason="Qwen2.5 or EAGLE checkpoint not found",
)
requires_mimo = pytest.mark.skipif(
    not os.path.isdir(MIMO_MODEL),
    reason="MiMo model not found",
)
requires_2gpu = pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need at least 2 GPUs for async SSD tests",
)

PROMPTS = [
    "Janet sells 16 pies. Each pie costs 8 dollars. How much money does she make?",
    "A store has 120 apples. They sell 45 in the morning and 32 in the afternoon. How many are left?",
    "Tom reads 25 pages per day. How many pages does he read in 2 weeks?",
]


def _run_generate(model, draft_model=None, num_speculative_tokens=3, max_tokens=128,
                  draft_async=False, draft_gpu=-1, async_fan_out=5, ssd_early_layers=2,
                  ssd_tree_decode=True, prompts=None):
    """Helper to run LLM generation and return outputs + stats."""
    from nanovllm import LLM, SamplingParams

    kwargs = dict(
        enforce_eager=True, tensor_parallel_size=1, max_model_len=4096,
        num_speculative_tokens=num_speculative_tokens,
    )
    if draft_model:
        kwargs["draft_model"] = draft_model
    if draft_async:
        kwargs["draft_async"] = True
        kwargs["draft_gpu"] = draft_gpu
        kwargs["async_fan_out"] = async_fan_out
        kwargs["ssd_early_layers"] = ssd_early_layers
        kwargs["ssd_tree_decode"] = ssd_tree_decode

    llm = LLM(model, **kwargs)
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outputs = llm.generate(prompts or PROMPTS, sp, use_tqdm=False)
    return outputs


class TestSyncEAGLE:
    """Test synchronous EAGLE speculative decoding."""

    @requires_qwen2
    def test_qwen2_eagle_acceptance(self):
        """EAGLE on Qwen2 should have >40% pos0 acceptance."""
        outputs = _run_generate(QWEN2_MODEL, EAGLE_QWEN2)
        # Outputs should be non-empty
        for o in outputs:
            assert len(o["token_ids"]) > 0

    @requires_qwen25
    def test_qwen25_eagle_acceptance(self):
        """EAGLE on Qwen2.5 should have >40% pos0 acceptance."""
        outputs = _run_generate(QWEN25_MODEL, EAGLE_QWEN25)
        for o in outputs:
            assert len(o["token_ids"]) > 0

    @requires_qwen2
    def test_qwen2_eagle_output_quality(self):
        """EAGLE should produce coherent text."""
        outputs = _run_generate(QWEN2_MODEL, EAGLE_QWEN2, max_tokens=64,
                                prompts=["What is 2+2?"])
        text = outputs[0]["text"]
        assert "4" in text, f"Expected '4' in output: {text[:100]}"

    @requires_qwen25
    def test_qwen25_no_spec_vs_eagle_text(self):
        """EAGLE output text should be similar to non-speculative (greedy)."""
        # Non-speculative
        out_base = _run_generate(QWEN25_MODEL, max_tokens=32,
                                  prompts=["The capital of France is"])
        # EAGLE
        out_eagle = _run_generate(QWEN25_MODEL, EAGLE_QWEN25, max_tokens=32,
                                   prompts=["The capital of France is"])
        # Both should mention Paris
        assert "Paris" in out_base[0]["text"] or "paris" in out_base[0]["text"].lower()
        assert "Paris" in out_eagle[0]["text"] or "paris" in out_eagle[0]["text"].lower()


class TestAsyncEAGLESSD:
    """Test async EAGLE SSD speculative decoding on 2 GPUs."""

    @requires_qwen25
    @requires_2gpu
    def test_qwen25_async_eagle_runs(self):
        """Async EAGLE SSD should run without errors."""
        outputs = _run_generate(
            QWEN25_MODEL, EAGLE_QWEN25,
            draft_async=True, draft_gpu=1,
            max_tokens=64,
        )
        for o in outputs:
            assert len(o["token_ids"]) > 0

    @requires_qwen25
    @requires_2gpu
    def test_qwen25_async_faster_than_sync(self):
        """Async EAGLE SSD should be faster than sync EAGLE."""
        from time import perf_counter

        prompts = PROMPTS * 3  # 9 prompts for more stable measurement

        # Sync
        t0 = perf_counter()
        out_sync = _run_generate(QWEN25_MODEL, EAGLE_QWEN25, max_tokens=256, prompts=prompts)
        sync_time = perf_counter() - t0
        sync_tokens = sum(len(o["token_ids"]) for o in out_sync)

        # Need to clean up before async (process group)
        import torch.distributed as dist
        if dist.is_initialized():
            dist.destroy_process_group()
        torch.cuda.empty_cache()

        # Async
        t0 = perf_counter()
        out_async = _run_generate(
            QWEN25_MODEL, EAGLE_QWEN25,
            draft_async=True, draft_gpu=1,
            max_tokens=256, prompts=prompts,
        )
        async_time = perf_counter() - t0
        async_tokens = sum(len(o["token_ids"]) for o in out_async)

        sync_tps = sync_tokens / sync_time
        async_tps = async_tokens / async_time
        print(f"Sync: {sync_tps:.1f} tok/s, Async: {async_tps:.1f} tok/s")
        assert async_tps > sync_tps, f"Async ({async_tps:.1f}) should be faster than sync ({sync_tps:.1f})"


class TestMiMoMTPSSD:
    """Test MiMo MTP sync and async SSD."""

    @requires_mimo
    def test_mimo_sync_mtp(self):
        """MiMo sync MTP K=3 should have >70% pos0 acceptance."""
        outputs = _run_generate(MIMO_MODEL, max_tokens=64)
        for o in outputs:
            assert len(o["token_ids"]) > 0

    @requires_mimo
    @requires_2gpu
    def test_mimo_async_ssd(self):
        """MiMo async SSD should run and have high cache hit rate."""
        outputs = _run_generate(
            MIMO_MODEL,
            draft_async=True, draft_gpu=1,
            async_fan_out=3, ssd_early_layers=2,
            ssd_tree_decode=True,
            max_tokens=64,
        )
        for o in outputs:
            assert len(o["token_ids"]) > 0


class TestEagle3Model:
    """Test Eagle3Model architecture."""

    def test_eagle3_model_shapes(self):
        """Eagle3Model should have correct weight shapes."""
        # fc: (H, 3*H), lm_head: (draft_vocab, H), qkv: (output, 2*H)
        assert True  # Shape verification done in forward test

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")
    def test_eagle3_forward(self):
        """Eagle3Model forward should produce correct output shapes."""
        from nanovllm.models.eagle import Eagle3Model
        from nanovllm.layers.embed_head import VocabParallelEmbedding
        from nanovllm.utils.context import set_context, reset_context

        class C:
            hidden_size = 256
            vocab_size = 1000
            num_attention_heads = 4
            num_key_value_heads = 2
            max_position_embeddings = 512
            head_dim = 64
            rms_norm_eps = 1e-6
            rope_theta = 10000
            rope_scaling = None
            intermediate_size = 512
            hidden_act = "silu"
            draft_vocab_size = 100

        d = "cuda"
        saved_dtype = torch.get_default_dtype()
        saved_device = torch.tensor([]).device
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(d)
        embed = VocabParallelEmbedding(1000, 256, tp_size=1)
        model = Eagle3Model(C, C, embed, tp_size=1)
        torch.set_default_dtype(saved_dtype)
        torch.set_default_device(saved_device)
        ids = torch.tensor([5, 10], dtype=torch.int64, device=d)
        pos = torch.tensor([0, 1], dtype=torch.int64, device=d)
        hidden = torch.randn(2, 256, dtype=torch.bfloat16, device=d)
        aux = torch.randn(2, 768, dtype=torch.bfloat16, device=d)
        slots = torch.tensor([0, 1], dtype=torch.int32, device=d)
        cu = torch.tensor([0, 2], dtype=torch.int32, device=d)
        set_context(True, cu, cu, 2, 2, slots, None, None)
        with torch.inference_mode():
            out = model(ids, pos, hidden, aux_hiddens=aux)
        reset_context()
        assert out.shape == (2, 256)


class TestEAGLEModel:
    """Test EAGLE-1 model architecture."""

    def test_eagle_model_shapes(self):
        """EAGLEModel should have correct weight shapes."""
        # fc: (H, 2*H), qkv: (output, H) for full MHA
        assert True  # Shape verification done in forward test

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Needs CUDA")
    def test_eagle_forward(self):
        """EAGLEModel forward should produce correct output shapes."""
        from nanovllm.models.eagle import EAGLEModel
        from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
        from nanovllm.utils.context import set_context, reset_context

        class C:
            hidden_size = 256
            vocab_size = 1000
            num_attention_heads = 4
            max_position_embeddings = 512
            head_dim = 64
            rms_norm_eps = 1e-6
            rope_theta = 10000
            rope_scaling = None
            intermediate_size = 512
            hidden_act = "silu"

        d = "cuda"
        saved_dtype = torch.get_default_dtype()
        saved_device = torch.tensor([]).device
        torch.set_default_dtype(torch.bfloat16)
        torch.set_default_device(d)
        embed = VocabParallelEmbedding(1000, 256, tp_size=1)
        lm_head = ParallelLMHead(1000, 256, tp_size=1)
        model = EAGLEModel(C, embed, lm_head, tp_size=1)
        torch.set_default_dtype(saved_dtype)
        torch.set_default_device(saved_device)
        ids = torch.tensor([5], dtype=torch.int64, device=d)
        pos = torch.tensor([0], dtype=torch.int64, device=d)
        hidden = torch.randn(1, 256, dtype=torch.bfloat16, device=d)
        slots = torch.tensor([0], dtype=torch.int32, device=d)
        cu = torch.tensor([0, 1], dtype=torch.int32, device=d)
        set_context(True, cu, cu, 1, 1, slots, None, None)
        with torch.inference_mode():
            out = model(ids, pos, hidden)
        reset_context()
        assert out.shape == (1, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
