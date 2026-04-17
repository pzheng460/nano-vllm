"""Instrumented nano-vllm runner for Spec-Bench evaluation.

`LLM.generate` aggregates metrics across the batch; to produce per-prompt
per-decoding-step `accept_lengths` (Spec-Bench schema) we bypass it and
manually drive `scheduler.add` + `llm.step()`.

With batch size 1, each `step()` reports metrics for the single active
sequence:
  * prefill step: `num_tokens > 0` (not counted as a decode step)
  * decode step:  `num_tokens < 0`; `-num_tokens` tokens were emitted
    (accepted drafts + bonus). Baseline is always 1; spec modes are 1..K+1.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Optional

from tests.harness.chat import build_turn_input_ids
from tests.harness.specbench import Choice, ModelAnswer, Question


@dataclass
class RunnerConfig:
    model: str
    draft_model: Optional[str] = None
    num_speculative_tokens: int = 3
    max_new_tokens: int = 512
    max_model_len: int = 4096
    temperature: float = 0.0
    # Async SSD options (only used when draft_async=True)
    draft_async: bool = False
    draft_gpu: int = 1
    async_fan_out: int = 3
    ssd_early_layers: int = 2
    ssd_tree_decode: bool = True
    # Engine options
    tensor_parallel_size: int = 1
    enforce_eager: bool = True

    def mode_tag(self) -> str:
        if self.draft_async:
            return f"async-ssd-k{self.num_speculative_tokens}"
        if self.draft_model is not None:
            return f"sync-eagle-k{self.num_speculative_tokens}"
        return "baseline"


def build_llm(cfg: RunnerConfig):
    from nanovllm import LLM

    kwargs: dict = dict(
        enforce_eager=cfg.enforce_eager,
        tensor_parallel_size=cfg.tensor_parallel_size,
        max_model_len=cfg.max_model_len,
    )
    if cfg.draft_model is not None:
        kwargs["draft_model"] = cfg.draft_model
        kwargs["num_speculative_tokens"] = cfg.num_speculative_tokens
    if cfg.draft_async:
        kwargs["draft_async"] = True
        kwargs["draft_gpu"] = cfg.draft_gpu
        kwargs["async_fan_out"] = cfg.async_fan_out
        kwargs["ssd_early_layers"] = cfg.ssd_early_layers
        kwargs["ssd_tree_decode"] = cfg.ssd_tree_decode
    return LLM(cfg.model, **kwargs)


def run_one_prompt(llm, input_ids: list[int], sampling_params) -> dict:
    """Run one prompt to completion; return per-step stats."""
    from nanovllm.engine.sequence import Sequence

    mr = llm.model_runner
    if hasattr(mr, "_cache_hit"):
        mr._cache_hit = 0
        mr._cache_miss = 0

    seq = Sequence(list(input_ids), sampling_params)
    llm.scheduler.add(seq)

    accept_lengths: list[int] = []
    completion_ids: list[int] = []
    t0 = perf_counter()
    while not llm.is_finished():
        outputs, num_tokens, _draft, _accept, _per_pos = llm.step()
        if num_tokens < 0:
            accept_lengths.append(-num_tokens)
        for _seq_id, token_ids in outputs:
            completion_ids = list(token_ids)
    wall = perf_counter() - t0

    return {
        "token_ids": completion_ids,
        "decoding_steps": len(accept_lengths),
        "accept_lengths": accept_lengths,
        "wall_time": wall,
    }


def run_questions(
    cfg: RunnerConfig,
    questions: list[Question],
    model_id: Optional[str] = None,
    llm=None,
    verbose: bool = False,
) -> list[ModelAnswer]:
    """Run Spec-Bench questions; return per-question ModelAnswer records.

    Multi-turn questions are handled by appending the previous assistant reply
    (raw decoded text) before encoding the next user turn with the chat
    template.
    """
    from nanovllm import SamplingParams

    if llm is None:
        llm = build_llm(cfg)

    tokenizer = llm.tokenizer
    sp = SamplingParams(temperature=cfg.temperature, max_tokens=cfg.max_new_tokens)
    if model_id is None:
        model_id = f"{Path(cfg.model).name}::{cfg.mode_tag()}"

    answers: list[ModelAnswer] = []
    for q in questions:
        prior_replies: list[str] = []
        turns_text: list[str] = []
        decoding_steps: list[int] = []
        new_tokens: list[int] = []
        wall_time: list[float] = []
        accept_lengths_all: list[int] = []

        for _ in range(len(q.turns)):
            input_ids = build_turn_input_ids(tokenizer, q.turns, prior_replies)
            stats = run_one_prompt(llm, input_ids, sp)
            reply_text = tokenizer.decode(stats["token_ids"], skip_special_tokens=True)

            prior_replies.append(reply_text)
            turns_text.append(reply_text)
            decoding_steps.append(stats["decoding_steps"])
            new_tokens.append(len(stats["token_ids"]))
            wall_time.append(stats["wall_time"])
            accept_lengths_all.extend(stats["accept_lengths"])

        answers.append(
            ModelAnswer(
                question_id=q.question_id,
                category=q.category,
                model_id=model_id,
                choices=[
                    Choice(
                        index=0,
                        turns=turns_text,
                        decoding_steps=decoding_steps,
                        new_tokens=new_tokens,
                        wall_time=wall_time,
                        accept_lengths=accept_lengths_all,
                    )
                ],
            )
        )
        if verbose:
            total_tok = sum(new_tokens)
            total_t = sum(wall_time)
            mean_acc = (
                sum(accept_lengths_all) / max(len(accept_lengths_all), 1)
            )
            print(
                f"[{cfg.mode_tag()}] q={q.question_id} cat={q.category} "
                f"tok={total_tok} t={total_t:.2f}s "
                f"tps={total_tok/max(total_t,1e-9):.1f} "
                f"mean_accept={mean_acc:.2f}"
            )

    return answers
