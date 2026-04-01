"""Separate module for draft process entry point.
MUST set TORCHDYNAMO_DISABLE before any torch import.
"""
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"

def draft_entry(config, rank, init_q=None):
    from nanovllm.engine.draft_runner import launch_draft_runner
    launch_draft_runner(config, rank, init_q)
