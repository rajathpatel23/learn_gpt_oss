"""
attn_sinks: Modular tools for Attention Sinks (StreamingLLM-style)
------------------------------------------------------------------
Submodules:
- cache_policies: eviction/retention policies (sinks + recent window)
- streaming: HF generator with attention-sink cache policy
- viz: simulation + live kept-index heatmaps
- cli: command-line entry points

Example:
    python -m attn_sinks.cli --simulate --sink_size 6 --window_size 768 --seq_start 1024
    python -m attn_sinks.cli --live --model google/gemma-2b-it --device mps --dtype float16
"""
from .cache_policies import evict_middle_with_sinks
from .streaming import StreamingGenerator
from .viz import build_keep_mask_over_time, plot_keep_mask, live_generate_and_record

__all__ = [
    "evict_middle_with_sinks",
    "StreamingGenerator",
    "build_keep_mask_over_time",
    "plot_keep_mask",
    "live_generate_and_record",
]
