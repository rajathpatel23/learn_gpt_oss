# attn_sinks — Attention Sinks Toolkit

Modular, Mac-friendly tools to **run** and **visualize** StreamingLLM-style attention sinks.

## Install (Mac MPS)

```bash
python -m venv venv && source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sentencepiece tokenizers matplotlib numpy
```

## Quick start

### 1) Simulation (no HF model needed)
```bash
python -m attn_sinks.cli --simulate --sink_size 6 --window_size 768 --seq_start 1024 --total_steps 120
```

### 2) Live with Gemma-2B (MPS)
```bash
python -m attn_sinks.cli --live   --model google/gemma-2b-it   --prompt "Explain attention sinks in one paragraph."   --max_new_tokens 128   --sink_size 6 --window_size 768   --device mps --dtype float16
```

## Programmatic usage

```python
from attn_sinks import StreamingGenerator, build_keep_mask_over_time, plot_keep_mask

gen = StreamingGenerator("google/gemma-2b-it", sink_size=6, window_size=768, device="mps")
print(gen.generate_stream("Explain attention sinks briefly:", max_new_tokens=120))
```
