from typing import List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from cache_policies import evict_middle_with_sinks

def build_keep_mask_over_time(total_steps: int, sink_size: int, window_size: int, seq_start: int) -> np.ndarray:
    """
    Simulate which token positions are kept at each time step.
    Returns a matrix (rows=token index, cols=time), values: 0 evicted, 1 sink, 2 recent.
    """
    max_len = seq_start + total_steps
    mask = np.zeros((max_len, total_steps), dtype=np.int8)
    for t in range(total_steps):
        L = seq_start + t
        first = np.arange(0, min(sink_size, L))
        last = np.arange(max(0, L - window_size), L)
        mask[first, t] = 1
        mask[last, t] = 2
    return mask

def plot_keep_mask(mask: np.ndarray, title: str = "Attention Sinks — Kept Indices Over Time"):
    colors = np.empty(mask.shape + (3,), dtype=float)
    colors[mask == 0] = [0.85, 0.85, 0.85]  # gray
    colors[mask == 1] = [0.29, 0.56, 0.89]  # blue
    colors[mask == 2] = [0.49, 0.83, 0.13]  # green

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(colors, aspect='auto', origin='lower')
    ax.set_xlabel("Time step (new tokens)")
    ax.set_ylabel("Token index")
    ax.set_title(title)

    legend = [
        patches.Patch(facecolor=(0.29, 0.56, 0.89), edgecolor='black', label='Sink (kept)'),
        patches.Patch(facecolor=(0.49, 0.83, 0.13), edgecolor='black', label='Recent (kept)'),
        patches.Patch(facecolor=(0.85, 0.85, 0.85), edgecolor='black', label='Evicted'),
    ]
    ax.legend(handles=legend, loc='upper right')
    plt.tight_layout()
    plt.show()

def live_generate_and_record(model_name: str, prompt: str, sink_size: int, window_size: int,
                             max_new_tokens: int, device: str = "mps", dtype_str: str = "float16") -> np.ndarray:
    """
    Stream with a HF model and record which indices are kept at each step.
    Returns a mask like build_keep_mask_over_time.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str)
    # Initialize tokenizer and model
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device); model.eval()

    inp = tok(prompt, return_tensors="pt").to(device)
    input_ids = inp["input_ids"]
    outputs = model(input_ids=input_ids, use_cache=True)
    past = outputs.past_key_values
    logits = outputs.logits[:, -1, :]

    keep_indices_over_time: List[np.ndarray] = []
    seq_lengths: List[int] = [input_ids.shape[1]]

    for _ in range(max_new_tokens):
        next_id = logits.argmax(dim=-1, keepdim=True)
        outputs = model(input_ids=next_id, use_cache=True, past_key_values=past)
        logits = outputs.logits[:, -1, :]
        past, keep_idx_np = evict_middle_with_sinks(outputs.past_key_values, sink_size, window_size)
        seq_lengths.append(seq_lengths[-1] + 1)
        if keep_idx_np is None:
            keep_indices_over_time.append(np.arange(seq_lengths[-1]))
        else:
            keep_indices_over_time.append(keep_idx_np)

    total_steps = len(keep_indices_over_time)
    max_len = max(seq_lengths)
    mask = np.zeros((max_len, total_steps), dtype=np.int8)

    for t, keep_idx in enumerate(keep_indices_over_time):
        L = seq_lengths[t+1]
        sinks = set(range(min(sink_size, L)))
        recent = set(range(max(0, L - window_size), L))
        for idx in keep_idx:
            if idx in sinks: mask[idx, t] = 1
            elif idx in recent: mask[idx, t] = 2
    return mask
