from typing import List, Tuple, Optional
import torch

def _slice_cache_layer(layer_k: torch.Tensor, layer_v: torch.Tensor, keep_idx: torch.Tensor):
    keep_idx = keep_idx.to(layer_k.device).long()
    k = layer_k.index_select(dim=2, index=keep_idx)
    v = layer_v.index_select(dim=2, index=keep_idx)
    return k, v

def evict_middle_with_sinks(
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]],
    sink_size: int,
    window_size: int,
):
    """
    Keep the first `sink_size` tokens and the last `window_size` tokens in the KV cache,
    evict everything in the middle. Returns (new_past, keep_idx_np)
    """
    if past_key_values is None:
        return None, None

    total_len = past_key_values[0][0].shape[2]
    # Build keep index regardless, for visualization callers
    import numpy as np
    first_np = list(range(min(sink_size, total_len)))
    last_np = list(range(max(0, total_len - window_size), total_len))
    keep_idx_np = np.array(first_np + last_np, dtype="int64")

    if total_len <= sink_size + window_size:
        return past_key_values, keep_idx_np

    first = torch.arange(0, sink_size)
    last = torch.arange(total_len - window_size, total_len)
    keep_idx = torch.cat([first, last], dim=0)

    new_past = []
    for (k, v) in past_key_values:
        nk, nv = _slice_cache_layer(k, v, keep_idx)
        print(nk.shape, nv.shape)
        new_past.append((nk, nv))
    return new_past, keep_idx_np
