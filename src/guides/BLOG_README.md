### Attention Variants in Transformers: A Practical, Intuition-First Guide

This guide explains how attention works in modern transformers, why variants like Multi-Head, Multi-Query, and Grouped-Query exist, and how they connect to sequential (causal) learning in practice.

---

### 0) Quick Overview

-   **Self-Attention**: Each token looks at other tokens to build a contextualized representation.
-   **Multi-Head Attention (MHA)**: Multiple attention "views" in parallel for richer patterns; highest quality, higher memory.
-   **Multi-Query Attention (MQA)**: Share the Key and Value projections (and KV cache) across query heads; fastest and most memory-efficient for inference.
-   **Grouped-Query Attention (GQA)**: Middle ground—share K/V within groups of heads; balances quality and speed.
-   **Sequential (Causal) Learning**: Train to predict the next token using only the past, then generate left-to-right at inference.

---

### 1) Self-Attention (single-head)

-   **What it is**: A content-based lookup over the sequence.
    -   Queries `Q` ask "what do I need?"; Keys `K` say "what do I offer?"; Values `V` are the information to aggregate.
    -   Core computation: `softmax((Q @ K^T) / sqrt(d)) @ V`.
-   **Causality**: For generation, we apply a causal mask so positions only attend to previous tokens.
-   **Intuition**: Each token gathers relevant context, weighted by semantic compatibility (query-key similarity).
-   **Limits**: A single head blends all relations into one pattern, which can be too coarse for complex structure.

#### 1.1 Formalism and Shapes

-   Notation: batch `B`, sequence length `L`, model width `D`.
-   Projections (single head):
    -   `Q = X W_q`, `K = X W_k`, `V = X W_v`
    -   Shapes: `X: [B, L, D]`, `W_q, W_k, W_v: [D, d]`, `Q,K,V: [B, L, d]`
-   Attention logits: `A = (Q @ K^T) / sqrt(d)` → shape `[B, L, L]`
-   Masking: add `-inf` to disallowed positions (e.g., `j > i` for causal).
-   Weights: `P = softmax(A, dim=-1)`
-   Output: `Y = P @ V` → shape `[B, L, d]`
-   Position info: with RoPE, apply rotations to `Q` and `K` before `Q @ K^T`.

#### 1.2 Worked Mini-Example

-   Suppose `L=3`, `d=2`, `Q = [[q11,q12], ...]`, `K = [[k11,k12], ...]`.
-   For token `i=2`, logits are dot products with all `j≤2`: `A[2,j] = (Q[2]·K[j]) / sqrt(2)`; apply softmax over `j ∈ {0,1,2}`.
-   The output `Y[2]` is a convex combination of `V[0..2]` weighted by those probabilities.

#### 1.3 Pseudocode (masking and stability)

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q,K,V: [B, L, d]
    scale = 1.0 / math.sqrt(Q.size(-1))
    logits = torch.einsum('bld,bmd->blm', Q, K) * scale  # [B,L,L]
    if mask is not None:
        logits = logits.masked_fill(mask == 0, float('-inf'))
    probs = torch.softmax(logits, dim=-1)
    return torch.einsum('blm,bmd->bld', probs, V)
```

---

### 2) Multi-Head Attention (MHA)

-   **Why multiple heads**: Different heads specialize (syntax, long-range links, entity tracking, etc.). Heads attend to different subspaces.
-   **How**: Split `d_model` across `num_heads` projections, compute attention per head, then concatenate and project.
-   **Trade-offs**:
    -   **Pros**: Highest model quality and representational richness.
    -   **Cons**: KV cache scales with `num_heads`, increasing inference memory and bandwidth.

#### 2.1 Shapes and Projections

-   Let `D` be model width, `H` be number of heads, head width `d_h = D / H`.
-   Combined projections: `W_q: [D, H*d_h]`, `W_k: [D, H*d_h]`, `W_v: [D, H*d_h]`.
-   After projection, reshape to heads: `Q,K,V: [B, H, L, d_h]`.
-   Compute attention independently per head, then `concat` along `d_h` → `[B, L, H*d_h] = [B, L, D]`, then apply output projection `W_o: [D, D]`.

#### 2.2 Cost and Typical Settings

-   Compute per layer roughly `O(B * H * L^2 * d_h)`; memory for attention logits `O(B * H * L^2)`.
-   Typical example: `D=4096, H=32, d_h=128`. Many open models use `H ∈ [24, 40]` for 7–70B scales.

```python
# Sketch for multi-head projection
Q = X @ W_q; K = X @ W_k; V = X @ W_v               # [B,L,D]
Q = Q.view(B, L, H, d_h).transpose(1, 2)            # [B,H,L,d_h]
# ... attention per head ...
Y = Y_heads.transpose(1, 2).contiguous().view(B, L, D) @ W_o
```

---

### 3) Multi-Query Attention (MQA)

-   **Idea**: Keep many query heads but share a single set of K/V across all heads.
-   **Why**: At inference, the KV cache dominates memory and latency. Sharing K/V reduces cache size by ~`num_heads`×.
-   **Effect**:
    -   **Memory/Speed**: Large improvement—smaller KV cache, less memory movement, faster decoding.
    -   **Quality**: Slightly lower than full MHA in some tasks, often negligible in practice for many deployments.
-   **When to use**: Production inference where throughput and cost matter (APIs, mobile, edge), long context windows.

#### 3.1 KV Cache Math (per layer, per token, float16)

-   KV cache stores `K` and `V` for past tokens.
-   Size (floats) per token: `2 * num_kv_heads * d_h`.
-   Bytes per token: above × 2 (for float16).
-   Example (`D=4096, H=32, d_h=128`):
    -   MHA: `num_kv_heads=32` → floats `2*32*128 = 8192` → `8192*2 = 16384` bytes ≈ 16 KB/token/layer.
    -   MQA: `num_kv_heads=1` → floats `2*1*128 = 256` → `256*2 = 512` bytes/token/layer.
-   With `L=8192`, `Layers=32`: MHA ≈ `16 KB * 8192 * 32 ≈ 4 TB` total bytes across all tokens (streamed over time), whereas per active context window the live cache per sequence is `16 KB * 8192 ≈ 128 MB` per layer; MQA reduces this by 32×.

#### 3.2 Practical Notes

-   Query heads remain `H` to preserve modeling capacity; only K/V are shared.
-   Very friendly to FlashAttention kernels and long contexts.
-   Quality gaps can often be closed with training tricks (better data, RoPE scaling, finetuning).

#### 3.3 What Exactly Is Shared?

-   Projection weights: use one `W_k: [D, d_h]` and one `W_v: [D, d_h]` for all heads (instead of per-head splits).
-   Runtime tensors and cache: a single `K: [B, L, d_h]` and `V: [B, L, d_h]` are referenced by every query head; the KV cache stores this single set per layer and position.
-   Queries remain multi-headed: `Q: [B, H, L, d_h]` so each head can attend differently over the same shared `K,V`.

---

### 4) Grouped-Query Attention (GQA)

-   **Idea**: A compromise between MHA and MQA—divide heads into `G` groups; each group shares one K/V set.
-   **Why**: Regain some of MHA's quality while keeping KV cache much smaller than MHA.
-   **Effect**:
    -   **Memory/Speed**: KV cache shrinks by ~`num_heads / G`.
    -   **Quality**: Closer to MHA than MQA, especially with tuned group sizes (e.g., 4–8 groups).
-   **When to use**: You need better quality than MQA but still want major inference savings.

#### 4.1 KV Cache Math and Example

-   `num_kv_heads = G` (groups).
-   Bytes per token per layer (float16): `2 * G * d_h * 2`.
-   With `D=4096, H=32, d_h=128, G=4`: `2*4*128*2 = 2048` bytes = 2 KB/token/layer → 8× smaller than MHA, 4× larger than MQA.

#### 4.2 Implementation Hint

-   Maintain a mapping from query head index → group index to route attention to the shared K/V for that group.

#### 4.3 What Exactly Is Shared?

-   Projection weights per group: one `W_k^g: [D, d_h]` and `W_v^g: [D, d_h]` per group `g ∈ {1..G}`.
-   Runtime tensors/cache per group: `K^g: [B, L, d_h]`, `V^g: [B, L, d_h]` that all heads in group `g` read from and write to in the KV cache.
-   Queries remain per-head: `Q: [B, H, L, d_h]`, but K/V are shared within their assigned group.

---

### 5) Sequential Learning with Causal Transformers

-   **Basic principle**: Autoregressive factorization of sequence probability.
    -   Model learns `p(x_1, …, x_T) = ∏_{t=1..T} p(x_t | x_{<t})`.
-   **Training (teacher forcing)**:
    -   Shift inputs vs. targets by one token; apply a causal mask; minimize next-token cross-entropy.
    -   Benefits: Stable optimization, parallelizable training across time steps.
-   **Inference (generation)**:
    -   Feed tokens left-to-right, caching K/V to avoid recomputing history.
    -   Sampling controls: temperature, top-k/top-p, repetition penalties.
-   **Why it works well**:
    -   Scales efficiently on accelerators; aligns with left-to-right language production.
    -   Composable with improvements: RoPE for position, SWA for long contexts, MQA/GQA for efficient KV cache.
-   **Limitations and mitigations**:
    -   No future context (by design); for bidirectional tasks, use retrieval, reranking, or different architectures.
    -   Long-range dependencies: use SWA with transitive propagation, attention sinks, or retrieval augmentation.

#### 5.1 Pseudocode for Training vs Inference

```python
# Training (teacher forcing)
logits = model(input_ids[:, :-1])           # predict next token
loss = cross_entropy(logits, input_ids[:, 1:])

# Inference (KV caching)
for t in range(T):
    logits, kv_cache = model.step(input_ids[:, t], kv_cache)
    next_id = sample(logits, temp=0.8, top_p=0.95)
    emit(next_id)
```

---

### 6) Choosing Between MHA, MQA, and GQA

-   **If quality is paramount and memory is ample**: Use **MHA**.
-   **If inference speed/cost and context length dominate**: Use **MQA**.
-   **If you need a middle ground**: Use **GQA** with 4–8 groups; tune group count empirically.
-   **Context window**:
    -   Longer windows greatly increase KV cache size; MQA/GQA scale better.
    -   With Sliding Window Attention (SWA), consider keeping small "attention sinks" to stabilize global context.
-   **Deployment tips**:
    -   Pair with RoPE for robust relative position handling and extrapolation.
    -   Use FlashAttention or fused kernels; fuse causal masks and biases for speed.
    -   Profile with your sequence lengths and batch sizes; KV bandwidth often bottlenecks throughput.

#### 6.1 Concrete Selection Heuristics

-   Batch=1 streaming, very long contexts (≥32k): prefer MQA.
-   Batch>4 with moderate contexts (4k–8k) and quality-sensitive tasks: try GQA with `G∈{4,8}`.
-   Offline training/evaluation where memory is ample: MHA for maximum quality.

---

### 7) Implementation Notes (KV Cache and Compatibility)

-   **KV cache size** (per layer, per token): proportional to `num_kv_heads * head_dim`.
    -   MHA: `num_kv_heads = num_heads`.
    -   MQA: `num_kv_heads = 1`.
    -   GQA: `num_kv_heads = num_groups`.
-   **Compatibility**: RoPE, causal masks, ALiBi/relative biases, FlashAttention all work with MHA/MQA/GQA.
-   **Attention sinks**: Reserve a few early positions, keep their K/V always in cache, and (optionally) apply a small positive bias toward them to stabilize streaming.

#### 7.1 Streaming and Sliding Window Attention (SWA)

-   With SWA of window `W`, keep K/V only for the last `W` tokens.
-   To maintain global context, retain `m` sink positions (e.g., BOS plus few early tokens) across evictions and add a small positive bias toward them.
-   Information propagates forward layer-by-layer even if direct attention to far past is cut, but sinks improve stability.

#### 7.2 Memory/Complexity Summary

-   Attention compute: `O(B * H * L^2 * d_h)`; softmax memory `O(B * H * L^2)`.
-   KV cache live memory per sequence: `(2 * num_kv_heads * d_h * bytes_per_scalar) * L * num_layers`.
-   Reductions with MQA/GQA are often the largest lever for faster inference.

#### 7.3 Common Pitfalls

-   Forgetting `sqrt(d_h)` scaling or applying it twice.
-   Mask dtype or broadcasting errors leading to leaking future tokens.
-   Applying RoPE after projection but before head split—ensure rotations act on per-head `d_h` pairs.
-   Mixed precision overflows: keep logits/bias in `float32` before softmax when needed.
-   Inconsistent KV eviction with SWA causing misaligned keys/values across layers.
-   Wrong head→group mapping in GQA.

---

### 8) Further Reading

-   [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
-   [RoPE: Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
-   [ALiBi: Train Short, Test Long](https://arxiv.org/abs/2108.12409)
-   [FlashAttention](https://arxiv.org/abs/2205.14135)
-   [Efficient Streaming with Attention Sinks](https://arxiv.org/abs/2309.17453)


