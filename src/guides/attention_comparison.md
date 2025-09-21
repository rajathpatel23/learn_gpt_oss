# Attention Mechanisms Comparison: MHA vs MQA vs GQA

## Overview

This document explains the different attention mechanisms implemented in our demonstration, focusing on Multi-Head Attention (MHA), Multi-Query Attention (MQA), and Grouped Query Attention (GQA). We'll explore both the theoretical foundations and practical implementations with real code outputs.

## Practical Demonstration Results

Our implementation demonstrates these mechanisms using the text: *"To be or not to be, that is the question"*

**Sample Output:**
```
🚀 Demonstrating Attention Mechanisms
============================================================
Sample text: 'To be or not to be, that is the question'
Tokens: ['To', 'be', 'or', 'not', 'to', 'be,', 'that', 'is', 'the', 'question']
Embeddings shape: torch.Size([1, 10, 256])

1. Basic Scaled Dot-Product Attention
Output shape: torch.Size([1, 10, 256])
Attention weights shape: torch.Size([1, 10, 10])

2. Multi-Head Attention (MHA)
Output shape: torch.Size([1, 10, 256])
Attention weights shape: torch.Size([1, 16, 10, 10])

3. Multi-Query Attention (MQA)
Output shape: torch.Size([1, 10, 256])
Attention weights shape: torch.Size([1, 16, 10, 10])

4. Grouped Query Attention (GQA)
Output shape: torch.Size([1, 10, 256])
Attention weights shape: torch.Size([1, 16, 10, 10])
```

## 1. Multi-Head Attention (MHA) - Traditional Approach

### How it works:
- **Separate projections**: Each head has its own Query (Q), Key (K), and Value (V) linear projections
- **Independent heads**: All heads learn different attention patterns independently
- **Full expressiveness**: Maximum capacity to learn diverse attention patterns

### Parameters:
- Q projection: `d_model × d_model`
- K projection: `d_model × d_model` 
- V projection: `d_model × d_model`
- Output projection: `d_model × d_model`
- **Total**: `4 × d_model²` parameters

### Used in:
- BERT, GPT-2, T5, original Transformer

## 2. Multi-Query Attention (MQA) - Maximum Efficiency

### How it works:
- **Shared K,V**: Single Key and Value projections shared across all query heads
- **Multiple Q**: Each head still has its own Query projection
- **Significant memory savings**: Reduces KV cache size during inference

### Parameters:
- Q projection: `d_model × d_model` (multiple heads)
- K projection: `d_model × d_k` (single head)
- V projection: `d_model × d_k` (single head) 
- Output projection: `d_model × d_model`
- **Total**: `~3 × d_model²` parameters (43.8% reduction in our demo)

### Used in:
- PaLM, Chinchilla, Falcon

### Trade-offs:
- ✅ Much lower memory usage during inference
- ✅ Faster generation (smaller KV cache)
- ⚠️ Slight performance degradation on some tasks

## 3. Grouped Query Attention (GQA) - Balanced Approach

### How it works:
- **Grouped sharing**: Multiple query heads share the same K,V projections
- **Configurable groups**: `num_heads / num_kv_heads` queries per K,V pair
- **Middle ground**: Between MHA expressiveness and MQA efficiency

### Parameters:
- Q projection: `d_model × d_model` (all heads)
- K projection: `d_model × (num_kv_heads × d_k)` 
- V projection: `d_model × (num_kv_heads × d_k)`
- Output projection: `d_model × d_model`
- **Total**: Varies based on `num_kv_heads` (37.5% reduction with 2 KV heads in our demo)

### Used in:
- Llama-2, Code Llama, Mistral

### Trade-offs:
- ✅ Good balance of efficiency and performance
- ✅ Configurable trade-off via `num_kv_heads`
- ✅ Better than MQA on most benchmarks

## 4. Implementation Insights

### Key Differences in Code:

```python
# MHA: Each head has separate K,V
for i in range(num_heads):
    Q_i, K_i, V_i = separate_projections[i]
    
# MQA: All heads share single K,V
K_shared, V_shared = single_projection()
for i in range(num_heads):
    Q_i, K_shared, V_shared
    
# GQA: Groups share K,V
for i in range(num_heads):
    group_idx = i // queries_per_group
    Q_i, K_group[group_idx], V_group[group_idx]
```

### Memory Usage During Inference:

| Mechanism | KV Cache Size | Relative Memory |
|-----------|---------------|-----------------|
| MHA       | `num_heads × seq_len × d_k` | 100% |
| MQA       | `1 × seq_len × d_k` | ~12.5% (8 heads) |
| GQA       | `num_kv_heads × seq_len × d_k` | 25% (2 KV heads) |

## 5. When to Use Each

### Choose MHA when:
- Maximum model quality is required
- Memory/compute is not a constraint
- Research/experimentation phase

### Choose MQA when:
- Memory efficiency is critical
- Fast inference is required
- Serving large models with long sequences

### Choose GQA when:
- Need balance between quality and efficiency
- Want configurable trade-off
- Building production systems

## 6. Performance Comparison

From our efficiency analysis with d_model=256, num_heads=16:

```
📊 Attention Mechanisms Efficiency Analysis
Multi-Head Attention (MHA) parameters: 263,168
Multi-Query Attention (MQA) parameters: 139,808  (46.9% reduction)
Grouped Query Attention (GQA) parameters: 164,480 (37.5% reduction)
```

### Parameter Breakdown Analysis

The significant parameter reduction comes from sharing Key and Value projections:

**MHA (Traditional):**
- Q projection: 256 × 256 = 65,536 parameters
- K projection: 256 × 256 = 65,536 parameters  
- V projection: 256 × 256 = 65,536 parameters
- Output projection: 256 × 256 = 65,536 parameters
- **Total: ~262K parameters**

**MQA (Maximum Efficiency):**
- Q projection: 256 × 256 = 65,536 parameters (16 heads)
- K projection: 256 × 16 = 4,096 parameters (1 shared head)
- V projection: 256 × 16 = 4,096 parameters (1 shared head)
- Output projection: 256 × 256 = 65,536 parameters
- **Total: ~139K parameters (46.9% reduction)**

**GQA (Balanced Approach):**
- Q projection: 256 × 256 = 65,536 parameters (16 heads)
- K projection: 256 × 64 = 16,384 parameters (4 KV heads)
- V projection: 256 × 64 = 16,384 parameters (4 KV heads)
- Output projection: 256 × 256 = 65,536 parameters
- **Total: ~164K parameters (37.5% reduction)**

## 7. Real-World Usage

- **Llama-2**: Uses GQA with 8 query heads sharing 1 KV head per group
- **PaLM**: Uses MQA for efficiency in large-scale deployment  
- **GPT-4**: Likely uses advanced variants of these mechanisms
- **Code models**: Often use GQA for balance of code understanding and efficiency

## 8. Code Implementation Insights

Our implementation reveals key architectural differences:

### MHA Implementation
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        # Each head gets full projections
        self.w_q = nn.Linear(d_model, d_model)  # 256 → 256
        self.w_k = nn.Linear(d_model, d_model)  # 256 → 256  
        self.w_v = nn.Linear(d_model, d_model)  # 256 → 256
        self.w_o = nn.Linear(d_model, d_model)
```

### MQA Implementation
```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        # Queries get full projection, K,V get reduced
        self.w_q = nn.Linear(d_model, d_model)      # 256 → 256
        self.w_k = nn.Linear(d_model, self.d_k)     # 256 → 16 (shared)
        self.w_v = nn.Linear(d_model, self.d_k)     # 256 → 16 (shared)
        self.w_o = nn.Linear(d_model, d_model)
```

### GQA Implementation  
```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int):
        # Grouped sharing of K,V projections
        self.w_q = nn.Linear(d_model, d_model)                    # 256 → 256
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)   # 256 → 64
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)   # 256 → 64
        self.w_o = nn.Linear(d_model, d_model)
```

### Attention Pattern Analysis

The attention weights reveal how each mechanism processes information:

- **MHA**: Each head learns completely independent patterns
- **MQA**: All heads share the same Key/Value representations but have different Query patterns
- **GQA**: Groups of heads share Key/Value representations, creating a middle ground

## Conclusion

The evolution from MHA → GQA → MQA represents the field's focus on making large language models more efficient while maintaining performance. Our analysis shows:

1. **Parameter Efficiency**: MQA achieves 46.9% parameter reduction, GQA achieves 37.5%
2. **Memory Efficiency**: During inference, KV cache size scales with number of KV heads
3. **Performance Trade-off**: GQA has emerged as the sweet spot for many applications

**Key Takeaway**: GQA offers most of MHA's expressiveness with significant efficiency gains, which is why it's adopted in modern models like Llama-2 and Mistral.
