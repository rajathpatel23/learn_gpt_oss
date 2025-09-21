# Learn GPT OSS

A comprehensive educational repository for understanding and implementing core components of modern GPT-style language models.

## 🎯 What's Inside

### Core Implementations
- **Attention Mechanisms**: Complete implementations of Multi-Head (MHA), Multi-Query (MQA), and Grouped-Query (GQA) attention with performance comparisons
- **Tokenization**: Byte-Pair Encoding (BPE) from scratch, custom tokenizer training
- **Attention Sinks**: Streaming LLM with efficient KV cache management for long contexts
- **Modern Optimizations**: RoPE positional encoding, RMS normalization, sliding window attention

### Key Features
- **Parameter Efficiency Analysis**: MQA achieves 46.9% reduction, GQA achieves 37.5% reduction vs MHA
- **Practical Examples**: Working implementations using Shakespeare dataset
- **Visualization Tools**: Attention pattern analysis and cache behavior visualization
- **Production Considerations**: Memory optimization, inference efficiency, streaming generation

## 📚 Learning Path

1. **Start Here**: `src/guides/BLOG_README.md` - Intuitive overview of attention variants
2. **Deep Dive**: `src/guides/attention_mechanisms_comprehensive.md` - Mathematical foundations
3. **Implementation**: `src/attention_mechanism.py` - Hands-on coding examples
4. **Advanced**: `src/attention_sink/` - Streaming and long-context handling
5. **Tokenization**: `src/tokenization/` - Complete tokenization pipeline

## 🚀 Quick Start

```bash
pip install -r requirements.txt
python src/attention_mechanism.py  # Run attention comparisons
python src/tokenization/tokenizer.py  # Train custom tokenizer
python src/attention_sink/cli.py  # Try streaming generation
```

**Focus**: Understanding transformer internals through practical implementation and clear explanations.

