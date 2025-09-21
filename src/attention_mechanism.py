import torch
import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
import numpy as np
from tokenizers import Tokenizer
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt

torch.manual_seed(123)


#Scaled Dot-Product Attention


class ScaledDotProductAttention(nn.Module):
    """ Basic scaled dot-production attention mechanism"""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.size()
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_model)
        if mask is not None:
            scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, value)
        return output, attention_weights

    
class MultiHeadAttention(nn.Module):
    """ Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """

        batch_size, seq_len, d_model = query.size()
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert key.size() == value.size()
        assert mask is None or mask.size() == (batch_size, seq_len, seq_len)

        # Linear projections
        query = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key = self.w_k(key).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = self.w_v(value).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        attention_values = []
        attention_weights_list = []

        for i in range(self.num_heads):
            head_output, head_attention = self.attention(query[:, i], key[:, i], value[:, i], mask)
            attention_values.append(head_output)
            attention_weights_list.append(head_attention)

        # Concatenate heads
        attention = torch.stack(attention_values, dim=1)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(attention)

        attention_weights = torch.stack(attention_weights_list, dim=1) # [batch_size, num_heads, seq_len, seq_len]

        return output, attention_weights


class MultiQueryAttention(nn.Module):
    """ Multi-query attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, self.d_k)
        self.w_v = nn.Linear(d_model, self.d_k)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """

        batch_size, seq_len, d_model = query.size()

        #Project queries (multiple heads)
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        #Project keys and values (single head each)
        K = self.w_k(key).unsqueeze(1)  # [batch, 1, seq_len, d_k]
        V = self.w_v(value).unsqueeze(1)  # [batch, 1, seq_len, d_k]

        #Expand K and V to match number of query heads
        K = K.expand(batch_size, self.num_heads, seq_len, self.d_k)
        V = V.expand(batch_size, self.num_heads, seq_len, self.d_k)

        # Apply Attention to each head (K and V are shared)
        attention_values = []
        attention_weights_list = []

        for i in range(self.num_heads):
            head_output, head_attention = self.attention(Q[:, i], K[:, i], V[:, i], mask)
            attention_values.append(head_output)
            attention_weights_list.append(head_attention)

        attention_output = torch.stack(attention_values, dim=2)
        # understand what is operation is doing
        # it is concatenating the attention values from each head
        # and then transposing the result to the correct shape
        # and then viewing the result as a contiguous tensor with the correct shape
        # and then applying the final linear projection
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(attention_output)

        attention_weights = torch.stack(attention_weights_list, dim=1)

        return output, attention_weights



class GroupedQueryAttention(nn.Module):
    """ Grouped query attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        self.dropout = nn.Dropout(dropout)
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.w_o = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k, dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len, d_model]
            key: [batch_size, seq_len, d_model]
            value: [batch_size, seq_len, d_model]
            mask: [batch_size, seq_len, seq_len] or None

        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = query.size()
        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        assert key.size() == value.size()
        assert mask is None or mask.size() == (batch_size, seq_len, seq_len)

        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Project keys and values (reduced number of heads)
        K = self.w_k(key).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)


        attention_values = []
        attention_weights_list = []

        for i in range(self.num_heads):
            kv_head_idx = i // self.num_queries_per_kv
            head_output, head_attention = self.attention(Q[:, i], K[:, kv_head_idx], V[:, kv_head_idx], mask)
            attention_values.append(head_output)
            attention_weights_list.append(head_attention)

        attention_output = torch.stack(attention_values, dim=2)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        output = self.w_o(attention_output)

        attention_weights = torch.stack(attention_weights_list, dim=1)

        return output, attention_weights


class CausalAttention(nn.Module):
    """ Causal attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create lower triangular mask for causal attention."""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0)  # Add batch dimension
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        
        Returns:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.size()
        mask = self.create_causal_mask(seq_len, x.device)
        output, attention_weights = self.multi_head_attention(x, x, x, mask)
        return output, attention_weights


class ParameterAnalyzer:
    """ Utility class for analyzing attention mechanism efficiency"""

    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def analyze_attention_efficiency(d_model: int, num_heads: int, num_kv_heads: int = None):
        """
        Analyze and compare efficiency of different attention mechanisms.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_kv_heads: Number of key-value heads (for GQA)
        """
        print(f"\n📊 Attention Mechanisms Efficiency Analysis")
        print(f"Model dimension (d_model): {d_model}")
        print(f"Number of attention heads (num_heads): {num_heads}")
        if num_kv_heads:
            print(f"Number of key-value heads (num_kv_heads): {num_kv_heads}")
        print("=" * 60)

        # Create instances
        mha = MultiHeadAttention(d_model, num_heads)
        mqa = MultiQueryAttention(d_model, num_heads)
        if num_kv_heads:
            gqa = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
        
        mha_params = ParameterAnalyzer.count_parameters(mha)
        mqa_params = ParameterAnalyzer.count_parameters(mqa)
        if num_kv_heads:
            gqa_params = ParameterAnalyzer.count_parameters(gqa)

        print(f"Multi-Head Attention (MHA) parameters: {mha_params:,}")
        print(f"Multi-Query Attention (MQA) parameters: {mqa_params:,}")
        print(f"MQA parameter reduction: {(1 - mqa_params/mha_params)*100:.1f}%")
        if num_kv_heads:
            print(f"Grouped-Query Attention (GQA) parameters: {gqa_params:,}")
            print(f"GQA parameter reduction: {(1 - gqa_params/mha_params)*100:.1f}%")
        
        return mha_params, mqa_params, gqa_params if num_kv_heads else None

class AttentionVisualizer:
    """ Utility class for visualizing attention weights"""

    @staticmethod
    def plot_attention_weights(attention_weights: torch.Tensor, tokens: list, 
    title: str = "Attention Weights", head_idx: int = 0):
        """
        Plot attention weights as a heatmap.

        Args:
            attention_weights: [batch_size, num_heads, seq_len, seq_len] or [batch_size, seq_len, seq_len]
            tokens: List of token strings
            title: Plot title
            head_idx: Which attention head to visualize (if multi-head)
        """
        if len(attention_weights.shape) == 4:
            weights = attention_weights[0, head_idx].detach().cpu().numpy()
        else:
            weights = attention_weights[0].detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(weights, xticklabels=tokens, 
                    yticklabels=tokens, 
                    annot=True if len(tokens) <= 10 else False,
                    fmt='.2f',
                    cmap="viridis")
        plt.title(f"{title} (Head {head_idx})")
        plt.xlabel("Key Tokens")
        plt.ylabel("Query Tokens")
        plt.tight_layout()
        plt.show()
    

    @staticmethod
    def compare_attention_patterns(attention_weights_list: list, tokens: list, 
    titles: list, head_idx: int = 0):
        """
        Compare multiple attention patterns side by side.
        """
        fig, axes = plt.subplots(1, len(attention_weights_list), figsize=(15, 5))
        if len(attention_weights_list) == 1:
            axes = [axes]
        for i, (weights, title) in enumerate(zip(attention_weights_list, titles)):
            if len(weights.shape) == 4:
                weights = weights[0, head_idx].detach().cpu().numpy()
            else:
                weights = weights[0].detach().cpu().numpy()
            sns.heatmap(weights, xticklabels=tokens, 
                            yticklabels=tokens, 
                            annot=True if len(tokens) <= 10 else False, 
                            fmt='.2f', cmap="viridis", 
                            ax=axes[i])
            axes[i].set_title(title)
            axes[i].set_xlabel("Key Tokens")
            if i == 0:
                axes[i].set_ylabel("Query Tokens")
        plt.tight_layout()
        plt.show()

def load_trained_tokenizer(tokenizer_path: str) -> Optional[Tokenizer]:
    """Load our trained BPE tokenizer with fallback."""
    try:
        tokenizer = Tokenizer.from_file(tokenizer_path)
        print(f"✅ Successfully loaded custom tokenizer (vocab size: {tokenizer.get_vocab_size()})")
        return tokenizer
    except Exception as e:
        print(f"⚠️  Could not load tokenizer: {e}")
        print("Using fallback tokenization...")
        return None


def prepare_sample_text(tokenizer: Optional[Tokenizer], text: str, max_length: int = 16) -> Tuple[torch.Tensor, list]:
    """
    Tokenize text and prepare embeddings for attention demo.
    
    Returns:
        embeddings: [1, seq_len, d_model] - random embeddings for demo
        tokens: List of decoded token strings
    """
    if tokenizer is not None:
        try:
            # Use trained tokenizer
            encoded = tokenizer.encode(text)
            token_ids = encoded.ids[:max_length]  # Limit sequence length
            tokens = [tokenizer.decode([token_id]) for token_id in token_ids]
        except Exception:
            # Fallback to simple word splitting
            tokens = text.split()[:max_length]
    else:
        # Fallback to simple word splitting
        tokens = text.split()[:max_length]
    
    d_model = 256
    embeddings = torch.randn(1, len(tokens), d_model)
    return embeddings, tokens



def demonstrate_attention_mechanisms():
    """Main demonstration function."""
    print("\n🚀 Demonstrating Attention Mechanisms")
    print("=" * 60)

    # Load our trained tokenizer
    tokenizer_path = "/Users/rajatpatel/research/learn_gpt_oss/src/tokenizer_data/tokenizer.json"
    tokenizer = load_trained_tokenizer(tokenizer_path)
    
    # Sample text from Shakespeare
    sample_text = "To be or not to be, that is the question"
    print(f"Sample text: '{sample_text}'")
    
    # Prepare embeddings and tokens
    embeddings, tokens = prepare_sample_text(tokenizer, sample_text, max_length=12)
    print(f"Tokens: {tokens}")
    print(f"Embeddings shape: {embeddings.shape}")
    print()

        # Initialize attention mechanisms
    d_model = embeddings.size(-1)
    num_heads = 16
    num_kv_heads = 4 # For GQA
    
    basic_attention = ScaledDotProductAttention(d_model)
    multi_head_attention = MultiHeadAttention(d_model, num_heads)
    multi_query_attention = MultiQueryAttention(d_model, num_heads)
    grouped_query_attention = GroupedQueryAttention(d_model, num_heads, num_kv_heads)
    causal_attention = CausalAttention(d_model, num_heads)


        # Demonstrate different attention mechanisms
    print("1. Basic Scaled Dot-Product Attention")
    print("-" * 40)
    basic_output, basic_weights = basic_attention(embeddings, embeddings, embeddings)
    print(f"Output shape: {basic_output.shape}")
    print(f"Attention weights shape: {basic_weights.shape}")
    
    print("\n2. Multi-Head Attention (MHA)")
    print("-" * 40)
    mh_output, mh_weights = multi_head_attention(embeddings, embeddings, embeddings)
    print(f"Output shape: {mh_output.shape}")
    print(f"Attention weights shape: {mh_weights.shape}")
    
    print("\n3. Multi-Query Attention (MQA)")
    print("-" * 40)
    mq_output, mq_weights = multi_query_attention(embeddings, embeddings, embeddings)
    print(f"Output shape: {mq_output.shape}")
    print(f"Attention weights shape: {mq_weights.shape}")
    
    print("\n4. Grouped Query Attention (GQA)")
    print("-" * 40)
    gq_output, gq_weights = grouped_query_attention(embeddings, embeddings, embeddings)
    print(f"Output shape: {gq_output.shape}")
    print(f"Attention weights shape: {gq_weights.shape}")
    
    print("\n5. Causal (Masked) Attention")
    print("-" * 40)
    causal_output, causal_weights = causal_attention(embeddings)
    print(f"Output shape: {causal_output.shape}")
    print(f"Attention weights shape: {causal_weights.shape}")
    
    # Efficiency Analysis
    print("\n" + "="*60)
    ParameterAnalyzer.analyze_attention_efficiency(d_model, num_heads, num_kv_heads)
    
    # Visualize attention patterns
    print("\n📊 Visualizing Attention Patterns...")
    visualizer = AttentionVisualizer()
    
    # Compare MHA vs MQA vs GQA
    attention_weights_list = [mh_weights, mq_weights, gq_weights]
    titles = ["Multi-Head Attention (MHA)", "Multi-Query Attention (MQA)", "Grouped Query Attention (GQA)"]
    
    visualizer.compare_attention_patterns(attention_weights_list, tokens, titles, head_idx=0)
    
    # Show causal attention pattern
    print("\n🔒 Causal Attention Pattern:")
    visualizer.plot_attention_weights(causal_weights, tokens, "Causal Attention", head_idx=0)
    
    # Compare different heads in GQA
    print("\n🔍 GQA - Different Query Heads sharing KV heads:")
    for head in [0, num_kv_heads, num_kv_heads*2]:  # Show heads from different groups
        if head < num_heads:
            visualizer.plot_attention_weights(gq_weights, tokens, 
                                            f"GQA Head {head} (KV group {head//4})", head_idx=head)
    
    print("\n✅ Demonstration complete!")
    print("\nKey Observations:")
    print("- MHA: Each head has its own Q, K, V projections - most expressive")
    print("- MQA: All heads share single K, V - most memory efficient") 
    print("- GQA: Groups of heads share K, V - balanced trade-off")
    print("- Causal: Each position can only attend to previous positions")
    print("- GQA is used in Llama-2, MQA in PaLM, traditional MHA in GPT-2/BERT")


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    demonstrate_attention_mechanisms()