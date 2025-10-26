import json 
import math
import os
from dataclasses import dataclass
from typing import Optional
import logging
logger = logging.getLogger(__name__)
import torch
import torch.distributed as dist



@dataclass
class ModelConfig:
    num_hidden_layers: int = 3
    num_experts: int = 4
    experts_per_token: int = 2
    vocab_size: int = 201088
    hidden_size: int = 512
    intermediate_size: int = 512
    swiglu_limit: float = 7.0
    head_dim: int = 32
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    sliding_window: int = 16
    initial_context_length: int = 1024
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_ntk_alpha: float = 1.0
    rope_ntk_beta: float = 32.0
    use_attention_sink: bool = True





class RMSNorm(torch.nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-05, device: Optional[torch.device] = None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.scale = torch.nn.Parameter(torch.ones(num_features, device=device, dtype=torch.float32))   

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.num_features
        t, dtype = x.float(), x.dtype
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.scale).to(dtype)

def _apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(device=x.device, dtype=x.dtype)
    sin = sin.unsqueeze(-2).to(device=x.device, dtype=x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim: int, base: int, dtype: torch.dtype, initial_context_length: int = 1024, scaling_factor: float = 32.0, ntk_alpha: float = 1.0, ntk_beta: float = 32.0, device: Optional[torch.device] = None):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.device = device
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        
    def _compute_concentration_and_inv_freq(self) -> tuple[torch.Tensor, torch.Tensor]:
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            d_half = self.head_dim / 2  # half of the head dimension
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)
        cos = cos.to(device=query.device)
        sin = sin.to(device=query.device)
        q_shape = query.shape
        k_shape = key.shape
        q_flat = query.view(num_tokens, -1, self.head_dim)
        k_flat = key.view(num_tokens, -1, self.head_dim)
        q_flat = _apply_rotary_emb(q_flat, cos, sin)
        k_flat = _apply_rotary_emb(k_flat, cos, sin)
        return q_flat.view(q_shape), k_flat.view(k_shape)


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    if S is not None:
        S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK = QK.float() * float(sm_scale)
    QK += mask[None, None, :, :]
    # stabilize before softmax
    QK = QK - torch.amax(QK, dim=-1, keepdim=True)
    if S is not None:
        QK = torch.cat([QK, S], dim=-1)
        W = torch.softmax(QK.float(), dim=-1).to(V.dtype)
        W = W[..., :-1]
    else:
        W = torch.softmax(QK.float(), dim=-1).to(V.dtype)
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)

class AttentionBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0, device: Optional[torch.device] = None):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        # Use float32 everywhere for stability
        dtype = torch.float32
        self.use_attention_sink = config.use_attention_sink
        if self.use_attention_sink:
            self.sinks = torch.nn.Parameter(
                torch.zeros(config.num_attention_heads, device=device, dtype=dtype)
            )
        else:
            self.sinks = None
        self.norm = RMSNorm(config.hidden_size, device=device)
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(config.hidden_size, qkv_dim, device=device, dtype=dtype)
        self.out = torch.nn.Linear(config.head_dim * config.num_attention_heads, config.hidden_size, device=device, dtype=dtype)
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        self.rope = RotaryEmbedding(config.head_dim, config.rope_theta, torch.float32, initial_context_length=config.initial_context_length, scaling_factor=config.rope_scaling_factor, ntk_alpha=config.rope_ntk_alpha, ntk_beta=config.rope_ntk_beta, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.norm(x)
        qkv = self.qkv(t)
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads * self.head_dim : (self.num_attention_heads + self.num_key_value_heads) * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads) * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
        ].contiguous()
        q = q.view(-1, self.num_key_value_heads, self.num_attention_heads // self.num_key_value_heads, self.head_dim)
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        q, k = self.rope(q, k)
        # adding sinks to the attention scores
        # sdpa is the scaled dot-product attention with sinks
        # q is the query tensor
        # k is the key tensor
        # v is the value tensor
        # self.sinks is the sinks tensor
        # self.sm_scale is the scale factor
        # self.sliding_window is the sliding window size
        # t is the attention scores
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        t = x + t
        return t

def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    return out_glu * (x_linear + 1)

class MLPBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.norm = RMSNorm(config.hidden_size, device=device)
        dtype = torch.float32
        self.gate = torch.nn.Linear(config.hidden_size, config.num_experts, device=device, dtype=dtype)
        assert config.intermediate_size % self.world_size == 0
        self.experts = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.intermediate_size * 2 // self.world_size, device=device, dtype=dtype),
                torch.nn.Linear(config.intermediate_size // self.world_size, config.hidden_size, device=device, dtype=dtype)
            ) for _ in range(config.num_experts)
        ])


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len, hidden_size = x.shape
        t = self.norm(x)
        g = self.gate(t)
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        expert_weights = torch.nn.functional.softmax(experts.values, dim=-1)
        expert_indices = experts.indices
        t_flat = t.view(-1, hidden_size)
        expert_indices_flat = expert_indices.view(-1, self.experts_per_token)
        expert_weights_flat = expert_weights.view(-1, self.experts_per_token)
        output = torch.zeros_like(t_flat)
        for expert_idx in range(self.num_experts):
            mask = (expert_indices_flat == expert_idx).any(dim=-1)
            if not mask.any():
                continue
            token_indices = torch.where(mask)[0]
            expert_pos = (expert_indices_flat[token_indices] == expert_idx).nonzero(as_tuple=True)[1]
            expert_input = t_flat[token_indices]
            weights = expert_weights_flat[token_indices, expert_pos]
            expert_out = expert_input
            expert_out = self.experts[expert_idx][0](expert_out)
            expert_out = swiglu(expert_out, limit=self.swiglu_limit)
            expert_out = self.experts[expert_idx][1](expert_out)
            output[token_indices] += expert_out * weights.unsqueeze(-1)
        if self.world_size > 1:
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
        output = output.view(seq_len, hidden_size)
        return x + output


class TransformerBlock(torch.nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int = 0, device: Optional[torch.device] = None):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config: ModelConfig, device: Optional[torch.device] = None):
        super().__init__()
        dtype = torch.float32
        self.embedding = torch.nn.Embedding(config.vocab_size, config.hidden_size, device=device, dtype=dtype)
        self.block = torch.nn.ModuleList([
            TransformerBlock(config, layer_idx, device) 
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
                    config.hidden_size, 
                    config.vocab_size, 
                    bias=False, 
                    device=device, 
                    dtype=dtype)
        self.config = config

    @staticmethod
    def from_checkpoint(path: str, device: Optional[torch.device] = None) -> "Transformer":
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        config = ModelConfig(**config)
        model = Transformer(config, device)
        model.load_state_dict(torch.load(os.path.join(path, "model.pt"), map_location=device))
        model.eval()
        return model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x   


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = Transformer.from_checkpoint("./", device=self.device)

    @torch.inference_mode()
    def generate(self, prompt_tokens: list[int], stop_tokens: list[int], temperature: float = 1.0, max_tokens: int = 0, return_logprobs: bool = False):
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]
            if temperature == 0.0:
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(predicted_token)
            num_generated_tokens += 1
            if return_logprobs:
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token
            if predicted_token in stop_tokens:
                break
                break