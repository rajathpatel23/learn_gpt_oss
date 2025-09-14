from typing import List, Optional
import torch
from .cache_policies import evict_middle_with_sinks

class StreamingGenerator:
    """
    Minimal streaming generator that applies the attention-sinks cache policy.

    Positional encoding note:
      - We keep global positions (do not reset after eviction), avoiding RoPE reindexing.
      - Works out-of-the-box with ALiBi and absolute pos emb; fine for RoPE if you don't reset.
    """
    def __init__(
        self,
        model_name: str,
        sink_size: int = 6,
        window_size: int = 768,
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype or torch.float16, trust_remote_code=trust_remote_code
        )
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        self.model.to(self.device)

        self.sink_size = int(sink_size)
        self.window_size = int(window_size)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> str:
        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inp["input_ids"]

        outputs = self.model(input_ids=input_ids, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            if temperature > 0:
                probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cumulative = torch.cumsum(sorted_probs, dim=-1)
                    cutoff = (cumulative > top_p).float().argmax(dim=-1)
                    k = cutoff.item() + 1
                    top_idx = sorted_idx[:, :k]
                    top_probs = sorted_probs[:, :k]
                    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
                    next_id = top_idx.gather(1, torch.multinomial(top_probs, num_samples=1))
                else:
                    next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None and next_id.item() == eos_token_id:
                break
            generated_ids.append(next_id.item())

            outputs = self.model(input_ids=next_id, use_cache=True, past_key_values=past)
            logits = outputs.logits[:, -1, :]
            past, _ = evict_middle_with_sinks(past, self.sink_size, self.window_size)

        out_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=self.device)], dim=1)
        return self.tokenizer.decode(out_ids[0], skip_special_tokens=True)