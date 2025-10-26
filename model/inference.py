import torch
from torch.nn import functional as F

from data_loader import get_tokenizer
import logging
logger = logging.getLogger(__name__)




context_len=8192
tokenizer= get_tokenizer()

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    return tokenizer.decode(token_ids.tolist())



def generate_text(model, prompt, max_tokens=100, temperature=0.8, top_k=50):
    """Generate text from a prompt using trained model."""
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else
        torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
        torch.device("cpu")
    )
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    
    idx = text_to_token_ids(prompt,tokenizer).to(device)
    # Generate
    for _ in range(max_tokens):
        idx_cond = idx[-context_len:]
        with torch.inference_mode():
            logits= model(idx_cond)
        logits = logits[-1, :]
        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if temperature <= 0:
            temperature = 1.0
        logits = logits / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[[-1]]] = -float('inf')
        # numerical stability
        logits = torch.clamp(logits - logits.max(), min=-20, max=20)
        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp_min(1e-8)
        probs = probs / probs.sum()
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            logger.warning("NaN/Inf in probs; using argmax fallback")
            idx_next = torch.argmax(probs).unsqueeze(0)
        else:
            idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=0)

    
    
       
    
    # Decode and return
    result = token_ids_to_text(idx,tokenizer)
    return result