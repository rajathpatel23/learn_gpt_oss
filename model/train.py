from data_loader import train_loader, val_loader
from gpt_oss import Transformer, ModelConfig
from trainer import trainer
import wandb
import torch
from dataclasses import asdict

device = (
    torch.device("cuda:0") if torch.cuda.is_available() else
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    torch.device("cpu")
)
print(f"Using device: {device}")

# Tiny config for local quick runs
tiny_cfg = ModelConfig(
    num_hidden_layers=2,
    num_experts=2,
    experts_per_token=1,
    hidden_size=192,
    intermediate_size=384,
    head_dim=12,
    num_attention_heads=4,
    num_key_value_heads=2,
    sliding_window=64,
    initial_context_length=256,
    rope_theta=150000.0,
    rope_scaling_factor=8.0,
    rope_ntk_alpha=1.0,
    rope_ntk_beta=16.0,
    use_attention_sink=False,
)
model = Transformer(config=tiny_cfg, device=device)
wandb.init(project="gptoss-learning-v1", name="gptoss-learning-v1", group="gptoss-learning-v1", config={
    "learning_rate": 1e-4,
    "max_iters": 10,
    "warmup_steps": 50,
    "min_lr": 1e-5,
    "eval_iters": 2,
    "eval_freq": 2,
    "device": str(device),
    "model_type": "gpt-oss",
    "model_config": asdict(model.config)
})

train_losses, val_losses, tokens_seen = trainer(model, train_loader, val_loader, device)
print(f"Train Losses: {train_losses}")
print(f"Val Losses: {val_losses}")
print(f"Tokens Seen: {tokens_seen}")