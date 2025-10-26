import torch
from inference import generate_text
import time, os, gc
import wandb
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def calcc(input_batch, target_batch, model, device):
    total_loss = 0
    for i in range(len(input_batch)):
        inp = input_batch[i].to(device, non_blocking=True)
        logits = model(inp)
        del inp
        tgt = target_batch[i].to(device, non_blocking=True)
        loss = torch.nn.functional.cross_entropy(logits, tgt)
        total_loss += loss
        del tgt, logits, loss
    clear_gpu_memory()
    return total_loss / len(input_batch)

def calc_loss_batch(input_batch, target_batch, model, device):
    loss = calcc(input_batch, target_batch, model, device)
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if len(data_loader) == 0:
        return float("nan")
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    batches_seen = 0
    for input_batch, target_batch in data_loader:
        loss = calc_loss_batch(input_batch, target_batch, model, device)
        total_loss += float(loss.item())
        batches_seen += 1
        if batches_seen >= num_batches:
            break
    return total_loss / max(1, batches_seen)


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def train_model(model, train_loader, val_loader, optimizer,scheduler, device, num_epochs,
                       eval_freq, eval_iter, start_context):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    best_val_loss = float("inf")
    last_val_logged = None
    for epoch in range(num_epochs):
        model.train()
        artifact = wandb.Artifact("gptoss-model", type="model")
        for input_batch,target_batch in tqdm(train_loader):
            loss= calc_loss_batch(input_batch,target_batch,model,device)
            if not torch.isfinite(loss):
                logger.warning("Non-finite loss encountered; skipping step")
                optimizer.zero_grad(set_to_none=True)
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            wandb.log({"train/step_loss": loss.item(), 
                       "tokens_seen": tokens_seen, 
                       "epoch": epoch,
                       "lr": optimizer.param_groups[0]["lr"]
                       }, step=global_step)

            logger.info(f"Epoch {epoch+1}, Step {global_step+1}, Loss {loss.item()}")

            # Evaluate every eval_freq steps
            if (global_step + 1) % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                logger.info(f"[Eval] Epoch {epoch+1}, Step {global_step+1}, Train Loss {train_loss}, Val Loss {val_loss}")
                wandb.log({"train/loss": float(train_loss),
                               "val/loss": float(val_loss)}, step=global_step)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), "saved_models/gptoss_best.pt")
                    artifact = wandb.Artifact("gptoss-model", type="model")
                    artifact.add_file("saved_models/gptoss_best.pt")
                    wandb.log_artifact(artifact)    
                    logger.info(f"Saved best model with val_loss {val_loss}")
                last_val_logged = val_loss
        
        # (Optional) ensure at least one eval per epoch if none happened in-loop
        if last_val_logged is None:
            train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            track_tokens_seen.append(tokens_seen)
            logger.info(f"[Eval-Epoch-End] Epoch {epoch+1}, Step {global_step+1}, Train Loss {train_loss}, Val Loss {val_loss}")
            wandb.log({"train/loss": float(train_loss),
                           "val/loss": float(val_loss)}, step=global_step)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), "saved_models/gptoss_best.pt")
                artifact = wandb.Artifact("gptoss-model", type="model")
                artifact.add_file("saved_models/gptoss_best.pt")
                wandb.log_artifact(artifact)    
                logger.info(f"Saved best model with val_loss {val_loss}")
            last_val_logged = val_loss

        # check if the model directory exists
        if not os.path.exists("saved_models"):
            os.makedirs("saved_models")
        torch.save(model.state_dict(), "saved_models/gptoss.pt")
        artifact = wandb.Artifact("gptoss-model", type="model")
        artifact.add_file("saved_models/gptoss.pt")
        wandb.log_artifact(artifact, aliases=["latest"])
        if last_val_logged is not None:
            logger.info(f"Saved model with val_loss {last_val_logged}")
        else:
            logger.info("Saved model")
        txt=generate_text(model,start_context)
        logger.info(f"Generated text: {txt}")
        wandb.log({"generated_text": wandb.Html(txt)}, step=global_step)
        clear_gpu_memory()
    return train_losses, val_losses, track_tokens_seen

def trainer(model, train_loader, val_loader, device):
    learning_rate = 5e-5
    max_iters = 10
    warmup_steps = 50
    min_lr = 1e-5
    eval_iters = 2
    eval_freq = 2
    num_epochs = max_iters    
    torch.manual_seed(123)
    print(f"Using {device}")
    if os.path.exists('saved_models/gptoss.pt'):
        model.load_state_dict(torch.load('saved_models/gptoss.pt'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_steps))
    scheduler_decay = CosineAnnealingLR(optimizer, T_max=max(1, max_iters - warmup_steps), eta_min=min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[warmup_steps])
    num_epochs = max_iters
    start_time = time.time()
    train_losses, val_losses, tokens_seen = train_model(
        model, train_loader, val_loader, optimizer,scheduler, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iters,
        start_context="a fast driver named Tim went for",
    )
    torch.save(model.state_dict(),"saved_models/gotoss.pt")

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    wandb.log({"training_time_min": execution_time_minutes})
    wandb.finish()
    print(f"Training completed in {execution_time_minutes:.2f} minutes.") 
    return train_losses, val_losses, tokens_seen