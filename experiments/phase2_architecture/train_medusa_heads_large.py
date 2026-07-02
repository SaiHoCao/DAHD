#!/usr/bin/env python3
"""Train Medusa Heads with LARGE dataset and more epochs.

Key differences from original train_medusa_heads.py:
- Loads from data/training_large/ (500K+ samples)
- Trains for 15 epochs (was 3)
- Better LR schedule with cosine annealing
- Differential LR: ResBlock 1e-3, lm_head 1e-5
- Tracks per-head accuracy per epoch
- Saves best checkpoint based on avg(head_1..4) accuracy

Usage:
    CUDA_VISIBLE_DEVICES=6 python experiments/phase2_architecture/train_medusa_heads_large.py
"""

import json
import os
import sys
import time
import math
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MedusaTrainingConfig:
    # Model
    hidden_size: int = 4096
    vocab_size: int = 151936
    num_heads: int = 5        # Medusa heads (predict t+1 .. t+5)
    num_layers: int = 1       # ResBlocks per head
    
    # Paths
    target_model_path: str = "/mnt/nas1/hf/Qwen3-8B/"
    train_data_dir: str = str(PROJECT_ROOT / "data/training_large/train")
    val_data_dir: str = str(PROJECT_ROOT / "data/training_large/val")
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints/medusa_large")
    results_dir: str = str(PROJECT_ROOT / "results/phase4_medusa_large")
    
    # Training
    batch_size: int = 64       # Larger batch since data is bigger
    gradient_accumulation_steps: int = 2
    num_epochs: int = 15       # Much more epochs
    lr_resblock: float = 1e-3
    lr_lm_head: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Misc
    log_interval: int = 100
    num_workers: int = 4
    seed: int = 42
    
    # Resume from checkpoint
    resume_from: str = ""  # Set to checkpoint path to resume


# ============================================================================
# Medusa Model Architecture (same as original)
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with SiLU activation."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class MedusaModel(nn.Module):
    """Standard Medusa: K independent heads, each predicts one future position."""
    
    def __init__(self, hidden_size: int = 4096, vocab_size: int = 151936,
                 num_heads: int = 5, num_layers: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        # Per-head residual blocks
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = nn.Sequential(*[ResBlock(hidden_size) for _ in range(num_layers)])
            self.heads.append(layers)
        
        # Shared lm_head (from Qwen3-8B)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> list:
        """
        Args:
            hidden_states: [B, hidden_size] - last layer hidden state
        Returns:
            list of [B, vocab_size] logits, one per head
        """
        all_logits = []
        for head in self.heads:
            h = head(hidden_states)
            logits = self.lm_head(h)
            all_logits.append(logits)
        return all_logits
    
    def get_param_groups(self, lr_resblock: float, lr_lm_head: float,
                         weight_decay: float) -> list:
        """Get parameter groups with different learning rates."""
        resblock_params = []
        lm_head_params = []
        
        for head in self.heads:
            resblock_params.extend(head.parameters())
        lm_head_params.extend(self.lm_head.parameters())
        
        return [
            {"params": resblock_params, "lr": lr_resblock, "weight_decay": weight_decay},
            {"params": lm_head_params, "lr": lr_lm_head, "weight_decay": 0.0},
        ]


# ============================================================================
# Dataset
# ============================================================================

class MedusaDataset(Dataset):
    """Dataset for Medusa training: last-layer hidden states + next K tokens."""
    
    def __init__(self, data_dir: str, num_heads: int = 5):
        self.num_heads = num_heads
        self.chunks = []
        self.cumulative_sizes = []
        
        data_path = Path(data_dir)
        chunk_files = sorted(data_path.glob("chunk_*.pt"))
        
        total = 0
        for f in tqdm(chunk_files, desc=f"Loading {data_dir}"):
            chunk = torch.load(f, map_location='cpu', weights_only=False)
            n = chunk['hidden_concat'].shape[0]
            self.chunks.append(chunk)
            total += n
            self.cumulative_sizes.append(total)
        
        self.total_size = total
        print(f"  Loaded {len(chunk_files)} chunks, {total:,} samples from {data_dir}")
    
    def __len__(self) -> int:
        return self.total_size
    
    def _find_chunk(self, idx: int):
        """Binary search for the correct chunk."""
        lo, hi = 0, len(self.cumulative_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cumulative_sizes[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        chunk_idx = lo
        if chunk_idx > 0:
            local_idx = idx - self.cumulative_sizes[chunk_idx - 1]
        else:
            local_idx = idx
        return chunk_idx, local_idx
    
    def __getitem__(self, idx: int):
        chunk_idx, local_idx = self._find_chunk(idx)
        chunk = self.chunks[chunk_idx]
        
        # Extract last-layer hidden state (last 4096 dims of 12288)
        hidden = chunk['hidden_concat'][local_idx, 8192:].float()  # [4096]
        
        # next_tokens: [6] → take first num_heads (5)
        targets = chunk['next_tokens'][local_idx, :self.num_heads].long()  # [5]
        
        return hidden, targets


# ============================================================================
# Training Logic
# ============================================================================

def compute_loss(logits_list: list, targets: torch.Tensor) -> tuple:
    """Compute average CE loss across all heads."""
    losses = []
    for i, logits in enumerate(logits_list):
        loss = F.cross_entropy(logits, targets[:, i])
        losses.append(loss)
    
    total_loss = sum(losses) / len(losses)
    return total_loss, [l.item() for l in losses]


def compute_accuracy(logits_list: list, targets: torch.Tensor) -> list:
    """Compute top-1 accuracy for each head."""
    accs = []
    for i, logits in enumerate(logits_list):
        preds = logits.argmax(dim=-1)
        acc = (preds == targets[:, i]).float().mean().item()
        accs.append(acc)
    return accs


def train_epoch(model, dataloader, optimizer, scheduler, scaler, config, epoch, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    head_accs = [0.0] * config.num_heads
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}", leave=True)
    for step, (hidden, targets) in enumerate(pbar):
        hidden = hidden.to(device)
        targets = targets.to(device)
        
        with autocast(device_type='cuda', enabled=config.use_amp):
            logits_list = model(hidden)
            loss, per_head_loss = compute_loss(logits_list, targets)
            loss = loss / config.gradient_accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
        
        # Metrics
        with torch.no_grad():
            accs = compute_accuracy(logits_list, targets)
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        for i in range(config.num_heads):
            head_accs[i] += accs[i]
        num_batches += 1
        
        if (step + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_accs = [a / num_batches for a in head_accs]
            lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'h0': f'{avg_accs[0]:.3f}',
                'h1': f'{avg_accs[1]:.3f}',
                'h2': f'{avg_accs[2]:.3f}',
                'lr': f'{lr:.2e}',
            })
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_head_accs = [a / max(num_batches, 1) for a in head_accs]
    return avg_loss, avg_head_accs


@torch.no_grad()
def evaluate(model, dataloader, config, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0.0
    head_accs = [0.0] * config.num_heads
    num_batches = 0
    
    for hidden, targets in tqdm(dataloader, desc="Evaluating", leave=False):
        hidden = hidden.to(device)
        targets = targets.to(device)
        
        with autocast(device_type='cuda', enabled=config.use_amp):
            logits_list = model(hidden)
            loss, _ = compute_loss(logits_list, targets)
        
        accs = compute_accuracy(logits_list, targets)
        
        total_loss += loss.item()
        for i in range(config.num_heads):
            head_accs[i] += accs[i]
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_head_accs = [a / max(num_batches, 1) for a in head_accs]
    return avg_loss, avg_head_accs


# ============================================================================
# Main
# ============================================================================

def main():
    config = MedusaTrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Set seed
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Create directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Medusa Heads LARGE-SCALE Training for Qwen3-8B")
    print("=" * 70)
    print(f"  Hidden size:   {config.hidden_size}")
    print(f"  Vocab size:    {config.vocab_size}")
    print(f"  Num heads:     {config.num_heads}")
    print(f"  Num layers:    {config.num_layers}")
    print(f"  Batch size:    {config.batch_size} × {config.gradient_accumulation_steps} (accum)")
    print(f"  LR (ResBlock): {config.lr_resblock}")
    print(f"  LR (lm_head):  {config.lr_lm_head}")
    print(f"  Epochs:        {config.num_epochs}")
    print(f"  Device:        {device}")
    print(f"  AMP:           {config.use_amp}")
    print()
    
    # ---- Step 1: Initialize model ----
    print("[1/4] Initializing Medusa model with Qwen3-8B lm_head...")
    
    import gc
    from safetensors import safe_open
    
    lm_head_shard = os.path.join(config.target_model_path, "model-00005-of-00005.safetensors")
    with safe_open(lm_head_shard, framework="pt", device="cpu") as f:
        lm_head_weight = f.get_tensor("lm_head.weight").float()
    print(f"  lm_head weight shape: {lm_head_weight.shape}")
    
    medusa_model = MedusaModel(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    
    # Initialize lm_head from Qwen3-8B
    medusa_model.lm_head.weight.data.copy_(lm_head_weight)
    del lm_head_weight
    gc.collect()
    
    medusa_model = medusa_model.to(device)
    
    total_params = sum(p.numel() for p in medusa_model.parameters())
    trainable_params = sum(p.numel() for p in medusa_model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params / 1e6:.1f}M")
    print(f"  Trainable params: {trainable_params / 1e6:.1f}M")
    print(f"  GPU memory after model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # ---- Step 2: Load data ----
    print("\n[2/4] Loading training data...")
    train_dataset = MedusaDataset(config.train_data_dir, num_heads=config.num_heads)
    val_dataset = MedusaDataset(config.val_data_dir, num_heads=config.num_heads)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader):,} batches/epoch")
    print(f"  Val:   {len(val_dataset):,} samples, {len(val_loader):,} batches")
    
    # ---- Step 3: Setup optimizer + scheduler ----
    print("\n[3/4] Setting up optimizer...")
    
    param_groups = medusa_model.get_param_groups(
        config.lr_resblock, config.lr_lm_head, config.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=config.use_amp)
    
    # Cosine schedule with warmup (per-step)
    steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
    total_steps = steps_per_epoch * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    print(f"  Total steps: {total_steps:,}, Warmup: {warmup_steps:,}")
    print(f"  Steps/epoch: {steps_per_epoch:,}")
    
    # ---- Step 4: Training loop ----
    print("\n[4/4] Training...")
    
    training_log = {
        "config": {
            "hidden_size": config.hidden_size,
            "vocab_size": config.vocab_size,
            "num_heads": config.num_heads,
            "num_layers": config.num_layers,
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "lr_resblock": config.lr_resblock,
            "lr_lm_head": config.lr_lm_head,
            "num_epochs": config.num_epochs,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        "epochs": [],
    }
    
    best_avg_acc = 0.0  # Track avg of head_1..4 accuracy (since head_0 is easy)
    train_start = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_accs = train_epoch(
            medusa_model, train_loader, optimizer, scheduler, scaler,
            config, epoch, device
        )
        
        # Evaluate
        val_loss, val_accs = evaluate(medusa_model, val_loader, config, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "epoch_time_min": epoch_time / 60,
        }
        training_log["epochs"].append(epoch_log)
        
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Train Acc:  " + " | ".join(f"h{i}={a*100:.1f}%" for i, a in enumerate(train_accs)))
        print(f"  Val Acc:    " + " | ".join(f"h{i}={a*100:.1f}%" for i, a in enumerate(val_accs)))
        print(f"  Epoch time: {epoch_time/60:.1f} min")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            ckpt_path = Path(config.checkpoint_dir) / f"medusa_epoch{epoch+1}.pt"
            torch.save({
                "model_state_dict": medusa_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accs": val_accs,
                "config": config.__dict__,
            }, ckpt_path)
            print(f"  Saved: {ckpt_path}")
        
        # Save best model (based on avg of head_1..4 accuracy)
        avg_acc_1_4 = sum(val_accs[1:]) / (config.num_heads - 1)
        if avg_acc_1_4 > best_avg_acc:
            best_avg_acc = avg_acc_1_4
            best_path = Path(config.checkpoint_dir) / "medusa_best.pt"
            torch.save({
                "model_state_dict": medusa_model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accs": val_accs,
                "config": config.__dict__,
            }, best_path)
            print(f"  ★ New best model (avg h1-h4 acc={avg_acc_1_4*100:.2f}%)")
        
        # Also save to the standard path for e2e evaluation
        standard_best = PROJECT_ROOT / "checkpoints/medusa/medusa_best.pt"
        if avg_acc_1_4 > best_avg_acc - 0.001:  # Current is best or very close
            standard_best.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state_dict": medusa_model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accs": val_accs,
                "config": config.__dict__,
            }, standard_best)
        
        # Save training log after each epoch
        log_path = Path(config.results_dir) / "medusa_large_training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
    
    total_time = time.time() - train_start
    
    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total training time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"  Best avg(h1-h4) val accuracy: {best_avg_acc*100:.2f}%")
    print(f"  Best checkpoint: {Path(config.checkpoint_dir) / 'medusa_best.pt'}")
    print(f"  Training log: {log_path}")
    
    # Print final per-head accuracies
    final_accs = training_log["epochs"][-1]["val_accs"]
    print(f"\n  Final Val Accuracies:")
    for i, acc in enumerate(final_accs):
        print(f"    head_{i}: {acc*100:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
