#!/usr/bin/env python3
"""Train Gumiho Parallel Heads with large-scale data (500K+ samples, 15-20 epochs).

Targets:
- head_1 accuracy > 40%
- head_2 accuracy > 20%

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase2_architecture/train_gumiho_large.py
"""

import json
import os
import sys
import time
import math
import gc
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class GumihoLargeConfig:
    # Model
    hidden_size: int = 4096
    vocab_size: int = 151936
    num_heads: int = 5
    num_layers: int = 1  # ResBlocks per head
    
    # Paths
    target_model_path: str = "/mnt/nas1/hf/Qwen3-8B/"
    train_data_dir: str = str(PROJECT_ROOT / "data/training_large/train")
    val_data_dir: str = str(PROJECT_ROOT / "data/training_large/val")
    checkpoint_dir: str = str(PROJECT_ROOT / "checkpoints/gumiho_large")
    results_dir: str = str(PROJECT_ROOT / "results/phase4_e2e_large")
    
    # Training
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    num_epochs: int = 15
    lr_resblock: float = 1e-3
    lr_lm_head: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_grad_norm: float = 1.0
    
    # Mixed precision
    use_amp: bool = True
    
    # Misc
    log_interval: int = 100
    eval_interval: int = 2000
    seed: int = 42
    num_workers: int = 4
    
    # Early stopping
    patience: int = 5  # epochs without improvement


# ============================================================================
# Model (same as original)
# ============================================================================

class ResBlock(nn.Module):
    """Residual block with SiLU activation."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


class GumihoModel(nn.Module):
    """Standard Gumiho: K independent heads with shared lm_head."""
    
    def __init__(self, hidden_size=4096, vocab_size=151936, num_heads=5, num_layers=1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        
        self.heads = nn.ModuleList()
        for _ in range(num_heads):
            layers = nn.Sequential(*[ResBlock(hidden_size) for _ in range(num_layers)])
            self.heads.append(layers)
        
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> list:
        all_logits = []
        for head in self.heads:
            h = head(hidden_states)
            logits = self.lm_head(h)
            all_logits.append(logits)
        return all_logits
    
    def get_param_groups(self, lr_resblock, lr_lm_head, weight_decay):
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
# Dataset (memory-mapped for large data)
# ============================================================================

class GumihoLargeDataset(Dataset):
    """Dataset for large-scale Gumiho training with lazy loading."""
    
    def __init__(self, data_dir: str, num_heads: int = 5):
        self.num_heads = num_heads
        self.data_dir = Path(data_dir)
        self.chunk_files = sorted(self.data_dir.glob("chunk_*.pt"))
        
        # Get sizes without loading full data
        self.chunk_sizes = []
        self.cumulative_sizes = []
        total = 0
        
        for f in self.chunk_files:
            # Load only metadata (shape)
            chunk = torch.load(f, map_location='cpu', weights_only=False)
            n = chunk['hidden_concat'].shape[0]
            self.chunk_sizes.append(n)
            total += n
            self.cumulative_sizes.append(total)
            del chunk
        
        self.total_size = total
        self._cache = {}
        self._cache_limit = 10  # Keep max 10 chunks in memory
        
        print(f"  Loaded index: {len(self.chunk_files)} chunks, {total} samples from {data_dir}")
    
    def __len__(self):
        return self.total_size
    
    def _find_chunk(self, idx: int):
        lo, hi = 0, len(self.cumulative_sizes) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cumulative_sizes[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        chunk_idx = lo
        local_idx = idx - (self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else 0)
        return chunk_idx, local_idx
    
    def _get_chunk(self, chunk_idx: int):
        if chunk_idx not in self._cache:
            if len(self._cache) >= self._cache_limit:
                # Evict oldest
                oldest = next(iter(self._cache))
                del self._cache[oldest]
            self._cache[chunk_idx] = torch.load(
                self.chunk_files[chunk_idx], map_location='cpu', weights_only=False
            )
        return self._cache[chunk_idx]
    
    def __getitem__(self, idx: int):
        chunk_idx, local_idx = self._find_chunk(idx)
        chunk = self._get_chunk(chunk_idx)
        
        # Extract last-layer hidden state (last 4096 of 12288)
        hidden = chunk['hidden_concat'][local_idx, 8192:].float()  # [4096]
        targets = chunk['next_tokens'][local_idx, :self.num_heads].long()  # [5]
        
        return hidden, targets


# ============================================================================
# Training Logic
# ============================================================================

def compute_loss(logits_list, targets):
    losses = []
    for i, logits in enumerate(logits_list):
        loss = F.cross_entropy(logits, targets[:, i])
        losses.append(loss)
    total_loss = sum(losses) / len(losses)
    return total_loss, [l.item() for l in losses]


def compute_accuracy(logits_list, targets):
    accs = []
    for i, logits in enumerate(logits_list):
        preds = logits.argmax(dim=-1)
        acc = (preds == targets[:, i]).float().mean().item()
        accs.append(acc)
    return accs


def train_epoch(model, dataloader, optimizer, scheduler, scaler, config, epoch, device):
    model.train()
    total_loss = 0.0
    head_accs = [0.0] * config.num_heads
    num_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    for step, (hidden, targets) in enumerate(pbar):
        hidden = hidden.to(device)
        targets = targets.to(device)
        
        with autocast(enabled=config.use_amp):
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
        
        with torch.no_grad():
            accs = compute_accuracy(logits_list, targets)
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        for i in range(config.num_heads):
            head_accs[i] += accs[i]
        num_batches += 1
        
        if (step + 1) % config.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_accs = [a / num_batches for a in head_accs]
            pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'h0': f'{avg_accs[0]:.3f}',
                'h1': f'{avg_accs[1]:.3f}',
                'h2': f'{avg_accs[2]:.3f}',
            })
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_accs = [a / max(num_batches, 1) for a in head_accs]
    return avg_loss, avg_accs


@torch.no_grad()
def evaluate(model, dataloader, config, device):
    model.eval()
    total_loss = 0.0
    head_accs = [0.0] * config.num_heads
    num_batches = 0
    
    for hidden, targets in tqdm(dataloader, desc="Eval", leave=False):
        hidden = hidden.to(device)
        targets = targets.to(device)
        
        with autocast(enabled=config.use_amp):
            logits_list = model(hidden)
            loss, _ = compute_loss(logits_list, targets)
        
        accs = compute_accuracy(logits_list, targets)
        total_loss += loss.item()
        for i in range(config.num_heads):
            head_accs[i] += accs[i]
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_accs = [a / max(num_batches, 1) for a in head_accs]
    return avg_loss, avg_accs


# ============================================================================
# Main
# ============================================================================

def main():
    config = GumihoLargeConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Gumiho Parallel Heads Large-Scale Training")
    print("=" * 70)
    print(f"  Hidden size:   {config.hidden_size}")
    print(f"  Vocab size:    {config.vocab_size}")
    print(f"  Num heads:     {config.num_heads}")
    print(f"  Batch size:    {config.batch_size} × {config.gradient_accumulation_steps} (accum)")
    print(f"  LR (ResBlock): {config.lr_resblock}")
    print(f"  LR (lm_head):  {config.lr_lm_head}")
    print(f"  Epochs:        {config.num_epochs}")
    print(f"  Device:        {device}")
    print()
    
    # ---- Initialize model ----
    print("[1/4] Initializing Gumiho model with Qwen3-8B lm_head...")
    from safetensors import safe_open
    
    lm_head_shard = os.path.join(config.target_model_path, "model-00005-of-00005.safetensors")
    with safe_open(lm_head_shard, framework="pt", device="cpu") as f:
        lm_head_weight = f.get_tensor("lm_head.weight").float()
    print(f"  lm_head shape: {lm_head_weight.shape}")
    
    gumiho_model = GumihoModel(
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
    )
    gumiho_model.lm_head.weight.data.copy_(lm_head_weight)
    del lm_head_weight
    gc.collect()
    
    gumiho_model = gumiho_model.to(device)
    total_params = sum(p.numel() for p in gumiho_model.parameters())
    trainable_params = sum(p.numel() for p in gumiho_model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params / 1e6:.1f}M, Trainable: {trainable_params / 1e6:.1f}M")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # ---- Load data ----
    print("\n[2/4] Loading training data...")
    
    # Check if large training data exists, otherwise fall back to original
    train_dir = config.train_data_dir
    val_dir = config.val_data_dir
    if not Path(train_dir).exists() or len(list(Path(train_dir).glob("chunk_*.pt"))) == 0:
        print(f"  WARNING: {train_dir} not found, falling back to original data")
        train_dir = str(PROJECT_ROOT / "data/training/train")
        val_dir = str(PROJECT_ROOT / "data/training/val")
    
    train_dataset = GumihoLargeDataset(train_dir, num_heads=config.num_heads)
    val_dataset = GumihoLargeDataset(val_dir, num_heads=config.num_heads)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=config.num_workers, pin_memory=True
    )
    
    print(f"  Train: {len(train_dataset)} samples, {len(train_loader)} batches/epoch")
    print(f"  Val:   {len(val_dataset)} samples")
    
    # ---- Setup optimizer ----
    print("\n[3/4] Setting up optimizer...")
    param_groups = gumiho_model.get_param_groups(
        config.lr_resblock, config.lr_lm_head, config.weight_decay
    )
    optimizer = torch.optim.AdamW(param_groups)
    scaler = GradScaler(enabled=config.use_amp)
    
    # Cosine scheduler with warmup
    total_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"  Total steps: {total_steps}, Warmup: {warmup_steps}")
    
    # ---- Training loop ----
    print("\n[4/4] Training...")
    
    training_log = {
        "config": config.__dict__.copy(),
        "epochs": [],
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
    }
    # Remove non-serializable
    training_log["config"].pop("dtype", None)
    
    best_val_loss = float('inf')
    best_epoch = 0
    no_improve_count = 0
    start_time = time.time()
    
    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.num_epochs} (elapsed: {(time.time()-start_time)/60:.1f} min)")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_accs = train_epoch(
            gumiho_model, train_loader, optimizer, scheduler, scaler, config, epoch, device
        )
        
        # Evaluate
        val_loss, val_accs = evaluate(gumiho_model, val_loader, config, device)
        
        epoch_time = time.time() - epoch_start
        
        # Log
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "epoch_time_min": epoch_time / 60,
            "lr": scheduler.get_last_lr()[0],
        }
        training_log["epochs"].append(epoch_log)
        
        print(f"\n  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"  Train Acc:  " + " | ".join(f"h{i}={a*100:.1f}%" for i, a in enumerate(train_accs)))
        print(f"  Val Acc:    " + " | ".join(f"h{i}={a*100:.1f}%" for i, a in enumerate(val_accs)))
        print(f"  Epoch time: {epoch_time/60:.1f} min")
        
        # Check targets
        if val_accs[1] > 0.40:
            print(f"  ★ TARGET MET: head_1 accuracy {val_accs[1]*100:.1f}% > 40%")
        if val_accs[2] > 0.20:
            print(f"  ★ TARGET MET: head_2 accuracy {val_accs[2]*100:.1f}% > 20%")
        
        # Save checkpoint every epoch
        ckpt_path = Path(config.checkpoint_dir) / f"gumiho_epoch{epoch+1}.pt"
        torch.save({
            "model_state_dict": gumiho_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accs": val_accs,
            "train_accs": train_accs,
        }, ckpt_path)
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            no_improve_count = 0
            best_path = Path(config.checkpoint_dir) / "gumiho_best.pt"
            torch.save({
                "model_state_dict": gumiho_model.state_dict(),
                "epoch": epoch + 1,
                "val_loss": val_loss,
                "val_accs": val_accs,
                "train_accs": train_accs,
            }, best_path)
            print(f"  ★ New best (val_loss={val_loss:.4f}, epoch {epoch+1})")
        else:
            no_improve_count += 1
            if no_improve_count >= config.patience:
                print(f"\n  Early stopping: no improvement for {config.patience} epochs")
                break
        
        # Save training log incrementally
        log_path = Path(config.results_dir) / "gumiho_large_training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        
        # Time estimate
        elapsed = time.time() - start_time
        remaining_epochs = config.num_epochs - epoch - 1
        eta = elapsed / (epoch + 1) * remaining_epochs
        print(f"  ETA: {eta/60:.0f} min remaining")
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print(f"  Best epoch: {best_epoch}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    
    # Load best and report
    best_ckpt = torch.load(Path(config.checkpoint_dir) / "gumiho_best.pt", map_location='cpu', weights_only=False)
    print(f"  Best val accs: " + " | ".join(f"h{i}={a*100:.1f}%" for i, a in enumerate(best_ckpt['val_accs'])))
    print(f"  Checkpoint: {config.checkpoint_dir}/gumiho_best.pt")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
