#!/usr/bin/env python3
"""Train Gumiho-style parallel draft heads for DAHD speculative decoding.

Architecture (Gumiho-inspired, adapted for Qwen3-8B):
  - Input: fc(cat(embed[t+1], hidden[t]))
      where hidden[t]  = target model's last-layer hidden at position t
            embed[t+1] = embedding of target model's greedy next-token prediction
      At inference, embed[t+1] = embed(target_next) is FREE (already computed).
  - K=4 independent heads, each with 3 ResBlocks + shared lm_head
  - Head k predicts token at position t+k+2  (i.e. t+2, t+3, t+4, t+5)
    (t+1 is already known = target_next, so we skip it and predict from t+2)

Loss:
  Weighted cross-entropy with exponential decay: weight_k = λ^k, λ=0.8
  total_loss = Σ_k λ^k · CE(logits_k, labels_k)

Training data format (from generate_training_data.py v2):
  chunk_*.pt  →  {
      'hidden_last': [N, 4096]   bf16
      'token_ids':   [N]
      'next_tokens': [N, 6]      long  (tokens at t+1..t+6)
  }

  We use:
    embed input  = next_tokens[:, 0]   (= token at t+1)
    hidden input = hidden_last         (= hidden at t)
    labels       = next_tokens[:, 1:5] (= tokens at t+2..t+5)

Key fixes vs v1:
  1. Scheduler.step() called per optimizer step (not per epoch).
  2. Exponential loss weighting [1.0, 0.8, 0.64, 0.512] per head.
  3. embed_tokens loaded from Qwen3-8B, frozen.
  4. Deeper MLPs (3 ResBlocks per head instead of 1).
  5. Backward-compatible dataset: old chunks (12288-dim hidden_concat) still
     supported via auto-detection.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase2_architecture/train_medusa_heads.py
"""

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Model dims
    hidden_size:  int   = 4096
    vocab_size:   int   = 151936
    num_heads:    int   = 4       # predict t+2, t+3, t+4, t+5
    mlp_depth:    int   = 3       # ResBlocks per head (Gumiho uses ~5, 3 is lighter)
    loss_lambda:  float = 0.8     # exponential weight decay per head

    # Paths
    target_model_path: str = "/mnt/nas1/hf/Qwen3-8B/"
    train_data_dir:    str = str(PROJECT_ROOT / "data/training/train")
    val_data_dir:      str = str(PROJECT_ROOT / "data/training/val")
    checkpoint_dir:    str = str(PROJECT_ROOT / "checkpoints/medusa")
    results_dir:       str = str(PROJECT_ROOT / "results/phase4_medusa")

    # Training hyper-params
    batch_size:                  int   = 16
    gradient_accumulation_steps: int   = 4
    num_epochs:                  int   = 10
    lr_mlp:                      float = 1e-3
    lr_fc:                       float = 1e-3
    lr_lm_head:                  float = 1e-5
    weight_decay:                float = 0.01
    warmup_ratio:                float = 0.05
    max_grad_norm:               float = 1.0

    # Misc
    use_amp:       bool = True
    log_interval:  int  = 50
    num_workers:   int  = 4
    seed:          int  = 42


# ── Blocks ────────────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.linear = nn.Linear(size, size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


# ── Model ─────────────────────────────────────────────────────────────────────

class GumihoBranch(nn.Module):
    """Gumiho-style parallel draft heads.

    Input at inference:
      mlp_input = fc(cat(embed(target_next), hidden_t))  →  [B, hidden_size]

    Each head_k predicts the token at position t+k+2 (for k=0..K-1),
    i.e. two or more positions ahead of the current context.
    """

    def __init__(self, hidden_size: int, vocab_size: int,
                 num_heads: int = 4, mlp_depth: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size  = vocab_size
        self.num_heads   = num_heads

        # Frozen embedding table (loaded from target model)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Projection from [embed || hidden] → hidden
        # Zero-init like Gumiho's noResBlock: keeps early training stable
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.fc.weight)

        # K independent MLP heads
        self.heads = nn.ModuleList([
            nn.Sequential(*[ResBlock(hidden_size) for _ in range(mlp_depth)])
            for _ in range(num_heads)
        ])

        # Shared output projection (init from target lm_head)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, hidden: torch.Tensor, next_token_ids: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            hidden:         [B, hidden_size]  – last-layer hidden at position t
            next_token_ids: [B]               – token_{t+1} (already known)
        Returns:
            list of K tensors, each [B, vocab_size]
        """
        embed = self.embed_tokens(next_token_ids)               # [B, hidden_size]
        x = self.fc(torch.cat([embed, hidden], dim=-1))         # [B, hidden_size]
        return [self.lm_head(head(x)) for head in self.heads]  # K × [B, vocab_size]

    def get_param_groups(self, cfg: TrainingConfig) -> list[dict]:
        return [
            {"params": list(self.fc.parameters()),
             "lr": cfg.lr_fc, "weight_decay": cfg.weight_decay},
            {"params": [p for h in self.heads for p in h.parameters()],
             "lr": cfg.lr_mlp, "weight_decay": cfg.weight_decay},
            {"params": list(self.lm_head.parameters()),
             "lr": cfg.lr_lm_head, "weight_decay": 0.0},
        ]


# ── Dataset ───────────────────────────────────────────────────────────────────

class GumihoDataset(Dataset):
    """Loads chunk_*.pt files produced by generate_training_data.py (v2).

    Supports both the new format (hidden_last [N,4096]) and the legacy
    format (hidden_concat [N,12288]) for backward compatibility.
    """

    def __init__(self, data_dir: str, num_heads: int = 4):
        self.num_heads = num_heads
        self.chunks: list[dict] = []
        self.cumulative: list[int] = []

        total = 0
        for f in sorted(Path(data_dir).glob("chunk_*.pt")):
            chunk = torch.load(f, map_location="cpu", weights_only=False)
            n = chunk["next_tokens"].shape[0]
            self.chunks.append(chunk)
            total += n
            self.cumulative.append(total)

        self.total_size = total
        print(f"  Dataset: {len(self.chunks)} chunks, {total:,} samples from {data_dir}")

    def __len__(self) -> int:
        return self.total_size

    def _locate(self, idx: int):
        lo, hi = 0, len(self.cumulative) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cumulative[mid] <= idx:
                lo = mid + 1
            else:
                hi = mid
        local = idx - (self.cumulative[lo - 1] if lo > 0 else 0)
        return self.chunks[lo], local

    def __getitem__(self, idx: int):
        chunk, li = self._locate(idx)

        # Support both new (hidden_last) and legacy (hidden_concat) formats
        if "hidden_last" in chunk:
            hidden = chunk["hidden_last"][li].float()           # [4096]
        else:
            # Legacy: last 4096 of 12288-dim concat = last-layer hidden
            hidden = chunk["hidden_concat"][li, 8192:].float()  # [4096]

        # next_tokens[0] = token_{t+1} (embed input, known at inference)
        # next_tokens[1:num_heads+1] = t+2..t+num_heads+1  (labels)
        next_toks = chunk["next_tokens"][li]                    # [≥num_heads+1]
        embed_id  = next_toks[0].long()
        labels    = next_toks[1 : 1 + self.num_heads].long()   # [num_heads]

        return hidden, embed_id, labels


# ── Loss ──────────────────────────────────────────────────────────────────────

def compute_loss_and_acc(
    logits_list: list[torch.Tensor],
    labels: torch.Tensor,
    loss_lambda: float,
) -> tuple[torch.Tensor, list[float], list[float]]:
    """
    Args:
        logits_list: K tensors of [B, vocab]
        labels:      [B, K]
        loss_lambda: exponential weight decay per head
    Returns:
        total_loss, per_head_loss (floats), per_head_acc (floats)
    """
    num_heads  = len(logits_list)
    losses, accs = [], []

    for k, logits in enumerate(logits_list):
        ce   = F.cross_entropy(logits, labels[:, k])
        acc  = (logits.argmax(-1) == labels[:, k]).float().mean().item()
        losses.append((loss_lambda ** k) * ce)
        accs.append(acc)

    total = sum(losses) / sum(loss_lambda ** k for k in range(num_heads))
    return total, [l.item() for l in losses], accs


# ── Train epoch ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, scaler, cfg, epoch, device):
    model.train()
    sum_loss = 0.0
    sum_accs = [0.0] * cfg.num_heads
    n_batches = 0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.num_epochs}")
    for step, (hidden, embed_id, labels) in enumerate(pbar):
        hidden   = hidden.to(device)
        embed_id = embed_id.to(device)
        labels   = labels.to(device)

        with autocast(enabled=cfg.use_amp):
            logits_list = model(hidden, embed_id)
            loss, _, accs = compute_loss_and_acc(logits_list, labels, cfg.loss_lambda)
            loss = loss / cfg.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()   # ← per optimizer step, not per epoch

        sum_loss += loss.item() * cfg.gradient_accumulation_steps
        for k in range(cfg.num_heads):
            sum_accs[k] += accs[k]
        n_batches += 1

        if (step + 1) % cfg.log_interval == 0:
            avg_loss = sum_loss / n_batches
            avg_accs = [a / n_batches for a in sum_accs]
            pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "acc0": f"{avg_accs[0]:.3f}",
                "acc3": f"{avg_accs[-1]:.3f}",
                "lr":   f"{scheduler.get_last_lr()[0]:.2e}",
            })

    avg_loss = sum_loss / max(n_batches, 1)
    avg_accs = [a / max(n_batches, 1) for a in sum_accs]
    return avg_loss, avg_accs


@torch.no_grad()
def evaluate(model, loader, cfg, device):
    model.eval()
    sum_loss = 0.0
    sum_accs = [0.0] * cfg.num_heads
    n_batches = 0

    for hidden, embed_id, labels in tqdm(loader, desc="  Val", leave=False):
        hidden   = hidden.to(device)
        embed_id = embed_id.to(device)
        labels   = labels.to(device)

        with autocast(enabled=cfg.use_amp):
            logits_list = model(hidden, embed_id)
            loss, _, accs = compute_loss_and_acc(logits_list, labels, cfg.loss_lambda)

        sum_loss += loss.item()
        for k in range(cfg.num_heads):
            sum_accs[k] += accs[k]
        n_batches += 1

    avg_loss = sum_loss / max(n_batches, 1)
    avg_accs = [a / max(n_batches, 1) for a in sum_accs]
    return avg_loss, avg_accs


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cfg    = TrainingConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Gumiho-style Parallel Branch Training (DAHD)")
    print("=" * 70)
    print(f"  hidden_size : {cfg.hidden_size}")
    print(f"  num_heads   : {cfg.num_heads}  (predict t+2 .. t+{cfg.num_heads+1})")
    print(f"  mlp_depth   : {cfg.mlp_depth} ResBlocks/head")
    print(f"  loss_lambda : {cfg.loss_lambda}  → weights {[cfg.loss_lambda**k for k in range(cfg.num_heads)]}")
    print(f"  batch_size  : {cfg.batch_size} × {cfg.gradient_accumulation_steps} (eff {cfg.batch_size*cfg.gradient_accumulation_steps})")
    print(f"  epochs      : {cfg.num_epochs}")
    print()

    # ── 1. Build model ───────────────────────────────────────────────────────
    print("[1/4] Building model & loading frozen weights from Qwen3-8B …")
    from safetensors import safe_open

    model = GumihoBranch(
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        num_heads=cfg.num_heads,
        mlp_depth=cfg.mlp_depth,
    )

    # Load embed_tokens (frozen during training)
    emb_shard = os.path.join(cfg.target_model_path, "model-00001-of-00005.safetensors")
    with safe_open(emb_shard, framework="pt", device="cpu") as f:
        emb_weight = f.get_tensor("model.embed_tokens.weight").float()
    model.embed_tokens.weight.data.copy_(emb_weight)
    model.embed_tokens.weight.requires_grad_(False)
    del emb_weight

    # Load lm_head (fine-tuned with small lr)
    lm_shard = os.path.join(cfg.target_model_path, "model-00005-of-00005.safetensors")
    with safe_open(lm_shard, framework="pt", device="cpu") as f:
        lm_weight = f.get_tensor("lm_head.weight").float()
    model.lm_head.weight.data.copy_(lm_weight)
    del lm_weight

    model = model.to(device)

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total_p/1e6:.1f}M total, {trainable_p/1e6:.1f}M trainable")
    print(f"  GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ── 2. Data ──────────────────────────────────────────────────────────────
    print("\n[2/4] Loading datasets …")
    train_ds = GumihoDataset(cfg.train_data_dir, num_heads=cfg.num_heads)
    val_ds   = GumihoDataset(cfg.val_data_dir,   num_heads=cfg.num_heads)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    print(f"  Train: {len(train_ds):,} samples  |  Val: {len(val_ds):,} samples")

    # ── 3. Optimizer & scheduler ─────────────────────────────────────────────
    print("\n[3/4] Setting up optimiser & cosine-with-warmup scheduler …")
    param_groups = model.get_param_groups(cfg)
    optimizer    = torch.optim.AdamW(param_groups)
    scaler       = GradScaler(enabled=cfg.use_amp)

    total_steps  = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    warmup_steps = int(total_steps * cfg.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"  total_steps={total_steps}  warmup={warmup_steps}")

    # ── 4. Training loop ─────────────────────────────────────────────────────
    print("\n[4/4] Training …")
    training_log = {"config": cfg.__dict__, "epochs": []}
    best_val_acc0 = 0.0

    for epoch in range(cfg.num_epochs):
        print(f"\n{'─'*50}")
        print(f"Epoch {epoch+1}/{cfg.num_epochs}")

        train_loss, train_accs = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, cfg, epoch, device
        )
        val_loss, val_accs = evaluate(model, val_loader, cfg, device)

        head_acc_str = " | ".join(f"h{k}={a:.3f}" for k, a in enumerate(val_accs))
        print(f"  Train loss: {train_loss:.4f}   Val loss: {val_loss:.4f}")
        print(f"  Val accs:   {head_acc_str}")

        epoch_record = {
            "epoch": epoch + 1,
            "train_loss": train_loss, "val_loss": val_loss,
            "train_accs": train_accs, "val_accs": val_accs,
        }
        training_log["epochs"].append(epoch_record)

        # Save per-epoch checkpoint
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss, "val_accs": val_accs,
            "config": cfg.__dict__,
            "arch": "GumihoBranch",
        }
        torch.save(ckpt, Path(cfg.checkpoint_dir) / f"gumiho_epoch{epoch+1}.pt")

        if val_accs[0] > best_val_acc0:
            best_val_acc0 = val_accs[0]
            torch.save(ckpt, Path(cfg.checkpoint_dir) / "gumiho_best.pt")
            print(f"  ★ New best (head_0 acc={best_val_acc0:.4f})")

    # Save log
    log_path = Path(cfg.results_dir) / "gumiho_training_log.json"
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"  Best head-0 val acc: {best_val_acc0:.4f}")
    print(f"  Best ckpt: {cfg.checkpoint_dir}/gumiho_best.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()
