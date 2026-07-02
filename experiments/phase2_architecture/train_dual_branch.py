#!/usr/bin/env python3
"""Phase 2: Dual-branch training script for DAHD speculative decoding.

Implements the two-phase training procedure:
  Phase 1 - Separate pre-training: Train AR and Parallel branches independently.
  Phase 2 - Joint fine-tuning: Unfreeze shared bottom and train end-to-end.

Usage:
    # Phase 1: Separate pre-training
    python experiments/phase2_architecture/train_dual_branch.py \
        --target_model Qwen/Qwen2.5-7B-Instruct \
        --phase 1 --epochs 3 --lr 1e-4 \
        --output_dir checkpoints

    # Phase 2: Joint fine-tuning (load phase1 checkpoint)
    python experiments/phase2_architecture/train_dual_branch.py \
        --target_model Qwen/Qwen2.5-7B-Instruct \
        --phase 2 --epochs 2 --lr 5e-5 \
        --data_path data/train.jsonl \
        --output_dir checkpoints
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import ExperimentConfig
from src.drafters.dahd_draft_module import DAHDDraftModule
from src.utils.device_utils import setup_deterministic
from src.utils.logging_utils import ExperimentLogger


class TokenDataset(Dataset):
    """Dataset that provides sequences of hidden states and target token IDs.

    For training the draft module, we need the target model's hidden states
    as input, and the actual next tokens as supervision labels.
    """

    def __init__(
        self,
        data_path: str | Path | None,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        num_synthetic_samples: int = 1000,
    ):
        """Initialize the dataset.

        Args:
            data_path: Path to training data JSONL (each line has "text" field).
                If None, generates synthetic prompts from a fixed set.
            tokenizer: Tokenizer for encoding text.
            max_length: Maximum sequence length.
            num_synthetic_samples: Number of synthetic samples if data_path is None.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: list[dict] = []

        if data_path and Path(data_path).exists():
            with open(data_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))
        else:
            # Generate synthetic training data from template prompts
            templates = [
                "Solve the following math problem step by step: What is {}?",
                "Write a Python function that {}.",
                "Explain the concept of {} in simple terms.",
                "Summarize the following: {}",
            ]
            for i in range(num_synthetic_samples):
                template = templates[i % len(templates)]
                text = template.format(f"example_{i}")
                self.samples.append({"text": text})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.samples[idx].get("text", "")
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }


def compute_ar_loss(
    draft_module: DAHDDraftModule,
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Compute autoregressive branch training loss.

    Args:
        draft_module: The DAHD draft module.
        hidden_states: Target model hidden states, shape (batch, seq, hidden_dim).
        target_ids: Ground truth next token IDs, shape (batch, seq).
        k: Number of draft positions to predict.

    Returns:
        Cross-entropy loss for the AR branch predictions.
    """
    shared_repr = draft_module.shared_bottom(hidden_states)
    draft_tokens, draft_logits = draft_module.ar_branch(shared_repr, k)
    # draft_logits: (batch, k, vocab_size)

    # Target tokens for positions 1..k (next tokens after current position)
    # We use the last `k` tokens from target_ids as labels
    batch_size, seq_len = target_ids.shape
    if seq_len < k:
        k = seq_len

    # Labels are the next k tokens starting from position seq_len - k
    labels = target_ids[:, -k:]  # (batch, k)

    # Reshape for cross-entropy
    logits_flat = draft_logits[:, :k].reshape(-1, draft_logits.size(-1))
    labels_flat = labels.reshape(-1)

    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
    return loss


def compute_parallel_loss(
    draft_module: DAHDDraftModule,
    hidden_states: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute parallel branch training loss.

    Each head independently predicts its corresponding future token.

    Args:
        draft_module: The DAHD draft module.
        hidden_states: Target model hidden states, shape (batch, seq, hidden_dim).
        target_ids: Ground truth next token IDs, shape (batch, seq).

    Returns:
        Cross-entropy loss for the parallel branch predictions.
    """
    shared_repr = draft_module.shared_bottom(hidden_states)
    draft_tokens, draft_logits = draft_module.parallel_branch(shared_repr)
    # draft_logits: (batch, max_k, vocab_size)

    max_k = draft_logits.size(1)
    batch_size, seq_len = target_ids.shape
    k = min(max_k, seq_len)

    # Labels are the next k tokens
    labels = target_ids[:, -k:]  # (batch, k)

    logits_flat = draft_logits[:, :k].reshape(-1, draft_logits.size(-1))
    labels_flat = labels.reshape(-1)

    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100)
    return loss


def train_phase1(
    draft_module: DAHDDraftModule,
    target_model: nn.Module,
    dataloader: DataLoader,
    optimizer_ar: torch.optim.Optimizer,
    optimizer_par: torch.optim.Optimizer,
    device: str,
    epochs: int,
    k_easy: int,
    k_hard: int,
    exp_logger: ExperimentLogger,
    max_grad_norm: float = 1.0,
) -> None:
    """Phase 1: Separately pre-train AR and Parallel branches.

    Args:
        draft_module: The DAHD draft module to train.
        target_model: Frozen target model for hidden state extraction.
        dataloader: Training data loader.
        optimizer_ar: Optimizer for AR branch parameters.
        optimizer_par: Optimizer for Parallel branch parameters.
        device: Device string.
        epochs: Number of training epochs.
        k_easy: Draft length for parallel branch.
        k_hard: Draft length for AR branch.
        exp_logger: Logger instance.
        max_grad_norm: Maximum gradient norm for clipping.
    """
    print("\n[Phase 1 Training] Separately pre-training AR and Parallel branches...")

    for epoch in range(epochs):
        epoch_ar_loss = 0.0
        epoch_par_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Extract hidden states from frozen target model
            with torch.no_grad():
                outputs = target_model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]  # (batch, seq, hidden_dim)

            # --- Train AR branch (freeze parallel) ---
            for p in draft_module.parallel_branch.parameters():
                p.requires_grad = False
            for p in draft_module.ar_branch.parameters():
                p.requires_grad = True
            for p in draft_module.shared_bottom.parameters():
                p.requires_grad = True

            ar_loss = compute_ar_loss(draft_module, hidden_states, input_ids, k_hard)
            optimizer_ar.zero_grad()
            ar_loss.backward()
            nn.utils.clip_grad_norm_(draft_module.parameters(), max_grad_norm)
            optimizer_ar.step()

            # --- Train Parallel branch (freeze AR) ---
            for p in draft_module.ar_branch.parameters():
                p.requires_grad = False
            for p in draft_module.parallel_branch.parameters():
                p.requires_grad = True
            for p in draft_module.shared_bottom.parameters():
                p.requires_grad = True

            par_loss = compute_parallel_loss(draft_module, hidden_states, input_ids)
            optimizer_par.zero_grad()
            par_loss.backward()
            nn.utils.clip_grad_norm_(draft_module.parameters(), max_grad_norm)
            optimizer_par.step()

            epoch_ar_loss += ar_loss.item()
            epoch_par_loss += par_loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}, Batch {batch_idx+1}: "
                      f"AR loss={ar_loss.item():.4f}, Parallel loss={par_loss.item():.4f}")

        avg_ar = epoch_ar_loss / max(num_batches, 1)
        avg_par = epoch_par_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs}: avg AR loss={avg_ar:.4f}, avg Parallel loss={avg_par:.4f}")
        exp_logger.log_metric("phase1_ar_loss", avg_ar, step=epoch)
        exp_logger.log_metric("phase1_parallel_loss", avg_par, step=epoch)


def train_phase2(
    draft_module: DAHDDraftModule,
    target_model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
    ar_loss_weight: float = 0.5,
    par_loss_weight: float = 0.5,
    k_hard: int = 3,
    exp_logger: ExperimentLogger = None,
    max_grad_norm: float = 1.0,
) -> None:
    """Phase 2: Joint fine-tuning with all parameters unfrozen.

    Args:
        draft_module: The DAHD draft module (loaded from phase1 checkpoint).
        target_model: Frozen target model.
        dataloader: Training data loader.
        optimizer: Joint optimizer for all draft module parameters.
        device: Device string.
        epochs: Number of joint training epochs.
        ar_loss_weight: Weight for AR branch loss.
        par_loss_weight: Weight for Parallel branch loss.
        k_hard: Draft length for AR branch.
        exp_logger: Logger instance.
        max_grad_norm: Maximum gradient norm for clipping.
    """
    print("\n[Phase 2 Training] Joint fine-tuning with shared bottom unfrozen...")

    # Unfreeze all parameters
    for p in draft_module.parameters():
        p.requires_grad = True

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Extract hidden states from frozen target model
            with torch.no_grad():
                outputs = target_model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]

            # Compute joint loss
            ar_loss = compute_ar_loss(draft_module, hidden_states, input_ids, k_hard)
            par_loss = compute_parallel_loss(draft_module, hidden_states, input_ids)
            total_loss = ar_loss_weight * ar_loss + par_loss_weight * par_loss

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(draft_module.parameters(), max_grad_norm)
            optimizer.step()

            epoch_loss += total_loss.item()
            num_batches += 1

            if (batch_idx + 1) % 50 == 0:
                print(f"    Epoch {epoch+1}, Batch {batch_idx+1}: "
                      f"total loss={total_loss.item():.4f} "
                      f"(AR={ar_loss.item():.4f}, Par={par_loss.item():.4f})")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs}: avg joint loss={avg_loss:.4f}")
        if exp_logger:
            exp_logger.log_metric("phase2_joint_loss", avg_loss, step=epoch)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 2: Train DAHD dual-branch draft module."
    )
    parser.add_argument(
        "--target_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
        help="Target model HuggingFace ID or path."
    )
    parser.add_argument("--hidden_dim", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--vocab_size", type=int, default=151936, help="Vocabulary size (Qwen2.5).")
    parser.add_argument("--k_easy", type=int, default=6, help="Draft length for easy samples.")
    parser.add_argument("--k_hard", type=int, default=3, help="Draft length for hard samples.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size.")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Checkpoint output dir.")
    parser.add_argument(
        "--phase", type=int, choices=[1, 2], default=1,
        help="Training phase: 1=separate pre-training, 2=joint fine-tuning."
    )
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="Path to training data JSONL. If None, uses synthetic data."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device.")
    parser.add_argument(
        "--checkpoint_path", type=str, default=None,
        help="Path to phase1 checkpoint (required for phase 2)."
    )
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm.")
    parser.add_argument("--ar_loss_weight", type=float, default=0.5, help="Weight for AR loss in phase 2.")
    parser.add_argument("--par_loss_weight", type=float, default=0.5, help="Weight for parallel loss in phase 2.")
    return parser.parse_args()


def main() -> None:
    """Main training entry point."""
    args = parse_args()
    setup_deterministic(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp_logger = ExperimentLogger(
        experiment_name=f"train_dual_branch_phase{args.phase}",
        log_dir=str(output_dir / "logs"),
    )
    exp_logger.log_phase_start(2, f"Dual-branch training (phase {args.phase})")

    print("=" * 60)
    print(f"DAHD Dual-Branch Training — Phase {args.phase}")
    print("=" * 60)
    print(f"  Target model:  {args.target_model}")
    print(f"  Hidden dim:    {args.hidden_dim}")
    print(f"  Vocab size:    {args.vocab_size}")
    print(f"  k_easy/k_hard: {args.k_easy}/{args.k_hard}")
    print(f"  LR:            {args.lr}")
    print(f"  Epochs:        {args.epochs}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Device:        {args.device}")
    print()

    # Load target model (frozen)
    print("[1/4] Loading target model (frozen)...")
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map=args.device,
        trust_remote_code=True,
    )
    target_model.eval()
    for p in target_model.parameters():
        p.requires_grad = False

    # Use actual model dimensions if available
    hidden_dim = target_model.config.hidden_size
    vocab_size = target_model.config.vocab_size
    print(f"  Model loaded: hidden_dim={hidden_dim}, vocab_size={vocab_size}")

    # Initialize or load DAHD draft module
    print("[2/4] Initializing DAHD draft module...")
    draft_module = DAHDDraftModule(
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        max_k=max(args.k_easy, args.k_hard),
        draft_length_easy=args.k_easy,
        draft_length_hard=args.k_hard,
    ).to(args.device)

    if args.phase == 2:
        # Load phase 1 checkpoint
        checkpoint_path = args.checkpoint_path or (output_dir / "dahd_phase1.pt")
        if not Path(checkpoint_path).exists():
            print(f"  ERROR: Phase 1 checkpoint not found at {checkpoint_path}")
            print("  Run phase 1 training first.")
            sys.exit(1)
        print(f"  Loading phase 1 checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=args.device)
        draft_module.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in draft_module.parameters())
    trainable_params = sum(p.numel() for p in draft_module.parameters() if p.requires_grad)
    print(f"  Draft module params: {total_params:,} total, {trainable_params:,} trainable")

    # Prepare dataset and dataloader
    print("[3/4] Preparing training data...")
    dataset = TokenDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=512,
        num_synthetic_samples=1000,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )
    print(f"  Dataset size: {len(dataset)} samples, {len(dataloader)} batches/epoch")

    # Train
    print("[4/4] Starting training...")
    start_time = time.time()

    if args.phase == 1:
        # Separate optimizers for AR and Parallel branches
        ar_params = list(draft_module.ar_branch.parameters()) + list(draft_module.shared_bottom.parameters())
        par_params = list(draft_module.parallel_branch.parameters()) + list(draft_module.shared_bottom.parameters())

        optimizer_ar = torch.optim.AdamW(ar_params, lr=args.lr, weight_decay=0.01)
        optimizer_par = torch.optim.AdamW(par_params, lr=args.lr, weight_decay=0.01)

        train_phase1(
            draft_module=draft_module,
            target_model=target_model,
            dataloader=dataloader,
            optimizer_ar=optimizer_ar,
            optimizer_par=optimizer_par,
            device=args.device,
            epochs=args.epochs,
            k_easy=args.k_easy,
            k_hard=args.k_hard,
            exp_logger=exp_logger,
            max_grad_norm=args.max_grad_norm,
        )

        # Save phase 1 checkpoint
        save_path = output_dir / "dahd_phase1.pt"
        torch.save(draft_module.state_dict(), save_path)
        print(f"\n  Phase 1 checkpoint saved: {save_path}")

    elif args.phase == 2:
        # Joint optimizer for all parameters
        optimizer = torch.optim.AdamW(
            draft_module.parameters(), lr=args.lr, weight_decay=0.01
        )

        train_phase2(
            draft_module=draft_module,
            target_model=target_model,
            dataloader=dataloader,
            optimizer=optimizer,
            device=args.device,
            epochs=args.epochs,
            ar_loss_weight=args.ar_loss_weight,
            par_loss_weight=args.par_loss_weight,
            k_hard=args.k_hard,
            exp_logger=exp_logger,
            max_grad_norm=args.max_grad_norm,
        )

        # Save final checkpoint
        save_path = output_dir / "dahd_final.pt"
        torch.save(draft_module.state_dict(), save_path)
        print(f"\n  Final checkpoint saved: {save_path}")

    elapsed = time.time() - start_time
    print(f"\n  Training complete in {elapsed:.1f}s")
    exp_logger.log_phase_end(2, summary={"elapsed_seconds": elapsed, "phase": args.phase})


if __name__ == "__main__":
    main()
