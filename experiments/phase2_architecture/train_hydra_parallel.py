#!/usr/bin/env python3
"""Train DAHD Hydra Parallel Branch + Router.

Phase A: Train Hydra Parallel Branch (sequential-dependent heads)
Phase B: Train Difficulty Router
Phase C: Quick validation (acceptance rate comparison)

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase2_architecture/train_hydra_parallel.py
"""

import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =============================================================================
# Config
# =============================================================================
EAGLE3_CKPT = "/mnt/nas1/hf/qwen3_8b_eagle3/pytorch_model.bin"
TRAIN_DATA_DIR = Path("/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/data/training/train")
VAL_DATA_DIR = Path("/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/data/training/val")
CKPT_DIR = Path("/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/checkpoints")

HIDDEN_SIZE = 4096
DRAFT_VOCAB_SIZE = 32000  # Same as EAGLE-3 (not full 151936!)
NUM_HEADS = 6
BATCH_SIZE = 128
LR = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 100
EPOCHS_PB = 5  # Parallel Branch epochs
EPOCHS_ROUTER = 3  # Router epochs
DEVICE = "cuda:0"


# =============================================================================
# Model Architecture
# =============================================================================

class HydraParallelBranch(nn.Module):
    """Hydra-style sequential-dependent parallel heads.

    Architecture:
    - SharedBottom fc: [hidden_size*3, hidden_size] (reused from EAGLE-3)
    - 6 sequential-dependent heads, each producing logits over draft_vocab_size
    - Each head receives the shared repr + projected hidden from previous head
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        num_heads: int = 6,
        draft_vocab_size: int = 32000,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.draft_vocab_size = draft_vocab_size

        # SharedBottom: fc layer (will load EAGLE-3 weights)
        self.fc = nn.Linear(hidden_size * 3, hidden_size, bias=False)

        # Shared lm_head (will load EAGLE-3 lm_head weights)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)

        # 6 heads: each has LayerNorm + Linear transform
        self.head_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_heads)
        ])
        self.head_linears = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_heads)
        ])

        # Sequential dependency: projection from previous head's hidden → current input
        # First head has no dependency, so we have (num_heads - 1) projections
        self.head_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_heads - 1)
        ])

    def forward(self, hidden_concat: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            hidden_concat: [B, 12288] - concat of last 3 layer hidden states

        Returns:
            List of [B, draft_vocab_size] logits, one per head
        """
        shared = self.fc(hidden_concat)  # [B, 4096]

        all_logits = []
        prev_hidden = shared

        for i in range(self.num_heads):
            # Sequential dependency: add projected previous hidden
            if i > 0:
                projected = self.head_projections[i - 1](prev_hidden)
                curr_input = shared + projected  # residual
            else:
                curr_input = shared

            # Head transform
            h = self.head_norms[i](curr_input)  # LayerNorm
            h = self.head_linears[i](h)  # Linear → hidden_size
            prev_hidden = h  # Save for next head

            h = F.silu(h)  # Activation

            # Project to vocab with shared lm_head
            logit = self.lm_head(h)  # [B, draft_vocab_size]
            all_logits.append(logit)

        return all_logits

    def load_eagle3_weights(self, eagle3_state_dict: dict):
        """Load fc and lm_head weights from EAGLE-3 checkpoint."""
        # Load fc
        if "fc.weight" in eagle3_state_dict:
            self.fc.weight.data.copy_(eagle3_state_dict["fc.weight"])
            print(f"  Loaded fc.weight: {eagle3_state_dict['fc.weight'].shape}")

        # Load lm_head
        if "lm_head.weight" in eagle3_state_dict:
            self.lm_head.weight.data.copy_(eagle3_state_dict["lm_head.weight"])
            print(f"  Loaded lm_head.weight: {eagle3_state_dict['lm_head.weight'].shape}")


class DifficultyRouter(nn.Module):
    """Simple MLP router that predicts whether a position is easy/hard."""

    def __init__(self, hidden_size: int = 4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, shared_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_repr: [B, hidden_size] - output of fc layer

        Returns:
            [B, 1] - probability of being "easy"
        """
        return torch.sigmoid(self.net(shared_repr))


# =============================================================================
# Dataset
# =============================================================================

def build_target_to_draft_mapping(d2t: torch.Tensor, vocab_size: int = 151936) -> torch.Tensor:
    """Build a target_token_id → draft_token_id mapping.

    Args:
        d2t: [draft_vocab_size] mapping draft_id → target_id
        vocab_size: target vocabulary size

    Returns:
        [vocab_size] tensor where target_to_draft[target_id] = draft_id,
        or -100 if not in draft vocab (will be ignored in CE loss)
    """
    target_to_draft = torch.full([vocab_size], fill_value=-100, dtype=torch.long)
    for draft_id in range(d2t.shape[0]):
        target_id = d2t[draft_id].item()
        target_to_draft[target_id] = draft_id
    return target_to_draft


class HiddenStateDataset(Dataset):
    """Dataset loading pre-generated hidden states and labels."""

    def __init__(self, data_dir: Path, target_to_draft: torch.Tensor | None = None):
        """
        Args:
            data_dir: Directory containing chunk_*.pt files
            target_to_draft: [vocab_size] mapping target_id → draft_id (or -100)
        """
        self.target_to_draft = target_to_draft
        self.chunks = []
        self.cumulative_sizes = []

        chunk_files = sorted(data_dir.glob("chunk_*.pt"))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk_*.pt files in {data_dir}")

        total = 0
        for cf in chunk_files:
            data = torch.load(cf, map_location="cpu")
            n = data["hidden_concat"].shape[0]
            self.chunks.append(data)
            total += n
            self.cumulative_sizes.append(total)

        print(f"  Loaded {len(chunk_files)} chunks, {total} samples from {data_dir}")

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict:
        # Find which chunk this index belongs to
        chunk_idx = 0
        for i, cs in enumerate(self.cumulative_sizes):
            if idx < cs:
                chunk_idx = i
                break

        # Local index within chunk
        local_idx = idx if chunk_idx == 0 else idx - self.cumulative_sizes[chunk_idx - 1]

        chunk = self.chunks[chunk_idx]
        hidden_concat = chunk["hidden_concat"][local_idx]  # [12288] bf16
        next_tokens = chunk["next_tokens"][local_idx]  # [6] long (target token ids)

        # Map target token ids → draft token ids
        if self.target_to_draft is not None:
            next_tokens_draft = self.target_to_draft[next_tokens]  # [6], -100 for OOV
        else:
            next_tokens_draft = next_tokens

        return {
            "hidden_concat": hidden_concat.float(),  # convert bf16 → float32 for training
            "next_tokens": next_tokens_draft,  # [6] in draft vocab (or -100)
        }


# =============================================================================
# Training utilities
# =============================================================================

def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Cosine learning rate schedule with linear warmup."""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate_parallel_branch(
    model: HydraParallelBranch,
    dataloader: DataLoader,
    device: str,
) -> dict:
    """Evaluate per-head accuracy on validation set."""
    model.eval()
    correct_per_head = [0] * model.num_heads
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            hidden = batch["hidden_concat"].to(device)
            targets = batch["next_tokens"].to(device)  # [B, 6]

            logits_list = model(hidden)  # list of [B, vocab]

            for i, logits in enumerate(logits_list):
                preds = logits.argmax(dim=-1)  # [B]
                valid_mask = targets[:, i] != -100  # ignore OOV
                correct_per_head[i] += (preds[valid_mask] == targets[:, i][valid_mask]).sum().item()

            # Count valid samples (non -100) for head 0
            total += (targets[:, 0] != -100).sum().item()

    acc_per_head = [c / max(total, 1) for c in correct_per_head]
    model.train()
    return {f"head_{i}_acc": acc for i, acc in enumerate(acc_per_head)}


# =============================================================================
# Phase A: Train Parallel Branch
# =============================================================================

def train_parallel_branch():
    """Train the Hydra Parallel Branch."""
    print("\n" + "=" * 60)
    print("Phase A: Training Hydra Parallel Branch")
    print("=" * 60)

    # Load EAGLE-3 weights for initialization
    print("\n[1/4] Loading EAGLE-3 weights...")
    eagle3_sd = torch.load(EAGLE3_CKPT, map_location="cpu")
    d2t_mapping = eagle3_sd["d2t"]  # [32000] → target vocab id
    target_to_draft = build_target_to_draft_mapping(d2t_mapping)
    print(f"  d2t shape: {d2t_mapping.shape}")
    valid_draft_count = (target_to_draft >= 0).sum().item()
    print(f"  Valid target→draft mappings: {valid_draft_count}")

    # Initialize model
    print("\n[2/4] Initializing HydraParallelBranch...")
    model = HydraParallelBranch(
        hidden_size=HIDDEN_SIZE,
        num_heads=NUM_HEADS,
        draft_vocab_size=DRAFT_VOCAB_SIZE,
    ).to(DEVICE)
    model.load_eagle3_weights(eagle3_sd)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total_params:,} ({total_params * 4 / 1e6:.1f} MB in fp32)")
    print(f"  Trainable: {trainable_params:,}")

    # Load datasets
    print("\n[3/4] Loading datasets...")
    train_dataset = HiddenStateDataset(TRAIN_DATA_DIR, target_to_draft)
    val_dataset = HiddenStateDataset(VAL_DATA_DIR, target_to_draft)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Setup optimizer and scheduler
    total_steps = len(train_loader) * EPOCHS_PB
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, WARMUP_STEPS, total_steps)

    print(f"\n[4/4] Training for {EPOCHS_PB} epochs ({total_steps} steps)")
    print(f"  Batch size: {BATCH_SIZE}, LR: {LR}")
    print(f"  Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Training loop
    best_val_acc = 0.0
    for epoch in range(EPOCHS_PB):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"  Epoch {epoch+1}/{EPOCHS_PB}")
        for batch in pbar:
            hidden = batch["hidden_concat"].to(DEVICE)  # [B, 12288]
            targets = batch["next_tokens"].to(DEVICE)  # [B, 6]

            # Forward
            logits_list = model(hidden)  # list of [B, draft_vocab]

            # Loss: average CE across all 6 heads (ignore -100 = OOV tokens)
            loss = 0.0
            for i, logits in enumerate(logits_list):
                loss += F.cross_entropy(logits, targets[:, i], ignore_index=-100)
            loss = loss / NUM_HEADS

            # Backward
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            num_batches += 1

            if num_batches % 50 == 0:
                pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        avg_loss = epoch_loss / max(num_batches, 1)

        # Evaluate
        val_metrics = evaluate_parallel_branch(model, val_loader, DEVICE)
        avg_val_acc = sum(val_metrics.values()) / len(val_metrics)

        print(f"\n  Epoch {epoch+1}: loss={avg_loss:.4f}, val_avg_acc={avg_val_acc:.4f}")
        for k, v in val_metrics.items():
            print(f"    {k}: {v:.4f}")

        # Save best model
        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            save_path = CKPT_DIR / "hydra_parallel_branch_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "val_acc": avg_val_acc,
                "val_metrics": val_metrics,
            }, save_path)
            print(f"  Saved best model (acc={avg_val_acc:.4f}) → {save_path}")

    # Save final model
    save_path = CKPT_DIR / "hydra_parallel_branch_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": EPOCHS_PB,
    }, save_path)
    print(f"\n  Final model saved → {save_path}")

    return model


# =============================================================================
# Phase B: Train Router
# =============================================================================

def train_router(parallel_branch: HydraParallelBranch | None = None):
    """Train the Difficulty Router.

    Labels: head_0 of parallel branch correct → easy (1), incorrect → hard (0)
    """
    print("\n" + "=" * 60)
    print("Phase B: Training Difficulty Router")
    print("=" * 60)

    # Load EAGLE-3 mappings
    eagle3_sd = torch.load(EAGLE3_CKPT, map_location="cpu")
    d2t_mapping = eagle3_sd["d2t"]
    target_to_draft = build_target_to_draft_mapping(d2t_mapping)

    # Load parallel branch if not provided
    if parallel_branch is None:
        print("\n[1/4] Loading trained Parallel Branch...")
        parallel_branch = HydraParallelBranch(
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            draft_vocab_size=DRAFT_VOCAB_SIZE,
        ).to(DEVICE)
        ckpt = torch.load(CKPT_DIR / "hydra_parallel_branch_best.pt", map_location=DEVICE)
        parallel_branch.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded from checkpoint (val_acc={ckpt.get('val_acc', 'N/A')})")

    parallel_branch.eval()
    for p in parallel_branch.parameters():
        p.requires_grad = False

    # Generate router labels
    print("\n[2/4] Generating router labels...")
    train_dataset = HiddenStateDataset(TRAIN_DATA_DIR, target_to_draft)
    val_dataset = HiddenStateDataset(VAL_DATA_DIR, target_to_draft)

    # Generate labels for training data
    def generate_router_labels(dataset, desc="Generating labels"):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)
        all_shared = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {desc}"):
                hidden = batch["hidden_concat"].to(DEVICE)
                targets = batch["next_tokens"].to(DEVICE)  # [B, 6]

                # Get shared representation (fc output)
                shared = parallel_branch.fc(hidden)  # [B, 4096]

                # Check if head_0 is correct
                logits_list = parallel_branch(hidden)
                head0_preds = logits_list[0].argmax(dim=-1)  # [B]
                labels = (head0_preds == targets[:, 0]).float()  # 1=easy, 0=hard

                all_shared.append(shared.cpu())
                all_labels.append(labels.cpu())

        return torch.cat(all_shared, dim=0), torch.cat(all_labels, dim=0)

    train_shared, train_labels = generate_router_labels(train_dataset, "Train labels")
    val_shared, val_labels = generate_router_labels(val_dataset, "Val labels")

    easy_ratio_train = train_labels.mean().item()
    easy_ratio_val = val_labels.mean().item()
    print(f"  Train: {len(train_labels)} samples, easy ratio: {easy_ratio_train:.4f}")
    print(f"  Val: {len(val_labels)} samples, easy ratio: {easy_ratio_val:.4f}")

    # Initialize router
    print("\n[3/4] Initializing Router...")
    router = DifficultyRouter(hidden_size=HIDDEN_SIZE).to(DEVICE)
    router_params = sum(p.numel() for p in router.parameters())
    print(f"  Router params: {router_params:,}")

    # Simple tensor dataset for router training
    class RouterDataset(Dataset):
        def __init__(self, shared_repr, labels):
            self.shared_repr = shared_repr
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.shared_repr[idx], self.labels[idx]

    router_train = RouterDataset(train_shared, train_labels)
    router_val = RouterDataset(val_shared, val_labels)

    router_train_loader = DataLoader(
        router_train, batch_size=BATCH_SIZE * 2, shuffle=True, num_workers=2
    )
    router_val_loader = DataLoader(
        router_val, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=2
    )

    # Train router
    print(f"\n[4/4] Training Router for {EPOCHS_ROUTER} epochs...")
    optimizer = torch.optim.AdamW(router.parameters(), lr=5e-4, weight_decay=0.01)

    for epoch in range(EPOCHS_ROUTER):
        router.train()
        epoch_loss = 0.0
        num_batches = 0

        for shared_repr, labels in tqdm(router_train_loader, desc=f"  Epoch {epoch+1}"):
            shared_repr = shared_repr.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(-1)  # [B, 1]

            preds = router(shared_repr)  # [B, 1]
            loss = F.binary_cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Evaluate
        router.eval()
        correct = 0
        total = 0
        tp = fp = fn = tn = 0
        with torch.no_grad():
            for shared_repr, labels in router_val_loader:
                shared_repr = shared_repr.to(DEVICE)
                labels = labels.to(DEVICE)

                preds = router(shared_repr).squeeze(-1)  # [B]
                pred_labels = (preds > 0.5).float()

                correct += (pred_labels == labels).sum().item()
                total += labels.shape[0]

                tp += ((pred_labels == 1) & (labels == 1)).sum().item()
                fp += ((pred_labels == 1) & (labels == 0)).sum().item()
                fn += ((pred_labels == 0) & (labels == 1)).sum().item()
                tn += ((pred_labels == 0) & (labels == 0)).sum().item()

        acc = correct / max(total, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, acc={acc:.4f}, "
              f"precision={precision:.4f}, recall={recall:.4f}")

    # Save router
    save_path = CKPT_DIR / "difficulty_router.pt"
    torch.save({
        "model_state_dict": router.state_dict(),
        "train_easy_ratio": easy_ratio_train,
        "val_easy_ratio": easy_ratio_val,
        "val_acc": acc,
    }, save_path)
    print(f"\n  Router saved → {save_path}")

    return router


# =============================================================================
# Phase C: Quick Validation
# =============================================================================

def validate_acceptance_rate(
    parallel_branch: HydraParallelBranch | None = None,
    router: DifficultyRouter | None = None,
):
    """Validate: compute average accepted tokens per decision."""
    print("\n" + "=" * 60)
    print("Phase C: Acceptance Rate Validation")
    print("=" * 60)

    eagle3_sd = torch.load(EAGLE3_CKPT, map_location="cpu")
    d2t_mapping = eagle3_sd["d2t"]
    target_to_draft = build_target_to_draft_mapping(d2t_mapping)

    # Load models if needed
    if parallel_branch is None:
        parallel_branch = HydraParallelBranch(
            hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS,
            draft_vocab_size=DRAFT_VOCAB_SIZE,
        ).to(DEVICE)
        ckpt = torch.load(CKPT_DIR / "hydra_parallel_branch_best.pt", map_location=DEVICE)
        parallel_branch.load_state_dict(ckpt["model_state_dict"])

    if router is None:
        router = DifficultyRouter(hidden_size=HIDDEN_SIZE).to(DEVICE)
        ckpt = torch.load(CKPT_DIR / "difficulty_router.pt", map_location=DEVICE)
        router.load_state_dict(ckpt["model_state_dict"])

    parallel_branch.eval()
    router.eval()

    # Load validation data
    val_dataset = HiddenStateDataset(VAL_DATA_DIR, target_to_draft)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4)

    # Compute acceptance rates
    total_easy_accepted = 0
    total_easy_count = 0
    total_hard_accepted = 0
    total_hard_count = 0
    total_parallel_only_accepted = 0  # baseline: always use parallel

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validating"):
            hidden = batch["hidden_concat"].to(DEVICE)
            targets = batch["next_tokens"].to(DEVICE)  # [B, 6]

            # Get shared repr and router decision
            shared = parallel_branch.fc(hidden)
            router_probs = router(shared).squeeze(-1)  # [B]
            easy_mask = router_probs > 0.5  # [B]

            # Get parallel branch predictions
            logits_list = parallel_branch(hidden)

            # For "easy" positions: count consecutive correct tokens from parallel branch
            for i in range(hidden.shape[0]):
                if easy_mask[i]:
                    # Easy: use all 6 parallel heads
                    accepted = 0
                    for h in range(NUM_HEADS):
                        if targets[i, h] == -100:  # OOV token, stop
                            break
                        pred = logits_list[h][i].argmax()
                        if pred == targets[i, h]:
                            accepted += 1
                        else:
                            break
                    total_easy_accepted += accepted
                    total_easy_count += 1
                else:
                    # Hard: use only first 3 heads (simulating shorter draft)
                    accepted = 0
                    for h in range(min(3, NUM_HEADS)):
                        if targets[i, h] == -100:  # OOV token, stop
                            break
                        pred = logits_list[h][i].argmax()
                        if pred == targets[i, h]:
                            accepted += 1
                        else:
                            break
                    total_hard_accepted += accepted
                    total_hard_count += 1

                # Baseline: always parallel, all 6
                accepted_baseline = 0
                for h in range(NUM_HEADS):
                    if targets[i, h] == -100:
                        break
                    pred = logits_list[h][i].argmax()
                    if pred == targets[i, h]:
                        accepted_baseline += 1
                    else:
                        break
                total_parallel_only_accepted += accepted_baseline

    total_count = total_easy_count + total_hard_count
    avg_easy = total_easy_accepted / max(total_easy_count, 1)
    avg_hard = total_hard_accepted / max(total_hard_count, 1)
    avg_overall = (total_easy_accepted + total_hard_accepted) / max(total_count, 1)
    avg_baseline = total_parallel_only_accepted / max(total_count, 1)

    print(f"\n  Results on validation set ({total_count} positions):")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Router decisions: easy={total_easy_count} ({total_easy_count/max(total_count,1)*100:.1f}%), "
          f"hard={total_hard_count} ({total_hard_count/max(total_count,1)*100:.1f}%)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Avg accepted tokens (easy, k=6):  {avg_easy:.3f}")
    print(f"  Avg accepted tokens (hard, k=3):  {avg_hard:.3f}")
    print(f"  Avg accepted tokens (DAHD overall): {avg_overall:.3f}")
    print(f"  Avg accepted tokens (baseline, always parallel k=6): {avg_baseline:.3f}")
    print(f"  ─────────────────────────────────────────────")

    # Save results
    results = {
        "total_positions": total_count,
        "easy_count": total_easy_count,
        "hard_count": total_hard_count,
        "avg_accepted_easy": avg_easy,
        "avg_accepted_hard": avg_hard,
        "avg_accepted_dahd": avg_overall,
        "avg_accepted_baseline_parallel": avg_baseline,
    }
    results_path = CKPT_DIR / "validation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved → {results_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DAHD Hydra Parallel Branch Training Pipeline")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Hidden size: {HIDDEN_SIZE}")
    print(f"  Draft vocab size: {DRAFT_VOCAB_SIZE}")
    print(f"  Num heads: {NUM_HEADS}")
    print(f"  Batch size: {BATCH_SIZE}")

    # Phase A: Train Parallel Branch
    parallel_branch = train_parallel_branch()

    # Phase B: Train Router
    router = train_router(parallel_branch)

    # Phase C: Validate
    validate_acceptance_rate(parallel_branch, router)

    print("\n" + "=" * 60)
    print("All phases complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
