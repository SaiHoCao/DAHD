#!/usr/bin/env python3
"""Generate LARGE training data for Gumiho Parallel Heads.

Generates 500K+ samples by using all 397 prompts with longer generation (1024 tokens).
Saves to data/training_large/ with same format as original (hidden_concat, token_ids, next_tokens).

Usage:
    CUDA_VISIBLE_DEVICES=6 python experiments/phase2_architecture/generate_training_data_large.py
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Config
TARGET_MODEL_PATH = "/mnt/nas1/hf/Qwen3-8B/"
DATA_SOURCE = "/mnt/nas1/hf/qwen3_8b_eagle3/eagle_data.jsonl"
OUTPUT_DIR = Path("/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/data/training_large")
MAX_NEW_TOKENS = 1024  # 4x original (256) for more samples per prompt
NUM_FUTURE_TOKENS = 6  # labels: next 6 tokens
TRAIN_SPLIT = 350  # first 350 prompts for training (was 300)
# last 47 for validation
CHUNK_SIZE = 15000  # samples per chunk file


def load_prompts(path: str) -> list:
    """Load prompts from eagle_data.jsonl."""
    prompts = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["query"])
    return prompts


@torch.no_grad()
def generate_and_collect(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 1024,
) -> dict | None:
    """Run greedy decoding and collect hidden states from last 3 layers."""
    # Tokenize prompt
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    # Generate with greedy decoding
    generated = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        output_hidden_states=False,
        return_dict_in_generate=False,
    )

    full_seq = generated[0]  # [total_len]
    total_len = full_seq.shape[0]
    gen_len = total_len - prompt_len

    if gen_len < NUM_FUTURE_TOKENS + 1:
        return None

    # Forward pass to get hidden states - process in chunks if sequence is very long
    # to avoid OOM on hidden states (37 layers × seq_len × 4096 × 2 bytes)
    if total_len > 2048:
        # Process in two halves with overlap
        mid = total_len // 2
        overlap = NUM_FUTURE_TOKENS + 10  # enough overlap for continuity
        
        # First half
        out1 = model(
            full_seq[:mid + overlap].unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
        h1_last3 = torch.stack([
            out1.hidden_states[-3][0],
            out1.hidden_states[-2][0],
            out1.hidden_states[-1][0],
        ], dim=1)[:mid]  # [mid, 3, 4096]
        del out1
        torch.cuda.empty_cache()
        
        # Second half
        out2 = model(
            full_seq[mid:].unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
        h2_last3 = torch.stack([
            out2.hidden_states[-3][0],
            out2.hidden_states[-2][0],
            out2.hidden_states[-1][0],
        ], dim=1)  # [total_len - mid, 3, 4096]
        del out2
        torch.cuda.empty_cache()
        
        h_last3 = torch.cat([h1_last3, h2_last3], dim=0)  # [total_len, 3, 4096]
        del h1_last3, h2_last3
    else:
        outputs = model(
            full_seq.unsqueeze(0),
            output_hidden_states=True,
            use_cache=False,
        )
        all_hidden = outputs.hidden_states
        h_last3 = torch.stack([
            all_hidden[-3][0],
            all_hidden[-2][0],
            all_hidden[-1][0],
        ], dim=1)  # [seq_len, 3, 4096]
        del outputs
        torch.cuda.empty_cache()

    return {
        "hidden_states_last3": h_last3.to(torch.bfloat16),
        "token_ids": full_seq.cpu(),
    }


def create_training_samples(data: dict, num_future: int = 6) -> list:
    """Convert a single prompt's data into per-position training samples."""
    hidden_last3 = data["hidden_states_last3"]  # [seq_len, 3, 4096]
    token_ids = data["token_ids"]  # [seq_len]
    seq_len = token_ids.shape[0]

    samples = []
    for t in range(seq_len - num_future):
        hidden_concat = hidden_last3[t].reshape(-1)  # [12288]
        token_id = token_ids[t].item()
        next_tokens = token_ids[t + 1: t + 1 + num_future].tolist()

        samples.append({
            "hidden_concat": hidden_concat,
            "token_id": token_id,
            "next_tokens": next_tokens,
        })

    return samples


def save_samples_batch(samples: list, output_path: Path):
    """Save a batch of samples as a single .pt file."""
    if not samples:
        return

    hidden_stack = torch.stack([s["hidden_concat"] for s in samples])
    token_ids = torch.tensor([s["token_id"] for s in samples], dtype=torch.long)
    next_tokens = torch.tensor([s["next_tokens"] for s in samples], dtype=torch.long)

    torch.save({
        "hidden_concat": hidden_stack,
        "token_ids": token_ids,
        "next_tokens": next_tokens,
    }, output_path)


def main():
    start_time = time.time()
    print("=" * 60)
    print("DAHD Large-Scale Training Data Generation")
    print("=" * 60)
    print(f"  Target: 500K+ samples")
    print(f"  max_new_tokens: {MAX_NEW_TOKENS}")
    print(f"  Train/Val split: {TRAIN_SPLIT}/{397 - TRAIN_SPLIT}")

    # Setup output directories
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"\n[1/3] Loading prompts from {DATA_SOURCE}")
    prompts = load_prompts(DATA_SOURCE)
    print(f"  Total prompts: {len(prompts)}")
    print(f"  Train split: {TRAIN_SPLIT}, Val split: {len(prompts) - TRAIN_SPLIT}")

    # Load model
    print(f"\n[2/3] Loading model from {TARGET_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        TARGET_MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on {model.device}")
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # Generate data
    print(f"\n[3/3] Generating training data (max_new_tokens={MAX_NEW_TOKENS})")

    for split_name, split_prompts, out_dir in [
        ("train", prompts[:TRAIN_SPLIT], train_dir),
        ("val", prompts[TRAIN_SPLIT:], val_dir),
    ]:
        print(f"\n  --- {split_name} split ({len(split_prompts)} prompts) ---")
        all_samples = []
        num_skipped = 0
        split_start = time.time()

        for idx, prompt in enumerate(tqdm(split_prompts, desc=f"  {split_name}")):
            result = generate_and_collect(model, tokenizer, prompt, MAX_NEW_TOKENS)

            if result is None:
                num_skipped += 1
                continue

            samples = create_training_samples(result, NUM_FUTURE_TOKENS)
            all_samples.extend(samples)

            # Save in chunks
            if len(all_samples) >= CHUNK_SIZE:
                chunk_idx = len(list(out_dir.glob("chunk_*.pt")))
                save_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
                save_samples_batch(all_samples[:CHUNK_SIZE], save_path)
                all_samples = all_samples[CHUNK_SIZE:]
                tqdm.write(f"    Saved chunk_{chunk_idx:04d}.pt ({CHUNK_SIZE} samples)")

            # Free GPU memory periodically
            del result
            if idx % 20 == 0:
                torch.cuda.empty_cache()

            # Progress update
            if (idx + 1) % 50 == 0:
                elapsed = time.time() - split_start
                rate = (idx + 1) / elapsed
                remaining = (len(split_prompts) - idx - 1) / rate
                tqdm.write(f"    Progress: {idx+1}/{len(split_prompts)}, "
                          f"ETA: {remaining/60:.1f} min, "
                          f"Samples so far: {len(all_samples) + chunk_idx * CHUNK_SIZE if 'chunk_idx' in dir() else len(all_samples)}")

        # Save remaining samples
        if all_samples:
            chunk_idx = len(list(out_dir.glob("chunk_*.pt")))
            save_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
            save_samples_batch(all_samples, save_path)
            print(f"    Saved final chunk_{chunk_idx:04d}.pt ({len(all_samples)} samples)")

        # Print stats
        total_samples = sum(
            torch.load(p, map_location="cpu", weights_only=False)["hidden_concat"].shape[0]
            for p in out_dir.glob("chunk_*.pt")
        )
        split_time = time.time() - split_start
        print(f"  {split_name}: {total_samples} total samples, "
              f"{num_skipped} prompts skipped, {split_time/60:.1f} min")

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Data generation complete!")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Total time: {total_time/60:.1f} min ({total_time/3600:.2f} hours)")
    print("=" * 60)


if __name__ == "__main__":
    main()
