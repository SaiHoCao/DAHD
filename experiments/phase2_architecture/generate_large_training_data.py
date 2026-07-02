#!/usr/bin/env python3
"""Generate large-scale training data for Gumiho parallel heads.

Uses all 397 prompts with max_new_tokens=512, multiple seeds for diversity.
Target: 500K-1M samples for robust Gumiho parallel training.

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase2_architecture/generate_large_training_data.py
"""

import json
import os
import sys
import gc
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Config
TARGET_MODEL_PATH = "/mnt/nas1/hf/Qwen3-8B/"
DATA_SOURCE = "/mnt/nas1/hf/qwen3_8b_eagle3/eagle_data.jsonl"
OUTPUT_DIR = Path("/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/data/training_large")
MAX_NEW_TOKENS = 512  # Increased from 256
NUM_FUTURE_TOKENS = 6  # labels: next 6 tokens
TRAIN_SPLIT = 340  # 340 train, 57 val (more train prompts)
SEEDS = [42, 123, 777]  # Multiple seeds for diversity via temperature sampling
CHUNK_SIZE = 10000  # samples per chunk


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
    max_new_tokens: int = 512,
    do_sample: bool = False,
    temperature: float = 1.0,
    seed: int = 42,
) -> dict | None:
    """Run decoding and collect hidden states from last 3 layers.

    Returns:
        Dict with hidden_states_last3 and token_ids, or None if too short.
    """
    # Tokenize with chat template
    messages = [{"role": "user", "content": prompt}]
    try:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )
    except Exception:
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    inputs = tokenizer(formatted, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    # Generate
    if do_sample:
        torch.manual_seed(seed)
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )
    else:
        generated = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_hidden_states=False,
            return_dict_in_generate=False,
        )

    full_seq = generated[0]  # [total_len]
    total_len = full_seq.shape[0]
    gen_len = total_len - prompt_len

    if gen_len < NUM_FUTURE_TOKENS + 1:
        return None

    # Forward pass to get hidden states
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

    return {
        "hidden_states_last3": h_last3.to(torch.bfloat16),
        "token_ids": full_seq.cpu(),
    }


def create_training_samples(data: dict, num_future: int = 6) -> list:
    """Convert data into per-position training samples."""
    hidden_last3 = data["hidden_states_last3"]
    token_ids = data["token_ids"]
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
    print("=" * 70)
    print("Large-Scale Gumiho Training Data Generation")
    print("=" * 70)
    print(f"  Target: 500K+ samples")
    print(f"  Prompts: ALL 397 (train={TRAIN_SPLIT}, val={397-TRAIN_SPLIT})")
    print(f"  Max new tokens: {MAX_NEW_TOKENS}")
    print(f"  Seeds: {SEEDS}")
    print()

    # Setup output
    train_dir = OUTPUT_DIR / "train"
    val_dir = OUTPUT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    print(f"[1/3] Loading prompts from {DATA_SOURCE}")
    prompts = load_prompts(DATA_SOURCE)
    print(f"  Total prompts: {len(prompts)}")

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
    print(f"  Model loaded. GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Generate data
    print(f"\n[3/3] Generating training data...")

    for split_name, split_prompts, out_dir in [
        ("train", prompts[:TRAIN_SPLIT], train_dir),
        ("val", prompts[TRAIN_SPLIT:], val_dir),
    ]:
        print(f"\n{'='*50}")
        print(f"  {split_name.upper()} split: {len(split_prompts)} prompts")
        print(f"{'='*50}")

        all_samples = []
        chunk_idx = 0
        total_saved = 0
        num_skipped = 0

        for seed_idx, seed in enumerate(SEEDS):
            # First seed: greedy (deterministic, highest quality)
            # Subsequent seeds: sampling with temperature for diversity
            do_sample = (seed_idx > 0)
            temperature = 0.7 if do_sample else 1.0

            mode_str = f"greedy" if not do_sample else f"sample(T={temperature},seed={seed})"
            print(f"\n  --- Pass {seed_idx+1}/{len(SEEDS)}: {mode_str} ---")

            for idx, prompt in enumerate(tqdm(split_prompts, desc=f"  {split_name} pass{seed_idx+1}")):
                result = generate_and_collect(
                    model, tokenizer, prompt, MAX_NEW_TOKENS,
                    do_sample=do_sample, temperature=temperature, seed=seed + idx
                )

                if result is None:
                    num_skipped += 1
                    continue

                samples = create_training_samples(result, NUM_FUTURE_TOKENS)
                all_samples.extend(samples)

                # Save in chunks
                while len(all_samples) >= CHUNK_SIZE:
                    save_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
                    save_samples_batch(all_samples[:CHUNK_SIZE], save_path)
                    total_saved += CHUNK_SIZE
                    all_samples = all_samples[CHUNK_SIZE:]
                    chunk_idx += 1

                    if chunk_idx % 10 == 0:
                        elapsed = time.time() - start_time
                        print(f"    [{elapsed/60:.1f}min] Saved {total_saved} samples ({chunk_idx} chunks)")

                # Free GPU memory
                del result
                if idx % 30 == 0:
                    torch.cuda.empty_cache()

        # Save remaining samples
        if all_samples:
            save_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
            save_samples_batch(all_samples, save_path)
            total_saved += len(all_samples)
            chunk_idx += 1

        print(f"\n  {split_name}: {total_saved} total samples, {chunk_idx} chunks, {num_skipped} skipped")

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Data generation complete! Time: {elapsed/60:.1f} minutes")
    print(f"Output: {OUTPUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
