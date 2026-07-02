#!/usr/bin/env python3
"""Generate training data for DAHD Gumiho-style parallel branch.

Runs Qwen3-8B greedy decoding on prompts from eagle_data.jsonl,
collects last-layer hidden states at each token position,
and saves them with ground truth next-6-tokens as training labels.

Key fix vs v1: only collect samples from the GENERATED portion (skip prompt
tokens), since prompt positions are in "reading mode" and have a different
hidden state distribution from decoding-time positions.

Data format per sample:
  - hidden_last [4096]: last-layer hidden state at position t  (bf16)
  - token_id    int:    token at position t                   (the "current" token)
  - next_tokens [6]:    tokens at t+1, t+2, ..., t+6         (labels)

During training (Gumiho-style):
  mlp_input = fc(cat(embed(next_tokens[0]), hidden_last))  →  predict next_tokens[1:]
  (next_tokens[0] = token_{t+1} = target model's greedy prediction = known at inference)

Usage:
    CUDA_VISIBLE_DEVICES=2 python experiments/phase2_architecture/generate_training_data.py
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ── Config ───────────────────────────────────────────────────────────────────

TARGET_MODEL_PATH = "/mnt/nas1/hf/Qwen3-8B/"
DATA_SOURCE       = "/mnt/nas1/hf/qwen3_8b_eagle3/eagle_data.jsonl"
OUTPUT_DIR        = Path("/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/data/training")
MAX_NEW_TOKENS    = 256
NUM_FUTURE_TOKENS = 6   # labels: next 6 tokens
TRAIN_SPLIT       = 300  # first 300 prompts for training; rest → val
CHUNK_SIZE        = 10_000  # samples per .pt file


# ── Data loading ─────────────────────────────────────────────────────────────

def load_prompts(path: str) -> list[str]:
    prompts = []
    with open(path) as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data["query"])
    return prompts


# ── Generation + hidden-state collection ─────────────────────────────────────

@torch.no_grad()
def generate_and_collect(model, tokenizer, prompt: str) -> dict | None:
    """Greedy-decode and return last-layer hidden states + token ids.

    Returns:
        {
          'hidden_last': [total_len, 4096] bf16  – last-layer hidden at each pos
          'token_ids':   [total_len]        long  – full token sequence
          'prompt_len':  int                      – number of prompt tokens
        }
    or None if the generated portion is too short.
    """
    inputs    = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    prompt_len = input_ids.shape[1]

    generated = model.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=1.0,
        output_hidden_states=False,
        return_dict_in_generate=False,
    )

    full_seq = generated[0]          # [total_len]
    total_len = full_seq.shape[0]
    gen_len   = total_len - prompt_len

    if gen_len < NUM_FUTURE_TOKENS + 1:
        return None                  # too short to yield even one sample

    # Single forward pass to collect hidden states (causal mask ⟹ position t
    # only sees tokens 0..t, same as online decoding).
    outputs = model(
        full_seq.unsqueeze(0),
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_last = outputs.hidden_states[-1][0]   # [total_len, 4096]

    return {
        "hidden_last": hidden_last.to(torch.bfloat16).cpu(),
        "token_ids":   full_seq.cpu(),
        "prompt_len":  prompt_len,
    }


# ── Sample construction ───────────────────────────────────────────────────────

def create_training_samples(data: dict, num_future: int = NUM_FUTURE_TOKENS) -> list[dict]:
    """Per-position samples from the GENERATED portion only.

    For position t in [prompt_len, total_len - num_future):
        hidden_last  = hidden[t]               (hidden that predicts token t+1)
        token_id     = token_ids[t]            (current token at position t)
        next_tokens  = token_ids[t+1..t+num_future]

    The first label (next_tokens[0] = token_{t+1}) is the target model's
    greedy output at position t, which is KNOWN at inference time.  The
    remaining labels (next_tokens[1:]) are what the Gumiho MLP heads predict.
    """
    hidden  = data["hidden_last"]    # [total_len, 4096]
    tids    = data["token_ids"]      # [total_len]
    plen    = data["prompt_len"]
    tlen    = tids.shape[0]

    samples = []
    for t in range(plen, tlen - num_future):       # ← only generated positions
        samples.append({
            "hidden_last": hidden[t],              # [4096]  bf16
            "token_id":    tids[t].item(),         # int
            "next_tokens": tids[t+1 : t+1+num_future].tolist(),  # [6] ints
        })
    return samples


# ── Chunk serialisation ───────────────────────────────────────────────────────

def save_chunk(samples: list[dict], out_path: Path) -> None:
    if not samples:
        return
    hidden_stack = torch.stack([s["hidden_last"] for s in samples])      # [N, 4096]
    token_ids    = torch.tensor([s["token_id"]   for s in samples], dtype=torch.long)   # [N]
    next_tokens  = torch.tensor([s["next_tokens"] for s in samples], dtype=torch.long)  # [N, 6]

    torch.save({
        "hidden_last": hidden_stack,   # [N, 4096]  bf16
        "token_ids":   token_ids,      # [N]
        "next_tokens": next_tokens,    # [N, 6]
    }, out_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("DAHD Training Data Generation (v2 – generated-only)")
    print("=" * 60)

    train_dir = OUTPUT_DIR / "train"
    val_dir   = OUTPUT_DIR / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[1/3] Loading prompts from {DATA_SOURCE}")
    prompts = load_prompts(DATA_SOURCE)
    print(f"  Total: {len(prompts)}  |  train: {TRAIN_SPLIT}  |  val: {len(prompts)-TRAIN_SPLIT}")

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
    print(f"  Loaded on {model.device}")

    print(f"\n[3/3] Generating training data (max_new_tokens={MAX_NEW_TOKENS})")

    for split_name, split_prompts, out_dir in [
        ("train", prompts[:TRAIN_SPLIT],       train_dir),
        ("val",   prompts[TRAIN_SPLIT:],        val_dir),
    ]:
        print(f"\n  --- {split_name} ({len(split_prompts)} prompts) ---")
        all_samples: list[dict] = []
        n_skipped = 0

        for idx, prompt in enumerate(tqdm(split_prompts, desc=f"  {split_name}")):
            result = generate_and_collect(model, tokenizer, prompt)
            if result is None:
                n_skipped += 1
                continue

            samples = create_training_samples(result)
            all_samples.extend(samples)

            if len(all_samples) >= CHUNK_SIZE:
                chunk_idx = len(list(out_dir.glob("chunk_*.pt")))
                save_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
                save_chunk(all_samples, save_path)
                print(f"    → saved {len(all_samples)} samples  ({save_path.name})")
                all_samples = []

            if idx % 50 == 0:
                torch.cuda.empty_cache()

        if all_samples:
            chunk_idx = len(list(out_dir.glob("chunk_*.pt")))
            save_path = out_dir / f"chunk_{chunk_idx:04d}.pt"
            save_chunk(all_samples, save_path)
            print(f"    → saved {len(all_samples)} samples  ({save_path.name})")

        total = 0
        for p in out_dir.glob("chunk_*.pt"):
            c = torch.load(p, map_location="cpu", weights_only=False)
            key = "hidden_last" if "hidden_last" in c else "hidden_concat"
            total += c[key].shape[0]
        print(f"  {split_name}: {total:,} samples  ({n_skipped} prompts skipped)")

    print("\n" + "=" * 60)
    print("Done.")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
