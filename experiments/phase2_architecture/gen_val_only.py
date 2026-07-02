"""One-shot: generate val split only (prompts[300:]), then exit.
Run after train split already exists.
"""
import sys
sys.path.insert(0, __file__.replace('/experiments/phase2_architecture/gen_val_only.py', ''))

# Patch constant before importing main module
import experiments.phase2_architecture.generate_training_data as gd
gd.TRAIN_SPLIT = 300   # keep same split boundary

import json, torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

OUTPUT_DIR = Path("/data1/caoshuaihu.csh/workspace/dahd_speculative_decoding/data/training")
val_dir = OUTPUT_DIR / "val"
val_dir.mkdir(parents=True, exist_ok=True)

print("Loading prompts …")
prompts = gd.load_prompts(gd.DATA_SOURCE)
val_prompts = prompts[gd.TRAIN_SPLIT:]   # 97 prompts
print(f"Val prompts: {len(val_prompts)}")

print("Loading model …")
tokenizer = AutoTokenizer.from_pretrained(gd.TARGET_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    gd.TARGET_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map={"": "cuda:0"},
    trust_remote_code=True,
)
model.eval()
print(f"Loaded on {model.device}")

print("Generating val data …")
all_samples = []
n_skipped = 0

for idx, prompt in enumerate(tqdm(val_prompts, desc="val")):
    result = gd.generate_and_collect(model, tokenizer, prompt)
    if result is None:
        n_skipped += 1
        continue
    all_samples.extend(gd.create_training_samples(result))
    if len(all_samples) >= gd.CHUNK_SIZE:
        chunk_idx = len(list(val_dir.glob("chunk_*.pt")))
        save_path = val_dir / f"chunk_{chunk_idx:04d}.pt"
        gd.save_chunk(all_samples, save_path)
        print(f"  → saved {len(all_samples)} samples ({save_path.name})")
        all_samples = []
    if idx % 20 == 0:
        torch.cuda.empty_cache()

if all_samples:
    chunk_idx = len(list(val_dir.glob("chunk_*.pt")))
    save_path = val_dir / f"chunk_{chunk_idx:04d}.pt"
    gd.save_chunk(all_samples, save_path)
    print(f"  → saved {len(all_samples)} samples ({save_path.name})")

# Count total
total = 0
for p in val_dir.glob("chunk_*.pt"):
    c = torch.load(p, map_location="cpu", weights_only=False)
    key = "hidden_last" if "hidden_last" in c else "hidden_concat"
    total += c[key].shape[0]
print(f"\nVal total: {total:,} samples ({n_skipped} prompts skipped)")
print("Done.")
