#!/usr/bin/env python3
"""Migrate ViT checkpoint from separate q_proj/kv_proj to packed qkv_proj.

Old layout (per block):
  blocks.{i}.attn.q_proj.weight   [H*d, d_model]
  blocks.{i}.attn.q_proj.bias     [H*d]
  blocks.{i}.attn.kv_proj.weight  [2*G*d, d_model]
  blocks.{i}.attn.kv_proj.bias    [2*G*d]

New layout (per block):
  blocks.{i}.attn.qkv_proj.weight [H*d + 2*G*d, d_model]
  blocks.{i}.attn.qkv_proj.bias   [H*d + 2*G*d]

Optimizer state is reset (cannot be remapped reliably across shape changes).

Usage:
  python tools/migrate_qkv.py --input training/checkpoints/training.pt \
                               --output training/checkpoints/training_migrated.pt
"""

import argparse
import re
import sys

import torch


def migrate(input_path, output_path):
    ckpt = torch.load(input_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
    else:
        sd = ckpt
        ckpt = {"model_state_dict": sd}

    # Check if migration is needed
    q_keys = [k for k in sd if ".attn.q_proj." in k]
    qkv_keys = [k for k in sd if ".attn.qkv_proj." in k]

    if not q_keys and qkv_keys:
        print("Checkpoint already uses qkv_proj — no migration needed.")
        return

    if not q_keys:
        print("No q_proj keys found — is this a ViT checkpoint?")
        sys.exit(1)

    # Find all block prefixes that need migration
    prefixes = sorted(set(re.match(r"(blocks\.\d+\.attn\.)", k).group(1)
                          for k in q_keys))

    new_sd = {}
    migrated = 0

    for key, val in sd.items():
        # Skip old q_proj/kv_proj — we'll add qkv_proj instead
        if ".attn.q_proj." in key or ".attn.kv_proj." in key:
            continue
        new_sd[key] = val

    for prefix in prefixes:
        q_w = sd[prefix + "q_proj.weight"]
        q_b = sd[prefix + "q_proj.bias"]
        kv_w = sd[prefix + "kv_proj.weight"]
        kv_b = sd[prefix + "kv_proj.bias"]

        # qkv_proj = [Q | K | V] where kv_proj was [K | V] already
        new_sd[prefix + "qkv_proj.weight"] = torch.cat([q_w, kv_w], dim=0)
        new_sd[prefix + "qkv_proj.bias"] = torch.cat([q_b, kv_b], dim=0)
        migrated += 1

    ckpt["model_state_dict"] = new_sd

    # Drop optimizer state — param shapes changed, old state is invalid
    if "optimizer_state_dict" in ckpt:
        del ckpt["optimizer_state_dict"]
        print("Optimizer state reset (shape mismatch after migration).")

    # Also drop group_size/scale if stored
    torch.save(ckpt, output_path)
    print(f"Migrated {migrated} blocks: q_proj + kv_proj → qkv_proj")
    print(f"Saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate ViT checkpoint from q_proj/kv_proj to qkv_proj")
    parser.add_argument("--input", required=True, help="Old checkpoint path")
    parser.add_argument("--output", required=True, help="New checkpoint path")
    args = parser.parse_args()

    if args.input == args.output:
        print("ERROR: input and output must be different paths")
        sys.exit(1)

    migrate(args.input, args.output)


if __name__ == "__main__":
    main()
