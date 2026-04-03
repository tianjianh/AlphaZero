#!/usr/bin/env python3
"""
MiniGo AlphaZero — Training Script

Reads binary self-play data from C++ and trains the PyTorch model.
Uses memory-mapped I/O + DataLoader for efficient streaming from disk.
Supports multi-GPU via DistributedDataParallel (launched with torchrun).

Single GPU:   python train.py --data ../training/selfplay --epochs 15
Multi GPU:    torchrun --nproc_per_node=2 train.py --data ../training/selfplay --epochs 15
"""

import argparse
import bisect
import glob
import mmap
import os
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))
from model import AlphaZeroNet


# ═══════════════════════════════════════════════════════════
# Memory-mapped dataset — zero-copy random access to .bin files
# ═══════════════════════════════════════════════════════════

class SelfPlayDataset(Dataset):
    """Lazily reads self-play records from memory-mapped .bin files.

    Each .bin file has a fixed-size header (4 bytes: num_records) followed
    by num_records fixed-size records.  Record size is determined by the
    board size (state=17*H*W floats, policy=H*W+1 floats, value=1 float).

    Files are memory-mapped on first access per worker process.  The OS
    page cache handles hot/cold data automatically — no explicit buffer.
    """

    def __init__(self, data_dirs, board_size=9, input_channels=17):
        self.board_size = board_size
        self.input_channels = input_channels
        state_floats = input_channels * board_size * board_size
        policy_floats = board_size * board_size + 1

        # Fixed record size: state_size(4) + state + policy_size(4) + policy + value(4)
        self.record_bytes = 4 + state_floats * 4 + 4 + policy_floats * 4 + 4
        self.state_floats = state_floats
        self.policy_floats = policy_floats

        # Build index: scan all .bin files, read only the 4-byte header
        self.files = []
        self.cumulative = [0]
        for d in data_dirs:
            if not os.path.isdir(d):
                continue
            for f in sorted(glob.glob(os.path.join(d, "**", "*.bin"), recursive=True)):
                try:
                    with open(f, "rb") as fh:
                        n = struct.unpack("i", fh.read(4))[0]
                    if n > 0:
                        self.files.append(os.path.abspath(f))
                        self.cumulative.append(self.cumulative[-1] + n)
                except (OSError, struct.error):
                    continue

        self.total = self.cumulative[-1]
        self._mmaps = {}  # per-worker mmap cache (lazy, fork-safe)

    def __len__(self):
        return self.total

    def _get_mmap(self, filepath):
        """Get or create a read-only mmap for a file (per-worker cache)."""
        if filepath not in self._mmaps:
            f = open(filepath, "rb")
            self._mmaps[filepath] = (f, mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ))
        return self._mmaps[filepath][1]

    def __getitem__(self, idx):
        # Binary search: which file contains this record?
        file_idx = bisect.bisect_right(self.cumulative, idx) - 1
        local_idx = idx - self.cumulative[file_idx]

        mm = self._get_mmap(self.files[file_idx])
        offset = 4 + local_idx * self.record_bytes  # skip 4-byte header

        # Parse record: [state_size:i32] [state:f32*N] [policy_size:i32] [policy:f32*M] [value:f32]
        pos = offset
        pos += 4  # skip state_size (known)
        state = np.frombuffer(mm, dtype=np.float32, count=self.state_floats, offset=pos).copy()
        pos += self.state_floats * 4
        pos += 4  # skip policy_size (known)
        policy = np.frombuffer(mm, dtype=np.float32, count=self.policy_floats, offset=pos).copy()
        pos += self.policy_floats * 4
        value = struct.unpack_from("f", mm, pos)[0]

        state = state.reshape(self.input_channels, self.board_size, self.board_size)
        return (torch.from_numpy(state),
                torch.from_numpy(policy),
                torch.tensor(value, dtype=torch.float32))


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train model on C++ self-play data")
    parser.add_argument("--data", default="training/selfplay",
                        help="Data directories (comma-separated for multiple)")
    parser.add_argument("--checkpoint", default="training/checkpoints/training.pt",
                        help="Model checkpoint to load/save")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--board", type=int, default=9)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker processes (default: 4)")
    parser.add_argument("--output-onnx", default="models/model.onnx",
                        help="Output ONNX model path")
    parser.add_argument("--log-file", default=None,
                        help="Append structured metrics to this file")
    args = parser.parse_args()

    # ── DDP setup ──────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    use_ddp = local_rank >= 0 and torch.cuda.is_available()

    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    is_main = (rank == 0)

    log_file = None
    if args.log_file and is_main:
        log_file = open(args.log_file, "a")

    def tlog(msg):
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    def mprint(*a, **kw):
        if is_main:
            print(*a, **kw)

    if use_ddp:
        mprint(f"DDP: {world_size} GPUs (rank {rank}, device cuda:{local_rank})")
        tlog(f"    DDP: {world_size} GPUs")
    else:
        mprint(f"Device: {device}")
        tlog(f"    Device: {device}")

    # ── Model + optimizer ──────────────────────────────────
    model = AlphaZeroNet(
        board_size=args.board,
        input_channels=17,
        num_filters=args.filters,
        num_res_blocks=args.blocks,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)

    start_iteration = 0
    if is_main:
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_iteration = ckpt.get("iteration", 0)
            mprint(f"Resumed from iteration {start_iteration}")
        else:
            model.load_state_dict(ckpt)
            mprint("Loaded legacy checkpoint")
    else:
        mprint("Starting from scratch")

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Dataset + DataLoader ───────────────────────────────
    data_dirs = [d.strip() for d in args.data.split(",")]
    t_index = time.time()
    dataset = SelfPlayDataset(data_dirs, board_size=args.board)
    index_time = time.time() - t_index

    if len(dataset) == 0:
        mprint("No training data found.")
        tlog("    ERROR: no training data found")
        if log_file:
            log_file.close()
        if use_ddp:
            dist.destroy_process_group()
        return

    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size,
                                     rank=rank, shuffle=True)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    n_batches = len(loader)
    samples_per_gpu = len(dataset) // world_size if world_size > 1 else len(dataset)

    mprint(f"Dataset: {len(dataset)} samples from {len(dataset.files)} files "
           f"({index_time:.1f}s index)")
    mprint(f"DataLoader: {n_batches} batches/epoch, batch={args.batch_size}, "
           f"{args.num_workers} workers"
           + (f", {world_size} GPUs ({samples_per_gpu} samples/GPU)" if world_size > 1 else ""))

    tlog(f"    Samples:       {len(dataset)} from {len(dataset.files)} files")
    tlog(f"    Batches/epoch: {n_batches}" +
         (f" x {world_size} GPUs" if world_size > 1 else ""))
    tlog(f"    Index time:    {index_time:.1f}s")

    # ── Train ──────────────────────────────────────────────
    mprint(f"\nTraining: {args.epochs} epochs, lr={args.lr}")
    mprint("-" * 60)

    t_train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if use_ddp:
            sampler.set_epoch(epoch)  # reshuffle per epoch

        model.train()
        total_loss = total_pl = total_vl = 0.0
        n = 0
        t0 = time.time()

        for states, policies, values in loader:
            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)

            logits, pred_value = model(states)

            policy_loss = -torch.sum(
                policies * torch.log_softmax(logits, dim=1)
            ) / states.size(0)
            value_loss = nn.functional.mse_loss(pred_value, values)
            loss = policy_loss + value_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            n += 1

        dt = time.time() - t0
        avg_loss = total_loss / max(n, 1)
        avg_pl = total_pl / max(n, 1)
        avg_vl = total_vl / max(n, 1)
        mprint(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
               f"policy={avg_pl:.4f}  value={avg_vl:.4f}  ({dt:.1f}s)")
        tlog(f"    Epoch {epoch:3d}/{args.epochs}  "
             f"loss={avg_loss:.4f}  policy={avg_pl:.4f}  value={avg_vl:.4f}  {dt:.1f}s")

    train_time = time.time() - t_train_start
    mprint(f"Training complete ({train_time:.1f}s)")
    tlog(f"    Training time: {train_time:.1f}s")

    # ── Save & export (main process only) ──────────────────
    if is_main:
        base_model = model.module if use_ddp else model

        checkpoint = {
            "model_state_dict": base_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": start_iteration + 1,
            "board_size": args.board,
            "num_filters": args.filters,
            "num_res_blocks": args.blocks,
        }
        torch.save(checkpoint, args.checkpoint)
        print(f"Saved checkpoint: {args.checkpoint}")

        print("Exporting ONNX model...")
        from export_onnx import export_to_onnx
        model_cpu = base_model.cpu()
        export_to_onnx(model_cpu, args.output_onnx, board_size=args.board)
        tlog(f"    Exported: {args.output_onnx}")

    if log_file:
        log_file.close()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
