#!/usr/bin/env python3
"""
MiniGo AlphaZero — Training Script

Reads binary self-play data from C++ and trains the PyTorch model.
Supports gzip-compressed .bin.gz files (streamed, ~100x smaller on disk).
Multi-GPU via DistributedDataParallel (launched with torchrun).

Single GPU:   python train.py --data ../training/selfplay --epochs 15
Multi GPU:    torchrun --nproc_per_node=2 train.py --data ../training/selfplay --epochs 15
"""

import argparse
import bisect
import glob
import gzip
import mmap
import os
import random
import struct
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(__file__))
from model import AlphaZeroNet


# ═══════════════════════════════════════════════════════════
# Dataset — supports both .bin (mmap) and .bin.gz (streaming)
# ═══════════════════════════════════════════════════════════

def _find_data_files(data_dirs):
    """Find all .bin and .bin.gz files across data directories."""
    files = []
    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        files.extend(sorted(glob.glob(os.path.join(d, "**", "*.bin"), recursive=True)))
        files.extend(sorted(glob.glob(os.path.join(d, "**", "*.bin.gz"), recursive=True)))
    return files


def _read_header(filepath):
    """Read the 4-byte record count header from a .bin or .bin.gz file."""
    try:
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rb") as f:
                return struct.unpack("i", f.read(4))[0]
        else:
            with open(filepath, "rb") as f:
                return struct.unpack("i", f.read(4))[0]
    except (OSError, struct.error):
        return 0


def _parse_records(data, board_size, input_channels=17):
    """Parse all records from raw bytes (after decompression)."""
    state_floats = input_channels * board_size * board_size
    policy_floats = board_size * board_size + 1
    record_bytes = 4 + state_floats * 4 + 4 + policy_floats * 4 + 4

    n = struct.unpack_from("i", data, 0)[0]
    records = []
    for i in range(n):
        off = 4 + i * record_bytes
        pos = off + 4  # skip state_size
        state = np.frombuffer(data, np.float32, state_floats, pos).copy()
        pos += state_floats * 4 + 4  # skip policy_size
        policy = np.frombuffer(data, np.float32, policy_floats, pos).copy()
        pos += policy_floats * 4
        value = struct.unpack_from("f", data, pos)[0]
        records.append((state.reshape(input_channels, board_size, board_size),
                        policy, value))
    return records


class MmapDataset(Dataset):
    """Map-style dataset for uncompressed .bin files via memory-mapped I/O.
    Supports random access — used with DistributedSampler for DDP."""

    def __init__(self, files, board_size=9, input_channels=17):
        self.board_size = board_size
        self.input_channels = input_channels
        state_floats = input_channels * board_size * board_size
        policy_floats = board_size * board_size + 1
        self.record_bytes = 4 + state_floats * 4 + 4 + policy_floats * 4 + 4
        self.state_floats = state_floats
        self.policy_floats = policy_floats

        self.files = []
        self.cumulative = [0]
        for f in files:
            n = _read_header(f)
            if n > 0:
                self.files.append(os.path.abspath(f))
                self.cumulative.append(self.cumulative[-1] + n)
        self.total = self.cumulative[-1]
        self._mmaps = {}

    def __len__(self):
        return self.total

    def _get_mmap(self, filepath):
        if filepath not in self._mmaps:
            fh = open(filepath, "rb")
            self._mmaps[filepath] = (fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ))
        return self._mmaps[filepath][1]

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative, idx) - 1
        local_idx = idx - self.cumulative[file_idx]
        mm = self._get_mmap(self.files[file_idx])
        off = 4 + local_idx * self.record_bytes

        pos = off + 4
        state = np.frombuffer(mm, np.float32, self.state_floats, pos).copy()
        pos += self.state_floats * 4 + 4
        policy = np.frombuffer(mm, np.float32, self.policy_floats, pos).copy()
        pos += self.policy_floats * 4
        value = struct.unpack_from("f", mm, pos)[0]

        return (torch.from_numpy(state.reshape(self.input_channels, self.board_size, self.board_size)),
                torch.from_numpy(policy),
                torch.tensor(value, dtype=torch.float32))


class StreamingDataset(IterableDataset):
    """Iterable dataset for .bin.gz files — decompresses one file at a time.
    Memory = 1 decompressed file per worker (~10MB). File-level + within-file
    shuffling. For DDP, files are partitioned across ranks."""

    def __init__(self, files, board_size=9, input_channels=17,
                 rank=0, world_size=1, epoch=0):
        self.board_size = board_size
        self.input_channels = input_channels
        self.rank = rank
        self.world_size = world_size
        self.epoch = epoch

        # Partition files across DDP ranks
        self.all_files = list(files)
        self.files = self.all_files[rank::world_size]

        # Count total records (read 4-byte headers only)
        self.total = sum(_read_header(f) for f in self.all_files)
        self.rank_total = sum(_read_header(f) for f in self.files)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.rank_total

    def __iter__(self):
        # Deterministic shuffle (same seed = same file order for reproducibility)
        rng = random.Random(self.epoch * 1000 + self.rank)
        files = list(self.files)
        rng.shuffle(files)

        # Split files across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            files = files[worker_info.id::worker_info.num_workers]

        for filepath in files:
            # Decompress one file at a time (~100KB → ~10MB in memory)
            if filepath.endswith(".gz"):
                with gzip.open(filepath, "rb") as f:
                    data = f.read()
            else:
                with open(filepath, "rb") as f:
                    data = f.read()

            records = _parse_records(data, self.board_size, self.input_channels)
            rng.shuffle(records)

            for state, policy, value in records:
                yield (torch.from_numpy(state),
                       torch.from_numpy(policy),
                       torch.tensor(value, dtype=torch.float32))


def create_dataset(data_dirs, board_size, rank=0, world_size=1):
    """Auto-select dataset type based on file format.
    .bin files → MmapDataset (random access, supports DistributedSampler)
    .bin.gz files → StreamingDataset (streaming, memory-efficient)
    """
    files = _find_data_files(data_dirs)
    if not files:
        return None, 0

    has_gz = any(f.endswith(".gz") for f in files)
    total = sum(_read_header(f) for f in files)

    if has_gz:
        ds = StreamingDataset(files, board_size, rank=rank, world_size=world_size)
    else:
        ds = MmapDataset(files, board_size)

    return ds, total


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train model on C++ self-play data")
    parser.add_argument("--data", default="training/selfplay",
                        help="Data directories (comma-separated for multiple)")
    parser.add_argument("--checkpoint", default="training/checkpoints/training.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--board", type=int, default=9)
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-onnx", default="models/model.onnx")
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
        board_size=args.board, input_channels=17,
        num_filters=args.filters, num_res_blocks=args.blocks,
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
    dataset, total_samples = create_dataset(
        data_dirs, args.board, rank=rank, world_size=world_size)
    index_time = time.time() - t_index

    if dataset is None or total_samples == 0:
        mprint("No training data found.")
        tlog("    ERROR: no training data found")
        if log_file:
            log_file.close()
        if use_ddp:
            dist.destroy_process_group()
        return

    is_streaming = isinstance(dataset, StreamingDataset)

    if is_streaming:
        # StreamingDataset handles DDP partitioning internally
        sampler = None
        shuffle = False
    else:
        # MmapDataset uses DistributedSampler for DDP
        if use_ddp:
            sampler = DistributedSampler(dataset, num_replicas=world_size,
                                         rank=rank, shuffle=True)
            shuffle = False
        else:
            sampler = None
            shuffle = True

    # drop_last=False for streaming DDP: max_batches cap handles sync.
    # drop_last=True for mmap: DistributedSampler guarantees equal counts.
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=(not is_streaming),
        persistent_workers=(args.num_workers > 0 and not is_streaming),
    )

    n_files = len(dataset.files) if hasattr(dataset, "files") else 0
    fmt = "streaming (.bin.gz)" if is_streaming else "mmap (.bin)"
    mprint(f"Dataset: {total_samples} samples, {n_files} files, {fmt} ({index_time:.1f}s)")
    tlog(f"    Samples:       {total_samples} from {n_files} files ({fmt})")
    tlog(f"    Index time:    {index_time:.1f}s")

    # For streaming DDP: sync batch count across ranks to prevent hangs.
    # Each rank may have slightly different record counts from file partitioning.
    # All ranks must run the same number of batches (all-reduce requires it).
    max_batches = None
    if is_streaming and use_ddp:
        rank_records = dataset.rank_total
        rank_batches = rank_records // args.batch_size
        t = torch.tensor([rank_batches], dtype=torch.long, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        max_batches = t.item()
        mprint(f"DDP batch sync: {max_batches} batches/epoch (min across ranks)")

    # ── Train ──────────────────────────────────────────────
    mprint(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    mprint("-" * 60)

    t_train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        if is_streaming:
            dataset.set_epoch(epoch)

        model.train()
        total_loss = total_pl = total_vl = 0.0
        n = 0
        t0 = time.time()

        for states, policies, values in loader:
            if max_batches is not None and n >= max_batches:
                break

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
               f"policy={avg_pl:.4f}  value={avg_vl:.4f}  ({dt:.1f}s, {n} batches)")
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
