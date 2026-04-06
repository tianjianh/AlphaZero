#!/usr/bin/env python3
"""
MiniGo AlphaZero — Training Script

Reads zstd-compressed self-play data (.bin.zst) and trains the PyTorch model.
Streams data from disk — one file decompressed at a time, no memory limit.
Multi-GPU via DistributedDataParallel (launched with torchrun).

Single GPU:   python train.py --data ../training/selfplay --epochs 15
Multi GPU:    torchrun --nproc_per_node=2 train.py --data ../training/selfplay --epochs 15
"""

import argparse
import glob
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
from torch.utils.data import DataLoader, IterableDataset
import zstandard as zstd

sys.path.insert(0, os.path.dirname(__file__))
from model import AlphaZeroNet, GoViT, create_model


# ═══════════════════════════════════════════════════════════
# Streaming dataset — decompresses one file at a time
# ═══════════════════════════════════════════════════════════

def _find_data_files(data_dirs):
    """Find all selfplay data files (.bin.zst, .bin.gz, .bin)."""
    files = []
    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        for ext in ("*.bin.zst", "*.bin.gz", "*.bin"):
            files.extend(sorted(glob.glob(os.path.join(d, "**", ext), recursive=True)))
    return files


def _decompress(filepath):
    """Read and decompress a data file. Supports .bin.zst, .bin.gz, .bin."""
    if filepath.endswith(".zst"):
        dctx = zstd.ZstdDecompressor()
        with open(filepath, "rb") as f:
            return dctx.decompress(f.read())
    elif filepath.endswith(".gz"):
        import gzip
        with gzip.open(filepath, "rb") as f:
            return f.read()
    else:
        with open(filepath, "rb") as f:
            return f.read()


def _read_header(filepath):
    """Read the 4-byte record count from a data file header."""
    try:
        data = _decompress(filepath)
        return struct.unpack_from("i", data, 0)[0]
    except (OSError, struct.error, zstd.ZstdError):
        return 0


def _parse_file(data, board_size, input_channels=17):
    """Parse all records from raw (decompressed) bytes into tensors."""
    state_floats = input_channels * board_size * board_size
    policy_floats = board_size * board_size + 1
    record_bytes = 4 + state_floats * 4 + 4 + policy_floats * 4 + 4 + 4  # +4 for score

    n = struct.unpack_from("i", data, 0)[0]
    if n == 0:
        return []

    # Bulk parse into numpy arrays, then convert to tensors once
    states = np.empty((n, input_channels, board_size, board_size), dtype=np.float32)
    policies = np.empty((n, policy_floats), dtype=np.float32)
    values = np.empty(n, dtype=np.float32)
    scores = np.empty(n, dtype=np.float32)

    for i in range(n):
        off = 4 + i * record_bytes + 4  # skip file header + record state_size
        states[i] = np.frombuffer(data, np.float32, state_floats, off).reshape(
            input_channels, board_size, board_size)
        off += state_floats * 4 + 4  # skip policy_size
        policies[i] = np.frombuffer(data, np.float32, policy_floats, off)
        off += policy_floats * 4
        values[i] = struct.unpack_from("f", data, off)[0]
        off += 4
        scores[i] = struct.unpack_from("f", data, off)[0]

    return list(zip(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values),
        torch.from_numpy(scores),
    ))


class SelfPlayDataset(IterableDataset):
    """Streams from compressed selfplay files, decompressing one at a time.

    Memory per worker = one decompressed file (~10MB).
    File-level + within-file shuffling for good data mixing.
    For DDP, files are partitioned across ranks.
    """

    def __init__(self, files, board_size=9, rank=0, world_size=1):
        self.board_size = board_size
        self.epoch = 0

        # Partition files across DDP ranks
        self.all_files = list(files)
        self.files = self.all_files[rank::world_size]

        # Count records (decompress headers only — fast for zstd)
        self.total = sum(_read_header(f) for f in self.all_files)
        self.rank_total = sum(_read_header(f) for f in self.files)
        self.rank = rank

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.rank_total

    def __iter__(self):
        rng = random.Random(self.epoch * 1000 + self.rank)
        files = list(self.files)
        rng.shuffle(files)

        # Split files across DataLoader workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            files = files[worker_info.id::worker_info.num_workers]

        for filepath in files:
            data = _decompress(filepath)
            records = _parse_file(data, self.board_size)
            del data  # free compressed bytes immediately
            rng.shuffle(records)
            yield from records


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
    parser.add_argument("--arch", default="resnet", choices=["resnet", "vit"])
    parser.add_argument("--filters", type=int, default=64)
    parser.add_argument("--blocks", type=int, default=5)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--kv-groups", type=int, default=2)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8,
                        help="DataLoader workers for prefetching (default: 8)")
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="Loss weight for policy cross-entropy (default: 1.0)")
    parser.add_argument("--value-weight", type=float, default=1.0,
                        help="Loss weight for value BCE (default: 1.0)")
    parser.add_argument("--score-weight-loss", type=float, default=1.0,
                        help="Loss weight for score cross-entropy (default: 1.0)")
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
    model = create_model(
        arch=args.arch, board_size=args.board, input_channels=17,
        num_filters=args.filters, num_res_blocks=args.blocks,
        d_model=args.d_model, depth=args.depth, heads=args.heads,
        kv_groups=args.kv_groups, mlp_ratio=args.mlp_ratio,
    ).to(device)

    if args.arch == "vit":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

    start_iteration = 0
    if is_main:
        os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, KeyError):
                    mprint("Optimizer state mismatch (new params), reinitializing")
            start_iteration = ckpt.get("iteration", 0)
            mprint(f"Resumed from iteration {start_iteration}")
        else:
            model.load_state_dict(ckpt, strict=False)
            mprint("Loaded legacy checkpoint")
    else:
        mprint("Starting from scratch")

    # ── Mixed precision ─────────────────────────────────────
    use_amp = device.type == "cuda"
    scaler = None
    if use_amp:
        if torch.cuda.is_bf16_supported():
            amp_dtype = torch.bfloat16
            mprint("Mixed precision: BF16")
            tlog(f"    AMP: BF16")
        else:
            amp_dtype = torch.float16
            scaler = torch.amp.GradScaler()
            mprint("Mixed precision: FP16 + GradScaler")
            tlog(f"    AMP: FP16 + GradScaler")
    else:
        amp_dtype = torch.float32

    if use_ddp:
        model = DDP(model, device_ids=[local_rank])

    # ── Dataset + DataLoader ───────────────────────────────
    data_dirs = [d.strip() for d in args.data.split(",")]
    t_index = time.time()
    files = _find_data_files(data_dirs)

    if not files:
        mprint("No training data found.")
        tlog("    ERROR: no training data found")
        if log_file:
            log_file.close()
        if use_ddp:
            dist.destroy_process_group()
        return

    dataset = SelfPlayDataset(files, args.board, rank=rank, world_size=world_size)
    index_time = time.time() - t_index

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,  # max_batches cap handles DDP sync
    )

    mprint(f"Dataset: {dataset.total} samples, {len(files)} files ({index_time:.1f}s)")
    mprint(f"DataLoader: batch={args.batch_size}, {args.num_workers} workers"
           + (f", {world_size} GPUs" if world_size > 1 else ""))
    tlog(f"    Samples:       {dataset.total} from {len(files)} files")
    tlog(f"    Workers:       {args.num_workers}")
    tlog(f"    Index time:    {index_time:.1f}s")

    # DDP batch sync: all ranks agree on batch count to prevent NCCL hangs
    max_batches = None
    if use_ddp:
        rank_batches = dataset.rank_total // args.batch_size
        t = torch.tensor([rank_batches], dtype=torch.long, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        max_batches = t.item()
        mprint(f"DDP sync: {max_batches} batches/epoch")

    # ── Train ──────────────────────────────────────────────
    mprint(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    mprint("-" * 60)

    t_train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        total_loss = total_pl = total_vl = total_sl = 0.0
        n = 0
        t0 = time.time()

        board_area = args.board * args.board
        num_bins = board_area * 2 + 1

        for states, policies, values, scores in loader:
            if max_batches is not None and n >= max_batches:
                break

            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True).unsqueeze(1)
            scores = scores.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits, pred_value, pred_score = model(states)

                policy_loss = -torch.sum(
                    policies * torch.log_softmax(logits, dim=1)
                ) / states.size(0)

                # Value: BCE with logits — targets {-1,0,1} → {0, 0.5, 1}
                value_targets = (values + 1.0) / 2.0
                value_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_value, value_targets)

                # Score: cross-entropy over bins
                score_bin = (torch.round(scores) + board_area).long()
                score_bin = score_bin.clamp(0, num_bins - 1)
                score_loss = nn.functional.cross_entropy(pred_score, score_bin)

                loss = (args.policy_weight * policy_loss +
                        args.value_weight * value_loss +
                        args.score_weight_loss * score_loss)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_pl += policy_loss.item()
            total_vl += value_loss.item()
            total_sl += score_loss.item()
            n += 1

        dt = time.time() - t0
        avg_loss = total_loss / max(n, 1)
        avg_pl = total_pl / max(n, 1)
        avg_vl = total_vl / max(n, 1)
        avg_sl = total_sl / max(n, 1)
        mprint(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  "
               f"policy={avg_pl:.4f}  value={avg_vl:.4f}  score={avg_sl:.4f}  ({dt:.1f}s, {n} batches)")
        tlog(f"    Epoch {epoch:3d}/{args.epochs}  "
             f"loss={avg_loss:.4f}  policy={avg_pl:.4f}  value={avg_vl:.4f}  score={avg_sl:.4f}  {dt:.1f}s")

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
            "arch": args.arch,
            "board_size": args.board,
            "num_filters": args.filters,
            "num_res_blocks": args.blocks,
            "d_model": args.d_model,
            "depth": args.depth,
            "heads": args.heads,
            "kv_groups": args.kv_groups,
            "mlp_ratio": args.mlp_ratio,
        }
        torch.save(checkpoint, args.checkpoint)
        print(f"Saved checkpoint: {args.checkpoint}")

        print("Exporting ONNX model...")
        from export_onnx import export_to_onnx
        model_cpu = base_model.cpu()
        export_to_onnx(model_cpu, args.output_onnx, board_size=args.board, arch=args.arch)
        tlog(f"    Exported: {args.output_onnx}")

    if log_file:
        log_file.close()
    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
