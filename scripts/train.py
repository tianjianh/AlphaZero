#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import random
import struct
import sys
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(__file__))
from export_onnx import export_to_onnx
from model import create_model


MAGIC = 0x4D47
VERSION = 3


def _find_data_files(data_dirs):
    files = []
    for d in data_dirs:
        if not os.path.isdir(d):
            continue
        for ext in ("*.bin.zst", "*.bin.gz", "*.bin"):
            files.extend(sorted(glob.glob(os.path.join(d, "**", ext), recursive=True)))
    return files


def _decompress(filepath):
    if filepath.endswith(".zst"):
        try:
            import zstandard as zstd
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "zstandard is required to read .zst self-play files. "
                "Install it with `pip install zstandard`."
            ) from exc
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


def _parse_file(data, board_rows, board_cols, input_channels):
    if len(data) < 16:
        return []

    magic, version, count, rows, cols = struct.unpack_from("<HHiii", data, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError("Unsupported self-play file format; expected Xiangqi V3 records")
    if rows != board_rows or cols != board_cols:
        raise ValueError(f"Data board {rows}x{cols} does not match trainer board {board_rows}x{board_cols}")

    pos = 16
    records = []
    for _ in range(count):
        state_size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        state = torch.tensor(
            struct.unpack_from(f"<{state_size}f", data, pos),
            dtype=torch.float32,
        ).view(input_channels, board_rows, board_cols)
        pos += state_size * 4

        policy_size = struct.unpack_from("<i", data, pos)[0]
        pos += 4
        policy = torch.tensor(
            struct.unpack_from(f"<{policy_size}f", data, pos),
            dtype=torch.float32,
        )
        pos += policy_size * 4

        value = torch.tensor(struct.unpack_from("<f", data, pos)[0], dtype=torch.float32)
        pos += 4

        score = struct.unpack_from("<f", data, pos)[0]
        pos += 4

        ownership_size = board_rows * board_cols
        pos += ownership_size * 4
        pos += 4  # opponent_action

        records.append((state, policy, value, score))

    return records


class SelfPlayDataset(IterableDataset):
    def __init__(self, files, board_rows, board_cols, input_channels):
        self.files = list(files)
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.input_channels = input_channels
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.epoch)
        files = list(self.files)
        rng.shuffle(files)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            files = files[worker_info.id::worker_info.num_workers]

        for filepath in files:
            data = _decompress(filepath)
            records = _parse_file(data, self.board_rows, self.board_cols, self.input_channels)
            rng.shuffle(records)
            yield from records


def soft_cross_entropy(logits, targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(targets * log_probs).sum(dim=1).mean()


def main():
    parser = argparse.ArgumentParser(description="Train the Xiangqi policy/value network")
    parser.add_argument("--data", default="training/selfplay",
                        help="Comma-separated self-play data directories")
    parser.add_argument("--checkpoint", default="training/checkpoints/training.pt",
                        help="Checkpoint file to resume/save")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Adam learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Adam weight decay")
    parser.add_argument("--rows", type=int, default=10,
                        help="Board row count")
    parser.add_argument("--cols", type=int, default=9,
                        help="Board column count")
    parser.add_argument("--history-length", type=int, default=4,
                        help="Number of board snapshots encoded into the input tensor")
    parser.add_argument("--filters", type=int, default=128,
                        help="Residual tower channel count")
    parser.add_argument("--blocks", type=int, default=10,
                        help="Residual block count")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader worker count")
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="Policy loss weight")
    parser.add_argument("--value-weight", type=float, default=1.0,
                        help="Value loss weight")
    parser.add_argument("--output-onnx", default="models/model.onnx",
                        help="ONNX export path written after training")
    args = parser.parse_args()

    input_channels = args.history_length * 14 + 1
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Device: {device}")

    model = create_model(
        board_rows=args.rows,
        board_cols=args.cols,
        input_channels=input_channels,
        num_filters=args.filters,
        num_res_blocks=args.blocks,
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {args.checkpoint}")

    data_dirs = [d.strip() for d in args.data.split(",") if d.strip()]
    files = _find_data_files(data_dirs)
    if not files:
        raise SystemExit("No self-play files found")

    dataset = SelfPlayDataset(files, args.rows, args.cols, input_channels)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type != "cpu"),
    )

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        t0 = time.time()
        total_loss = total_policy = total_value = 0.0
        batches = 0

        for states, policies, values, _scores in loader:
            states = states.to(device)
            policies = policies.to(device)
            values = values.to(device).view(-1, 1)

            pred_policy, pred_value = model(states)
            policy_loss = soft_cross_entropy(pred_policy, policies)
            value_loss = F.mse_loss(pred_value, values)
            loss = args.policy_weight * policy_loss + args.value_weight * value_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_policy += float(policy_loss.item())
            total_value += float(value_loss.item())
            batches += 1

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch + 1}/{args.epochs} "
            f"loss={total_loss / batches:.4f} "
            f"policy={total_policy / batches:.4f} "
            f"value={total_value / batches:.4f} "
            f"time={elapsed:.1f}s"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "board_rows": args.rows,
                "board_cols": args.cols,
                "history_length": args.history_length,
                "filters": args.filters,
                "blocks": args.blocks,
            },
            args.checkpoint,
        )

    model_cpu = model.to("cpu")
    export_to_onnx(
        model_cpu,
        args.output_onnx,
        board_rows=args.rows,
        board_cols=args.cols,
        input_channels=input_channels,
    )


if __name__ == "__main__":
    main()
