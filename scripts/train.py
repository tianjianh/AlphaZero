#!/usr/bin/env python3
"""Train the Xiangqi policy / value network.

Mirrors the multi-gpu Go trainer where it makes sense:
  * DistributedDataParallel (torchrun / LOCAL_RANK) for multi-GPU runs
  * Mixed precision (BF16 autocast on CUDA, FP16 + GradScaler fallback)
  * Bulk numpy parsing of the V4 self-play format (no per-record Python loop)
  * 3-class WLD cross-entropy on the value head
  * Training-only auxiliary heads: opponent policy, game length, end-of-game
    ownership.  These do not appear in the exported ONNX graph.
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import struct
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from torch.distributed.algorithms.join import Join
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.dirname(__file__))
from export_onnx import export_to_onnx
from model import create_model


MAGIC = 0x4D47
VERSION = 4
HEADER_STRUCT = struct.Struct("<HHiii")         # magic, version, count, rows, cols
HEADER_BYTES = HEADER_STRUCT.size               # 16

FP32 = np.float32
INT32 = np.int32


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
    if filepath.endswith(".gz"):
        import gzip
        with gzip.open(filepath, "rb") as f:
            return f.read()
    with open(filepath, "rb") as f:
        return f.read()


def _read_header(data: bytes):
    if len(data) < HEADER_BYTES:
        raise ValueError("Self-play file too short (no header)")
    magic, version, count, rows, cols = HEADER_STRUCT.unpack_from(data, 0)
    if magic != MAGIC or version != VERSION:
        raise ValueError(
            f"Unsupported self-play file format: magic=0x{magic:04x} version={version}"
        )
    return count, rows, cols


def _parse_file_bulk(data: bytes, board_rows: int, board_cols: int, input_channels: int):
    """Bulk numpy parser for V4 self-play records.

    Record layout (all little-endian):
        state_size:i32  state:f32*state_size
        policy_size:i32 policy:f32*policy_size
        value:f32
        ownership:f32*(rows*cols)   — terminal-board, current player's POV
        opponent_action:i32
    """
    count, rows, cols = _read_header(data)
    if rows != board_rows or cols != board_cols:
        raise ValueError(
            f"Data board {rows}x{cols} does not match trainer board "
            f"{board_rows}x{board_cols}"
        )
    if count == 0:
        return None

    hw = rows * cols
    state_size = input_channels * hw
    action_size = hw * hw
    record_bytes = (
        4 + state_size * 4
        + 4 + action_size * 4
        + 4                      # value (f32)
        + hw * 4                 # ownership
        + 4                      # opponent_action (i32)
    )

    payload = data[HEADER_BYTES:]
    expected = count * record_bytes
    if len(payload) < expected:
        raise ValueError(
            f"Self-play file truncated: expected {expected} bytes, got {len(payload)}"
        )

    arr = np.frombuffer(payload, dtype=np.uint8, count=expected).reshape(count, record_bytes)

    cursor = 0
    cursor += 4  # state_size prefix
    states = np.frombuffer(
        arr[:, cursor:cursor + state_size * 4].tobytes(),
        dtype=FP32,
    ).reshape(count, input_channels, rows, cols).copy()
    cursor += state_size * 4

    cursor += 4  # policy_size prefix
    policies = np.frombuffer(
        arr[:, cursor:cursor + action_size * 4].tobytes(),
        dtype=FP32,
    ).reshape(count, action_size).copy()
    cursor += action_size * 4

    values = np.frombuffer(
        arr[:, cursor:cursor + 4].tobytes(),
        dtype=FP32,
    ).reshape(count).copy()
    cursor += 4

    ownerships = np.frombuffer(
        arr[:, cursor:cursor + hw * 4].tobytes(),
        dtype=FP32,
    ).reshape(count, hw).copy()
    cursor += hw * 4

    opp_actions = np.frombuffer(
        arr[:, cursor:cursor + 4].tobytes(),
        dtype=INT32,
    ).reshape(count).copy()
    cursor += 4

    assert cursor == record_bytes
    return states, policies, values, ownerships, opp_actions


class SelfPlayDataset(IterableDataset):
    def __init__(self, files, board_rows, board_cols, input_channels,
                 rank: int = 0, world_size: int = 1):
        self.files = list(files)
        self.board_rows = board_rows
        self.board_cols = board_cols
        self.input_channels = input_channels
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.epoch * 997 + self.rank)
        files = list(self.files)
        # Deterministic shuffle, then rank-partition so DDP ranks read disjoint files.
        random.Random(self.epoch).shuffle(files)
        if self.world_size > 1:
            files = files[self.rank::self.world_size]

        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            files = files[worker_info.id::worker_info.num_workers]

        for filepath in files:
            try:
                data = _decompress(filepath)
                parsed = _parse_file_bulk(
                    data, self.board_rows, self.board_cols, self.input_channels)
            except Exception as exc:
                print(f"[warn] skipping {filepath}: {exc}", flush=True)
                continue
            if parsed is None:
                continue
            states, policies, values, ownerships, opp_actions = parsed
            n = len(values)
            # Records are (original, mirror) pairs per move — the move index
            # of record i is i // 2.  Remaining moves from that position is
            # therefore (last_move_idx - move_idx), in MOVE units.
            total_moves = n // 2
            move_idx = np.arange(n, dtype=np.float32) // 2
            remaining_plies = (total_moves - 1) - move_idx
            remaining_plies = np.clip(remaining_plies, 0.0, None).astype(np.float32)
            idx = np.arange(n)
            rng.shuffle(idx)
            for i in idx:
                yield (
                    torch.from_numpy(states[i]),
                    torch.from_numpy(policies[i]),
                    torch.tensor(values[i], dtype=torch.float32),
                    torch.from_numpy(ownerships[i]),
                    torch.tensor(opp_actions[i], dtype=torch.long),
                    torch.tensor(remaining_plies[i], dtype=torch.float32),
                )


def soft_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


def wdl_target_from_value(values: torch.Tensor) -> torch.Tensor:
    """Map scalar game outcome in {-1, 0, +1} to a WDL class index.

    0 = win (+1), 1 = draw (0), 2 = loss (-1).  Values outside these exact
    targets are bucketed by sign.
    """
    # torch.where returns long via indexing; compose with sign.
    sign = torch.sign(values).long()        # -1, 0, +1
    # map:  +1 -> 0, 0 -> 1, -1 -> 2
    return 1 - sign


def is_main_process(rank: int) -> bool:
    return rank == 0


def main():
    parser = argparse.ArgumentParser(description="Train the Xiangqi policy/value network")
    parser.add_argument("--data", default="training/selfplay",
                        help="Comma-separated self-play data directories")
    parser.add_argument("--checkpoint", default="training/checkpoints/training.pt",
                        help="Checkpoint file to resume/save")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--rows", type=int, default=10)
    parser.add_argument("--cols", type=int, default=9)
    parser.add_argument("--history-length", type=int, default=4)
    parser.add_argument("--filters", type=int, default=128)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=1.0)
    parser.add_argument("--opp-policy-weight", type=float, default=0.25,
                        help="Loss weight on the opponent-policy aux head")
    parser.add_argument("--length-weight", type=float, default=0.10,
                        help="Loss weight on the remaining-plies aux head")
    parser.add_argument("--ownership-weight", type=float, default=0.10,
                        help="Loss weight on the end-of-game ownership aux head")
    parser.add_argument("--length-scale", type=float, default=50.0,
                        help="Ply normalization for tanh length target")
    parser.add_argument("--output-onnx", default="models/model.onnx")
    parser.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto",
                        help="Mixed precision mode (auto picks bf16 when the GPU supports it)")
    args = parser.parse_args()

    input_channels = args.history_length * 14 + 1

    # ── Distributed setup ───────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp_enabled = world_size > 1

    if ddp_enabled:
        if not torch.cuda.is_available():
            raise SystemExit("DDP requested but CUDA is not available")
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
    else:
        rank = 0
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    if is_main_process(rank):
        print(f"Device: {device}  world_size={world_size}  local_rank={local_rank}")

    # ── Model ───────────────────────────────────────────────────
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
        map_location = f"cuda:{local_rank}" if ddp_enabled else device
        ckpt = torch.load(args.checkpoint, map_location=map_location, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if is_main_process(rank):
            if missing:
                print(f"[ckpt] missing keys: {len(missing)}")
            if unexpected:
                print(f"[ckpt] unexpected keys: {len(unexpected)}")
            if "optimizer_state_dict" in ckpt:
                try:
                    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                except (ValueError, RuntimeError) as exc:
                    print(f"[ckpt] optimizer state not reused: {exc}")
            print(f"Resumed from {args.checkpoint}")

    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank])

    # ── Data ────────────────────────────────────────────────────
    data_dirs = [d.strip() for d in args.data.split(",") if d.strip()]
    files = _find_data_files(data_dirs)
    if not files:
        if is_main_process(rank):
            print("No self-play files found", flush=True)
        if ddp_enabled:
            dist.destroy_process_group()
        return

    dataset = SelfPlayDataset(
        files, args.rows, args.cols, input_channels,
        rank=rank, world_size=world_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # ── AMP config ──────────────────────────────────────────────
    # torch.cuda.is_bf16_supported() returns True on Turing (SM 7.5,
    # e.g. RTX 2080 Ti) because BF16 is emulated through the FP32 path
    # on pre-Ampere GPUs — SM 7.x has FP16 tensor cores but no native
    # BF16 tensor cores.  Emulated BF16 keeps the SMs busy (hence "GPU
    # 100%") while the tensor cores stay idle, which shows up as low
    # power draw.  Prefer FP16 on Turing to actually use the tensor
    # cores; BF16 on Ampere+ (SM 8.0+) where it's native.
    if args.amp == "auto":
        if device.type == "cuda":
            dev_idx = local_rank if ddp_enabled else 0
            major = torch.cuda.get_device_capability(dev_idx)[0]
            amp_mode = "bf16" if major >= 8 else "fp16"
        else:
            amp_mode = "off"
    else:
        amp_mode = args.amp

    scaler = None
    if amp_mode == "fp16" and device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")

    if is_main_process(rank):
        print(f"AMP mode: {amp_mode}")

    os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)

    # ── Training loop ───────────────────────────────────────────
    action_size = args.rows * args.cols * args.rows * args.cols
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        model.train()
        t0 = time.time()
        total_loss = total_policy = total_value = 0.0
        total_opp = total_length = total_own = 0.0
        opp_batches = 0
        batches = 0

        # DDP ranks see unequal record counts per epoch because the
        # IterableDataset partitions *files* (not records), and games
        # vary in length.  Join lets ranks that run out of data keep
        # participating in DDP collectives with shadow forwards until
        # every rank is done, avoiding the mismatched-collective hang
        # (rank 0 end-of-epoch all_reduce vs rank 1 mid-batch broadcast).
        join_ctx = Join([model]) if ddp_enabled else _null_context()

        with join_ctx:
            for states, policies, values, ownerships, opp_actions, rem_plies in loader:
                states = states.to(device, non_blocking=True)
                policies = policies.to(device, non_blocking=True)
                values = values.to(device, non_blocking=True).view(-1)
                ownerships = ownerships.to(device, non_blocking=True)
                opp_actions = opp_actions.to(device, non_blocking=True).view(-1)
                rem_plies = rem_plies.to(device, non_blocking=True).view(-1)
                wdl_target = wdl_target_from_value(values)

                optimizer.zero_grad(set_to_none=True)

                if amp_mode == "bf16":
                    ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
                elif amp_mode == "fp16":
                    ctx = torch.amp.autocast("cuda", dtype=torch.float16)
                else:
                    ctx = _null_context()

                with ctx:
                    pred_policy, pred_wdl, pred_opp, pred_len, pred_own = model(states)
                    policy_loss = soft_cross_entropy(pred_policy, policies)
                    value_loss = F.cross_entropy(pred_wdl, wdl_target)

                    # Opp-policy: masked cross-entropy (skip terminal records
                    # where no opponent response exists).
                    opp_mask = opp_actions.ge(0) & opp_actions.lt(action_size)
                    n_opp = int(opp_mask.sum().item())
                    if n_opp > 0:
                        opp_targets = opp_actions.clamp(min=0, max=action_size - 1)
                        opp_ce = F.cross_entropy(
                            pred_opp, opp_targets, reduction="none")
                        opp_loss = (opp_ce * opp_mask.float()).sum() / n_opp
                    else:
                        opp_loss = pred_opp.sum() * 0.0

                    # Length: Huber on tanh-squashed remaining plies.
                    length_target = torch.tanh(rem_plies / args.length_scale)
                    length_loss = F.smooth_l1_loss(
                        pred_len.view(-1), length_target)

                    # Ownership: MSE per square.
                    ownership_loss = F.mse_loss(pred_own, ownerships)

                    loss = (
                        args.policy_weight * policy_loss
                        + args.value_weight * value_loss
                        + args.opp_policy_weight * opp_loss
                        + args.length_weight * length_loss
                        + args.ownership_weight * ownership_loss
                    )

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_opp += float(opp_loss.item())
                total_length += float(length_loss.item())
                total_own += float(ownership_loss.item())
                if n_opp > 0:
                    opp_batches += 1
                batches += 1

        if is_main_process(rank):
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch + 1}/{args.epochs} "
                f"loss={total_loss / max(1, batches):.4f} "
                f"policy={total_policy / max(1, batches):.4f} "
                f"value={total_value / max(1, batches):.4f} "
                f"opp={total_opp / max(1, opp_batches):.4f} "
                f"length={total_length / max(1, batches):.4f} "
                f"own={total_own / max(1, batches):.4f} "
                f"batches={batches} "
                f"time={elapsed:.1f}s",
                flush=True,
            )

            raw_model = model.module if ddp_enabled else model
            torch.save(
                {
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "board_rows": args.rows,
                    "board_cols": args.cols,
                    "history_length": args.history_length,
                    "filters": args.filters,
                    "blocks": args.blocks,
                },
                args.checkpoint,
            )

    if is_main_process(rank):
        raw_model = model.module if ddp_enabled else model
        raw_model_cpu = raw_model.to("cpu")
        export_to_onnx(
            raw_model_cpu,
            args.output_onnx,
            board_rows=args.rows,
            board_cols=args.cols,
            input_channels=input_channels,
        )

    if ddp_enabled:
        dist.barrier()
        dist.destroy_process_group()


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
