#!/usr/bin/env python3
"""
MiniGo AlphaZero — Training Script

Reads zstd-compressed self-play data (.bin.zst) and trains the PyTorch model.
Streams data from disk — one file decompressed at a time, no memory limit.
Multi-GPU via DistributedDataParallel (launched with torchrun).

Seven-headed training: policy, value (W/L/D), scoreMean, scoreStdev,
ownership, scoreBelief (soft Gaussian), opponentPolicy.  All loss weights
are individually configurable via CLI.

Binary format: V2 only (0x4D47 magic header).

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
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset
import zstandard as zstd

sys.path.insert(0, os.path.dirname(__file__))
from model import AlphaZeroNet, GoViT, create_model

# Optional: NVIDIA Transformer Engine for FP8 training
try:
    import transformer_engine.pytorch as te
except ImportError:
    te = None


# ═══════════════════════════════════════════════════════════
# Streaming dataset — V2 binary format (0x4D47 magic)
# ═══════════════════════════════════════════════════════════

_V2_MAGIC = 0x4D47  # 'MG'

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
    """Read record count from V2 header without full decompression."""
    try:
        if filepath.endswith(".zst"):
            dctx = zstd.ZstdDecompressor()
            with open(filepath, "rb") as f:
                reader = dctx.stream_reader(f)
                hdr = reader.read(12)  # magic(2) + version(2) + count(4) + board_size(4)
        else:
            with open(filepath, "rb") as f:
                hdr = f.read(12)
        if len(hdr) < 12:
            return 0
        magic, version, count = struct.unpack_from("<HHi", hdr, 0)
        if magic != _V2_MAGIC:
            return 0  # not V2 — skip
        return count
    except (OSError, struct.error, zstd.ZstdError):
        return 0


def _parse_file(data, board_size, input_channels=17):
    """Parse all records from V2 binary format into tensors.

    V2 header: [magic:u16][version:u16][count:i32][board_size:i32]
    Per record: [state_size:i32][state:f32×S][policy_size:i32][policy:f32×P]
                [value:f32][score:f32][ownership:f32×board²][opponent_action:i32]
    """
    if len(data) < 12:
        return []

    magic, version, n, file_board = struct.unpack_from("<HHii", data, 0)
    if magic != _V2_MAGIC:
        raise ValueError(f"Not a V2 data file (magic=0x{magic:04X}, expected 0x4D47). "
                         "Training requires V2 format — regenerate selfplay data.")
    if n == 0:
        return []

    state_floats = input_channels * board_size * board_size
    policy_floats = board_size * board_size + 1
    ownership_floats = board_size * board_size

    # Record layout (V2):
    # [state_size:i32][state:f32×S][policy_size:i32][policy:f32×P]
    # [value:f32][score:f32][ownership:f32×B²][opponent_action:i32]
    record_bytes = (4 + state_floats * 4 + 4 + policy_floats * 4
                    + 4 + 4 + ownership_floats * 4 + 4)

    header_bytes = 12  # magic(2) + version(2) + count(4) + board_size(4)
    payload = np.frombuffer(data, dtype=np.uint8, offset=header_bytes, count=n * record_bytes)
    records = payload.reshape(n, record_bytes)

    # Byte offsets within each record
    s_off = 4
    s_end = s_off + state_floats * 4
    p_off = s_end + 4
    p_end = p_off + policy_floats * 4
    v_off = p_end
    v_end = v_off + 4
    sc_off = v_end
    sc_end = sc_off + 4
    own_off = sc_end
    own_end = own_off + ownership_floats * 4
    opp_off = own_end

    # Bulk extract
    states = np.ndarray((n, state_floats), dtype=np.float32,
                        buffer=records[:, s_off:s_end].tobytes()
                        ).reshape(n, input_channels, board_size, board_size).copy()
    policies = np.ndarray((n, policy_floats), dtype=np.float32,
                          buffer=records[:, p_off:p_end].tobytes()).copy()
    values = np.ndarray((n,), dtype=np.float32,
                        buffer=records[:, v_off:v_end].tobytes()).copy()
    scores = np.ndarray((n,), dtype=np.float32,
                        buffer=records[:, sc_off:sc_end].tobytes()).copy()
    ownerships = np.ndarray((n, ownership_floats), dtype=np.float32,
                            buffer=records[:, own_off:own_end].tobytes()).copy()
    opp_actions = np.ndarray((n,), dtype=np.int32,
                             buffer=records[:, opp_off:opp_off+4].tobytes()).copy()

    return list(zip(
        torch.from_numpy(states),
        torch.from_numpy(policies),
        torch.from_numpy(values),
        torch.from_numpy(scores),
        torch.from_numpy(ownerships),
        torch.from_numpy(opp_actions.astype(np.int64)),
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
    # --- All 7 loss weights ---
    # Loss weights.  Designed so mid-training weighted contributions are
    # balanced: policy ~3.0 (dominant), value ~1.0, auxiliaries 0.1-0.5.
    # scoreMean/Stdev weights are small because their raw MSE is in
    # points² (typical magnitude 20-100), matching KataGo's approach.
    parser.add_argument("--policy-weight", type=float, default=1.0,
                        help="Loss weight for policy CE (KataGo: 1.0)")
    parser.add_argument("--value-weight", type=float, default=1.5,
                        help="Loss weight for value CE W/L/D (KataGo: 1.5)")
    parser.add_argument("--score-mean-weight", type=float, default=0.005,
                        help="Loss weight for scoreMean MSE in points² "
                             "(~25 raw × 0.005 = 0.125 weighted)")
    parser.add_argument("--score-stdev-weight", type=float, default=0.005,
                        help="Loss weight for scoreStdev MSE in points² "
                             "(~12 raw × 0.005 = 0.06 weighted)")
    parser.add_argument("--ownership-weight", type=float, default=1.5,
                        help="Loss weight for ownership BCE per-intersection "
                             "mean (~0.3 × 1.5 = 0.45 weighted; matches KataGo)")
    parser.add_argument("--score-belief-weight", type=float, default=0.02,
                        help="Loss weight for score belief CE (soft Gaussian; "
                             "~3 × 0.02 = 0.06 weighted, matches KataGo)")
    parser.add_argument("--opp-policy-weight", type=float, default=0.1,
                        help="Loss weight for opponent policy CE "
                             "(~3 × 0.1 = 0.3 weighted)")
    parser.add_argument("--fp8", action="store_true",
                        help="Use FP8 training via NVIDIA Transformer Engine (Blackwell+)")
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
        use_fp8=args.fp8,
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

    # ── Mixed precision: FP8 > BF16 > FP16 > FP32 ──────────
    use_amp = device.type == "cuda"
    use_te_fp8 = False
    scaler = None
    if use_amp:
        sm = torch.cuda.get_device_capability()
        if args.fp8:
            try:
                import transformer_engine.pytorch as te
                use_te_fp8 = True
                amp_dtype = torch.bfloat16
                mprint(f"Mixed precision: FP8 via Transformer Engine (SM {sm[0]}.{sm[1]})")
                tlog(f"    AMP: FP8 (Transformer Engine)")
            except ImportError:
                mprint("WARNING: --fp8 requested but transformer_engine not installed, falling back")
                args.fp8 = False
        if not use_te_fp8:
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
                mprint(f"Mixed precision: BF16 (SM {sm[0]}.{sm[1]})")
                tlog(f"    AMP: BF16")
            else:
                amp_dtype = torch.float16
                scaler = torch.amp.GradScaler()
                mprint(f"Mixed precision: FP16 + GradScaler (SM {sm[0]}.{sm[1]})")
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
        drop_last=False,
        persistent_workers=args.num_workers > 0,
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

    # ── Pre-compute score belief bins ──────────────────────
    board_area = args.board * args.board
    num_bins = board_area * 2 + 1
    belief_sigma = 3.0
    bin_centers = torch.arange(num_bins, dtype=torch.float32, device=device) - board_area

    # ── Train ──────────────────────────────────────────────
    mprint(f"\nTraining: {args.epochs} epochs, batch={args.batch_size}, lr={args.lr}")
    mprint(f"  Weights: policy={args.policy_weight} value={args.value_weight} "
           f"score_mean={args.score_mean_weight} score_stdev={args.score_stdev_weight} "
           f"own={args.ownership_weight} belief={args.score_belief_weight} "
           f"opp={args.opp_policy_weight}")
    mprint("-" * 60)

    t_train_start = time.time()
    # Value-head diagnostic bin edges (KataGo-style calibration tracking).
    # Phase bins by stone-count proxy for move number on 9x9 Go:
    #   early 0-15   — opening (position fundamentally fluid)
    #   mid  16-40   — middle game / fighting (decisive moves)
    #   late 41+     — endgame (usually decided; low-CE zone)
    # Binning is per-sample not per-game — a 100-move game contributes
    # ~100 samples spread across all phases, so even iterations with
    # many short games produce plenty of late-bin samples as long as
    # the average game length is 60+ moves (both iter-4 and iter-40
    # satisfy this).  Typical split: early ~15%, mid ~25%, late ~55%.
    phase_edges = [0, 16, 41, 999]
    # Confidence bins on max softmax prob of 3-class value (W/L/D; chance = 1/3)
    conf_edges = [0.333, 0.50, 0.70, 0.90, 1.0001]

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        total_pl = total_vl = total_sml = total_ssl = 0.0
        total_ol = total_bl = total_opl = 0.0
        n = 0
        t0 = time.time()

        # Value diagnostics: CE bucketed by game phase; calibration curve
        # (correct count + confidence sum per confidence bin).  Per-epoch
        # accumulators so the overhead is O(B) per batch, not O(B²).
        phase_ce_sum  = [0.0, 0.0, 0.0]  # early/mid/late
        phase_count   = [0,   0,   0]
        conf_correct  = [0, 0, 0, 0]     # correct predictions per bin
        conf_sum      = [0.0, 0.0, 0.0, 0.0]  # sum of max-probs per bin
        conf_count    = [0, 0, 0, 0]

        for states, policies, values, scores, ownerships, opp_actions in loader:
            if max_batches is not None and n >= max_batches:
                break

            states = states.to(device, non_blocking=True)
            policies = policies.to(device, non_blocking=True)
            values = values.to(device, non_blocking=True)
            scores = scores.to(device, non_blocking=True)
            ownerships = ownerships.to(device, non_blocking=True)
            opp_actions = opp_actions.to(device, non_blocking=True)

            # Value target: {1 → win(0), -1 → loss(1), 0 → draw(2)}
            value_target = torch.where(values > 0, 0,
                           torch.where(values < 0, 1, 2)).long()

            optimizer.zero_grad()

            if use_te_fp8:
                te_ctx = te.fp8_autocast(enabled=True)
                amp_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16)
            else:
                te_ctx = torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp)
                amp_ctx = torch.amp.autocast("cuda", enabled=False)
            with te_ctx, amp_ctx:
                (pred_policy, pred_value, pred_score_mean, pred_score_stdev,
                 pred_ownership, pred_score_belief, pred_opp_policy) = model(states)

                # 1. Policy: soft CE against MCTS visit distribution
                policy_loss = -torch.sum(
                    policies * torch.log_softmax(pred_policy, dim=1)
                ) / states.size(0)

                # 2. Value: 3-class CE (win/loss/draw)
                value_loss = F.cross_entropy(pred_value, value_target)

                # 3. ScoreMean: raw MSE in points² (weight is small to compensate,
                # matching KataGo — scale absorbed into weight, not the loss).
                score_mean_loss = F.mse_loss(pred_score_mean.squeeze(1), scores)

                # 4. ScoreStdev: raw MSE against |actual - predicted_mean|
                with torch.no_grad():
                    stdev_target = (scores - pred_score_mean.squeeze(1).detach()).abs()
                score_stdev_loss = F.mse_loss(pred_score_stdev.squeeze(1), stdev_target)

                # 5. Ownership: per-intersection BCE
                ownership_loss = F.binary_cross_entropy_with_logits(
                    pred_ownership, ownerships)

                # 6. Score Belief: CE with soft Gaussian target (from score, not stored)
                score_expanded = scores.unsqueeze(1)  # [B, 1]
                belief_logits = -0.5 * ((bin_centers.unsqueeze(0) - score_expanded) / belief_sigma) ** 2
                soft_target = F.softmax(belief_logits, dim=1)
                score_belief_loss = -(soft_target * F.log_softmax(pred_score_belief, dim=1)).sum(dim=1).mean()

                # 7. Opponent Policy: CE against actual next move (mask -1)
                opp_mask = opp_actions >= 0
                if opp_mask.any():
                    opp_policy_loss = F.cross_entropy(
                        pred_opp_policy[opp_mask], opp_actions[opp_mask])
                else:
                    opp_policy_loss = torch.tensor(0.0, device=device)

                # Total weighted loss
                loss = (args.policy_weight       * policy_loss
                      + args.value_weight        * value_loss
                      + args.score_mean_weight   * score_mean_loss
                      + args.score_stdev_weight  * score_stdev_loss
                      + args.ownership_weight    * ownership_loss
                      + args.score_belief_weight * score_belief_loss
                      + args.opp_policy_weight   * opp_policy_loss)

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
            total_sml += score_mean_loss.item()
            total_ssl += score_stdev_loss.item()
            total_ol += ownership_loss.item()
            total_bl += score_belief_loss.item()
            total_opl += opp_policy_loss.item()
            n += 1

            # ── Value-head diagnostics (cheap, no extra fwd pass) ────
            # Stone count as a proxy for game phase.  Encoding: channels
            # 0 and 8 are the most-recent current/opponent snapshots, so
            # their sum across HxW gives total stones on board.  Works
            # for any history_length the encoder uses.
            with torch.no_grad():
                stone_count = (states[:, 0] + states[:, 8]).reshape(states.size(0), -1).sum(dim=1)
                v_probs = F.softmax(pred_value.float(), dim=1)
                v_max, v_pred = v_probs.max(dim=1)
                per_sample_ce = F.cross_entropy(pred_value.float(), value_target, reduction="none")
                correct = (v_pred == value_target)

                for b, (lo, hi) in enumerate(zip(phase_edges[:-1], phase_edges[1:])):
                    m = (stone_count >= lo) & (stone_count < hi)
                    c = m.sum().item()
                    if c:
                        phase_count[b]  += c
                        phase_ce_sum[b] += per_sample_ce[m].sum().item()

                for b, (lo, hi) in enumerate(zip(conf_edges[:-1], conf_edges[1:])):
                    m = (v_max >= lo) & (v_max < hi)
                    c = m.sum().item()
                    if c:
                        conf_count[b]   += c
                        conf_sum[b]     += v_max[m].sum().item()
                        conf_correct[b] += correct[m].sum().item()

        dt = time.time() - t0
        d = max(n, 1)
        avg = lambda t: t / d
        mprint(f"  Epoch {epoch:3d}/{args.epochs}  loss={avg(total_loss):.4f}  "
               f"pol={avg(total_pl):.4f}  val={avg(total_vl):.4f}  "
               f"smn={avg(total_sml):.4f}  ssd={avg(total_ssl):.4f}  "
               f"own={avg(total_ol):.4f}  bel={avg(total_bl):.4f}  "
               f"opp={avg(total_opl):.4f}  ({dt:.1f}s, {n}b)")
        tlog(f"    Epoch {epoch:3d}/{args.epochs}  "
             f"loss={avg(total_loss):.4f}  pol={avg(total_pl):.4f}  val={avg(total_vl):.4f}  "
             f"smn={avg(total_sml):.4f}  ssd={avg(total_ssl):.4f}  own={avg(total_ol):.4f}  "
             f"bel={avg(total_bl):.4f}  opp={avg(total_opl):.4f}  {dt:.1f}s")

        # ── Value diagnostics: phase CE + calibration (ECE) ──────
        # Tells us whether low val_loss is genuinely well-calibrated
        # (scenario 1) or dominated by easy late-game positions
        # (scenario 2) or overconfident (scenario 3).  Bucket CE by
        # game phase (stone count proxies for move number); bucket
        # max-softmax-prob by confidence and compare to accuracy.
        phase_ce = [s / c if c else 0.0
                    for s, c in zip(phase_ce_sum, phase_count)]
        phase_tot = sum(phase_count) or 1
        phase_frac = [c / phase_tot for c in phase_count]

        # Expected Calibration Error:  Σ (|B|/N) × |acc_B − conf_B|
        ece = 0.0
        conf_tot = sum(conf_count) or 1
        calib_parts = []
        for b in range(len(conf_count)):
            if conf_count[b]:
                conf_b = conf_sum[b] / conf_count[b]
                acc_b  = conf_correct[b] / conf_count[b]
                ece += (conf_count[b] / conf_tot) * abs(acc_b - conf_b)
                calib_parts.append(f"{conf_edges[b]:.2f}-{conf_edges[b+1]:.2f}:"
                                   f"conf={conf_b:.3f}/acc={acc_b:.3f}"
                                   f"@{conf_count[b]/conf_tot*100:.0f}%")
            else:
                calib_parts.append(f"{conf_edges[b]:.2f}-{conf_edges[b+1]:.2f}:-")

        diag_line = (f"      val-diag  "
                     f"phase CE: early={phase_ce[0]:.3f}({phase_frac[0]*100:.0f}%) "
                     f"mid={phase_ce[1]:.3f}({phase_frac[1]*100:.0f}%) "
                     f"late={phase_ce[2]:.3f}({phase_frac[2]*100:.0f}%)  "
                     f"ECE={ece:.3f}  calib[{' '.join(calib_parts)}]")
        mprint(diag_line)
        tlog(diag_line)

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
