#!/usr/bin/env python3
"""Warm-initialize MiniGo's trunk from a KataGo b10c128 network.

Builds a fresh MiniGo model in memory, copies KataGo regular-block conv
weights into matching MiniGo SE blocks, and writes the result as a
PyTorch checkpoint and/or ONNX file. At least one of --checkpoint /
--onnx must be specified.

Workflow with run_continuous.py (replaces the random-init seed ONNX,
optionally also seeds training.pt so the train worker resumes warm):

    python scripts/run_continuous.py init --filters 128 --blocks 10 -y
    python tools/warm_init_from_katago.py \
        --katago-bin path/to/g170e-b10c128-.../model.bin.gz \
        --arch resnet --filters 128 --blocks 10 \
        --onnx       models/accepted/v000000000.onnx \
        --checkpoint training/checkpoints/training.pt
    python scripts/run_continuous.py run --filters 128 --blocks 10 [...]

Workflow with run_loop.py (single-process, simpler):

    python run_loop.py init large
    python tools/warm_init_from_katago.py \
        --katago-bin path/to/g170e-b10c128-.../model.bin.gz \
        --arch resnet --filters 128 --blocks 10 \
        --checkpoint training/checkpoints/training.pt
    python run_loop.py train

What gets transferred (with `large` preset, 128f/10b):
    - MiniGo's 5 SE residual blocks (positions 0,2,4,6,8) get their
      conv1+conv2 weights from KataGo's regular (non-gpool) blocks,
      paired closest-depth-first.
    - Everything else stays at MiniGo's default init (stem, heads,
      GPool blocks, BN stats, SE attention modules).

Why the rest is skipped:
    - Stem:    17 input channels (MG) vs 22 spatial + 19 global (KG).
    - Heads:   KataGo has additional outputs (futurepos, seki, scoring,
               miscvalue, mixture-of-Gaussians scorebelief) and a
               different policy head topology (spatial + pass via gpool).
    - GPool:   MG keeps 128 channels through the block; KG narrows the
               regular path to 96 to make room for 32 channels of gpool
               injection. The conv shapes differ.
    - BN:      KG is pre-activation BN (norm before conv); MG is
               post-activation (BN after conv). The running_mean/var
               describe different distributions, so reuse is harmful.
               PyTorch defaults (mean=0, var=1) get recomputed from the
               first training batches.
    - SE:      KG b10c128 has no SE modules; MG's stay zero-initialised
               so the SE block acts as a sigmoid(0)=0.5 identity gate.

Estimated coverage: ~5 of 10 trunk blocks → ~55% of trunk params,
~50% of total model params. Optimizer state is cleared.

Source format: this script parses KataGo's .bin.gz / .txt.gz format
directly (Python port of cpp/neuralnet/desc.cpp). No KataGo or
TensorFlow dependencies; only torch + numpy.
"""

import argparse
import gzip
import io
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

# scripts/ is a sibling of tools/ — needed for `model` and `export_onnx`
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ────────────────────────────────────────────────────────────
#  KataGo .bin.gz / .txt.gz parser
#  (Python port of KataGo/cpp/neuralnet/desc.cpp)
# ────────────────────────────────────────────────────────────


class TokenStream:
    """Whitespace-tokenized reader over a byte buffer.

    Headers are always ASCII text. Float arrays are either ASCII (in
    .txt.gz) or a `@BIN@`-prefixed little-endian f32 block (in .bin.gz).
    """

    def __init__(self, raw: bytes):
        self.buf = io.BytesIO(raw)
        self.size = len(raw)

    def _peek(self) -> bytes:
        pos = self.buf.tell()
        ch = self.buf.read(1)
        self.buf.seek(pos)
        return ch

    def read_token(self) -> str:
        # Skip leading whitespace
        while True:
            ch = self.buf.read(1)
            if not ch:
                return ""
            if not ch.isspace():
                token = bytearray(ch)
                break
        # Read until next whitespace
        while True:
            ch = self.buf.read(1)
            if not ch or ch.isspace():
                break
            token.extend(ch)
        return token.decode("ascii")

    def read_int(self) -> int:
        return int(self.read_token())

    def read_float(self) -> float:
        return float(self.read_token())

    def read_bool(self) -> bool:
        # KataGo writes bools via std::ostream which produces "0"/"1".
        return self.read_int() != 0

    def read_floats(self, n: int, binary: bool) -> np.ndarray:
        if not binary:
            arr = np.empty(n, dtype=np.float32)
            for i in range(n):
                arr[i] = self.read_float()
            return arr

        # Binary block: skip whitespace until '@', then expect "BIN@"
        skipped = 0
        while True:
            ch = self.buf.read(1)
            if not ch:
                raise ValueError("Unexpected EOF before @BIN@ marker")
            if ch == b"@":
                break
            if not ch.isspace() or skipped > 100:
                raise ValueError(
                    f"Unexpected byte {ch!r} before @BIN@ marker "
                    f"(perhaps this is a .txt.gz file? pass --txt)"
                )
            skipped += 1
        tag = self.buf.read(4)
        if tag != b"BIN@":
            raise ValueError(f"Expected b'BIN@' after '@', got {tag!r}")
        data = self.buf.read(n * 4)
        if len(data) != n * 4:
            raise ValueError(
                f"Short read: wanted {n*4} bytes, got {len(data)} "
                f"(file truncated or format mismatch)"
            )
        # KataGo writes little-endian f32; force LE in case host differs
        return np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)


# ── layer descriptors ────────────────────────────────────────


@dataclass
class ConvLayer:
    name: str
    y: int
    x: int
    in_ch: int
    out_ch: int
    weights: np.ndarray  # [out_ch, in_ch, y, x] — PyTorch-compatible

    @classmethod
    def parse(cls, r: TokenStream, binary: bool) -> "ConvLayer":
        name = r.read_token()
        y = r.read_int()
        x = r.read_int()
        in_ch = r.read_int()
        out_ch = r.read_int()
        _dy = r.read_int()
        _dx = r.read_int()
        # File order is y, x, ic, oc with oc varying fastest.
        # Reshape to (y, x, ic, oc) then transpose to (oc, ic, y, x).
        flat = r.read_floats(y * x * in_ch * out_ch, binary)
        w = flat.reshape(y, x, in_ch, out_ch).transpose(3, 2, 0, 1).copy()
        return cls(name=name, y=y, x=x, in_ch=in_ch, out_ch=out_ch, weights=w)


@dataclass
class BatchNormLayer:
    name: str
    num_channels: int
    epsilon: float
    has_scale: bool
    has_bias: bool
    mean: np.ndarray
    variance: np.ndarray
    scale: np.ndarray
    bias: np.ndarray

    @classmethod
    def parse(cls, r: TokenStream, binary: bool) -> "BatchNormLayer":
        name = r.read_token()
        num_channels = r.read_int()
        epsilon = r.read_float()
        has_scale = r.read_bool()
        has_bias = r.read_bool()
        mean = r.read_floats(num_channels, binary)
        variance = r.read_floats(num_channels, binary)
        scale = (
            r.read_floats(num_channels, binary)
            if has_scale
            else np.ones(num_channels, dtype=np.float32)
        )
        bias = (
            r.read_floats(num_channels, binary)
            if has_bias
            else np.zeros(num_channels, dtype=np.float32)
        )
        return cls(
            name=name, num_channels=num_channels, epsilon=epsilon,
            has_scale=has_scale, has_bias=has_bias,
            mean=mean, variance=variance, scale=scale, bias=bias,
        )


@dataclass
class ActivationLayer:
    name: str
    activation: str  # "relu" / "mish" / "identity"

    @classmethod
    def parse(cls, r: TokenStream, model_version: int) -> "ActivationLayer":
        name = r.read_token()
        if model_version >= 11:
            kind = r.read_token()
            mapping = {
                "ACTIVATION_IDENTITY": "identity",
                "ACTIVATION_RELU": "relu",
                "ACTIVATION_MISH": "mish",
            }
            if kind not in mapping:
                raise ValueError(f"Unknown activation: {kind}")
            return cls(name=name, activation=mapping[kind])
        return cls(name=name, activation="relu")


@dataclass
class MatMulLayer:
    name: str
    in_ch: int
    out_ch: int
    weights: np.ndarray  # [out_ch, in_ch] — PyTorch-compatible

    @classmethod
    def parse(cls, r: TokenStream, binary: bool) -> "MatMulLayer":
        name = r.read_token()
        in_ch = r.read_int()
        out_ch = r.read_int()
        flat = r.read_floats(in_ch * out_ch, binary)
        # File order is ic, oc with oc varying fastest.
        w = flat.reshape(in_ch, out_ch).T.copy()
        return cls(name=name, in_ch=in_ch, out_ch=out_ch, weights=w)


@dataclass
class MatBiasLayer:
    name: str
    num_channels: int
    weights: np.ndarray

    @classmethod
    def parse(cls, r: TokenStream, binary: bool) -> "MatBiasLayer":
        name = r.read_token()
        num_channels = r.read_int()
        weights = r.read_floats(num_channels, binary)
        return cls(name=name, num_channels=num_channels, weights=weights)


@dataclass
class ResidualBlock:
    name: str
    pre_bn: BatchNormLayer
    pre_act: ActivationLayer
    regular_conv: ConvLayer
    mid_bn: BatchNormLayer
    mid_act: ActivationLayer
    final_conv: ConvLayer

    @classmethod
    def parse(cls, r: TokenStream, model_version: int, binary: bool) -> "ResidualBlock":
        name = r.read_token()
        return cls(
            name=name,
            pre_bn=BatchNormLayer.parse(r, binary),
            pre_act=ActivationLayer.parse(r, model_version),
            regular_conv=ConvLayer.parse(r, binary),
            mid_bn=BatchNormLayer.parse(r, binary),
            mid_act=ActivationLayer.parse(r, model_version),
            final_conv=ConvLayer.parse(r, binary),
        )


@dataclass
class GPoolBlock:
    name: str
    pre_bn: BatchNormLayer
    pre_act: ActivationLayer
    regular_conv: ConvLayer
    gpool_conv: ConvLayer
    gpool_bn: BatchNormLayer
    gpool_act: ActivationLayer
    gpool_to_bias_mul: MatMulLayer
    mid_bn: BatchNormLayer
    mid_act: ActivationLayer
    final_conv: ConvLayer

    @classmethod
    def parse(cls, r: TokenStream, model_version: int, binary: bool) -> "GPoolBlock":
        name = r.read_token()
        return cls(
            name=name,
            pre_bn=BatchNormLayer.parse(r, binary),
            pre_act=ActivationLayer.parse(r, model_version),
            regular_conv=ConvLayer.parse(r, binary),
            gpool_conv=ConvLayer.parse(r, binary),
            gpool_bn=BatchNormLayer.parse(r, binary),
            gpool_act=ActivationLayer.parse(r, model_version),
            gpool_to_bias_mul=MatMulLayer.parse(r, binary),
            mid_bn=BatchNormLayer.parse(r, binary),
            mid_act=ActivationLayer.parse(r, model_version),
            final_conv=ConvLayer.parse(r, binary),
        )


@dataclass
class Trunk:
    name: str
    model_version: int
    num_blocks: int
    trunk_num_channels: int
    mid_num_channels: int
    regular_num_channels: int
    gpool_num_channels: int
    initial_conv: ConvLayer
    initial_matmul: MatMulLayer
    blocks: List[Union[ResidualBlock, GPoolBlock]]
    block_kinds: List[str]  # 'regular' / 'gpool'


@dataclass
class KataGoModel:
    name: str
    model_version: int
    num_input_channels: int
    num_input_global_channels: int
    trunk: Trunk


def _parse_model_header(r: TokenStream) -> tuple:
    name = r.read_token()
    model_version = r.read_int()
    if model_version < 8:
        raise ValueError(
            f"Model version {model_version} is older than g170-era; "
            f"this script targets v8-v15 (script tested against g170 b10c128)."
        )
    if model_version > 15:
        print(
            f"WARN: model version {model_version} is newer than what this "
            f"script was written for (v15). Proceeding optimistically."
        )

    num_input_channels = r.read_int()
    num_input_global_channels = r.read_int()

    if model_version >= 13:
        # postProcessParams: 7 floats we don't need
        for _ in range(7):
            r.read_float()

    if model_version >= 15:
        meta_enc = r.read_int()
        for _ in range(7):
            r.read_int()
        if meta_enc != 0:
            raise NotImplementedError(
                "Metadata encoder is present but not supported by this script"
            )

    return name, model_version, num_input_channels, num_input_global_channels


def _parse_trunk(r: TokenStream, model_version: int, binary: bool) -> Trunk:
    name = r.read_token()
    num_blocks = r.read_int()
    trunk_c = r.read_int()
    mid_c = r.read_int()
    regular_c = r.read_int()
    _dilated_c = r.read_int()  # unused field
    gpool_c = r.read_int()

    if model_version >= 15:
        for _ in range(6):
            r.read_int()

    initial_conv = ConvLayer.parse(r, binary)
    initial_matmul = MatMulLayer.parse(r, binary)

    blocks: List[Union[ResidualBlock, GPoolBlock]] = []
    block_kinds: List[str] = []
    for _ in range(num_blocks):
        kind = r.read_token()
        if kind == "ordinary_block":
            blocks.append(ResidualBlock.parse(r, model_version, binary))
            block_kinds.append("regular")
        elif kind == "gpool_block":
            blocks.append(GPoolBlock.parse(r, model_version, binary))
            block_kinds.append("gpool")
        elif kind == "nested_bottleneck_block":
            raise NotImplementedError(
                "Nested bottleneck blocks are not supported "
                "(b10c128 from g170 should not contain them)."
            )
        else:
            raise ValueError(f"Unknown block kind: {kind}")

    # We don't need the trunk-tip BN/activation for warm init, but parse to
    # keep the stream position consistent if anyone extends this later.
    BatchNormLayer.parse(r, binary)
    ActivationLayer.parse(r, model_version)

    return Trunk(
        name=name, model_version=model_version, num_blocks=num_blocks,
        trunk_num_channels=trunk_c, mid_num_channels=mid_c,
        regular_num_channels=regular_c, gpool_num_channels=gpool_c,
        initial_conv=initial_conv, initial_matmul=initial_matmul,
        blocks=blocks, block_kinds=block_kinds,
    )


def parse_katago_model(path: str, force_binary: bool = None) -> KataGoModel:
    """Parse a KataGo model file (.bin.gz / .txt.gz / .bin / .txt).

    Returns the trunk only — heads are not parsed, to keep the script
    short. Adding head parsing is straightforward by following
    desc.cpp's PolicyHeadDesc / ValueHeadDesc constructors.
    """
    if force_binary is None:
        binary = ".bin" in os.path.basename(path)
    else:
        binary = force_binary

    with open(path, "rb") as f:
        magic = f.read(2)
        f.seek(0)
        if magic == b"\x1f\x8b":
            raw = gzip.decompress(f.read())
        else:
            raw = f.read()

    r = TokenStream(raw)
    name, model_version, num_in, num_global = _parse_model_header(r)
    trunk = _parse_trunk(r, model_version, binary)

    return KataGoModel(
        name=name, model_version=model_version,
        num_input_channels=num_in, num_input_global_channels=num_global,
        trunk=trunk,
    )


# ────────────────────────────────────────────────────────────
#  Trunk weight transfer
# ────────────────────────────────────────────────────────────


def _detect_minigo_topology(sd: dict) -> tuple:
    """Inspect a MiniGo state_dict and return (block_indices, block_kinds, num_filters)."""
    block_indices = sorted({
        int(k.split(".")[1]) for k in sd
        if k.startswith("trunk.") and k.split(".")[1].isdigit()
    })
    if not block_indices:
        raise ValueError("No 'trunk.N.*' keys found — is this an AlphaZeroNet checkpoint?")

    kinds = []
    for i in block_indices:
        if f"trunk.{i}.se.fc1.weight" in sd:
            kinds.append("se")
        elif f"trunk.{i}.conv_pool.weight" in sd:
            kinds.append("gpool")
        else:
            raise ValueError(
                f"Cannot identify trunk[{i}] kind — neither SE nor GPool keys present"
            )

    num_filters = sd[f"trunk.{block_indices[0]}.conv1.weight"].shape[0]
    return block_indices, kinds, num_filters


def _reset_bn(sd: dict, prefix: str, num_features: int) -> None:
    """Reset a BN layer to PyTorch defaults."""
    sd[f"{prefix}.weight"] = torch.ones(num_features)
    sd[f"{prefix}.bias"] = torch.zeros(num_features)
    sd[f"{prefix}.running_mean"] = torch.zeros(num_features)
    sd[f"{prefix}.running_var"] = torch.ones(num_features)
    if f"{prefix}.num_batches_tracked" in sd:
        sd[f"{prefix}.num_batches_tracked"] = torch.tensor(0, dtype=torch.long)


def warm_init(sd: dict, kg: KataGoModel, verbose: bool = True) -> dict:
    """Mutate `sd` to copy KG regular-block conv weights into MG SE blocks.

    Returns a dict with summary stats: blocks_paired, params_transferred,
    pairs_used (list of (mg_pos, kg_pos)).
    """
    mg_indices, mg_kinds, mg_filters = _detect_minigo_topology(sd)

    if mg_filters != kg.trunk.trunk_num_channels:
        raise ValueError(
            f"Channel count mismatch: MiniGo trunk has {mg_filters} filters, "
            f"KataGo has {kg.trunk.trunk_num_channels}. "
            f"For MG `large` (128f/10b) target b10c128; for `small` (64f/5b) target b6c64 instead."
        )

    if verbose:
        print(f"  MiniGo topology: {len(mg_indices)} blocks, {mg_filters} filters")
        print(f"  MiniGo kinds:    {mg_kinds}")
        print(f"  KataGo topology: {kg.trunk.num_blocks} blocks, {kg.trunk.trunk_num_channels} filters")
        print(f"  KataGo kinds:    {kg.trunk.block_kinds}")

    kg_regular_idx = [i for i, k in enumerate(kg.trunk.block_kinds) if k == "regular"]
    kg_used = set()
    pairs_used = []
    params_transferred = 0
    blocks_paired = 0

    if verbose:
        print()
        print("  Pairing (MG SE blocks ← closest-depth unused KG regular block):")

    for mg_pos, kind in zip(mg_indices, mg_kinds):
        if kind != "se":
            if verbose:
                print(f"    MG[{mg_pos}] {kind:>5s}: skip (channel layout incompatible)")
            continue

        # Closest-depth unused KG regular block
        candidates = [(abs(j - mg_pos), j) for j in kg_regular_idx if j not in kg_used]
        if not candidates:
            if verbose:
                print(f"    MG[{mg_pos}] se   : skip (no KG regular blocks remaining)")
            continue
        candidates.sort()
        kg_pos = candidates[0][1]
        kg_used.add(kg_pos)
        kg_blk = kg.trunk.blocks[kg_pos]
        assert isinstance(kg_blk, ResidualBlock)

        # Shape checks before copy
        c1_key = f"trunk.{mg_pos}.conv1.weight"
        c2_key = f"trunk.{mg_pos}.conv2.weight"
        c1 = sd[c1_key]
        c2 = sd[c2_key]
        kw1 = torch.from_numpy(kg_blk.regular_conv.weights).to(c1.dtype)
        kw2 = torch.from_numpy(kg_blk.final_conv.weights).to(c2.dtype)
        if tuple(c1.shape) != tuple(kw1.shape):
            raise ValueError(
                f"Shape mismatch {c1_key}: MG {tuple(c1.shape)} vs KG {tuple(kw1.shape)}"
            )
        if tuple(c2.shape) != tuple(kw2.shape):
            raise ValueError(
                f"Shape mismatch {c2_key}: MG {tuple(c2.shape)} vs KG {tuple(kw2.shape)}"
            )

        sd[c1_key] = kw1
        sd[c2_key] = kw2

        # Reset BN stats — pre-act vs post-act semantics differ.
        _reset_bn(sd, f"trunk.{mg_pos}.bn1", mg_filters)
        _reset_bn(sd, f"trunk.{mg_pos}.bn2", mg_filters)

        # SE module: leave as-is (zero-init from create_model gives sigmoid(0)=0.5
        # which is a near-identity gate).

        params_transferred += kw1.numel() + kw2.numel()
        blocks_paired += 1
        pairs_used.append((mg_pos, kg_pos))
        if verbose:
            print(
                f"    MG[{mg_pos}] se   ← KG[{kg_pos}] {kg_blk.name:<12s} "
                f"conv1+conv2 = {kw1.numel() + kw2.numel():,} params"
            )

    return {
        "blocks_paired": blocks_paired,
        "params_transferred": params_transferred,
        "pairs_used": pairs_used,
    }


# ────────────────────────────────────────────────────────────
#  CLI
# ────────────────────────────────────────────────────────────


def _save_checkpoint(model, path: str) -> None:
    """Write a checkpoint compatible with both train.py and train_continuous.py."""
    obj = {
        "model_state_dict": model.state_dict(),
        # Bare-minimum keys the resume paths look for. Defaults to 0
        # would also work (they all use .get(..., 0)) but writing them
        # explicitly silences "starting fresh" log messages.
        "step": 0,
        "iteration": 0,
        "bucket_level": 0,
        "watermark_id": 0,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(obj, path)


def _save_onnx(model, path: str, board_size: int, arch: str) -> None:
    """Export the model as ONNX, mirroring scripts/export_onnx.py."""
    from export_onnx import export_to_onnx  # imported lazily; pulls onnx dep
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    export_to_onnx(model, path, board_size=board_size,
                   input_channels=17, arch=arch)


def main():
    parser = argparse.ArgumentParser(
        description="Warm-initialize MiniGo from a KataGo b10c128 .bin.gz",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Recommended source: g170e-b10c128-s1141046784-d204142634 from\n"
            "https://katagoarchive.org/g170/neuralnets/ — the strongest g170\n"
            "b10c128 (Extended Training Net). Download the per-network\n"
            "directory's model.bin.gz (NOT the .zip — that's a TensorFlow\n"
            "checkpoint).\n"
            "\n"
            "At least one of --onnx / --checkpoint must be given. Both are\n"
            "the same warm-initialized weights — pick whichever the consumer\n"
            "needs. With run_continuous.py, write both: --onnx replaces the\n"
            "seed for selfplay/gate, --checkpoint seeds the train worker."
        ),
    )
    parser.add_argument("--katago-bin", required=True,
                        help="Path to KataGo .bin.gz / .txt.gz / .bin / .txt model file")

    parser.add_argument("--onnx", default=None,
                        help="Output path for ONNX (e.g. models/accepted/v000000000.onnx)")
    parser.add_argument("--checkpoint", default=None,
                        help="Output path for PyTorch .pt (e.g. training/checkpoints/training.pt)")

    # Fresh-model architecture (must match what `init` was called with)
    parser.add_argument("--arch", default="resnet", choices=["resnet"],
                        help="MiniGo architecture (only resnet supported by this tool)")
    parser.add_argument("--board", type=int, default=9)
    parser.add_argument("--filters", type=int, default=128,
                        help="Trunk channels (128 for b10c128 transfer)")
    parser.add_argument("--blocks", type=int, default=10,
                        help="Trunk blocks (10 for b10c128 transfer)")

    parser.add_argument("--dry-run", action="store_true",
                        help="Parse + plan + report, but don't write outputs")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-block pairing log")
    parser.add_argument("--txt", action="store_true",
                        help="Force text float parsing (override filename auto-detect)")
    parser.add_argument("--bin", action="store_true",
                        help="Force binary float parsing (override filename auto-detect)")
    args = parser.parse_args()

    if args.txt and args.bin:
        sys.exit("Pass at most one of --txt / --bin")
    if not args.onnx and not args.checkpoint:
        sys.exit("ERROR: pass at least one of --onnx / --checkpoint")
    force_binary = True if args.bin else (False if args.txt else None)

    print(f"Reading KataGo model from {args.katago_bin}...")
    kg = parse_katago_model(args.katago_bin, force_binary=force_binary)
    print(f"  Name:                   {kg.name}")
    print(f"  Model version:          {kg.model_version}")
    print(f"  Input spatial channels: {kg.num_input_channels}")
    print(f"  Input global channels:  {kg.num_input_global_channels}")
    print(f"  Trunk: {kg.trunk.num_blocks} blocks × {kg.trunk.trunk_num_channels} channels")
    n_reg = sum(1 for k in kg.trunk.block_kinds if k == "regular")
    n_gp = sum(1 for k in kg.trunk.block_kinds if k == "gpool")
    print(f"         {n_reg} regular + {n_gp} gpool")

    if kg.trunk.trunk_num_channels != 128:
        print(f"  WARN: expected b10c128 (128 channels) but got {kg.trunk.trunk_num_channels}")
    if kg.trunk.num_blocks != 10:
        print(f"  WARN: expected b10c128 (10 blocks) but got {kg.trunk.num_blocks}")

    print(f"\nBuilding fresh MiniGo {args.arch} ({args.filters}f/{args.blocks}b)...")
    from model import create_model
    model = create_model(arch=args.arch, board_size=args.board, input_channels=17,
                         num_filters=args.filters, num_res_blocks=args.blocks)
    sd = model.state_dict()
    sd_total = sum(
        v.numel() for v in sd.values()
        if isinstance(v, torch.Tensor) and v.dtype.is_floating_point
    )
    print(f"  Total float params: {sd_total:,}")

    print()
    print("Warm-initializing MiniGo trunk:")
    summary = warm_init(sd, kg, verbose=not args.quiet)

    print()
    print("Summary:")
    print(f"  Blocks paired:        {summary['blocks_paired']}")
    print(f"  Params transferred:   {summary['params_transferred']:,}")
    print(f"  Coverage of model:    {100.0 * summary['params_transferred'] / sd_total:.1f}%")

    # Apply warm-init back into the model so we can ONNX-export it.
    # strict=False so any extra/missing keys (e.g. the BN num_batches_tracked
    # buffers we may have added or dropped) don't fail the load.
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  WARN: load_state_dict missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  WARN: load_state_dict unexpected keys: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    if args.dry_run:
        print("\n--dry-run: not writing outputs.")
        return

    if args.checkpoint:
        _save_checkpoint(model, args.checkpoint)
        print(f"\nWrote checkpoint: {args.checkpoint}")

    if args.onnx:
        print()
        _save_onnx(model, args.onnx, board_size=args.board, arch=args.arch)


if __name__ == "__main__":
    main()
