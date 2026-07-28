#!/usr/bin/env python3
"""Parser for KataGo native model files (.bin.gz / .txt.gz).

Reads a kata1-style network downloaded from katagotraining.org into
plain numpy descriptor objects (Trunk / PolicyHead / ValueHead with
ConvLayer / BatchNormLayer / MatMul / ... leaves).  Consumers:

  tools/katago_to_onnx.py       — kata1 -> ONNX conversion (via katago_arch)
  tools/kata_export_for_rknn.py — kata1 -> RKNN-friendly ONNX
  tools/katago_parity_test.py   — PyTorch <-> ONNX Runtime parity check

Supports model_version 8-15 (v15+ extra policy outputs are parsed and
ignored).  Pure numpy — no torch dependency.
"""

import gzip
import io
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np


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
    # Trunk-tip BN/activation applied after the last block before heads.
    # Optional[...] = None so consumers that don't read these are unchanged.
    trunk_tip_bn: Optional[BatchNormLayer] = None
    trunk_tip_act: Optional[ActivationLayer] = None


@dataclass
class PolicyHead:
    """KataGo's policy head.

    Required fields (used by katago_arch): p1_conv, p1_bn.

    Optional fields (Optional[...] = None) capture the previously-discarded
    layers needed by the inference graph (katago_to_onnx.py). Warm-init
    callers never read these, so existing behavior is byte-for-byte
    preserved (validated by Phase 0.1 byte-diff of training.pt).
    """
    name: str
    p1_conv: ConvLayer
    p1_bn: BatchNormLayer
    # Inference-only fields (consumed by katago_arch for the full graph):
    g1_conv:        Optional[ConvLayer]       = None
    g1_bn:          Optional[BatchNormLayer]  = None
    g1_act:         Optional[ActivationLayer] = None
    gpool_to_bias:  Optional[MatMulLayer]     = None
    p1_act:         Optional[ActivationLayer] = None
    p2_conv:        Optional[ConvLayer]       = None
    gpool_to_pass:  Optional[MatMulLayer]     = None
    # v15+ extras (still Optional):
    gpool_to_pass_bias:  Optional[MatBiasLayer]    = None
    pass_act:            Optional[ActivationLayer] = None
    gpool_to_pass_mul2:  Optional[MatMulLayer]     = None


@dataclass
class ValueHead:
    """KataGo's value head.

    Required fields (used by katago_arch): v1_conv, v1_bn.

    Optional fields capture the previously-discarded layers needed by the
    inference graph (katago_to_onnx.py).
    """
    name: str
    v1_conv: ConvLayer
    v1_bn: BatchNormLayer
    # Inference-only fields (consumed by katago_arch for the full graph):
    v1_act:           Optional[ActivationLayer] = None
    v2_mul:           Optional[MatMulLayer]     = None
    v2_bias:          Optional[MatBiasLayer]    = None
    v2_act:           Optional[ActivationLayer] = None
    v3_mul:           Optional[MatMulLayer]     = None
    v3_bias:          Optional[MatBiasLayer]    = None
    sv3_mul:          Optional[MatMulLayer]     = None
    sv3_bias:         Optional[MatBiasLayer]    = None
    v_ownership_conv: Optional[ConvLayer]       = None


@dataclass
class KataGoModel:
    name: str
    model_version: int
    num_input_channels: int
    num_input_global_channels: int
    trunk: Trunk
    policy_head: Union[PolicyHead, None] = None
    value_head: Union[ValueHead, None] = None


def _parse_model_header(r: TokenStream) -> tuple:
    name = r.read_token()
    model_version = r.read_int()
    if model_version < 8:
        raise ValueError(
            f"Model version {model_version} is too old for this script "
            f"(targets v8-v15, tested against kata1 b10c128 v10)."
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
                "(b10c128 should not contain them)."
            )
        else:
            raise ValueError(f"Unknown block kind: {kind}")

    # Trunk-tip BN/activation: applied after the last block before heads.
    # Warm-init doesn't read these; the inference graph (katago_to_onnx.py)
    # does.
    trunk_tip_bn = BatchNormLayer.parse(r, binary)
    trunk_tip_act = ActivationLayer.parse(r, model_version)

    return Trunk(
        name=name, model_version=model_version, num_blocks=num_blocks,
        trunk_num_channels=trunk_c, mid_num_channels=mid_c,
        regular_num_channels=regular_c, gpool_num_channels=gpool_c,
        initial_conv=initial_conv, initial_matmul=initial_matmul,
        blocks=blocks, block_kinds=block_kinds,
        trunk_tip_bn=trunk_tip_bn, trunk_tip_act=trunk_tip_act,
    )


def _parse_policy_head(r: TokenStream, model_version: int, binary: bool) -> PolicyHead:
    """Parse the policy head. All layers retained as Optional[...] fields
    so katago_to_onnx.py can build a full inference graph; trunk-only
    reads p1_conv and p1_bn.
    """
    name = r.read_token()
    p1_conv     = ConvLayer.parse(r, binary)
    g1_conv     = ConvLayer.parse(r, binary)
    g1_bn       = BatchNormLayer.parse(r, binary)
    g1_act      = ActivationLayer.parse(r, model_version)
    gpool_bias  = MatMulLayer.parse(r, binary)
    p1_bn       = BatchNormLayer.parse(r, binary)
    p1_act      = ActivationLayer.parse(r, model_version)
    p2_conv     = ConvLayer.parse(r, binary)
    gpool_pass  = MatMulLayer.parse(r, binary)
    extras = {}
    if model_version >= 15:
        extras['gpool_to_pass_bias'] = MatBiasLayer.parse(r, binary)
        extras['pass_act']           = ActivationLayer.parse(r, model_version)
        extras['gpool_to_pass_mul2'] = MatMulLayer.parse(r, binary)
    return PolicyHead(
        name=name, p1_conv=p1_conv, p1_bn=p1_bn,
        g1_conv=g1_conv, g1_bn=g1_bn, g1_act=g1_act,
        gpool_to_bias=gpool_bias, p1_act=p1_act,
        p2_conv=p2_conv, gpool_to_pass=gpool_pass, **extras,
    )


def _parse_value_head(r: TokenStream, model_version: int, binary: bool) -> ValueHead:
    """Parse the value head. All layers retained as Optional[...] fields
    so katago_to_onnx.py can build a full inference graph; trunk-only
    reads v1_conv and v1_bn.
    """
    name = r.read_token()
    v1_conv          = ConvLayer.parse(r, binary)
    v1_bn            = BatchNormLayer.parse(r, binary)
    v1_act           = ActivationLayer.parse(r, model_version)
    v2_mul           = MatMulLayer.parse(r, binary)
    v2_bias          = MatBiasLayer.parse(r, binary)
    v2_act           = ActivationLayer.parse(r, model_version)
    v3_mul           = MatMulLayer.parse(r, binary)
    v3_bias          = MatBiasLayer.parse(r, binary)
    sv3_mul          = MatMulLayer.parse(r, binary)
    sv3_bias         = MatBiasLayer.parse(r, binary)
    v_ownership_conv = ConvLayer.parse(r, binary)
    return ValueHead(
        name=name, v1_conv=v1_conv, v1_bn=v1_bn,
        v1_act=v1_act,
        v2_mul=v2_mul, v2_bias=v2_bias, v2_act=v2_act,
        v3_mul=v3_mul, v3_bias=v3_bias,
        sv3_mul=sv3_mul, sv3_bias=sv3_bias,
        v_ownership_conv=v_ownership_conv,
    )


def parse_katago_model(path: str, force_binary: bool = None) -> KataGoModel:
    """Parse a KataGo model file (.bin.gz / .txt.gz / .bin / .txt).

    Returns trunk + value-head's v1Conv/v1BN. Policy head and the rest
    of the value head are walked over but not retained — see
    _skip_policy_head and _parse_value_head.
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

    # Parse heads. If anything goes wrong (e.g. unsupported version),
    # warn and continue with trunk-only — head transfer is optional.
    policy_head = None
    value_head = None
    try:
        policy_head = _parse_policy_head(r, model_version, binary)
        value_head = _parse_value_head(r, model_version, binary)
    except (ValueError, NotImplementedError) as e:
        print(f"WARN: head parse failed ({e}); continuing with trunk-only transfer")

    return KataGoModel(
        name=name, model_version=model_version,
        num_input_channels=num_in, num_input_global_channels=num_global,
        trunk=trunk, policy_head=policy_head, value_head=value_head,
    )
