#!/usr/bin/env python3
"""Generate ONNX models + reference input/output vectors for backend
numerical verification (build/verify reads the .vec files).

Covers every model format the C++ backends must support:
  resnet       — MiniGo KataGo-style ResNet (SE + GPool blocks)
  vit          — GoViT (GQA attention, directional rel-bias)
  katanet      — trainable KataGoNet (dual input, MiniGo heads)
  kata1_relu   — synthetic converted-kata1 network, ReLU
  kata1_mish   — synthetic converted-kata1 network, Mish

Vector file format (little-endian):
  u32 magic 'MGTV' (0x4D475456) | u32 board | u32 state_len | u32 n
  n * state_len f32   inputs   (flat; kata: 22*hw spatial then 19 global)
  n * (2*hw + 4) f32  expected  [policy(hw+1) | value | score | score_sd | own(hw)]

Usage: python3 scripts/make_test_vectors.py [outdir]
"""

import os
import struct
import sys
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "tools"))

from model import create_model                      # noqa: E402
from export_onnx import _InferenceWrapper, export_to_onnx  # noqa: E402

BOARD = 9
HW = BOARD * BOARD
N_STATES = 5
SEED = 20260713


def randomize_degenerate(model):
    """Give zero-init'ed parameters and fresh BN stats non-trivial values
    so the numeric test actually exercises those paths (SE gates, gpool
    bias injections, rel_bias, BN folding)."""
    g = torch.Generator().manual_seed(SEED)
    for p in model.parameters():
        if p.numel() > 0 and float(p.detach().abs().max()) == 0.0:
            with torch.no_grad():
                p.uniform_(-0.2, 0.2, generator=g)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            with torch.no_grad():
                m.running_mean.uniform_(-0.5, 0.5, generator=g)
                m.running_var.uniform_(0.5, 2.0, generator=g)
                m.weight.uniform_(0.5, 1.5, generator=g)
                m.bias.uniform_(-0.3, 0.3, generator=g)


def write_vec(path, board, states, refs):
    """states: [n, state_len]; refs: [n, 2*hw+4]"""
    states = np.asarray(states, dtype=np.float32)
    refs = np.asarray(refs, dtype=np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIII", 0x4D475456, board, states.shape[1],
                            states.shape[0]))
        f.write(states.tobytes())
        f.write(refs.tobytes())
    print(f"  wrote {path}: {states.shape[0]} states x {states.shape[1]}")


def refs_from_outputs(policy, value, score, sd, own):
    return np.concatenate([
        policy.numpy(), value.numpy(), score.numpy(), sd.numpy(), own.numpy(),
    ], axis=1)


def gen_single_input(name, model, outdir):
    randomize_degenerate(model)
    model.eval()
    onnx_path = os.path.join(outdir, f"{name}.onnx")
    export_to_onnx(model, onnx_path, board_size=BOARD)

    torch.manual_seed(SEED)
    x = torch.randn(N_STATES, 17, BOARD, BOARD)
    with torch.no_grad():
        pol, val, sm, sd, own = _InferenceWrapper(model)(x)
    states = x.reshape(N_STATES, -1)
    write_vec(os.path.join(outdir, f"{name}.vec"), BOARD,
              states.numpy(), refs_from_outputs(pol, val, sm, sd, own))


def gen_katanet(outdir):
    model = create_model("katago", board_size=BOARD, num_filters=32,
                         num_res_blocks=3)
    randomize_degenerate(model)
    model.eval()
    onnx_path = os.path.join(outdir, "katanet.onnx")
    export_to_onnx(model, onnx_path, board_size=BOARD)

    torch.manual_seed(SEED + 1)
    sp = torch.randn(N_STATES, 22, BOARD, BOARD)
    gl = torch.randn(N_STATES, 19)
    with torch.no_grad():
        pol, val, sm, sd, own = _InferenceWrapper(model)(sp, gl)
    states = np.concatenate([sp.reshape(N_STATES, -1).numpy(),
                             gl.numpy()], axis=1)
    write_vec(os.path.join(outdir, "katanet.vec"), BOARD,
              states, refs_from_outputs(pol, val, sm, sd, own))


# ── Synthetic converted-kata1 network ────────────────────────────
def _mk_conv(rng, in_ch, out_ch, k):
    std = (2.0 / (in_ch * k * k)) ** 0.5
    return types.SimpleNamespace(
        in_ch=in_ch, out_ch=out_ch, y=k, x=k,
        weights=rng.normal(0, std, (out_ch, in_ch, k, k)).astype(np.float32))


def _mk_bn(rng, ch):
    return types.SimpleNamespace(
        num_channels=ch, epsilon=1e-20,
        mean=rng.normal(0, 0.5, ch).astype(np.float32),
        variance=rng.uniform(0.5, 2.0, ch).astype(np.float32),
        scale=rng.uniform(0.5, 1.5, ch).astype(np.float32),
        bias=rng.normal(0, 0.3, ch).astype(np.float32))


def _mk_mm(rng, in_ch, out_ch):
    std = (1.0 / in_ch) ** 0.5
    return types.SimpleNamespace(
        in_ch=in_ch, out_ch=out_ch,
        weights=rng.normal(0, std, (out_ch, in_ch)).astype(np.float32))


def _mk_bias(rng, ch):
    return types.SimpleNamespace(
        num_channels=ch,
        weights=rng.normal(0, 0.2, ch).astype(np.float32))


def _mk_act(name):
    return types.SimpleNamespace(activation=name)


def gen_kata1(outdir, act, tag):
    """Build a small kata1-style net through tools/katago_arch.py with mock
    parsed-binary descriptors, export it the same way katago_to_onnx does."""
    from katago_arch import KataGoNet as Kata1Net

    rng = np.random.default_rng(SEED + 7)
    C, PC, RC = 32, 16, 32           # trunk / gpool / regular channels
    blk_reg = types.SimpleNamespace(
        pre_bn=_mk_bn(rng, C), pre_act=_mk_act(act),
        regular_conv=_mk_conv(rng, C, RC, 3),
        mid_bn=_mk_bn(rng, RC), mid_act=_mk_act(act),
        final_conv=_mk_conv(rng, RC, C, 3))
    blk_gp = types.SimpleNamespace(
        pre_bn=_mk_bn(rng, C), pre_act=_mk_act(act),
        regular_conv=_mk_conv(rng, C, RC, 3),
        gpool_conv=_mk_conv(rng, C, PC, 3),
        gpool_bn=_mk_bn(rng, PC), gpool_act=_mk_act(act),
        gpool_to_bias_mul=_mk_mm(rng, 3 * PC, RC),
        mid_bn=_mk_bn(rng, RC), mid_act=_mk_act(act),
        final_conv=_mk_conv(rng, RC, C, 3))
    trunk = types.SimpleNamespace(
        trunk_num_channels=C, num_blocks=3,
        initial_conv=_mk_conv(rng, 22, C, 5),
        initial_matmul=_mk_mm(rng, 19, C),
        block_kinds=["regular", "gpool", "regular"],
        blocks=[blk_reg, blk_gp, types.SimpleNamespace(
            pre_bn=_mk_bn(rng, C), pre_act=_mk_act(act),
            regular_conv=_mk_conv(rng, C, RC, 3),
            mid_bn=_mk_bn(rng, RC), mid_act=_mk_act(act),
            final_conv=_mk_conv(rng, RC, C, 3))],
        trunk_tip_bn=_mk_bn(rng, C), trunk_tip_act=_mk_act(act))
    policy = types.SimpleNamespace(
        p1_conv=_mk_conv(rng, C, 24, 1),
        g1_conv=_mk_conv(rng, C, 24, 1),
        g1_bn=_mk_bn(rng, 24), g1_act=_mk_act(act),
        gpool_to_bias=_mk_mm(rng, 3 * 24, 24),
        p1_bn=_mk_bn(rng, 24), p1_act=_mk_act(act),
        p2_conv=_mk_conv(rng, 24, 2, 1),
        gpool_to_pass=_mk_mm(rng, 3 * 24, 1))
    value = types.SimpleNamespace(
        v1_conv=_mk_conv(rng, C, 24, 1),
        v1_bn=_mk_bn(rng, 24), v1_act=_mk_act(act),
        v2_mul=_mk_mm(rng, 3 * 24, 48), v2_bias=_mk_bias(rng, 48),
        v2_act=_mk_act(act),
        v3_mul=_mk_mm(rng, 48, 3), v3_bias=_mk_bias(rng, 3),
        sv3_mul=_mk_mm(rng, 48, 4), sv3_bias=_mk_bias(rng, 4),
        v_ownership_conv=_mk_conv(rng, 24, 1, 1))
    kmodel = types.SimpleNamespace(
        num_input_channels=22, num_input_global_channels=19,
        trunk=trunk, policy_head=policy, value_head=value)

    net = Kata1Net(kmodel).eval()
    onnx_path = os.path.join(outdir, f"{tag}.onnx")
    sp0 = torch.zeros(1, 22, BOARD, BOARD)
    gl0 = torch.zeros(1, 19)
    torch.onnx.export(
        net, (sp0, gl0), onnx_path,
        input_names=["state_spatial", "state_global"],
        output_names=["policy_logits", "value", "score_mean",
                      "score_stdev", "ownership"],
        dynamic_axes={n: {0: "batch"} for n in
                      ["state_spatial", "state_global", "policy_logits",
                       "value", "score_mean", "score_stdev", "ownership"]},
        opset_version=17, do_constant_folding=True, dynamo=False)
    import onnx
    from onnx import numpy_helper
    mdl = onnx.load(onnx_path)
    for name, tensor in net.state_dict().items():
        mdl.graph.initializer.append(
            numpy_helper.from_array(tensor.numpy(), name="_sd_" + name))
    onnx.save(mdl, onnx_path)
    print(f"  exported {onnx_path}")

    torch.manual_seed(SEED + 2)
    sp = torch.randn(N_STATES, 22, BOARD, BOARD)
    gl = torch.randn(N_STATES, 19)
    with torch.no_grad():
        pol, val, sm, sd, own = net(sp, gl)
    states = np.concatenate([sp.reshape(N_STATES, -1).numpy(), gl.numpy()], axis=1)
    write_vec(os.path.join(outdir, f"{tag}.vec"), BOARD,
              states, refs_from_outputs(pol, val, sm, sd, own))


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "test_vectors"
    os.makedirs(outdir, exist_ok=True)
    torch.manual_seed(SEED)

    print("resnet (SE + GPool blocks, f32 b3):")
    gen_single_input("resnet",
                     create_model("resnet", board_size=BOARD,
                                  num_filters=32, num_res_blocks=3), outdir)

    print("vit (d64 depth2 h4 g2):")
    gen_single_input("vit",
                     create_model("vit", board_size=BOARD, d_model=64,
                                  depth=2, heads=4, kv_groups=2,
                                  mlp_ratio=4), outdir)

    print("katanet (trainable KataGoNet c32 b3):")
    gen_katanet(outdir)

    print("kata1_relu (synthetic converted kata1):")
    gen_kata1(outdir, "relu", "kata1_relu")

    print("kata1_mish (synthetic converted kata1):")
    gen_kata1(outdir, "mish", "kata1_mish")

    print("done.")


if __name__ == "__main__":
    main()
