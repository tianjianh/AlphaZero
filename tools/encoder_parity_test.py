#!/usr/bin/env python3
"""Encoder parity test: C++ encode_for_katago vs Python encode_katago.

The repo invariant is that the two KataGo V7 encoders (src/
katago_inputs.cpp and scripts/gamedata.py) are byte-identical.  This
harness proves it on (a) synthetic ladder scenarios and (b) real V3
selfplay records:

  1. writes each game as an UNCOMPRESSED V3 .bin,
  2. runs build/encode_dump (replays with the C++ GoGame + encoder),
  3. replays the same record with gamedata.Replay + encode_katago,
  4. compares every position: spatial planes must match EXACTLY,
     globals to 1e-6 (float arithmetic only, both fp32-representable).

Usage:
  python3 tools/encoder_parity_test.py [--records 'training/selfplay/g_*.bin.zst']
                                       [--limit 20] [--build-dir build]
"""

import argparse
import glob
import io
import os
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np

# --force-py must land in the environment BEFORE gamedata is imported
# (the native-lib probe result is cached at first use).
if "--force-py" in sys.argv:
    sys.argv.remove("--force-py")
    os.environ["MINIGO_LADDER_FORCE_PY"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from gamedata import (parse_v3, Replay, encode_katago,       # noqa: E402
                      KATAGO_SPATIAL, KATAGO_GLOBAL,
                      ladder_native_available)

PASS = None  # placeholder; per-board pass action = n*n


def write_v3(path, board_size, komi, actions):
    hw = board_size * board_size
    with open(path, "wb") as f:
        f.write(struct.pack("<HHifi", 0x4D47, 3, board_size, komi,
                            len(actions)))
        f.write(struct.pack("<bf", 0, 0.0))
        zero_policy = np.zeros(hw + 1, dtype=np.float32).tobytes()
        for a in actions:
            f.write(struct.pack("<h", a))
            f.write(zero_policy)
        f.write(np.zeros(hw, dtype=np.int8).tobytes())


def synthetic_games(n=9):
    """Ladder-heavy hand-built games (all moves legal by construction)."""
    p = n * n  # pass
    A = lambda r, c: r * n + c
    games = []
    # 1. Working ladder (white runs and dies) — plus play it out a few plies.
    games.append([A(2, 1), A(2, 2), A(1, 2), p, A(3, 1), p,
                  A(2, 3), A(3, 2), A(4, 2), A(3, 3), A(3, 4), A(4, 3),
                  A(5, 3), A(4, 4), A(4, 5), A(5, 4), A(6, 4), A(5, 5)])
    # 2. Same start with a ladder breaker at (6,6).
    games.append([A(2, 1), A(6, 6), A(1, 2), A(2, 2), A(3, 1), p,
                  A(2, 3), A(3, 2), A(4, 2), A(3, 3)])
    # 3. Corner captures + simple ko cycle.
    games.append([A(0, 1), A(0, 0), A(1, 0), p, p, A(1, 1),
                  A(2, 1), A(0, 0), A(0, 1)])
    # 4. Atari chains along the edge (1-lib defender-first cases).
    games.append([A(0, 0), A(0, 1), A(1, 0), A(1, 1), A(2, 0), A(2, 1),
                  A(3, 1), A(3, 0), p, A(4, 0)])
    # 5. Passes only (initial-board clamping of planes 15/16).
    games.append([p, p])
    return games


def cxx_dump(dump_bin, record_path, out_path, time_it=False):
    cmd = [dump_bin, "--record", record_path, "--output", out_path]
    if time_it:
        cmd.append("--time")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"encode_dump failed: {r.stderr}")
    return r.stdout.strip()


def compare_record(dump_bin, raw_bytes, name, time_it=False):
    """Returns (n_positions, max_global_diff).  Raises on any mismatch."""
    game = parse_v3(raw_bytes)
    n = game.board_size
    hw = n * n

    with tempfile.TemporaryDirectory() as td:
        rec_path = os.path.join(td, "g.bin")
        with open(rec_path, "wb") as f:
            f.write(raw_bytes)
        out_path = os.path.join(td, "planes.bin")
        timing = cxx_dump(dump_bin, rec_path, out_path, time_it)
        blob = open(out_path, "rb").read()

    magic, bs, n_moves = struct.unpack_from("<III", blob, 0)
    assert magic == 0x4B454E43 and bs == n and n_moves == game.n_moves, \
        f"{name}: dump header mismatch"
    per = KATAGO_SPATIAL * hw + KATAGO_GLOBAL
    data = np.frombuffer(blob, dtype=np.float32, offset=12)
    assert data.size == n_moves * per, f"{name}: dump size mismatch"
    data = data.reshape(n_moves, per)

    rep = Replay(n)
    max_gdiff = 0.0
    for m in range(n_moves):
        sp_py, gl_py = encode_katago(rep, game.komi)
        sp_cc = data[m, :KATAGO_SPATIAL * hw].reshape(KATAGO_SPATIAL, n, n)
        gl_cc = data[m, KATAGO_SPATIAL * hw:]

        if not np.array_equal(sp_py, sp_cc):
            for pl in range(KATAGO_SPATIAL):
                if not np.array_equal(sp_py[pl], sp_cc[pl]):
                    diff = np.argwhere(sp_py[pl] != sp_cc[pl])
                    raise AssertionError(
                        f"{name}: move {m} plane {pl} differs at {diff[:6].tolist()}\n"
                        f"py:\n{sp_py[pl]}\ncc:\n{sp_cc[pl]}")
        gdiff = float(np.abs(gl_py - gl_cc).max())
        max_gdiff = max(max_gdiff, gdiff)
        if gdiff > 1e-6:
            raise AssertionError(f"{name}: move {m} globals differ by {gdiff}")

        rep.play(int(game.actions[m]))

    if time_it and timing:
        print(f"    {timing}")
    return n_moves, max_gdiff


def main():
    ap = argparse.ArgumentParser(description="C++/Python encoder parity test")
    ap.add_argument("--records", default="",
                    help="glob of real g_*.bin.zst records to also check")
    ap.add_argument("--limit", type=int, default=20,
                    help="max real records to check")
    ap.add_argument("--build-dir", default="build")
    args = ap.parse_args()

    dump_bin = os.path.join(args.build_dir, "encode_dump")
    if not os.path.isfile(dump_bin):
        print(f"ERROR: {dump_bin} not found — build it first (make encode_dump)")
        sys.exit(2)

    total_pos = 0
    worst_g = 0.0

    mode = "native ladder lib" if ladder_native_available() else "pure-Python ladder"
    print(f"python encoder mode: {mode}")
    print("── synthetic ladder scenarios ──")
    for i, moves in enumerate(synthetic_games()):
        buf = io.BytesIO()
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tf:
            write_v3(tf.name, 9, 7.5, moves)
            raw = open(tf.name, "rb").read()
            os.unlink(tf.name)
        npos, g = compare_record(dump_bin, raw, f"synthetic#{i}",
                                 time_it=(i == 0))
        total_pos += npos
        worst_g = max(worst_g, g)
        print(f"  synthetic#{i}: {npos} positions OK")

    if args.records:
        import zstandard as zstd
        files = sorted(glob.glob(args.records))[: args.limit]
        print(f"── real records ({len(files)}) ──")
        t0 = time.time()
        py_pos = 0
        for path in files:
            with open(path, "rb") as f:
                if path.endswith(".zst"):
                    raw = zstd.ZstdDecompressor().stream_reader(f).read()
                else:
                    raw = f.read()
            npos, g = compare_record(dump_bin, raw, os.path.basename(path),
                                     time_it=(py_pos == 0))
            total_pos += npos
            py_pos += npos
            worst_g = max(worst_g, g)
        dt = time.time() - t0
        if py_pos:
            print(f"  {len(files)} records, {py_pos} positions, "
                  f"{dt / py_pos * 1e3:.2f} ms/position (py+cc+compare)")

    print(f"PARITY OK: {total_pos} positions, spatial exact, "
          f"max |global diff| = {worst_g:.2e}")


if __name__ == "__main__":
    main()
