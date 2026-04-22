#!/usr/bin/env bash
# Compress game_*.bin files under a directory to .bin.zst in place.
# Same write-then-verify-then-delete semantics as run_loop.py's
# compress_selfplay() — safe to interrupt and resume at any time.
#
# Usage:
#   tools/compress_bootstrap.sh                           # defaults to training/bootstrap
#   tools/compress_bootstrap.sh /path/to/dir              # any dir
#   tools/compress_bootstrap.sh /path/to/dir 200          # progress every N files

set -u

DIR="${1:-training/bootstrap}"
REPORT="${2:-200}"

if [ ! -d "$DIR" ]; then
    echo "not a directory: $DIR" >&2
    exit 1
fi

cd "$DIR" || exit 1

shopt -s nullglob
bins=( game_*.bin )
total=${#bins[@]}
if [ "$total" -eq 0 ]; then
    echo "no game_*.bin files in $DIR — nothing to compress."
    exit 0
fi

echo "Compressing $total .bin files in $DIR"

ok=0
skipped=0
failed=0
for f in "${bins[@]}"; do
    [ -e "$f" ] || continue
    dst="$f.zst"
    if [ -s "$dst" ]; then
        rm -f "$f"
        skipped=$((skipped + 1))
        continue
    fi
    if zstd -q -f -o "$dst" "$f" && [ -s "$dst" ]; then
        rm -f "$f"
        ok=$((ok + 1))
    else
        rm -f "$dst"
        failed=$((failed + 1))
        echo "  FAILED: $f" >&2
    fi
    done_now=$((ok + skipped + failed))
    if [ "$((done_now % REPORT))" -eq 0 ]; then
        printf "  %d/%d  (ok=%d skipped=%d failed=%d)\n" \
               "$done_now" "$total" "$ok" "$skipped" "$failed"
    fi
done

printf "Done.  ok=%d skipped=%d failed=%d  of %d\n" \
       "$ok" "$skipped" "$failed" "$total"
[ "$failed" -eq 0 ]
