#!/bin/sh
# Run on the board; no NPU initialization, no video playback.
set -eu
export HERMES_FILE_OUTPUT=1
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
BIN="$ROOT/build/npu_people/abi_diagnostic"
[ -x "$BIN" ] || { echo 'Build on PC: make -f experiments/yolo_bg/Makefile.npu diagnostic' >&2; exit 2; }
umask 022
OUT=$(mktemp -d "$ROOT/build/npu_people/abi_diag_XXXXXX")
chmod 755 "$OUT"
echo "Diagnostic output: $OUT"
if "$BIN" "$OUT/files" > "$OUT/diagnostic.log" 2>&1; then
    RC=0
else
    RC=$?
fi
cat "$OUT/diagnostic.log"
echo "Diagnostic exit code: $RC; report: $OUT/diagnostic.log"
exit "$RC"
