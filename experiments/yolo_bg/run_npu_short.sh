#!/bin/sh
# Run ON THE BOARD from its NFS-mounted checkout. Never starts a full batch.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
VIDEO=${1:-data15}
FACE=${2:-0}
FRAMES=${3:-120}
case "$VIDEO" in data[0-9]|data[0-9][0-9]) ;; *) echo 'video must be data1..data18' >&2; exit 2;; esac
case "$FACE" in 0|1) ;; *) echo 'face must be 0 or 1' >&2; exit 2;; esac
case "$FRAMES" in ''|*[!0-9]*) echo 'frames must be 1..300' >&2; exit 2;; esac
[ "$FRAMES" -ge 1 ] && [ "$FRAMES" -le 300 ] || { echo 'short test limit: 1..300 frames' >&2; exit 2; }
umask 022
OUT=$(mktemp -d "$ROOT/build/npu_people/short_${VIDEO}_face${FACE}_XXXXXX")
# Keep root-squashed NFS results readable from the PC, as in the batch runner.
chmod 755 "$OUT"
# Capture the concrete settings used for this short run.
cp "$ROOT/Parameter/sdk_gray_count_falls.ini" "$OUT/parameters.ini"
export HERMES_FILE_OUTPUT=1
export HERMES_PERSON_BG=1 HERMES_PERSON_MODE=async HERMES_PERSON_INTERVAL_MS=1600 HERMES_PERSON_DELAY_MS=800
export HERMES_PERSON_MODEL="$ROOT/handoff_yolov7t_people/yolov7t_people_320x320_v2.ty"
export HERMES_PERSON_CONF=0.25 HERMES_PERSON_NMS=0.7 HERMES_PERSON_PROTECTION_V2=1
export HERMES_PERSON_HISTORY_TWO=1 HERMES_FALL_PERSON_CONFIRM=1 HERMES_EXCLUDE_TOP_ROW=1
export HERMES_IDENTITY_A=1 HERMES_COAST_VALIDITY=0 HERMES_BG_STABILITY=1 HERMES_BG_STABILITY_SECONDS=6
export HERMES_PERSON_TRACE=1
printf 'video=%s\nface=%s\nmax_frames=%s\nframe_period_ms=133.3333333333333\n' "$VIDEO" "$FACE" "$FRAMES" > "$OUT/run.txt"
echo "Short test output: $OUT"
cd "$OUT"
exec "$ROOT/build/npu_people/sdk_gray_npu" "$ROOT/Bug_Video/Fall_17/$VIDEO.gray" \
 --width=800 --height=450 --interval=1 --frame-period-ms=133.3333333333333 \
 --realtime --max-frames="$FRAMES" --face="$FACE" \
 --face-model="$ROOT/models/blaze_face_detect_nnp310_128x128.ty" \
 --ini="$OUT/parameters.ini" --write-files --output-prefix="$OUT/$VIDEO" > "$OUT/runner.log" 2>&1
