#!/bin/sh
# Run ON THE BOARD. Reads every frame; keeps the 133.33 ms playback schedule.
set -eu
trap 'echo "Batch interrupted" >&2; exit 130' INT
trap 'echo "Batch terminated" >&2; exit 143' TERM
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
usage() {
    echo "Usage: sh $0 [first=4] [last=17] [face=0] [person_model.ty]"
    echo 'Runs every frame, in order, at 133.3333333333333 ms per frame.'
}
case "${1:-}" in -h|--help) usage; exit 0;; esac
[ "$#" -le 4 ] || { usage >&2; exit 2; }
FIRST=${1:-4}
LAST=${2:-17}
FACE=${3:-0}
MODEL=${4:-${HERMES_PERSON_MODEL:-$ROOT/handoff_yolov7t_people/yolov7t_people_320x320_v2.ty}}
for VALUE in "$FIRST" "$LAST"; do
    case "$VALUE" in [1-9]|1[0-8]) ;; *) echo 'Video indices must be 1..18' >&2; exit 2;; esac
done
[ "$FIRST" -le "$LAST" ] || { echo 'first must be <= last' >&2; exit 2; }
case "$FACE" in 0|1) ;; *) echo 'face must be 0 or 1' >&2; exit 2;; esac
# Resolve a relative model before changing into the output directory.
case "$MODEL" in /*) ;; *) MODEL="$(pwd)/$MODEL";; esac
RUNNER="$ROOT/build/npu_people/sdk_gray_npu"
INI="$ROOT/Parameter/sdk_gray_count_falls.ini"
FACE_MODEL="$ROOT/models/blaze_face_detect_nnp310_128x128.ty"
[ -x "$RUNNER" ] || { echo "Runner missing/not executable: $RUNNER" >&2; exit 1; }
for INPUT in "$MODEL" "$INI"; do
    [ -r "$INPUT" ] || { echo "Cannot read: $INPUT" >&2; exit 1; }
done
if [ "$FACE" = 1 ] && [ ! -r "$FACE_MODEL" ]; then
    echo "Cannot read: $FACE_MODEL" >&2; exit 1
fi
N=$FIRST
while [ "$N" -le "$LAST" ]; do
    INPUT="$ROOT/Bug_Video/Fall_17/data$N.gray"
    [ -r "$INPUT" ] || { echo "Cannot read: $INPUT" >&2; exit 1; }
    N=$((N + 1))
done

umask 022
OUT=$(mktemp -d "$ROOT/build/npu_people/batch_data${FIRST}_${LAST}_face${FACE}_XXXXXX")
# mktemp uses 0700; allow the PC user to read NFS root-squashed results.
chmod 755 "$OUT"
export HERMES_PERSON_BG=1 HERMES_PERSON_MODE=async
export HERMES_PERSON_INTERVAL_MS=1600 HERMES_PERSON_DELAY_MS=800
export HERMES_PERSON_MODEL="$MODEL"
export HERMES_PERSON_CONF=0.25 HERMES_PERSON_NMS=0.7
export HERMES_PERSON_PROTECTION_V2=1 HERMES_PERSON_HISTORY_TWO=1
export HERMES_FALL_PERSON_CONFIRM=1 HERMES_EXCLUDE_TOP_ROW=1
export HERMES_IDENTITY_A=1 HERMES_COAST_VALIDITY=0
export HERMES_BG_STABILITY=1 HERMES_BG_STABILITY_SECONDS=6
export HERMES_PERSON_TRACE=1
printf 'first=%s\nlast=%s\nface=%s\nmodel=%s\nmax_frames=0\nframe_period_ms=133.3333333333333\n' \
    "$FIRST" "$LAST" "$FACE" "$MODEL" > "$OUT/run.txt"
printf 'video,status\n' > "$OUT/status.csv"
echo "Batch output: $OUT"
N=$FIRST
while [ "$N" -le "$LAST" ]; do
    RUN="$OUT/data$N"
    mkdir "$RUN"
    cp "$INI" "$RUN/parameters.ini"
    echo "Starting data$N (face=$FACE): $RUN"
    if (
        cd "$RUN"
        exec "$RUNNER" "$ROOT/Bug_Video/Fall_17/data$N.gray" \
            --width=800 --height=450 --interval=1 \
            --frame-period-ms=133.3333333333333 --realtime \
            --max-frames=0 --face="$FACE" --face-model="$FACE_MODEL" \
            --ini="$RUN/parameters.ini" --output-prefix="$RUN/data$N" \
            > "$RUN/runner.log" 2>&1
    ); then
        printf 'data%s,completed\n' "$N" >> "$OUT/status.csv"
    else
        RC=$?
        printf 'data%s,failed_%s\n' "$N" "$RC" >> "$OUT/status.csv"
        echo "data$N failed ($RC); see $RUN/runner.log" >&2
        exit "$RC"
    fi
    N=$((N + 1))
done
echo "Batch completed: $OUT"
