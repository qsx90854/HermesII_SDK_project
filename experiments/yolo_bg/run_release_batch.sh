#!/bin/sh
# Board validation: only this shell writes logs; the release SDK cannot write files.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
case "${1:-}" in -h|--help)
    echo "Usage: sh $0 [first=4] [last=17] [face=1] [output_parent]"
    echo 'Overrides: RUNNER, DATA_DIR, GROUND_TRUTH, OUTPUT_DIR, YOLO_MODEL, FACE_MODEL, INI, MAX_FRAMES (0=all).'
    exit 0;;
esac
[ "$#" -le 4 ] || { echo 'Too many arguments' >&2; exit 2; }
FIRST=${1:-4}; LAST=${2:-17}; FACE=${3:-1}
for N in "$FIRST" "$LAST"; do
    case "$N" in [1-9]|1[0-8]) ;; *) echo 'Data indices must be 1..18' >&2; exit 2;; esac
done
[ "$FIRST" -le "$LAST" ] || { echo 'first must be <= last' >&2; exit 2; }
case "$FACE" in 0|1) ;; *) echo 'face must be 0 or 1' >&2; exit 2;; esac
MAX_FRAMES=${MAX_FRAMES:-0}
case "$MAX_FRAMES" in ''|*[!0-9]*) echo 'Invalid MAX_FRAMES' >&2; exit 2;; esac
RUNNER=${RUNNER:-$ROOT/build/release_npu/sdk_gray_count_falls_edge}
DATA_DIR=${DATA_DIR:-$ROOT/Bug_Video/Fall_17}
YOLO_MODEL=${YOLO_MODEL:-$ROOT/handoff_yolov7t_people/yolov7t_people_320x320_v2.ty}
FACE_MODEL=${FACE_MODEL:-$ROOT/models/blaze_face_detect_nnp310_128x128.ty}
INI=${INI:-$ROOT/Parameter/sdk_gray_count_falls.ini}
GROUND_TRUTH=${GROUND_TRUTH:-$DATA_DIR/groundTruth}
PARENT=${4:-${OUTPUT_DIR:-$ROOT/build/release_validation}}
# Resolve overrides before changing the working directory.
case "$RUNNER" in /*) ;; *) RUNNER="$(pwd)/$RUNNER";; esac
case "$DATA_DIR" in /*) ;; *) DATA_DIR="$(pwd)/$DATA_DIR";; esac
case "$YOLO_MODEL" in /*) ;; *) YOLO_MODEL="$(pwd)/$YOLO_MODEL";; esac
case "$FACE_MODEL" in /*) ;; *) FACE_MODEL="$(pwd)/$FACE_MODEL";; esac
case "$INI" in /*) ;; *) INI="$(pwd)/$INI";; esac
case "$GROUND_TRUTH" in /*) ;; *) GROUND_TRUTH="$(pwd)/$GROUND_TRUTH";; esac
case "$PARENT" in /*) ;; *) PARENT="$(pwd)/$PARENT";; esac
[ -x "$RUNNER" ] || { echo "Missing executable: $RUNNER" >&2; exit 1; }
for FILE in "$YOLO_MODEL" "$INI"; do [ -r "$FILE" ] || { echo "Cannot read $FILE" >&2; exit 1; }; done
if [ "$FACE" = 1 ]; then [ -r "$FACE_MODEL" ] || { echo "Cannot read $FACE_MODEL" >&2; exit 1; }; fi
N=$FIRST
while [ "$N" -le "$LAST" ]; do
    [ -r "$DATA_DIR/data$N.gray" ] || { echo "Cannot read $DATA_DIR/data$N.gray" >&2; exit 1; }
    N=$((N+1))
done
umask 022
mkdir -p "$PARENT"
OUT=$(mktemp -d "$PARENT/batch_data${FIRST}_${LAST}_face${FACE}_XXXXXX")
chmod 755 "$OUT"
export HERMES_FILE_OUTPUT=0
cp "$INI" "$OUT/parameters.ini"
printf 'first=%s\nlast=%s\nface=%s\nmax_frames=%s\nrunner=%s\nyolo=%s\nface_model=%s\ndata_dir=%s\nframe_period_ms=133.3333333333333\n' \
 "$FIRST" "$LAST" "$FACE" "$MAX_FRAMES" "$RUNNER" "$YOLO_MODEL" "$FACE_MODEL" "$DATA_DIR" > "$OUT/run.txt"
printf 'video,status\n' > "$OUT/status.csv"
printf 'video,frames_read,frames_fed,fall_callbacks,bedexit_callbacks,first_stable_sdk_index,first_stable_gray_index,first_stable_timestamp_ms\n' > "$OUT/summary.csv"
CHILD=
interrupt() {
    trap '' INT TERM
    if [ -n "$CHILD" ]; then kill -TERM "$CHILD" 2>/dev/null || :; wait "$CHILD" 2>/dev/null || :; fi
    echo "Batch interrupted; partial logs: $OUT" >&2
    exit 130
}
trap interrupt INT TERM
printf 'ground_truth=%s\n' "$GROUND_TRUTH" >> "$OUT/run.txt"
echo "Release batch output: $OUT"
N=$FIRST
while [ "$N" -le "$LAST" ]; do
    RUN="$OUT/data$N"; mkdir "$RUN"
    echo "Starting data$N (YOLO=1, face=$FACE)"
    (
        cd "$RUN"
        exec "$RUNNER" "$DATA_DIR/data$N.gray" \
          --width=800 --height=450 --interval=1 --frame-period-ms=133.3333333333333 \
          --realtime --max-frames="$MAX_FRAMES" --ini="$OUT/parameters.ini" \
          --bg-stability=1 --bg-stability-ms=6000 --yolo=1 --yolo-model="$YOLO_MODEL" \
          --face="$FACE" --face-model="$FACE_MODEL" \
          --write-events --write-analysis --output-prefix="$RUN/data$N"
    ) > "$RUN/runner.log" 2>&1 &
    CHILD=$!
    if wait "$CHILD"; then RC=0; else RC=$?; fi
    CHILD=
    if [ "$RC" -ne 0 ]; then
        printf 'data%s,failed_%s\n' "$N" "$RC" >> "$OUT/status.csv"
        echo "data$N failed ($RC): $RUN/runner.log" >&2
        tail -n 12 "$RUN/runner.log" >&2
        exit "$RC"
    fi
    printf 'data%s,completed\n' "$N" >> "$OUT/status.csv"
    awk -v video="data$N" '
      BEGIN {sdk=gray=stamp="NA"}
      /^Frames read from file/ {read_count=$NF}
      /^Frames sent to SDK/ {fed=$NF}
      /^FALL_COUNT=/ {split($1,a,"=");fall=a[2];split($2,a,"=");bed=a[2]}
      /^BG_STABLE / && sdk=="NA" {split($2,a,"=");sdk=a[2];split($3,a,"=");gray=a[2];split($4,a,"=");stamp=a[2]}
      END {printf "%s,%s,%s,%s,%s,%s,%s,%s\n",video,read_count,fed,fall,bed,sdk,gray,stamp}
    ' "$RUN/runner.log" >> "$OUT/summary.csv"
    tail -n 1 "$OUT/summary.csv"
    N=$((N+1))
done
echo "Batch completed: $OUT"
echo "Summary: $OUT/summary.csv (NA = BG did not become stable)"
if [ "$MAX_FRAMES" = 0 ] && command -v python3 >/dev/null 2>&1; then
    python3 "$ROOT/experiments/yolo_bg/summarize_release_batch.py" "$OUT" --ground-truth "$GROUND_TRUTH"
else
    echo 'Accuracy can be evaluated on the Linux host with summarize_release_batch.py.'
fi
