#!/bin/sh
# Full pipeline with eligibility, retained-trigger and bottom-occupancy diagnostics.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
case "${1:-}" in -h|--help)
 echo "Usage: sh $0 [first=4] [last=17] [face=1] [person_model.ty]"
 echo 'Runs every gray frame with YOLO, BG stability, eligibility/reason/identity traces.'
 exit 0;;
esac
[ "$#" -le 4 ] || { echo 'Too many arguments' >&2; exit 2; }
export HERMES_ELIGIBILITY_TRACE=1 HERMES_FALL_REASON_TRACE=1 HERMES_TRACK_REPLAY=1
if [ "$#" -eq 4 ]; then
 exec sh "$ROOT/experiments/yolo_bg/run_npu_batch.sh" "$1" "$2" "$3" "$4"
else
 exec sh "$ROOT/experiments/yolo_bg/run_npu_batch.sh" "${1:-4}" "${2:-17}" "${3:-1}"
fi
