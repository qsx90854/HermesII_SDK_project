#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
MPP=${MPP:-/home/mark/nn_release/npu_inference/mpp}
OUT=$(mktemp -d /tmp/hermes-npu-tests-XXXXXX)
CXX=${CXX:-g++}
"$CXX" -std=c++17 -Wall -Wextra -I"$ROOT/src" "$ROOT/experiments/yolo_bg/tests/npu_policy_test.cpp" -o "$OUT/policy"
"$OUT/policy"
"$CXX" -std=c++17 -g -DHERMES_ENABLE_NPU_PERSON=1 -I"$ROOT/src" -I"$ROOT/include" -I"$MPP/include" \
 "$ROOT/experiments/yolo_bg/tests/npu_backend_test.cpp" "$ROOT/src/ai/person_detector_npu.cpp" \
 "$ROOT/src/ai/npu_runtime.cpp" "$ROOT/src/ai/face_detector.cpp" -o "$OUT/backend"
"$OUT/backend" "$OUT/driver_double.ty"
echo "Host tests passed (driver double, no hardware inference). Outputs: $OUT"

"$CXX" -std=c++17 -DHERMES_ENABLE_FILE_OUTPUT=1 -Wall -Wextra -I"$ROOT/src" "$ROOT/experiments/yolo_bg/tests/stage_timing_test.cpp" -o "$OUT/stage_timing"
(cd "$OUT" && HERMES_FILE_OUTPUT=1 ./stage_timing)

"$CXX" -std=c++17 -O3 -Wall -Wextra -I"$ROOT/src" "$ROOT/experiments/yolo_bg/tests/bg_pixel_update_test.cpp" -o "$OUT/bg_pixels"
"$OUT/bg_pixels"

"$CXX" -std=c++17 -O2 -Wall -Wextra -I"$ROOT/src" -I"$ROOT/include" "$ROOT/experiments/yolo_bg/tests/identity_blocks_test.cpp" -o "$OUT/identity_blocks"
"$OUT/identity_blocks"

"$CXX" -std=c++17 -O3 -Wall -Wextra -I"$ROOT/src" "$ROOT/experiments/yolo_bg/tests/identity_geometry_test.cpp" -o "$OUT/identity_geometry"
"$OUT/identity_geometry"

"$CXX" -std=c++17 -O2 -Wall -Wextra -I"$ROOT/src" -I"$ROOT/include" "$ROOT/experiments/yolo_bg/tests/identity_spans_test.cpp" -o "$OUT/identity_spans"
"$OUT/identity_spans"
