#!/bin/sh
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../../.." && pwd)
OUT=$(mktemp -d /tmp/hermes-background-api-XXXXXX)
cd "$ROOT"
${CXX:-g++} -std=c++17 -O1 -pthread -DDISABLE_NPU -DHERMES_ENABLE_PERSON=1 -DHERMES_BG_STABILITY=1 -Isrc -Iinclude \
 src/HermesII_sdk.cpp src/fall/fall_detector.cpp src/fall/person_bg_controller.cpp \
 src/ai/face_detector.cpp src/ai/model_runner.cpp src/image_process/image_process.cpp \
 src/fusion/image_fusion.cpp src/event_recorder.cpp experiments/yolo_bg/tests/background_api_test.cpp -o "$OUT/test"
mkdir "$OUT/work"
(cd "$OUT/work" && "$OUT/test")
echo "API test output: $OUT"
