# Hermes v1 + history-two delivery

Model snapshot: models/hermes-v1/yolov7t_HermesII_v1.onnx; output [1,6,8400], class0 normal_person/class1 fallen_person. Both support BG protection and fall confirmation, neither directly triggers a fall. Gray input replicated to RGB, letterbox640, confidence0.25, class-aware NMS0.7.

HERMES_PERSON_HISTORY_TWO=1 checks the latest two completed successful inferences at candidate creation, any matching class confirms. No observation/stationary cutoff, no support/departure or mixed FG/motion displacement veto. If neither matches, wait for the next successful capture >=candidate timestamp; failure is unknown; negative clears the episode; timeout10s; lost track cancels. Existing overlap thresholds remain 50% of candidate and15% of person. Old confirmation remains available with the flag off. Coasting validity is OFF in the new entry, top-row exclusion ON. Cross-ID deduplication/track transfer is not added.

Build: fall_eval/build_hermes_v1_history2, Release O3 NDEBUG. Previous executables preserved. READY.json stores final source/model/executable hashes.

Short test: data16 first2800 original frames, stride4 (700 SDK inputs), two complete GT events inside this prefix. TP2 FP0 FN0, onset940/2260. This is not full dataset accuracy. 23 successful applied YOLO results, 0 failures; one in flight at clip end. Five UI modes verified; history-two unit tests passed; prior-model format still loads. Two C++ vs supplied Python reference image checks: identical classes/counts, maximum coordinate difference0.56pixel and score difference0.0115 (resize rounding). No claim of bit-identical preprocessing.

Short-test results: D:\HermesII_202610\fall_eval\results\fall_safety_20261010_193500_727806

Full experiment entry prepared but NOT executed (data4-16, strides4/8; data14 first6066 frames):

    D:\Lightglue\lg_env\Scripts\python.exe -X utf8 D:\HermesII_202610\HermesII_SDK_project\experiments\yolo_bg\full_hermes_v1.py

UI uses OPEN_UI.ps1 inside short-test results. New person_confirmation_checks.csv records history/next request IDs, source/application times, class, score and both overlap ratios. Existing track log shows person_confirmed_history/person_confirmed_next or cancellation.

Backup: D:\HermesII_202610\backups\hermes_v1_history2_20261010_192922_259594
Restore only listed touched files from the matching relative paths, after preserving any later edits. New files: run_hermes_v1.py, full_hermes_v1.py, history_two_test.cpp, model_parity_probe.cpp, this document; remove only if no later modifications depend on them. Model snapshot and independent build are additional artifacts. No commit/reset/stash performed.

Review: bounded lightweight reviewer checked parser/history path. A claimed rejection-latch issue was checked against fall_detector.cpp new-trigger reset and was not a defect. Existing tracked fall_detector.cpp has pre-existing trailing whitespace reported by git diff --check; it was not changed in this task. Diff against the timestamp backup is fall_eval/hermes_v1_delivery.diff.
