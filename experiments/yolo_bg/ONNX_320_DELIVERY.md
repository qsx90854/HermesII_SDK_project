# 320 model experiment

The ONNX backend now reads static float [1,3,H,W] input dimensions and accepts the two-class [1,6,N] output. 640/8400 and 320/2100 use the same letterbox, class-aware NMS and coordinate restoration. SDK input is still 800x450.

A tracking and history-two confirmation, BG gating, thresholds, 4-second inference interval and 2-second simulated delay are retained. Full cohort: data4-16, stride4/8, with data14 scored only on the verified 6066-frame prefix. Text diagnostics only; no new input bins or replay PNGs.

New executable: fall_eval/build_identity_a_320/fall_eval_runner.exe. READY.json and READY_320.json pin it. New launcher: experiments/yolo_bg/full_identity_a_320.py. Model snapshot: models/hermes-v1/yolov7t_HermesII_v1_320x320.onnx.

Validation: one data16 frame compared with Python ONNX Runtime (one box, maximum coordinate difference 0.160 pixel, score difference 0.000506); old 640 probe CSV identical to the previous executable on the same frame. Existing A and history-two unit tests passed. data16 stride4 first1200 source frames ran successfully with 10 successful YOLO results and no inference failures. These checks do not establish full-model accuracy; see the full experiment summary when complete.

Validation folder: fall_eval/results/onnx320_validation_20261011_210845_821586.

Backup of existing src/ai/person_detector_onnx.cpp and experiments/yolo_bg/READY.json, retaining uncommitted content: D:\HermesII_202610\backups\onnx_320_20261011_210728_422546. Restore only these relative paths if needed, preserving later edits. New files can be removed separately. No Git commit or branch switch.

Full result folder: D:\HermesII_202610\fall_eval\results\identity_a_320_all_20261011_211015_203868.
