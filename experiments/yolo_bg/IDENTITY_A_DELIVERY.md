# Identity A port

Current branch: `experiment/bg-stability`. Source: `experiment/long-fg-local-activity:experiment_snapshot/fall_eval/fg_association.h` and A ownership rules. This is an adapted port, not a complete branch merge.

## Scope

- Instance-local, per-frame motion/FG association and stable track ownership.
- Mature-track priority (600 ms and at least 3 independent supported observations); supported motion fragments can share the prior ID.
- Competing mature owners remain ambiguous and pause fall accumulation; conflict footprint grace is 2 seconds.
- Supported coasting cannot enlarge its saved pixel footprint; unsupported prediction expires after 700 ms, and absence of actual motion is capped at 30 seconds.
- Replaces legacy TrackObjects, ghost injection, post-tracking merging, nearest-FG recovery, and cross-ID observation inheritance only when enabled.
- Current YOLO-assisted BG foreground is the support source. No long/short BG, B/D/E, pixel temporal history, or local-activity changes were imported. Historical JSON fields `long_ids` identify the current raw FG components, not a separate long BG.
- Retains the current YOLO model and history-two confirmation. Tracking maturity is not a person classifier.

## Enable

Default is OFF. The generic launcher explicitly sets the flag, avoiding inherited environment settings:

```powershell
python -X utf8 D:\HermesII_202610\HermesII_SDK_project\experiments\yolo_bg\run_hermes_v1.py --identity-a --videos data16 --strides 4 --limit 2800
```

Omit `--identity-a` for the current pre-port algorithm. Direct SDK processes can set `HERMES_IDENTITY_A=1` before SDK construction. No public SDK ABI was changed. Do not assume the existing `full_hermes_v1.py` automatically enables A.

With track replay enabled, `identity_a.jsonl` records maturity, current FG ownership, conflicts, reacquisition, and expiry. The existing `view.py` displays these records alongside motion/FG IDs. Without track replay, candidate/event JSON serialization and file output are disabled.

## Verification

Independent Release build: `D:\HermesII_202610\fall_eval\build_identity_a`.

- C++ unit tests passed for maturity, fragment retention, bounded coasting, unsupported expiry, competing owners, tentative ID rejection, and clock rollback.
- Existing history-two confirmation tests passed.
- data16, stride4, first 2800 source frames (700 samples): TP=2, FP=0, FN=0; event starts 940 and 2312.
- OFF outputs match the prior executable's saved run on every frame, excluding process time; track replay rows also match exactly.
- ON has 7 distinct IDs versus OFF's 12; legacy ghost rows 0 versus 19. These counts alone do not prove every merge is correct.
- Five UI modes rendered successfully and A ownership text was present.
- 549 conservative position-uncertainty events remain in this clip. This is not a full-data accuracy or performance validation.

Results and `OPEN_UI.ps1`: `D:\HermesII_202610\fall_eval\results\fall_safety_20261011_112848_201004`.
`validation.json` contains metrics and checks; `delivery.diff` compares existing files with the pre-edit backup. READY.json pins the new executable and source hashes.

## Backup and restore

Verified pre-edit copies, including pre-existing uncommitted changes:
`D:\HermesII_202610\backups\identity_a_port_20261011_112228_263606`.

Existing files changed: `src/fall/fall_detector.cpp`, `experiments/yolo_bg/run.py`, `view.py`, `CMakeLists.txt`, `READY.json`.
New files: `src/fall/identity_a.h`, `experiments/yolo_bg/identity_a_test.cpp`, this document.

For rollback, restore only those existing files from the same relative paths in the backup and remove the new A files, after checking that no later edits need preservation. For a behavioral comparison, simply leave A disabled. No commit, stash, branch switch, or full experiment was performed.
