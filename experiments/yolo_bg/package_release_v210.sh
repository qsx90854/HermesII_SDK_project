#!/bin/sh
# Run on the build host after: make -f experiments/yolo_bg/Makefile.npu release
set -eu
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
OUT="$ROOT/HermesII_TestTool_release_v210"
BIN="$ROOT/build/release_npu"
for FILE in sdk_gray_count_falls_edge libHermesII_sdk.so libHermesII_npu_vendor.so; do
    [ -r "$BIN/$FILE" ] || { echo "Missing release artifact: $BIN/$FILE" >&2; exit 1; }
done
# Refuse stale artifacts rather than label an older SDK as v2.1.0.
strings "$BIN/libHermesII_sdk.so" | awk '$0=="2.1.0" {found=1} END {exit !found}' || {
    echo 'Release SDK does not contain version 2.1.0; rebuild first.' >&2; exit 1;
}
mkdir -p "$OUT/bin" "$OUT/models" "$OUT/config" "$OUT/include" "$OUT/examples" "$OUT/groundTruth" "$OUT/experiments/yolo_bg"
for FILE in sdk_gray_count_falls_edge libHermesII_sdk.so libHermesII_npu_vendor.so; do cp "$BIN/$FILE" "$OUT/bin/$FILE"; done
cp "$ROOT/models/blaze_face_detect_nnp310_128x128.ty" "$OUT/models/"
cp "$ROOT/handoff_yolov7t_people/yolov7t_people_320x320_v2.ty" "$OUT/models/"
cp "$ROOT/Parameter/sdk_gray_count_falls.ini" "$OUT/config/"
cp "$ROOT/include/HermesII_sdk.h" "$ROOT/include/Image.h" "$OUT/include/"
cp "$ROOT/examples/sdk_gray_count_falls.cpp" "$OUT/examples/"
cp "$ROOT/view_gray_events.py" "$ROOT/view_gray_analysis.py" "$OUT/"
cp "$ROOT/experiments/yolo_bg/RELEASE_USAGE_v210.md" "$OUT/USAGE.md"
cp "$ROOT/experiments/yolo_bg/run_release_batch.sh" "$ROOT/experiments/yolo_bg/summarize_release_batch.py" "$OUT/experiments/yolo_bg/"
# Explicit whitelist: never copy videos or previous experiment outputs.
for FILE in "$ROOT"/Bug_Video/Fall_17/groundTruth/*_fall_frameindex.txt; do cp "$FILE" "$OUT/groundTruth/"; done
printf '2.1.0\n' > "$OUT/VERSION"
# Preserve a tester's path configuration when regenerating the package.
if [ ! -f "$OUT/test_paths.conf" ]; then
cat > "$OUT/test_paths.conf" <<'CONF'
# Use absolute paths as seen ON THE BOARD, after mounting NFS.
# Environment variables override these defaults. Set DATA_DIR before running.
DATA_DIR=${DATA_DIR:-}
GROUND_TRUTH=${GROUND_TRUTH:-$PACKAGE_DIR/groundTruth}
OUTPUT_DIR=${OUTPUT_DIR:-$PACKAGE_DIR/results}
# Optional overrides:
# DATA_DIR=${DATA_DIR:-/nfs/test_data/Fall_17}
# OUTPUT_DIR=${OUTPUT_DIR:-/nfs/test_results/v210}
CONF
fi
cat > "$OUT/run_batch.sh" <<'WRAPPER'
#!/bin/sh
set -eu
PACKAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$PACKAGE_DIR/test_paths.conf"
case "${1:-}" in -h|--help)
    echo 'Usage: DATA_DIR=/board/path/to/gray sh run_batch.sh [first=4] [last=17] [face=1] [output_parent]'
    echo 'Set DATA_DIR in test_paths.conf or the environment; GROUND_TRUTH and OUTPUT_DIR can also be overridden.'
    exit 0;;
esac
[ -n "${DATA_DIR:-}" ] || { echo 'Set DATA_DIR to the directory containing dataN.gray (in test_paths.conf or the environment).' >&2; exit 2; }
export DATA_DIR GROUND_TRUTH OUTPUT_DIR
RUNNER=${RUNNER:-$PACKAGE_DIR/bin/sdk_gray_count_falls_edge}
YOLO_MODEL=${YOLO_MODEL:-$PACKAGE_DIR/models/yolov7t_people_320x320_v2.ty}
FACE_MODEL=${FACE_MODEL:-$PACKAGE_DIR/models/blaze_face_detect_nnp310_128x128.ty}
INI=${INI:-$PACKAGE_DIR/config/sdk_gray_count_falls.ini}
export RUNNER YOLO_MODEL FACE_MODEL INI
exec sh "$PACKAGE_DIR/experiments/yolo_bg/run_release_batch.sh" "$@"
WRAPPER
cat > "$OUT/summarize.sh" <<'WRAPPER'
#!/bin/sh
set -eu
PACKAGE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$PACKAGE_DIR/test_paths.conf"
[ "$#" -eq 1 ] || { echo 'Usage: sh summarize.sh /path/to/batch_directory' >&2; exit 2; }
exec python3 "$PACKAGE_DIR/experiments/yolo_bg/summarize_release_batch.py" "$1" --ground-truth "$GROUND_TRUTH"
WRAPPER
cat > "$OUT/README.md" <<'DOC'
# HermesII TestTool 2.1.0

本包供相同 ARM Cortex-A7 / uClibc 板端測試，不需要 HermesII_SDK 原始專案。
SDK 版本為 2.1.0；不包含 .gray。bin/ 內執行檔與兩個 .so 必須放在一起，
可整包搬到任意 NFS 路徑。板端需具備原本相容的 libstdc++、libgcc、uClibc。
不要將本包 ARM 執行檔直接放在一般 x86 Linux 電腦執行。

## 1. 設定影片與輸出位置

下列路徑都是「板端掛載 NFS 後看得到的路徑」，不是電腦端匯出目錄。
例如電腦的 /home/tester/nfs_share 掛到板端 /mnt/test，則設定 /mnt/test/...。

可在每次執行時設定：

```sh
cd /mnt/test/HermesII_TestTool_release_v210
DATA_DIR=/mnt/test/videos/Fall_17 sh run_batch.sh 4 17 1
```

DATA_DIR 必須直接包含 data4.gray、data5.gray 等檔案；資料格式為 800x450、
每 pixel 一 byte、無檔頭的連續灰階影格，每張假設 133.3333333333333 ms。
此測試資料已從原影片每四幀抽一幀，腳本會讀取所有 gray frames，不會再每四幀抽樣。

若不想每次輸入路徑，編輯 test_paths.conf 的 DATA_DIR 行，例如：

```sh
DATA_DIR=${DATA_DIR:-/mnt/test/videos/Fall_17}
```

然後執行 `sh run_batch.sh 4 17 1`。外部 DATA_DIR 環境設定優先。
GT 預設使用本包 groundTruth/，不需要和 .gray 放一起；新資料必須提供對應 GT。
要改 GT 和輸出位置，可設定：

```sh
DATA_DIR=/mnt/other/videos \
GROUND_TRUTH=/mnt/other/groundTruth \
OUTPUT_DIR=/mnt/test/results/v210 \
sh run_batch.sh 4 17 1
```

## 2. 執行測試

```sh
# data4..17，全幀；YOLO 與 Face 都開啟。
sh run_batch.sh 4 17 1
# 所有 data1..18。
sh run_batch.sh 1 18 1
# 僅 data4，全幀。
sh run_batch.sh 4 4 1
# 先跑 120 幀確認環境，短測不計整片準確率。
MAX_FRAMES=120 sh run_batch.sh 4 4 1
# 第三個參數 0：關閉 Face；YOLO 仍開啟。
sh run_batch.sh 4 17 0
```

正式 SDK 禁止寫檔。驗證 shell 會把終端 log 寫到 OUTPUT_DIR
（預設本包 results/）下新建的 batch_... 目錄。
批次腳本使用 --write-analysis，由測試執行檔輸出 dataN.events.json、
dataN.analysis.json 與 dataN.meta.json。
SDK 本身禁止寫檔；--write-files 仍僅供實驗版使用。
單片執行與事件回放操作請見 [USAGE.md](USAGE.md)。
每幀依 133.33 ms 時程送入；讀檔或處理超過期限時不會額外等固定 133.33 ms。
Ctrl+C 會停止目前影片與後續批次；失敗時停止並印出 runner.log 位置。

## 3. 查看結果與準確率

- dataN/dataN.analysis.json、dataN/dataN.meta.json：逐幀分析與基本資訊，使用 view_gray_analysis.py。
- dataN/dataN.events.json：由測試執行檔寫出的跌倒／離床事件，可用 view_gray_events.py 回放。
- dataN/runner.log：啟動版本、設定、事件與 BG_STABLE 訊息。
- summary.csv：影格數、跌倒/離床 callback 次數、首次穩定 SDK/gray index 與 timestamp。
  callback 次數未合併；事件準確率請看 accuracy.csv 的 detected_events、TP/FP/FN。
- status.csv：各影片是否完成。
- run.txt、parameters.ini：本次路徑、選項及演算法設定。
- accuracy.csv：板端有 Python 3 時，完整批次後自動產生。

BG index 從 0 開始；在本腳本每幀都送入時，SDK index 等於 gray index。
NA 表示該影片沒有觀察到 BG 進入穩定。以每四幀抽一幀的資料來說，
對應原 MP4 index = gray index * 4。首次穩定不是持續穩定性監測結果。

若板端沒有 Python 3，可回到 Linux 電腦，在同一包內執行：

```sh
sh summarize.sh /電腦看得到的路徑/results/batch_實際目錄
```

或直接：

```sh
python3 experiments/yolo_bg/summarize_release_batch.py \
  /電腦看得到的路徑/results/batch_實際目錄 --ground-truth groundTruth
```

計分沿用 score_falls.py：先將相鄰間隔 <=15 個 gray 幀的跌倒 callbacks 合併成事件。
每個 GT 區間與任一偵測事件區間重疊即算一次 TP；沒有重疊的 GT 區間算 FN。
只有整段都不與任何 GT 重疊的偵測事件算 FP；命中同一 GT 的額外事件列為
duplicate_events，不另計 FP/FN。同一事件內的逐幀 callbacks 不是獨立事件，
另列 fall_callbacks、detected_events、repeated_callbacks 供核對。
Recall=TP/(TP+FN)，Precision=TP/(TP+FP)。
GT 區間為原 MP4 index，端點含在內；0,0 為無跌倒標記。
離床事件只統計次數，沒有對應離床 GT 評分。
完整準確率必須跑完所有幀，不能使用 MAX_FRAMES 截短測試。

## 檔案與完整性

bin/：板端執行檔及兩個必要 SO；models/：YOLO 與 Face .ty 模型；
config/：測試參數；groundTruth/：原 MP4 跌倒區間；
include/、examples/：公開 API 與測試範例原始碼；
experiments/yolo_bg/：批次腳本與 Python 3 統計工具。
SHA256SUMS 可在 Linux 電腦執行 `sha256sum -c SHA256SUMS` 檢查交付檔案。
test_paths.conf 為使用者可修改設定，未納入雜湊。
DOC
chmod 755 "$OUT/bin/"* "$OUT/run_batch.sh" "$OUT/summarize.sh"
(cd "$OUT" && find bin models config include examples groundTruth experiments -type f -exec sha256sum '{}' \; > SHA256SUMS && sha256sum VERSION README.md USAGE.md view_gray_events.py view_gray_analysis.py run_batch.sh summarize.sh >> SHA256SUMS)
echo "Release package: $OUT"
