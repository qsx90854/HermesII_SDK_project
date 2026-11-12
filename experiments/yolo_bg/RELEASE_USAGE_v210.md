# v2.1.0 測試執行檔與事件檢視工具

本版包含身分確認前觸發證據保留（最多 4 秒）及底部 40 列 FG 像素占比過半的下緣判斷。
SDK 編譯時使用 HERMES_ENABLE_FILE_OUTPUT=0，不輸出診斷、錄影或分析檔；即使設定
HERMES_FILE_OUTPUT=1 也無法啟用。終端訊息仍可能顯示。
.events.json 由 sdk_gray_count_falls_edge 的事件 callback 處理程式寫出，不由 SDK 寫出。

## 板端批次測試

在板端進入此套件目錄，指定包含 dataN.gray 的路徑：

```sh
DATA_DIR=/nfs/videos/Fall_17 sh run_batch.sh 4 17 1
```

最後參數 1 啟用 Face；YOLO 與 BG 穩定機制均開啟。所有 gray frame 都會送入 SDK，
依 133.3333333333333 ms 排程。也可把 DATA_DIR 設於 test_paths.conf；
OUTPUT_DIR 可改變結果儲存位置，預設為套件 results/。
每片事件位於 results/batch_.../dataN/dataN.events.json。

## 板端單片測試

以下從套件根目錄執行；輸出目錄需事先建立且可寫：

```sh
mkdir -p results/single_data9
./bin/sdk_gray_count_falls_edge /nfs/videos/Fall_17/data9.gray \
  --width=800 --height=450 --interval=1 \
  --frame-period-ms=133.3333333333333 --realtime \
  --ini=config/sdk_gray_count_falls.ini \
  --bg-stability=1 --bg-stability-ms=6000 \
  --yolo=1 --yolo-model=models/yolov7t_people_320x320_v2.ty \
  --face=1 --face-model=models/blaze_face_detect_nnp310_128x128.ty \
  --write-events --output-prefix=results/single_data9/data9
```

- `--max-frames=120`：可加上此參數進行短測；省略時跑完整檔案。
- `--write-events`：啟用測試程式的事件 JSON 輸出，不需要任何 SDK 寫檔環境變數。
- `--output-prefix`：輸出檔名前綴；範例產生 data9.events.json。
  省略時寫在輸入 .gray 旁邊，需有該目錄寫入權限。
- 省略 `--write-events` 時不輸出事件檔；正式版不要使用 `--write-files`。
- 即使沒有事件，正常結束也會產生 events 空陣列；跌倒／離床的連續 callback 會逐一記錄。
- 執行檔與兩個 .so 必須保留在同一個 bin 目錄，模型路徑必須指向板端可讀的檔案。

## 電腦端查看事件

使用有桌面視窗的 Python 3 環境，安裝 NumPy 與具 GUI 支援的 OpenCV
（例如在虛擬環境執行 `python3 -m pip install numpy opencv-python`）。
ARM 執行檔在板端跑；Python 檢視器在電腦端跑。

```sh
python3 view_gray_events.py /home/tester/videos/data9.gray \
  --events /home/tester/HermesII_TestTool_release_v210/results/single_data9/data9.events.json \
  --interval 1 --fps 8
```

若 JSON 與 gray 同目錄且同名，直接 `python3 view_gray_events.py /path/data9.gray`。
兩個路徑都必須使用電腦看得到的路徑；JSON 裡記錄的板端來源路徑不必相同。
`--interval 1` 指測試程式送入每張 gray，不是原始 MP4 的抽樣倍率，請勿填 4。
目前檢視器 fps 接受整數，8 是近似播放速度；不影響幀對應與事件判讀。

| 操作 | 功能 |
|---|---|
| 空白鍵 | 播放／暫停 |
| a / d | 前／後一幀 |
| j / l | 前／後 30 幀 |
| n / p | 下一個／上一個事件幀（含連續 callback） |
| 拖曳 frame 滑桿 | 跳到指定幀 |
| s | 截圖，存到 gray 旁邊，需要寫入權限 |
| q / Esc | 離開 |

此工具顯示事件時的物件框及跌倒／離床標記，不能呈現每幀完整計算過程；
.events.json 也不能直接交給 view_gray_analysis.py。
連續 callback 不等於多次跌倒，事件準確率使用 `sh summarize.sh /path/to/batch` 統計。

## 逐幀分析與 meta（與舊格式相容）

批次腳本現在預設加上 `--write-analysis`，同時輸出 dataN.analysis.json、
dataN.meta.json、dataN.events.json。單片命令也可加上 `--write-analysis`；
此選項自動啟用 events 輸出。只用 `--write-events` 則仍只輸出事件 JSON。

```sh
python3 view_gray_analysis.py /home/tester/videos/data17.gray \
  --analysis /home/tester/HermesII_TestTool_release_v210/results/batch_實際目錄/data17/data17.analysis.json \
  --interval 1
```

analysis 保留舊的影像／grid／床區、total_frames、events、frames 欄位。
每幀包含 seq、ts、proc_us、total_fg 與最多 8 個物件：id、cx、cy、dx、dy、
strength、accel、pixels、safe_ratio、dir_var、obs、is_fall、is_bed_exit、fg_area、coasting、blocks。
meta 保留影像／grid／床區、total_frames、analysis_file 與事件清單。
`proc_us` 使用 SDK 內部 Detect() 耗時，與舊記錄器相同；不包含 JSON 寫檔時間。
`ts` 使用本次送入的時間戳，現在每幀約 133.33 ms，並非舊檔的 33 ms。
`seq` 為送入 SDK 的零起算序號；`--interval 1` 時等於 gray index。
`total_fg` 沿用舊版從物件讀取的值；無物件時是 0，不代表整張影像真的沒有前景。

所有 JSON 都由測試執行檔寫出，SDK 不寫檔。分析逐幀寫入 .tmp，正常結束或
Ctrl+C 正常收尾後完成 .analysis.json 與 .meta.json；斷電或強制 kill 時可能只剩未完成的 .tmp。
寫檔有額外 I/O 耗時，若只量測不含分析輸出的整體吞吐，單片測試請省略 --write-analysis。
JSON 欄位順序及空白排版可能與舊版不同，不影響原檢視器讀取。
