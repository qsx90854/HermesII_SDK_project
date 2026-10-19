# NPU 人物／face 整合（2026-10-14）

本次提供獨立 ARM build、NPU 人物後端、非同步 face 和短測入口。2026-10-15 使用者已在板端跑完 data4 的 2054 張，但舊連結方式造成結束時診斷寫檔失敗，批次沒有繼續；新連結版本待板端重新驗證。編譯通過與 host driver-double 測試不代表 NPU 實際時延或辨識精度已驗證。

## 編譯與檔案

在 Linux 專案根目錄：

```sh
make -f experiments/yolo_bg/Makefile.npu -j2
sh experiments/yolo_bg/tests/run_host_tests.sh
```

輸出 `build/npu_people/libHermesII_sdk.so`、`build/npu_people/libHermesII_npu_vendor.so`、`build/npu_people/sdk_gray_npu`（ARM Cortex-A7 / uClibc）。三個檔案需一同部署，放在同一目錄；SDK 透過 `$ORIGIN` 載入 vendor 共享庫。物件檔獨立存放，不覆蓋既有 makefile 的產物。編譯選項：C++17、`HERMES_ENABLE_NPU_PERSON=1`、`HERMES_ENABLE_PERSON=1`、`HERMES_BG_STABILITY=1`。不可同時設定 ONNX 與 NPU 人物後端，也不可搭配 `DISABLE_NPU`。

2026-10-15 連結修正：廠商靜態庫內有 HIDDEN 的 `std::num_put<...>::id`，直接合併進 SDK 時，將 SDK 的同名符號變成私有副本。板端診斷顯示 runner 數值輸出正常、SDK 的 `num_put` 不存在，第一個整數即設 badbit；字串布局／往返檢查與 C stdio 均正常。現將原本同一組靜態庫隔離至 `libHermesII_npu_vendor.so`，用 `npu_vendor.exports` 僅輸出 `TY_*`／`FH_*` C API；SDK 與 runner 不再合併廠商物件。沒有更換廠商庫版本或板端 C++ runtime。以 `-z defs` 檢查共享庫的未解決引用；若未來新增 NPU 呼叫，需將對應入口補進 Makefile 的 `VENDOR_API`。

新版 ARM 編譯／連結及符號檢查已通過，SDK 的 `num_put::id` 已恢復 WEAK DEFAULT。板端先執行 `sh experiments/yolo_bg/run_abi_diagnostic.sh`，全部 PASS 後，再執行 `sh experiments/yolo_bg/run_npu_short.sh data4 1 120`，確認人物 CSV 有資料、summary 為完整 JSON、正常結束。此修正尚未取得板端執行結果。

- `src/ai/person_backend.h`：通用 Detector factory；Windows 的同步離線 RunSync 不進入 NPU 路徑。
- `src/ai/person_detector_npu.*`：.ty 載入、MMZ、cache、非同步提交／輪詢、結果所有權。
- `src/ai/person_postprocess.h`：`[1,6300,7]` 解碼、逐類別 NMS、左上 letterbox 座標還原。
- `src/ai/npu_runtime.*`：共用 runtime 與最早到期優先的單筆 NPU admission。
- `src/ai/face_detector.*`：整合 build 使用 TrySubmit/TryGetResult；其他 build 保留同步 Detect。
- `src/fall/fall_detector.cpp`：face 結果快取、裁切資訊、提交時序；共用人物流程不再只受 ONNX 巨集控制。

## 排程與行為

所有 SDK/NPU client 操作須由同一條執行緒串行呼叫。本版本不建立 worker，也不支援多個 SDK instance 分別由不同執行緒操作同一 NPU。

- 兩模型可常駐，但整個 SDK 同時只有一筆 NPU 任務。僅記錄待執行的到期時間，不排隊保存舊影像。
- face 在 frame 開頭非阻塞收取結果，保留原裁切、128×128 縮放與垂直翻轉。提交延後至 CPU 偵測流程末端。每筆保存其來源時間與裁切座標；最近有效結果最長保留 2500 ms，過期或停用後清除人臉標記。
- YOLO 維持 controller 原有位置收取結果／更新 BG，再提交本幀影像。BG 仍在本幀偵測後更新；歷史框與移動後的保護框仍各自保存。
- NPU 忙碌而尚未接受請求時，下個 frame 用最新影像重試；已送出且尚未取回的請求不另排隊。
- 正常 frame 不阻塞等待推論；初始化、SetConfig/SetBedRegion、來源時間／尺寸重設與 Release 是明確邊界，可以阻塞排空在途工作。不要把這些邊界操作當成穩態 frame 耗時。
- stream 重設會排空／丟棄舊結果，再清除歷史與快取；因此不會把舊結果套到新 stream，無需讓 vendor task 跨 stream 存活。
- 推論失敗不是無人。Query 錯誤會明確報錯；不能確認任務結束時保留資源，禁止重用正在 DMA 的 buffer 或直接 SysExit。這種故障需停止該 SDK 工作流程並處理板端狀態，不能吞掉例外後繼續使用。
- 本版本的 NPU `inference_ms` 是提交至下一次成功輪詢的牆鐘時間，包含輪詢間隔，屬完成延遲上界，不能視為純硬體 kernel 時間。

`SysInit/SysExit` 由 runtime 共用管理；最後一個 model client 釋放後退出。如果宿主已擁有初始化，必須在建立 SDK **之前**設定 `HERMES_NPU_EXTERNAL_RUNTIME=1`，由宿主在全部 SDK client 釋放後退出。不得與另一套未協調的 face/runtime 管理混用。

## 模型與座標

目標是交接包的 `yolov7t_people_320x320_v2.ty`：AIPP RGB planar uint8、320×320、LEFT_TOP、padding 128、/255；輸出必須為 float32 `[1,6300,7]`。兩類都作為人物證據，不直接產生跌倒事件。

SDK 傳 packed gray，後端複製到自己的三個 MMZ 平面。原 buffer 可立即重用；NPU 輸入／輸出在任務完成前保持不變。`picWidth/picHeight` 是提交影像尺寸，不是固定 320。

後端將框除以縮放比例並裁切至提交影像邊界。頂列排除時，800×450 / 16 rows 會提交 800×422；controller 再補回 28 pixel。此尺寸與硬體 AIPP 的 stride／縮放細節仍需板端短測確認。

Windows 舊模型是 `[1,6,N]`，且使用置中 padding 114；因此此次同時更換模型與前處理，不能宣稱偵測結果與 Windows 完全相同。NMS 明確設定，不依賴範例預設值。

## 板端短測（本次未執行）

板子完成既有 NPU driver/clock 初始化並可存取 NFS 後，在板端 checkout 執行：

```sh
# data15 前 120 張，約 16 秒；先只跑 YOLO
sh experiments/yolo_bg/run_npu_short.sh data15 0 120
# 同一段，打開非同步 face
sh experiments/yolo_bg/run_npu_short.sh data15 1 120
```

入口限制 1～300 張，預設 120 張；不會自動批次跑全部 data。每次建立獨立 `build/npu_people/short_*` 輸出目錄，保存 INI、run.txt、runner.log、frames.csv、events.json 與人物 BG 診斷 CSV。原始 gray/GT/Parameter 不被修改。

固定設定：

- packed 800×450 gray，`--interval=1`，timestamp = round(gray_index × 1000 / 7.5)。
- `--realtime` 依牆鐘 sleep_until 餵圖；如果 CPU 太慢會記錄 lag，不偷偷丟幀。無此選項是快速離線回放，不能用來評估實機非同步結果時序。
- YOLO interval=1600ms，async 真實完成，沒有 800ms sleep。delay=800ms 仍參與既有 freshness 上限（共 3400ms）。
- confidence=.25，NMS=.7，Identity A / BG protection v2 / history-two / person confirmation / top-row exclusion / BG stability 開啟，coast-validity 關閉。
- face 頻率取 INI（目前 8 張，約 1066.67ms），只有第二個測試開啟 face。

短測檢查：實際有成功的 person results、框位置與 BG 保護對得上、正常幀不被 NPU Forward 卡住、face callback 正確、停止後 MMZ 無新增殘留。`frames.csv` 的灰階索引是 0-based，groundTruth 仍用原 MP4 索引，可將 gray_index 乘 4 比對。events.json 內原有 SDK index 不應直接假設與 gray_index 相等。

後續全測交給使用者切換輕量模型後執行。短測不是跌倒準確率驗收；data15 的前 120 張也尚未到標註跌倒開始位置。

## 板端批次回放

在板端專案根目錄執行（每支影片獨立啟動 runner）：

```sh
# data4～17 全部 frame，face 關閉
sh experiments/yolo_bg/run_npu_batch.sh
# data4～17 全部 frame，face 開啟
sh experiments/yolo_bg/run_npu_batch.sh 4 17 1
# 指定符合現有輸入／輸出格式的輕量模型
sh experiments/yolo_bg/run_npu_batch.sh 4 17 0 /path/to/light_model.ty
```

參數依序為起始影片、結束影片、face 0/1、人物模型路徑；未指定模型時先取 `HERMES_PERSON_MODEL`，再使用交接資料夾的原模型。固定 `--max-frames=0 --realtime`，讀到檔案結束，仍依 133.33ms 時間表送幀。啟動前檢查所有輸入存在，失敗時停止後續影片。

結果存至 `build/npu_people/batch_*`，內有本次設定 `run.txt`、完成狀態 `status.csv`，及各影片自己的 INI、runner.log、frames.csv、events.json 和人物診斷。輸出目錄開放讀取權限，讓 PC 可讀 NFS root-squash 產生的資料；擁有者仍可能是 nobody，需要修改／刪除時可在 PC 使用 `sudo chown -R mark:mark <輸出目錄>`。本次僅檢查腳本，沒有執行批次資料測試。

## 板端 ABI／數值寫檔診斷

在 PC 編譯：`make -f experiments/yolo_bg/Makefile.npu diagnostic`。
此目標也會在實際 `libHermesII_sdk.so` 加入三個 C 診斷入口；資料由純數值結構傳遞，檢查布局一致後才測試跨邊界字串。

板端專案根目錄執行：

```sh
sh experiments/yolo_bg/run_abi_diagnostic.sh
```

不用啟動 SDK／NPU 或讀取影片。結果保存在 `build/npu_people/abi_diag_*/diagnostic.log` 與 `files/`，不覆蓋先前測試結果。

- 比對 runner 與真正 SDK `.so` 的 `sizeof(std::string)`、alignment、ImageRelated 設定大小及字串欄位 offset、motion/fall 設定大小、stream 大小、C++ ABI 與編譯標頭版本。ARM 與 PC 的大小可以不同，重點是同一板端程式兩側一致。
- 布局一致才測短字串、長字串及 ImageRelated 設定，讓 SDK 修改後回到 runner 驗證；不一致則跳過並報 FAIL。
- 各自在 runner／SDK 內測預設 ofstream、classic locale ofstream、ostringstream、C stdio。輸出固定文字、整數、64 位元整數、小數；逐步記錄 stream state 和數值 locale facet 是否存在，並由 C 讀檔確認實際內容。
- 顯示實際載入的 SDK 與 `/proc/self/maps` 中 C++ runtime 路徑。`glibcxx_date` 是編譯標頭版本，不能當作板端動態庫版本。

`MISMATCH` 表示編譯布局／選項不同，需先統一；字串往返 FAIL 則表示跨邊界操作異常。若 C stdio 正常、stream 在第一個整數時變成 bad/fail，則將問題縮小到 C++ stream／locale 路徑；若只有 SDK 一側失敗，需查 DSO 邊界與符號解析。這些是定位線索，不能單憑它斷言 runtime 已損壞。全部 PASS 只代表 SDK 初始化前的小型測試正常，仍不能排除後續運行時的記憶體覆寫。

## 本機驗證

### 2026-10-16：CPU 路徑七項精簡

- `ENABLE_PERF_PROFILING` 預設跟隨 `ENABLE_DEBUG_PRINT`（目前為 0）；關閉時兩套計時統計均不執行。可在編譯時指定 `-DENABLE_PERF_PROFILING=1` 恢復計時，輸出仍由 debug 開關控制。
- 移除只有寫入、沒有讀取的 `object_y_history_buffer` 和 `object_safe_ratio_history_buffer`。
- Identity A 啟用時省略 legacy `previous_objects` 深複製，以及 ghost injection 的 `current_ids` 集合建構。
- 人物 BG 啟用時省略僅供 legacy BG 使用的 `block_fg_scores_` 複製；本幀實際計算用的 scores 保留。
- BG stability 在區域 mask 重建時快取有效像素數，ready 後直接使用；reset、尺寸及時間戳中斷仍依原流程重建。
- motion predictor 只保留上一組向量；影像歷史與 SAD 計算保持原樣。

短測：ARM SDK／runner／ABI diagnostic 編譯連結通過；NPU/face driver-double 測試通過；profiling 開啟分支通過語法編譯。BG 原有邊界測試通過，另以修改前版本比對 240 次更新，涵蓋 latch、reset、尺寸切換、時間戳中斷、空 mask 及頂端排除區域。

CPU 差異測試使用 data6 前 260 幀、133.333333 ms 來源時間、face 關閉。分別比較人物 BG + Identity A 與 legacy BG + legacy tracking，人物模型使用固定的離線測試回覆（包含兩類人物及空結果），不是實際 YOLO 推論。前後逐幀 fall flag、事件 JSON、追蹤紀錄、BG/FG PNG 一致；人物路徑有 531 個檔案、legacy 路徑有 520 個檔案逐 byte 相同。frames CSV 排除耗時／wall lag，人物 summary 排除 controller 耗時後亦一致。

驗證紀錄與前後來源快照：`build/npu_people/edge_opt_validation_20261016/`。人物路徑未產生跌倒事件；legacy 路徑產生 30 筆事件紀錄，前後內容相同。這不能取代完整 TP/FP/FN 回歸；沒有執行全資料測試或板端 NPU 推論。實際加速幅度需在板端量測；非同步回覆落在哪一幀仍可能隨執行速度改變。

- ARM build：人物 NPU、非同步 face、SDK shared library 及 gray runner 編譯／連結。
- Host policy test：最早到期、取消、單筆限制、兩類人物、NMS、左上補邊、裁切座標、NaN 和空結果。
- Host driver-double test：輸入複製／所有權、未完成拒收、timestamp/request ID、無人成功、提交／輪詢錯誤、Shutdown 排空、錯誤模型規格、共享 runtime、face 翻轉與雙模型 admission。這不是板端推論。
- CPU runner：data15 前 60 張，驗證新 CLI、輸入張數、時間戳及輸出紀錄；未啟用 NPU。
