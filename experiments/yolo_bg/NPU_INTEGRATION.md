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

## 分段耗時計量（2026-10-19）

預設關閉，不需要開啟舊的 `ENABLE_PERF_PROFILING`。板端短測：

```sh
HERMES_STAGE_TIMING=1 sh experiments/yolo_bg/run_npu_short.sh data4 1 120
```

正常結束後，同一輸出資料夾新增 `algorithm_timing.csv` 與 `bg_controller_timing.csv`（人物 BG 啟用時）。在 PC 統計單次輸出或整批目錄：

```sh
python3 experiments/yolo_bg/summarize_stage_timing.py build/npu_people/short_data4_face1_XXXXXX --output /tmp/stage_timing_report.md
```

主流程的 `total_ms` 是 `FallDetector::Detect` 的時間，與 runner 的 `SetInputMemory + ProcessNextFrame` 範圍不同。報表依累積耗時排序，列出攤每幀平均、有進入該區段影格的平均、全幀 P95、最大值與 frame index／timestamp；並依 person events 分開列出套用 BG 更新及其他影格。不同 CSV 的 frame index 不保證一致，跨檔以 timestamp 對照；重複 timestamp 或重設串流應分開測試。

主流程互斥區段：

- `input`：頂端排除複製、輸入／timestamp 準備。
- `face_prepare_poll`：Face 收取與輸入準備；提交另列 `face_submit`。不包含非同步 NPU 等待期間。
- `gray_geometry`：尺寸／mask 準備、灰階轉換。
- `bg_mask_unknown`：未知背景區填補與必要的背景配置。
- `foreground`：前景周邊處理與啟用時的 FG replay 寫檔；子工作另列 `fg_mask`（差分）、`fg_components`（連通區域）、`fg_sort`（物件排序）。
- `motion`：運動估計、床區遮罩及追蹤前整理。
- `tracking`：物件抽取／關聯、Identity A 或 legacy tracking。
- `pixel_stats`：物件像素與區域統計。
- `bg_stability`：背景穩定判斷及暖機狀態清除。
- `safety_history`：安全檢查準備、FG 關聯與物件歷史。
- `fall_logic`、`post_logic`、`confirmation`：跌倒判斷、後處理與人物確認。
- `bg_controller`：人物 BG Step 或 legacy BG 更新。
- `trace_output`、`debug_csv`、`callback`：追蹤輸出、每物件 debug CSV 計算與寫入／flush、事件 callback（含 runner 的 events JSON 重寫）。
- `tail`：函式末尾清理／診斷。

BG 明細互斥區段：`setup`、`maintenance`（保護區維護）、`pending_change`（等待結果時比對來源影像）、`poll`、`apply_result`（人物關聯與保護區套用）、`region_masks`、`block_policy`、`pixel_update`（各可更新 block 的背景寫入）、`block_log`、`event_log`、`submit`（含來源複製）、`tail`。`maintenance`／`apply_result` 仍包含原有 protection／box log，不能當作完全不含 IO 的純算法數字。離線模式的 `submit` 也包含同步模型執行，不能拿來估計板端非同步提交成本。

使用 `steady_clock` 牆鐘，會包含被作業系統搶占的時間。每幀只寫預先配置的記憶體，Release／Finish 才輸出；強制終止可能沒有計時檔。預設每個 recorder 保留最多 4096 幀（兩個 recorder 約 3.3 MiB row storage），超過後 `dropped_frames` 明確記錄漏收數；可用 `HERMES_STAGE_TIMING_MAX_FRAMES` 設為 1～100000。每支影片獨立程序，所以目前 data4～17 均不會超過預設容量。

每個 CSV 中子區段與父區段採排他計時，區段和等於 total；不要再加 total，或把 BG 明細再次加到主流程。`*_calls` 是區段進入次數，包含巢狀計時返回，不是更新次數；`pixel_update` 一幀可能進入多次，代表多個 block。分支未執行時也可能有極小的條件判斷時間。總計包含少量計時器成本，應另以未開計時的測試量測實際產品速度。

此版驗證：ARM SDK／runner／diagnostic 編譯連結通過；主機 NPU/face driver-double 及計時器的關閉、巢狀排他、容量限制、丟棄計數與輸出重設測試通過。data6 前 120 幀以前後版本比較，開啟計時後人物路徑 251 個檔案、legacy 路徑 240 個檔案逐 byte 一致（事件、追蹤、BG/FG 等），逐幀判斷亦一致；耗時欄位排除比較。人物回覆使用固定離線測試資料、Face 關閉，非板端 NPU 效能測試。報表程式已驗證兩條路徑的區段加總。紀錄在 `build/npu_people/stage_timing_validation_20261019/`；其中主機耗時含 FG replay PNG 輸出，不能當作板端效能數據。

## BG 像素迴圈第一階段優化

`bg_pixel_update.h` 將每個 block 分成無效背景的逐列複製與有效背景的混合；混合係數及 `1-alpha` 固定於 block 層級，內層使用列指標。逐列複製使用 `memmove`，亦支援來源與目的完全相同的緩衝區。保留原浮點公式及 `std::lround`，本次未使用固定小數點、近似取整或 NEON 改寫。

ARM 編譯連結通過。檢查實際 controller 物件檔，混合路徑保留原版 `alpha*now` 的浮點乘法，接著以 `(1-alpha)*bg` 做 FMA，再呼叫 `lroundf`；係數讀取與 `1-alpha` 減法移出像素內層。這不是板端執行的逐值測試。

主機測試以原公式比對 983040 組混合，涵蓋預設各階段 alpha、0/1 與其他係數；另測矩形邊界、奇數寬、初始化複製、來源目的相同與空區域。data6 前 120 幀以前後版本、固定離線人物回覆比較：人物路徑 251 個檔案與 legacy 路徑 240 個檔案逐 byte 一致（BG/FG、追蹤、事件等），逐幀 fall 判斷一致。排除計時欄位與計時 CSV，Face 關閉。未跑全資料。

板端仍請開啟 `HERMES_STAGE_TIMING=1` 做短測，對照 `pixel_update` 在更新幀的平均與 P95；不能以主機耗時推算板端加速幅度。先前全測 `batch_data4_17_face1_dWcHEL` 可作為此修改前的基準。

## 追蹤子區段與工作量紀錄

`HERMES_STAGE_TIMING=1` 現在同時輸出 `tracking_timing.csv`，仍採逐幀記憶體暫存、結束後寫檔與相同容量上限。額外配置約 2.35 MiB（4096 幀），只在計時開啟時配置。計數只加總容器的 `.size()`，不逐像素額外掃描，不開啟完整 identity replay。

子區段包含 `extract`、`merge`、`identity_begin`、`associate`、`reconstruct`、`copy_results`、`metadata`、`expired_cleanup`、`stale_cleanup`、`coasting`、`safe_area`、`bed_history`、`legacy_associate`。`workload` 單獨記錄收集工作量的成本；總時間已包含在主流程的 `tracking`，不可重複相加。

工作量欄位：

- `identity_enabled`：Identity A 是否啟用。
- `active_blocks`、`fg_count`、`fg_pixels`、`fg_max_pixels`：追蹤進入前的運動區塊數、前景區域數、區域像素數總和／最大值。
- `extracted_objects`：抽取完成、同幀合併前的物件數。
- `association_objects`、`tracks_before`、`mature_tracks`：Identity A begin 完成後、associate 前的輸入數量。
- `footprint_pixels`、`footprint_max_pixels`：關聯前各 track 的 footprint 大小總和／最大值（可能包含重複像素，不是全畫面聯集）。
- `cost_pairs`：`tracks_before × association_objects`，成本矩陣實際物件配對項目數，並非 gate 通過數。
- `cost_matrix_cells`：`tracks_before × (association_objects + tracks_before)`，包含未匹配用的額外欄。
- `tracks_after`、`assisted_count`、`assisted_pixels`：重建與結果複製後的 track／重建區域／像素量。
- `expired_tracks`、`stale_tracks`：本幀進入各清理流程的 ID 數。
- `objects_after`、`persistent_ids`、`observation_ids`、`object_history_frames`、`bed_history_ids`、`bed_history_samples`：追蹤區段結束時的物件與歷史容器大小。

Identity A 專用計數在 legacy 模式保持 0，請用 `identity_enabled` 區分「不適用」和實際零工作量。時間戳與主流程一致，預設 interval=1 回放時 frame_index 對應從 0 起算的 gray index。重設或跳幀回放時應以 timestamp 對照。

`summarize_stage_timing.py` 已支援第三份 CSV，列出追蹤子區段排名、最慢 10 幀與同幀工作量。若要定位 data4 最後的 2040～2053 幀，需從頭跑以保留追蹤歷史：

```sh
HERMES_STAGE_TIMING=1 sh experiments/yolo_bg/run_npu_batch.sh 4 4 1
```

驗證：ARM SDK／runner／diagnostic 編譯連結成功；主機計時器測試涵蓋工作量欄位輸出、容量丟棄與重設。data6 前 120 幀固定離線人物回覆、Face 關閉，人物／legacy 兩路徑修改前後 BG/FG、追蹤、事件及逐幀判斷一致；不比較耗時欄位。兩路徑各 120 筆 tracking 計時均通過 frame/timestamp 對齊、子計時不超過主 tracking 時間、工作量矩陣大小與報表加總驗證。未跑全資料，板端尚待使用者回放。驗證紀錄在 `build/npu_people/tracking_log_validation/`。

## Identity A block 計算等價優化

- `mask_blocks()` 改用 grid 大小的 byte 標記，按 block ID 遞增輸出。結果仍是原本 sort/unique 的遞增集合，不再排序每個像素產生的大量重複 block ID。
- `supported_pixels()` 的 block membership 改為標記陣列，替代逐像素 `std::set` 查詢；像素輸出本身仍照原本方式排序去重。
- `begin()` 在幾何改變時建立每像素的 block ID 快取。800×450、32-bit int 約 1.37 MiB；同尺寸不重建，reset 使快取失效。保留原本整數除法與最後 row/column 的 clamp 規則，非整除尺寸也一致。無可用快取或超出像素範圍時仍走原映射公式。

未改動像素 overlap、追蹤門檻、Hungarian 配對、footprint 更新與成熟／coasting 規則；未引入像素集合排序快取，以避免未證明的失效條件改變結果。

驗證：ARM SDK／runner／diagnostic 建置通過。主機 769539 個原始 block 映射比對，涵蓋 800×450、801×451 等尺寸、末端區塊、亂序／重複／空集合、membership、geometry/reset。原 identity_a 測試通過。另以修改前後模型做 24 次大型前景更新（26000～75400 像素），track ID、footprint、重建像素及診斷 JSON 一致。data6 前 120 幀固定離線回覆、Face 關閉，人物／legacy 路徑之 BG/FG、追蹤、事件及逐幀判斷一致，tracking 工作量欄位也一致；排除耗時欄位。沒有跑全資料或板端推論。紀錄在 `build/npu_people/identity_block_validation/`。

板端可用相同指令回放 data4，以 `batch_data4_4_face1_Puhoo5/data4` 為修改前基準，重點比較 `associate`、`reconstruct` 與 gray index 2040～2053；幾何快取建立成本會計入第一幀的 `identity_begin`。

### Identity A 細分計時（2026-10-20）

`HERMES_STAGE_TIMING=1` 現在另存兩份明細，正常 Release 時寫出；沿用
`HERMES_STAGE_TIMING_MAX_FRAMES`（預設 4096），兩份合計額外約 3.2 MiB
列緩衝。關閉計時不配置列緩衝、不讀取時鐘。未改 association／reconstruction
的判定或像素運算；計時有額外成本，數值包含執行緒排程等待。

- `association_detail_timing.csv`：`support_scan` 建立 block membership 與收集
  supported pixels（不含 `support_sort`）；`support_sort` 排序去重；
  `prior_local` 逐點搜尋 detection blocks；`prior_sort` 複製與排序 FG；
  `prior_shared` 搜尋 footprint 交集；`prior_coverage` footprint 轉 blocks 與鄰近覆蓋；
  `cost_matrix` 成本、overlap 與 gate；`assignment` 模糊判定與 Hungarian；
  `fragment_overlap` 額外 fragment 歸屬；`group_tracks` 分組及 track 更新；
  `other` 初始化及其餘控制工作。
- `reconstruction_detail_timing.csv`：`allowed_masks` 允許區域；`fg_histogram`
  FG block 計數；`owner_selection` 所有權判定；`pixel_collect` 收集像素；
  `pixel_sort` 排序去重；`coast_clip` coasting 第一次交集；`position` 位置與速度；
  `independence` block 支援獨立性；`make_fg` 像素複製、bbox 與重心；
  `footprint_write` footprint 寫回／第二次交集；`expiry` 過期判定；
  `assisted_repack` assisted 複製與重新編號；`other` 其餘狀態處理。

各明細內區段互斥；兩份 total 分別已包含在 `tracking_timing.csv` 的
`associate_ms`／`reconstruct_ms`，不可重複加總。以 SDK 的 zero-based gray
`frame_index` 與 `timestamp_ms` 對照；`*_calls` 包含巢狀區段返回次數。
`summarize_stage_timing.py` 會自動彙整新增明細，亦相容舊版輸出。

板端基本短測：

```sh
HERMES_STAGE_TIMING=1 sh experiments/yolo_bg/run_npu_short.sh data4 1 120
```

此短測只驗證前 120 幀，無法涵蓋 frame 2000／2053 的大量前景。
要比較同一段追蹤歷史與相同 frame index，請由使用者執行 data4 全幀：

```sh
HERMES_STAGE_TIMING=1 sh experiments/yolo_bg/run_npu_batch.sh 4 4 1
```

此次驗證：host tests、Identity A maturity/fragmentation/coasting/conflict/reset
測試及 ARM `all diagnostic` 建置通過。24 幀 26,000..75,400 pixels 的修改前後
差異比對，在計時開啟與關閉下，IDs、footprints、assisted pixels 與診斷 JSON
一致；兩份明細皆 24 列、frame index 對應正確、無 dropped rows、分區總和等於
各自 total（CSV 四捨五入容差 0.001 ms）。尚未執行板端測試或完整資料測試。

### Association／reconstruction 與整數 NEON 優化（2026-10-20）

本輪同時修改兩區，保留 thresholds、成本公式、處理順序與 public SDK 型別。

- `associate`：supported pixels 分 FG 篩選、排序檢查與有序聯集；prior local
  改 block membership 查表，累計達到原判斷所需的 20 點即停止（保留重複點計數）。
  FG 已嚴格遞增時直接引用；其他輸入建立排序去重副本，在本次 associate 重用。
  prior shared 以有序集合交集計數，小 query（少於 footprint 的 1/16）保留二分搜尋。
  association 期間快取 footprint blocks，以及 detection/track 的 pixel overlap。
- `reconstruct`：篩選後用有序聯集產生相同 pixels；只有存在其他 mature track
  才建立 independence 所需 blocks。位置與 bbox 共用一次幾何掃描；coasting
  已在前面取過交集，寫回時不再重算；assisted 重整時只對唯一引用的 FG 使用 move。
- `pixel_geometry.h`：ARM NEON 下，width=800、點數<=360000 且 indices 全在
  0..359999 時，以精確整數乘法／位移計算 y，再計算 x，向量化 sums/min/max。
  累加總和在此界限內不溢位，轉 double 後保留原除法順序。任意尺寸、越界索引
  與其他平台退回原順序 scalar double 計算。未使用近似倒數、float 累加或 fast-math。
  可用編譯巨集 `HERMES_DISABLE_GEOMETRY_NEON` 關閉 NEON 供對照。

細分 log 名稱沿用：`support_sort`／`pixel_sort` 現包含有序檢查與合併；
`prior_sort` 包含來源順序檢查及必要時的排序；`position` 現包含共用 bbox/sums
掃描，`make_fg` 主要為 FG 結構及 pixels 複製。因此這兩欄應合併比較。
此次沒有實作 bitmap 交集；先使用列表交集，避免在沒有量測下增加每 track bitmap 成本。

板端先執行精確性短測（不開 SDK/NPU、不讀影片）：

```sh
/nfs/HermesII_SDK/build/npu_people/identity_geometry_test
```

必須顯示 `PASS: NEON geometry equals original double scalar`。這會驗證所有
360000 個 index 的座標轉換、向量路徑、尾端、亂序／重複點、bbox／總和及 fallback。
接著可做 SDK 短測：

```sh
HERMES_STAGE_TIMING=1 sh experiments/yolo_bg/run_npu_short.sh data4 1 120
```

比較 frame 2000/2053 時由使用者執行：

```sh
HERMES_STAGE_TIMING=1 sh experiments/yolo_bg/run_npu_batch.sh 4 4 1
```

主機可驗證 scalar 邏輯和整數除法公式，不能代替實際 ARM NEON 路徑測試；
板端效能收益須以上述實際測試結果判定。

本輪驗證結果：host tests、24 幀大型前景差異比對、160 幀多物件／重疊／亂序／
重複像素／coasting／時鐘重設比對均通過，位置速度與 pixels/bbox/IDs/診斷一致。
120 幀 data6 主機固定 AI 輸出短測中，person 路徑 251 個、legacy 路徑 240 個
非耗時輸出檔逐位元一致，frame 結果、BG summary（排除耗時）、tracking workload
也一致；每份計時的分區總和驗證通過。這是主機 scalar 與模擬 AI 測試，未宣稱板端
NEON 已執行。ARM all/diagnostic（含 identity_geometry_test）建置通過。
驗證原始碼與記錄保存在 `build/npu_people/identity_neon_validation/`。

### 大量 FG pixels 的 span 處理（2026-10-20）

繼 hm8EeK 全測版本，新增 `visit_pixel_spans`／`select_pixels`。相鄰 vector
元素若位於同一列、同一 grid block，以一段處理；每段只做一次 block 計算、
歸屬判定、批次像素複製或直方圖加總。仍保留所有 pixels 與重複點計數，
不是抽點，也不變更解析度、門檻或 BG 更新時序。亂序資料會在跨 cell 時切段，
仍保持原本資料順序；out-of-range membership 採既有 fallback。

- associate supported pixels 與 reconstruct pixel collection 改批次篩選。
- reconstruct FG histogram 用段長累計；footprint mask_blocks 也按 row/block
  段標記，省掉同一 block 的大量重複寫入與 block 檢查。
- 已嚴格遞增的 pixels 一次檢查即完成 unique，不再額外掃描 std::unique。
- masks 經 merge_pixels 已保證有序去重，移除 reconstruct 後續重複 unique。
- 原本精確整數 NEON 座標 kernel 保留；本輪主要減少每點重複工作。

驗證：host tests 與原本 24 幀大型前景／160 幀多物件等差異比對通過。
另以75000、150000、225000、300000、360000點的24幀追蹤測試比對 IDs、
footprints、assisted pixels 與診斷；以及 span helper 的逐點參考測試，覆蓋
完整畫面、不同解析度／grid、亂序／重複／越界點與幾何變更，全部通過。
120幀data6的person／legacy主機固定AI短測，251／240個非耗時輸出檔、frame
結果與tracking工作量一致。ARM SDK建置通過。記錄見
`build/npu_people/fg_span_validation/`。

段數會受前景碎裂程度影響；複雜度仍包含讀取／輸出 pixels，不能保證任意
前景量、物件數都在130ms以內。主機測試不代表板端速度；保持細分計時，
由使用者先以data4全幀對照hm8EeK／YLlujW，再決定是否全資料回歸。

## 正式 API 與不寫檔版本（2026-10-21）

正式版使用獨立目錄編譯，避免混用實驗版 object：

```sh
make -f experiments/yolo_bg/Makefile.npu -j4 release
```

產物在 `build/release_npu/`：`libHermesII_sdk.so`、
`libHermesII_npu_vendor.so`、`sdk_gray_count_falls_edge`。部署時兩個 SO 必須一起帶上。
正式版 `HERMES_ENABLE_FILE_OUTPUT=0`，SDK 與範例不產生 CSV、JSON、TXT、
圖片、raw 或事件錄影檔；舊 sdk.ini、環境變數與 EventRecording 設定無法開啟寫檔。
仍會讀取模型、設定與輸入影片，並輸出終端訊息。外部 shell 若自行使用 `>`，
則由 shell 建立輸出檔，與 SDK 寫檔功能無關。

新的 `BackgroundControl_v1` 加在 ConfigType 最後，未改變 FallDetection_v3
或既有 enum 值。設定必須在 Init 前完成，SDK 複製設定，路徑最多 511 bytes 加 NUL。
執行中 SetConfig 此類型會回傳 ERROR_INVALID_INPUT；Release 後可重新設定、Init。

```cpp
VisionSDK::BackgroundControl_v1 bg;
bg.enable_bg_stability = 1;
bg.bg_stability_window_ms = 6000; // 100..600000
bg.enable_yolo = 1;
std::snprintf(bg.yolo_model_path, sizeof(bg.yolo_model_path), "%s", model_path);
// 呼叫端應先確認 model_path 長度，避免 snprintf 截斷。
if (sdk.SetConfig(&bg) != VisionSDK::StatusCode::OK) { /* 設定錯誤 */ }
// face 的開關仍由 FallDetection_v3 控制；Init 的路徑仍為 face 模型。
if (sdk.Init(face_model_path, 4) != VisionSDK::StatusCode::OK) { /* 初始化失敗 */ }
```

明確呼叫新 API 會覆蓋相關舊環境設定；未呼叫則保留實驗腳本環境設定相容性。
新 API 的 YOLO 使用 async 模式，1600 ms 排程、800 ms 最早套用延遲、
confidence .25 / NMS .7，並啟用目前已驗證的 Identity A、history-two、
person confirmation、top-row exclusion、protection-v2 流程。兩類模型輸出皆視為人。
沒有編入對應功能的 SDK，要求開啟時 Init 會失敗。YOLO 模型載入錯誤回傳
ERROR_INIT_FAILED。Face 仍沿用現有非阻塞實作。

每次成功 ProcessNextFrame 後可查詢：

```cpp
VisionSDK::BackgroundStatus_v1 state;
if (sdk.GetBackgroundStatus(state) == VisionSDK::StatusCode::OK &&
    state.state == VisionSDK::BackgroundState::Stable) {
    // state.stable_frame_index：0-based SDK 輸入幀號
    // state.stable_timestamp_ms：首次穩定的輸入 timestamp
}
```

狀態為 Disabled / WaitingInitialBackground / Observing / Stable。
尚未穩定的 stable_frame_index 為 UINT64_MAX；timestamp=0 本身不代表無效。
has_processed_frame、last_timestamp_ms 表示是否已有處理結果及其時間。
穩定會鎖定；時間倒退、間隔超過 1000 ms、影像尺寸或床區改變時重新觀察。
Release 會清除查詢結果，下次 Init 重新開始。
請在同一處理執行緒依序呼叫，這個查詢不額外提供跨執行緒同步。
SDK index 不一定等於原始 mp4 index；範例同時列出 gray_index。

板端不寫檔短測範例（從專案根目錄執行）：

```sh
./build/release_npu/sdk_gray_count_falls_edge Bug_Video/Fall_17/data4.gray \
  --ini=Parameter/sdk_gray_count_falls.ini \
  --frame-period-ms=133.3333333333333 --realtime --max-frames=120 \
  --bg-stability=1 --bg-stability-ms=6000 --yolo=1 \
  --yolo-model=handoff_yolov7t_people/yolov7t_people_320x320_v2.ty \
  --face=1 --face-model=models/blaze_face_detect_nnp310_128x128.ty
```

移除 `--max-frames=120` 可跑完該檔。範例首次穩定會印
`BG_STABLE sdk_frame_index=... gray_index=... timestamp_ms=...`。
CLI 新設定預設 BG / YOLO 關閉；使用 `--yolo-model` 時也須明確指定 `--yolo=1`。

實驗版仍使用 `make -f experiments/yolo_bg/Makefile.npu all`，輸出位於
`build/npu_people`。寫檔必須編譯 `HERMES_ENABLE_FILE_OUTPUT=1` 並在啟動前設定
`HERMES_FILE_OUTPUT=1`；範例自身輸出另外需要 `--write-files`。
run_npu_short.sh / run_npu_batch.sh 已加上這些實驗專用選項，原指令不變。
run_abi_diagnostic.sh 也明確開啟診斷用寫檔。

驗證：`sh experiments/yolo_bg/tests/run_background_api_test.sh` 使用 CPU model double，
檢查設定驗證、多實例、環境覆蓋、穩定/重置、重啟、模型失敗重試與禁止輸出。
`sh experiments/yolo_bg/tests/run_host_tests.sh` 包含 NPU driver double 與既有精確性測試。
這些 host 測試不代表已完成實際板端 YOLO/Face 推論驗證。

本次驗證結果：ARM 實驗版與正式版編譯成功；host API/lifecycle 與既有測試通過。
data6 前 120 幀在正式寫檔策略下沒有產生輸出檔，BG 首次穩定為 gray 63 / 8400 ms。
使用相同 CPU model double 對照修改前版本，person/legacy 各 120 幀的非耗時輸出
分別 251 / 240 檔逐 byte 相同，frame 結果與 tracking workload 相同。
實際 NPU/Face 完整流程仍需板端短測，未執行全資料測試。

### 正式範例批次驗證

```sh
# 板端：data4..17，全幀、133.33 ms，YOLO 與 Face 都開啟。
sh experiments/yolo_bg/run_release_batch.sh 4 17 1
# 僅 data4，全幀。
sh experiments/yolo_bg/run_release_batch.sh 4 4 1
# 腳本啟動流程短測（短測不計算整片準確率）。
MAX_FRAMES=120 sh experiments/yolo_bg/run_release_batch.sh 4 4 1
```

使用 `build/release_npu/sdk_gray_count_falls_edge`，沒有 `--write-files`。
SDK 不寫檔；外層驗證 shell 會將 stdout/stderr 保存到
`build/release_validation/batch_.../dataN/runner.log`，並產生 `summary.csv`。
`summary.csv` 含首次穩定 SDK/gray 幀號及 timestamp，未穩定填 NA。
第四個參數可指定驗證輸出目錄，例如 `/nfs/validation`。

板端有 Python 3 時，完整測試後自動產生 `accuracy.csv` 並印 TP/FP/FN、
Recall、Precision。板端沒有 Python 時，在 Linux 主機執行：

```sh
python3 experiments/yolo_bg/summarize_release_batch.py build/release_validation/batch_實際目錄
```

GT 的 `0,0` 視為沒有跌倒；原始 MP4 區間與 gray_index*4 比較，不放寬時間窗。
計分沿用 score_falls.py：先將相鄰間隔 <=15 個 gray 幀的跌倒 callbacks 合併成事件。
每個 GT 區間與任一偵測事件區間重疊即算一次 TP；沒有重疊的 GT 區間算 FN。
只有整段都不與任何 GT 重疊的偵測事件算 FP；命中同一 GT 的額外事件列為
duplicate_events，不另計 FP/FN。同一事件內的逐幀 callbacks 不是獨立事件，
另列 fall_callbacks、detected_events、repeated_callbacks 供核對。
Recall=TP/(TP+FN)，Precision=TP/(TP+FP)。
outside_events 即未與 GT 重疊的事件段數（FP），不是 callback 幀數。
這裡只對跌倒事件評分，離床事件仍列於 summary.csv。
腳本可用 RUNNER / DATA_DIR / INI / YOLO_MODEL / FACE_MODEL 環境變數覆寫路徑。
已用模擬 runner 驗證 shell、BG 摘要與重複事件計分，未代跑板端全測。

### v2.1.0 測試人員交付包

SDK GetVersion 現為 `2.1.0`。重建與整理交付包：

```sh
make -f experiments/yolo_bg/Makefile.npu -j4 release
sh experiments/yolo_bg/package_release_v210.sh
```

輸出 `HermesII_TestTool_release_v210/`，包含 ARM 執行檔、必要 SO、兩個模型、
測試 INI、GT、公開 header/範例原始碼、批次與統計工具、使用說明及 SHA256SUMS。
不包含 .gray。搬到任意 NFS 掛載位置後，從包內執行：

```sh
DATA_DIR=/板端看得到的影片目錄 sh run_batch.sh 4 17 1
```

`test_paths.conf` 可保存 DATA_DIR、GROUND_TRUTH、OUTPUT_DIR 的預設值。
GROUND_TRUTH 預設包內 groundTruth，OUTPUT_DIR 預設包內 results；影片可獨立放置。
路徑一律使用板端掛載後的路徑；回主機執行 `sh summarize.sh` 時，批次目錄則使用
主機看得到的路徑。README.md 含完整例子。
已檢查包內無 gray、檔案雜湊、ARM 相依與 $ORIGIN SO 路徑，並用模擬 runner
驗證搬移整包、含空白路徑、外部 gray/output 目錄與 GT 配對；沒有執行板端全測。

2026-10-21 計分修正：最初 release 統計工具誤將逐幀 callback 當獨立事件，
造成 QVFuF8 的 FP=59、Precision=26.25%，以及大量 callback 級 duplicates。
已恢復 score_falls.py 的 segment 級口徑（gap=15 gray 幀、整段與 GT 重疊）。
同批重算 TP=21、FP=1、FN=3、Recall=87.50%、Precision=95.45%，額外命中事件 2 次。
14 支影片的事件起訖幀與 oLE9gC 全部一致；此問題屬於統計工具，不是 SDK 偵測變更。
測試：`python3 experiments/yolo_bg/tests/release_scoring_test.py`，包含 GT 尾端重疊、
持續 30 幀、重複事件、無跌倒、漏報，以及 1000 組與舊 scorer 的差異比對。
新版 summary.csv 的 callback 計數欄改名 fall_callbacks / bedexit_callbacks，
避免誤讀為已合併事件；已存在的 summary.csv 不會自動改寫。

### 身分確認前的候選診斷（只記錄，不改判斷）

```sh
# 板端 data4..17，全幀、YOLO + Face；診斷版才能寫檔。
sh experiments/yolo_bg/run_eligibility_batch.sh 4 17 1
# 單片 data9 全幀。
sh experiments/yolo_bg/run_eligibility_batch.sh 9 9 1
```

新檔 eligibility_diagnostics.csv 在正常 BG/identity gate 前建立唯讀快照。
使用原本相同的最後 20 幀歷史、最近 3 筆動量、至少 10 筆物件歷史、
門檻 20（有離床倍率時沿用原值），以及 pixel_count/fg_area 的原面積門檻。
欄位含 mature、ambiguous、identity_eligible、面積、有效量測、當前觀察/等待狀態、
最近動量與候選阻擋原因。candidate_blocked=1 只表示移除 BG/identity gate 後
符合「新觸發的前置條件」，不表示已通過減速、45 幀觀察、形狀/床區/邊界或 YOLO
確認，更不代表實際發出跌倒事件。不會回補動量、啟用候選或改任何追蹤/觀察狀態。

啟用方式 HERMES_ELIGIBILITY_TRACE=1；亦需診斷編譯與 HERMES_FILE_OUTPUT=1。
全測 wrapper 另外開啟 HERMES_FALL_REASON_TRACE / HERMES_TRACK_REPLAY。
記錄先存記憶體，在 Release 輸出；eligibility_diagnostics_summary.csv 應有 dropped=0。
262144 筆上限避免無界記憶體增長，滿時只丟棄診斷紀錄，不影響判斷。
正式 SDK 寫檔禁令仍有效；本輪不更新 v210 正式交付包。

後續以相同 segment 級評分比對實際 TP/FP/FN，並檢視 data4/5/12 等無跌倒影片
是否也有 blocked 候選，以評估未來保留證據方案的誤報風險。只加 log 的測試
可驗證輸出是否變動，但不能證明尚未實作的候選回補方案不增加 FP。

### 2026-10-21 retained identity trigger and bottom occupancy experiment

The experimental SDK retains an otherwise-qualified fresh momentum/area trigger
blocked by identity eligibility for at most 4000 ms. Evidence is tied to the same
continuously present track ID, cleared on BG reset/release, and consumed once on
confirmation with current measured support. Current momentum still controls
waiting/deceleration; observation, projection, person confirmation and other
filters remain active. `TRIGGER_RETAIN`, `TRIGGER_REPLAY`, `TRIGGER_EXPIRE` expose
this path in `fall_reasons.csv`. Retained area is used only to qualify the trigger.

The bottom edge history now requires over 50% of the matched full foreground
pixels in the last 40 image rows, rather than a bounding-box touch. Side/top bbox
rules remain unchanged. Over half of observation frames must meet the combined
side/top or bottom-majority condition. The final center check retains side/top
checks but no longer independently rejects a bottom-center position. Corner and
other filters remain unchanged. `EDGE_BOTTOM_SAMPLE` records bottom pixel count,
total count, ratio and history. No match adds no history sample, as before.

Build: `make -f experiments/yolo_bg/Makefile.npu -j2 all`.
Board full regression: `sh experiments/yolo_bg/run_eligibility_batch.sh 4 17 1`.
This updates the experimental build only, not the v210 release package. Board
accuracy must be compared to be6IiG (TP21/FP1/FN3) using the same segment gap15.
