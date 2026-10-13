# Linux / Edge 人物推論後端移植交接

更新：2026-10-13。本文描述目前 Windows SDK 程式；Linux 接線尚未實作。此次只新增註解／文件，不更動算法、模型、參數或效能。

## 1. 替換範圍

| 檔案 | 責任 | Linux 處理方式 |
|---|---|---|
| `src/ai/person_detector.h` | Detector、Box、Result 通用契約 | 保留，實作新後端 |
| `src/ai/person_detector_onnx.h/.cpp` | Windows DLL、ORT、前後處理、worker | 不編入 Linux，改接板端推論 |
| `src/fall/person_bg_controller.h/.cpp` | 排程、延遲結果保護、BG 權重、人物結果歷史 | 保留政策；替換後端建立與離線測試接線 |
| `src/fall/fall_safety.h` | 候選跌倒的歷史人物確認 | 保留；不要讓推論後端自行發布跌倒 |
| `src/fall/fall_detector.cpp` | SDK 每幀流程及上述模組接線 | 保留共用接線，拆除與 ONNX 專用巨集的耦合 |

`PersonBgController::Impl` 目前持有 `unique_ptr<PersonDetectorOnnx>`，建構與時間／尺寸重設時各有後端建立位置。建議改成持有 `unique_ptr<Detector>`，由 factory 或注入建立 Linux 後端；兩個建立位置及 Shutdown 都要處理。

`RunSync` 不屬於通用 Detector 介面，僅供 Windows 離線測試。改成通用指標時，需將 offline 測試路徑另外隔離，不能直接照搬對 RunSync 的呼叫。

`HERMES_ENABLE_ONNX_PERSON` 目前不只控制 ORT，也包住 controller 與人物確認接線。不能單純關掉巨集就視為完成移植。應另分「共用人物輔助功能」及「ONNX 後端」編譯選項，確認 Linux 仍呼叫 MaskUnknown、Step、History 與人物確認。本文不提供未驗證的板端編譯命令。

## 2. 輸入、輸出與 buffer 所有權

- `TrySubmit(gray,w,h,stamp,id)`：連續 `uint8_t` 灰階影像，row stride 等於 w，至少 w*h bytes。沒有額外 pitch 參數。板端 camera buffer 若有 padding，需先處理或正式擴充介面，不能直接當 packed 使用。
- Windows 接受請求後會複製影像；呼叫者可以立即重用原 buffer。Linux 也必須複製或持有有效所有權，不能只保留短命指標給非同步 NPU。
- Box 為浮點像素邊界 `x1,y1,x2,y2`，相對於「提交給後端的影像」，不是 tensor／letterbox 座標或 0～1 座標；x2/y2 可等於 w/h。
- class 0 為 normal_person，class 1 為 fallen_person。兩者都作為人物存在證據；class 1 並不直接等同 SDK 跌倒事件。
- Result 必須原樣回傳 request_id 與 source_timestamp_ms。confidence 是後處理後的人物分數。
- `success=true, persons=[]` 是有效的無人結果；`success=false` 是錯誤／未知，error 保留原因。不能把 NPU 失敗、忙碌或超時假裝成無人。
- `inference_ms` 表示推論耗時，`total_ms` 包含前後處理；均為量測資訊，不作為來源 timestamp。

## 3. 時間、執行緒與重設

來源 timestamp 使用毫秒，與 SDK 每幀 Step 相同時間軸，正常連續遞增。不要以完成時間覆蓋拍攝時間，也不要混用不同起點的系統時鐘。

TrySubmit 為非阻塞接收：false 表示忙碌／停止，尚未產生推論結果；TryGetResult 為非阻塞輪詢，true 才交出一筆完整 Result。Windows 同時最多一筆 outstanding，已完成但尚未被取走也占用該名額。SDK 狀態與 BG 僅由 SDK 執行緒修改，NPU callback 應只安全發布 Result。

Shutdown 停止接收並安全等待正在使用資源的工作完成，再釋放資源；Windows 可能捨棄尚未執行的 queued 工作，不保證最後一定有結果。重設後舊結果不得混入新 stream。

controller 在解析度／grid 改變、timestamp 非遞增或相鄰 SDK frame 間隔超過 1000ms 時重設。即使 AI 週期較長，SDK 影像輸入仍需連續；只每 1.6 秒呼叫一次 SDK 會反覆重設。

## 4. 座標還原與 BG 保護的差別

啟用 HERMES_EXCLUDE_TOP_ROW 時，controller 裁掉最上面一個 grid row：crop_top = h / rows，提交 now + crop_top*w、尺寸 w × (h-crop_top)。後端先還原到這個裁切影像座標；controller 收結果時再加一次 crop_top。不能漏加或重複加。

完整座標流程：SDK 灰階 frame → controller 裁切頂列 → 後端 resize/letterbox → 推論 → 後端 inverse letterbox → controller 補回頂列偏移。

History 保存來源影像的原人物框。BG 保護框則可能依可靠 motion track 位移、外擴並維持限時保護。這兩種框用途不同，不能用移動後的保護框覆蓋歷史原始偵測框。

## 5. Windows 模型參考行為

目前 SDK 輸入通常為 800×450，但後端按傳入 w/h 處理；模型 input 尺寸由 ONNX metadata 取得。

Windows 前處理：灰階複製三通道、雙線性縮放、居中 letterbox 填 114、float32 NCHW 除 255。這不是 Python UI 直接使用原影片 RGB 的同一條路徑。Linux 若改用彩色、NV12、量化張量或硬體 resize，必須另外驗證模型效果與座標。

目前 C++ 支援：

| 輸出 | 解碼 |
|---|---|
| `[1,6,N]` | cx,cy,w,h + 兩個 class score，沒有 objectness；640 模型 N=8400，320 模型 N=2100 |
| `[1,25200,85]` | COCO；objectness × person score，僅最高分類為 person 的候選 |

後處理為 class-aware NMS，保留最多 300 框，NMS 後裁切邊界及刪除空框。預設 confidence=.25、NMS IoU=.45；應以實際實驗 manifest 為準。

目前 SDK 的 Hermes320模型為 `models/hermes-v1/yolov7t_HermesII_v1_320x320.onnx`。新 `yolov7-tiny-20261013.onnx` 的 `[1,6300,7]` 僅 Python viewer 已支援，SDK C++ 會拒絕；若後續接入，分數應為 objectness × class score，不能套用上述 6-channel 解碼。不要將這份交接視為已完成新模型移植。

## 6. 排程與 BG／跌倒確認政策

程式預設 interval=4000ms、delay=2000ms；最近 timing16 實驗覆寫成 interval=1600ms、delay=800ms。預設值與實驗設定不同，需隨交接保存有效設定。

- async：實際 worker 完成後輪詢採用，不加模擬 sleep。
- offline：提交時 RunSync，直到來源時間到 request_time+delay 才發布；用於可重現離線測試。
- cadence 由提交時來源時間決定，不是「完成後再等 interval」。忙碌則跳過，不排隊追趕。
- freshness 現在使用 interval+delay+1000ms，async 也包含 delay 參數；Linux 不應誤認它只影響離線，需明確配置。
- 只有成功且新鮮的結果才執行一輪 BG 更新。人物保護 alpha=0 並清除非人物次數；motion 或請求後變動區本輪不吸收。
- 可更新非人物區每輪先增加 misses，再算 min(.32,.08+.04*misses)，所以第一輪為 12%。首次有效 BG 直接取 current，尚未有效區以 current 佔位，不能解讀成已知地板。
- v2 人物保護含 motion bridge、失配外觀支持／解除與期限。移植推論不應順便移除這些機制。
- HERMES_PERSON_HISTORY_TWO=1 時，跌倒候選檢查前兩次成功結果，任一次位置匹配即可；都不符則等下一次合格結果。推論失敗不等於無人。細節以 fall_safety.h 為準。

相關開關：HERMES_PERSON_BG、HERMES_PERSON_MODE、HERMES_PERSON_INTERVAL_MS、HERMES_PERSON_DELAY_MS、HERMES_PERSON_PROTECTION_V2、HERMES_FALL_PERSON_CONFIRM、HERMES_PERSON_HISTORY_TWO、HERMES_EXCLUDE_TOP_ROW。Identity A、coasting、BG stability、observation frame 參數另行維持相同實驗設定，不因換後端改動。

## 7. Linux 接線驗收建議（尚未執行）

1. 使用同一份 SDK 灰階輸入，比對後端預處理、框座標、類別、confidence；量化誤差需記錄，不能默認框完全相同。
2. 驗證無人成功、推論失敗、忙碌拒絕、結果延遲、取走結果前不得再接收、Shutdown 與重設。接受後重用輸入 buffer 不得破壞結果。
3. 驗證頂列排除 ON/OFF 座標還原，以及 capture timestamp 與 applied timestamp 的差別。
4. 先固定 AI 結果序列，確認 BG alpha、人物保護、ID、歷史確認與跌倒輸出一致，區分接線問題與模型差異。
5. 最後以真實板端節奏量測推論延遲、跳過請求數及 SDK 持續运行記憶體。Windows 離線延遲不是板端實測。

目前 person_bg_events.csv／summary 仍有未受 TRACE 控制的輸出；移植時須安排可寫路徑或另外設計關閉功能。本次未修改該行為。README 中早期實驗說明是歷史紀錄；部署參數應從欲對照的 manifest 核對。
