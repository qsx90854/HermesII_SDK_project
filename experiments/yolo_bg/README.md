# YOLO-assisted BG experiment (Windows)
Linux／NPU 交接請先讀 [LINUX_AI_PORTING.md](LINUX_AI_PORTING.md)（2026-10-13；介面契約、替換範圍與驗收）。

獨立Windows後端：src/ai/person_detector_onnx.h/.cpp，經C API動態載入ONNX Runtime，不將Runtime header暴露到SDK。
人物BG控制：src/fall/person_bg_controller.h/.cpp。fall_detector.cpp僅加編譯開關內的接線。
未接上Edge NPU；此版本不改SDK公開ABI。實驗設定先透過環境變數傳入。

## 執行（已進入lg_env）

預設data13、stride4、前900個原影片frames，baseline_bg與yolo_bg兩組；短測不產生準確率。

```powershell
python -X utf8 D:\HermesII_202610\HermesII_SDK_project\experiments\yolo_bg\run.py --trace
```

完整測試data4–8、10–13、15、16，stride4/8，兩模式共44組：

```powershell
python -X utf8 D:\HermesII_202610\HermesII_SDK_project\experiments\yolo_bg\run.py --full --trace
```

只跑指定完整影片：`--videos data6 data13 --strides 4 --limit 0 --trace`。
只跑新版：加`--only-yolo`。關閉UI細節輸出：省略`--trace`。
兩模式都啟用6秒/嚴格25%穩定gate，唯一組間改動為人物BG模式。
讀MP4串流，不寫或複製input bin。控制目錄自動輸出到fall_eval/results/yolo_bg_時間戳。

## 結果與UI

完整測試的summary.csv有兩模式TP/FP/FN/Recall/Precision/F1/FP每小時及延遲；per_event.csv保留各事件。
person_summary.json彙整推論及更新次數。每個yolo_bg/dataX/stride_Y內：
- person_bg_events.csv：取圖、採用時間、人物數、耗時、有效/受保護block數。
- person_bg_summary.json：提交、採用、失敗、過期及影片結束尚未採用的結果數。
- --trace才產生person_bg_blocks.csv與person_bg_boxes.csv。

```powershell
python -X utf8 D:\HermesII_202610\HermesII_SDK_project\experiments\yolo_bg\view.py
```

依序選原MP4及上述stride資料夾。UI顯示每輪BG更新的實際採用frame，可前後比較block alpha。
藍框是舊的取圖時YOLO結果，不代表採用時人物所在；綠色block才是採用時保護區。
這是BG更新快照檢視器，不是每幀人物追蹤播放器。

## 時序與策略

預設每4000ms取圖。離線入口同步計算模型結果，但等source timestamp+2000ms才採用；採用幀為第一個達到該時間的輸入幀。沒有使用未來frame。模型推論實際wall time不影響此排程。
實時模式HERMES_PERSON_MODE=async由worker非同步執行；此模式輸入須依真實節奏送入，不能用快速離線串流估計Edge效果。
只允許一個尚未採用請求，過忙跳過，不累積佇列。錯誤和過期结果不增加alpha；最後尚未到採用時間的结果不強制套到最後一幀。
SDK timestamp間隔>1000ms、逆序或解析度/grid改變會清除背景有效性及人物保護狀態。

人物框外擴10%；人物保護alpha=0，非人物計數歸零。離開保護後從12%開始，每次成功可用結果加4百分點，上限32%。
人物資格預設保留14秒（可設--hold-ms），期間可靠實測track會平移保護框；coasting不刷新資格。此保護會使YOLO短暫漏框不立即觸發吸收。
track失配則保留限時的來源範圍，標記track=-1；此為保守保護，不宣稱已確認當前人物位置。重新YOLO命中重疊區會替換舊框。
當前motion區或取圖後超過20%像素灰階變化>12的block，本輪不吸收、不增加非人物次數。
只在成功採用時更新BG；人物模式停用舊的每12frame節奏與床區alpha倍率。
未知block在第一次非人物且安靜時建立背景；此前背景buffer為current的佔位副本，使既有BG差分不以未知值提供FG證據。不是有效地板背景。若YOLO漏掉開場人物，仍可能初始化污染，這層無法保證語義正確。

## 設定

HERMES_PERSON_BG=1（預設關閉）；HERMES_PERSON_MODEL/ HERMES_PERSON_RUNTIME為絕對路徑。
HERMES_PERSON_INTERVAL_MS=4000、DELAY_MS=2000（離線）、HOLD_MS=14000、CONF=.25、NMS=.45、MARGIN=.10。
其餘環境變數全名：HERMES_PERSON_BASE_ALPHA=.08、HERMES_PERSON_ALPHA_STEP=.04、HERMES_PERSON_MAX_ALPHA=.32、HERMES_PERSON_TRACE=0。
入口--interval-ms/--delay-ms/--confidence/--hold-ms可調，會保存manifest。其餘參數目前使用固定預設，以免隱藏環境影響實驗。

## 編譯/驗證

獨立build：fall_eval/build_yolo_bg，Release -O3 -DNDEBUG。CMake選項HERMES_ENABLE_ONNX_PERSON=OFF可不編譯ONNX模組；執行期開關OFF也不載入DLL/模型。
READY.json記錄模型、來源、執行檔、Runtime與設定hash；修改來源需重新編譯驗證，不能略過檢查。
短測：data13的674輸入幀OFF與舊BG版跌倒/ID輸出一致；C++與Python人物框比對；單一進行中請求與shutdown；合成無人影像驗證alpha12/16/20/24/28/32/32/32與2000ms延遲。
尚未跑全量，也尚未驗證人物模型在IR影片上的準確率或真實Edge耗時。
備份：D:/HermesII_202610/backups/yolo_bg_20261008_172426_793663。只修改既有fall_detector.cpp，其餘為新增檔案。
還原時先確認無後續修改，再以備份的相對路徑還原fall_detector.cpp；停用新入口即可保留舊執行檔。不要覆蓋其他使用者變更。


## 2026-10-09 人物保護修正 v2

run.py預設啟用v2，結果前綴yolo_bg_v2；--legacy-protection可對照舊保護，獨立build_yolo_bg_v2保留原build。
保留每4秒更新、12%起每次加4%到32%的設定，這次只改保護資格延續與新增診斷。
實測track連續時保護區平移並保存8x8灰階參考；失配時以64個樣本中至少65%灰階差<=12作為位置仍受支持的條件。
失配且外觀不受支持持續1000ms就解除。靜止外觀支持不刷新YOLO確認時間，仍受14秒上限約束。
此判斷是位置外觀支持，不是新增人體辨識；小幅動作也可能影響支持比例，需完整實驗驗證。
--trace新增person_bg_protection.csv（track_lost、stationary_supported、unmatched_grace、appearance_changed_release、expired等）及fall_reasons.csv（SDK frame、ID、觀察期觸發與篩選原因）。
fall_reasons.csv使用SDK frame；請以同資料夾frames.csv映射原影片frame，不能當作原影片index直接讀。
舊框保護只能解釋data4第二次新增FP；第一次仍受FG幾何與既有高窄過濾器影響，本次不放寬或加嚴跌倒門檻。
本次修改前備份由fall_eval/yolo_v2_backup.txt記錄。還原時只取回該備份中的檔案，保留後續其他修改。


### 實際 FG 逐幀檢視
執行時同時加上 `--trace --fg-replay`，每個 stride 目錄會產生 `fg_replay/index.csv` 與每幀 BG/FG PNG。這是差分與 FG 分組使用的實際 mask，保存時機在該幀 BG 更新之前；未知背景區按 SDK 邏輯排除。PNG 僅作診斷，預設關閉，未產生額外 input bin。

`view.py` 自動辨識 FG replay，可切換 FG 疊圖、FG 二值圖、BG、原圖與更新權重；支援原影片 frame 跳轉、逐 SDK 幀與播放。紅框及上方文字表示當前算法的跌倒輸出。FG 是分組前 mask，此版沒有新增 FG 或 motion 物件 ID 輸出。

單跑 data4 stride4：
```powershell
python -X utf8 HermesII_SDK_project/experiments/yolo_bg/run.py --videos data4 --strides 4 --limit 0 --only-yolo --trace --fg-replay
```


### Fall safety（2026-10-10）
三個開關預設 OFF，彼此獨立：`--coast-validity`、`--confirm-person`、`--exclude-top-row`。診斷用 `--track-replay --fg-replay`；這兩個選項會自動開啟 `--trace`。新功能結果目錄前綴為 `fall_safety_`。

* Coasting/ghost 不是實測動量，不放入 momentum_samples，也不能用預設零直接通過減速。無 motion 時，必須有最近實測區域的 FG 支持與連續影像安靜才可累積。固定起始參數：取樣間隔4px、至少16支持點、仍為FG >=50%、與最後實測灰階差 <=12 的點 >=65%、相鄰幀變動點 <=20%、連續3個輸入幀、參考最多12秒；重疊的其他實測 track 構成歸屬衝突。觀察自觸發起最多30秒。45張原門檻不變，quiet 張數與實測樣本分開記錄。全部由 quiet 支持時，實測平均值標示 NaN；獨立檢查 quiet 張數，不以0充當實測平均值。
* YOLO 使用最近兩個已返回、未過期的取圖結果。歷史正向結果可在候選成立前，但必須晚於觀察開始及最近明顯位移；最新有效結果優先。優先使用1秒內實際配對的FG範圍，否則使用motion範圍。候選覆蓋率 >=50%、人物框覆蓋率 >=15%。不足則 pending，等待候選成立後的新取圖：命中確認，成功推論但未命中取消。推論錯誤不是未命中。等待最多10秒；取消清空本輪觀察及持續跌倒計數，需新的實測觸發才能再開始。
* 頂列排除是目前網格的第一列（800x450、12x16時 y=0..27），原座標不變；motion SAD、FG、BG更新及穩定度排除，YOLO裁切有效區域後補回Y偏移。初始化固定，不支援中途切換。
* `track_replay.csv` 含各幀 motion/ghost/coasting 來源、量測有效性、FG配對是否真的執行、觀察張數與人物確認狀態。FG ID為每幀 local ID，配對線代表算法實際配對，不是假設全域FG追蹤。`fall_confirmation.csv` 紀錄 pending/確認/取消原因。
* 此版本不是原先 long_fg_local_activity 分支；此處只在目前 YOLO BG 基準上增加上述功能。

使用者切換輕量模型後，可執行全量（11個影片×stride4/8，排除data1–3、9及不可用data14）：
```powershell
python -X utf8 D:\HermesII_202610\HermesII_SDK_project\experiments\yolo_bg\run.py --full --only-yolo --coast-validity --confirm-person --exclude-top-row --track-replay --fg-replay
```
結果含 summary.csv、per_video.csv、per_event.csv 與逐幀UI資料；不產生 input bin。PNG是額外診斷輸出，若只看分數可移除 `--fg-replay`。
短測不是完整準確率評估；data6 第一個原先計為TP的事件在新增檢查後消失，需保留此回歸資訊，不能稱 Recall 已通過。
本次修改前備份位置記錄在 `fall_eval/fall_safety_backup.txt`。還原時只恢復其中列出的對應檔案，移除本次新增且沒有後續修改的 safety 檔案；不要重設整個工作樹。
