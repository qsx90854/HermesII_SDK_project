# 基準版 BG 啟動穩定度實驗

目前實作只增加啟動gate；沒有YOLO，也沒有新版長短期BG/FG算法。
初始BG存在後，至少觀察N=6秒；有效區域為800x450畫面四邊各扣40pixels，再扣床。
最近N秒相鄰輸入影像灰階差>12的pixel聯集，嚴格小於有效區域25%才通過。
每個pixel保存最後變動時間，滑动視窗到期後不再算入。BG更新/追蹤持續，gate關閉時不累積跌倒候選/觀察與預啟用motion歷史。
通過後鎖定；不再因正常走動重新判不穩定。時間戳逆序/間隔>1000ms或解析度改變會重新建立視窗。
沒有YOLO時不能排除初始BG含靜止人體，這一版也不實作可靠空地覆蓋率或人物BG保護。

工作目錄D:/HermesII_202610，已進虛擬環境：

先測data6/data13，OFF/ON × stride4/8，共8組：
```
python -X utf8 HermesII_SDK_project/experiments/bg_stability/run.py
```
完整11片OFF/ON × stride4/8，共44組：
```
python -X utf8 HermesII_SDK_project/experiments/bg_stability/run.py --full
```
指定N秒：加上 --seconds 6。
完整cohort=data4,5,6,7,8,10,11,12,13,15,16。排除1/2/3/9及解碼失敗14。

結果fall_eval/results/bg_stability_時間戳/：
summary.csv / comparison.csv：兩版TP/FP/FN、Recall/Precision/F1、FP/hour、延遲。
bg_ready_summary.csv：啟用時的source_frame、sdk_frame與秒數（frame從0起算）。
readiness.csv：每片各stride啟用秒數及是否從未啟用。
fn_initialization.csv：FN位於完全未啟用期間、開始早於啟用、或啟用後漏判；屬於時間分類，不等同已確認root cause。
baseline_off/與bg_stability_on/：原有per_video.csv/per_event.csv、每片frames.csv與log。
bg_stability_on/dataX/stride_Y/bg_stability.csv：時間、變動pixels/比例、ready/reason；ready_latched時ratio=-1代表已停止評估，不能解讀成零變動。
主評分保留完整GT；不排除初始化期間跌倒。沿用原始0-basedinclusive標記與alarm onset一對一配對。
直接串流MP4至C++，不寫任何新的bin，也不複製既有bin。

READY.json驗證source/binary/config hashes。若後續修改來源，需重新編譯與驗證後更新，不能直接略過。

驗證：6秒／嚴格25%版本只短測，完整矩陣由使用者手動執行。
