"""HermesII two-class person model + two-history confirmation. No coasting validity gate."""
import sys
from pathlib import Path
import run
if __name__=='__main__':
 sys.argv[1:1]=['--model',str(run.ROOT/'models/hermes-v1/yolov7t_HermesII_v1.onnx'),'--nms-iou','0.7','--history-two','--confirm-person','--exclude-top-row','--only-yolo','--track-replay','--fg-replay']
 run.main()
