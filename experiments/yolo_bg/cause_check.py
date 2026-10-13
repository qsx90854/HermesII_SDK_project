"""Bounded causal comparison: one MP4 decode per video, three SDK processes."""
import sys,os,subprocess,struct,csv,json
from pathlib import Path
from datetime import datetime
import cv2
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'fall_eval'));import batch_eval as batch
out=ROOT/'fall_eval/results'/('yolo_bg_v2_check_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir();print('CONTROL_DIR='+str(out),flush=True);(ROOT/'fall_eval/yolo_v2_check_path.txt').write_text(str(out))
all_rows=[];events=[]
for video in ['data4','data6','data13']:
 path=next((ROOT/'test_video_mp4_all').glob(video+'*.mp4'));cap=cv2.VideoCapture(str(path));w,h=int(cap.get(3)),int(cap.get(4));fps=cap.get(5);expected=int(cap.get(7));assert cap.isOpened()
 gt=batch.labels(ROOT/'fall_frameindex_data_all'/(video+'_fall_frameindex.txt'),0)
 bed=out/(video+'_bed.txt');points=[list(map(int,line.split(','))) for line in (ROOT/'Parameter/Bed_Position.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()];bed.write_text(''.join(f'{x*800//w},{y*450//h}\n' for x,y in points));jobs=[]
 try:
  for mode in ['baseline','legacy','v2']:
   d=out/mode/video/'stride_4';d.mkdir(parents=True);env={k:v for k,v in os.environ.items() if not k.startswith('HERMES_')};env.update(HERMES_BG_STABILITY='1',HERMES_PERSON_BG='0' if mode=='baseline' else '1',HERMES_PERSON_PROTECTION_V2='1' if mode=='v2' else '0',HERMES_PERSON_MODE='offline',HERMES_PERSON_TRACE='1',HERMES_FALL_REASON_TRACE='1',HERMES_PERSON_MODEL=str(ROOT/'models/yolov7-tiny/yolov7-tiny.onnx'),HERMES_PERSON_RUNTIME=r'D:\Lightglue\lg_env\Lib\site-packages\onnxruntime\capi\onnxruntime.dll')
   log=(d/'runner.log').open('w');process=subprocess.Popen([str(ROOT/'fall_eval/build_yolo_bg_v2/fall_eval_runner.exe'),str(ROOT/'Parameter/sdk_gray_count_falls.ini'),str(bed),str(d/'frames.csv'),'800','450'],cwd=d,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT);jobs.append((mode,d,process,log))
  count=0
  while True:
   ok,image=cap.read()
   if not ok:break
   if count%4==0:
    gray=cv2.cvtColor(cv2.resize(image,(800,450),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY);packet=struct.pack('<QQ',count,round(count*1000/fps)+1)+gray.tobytes()
    for _,_,process,_ in jobs:process.stdin.write(packet)
   count+=1
  assert count==expected,(count,expected)
  for mode,d,process,log in jobs:
   process.stdin.write(struct.pack('<Q',2**64-1));process.stdin.close();assert process.wait(timeout=60)==0,str(d);log.close();rows=list(csv.DictReader((d/'frames.csv').open()));assert [int(x['source_frame']) for x in rows]==list(range(0,count,4));metrics,detail=batch.score(rows,gt,fps);all_rows.append(dict(mode=mode,video=video,stride=4,**metrics));events += [dict(mode=mode,video=video,**x) for x in detail];batch.dump(d.parent/'video.json',dict(path=str(path),fps=fps,width=w,height=h,frames=count));print(mode,video,metrics['tp'],metrics['fp'],metrics['fn'],flush=True)
 finally:
  cap.release()
  for _,_,process,log in jobs:
   if process.poll() is None:process.terminate();process.wait(timeout=10)
   log.close()
 batch.write_csv(out/'summary.csv',all_rows);batch.write_csv(out/'per_event.csv',events)
batch.dump(out/'completion.json',dict(complete=True,completed=len(all_rows)));print('COMPLETE',flush=True)
