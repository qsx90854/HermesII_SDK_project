"""Bounded safety pilot; MP4 streams, no input bins, no full dataset run."""
import os,sys,subprocess,struct,csv,json,hashlib
from pathlib import Path
from datetime import datetime
import cv2
ROOT=Path(__file__).resolve().parents[3]
runner=ROOT/'fall_eval/build_fall_safety/fall_eval_runner.exe'
out=ROOT/'fall_eval/results'/('fall_safety_smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));out.mkdir()
(ROOT/'fall_eval/safety_smoke_path.txt').write_text(str(out),encoding='utf-8')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
print('CONTROL_DIR='+str(out),flush=True);results=[]
dump(out/'manifest.json',{'runner':str(runner),'sha256':hashlib.sha256(runner.read_bytes()).hexdigest(),'scope':'data4 first4800 source frames; data6 first1200; stride4; not full accuracy','input':'streamed MP4; no bins'})
dump(out/'status.json',{'complete':False,'stage':'running'})
for video,limit,modes in [('data4',4800,{'off':(0,0,0),'coast':(1,0,0),'confirm':(0,1,0),'combined_top':(1,1,1)}),('data6',1200,{'combined':(1,1,0),'combined_top':(1,1,1)}),('data13',1600,{'combined_top':(1,1,1)})]:
 if len(sys.argv)>1 and video not in sys.argv[1:]:continue
 path=next((ROOT/'test_video_mp4_all').glob(video+'*.mp4'));cap=cv2.VideoCapture(str(path));w,h=int(cap.get(3)),int(cap.get(4));fps=cap.get(5);assert cap.isOpened();bed=out/(video+'_bed.txt');points=[list(map(int,line.split(','))) for line in (ROOT/'Parameter/Bed_Position.txt').read_text(encoding='utf-8-sig').splitlines() if line.strip()];bed.write_text(''.join(f'{x*800//w},{y*450//h}\n' for x,y in points),encoding='utf-8');jobs=[]
 try:
  for mode,(coast,confirm,top) in modes.items():
   d=out/mode/video/'stride_4';d.mkdir(parents=True);env={k:v for k,v in os.environ.items() if not k.startswith('HERMES_')};env.update(HERMES_BG_STABILITY='1',HERMES_PERSON_BG='1',HERMES_PERSON_PROTECTION_V2='1',HERMES_PERSON_MODE='offline',HERMES_PERSON_TRACE='1',HERMES_FALL_REASON_TRACE='1',HERMES_TRACK_REPLAY='1',HERMES_PERSON_FG_REPLAY='1' if mode=='combined_top' else '0',HERMES_COAST_VALIDITY=str(coast),HERMES_FALL_PERSON_CONFIRM=str(confirm),HERMES_EXCLUDE_TOP_ROW=str(top),HERMES_PERSON_MODEL=str(ROOT/'models/yolov7-tiny/yolov7-tiny.onnx'),HERMES_PERSON_RUNTIME=r'D:\Lightglue\lg_env\Lib\site-packages\onnxruntime\capi\onnxruntime.dll')
   log=(d/'runner.log').open('w',encoding='utf-8');proc=subprocess.Popen([str(runner),str(ROOT/'Parameter/sdk_gray_count_falls.ini'),str(bed),str(d/'frames.csv'),'800','450'],cwd=d,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT);jobs.append((mode,d,proc,log));dump(out/mode/'manifest.json',{'exclude_top_row':bool(top)})
  for n in range(limit):
   ok,img=cap.read();assert ok,(video,n)
   if n%4==0:
    gray=cv2.cvtColor(cv2.resize(img,(800,450),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY);packet=struct.pack('<QQ',n,round(n*1000/fps)+1)+gray.tobytes()
    for _,_,proc,_ in jobs:proc.stdin.write(packet)
  for mode,d,proc,log in jobs:
   proc.stdin.write(struct.pack('<Q',2**64-1));proc.stdin.close();assert proc.wait(timeout=60)==0,str(d);log.close();rows=read(d/'frames.csv');assert len(rows)==limit//4
   fall_frames=[int(x['source_frame']) for x in rows if x['is_fall']=='1'];r={'mode':mode,'video':video,'sdk_frames':len(rows),'fall_first':min(fall_frames) if fall_frames else None,'fall_last':max(fall_frames) if fall_frames else None};results.append(r);dump(d.parent/'video.json',dict(path=str(path),fps=fps,width=w,height=h,frames=limit));print(r,flush=True)
   if mode=='off':
    old=read(ROOT/'fall_eval/results/yolo_bg_v2_20261009_184445_988399/yolo_bg/data4/stride_4/frames.csv');keys=['source_frame','sdk_frame','timestamp_ms','is_fall','fall_object_ids'];assert [[x[k] for k in keys] for x in rows]==[[x[k] for k in keys] for x in old[:len(rows)]],'OFF regression';r['off_regression']=True
   if mode=='combined_top':
    for row in read(d/'track_replay.csv'):assert all(int(b)>=12 for b in row['blocks'].split(';') if b)
    for row in read(d/'fg_objects.csv'):assert float(row['y1'])>=28
    for row in read(d/'person_bg_blocks.csv'):
     if int(row['block'])<12:assert row['reason']=='excluded' and float(row['alpha'])==0
    for row in read(d/'person_bg_boxes.csv'):
     if row['kind']=='source':assert float(row['y1'])>=28
    r['top_exclusion']=True
 finally:
  cap.release()
  for _,_,proc,log in jobs:
   if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
   log.close()
 dump(out/'summary.json',results)
dump(out/'status.json',{'complete':True,'stage':'complete','completed':len(results)});print('COMPLETE',flush=True)
