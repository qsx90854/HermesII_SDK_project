"""YOLO-assisted BG experiment; no input bins are generated."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--model',type=Path,default=ROOT/'models/yolov7-tiny/yolov7-tiny.onnx');p.add_argument('--history-two',action='store_true');p.add_argument('--nms-iou',type=float,default=.45)
 p.add_argument('--identity-a',action='store_true',help='Enable persistent FG ownership and mature-track fragment protection')
 p.add_argument('--full',action='store_true',help='11 videos, strides 4/8, both modes; full scoring')
 p.add_argument('--videos',nargs='+',default=None);p.add_argument('--strides',nargs='+',type=int,default=None)
 p.add_argument('--limit',type=int,default=None,help='Source frame limit; 0=whole video. Limited runs have no accuracy scoring.')
 p.add_argument('--legacy-protection',action='store_true',help='Ablation: retain old blind 14-second protection')
 p.add_argument('--coast-validity',action='store_true');p.add_argument('--confirm-person',action='store_true');p.add_argument('--exclude-top-row',action='store_true');p.add_argument('--track-replay',action='store_true')
 p.add_argument('--fg-replay',action='store_true',help='Save per-frame BG/FG PNGs for FG viewer')
 p.add_argument('--only-yolo',action='store_true');p.add_argument('--trace',action='store_true',help='Write block and box CSV for replay viewer')
 p.add_argument('--delay-ms',type=int,default=2000);p.add_argument('--interval-ms',type=int,default=4000)
 p.add_argument('--confidence',type=float,default=.25);p.add_argument('--hold-ms',type=int,default=14000)
 a=p.parse_args()
 if not 0<a.nms_iou<=1:p.error('Invalid NMS IoU')
 a.trace=a.trace or a.fg_replay or a.track_replay
 if a.interval_ms<100 or a.interval_ms>60000 or not 0<=a.delay_ms<=60000 or not 0<a.confidence<=1 or not 0<=a.hold_ms<=120000:p.error('Invalid interval/delay/confidence/hold')
 videos=a.videos or (['data'+str(x) for x in [4,5,6,7,8,10,11,12,13,15,16]] if a.full else ['data13'])
 strides=a.strides or ([4,8] if a.full else [4]);limit=a.limit if a.limit is not None else (0 if a.full else 900)
 if limit<0 or any(x<1 for x in strides) or len(set(strides))!=len(strides):p.error('Invalid limit or strides')
 ready=json.loads((HERE/'READY.json').read_text(encoding='utf-8'))
 for name,digest in ready['hashes'].items():
  if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed after verification: '+name)
 runner=Path(ready['runner']);runtime=Path(ready['runtime']);model=a.model.resolve()
 control=ROOT/'fall_eval/results'/(('fall_safety_' if a.coast_validity or a.confirm_person or a.exclude_top_row else 'yolo_bg_v2_')+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));control.mkdir()
 modes=['yolo_bg'] if a.only_yolo else ['baseline_bg','yolo_bg']
 state={'complete':False,'stage':'running','completed':0,'expected':len(modes)*len(videos)*len(strides),'smoke_test':bool(limit)}
 dump(control/'status.json',state);dump(control/'manifest.json',{'model':str(model),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),'identity_a':a.identity_a,'history_two':a.history_two,'nms_iou':a.nms_iou,'ready':ready,'videos':videos,'strides':strides,'limit':limit,'interval_ms':a.interval_ms,'delay_ms':a.delay_ms,'confidence':a.confidence,'hold_ms':a.hold_ms,'trace':a.trace,'fg_replay':a.fg_replay,'coast_validity':a.coast_validity,'confirm_person':a.confirm_person,'exclude_top_row':a.exclude_top_row,'track_replay':a.track_replay,'protection_v2':not a.legacy_protection,'execution':'deterministic offline: run on sampled image, withhold result until source timestamp + delay','bg_gate':'6 seconds, strict25%','bg_alpha':{'base':.08,'step':.04,'max':.32},'input':'MP4 stream; no bin copies'})
 print('CONTROL_DIR='+str(control),flush=True)
 try:
  summary=[]
  for mode in modes:
   state['current']=mode;dump(control/'status.json',state)
   env={k:v for k,v in os.environ.items() if not k.startswith('HERMES_')}
   env.update(HERMES_IDENTITY_A='1' if a.identity_a else '0',HERMES_PERSON_HISTORY_TWO='1' if a.history_two else '0',HERMES_PERSON_NMS=str(a.nms_iou),HERMES_COAST_VALIDITY='1' if a.coast_validity and mode=='yolo_bg' else '0',HERMES_FALL_PERSON_CONFIRM='1' if a.confirm_person and mode=='yolo_bg' else '0',HERMES_EXCLUDE_TOP_ROW='1' if a.exclude_top_row else '0',HERMES_TRACK_REPLAY='1' if a.track_replay else '0',HERMES_PERSON_FG_REPLAY='1' if a.fg_replay else '0',HERMES_PERSON_PROTECTION_V2='0' if a.legacy_protection else '1',HERMES_FALL_REASON_TRACE='1' if a.trace else '0',HERMES_BG_STABILITY='1',HERMES_BG_STABILITY_SECONDS='6',HERMES_PERSON_BG='1' if mode=='yolo_bg' else '0',HERMES_PERSON_MODE='offline',HERMES_PERSON_MODEL=str(model),HERMES_PERSON_RUNTIME=str(runtime),HERMES_PERSON_TRACE='1' if a.trace else '0',HERMES_PERSON_INTERVAL_MS=str(a.interval_ms),HERMES_PERSON_DELAY_MS=str(a.delay_ms),HERMES_PERSON_CONF=str(a.confidence),HERMES_PERSON_HOLD_MS=str(a.hold_ms))
   cmd=[sys.executable,'-X','utf8',str(ROOT/'fall_eval/batch_eval.py'),'--runner',str(runner),'--experiment','baseline','--videos',*videos,'--strides',*map(str,strides),'--limit',str(limit),'--output-dir',str(control/mode)]
   with (control/(mode+'.log')).open('w',encoding='utf-8') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
   done=json.loads((control/mode/'completion.json').read_text(encoding='utf-8-sig'))
   if not done['complete'] or done['completed']!=len(videos)*len(strides):raise RuntimeError('Incomplete mode '+mode)
   if mode=='yolo_bg':
    for file in (control/mode).glob('data*/stride_*/person_bg_summary.json'):
     if json.loads(file.read_text())['failed']:raise RuntimeError('YOLO inference failed; see '+str(file.parent/'person_bg_errors.log'))
   state['completed']+=done['completed'];dump(control/'status.json',state)
   if not limit:
    with (control/mode/'summary.csv').open(encoding='utf-8-sig') as f:summary += [dict(mode=mode,**row) for row in csv.DictReader(f)]
  if summary:
   with (control/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
  person=[]
  for file in (control/'yolo_bg').glob('data*/stride_*/person_bg_summary.json'):
   person.append(dict(video=file.parent.parent.name,stride=int(file.parent.name.split('_')[1]),**json.loads(file.read_text())))
  dump(control/'person_summary.json',person)
  state.update(complete=True,stage='complete');dump(control/'status.json',state)
  print('Completed. '+('Smoke only: no TP/FP/FN scoring.' if limit else 'Metrics: '+str(control/'summary.csv')),flush=True)
 except Exception as e:
  state.update(stage='failed',error=str(e));dump(control/'status.json',state);raise
if __name__=='__main__':main()
