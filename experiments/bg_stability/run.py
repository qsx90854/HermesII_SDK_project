"""Baseline BG startup stability experiment. Default pilot; --full = 44 combinations."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
from datetime import datetime
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];EVAL=ROOT/'fall_eval'
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def write_csv(p,rows):
 if not rows:return
 with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def summarize(control):
 control=Path(control).resolve()
 delivery=json.loads((control/'delivery_manifest.json').read_text(encoding='utf-8-sig'));videos=delivery['videos']
 state=json.loads((control/'status.json').read_text(encoding='utf-8-sig'))
 completed=0
 for mode in ['baseline_off','bg_stability_on']:
  done=json.loads((control/mode/'completion.json').read_text(encoding='utf-8-sig'))
  if not done['complete'] or done['completed']!=len(videos)*2:raise RuntimeError('Cannot summarize incomplete mode: '+mode)
  completed+=done['completed']
 if state.get('error'):state.setdefault('previous_errors',[]).append(state.pop('error'))
 state['completed']=completed
 readiness=[];ready_frames=[];fn=[]
 for video in videos:
  for stride in [4,8]:
   target=control/'bg_stability_on'/video/f'stride_{stride}';meta=json.loads((target/'bg_stability_summary.json').read_text(encoding='utf-8-sig'));vid=json.loads((target.parent/'video.json').read_text(encoding='utf-8-sig'));fps=vid['fps'];activation=meta['activation_timestamp_ms']
   readiness.append(dict(video=video,stride=stride,enabled=meta['ready'],enable_seconds=(activation-1)/1000 if activation else '',window_ms=meta['window_ms']))
   frame={'source_frame':'','sdk_frame':''}
   if activation:
    with (target/'frames.csv').open(encoding='utf-8-sig',newline='') as f:
     frame=next((r for r in csv.DictReader(f) if int(r['timestamp_ms'])==activation),None)
    if frame is None:raise RuntimeError('Activation timestamp missing: '+str(target))
   ready_frames.append(dict(video=video,stride=stride,ready=meta['ready'],source_frame=frame['source_frame'],sdk_frame=frame['sdk_frame'],enable_seconds=(activation-1)/1000 if activation else '',timestamp_ms=activation,window_seconds=meta['window_ms']/1000))
   events=list(csv.DictReader((control/'bg_stability_on/per_event.csv').open(encoding='utf-8-sig')))
   for e in events:
    if e['video']!=video or int(e['stride'])!=stride or e['result']!='FN':continue
    start=int(e['start']);end=int(e['end']);start_ms=round(start*1000/fps)+1;end_ms=round(end*1000/fps)+1
    cause='not_enabled_during_interval' if not activation or end_ms<activation else 'started_before_enable' if start_ms<activation else 'miss_after_enable'
    fn.append(dict(video=video,stride=stride,start_frame=start,end_frame=end,start_seconds=start/fps,end_seconds=end/fps,enable_timestamp_ms=activation,category=cause))
 write_csv(control/'bg_ready_summary.csv',ready_frames)
 write_csv(control/'readiness.csv',readiness);write_csv(control/'fn_initialization.csv',fn)
 summary=[]
 for mode in ['baseline_off','bg_stability_on']:
  for row in csv.DictReader((control/mode/'summary.csv').open(encoding='utf-8-sig')):summary.append(dict(mode=mode,**row))
 write_csv(control/'summary.csv',summary)
 write_csv(control/'comparison.csv',summary);state.update(complete=True,stage='complete');dump(control/'status.json',state);print(json.dumps(summary,ensure_ascii=False),flush=True)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--full',action='store_true');ap.add_argument('--seconds',type=float,default=6);ap.add_argument('--summarize-only',type=Path,help='Rebuild reports from completed run; never rerun videos');args=ap.parse_args()
 if args.summarize_only:return summarize(args.summarize_only)
 if not .1<=args.seconds<=600:ap.error('seconds must be .1..600')
 ready=json.loads((HERE/'READY.json').read_text(encoding='utf-8-sig'));
 for filename,digest in ready['hashes'].items():
  if hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=digest:raise RuntimeError('Changed after verification: '+filename)
 videos=['data'+str(n) for n in [4,5,6,7,8,10,11,12,13,15,16]] if args.full else ['data6','data13']
 control=EVAL/'results'/('bg_stability_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));control.mkdir();state=dict(complete=False,stage='running',expected=len(videos)*4,completed=0);dump(control/'status.json',state);print('CONTROL_DIR='+str(control),flush=True)
 dump(control/'delivery_manifest.json',dict(ready=ready,videos=videos,strides=[4,8],window_seconds=args.seconds,changed_ratio_strict=.25,gray_delta=12,input_method='direct MP4 stream; no disk bins generated',scoring='original full GT including initialization periods'))
 try:
  for mode in ['baseline_off','bg_stability_on']:
   state['current']=mode;dump(control/'status.json',state)
   env={k:v for k,v in os.environ.items() if not k.startswith('HERMES_')};env.update(HERMES_BG_STABILITY='1' if mode=='bg_stability_on' else '0',HERMES_BG_STABILITY_SECONDS=str(args.seconds),HERMES_BG_STABILITY_DELTA='12')
   cmd=[sys.executable,'-X','utf8',str(EVAL/'batch_eval.py'),'--runner',ready['runner'],'--experiment','baseline','--videos',*videos,'--strides','4','8','--output-dir',str(control/mode)]
   with (control/(mode+'.log')).open('w',encoding='utf-8') as log:rc=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
   if rc:raise RuntimeError(mode+' failed; see '+mode+'.log')
   done=json.loads((control/mode/'completion.json').read_text(encoding='utf-8-sig'));assert done['complete'] and done['completed']==len(videos)*2
   state['completed']+=done['completed'];dump(control/'status.json',state)
  summarize(control)
 except Exception as e:state.update(stage='failed',error=str(e));dump(control/'status.json',state);raise
if __name__=='__main__':main()
