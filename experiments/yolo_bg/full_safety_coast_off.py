"""All available data except data1-3; data14 scored on verified 6066-frame prefix."""
import os,sys,csv,json,hashlib,subprocess,time
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'fall_eval'));import batch_eval as batch
control=ROOT/'fall_eval/results'/('fall_safety_coast_off_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));control.mkdir()
(ROOT/'fall_eval/safety_coast_off_path.txt').write_text(str(control),encoding='utf-8')
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def read(p):return list(csv.DictReader(p.open(encoding='utf-8-sig')))
state={'complete':False,'stage':'verifying','completed':0,'expected':26};dump(control/'status.json',state)
print('CONTROL_DIR='+str(control),flush=True)
try:
 ready=json.loads((HERE/'READY.json').read_text(encoding='utf-8'))
 for n,h in ready['hashes'].items():assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,'Changed after verification: '+n
 videos=['data'+str(n) for n in range(4,17) if n!=14]
 manifest={'ready':ready,'videos':videos+['data14'],'strides':[4,8],'exclude':['data1','data2','data3'],'data14':{'source_frames':6066,'metadata_frames':6114,'scope':'decodable prefix only; all GT must be inside prefix'},'coast_validity':False,'confirm_person':True,'exclude_top_row':True,'track_replay':True,'fg_replay':True,'trace':True,'bg_gate':'6s strict25%','interval_ms':4000,'delay_ms':2000,'confidence':.25,'hold_ms':14000,'input':'MP4 streams; no generated input bins'};dump(control/'manifest.json',manifest)
 env={k:v for k,v in os.environ.items() if not k.startswith('HERMES_')};env.update(HERMES_BG_STABILITY='1',HERMES_BG_STABILITY_SECONDS='6',HERMES_PERSON_BG='1',HERMES_PERSON_PROTECTION_V2='1',HERMES_PERSON_MODE='offline',HERMES_PERSON_TRACE='1',HERMES_FALL_REASON_TRACE='1',HERMES_TRACK_REPLAY='1',HERMES_PERSON_FG_REPLAY='1',HERMES_COAST_VALIDITY='0',HERMES_FALL_PERSON_CONFIRM='1',HERMES_EXCLUDE_TOP_ROW='1',HERMES_PERSON_MODEL=str(ROOT/'models/yolov7-tiny/yolov7-tiny.onnx'),HERMES_PERSON_RUNTIME=ready['runtime'])
 out=control/'yolo_bg';state.update(stage='full_videos',current='data4–13,15,16');dump(control/'status.json',state)
 cmd=[sys.executable,'-X','utf8',str(ROOT/'fall_eval/batch_eval.py'),'--runner',ready['runner'],'--experiment','baseline','--videos',*videos,'--strides','4','8','--output-dir',str(out)]
 with (control/'batch.log').open('w',encoding='utf-8') as log:
  proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   completion=out/'completion.json'
   if completion.exists():
    try:state['completed']=json.loads(completion.read_text(encoding='utf-8'))['completed'];dump(control/'status.json',state)
    except (ValueError,KeyError):pass
   time.sleep(30)
  if proc.returncode:raise RuntimeError('Full batch failed: see batch.log')
 done=json.loads((out/'completion.json').read_text(encoding='utf-8'));assert done['complete'] and done['completed']==24,done
 state.update(stage='data14_prefix',completed=24,current='data14 first6066 frames');dump(control/'status.json',state)
 os.environ.clear();os.environ.update(env)
 video=next((ROOT/'test_video_mp4_all').glob('data14*.mp4'));gt=batch.labels(ROOT/'fall_frameindex_data_all/data14_fall_frameindex.txt',0);assert all(end<6066 for start,end in gt),'GT exceeds decoded prefix'
 target=out/'data14';target.mkdir();snapshot=out/'inputs'
 with (control/'data14.log').open('w',encoding='utf-8') as log:
  original=sys.stdout;sys.stdout=log
  try:metrics,events=batch.execute_video(video,gt,[4,8],target,Path(ready['runner']),snapshot/'sdk_gray_count_falls.ini',snapshot/'Bed_Position.txt',6066)
  finally:sys.stdout=original
 per_video=read(out/'per_video.csv');per_event=read(out/'per_event.csv')
 for stride in [4,8]:
  rows=read(target/f'stride_{stride}/frames.csv');fps=metrics[[m['stride'] for m in metrics].index(stride)]['fps'];scored,details=batch.score(rows,gt,fps);row=dict(video='data14',**metrics[[m['stride'] for m in metrics].index(stride)],**scored);row['fp_per_hour']=row['fp']/(6066/fps/3600);per_video.append(row);per_event.extend(dict(video='data14',stride=stride,**x) for x in details)
 # Preserve first24 summary, then replace aggregate with all26.
 for name in ['per_video.csv','per_event.csv','summary.csv','completion.json']:(out/('full12_'+name)).write_bytes((out/name).read_bytes())
 def write(p,rows):
  fields=list(dict.fromkeys(k for row in rows for k in row));
  with p.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
 write(out/'per_video.csv',per_video);write(out/'per_event.csv',per_event)
 summary=[]
 for stride in [4,8]:
  group=[x for x in per_video if int(x['stride'])==stride];tp=sum(int(x['tp']) for x in group);fp=sum(int(x['fp']) for x in group);fn=sum(int(x['fn']) for x in group);seconds=sum(float(x['duration_seconds']) for x in group)
  summary.append(dict(stride=stride,videos=len(group),tp=tp,fp=fp,fn=fn,recall=tp/(tp+fn) if tp+fn else None,precision=tp/(tp+fp) if tp+fp else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,fp_per_hour=fp/(seconds/3600),duplicates=sum(int(x.get('duplicates',0)) for x in group),scope='12 full videos + data14 decoded prefix'))
 write(out/'summary.csv',summary);write(control/'summary.csv',summary)
 meta=json.loads((target/'video.json').read_text(encoding='utf-8'));meta.update(smoke_test=False,scope='decoded_prefix_6066_of_6114',results=[x for x in per_video if x['video']=='data14']);dump(target/'video.json',meta)
 for file in out.glob('data*/stride_*/person_bg_summary.json'):assert not json.loads(file.read_text(encoding='utf-8'))['failed'],str(file)
 dump(out/'completion.json',{'complete':True,'completed':26,'expected':26,'data14_scope':'6066-frame prefix'})
 dump(control/'validation.json',{'complete':True,'completed':26,'expected':26,'sdk_sources_verified':True,'data14_all_gt_inside_prefix':True,'full_videos':12,'partial_videos':1,'no_input_bins_generated':True})
 (control/'OPEN_UI.ps1').write_text('& "D:\\Lightglue\\lg_env\\Scripts\\python.exe" -X utf8 "'+str(HERE/'view.py')+'"\n',encoding='utf-8-sig')
 state.update(complete=True,stage='complete',completed=26,current=None);dump(control/'status.json',state);print('COMPLETE '+str(control/'summary.csv'),flush=True)
except Exception as e:
 state.update(stage='failed',error=str(e));dump(control/'status.json',state);raise
