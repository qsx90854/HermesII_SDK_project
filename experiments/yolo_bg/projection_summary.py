"""Summarize projection evaluations, including rejected candidates, from saved text traces."""
import csv,re,json
from pathlib import Path
FIELDS=['video','stride','track_id','observation_start_source_frame','evaluation_source_frame','evaluation_sdk_frame','mean_area','median_area','max_area','final_area','valid_projection_samples','observed_frames','evaluation_events','reported_at_evaluation','stats_source']
def read(path):
 with path.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def fields(message):return dict(re.findall(r'([a-z_]+)=([-+]?[0-9]+(?:[.][0-9]+)?(?:[eE][-+]?[0-9]+)?)',message))
def summarize(root):
 output=[]
 for file in sorted(Path(root).rglob('stride_*/fall_reasons.csv')):
  run=file.parent;frames={r['sdk_frame']:r for r in read(run/'frames.csv')};reasons=read(file);same={};buffers={};evaluations=[]
  for r in reasons:same.setdefault((r['sdk_frame'],r['track_id']),[]).append(r)
  for r in reasons:
   track=r['track_id'];event=r['event']
   if event in ['WAIT_START','OBS_START']:buffers[track]=[]
   if event=='PROJ_SAMPLE':buffers.setdefault(track,[]).append(r)
   if event=='PROJ_FINAL':evaluations.append((r,list(buffers.get(track,[]))))
   # Bed-filtered candidates skip geometry checks. Preserve their rounded per-sample statistics too.
   if event=='FILTER_BED' and not any(x['event']=='PROJ_FINAL' for x in same[(r['sdk_frame'],track)]):
    samples=list(buffers.get(track,[]));values=sorted(float(x['value']) for x in samples)
    if values:
     meta=fields(samples[-1]['message']);synthetic=dict(r,event='PROJ_FINAL',value=str(float(samples[-1]['value'])),message=f"mean_area={sum(values)/len(values)}|median_area={values[len(values)//2]}|max_area={max(values)}|samples={len(values)}|observed={meta.get('observed',len(values))}");evaluations.append((synthetic,samples))
  for final,samples in evaluations:
   meta=fields(final['message']);count=int(meta['samples']);samples=samples[-count:] if count else [];assert len(samples)==count,(run,final)
   fr=frames[final['sdk_frame']];events=same[(final['sdk_frame'],final['track_id'])];direct=any(x['event']=='PROJ_FINAL' for x in events)
   output.append(dict(video=run.parent.name,stride=int(run.name.split('_')[1]),track_id=final['track_id'],observation_start_source_frame=frames[samples[0]['sdk_frame']]['source_frame'] if samples else '',evaluation_source_frame=fr['source_frame'],evaluation_sdk_frame=final['sdk_frame'],mean_area=meta['mean_area'],median_area=meta['median_area'],max_area=meta['max_area'],final_area=final['value'],valid_projection_samples=count,observed_frames=meta['observed'],evaluation_events=';'.join(dict.fromkeys(x['event'] for x in events if x['event'] not in ['OBS_SAMPLE','FG_ASSOC','PROJ_SAMPLE','PROJ_FINAL'])),reported_at_evaluation=int(fr['is_fall']=='1' and final['track_id'] in fr['fall_object_ids'].split(';')),stats_source='CPP_PROJ_FINAL' if direct else 'rounded_PROJ_SAMPLE'))
  # Every real alarm has a reference to its latest projection evaluation; persistent frames form one event.
 alarms=[]
 for file in sorted(Path(root).rglob('stride_*/frames.csv')):
  run=file.parent;previous=False
  for fr in read(file):
   positive=fr['is_fall']=='1'
   if positive and not previous:
    for track in filter(None,fr['fall_object_ids'].split(';')):
     matches=[r for r in output if r['video']==run.parent.name and r['stride']==int(run.name.split('_')[1]) and r['track_id']==track and int(r['evaluation_source_frame'])<=int(fr['source_frame'])]
     alarm=dict(alarm_source_frame=fr['source_frame'],video=run.parent.name,stride=int(run.name.split('_')[1]),track_id=track,projection_matched=bool(matches))
     if matches:alarm.update(max(matches,key=lambda r:int(r['evaluation_source_frame'])))
     alarms.append(alarm)
   previous=positive
 def write(path,data,keys):
  with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
 root=Path(root);write(root/'projection_event_summary.csv',output,FIELDS)
 alarm_fields=['alarm_source_frame','projection_matched']+FIELDS
 for a in alarms:
  for key in alarm_fields:a.setdefault(key,'')
 write(root/'fall_projection_summary.csv',alarms,alarm_fields)
 return {'evaluations':len(output),'alarm_tracks':len(alarms),'alarms_missing_projection':sum(not r['projection_matched'] for r in alarms)}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();print(json.dumps(summarize(args.root)))
