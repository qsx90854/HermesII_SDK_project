import os,sys,subprocess,struct,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];out=ROOT/'fall_eval/results/top_row_synthetic';out.mkdir(exist_ok=True)
from datetime import datetime
out=out/datetime.now().strftime('%Y%m%d_%H%M%S_%f');out.mkdir()
results=[]
for mode in ['zero','top_noise','second_row_noise']:
 d=out/mode;d.mkdir();env={k:v for k,v in os.environ.items() if not k.startswith('HERMES_')};env.update(HERMES_EXCLUDE_TOP_ROW='1',HERMES_TRACK_REPLAY='1',HERMES_PERSON_BG='0',HERMES_BG_STABILITY='0',HERMES_PERSON_FG_REPLAY='1')
 bed=d/'bed.txt';bed.write_text('200,50\n600,50\n650,400\n150,400\n');log=(d/'runner.log').open('w');proc=subprocess.Popen([str(ROOT/'fall_eval/build_fall_safety/fall_eval_runner.exe'),str(ROOT/'Parameter/sdk_gray_count_falls.ini'),str(bed),str(d/'frames.csv'),'800','450'],cwd=d,env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
 for n in range(90):
  im=np.full((450,800),80,dtype=np.uint8)
  if mode=='top_noise':im[:28]=255 if n%2 else 0
  elif mode=='second_row_noise':im[28:56]=255 if n%2 else 0
  proc.stdin.write(struct.pack('<QQ',n*4,n*133+1)+im.tobytes())
 proc.stdin.write(struct.pack('<Q',2**64-1));proc.stdin.close();assert proc.wait(timeout=30)==0;log.close()
 def read(name):return list(csv.DictReader((d/name).open(encoding='utf-8-sig')))
 rows=read('frames.csv');keys=['source_frame','sdk_frame','timestamp_ms','is_fall','fall_object_ids'];results.append([[x[k] for k in keys] for x in rows]);objs=read('track_replay.csv');fg=read('fg_objects.csv');assert all(float(x['y1'])>=28 for x in fg);assert all(all(int(b)>=12 for b in x['blocks'].split(';') if b) for x in objs)
 if mode!='second_row_noise':assert not objs and not fg
 else:assert objs,'positive control failed: second row must remain active'
assert results[0]==results[1]
(out/'validation.json').write_text(json.dumps({'complete':True,'top_noise_invariant':True,'second_row_active':True},indent=2));print('PASS top row invariance and active second row:',out)
