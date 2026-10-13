"""Replay BG update snapshots. Select video and a stride directory produced with --trace."""
import argparse,csv,json
from pathlib import Path
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
import cv2,numpy as np
from PIL import Image,ImageTk
class Replay:
 def __init__(self,root,video,run):
  self.root=root;self.cap=cv2.VideoCapture(str(video))
  if not self.cap.isOpened():raise ValueError('Cannot open video')
  self.meta=json.loads((run/'person_bg_summary.json').read_text());self.blocks={};self.boxes={};self.frames={}
  for row in csv.DictReader((run/'frames.csv').open(encoding='utf-8-sig')):self.frames[int(row['timestamp_ms'])]=int(row['source_frame'])
  for row in csv.DictReader((run/'person_bg_blocks.csv').open(encoding='utf-8-sig')):self.blocks.setdefault(int(row['timestamp_ms']),[]).append(row)
  for row in csv.DictReader((run/'person_bg_boxes.csv').open(encoding='utf-8-sig')):self.boxes.setdefault(int(row['applied_timestamp_ms']),[]).append(row)
  self.times=sorted(self.blocks)
  if not self.times:raise ValueError('No applied BG snapshots in this run')
  self.index=0;root.title('人物BG更新快照');root.geometry('1150x760');root.protocol('WM_DELETE_WINDOW',self.close)
  bar=ttk.Frame(root);bar.pack(fill='x')
  ttk.Button(bar,text='上一輪',command=lambda:self.step(-1)).pack(side='left');ttk.Button(bar,text='下一輪',command=lambda:self.step(1)).pack(side='left')
  self.status=tk.StringVar();ttk.Label(root,textvariable=self.status,padding=6).pack(fill='x')
  self.canvas=tk.Canvas(root,bg='#202020');self.canvas.pack(fill='both',expand=True)
  self.scale=tk.Scale(root,from_=0,to=len(self.times)-1,orient='horizontal',command=self.seek);self.scale.pack(fill='x')
  ttk.Label(root,text='綠：人物保護／alpha 0；黃：motion或新變動／alpha 0；紅：更新（顏色越深alpha越大）；灰叉：未知BG。藍框是取圖時YOLO框，只作參考。',wraplength=1100).pack(fill='x')
  root.after(100,self.show)
 def seek(self,v):self.index=int(float(v));self.show()
 def step(self,n):self.index=max(0,min(len(self.times)-1,self.index+n));self.scale.set(self.index);self.show()
 def show(self):
  ts=self.times[self.index];frame=self.frames[ts];self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=self.cap.read()
  if not ok:messagebox.showerror('讀取失敗',f'Cannot decode frame {frame}');return
  w,h=self.meta['width'],self.meta['height'];cols,rows=self.meta['grid_cols'],self.meta['grid_rows'];img=cv2.resize(img,(w,h));overlay=img.copy();protected=0
  for row in self.blocks[ts]:
   b=int(row['block']);c,r=b%cols,b//cols;x1,y1=c*(w//cols),r*(h//rows);x2=w if c==cols-1 else (c+1)*(w//cols);y2=h if r==rows-1 else (r+1)*(h//rows);alpha=float(row['alpha'])
   color=(0,210,0) if row['reason']=='person_hold' else (0,210,255) if alpha==0 else (0,0,int(100+155*alpha/.32))
   if row['reason']=='person_hold':protected+=1
   cv2.rectangle(overlay,(x1,y1),(x2,y2),color,-1)
   cv2.putText(img,f'{alpha:.0%}',(x1+1,y1+13),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
   if row['valid']=='0':cv2.line(img,(x1,y1),(x2,y2),(150,150,150),1)
  img=cv2.addWeighted(img,.7,overlay,.3,0)
  source_times=[]
  for box in self.boxes.get(ts,[]):
   if box['kind']!='source':continue
   source_times.append(int(box['source_timestamp_ms']));coords=[int(float(box[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(img,tuple(coords[:2]),tuple(coords[2:]),(255,100,0),2)
  self.status.set(f'更新 {self.index+1}/{len(self.times)} | 原影片 frame {frame} | 採用 {(ts-1)/1000:.3f}s | 人物保護 {protected}/{cols*rows} blocks | YOLO取圖 '+(','.join(f'{(t-1)/1000:.3f}s' for t in sorted(set(source_times))) or '本次未檢出人物'))
  s=min(max(1,self.canvas.winfo_width())/w,max(1,self.canvas.winfo_height())/h);img=cv2.resize(img,(max(1,int(w*s)),max(1,int(h*s))));self.photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)));self.canvas.delete('all');self.canvas.create_image(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,image=self.photo)
 def close(self):self.cap.release();self.root.destroy()

class FgReplay(Replay):
 def __init__(self,root,video,run):
  self.run=run;self.fg_index={};self.frame_rows={};self.playing=False
  self.identity_rows={}
  if (run/'identity_a.jsonl').exists():
   with (run/'identity_a.jsonl').open(encoding='utf-8') as f:
    for line in f:
     entry=json.loads(line);self.identity_rows[entry['sdk_frame']]=entry['association']
  self.track_rows={};self.fg_rows={};self.exclude_top=False;self.confirm_checks=[]
  if (run/'person_confirmation_checks.csv').exists():
   with (run/'person_confirmation_checks.csv').open(encoding='utf-8-sig') as f:self.confirm_checks=list(csv.DictReader(f))
  for filename,target,key in [('track_replay.csv',self.track_rows,'sdk_frame'),('fg_objects.csv',self.fg_rows,'sdk_frame')]:
   if (run/filename).exists():
    with (run/filename).open(encoding='utf-8-sig') as f:
     for row in csv.DictReader(f):target.setdefault(int(row[key]),[]).append(row)
  for parent in [run.parents[1],run.parents[2]]:
   manifest=parent/'manifest.json'
   if manifest.exists():
    settings=json.loads(manifest.read_text(encoding='utf-8'))
    if 'exclude_top_row' in settings:self.exclude_top=settings['exclude_top_row'];break

  for row in csv.DictReader((run/'fg_replay/index.csv').open(encoding='utf-8-sig')):self.fg_index[int(row['timestamp_ms'])]=row
  for row in csv.DictReader((run/'frames.csv').open(encoding='utf-8-sig')):self.frame_rows[int(row['timestamp_ms'])]=row
  super().__init__(root,video,run)
  self.times=sorted(self.fg_index);self.index=0;self.scale.configure(to=len(self.times)-1)
  root.title('YOLO BG / 實際 FG 逐幀檢視')
  bar=ttk.Frame(root);bar.pack(fill='x',before=self.canvas)
  self.mode=tk.StringVar(value='FG 疊圖');choices=ttk.Combobox(bar,textvariable=self.mode,values=['FG 疊圖','FG 二值圖','BG','原圖','更新權重'],state='readonly',width=15);choices.pack(side='left');choices.bind('<<ComboboxSelected>>',lambda e:self.show())
  self.show_ids=tk.BooleanVar(value=True);ttk.Checkbutton(bar,text='motion / FG ID 與配對',variable=self.show_ids,command=self.show).pack(side='left')
  self.detail=tk.StringVar();ttk.Label(root,textvariable=self.detail,justify='left',font=('Consolas',10),wraplength=1100).pack(fill='x',before=self.canvas)
  self.jump=tk.StringVar(value='4576');ttk.Label(bar,text='原影片 frame：').pack(side='left');ttk.Entry(bar,textvariable=self.jump,width=9).pack(side='left');ttk.Button(bar,text='跳轉',command=self.jump_to).pack(side='left');ttk.Button(bar,text='播放／暫停',command=self.toggle).pack(side='left')
  ttk.Label(root,text='FG/BG 是該幀 BG 更新前的實際差分輸入；未知背景區已按SDK邏輯排除。更新權重顯示最近一次更新快照。上一輪／下一輪按鈕在此模式會切換相鄰SDK輸入幀。',wraplength=1100).pack(fill='x')
 def jump_to(self):
  try:target=int(self.jump.get())
  except ValueError:messagebox.showerror('參數錯誤','Frame 必須是整數');return
  self.playing=False;self.index=min(range(len(self.times)),key=lambda i:abs(self.frames[self.times[i]]-target));self.scale.set(self.index);self.show()
 def toggle(self):
  self.playing=not self.playing
  if self.playing:self.root.after(1,self.play_next)
 def play_next(self):
  if not self.playing:return
  if self.index+1>=len(self.times):self.playing=False;return
  self.step(1);self.root.after(130,self.play_next)
 def show(self):
  if not hasattr(self,'mode'):return
  ts=self.times[self.index];row=self.fg_index[ts];frame=self.frames[ts];self.cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,img=self.cap.read()
  if not ok:messagebox.showerror('讀取失敗',f'Cannot decode frame {frame}');return
  w,h=int(row['width']),int(row['height']);img=cv2.resize(img,(w,h),interpolation=cv2.INTER_AREA)
  mode=self.mode.get();fg=cv2.imread(str(self.run/'fg_replay'/row['fg']),cv2.IMREAD_GRAYSCALE)
  if fg is None or fg.shape!=(h,w):raise ValueError('Missing or invalid FG snapshot')
  if mode=='FG 疊圖':
   selected=fg!=0;color=np.zeros_like(img);color[:]=(0,255,0);img[selected]=cv2.addWeighted(img,.55,color,.45,0)[selected]
  elif mode=='FG 二值圖':img=cv2.cvtColor(fg,cv2.COLOR_GRAY2BGR)
  elif mode=='BG':
   bg=cv2.imread(str(self.run/'fg_replay'/row['bg']),cv2.IMREAD_GRAYSCALE)
   if bg is None:raise ValueError('Missing BG snapshot')
   img=cv2.cvtColor(bg,cv2.COLOR_GRAY2BGR)
  elif mode=='更新權重':
   previous=[t for t in self.blocks if t<=ts]
   if previous:
    updated=max(previous);overlay=img.copy();cols,rows=self.meta['grid_cols'],self.meta['grid_rows']
    for block in self.blocks[updated]:
     b=int(block['block']);c,r=b%cols,b//cols;x1,y1=c*(w//cols),r*(h//rows);x2=w if c==cols-1 else (c+1)*(w//cols);y2=h if r==rows-1 else (r+1)*(h//rows);a=float(block['alpha']);color=(0,210,0) if block['reason']=='person_hold' else (0,210,255) if a==0 else (0,0,int(100+155*a/.32));cv2.rectangle(overlay,(x1,y1),(x2,y2),color,-1);cv2.putText(img,f'{a:.0%}',(x1+1,y1+13),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,255,255),1)
    img=cv2.addWeighted(img,.7,overlay,.3,0)
  sid=int(row['sdk_frame']);tracks=self.track_rows.get(sid,[]);fgs=self.fg_rows.get(sid,[])
  lines=['ID | origin | valid momentum | FG | phase / count | evidence | confirmation']
  for t in tracks:
   lines.append(f"T{t['track_id']} | {t['origin']} | {t['strength'] if t['momentum_valid']=='1' else 'unmeasured'} | {'F'+t['fg_id'] if t['fg_checked']=='1' else 'not checked'} | {t['phase']} / {t['observed']}/{t.get('observation_target','45')} (M:{t['measured_samples']} Q:{t['quiet_samples']}) | {t['evidence']} | {t['confirmation']}")
  for t in tracks:
   evidence=[c for c in self.confirm_checks if c['track_id']==t['track_id'] and int(c['timestamp_ms'])<=ts]
   if evidence:
    newest=max(int(c['timestamp_ms']) for c in evidence);current=[c for c in evidence if int(c['timestamp_ms'])==newest]
    lines.append('T'+t['track_id']+' YOLO checks: '+'; '.join(f"{c['stage']} #{c['request_id']} @{int(c['source_ms'])/1000:.2f}s class={c['class_id']} overlap={float(c['candidate_overlap']):.0%}/{float(c['person_overlap']):.0%} {'MATCH' if c['matched']=='1' else 'no match'}" for c in current))
  identity=self.identity_rows.get(sid)
  if identity:
   maturity={t['id']:t['mature'] for t in identity['person_tracks']}
   lines.append('A ownership: '+'; '.join(f"T{t['id']} {'mature' if maturity.get(t['id']) else 'tentative'} / {t['state']} / F{t['assisted_id']}" for t in identity['tracks']))
   if identity['events']:lines.append('A events: '+'; '.join(f"{e['type']} T{e['id']} other={e['other']}" for e in identity['events']))
  self.detail.set('\n'.join(lines) if tracks else '此結果未提供逐幀物件 log。')
  if self.show_ids.get():
   centers={}
   for f in fgs:
    box=[int(float(f[k])) for k in ('x1','y1','x2','y2')];centers[f['fg_id']]=(int(float(f['cx'])),int(float(f['cy'])))
    cv2.rectangle(img,tuple(box[:2]),tuple(box[2:]),(255,255,0),1);cv2.putText(img,'F'+f['fg_id'],(box[0],max(12,box[1])),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,0),1)
   for t in tracks:
    box=[int(float(t[k])) for k in ('x1','y1','x2','y2')];color=(0,128,255) if t['momentum_valid']=='0' else (255,0,255)
    cv2.rectangle(img,tuple(box[:2]),tuple(box[2:]),color,2);cv2.putText(img,'T'+t['track_id']+' '+t['origin'],(box[0],min(h-5,box[3]+12)),cv2.FONT_HERSHEY_SIMPLEX,.4,color,1)
    if t['fg_checked']=='1' and t['fg_id'] in centers:cv2.line(img,((box[0]+box[2])//2,(box[1]+box[3])//2),centers[t['fg_id']],color,1)
  latest=max((t for t in self.boxes if t<=ts),default=None)
  if latest is not None and self.show_ids.get():
   for ybox in self.boxes[latest]:
    if ybox['kind']!='source':continue
    bb=[int(float(ybox[k])) for k in ('x1','y1','x2','y2')];cv2.rectangle(img,tuple(bb[:2]),tuple(bb[2:]),(255,100,0),1);cv2.putText(img,'YOLO '+({'0':'normal','1':'fallen'}.get(ybox.get('class_id'),'person'))+' '+format(float(ybox['confidence']),'.2f')+' @'+str(round((int(ybox['source_timestamp_ms'])-1)/1000,1))+'s',(bb[0],min(h-5,bb[1]+14)),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,100,0),1)
  if self.exclude_top:
   top=h//self.meta['grid_rows'];img[:top]=60
   for x in range(0,w,16):cv2.line(img,(x,0),(x+top,top-1),(150,150,150),1)
   cv2.putText(img,'EXCLUDED',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
  state=self.frame_rows[ts];fall=state['is_fall']=='1';label='判定跌倒' if fall else '未判跌倒';ids=state['fall_object_ids']
  self.status.set(f'原影片 frame {frame} | SDK frame {row["sdk_frame"]} | {(ts-1)/1000:.3f}s | FG pixels {np.count_nonzero(fg)} | {label} {ids}')
  if fall:cv2.rectangle(img,(0,0),(w-1,h-1),(0,0,255),4)
  z=min(max(1,self.canvas.winfo_width())/w,max(1,self.canvas.winfo_height())/h);img=cv2.resize(img,(max(1,int(w*z)),max(1,int(h*z))));self.photo=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)));self.canvas.delete('all');self.canvas.create_image(self.canvas.winfo_width()/2,self.canvas.winfo_height()/2,image=self.photo)
 def close(self):self.playing=False;super().close()

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--video',type=Path);p.add_argument('--run',type=Path);a=p.parse_args();root=tk.Tk();root.withdraw()
 try:
  video=str(a.video) if a.video else filedialog.askopenfilename(title='選原始MP4',filetypes=[('MP4','*.mp4')])
  run=str(a.run) if a.run else filedialog.askdirectory(title='選含person_bg_blocks.csv的stride資料夾')
  if video and run:app=(FgReplay if (Path(run)/"fg_replay/index.csv").exists() else Replay)(root,video,Path(run));root.deiconify();root.mainloop()
  else:root.destroy()
 except Exception as e:messagebox.showerror('無法開啟',str(e));root.destroy()
