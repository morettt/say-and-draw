"""Mixed text/drawing continuation of the best digit+A-Z checkpoint."""
from project_paths import ROOT, SMOOTH_VQ, QWEN_MULTITURN_DIR, QWEN_CONFIRMATION_DIR
import json,random,time,sys,gc
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
SRC=QWEN_MULTITURN_DIR/'best';OUT=QWEN_CONFIRMATION_DIR
OUT.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ROOT))
from aligned_digit_data import load
from aligned_vq_model import AlignedVQ
from qwen_chat_data import load_rows
SYSTEM='You are a helpful assistant. Reply normally. For drawing, say a short sentence and then emit <DRAW>, 16 motion codes, <END_DRAW>. Do not emit <CHAT>.'
CHARS='0123456789NBACDEFGHIJKLMOPQRSTUVWXYZ'
torch.set_num_threads(4);torch.manual_seed(1610);rng=random.Random(1610)
xy,lab,_=load();split=random.Random(42);tr=[];va=[]
for c in range(36):
 ix=np.flatnonzero(lab==c).tolist();split.shuffle(ix);n=max(1,round(len(ix)*.15));va+=ix[:n];tr+=ix[n:]
v=AlignedVQ().cuda().eval();v.load_state_dict(torch.load(SMOOTH_VQ,weights_only=False)['model'])
with torch.no_grad():raw=v.encode(torch.tensor(xy,device='cuda'))[1].cpu().numpy()
del v;torch.cuda.empty_cache()
tok=AutoTokenizer.from_pretrained(SRC);draw=tok.convert_tokens_to_ids('<DRAW>');end=tok.convert_tokens_to_ids('<END_DRAW>');cid=[tok.convert_tokens_to_ids(f'<M{i:03d}>') for i in range(256)];ct=torch.tensor(cid,device='cuda')
def prefix(q,history=None):return tok.apply_chat_template([{'role':'system','content':SYSTEM}]+(history or [])+[{'role':'user','content':q}],tokenize=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
def textrow(q,a,history=None):
 p=prefix(q,history);return (p+tok.encode(a,add_special_tokens=False)+[tok.eos_token_id],len(p),None)
templates=['请帮我写一个{kind}{c}。','画一个{kind}{c}给我看看。','请用手写的方式写出{c}。','帮我画{c}。','我想看看{c}的笔迹，请画出来。','写一个{c}。','请画一个{kind}{c}。','Draw the character {c}.','Write character {c}.','Generate handwriting for {c}.']
def drawrow(i,t=0):
 c=CHARS[lab[i]];q=templates[t].format(c=c,kind='数字' if c.isdigit() else '字母');intro=f'好的，我给你写一个{c}。\n' if t<7 else f'Sure, here is {c}.\n';p=prefix(q);a=tok.encode(intro,add_special_tokens=False)+[draw];start=len(p)+len(a)
 return p+a+[cid[int(z)] for z in raw[i]]+[end,tok.eos_token_id],len(p),start
CHAT_FILE=ROOT/'data'/'qwen'/'chat_stage5.jsonl'
chat,chat_raw=load_rows(CHAT_FILE)
chat_rows=[textrow(q,a,h) for q,a,h in chat]
draw_rows={i:[drawrow(i,t) for t in range(len(templates))] for i in tr+va}
# Inserted into the training script after single-turn rows have been constructed.
byclass={c:[i for i in tr if lab[i]==c] for c in range(36)}
hr=random.Random(9016)
def ink_answer(i):
 c=CHARS[lab[i]]
 return f'好的，我给你写一个{c}。\n<DRAW>'+''.join(f'<M{int(z):03d}>' for z in raw[i])+'<END_DRAW>'
def make_history(classes):
 h=[]
 for c in classes:
  i=hr.choice(byclass[c]);h.extend([{'role':'user','content':f'帮我写一个{CHARS[c]}。'},{'role':'assistant','content':ink_answer(i)}])
 return h
def make_draw(i,q,h):
 c=CHARS[lab[i]];p=prefix(q,h);a=tok.encode(f'好的，这次写{c}。\n',add_special_tokens=False)+[draw]
 return p+a+[cid[int(z)] for z in raw[i]]+[end,tok.eos_token_id],len(p),len(p)+len(a)
for i in tr:
 c=CHARS[lab[i]]
 # Chinese-only new draw instructions, plus the original Chinese single-turn rows.
 draw_rows[i]=draw_rows[i][:7]
 for j,q in enumerate([f'接下来改成{c}。',f'刚才那个不要了，写{c}。',f'你画错了，我要的是{c}，请重新写。',f'现在请写{c}，不要重复前一个。',f'这次我要的是{"数字" if c.isdigit() else "字母"}{c}。',f'谢谢。再帮我画一个{c}。']):
  previous=hr.randrange(10) if not c.isdigit() else hr.randrange(10,36)
  sequence=[previous] if j%2==0 else [hr.randrange(36),previous]
  draw_rows[i].append(make_draw(i,q,make_history(sequence)))
  old=CHARS[previous]
  draw_rows[i].append(make_draw(i,f'不要写{old}了，我现在要{c}。',make_history(sequence)))
 draw_rows[i].append(make_draw(i,'再写一遍刚才的那个。',make_history([lab[i]])))
(OUT/'multiturn_recipe.json').write_text(json.dumps({'draw_variants_per_sample':len(draw_rows[tr[0]]),'chat_rows':len(chat_rows),'history_contains_actual_codes':True,'history_source':'training split only','seed':9016},ensure_ascii=False,indent=2))

for i in tr:
 target=CHARS[lab[i]]
 for j,confirm in enumerate(['那你画吧','那就画出来吧','好，开始吧','可以，写出来','行，帮我写吧','那你试试看','好，画给我看','现在画吧','那你画吧','好，开始吧','那就写吧','开始']):
  oldclass=hr.choice([x for x in range(36) if x!=lab[i]]);old=CHARS[oldclass];oldi=hr.choice(byclass[oldclass])
  h=[{'role':'user','content':f'能给我画一个{old}吗'},{'role':'assistant','content':f'能，我可以画{old}。'},{'role':'user','content':'那你画吧'},{'role':'assistant','content':ink_answer(oldi)}]
  alias=target.lower() if j%2 and target.isalpha() else target
  h.extend([{'role':'user','content':f'能再给我画一个{"字母" if target.isalpha() else "数字"}{alias}吗'},{'role':'assistant','content':f'能，我可以画{alias}。'}])
  draw_rows[i].append(make_draw(i,confirm,h))
  if j<4:
   # New target comes before the forbidden old target, too.
   draw_rows[i].append(make_draw(i,f'我要{target}，不要再写{old}。',h[:4]))
 # An unqualified typed lowercase letter refers to its trained capital form;
 # explicit requests for lowercase remain capability clarification examples.
 if target.isalpha():
  draw_rows[i].append(make_draw(i,f'帮我写字母{target.lower()}。',[]))
(OUT/'training_data.json').write_text(json.dumps({'chat':chat_raw,'chat_source':str(CHAT_FILE),'draw_templates':templates,'train_indices':tr,'valid_indices':va,'system':SYSTEM},ensure_ascii=False,indent=2))
def batch(rows):
 x=torch.full((len(rows),max(len(r[0]) for r in rows)),tok.pad_token_id,device='cuda',dtype=torch.long);mask=torch.zeros_like(x);y=torch.full_like(x,-100)
 for j,(s,p,_) in enumerate(rows):x[j,:len(s)]=torch.tensor(s,device='cuda');mask[j,:len(s)]=1;y[j,p:len(s)]=x[j,p:len(s)]
 return x,mask,y
m=Qwen3_5ForConditionalGeneration.from_pretrained(SRC,dtype=torch.bfloat16,attn_implementation='eager').cuda();m.config.use_cache=False;m.gradient_checkpointing_enable()
named=list(m.named_parameters());params=[p for _,p in named]
masters=[torch.nn.Parameter(p.detach().float().clone()) for p in params]
opt=torch.optim.AdamW(masters,lr=2e-6,weight_decay=.01)
probes_chat=['你好呀，今天怎么样？','请只用文字告诉我5加3的结果。','不要画图，解释一下字母A。','谢谢你的帮助！','和我聊聊熊猫吧。','先别画图，我们聊聊今天的心情。']
probes_draw=['请替我手写数字3。','麻烦写一下数字5。','我想看大写A的手写样子，请写出来。','请写出字母Z给我看。']
@torch.no_grad()
def probe(q,h=None):
 p=prefix(q,h);x=torch.tensor([p],device='cuda');generated=[]
 for _ in range(64):
  n=int(m(input_ids=x,use_cache=False,logits_to_keep=1).logits[0,-1].argmax());generated.append(n)
  if n in [tok.eos_token_id,draw] or n in cid:break
  x=torch.cat([x,torch.tensor([[n]],device='cuda')],1)
 return {'text':tok.decode(generated,skip_special_tokens=False),'draw':draw in generated,'stray_code':any(n in cid for n in generated),'eos':tok.eos_token_id in generated}
@torch.no_grad()
def evaluate():
 m.eval();right=tot=0
 for off in range(0,len(va),4):
  rows=[draw_rows[i][0] for i in va[off:off+4]];x,mask,y=batch(rows);z=m(input_ids=x,attention_mask=mask,use_cache=False).logits
  for j,(_,_,st) in enumerate(rows):right+=int((z[j,st-1:st+15,ct].argmax(-1)==torch.tensor(raw[va[off+j]],device='cuda')).sum());tot+=16
 samples={q:probe(q) for q in probes_chat+probes_draw}
 chat_ok=sum(not samples[q]['draw'] and not samples[q]['stray_code'] and samples[q]['eos'] and len(samples[q]['text'])>12 for q in probes_chat)
 draw_ok=sum(samples[q]['draw'] for q in probes_draw)
 multi=[]
 for old,new in [('3','A'),('5','Z'),('A','7'),('B','K'),('8','D'),('X','2')]:
  h=make_history([CHARS.index(old)]);r=probe(f'换一个，请写{new}。',h)
  import re
  mentioned=re.findall(r'[A-Z0-9]',r['text'].split('<DRAW>')[0])
  multi.append({'old':old,'expected':new,**r,'target_ok':r['draw'] and bool(mentioned) and mentioned[-1]==new})
 stops=[probe('停止画图，只回答：刚才写了什么？',make_history([CHARS.index(c)])) for c in '35AB']
 confirm_checks=[]
 for old,new in [('4','W'),('3','B'),('A','7'),('5','K'),('9','M'),('Z','2')]:
  oldi=byclass[CHARS.index(old)][0]
  h=[{'role':'user','content':f'能画{old}吗？'},{'role':'assistant','content':f'能，我可以画{old}。'},{'role':'user','content':'那你画吧'},{'role':'assistant','content':ink_answer(oldi)},{'role':'user','content':f'能再给我画{new.lower()}吗'},{'role':'assistant','content':f'可以画{new.lower()}。'}]
  r=probe('那你画吧',h);mentioned=re.findall(r'[A-Z0-9]',r['text'].split('<DRAW>')[0]);confirm_checks.append({'old':old,'target':new,**r,'ok':r['draw'] and bool(mentioned) and mentioned[-1]==new})
 return {'confirmation_checks':confirm_checks,'confirmation_ok':sum(r['ok'] for r in confirm_checks),'code_accuracy':right/tot,'chat_plain':chat_ok,'chat_total':len(probes_chat),'draw_mode':draw_ok,'draw_total':len(probes_draw),'samples':samples,'multi':multi,'multi_target_ok':sum(r['target_ok'] for r in multi),'stop_ok':sum(not r['draw'] and not r['stray_code'] and r['eos'] for r in stops),'stops':stops}
meta={'source':str(SRC),'epochs':3,'batch_draw':2,'batch_chat':2,'lr':2e-6,'seed':1610,'master_dtype':'fp32','compute_dtype':'bf16','chat_rows':len(chat_rows),'draw_train':len(tr),'draw_valid':len(va),'selection':'maximize chat/draw probe success, then validation code accuracy; probes are development checks, not independent final test'}
(OUT/'experiment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
print(json.dumps({'epoch':0,'validation':evaluate()},ensure_ascii=False),flush=True)
best=(-1,-1);start=time.time()
for ep in range(1,4):
 rng.shuffle(tr);m.train();ls=[]
 for off in range(0,len(tr),2):
  rows=[rng.choice(draw_rows[i]) for i in tr[off:off+2]];rows+=rng.choices(chat_rows,k=len(rows));rng.shuffle(rows)
  m.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True);x,mask,y=batch(rows)
  loss=m(input_ids=x,attention_mask=mask,labels=y,use_cache=False).loss;loss.backward();torch.nn.utils.clip_grad_norm_(params,1)
  for p,q in zip(params,masters):q.grad=None if p.grad is None else p.grad.float()
  opt.step()
  with torch.no_grad():
   for p,q in zip(params,masters):
    if q.grad is not None:p.copy_(q)
  ls.append(float(loss.detach()))
  if ep==1 and off==0:print(json.dumps({'first_step_ok':True,'peak_gb':torch.cuda.max_memory_allocated()/1e9}),flush=True)
 rec={'epoch':ep,'train_loss':float(np.mean(ls)),'seconds':round(time.time()-start)}
 if ep%1==0:
  val=evaluate();rec['validation']=val;score=(val['confirmation_ok'],val['chat_plain']+val['draw_mode']+val['multi_target_ok']+val['stop_ok'],val['code_accuracy'])
  if score>best:
   best=score;m.save_pretrained(OUT/'best',safe_serialization=True);tok.save_pretrained(OUT/'best');(OUT/'best_meta.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2))
 print(json.dumps(rec,ensure_ascii=False),flush=True)
 with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
(OUT/'TRAIN_DONE').write_text('3 epochs completed')
print('TRAIN_DONE',flush=True)

