"""Controlled continuation: identical BF16 compute, differing AdamW/master precision."""
from project_paths import ROOT, SMOOTH_VQ, QWEN_STAGE1_DIR, QWEN_PRECISION_DIR
from training_runs import latest_training_run
import hashlib,json,random,sys,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration

SRC=latest_training_run(QWEN_STAGE1_DIR)
RUN=QWEN_PRECISION_DIR;arm=sys.argv[1]
assert arm in ('bf16','fp32_master')
OUT=RUN/arm;OUT.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(ROOT))
from aligned_digit_data import load
from aligned_vq_model import AlignedVQ
torch.set_num_threads(4)
xy,lab,_=load();rng=random.Random(42);tr=[];va=[]
for c in range(36):
 ix=np.flatnonzero(lab==c).tolist();rng.shuffle(ix);n=max(1,round(len(ix)*.15));va+=ix[:n];tr+=ix[n:]
v=AlignedVQ().cuda().eval();v.load_state_dict(torch.load(SMOOTH_VQ,weights_only=False)['model'])
with torch.no_grad():raw=v.encode(torch.tensor(xy,device='cuda'))[1].cpu().numpy()
del v;torch.cuda.empty_cache()
tok=AutoTokenizer.from_pretrained(SRC)
chars='0123456789NBACDEFGHIJKLMOPQRSTUVWXYZ';system='You generate handwriting codes. Output exactly 16 code tokens.'
templates=['Draw the character {c}.','Generate handwriting for {c}.','Write character {c}.']
prefixes={(c,t):tok.apply_chat_template([{'role':'system','content':system},{'role':'user','content':p.format(c=c)}],tokenize=True,add_generation_prompt=True,enable_thinking=False)['input_ids'] for c in chars for t,p in enumerate(templates)}
draw=tok.convert_tokens_to_ids('<DRAW>');end=tok.convert_tokens_to_ids('<END_DRAW>')
cid=np.array([tok.convert_tokens_to_ids(f'<M{i:03d}>') for i in range(256)])
schedule_rng=random.Random(20261015);schedule=[]
for ep in range(10):
 order=tr.copy();schedule_rng.shuffle(order);schedule.append([(i,schedule_rng.randrange(3)) for i in order])
schedule_hash=hashlib.sha256(json.dumps(schedule).encode()).hexdigest()
def batch(rows):
 seq=[];starts=[]
 for i,t in rows:
  pre=prefixes[chars[lab[i]],t];starts.append(len(pre));seq.append(pre+[draw]+cid[raw[i]].tolist()+[end,tok.eos_token_id])
 x=torch.full((len(rows),max(map(len,seq))),tok.pad_token_id,dtype=torch.long,device='cuda');mask=torch.zeros_like(x);y=torch.full_like(x,-100)
 for j,(s,p) in enumerate(zip(seq,starts)):
  x[j,:len(s)]=torch.tensor(s,device='cuda');mask[j,:len(s)]=1;y[j,p:len(s)]=x[j,p:len(s)]
 return x,mask,y,starts

torch.manual_seed(20261015)
m=Qwen3_5ForConditionalGeneration.from_pretrained(SRC,dtype=torch.bfloat16,attn_implementation='eager').cuda()
m.config.use_cache=False;m.gradient_checkpointing_enable()
named=list(m.named_parameters());params=[p for _,p in named]
masters=[torch.nn.Parameter(p.detach().float().clone()) for p in params] if arm=='fp32_master' else params
opt=torch.optim.AdamW(masters,lr=1e-5,weight_decay=.01)
meta={'arm':arm,'source':str(SRC),'epochs':10,'batch':4,'lr':1e-5,'seed':20261015,'schedule_sha256':schedule_hash,'train':len(tr),'validation':len(va),'compute_dtype':'bfloat16','master_dtype':'float32' if arm=='fp32_master' else 'bfloat16','optimizer_reset_for_both':True,'selection':'fixed epoch 10, no best-checkpoint selection','source_weights_sha256':hashlib.sha256((SRC/'model.safetensors').read_bytes()).hexdigest()}
(OUT/'experiment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
@torch.no_grad()
def evaluate():
 m.eval();right=0;total=0;nll=0;nt=0
 for k in range(0,len(va),4):
  rows=[(i,0) for i in va[k:k+4]];x,mask,y,starts=batch(rows);o=m(input_ids=x,attention_mask=mask,labels=y,use_cache=False)
  count=int((y[:,1:]!=-100).sum());nll+=float(o.loss)*count;nt+=count
  for j,p in enumerate(starts):
   logits=o.logits[j,p:p+16,torch.tensor(cid,device='cuda')];target=torch.tensor(raw[rows[j][0]],device='cuda');right+=int((logits.argmax(-1)==target).sum());total+=16
 return {'loss':nll/nt,'code_accuracy':right/total}
print(json.dumps({'epoch':0,'validation':evaluate()}),flush=True)
started=time.time()
for ep,rows in enumerate(schedule,1):
 torch.manual_seed(20261015+ep);m.train();ls=[]
 for k in range(0,len(rows),4):
  m.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
  x,mask,y,_=batch(rows[k:k+4]);loss=m(input_ids=x,attention_mask=mask,labels=y,use_cache=False).loss;loss.backward()
  # Both arms use identical gradient computation and clipping precision.
  torch.nn.utils.clip_grad_norm_(params,1)
  if arm=='fp32_master':
   for p,q in zip(params,masters):q.grad=None if p.grad is None else p.grad.float()
  opt.step()
  if arm=='fp32_master':
   with torch.no_grad():
    for p,q in zip(params,masters):
     if q.grad is not None:p.copy_(q)
  ls.append(float(loss.detach()))
  if ep==1 and k==0:
   state=next(iter(opt.state.values()));print(json.dumps({'first_step_ok':True,'optimizer_state_dtype':str(state['exp_avg'].dtype),'gpu_peak_gb':torch.cuda.max_memory_allocated()/1e9}),flush=True)
 rec={'epoch':ep,'train_loss':float(np.mean(ls)),'validation':evaluate(),'seconds':round(time.time()-started)}
 print(json.dumps(rec),flush=True)
 with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
m.save_pretrained(OUT,safe_serialization=True);tok.save_pretrained(OUT)
if arm=='fp32_master':torch.save({n:q.detach().cpu() for (n,_),q in zip(named,masters)},OUT/'fp32_master_weights.pt')
(OUT/'TRAIN_DONE').write_text('completed all 10 epochs\n');print('TRAIN_DONE',flush=True)

