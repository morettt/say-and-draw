from project_paths import QWEN_BASE_DIR, QWEN_STAGE1_DIR
from qwen_model_setup import ensure_qwen_model
from training_runs import create_training_run, complete_training_run
import argparse
import json,random,time
import sys
from pathlib import Path
import numpy as np, torch
from transformers import Qwen3_5ForConditionalGeneration, AutoTokenizer
import sys
PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT))
from data_tools.qwen import load_handwriting, encode_example
parser=argparse.ArgumentParser()
parser.add_argument('--output-dir',type=Path,help='本次训练的独立输出目录；已存在的目录不会覆盖')
args=parser.parse_args()
SRC=ensure_qwen_model(model_dir=QWEN_BASE_DIR); OUT=create_training_run(QWEN_STAGE1_DIR,args.output_dir)
print(json.dumps({'stage':'output','path':str(OUT)},ensure_ascii=False),flush=True)
CHARS='0123456789NBACDEFGHIJKLMOPQRSTUVWXYZ'; SYSTEM='你负责生成手写笔迹编号。输出恰好16个编号。'
random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4); dev='cuda'
dataset=load_handwriting('stage1')
lab=np.asarray(dataset.labels);raw=np.asarray(dataset.codes);tr=dataset.train.copy();va=dataset.validation.copy()
SYSTEM=dataset.system

tok=AutoTokenizer.from_pretrained(SRC,local_files_only=True)
special=['<DRAW>','<END_DRAW>']+[f'<M{i:03d}>' for i in range(256)];tok.add_special_tokens({'additional_special_tokens':special})
draw=tok.convert_tokens_to_ids('<DRAW>');end=tok.convert_tokens_to_ids('<END_DRAW>');code_ids=np.array([tok.convert_tokens_to_ids(f'<M{i:03d}>') for i in range(256)])
model=Qwen3_5ForConditionalGeneration.from_pretrained(SRC,local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').to(dev);model.resize_token_embeddings(len(tok));model.config.use_cache=False;model.gradient_checkpointing_enable()
def seq(i):
 row=random.choice(dataset.examples[i]);s,p,_=encode_example(tok,row);return s,p
def batch(rows):
 w=max(len(x[0]) for x in rows);ids=torch.full((len(rows),w),tok.pad_token_id,dtype=torch.long);mask=torch.zeros_like(ids);ys=torch.full_like(ids,-100)
 for j,(s,p) in enumerate(rows):ids[j,:len(s)]=torch.tensor(s);mask[j,:len(s)]=1;ys[j,p:len(s)]=ids[j,p:len(s)]
 return ids.to(dev),mask.to(dev),ys.to(dev)
@torch.no_grad()
def ev(ids):
 model.eval();right=tot=0;losses=[]
 for k in range(0,len(ids),4):
  rows=[seq(i) for i in ids[k:k+4]];x,m,y=batch(rows);o=model(input_ids=x,attention_mask=m);z=o.logits[:,:-1];t=y[:,1:];good=t>=0;losses.append(float(torch.nn.functional.cross_entropy(z.float().reshape(-1,z.shape[-1]),t.reshape(-1),ignore_index=-100)));p=z.argmax(-1)
  for j,(_,pr) in enumerate(rows):right+=int((p[j,pr:pr+17]==t[j,pr:pr+17]).sum());tot+=17
 return {'loss':float(np.mean(losses)),'code_token_accuracy':right/tot,'samples':len(ids)}
opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01);best=1e9;start=time.time()
print(json.dumps({'stage':'setup','train':len(tr),'validation':len(va),'parameters':sum(p.numel() for p in model.parameters()),'codes':256}),flush=True)
for ep in range(1,36):
 random.shuffle(tr);model.train();ls=[]
 for k in range(0,len(tr),4):
  x,m,y=batch([seq(i) for i in tr[k:k+4]]);loss=model(input_ids=x,attention_mask=m,labels=y).loss;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ls.append(float(loss.detach()))
 score=ev(va);print(json.dumps({'stage':'epoch','epoch':ep,'train_loss':float(np.mean(ls)),'validation':score,'elapsed_s':round(time.time()-start)},ensure_ascii=False),flush=True)
 if score['loss']<best:
  best=score['loss'];model.save_pretrained(OUT,safe_serialization=True);tok.save_pretrained(OUT);(OUT/'training_info.json').write_text(json.dumps({'epoch':ep,'validation':score,'chars':CHARS},ensure_ascii=False))
complete_training_run(QWEN_STAGE1_DIR,OUT)
print(json.dumps({'stage':'finished','best_validation_loss':best,'path':str(OUT)},ensure_ascii=False),flush=True)


