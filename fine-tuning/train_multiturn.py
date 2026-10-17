"""Mixed text/drawing continuation of the best digit+A-Z checkpoint."""
import json,random,time,sys,gc
from pathlib import Path
import numpy as np,torch
from transformers import AutoTokenizer,Qwen3_5ForConditionalGeneration
ROOT=Path(__file__).resolve().parent;SRC=ROOT/'runs'/'qwen_chat_letters'/'best';OUT=ROOT/'runs'/'qwen_multiturn'
OUT.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(ROOT))
from aligned_digit_data import load
from aligned_vq_model import AlignedVQ
SYSTEM='You are a helpful assistant. Reply normally. For drawing, say a short sentence and then emit <DRAW>, 16 motion codes, <END_DRAW>. Do not emit <CHAT>.'
CHARS='0123456789NBACDEFGHIJKLMOPQRSTUVWXYZ'
torch.set_num_threads(4);torch.manual_seed(1610);rng=random.Random(1610)
xy,lab,_=load();split=random.Random(42);tr=[];va=[]
for c in range(36):
 ix=np.flatnonzero(lab==c).tolist();split.shuffle(ix);n=max(1,round(len(ix)*.15));va+=ix[:n];tr+=ix[n:]
v=AlignedVQ().cuda().eval();v.load_state_dict(torch.load(ROOT/'data'/'aligned_vq_smooth.pt',weights_only=False)['model'])
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
chat=[('你好','你好！有什么想聊的？'),('你好啊','你好！很高兴和你聊天。'),('您好','您好！我可以帮您做什么？'),('谢谢','不客气！'),('再见','再见，祝你今天愉快！'),('早上好','早上好！祝你有美好的一天。'),('你是谁','我是一个可以回答问题、聊天和生成手写数字与大写英文字母的助手。'),('你能做什么','我可以和你聊天、回答问题，也可以按要求写数字和大写英文字母。'),('我们聊聊天吧','好呀，你想聊什么？'),('不要画图，陪我聊聊','好的，我们只聊天。你今天过得怎么样？'),('我今天有点难过','听起来你今天不太开心。愿意说说发生了什么吗？'),('我有点累','辛苦了，先休息一会儿、喝点水吧。'),('讲一个笑话','为什么电脑总是很冷？因为它开了太多窗口。'),('介绍一下熊猫','大熊猫是中国的珍稀哺乳动物，主要以竹子为食。'),('天空为什么是蓝色的','阳光经过大气时，蓝光比红光更容易被散射，因此天空通常看起来是蓝色的。'),('水在几摄氏度结冰','在标准大气压下，纯水通常在零摄氏度结冰。'),('中国的首都是哪里','中国的首都是北京。'),('一年有几个月','一年有十二个月。'),('一周有几天','一周有七天。'),('帮我安排学习时间','可以先学习二十五分钟，再休息五分钟，按任务难度调整。'),('为什么需要睡觉','睡眠帮助身体恢复，也有助于记忆和注意力。'),('今天的天气怎么样','我没有实时天气信息。你可以告诉我所在城市，或查看当地天气预报。'),('你能画猫吗','目前我主要支持手写数字和大写英文字母，还不能可靠地画猫。'),('请画一只猫','目前我还不能可靠地画猫，可以帮你写一个数字或大写英文字母。'),('你能写中文吗','目前手写部分主要支持数字和大写英文字母，中文可以用文字回复。'),('Hello','Hello! How can I help you?'),('How are you?','I am ready to help. How are you?'),('Do not draw. Just say hello.','Hello!'),('What can you do?','I can chat, answer questions, and draw handwritten digits and uppercase letters.')]
for c in CHARS:
 kind='数字' if c.isdigit() else '大写英文字母'
 chat.extend([(f'{c}是什么？',f'{c}是一个{kind}。'),(f'不要画，告诉我{c}是什么。',f'{c}是一个{kind}。'),(f'你会写{c}吗？',f'会，我可以按要求写出{c}。'),(f'只用文字回答，你能画{c}吗？',f'能，我可以画{c}。'),(f'我提到了{c}，但不需要你画出来。','好的，我只回复文字。'),(f'Tell me what {c} is. Do not draw.',f'{c} is a '+('digit.' if c.isdigit() else 'capital letter.'))])
for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
 chat.extend([(f'字母{c}的小写是什么？',f'字母{c}的小写是{c.lower()}。'),(f'不要画，{c}是第几个字母？',f'{c}是英文字母表中的第{ord(c)-64}个字母。')])
for a in range(10):
 for b in [1,3,5]:chat.append((f'{a}+{b}等于多少？',f'{a}加{b}等于{a+b}。'))
chat=[(q,a) for q,a in chat if any(ord(z)>127 for z in q)]
chat_rows=[textrow(q,a) for q,a in chat]
# Short multi-turn examples teach the same interface used by the deployed chat page.
for c in CHARS:
 chat_rows.append(textrow('先别画，告诉我你准备写什么。',f'我准备写{c}。',[{'role':'user','content':f'我想写{c}。'},{'role':'assistant','content':f'你想让我帮你写{c}吗？'}]))
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
for c in range(36):
 char=CHARS[c];h=make_history([c])
 for q,a in [('好了，停下，不要再画了。','好的，我停止画图。'),('现在只聊天。你好！','你好！我们接下来只聊天。'),('不要再画，告诉我刚才写的是什么。',f'刚才写的是{char}。'),('我没让你继续写。','明白，我会等你提出新的要求。'),('不画了，告诉我2加3的答案。','2加3等于5。'),('不要画，字母A的小写是什么？','字母A的小写是a。'),('下一步先别动笔，解释一下英文字母。','英文字母是用来书写英语的符号，共有二十六个。')]:
  chat_rows.append(textrow(q,a,h))
# Additional varied negatives, with answers addressing the actual question.
for c in CHARS:
 for q,a in [(f'先不要画{c}，说说它是什么。',f'{c}是一个'+('数字。' if c.isdigit() else '大写英文字母。')),(f'我只问你会不会写{c}，不用写出来。',f'会，我可以写{c}。')]:chat_rows.append(textrow(q,a))
 for q in [f'暂时不画，请解释{c}。',f'这里只需要回答问题：{c}是什么？',f'我不想要笔迹，请介绍一下{c}。']:
  chat_rows.append(textrow(q,f'{c}是一个'+('数字。' if c.isdigit() else '大写英文字母。')))
for a in range(10):
 for b in [2,4,6]:
  chat_rows.append(textrow(f'只用文字回答，{a}加{b}是多少？',f'{a}加{b}等于{a+b}。',make_history([hr.randrange(36)])))
# Capability boundaries: this version supports one character per request.
for q in ['帮我写HELLO','一次写出123','画一只猫','帮我画一栋房子','写一个汉字好']:
 chat_rows.append(textrow(q,'目前我一次能手写一个数字或大写英文字母，请指定一个字符。'))
for c in 'abcdefghijklmnopqrstuvwxyz':chat_rows.append(textrow(f'请写一个小写{c}。',f'目前手写部分支持大写字母，可以帮你写{c.upper()}。'))
(OUT/'multiturn_recipe.json').write_text(json.dumps({'draw_variants_per_sample':len(draw_rows[tr[0]]),'chat_rows':len(chat_rows),'history_contains_actual_codes':True,'history_source':'training split only','seed':9016},ensure_ascii=False,indent=2))

(OUT/'training_data.json').write_text(json.dumps({'chat':chat,'draw_templates':templates,'train_indices':tr,'valid_indices':va,'system':SYSTEM},ensure_ascii=False,indent=2))
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
 return {'code_accuracy':right/tot,'chat_plain':chat_ok,'chat_total':len(probes_chat),'draw_mode':draw_ok,'draw_total':len(probes_draw),'samples':samples,'multi':multi,'multi_target_ok':sum(r['target_ok'] for r in multi),'stop_ok':sum(not r['draw'] and not r['stray_code'] and r['eos'] for r in stops),'stops':stops}
meta={'source':str(SRC),'epochs':6,'batch_draw':2,'batch_chat':2,'lr':2e-6,'seed':1610,'master_dtype':'fp32','compute_dtype':'bf16','chat_rows':len(chat_rows),'draw_train':len(tr),'draw_valid':len(va),'selection':'maximize chat/draw probe success, then validation code accuracy; probes are development checks, not independent final test'}
(OUT/'experiment.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True)
print(json.dumps({'epoch':0,'validation':evaluate()},ensure_ascii=False),flush=True)
best=(-1,-1);start=time.time()
for ep in range(1,7):
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
  val=evaluate();rec['validation']=val;score=(val['chat_plain']+val['draw_mode']+val['multi_target_ok']+val['stop_ok'],val['code_accuracy'])
  if score>best:
   best=score;m.save_pretrained(OUT/'best',safe_serialization=True);tok.save_pretrained(OUT/'best');(OUT/'best_meta.json').write_text(json.dumps(rec,ensure_ascii=False,indent=2))
 print(json.dumps(rec,ensure_ascii=False),flush=True)
 with (OUT/'metrics.jsonl').open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
(OUT/'TRAIN_DONE').write_text('6 epochs completed')
print('TRAIN_DONE',flush=True)

