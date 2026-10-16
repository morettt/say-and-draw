import json,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from aligned_digit_data import load,BASE
from aligned_vq_model import AlignedVQ

torch.manual_seed(801);torch.set_num_threads(4)
xy,ds,_=load();x=torch.tensor(xy,device='cuda');digits=torch.tensor(ds,device='cuda')
vq=AlignedVQ().cuda().eval();vq.load_state_dict(torch.load(BASE/'aligned_vq_smooth.pt',map_location='cuda',weights_only=False)['model'])
with torch.no_grad():raw=vq.encode(x)[1]
used=torch.unique(raw);compact=torch.searchsorted(used,raw);K=len(used)
BOS=0;DIGIT0=1;CODE0=37;VOCAB=CODE0+K
rng=np.random.default_rng(801);train=[];valid=[]
for d in range(10):
    ix=np.where(ds==d)[0];rng.shuffle(ix);n=max(2,round(len(ix)*.1));valid.extend(ix[:n]);train.extend(ix[n:])
train.extend(np.where(ds==10)[0]); b=np.where(ds==11)[0]; train.extend(b[:12]); valid.extend(b[12:]); a=np.where(ds==12)[0];train.extend(a[:12]); valid.extend(a[12:]); c=np.where(ds==13)[0]; train.extend(c[:12]); valid.extend(c[12:]); e=np.where(ds==14)[0]; train.extend(e[:12]); valid.extend(e[12:]); f=np.where(ds==15)[0]; train.extend(f[:12]); valid.extend(f[12:]); g=np.where(ds==16)[0]; train.extend(g[:12]); valid.extend(g[12:]); h0=np.where(ds==17)[0]; train.extend(h0[:12]); valid.extend(h0[12:]); h1=np.where(ds==18)[0]; train.extend(h1[:12]); valid.extend(h1[12:]); h2=np.where(ds==19)[0]; train.extend(h2[:12]); valid.extend(h2[12:]); h3=np.where(ds==20)[0]; train.extend(h3[:12]); valid.extend(h3[12:]); h4=np.where(ds==21)[0]; train.extend(h4[:12]); valid.extend(h4[12:]); z0=np.where(ds==22)[0]; train.extend(z0[:12]); valid.extend(z0[12:]); z1=np.where(ds==23)[0]; train.extend(z1[:12]); valid.extend(z1[12:]); z2=np.where(ds==24)[0]; train.extend(z2[:12]); valid.extend(z2[12:]); z3=np.where(ds==25)[0]; train.extend(z3[:12]); valid.extend(z3[12:]); z4=np.where(ds==26)[0]; train.extend(z4[:12]); valid.extend(z4[12:]); z5=np.where(ds==27)[0]; train.extend(z5[:12]); valid.extend(z5[12:]); z6=np.where(ds==28)[0]; train.extend(z6[:12]); valid.extend(z6[12:]); z7=np.where(ds==29)[0]; train.extend(z7[:12]); valid.extend(z7[12:]); z0=np.where(ds==30)[0]; train.extend(z0[:12]); valid.extend(z0[12:]); z1=np.where(ds==31)[0]; train.extend(z1[:12]); valid.extend(z1[12:]); z2=np.where(ds==32)[0]; train.extend(z2[:12]); valid.extend(z2[12:]); z3=np.where(ds==33)[0]; train.extend(z3[:12]); valid.extend(z3[12:]); z4=np.where(ds==34)[0]; train.extend(z4[:12]); valid.extend(z4[12:]); z5=np.where(ds==35)[0]; train.extend(z5[:12]); valid.extend(z5[12:])
train=torch.tensor(train,device='cuda');valid=torch.tensor(valid,device='cuda')
class CausalCodeLM(nn.Module):
    def __init__(self,vocab):
        super().__init__();self.tok=nn.Embedding(vocab,128);self.pos=nn.Embedding(18,128)
        layer=nn.TransformerEncoderLayer(128,4,512,dropout=.1,batch_first=True,activation='gelu',norm_first=True)
        self.blocks=nn.TransformerEncoder(layer,4,enable_nested_tensor=False)
        self.norm=nn.LayerNorm(128);self.head=nn.Linear(128,vocab,bias=False)
    def forward(self,ids):
        t=ids.shape[1];h=self.tok(ids)+self.pos(torch.arange(t,device=ids.device))[None]
        mask=torch.ones(t,t,device=ids.device,dtype=torch.bool).triu(1)
        return self.head(self.norm(self.blocks(h,mask=mask)))

def batch(ix):
    target=compact[ix]+CODE0;B=len(ix)
    inputs=torch.empty((B,17),dtype=torch.long,device='cuda')
    inputs[:,0]=BOS;inputs[:,1]=digits[ix]+DIGIT0;inputs[:,2:]=target[:,:-1]
    labels=target
    return inputs,labels

model=CausalCodeLM(VOCAB).cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01)
weights=torch.ones(len(train),device='cuda');weights[(digits[train]==4)|(digits[train]==5)]=2.5;weights[digits[train]==10]=2.5;weights[digits[train]==11]=2.5;weights[digits[train]==12]=2.5;weights[digits[train]==13]=2.5;weights[digits[train]==14]=2.5;weights[digits[train]==15]=2.5;weights[digits[train]==16]=2.5;weights[digits[train]==17]=2.5;weights[digits[train]==18]=2.5;weights[digits[train]==19]=2.5;weights[digits[train]==20]=2.5;weights[digits[train]==21]=2.5;weights[digits[train]==22]=2.5;weights[digits[train]==23]=2.5;weights[digits[train]==24]=2.5;weights[digits[train]==25]=2.5;weights[digits[train]==26]=2.5;weights[digits[train]==27]=2.5;weights[digits[train]==28]=2.5;weights[digits[train]==29]=2.5;weights[digits[train]==30]=2.5;weights[digits[train]==31]=2.5;weights[digits[train]==32]=2.5;weights[digits[train]==33]=2.5;weights[digits[train]==34]=2.5;weights[digits[train]==35]=2.5
@torch.no_grad()
def evaluate():
    model.eval();inp,tgt=batch(valid);out=model(inp)[:,1:]
    loss=float(F.cross_entropy(out.reshape(-1,VOCAB),tgt.reshape(-1)))
    acc=float((out.argmax(-1)==tgt).float().mean())
    return loss,acc
@torch.no_grad()
def generate(d):
    model.eval();seq=torch.tensor([[BOS,DIGIT0+d]],device='cuda');codes=[]
    for _ in range(16):
        logits=model(seq)[0,-1,CODE0:]
        code=int(logits.argmax());codes.append(code);seq=torch.cat([seq,torch.tensor([[CODE0+code]],device='cuda')],1)
    return codes
best=1e9;best_step=0;start=time.time()
for step in range(1,3001):
    model.train();ix=train[torch.multinomial(weights,64,replacement=True)]
    inp,tgt=batch(ix);out=model(inp)[:,1:]
    loss=F.cross_entropy(out.reshape(-1,VOCAB),tgt.reshape(-1))
    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    if step%100==0:
        val,acc=evaluate()
        if val<best:
            best=val;best_step=step
            torch.save({'model':model.state_dict(),'used':used.cpu(),'vocab':VOCAB,'step':step,'val':val},BASE/'causal_vq_generator.pt')
        rec={'step':step,'val':round(val,4),'accuracy':round(acc,4),'best_step':best_step,'seconds':round(time.time()-start)}
        if step%300==0:rec['samples']={str(d):generate(d) for d in (4,5,8,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35)}
        print(json.dumps(rec),flush=True)
print(json.dumps({'finished':True,'best_step':best_step,'best_val':best,'seconds':round(time.time()-start)}),flush=True)
