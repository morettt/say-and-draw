import sys,json,time,numpy as np,torch
from pathlib import Path
import torch.nn.functional as F
u=Path(__file__).resolve().parent;sys.path.insert(0,str(u))
from aligned_digit_data import load
from aligned_vq_model import AlignedVQ
from causal_vq_model import CausalCodeLM
from conditioned_model import ConditionedCodeLM
torch.set_num_threads(4)
xy,labels,held=load();x=torch.tensor(xy,device='cuda');d=torch.tensor(labels,device='cuda')
v=AlignedVQ().cuda().eval();v.load_state_dict(torch.load(u/'data'/'aligned_vq_smooth.pt',weights_only=False)['model'])
cp=torch.load(u/'data'/'causal_vq_generator.pt',weights_only=False);used=cp['used'].cuda()
with torch.no_grad():raw=v.encode(x)[1];codes=torch.searchsorted(used,raw)
assert torch.equal(used[codes],raw)
tr=torch.tensor(np.flatnonzero(~held),device='cuda');va=torch.tensor(np.flatnonzero(held),device='cuda')
weights=torch.ones(len(tr),device='cuda');weights[(d[tr]>=10)|(d[tr]==4)|(d[tr]==5)]=2.5
def batch(ix):
 target=codes[ix]+37
 inp=torch.cat([torch.zeros((len(ix),1),device='cuda',dtype=torch.long),(d[ix]+1)[:,None],target[:,:-1]],1)
 return inp,target
for name,cls in [('control_gen',CausalCodeLM),('condition_gen',ConditionedCodeLM)]:
 torch.manual_seed(221);m=cls(cp['vocab']).cuda();m.load_state_dict(cp['model'])
 opt=torch.optim.AdamW(m.parameters(),lr=5e-5,weight_decay=.01);start=time.time()
 for step in range(2401):
  if step:
   m.train();ix=tr[torch.multinomial(weights,64,replacement=True)];inp,tgt=batch(ix)
   out=m(inp)[:,1:];loss=F.cross_entropy(out.reshape(-1,cp['vocab']),tgt.reshape(-1))
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
  if step==1200:
   for g in opt.param_groups:g['lr']=1e-5
  if step%400==0:
   m.eval()
   with torch.no_grad():
    inp,tgt=batch(va);out=m(inp)[:,1:]
    val=F.cross_entropy(out.reshape(-1,cp['vocab']),tgt.reshape(-1)).item()
    acc=(out.argmax(-1)==tgt).float().mean().item()
   print(json.dumps(dict(arm=name,step=step,val=val,acc=acc,seconds=round(time.time()-start))),flush=True)
  if step in (800,1600,2400):
   torch.save(dict(model=m.state_dict(),used=used.cpu(),vocab=cp['vocab'],step=step,condition_all_positions=(name=='condition_gen')),str(u/(name+'_'+str(step)+'.pt')))
print('DONE',flush=True)



