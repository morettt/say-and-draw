import json,time
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from aligned_digit_data import load,BASE
from aligned_vq_model import AlignedVQ

torch.manual_seed(925);torch.set_num_threads(4)
xy,digits,held=load();x=torch.tensor(xy,device='cuda');d=torch.tensor(digits,device='cuda')
tr=torch.tensor(np.where(~held)[0],device='cuda');va=torch.tensor(np.where(held)[0],device='cuda')
weights=torch.ones(len(tr),device='cuda');weights[(d[tr]==4)|(d[tr]==5)]=2.5;weights[d[tr]==10]=2.5;weights[d[tr]==11]=2.5;weights[d[tr]==12]=2.5;weights[d[tr]==13]=2.5;weights[d[tr]==14]=2.5;weights[d[tr]==15]=2.5;weights[d[tr]==16]=2.5;weights[d[tr]==17]=2.5;weights[d[tr]==18]=2.5;weights[d[tr]==19]=2.5;weights[d[tr]==20]=2.5;weights[d[tr]==21]=2.5;weights[d[tr]==22]=2.5;weights[d[tr]==23]=2.5;weights[d[tr]==24]=2.5;weights[d[tr]==25]=2.5;weights[d[tr]==26]=2.5;weights[d[tr]==27]=2.5;weights[d[tr]==28]=2.5;weights[d[tr]==29]=2.5;weights[d[tr]==30]=2.5;weights[d[tr]==31]=2.5;weights[d[tr]==32]=2.5;weights[d[tr]==33]=2.5;weights[d[tr]==34]=2.5;weights[d[tr]==35]=2.5
m=AlignedVQ().cuda();m.load_state_dict(torch.load(BASE/'aligned_vq.pt',map_location='cuda',weights_only=False)['model'])
opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-5)
def metrics(y,x,z,q,d):
 coord=(y-x).abs().mean()
 vy=y[:,1:]-y[:,:-1];vx=x[:,1:]-x[:,:-1]
 mask=torch.ones((len(x),63,1),device=x.device);mask[d==4,31]=0;mask[d==5,53]=0
 vel=((vy-vx).abs()*mask).sum()/mask.sum()/2
 ay=vy[:,1:]-vy[:,:-1];ax=vx[:,1:]-vx[:,:-1]
 cmask=mask[:,1:]*mask[:,:-1]
 accel=((ay-ax).abs()*cmask).sum()/cmask.sum()/2
 vq=F.mse_loss(q,z.detach())+.25*F.mse_loss(z,q.detach())
 return coord,vel,accel,vq
best=1e9;best_step=0;start=time.time()
for step in range(1,6001):
 m.train();ix=tr[torch.multinomial(weights,64,replacement=True)]
 y,z,q,_=m(x[ix]);c,v,a,qerr=metrics(y,x[ix],z,q,d[ix])
 loss=c+.8*v+.15*a+.2*qerr
 opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step%500==0:
  m.eval()
  with torch.no_grad():
   y,z,q,ids=m(x[va]);c,v,a,qerr=metrics(y,x[va],z,q,d[va]);score=float(c+.8*v+.15*a)
   used=int(m.encode(x)[1].unique().numel())
  if score<best:
   best=score;best_step=step
   torch.save({'model':m.state_dict(),'step':step,'score':score,'codes':256,'groups':16},BASE/'aligned_vq_smooth.pt')
  print(json.dumps({'step':step,'coord':round(float(c),5),'vel':round(float(v),5),'accel':round(float(a),5),'used':used,'best_step':best_step,'seconds':round(time.time()-start)}),flush=True)
print(json.dumps({'finished':True,'best_step':best_step,'best_score':best,'seconds':round(time.time()-start)}),flush=True)
