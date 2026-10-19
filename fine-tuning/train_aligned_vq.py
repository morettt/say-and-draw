from project_paths import ALIGNED_VQ, prepare_handwriting_outputs
import json,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from aligned_digit_data import load

prepare_handwriting_outputs()
torch.manual_seed(924);torch.set_num_threads(4)
class AlignedVQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Conv1d(2,64,5,padding=2),nn.GELU(),nn.Conv1d(64,128,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(128,64,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(64,32,3,padding=1))
        self.book=nn.Embedding(256,32);nn.init.uniform_(self.book.weight,-.1,.1)
        self.decoder=nn.Sequential(nn.Conv1d(32,64,3,padding=1),nn.GELU(),nn.ConvTranspose1d(64,128,4,stride=2,padding=1),nn.GELU(),nn.ConvTranspose1d(128,64,4,stride=2,padding=1),nn.GELU(),nn.Conv1d(64,2,5,padding=2),nn.Sigmoid())
    def encode(self,x):
        z=self.encoder(x.transpose(1,2)).transpose(1,2)
        dist=(z.square().sum(-1,keepdim=True)-2*z@self.book.weight.T+self.book.weight.square().sum(-1))
        ids=dist.argmin(-1)
        return z,ids
    def decode(self,ids):
        return self.decoder(self.book(ids).transpose(1,2)).transpose(1,2)
    def forward(self,x):
        z,ids=self.encode(x);q=self.book(ids)
        y=self.decoder((z+(q-z).detach()).transpose(1,2)).transpose(1,2)
        return y,z,q,ids

xy,digits,held=load();x=torch.tensor(xy,device='cuda');d=torch.tensor(digits,device='cuda')
tr=torch.tensor((~held).nonzero()[0],device='cuda');va=torch.tensor(held.nonzero()[0],device='cuda')
weights=torch.ones(len(tr),device='cuda');weights[(d[tr]==4)|(d[tr]==5)]=2.5;weights[d[tr]==10]=2.5;weights[d[tr]==11]=2.5;weights[d[tr]==12]=2.5;weights[d[tr]==13]=2.5;weights[d[tr]==14]=2.5;weights[d[tr]==15]=2.5;weights[d[tr]==16]=2.5;weights[d[tr]==17]=2.5;weights[d[tr]==18]=2.5;weights[d[tr]==19]=2.5;weights[d[tr]==20]=2.5;weights[d[tr]==21]=2.5;weights[d[tr]==22]=2.5;weights[d[tr]==23]=2.5;weights[d[tr]==24]=2.5;weights[d[tr]==25]=2.5;weights[d[tr]==26]=2.5;weights[d[tr]==27]=2.5;weights[d[tr]==28]=2.5;weights[d[tr]==29]=2.5;weights[d[tr]==30]=2.5;weights[d[tr]==31]=2.5;weights[d[tr]==32]=2.5;weights[d[tr]==33]=2.5;weights[d[tr]==34]=2.5;weights[d[tr]==35]=2.5
model=AlignedVQ().cuda();opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-5)
def calc(y,x,z,q,d):
    coord=(y-x).abs().mean()
    vel=(y[:,1:]-y[:,:-1])-(x[:,1:]-x[:,:-1])
    mask=torch.ones((len(x),63,1),device=x.device)
    mask[d==4,31]=0;mask[d==5,53]=0
    speed=(vel.abs()*mask).sum()/mask.sum()/2
    vq=F.mse_loss(q,z.detach())+.25*F.mse_loss(z,q.detach())
    return coord+.3*speed+vq,coord,speed
best=1e9;best_step=0;start=time.time()
for step in range(1,12001):
    model.train();ix=tr[torch.multinomial(weights,64,replacement=True)]
    y,z,q,_=model(x[ix]);loss,_,_=calc(y,x[ix],z,q,d[ix])
    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
    if step%500==0:
        model.eval()
        with torch.no_grad():
            y,z,q,ids=model(x[va]);val,coord,speed=calc(y,x[va],z,q,d[va])
            used=int(model.encode(x)[1].unique().numel())
            value=float(val)
        if value<best:
            best=value;best_step=step
            torch.save({'model':model.state_dict(),'step':step,'val':value,'codes':256,'groups':16},ALIGNED_VQ)
        print(json.dumps({'step':step,'val':round(value,5),'coord':round(float(coord),5),'speed':round(float(speed),5),'used':used,'best_step':best_step,'seconds':round(time.time()-start)}),flush=True)
print(json.dumps({'finished':True,'best_step':best_step,'best_val':best,'seconds':round(time.time()-start)}),flush=True)
