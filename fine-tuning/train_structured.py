from project_paths import SMOOTH_VQ, STRUCTURED_DECODER, prepare_handwriting_outputs
import sys,json,time,numpy as np,torch
from pathlib import Path
u=Path(__file__).resolve().parent
sys.path.insert(0,str(u))
from aligned_digit_data import load
from aligned_vq_model import AlignedVQ
prepare_handwriting_outputs()
torch.set_num_threads(4);torch.manual_seed(1901)
xy,labels,held=load()
a=np.load(u/'data'/'digits_all.npz')
keep=[i for i,d in enumerate(a['digit']) if not(d==1 and np.any(a['points'][i,1:,2]>0))]
pen=(a['points'][keep,:,2]>0)
for i,c in enumerate(labels):
 if c in (4,5):
  pen[i]=False;pen[i,32 if c==4 else 54]=True
pen[:,0]=True
x=torch.tensor(xy,device='cuda')
labs=torch.tensor(labels,device='cuda')
p=torch.tensor(pen,device='cuda')
tr=torch.tensor(np.where(~held)[0],device='cuda')
va=torch.tensor(np.where(held)[0],device='cuda')
vq=AlignedVQ().cuda().eval()
vq.load_state_dict(torch.load(SMOOTH_VQ,weights_only=False)['model'])
with torch.no_grad():q=vq.book(vq.encode(x)[1]).detach()
fullmask=(~p[:,1:]).float()[:,:,None]
oldmask=torch.ones_like(fullmask);oldmask[labs==4,31]=0;oldmask[labs==5,53]=0
endpoint=p.clone()
endpoint[:,:-1]|=p[:,1:];endpoint[:,-1]=True
def metrics(y,ix,mask):
 t=x[ix];vm=mask[ix];am=vm[:,1:]*vm[:,:-1]
 v=(y[:,1:]-y[:,:-1])-(t[:,1:]-t[:,:-1])
 ac=v[:,1:]-v[:,:-1]
 c=(y-t).abs().mean()
 vl=(v.abs()*vm).sum()/(2*vm.sum())
 al=(ac.abs()*am).sum()/(2*am.sum())
 return c+.8*vl+.15*al
def evaluate(m):
 with torch.no_grad():
  y=m(q[va])
  return dict(score=metrics(y,va,fullmask).item(),
   coord=(y-x[va]).abs().mean().item(),
   endpoints=(y-x[va]).abs()[endpoint[va]].mean().item())
weights=torch.ones(len(tr),device='cuda')
weights[(labs[tr]>=10)|(labs[tr]==4)|(labs[tr]==5)]=2.5
from structured_decoder import StructuredStrokeDecoder
def resample(z):
 dist=np.r_[0,np.linalg.norm(np.diff(z,axis=0),axis=1).cumsum()]
 if dist[-1]<1e-8:return np.repeat(z[:1],32,axis=0)
 return np.stack([np.interp(np.linspace(0,dist[-1],32),dist,z[:,j]) for j in range(2)],-1)
target=np.zeros((len(xy),8,32,2),np.float32);counts=[]
for i,(z,mark) in enumerate(zip(xy,pen)):
 cuts=np.r_[0,np.flatnonzero(mark[1:])+1,64];counts.append(len(cuts)-1)
 assert 1<=counts[-1]<=8
 for j,(start,end) in enumerate(zip(cuts[:-1],cuts[1:])):
  target[i,j]=resample(z[start:end])
t=torch.tensor(target,device='cuda');count=torch.tensor(counts,device='cuda')
active=(torch.arange(8,device='cuda')[None]<count[:,None])[:,:,None,None].float()
def geo(y,ix):
 mask=active[ix];den=mask.sum()
 coord=((y-t[ix]).abs()*mask).sum()/(den*32*2)
 v=(y[:,:,1:]-y[:,:,:-1])-(t[ix,:,1:]-t[ix,:,:-1])
 vel=(v.abs()*mask).sum()/(den*31*2)
 ac=v[:,:,1:]-v[:,:,:-1]
 accel=(ac.abs()*mask).sum()/(den*30*2)
 return coord+.8*vel+.15*accel,coord
torch.manual_seed(1901);m=StructuredStrokeDecoder().cuda()
opt=torch.optim.AdamW(m.parameters(),lr=2e-4,weight_decay=1e-4)
best=1e9;start=time.time()
for step in range(1,8001):
 m.train();ix=tr[torch.multinomial(weights,64,replacement=True)]
 y,logits=m(q[ix]);g,c=geo(y,ix)
 loss=g+.03*torch.nn.functional.cross_entropy(logits,count[ix]-1)
 opt.zero_grad(set_to_none=True);loss.backward()
 torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
 if step==4000:
  for group in opt.param_groups:group['lr']=5e-5
 if step%400==0:
  m.eval()
  with torch.no_grad():
   y,z=m(q[va]);g,c=geo(y,va)
   correct=int(((z.argmax(-1)+1)==count[va]).sum())
   ep=((y[:,:,0]-t[va,:,0]).abs()+(y[:,:,-1]-t[va,:,-1]).abs())
   ep=(ep*active[va,:,0,:]).sum()/(active[va].sum()*4)
  criterion=g.item()+.03*(1-correct/len(va))
  if criterion<best:
   best=criterion
   torch.save({'model':m.state_dict(),'step':step,'score':g.item(),'count_correct':correct},str(STRUCTURED_DECODER))
  print(json.dumps(dict(step=step,score=g.item(),coord=c.item(),endpoints=ep.item(),count=correct,total=len(va),seconds=round(time.time()-start))),flush=True)
print('DONE',flush=True)


