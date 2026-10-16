import sys,json,os,numpy as np,torch
from pathlib import Path
from PIL import Image,ImageDraw
from PIL import ImageFilter
u=Path(__file__).resolve().parent
sys.path.insert(0,str(u))
from aligned_digit_data import load
from aligned_vq_model import AlignedVQ
from causal_vq_model import CausalCodeLM
from structured_decoder import StructuredStrokeDecoder
torch.set_num_threads(4)
chars='0123456789NBACDEFGHIJKLMOPQRSTUVWXYZ'
xy,labels,held=load();a=np.load(u/'data'/'digits_all.npz')
keep=[i for i,d in enumerate(a['digit']) if not(d==1 and np.any(a['points'][i,1:,2]>0))]
pen=(a['points'][keep,:,2]>0)
for i,c in enumerate(labels):
 if c in (4,5):pen[i]=False;pen[i,32 if c==4 else 54]=True
pen[:,0]=True
v=AlignedVQ().cuda().eval();v.load_state_dict(torch.load(u/'data'/'aligned_vq_smooth.pt',weights_only=False)['model'])
tag=os.getenv('EVAL_TAG','');cp=torch.load(os.getenv('GENERATOR',str(u/'control_gen_1600.pt')),weights_only=False)
from conditioned_model import ConditionedCodeLM
gen=(ConditionedCodeLM if cp.get('condition_all_positions') else CausalCodeLM)(cp['vocab']).cuda().eval();gen.load_state_dict(cp['model']);used=cp['used'].cuda()
bank=u/('bank'+tag+'.pt')
with torch.inference_mode():
 trueq=v.book(v.encode(torch.tensor(xy,device='cuda'))[1])
 if not bank.exists():
  torch.manual_seed(4242);gl=torch.arange(36,device='cuda').repeat_interleave(12)
  seq=torch.stack([torch.zeros_like(gl),gl+1],1);ids=[]
  for j in range(16):
   val,pos=gen(seq)[:,-1,37:].topk(2,dim=-1)
   k=torch.multinomial((val/.7).softmax(-1),1)
   c=pos.gather(1,k);ids.append(c[:,0]);seq=torch.cat([seq,c+37],1)
  codes=used[torch.stack(ids,1)]
  torch.save(dict(codes=codes.cpu(),labels=gl.cpu()),bank)
 z=torch.load(bank,weights_only=False);gl=z['labels'].numpy();freeq=v.book(z['codes'].cuda())
def predict(name,q,l):
 if name=='structured':
  m=StructuredStrokeDecoder().cuda().eval();m.load_state_dict(torch.load(str(u/'structured_decoder.pt'),weights_only=False)['model'])
 elif name=='attention':
  m=AttentionStrokeDecoder().cuda().eval();m.load_state_dict(torch.load('/tmp/pen_fix/attention.pt',weights_only=False)['model'])
 else:
  m=(RangeDecoder() if name=='range' else ResidualDecoder()).cuda().eval()
  path=u/'residual_decoder.pt' if name=='baseline' else Path('/tmp/pen_fix/'+name+'.pt')
  m.load_state_dict(torch.load(path,weights_only=False)['decoder'])
 ys=[];ps=[]
 with torch.inference_mode():
  for st in range(0,len(q),128):
   z=q[st:st+128];ls=torch.tensor(l[st:st+128],device='cuda')
   if name=='structured':
    sy,logits=m(z);n=logits.argmax(-1)+1
    y=sy.reshape(-1,256,2)
    valid=torch.arange(256,device='cuda')[None]<n[:,None]*32
    y[~valid]=float('nan')
    p=(torch.arange(256,device='cuda')[None]%32==0)&valid
   elif name=='attention':y,logits=m(z);p=logits>0
   else:
    y=m(z);cuts=head(z,y,ls).argmax(-1)
    p=torch.zeros((len(z),64),device='cuda',dtype=torch.bool)
    for j in range(3):
     rows=torch.where(cuts[:,j]<64)[0];p[rows,cuts[rows,j]]=True
   p[:,0]=True;ys.append(y.cpu().numpy());ps.append(p.cpu().numpy())
 return np.concatenate(ys),np.concatenate(ps)
S=160
def raster(x,p):
 finite=np.isfinite(x).all(-1);x=x[finite];p=p[finite]
 im=Image.new('1',(S,S));dr=ImageDraw.Draw(im)
 cuts=np.r_[0,np.flatnonzero(p[1:])+1,len(x)]
 for a,b in zip(cuts[:-1],cuts[1:]):
  points=[((.5+(float(t[0])-.5)*.43)*(S-1),float(t[1])*(S-1)) for t in x[a:b]]
  if len(points)>1:dr.line(points,fill=1,width=2)
 return np.asarray(im,dtype=bool)
refs=np.stack([raster(x,p) for x,p in zip(xy,pen)])
r_dil=np.stack([np.asarray(Image.fromarray(m).filter(ImageFilter.MaxFilter(5)),dtype=bool) for m in refs])
r_sizes=refs.sum((1,2)).clip(1)
def compare(mask,indices):
 if not mask.any():return 0.,1.
 dil=np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(5)),dtype=bool)
 precision=r_dil[indices][:,mask].mean(1)
 recall=(refs[indices]*dil).sum((1,2))/r_sizes[indices]
 f1=2*precision*recall/(precision+recall+1e-9)
 best=float(f1.max())
 return best,1-best
va=np.flatnonzero(held)
all_results={}
for name in ['structured']:
 y,p=predict(name,trueq[va],labels[va])
 rec=[compare(raster(yy,pp),[i]) for yy,pp,i in zip(y,p,va)]
 f1=np.array([t[0] for t in rec]);ch=np.array([t[1] for t in rec])
 fy,fp=predict(name,freeq,gl)
 free=[compare(raster(yy,pp),np.flatnonzero(labels==i)) for yy,pp,i in zip(fy,fp,gl)]
 ff=np.array([t[0] for t in free]);fc=np.array([t[1] for t in free])
 r=dict(recon_f1=float(f1.mean()),recon_mismatch=float(ch.mean()),
  pen_exact=int((p[:,1:]==pen[va,1:]).all(1).sum()) if p.shape[1]==64 else None,
  pen_count=int((p[:,1:].sum(1)==pen[va,1:].sum(1)).sum()),total=len(va),
  free_f1=float(ff.mean()),free_mismatch=float(fc.mean()),
  free_digits=float(ff[gl<10].mean()),free_letters=float(ff[gl>=10].mean()))
 r['per_char']={c:round(float(ff[gl==i].mean()),4) for i,c in enumerate(chars)}
 print(json.dumps(dict(model=name,**r)),flush=True)
 np.savez_compressed(str(u/(name+tag+'_eval.npz')),recon=y,recon_pen=p,free=fy,free_pen=fp,f1=ff,labels=gl)
 all_results[name]=r
(u/('evaluation'+tag+'.json')).write_text(json.dumps(all_results,indent=2))



