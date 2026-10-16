from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parent

def resample(p,n):
    xy=p[:,:2].astype(np.float64)
    dist=np.r_[0,np.linalg.norm(np.diff(xy,axis=0),axis=1).cumsum()]
    if dist[-1]<1e-6:return np.repeat(xy[:1],n,axis=0)
    return np.stack([np.interp(np.linspace(0,dist[-1],n),dist,xy[:,j]) for j in range(2)],-1)

def aligned(p,d):
    if d in (4,5):
        cut=int(np.where(p[1:,2]>0)[0][0]+1)
        n=32 if d==4 else 54
        return np.concatenate([resample(p[:cut],n),resample(p[cut:],64-n)])
    if len(p)==64 and np.any(p[1:,2]>0): return p[:,:2].astype(np.float64)
    return resample(p,64)

def load():
    a=np.load(BASE/'digits_all.npz')
    ids=np.array([i for i,d in enumerate(a['digit']) if not (d==1 and np.any(a['points'][i,1:,2]>0))])
    xy=np.stack([aligned(a['points'][i],int(a['digit'][i])) for i in ids]).astype('float32')
    labels=a['digit'][ids].astype('int64')
    held=np.isin(a['number'][ids],[6,16,26,36,46,56,66,76,86,96,-3,-5,-7,-9,-11,-13,-15,-17,-19,-21,-23,-25,-27,-29,-31,-33,-35,-37,-39,-41,-43,-45,-47,-49,-51])
    return xy,labels,held

