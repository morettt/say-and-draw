from project_paths import DATA_DIR, DIGIT_DATA
import json
from pathlib import Path
import numpy as np

BASE = DATA_DIR
CHARS = '0123456789NBACDEFGHIJKLMOPQRSTUVWXYZ'


def load_raw(path=DIGIT_DATA):
    path = Path(path)
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'轨迹数据不能为空：{path}')
    for i, row in enumerate(rows):
        if any(type(row.get(name)) is not int for name in ('source', 'number', 'position', 'digit')):
            raise ValueError(f'第 {i+1} 行缺少整数标签或来源字段')
        if not 0 <= row['digit'] < len(CHARS) or row.get('character', CHARS[row['digit']]) != CHARS[row['digit']]:
            raise ValueError(f'第 {i+1} 行字符标签无效')
    data = {name: np.asarray([row[name] for row in rows], dtype='int64') for name in ('source','number','position','digit')}
    data['points'] = np.asarray([row['points'] for row in rows], dtype='float32')
    if data['points'].shape != (len(rows),64,3) or not np.isfinite(data['points']).all():
        raise ValueError('每条轨迹必须包含 64 个有效的 [x,y,新笔画标记] 点')
    if not np.isin(data['points'][:,:,2], [0,1]).all():
        raise ValueError('新笔画标记必须为 0 或 1')
    return data

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
    a=load_raw()
    ids=np.array([i for i,d in enumerate(a['digit']) if not (d==1 and np.any(a['points'][i,1:,2]>0))])
    xy=np.stack([aligned(a['points'][i],int(a['digit'][i])) for i in ids]).astype('float32')
    labels=a['digit'][ids].astype('int64')
    held=np.isin(a['number'][ids],[6,16,26,36,46,56,66,76,86,96,-3,-5,-7,-9,-11,-13,-15,-17,-19,-21,-23,-25,-27,-29,-31,-33,-35,-37,-39,-41,-43,-45,-47,-49,-51])
    return xy,labels,held

