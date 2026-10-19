"""Export Chinese handwriting question/answer JSONL from trajectory JSONL."""

import argparse
import json
import os
from pathlib import Path
import random
import tempfile

from project_paths import DIGIT_DATA, SMOOTH_VQ, QWEN_HANDWRITING, QWEN_DRAWING
from data_tools.qwen import CHARS, STAGE1_SYSTEM, DRAW_SYSTEM, STAGE1_TEMPLATES, DRAW_TEMPLATES, sha256, read_handwriting


def export(force=False):
    import numpy as np
    import torch
    from aligned_digit_data import load, load_raw
    from aligned_vq_model import AlignedVQ

    outputs = (QWEN_HANDWRITING,QWEN_DRAWING)
    if not force and any(path.exists() for path in outputs):
        raise FileExistsError('笔迹 JSONL 已存在；如需重新导出，请加 --force')
    if not SMOOTH_VQ.is_file():
        raise FileNotFoundError(f'导出笔迹编号需要 VQ 权重：{SMOOTH_VQ}')
    xy,labels,_ = load()
    original = load_raw()
    kept = [i for i,d in enumerate(original['digit']) if not(d==1 and np.any(original['points'][i,1:,2]>0))]
    torch.set_num_threads(4)
    codec = AlignedVQ().eval()
    codec.load_state_dict(torch.load(SMOOTH_VQ,map_location='cpu',weights_only=False)['model'])
    with torch.inference_mode():
        codes = codec.encode(torch.tensor(xy))[1].tolist()
    rng = random.Random(42)
    train,validation = [],[]
    for label in range(len(CHARS)):
        ids = np.flatnonzero(labels==label).tolist()
        rng.shuffle(ids)
        count = max(1,round(len(ids)*.15))
        validation.extend(ids[:count]);train.extend(ids[count:])
    splits = {i:(split,rank) for split,indices in (('train',train),('validation',validation)) for rank,i in enumerate(indices)}
    temporary = []
    try:
        for path,system,templates in ((QWEN_HANDWRITING,STAGE1_SYSTEM,STAGE1_TEMPLATES),(QWEN_DRAWING,DRAW_SYSTEM,DRAW_TEMPLATES)):
            path.parent.mkdir(parents=True,exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,suffix='.jsonl',delete=False) as stream:
                temp = Path(stream.name);temporary.append(temp)
                for i,label in enumerate(labels):
                    c = CHARS[int(label)];split,rank = splits[i]
                    code_text = '<DRAW>'+''.join(f'<M{v:03d}>' for v in codes[i])+'<END_DRAW>'
                    for variant,template in enumerate(templates):
                        row = {'id':f'stroke_{kept[i]:06d}_{variant}','sample_id':f'stroke_{kept[i]:06d}',
                               'character':c,'label':int(label),'split':split,'split_order':rank,'variant':variant,
                               'codes':codes[i],'system':system,'user':template.format(c=c,kind='数字' if c.isdigit() else '字母'),
                               'assistant':('' if path==QWEN_HANDWRITING else f'好的，我给你写一个{c}。\n')+code_text}
                        stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            read_handwriting(temp)
        for temp,path in zip(temporary,outputs):
            os.replace(temp,path)
        manifest = {'source_sha256':sha256(DIGIT_DATA),'codec_sha256':sha256(SMOOTH_VQ),'samples':len(labels),'train':len(train),'validation':len(validation)}
        (QWEN_HANDWRITING.parent/'handwriting_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)
    print(json.dumps({'stage':'export_handwriting',**manifest},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--force',action='store_true',help='重新导出并覆盖已有的笔迹 JSONL')
    export(parser.parse_args().force)
