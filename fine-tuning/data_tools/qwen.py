"""Validated, editable question/answer JSONL for handwriting training."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from project_paths import QWEN_HANDWRITING, QWEN_DRAWING, SMOOTH_VQ, DIGIT_DATA

CHARS = '0123456789NBACDEFGHIJKLMOPQRSTUVWXYZ'
STAGE1_SYSTEM = '你负责生成手写笔迹编号。输出恰好16个编号。'
DRAW_SYSTEM = '你是一个乐于助人的助手。普通问题正常回复。绘画时先简短回复，再输出<DRAW>、16个笔迹编号和<END_DRAW>。不要输出<CHAT>。'
STAGE1_TEMPLATES = ['请写一个{c}。','请生成{c}的手写笔迹。','请把{c}写出来。']
DRAW_TEMPLATES = ['请帮我写一个{kind}{c}。','画一个{kind}{c}给我看看。','请用手写的方式写出{c}。','帮我画{c}。','我想看看{c}的笔迹，请画出来。','写一个{c}。','请画一个{kind}{c}。','请写出字符{c}。','请手写{c}。','请画出{c}的笔迹。']


def sha256(path):
    path = Path(path)
    content = path.read_bytes()
    if path.suffix == '.jsonl':
        content = b'\n'.join(content.splitlines())
    return hashlib.sha256(content).hexdigest()


@dataclass
class HandwritingData:
    labels: list[int]
    codes: list[list[int]]
    train: list[int]
    validation: list[int]
    examples: list[list[dict]]
    system: str


def read_handwriting(path: Path) -> HandwritingData:
    groups = {}
    system = None
    identifiers = set()
    for number, line in enumerate(Path(path).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row,dict):
            raise ValueError(f'{path}:{number}: 必须是 JSON 对象')
        codes = row.get('codes')
        label = row.get('label')
        if type(label) is not int or not 0 <= label < len(CHARS) or row.get('character') != CHARS[label]:
            raise ValueError(f'{path}:{number}: 字符标签无效')
        if not isinstance(codes,list) or len(codes)!=16 or any(type(v) is not int or not 0<=v<256 for v in codes):
            raise ValueError(f'{path}:{number}: codes 必须是 16 个 0–255 的整数')
        if row.get('split') not in ('train','validation') or type(row.get('split_order')) is not int or row['split_order']<0:
            raise ValueError(f'{path}:{number}: 数据划分无效')
        for name in ('id','sample_id','system','user','assistant'):
            if not isinstance(row.get(name),str) or not row[name].strip():
                raise ValueError(f'{path}:{number}: 缺少 {name}')
        if row['id'] in identifiers:
            raise ValueError(f'{path}:{number}: 重复 id')
        identifiers.add(row['id'])
        if type(row.get('variant')) is not int or row['variant']<0:
            raise ValueError(f'{path}:{number}: variant 无效')
        expected = '<DRAW>'+''.join(f'<M{v:03d}>' for v in codes)+'<END_DRAW>'
        if not row['assistant'].endswith(expected) or row['assistant'].count('<DRAW>')!=1:
            raise ValueError(f'{path}:{number}: 回答中的笔迹编号与 codes 不一致')
        if system is None:
            system = row['system']
        elif row['system'] != system:
            raise ValueError(f'{path}:{number}: 同一数据集的系统提示必须一致')
        group = groups.setdefault(row['sample_id'],[])
        if group and any(row[key]!=group[0][key] for key in ('label','codes','split','split_order')):
            raise ValueError(f'{path}:{number}: 同一轨迹的标签、编号或划分不一致')
        if any(other['variant']==row['variant'] for other in group):
            raise ValueError(f'{path}:{number}: 重复模板编号')
        group.append(row)
    if not groups:
        raise ValueError(f'笔迹训练数据不能为空：{path}')
    examples = [sorted(group,key=lambda row:row['variant']) for group in groups.values()]
    if len({len(group) for group in examples})!=1 or any([r['variant'] for r in group]!=list(range(len(group))) for group in examples):
        raise ValueError('各轨迹必须包含相同数量、从 0 开始连续编号的模板')
    indices = {split:sorted([i for i,group in enumerate(examples) if group[0]['split']==split],key=lambda i:examples[i][0]['split_order']) for split in ('train','validation')}
    if any(not indices[split] or len({examples[i][0]['split_order'] for i in indices[split]})!=len(indices[split]) for split in indices):
        raise ValueError('训练和验证划分必须非空，且各自的 split_order 不能重复')
    return HandwritingData([group[0]['label'] for group in examples], [group[0]['codes'] for group in examples], indices['train'], indices['validation'], examples, system)


def ensure_handwriting_data() -> None:
    if not QWEN_HANDWRITING.is_file() or not QWEN_DRAWING.is_file():
        from data_tools.export_qwen import export
        export()
    manifest = QWEN_HANDWRITING.parent/'handwriting_manifest.json'
    if manifest.is_file():
        info = json.loads(manifest.read_text(encoding='utf-8'))
        for path,key in ((DIGIT_DATA,'source_sha256'),(SMOOTH_VQ,'codec_sha256')):
            if path.is_file() and info.get(key)!=sha256(path):
                raise ValueError(f'{path} 已更新，请运行 python -m data_tools.export_qwen --force 重新导出笔迹 JSONL')


def load_handwriting(kind='stage1') -> HandwritingData:
    if kind not in ('stage1','draw'):
        raise ValueError(f'未知笔迹数据集：{kind}')
    ensure_handwriting_data()
    return read_handwriting(QWEN_HANDWRITING if kind=='stage1' else QWEN_DRAWING)


def encode_example(tokenizer, row):
    prompt = tokenizer.apply_chat_template([{'role':'system','content':row['system']},{'role':'user','content':row['user']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)['input_ids']
    answer = tokenizer.encode(row['assistant'],add_special_tokens=False)
    draw = tokenizer.convert_tokens_to_ids('<DRAW>')
    start = len(prompt)+answer.index(draw)+1
    return prompt+answer+[tokenizer.eos_token_id],len(prompt),start
