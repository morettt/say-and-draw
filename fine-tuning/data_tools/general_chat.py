"""Import 问：/答： text; blank lines before questions separate conversations."""

import argparse
import hashlib
import json
from pathlib import Path
import random
import re

from project_paths import ROOT, DATA_DIR

DEFAULT_SOURCE = ROOT / '通用数据集.txt'
DEFAULT_DATA = DATA_DIR / 'qwen' / 'chat_stage6.jsonl'


def parse_conversations(text):
    conversations, messages = [], []
    blank = False
    for number, line in enumerate(text.lstrip('\ufeff').splitlines(), 1):
        if not line.strip():
            blank = True
            continue
        match = re.match(r'^\s*(问|答)[：:]\s*(.*)$', line)
        if match:
            role = 'user' if match[1] == '问' else 'assistant'
            if role == 'user' and blank and messages:
                if messages[-1]['role'] != 'assistant':
                    raise ValueError(f'第 {number} 行之前的问题没有回答')
                conversations.append(messages)
                messages = []
            expected = 'user' if len(messages) % 2 == 0 else 'assistant'
            if role != expected:
                raise ValueError(f'第 {number} 行：预期 {expected}，实际 {role}')
            messages.append({'role': role, 'content': match[2]})
        elif messages:
            messages[-1]['content'] += ('\n\n' if blank else '\n') + line
        else:
            raise ValueError(f'第 {number} 行：找不到“问：”')
        blank = False
    if messages:
        if messages[-1]['role'] != 'assistant':
            raise ValueError('文件末尾的问题没有回答')
        conversations.append(messages)
    if len(conversations) < 2:
        raise ValueError('至少需要两个由空行分隔的完整对话')
    if any(not m['content'].strip() for c in conversations for m in c):
        raise ValueError('问题和回答不能为空')
    return conversations


def export(source=DEFAULT_SOURCE, output=DEFAULT_DATA):
    text = Path(source).read_text(encoding='utf-8-sig')
    conversations = parse_conversations(text)
    # Identical conversations always stay in the same split.
    unique = {}
    for messages in conversations:
        digest = hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()
        unique.setdefault(digest, messages)
    keys = sorted(unique)
    if len(keys) < 2:
        raise ValueError('去重后至少需要两个对话')
    random.Random(61019).shuffle(keys)
    validation = set(keys[:max(1, round(len(keys) * .1))])
    rows = [{'id': key, 'split': 'validation' if key in validation else 'train',
             'messages': unique[key]} for key in sorted(unique)]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    return rows


def load_data(path=DEFAULT_DATA):
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        messages = row.get('messages', [])
        if row.get('split') not in ('train', 'validation') or not messages or len(messages) % 2:
            raise ValueError(f'无效对话：{row.get("id")}')
        for i, message in enumerate(messages):
            if message.get('role') != ('user' if i % 2 == 0 else 'assistant') or not isinstance(message.get('content'), str) or not message['content'].strip():
                raise ValueError(f'无效消息：{row.get("id")}')
    if not all(any(r['split'] == split for r in rows) for split in ('train', 'validation')):
        raise ValueError('训练集和验证集都不能为空')
    return rows


def encode_turn(tokenizer, system, messages, turn, max_length):
    """Supervise one answer with complete history; never truncate target answers."""
    history = messages[:turn * 2]
    user, answer = messages[turn * 2:turn * 2 + 2]
    target = tokenizer.encode(answer['content'], add_special_tokens=False) + [tokenizer.eos_token_id]
    while True:
        prompt = tokenizer.apply_chat_template(
            [{'role': 'system', 'content': system}] + history + [user],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        if hasattr(prompt, 'keys'):
            prompt = prompt['input_ids']
        prompt = list(prompt)
        if len(prompt) + len(target) <= max_length:
            return prompt + target, len(prompt)
        if not history:
            return None
        history = history[2:]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()
    rows = export(args.source, args.output)
    print(json.dumps({'conversations': len(rows), 'turns': sum(len(r['messages']) // 2 for r in rows),
                      'multiturn': sum(len(r['messages']) > 2 for r in rows), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
