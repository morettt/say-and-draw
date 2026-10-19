"""Stage 6: general single/multi-turn dialogue with handwriting replay."""

import argparse
import json
import math
from pathlib import Path
import random
import time

from project_paths import ROOT, QWEN_CONFIRMATION_DIR, QWEN_GENERAL_DIR
from training_runs import create_training_run, complete_training_run
from data_tools.general_chat import DEFAULT_DATA, load_data, encode_turn


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=QWEN_CONFIRMATION_DIR / 'best')
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--gradient-accumulation', type=int, default=4)
    p.add_argument('--replay-ratio', type=float, default=.2, help='drawing examples per dialogue example')
    p.add_argument('--prepare-only', action='store_true', help='validate/tokenize data without loading weights')
    args = p.parse_args()
    if min(args.epochs, args.max_length, args.batch_size, args.gradient_accumulation) < 1 or args.lr <= 0 or not 0 <= args.replay_ratio <= 1:
        p.error('训练参数必须为正，replay-ratio 必须在 0 到 1 之间')
    return args


def main():
    args = parse_args()
    import torch
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from data_tools.qwen import load_handwriting, encode_example

    if not (args.model / 'config.json').is_file():
        raise FileNotFoundError(f'找不到第五阶段权重：{args.model}；也可用 --model 指定源模型')
    rng = random.Random(61019)
    torch.manual_seed(61019)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    handwriting = load_handwriting('draw')
    system = handwriting.system
    rows = load_data(args.data)
    chat = {'train': [], 'validation': []}
    skipped = []
    for row in rows:
        for turn in range(len(row['messages']) // 2):
            encoded = encode_turn(tokenizer, system, row['messages'], turn, args.max_length)
            if encoded is None:
                skipped.append({'id': row['id'], 'turn': turn})
            else:
                chat[row['split']].append(encoded)
    if not chat['train'] or not chat['validation']:
        raise ValueError('长度过滤后训练集或验证集为空；请调大 --max-length')
    replay = {}
    for split, indices in [('train', handwriting.train), ('validation', handwriting.validation)]:
        replay[split] = []
        for i in indices:
            sequence, prefix, _ = encode_example(tokenizer, handwriting.examples[i][0])
            if len(sequence) <= args.max_length:
                replay[split].append((sequence, prefix))
        if not replay[split]:
            raise ValueError(f'{split} 没有可用笔迹样本')
    meta = {'source': str(args.model.resolve()), 'data': str(args.data.resolve()),
            'epochs': args.epochs, 'lr': args.lr, 'max_length': args.max_length,
            'train_turns': len(chat['train']), 'validation_turns': len(chat['validation']),
            'skipped_long_turns': skipped, 'replay_ratio': args.replay_ratio,
            'batch_size': args.batch_size, 'gradient_accumulation': args.gradient_accumulation,
            'compute_dtype': 'bfloat16', 'master_dtype': 'float32',
            'selection': 'minimum validation dialogue loss + replay_ratio * handwriting loss'}
    print(json.dumps(meta, ensure_ascii=False), flush=True)
    if args.prepare_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError('第六阶段训练需要 CUDA')
    out = create_training_run(QWEN_GENERAL_DIR, args.output_dir)
    print(json.dumps({'stage': 'output', 'path': str(out)}, ensure_ascii=False), flush=True)
    (out / 'experiment.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'training_data.jsonl').write_text(Path(args.data).read_text(encoding='utf-8'), encoding='utf-8')
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, local_files_only=True, dtype=torch.bfloat16, attn_implementation='eager').cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    params = list(model.parameters())
    masters = [torch.nn.Parameter(p.detach().float().clone()) for p in params]
    optimizer = torch.optim.AdamW(masters, lr=args.lr, weight_decay=.01)

    def batch(examples):
        width = max(len(s) for s, _ in examples)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        ids = torch.full((len(examples), width), pad, dtype=torch.long, device='cuda')
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for i, (sequence, prefix) in enumerate(examples):
            ids[i, :len(sequence)] = torch.tensor(sequence, device='cuda')
            mask[i, :len(sequence)] = 1
            labels[i, prefix:len(sequence)] = ids[i, prefix:len(sequence)]
        return {'input_ids': ids, 'attention_mask': mask, 'labels': labels, 'use_cache': False}

    @torch.no_grad()
    def loss_on(examples):
        model.eval()
        total = count = 0
        for offset in range(0, len(examples), args.batch_size):
            data = batch(examples[offset:offset + args.batch_size])
            tokens = int((data['labels'][:, 1:] != -100).sum())
            loss = model(**data).loss
            total += float(loss) * tokens
            count += tokens
        return total / count

    best = math.inf
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        examples = list(chat['train'])
        examples += rng.choices(replay['train'], k=round(len(examples) * args.replay_ratio))
        rng.shuffle(examples)
        model.train()
        steps = math.ceil(len(examples) / args.batch_size)
        losses = []
        for group_start in range(0, steps, args.gradient_accumulation):
            model.zero_grad(set_to_none=True)
            optimizer.zero_grad(set_to_none=True)
            group_end = min(steps, group_start + args.gradient_accumulation)
            for step in range(group_start, group_end):
                loss = model(**batch(examples[step * args.batch_size:(step + 1) * args.batch_size])).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('训练 loss 非有限值，停止保存')
                (loss / (group_end - group_start)).backward()
                losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
            for p, master in zip(params, masters):
                master.grad = None if p.grad is None else p.grad.float()
            optimizer.step()
            with torch.no_grad():
                for p, master in zip(params, masters):
                    p.copy_(master)
            print(json.dumps({'epoch': epoch, 'step': group_end, 'steps': steps,
                              'train_loss': losses[-1], 'elapsed_s': round(time.time() - start)}), flush=True)
        validation = {'dialogue_loss': loss_on(chat['validation']), 'handwriting_loss': loss_on(replay['validation'])}
        score = validation['dialogue_loss'] + args.replay_ratio * validation['handwriting_loss']
        record = {'epoch': epoch, 'train_loss': sum(losses) / len(losses), 'validation': validation, 'selection_score': score}
        with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        if math.isfinite(score) and score < best:
            best = score
            model.save_pretrained(out, safe_serialization=True)
            tokenizer.save_pretrained(out)
            (out / 'training_info.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(record, ensure_ascii=False), flush=True)
    complete_training_run(QWEN_GENERAL_DIR, out)
    print(json.dumps({'stage': 'finished', 'model': str(out), 'best_score': best}), flush=True)


if __name__ == '__main__':
    main()
