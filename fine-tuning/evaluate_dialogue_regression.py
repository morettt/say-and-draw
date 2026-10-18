"""Blind automated regression test for Qwen's ordinary dialogue ability.

The candidate model is compared with the unchanged base model on a fixed
holdout set.  An independent Alibaba-compatible Qwen endpoint judges the two
answers.  The project key is built in and can be overridden with
ALI_QWEN_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration


ROOT = Path(__file__).resolve().parent
DEFAULT_CASES = ROOT / "eval" / "dialogue_holdout.jsonl"
DEFAULT_API_URL = "https://llm-1kogbdw5rh8bt640.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
DEFAULT_JUDGE_MODEL = "qwen3.8-flash"
# Configured by the project owner for the built-in dialogue regression judge.
DEFAULT_API_KEY = "sk-ws-H.PYLEMME.Nu05.MEUCIQD_BVpitVBG56BAMErEc45mZeyR2jeesB9zPn8mT2IjAgIgWuwRuxi9tUG-NEAsogX1FtAi3jiK72qwYrifuOofkvo"
JUDGE_SYSTEM = """你是独立的对话质量评审员。比较两个模型对同一个用户问题的回答。
不要因为回答更长就给高分，也不要要求固定措辞。重点看：是否理解问题、事实是否合理、是否遵守用户限制、是否自然连贯。
普通聊天中如果错误输出 <DRAW>、<Mxxx> 或其他绘画控制标记，应该扣分。
只输出一个 JSON 对象，不要输出 Markdown。格式：
{"winner":"A|B|tie","a_score":1,"b_score":1,"reason":"简短理由"}
分数为 1 到 5。"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", type=Path, required=True, help="训练前的 Qwen 模型目录")
    p.add_argument("--candidate", type=Path, required=True, help="当前阶段的 Qwen 模型目录")
    p.add_argument("--label", required=True, help="阶段名称，例如 stage1")
    p.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--api-url", default=DEFAULT_API_URL)
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--max-new-tokens", type=int, default=160)
    return p.parse_args()


def load_cases(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def generate_answers(model_dir: Path, cases: list[dict], max_new_tokens: int) -> list[str]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_dir, dtype=dtype, attn_implementation="eager"
    ).to(device).eval()
    answers = []
    with torch.inference_mode():
        for case in cases:
            rendered = tokenizer.apply_chat_template(
                case["messages"], tokenize=True, add_generation_prompt=True, enable_thinking=False
            )
            input_ids = rendered["input_ids"] if isinstance(rendered, dict) else rendered
            x = torch.tensor([input_ids], device=device)
            output = model.generate(
                input_ids=x,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )[0, len(input_ids) :]
            answers.append(tokenizer.decode(output, skip_special_tokens=False).strip())
    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return answers


def parse_judgement(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"评审模型没有返回 JSON：{text[:300]}")
    result = json.loads(match.group(0))
    winner = str(result.get("winner", "tie")).lower()
    if winner not in {"a", "b", "tie"}:
        winner = "tie"
    return {
        "winner": winner,
        "a_score": max(1, min(5, int(result.get("a_score", 3)))),
        "b_score": max(1, min(5, int(result.get("b_score", 3)))),
        "reason": str(result.get("reason", "")),
    }


def judge(api_url: str, api_key: str, model: str, question: str, answer_a: str, answer_b: str) -> dict:
    prompt = f"""请比较下面两个回答。用户问题：\n<QUESTION>\n{question}\n</QUESTION>\n\n回答 A：\n<ANSWER_A>\n{answer_a}\n</ANSWER_A>\n\n回答 B：\n<ANSWER_B>\n{answer_b}\n</ANSWER_B>"""
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}],
            "temperature": 0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(
        api_url.rstrip("/") + "/chat/completions",
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"评审 API 返回 HTTP {exc.code}: {detail[:500]}") from exc
    text = body["choices"][0]["message"]["content"]
    if isinstance(text, list):
        text = "".join(str(part.get("text", "")) if isinstance(part, dict) else str(part) for part in text)
    return parse_judgement(text)


def main() -> None:
    args = parse_args()
    api_key = os.getenv("ALI_QWEN_API_KEY", DEFAULT_API_KEY)
    cases = load_cases(args.cases)
    baseline = generate_answers(args.baseline, cases, args.max_new_tokens)
    candidate = generate_answers(args.candidate, cases, args.max_new_tokens)
    rng = random.Random(20261018)
    rows = []
    wins = {"baseline": 0, "candidate": 0, "tie": 0}
    scores = {"baseline": [], "candidate": []}
    for case, base_answer, candidate_answer in zip(cases, baseline, candidate):
        swap = rng.choice([False, True])
        a, b = (candidate_answer, base_answer) if swap else (base_answer, candidate_answer)
        result = judge(args.api_url, api_key, args.judge_model, json.dumps(case["messages"], ensure_ascii=False), a, b)
        winner = result["winner"]
        mapped = "tie" if winner == "tie" else (("candidate" if winner == "a" else "baseline") if swap else ("baseline" if winner == "a" else "candidate"))
        wins[mapped] += 1
        scores["baseline"].append(result["b_score"] if swap else result["a_score"])
        scores["candidate"].append(result["a_score"] if swap else result["b_score"])
        rows.append({"id": case["id"], "messages": case["messages"], "baseline": base_answer, "candidate": candidate_answer, "judgement": {**result, "mapped_winner": mapped}})
        print(json.dumps({"id": case["id"], "winner": mapped}, ensure_ascii=False), flush=True)
        time.sleep(0.05)
    report = {
        "label": args.label,
        "baseline_model": str(args.baseline),
        "candidate_model": str(args.candidate),
        "judge_model": args.judge_model,
        "cases": len(cases),
        "summary": {
            "baseline_wins": wins["baseline"],
            "candidate_wins": wins["candidate"],
            "ties": wins["tie"],
            "baseline_mean_score": sum(scores["baseline"]) / len(cases),
            "candidate_mean_score": sum(scores["candidate"]) / len(cases),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"label": args.label, **report["summary"], "output": str(args.output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
