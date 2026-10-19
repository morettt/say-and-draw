"""Run the complete six-stage Qwen3.5 handwriting/chat training pipeline.

Usage:
    python run_qwen35_pipeline.py
    python run_qwen35_pipeline.py --from-stage 3
    python run_qwen35_pipeline.py --dry-run

The training scripts own their hyperparameters.  This launcher only handles
ordering, dependency checks, logging, and safe continuation after completed
stages.
"""

from __future__ import annotations
from project_paths import ROOT, QWEN_MODEL_ID, QWEN_BASE_DIR, QWEN_HANDWRITING, QWEN_DRAWING, QWEN_STAGE1_DIR, QWEN_FP32_DIR, QWEN_CHAT_DIR, QWEN_MULTITURN_DIR, QWEN_CONFIRMATION_DIR

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from qwen_model_setup import ensure_qwen_model as prepare_qwen_model
from training_runs import latest_training_run
from project_paths import QWEN_GENERAL_DIR
from data_tools.general_chat import DEFAULT_DATA
from data_tools.qwen import ensure_handwriting_data


STATE_PATH = ROOT / "qwen35_pipeline_state.json"
MODEL_ID = QWEN_MODEL_ID
MODEL_DIR = QWEN_BASE_DIR
EVAL_SCRIPT = ROOT / "eval" / "evaluate_dialogue_regression.py"
EVAL_DIR = ROOT / "eval" / "dialogue_runs"

STAGES = {
    1: {
        "name": "Qwen3.5 handwriting stage 1",
        "script": "train_qwen35_handwriting_stage1_repro.py",
        "args": [],
        "model": QWEN_STAGE1_DIR,
        "outputs": [QWEN_STAGE1_DIR / "TRAIN_DONE"],
    },
    2: {
        "name": "FP32 master continuation",
        "script": "train_precision_ablation_oldmethod.py",
        "args": ["fp32_master"],
        "model": QWEN_FP32_DIR,
        "outputs": [QWEN_FP32_DIR / "TRAIN_DONE"],
    },
    3: {
        "name": "Chat and handwriting mixed training",
        "script": "train_chat_letters.py",
        "args": [],
        "model": QWEN_CHAT_DIR / "best",
        "outputs": [QWEN_CHAT_DIR / "TRAIN_DONE"],
    },
    4: {
        "name": "Multi-turn training",
        "script": "train_multiturn.py",
        "args": [],
        "model": QWEN_MULTITURN_DIR / "best",
        "outputs": [QWEN_MULTITURN_DIR / "TRAIN_DONE"],
    },
    5: {
        "name": "Confirmation-dialog training",
        "script": "train_confirmation.py",
        "args": [],
        "model": QWEN_CONFIRMATION_DIR / "best",
        "outputs": [QWEN_CONFIRMATION_DIR / "TRAIN_DONE"],
    },
    6: {
        "name": "General single/multi-turn dialogue training",
        "script": "train_general_chat.py",
        "args": [],
        "model": QWEN_GENERAL_DIR,
        "outputs": [QWEN_GENERAL_DIR / "TRAIN_DONE"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=range(1, 7),
        default=1,
        help="start at this stage (1-6); earlier stages must already be complete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without starting training",
    )
    parser.add_argument(
        "--skip-dialogue-eval",
        action="store_true",
        help="do not call the independent dialogue evaluator after stages",
    )
    return parser.parse_args()


def load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def outputs_exist(stage: dict) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in stage["outputs"])


def ensure_qwen_model(dry_run: bool) -> None:
    prepare_qwen_model(dry_run=dry_run, model_dir=MODEL_DIR)


def check_environment(dry_run: bool, require_dialogue_eval: bool) -> None:
    ensure_qwen_model(dry_run)
    if dry_run:
        return

    ensure_handwriting_data()
    required = [QWEN_HANDWRITING, QWEN_DRAWING, DEFAULT_DATA]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "Qwen3.5 训练缺少必要文件或目录：\n  - " + "\n  - ".join(missing)
        )

    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if probe.stdout.strip() != "1":
        raise SystemExit(
            "Qwen3.5 训练脚本目前要求 CUDA。请在有 NVIDIA GPU 的环境运行。"
        )


def run_dialogue_eval(label: str, model: Path, dry_run: bool) -> None:
    if not EVAL_SCRIPT.is_file():
        raise SystemExit(f"缺少对话评估脚本：{EVAL_SCRIPT.name}")
    output = EVAL_DIR / f"{label}.json"
    command = [
        sys.executable,
        "-u",
        str(EVAL_SCRIPT.relative_to(ROOT)),
        "--model",
        str(model),
        "--label",
        label,
        "--output",
        str(output),
    ]
    print("对话回归评估：$ " + " ".join(command), flush=True)
    if dry_run:
        return
    result = subprocess.run(command, cwd=ROOT, env=os.environ.copy())
    if result.returncode != 0:
        raise SystemExit(f"{label} 对话回归评估失败，已停止。")


def get_stage(number: int) -> dict:
    stage = dict(STAGES[number])
    if number == 6 and (QWEN_GENERAL_DIR / 'latest.json').is_file():
        model_dir = latest_training_run(QWEN_GENERAL_DIR)
        stage['model'] = model_dir
        stage['outputs'] = [model_dir / 'TRAIN_DONE']
    if number == 1:
        try:
            model_dir = latest_training_run(QWEN_STAGE1_DIR)
        except FileNotFoundError:
            if (QWEN_STAGE1_DIR / "latest.json").exists() or (QWEN_STAGE1_DIR / "training_info.json").exists():
                raise
            stage["model"] = QWEN_STAGE1_DIR
            stage["outputs"] = [QWEN_STAGE1_DIR / "TRAIN_DONE"]
        else:
            stage["model"] = model_dir
            # Legacy checkpoints have no TRAIN_DONE; retain existing compatibility.
            marker = "TRAIN_DONE" if (model_dir / "TRAIN_DONE").is_file() else "training_info.json"
            stage["outputs"] = [model_dir / marker]
    return stage


def validate_previous_stages(number: int) -> None:
    for previous in range(1, number):
        missing = [p.name for p in get_stage(previous)["outputs"] if not p.is_file()]
        if missing:
            raise SystemExit(
                f"阶段 {number} 需要先完成阶段 {previous}，缺少产物：{', '.join(missing)}"
            )


def run_stage(number: int, state: dict, dry_run: bool) -> None:
    stage = get_stage(number)
    script = ROOT / stage["script"]
    if not script.is_file():
        raise SystemExit(f"阶段 {number} 缺少脚本：{script.name}")
    if not dry_run:
        validate_previous_stages(number)

    command = [sys.executable, "-u", script.name, *stage["args"]]
    print(f"\n=== 阶段 {number}/6：{stage['name']} ===", flush=True)
    print("$ " + " ".join(command), flush=True)

    if outputs_exist(stage):
        print("已找到完整产物，跳过。", flush=True)
        state[str(number)] = {"status": "skipped", "outputs": [str(p) for p in stage["outputs"]]}
        if not dry_run:
            save_state(state)
        return
    if dry_run:
        return

    # Stages 2-5 intentionally use OUT.mkdir(exist_ok=False).  If a previous
    # run left an incomplete directory, stop with a clear message instead of
    # deleting training artifacts automatically.
    output_dirs = {path.parent for path in stage["outputs"]}
    blocked = [str(path.relative_to(ROOT)) for path in output_dirs if path.exists() and not outputs_exist(stage)] if 2 <= number <= 5 else []
    if blocked and number >= 2:
        raise SystemExit(
            "发现未完成的阶段目录，原训练脚本不会覆盖它们：\n  - "
            + "\n  - ".join(blocked)
            + "\n请确认后手动处理，再重新运行对应阶段。"
        )

    started = time.time()
    state[str(number)] = {"status": "running", "script": script.name}
    save_state(state)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        state[str(number)] = {"status": "failed", "returncode": result.returncode}
        save_state(state)
        raise SystemExit(f"阶段 {number} 失败，已停止；可用 --from-stage {number} 继续。")

    stage = get_stage(number)
    missing = [p.name for p in stage["outputs"] if not p.is_file() or p.stat().st_size == 0]
    if missing:
        state[str(number)] = {"status": "failed", "missing_outputs": missing}
        save_state(state)
        raise SystemExit(f"阶段 {number} 退出成功，但缺少产物：{', '.join(missing)}")

    state[str(number)] = {
        "status": "done",
        "seconds": round(time.time() - started),
        "outputs": [str(p) for p in stage["outputs"]],
    }
    save_state(state)
    print(f"阶段 {number} 完成，用时 {state[str(number)]['seconds']} 秒。", flush=True)


def main() -> None:
    args = parse_args()
    check_environment(args.dry_run, not args.skip_dialogue_eval)
    state = load_state()
    for number in range(args.from_stage, 7):
        run_stage(number, state, args.dry_run)
        if not args.skip_dialogue_eval:
            run_dialogue_eval(f"stage{number}", get_stage(number)["model"], args.dry_run)
    if not args.dry_run:
        print("\nQwen3.5 六个训练阶段全部完成。", flush=True)


if __name__ == "__main__":
    main()
