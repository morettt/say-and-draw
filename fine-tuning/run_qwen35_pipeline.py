"""Run the complete five-stage Qwen3.5 handwriting/chat training pipeline.

Usage:
    python run_qwen35_pipeline.py
    python run_qwen35_pipeline.py --from-stage 3
    python run_qwen35_pipeline.py --dry-run

The training scripts own their hyperparameters.  This launcher only handles
ordering, dependency checks, logging, and safe continuation after completed
stages.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
STATE_PATH = ROOT / "qwen35_pipeline_state.json"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_DIR = ROOT / "qwen3.5-0.8b"

STAGES = {
    1: {
        "name": "Qwen3.5 handwriting stage 1",
        "script": "train_qwen35_handwriting_stage1_repro.py",
        "args": [],
        "outputs": [ROOT / "runs" / "qwen_stage1_oldmethod_restart" / "training_info.json"],
    },
    2: {
        "name": "FP32 master continuation",
        "script": "train_precision_ablation_oldmethod.py",
        "args": ["fp32_master"],
        "outputs": [ROOT / "runs" / "qwen_precision_oldmethod" / "fp32_master" / "TRAIN_DONE"],
    },
    3: {
        "name": "Chat and handwriting mixed training",
        "script": "train_chat_letters.py",
        "args": [],
        "outputs": [ROOT / "runs" / "qwen_chat_letters" / "TRAIN_DONE"],
    },
    4: {
        "name": "Multi-turn training",
        "script": "train_multiturn.py",
        "args": [],
        "outputs": [ROOT / "runs" / "qwen_multiturn" / "TRAIN_DONE"],
    },
    5: {
        "name": "Confirmation-dialog training",
        "script": "train_confirmation.py",
        "args": [],
        "outputs": [ROOT / "runs" / "qwen_confirmation" / "TRAIN_DONE"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=range(1, 6),
        default=1,
        help="start at this stage (1-5); earlier stages must already be complete",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without starting training",
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


def model_dir_has_files() -> bool:
    return MODEL_DIR.is_dir() and any(path.is_file() for path in MODEL_DIR.rglob("*"))


def modelscope_executable() -> str | None:
    found = shutil.which("modelscope")
    if found:
        return found
    for candidate in (Path(sys.executable).with_name("modelscope.exe"), Path(sys.executable).with_name("modelscope")):
        if candidate.is_file():
            return str(candidate)
    return None


def ensure_qwen_model(dry_run: bool) -> None:
    if model_dir_has_files():
        print(f"已找到 Qwen3.5 基础模型：{MODEL_DIR}", flush=True)
        return

    executable = modelscope_executable() or "modelscope"
    command = [
        executable,
        "download",
        "--model",
        MODEL_ID,
        "--local_dir",
        str(MODEL_DIR),
    ]
    print("未找到有效的 Qwen3.5 基础模型，将从 ModelScope 下载：", flush=True)
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return
    if modelscope_executable() is None:
        raise SystemExit(
            "找不到 modelscope 命令。请先安装依赖（pip install modelscope），然后重新运行。"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        raise SystemExit(f"ModelScope 下载失败，退出码：{result.returncode}")
    if not model_dir_has_files():
        raise SystemExit(f"ModelScope 下载完成，但模型目录仍为空：{MODEL_DIR}")
    print(f"Qwen3.5 基础模型下载完成：{MODEL_DIR}", flush=True)


def check_environment(dry_run: bool) -> None:
    ensure_qwen_model(dry_run)
    if dry_run:
        return

    required = [
        ROOT / "data" / "digits_all.npz",
        ROOT / "data" / "aligned_vq_smooth.pt",
    ]
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


def validate_previous_stages(number: int) -> None:
    for previous in range(1, number):
        missing = [p.name for p in STAGES[previous]["outputs"] if not p.is_file()]
        if missing:
            raise SystemExit(
                f"阶段 {number} 需要先完成阶段 {previous}，缺少产物：{', '.join(missing)}"
            )


def run_stage(number: int, state: dict, dry_run: bool) -> None:
    stage = STAGES[number]
    script = ROOT / stage["script"]
    if not script.is_file():
        raise SystemExit(f"阶段 {number} 缺少脚本：{script.name}")
    if not dry_run:
        validate_previous_stages(number)

    command = [sys.executable, "-u", script.name, *stage["args"]]
    print(f"\n=== 阶段 {number}/5：{stage['name']} ===", flush=True)
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
    blocked = [str(path.relative_to(ROOT)) for path in output_dirs if path.exists() and not outputs_exist(stage)]
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
    check_environment(args.dry_run)
    state = load_state()
    for number in range(args.from_stage, 6):
        run_stage(number, state, args.dry_run)
    if not args.dry_run:
        print("\nQwen3.5 五个训练阶段全部完成。", flush=True)


if __name__ == "__main__":
    main()
