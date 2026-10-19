"""Run the complete five-stage handwriting training pipeline.

Usage:
    python run_training_pipeline.py
    python run_training_pipeline.py --from-stage 3
    python run_training_pipeline.py --dry-run

The individual training files keep their original hyperparameters.  This
launcher only handles ordering, preflight checks, logging, and resuming from
completed stages.
"""

from __future__ import annotations
from project_paths import ROOT, ALIGNED_VQ, SMOOTH_VQ, STRUCTURED_DECODER, SMALL_RUN_DIR, CAUSAL_GENERATOR

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


STATE_PATH = ROOT / "pipeline_state.json"

STAGES = {
    1: {
        "name": "AlignedVQ",
        "script": "train_aligned_vq.py",
        "outputs": [ALIGNED_VQ],
    },
    2: {
        "name": "AlignedVQ smooth",
        "script": "train_aligned_vq_smooth.py",
        "outputs": [SMOOTH_VQ],
    },
    3: {
        "name": "Causal Transformer",
        "script": "train_causal_vq.py",
        "outputs": [CAUSAL_GENERATOR],
    },
    4: {
        "name": "Structured stroke decoder",
        "script": "train_structured.py",
        "outputs": [STRUCTURED_DECODER],
    },
    5: {
        "name": "Transformer continuation",
        "script": "train_control.py",
        "outputs": [SMALL_RUN_DIR / "control_gen_2400.pt", SMALL_RUN_DIR / "condition_gen_2400.pt"],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-stage",
        type=int,
        choices=range(1, 6),
        default=1,
        help="start at this stage (1-5); earlier stages must already have outputs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rerun stages even when all expected output files already exist",
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


def check_environment() -> None:
    if not (ROOT / "data" / "digits_all.npz").is_file():
        raise SystemExit("找不到 data/digits_all.npz，请确认当前目录是 fine-tuning。")

    # All five original scripts use .cuda() directly.  Fail early with a
    # useful message instead of waiting for the first child process to crash.
    probe = subprocess.run(
        [sys.executable, "-c", "import torch; print(int(torch.cuda.is_available()))"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    if probe.stdout.strip() != "1":
        raise SystemExit(
            "这套训练脚本目前要求 CUDA。请在有 NVIDIA GPU 的环境中运行，"
            "或先把各训练脚本里的 .cuda()/device='cuda' 改成可配置设备。"
        )


def run_stage(number: int, state: dict, force: bool, dry_run: bool) -> None:
    stage = STAGES[number]
    script = ROOT / stage["script"]
    if not script.is_file():
        raise SystemExit(f"阶段 {number} 缺少脚本：{script.name}")

    if not dry_run:
        for previous in range(1, number):
            missing = [p.name for p in STAGES[previous]["outputs"] if not p.is_file()]
            if missing:
                raise SystemExit(
                    f"阶段 {number} 需要先完成阶段 {previous}，缺少产物：{', '.join(missing)}"
                )

    command = [sys.executable, "-u", script.name]
    print(f"\n=== 阶段 {number}/5：{stage['name']} ===", flush=True)
    print("$ " + " ".join(command), flush=True)

    if not force and outputs_exist(stage):
        print("已找到完整产物，跳过（如需重跑请加 --force）。", flush=True)
        state[str(number)] = {"status": "skipped", "outputs": [str(p) for p in stage["outputs"]]}
        save_state(state)
        return

    if dry_run:
        return

    started = time.time()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    state[str(number)] = {"status": "running", "script": script.name}
    save_state(state)
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode != 0:
        state[str(number)] = {"status": "failed", "returncode": result.returncode}
        save_state(state)
        raise SystemExit(f"阶段 {number} 失败，已停止；修复后可用 --from-stage {number} 继续。")

    missing = [p.name for p in stage["outputs"] if not p.is_file() or p.stat().st_size == 0]
    if missing:
        state[str(number)] = {"status": "failed", "missing_outputs": missing}
        save_state(state)
        raise SystemExit(f"阶段 {number} 虽然退出成功，但缺少产物：{', '.join(missing)}")

    state[str(number)] = {
        "status": "done",
        "seconds": round(time.time() - started),
        "outputs": [str(p) for p in stage["outputs"]],
    }
    save_state(state)
    print(f"阶段 {number} 完成，用时 {state[str(number)]['seconds']} 秒。", flush=True)


def main() -> None:
    args = parse_args()
    if not args.dry_run:
        check_environment()

    state = load_state()
    for number in range(args.from_stage, 6):
        run_stage(number, state, args.force, args.dry_run)
    if not args.dry_run:
        print("\n五个训练阶段全部完成。", flush=True)


if __name__ == "__main__":
    main()
