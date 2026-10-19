"""Check and download the local Qwen base model before training."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from project_paths import ROOT, QWEN_BASE_DIR, QWEN_MODEL_ID


def missing_model_files(model_dir: Path) -> list[str]:
    def exists(name: str) -> bool:
        path = model_dir / name
        return path.is_file() and path.stat().st_size > 0

    missing = [name for name in ("config.json", "tokenizer_config.json", "tokenizer.json") if not exists(name)]
    if exists("model.safetensors") or exists("pytorch_model.bin"):
        return missing
    for index in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        if not exists(index):
            continue
        try:
            weight_map = json.loads((model_dir / index).read_text(encoding="utf-8"))["weight_map"]
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError("empty weight map")
            shards = set(weight_map.values())
            if any(not isinstance(name, str) for name in shards):
                raise ValueError("invalid weight shard")
        except (OSError, ValueError, KeyError, TypeError):
            return missing + [f"有效的 {index}"]
        return missing + sorted(name for name in shards if not exists(name))
    return missing + ["模型权重（model.safetensors、pytorch_model.bin 或完整分片）"]


def modelscope_executable() -> str | None:
    # Prefer the executable installed with the active Python environment.
    for directory in (Path(sys.executable).parent / "Scripts", Path(sys.executable).parent):
        for name in ("modelscope.exe", "modelscope"):
            candidate = directory / name
            if candidate.is_file():
                return str(candidate)
    return shutil.which("modelscope")


def ensure_qwen_model(dry_run: bool = False, model_dir: Path = QWEN_BASE_DIR) -> Path:
    model_dir = Path(model_dir)
    missing = missing_model_files(model_dir)
    if not missing:
        print(f"已找到 Qwen3.5 基础模型：{model_dir}", flush=True)
        return model_dir

    executable = modelscope_executable()
    command = [executable or "modelscope", "download", "--model", QWEN_MODEL_ID, "--local_dir", str(model_dir)]
    print(f"Qwen3.5 基础模型缺少：{', '.join(missing)}", flush=True)
    print("将从 ModelScope 下载到：" + str(model_dir), flush=True)
    print("$ " + " ".join(command), flush=True)
    if dry_run:
        return model_dir
    if executable is None:
        raise SystemExit("找不到 modelscope 命令，请在当前 Python 环境执行 pip install modelscope 后重试。")

    model_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
    if result.returncode != 0:
        raise SystemExit(f"ModelScope 下载失败，退出码：{result.returncode}；请重新运行原训练命令以重试。")
    missing = missing_model_files(model_dir)
    if missing:
        raise SystemExit(f"ModelScope 下载后模型仍不完整，缺少：{', '.join(missing)}；目录：{model_dir}")
    print(f"Qwen3.5 基础模型已就绪：{model_dir}", flush=True)
    return model_dir
