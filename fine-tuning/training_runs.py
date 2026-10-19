"""Create separate training runs and publish only completed model checkpoints."""

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile

from project_paths import QWEN_STAGE1_DIR
from qwen_model_setup import missing_model_files


def create_training_run(stage_dir: Path, output_dir: Path | None = None) -> Path:
    stage_dir = Path(stage_dir).resolve()
    if output_dir is not None:
        output_dir = Path(output_dir).resolve()
        # Never write into an existing directory, even if it is empty.
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    prefix = datetime.now().strftime("run_%Y%m%d_%H%M%S_")
    return Path(tempfile.mkdtemp(prefix=prefix, dir=stage_dir))


def _check_checkpoint(model_dir: Path, require_done: bool) -> None:
    required = ["training_info.json"] + (["TRAIN_DONE"] if require_done else [])
    missing = missing_model_files(model_dir) + [
        name for name in required
        if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
    ]
    if missing:
        raise FileNotFoundError(f"训练产物不完整：{model_dir}，缺少：{', '.join(missing)}")


def complete_training_run(stage_dir: Path, model_dir: Path) -> None:
    stage_dir, model_dir = Path(stage_dir).resolve(), Path(model_dir).resolve()
    _check_checkpoint(model_dir, require_done=False)
    (model_dir / "TRAIN_DONE").write_text("training completed\n", encoding="utf-8")
    stage_dir.mkdir(parents=True, exist_ok=True)
    saved_path = model_dir.relative_to(stage_dir) if model_dir.is_relative_to(stage_dir) else model_dir
    record = {"model_dir": str(saved_path), "completed_at": datetime.now().isoformat()}
    # Publish atomically, so interrupted writes cannot corrupt the previous pointer.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=stage_dir, prefix=".latest-", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        json.dump(record, file, ensure_ascii=False, indent=2)
    try:
        os.replace(temporary, stage_dir / "latest.json")
    finally:
        temporary.unlink(missing_ok=True)


def latest_training_run(stage_dir: Path) -> Path:
    stage_dir = Path(stage_dir).resolve()
    pointer = stage_dir / "latest.json"
    if pointer.is_file():
        try:
            record = json.loads(pointer.read_text(encoding="utf-8"))
            path = record["model_dir"]
            if not isinstance(path, str) or not path:
                raise ValueError("empty model path")
            model_dir = Path(path)
            if not model_dir.is_absolute():
                model_dir = stage_dir / model_dir
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"无效的训练记录：{pointer}") from exc
        _check_checkpoint(model_dir, require_done=True)
        return model_dir.resolve()
    # Keep checkpoints produced by the previous flat directory layout usable.
    if (stage_dir / "training_info.json").is_file():
        _check_checkpoint(stage_dir, require_done=False)
        return stage_dir
    raise FileNotFoundError(f"找不到已完成的第一阶段模型：{stage_dir}，请先运行第一阶段训练。")


def resolve_model_path(model_dir: Path) -> Path:
    model_dir = Path(model_dir)
    if model_dir.resolve() == QWEN_STAGE1_DIR.resolve() or (model_dir / "latest.json").is_file():
        return latest_training_run(model_dir)
    return model_dir
