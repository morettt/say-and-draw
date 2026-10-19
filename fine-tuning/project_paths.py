"""Shared paths for data, downloaded models, checkpoints, and run artifacts.

Paths are relative to this file, independent of the shell's working directory.
Importing this module never creates directories or downloads files.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs"

DIGIT_DATA = DATA_DIR / "digits_all.jsonl"
QWEN_HANDWRITING = DATA_DIR / "qwen" / "handwriting_stage1.jsonl"
QWEN_DRAWING = DATA_DIR / "qwen" / "handwriting_draw.jsonl"
QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
QWEN_BASE_DIR = MODELS_DIR / "qwen3.5-0.8b"
CODEC_DIR = MODELS_DIR / "codecs"
ALIGNED_VQ = CODEC_DIR / "aligned_vq.pt"
SMOOTH_VQ = CODEC_DIR / "aligned_vq_smooth.pt"
STRUCTURED_DECODER = CODEC_DIR / "structured_decoder.pt"

SMALL_RUN_DIR = RUNS_DIR / "small_transformer"
CAUSAL_GENERATOR = SMALL_RUN_DIR / "causal_vq_generator.pt"
CONTROL_GENERATOR = SMALL_RUN_DIR / "control_gen_1600.pt"
HANDWRITING_EVAL_DIR = SMALL_RUN_DIR / "eval"

QWEN_STAGE1_DIR = RUNS_DIR / "qwen_stage1_oldmethod_restart"
QWEN_PRECISION_DIR = RUNS_DIR / "qwen_precision_oldmethod"
QWEN_FP32_DIR = QWEN_PRECISION_DIR / "fp32_master"
QWEN_CHAT_DIR = RUNS_DIR / "qwen_chat_letters"
QWEN_MULTITURN_DIR = RUNS_DIR / "qwen_multiturn"
QWEN_CONFIRMATION_DIR = RUNS_DIR / "qwen_confirmation"
QWEN_FINAL_DIR = QWEN_CONFIRMATION_DIR / "best"


def prepare_handwriting_outputs() -> None:
    """Create output directories before running handwriting training."""
    CODEC_DIR.mkdir(parents=True, exist_ok=True)
    SMALL_RUN_DIR.mkdir(parents=True, exist_ok=True)
