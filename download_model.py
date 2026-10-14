"""Download the published SayAndDraw weights into this repository's models folder."""

from pathlib import Path

from modelscope import snapshot_download


MODEL_ID = "morelle/SayAndDraw-v0.1"
MODEL_DIR = Path(__file__).resolve().parent / "models"
REQUIRED_FILES = (
    "llm/config.json",
    "aligned_vq_smooth.pt",
    "residual_decoder.pt",
)


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {MODEL_ID} to {MODEL_DIR}", flush=True)
    snapshot_download(model_id=MODEL_ID, local_dir=str(MODEL_DIR))
    missing = [name for name in REQUIRED_FILES if not (MODEL_DIR / name).is_file()]
    if missing:
        raise RuntimeError(f"Download incomplete; missing: {', '.join(missing)}")
    print("Model ready. Run: python app.py", flush=True)


if __name__ == "__main__":
    main()
