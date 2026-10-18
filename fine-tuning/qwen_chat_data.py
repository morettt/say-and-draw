"""Load editable JSONL chat data used by the Qwen continuation stages."""

import json
from pathlib import Path


def load_rows(path: Path) -> tuple[list[tuple[str, str, list[dict]]], list[dict]]:
    raw = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for row in raw:
        if not isinstance(row.get("user"), str) or not isinstance(row.get("assistant"), str):
            raise ValueError(f"invalid chat row in {path}: user/assistant are required")
        history = row.get("history", [])
        if not isinstance(history, list):
            raise ValueError(f"invalid history in {path}: expected a list")
        rows.append((row["user"], row["assistant"], history))
    return rows, raw
