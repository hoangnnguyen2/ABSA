import json
from pathlib import Path

from src.utils.config import OUTPUT_DIR


def save_metrics(metrics, filename):
    output_dir = Path(OUTPUT_DIR) / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation metrics -> {path}")
    return path
