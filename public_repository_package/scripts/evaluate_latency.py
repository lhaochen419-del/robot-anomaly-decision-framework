from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import build_datasets_from_config
from src.evaluation import measure_cirfl_latency
from src.utils.config import load_config


def write_latency_report(row: dict, output_dir: Path) -> None:
    lines = [
        "# Complexity and Latency",
        "",
        f"- Parameters: {row['parameters']}",
        f"- Model size MB: {row['model_size_mb']:.4f}",
        f"- Device: {row['device']}",
        f"- Latency ms/window: {row['latency_ms_per_window']:.4f}",
        f"- CUDA available: {row['cuda_available']}",
        f"- GPU: {row['gpu_name']}",
        f"- Platform: {row['platform']}",
        f"- Torch: {row['torch']}",
    ]
    (output_dir / "complexity_latency.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_config.yaml")
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["project"]["output_dir"]
    train_ds, _, _, feature_cols, _ = build_datasets_from_config(config, ROOT, split_mode="main")
    ckpt = args.checkpoint or output_dir / f"CIRFL_seed{config['seeds'][0]}_main.pt"
    row = measure_cirfl_latency(config, len(feature_cols), checkpoint_path=ckpt if Path(ckpt).exists() else None)
    pd.DataFrame([row]).to_csv(output_dir / "latency.csv", index=False)
    write_latency_report(row, output_dir)
    print(f"Wrote latency report to {output_dir}")


if __name__ == "__main__":
    main()
