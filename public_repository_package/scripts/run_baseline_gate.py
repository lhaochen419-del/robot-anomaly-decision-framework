from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines import run_all_baselines
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_config.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["project"]["output_dir"]
    frames = []
    errors = []
    for protocol in ["main", "cross_condition"]:
        df, err = run_all_baselines(config, ROOT, output_dir, protocol=protocol)
        frames.append(df)
        errors.extend(err)
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output_dir / "baseline_comparison.csv", index=False)
    if errors:
        (output_dir / "baseline_errors.md").write_text("\n".join(f"- {e}" for e in errors), encoding="utf-8")
    print(f"Wrote baseline comparison to {output_dir / 'baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
