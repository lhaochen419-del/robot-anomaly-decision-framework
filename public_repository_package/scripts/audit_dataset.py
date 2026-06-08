from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import audit_dataset, build_datasets_from_config, write_audit_reports
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_config.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["project"]["output_dir"]
    _, _, _, feature_cols, frame = build_datasets_from_config(config, ROOT, split_mode="main")
    audit = audit_dataset(frame, feature_cols, config)
    write_audit_reports(audit, output_dir)
    print(f"Wrote audit reports to {output_dir}")


if __name__ == "__main__":
    main()
