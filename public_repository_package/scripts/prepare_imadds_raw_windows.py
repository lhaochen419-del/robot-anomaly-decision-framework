from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.adapters.imadds_raw import build_imadds_raw_windows, load_imadds_attributes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/imadds_roboticarm_raw.yaml")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))["dataset"]
    raw_dir = ROOT / cfg["raw_dir"]
    attrs = load_imadds_attributes(raw_dir)
    result = build_imadds_raw_windows(
        raw_dir=raw_dir,
        output_npz=ROOT / cfg["processed_npz"],
        metadata_csv=ROOT / cfg["metadata_csv"],
        segment_length=int(cfg.get("segment_length", 512)),
        window_size=int(cfg.get("window_size", 128)),
        stride=int(cfg.get("stride", 128)),
        max_train_source_segments=cfg.get("max_train_source_segments", 600),
        max_train_target_segments=cfg.get("max_train_target_segments"),
        seed=int(cfg.get("seed", 7)),
    )
    result["attribute_rows_total"] = int(len(attrs))
    result["attribute_files"] = int(attrs["attribute_file"].nunique()) if len(attrs) else 0
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(result)


if __name__ == "__main__":
    main()
