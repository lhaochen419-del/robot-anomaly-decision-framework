from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.adapters import adapt_imadds_robotic_arm, adapt_kuka_torque, adapt_nist_ur, adapt_road

ADAPTERS = {
    "road": adapt_road,
    "imadds_robotic_arm": adapt_imadds_robotic_arm,
    "nist_ur": adapt_nist_ur,
    "kuka_torque": adapt_kuka_torque,
}
CONFIGS = {
    "road": "configs/datasets/road.yaml",
    "imadds_robotic_arm": "configs/datasets/imadds_robotic_arm.yaml",
    "nist_ur": "configs/datasets/nist_ur.yaml",
    "kuka_torque": "configs/datasets/kuka_torque.yaml",
}


def _load_dataset_config(name: str) -> dict:
    with (ROOT / CONFIGS[name]).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)["dataset"]


def _write_acquisition_report(results: list[dict]) -> None:
    out = ROOT / "data" / "acquisition_status.md"
    lines = ["# Real Data Acquisition Status", ""]
    for item in results:
        lines.extend(
            [
                f"## {item.get('dataset', item.get('name', item.get('adapter', 'dataset')))}",
                f"- Status: {item.get('status', 'UNKNOWN')}",
                f"- Official source: {item.get('official_source', 'see dataset config')}",
                f"- Output CSV: {item.get('output_csv', 'NA')}",
                f"- Reason: {item.get('reason', 'NA')}",
                "",
            ]
        )
    lines.append("See `data/download_instructions.md` for manual download and placement instructions.")
    out.write_text("\n".join(lines), encoding="utf-8")


def prepare_one(name: str) -> dict:
    cfg = _load_dataset_config(name)
    adapter = ADAPTERS[cfg["adapter"]]
    kwargs = {}
    if name == "road":
        kwargs["max_rows_per_run"] = None
    if name == "kuka_torque":
        kwargs["max_segments_per_class"] = 200
    result = adapter(ROOT / cfg["raw_dir"], ROOT / cfg["processed_csv"], **kwargs)
    result.setdefault("dataset", cfg["name"])
    result.setdefault("official_source", cfg.get("official_source", ""))
    result.setdefault("adapter", cfg["adapter"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all"] + sorted(CONFIGS))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    names = sorted(CONFIGS) if args.dataset == "all" else [args.dataset]
    results = [prepare_one(name) for name in names]
    _write_acquisition_report(results)
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for result in results:
            print(f"{result.get('dataset')}: {result.get('status')} -> {result.get('output_csv')}")


if __name__ == "__main__":
    main()
