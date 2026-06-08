from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.provenance import validate_metric_file, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-dir", default="progress_for_chatgpt/latest")
    parser.add_argument("--dataset", default="RoAD")
    parser.add_argument("--report-path", default="outputs/packet_integrity_check.md")
    args = parser.parse_args()
    packet = ROOT / args.packet_dir
    checks = {}
    for name in ["08_gate_v2_metrics.csv", "09_gate_v2_baseline_comparison.csv"]:
        checks[name] = validate_metric_file(packet / name, expected_dataset=args.dataset, require_real=True)
    report_path = ROOT / args.report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_validation_report(report_path, checks)
    ok = all(v[0] for v in checks.values())
    print(f"packet_integrity_ok={ok}")
    if not ok:
        for name, (_, issues) in checks.items():
            for issue in issues:
                print(f"{name}: {issue}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
