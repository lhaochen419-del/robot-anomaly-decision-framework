from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.utils.provenance import validate_metric_file


def _copy_or_stub(src: Path, dst: Path, stub: str) -> None:
    if src.exists():
        shutil.copyfile(src, dst)
    else:
        dst.write_text(stub, encoding="utf-8")


def _read(src: Path, fallback: str) -> str:
    return src.read_text(encoding="utf-8") if src.exists() else fallback


def make_review_packet(config: dict, root: Path = ROOT) -> Path:
    output_dir = root / config["project"].get("output_dir", "outputs/real_gate_v2")
    review_dir = root / config.get("review_packet", {}).get("output_dir", "progress_for_chatgpt/latest")
    review_dir.mkdir(parents=True, exist_ok=True)
    for item in review_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    status = config.get("project", {}).get("computed_status", "NO-GO")
    metrics_ok, metrics_issues = validate_metric_file(output_dir / "gate_v2_metrics.csv", "RoAD", require_real=True)
    baselines_ok, baselines_issues = validate_metric_file(output_dir / "gate_v2_baseline_comparison.csv", "RoAD", require_real=True)
    if not metrics_ok or not baselines_ok:
        raise RuntimeError(f"Refusing to build packet from invalid/stale metric files: metrics={metrics_issues}, baselines={baselines_issues}")

    # 00
    (review_dir / "00_readme_for_chatgpt.md").write_text(f"""# RoAD Protocol Repair + CIRFL v2 Redesign + Real Gate v2

- Current stage: Real Gate v2 algorithm-improvement only; no full experiments, no manuscript writing, and no figures.
- Current status: {status}.
- Synthetic results included in this packet: NO.
- Valid protocols are listed in `01_protocol_validity_report.md`.
- Invalid/obsolete synthetic debug outputs are not copied into this packet.
""", encoding="utf-8")
    _copy_or_stub(output_dir / "protocol_validity_report_v2.md", review_dir / "01_protocol_validity_report.md", "# Protocol Validity\n\nNo protocol report found.\n")
    _copy_or_stub(root / "outputs/road_feasibility/label_feasibility_report.md", review_dir / "02_road_label_feasibility.md", "# RoAD Label Feasibility\n\nNo feasibility report found.\n")
    _copy_or_stub(output_dir / "data_audit.md", review_dir / "03_real_data_audit_v2.md", "# Real Data Audit v2\n\nNo audit report found.\n")
    _copy_or_stub(output_dir / "leakage_split_protocol.md", review_dir / "04_leakage_split_protocol_v2.md", "# Leakage Split Protocol v2\n\nNo leakage report found.\n")
    (review_dir / "05_cirfl_v2_algorithm_spec.md").write_text("""# CIRFL_v2 Algorithm Specification

CIRFL_v2 keeps the unified residual-field hypothesis. The changes are internal to the residual-field mechanism:
- train-only median/IQR normalization with train-derived channel clipping;
- stable heteroscedastic scale `scale = min_scale + softplus(clamped_raw_scale)`;
- calibrated energy clipping before aggregation to prevent score explosion;
- scheduled gradient reversal plus z_h/z_c covariance and orthogonality penalties;
- normalized signed relation-atom dictionary with atom repulsion, usage balance, and assignment entropy regularization;
- residual-field anomaly-axis scoring inside z_h, combined with prototype margin and calibrated residual energy;
- validation-only score orientation and target-FAR threshold calibration;
- binary anomaly detection as the main RoAD task, with multiclass prototype diagnosis retained only as auxiliary analysis.

The anomaly score is a residual-field signature score composed of calibrated mismatch energy, prototype margin, and the learned z_h anomaly axis. Source localization remains per-time/channel calibrated energy and is not replaced by a black-box classifier.
""", encoding="utf-8")
    (review_dir / "06_novelty_guardrail_v2.md").write_text("""# Novelty Guardrail v2

CIRFL_v2 is not an improved Transformer/GNN/LSTM/CNN/AE and does not add a stacked large module. The redesign only repairs the internal residual-field mechanism: condition decoupling, relation atom dictionary diversity, calibrated residual scoring, and source contribution.

Difference from AE/USAD: CIRFL_v2 scores condition-decoupled residual relation mismatch rather than reconstruction error. Difference from GNN/attention methods: relation atoms are a residual-field dictionary, not graph message passing or attention stacking.

Current risk: the relaxed RoAD Gate v2 is GO, but XGBoost still reaches near-perfect AUROC/PR-AUC on the binary main protocol. Full experiments must test whether CIRFL_v2 keeps its advantage under stricter cross-scenario, cross-condition, and external-dataset protocols.
""", encoding="utf-8")
    with (review_dir / "07_gate_v2_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    shutil.copyfile(output_dir / "gate_v2_metrics.csv", review_dir / "08_gate_v2_metrics.csv")
    shutil.copyfile(output_dir / "gate_v2_baseline_comparison.csv", review_dir / "09_gate_v2_baseline_comparison.csv")
    _copy_or_stub(output_dir / "threshold_score_sanity.md", review_dir / "10_threshold_score_sanity.md", "# Threshold Score Sanity\n\nNo report found.\n")
    _copy_or_stub(output_dir / "mechanism_diagnosis_v2.md", review_dir / "11_mechanism_diagnosis_v2.md", "# Mechanism Diagnosis v2\n\nNo report found.\n")
    _copy_or_stub(output_dir / "ablation_v2.md", review_dir / "12_ablation_v2.md", "# Ablation v2\n\nNo ablation report found.\n")
    _copy_or_stub(output_dir / "statistical_summary_v2.md", review_dir / "13_statistical_summary_v2.md", "# Statistical Summary v2\n\nNo summary found.\n")
    _copy_or_stub(output_dir / "complexity_latency_v2.md", review_dir / "14_complexity_latency_v2.md", "# Complexity Latency v2\n\nNo latency report found.\n")
    _copy_or_stub(output_dir / "go_no_go_report_v2.md", review_dir / "15_go_no_go_report_v2.md", "# GO/NO-GO v2\n\nNo report found.\n")
    errors = _read(output_dir / "errors_and_risks.md", "# Errors and Risks\n\n- No error report found.\n")
    figure_note = "- Figures are intentionally not generated or copied in this algorithm-improvement stage."
    if figure_note not in errors:
        errors += "\n" + figure_note + "\n"
    (review_dir / "16_errors_and_risks.md").write_text(errors, encoding="utf-8")
    (review_dir / "17_code_index.md").write_text("""# Code Index

- `src/utils/torch_utils.py`: automatic CPU/GPU benchmark-based device selection and checkpoint utilities.
- `src/utils/provenance.py`: provenance validation for real-only metric files.
- `scripts/check_packet_integrity.py`: packet metric integrity checker.
- `scripts/analyze_road_feasibility.py`: RoAD label/run/condition feasibility report.
- `scripts/build_road_protocols.py`: valid RoAD v2 protocol construction and YAML generation.
- `configs/real_gate_v2_road_binary.yaml`: binary main protocol config.
- `configs/real_gate_v2_road_scenario_holdout.yaml`: scenario holdout config collection.
- `configs/real_gate_v2_road_condition_holdout.yaml`: condition holdout config.
- `configs/real_gate_v2_road_all.yaml`: all v2 protocols used by `run_real_gate_v2.py`.
- `src/datasets/timeseries.py`: train-only robust normalization, clipping, binary labels.
- `src/models/cirfl.py`: CIRFL_v2 stable scale, atom diversity, condition decorrelation.
- `src/training/train_cirfl.py`: CIRFL_v2 params, class-balanced loss, GRL schedule.
- `src/baselines/gate_baselines.py`: separate XGBoost and LightGBM baselines.
- `scripts/run_real_gate_v2.py`: Real Gate v2 runner.

Commands:
- `python scripts/analyze_road_feasibility.py`
- `python scripts/build_road_protocols.py --write-configs`
- `python scripts/run_real_gate_v2.py --config configs/real_gate_v2_road_all.yaml`
- `python scripts/check_packet_integrity.py`
""", encoding="utf-8")
    # No figures are copied in this stage by user request.
    files = [p for p in review_dir.iterdir() if p.is_file()]
    if len(files) > 20:
        raise RuntimeError(f"review packet has {len(files)} files, expected <= 20")
    return review_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_gate_v2_road_all.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    out = make_review_packet(config, ROOT)
    print(f"Wrote review packet to {out} ({len([p for p in out.iterdir() if p.is_file()])} files)")


if __name__ == "__main__":
    main()
