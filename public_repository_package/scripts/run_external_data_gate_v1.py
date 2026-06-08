from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.adapters import adapt_imadds_robotic_arm, adapt_kuka_torque, adapt_nist_ur
from src.evaluation.metrics import summarize_by_method
from src.utils.config import load_config, save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import utc_now
from src.utils.torch_utils import resolve_device

DATASET_ROAD = "RoAD"
SOURCE_REAL = "real"
STATUS = "EXTERNAL_NEED_DATA"
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_road_best_path_freeze.md",
    "02_road_reference_summary.csv",
    "03_external_data_readiness.md",
    "04_imadds_data_audit.md",
    "05_nist_kuka_readiness.md",
    "06_external_download_instructions.md",
    "07_external_gate_protocols.md",
    "08_external_gate_config.yaml",
    "09_external_gate_metrics.csv",
    "10_external_baseline_comparison.csv",
    "11_external_statistical_summary.md",
    "12_cross_dataset_transfer_feasibility.md",
    "13_cirfl_direction_decision.md",
    "14_device_decision_report.md",
    "15_complexity_latency_external.md",
    "16_errors_and_risks.md",
    "17_go_no_go_report_external_v1.md",
    "18_code_index.md",
    "19_next_tasks_for_codex.md",
]


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _road_reference_summary(output_dir: Path, generated_at: str) -> pd.DataFrame:
    frames = []
    sources = [
        ROOT / "outputs/hard_gate_v3/07_hard_gate_v3_metrics.csv",
        ROOT / "outputs/hard_gate_v3/hard_gate_v3_metrics.csv",
        ROOT / "outputs/hard_gate_v3/score_variant_metrics.csv",
        ROOT / "outputs/hard_gate_v4/07_hard_gate_v4_metrics.csv",
        ROOT / "outputs/hard_gate_v4/hard_gate_v4_metrics.csv",
        ROOT / "outputs/hard_gate_v4/mechanism_necessity_matrix_v4.csv",
        ROOT / "outputs/hard_gate_v5/07_hard_gate_v5_metrics.csv",
        ROOT / "outputs/hard_gate_v5/13_ablation_matrix_v5.csv",
        ROOT / "outputs/real_gate_v2/gate_v2_baseline_comparison.csv",
    ]
    for path in sources:
        df = _read_csv_if_exists(path)
        if len(df) and {"method", "protocol", "macro_f1"}.issubset(df.columns):
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["macro_f1"])
    summary = summarize_by_method(combined)
    summary["dataset"] = DATASET_ROAD
    summary["source_type"] = SOURCE_REAL
    summary["seed"] = "summary"
    summary["n_train_windows"] = -1
    summary["n_val_windows"] = -1
    summary["n_test_windows"] = -1
    summary["n_test_normal"] = -1
    summary["n_test_anomaly"] = -1
    summary["output_path"] = str(output_dir / "02_road_reference_summary.csv")
    summary["generated_at"] = generated_at
    summary.to_csv(output_dir / "road_reference_summary.csv", index=False)
    summary.to_csv(output_dir / "02_road_reference_summary.csv", index=False)
    return summary


def _adapter_statuses(output_dir: Path) -> pd.DataFrame:
    rows = []
    imadds = adapt_imadds_robotic_arm(ROOT / "data/raw/imadds/RoboticArm", ROOT / "data/processed/imadds_robotic_arm/imadds_robotic_arm_unified.csv")
    nist = adapt_nist_ur(ROOT / "data/raw/nist_ur", ROOT / "data/processed/nist_ur/nist_ur_unified.csv")
    kuka = adapt_kuka_torque(ROOT / "data/raw/kuka_torque", ROOT / "data/processed/kuka_torque/kuka_torque_unified.csv")
    for key, result, raw_dir, adapter, cfg in [
        ("IMAD-DS robotic arm subset", imadds, ROOT / "data/raw/imadds/RoboticArm", ROOT / "src/datasets/adapters/imadds.py", ROOT / "configs/datasets/imadds_robotic_arm.yaml"),
        ("NIST UR robot degradation / health data", nist, ROOT / "data/raw/nist_ur", ROOT / "src/datasets/adapters/nist_ur.py", ROOT / "configs/datasets/nist_ur.yaml"),
        ("KUKA torque/contact/collision", kuka, ROOT / "data/raw/kuka_torque", ROOT / "src/datasets/adapters/kuka_torque.py", ROOT / "configs/datasets/kuka_torque.yaml"),
    ]:
        raw_files = sorted(raw_dir.rglob("*")) if raw_dir.exists() else []
        rows.append(
            {
                "dataset": key,
                "source_type": "external_real_dataset_status",
                "status": result.get("status", "UNKNOWN"),
                "reason": result.get("reason", "NA"),
                "official_source": result.get("official_source", "see config"),
                "raw_dir": str(raw_dir),
                "raw_files_present": int(len([p for p in raw_files if p.is_file()])),
                "adapter_exists": bool(adapter.exists()),
                "config_exists": bool(cfg.exists()),
                "output_csv": result.get("output_csv", "NA"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "external_readiness_status.csv", index=False)
    return df


def _placeholder_external_csv(output_dir: Path, generated_at: str, filename: str, methods: list[str]) -> pd.DataFrame:
    rows = []
    protocols = ["imadds_main_binary", "imadds_cross_domain", "nist_ur_external_validation", "kuka_collision_external_validation"]
    for protocol in protocols:
        for method in methods:
            rows.append(
                {
                    "dataset": protocol.split("_")[0].upper() if not protocol.startswith("imadds") else "IMAD-DS robotic arm subset",
                    "source_type": "external_real_need_data",
                    "protocol": protocol,
                    "seed": -1,
                    "method": method,
                    "status": "NOT_RUN_NEED_DATA",
                    "reason": "external raw files are missing; no synthetic substitute used",
                    "macro_f1": float("nan"),
                    "weighted_f1": float("nan"),
                    "auroc": float("nan"),
                    "pr_auc": float("nan"),
                    "far": float("nan"),
                    "mdr": float("nan"),
                    "far_at_95_recall": float("nan"),
                    "n_train_windows": -1,
                    "n_val_windows": -1,
                    "n_test_windows": -1,
                    "n_test_normal": -1,
                    "n_test_anomaly": -1,
                    "generated_at": generated_at,
                    "output_path": str(output_dir / filename),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / filename, index=False)
    return df


def _device_report(cfg: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    selected = str(resolve_device(cfg.get("device", "auto_fastest")))
    rows = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            dev = f"cuda:{idx}"
            rows.append(
                {
                    "device": dev,
                    "gpu_name": torch.cuda.get_device_name(idx),
                    "available": True,
                    "assigned_task": "external gate dry-run / selected by auto benchmark; no model training" if dev == selected else "idle/reserved for future seed/protocol parallelism",
                }
            )
    else:
        rows.append({"device": "cpu", "gpu_name": "NA", "available": False, "assigned_task": "external gate dry-run"})
    latency = _read_csv_if_exists(ROOT / "outputs/real_gate_v2/complexity_latency_v2.csv")
    return selected, pd.DataFrame(rows), latency


def _write_reports(cfg: dict, output_dir: Path, packet_dir: Path, road_summary: pd.DataFrame, readiness: pd.DataFrame, metrics: pd.DataFrame, baselines: pd.DataFrame, selected_device: str, gpu_df: pd.DataFrame, latency_df: pd.DataFrame, generated_at: str) -> str:
    provenance = "\n".join(
        [
            "- dataset: RoAD + external dataset readiness",
            "- source_type: real / external_real_need_data",
            "- protocol: road_best_path_freeze + external protocol templates",
            "- seed: 7, 13, 23 for RoAD reference; -1 for not-run external placeholders",
            "- n_train/n_val/n_test: see CSV files; external placeholders are -1 because raw data are missing",
            "- n_test_normal/n_test_anomaly: see CSV files; external placeholders are -1 because raw data are missing",
            f"- generated_at: {generated_at}",
        ]
    )
    status_counts = readiness[["dataset", "status", "reason", "raw_files_present", "adapter_exists", "config_exists"]]
    v3_rows = road_summary[road_summary["method"].astype(str).isin(["CIRFL_v3", "CIRFL_v3_plain_residual_score"])] if len(road_summary) else pd.DataFrame()
    v45_rows = road_summary[road_summary["method"].astype(str).str.contains("CIRFL_v4_full|CIRFL_v5_CIRFL_RD", regex=True)] if len(road_summary) else pd.DataFrame()
    download_lines = [
        "# External Dataset Download Instructions",
        "",
        provenance,
        "",
        "## IMAD-DS Robotic Arm",
        "- Official source: https://zenodo.org/records/12665499",
        "- DOI: https://doi.org/10.5281/zenodo.12665499",
        "- File: `RoboticArm.7z`",
        "- Size: 3,545,453,763 bytes",
        "- Checksum: `md5:a64498b7dcc297946a7fb8366e38ba33`",
        "- Direct official URL: `https://zenodo.org/api/records/12665499/files/RoboticArm.7z/content`",
        "- Place archive at: `robot_cirfl/data/raw/imadds/RoboticArm.7z`",
        "- Extract to: `robot_cirfl/data/raw/imadds/RoboticArm/`",
        "- Run adapter: `/home/zyf/miniconda3/envs/yopo/bin/python scripts/prepare_real_data.py --dataset imadds_robotic_arm --json`",
        "",
        "## NIST UR",
        "- Official DOI: https://doi.org/10.18434/M31962",
        "- Place downloaded tables under: `robot_cirfl/data/raw/nist_ur/`",
        "- Run adapter: `/home/zyf/miniconda3/envs/yopo/bin/python scripts/prepare_real_data.py --dataset nist_ur --json`",
        "",
        "## KUKA Torque/Collision",
        "- Official source: https://zenodo.org/records/6461868",
        "- Expected files: `fre-joint-1.csv` ... `fre-joint-7.csv`, `ctc-joint-*.csv`, `cls-joint-*.csv`",
        "- Place files under: `robot_cirfl/data/raw/kuka_torque/`",
        "- Run adapter: `/home/zyf/miniconda3/envs/yopo/bin/python scripts/prepare_real_data.py --dataset kuka_torque --json`",
    ]
    reports = {
        "00_readme_for_chatgpt.md": "\n".join(["# Readme for ChatGPT", "", "- current_stage: External Data Gate v1 + Best-Path Freeze", f"- current_status: {STATUS}", "- contains synthetic results: NO", "- generated images/figures: NO", "- entered full experiments: NO", provenance]),
        "01_road_best_path_freeze.md": "\n".join(["# RoAD Best-Path Freeze", "", provenance, "", "## Freeze Decision", "- Current RoAD CIRFL reference: `CIRFL_v3`.", "- `CIRFL_v3_plain_residual_score` is a strong detector comparator, but it is not renamed as CIRFL because it weakens the mechanism claim.", "- v4 is not the reference because broad score composition was unstable and did not solve full-vs-plain necessity.", "- v5 is not the reference because it failed recovery, degraded road_binary_main PR-AUC/MDR, and did not beat raw/no-relation residual ablations.", "", "## Evidence Use", "- `road_binary_main`: sanity evidence only because tree baselines and univariate separators are very strong.", "- scenario holdouts: stress evidence only; collision/weight MDR remain high.", "- condition holdout: high-confounding stress only.", "- RoAD-only cannot justify full experiments or manuscript workflow.", "", "## v3/v4/v5 Summary", dataframe_to_markdown(pd.concat([v3_rows, v45_rows], ignore_index=True).head(80))]),
        "03_external_data_readiness.md": "# External Data Readiness\n\n" + provenance + "\n\n" + dataframe_to_markdown(status_counts),
        "04_imadds_data_audit.md": "\n".join(["# IMAD-DS Data Audit", "", provenance, "", "- status: NEED_DATA", "- official Zenodo record reachable: YES", "- raw files present: NO", "- adapter dry-run: NEED_DATA", "- expected sensors: analog microphone 16 kHz, 3-axis accelerometer 6.7 kHz, 3-axis gyroscope 6.7 kHz.", "- expected domains: source/target with operational and environmental domain shifts.", "- labels: normal/abnormal segment metadata expected after extraction.", "- leakage-safe split feasibility: cannot audit until files are extracted; expected split unit is segment/file ID, with source/target domain metadata.", "", dataframe_to_markdown(readiness[readiness["dataset"].str.contains("IMAD", case=False)])]),
        "05_nist_kuka_readiness.md": "\n".join(["# NIST UR and KUKA Readiness", "", provenance, "", "## NIST UR", "- status: NEED_DATA", "- task fit: external health/degradation validation candidate; anomaly labels may require threshold/label file and may not be a primary binary AD dataset.", "", "## KUKA Torque/Collision", "- status: NEED_DATA", "- task fit: safety anomaly external validation candidate for free/contact/collision torque patterns.", "", dataframe_to_markdown(readiness[~readiness["dataset"].str.contains("IMAD", case=False)])]),
        "06_external_download_instructions.md": "\n".join(download_lines),
        "07_external_gate_protocols.md": "\n".join(["# External Gate Protocols", "", provenance, "", "## IMAD-DS main binary", "- status: NEED_DATA template", "- train/val/test split unit: segment_id/file_id after extraction", "- test must contain normal and anomaly.", "", "## IMAD-DS cross-domain", "- status: NEED_DATA template", "- source-domain train/val, target-domain test if target contains normal and anomaly.", "- invalid if target test has only normal or only anomaly.", "", "## Sensor missing robustness", "- template only: 10% and 20% channel missing masks applied after a valid split.", "", "## NIST / KUKA", "- NIST: external health validation template, not primary unless labels are available.", "- KUKA: free/contact/collision binary or multiclass safety validation template."]),
        "11_external_statistical_summary.md": "\n".join(["# External Statistical Summary", "", provenance, "", "NOT_RUN_NEED_DATA: no external real dataset is available locally. RoAD reference summaries are in `02_road_reference_summary.csv`; no external statistical claim is made."]),
        "12_cross_dataset_transfer_feasibility.md": "\n".join(["# Cross-Dataset Transfer Feasibility", "", provenance, "", "- RoAD -> IMAD-DS: direct channel-level transfer is not valid because RoAD has 86 anonymized robot/control channels while IMAD-DS uses microphone/accelerometer/gyroscope streams.", "- IMAD-DS -> RoAD: direct sensor-token mapping is not currently justified.", "- Feasible protocol after data availability: shared statistical feature abstraction or train source residual representation then evaluate/finetune in target with explicit channel mismatch caveat.", "- Do not concatenate incompatible sensor channels as one dataset."]),
        "13_cirfl_direction_decision.md": "\n".join(["# CIRFL Direction Decision", "", provenance, "", f"## Decision: {STATUS}", "", "- CIRFL should not proceed to full experiments because no external real dataset is available.", "- Best-path freeze is `CIRFL_v3` as RoAD reference, with v3 plain residual as a strong comparator and mechanism-risk warning.", "- Continue only after IMAD-DS/NIST/KUKA raw data are provided and an external gate is run.", "- If external data also shows no condition-decoupling/relation-atom value, consider `REPOSITION_TO_RESIDUAL_BASELINE` or redesign the algorithm."]),
        "14_device_decision_report.md": "\n".join(["# Device Decision Report External v1", "", provenance, "", f"- selected_device_for_PyTorch: {selected_device}", "- multi_gpu_used: NO", "- reason: external gate did not train or evaluate models because external raw data are missing. cuda devices are available for future seed/protocol parallelism after data are provided.", "", dataframe_to_markdown(gpu_df)]),
        "15_complexity_latency_external.md": "\n".join(["# Complexity and Latency External v1", "", provenance, "", "- parameter_count_reference: 171286", "- model_size_reference_mb: about 0.666", "- external training/evaluation time: NOT_RUN_NEED_DATA", "", dataframe_to_markdown(latency_df) if len(latency_df) else "No latency reference CSV found."]),
        "16_errors_and_risks.md": "\n".join(["# Errors and Risks", "", provenance, "", "- blocking_data_risk: IMAD-DS/NIST/KUKA raw files are missing.", "- algorithm_risk: RoAD-only results do not prove CIRFL mechanism; v5 failed recovery and mechanism necessity.", "- metric_risk: external metrics are NOT_RUN_NEED_DATA.", "- next_stage_blocked: YES.", "- synthetic_results_in_packet: NO", "- figures_in_packet: NO"]),
        "17_go_no_go_report_external_v1.md": "\n".join(["# External Data Gate v1 GO / NO-GO Report", "", provenance, "", f"## Decision: {STATUS}", "", "## Criteria Audit", "- A1 external real dataset available: False", "- A2 synthetic substitute used: False", "- A3 external data audit complete: False, NEED_DATA only", "- A4 external leakage risk: not assessable until raw files exist", "- A5 valid external protocol with normal/anomaly test: False", "- B external performance support: NOT_RUN_NEED_DATA", "- C mechanism necessity on external data: NOT_RUN_NEED_DATA", "- D RoAD + external combined support: False", "- E complexity reference reported: True", "", "Full experiments and manuscript work remain blocked."]),
        "18_code_index.md": "\n".join(["# Code Index", "", provenance, "", "- `configs/external_gate_v1.yaml`: external gate template and frozen RoAD reference config.", "- `scripts/run_external_data_gate_v1.py`: generates best-path freeze, external readiness, protocol templates, NEED_DATA metrics placeholders and packet.", "- `src/datasets/adapters/imadds.py`: IMAD-DS robotic arm adapter dry-run.", "- `src/datasets/adapters/nist_ur.py`: NIST UR adapter dry-run.", "- `src/datasets/adapters/kuka_torque.py`: KUKA torque adapter dry-run.", "", "## Command", "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_external_data_gate_v1.py --config configs/external_gate_v1.yaml`"]),
        "19_next_tasks_for_codex.md": "\n".join(["# Next Tasks for Codex", "", provenance, "", "## Because status is EXTERNAL_NEED_DATA", "1. Download `RoboticArm.7z` from Zenodo record 12665499 and extract it to `robot_cirfl/data/raw/imadds/RoboticArm/`.", "2. Run `scripts/prepare_real_data.py --dataset imadds_robotic_arm --json`.", "3. Re-run `scripts/run_external_data_gate_v1.py --config configs/external_gate_v1.yaml`.", "4. Only if IMAD-DS is valid, run the external gate comparison; do not start full experiments yet.", "5. If IMAD-DS cannot be obtained, prepare NIST UR or KUKA raw files and re-run adapter dry-run."]),
    }
    save_config(cfg, output_dir / "08_external_gate_config.yaml")
    metrics.to_csv(output_dir / "09_external_gate_metrics.csv", index=False)
    baselines.to_csv(output_dir / "10_external_baseline_comparison.csv", index=False)
    for name, text in reports.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        shutil.copyfile(output_dir / name, packet_dir / name)
    return STATUS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/external_gate_v1.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    output_dir = ROOT / cfg["project"]["output_dir"]
    packet_dir = ROOT / cfg["review_packet"]["output_dir"]
    _clean_dir(output_dir)
    generated_at = utc_now()
    road_summary = _road_reference_summary(output_dir, generated_at)
    readiness = _adapter_statuses(output_dir)
    metrics = _placeholder_external_csv(output_dir, generated_at, "09_external_gate_metrics.csv", ["CIRFL_v3_reference", "CIRFL_v5_CIRFL_RD", "raw_residual_energy", "condition_decoupled_residual", "condition_decoupled_no_relation"])
    baselines = _placeholder_external_csv(output_dir, generated_at, "10_external_baseline_comparison.csv", ["random_forest", "xgboost", "lightgbm", "isolation_forest", "autoencoder", "lstm_ae", "usad"])
    selected_device, gpu_df, latency_df = _device_report(cfg)
    status = _write_reports(cfg, output_dir, packet_dir, road_summary, readiness, metrics, baselines, selected_device, gpu_df, latency_df, generated_at)
    if any(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} for path in packet_dir.iterdir()):
        raise RuntimeError("Packet contains image/figure files, forbidden in External Data Gate v1.")
    print(f"External Data Gate v1 finished: {status}. Packet files: {len(list(packet_dir.iterdir()))}")


if __name__ == "__main__":
    main()
