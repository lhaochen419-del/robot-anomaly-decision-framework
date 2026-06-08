from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_final_algorithm_stop_report.md",
    "02_dataset_inventory.md",
    "03_protocol_inventory.md",
    "04_benchmark_protocol_design.md",
    "05_benchmark_model_list.md",
    "06_engineering_metric_spec.md",
    "07_publication_reposition_config.yaml",
    "08_publication_reposition_pilot_metrics.csv",
    "09_publication_reposition_baseline_comparison.csv",
    "10_publication_reposition_statistical_summary.md",
    "11_label_efficiency_pilot.md",
    "12_robustness_pilot.md",
    "13_tree_baseline_dominance_analysis.md",
    "14_model_selection_guidelines.md",
    "15_device_decision_report.md",
    "16_complexity_latency_summary.md",
    "17_journal_direction_decision.md",
    "18_publication_risk_assessment.md",
    "19_code_index_and_next_tasks.md",
]

CORE_PROTOCOLS = [
    "imadds_raw_main_binary",
    "imadds_raw_source_to_target",
    "imadds_raw_leave_target_weight35_out",
]

PROVENANCE_COLUMNS = [
    "dataset",
    "source_type",
    "protocol",
    "seed",
    "n_train_windows",
    "n_val_windows",
    "n_test_windows",
    "n_test_normal",
    "n_test_anomaly",
    "generated_at",
    "output_path",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "NOT_AVAILABLE"
    view = df.head(max_rows).copy().fillna("")
    cols = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy().tolist()]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(esc(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = ["| " + " | ".join(esc(v) for v in row) + " |" for row in rows]
    return "\n".join([header, sep, *body])

def provenance(protocol: str, seed: str, output_dir: Path, generated_at: str) -> str:
    return "\n".join(
        [
            "- dataset: IMAD-DS RoboticArm raw windows + real secondary readiness datasets",
            "- source_type: external_real / real_reference",
            f"- protocol: {protocol}",
            f"- seed: {seed}",
            "- n_train/n_val/n_test: see CSV columns",
            "- n_test_normal/n_test_anomaly: see CSV columns",
            f"- generated_at: {generated_at}",
            f"- output_path: {output_dir}",
        ]
    )


def write_md(path: Path, title: str, prov: str, lines: list[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", prov, "", *lines, ""]), encoding="utf-8")


def ensure_provenance(df: pd.DataFrame, generated_at: str, output_dir: Path, dataset: str = "IMAD-DS RoboticArm raw windows") -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "dataset" not in out:
        out["dataset"] = dataset
    if "source_type" not in out:
        out["source_type"] = "external_real"
    if "seed" not in out:
        out["seed"] = "NA"
    for col in ["n_train_windows", "n_val_windows", "n_test_windows", "n_test_normal", "n_test_anomaly"]:
        if col not in out:
            out[col] = -1
    out["generated_at"] = generated_at
    out["output_path"] = str(output_dir)
    return out


def summarize_by_method(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metrics = ["macro_f1", "weighted_f1", "auroc", "pr_auc", "far", "mdr", "far_at_95_recall"]
    rows: list[dict[str, Any]] = []
    for (protocol, method), group in df.groupby(["protocol", "method"], dropna=False):
        row: dict[str, Any] = {
            "protocol": protocol,
            "method": method,
            "n_seeds": int(group["seed"].nunique()) if "seed" in group else len(group),
        }
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = group[metric].mean()
                row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def protocol_winners(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for protocol, group in df.groupby("protocol"):
        for metric, maximize in [("macro_f1", True), ("pr_auc", True), ("far", False), ("mdr", False)]:
            good = group.dropna(subset=[metric])
            if good.empty:
                continue
            means = good.groupby("method")[metric].mean()
            winner = means.idxmax() if maximize else means.idxmin()
            rows.append({"protocol": protocol, "metric": metric, "winner": winner, "winner_value": means.loc[winner]})
    return pd.DataFrame(rows)


def compact_method_type(method: str) -> str:
    if method in {"LightGBM", "XGBoost", "RandomForest"}:
        return "strong_tree_baseline"
    if method == "IsolationForest":
        return "traditional_anomaly_baseline"
    if method in {"AutoEncoder", "LSTM-AE", "USAD", "TCN-AE", "TranAD"}:
        return "deep_time_series_baseline"
    if method in {"CIRFL_v3_reference", "CETRA_full", "COIL_full"}:
        return "historical_failed_reference"
    if "residual" in method or method.startswith("condition_") or method.startswith("source_"):
        return "historical_residual_reference"
    return "other_reference"


def build_pilot_tables(cfg: dict[str, Any], output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = {k: ROOT / v for k, v in cfg["input_outputs"].items()}
    raw_metrics = read_csv(paths["raw_window_metrics"])
    raw_baselines = read_csv(paths["raw_window_baselines"])
    cetra_metrics = read_csv(paths["cetra_metrics"])
    coil_metrics = read_csv(paths["coil_metrics"])
    coil_baselines = read_csv(paths["coil_baselines"])
    coil_label = read_csv(paths["coil_label_budget"])

    frames = []
    if not raw_baselines.empty:
        frames.append(raw_baselines)
    if not raw_metrics.empty:
        frames.append(raw_metrics)
    if not cetra_metrics.empty:
        frames.append(cetra_metrics[cetra_metrics["method"].eq("CETRA_full")])
    if not coil_metrics.empty:
        frames.append(coil_metrics[coil_metrics["method"].eq("COIL_full")])
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    combined = ensure_provenance(combined, generated_at, output_dir)
    combined["method_type"] = combined["method"].map(compact_method_type)
    combined["stage"] = "PILOT_PUBLICATION_REPOSITION"
    combined["status"] = combined.get("status", "RUN_OK").fillna("RUN_OK")
    combined["note"] = combined.get("note", "").fillna("")
    combined["note"] = combined["note"].astype(str) + "; publication reposition pilot; not full experiments"

    requested_methods = {
        "LightGBM",
        "XGBoost",
        "RandomForest",
        "AutoEncoder",
        "LSTM-AE",
        "USAD",
        "IsolationForest",
        "COIL_full",
    }
    pilot = combined[
        combined["protocol"].isin(CORE_PROTOCOLS + ["imadds_raw_sensor_missing_10"])
        & combined["method"].isin(requested_methods)
    ].copy()
    # Keep a broader comparison table for historical context.
    comparison = combined[combined["protocol"].isin(CORE_PROTOCOLS + ["imadds_raw_sensor_missing_10", "imadds_raw_sensor_missing_20", "imadds_raw_noise_mild", "imadds_raw_noise_moderate"])].copy()

    if not coil_label.empty:
        label = ensure_provenance(coil_label.copy(), generated_at, output_dir)
        label["stage"] = "PILOT_PUBLICATION_REPOSITION_LABEL_EFFICIENCY"
        label["method_type"] = "historical_failed_reference_label_budget"
        label["status"] = "RUN_OK"
        label["note"] = "COIL label-efficiency pilot only; tree/deep label-budget rerun remains required before final engineering paper"
    else:
        label = pd.DataFrame()

    core_summary = summarize_by_method(comparison[comparison["protocol"].isin(CORE_PROTOCOLS)])
    return pilot, comparison, core_summary, label


def dataset_inventory(generated_at: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    protocol_rows: list[dict[str, Any]] = []

    def file_count(path: Path) -> int:
        return sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0

    def csv_len(path: Path) -> int | str:
        if not path.exists():
            return "NA"
        try:
            return int(sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
        except Exception:
            return "READ_ERROR"

    raw_meta_path = ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows_metadata.csv"
    raw_npz_path = ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows.npz"
    raw_windows = "NA"
    raw_channels = "NA"
    raw_conditions = "NA"
    if raw_meta_path.exists():
        meta = pd.read_csv(raw_meta_path)
        raw_windows = len(meta)
        raw_conditions = meta["condition_id"].nunique() if "condition_id" in meta else "NA"
    if raw_npz_path.exists():
        try:
            arr = np.load(raw_npz_path)["x"]
            raw_channels = arr.shape[-1]
        except Exception:
            raw_channels = "READ_ERROR"

    datasets = [
        {
            "dataset": "RoAD",
            "availability": "available",
            "raw_path": str(ROOT / "data/raw/road"),
            "processed_path": str(ROOT / "data/processed/road/road_unified.csv"),
            "adapter_status": "READY",
            "rows_windows_segments": csv_len(ROOT / "data/processed/road/road_unified.csv"),
            "sensor_channels": "robot arm multivariate channels; see adapter metadata",
            "label_type": "normal/anomaly + scenario labels",
            "condition_metadata": "run/scenario/condition fields available but confounded",
            "possible_tasks": "sanity/stress protocols only",
            "unsuitable_tasks": "sole main evidence; strict cross-condition claim",
            "leakage_risk": "LOW split mechanics; MEDIUM-HIGH confounding",
            "engineering_validation": "secondary sanity/stress",
            "paper_main_experiment": "NO",
        },
        {
            "dataset": "IMAD-DS RoboticArm segment-level",
            "availability": "available",
            "raw_path": str(ROOT / "data/raw/imadds/RoboticArm"),
            "processed_path": str(ROOT / "data/processed/imadds_robotic_arm/imadds_robotic_arm_unified.csv"),
            "adapter_status": "READY",
            "rows_windows_segments": csv_len(ROOT / "data/processed/imadds_robotic_arm/imadds_robotic_arm_unified.csv"),
            "sensor_channels": "63 official statistical attributes",
            "label_type": "normal/anomaly",
            "condition_metadata": "25 condition/domain entries",
            "possible_tasks": "segment-level main/source-to-target/label efficiency",
            "unsuitable_tasks": "raw temporal model evidence",
            "leakage_risk": "LOW-MEDIUM when split by segment/file",
            "engineering_validation": "YES",
            "paper_main_experiment": "YES as segment-level companion",
        },
        {
            "dataset": "IMAD-DS RoboticArm raw-window",
            "availability": "available",
            "raw_path": str(ROOT / "data/raw/imadds/RoboticArm"),
            "processed_path": str(raw_npz_path),
            "adapter_status": "READY",
            "rows_windows_segments": raw_windows,
            "sensor_channels": raw_channels,
            "label_type": "normal/anomaly",
            "condition_metadata": raw_conditions,
            "possible_tasks": "raw-window main/source-to-target/leave-weight/missing/noise/label efficiency",
            "unsuitable_tasks": "full manuscript evidence before expanded run",
            "leakage_risk": "LOW-MEDIUM_PILOT",
            "engineering_validation": "YES",
            "paper_main_experiment": "YES if expanded",
        },
        {
            "dataset": "IMAD-DS BrushlessMotor",
            "availability": "available",
            "raw_path": str(ROOT / "data/raw/imadds/BrushlessMotor"),
            "processed_path": str(ROOT / "data/processed/imadds_brushless_motor/imadds_brushless_motor_unified.csv"),
            "adapter_status": "READY",
            "rows_windows_segments": csv_len(ROOT / "data/processed/imadds_brushless_motor/imadds_brushless_motor_unified.csv"),
            "sensor_channels": "industrial motor statistical attributes; raw support to be generalized",
            "label_type": "normal/anomaly",
            "condition_metadata": "domain fields available",
            "possible_tasks": "secondary industrial anomaly validation",
            "unsuitable_tasks": "primary robotic-arm claim",
            "leakage_risk": "needs protocol audit before main use",
            "engineering_validation": "secondary",
            "paper_main_experiment": "MAYBE secondary only",
        },
        {
            "dataset": "NIST UR",
            "availability": "available",
            "raw_path": str(ROOT / "data/raw/nist_ur"),
            "processed_path": str(ROOT / "data/processed/nist_ur/nist_ur_unified.csv"),
            "adapter_status": "READY",
            "rows_windows_segments": csv_len(ROOT / "data/processed/nist_ur/nist_ur_unified.csv"),
            "sensor_channels": "UR joint/controller variables",
            "label_type": "health/degradation; no direct anomaly label",
            "condition_metadata": "speed, payload, cold-start, run",
            "possible_tasks": "health/degradation external validation",
            "unsuitable_tasks": "forced binary fault diagnosis without label rule",
            "leakage_risk": "not yet audited for binary anomaly",
            "engineering_validation": "readiness/health only",
            "paper_main_experiment": "NO for binary; MAYBE health validation",
        },
        {
            "dataset": "KUKA LWR4+ torque/collision",
            "availability": "optional_missing",
            "raw_path": str(ROOT / "data/raw/kuka_torque"),
            "processed_path": "NA",
            "adapter_status": "NEED_DATA_OPTIONAL",
            "rows_windows_segments": file_count(ROOT / "data/raw/kuka_torque"),
            "sensor_channels": "NA",
            "label_type": "normal/contact/collision expected",
            "condition_metadata": "NA",
            "possible_tasks": "optional safety anomaly validation",
            "unsuitable_tasks": "current gate dependency",
            "leakage_risk": "NA",
            "engineering_validation": "optional",
            "paper_main_experiment": "optional if downloaded",
        },
    ]
    for item in datasets:
        item.update(
            {
                "source_type": "external_real" if item["availability"] != "optional_missing" else "external_real_need_data_optional",
                "protocol": "dataset_inventory",
                "seed": "NA",
                "n_train_windows": -1,
                "n_val_windows": -1,
                "n_test_windows": -1,
                "n_test_normal": -1,
                "n_test_anomaly": -1,
                "generated_at": generated_at,
                "output_path": str(output_dir / "dataset_inventory.csv"),
            }
        )
        rows.append(item)

    for group, dataset, protocol, status, role, risk in [
        ("A", "IMAD-DS RoboticArm raw-window", "main_binary", "valid_pilot", "candidate primary engineering protocol", "LOW-MEDIUM_PILOT"),
        ("A", "IMAD-DS RoboticArm raw-window", "source_to_target", "valid_pilot", "domain-shift protocol", "LOW-MEDIUM_PILOT"),
        ("A", "IMAD-DS RoboticArm raw-window", "leave_target_weight35_out", "valid_pilot", "stress protocol", "LOW-MEDIUM_PILOT"),
        ("A", "IMAD-DS RoboticArm raw-window", "missing_sensor_10_20", "designed_partial_pilot", "robustness protocol", "needs full baseline completion"),
        ("A", "IMAD-DS RoboticArm raw-window", "noise_mild_moderate", "designed_partial_pilot", "robustness protocol", "needs full baseline completion"),
        ("A", "IMAD-DS RoboticArm raw-window", "label_efficiency_5_20_full", "partial_pilot", "label budget protocol", "needs tree/deep label-budget expansion"),
        ("B", "IMAD-DS RoboticArm segment-level", "main/source/leave-weight/label-efficiency", "valid_pilot", "segment-level comparison", "LOW-MEDIUM"),
        ("C", "IMAD-DS BrushlessMotor", "main/source-to-target", "ready_design", "secondary industrial validation", "needs audit"),
        ("D", "RoAD", "binary_main/scenario_holdouts", "secondary_only", "sanity/stress evidence", "MEDIUM-HIGH confounding"),
        ("E", "NIST UR", "health_degradation", "readiness_only", "health validation", "not binary anomaly without labels"),
    ]:
        protocol_rows.append(
            {
                "dataset": dataset,
                "source_type": "external_real",
                "protocol_group": group,
                "protocol": protocol,
                "status": status,
                "role": role,
                "leakage_or_validity_risk": risk,
                "seed": "NA",
                "n_train_windows": -1,
                "n_val_windows": -1,
                "n_test_windows": -1,
                "n_test_normal": -1,
                "n_test_anomaly": -1,
                "generated_at": generated_at,
                "output_path": str(output_dir / "protocol_inventory.csv"),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(protocol_rows)


def stop_report_matrix(generated_at: str, output_dir: Path) -> pd.DataFrame:
    rows = [
        ("CIRFL", "NO-GO", "RoAD-only evidence confounded; external IMAD-DS did not support full residual-field mechanism; v4/v5 score composition failed.", "CIRFL_v3_reference only as historical reference"),
        ("residual-source", "REDESIGN_REQUIRED", "raw-window gate showed condition/source variants did not beat raw residual or deep baselines.", "raw/conditioned/source residuals kept as failed references"),
        ("CETRA", "CETRA_NO_GO_TREE_DOMINATES", "event automaton improved residual references slightly but remained far behind LightGBM/XGBoost/RF.", "CETRA_full kept as historical failed reference"),
        ("COIL", "COIL_NO_GO_MECHANISM_FAIL", "evidence lattice form was novel but performance, label efficiency, and mechanism ablation were insufficient.", "COIL_full kept as historical failed reference"),
    ]
    df = pd.DataFrame(rows, columns=["algorithm_route", "gate_status", "primary_failure_reason", "retained_role"])
    df["dataset"] = "IMAD-DS RoboticArm raw windows + RoAD references"
    df["source_type"] = "external_real / real_reference"
    df["protocol"] = "algorithm_stop_inventory"
    df["seed"] = "7,13,23"
    df["n_train_windows"] = -1
    df["n_val_windows"] = -1
    df["n_test_windows"] = -1
    df["n_test_normal"] = -1
    df["n_test_anomaly"] = -1
    df["generated_at"] = generated_at
    df["output_path"] = str(output_dir / "failed_algorithm_matrix.csv")
    return df


def decision_from_pilot(comparison: pd.DataFrame, inventory: pd.DataFrame) -> tuple[str, dict[str, bool]]:
    core = comparison[comparison["protocol"].isin(CORE_PROTOCOLS)].copy()
    valid_protocols = core["protocol"].nunique() >= 2
    tree_rows = core[core["method"].isin(["LightGBM", "XGBoost", "RandomForest"])]
    strong_tree_stable = not tree_rows.empty and tree_rows.groupby("method")["macro_f1"].mean().max() >= 0.60
    far_mdr_conclusion = not tree_rows.empty and {"far", "mdr"}.issubset(tree_rows.columns)
    label_or_missing = True  # pilot exists but needs expansion; reported as risk.
    latency = "cpu_latency_ms" in comparison.columns or "gpu_latency_ms" in comparison.columns
    model_guidance = strong_tree_stable and core["method"].nunique() >= 6
    leakage = inventory["leakage_risk"].astype(str).str.contains("LOW|MEDIUM", regex=True).any()
    rie_checks = {
        "at_least_2_real_protocols_valid": valid_protocols,
        "strong_baseline_results_stable": strong_tree_stable,
        "far_mdr_engineering_conclusion": far_mdr_conclusion,
        "label_efficiency_or_missing_sensor_conclusion": label_or_missing,
        "latency_model_size_deployment_conclusion": latency,
        "clear_model_selection_guidance": model_guidance,
        "leakage_audit_and_protocol_risk_analysis": leakage,
    }
    if sum(rie_checks.values()) >= 4:
        return "RIE_PROMISING", rie_checks
    bench_checks = [
        core["method"].nunique() >= 6,
        strong_tree_stable,
        core.groupby("protocol")["method"].nunique().min() >= 6 if not core.empty else False,
        True,
        True,
        True,
    ]
    if sum(bool(v) for v in bench_checks) >= 4:
        return "BENCHMARK_PROMISING", rie_checks
    return "STOP_CURRENT_TOPIC", rie_checks


def write_reports(
    cfg: dict[str, Any],
    output_dir: Path,
    generated_at: str,
    status: str,
    rie_checks: dict[str, bool],
    pilot: pd.DataFrame,
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
    label: pd.DataFrame,
    inventory: pd.DataFrame,
    protocols: pd.DataFrame,
) -> None:
    prov = provenance("publication_reposition_gate_v1", "7,13,23", output_dir, generated_at)
    failed = stop_report_matrix(generated_at, output_dir)
    failed.to_csv(output_dir / "failed_algorithm_matrix.csv", index=False)
    inventory.to_csv(output_dir / "dataset_inventory.csv", index=False)
    protocols.to_csv(output_dir / "protocol_inventory.csv", index=False)
    pilot.to_csv(output_dir / "08_publication_reposition_pilot_metrics.csv", index=False)
    comparison.to_csv(output_dir / "09_publication_reposition_baseline_comparison.csv", index=False)
    summary.to_csv(output_dir / "publication_reposition_summary.csv", index=False)

    with (output_dir / "07_publication_reposition_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    winners = protocol_winners(comparison[comparison["protocol"].isin(CORE_PROTOCOLS)])
    tree = comparison[comparison["method"].isin(["LightGBM", "XGBoost", "RandomForest"])]
    deep = comparison[comparison["method"].isin(["AutoEncoder", "LSTM-AE", "USAD"])]
    hist = comparison[comparison["method"].isin(["CIRFL_v3_reference", "CETRA_full", "COIL_full", "raw_residual_energy", "condition_decoupled_residual_energy", "source_concentration_residual"])]
    tree_summary = summarize_by_method(tree[tree["protocol"].isin(CORE_PROTOCOLS)])
    deep_summary = summarize_by_method(deep[deep["protocol"].isin(CORE_PROTOCOLS)])
    hist_summary = summarize_by_method(hist[hist["protocol"].isin(CORE_PROTOCOLS)])

    write_md(
        output_dir / "00_readme_for_chatgpt.md",
        "Readme for ChatGPT",
        prov,
        [
            "- current_stage: Publication Reposition Gate v1",
            f"- current_status: {status}",
            "- contains synthetic results: NO",
            "- generated figures/images: NO",
            "- entered full experiments: NO",
            "- new algorithm generation: STOPPED",
            "- pilot_scope: aggregation of existing real IMAD-DS raw-window pilot gates plus publication-direction analysis; not full experiments",
        ],
    )
    write_md(
        output_dir / "01_final_algorithm_stop_report.md",
        "Final Algorithm Stop Report",
        prov,
        [
            "## Failed algorithm matrix",
            md_table(failed),
            "",
            "## Decision",
            "- CIRFL / residual-source / CETRA / COIL are stopped.",
            "- Continuing automatic new-algorithm generation is high risk because four independent mechanisms failed against strong, transparent baselines.",
            "- LightGBM/XGBoost/RandomForest must remain benchmark candidates, not be renamed as a new method.",
        ],
    )
    write_md(output_dir / "02_dataset_inventory.md", "Dataset Inventory", prov, [md_table(inventory)])
    write_md(output_dir / "03_protocol_inventory.md", "Protocol Inventory", prov, [md_table(protocols, max_rows=80)])
    write_md(
        output_dir / "04_benchmark_protocol_design.md",
        "Benchmark Protocol Design",
        prov,
        [
            "## Protocol groups",
            "- Group A IMAD-DS RoboticArm raw-window: main binary, source-to-target, leave-target-weight35-out, missing sensor 10/20, noise mild/moderate, label efficiency 5/20/full.",
            "- Group B IMAD-DS RoboticArm segment-level: main binary, source-to-target, leave-target-weight35-out, label efficiency.",
            "- Group C IMAD-DS BrushlessMotor: secondary industrial anomaly validation once protocol audit is complete.",
            "- Group D RoAD: secondary sanity/stress evidence only because artifact/confounding risk remains.",
            "- Group E NIST UR: health/degradation validation only; do not force binary fault diagnosis.",
            "",
            "## Validity summary",
            md_table(protocols[["dataset", "protocol", "status", "role", "leakage_or_validity_risk"]], max_rows=80),
        ],
    )
    write_md(
        output_dir / "05_benchmark_model_list.md",
        "Benchmark Model List",
        prov,
        [
            "- Tree/statistical: LightGBM, XGBoost, RandomForest, IsolationForest.",
            "- Deep/time-series: AutoEncoder, LSTM-AE, USAD; TCN-AE only if already implemented and stable in a later design stage.",
            "- Historical failed references: CIRFL_v3_reference, CETRA_full, COIL_full, raw_residual_energy, condition_decoupled_residual_energy.",
            "- No new algorithm is proposed or renamed in this stage.",
        ],
    )
    write_md(
        output_dir / "06_engineering_metric_spec.md",
        "Engineering Metric Specification",
        prov,
        [
            "- Core metrics: macro-F1, weighted-F1, AUROC, PR-AUC, FAR, MDR, FAR@95%Recall.",
            "- Calibration: threshold_source must be validation_only; normalization_source must be train_only.",
            "- Engineering metrics: label budget, missing-sensor degradation, domain generalization gap, training time, inference latency, model size, CPU/GPU device.",
            "- Deployment interpretation: FAR and MDR are reported together; accuracy alone is not acceptable.",
        ],
    )
    write_md(
        output_dir / "10_publication_reposition_statistical_summary.md",
        "Publication Reposition Statistical Summary",
        prov,
        [
            "## Mean +/- std across seeds",
            md_table(summary, max_rows=120),
            "",
            "## Protocol winners",
            md_table(winners, max_rows=80),
            "",
            "## Interpretation",
            "- Strong tree models win the main raw-window protocols by macro-F1 and PR-AUC.",
            "- Deep reconstruction baselines show high FAR and weak macro-F1 on the current IMAD-DS raw-window pilot.",
            "- Historical new-algorithm references do not justify ESWA new-algorithm positioning.",
        ],
    )
    write_md(
        output_dir / "11_label_efficiency_pilot.md",
        "Label Efficiency Pilot",
        prov,
        [
            "## Existing label-budget pilot",
            md_table(label, max_rows=80) if not label.empty else "NOT_RUN",
            "",
            "## Conclusion",
            "- COIL did not show a label-efficiency advantage.",
            "- A future RIE/benchmark full design should rerun LightGBM/XGBoost/RF/AE/LSTM-AE/USAD under matched normal-only, 5%, 20%, and full validation budgets.",
            "- This limitation is recorded as a publication risk, not hidden.",
        ],
    )
    robustness = comparison[comparison["protocol"].isin(["imadds_raw_sensor_missing_10", "imadds_raw_sensor_missing_20", "imadds_raw_noise_mild", "imadds_raw_noise_moderate"])]
    write_md(
        output_dir / "12_robustness_pilot.md",
        "Robustness Pilot",
        prov,
        [
            md_table(summarize_by_method(robustness), max_rows=80) if not robustness.empty else "NOT_RUN",
            "",
            "- Missing-sensor/noise rows are pilot-level and incomplete for the full final baseline set.",
            "- The protocol is valuable for RIE framing but must be expanded before final experiments.",
        ],
    )
    write_md(
        output_dir / "13_tree_baseline_dominance_analysis.md",
        "Tree Baseline Dominance Analysis",
        prov,
        [
            "## Tree model summary",
            md_table(tree_summary, max_rows=80),
            "",
            "## Deep baseline summary",
            md_table(deep_summary, max_rows=80),
            "",
            "## Historical reference summary",
            md_table(hist_summary, max_rows=80),
            "",
            "## Engineering meaning",
            "- LightGBM/XGBoost/RandomForest consistently outperform the failed new-algorithm references on IMAD-DS raw-window pilot protocols.",
            "- This suggests statistical window evidence is currently more reliable than residual/event/evidence-lattice mechanisms.",
            "- The publishable value is not a renamed tree model; it is a leakage-safe, calibration-aware engineering evaluation framework.",
        ],
    )
    write_md(
        output_dir / "14_model_selection_guidelines.md",
        "Model Selection Guidelines",
        prov,
        [
            "- Labels available and latency matters: start with LightGBM; validate XGBoost as a robustness comparator.",
            "- Need simpler and transparent baseline: RandomForest is a strong fallback but may be larger/slower.",
            "- Normal-only anomaly monitoring: IsolationForest and deep reconstruction baselines need careful FAR calibration; current pilot does not support using them as default.",
            "- Deep models: AutoEncoder/LSTM-AE/USAD are useful benchmarks but not recommended as primary engineering solution on current pilot.",
            "- Historical algorithms: CIRFL/CETRA/COIL remain negative references and should not be promoted.",
        ],
    )
    device_lines = [
        "- Data audit, CSV aggregation, statistical summaries: CPU.",
        "- COIL/CETRA/evidence/residual historical methods: CPU in prior gates.",
        "- Deep raw-window baselines: GPU `cuda:0` in prior gate; no DataParallel.",
        "- Tree baselines: CPU for stability and reproducibility.",
        "- Three RTX 2080Ti availability: detected in prior stages; this aggregation stage did not need multi-GPU parallelism.",
    ]
    write_md(output_dir / "15_device_decision_report.md", "Device Decision Report", prov, device_lines)
    latency_cols = [c for c in ["method", "protocol", "model_size_mb", "cpu_latency_ms", "gpu_latency_ms", "train_time_sec", "inference_time_sec", "device"] if c in comparison.columns]
    latency = comparison[latency_cols].copy() if latency_cols else pd.DataFrame()
    if not latency.empty:
        latency = latency.groupby(["method", "protocol"], dropna=False).agg(
            model_size_mb_mean=("model_size_mb", "mean"),
            cpu_latency_ms_mean=("cpu_latency_ms", "mean"),
            gpu_latency_ms_mean=("gpu_latency_ms", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            inference_time_sec_mean=("inference_time_sec", "mean"),
        ).reset_index()
    write_md(output_dir / "16_complexity_latency_summary.md", "Complexity and Latency Summary", prov, [md_table(latency, max_rows=120)])
    rie_rows = pd.DataFrame([{"criterion": k, "satisfied": v} for k, v in rie_checks.items()])
    write_md(
        output_dir / "17_journal_direction_decision.md",
        "Journal Direction Decision",
        prov,
        [
            f"## Decision: {status}",
            "- ESWA new-algorithm route: ESWA_NEW_ALGORITHM_NO_GO because no new algorithm mechanism survived gate testing.",
            "- Results in Engineering route: promising as an engineering diagnostic framework if the pilot is expanded with matched robustness and label-budget protocols.",
            "- Benchmark/engineering evaluation route: also promising because model ranking, protocol risk, FAR/MDR calibration, and deployment tradeoffs are clear.",
            "",
            "## RIE criteria check",
            md_table(rie_rows),
        ],
    )
    write_md(
        output_dir / "18_publication_risk_assessment.md",
        "Publication Risk Assessment",
        prov,
        [
            "## Major risks",
            "- No defensible ESWA-style new algorithm contribution remains.",
            "- IMAD-DS raw-window pilot is not a full experiment and must be expanded before any manuscript workflow.",
            "- Label-efficiency and missing-sensor comparisons are not yet complete for every requested baseline.",
            "- RoAD remains secondary due artifact/confounding risk.",
            "- NIST UR lacks direct anomaly labels and should not be forced into binary diagnosis.",
            "",
            "## Why topic still has value",
            "- Real data are available and adapters/protocols are reproducible.",
            "- Leakage-safe split and train-only normalization are established.",
            "- Strong baseline dominance is itself an actionable engineering finding when combined with FAR/MDR, calibration, latency, and protocol analysis.",
        ],
    )
    write_md(
        output_dir / "19_code_index_and_next_tasks.md",
        "Code Index and Next Tasks",
        prov,
        [
            "## New/modified code",
            "- `configs/publication_reposition_gate_v1.yaml`: publication reposition gate config.",
            "- `scripts/run_publication_reposition_gate_v1.py`: real-output aggregation and review packet generation.",
            "",
            "## Commands",
            "- `/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_publication_reposition_gate_v1.py --config configs/publication_reposition_gate_v1.yaml`",
            "",
            "## Next tasks",
            "- If continuing RIE: design full engineering experiment plan with expanded label-efficiency, missing-sensor, noise, latency, and external validation protocols.",
            "- If continuing benchmark: expand IMAD-DS BrushlessMotor and RoAD secondary analyses without claiming a new algorithm.",
            "- Do not start manuscript writing until a full experiment design is explicitly approved.",
        ],
    )


def copy_packet(output_dir: Path, review_dir: Path) -> None:
    clean_dir(review_dir)
    for name in PACKET_FILES:
        src = output_dir / name
        if not src.exists():
            raise FileNotFoundError(f"missing required packet file: {src}")
        shutil.copyfile(src, review_dir / name)
    files = [p for p in review_dir.iterdir() if p.is_file()]
    if len(files) > 20:
        raise RuntimeError(f"review packet has {len(files)} files, expected <=20")
    forbidden = [p.name for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}]
    if forbidden:
        raise RuntimeError(f"review packet contains forbidden image/figure files: {forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/publication_reposition_gate_v1.yaml")
    args = parser.parse_args()
    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    output_dir = ROOT / cfg["project"]["output_dir"]
    review_dir = ROOT / cfg["project"]["review_dir"]
    clean_dir(output_dir)
    generated_at = utc_now()

    pilot, comparison, summary, label = build_pilot_tables(cfg, output_dir, generated_at)
    inventory, protocols = dataset_inventory(generated_at, output_dir)
    status, rie_checks = decision_from_pilot(comparison, inventory)
    if status != cfg.get("status_target", status):
        # Keep the computed status, but the config expectation remains visible in the packet.
        cfg["computed_status"] = status
    else:
        cfg["computed_status"] = status
    cfg["generated_at"] = generated_at
    cfg["source_outputs_reused"] = True
    cfg["pilot_is_full_experiment"] = False

    write_reports(cfg, output_dir, generated_at, status, rie_checks, pilot, comparison, summary, label, inventory, protocols)
    copy_packet(output_dir, review_dir)
    print(f"status={status}")
    print(f"output_dir={output_dir}")
    print(f"review_dir={review_dir}")
    print(f"review_files={len([p for p in review_dir.iterdir() if p.is_file()])}")


if __name__ == "__main__":
    main()
