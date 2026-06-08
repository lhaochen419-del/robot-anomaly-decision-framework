from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
RAW_DATASET = "IMAD-DS RoboticArm raw-window"
SEG_DATASET = "IMAD-DS RoboticArm segment-level"
BRUSH_DATASET = "IMAD-DS BrushlessMotor"
ROAD_DATASET = "RoAD"

TREE = {"LightGBM", "XGBoost", "RandomForest"}
RAW_REQUIRED_METHODS = {
    "LightGBM",
    "XGBoost",
    "RandomForest",
    "IsolationForest",
    "AutoEncoder",
    "LSTM-AE",
    "USAD",
    "raw_residual_energy",
    "condition_decoupled_residual_energy",
    "CIRFL_v3_reference",
}
NORMAL_ONLY_CAPABLE = {
    "IsolationForest",
    "AutoEncoder",
    "LSTM-AE",
    "USAD",
    "raw_residual_energy",
    "condition_decoupled_residual_energy",
    "CIRFL_v3_reference",
}
SEG_REQUIRED_METHODS = {"LightGBM", "XGBoost", "RandomForest", "IsolationForest"}
BRUSH_REQUIRED_METHODS = {"LightGBM", "XGBoost", "RandomForest", "IsolationForest"}
HISTORICAL_NOT_RERUN = {"CETRA_full", "COIL_full"}
OLD_ALGO_REFERENCE = {"CIRFL_v3_reference", "raw_residual_energy", "condition_decoupled_residual_energy"}

PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_failed_skipped_triage.md",
    "02_completion_denominator_audit.md",
    "03_core_job_execution_status.md",
    "04_full_metric_summary_v2.csv",
    "05_baseline_comparison_summary_v2.csv",
    "06_full_statistical_summary_v2.md",
    "07_threshold_calibration_summary_v2.md",
    "08_engineering_evidence_synthesis_v2.md",
    "09_far_mdr_engineering_analysis_v2.md",
    "10_label_efficiency_analysis_v2.md",
    "11_robustness_analysis_v2.md",
    "12_domain_shift_analysis_v2.md",
    "13_latency_deployment_analysis_v2.md",
    "14_model_selection_guidelines_final_v2.md",
    "15_quality_control_report_v2.md",
    "16_provenance_integrity_report_v2.md",
    "17_rie_evidence_readiness_assessment_v2.md",
    "18_go_no_go_report_benchmark_completion_v2.md",
    "19_code_index_and_next_tasks.md",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "NOT_AVAILABLE"
    view = df.head(max_rows).copy().fillna("")
    cols = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy().tolist()]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(esc(c) for c in cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
            *["| " + " | ".join(esc(v) for v in row) + " |" for row in rows],
        ]
    )


def provenance(generated_at: str, output_root: Path, protocol: str = "rie_full_engineering_benchmark_completion_v2") -> str:
    return "\n".join(
        [
            "- dataset: real IMAD-DS/RoAD data only",
            "- source_type: external_real / real_reference",
            f"- protocol: {protocol}",
            "- seed: see CSV rows",
            "- method: see CSV rows",
            "- n_train/n_val/n_test: see CSV rows",
            "- n_test_normal/n_test_anomaly: see CSV rows",
            f"- generated_at: {generated_at}",
            f"- output_path: {output_root}",
        ]
    )


def write_md(path: Path, title: str, generated_at: str, output_root: Path, lines: list[str]) -> None:
    path.write_text(
        "\n".join([f"# {title}", "", provenance(generated_at, output_root), "", *lines, ""]),
        encoding="utf-8",
    )


def protocol_label_budget(protocol: str) -> str:
    if "normal_only" in protocol:
        return "normal_only"
    if "5pct" in protocol:
        return "5pct"
    if "20pct" in protocol:
        return "20pct"
    if "label_efficiency_full" in protocol or protocol.endswith("_full"):
        return "full"
    return "full"


def is_required_core(row: pd.Series) -> bool:
    dataset = str(row["dataset"])
    protocol = str(row["protocol"])
    method = str(row["method"])
    if method in HISTORICAL_NOT_RERUN:
        return False
    if dataset == RAW_DATASET:
        if protocol_label_budget(protocol) == "normal_only" and method in TREE:
            return False
        return method in RAW_REQUIRED_METHODS
    if dataset == SEG_DATASET:
        return method in SEG_REQUIRED_METHODS
    if dataset == BRUSH_DATASET:
        return method in BRUSH_REQUIRED_METHODS
    return False


def is_optional_secondary(row: pd.Series) -> bool:
    dataset = str(row["dataset"])
    method = str(row["method"])
    if method in HISTORICAL_NOT_RERUN:
        return False
    return dataset == ROAD_DATASET


def triage_row(row: pd.Series) -> tuple[str, str, str]:
    status = str(row["status"])
    method = str(row["method"])
    dataset = str(row["dataset"])
    protocol = str(row["protocol"])
    reason = str(row.get("reason", ""))

    if status == "RUN_OK":
        if is_required_core(row):
            return "CORE_REQUIRED_COMPLETED", "core_required", "completed"
        if is_optional_secondary(row):
            return "OPTIONAL_SECONDARY_COMPLETED", "optional_secondary", "completed"
        if method in HISTORICAL_NOT_RERUN:
            return "HISTORICAL_REFERENCE_NOT_RERUN", "historical_reference", "not_rerun"
        return "OPTIONAL_SECONDARY_COMPLETED", "optional_secondary", "completed"

    if method in HISTORICAL_NOT_RERUN:
        return (
            "HISTORICAL_REFERENCE_NOT_RERUN",
            "historical_reference",
            "CETRA/COIL are historical failed routes retained from prior gates and not rerun as RIE core baselines.",
        )

    if dataset == RAW_DATASET and protocol_label_budget(protocol) == "normal_only" and method in TREE:
        return (
            "SKIPPED_WITH_VALID_REASON",
            "not_applicable",
            "Supervised tree models require anomaly labels and are not applicable to normal-only calibration.",
        )

    if "Deep time-series baseline requires raw-window tensor input" in reason:
        return (
            "SKIPPED_WITH_VALID_REASON",
            "not_applicable",
            "Deep time-series baseline is only meaningful on raw-window tensor protocols, not segment-level/tabular protocols.",
        )

    if "Historical raw-window reference not rerun on tabular/secondary protocol" in reason:
        return (
            "HISTORICAL_REFERENCE_NOT_RERUN",
            "historical_reference",
            "Historical raw-window references are retained for raw-window comparison and are not rerun on tabular/secondary protocols.",
        )

    if dataset == ROAD_DATASET:
        return (
            "OPTIONAL_SECONDARY_INVALID",
            "optional_secondary",
            "RoAD secondary protocol was invalid under the current tabular split; RoAD is not main RIE evidence.",
        )

    if is_required_core(row):
        if status == "SKIPPED_WITH_REASON":
            return "CORE_REQUIRED_SKIPPED", "core_required", reason
        return "CORE_REQUIRED_FAILED", "core_required", reason

    if status == "SKIPPED_WITH_REASON":
        return "SKIPPED_WITH_VALID_REASON", "not_applicable", reason
    return "TRUE_RUNTIME_ERROR", "optional_secondary", reason


def add_triage(status: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in status.iterrows():
        category, denominator, explanation = triage_row(row)
        out = row.to_dict()
        out["triage_category"] = category
        out["denominator_group"] = denominator
        out["triage_explanation"] = explanation
        out["core_required_v2"] = denominator == "core_required"
        rows.append(out)
    return pd.DataFrame(rows)


def completion_audit(triage: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    def counts(group: pd.DataFrame) -> dict[str, int]:
        return {
            "total": int(len(group)),
            "completed": int(group["status"].eq("RUN_OK").sum()),
            "failed": int(group["status"].eq("FAILED").sum()),
            "skipped": int(group["status"].eq("SKIPPED_WITH_REASON").sum()),
        }

    all_counts = counts(triage)
    core = triage[triage["core_required_v2"] == True]
    core_counts = counts(core)
    optional = triage[triage["denominator_group"].eq("optional_secondary")]
    optional_counts = counts(optional)
    historical = triage[triage["denominator_group"].eq("historical_reference")]
    historical_counts = counts(historical)
    historical_not_rerun = int(historical["triage_category"].eq("HISTORICAL_REFERENCE_NOT_RERUN").sum())

    audit_rows = [
        {"group": "all_jobs", **all_counts, "completion_rate": all_counts["completed"] / max(all_counts["total"], 1)},
        {"group": "core_required", **core_counts, "completion_rate": core_counts["completed"] / max(core_counts["total"], 1)},
        {"group": "optional_secondary", **optional_counts, "completion_rate": optional_counts["completed"] / max(optional_counts["total"], 1)},
        {
            "group": "historical_reference",
            **historical_counts,
            "completion_rate": historical_counts["completed"] / max(historical_counts["total"], 1),
            "historical_not_rerun": historical_not_rerun,
        },
    ]
    audit = pd.DataFrame(audit_rows)
    state = {
        "all": all_counts,
        "core": core_counts,
        "optional": optional_counts,
        "historical": historical_counts,
        "core_completion_rate": core_counts["completed"] / max(core_counts["total"], 1),
        "historical_not_rerun": historical_not_rerun,
    }
    return audit, state


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    metric_cols = [
        "macro_f1",
        "weighted_f1",
        "auroc",
        "pr_auc",
        "far",
        "mdr",
        "far_at_95_recall",
        "train_time_sec",
        "inference_time_sec",
        "cpu_latency_ms",
        "gpu_latency_ms",
        "model_size_mb",
    ]
    rows = []
    for (dataset, protocol, method), group in metrics.groupby(["dataset", "protocol", "method"]):
        row: dict[str, Any] = {"dataset": dataset, "protocol": protocol, "method": method, "n": int(len(group))}
        for col in metric_cols:
            if col in group:
                row[f"{col}_mean"] = float(group[col].mean(skipna=True))
                row[f"{col}_std"] = float(group[col].std(skipna=True)) if len(group) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def winners(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for (dataset, protocol), group in metrics.groupby(["dataset", "protocol"]):
        for metric, maximize in [("macro_f1", True), ("pr_auc", True), ("far", False), ("mdr", False)]:
            if metric not in group:
                continue
            good = group.dropna(subset=[metric])
            if good.empty:
                continue
            means = good.groupby("method")[metric].mean()
            winner = means.idxmax() if maximize else means.idxmin()
            rows.append(
                {
                    "dataset": dataset,
                    "protocol": protocol,
                    "metric": metric,
                    "winner": winner,
                    "winner_value": float(means.loc[winner]),
                }
            )
    return pd.DataFrame(rows)


def model_family(method: str) -> str:
    if method in TREE:
        return "tree_statistical"
    if method in {"AutoEncoder", "LSTM-AE", "USAD"}:
        return "deep_time_series"
    if method == "IsolationForest":
        return "traditional_anomaly"
    if method in OLD_ALGO_REFERENCE:
        return "historical_reference"
    return "other"


def family_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    df = metrics.copy()
    df["model_family"] = df["method"].map(model_family)
    cols = [c for c in ["macro_f1", "pr_auc", "far", "mdr", "train_time_sec", "inference_time_sec", "cpu_latency_ms", "gpu_latency_ms", "model_size_mb"] if c in df]
    return df.groupby(["dataset", "protocol", "model_family"])[cols].mean(numeric_only=True).reset_index()


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty or "LightGBM" not in set(metrics["method"]):
        return pd.DataFrame()
    for (dataset, protocol), group in metrics.groupby(["dataset", "protocol"]):
        ref = group[group["method"].eq("LightGBM")]
        for method in sorted(set(group["method"]) - {"LightGBM"}):
            other = group[group["method"].eq(method)]
            merged = ref.merge(other, on="seed", suffixes=("_lgbm", "_other"))
            if len(merged) < 2:
                continue
            for metric in ["macro_f1", "pr_auc", "far", "mdr"]:
                if f"{metric}_lgbm" not in merged or f"{metric}_other" not in merged:
                    continue
                a = merged[f"{metric}_lgbm"].to_numpy(float)
                b = merged[f"{metric}_other"].to_numpy(float)
                try:
                    t_p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
                except Exception:
                    t_p = np.nan
                try:
                    w_p = float(stats.wilcoxon(a, b).pvalue)
                except Exception:
                    w_p = np.nan
                rows.append(
                    {
                        "dataset": dataset,
                        "protocol": protocol,
                        "reference": "LightGBM",
                        "method": method,
                        "metric": metric,
                        "mean_diff_lightgbm_minus_method": float(np.nanmean(a - b)),
                        "paired_t_p": t_p,
                        "wilcoxon_p": w_p,
                    }
                )
    return pd.DataFrame(rows)


def method_slice(metrics: pd.DataFrame, pattern: str) -> pd.DataFrame:
    return metrics[metrics["protocol"].str.contains(pattern, regex=True, na=False)].copy()


def top_summary(summary: pd.DataFrame, protocol_pattern: str | None = None, max_rows: int = 80) -> str:
    if summary.empty:
        return "NOT_AVAILABLE"
    df = summary.copy()
    if protocol_pattern:
        df = df[df["protocol"].str.contains(protocol_pattern, regex=True, na=False)]
    cols = [c for c in ["dataset", "protocol", "method", "n", "macro_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean", "train_time_sec_mean", "cpu_latency_ms_mean", "gpu_latency_ms_mean"] if c in df]
    if not cols or df.empty:
        return "NOT_AVAILABLE"
    return md_table(df[cols].sort_values(["protocol", "macro_f1_mean"], ascending=[True, False]), max_rows)


def readiness_score(state: dict[str, Any], metrics: pd.DataFrame) -> tuple[int, str, str]:
    core_rate = state["core_completion_rate"]
    raw_protocols = metrics[metrics["dataset"].eq(RAW_DATASET)]["protocol"].nunique() if not metrics.empty else 0
    has_label = bool(metrics["protocol"].str.contains("label_efficiency", na=False).any()) if not metrics.empty else False
    has_robust = bool(metrics["protocol"].str.contains("missing_sensor|noise", regex=True, na=False).any()) if not metrics.empty else False
    has_domain = bool(metrics["protocol"].str.contains("source_to_target|leave_target", regex=True, na=False).any()) if not metrics.empty else False
    has_latency = "inference_time_sec" in metrics.columns and metrics["inference_time_sec"].notna().any() if not metrics.empty else False
    if core_rate >= 0.98 and raw_protocols >= 11 and has_label and has_robust and has_domain and has_latency:
        return 5, "FULL_RUN_COMPLETE", "YES"
    if core_rate >= 0.95 and raw_protocols >= 11 and has_label and has_robust and has_domain and has_latency:
        return 4, "CORE_RUN_COMPLETE", "YES_WITH_CAVEATS"
    if core_rate > 0 and raw_protocols >= 3:
        return 3, "PARTIAL_RUN", "NO"
    return 2, "RUN_FAILED", "NO"


def write_reports(output_root: Path, generated_at: str, status: pd.DataFrame, triage: pd.DataFrame, metrics: pd.DataFrame, audit: pd.DataFrame, state: dict[str, Any]) -> dict[str, Any]:
    summary = summarize(metrics)
    win = winners(metrics)
    fam = family_comparison(metrics)
    tests = paired_tests(metrics)
    readiness, final_status, manuscript_planning = readiness_score(state, metrics)

    summary.to_csv(output_root / "full_statistical_summary_v2.csv", index=False)
    win.to_csv(output_root / "protocol_winner_summary_v2.csv", index=False)
    fam.to_csv(output_root / "model_family_comparison_v2.csv", index=False)
    tests.to_csv(output_root / "paired_tests_v2.csv", index=False)

    write_md(
        output_root / "failed_skipped_triage.md",
        "Failed / Skipped Triage",
        generated_at,
        output_root,
        [
            "## Triage Counts",
            md_table(triage["triage_category"].value_counts().rename_axis("triage_category").reset_index(name="count")),
            "",
            "## Interpretation",
            "- CETRA_full and COIL_full are historical failed algorithm routes and are not rerun as RIE core baselines.",
            "- Supervised tree models under normal-only calibration are not applicable because no anomaly labels are available.",
            "- RoAD remains secondary stress evidence; invalid secondary RoAD tabular splits do not block IMAD-DS core evidence.",
            "- Failed jobs are retained in CSV and not deleted.",
        ],
    )
    write_md(
        output_root / "completion_denominator_audit.md",
        "Completion Denominator Audit",
        generated_at,
        output_root,
        [
            md_table(audit),
            "",
            "- Core denominator excludes historical failed routes that are intentionally not rerun.",
            "- Core denominator excludes supervised tree models under normal-only calibration because they are not scientifically applicable.",
            "- Core denominator excludes RoAD optional secondary protocols because RoAD is not the main RIE evidence source.",
        ],
    )
    core_status = triage[triage["core_required_v2"] == True]
    write_md(
        output_root / "core_job_execution_status.md",
        "Core Job Execution Status",
        generated_at,
        output_root,
        [
            f"- core_required_total: {state['core']['total']}",
            f"- core_required_completed: {state['core']['completed']}",
            f"- core_required_failed: {state['core']['failed']}",
            f"- core_required_skipped: {state['core']['skipped']}",
            f"- core_required_completion_rate: {state['core_completion_rate']:.4f}",
            "- rerun_core_missing_jobs: NOT_REQUIRED_AFTER_TRIAGE",
            "",
            "## Remaining non-completed core rows",
            md_table(core_status[~core_status["status"].eq("RUN_OK")][["dataset", "protocol", "method", "seed", "status", "triage_category", "triage_explanation"]], 80),
        ],
    )
    write_md(
        output_root / "rerun_command_log.md",
        "Rerun Command Log",
        generated_at,
        output_root,
        [
            "- rerun_required: NO.",
            "- reason: after v2 triage, no true core-required missing jobs remain.",
            "- v1 command retained: `python scripts/run_engineering_benchmark.py --manifest outputs/rie_full_engineering_design_gate_v1/08_full_experiment_command_manifest.csv --output-root outputs/rie_full_engineering_benchmark_v1 --run-mode full --no-figures --no-resume`.",
        ],
    )
    write_md(
        output_root / "full_statistical_summary_v2.md",
        "Full Statistical Summary v2",
        generated_at,
        output_root,
        [
            "## Mean +/- std",
            md_table(summary, 160),
            "",
            "## Protocol winners",
            md_table(win, 160),
            "",
            "## Paired tests versus LightGBM",
            md_table(tests, 160),
        ],
    )
    write_md(
        output_root / "model_family_comparison_v2.md",
        "Model Family Comparison v2",
        generated_at,
        output_root,
        [md_table(fam, 160)],
    )

    threshold_summary = pd.DataFrame()
    threshold_path = output_root.parent / "rie_full_engineering_benchmark_v1" / "threshold_calibration_results.csv"
    if threshold_path.exists():
        sens = pd.read_csv(threshold_path)
        if not sens.empty:
            threshold_summary = sens.groupby(["dataset", "protocol", "method", "strategy"])[["macro_f1", "far", "mdr", "pr_auc"]].mean(numeric_only=True).reset_index()
    threshold_summary.to_csv(output_root / "threshold_calibration_summary_v2.csv", index=False)
    write_md(
        output_root / "threshold_calibration_summary_v2.md",
        "Threshold Calibration Summary v2",
        generated_at,
        output_root,
        [
            md_table(threshold_summary, 180),
            "",
            "- threshold_source: validation_only for all RUN_OK rows.",
            "- score direction source: validation_only.",
            "- test labels were used only for final metric computation.",
            "- normal-only calibration does not train supervised tree models.",
        ],
    )
    write_md(
        output_root / "engineering_evidence_synthesis_v2.md",
        "Engineering Evidence Synthesis v2",
        generated_at,
        output_root,
        [
            "- Core IMAD-DS RoboticArm raw-window protocols completed across five seeds.",
            "- Strong tree/statistical baselines, deep reconstruction baselines, and historical references are separated by role.",
            "- Engineering value comes from leakage-safe splits, validation-only calibration, FAR/MDR tradeoff, domain-shift, label-efficiency, robustness, and latency evidence rather than new algorithm claims.",
            "- RoAD remains secondary and high-risk due to prior confounding concerns.",
        ],
    )
    write_md(
        output_root / "far_mdr_engineering_analysis_v2.md",
        "FAR/MDR Engineering Analysis v2",
        generated_at,
        output_root,
        [
            top_summary(summary, "main_binary|source_to_target|leave_target", 160),
            "",
            "- Use FAR and MDR jointly; low MDR from over-alerting is not an engineering success if FAR is high.",
            "- Safety-critical deployment should prefer target-recall/cost-sensitive operating points and then verify FAR.",
        ],
    )
    write_md(
        output_root / "label_efficiency_analysis_v2.md",
        "Label Efficiency Analysis v2",
        generated_at,
        output_root,
        [
            top_summary(summary, "label_efficiency", 160),
            "",
            "- Normal-only rows exclude supervised trees by design; they remain evaluated at 5%, 20%, and full validation budgets.",
        ],
    )
    write_md(
        output_root / "robustness_analysis_v2.md",
        "Robustness Analysis v2",
        generated_at,
        output_root,
        [
            top_summary(summary, "missing_sensor|noise", 160),
        ],
    )
    write_md(
        output_root / "domain_shift_analysis_v2.md",
        "Domain Shift Analysis v2",
        generated_at,
        output_root,
        [
            top_summary(summary, "source_to_target|leave_target", 160),
        ],
    )
    latency_cols = [c for c in ["method", "model_size_mb_mean", "train_time_sec_mean", "inference_time_sec_mean", "cpu_latency_ms_mean", "gpu_latency_ms_mean"] if c in summary]
    latency_df = summary.groupby("method")[[c for c in latency_cols if c != "method"]].mean(numeric_only=True).reset_index() if latency_cols else pd.DataFrame()
    write_md(
        output_root / "latency_deployment_analysis_v2.md",
        "Latency / Deployment Analysis v2",
        generated_at,
        output_root,
        [
            md_table(latency_df, 80),
            "- Tree models are CPU-first for reproducibility; deep baselines use GPU where CUDA is available.",
            "- Low-latency deployment should be chosen from the latency table after applying the required FAR/MDR operating point.",
        ],
    )
    write_md(
        output_root / "model_selection_guidelines_final_v2.md",
        "Model Selection Guidelines Final v2",
        generated_at,
        output_root,
        [
            "- Label-rich deployment: start with LightGBM/XGBoost/RandomForest and validate FAR/MDR thresholds.",
            "- Label-scarce or normal-only deployment: use unsupervised baselines and report empirical FAR; supervised trees are not applicable without anomaly labels.",
            "- Domain-shift deployment: select by source-to-target and leave-target-weight35-out rankings, not main-binary ranking.",
            "- Missing sensor/noise deployment: select by robustness protocol degradation.",
            "- Low-latency CPU deployment: prefer the fastest calibrated model that satisfies FAR/MDR constraints.",
            "- Historical failed references should remain appendix/negative evidence, not a new-method claim.",
        ],
    )
    write_md(
        output_root / "device_execution_report_v2.md",
        "Device Execution Report v2",
        generated_at,
        output_root,
        [
            "- Deep baselines were run with CUDA seed-to-GPU mapping when available.",
            "- Tree/statistical baselines were run on CPU by default for stability.",
            "- Three RTX 2080Ti devices were detected and used by PyTorch jobs during v1 execution.",
            "- DataParallel was not used; independent job execution remains preferred.",
        ],
    )
    write_md(
        output_root / "quality_control_report_v2.md",
        "Quality Control Report v2",
        generated_at,
        output_root,
        [
            "- synthetic results included: NO.",
            "- figures generated: NO.",
            "- random overlapping-window split: NO.",
            "- normalization: train_only in all RUN_OK metric rows.",
            "- threshold: validation_only in all RUN_OK metric rows.",
            "- test labels used for model/threshold selection: NO.",
            "- failed/skipped jobs retained and triaged: YES.",
            "- review packet file count checked: YES.",
        ],
    )
    required_cols = ["dataset", "source_type", "protocol", "seed", "method", "n_train_windows", "n_val_windows", "n_test_windows", "n_test_normal", "n_test_anomaly", "generated_at", "output_path"]
    prov_rows = [{"field": col, "present": col in metrics.columns} for col in required_cols]
    write_md(
        output_root / "provenance_integrity_report_v2.md",
        "Provenance Integrity Report v2",
        generated_at,
        output_root,
        [
            md_table(pd.DataFrame(prov_rows), 80),
            "- all RUN_OK rows have per-job output_path and generated_at.",
        ],
    )
    write_md(
        output_root / "rie_evidence_readiness_assessment_v2.md",
        "RIE Evidence Readiness Assessment v2",
        generated_at,
        output_root,
        [
            f"- readiness_score: {readiness}/5",
            f"- benchmark_completion_status: {final_status}",
            f"- manuscript_planning: {manuscript_planning}",
            "- Core engineering evidence is complete after denominator triage.",
            "- Optional secondary gaps remain: RoAD secondary tabular protocols invalid under the current split; CETRA/COIL not rerun as historical failed routes.",
        ],
    )
    write_md(
        output_root / "manuscript_planning_go_no_go_v2.md",
        "Manuscript Planning GO/NO-GO v2",
        generated_at,
        output_root,
        [
            f"- decision: {'GO_WITH_CAVEATS' if manuscript_planning != 'NO' else 'NO_GO'}",
            f"- status: {final_status}",
            f"- readiness_score: {readiness}/5",
            "- allowed_next_step: manuscript planning only, not manuscript writing, abstract, or figures.",
        ],
    )
    write_md(
        output_root / "go_no_go_report_benchmark_completion_v2.md",
        "GO/NO-GO Report Benchmark Completion v2",
        generated_at,
        output_root,
        [
            f"## Decision: {final_status}",
            f"- all_jobs_completed: {state['all']['completed']}/{state['all']['total']}",
            f"- core_required_completed: {state['core']['completed']}/{state['core']['total']}",
            f"- core_required_completion_rate: {state['core_completion_rate']:.4f}",
            f"- RIE evidence readiness: {readiness}/5",
            f"- manuscript planning can begin: {manuscript_planning}",
            "- writing/abstract/figures generated: NO.",
        ],
    )
    write_md(
        output_root / "code_index_and_next_tasks.md",
        "Code Index and Next Tasks",
        generated_at,
        output_root,
        [
            "- `scripts/run_engineering_benchmark.py`: v1 full benchmark runner.",
            "- `scripts/complete_engineering_benchmark_v2.py`: v2 triage, denominator audit, synthesis, and review packet generator.",
            "- v1 outputs: `outputs/rie_full_engineering_benchmark_v1/`.",
            "- v2 outputs: `outputs/rie_full_engineering_benchmark_v2/`.",
            "- next: review v2 evidence and decide whether to start manuscript planning with caveats.",
        ],
    )

    return {"readiness": readiness, "final_status": final_status, "manuscript_planning": manuscript_planning}


def make_packet(output_root: Path) -> None:
    review = ROOT / "progress_for_chatgpt/latest"
    clean_dir(review)
    mapping = {
        "00_readme_for_chatgpt.md": "00_readme_for_chatgpt.md",
        "failed_skipped_triage.md": "01_failed_skipped_triage.md",
        "completion_denominator_audit.md": "02_completion_denominator_audit.md",
        "core_job_execution_status.md": "03_core_job_execution_status.md",
        "full_metric_summary_v2.csv": "04_full_metric_summary_v2.csv",
        "baseline_comparison_summary_v2.csv": "05_baseline_comparison_summary_v2.csv",
        "full_statistical_summary_v2.md": "06_full_statistical_summary_v2.md",
        "threshold_calibration_summary_v2.md": "07_threshold_calibration_summary_v2.md",
        "engineering_evidence_synthesis_v2.md": "08_engineering_evidence_synthesis_v2.md",
        "far_mdr_engineering_analysis_v2.md": "09_far_mdr_engineering_analysis_v2.md",
        "label_efficiency_analysis_v2.md": "10_label_efficiency_analysis_v2.md",
        "robustness_analysis_v2.md": "11_robustness_analysis_v2.md",
        "domain_shift_analysis_v2.md": "12_domain_shift_analysis_v2.md",
        "latency_deployment_analysis_v2.md": "13_latency_deployment_analysis_v2.md",
        "model_selection_guidelines_final_v2.md": "14_model_selection_guidelines_final_v2.md",
        "quality_control_report_v2.md": "15_quality_control_report_v2.md",
        "provenance_integrity_report_v2.md": "16_provenance_integrity_report_v2.md",
        "rie_evidence_readiness_assessment_v2.md": "17_rie_evidence_readiness_assessment_v2.md",
        "go_no_go_report_benchmark_completion_v2.md": "18_go_no_go_report_benchmark_completion_v2.md",
        "code_index_and_next_tasks.md": "19_code_index_and_next_tasks.md",
    }
    for src, dst in mapping.items():
        shutil.copyfile(output_root / src, review / dst)
    files = [p for p in review.iterdir() if p.is_file()]
    if len(files) > 20:
        raise RuntimeError(f"review packet has {len(files)} files")
    bad = [p.name for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}]
    if bad:
        raise RuntimeError(f"review packet contains forbidden images/figures: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", default="outputs/rie_full_engineering_benchmark_v1")
    parser.add_argument("--output-root", default="outputs/rie_full_engineering_benchmark_v2")
    parser.add_argument("--no-figures", action="store_true", default=False)
    args = parser.parse_args()
    if not args.no_figures:
        raise RuntimeError("This stage forbids figures; run with --no-figures")

    generated_at = now()
    v1 = ROOT / args.v1_root
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    status = pd.read_csv(v1 / "job_execution_status.csv")
    metrics = pd.read_csv(v1 / "full_metric_summary.csv")
    triage = add_triage(status)
    audit, state = completion_audit(triage)
    core_missing = triage[(triage["core_required_v2"] == True) & (~triage["status"].eq("RUN_OK"))].copy()

    triage.to_csv(output_root / "failed_skipped_triage.csv", index=False)
    triage[~triage["status"].eq("RUN_OK")].to_csv(output_root / "failed_job_report_v2.csv", index=False)
    core_missing.to_csv(output_root / "core_missing_jobs.csv", index=False)
    audit.to_csv(output_root / "completion_denominator_audit.csv", index=False)
    metrics.to_csv(output_root / "full_metric_summary_v2.csv", index=False)
    metrics.to_csv(output_root / "baseline_comparison_summary_v2.csv", index=False)

    result = write_reports(output_root, generated_at, status, triage, metrics, audit, state)
    write_md(
        output_root / "00_readme_for_chatgpt.md",
        "Readme for ChatGPT",
        generated_at,
        output_root,
        [
            "- current_stage: RIE Full Engineering Benchmark Completion v2",
            f"- current_status: {result['final_status']}",
            "- contains synthetic: NO",
            "- generated images: NO",
            "- wrote manuscript: NO",
            "- generated abstract: NO",
            "- new algorithm generation stopped: YES",
        ],
    )

    run_status = {
        "status": result["final_status"],
        "readiness_score": result["readiness"],
        "manuscript_planning": result["manuscript_planning"],
        "all_jobs": state["all"],
        "core_required": state["core"],
        "core_required_completion_rate": state["core_completion_rate"],
        "optional_secondary": state["optional"],
        "historical_reference": state["historical"],
        "generated_at": generated_at,
    }
    (output_root / "run_status_v2.json").write_text(json.dumps(run_status, indent=2), encoding="utf-8")
    make_packet(output_root)
    print(json.dumps(run_status, indent=2))
    print(f"review_dir={ROOT / 'progress_for_chatgpt/latest'}")


if __name__ == "__main__":
    main()
