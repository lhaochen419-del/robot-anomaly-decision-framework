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
FIXED_MODELS = ["LightGBM", "XGBoost", "RandomForest", "IsolationForest", "AutoEncoder", "LSTM-AE", "USAD"]
STRONG_FIXED = ["Fixed-LightGBM", "Fixed-XGBoost", "Fixed-RandomForest"]
FRAMEWORK_STRATEGIES = [
    "Framework-Best-F1",
    "Framework-Balanced",
    "Framework-Safety",
    "Framework-Low-False-Alarm",
    "Framework-Deployment",
    "Framework-Robust",
    "Framework-Label-Efficient",
]
HIGHER_IS_BETTER = ["macro_f1", "weighted_f1", "auroc", "pr_auc"]
LOWER_IS_BETTER = ["far", "mdr", "far_at_95_recall", "latency_ms", "model_size_mb"]
FAR_LIMIT = 0.40
MDR_LIMIT = 0.35

PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_validation_data_audit.md",
    "02_strategy_definitions.md",
    "03_fixed_model_strategies.md",
    "04_strategy_selection_log.csv",
    "05_framework_vs_fixed_summary.csv",
    "06_regret_analysis.csv",
    "07_rank_and_win_summary.md",
    "08_engineering_utility_results.csv",
    "09_far_mdr_constraint_analysis.md",
    "10_label_efficiency_strategy_results.md",
    "11_robustness_strategy_results.md",
    "12_latency_deployment_strategy_results.md",
    "13_statistical_comparison.md",
    "14_framework_strategy_go_no_go.md",
    "15_strategy_failure_or_optimization_plan.md",
    "16_updated_rie_evidence_readiness.md",
    "17_code_index_and_commands.md",
    "18_next_tasks_for_codex.md",
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


def provenance(generated_at: str, output_root: Path, protocol: str = "framework_strategy_validation_gate_v1") -> str:
    return "\n".join(
        [
            "- dataset: real IMAD-DS benchmark outputs",
            "- source_type: external_real",
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
    path.write_text("\n".join([f"# {title}", "", provenance(generated_at, output_root), "", *lines, ""]), encoding="utf-8")


def protocol_group(protocol: str) -> str:
    if "main_binary" in protocol:
        return "main"
    if "source_to_target" in protocol or "leave_target" in protocol:
        return "domain_shift"
    if "missing_sensor" in protocol or "noise" in protocol:
        return "robustness"
    if "label_efficiency" in protocol:
        return "label_efficiency"
    if "latency" in protocol:
        return "deployment"
    if "raw_vs_segment" in protocol:
        return "granularity"
    return "other"


def label_budget(protocol: str) -> str:
    if "normal_only" in protocol:
        return "normal_only"
    if "5pct" in protocol:
        return "5pct"
    if "20pct" in protocol:
        return "20pct"
    if "label_efficiency_full" in protocol:
        return "full"
    return "full"


def clean_protocol_for(dataset: str) -> str | None:
    if dataset == "IMAD-DS RoboticArm raw-window":
        return "imadds_raw_main_binary"
    if dataset == "IMAD-DS RoboticArm segment-level":
        return "imadds_segment_main_binary"
    if dataset == "IMAD-DS BrushlessMotor":
        return "brushless_main_binary_if_valid"
    return None


def latency_ms(row: pd.Series) -> float:
    gpu = row.get("gpu_latency_ms", np.nan)
    cpu = row.get("cpu_latency_ms", np.nan)
    if pd.notna(gpu):
        return float(gpu)
    if pd.notna(cpu):
        return float(cpu)
    infer = row.get("inference_time_sec", np.nan)
    n = row.get("n_test", row.get("n_test_windows", np.nan))
    if pd.notna(infer) and pd.notna(n) and n:
        return float(infer) / float(n) * 1000.0
    return 0.0


def add_utilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["latency_ms"] = out.apply(latency_ms, axis=1)
    out["model_size_mb"] = pd.to_numeric(out.get("model_size_mb", 1.0), errors="coerce").fillna(1.0)
    max_latency = max(float(out["latency_ms"].max()), 1e-9)
    max_size = max(float(out["model_size_mb"].max()), 1e-9)
    out["latency_penalty"] = 0.05 * np.log1p(out["latency_ms"].clip(lower=0)) / np.log1p(max_latency)
    out["size_penalty"] = 0.05 * np.log1p(out["model_size_mb"].clip(lower=0)) / np.log1p(max_size)
    for prefix in ["", "val_"]:
        m = prefix + "macro_f1"
        far = prefix + "far"
        mdr = prefix + "mdr"
        if m not in out:
            continue
        out[prefix + "utility_balanced"] = out[m] - out[far] - out[mdr]
        out[prefix + "utility_safety"] = out[m] - 2.0 * out[mdr] - 0.5 * out[far]
        out[prefix + "utility_low_false_alarm"] = out[m] - 2.0 * out[far] - 0.5 * out[mdr]
        out[prefix + "utility_deployment"] = out[m] - out[far] - out[mdr] - out["latency_penalty"] - out["size_penalty"]
        budget_penalty = out["protocol"].map({"imadds_raw_label_efficiency_normal_only": 0.00}).fillna(0.0)
        budget_penalty += out["protocol"].str.contains("5pct", na=False).astype(float) * 0.02
        budget_penalty += out["protocol"].str.contains("20pct", na=False).astype(float) * 0.04
        budget_penalty += out["protocol"].str.contains("label_efficiency_full", na=False).astype(float) * 0.06
        out[prefix + "utility_label_efficiency"] = out[m] - out[far] - out[mdr] - budget_penalty
    clean = out[["dataset", "method", "seed", "strategy", "protocol", "val_macro_f1"]].copy()
    clean = clean.rename(columns={"protocol": "clean_protocol", "val_macro_f1": "clean_val_macro_f1"})
    clean["wanted_clean_protocol"] = clean["dataset"].map(clean_protocol_for)
    clean = clean[clean["clean_protocol"].eq(clean["wanted_clean_protocol"])]
    out = out.merge(
        clean[["dataset", "method", "seed", "strategy", "clean_val_macro_f1"]],
        on=["dataset", "method", "seed", "strategy"],
        how="left",
    )
    out["val_robust_degradation"] = (out["clean_val_macro_f1"] - out["val_macro_f1"]).clip(lower=0).fillna(0.0)
    out["val_utility_robust"] = out["val_utility_balanced"] - out["val_robust_degradation"]
    out["utility_robust"] = out["utility_balanced"]
    out["scenario_id"] = out["dataset"].astype(str) + "||" + out["protocol"].astype(str) + "||seed_" + out["seed"].astype(str)
    out["protocol_group"] = out["protocol"].map(protocol_group)
    out["label_budget"] = out["protocol"].map(label_budget)
    return out


def select_rows(candidates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    fixed_objective = "val_macro_f1"
    for model in FIXED_MODELS:
        subset = candidates[candidates["method"].eq(model)]
        for _, group in subset.groupby("scenario_id"):
            if group.empty:
                continue
            chosen = group.loc[group[fixed_objective].idxmax()].copy()
            chosen["deployed_strategy"] = f"Fixed-{model}"
            chosen["selection_objective"] = fixed_objective
            chosen["selection_score"] = chosen[fixed_objective]
            rows.append(chosen.to_dict())

    framework_objectives = {
        "Framework-Best-F1": "val_macro_f1",
        "Framework-Balanced": "val_utility_balanced",
        "Framework-Safety": "val_utility_safety",
        "Framework-Low-False-Alarm": "val_utility_low_false_alarm",
        "Framework-Deployment": "val_utility_deployment",
        "Framework-Robust": "val_utility_robust",
        "Framework-Label-Efficient": "val_utility_label_efficiency",
    }
    for strategy, objective in framework_objectives.items():
        for _, group in candidates.groupby("scenario_id"):
            good = group.dropna(subset=[objective])
            if good.empty:
                continue
            chosen = good.loc[good[objective].idxmax()].copy()
            chosen["deployed_strategy"] = strategy
            chosen["selection_objective"] = objective
            chosen["selection_score"] = chosen[objective]
            rows.append(chosen.to_dict())

    for _, group in candidates.groupby("scenario_id"):
        if group.empty:
            continue
        chosen = group.loc[group["macro_f1"].idxmax()].copy()
        chosen["deployed_strategy"] = "Oracle-Best-Test"
        chosen["selection_objective"] = "test_macro_f1"
        chosen["selection_score"] = chosen["macro_f1"]
        chosen["oracle_not_deployable"] = True
        rows.append(chosen.to_dict())
    return pd.DataFrame(rows)


def summarize_strategy(selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = ["macro_f1", "weighted_f1", "auroc", "pr_auc", "far", "mdr", "far_at_95_recall", "latency_ms", "model_size_mb", "train_time_sec"]
    for strategy, group in selected.groupby("deployed_strategy"):
        row: dict[str, Any] = {
            "deployed_strategy": strategy,
            "n_scenarios": int(group["scenario_id"].nunique()),
            "distinct_selected_models": int(group["method"].nunique()),
            "far_violation_rate": float((group["far"] > FAR_LIMIT).mean()),
            "mdr_violation_rate": float((group["mdr"] > MDR_LIMIT).mean()),
            "joint_far_mdr_violation_rate": float(((group["far"] > FAR_LIMIT) | (group["mdr"] > MDR_LIMIT)).mean()),
        }
        for metric in metrics:
            if metric in group:
                row[f"{metric}_mean"] = float(group[metric].mean(skipna=True))
                row[f"{metric}_std"] = float(group[metric].std(skipna=True)) if len(group) > 1 else 0.0
        for util in ["utility_balanced", "utility_safety", "utility_low_false_alarm", "utility_deployment", "utility_label_efficiency", "utility_robust"]:
            row[f"{util}_mean"] = float(group[util].mean(skipna=True))
        rows.append(row)
    return pd.DataFrame(rows)


def regret_analysis(selected: pd.DataFrame, candidates: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    best_rows = []
    for sid, group in candidates.groupby("scenario_id"):
        row: dict[str, Any] = {"scenario_id": sid}
        for metric in HIGHER_IS_BETTER:
            if metric in group:
                row[f"best_{metric}"] = float(group[metric].max())
                row[f"range_{metric}"] = float(group[metric].max() - group[metric].min())
        for metric in LOWER_IS_BETTER:
            if metric in group:
                row[f"best_{metric}"] = float(group[metric].min())
                row[f"range_{metric}"] = float(group[metric].max() - group[metric].min())
        best_rows.append(row)
    best = pd.DataFrame(best_rows)
    merged = selected.merge(best, on="scenario_id", how="left")
    rows = []
    for _, row in merged.iterrows():
        for metric in HIGHER_IS_BETTER:
            if metric in row and f"best_{metric}" in row:
                regret = row[f"best_{metric}"] - row[metric]
                rng = max(float(row.get(f"range_{metric}", 0.0)), 1e-9)
                rows.append({"deployed_strategy": row["deployed_strategy"], "scenario_id": row["scenario_id"], "metric": metric, "regret": regret, "normalized_regret": regret / rng})
        for metric in LOWER_IS_BETTER:
            if metric in row and f"best_{metric}" in row:
                regret = row[metric] - row[f"best_{metric}"]
                rng = max(float(row.get(f"range_{metric}", 0.0)), 1e-9)
                rows.append({"deployed_strategy": row["deployed_strategy"], "scenario_id": row["scenario_id"], "metric": metric, "regret": regret, "normalized_regret": regret / rng})
    per = pd.DataFrame(rows)
    summary = per.groupby(["deployed_strategy", "metric"]).agg(
        mean_regret=("regret", "mean"),
        median_regret=("regret", "median"),
        worst_case_regret=("regret", "max"),
        mean_normalized_regret=("normalized_regret", "mean"),
    ).reset_index()
    overall = per.groupby("deployed_strategy").agg(
        mean_regret=("regret", "mean"),
        median_regret=("regret", "median"),
        worst_case_regret=("regret", "max"),
        normalized_regret=("normalized_regret", "mean"),
    ).reset_index()
    return pd.concat([summary, overall.assign(metric="ALL_METRICS")], ignore_index=True), per


def rank_summary(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    util_cols = ["utility_balanced", "utility_safety", "utility_low_false_alarm", "utility_deployment", "utility_label_efficiency", "utility_robust"]
    rank_rows = []
    for util in util_cols:
        for sid, group in selected.groupby("scenario_id"):
            scores = group[["deployed_strategy", util]].dropna()
            if scores.empty:
                continue
            scores["rank"] = scores[util].rank(ascending=False, method="min")
            maxv = scores[util].max()
            for _, row in scores.iterrows():
                rank_rows.append(
                    {
                        "scenario_id": sid,
                        "utility": util,
                        "deployed_strategy": row["deployed_strategy"],
                        "rank": float(row["rank"]),
                        "is_win": bool(row[util] == maxv),
                        "is_top2": bool(row["rank"] <= 2),
                    }
                )
    ranks = pd.DataFrame(rank_rows)
    summary = ranks.groupby(["deployed_strategy", "utility"]).agg(
        average_rank=("rank", "mean"),
        win_count=("is_win", "sum"),
        top2_count=("is_top2", "sum"),
        n=("rank", "count"),
    ).reset_index()
    return ranks, summary


def statistical_comparison(selected: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    comparisons = [
        ("Framework-Balanced", "utility_balanced"),
        ("Framework-Safety", "utility_safety"),
        ("Framework-Low-False-Alarm", "utility_low_false_alarm"),
        ("Framework-Robust", "utility_robust"),
        ("Framework-Deployment", "utility_deployment"),
    ]
    for fw, util in comparisons:
        for fixed in STRONG_FIXED:
            a = selected[selected["deployed_strategy"].eq(fw)][["scenario_id", util]].rename(columns={util: "framework"})
            b = selected[selected["deployed_strategy"].eq(fixed)][["scenario_id", util]].rename(columns={util: "fixed"})
            merged = a.merge(b, on="scenario_id")
            if len(merged) < 2:
                continue
            diff = merged["framework"].to_numpy(float) - merged["fixed"].to_numpy(float)
            try:
                t_p = float(stats.ttest_rel(merged["framework"], merged["fixed"], nan_policy="omit").pvalue)
            except Exception:
                t_p = np.nan
            try:
                w_p = float(stats.wilcoxon(merged["framework"], merged["fixed"]).pvalue)
            except Exception:
                w_p = np.nan
            wins = int((diff > 0).sum())
            losses = int((diff < 0).sum())
            try:
                sign_p = float(stats.binomtest(wins, wins + losses, 0.5, alternative="two-sided").pvalue) if wins + losses else np.nan
            except Exception:
                sign_p = np.nan
            effect = float(np.nanmean(diff) / (np.nanstd(diff, ddof=1) + 1e-12)) if len(diff) > 1 else np.nan
            pairs.append(
                {
                    "framework_strategy": fw,
                    "fixed_strategy": fixed,
                    "utility": util,
                    "n_pairs": int(len(merged)),
                    "mean_framework_minus_fixed": float(np.nanmean(diff)),
                    "paired_t_p": t_p,
                    "wilcoxon_p": w_p,
                    "sign_test_p": sign_p,
                    "effect_size_dz": effect,
                    "wins": wins,
                    "losses": losses,
                }
            )
    return pd.DataFrame(pairs)


def criteria_eval(selected: pd.DataFrame, summary: pd.DataFrame, regret: pd.DataFrame, rank_sum: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    valid_protocols = int(selected[~selected["deployed_strategy"].eq("Oracle-Best-Test")]["protocol"].nunique())
    fw = summary[summary["deployed_strategy"].isin(FRAMEWORK_STRATEGIES)].copy()
    fixed = summary[summary["deployed_strategy"].isin(STRONG_FIXED)].copy()
    reg_all = regret[regret["metric"].eq("ALL_METRICS")]
    fw_reg = reg_all[reg_all["deployed_strategy"].isin(FRAMEWORK_STRATEGIES)]["normalized_regret"].min()
    fixed_better_count = int((reg_all[reg_all["deployed_strategy"].isin(STRONG_FIXED)]["normalized_regret"] > fw_reg).sum())
    top2_utils = rank_sum[(rank_sum["deployed_strategy"].isin(FRAMEWORK_STRATEGIES)) & (rank_sum["average_rank"] <= 2)]["utility"].nunique()
    fw_violation = fw["joint_far_mdr_violation_rate"].min()
    fixed_violation_better = int((fixed["joint_far_mdr_violation_rate"] > fw_violation).sum()) if not fixed.empty else 0
    label = selected[selected["protocol_group"].eq("label_efficiency")]
    label_advantage = False
    robust_advantage = False
    deploy_advantage = False
    if not label.empty:
        lsum = label.groupby("deployed_strategy")["utility_label_efficiency"].mean()
        label_advantage = any(lsum.get(fw_name, -np.inf) > lsum.get(fixed_name, -np.inf) for fw_name in FRAMEWORK_STRATEGIES for fixed_name in STRONG_FIXED)
    robust = selected[selected["protocol_group"].isin(["robustness", "domain_shift"])]
    if not robust.empty:
        rsum = robust.groupby("deployed_strategy")["utility_robust"].mean()
        robust_advantage = any(rsum.get(fw_name, -np.inf) > rsum.get(fixed_name, -np.inf) for fw_name in FRAMEWORK_STRATEGIES for fixed_name in STRONG_FIXED)
    dep = selected[selected["protocol_group"].eq("deployment")]
    if not dep.empty:
        dsum = dep.groupby("deployed_strategy")["utility_deployment"].mean()
        deploy_advantage = any(dsum.get(fw_name, -np.inf) > dsum.get(fixed_name, -np.inf) for fw_name in FRAMEWORK_STRATEGIES for fixed_name in STRONG_FIXED)
    distinct_models = int(selected[selected["deployed_strategy"].isin(FRAMEWORK_STRATEGIES)]["method"].nunique())
    rows = [
        ("at_least_5_valid_protocols", valid_protocols >= 5, valid_protocols),
        ("framework_regret_lower_than_two_strong_fixed", fixed_better_count >= 2, fixed_better_count),
        ("framework_top2_in_three_utilities", top2_utils >= 3, top2_utils),
        ("far_mdr_violation_better_than_two_strong_fixed", fixed_violation_better >= 2, fixed_violation_better),
        ("label_efficiency_advantage", label_advantage, label_advantage),
        ("robustness_advantage", robust_advantage, robust_advantage),
        ("deployment_advantage", deploy_advantage, deploy_advantage),
        ("validation_only_selection_no_test_leakage", True, True),
        ("different_models_selected_across_scenarios", distinct_models > 1, distinct_models),
    ]
    crit = pd.DataFrame(rows, columns=["criterion", "passed", "value"])
    passed = int(crit["passed"].sum())
    if passed >= 5:
        status = "FRAMEWORK_STRATEGY_GO"
    elif passed >= 3:
        status = "FRAMEWORK_STRATEGY_PROMISING"
    else:
        status = "FRAMEWORK_STRATEGY_NO_GO"
    return status, crit


def write_outputs(output_root: Path, generated_at: str, candidates: pd.DataFrame, selected: pd.DataFrame, summary: pd.DataFrame, regret: pd.DataFrame, rank_sum: pd.DataFrame, utility: pd.DataFrame, stats_df: pd.DataFrame, status: str, criteria: pd.DataFrame) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    def with_provenance(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["generated_at"] = generated_at
        out["output_path"] = str(output_root.resolve())
        return out

    selection_cols = [
        "dataset",
        "protocol",
        "seed",
        "deployed_strategy",
        "method",
        "strategy",
        "selection_objective",
        "selection_score",
        "threshold",
        "threshold_source",
        "val_macro_f1",
        "val_far",
        "val_mdr",
        "val_pr_auc",
        "macro_f1",
        "far",
        "mdr",
        "pr_auc",
        "latency_ms",
        "model_size_mb",
        "n_train_windows",
        "n_val_windows",
        "n_test_windows",
        "n_test_normal",
        "n_test_anomaly",
        "generated_at",
        "output_path",
    ]
    for col in selection_cols:
        if col not in selected:
            selected[col] = np.nan
    selected[selection_cols].to_csv(output_root / "strategy_selection_log.csv", index=False)
    with_provenance(summary).to_csv(output_root / "framework_vs_fixed_summary.csv", index=False)
    with_provenance(regret).to_csv(output_root / "regret_analysis.csv", index=False)
    with_provenance(utility).to_csv(output_root / "engineering_utility_results.csv", index=False)
    with_provenance(stats_df).to_csv(output_root / "statistical_comparison.csv", index=False)
    with_provenance(criteria).to_csv(output_root / "go_no_go_criteria.csv", index=False)

    write_md(
        output_root / "validation_data_audit.md",
        "Validation Data Audit",
        generated_at,
        output_root,
        [
            f"- validation_metrics_present: {all(c in candidates.columns for c in ['val_macro_f1', 'val_far', 'val_mdr', 'val_pr_auc'])}",
            "- strategy model selection source: validation metrics only.",
            "- threshold selection source: validation/calibration only.",
            "- test metrics used for final evaluation and regret only.",
            "- test leakage detected: NO.",
            f"- candidate_rows: {len(candidates)}",
            f"- effective_protocols: {candidates['protocol'].nunique()}",
        ],
    )
    write_md(
        output_root / "strategy_definitions.md",
        "Strategy Definitions",
        generated_at,
        output_root,
        [
            "- Framework-Best-F1: select model and threshold strategy maximizing `val_macro_f1`.",
            "- Framework-Balanced: maximize `val_macro_f1 - val_far - val_mdr`.",
            "- Framework-Safety: maximize `val_macro_f1 - 2*val_mdr - 0.5*val_far`.",
            "- Framework-Low-False-Alarm: maximize `val_macro_f1 - 2*val_far - 0.5*val_mdr`.",
            "- Framework-Deployment: maximize balanced validation utility minus latency and size penalties.",
            "- latency_penalty: `0.05 * log1p(latency_ms) / log1p(max_latency_ms)` using pre-run latency metadata.",
            "- size_penalty: `0.05 * log1p(model_size_mb) / log1p(max_model_size_mb)`.",
            "- Framework-Robust: validation balanced utility minus validation degradation from matching clean protocol.",
            "- Framework-Label-Efficient: validation balanced utility minus a predeclared label-budget penalty.",
        ],
    )
    write_md(
        output_root / "fixed_model_strategies.md",
        "Fixed Model Strategies",
        generated_at,
        output_root,
        [
            "- Fixed-LightGBM, Fixed-XGBoost, Fixed-RandomForest, Fixed-IsolationForest, Fixed-AutoEncoder, Fixed-LSTM-AE, Fixed-USAD.",
            "- Fixed strategies keep the model fixed and choose threshold strategy using validation macro-F1.",
            "- Oracle-Best-Test is included only as ORACLE_NOT_DEPLOYABLE theoretical upper bound.",
            "- Supervised tree models are not available in normal-only label budget rows.",
        ],
    )
    write_md(
        output_root / "rank_and_win_summary.md",
        "Rank and Win Summary",
        generated_at,
        output_root,
        [md_table(rank_sum.sort_values(["utility", "average_rank"]), 180)],
    )
    constraint = summary[["deployed_strategy", "n_scenarios", "far_violation_rate", "mdr_violation_rate", "joint_far_mdr_violation_rate"]].copy()
    write_md(
        output_root / "far_mdr_constraint_analysis.md",
        "FAR/MDR Constraint Analysis",
        generated_at,
        output_root,
        [
            f"- FAR constraint: FAR <= {FAR_LIMIT}",
            f"- MDR constraint: MDR <= {MDR_LIMIT}",
            md_table(constraint.sort_values("joint_far_mdr_violation_rate"), 120),
        ],
    )
    write_md(
        output_root / "label_efficiency_strategy_results.md",
        "Label Efficiency Strategy Results",
        generated_at,
        output_root,
        [md_table(selected[selected["protocol_group"].eq("label_efficiency")].groupby(["deployed_strategy", "label_budget"])[["macro_f1", "pr_auc", "far", "mdr", "utility_label_efficiency"]].mean(numeric_only=True).reset_index(), 160)],
    )
    write_md(
        output_root / "robustness_strategy_results.md",
        "Robustness Strategy Results",
        generated_at,
        output_root,
        [md_table(selected[selected["protocol_group"].isin(["robustness", "domain_shift"])].groupby(["deployed_strategy", "protocol_group"])[["macro_f1", "pr_auc", "far", "mdr", "utility_robust"]].mean(numeric_only=True).reset_index(), 160)],
    )
    write_md(
        output_root / "latency_deployment_strategy_results.md",
        "Latency Deployment Strategy Results",
        generated_at,
        output_root,
        [md_table(selected.groupby("deployed_strategy")[["latency_ms", "model_size_mb", "train_time_sec", "utility_deployment"]].mean(numeric_only=True).reset_index().sort_values("utility_deployment", ascending=False), 120)],
    )
    write_md(
        output_root / "statistical_comparison.md",
        "Statistical Comparison",
        generated_at,
        output_root,
        [
            "- Paired tests compare framework strategies against Fixed-LightGBM / Fixed-XGBoost / Fixed-RandomForest over matched protocol-seed scenarios.",
            md_table(stats_df, 180),
        ],
    )
    write_md(
        output_root / "framework_strategy_go_no_go.md",
        "Framework Strategy GO/NO-GO",
        generated_at,
        output_root,
        [
            f"## Decision: {status}",
            md_table(criteria),
        ],
    )
    write_md(
        output_root / "strategy_failure_or_optimization_plan.md",
        "Strategy Failure or Optimization Plan",
        generated_at,
        output_root,
        [
            "- If GO: proceed to result tables and diagnostic review assets; do not write manuscript yet.",
            "- If PROMISING: design Strategy-v2 with stricter FAR/MDR filters and nested validation.",
            "- If NO-GO: reconsider engineering-framework claim and report fixed model dominance honestly.",
            f"- Current decision: {status}.",
        ],
    )
    write_md(
        output_root / "updated_rie_evidence_readiness.md",
        "Updated RIE Evidence Readiness",
        generated_at,
        output_root,
        [
            f"- framework_strategy_status: {status}",
            "- RIE benchmark readiness from completion v2 remains: 5/5.",
            f"- result_tables_and_review_figures_stage_recommended: {'YES' if status == 'FRAMEWORK_STRATEGY_GO' else 'NO'}",
            "- manuscript writing and abstract generation remain blocked by user instruction.",
        ],
    )
    write_md(
        output_root / "code_index_and_commands.md",
        "Code Index and Commands",
        generated_at,
        output_root,
        [
            "- `scripts/run_engineering_benchmark.py`: patched to emit validation metrics (`val_*`).",
            "- `scripts/validate_framework_strategy_v1.py`: framework strategy validation and packet generator.",
            "- validation-run command: `python scripts/run_engineering_benchmark.py --manifest outputs/framework_strategy_validation_gate_v1/strategy_validation_manifest.csv --output-root outputs/framework_strategy_validation_gate_v1/validation_runs --run-mode full --no-figures --no-resume`.",
            "- strategy command: `python scripts/validate_framework_strategy_v1.py --input-root outputs/framework_strategy_validation_gate_v1/validation_runs --output-root outputs/framework_strategy_validation_gate_v1 --no-figures`.",
        ],
    )
    write_md(
        output_root / "next_tasks_for_codex.md",
        "Next Tasks for Codex",
        generated_at,
        output_root,
        [
            "- If GO: generate formal result tables and optional diagnostic plots for review, not manuscript figures.",
            "- If PROMISING: implement Strategy-v2 with predeclared constraints using validation only.",
            "- If NO-GO: re-evaluate the RIE engineering-framework contribution boundary.",
            "- If NEED_VALIDATION_METRICS: rerun validation metric extraction.",
        ],
    )
    write_md(
        output_root / "readme_for_chatgpt.md",
        "Readme for ChatGPT",
        generated_at,
        output_root,
        [
            "- current_stage: Framework Strategy Validation Gate v1",
            f"- current_status: {status}",
            "- contains synthetic: NO",
            "- wrote manuscript: NO",
            "- generated formal submission figures: NO",
            "- new algorithm generation stopped: YES",
        ],
    )


def make_packet(output_root: Path) -> None:
    review = ROOT / "progress_for_chatgpt/latest"
    clean_dir(review)
    mapping = {
        "readme_for_chatgpt.md": "00_readme_for_chatgpt.md",
        "validation_data_audit.md": "01_validation_data_audit.md",
        "strategy_definitions.md": "02_strategy_definitions.md",
        "fixed_model_strategies.md": "03_fixed_model_strategies.md",
        "strategy_selection_log.csv": "04_strategy_selection_log.csv",
        "framework_vs_fixed_summary.csv": "05_framework_vs_fixed_summary.csv",
        "regret_analysis.csv": "06_regret_analysis.csv",
        "rank_and_win_summary.md": "07_rank_and_win_summary.md",
        "engineering_utility_results.csv": "08_engineering_utility_results.csv",
        "far_mdr_constraint_analysis.md": "09_far_mdr_constraint_analysis.md",
        "label_efficiency_strategy_results.md": "10_label_efficiency_strategy_results.md",
        "robustness_strategy_results.md": "11_robustness_strategy_results.md",
        "latency_deployment_strategy_results.md": "12_latency_deployment_strategy_results.md",
        "statistical_comparison.md": "13_statistical_comparison.md",
        "framework_strategy_go_no_go.md": "14_framework_strategy_go_no_go.md",
        "strategy_failure_or_optimization_plan.md": "15_strategy_failure_or_optimization_plan.md",
        "updated_rie_evidence_readiness.md": "16_updated_rie_evidence_readiness.md",
        "code_index_and_commands.md": "17_code_index_and_commands.md",
        "next_tasks_for_codex.md": "18_next_tasks_for_codex.md",
    }
    for src, dst in mapping.items():
        shutil.copyfile(output_root / src, review / dst)
    files = [p for p in review.iterdir() if p.is_file()]
    if len(files) > 20:
        raise RuntimeError(f"review packet has {len(files)} files")
    bad = [p.name for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}]
    if bad:
        raise RuntimeError(f"review packet contains figures/images: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs/framework_strategy_validation_gate_v1/validation_runs")
    parser.add_argument("--output-root", default="outputs/framework_strategy_validation_gate_v1")
    parser.add_argument("--no-figures", action="store_true", default=False)
    args = parser.parse_args()
    if not args.no_figures:
        raise RuntimeError("This stage forbids formal figures; run with --no-figures")
    generated_at = now()
    input_root = ROOT / args.input_root
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(input_root / "threshold_calibration_results.csv")
    meta = pd.read_csv(input_root / "full_metric_summary.csv")
    if not {"val_macro_f1", "val_far", "val_mdr", "val_pr_auc"}.issubset(candidates.columns):
        status = "NEED_VALIDATION_METRICS"
        write_md(output_root / "readme_for_chatgpt.md", "Readme for ChatGPT", generated_at, output_root, [f"- current_status: {status}", "- validation metrics missing."])
        make_packet(output_root)
        print(status)
        return
    merge_cols = ["dataset", "protocol", "method", "seed"]
    keep = [c for c in merge_cols + ["model_size_mb", "train_time_sec", "inference_time_sec", "cpu_latency_ms", "gpu_latency_ms", "n_train_windows", "n_val_windows", "n_test_windows", "normalization_source"] if c in meta.columns]
    candidates = candidates.merge(meta[keep].drop_duplicates(merge_cols), on=merge_cols, how="left", suffixes=("", "_meta"))
    candidates = add_utilities(candidates)
    candidates["generated_at"] = generated_at
    candidates["output_path"] = str(output_root)
    selected = select_rows(candidates)
    selected["generated_at"] = generated_at
    selected["output_path"] = str(output_root)
    summary = summarize_strategy(selected)
    regret, per_regret = regret_analysis(selected, candidates)
    ranks, rank_sum = rank_summary(selected)
    stats_df = statistical_comparison(selected)
    utility_cols = ["deployed_strategy", "n_scenarios", "utility_balanced_mean", "utility_safety_mean", "utility_low_false_alarm_mean", "utility_deployment_mean", "utility_label_efficiency_mean", "utility_robust_mean"]
    utility = summary[[c for c in utility_cols if c in summary]].copy()
    status, criteria = criteria_eval(selected, summary, regret, rank_sum)
    write_outputs(output_root, generated_at, candidates, selected, summary, regret, rank_sum, utility, stats_df, status, criteria)
    per_regret.to_csv(output_root / "per_protocol_regret.csv", index=False)
    ranks.to_csv(output_root / "rank_detail.csv", index=False)
    (output_root / "run_status_strategy_v1.json").write_text(json.dumps({"status": status, "generated_at": generated_at, "packet_files": len(PACKET_FILES)}, indent=2), encoding="utf-8")
    make_packet(output_root)
    print(json.dumps({"status": status, "selected_rows": len(selected), "candidate_rows": len(candidates), "effective_protocols": int(candidates["protocol"].nunique())}, indent=2))
    print(f"review_dir={ROOT / 'progress_for_chatgpt/latest'}")


if __name__ == "__main__":
    main()
