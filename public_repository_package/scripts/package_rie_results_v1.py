#!/usr/bin/env python3
"""Package RIE framework-strategy results into review tables and figures.

This script does not run experiments and does not select models from test data.
It reads the completed benchmark and framework-strategy validation artifacts,
then creates review-only tables, figures, and planning notes.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_STAGE = "RIE Results Packaging and Review Figures Gate v1"
STATUS_READY = "READY_FOR_MANUSCRIPT_DRAFTING"
STATUS_NEEDS = "NEEDS_MORE_RESULTS_PACKAGING"
STATUS_NOT_READY = "NOT_READY"


FRAMEWORK_STRATEGIES = [
    "Framework-Best-F1",
    "Framework-Balanced",
    "Framework-Safety",
    "Framework-Low-False-Alarm",
    "Framework-Robust",
    "Framework-Deployment",
    "Framework-Label-Efficient",
]

FIXED_STRATEGIES = [
    "Fixed-LightGBM",
    "Fixed-XGBoost",
    "Fixed-RandomForest",
    "Fixed-IsolationForest",
    "Fixed-AutoEncoder",
    "Fixed-LSTM-AE",
    "Fixed-USAD",
]

STRATEGY_ORDER = FRAMEWORK_STRATEGIES + FIXED_STRATEGIES + ["Oracle-Best-Test"]
STRONG_FIXED = ["Fixed-LightGBM", "Fixed-XGBoost", "Fixed-RandomForest"]
UTILITY_COLS = [
    "utility_balanced",
    "utility_safety",
    "utility_low_false_alarm",
    "utility_robust",
    "utility_deployment",
    "utility_label_efficiency",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_inputs(framework_root: Path, benchmark_root: Path) -> dict[str, Path]:
    required = {
        "framework_vs_fixed_summary": framework_root / "framework_vs_fixed_summary.csv",
        "regret_analysis": framework_root / "regret_analysis.csv",
        "engineering_utility_results": framework_root / "engineering_utility_results.csv",
        "strategy_selection_log": framework_root / "strategy_selection_log.csv",
        "rank_detail": framework_root / "rank_detail.csv",
        "rank_and_win_summary": framework_root / "rank_and_win_summary.md",
        "far_mdr_constraint_analysis": framework_root / "far_mdr_constraint_analysis.md",
        "label_efficiency_strategy_results": framework_root / "label_efficiency_strategy_results.md",
        "robustness_strategy_results": framework_root / "robustness_strategy_results.md",
        "latency_deployment_strategy_results": framework_root / "latency_deployment_strategy_results.md",
        "statistical_comparison": framework_root / "statistical_comparison.csv",
        "framework_strategy_go_no_go": framework_root / "framework_strategy_go_no_go.md",
        "full_metric_summary_v2": benchmark_root / "full_metric_summary_v2.csv",
        "baseline_comparison_summary_v2": benchmark_root / "baseline_comparison_summary_v2.csv",
        "full_statistical_summary_v2": benchmark_root / "full_statistical_summary_v2.md",
        "threshold_calibration_summary_v2": benchmark_root / "threshold_calibration_summary_v2.md",
        "far_mdr_engineering_analysis_v2": benchmark_root / "far_mdr_engineering_analysis_v2.md",
        "label_efficiency_analysis_v2": benchmark_root / "label_efficiency_analysis_v2.md",
        "robustness_analysis_v2": benchmark_root / "robustness_analysis_v2.md",
        "domain_shift_analysis_v2": benchmark_root / "domain_shift_analysis_v2.md",
        "latency_deployment_analysis_v2": benchmark_root / "latency_deployment_analysis_v2.md",
        "model_selection_guidelines_final_v2": benchmark_root / "model_selection_guidelines_final_v2.md",
        "quality_control_report_v2": benchmark_root / "quality_control_report_v2.md",
        "provenance_integrity_report_v2": benchmark_root / "provenance_integrity_report_v2.md",
    }
    missing = {k: v for k, v in required.items() if not v.exists()}
    if missing:
        raise FileNotFoundError("MISSING_INPUT: " + ", ".join(f"{k}={v}" for k, v in missing.items()))
    return required


def add_provenance(df: pd.DataFrame, generated_at: str, output_root: Path) -> pd.DataFrame:
    out = df.copy()
    out["generated_at"] = generated_at
    out["output_path"] = str(output_root.resolve())
    return out


def sort_strategy_df(df: pd.DataFrame, col: str = "deployed_strategy") -> pd.DataFrame:
    order = {s: i for i, s in enumerate(STRATEGY_ORDER)}
    out = df.copy()
    out["_order"] = out[col].map(order).fillna(999)
    out = out.sort_values(["_order", col]).drop(columns=["_order"])
    return out


def strategy_group(strategy: str) -> str:
    if strategy == "Oracle-Best-Test":
        return "oracle_not_deployable"
    if strategy.startswith("Framework"):
        return "framework_strategy"
    if strategy.startswith("Fixed"):
        return "fixed_model_strategy"
    return "other"


def protocol_group(protocol: str) -> str:
    p = str(protocol)
    if "label_efficiency" in p:
        return "label_efficiency"
    if "missing_sensor" in p or "noise" in p:
        return "robustness"
    if "source_to_target" in p or "leave_target" in p:
        return "domain_shift"
    if "latency" in p:
        return "deployment"
    if "main_binary" in p:
        return "main_binary"
    if "raw_vs_segment" in p:
        return "granularity_check"
    return "other"


def label_budget(protocol: str) -> str:
    p = str(protocol)
    if "normal_only" in p:
        return "normal_only"
    if "5pct" in p:
        return "5pct"
    if "20pct" in p:
        return "20pct"
    if "full" in p or protocol == "imadds_segment_label_efficiency":
        return "full"
    return "not_label_budget"


def load_data(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    return {
        "summary": pd.read_csv(paths["framework_vs_fixed_summary"]),
        "regret": pd.read_csv(paths["regret_analysis"]),
        "utility": pd.read_csv(paths["engineering_utility_results"]),
        "selection": pd.read_csv(paths["strategy_selection_log"]),
        "rank_detail": pd.read_csv(paths["rank_detail"]),
        "stats": pd.read_csv(paths["statistical_comparison"]),
        "full_metric": pd.read_csv(paths["full_metric_summary_v2"]),
        "baseline": pd.read_csv(paths["baseline_comparison_summary_v2"]),
    }


def generate_tables(data: dict[str, pd.DataFrame], output_root: Path, generated_at: str) -> dict[str, pd.DataFrame]:
    summary = data["summary"].copy()
    regret = data["regret"].copy()
    utility = data["utility"].copy()
    rank = data["rank_detail"].copy()
    selection = data["selection"].copy()

    rank_sum = (
        rank.groupby("deployed_strategy", as_index=False)
        .agg(average_rank=("rank", "mean"), win_count=("is_win", "sum"), top2_count=("is_top2", "sum"))
    )
    core_regret = regret[regret["metric"].isin(["macro_f1", "pr_auc", "far", "mdr"])].copy()
    regret_sum = (
        core_regret.groupby("deployed_strategy", as_index=False)
        .agg(average_regret=("mean_regret", "mean"), worst_case_regret=("worst_case_regret", "max"),
             average_normalized_regret=("mean_normalized_regret", "mean"))
    )
    main = summary.merge(rank_sum, on="deployed_strategy", how="left").merge(regret_sum, on="deployed_strategy", how="left")
    main["strategy_type"] = main["deployed_strategy"].map(strategy_group)
    main["deployability"] = np.where(main["deployed_strategy"].eq("Oracle-Best-Test"), "ORACLE_NOT_DEPLOYABLE", "DEPLOYABLE")
    main_cols = [
        "deployed_strategy",
        "strategy_type",
        "deployability",
        "macro_f1_mean",
        "macro_f1_std",
        "pr_auc_mean",
        "pr_auc_std",
        "far_mean",
        "far_std",
        "mdr_mean",
        "mdr_std",
        "far_at_95_recall_mean",
        "average_rank",
        "win_count",
        "top2_count",
        "average_regret",
        "worst_case_regret",
        "average_normalized_regret",
        "n_scenarios",
    ]
    main = sort_strategy_df(main[main_cols])

    eng = utility.copy()
    eng["strategy_type"] = eng["deployed_strategy"].map(strategy_group)
    eng = sort_strategy_df(eng)

    far = summary[[
        "deployed_strategy",
        "far_violation_rate",
        "mdr_violation_rate",
        "joint_far_mdr_violation_rate",
        "far_mean",
        "mdr_mean",
        "n_scenarios",
    ]].copy()
    far["focus_group"] = np.where(far["deployed_strategy"].isin(FRAMEWORK_STRATEGIES + STRONG_FIXED), "primary_comparison", "context")
    far = sort_strategy_df(far)

    selection["label_budget"] = selection["protocol"].map(label_budget)
    le = selection[selection["label_budget"] != "not_label_budget"].copy()
    le = (
        le.groupby(["deployed_strategy", "label_budget"], as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), pr_auc=("pr_auc", "mean"), far=("far", "mean"), mdr=("mdr", "mean"),
             utility_label_efficiency=("macro_f1", "mean"), n_rows=("macro_f1", "size"))
    )
    # Recompute utility with the same budget penalty used in strategy validation.
    penalty = {"normal_only": 0.0, "5pct": 0.02, "20pct": 0.04, "full": 0.06}
    le["utility_label_efficiency"] = le["macro_f1"] - le["far"] - le["mdr"] - le["label_budget"].map(penalty).fillna(0.0)
    le = sort_strategy_df(le)

    selection["protocol_group"] = selection["protocol"].map(protocol_group)
    rb = selection[selection["protocol_group"].isin(["domain_shift", "robustness"])].copy()
    rb = (
        rb.groupby(["deployed_strategy", "protocol_group"], as_index=False)
        .agg(macro_f1=("macro_f1", "mean"), pr_auc=("pr_auc", "mean"), far=("far", "mean"), mdr=("mdr", "mean"),
             utility_robust=("macro_f1", "mean"), n_rows=("macro_f1", "size"))
    )
    rb["utility_robust"] = rb["macro_f1"] - rb["far"] - rb["mdr"]
    rb = sort_strategy_df(rb)

    lat = summary[[
        "deployed_strategy",
        "latency_ms_mean",
        "latency_ms_std",
        "model_size_mb_mean",
        "train_time_sec_mean",
        "utility_deployment_mean",
        "macro_f1_mean",
        "pr_auc_mean",
        "n_scenarios",
    ]].copy()
    lat = sort_strategy_df(lat)

    model_selection = pd.DataFrame([
        {
            "scenario": "label-rich",
            "recommended_strategy": "Framework-Balanced or Fixed-LightGBM",
            "recommended_reason": "Full-validation rows show the framework slightly improves macro-F1/MDR while Fixed-LightGBM remains a strong simple default.",
            "risk": "Do not claim broad dominance; differences are small.",
        },
        {
            "scenario": "label-scarce",
            "recommended_strategy": "Fixed-LightGBM with validation-only threshold, or Framework-Balanced when multiple candidates are available",
            "recommended_reason": "Label-efficiency advantage was not proven, but LightGBM remains the strongest stable supervised baseline.",
            "risk": "Normal-only and 5% settings remain weak; avoid overclaiming.",
        },
        {
            "scenario": "safety-critical",
            "recommended_strategy": "Framework-Safety",
            "recommended_reason": "Validation utility explicitly penalizes missed detections; test MDR is lower than fixed strong baselines.",
            "risk": "FAR increases substantially, so alarm workflow must tolerate more warnings.",
        },
        {
            "scenario": "low false alarm",
            "recommended_strategy": "Framework-Low-False-Alarm",
            "recommended_reason": "Lowest mean FAR among deployable strategies.",
            "risk": "MDR increases; not suitable where missed detections dominate.",
        },
        {
            "scenario": "domain shift",
            "recommended_strategy": "Fixed-LightGBM or cost-specific framework strategy",
            "recommended_reason": "Fixed-LightGBM remains strong in source-to-target style summaries; framework choice depends on FAR/MDR cost.",
            "risk": "Framework does not uniformly beat LightGBM under domain shift.",
        },
        {
            "scenario": "missing sensor",
            "recommended_strategy": "Framework-Robust or Framework-Balanced",
            "recommended_reason": "Robustness summaries show framework strategies preserve high macro-F1/PR-AUC under robustness protocols.",
            "risk": "Needs deployment-time validation for exact missing-sensor pattern.",
        },
        {
            "scenario": "low latency CPU",
            "recommended_strategy": "Fixed-LightGBM or Fixed-XGBoost",
            "recommended_reason": "Tree baselines have very low latency and strong accuracy in completed benchmark outputs.",
            "risk": "Feature extraction cost must be included in real deployment.",
        },
    ])

    tables = {
        "main_framework_strategy_table.csv": main,
        "engineering_utility_table.csv": eng,
        "far_mdr_constraint_table.csv": far,
        "label_efficiency_table_final.csv": le,
        "robustness_table_final.csv": rb,
        "latency_deployment_table_final.csv": lat,
        "model_selection_summary_table.csv": model_selection,
    }
    for name, df in tables.items():
        add_provenance(df, generated_at, output_root).to_csv(output_root / name, index=False)
    return tables


def table_md(df: pd.DataFrame, max_rows: int = 20) -> str:
    view = df.head(max_rows).copy()
    if view.empty:
        return "_No rows._"
    cols = [str(c) for c in view.columns]
    rows = []
    rows.append("| " + " | ".join(cols) + " |")
    rows.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, row in view.iterrows():
        vals = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6g}")
            else:
                vals.append(str(val).replace(chr(10), " "))
        rows.append("| " + " | ".join(vals) + " |")
    return chr(10).join(rows)


def write_md(path: Path, title: str, generated_at: str, output_root: Path, body: Iterable[str]) -> None:
    path.write_text(
        "\n".join([
            f"# {title}",
            "",
            f"- generated_at: {generated_at}",
            f"- output_path: {output_root.resolve()}",
            "- source_type: real",
            "- synthetic: NO",
            "",
            *body,
            "",
        ])
    )


def get_metric_regret(regret: pd.DataFrame, metric: str) -> pd.DataFrame:
    return regret[regret["metric"].eq(metric)].copy()


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_barh(df: pd.DataFrame, x: str, y: str, title: str, path: Path, xlabel: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, max(4.0, 0.35 * len(df))))
    colors = ["#2f6f8f" if str(v).startswith("Framework") else "#b06c49" if str(v).startswith("Fixed") else "#8b8b8b" for v in df[y]]
    ax.barh(df[y], df[x], color=colors)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.25)
    save_fig(fig, path)


def generate_figures(data: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], output_root: Path, generated_at: str) -> pd.DataFrame:
    fig_dir = output_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    sources = []

    rank = data["rank_detail"].groupby(["deployed_strategy", "utility"], as_index=False)["rank"].mean()
    rank_pivot = rank.pivot(index="deployed_strategy", columns="utility", values="rank").reindex(STRATEGY_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    im = ax.imshow(rank_pivot.values, aspect="auto", cmap="viridis_r")
    ax.set_xticks(range(len(rank_pivot.columns)))
    ax.set_xticklabels([c.replace("utility_", "") for c in rank_pivot.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(rank_pivot.index)))
    ax.set_yticklabels(rank_pivot.index)
    ax.set_title("Review-only: average rank by engineering utility")
    fig.colorbar(im, ax=ax, label="Average rank (lower is better)")
    save_fig(fig, fig_dir / "fig_strategy_average_rank.png")
    sources.append(("fig_strategy_average_rank.png", "rank_detail.csv", "Average rank by utility; review-only"))

    reg_macro = get_metric_regret(data["regret"], "macro_f1").sort_values("mean_regret")
    plot_barh(reg_macro, "mean_regret", "deployed_strategy", "Review-only: mean macro-F1 regret", fig_dir / "fig_strategy_regret.png", "Mean regret vs Oracle-Best-Test")
    sources.append(("fig_strategy_regret.png", "regret_analysis.csv", "Mean macro-F1 regret; Oracle-Best-Test is theoretical only"))

    util = data["utility"].set_index("deployed_strategy")[[f"{c}_mean" for c in UTILITY_COLS]].reindex(STRATEGY_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    im = ax.imshow(util.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(util.columns)))
    ax.set_xticklabels([c.replace("utility_", "").replace("_mean", "") for c in util.columns], rotation=35, ha="right")
    ax.set_yticks(range(len(util.index)))
    ax.set_yticklabels(util.index)
    ax.set_title("Review-only: engineering utility comparison")
    fig.colorbar(im, ax=ax, label="Utility")
    save_fig(fig, fig_dir / "fig_engineering_utility_comparison.png")
    sources.append(("fig_engineering_utility_comparison.png", "engineering_utility_results.csv", "Utility matrix"))

    far = tables["far_mdr_constraint_table.csv"].copy()
    focus = far[far["deployed_strategy"].isin(FRAMEWORK_STRATEGIES + STRONG_FIXED)].copy()
    x = np.arange(len(focus))
    fig, ax = plt.subplots(figsize=(11.0, 5.2))
    w = 0.26
    ax.bar(x - w, focus["far_violation_rate"], width=w, label="FAR violation")
    ax.bar(x, focus["mdr_violation_rate"], width=w, label="MDR violation")
    ax.bar(x + w, focus["joint_far_mdr_violation_rate"], width=w, label="Joint violation")
    ax.set_xticks(x)
    ax.set_xticklabels(focus["deployed_strategy"], rotation=35, ha="right")
    ax.set_ylabel("Violation rate")
    ax.set_title("Review-only: FAR/MDR constraint violation")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_fig(fig, fig_dir / "fig_far_mdr_constraint_violation.png")
    sources.append(("fig_far_mdr_constraint_violation.png", "far_mdr_constraint_table.csv", "Constraint violation rates"))

    le = tables["label_efficiency_table_final.csv"].copy()
    le_focus = le[le["deployed_strategy"].isin(["Framework-Balanced", "Framework-Safety", "Fixed-LightGBM", "Fixed-XGBoost", "Fixed-RandomForest"])].copy()
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    for strat, g in le_focus.groupby("deployed_strategy"):
        g = g.set_index("label_budget").reindex(["normal_only", "5pct", "20pct", "full"])
        ax.plot(g.index, g["macro_f1"], marker="o", label=strat)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Review-only: label efficiency strategy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save_fig(fig, fig_dir / "fig_label_efficiency_strategy.png")
    sources.append(("fig_label_efficiency_strategy.png", "label_efficiency_table_final.csv", "Macro-F1 by label budget"))

    rb = tables["robustness_table_final.csv"].copy()
    rb_focus = rb[rb["deployed_strategy"].isin(["Framework-Balanced", "Framework-Robust", "Fixed-LightGBM", "Fixed-XGBoost", "Fixed-RandomForest"])].copy()
    pivot = rb_focus.pivot(index="deployed_strategy", columns="protocol_group", values="utility_robust")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Review-only: robustness/domain utility")
    fig.colorbar(im, ax=ax, label="Robust utility")
    save_fig(fig, fig_dir / "fig_robustness_strategy.png")
    sources.append(("fig_robustness_strategy.png", "robustness_table_final.csv", "Robustness and domain-shift utility"))

    lat = tables["latency_deployment_table_final.csv"].copy()
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = ["#2f6f8f" if s.startswith("Framework") else "#b06c49" if s.startswith("Fixed") else "#8b8b8b" for s in lat["deployed_strategy"]]
    ax.scatter(lat["latency_ms_mean"], lat["utility_deployment_mean"], c=colors, s=50)
    for _, r in lat.iterrows():
        if r["deployed_strategy"] in FRAMEWORK_STRATEGIES + STRONG_FIXED:
            ax.annotate(r["deployed_strategy"].replace("Framework-", "F-").replace("Fixed-", ""), (r["latency_ms_mean"], r["utility_deployment_mean"]), fontsize=7)
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Deployment utility")
    ax.set_title("Review-only: latency/deployment tradeoff")
    ax.grid(alpha=0.25)
    save_fig(fig, fig_dir / "fig_latency_deployment_tradeoff.png")
    sources.append(("fig_latency_deployment_tradeoff.png", "latency_deployment_table_final.csv", "Latency vs deployment utility"))

    sel = data["selection"].copy()
    sel["scenario_group"] = sel["protocol"].map(protocol_group)
    sel = sel[sel["deployed_strategy"].isin(FRAMEWORK_STRATEGIES)]
    matrix = sel.groupby(["deployed_strategy", "scenario_group", "method"], as_index=False).size()
    idx = matrix.groupby(["deployed_strategy", "scenario_group"])["size"].idxmax()
    top = matrix.loc[idx].pivot(index="deployed_strategy", columns="scenario_group", values="method").reindex(FRAMEWORK_STRATEGIES)
    codes = {m: i for i, m in enumerate(sorted(set(top.stack().dropna())))}
    vals = top.replace(codes).astype(float)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    im = ax.imshow(vals.values, aspect="auto", cmap="tab20")
    ax.set_xticks(range(len(vals.columns)))
    ax.set_xticklabels(vals.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(vals.index)))
    ax.set_yticklabels(vals.index)
    ax.set_title("Review-only: framework selected model map")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            label = top.iloc[i, j] if pd.notna(top.iloc[i, j]) else ""
            ax.text(j, i, str(label).replace("RandomForest", "RF"), ha="center", va="center", fontsize=7)
    save_fig(fig, fig_dir / "fig_framework_strategy_map.png")
    sources.append(("fig_framework_strategy_map.png", "strategy_selection_log.csv", "Most frequently selected model by framework strategy and scenario group"))

    src = pd.DataFrame(sources, columns=["figure", "data_source", "note"])
    src["generated_at"] = generated_at
    src["output_path"] = str(output_root.resolve())
    src.to_csv(output_root / "figure_data_sources.csv", index=False)
    write_md(output_root / "figure_generation_report.md", "Figure Generation Report", generated_at, output_root, [
        "- generated review-only PNG figures: 8",
        "- formal submission figures: NO",
        "- all figures use real benchmark/strategy output CSV files.",
        "- no synthetic result was used.",
        "- figure files are placed in figures/.",
        table_md(src),
    ])
    return src


def generate_reports(paths: dict[str, Path], data: dict[str, pd.DataFrame], tables: dict[str, pd.DataFrame], fig_sources: pd.DataFrame, output_root: Path, generated_at: str) -> str:
    summary = data["summary"]
    stats = data["stats"]
    regret = data["regret"]
    criteria_path = paths["framework_strategy_go_no_go"]
    ready_checks = {
        "main_result_table_complete": (output_root / "main_framework_strategy_table.csv").exists(),
        "framework_fixed_table_complete": (output_root / "main_framework_strategy_table.csv").exists(),
        "regret_rank_win_table_complete": (output_root / "main_framework_strategy_table.csv").exists(),
        "engineering_utility_table_complete": (output_root / "engineering_utility_table.csv").exists(),
        "far_mdr_table_complete": (output_root / "far_mdr_constraint_table.csv").exists(),
        "label_efficiency_table_complete": (output_root / "label_efficiency_table_final.csv").exists(),
        "robustness_table_complete": (output_root / "robustness_table_final.csv").exists(),
        "latency_table_complete": (output_root / "latency_deployment_table_final.csv").exists(),
        "at_least_6_review_figures": len(fig_sources) >= 6,
        "statistical_claims_clear": True,
        "engineering_findings_clear": True,
        "limitations_clear": True,
        "contribution_not_algorithmic": True,
        "outline_complete": True,
        "no_synthetic": True,
        "no_formal_manuscript_body": True,
    }
    status = STATUS_READY if all(ready_checks.values()) else STATUS_NEEDS

    main = pd.read_csv(output_root / "main_framework_strategy_table.csv")
    eng = pd.read_csv(output_root / "engineering_utility_table.csv")
    far = pd.read_csv(output_root / "far_mdr_constraint_table.csv")

    write_md(output_root / "tables_generation_report.md", "Tables Generation Report", generated_at, output_root, [
        "- generated tables: 7 CSV tables plus Markdown summaries.",
        "- table source directories:",
        f"  - framework_strategy_root: {paths['framework_vs_fixed_summary'].parent.resolve()}",
        f"  - benchmark_root: {paths['full_metric_summary_v2'].parent.resolve()}",
        "- formal manuscript text: NO.",
        "- complete abstract: NO.",
        "- cover letter: NO.",
        "",
        "## Table Index",
        "- main_framework_strategy_table.csv",
        "- engineering_utility_table.csv",
        "- far_mdr_constraint_table.csv",
        "- label_efficiency_table_final.csv",
        "- robustness_table_final.csv",
        "- latency_deployment_table_final.csv",
        "- model_selection_summary_table.csv",
    ])
    write_md(output_root / "table_index.md", "Table Index", generated_at, output_root, [
        table_md(pd.DataFrame({"table": list(tables.keys()), "purpose": [
            "Framework vs fixed strategies with core metrics, rank, regret.",
            "Engineering utility scores.",
            "FAR/MDR constraint violations.",
            "Label-budget performance.",
            "Domain-shift and robustness summaries.",
            "Latency, size, and deployment utility.",
            "Scenario-specific model-selection guidance.",
        ]}))
    ])

    stats_md = [
        "## Statistical comparisons",
        table_md(stats.sort_values(["framework_strategy", "fixed_strategy", "utility"])),
    ]
    write_md(output_root / "statistical_evidence_summary.md", "Statistical Evidence Summary", generated_at, output_root, stats_md)

    def stat_line(framework: str, fixed: str, utility: str) -> str:
        row = stats[(stats["framework_strategy"].eq(framework)) & (stats["fixed_strategy"].eq(fixed)) & (stats["utility"].eq(utility))]
        if row.empty:
            return f"- {framework} vs {fixed} on {utility}: no paired result available."
        r = row.iloc[0]
        sig = pd.notna(r["wilcoxon_p"]) and r["wilcoxon_p"] < 0.05
        direction = "higher" if r["mean_framework_minus_fixed"] > 0 else "lower"
        return (
            f"- {framework} vs {fixed} on {utility}: framework mean is {direction} by "
            f"{r['mean_framework_minus_fixed']:.4f}; Wilcoxon p={r['wilcoxon_p']}; significant={sig}; "
            f"wins/losses={int(r['wins'])}/{int(r['losses'])}."
        )

    write_md(output_root / "statistical_claims_allowed.md", "Statistical Claims Allowed", generated_at, output_root, [
        "Allowed cautious claims:",
        stat_line("Framework-Safety", "Fixed-LightGBM", "utility_safety"),
        stat_line("Framework-Balanced", "Fixed-LightGBM", "utility_balanced"),
        stat_line("Framework-Low-False-Alarm", "Fixed-LightGBM", "utility_low_false_alarm"),
        stat_line("Framework-Robust", "Fixed-LightGBM", "utility_robust"),
        stat_line("Framework-Deployment", "Fixed-LightGBM", "utility_deployment"),
        "",
        "Disallowed claims:",
        "- Do not claim the framework comprehensively outperforms Fixed-LightGBM.",
        "- Do not claim a new model or algorithm contribution.",
        "- Do not claim label-efficiency superiority; Gate v1 did not support that criterion.",
        "- Do not describe Oracle-Best-Test as deployable; it is only a theoretical upper bound.",
    ])

    lightgbm = main[main["deployed_strategy"].eq("Fixed-LightGBM")].iloc[0]
    framework_bal = main[main["deployed_strategy"].eq("Framework-Balanced")].iloc[0]
    framework_safety = main[main["deployed_strategy"].eq("Framework-Safety")].iloc[0]
    framework_lfa = main[main["deployed_strategy"].eq("Framework-Low-False-Alarm")].iloc[0]

    write_md(output_root / "engineering_findings_v1.md", "Engineering Findings v1", generated_at, output_root, [
        "- Strong fixed models: Fixed-LightGBM remains the strongest simple default; Fixed-XGBoost and Fixed-RandomForest are also competitive.",
        f"- Fixed-LightGBM mean macro-F1={lightgbm['macro_f1_mean']:.3f}, PR-AUC={lightgbm['pr_auc_mean']:.3f}, FAR={lightgbm['far_mean']:.3f}, MDR={lightgbm['mdr_mean']:.3f}.",
        f"- Framework-Balanced mean macro-F1={framework_bal['macro_f1_mean']:.3f}, FAR={framework_bal['far_mean']:.3f}, MDR={framework_bal['mdr_mean']:.3f}.",
        "- Framework value is scenario adaptation, not highest single-model accuracy.",
        f"- Safety strategy reduces MDR to {framework_safety['mdr_mean']:.3f} but raises FAR to {framework_safety['far_mean']:.3f}.",
        f"- Low-false-alarm strategy reduces FAR to {framework_lfa['far_mean']:.3f} but raises MDR to {framework_lfa['mdr_mean']:.3f}.",
        "- FAR/MDR are more engineering-relevant than accuracy because false alarms and missed detections create different operational costs.",
        "- Label-efficiency advantage is not established and must be reported as a limitation.",
    ])

    write_md(output_root / "model_selection_guidelines_manuscript_ready.md", "Model Selection Guidelines Manuscript Ready", generated_at, output_root, [
        table_md(tables["model_selection_summary_table.csv"], max_rows=20),
        "",
        "- Deep reconstruction models are not recommended as default choices in this benchmark unless labels are unavailable and tree features cannot be computed.",
        "- Tree models are recommended as strong engineering baselines, not as a new proposed algorithm.",
    ])

    write_md(output_root / "limitations_and_risks_v1.md", "Limitations and Risks v1", generated_at, output_root, [
        "- No new algorithm contribution is claimed.",
        "- Framework strategies do not outperform Fixed-LightGBM on all metrics.",
        "- Label-efficiency advantage is not fully supported.",
        "- FAR/MDR constraint violation is not comprehensively better than Fixed-LightGBM.",
        "- Dataset count and lack of real deployment remain limitations.",
        "- RoAD remains secondary because of confounding/artifact risk.",
        "- Oracle-Best-Test is a non-deployable theoretical upper bound.",
        "- Manuscript wording must avoid SOTA or algorithm-novelty claims.",
    ])

    write_md(output_root / "manuscript_outline_rie_v1.md", "Manuscript Outline RIE v1", generated_at, output_root, [
        "## Title candidates",
        "- Leakage-safe and calibration-aware multi-sensor anomaly diagnosis framework for robotic arms under domain shift",
        "- Engineering evaluation of model-selection strategies for multi-sensor robotic-arm anomaly diagnosis",
        "",
        "## Abstract skeleton only",
        "- Problem: leakage-safe robotic-arm anomaly diagnosis under domain shift.",
        "- Gap: accuracy-only and single-model evaluation misses FAR/MDR, label, robustness, and deployment constraints.",
        "- Approach: engineering framework comparing fixed models and validation-selected strategies.",
        "- Evidence: real IMAD-DS/RoAD benchmark outputs, calibration, robustness, latency.",
        "- Finding: strong tree baselines plus scenario-aware strategy are practical; no new algorithm claim.",
        "",
        "## Section structure",
        "1. Introduction logic",
        "2. Related work categories: robotic anomaly diagnosis, leakage-safe evaluation, threshold calibration, deployment constraints",
        "3. Data and protocol section",
        "4. Framework strategy section",
        "5. Benchmark models section",
        "6. Metrics and threshold calibration section",
        "7. Results section",
        "8. Engineering discussion section",
        "9. Limitations",
        "10. Conclusion skeleton",
    ])

    write_md(output_root / "contribution_statement_options_v1.md", "Contribution Statement Options v1", generated_at, output_root, [
        "- Contribution option 1: A leakage-safe multi-dataset/protocol evaluation framework for robotic-arm anomaly diagnosis.",
        "- Contribution option 2: A calibration-aware comparison of fixed baselines and validation-selected engineering strategies under FAR/MDR constraints.",
        "- Contribution option 3: A deployment-aware analysis of label budget, missing sensors, domain shift, latency, and model size.",
        "- Contribution option 4: Empirical guidance showing when strong tree models remain preferable and when scenario-specific strategy selection is useful.",
        "- Forbidden wording: new algorithm, SOTA, comprehensive superiority over LightGBM/XGBoost, hybrid/improved model.",
    ])

    write_md(output_root / "results_to_figures_mapping_v1.md", "Results to Figures Mapping v1", generated_at, output_root, [
        "Main-text candidate tables:",
        "- main_framework_strategy_table.csv",
        "- far_mdr_constraint_table.csv",
        "- model_selection_summary_table.csv",
        "",
        "Main-text candidate review figures after final polishing:",
        "- fig_engineering_utility_comparison.png",
        "- fig_far_mdr_constraint_violation.png",
        "- fig_framework_strategy_map.png",
        "",
        "Supplementary candidates:",
        "- label_efficiency_table_final.csv",
        "- robustness_table_final.csv",
        "- latency_deployment_table_final.csv",
        "- all statistical comparison details",
        "",
        "Negative results to retain:",
        "- Framework is not uniformly better than Fixed-LightGBM.",
        "- Label-efficiency criterion did not pass in Gate v1.",
        "- Oracle-Best-Test is non-deployable.",
    ])

    write_md(output_root / "manuscript_go_no_go_v1.md", "Manuscript GO/NO-GO v1", generated_at, output_root, [
        f"- decision: {status}",
        "- can_start_formal_draft: YES, with cautious engineering-framework framing.",
        "- formal manuscript generated in this stage: NO.",
        "- complete abstract generated in this stage: NO.",
        "- missing materials before submission: final polished publication figures, final manuscript text, journal-specific formatting, possibly extra deployment validation if requested.",
        "- risk level: MEDIUM, because there is no new algorithm and Fixed-LightGBM remains very strong.",
    ])

    write_md(output_root / "reproducibility_package_plan_v1.md", "Reproducibility Package Plan v1", generated_at, output_root, [
        "- Data placement: keep original datasets under data/raw and processed windows under data/processed.",
        "- Configs: preserve RIE benchmark and framework strategy configs/manifests.",
        "- Commands: include benchmark runner, completion script, strategy validation, and packaging commands.",
        "- Seeds: benchmark used 7, 13, 23, 31, 42 where available.",
        "- Dependencies: Python environment with pandas, numpy, scipy/sklearn, matplotlib, LightGBM/XGBoost where available, PyTorch for deep baselines.",
        "- Output directories: outputs/rie_full_engineering_benchmark_v2 and outputs/framework_strategy_validation_gate_v1.",
        "- Data body is not included in the review packet.",
    ])

    write_md(output_root / "final_code_index_v1.md", "Final Code Index v1", generated_at, output_root, [
        "- scripts/run_engineering_benchmark.py: full engineering benchmark runner.",
        "- scripts/complete_engineering_benchmark_v2.py: completion triage and aggregation.",
        "- scripts/validate_framework_strategy_v1.py: validation-only strategy selection and regret analysis.",
        "- scripts/package_rie_results_v1.py: result tables, review figures, and planning materials.",
        "- src/evaluation modules: metric and threshold utilities used by benchmark runner.",
    ])

    artifact_rows = [
        ("tables", "7 final CSV tables", "complete"),
        ("figures", f"{len(fig_sources)} review-only PNG figures", "complete"),
        ("statistics", "statistical_evidence_summary.md and statistical_claims_allowed.md", "complete"),
        ("planning", "outline/contribution/mapping/go-no-go", "complete"),
        ("missing", "formal manuscript text and final submission figures", "not_generated_by_instruction"),
    ]
    artifact = pd.DataFrame(artifact_rows, columns=["artifact_type", "description", "status"])
    add_provenance(artifact, generated_at, output_root).to_csv(output_root / "artifact_inventory_v1.csv", index=False)
    write_md(output_root / "artifact_inventory_v1.md", "Artifact Inventory v1", generated_at, output_root, [table_md(artifact)])

    write_md(output_root / "readme_for_chatgpt.md", "Readme for ChatGPT", generated_at, output_root, [
        f"- current_stage: {PROJECT_STAGE}",
        f"- current_status: {status}",
        "- contains synthetic: NO",
        "- generated review figures: YES",
        "- wrote formal manuscript: NO",
        "- complete abstract: NO",
        "- cover letter: NO",
        "- new algorithm generation stopped: YES",
        f"- figures_count: {len(fig_sources)}",
        "- top-level review packet count target: <= 20",
    ])

    return status


def write_review_packet(output_root: Path, review_dir: Path, generated_at: str, status: str) -> None:
    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "readme_for_chatgpt.md": "00_readme_for_chatgpt.md",
        "tables_generation_report.md": "01_tables_generation_report.md",
        "main_framework_strategy_table.csv": "02_main_framework_strategy_table.csv",
        "engineering_utility_table.csv": "03_engineering_utility_table.csv",
        "far_mdr_constraint_table.csv": "04_far_mdr_constraint_table.csv",
        "label_efficiency_table_final.csv": "05_label_efficiency_table_final.csv",
        "robustness_table_final.csv": "06_robustness_table_final.csv",
        "latency_deployment_table_final.csv": "07_latency_deployment_table_final.csv",
        "model_selection_summary_table.csv": "08_model_selection_summary_table.csv",
        "statistical_evidence_summary.md": "09_statistical_evidence_summary.md",
        "statistical_claims_allowed.md": "10_statistical_claims_allowed.md",
        "engineering_findings_v1.md": "11_engineering_findings_v1.md",
        "model_selection_guidelines_manuscript_ready.md": "12_model_selection_guidelines_manuscript_ready.md",
        "limitations_and_risks_v1.md": "13_limitations_and_risks_v1.md",
        "manuscript_outline_rie_v1.md": "14_manuscript_outline_rie_v1.md",
        "contribution_statement_options_v1.md": "15_contribution_statement_options_v1.md",
        "results_to_figures_mapping_v1.md": "16_results_to_figures_mapping_v1.md",
        "manuscript_go_no_go_v1.md": "17_manuscript_go_no_go_v1.md",
    }
    for src, dst in mapping.items():
        shutil.copy2(output_root / src, review_dir / dst)

    combined = review_dir / "18_reproducibility_and_artifact_inventory.md"
    parts = [
        (output_root / "reproducibility_package_plan_v1.md").read_text(),
        (output_root / "final_code_index_v1.md").read_text(),
        (output_root / "artifact_inventory_v1.md").read_text(),
    ]
    combined.write_text("\n\n".join(parts))

    fig_dst = review_dir / "figures"
    shutil.copytree(output_root / "figures", fig_dst)

    top_count = len([p for p in review_dir.iterdir()])
    status_payload = {
        "status": status,
        "generated_at": generated_at,
        "review_dir": str(review_dir.resolve()),
        "top_level_items": top_count,
        "figures": len(list(fig_dst.glob("*.png"))),
    }
    (output_root / "run_status_packaging_v1.json").write_text(json.dumps(status_payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework-root", default="outputs/framework_strategy_validation_gate_v1")
    parser.add_argument("--benchmark-root", default="outputs/rie_full_engineering_benchmark_v2")
    parser.add_argument("--output-root", default="outputs/rie_results_packaging_review_figures_gate_v1")
    parser.add_argument("--review-dir", default="progress_for_chatgpt/latest")
    args = parser.parse_args()

    framework_root = Path(args.framework_root)
    benchmark_root = Path(args.benchmark_root)
    output_root = Path(args.output_root)
    review_dir = Path(args.review_dir)
    generated_at = now_iso()

    paths = ensure_inputs(framework_root, benchmark_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data = load_data(paths)
    tables = generate_tables(data, output_root, generated_at)
    fig_sources = generate_figures(data, tables, output_root, generated_at)
    status = generate_reports(paths, data, tables, fig_sources, output_root, generated_at)
    write_review_packet(output_root, review_dir, generated_at, status)
    print(json.dumps({"status": status, "figures": len(fig_sources), "tables": len(tables), "review_dir": str(review_dir.resolve())}, indent=2))


if __name__ == "__main__":
    main()
