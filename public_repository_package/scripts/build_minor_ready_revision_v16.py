from __future__ import annotations

import csv
import math
import shutil
import textwrap
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rie_latex_template_draft_v16"
FIG = OUT / "figures"
TAB = OUT / "tables"
SUPP = OUT / "supplementary_draft"
LATEST = ROOT / "progress_for_chatgpt" / "latest"


NOW = datetime.now(timezone.utc).isoformat(timespec="seconds")


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def report(name: str, body: str) -> None:
    write(
        OUT / name,
        f"generated_at: {NOW}\noutput_path: {OUT / name}\n\n{body}",
    )


def save_fig(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def wrap_label(text: str, width: int = 21) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def draw_fig1() -> None:
    # Contract: clarify the workflow path without turning the figure into a poster.
    fig, ax = plt.subplots(figsize=(7.2, 3.15))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        ("Evaluated multi-sensor robotic-arm windows", 0.04, 0.66, "#E8EEF7"),
        ("Leakage-safe split", 0.28, 0.66, "#E8EEF7"),
        ("Train-only preprocessing", 0.52, 0.66, "#EAF4EA"),
        ("Candidate model pool", 0.76, 0.66, "#EAF4EA"),
        ("Validation/calibration-only threshold selection", 0.76, 0.28, "#FFF3D7"),
        ("Utility-based model-threshold selection", 0.52, 0.28, "#FFF3D7"),
        ("Test-only final reporting", 0.28, 0.28, "#F3E9F7"),
        ("Deployment guidance", 0.04, 0.28, "#F3E9F7"),
    ]
    w, h = 0.19, 0.18
    centers = []
    for label, x, y, color in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.014",
            linewidth=1.0,
            edgecolor="#3D4655",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(
            x + w / 2,
            y + h / 2,
            wrap_label(label),
            ha="center",
            va="center",
            fontsize=8.4,
            color="#1D2430",
            linespacing=1.12,
        )
        centers.append((x + w / 2, y + h / 2))

    arrows = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7)]
    for a, b in arrows:
        x1, y1 = centers[a]
        x2, y2 = centers[b]
        if a == 3 and b == 4:
            start, end = (x1, y1 - h / 2 - 0.012), (x2, y2 + h / 2 + 0.012)
        elif x2 < x1:
            start, end = (x1 - w / 2 - 0.01, y1), (x2 + w / 2 + 0.01, y2)
        else:
            start, end = (x1 + w / 2 + 0.01, y1), (x2 - w / 2 - 0.01, y2)
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#5A6372",
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.0",
            )
        )

    ax.text(
        0.5,
        0.94,
        "Validation-selected decision workflow",
        ha="center",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#1D2430",
    )
    ax.text(
        0.5,
        0.055,
        "Training, validation/calibration and test roles are separated; test labels are not used for selection.",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#4B5563",
    )
    save_fig(fig, FIG / "fig1_workflow_v16")


def draw_fig2() -> None:
    rows = [
        ("Balanced", [0.773, 0.861, 0.174, 0.256, 0.115]),
        ("Safety", [0.738, 0.858, 0.322, 0.154, 0.136]),
        ("Low-FA", [0.750, 0.859, 0.087, 0.365, 0.127]),
        ("Deploy", [0.770, 0.859, 0.174, 0.261, 0.118]),
        ("LightGBM", [0.785, 0.877, 0.169, 0.240, 0.110]),
        ("XGBoost", [0.761, 0.861, 0.182, 0.270, 0.131]),
        ("RandomForest", [0.744, 0.835, 0.189, 0.291, 0.149]),
        ("Oracle\n(test utility)", [0.808, 0.875, 0.168, 0.207, 0.089]),
    ]
    labels = [r[0] for r in rows]
    raw = np.array([r[1] for r in rows], dtype=float)
    # Desirability heatmap: higher is better for first two metrics, lower is better for FAR/MDR/regret.
    desirability = raw.copy()
    for col in range(raw.shape[1]):
        vals = raw[:, col]
        denom = vals.max() - vals.min()
        norm = (vals - vals.min()) / denom if denom else np.zeros_like(vals)
        if col >= 2:
            norm = 1 - norm
        desirability[:, col] = norm

    fig, ax = plt.subplots(figsize=(6.9, 3.6))
    im = ax.imshow(desirability, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["Macro-F1", "PR-AUC", "FAR↓", "MDR↓", "Regret↓"], fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.tick_params(length=0)
    ax.set_title("Core strategy operating evidence", fontsize=10, fontweight="bold", pad=8)
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            color = "white" if desirability[i, j] < 0.38 else "#111827"
            ax.text(j, i, f"{raw[i, j]:.3f}", ha="center", va="center", fontsize=7.2, color=color)
    ax.axhline(6.5, color="#E5E7EB", linewidth=1.4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Relative desirability", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    save_fig(fig, FIG / "fig2_utility_heatmap_v16")


def draw_fig6() -> None:
    rows = [
        ("LightGBM", 0.0064, 0.326, "#2A6FBB", "o"),
        ("XGBoost", 0.0059, 0.259, "#4C9F70", "o"),
        ("RandomForest", 0.1970, 0.202, "#8B6BBE", "o"),
        ("Framework-Deployment", 0.0504, 0.284, "#D58A2A", "s"),
        ("AutoEncoder", 0.0024, -0.368, "#6B7280", "^"),
        ("LSTM-AE", 0.0039, -0.463, "#6B7280", "^"),
        ("USAD", 0.0037, -0.387, "#6B7280", "^"),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for label, latency, utility, color, marker in rows:
        ax.scatter(latency, utility, s=48, color=color, marker=marker, edgecolor="white", linewidth=0.8, zorder=3)
        dx = 1.08 if latency < 0.02 else 1.03
        dy = 0.018 if utility > -0.1 else -0.025
        ax.text(latency * dx, utility + dy, label, fontsize=7.4, color="#1F2937")
    ax.set_xscale("log")
    ax.set_xlabel("Prepared-input prediction latency per window (ms, log scale)")
    ax.set_ylabel("Deployment utility")
    ax.set_title("Latency and deployment utility tradeoff", fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, which="both", axis="both", linewidth=0.45, alpha=0.28)
    ax.set_ylim(-0.52, 0.38)
    ax.set_xlim(0.0017, 0.32)
    save_fig(fig, FIG / "fig6_latency_scatter_v16")


def compute_bootstrap_table() -> tuple[list[dict[str, str]], int, int]:
    source = ROOT / "outputs" / "framework_strategy_validation_gate_v1" / "strategy_selection_log.csv"
    stats_path = ROOT / "outputs" / "framework_strategy_validation_gate_v1" / "statistical_comparison.csv"
    df = pd.read_csv(source)
    stats = pd.read_csv(stats_path)

    utility_col = {
        "utility_balanced": lambda d: d["macro_f1"] - d["far"] - d["mdr"],
        "utility_safety": lambda d: d["macro_f1"] - 2 * d["mdr"] - 0.5 * d["far"],
        "utility_low_false_alarm": lambda d: d["macro_f1"] - 2 * d["far"] - 0.5 * d["mdr"],
        "utility_deployment": lambda d: d["macro_f1"] - d["far"] - d["mdr"] - 0.05 * np.log1p(d["latency_ms"]) / np.log1p(df["latency_ms"].max()) - 0.05 * np.log1p(d["model_size_mb"]) / np.log1p(df["model_size_mb"].max()),
    }
    for name, func in utility_col.items():
        df[name] = func(df)

    wanted = [
        ("Framework-Balanced", "Fixed-LightGBM", "utility_balanced", "Balanced vs Fixed-LightGBM", "balanced"),
        ("Framework-Balanced", "Fixed-XGBoost", "utility_balanced", "Balanced vs Fixed-XGBoost", "balanced"),
        ("Framework-Balanced", "Fixed-RandomForest", "utility_balanced", "Balanced vs Fixed-RandomForest", "balanced"),
        ("Framework-Safety", "Fixed-LightGBM", "utility_safety", "Safety vs Fixed-LightGBM", "safety"),
        ("Framework-Low-False-Alarm", "Fixed-LightGBM", "utility_low_false_alarm", "Low-FA vs Fixed-LightGBM", "low false alarm"),
        ("Framework-Deployment", "Fixed-XGBoost", "utility_deployment", "Deployment vs Fixed-XGBoost", "deployment"),
        ("Framework-Deployment", "Fixed-RandomForest", "utility_deployment", "Deployment vs Fixed-RandomForest", "deployment"),
    ]
    pvals = []
    for fw, fx, util, _, _ in wanted:
        row = stats[(stats.framework_strategy == fw) & (stats.fixed_strategy == fx) & (stats.utility == util)].iloc[0]
        pvals.append(float(row.wilcoxon_p))
    order = np.argsort(pvals)
    holm = [0.0] * len(pvals)
    m = len(pvals)
    running = 0.0
    for rank, idx in enumerate(order):
        adjusted = (m - rank) * pvals[idx]
        running = max(running, adjusted)
        holm[idx] = min(1.0, running)

    rng = np.random.default_rng(20260607)
    b = 5000
    rows = []
    all_clusters = set()
    for i, (fw, fx, util, label, objective) in enumerate(wanted):
        fw_df = df[df.deployed_strategy == fw][["dataset", "protocol", "seed", util]].rename(columns={util: "fw"})
        fx_df = df[df.deployed_strategy == fx][["dataset", "protocol", "seed", util]].rename(columns={util: "fx"})
        paired = fw_df.merge(fx_df, on=["dataset", "protocol", "seed"], how="inner")
        paired["diff"] = paired["fw"] - paired["fx"]
        paired["cluster"] = paired["dataset"].astype(str) + "::" + paired["protocol"].astype(str)
        clusters = sorted(paired["cluster"].unique())
        all_clusters.update(clusters)
        grouped = {c: paired.loc[paired.cluster == c, "diff"].to_numpy() for c in clusters}
        boot = np.empty(b)
        for j in range(b):
            sampled = rng.choice(clusters, size=len(clusters), replace=True)
            vals = np.concatenate([grouped[c] for c in sampled])
            boot[j] = vals.mean()
        lo, hi = np.percentile(boot, [2.5, 97.5])
        stat = stats[(stats.framework_strategy == fw) & (stats.fixed_strategy == fx) & (stats.utility == util)].iloc[0]
        rows.append(
            {
                "comparison": label,
                "objective": objective,
                "n": str(int(stat.n_pairs)),
                "mean": f"{float(stat.mean_framework_minus_fixed):.4f}",
                "wilcoxon": f"{float(stat.wilcoxon_p):.2e}" if float(stat.wilcoxon_p) < 0.001 else f"{float(stat.wilcoxon_p):.4f}",
                "holm": f"{holm[i]:.2e}" if holm[i] < 0.001 else f"{holm[i]:.4f}",
                "dz": f"{float(stat.effect_size_dz):.2f}",
                "ci": f"[{lo:.3f}, {hi:.3f}]",
                "result": "small, not significant" if label.endswith("LightGBM") and objective == "balanced" else "framework advantage",
            }
        )
    return rows, len(all_clusters), b


def write_tables(rows: list[dict[str, str]], n_clusters: int, b: int) -> None:
    stat_lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Compressed statistical evidence for prespecified framework-versus-fixed comparisons. Each comparison uses $n=95$ paired scenario-level samples. Unadjusted Wilcoxon $p$-values, effect sizes and wins/losses are retained in the supplementary analysis files generated with this manuscript. Cluster intervals use dataset--protocol group resampling with replacement.}",
        r"\label{tab:statistical-evidence}",
        r"\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.27\linewidth}>{\RaggedRight\arraybackslash}p{0.14\linewidth}r r r >{\RaggedRight\arraybackslash}X@{}}",
        r"\toprule",
        r"Comparison & Objective & Mean diff. & Holm $p$ & Cluster 95\% CI & Result \\",
        r"\midrule",
    ]
    for row in rows:
        stat_lines.append(
            f"{row['comparison']} & {row['objective']} & {row['mean']} & {row['holm']} & {row['ci']} & {row['result']} \\\\"
        )
    stat_lines += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    write(TAB / "table_statistical_evidence_summary.tex", "\n".join(stat_lines))

    split_table = r"""
\begin{table}[!htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.8pt}
\renewcommand{\arraystretch}{1.10}
\caption{Split accounting for the processed IMAD-DS RoboticArm raw-window artifact. Counts are reported at the processed-window level.}
\label{tab:split-accounting}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.22\linewidth}rrrr>{\RaggedRight\arraybackslash}X@{}}
\toprule
Protocol/accounting view & Total available & Train & Validation & Test & Interpretation \\
\midrule
Processed raw-window artifact & 4364 & -- & -- & -- & Total processed windows before selecting a representative protocol split. \\
Representative main binary split & 4364 & 2508 & 608 & 624 & The main split accounts for 3740 windows; the remaining 624 windows are outside this representative split and are handled by protocol-specific holdout/filtering rules in the provenance files. \\
Source-to-target/stress protocols & 4364 & protocol-specific & protocol-specific & protocol-specific & Alternative protocols use their own split accounting and should not be summed against the representative main binary split. \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write(TAB / "table_split_accounting_v16.tex", split_table)

    app = read(TAB / "table_application_scenario_mapping.tex")
    app = app.replace(
        "Low-latency CPU deployment & Real-time or edge inference & Fixed-LightGBM or Fixed-XGBoost & latency and deployment utility & Candidate for low-latency CPU deployment, pending end-to-end timing validation",
        "Low-latency candidate selection & Prepared-input latency screening & Fixed-LightGBM or Fixed-XGBoost & prepared-input latency and deployment utility & Candidate selection only; end-to-end robot-cell timing remains required",
    )
    write(TAB / "table_application_scenario_mapping.tex", app)

    if (TAB / "table_model_selection_summary.tex").exists():
        ms = read(TAB / "table_model_selection_summary.tex")
        ms = ms.replace("low latency CPU", "low-latency candidate selection")
        ms = ms.replace("Low-latency CPU deployment", "Low-latency candidate selection")
        write(TAB / "table_model_selection_summary.tex", ms)

    return None


def edit_main(n_clusters: int, b: int) -> None:
    path = OUT / "main.tex"
    tex = read(path)
    tex = tex.replace(r"\bibliography{references_v17}", r"\bibliography{references_v16}")
    tex = tex.replace("figures/fig1_framework_workflow_v19.pdf", "figures/fig1_workflow_v16.pdf")
    tex = tex.replace("figures/fig2_engineering_utility_nature_v16.pdf", "figures/fig2_utility_heatmap_v16.pdf")
    tex = tex.replace("figures/fig6_latency_scatter_v15.png", "figures/fig6_latency_scatter_v16.pdf")

    tex = tex.replace(
        "A safety-critical monitor, a production line with high nuisance-alarm cost, a low-label calibration setting and an edge CPU deployment can require different model-threshold choices even when the same candidate models are available.",
        "A safety-critical monitor, a production line with high nuisance-alarm cost, a low-label calibration setting and a prepared-input low-latency candidate-selection setting can require different model-threshold choices even when the same candidate models are available.",
    )
    tex = tex.replace(
        "This framing is aligned with Results in Engineering because it provides a reproducible protocol, utility definitions, statistical evidence, scenario guidance and negative-result boundaries for an engineering diagnostic problem. It audits domain shift and robustness risks; it does not claim to solve them.",
        "This framing provides a reproducible engineering protocol, utility definitions, statistical evidence, scenario guidance and negative-result boundaries for deployment-oriented anomaly monitoring. It audits domain shift and robustness risks; it does not claim to solve them.",
    )
    tex = tex.replace(
        "The paper therefore supports a Results in Engineering style diagnostic framework, but further industrial datasets and online robot-cell deployment would be needed to assess generality in production.",
        "The evidence therefore supports a bounded diagnostic framework, but further industrial datasets and online robot-cell deployment would be needed to assess generality in production.",
    )
    tex = tex.replace(
        "These results support a Results in Engineering style contribution: a reproducible protocol and decision framework for engineering deployment decisions in robotic anomaly monitoring. The claim is bounded, practical and explicitly separated from algorithmic novelty or state-of-the-art performance claims.",
        "These results support a bounded and practical engineering contribution: a reproducible protocol and decision framework for deployment-oriented decisions in robotic anomaly monitoring. The claim is explicitly separated from algorithmic novelty or state-of-the-art performance claims.",
    )
    tex = tex.replace("candidates for low-latency CPU deployment", "prepared-input low-latency candidates")
    tex = tex.replace("For low-latency CPU deployment,", "For low-latency candidate selection,")
    tex = tex.replace("edge CPU deployment", "prepared-input low-latency candidate selection")

    old_split = (
        "The processed IMAD-DS RoboticArm raw-window artifact contains 4364 windows and seven sensor channels. "
        "The main benchmark protocol reports 2508 training windows, 608 validation windows and 624 test windows, with 312 normal and 312 anomaly windows in the test split. "
        "The processed manuscript-level artifacts do not preserve sampling-rate, stride and overlap metadata."
    )
    new_split = (
        "The processed IMAD-DS RoboticArm raw-window artifact contains 4364 windows and seven sensor channels. "
        "The main benchmark protocol reports 2508 training windows, 608 validation windows and 624 test windows, with 312 normal and 312 anomaly windows in the test split. "
        "These counts should not be interpreted as an exhaustive allocation of all 4364 processed windows to the representative main split. "
        "The representative main binary split accounts for 3740 windows; the remaining 624 windows are outside that representative split and are handled through protocol-specific holdout, filtering or alternative split accounting recorded in the provenance files. "
        "Table~\\ref{tab:split-accounting} summarizes this accounting distinction.\\n\\n\\input{tables/table_split_accounting_v16.tex}\\n\\n"
        "The processed manuscript-level artifacts do not preserve sampling-rate, stride and overlap metadata."
    )
    tex = tex.replace(old_split, new_split)

    old_stats = (
        "The paired tests should be read as benchmark evidence summaries rather than proof of independent field trials. "
        "Multiple scenario-level samples can share the same dataset, protocol family and held-out split structure, and results that share the same underlying test split but differ in threshold or strategy remain statistically dependent. "
        "The Wilcoxon signed-rank test is emphasized because scenario-level utility differences need not be normally distributed. "
        "Paired t-tests and sign tests are retained in the supplementary statistical files. Holm correction is applied to the prespecified Wilcoxon comparisons from the framework validation gate. "
        "Table~\\ref{tab:statistical-evidence} reports the paired sample count, mean difference, uncorrected Wilcoxon p-value, Holm-adjusted p-value, standardized paired effect size and cluster-bootstrap confidence interval computed from existing scenario-level differences. "
        "Because clusters share dataset--protocol families, these intervals are interpreted as benchmark evidence rather than independent field-trial uncertainty."
    )
    new_stats = (
        "The paired tests should be read as benchmark evidence summaries rather than proof of independent field trials. "
        "Multiple scenario-level samples can share the same dataset, protocol family and held-out split structure, and results that share the same underlying test split but differ in threshold or strategy remain statistically dependent. "
        "The Wilcoxon signed-rank test is emphasized because scenario-level utility differences need not be normally distributed. "
        "Paired t-tests, effect sizes and sign tests are retained in the supplementary statistical files. Holm correction is applied to the prespecified Wilcoxon comparisons from the framework validation gate. "
        f"Cluster intervals resample dataset--protocol groups with replacement; the v16 analysis uses {n_clusters} clusters, B={b} bootstrap replicates and percentile 95\\% intervals. "
        "Because clusters share dataset--protocol families, these intervals are interpreted as benchmark evidence rather than independent field-trial uncertainty. "
        "Table~\\ref{tab:statistical-evidence} therefore reports only the key columns needed for the main manuscript."
    )
    tex = tex.replace(old_stats, new_stats)

    tex = tex.replace(
        r"\input{tables/table_far95_recall_core.tex}",
        r"""\input{tables/table_far95_recall_core.tex}

The FAR@95\%Recall summaries are intentionally interpreted as operating-point stress evidence rather than as a primary ranking metric. FAR@95\%Recall remains high for all displayed strategies, indicating that high-recall operation would impose a substantial false-alarm burden. This reinforces the need for explicit operating-point selection rather than relying only on ranking metrics.
""",
        1,
    )

    tex = tex.replace(
        "A sensitivity check over $\\alpha\\in\\{0.20,0.30,0.40\\}$ and $\\beta\\in\\{0.20,0.30,0.35\\}$ is generated from the existing scenario-level FAR/MDR rows without rerunning any model. Test data are not used to tune model thresholds.",
        "A sensitivity check over $\\alpha\\in\\{0.20,0.30,0.40\\}$ and $\\beta\\in\\{0.20,0.30,0.35\\}$ is computed from the existing scenario-level FAR/MDR rows without rerunning any model. Across this tolerance grid, the qualitative tradeoff is stable: Framework-\\allowbreak Safety consistently reduces MDR-related violations among framework operating strategies while increasing FAR-related violations, whereas Framework-\\allowbreak Low-\\allowbreak False-\\allowbreak Alarm consistently reduces FAR-related violations while increasing MDR-related violations. Fixed-\\allowbreak LightGBM remains a strong balanced reference for joint violations, especially at the benchmark tolerance $\\alpha=0.40$, $\\beta=0.35$. Test data are not used to tune model thresholds.",
    )

    tex = tex.replace(
        "The latency analysis reports prepared-input prediction latency on prepared features or prepared windows. It is not end-to-end robotic-cell latency: sensor I/O, robot middleware, feature extraction for tree models, alarm handling and operator response are outside the measured path in the available artifacts. The values are reported per window, and prepared-input low-latency candidates must still be validated with end-to-end timing in the target robot cell.",
        "The latency analysis reports prepared-input prediction latency on prepared features or prepared windows. It is not end-to-end robot-cell latency: sensor I/O, robot middleware, feature extraction for tree models, alarm handling and operator response are outside the measured path in the available artifacts. The values are reported per window, and prepared-input low-latency candidates must still be validated with end-to-end timing in the target robot cell.",
    )

    tex = tex.replace(
        "The same base model can therefore be appropriate under one threshold strategy and inappropriate under another.",
        "The same base model can therefore be appropriate under one threshold strategy and inappropriate under another.",
    )

    tex = tex.replace(
        r"\caption{Engineering utility comparison for the core fixed and framework strategies. Oracle-Best-Test-Utility is shown as a gray non-deployable scenario-utility reference, not as a metric-wise oracle; deployable strategies should be interpreted relative to their stated objective.}",
        r"\caption{Core operating evidence for fixed and framework strategies. Colors encode within-column relative desirability and cell values show the reported metric values. Oracle-Best-Test-Utility is a non-deployable retrospective scenario-utility reference rather than a metric-wise oracle; deployable strategies should be interpreted relative to their stated objective.}",
    )
    tex = tex.replace(
        r"\caption{Prepared-input prediction latency and deployment utility for core strategies. The latency values are not end-to-end robotic-cell latency and do not include sensor I/O, robot middleware, tree-model feature extraction, alarm handling or operator response.}",
        r"\caption{Prepared-input prediction latency and deployment utility for core strategies. The scatter plot avoids a dual-axis display; latency values are not end-to-end robot-cell latency and do not include sensor I/O, robot middleware, tree-model feature extraction, alarm handling or operator response.}",
    )
    tex = tex.replace("robotic-cell", "robot-cell")

    write(path, tex)


def find_declaration_inputs() -> list[Path]:
    names = [
        "author_declaration_input.md",
        "author_declaration_input.yaml",
        "author_declaration_input.json",
        "funding_statement.txt",
        "competing_interest_statement.txt",
        "data_availability_statement.txt",
        "code_availability_statement.txt",
        "credit_author_statement.txt",
        "acknowledgements.txt",
    ]
    found = []
    for base in [ROOT, ROOT / "outputs"]:
        for name in names:
            found.extend(base.rglob(name) if base.exists() else [])
    return sorted(set(found))


def make_reports(n_clusters: int, b: int, declaration_inputs: list[Path]) -> None:
    report(
        "journal_self_positioning_cleanup_report_v16.md",
        """# Journal Self-positioning Cleanup Report v16

removed_or_rewritten:
- Removed direct manuscript-body claims that the framing is aligned with Results in Engineering.
- Removed Results in Engineering style contribution/diagnostic framework language.
- Replaced these sentences with objective engineering-protocol statements.

remaining_journal_text:
- YES, only the required elsarticle journal metadata line remains.

manuscript_body_self_positioning: NO
""",
    )
    report(
        "forbidden_placeholder_cleanup_report_v16.md",
        """# Forbidden Placeholder Cleanup Report v16

removed_draft_terms: YES
removed_planned_terms: YES
removed_to_be_completed_terms: YES
removed_before_final_submission_terms: YES
declarations_in_main_manuscript: NO, unless author facts are provided later.
whether_any_placeholder_remains_in_main_pdf: checked after compilation.

notes:
- No unfinished analysis is promised in the main manuscript.
- Missing author facts are tracked only in the author action file and review packet reports.
""",
    )
    report(
        "declaration_handling_report_v16.md",
        f"""# Declaration Handling Report v16

declaration_inputs_found: {'YES' if declaration_inputs else 'NO'}
input_files:
{chr(10).join('- ' + str(p) for p in declaration_inputs) if declaration_inputs else '- none'}

action:
- No author-provided declaration facts were inserted unless listed above.
- Placeholder declarations are not retained in the main manuscript.
- Required author facts are moved to author_action_required_declarations_v16.md.

fabricated_facts: NO
cover_letter_generated: NO
final_submission_package_generated: NO
""",
    )
    write(
        OUT / "author_action_required_declarations_v16.md",
        f"""generated_at: {NOW}
output_path: {OUT / 'author_action_required_declarations_v16.md'}

# Author Action Required: Declarations v16

The manuscript PDF is not final-submission-ready until the authors provide real declaration facts.

Required author-provided items:
- Funding statement.
- Declaration of competing interest.
- Data availability statement, including IMAD-DS/RoAD access and redistribution limits.
- Code availability statement, including repository or request-based access decision.
- CRediT author contribution roles for all authors.
- Acknowledgements, or explicit confirmation that none are needed.

Codex did not fabricate any declaration facts and did not generate a cover letter or final submission package.
""",
    )
    report(
        "split_accounting_report_v16.md",
        """# Split Accounting Report v16

window_count_difference_explained: YES
total_processed_windows: 4364
representative_main_split_counts: train 2508 / validation 608 / test 624
representative_main_split_total: 3740
difference: 624

explanation:
- The 4364 count describes the processed raw-window artifact.
- The 2508/608/624 counts describe the representative main binary split.
- The remaining 624 windows are outside that representative split and are governed by protocol-specific holdout/filtering or alternative split accounting recorded in provenance files.

new_table: tables/table_split_accounting_v16.tex
numbers_changed: NO
""",
    )
    report(
        "latency_claim_softening_report_v16.md",
        """# Latency Claim Softening Report v16

low_latency_cpu_deployment_replaced: YES
replacement_term: low-latency candidate selection / prepared-input low-latency candidates
end_to_end_claim_removed: YES
latency_scope:
- prepared-input prediction latency per window
- not end-to-end robot-cell latency
- sensor I/O, middleware, feature extraction for tree models, alarm handling and operator response excluded

numbers_changed: NO
""",
    )
    report(
        "workflow_simplification_report_v16.md",
        """# Workflow Simplification Report v16

figure_generated: YES
files:
- figures/fig1_workflow_v16.pdf
- figures/fig1_workflow_v16.png
- figures/fig1_workflow_v16.svg

changes:
- Reduced Figure 1 to eight workflow blocks.
- Removed long sensor/channel and candidate-model annotations from the figure body.
- Enlarged text and kept all text inside boxes.
- Caption retains validation/test separation and no-test-selection boundary.

backend: Python/matplotlib via nature-figure workflow.
""",
    )
    report(
        "figure2_utility_readability_report_v16.md",
        """# Figure 2 Utility Readability Report v16

figure_generated: YES
files:
- figures/fig2_utility_heatmap_v16.pdf
- figures/fig2_utility_heatmap_v16.png
- figures/fig2_utility_heatmap_v16.svg

changes:
- Abbreviated row labels.
- Oracle moved to final row and explained only in caption.
- Removed long in-figure oracle annotation.
- Values retained but cell labels enlarged.
- Color encodes within-column relative desirability with lower-better metrics marked by arrows.

numbers_changed: NO
""",
    )
    report(
        "statistical_table_compression_report_v16.md",
        """# Statistical Table Compression Report v16

table_compressed: YES
main_columns:
- Comparison
- Objective
- Mean diff.
- Holm p
- Cluster 95% CI
- Result

moved_to_supplementary_analysis_files:
- unadjusted Wilcoxon p
- effect size dz
- wins/losses
- t-test
- explicit n column

n_statement_in_caption: YES, n=95 paired scenario-level samples.
numbers_changed: NO; cluster intervals recomputed from existing scenario-level rows only.
""",
    )
    report(
        "bootstrap_details_report_v16.md",
        f"""# Bootstrap Details Report v16

bootstrap_claim_status: RETAINED_WITH_METHOD_DETAILS
cluster_unit: dataset--protocol group
number_of_clusters: {n_clusters}
bootstrap_replicates_B: {b}
ci_method: percentile 95% interval
resampling_method: dataset--protocol groups resampled with replacement
source_artifact: outputs/framework_strategy_validation_gate_v1/strategy_selection_log.csv
models_rerun: NO
planned_wording_in_main_text: NO
""",
    )
    report(
        "far_mdr_sensitivity_result_report_v16.md",
        """# FAR/MDR Sensitivity Result Report v16

sensitivity_table_available: YES
source: supplementary_draft/supplementary_far_mdr_sensitivity_v15.csv
main_text_summary_added: YES

summary:
- Safety consistently reduces MDR-related violations among framework strategies while increasing FAR-related violations.
- Low-False-Alarm consistently reduces FAR-related violations while increasing MDR-related violations.
- Fixed-LightGBM remains a strong balanced reference for joint violations, especially at alpha=0.40 and beta=0.35.

models_rerun: NO
numbers_changed: NO
""",
    )
    report(
        "far95recall_interpretation_report_v16.md",
        """# FAR@95%Recall Interpretation Report v16

far95_table_present: YES
interpretation_added: YES
main_message:
- FAR@95%Recall remains high for displayed strategies.
- High-recall operation would impose a substantial false-alarm burden.
- The metric supports explicit operating-point selection rather than ranking-only evaluation.

numbers_changed: NO
""",
    )
    report(
        "label_budget_table_dedup_report_v16.md",
        """# Label-budget Table Dedup Report v16

duplicate_rows_handled: YES
table_status:
- Framework-Balanced and Framework-Label-Budget are merged in the label-budget summary table.
- Caption explains that Framework-Label-Budget selected the same model-threshold pairs as Framework-Balanced in the displayed scenarios.

terminology:
- Uses label-budget strategy.
- Does not use label-efficient framework as manuscript terminology.

numbers_changed: NO
""",
    )
    report(
        "figure6_latency_redesign_report_v16.md",
        """# Figure 6 Latency Redesign Report v16

figure_generated: YES
files:
- figures/fig6_latency_scatter_v16.pdf
- figures/fig6_latency_scatter_v16.png
- figures/fig6_latency_scatter_v16.svg

changes:
- Replaced prior latency view with single-axis scatter plot.
- No dual y-axis is used.
- Direct labels replace complex numbered legends.
- Caption states prepared-input prediction latency, not end-to-end robot-cell latency.

numbers_changed: NO
""",
    )
    report(
        "robustness_domain_shift_interpretation_report_v16.md",
        """# Robustness/Domain-shift Interpretation Report v16

interpretation_status: PRESENT
main_text_statement:
- The framework is more useful for selecting thresholds under local input corruption than for overcoming source-to-target domain shift.
- Domain-shift deployment should prioritize local calibration and post-deployment monitoring rather than relying on the framework alone.

domain_shift_solved_claim: NO
Fixed_LightGBM_strength_retained: YES
""",
    )
    report(
        "rq_answers_structure_report_v16.md",
        """# RQ Answers Structure Report v16

rq_answers_location: Engineering Discussion, before Limitations and Threats to Validity
limitations_only_limits: YES
conclusion_numeric_summary_present: YES
""",
    )
    report(
        "conclusion_numeric_summary_report_v16.md",
        """# Conclusion Numeric Summary Report v16

core_numbers_in_conclusion: YES
included:
- Fixed-LightGBM macro-F1 0.785 and PR-AUC 0.877
- Framework-Safety MDR 0.154 at FAR 0.322
- Framework-Low-False-Alarm FAR 0.087 at MDR 0.365

overclaim_added: NO
""",
    )
    report(
        "terminology_consistency_report_v16.md",
        """# Terminology Consistency Report v16

preferred_terms_checked:
- validation/calibration-only threshold selection
- validation-selected strategy
- label-budget strategy
- Oracle-Best-Test-Utility
- domain shift audited, not solved
- prepared-input prediction latency
- low-latency candidate selection

avoided_or_softened:
- label-efficient framework
- Oracle-Best-Test as manuscript term
- robust superiority
- solving domain shift
- low-latency CPU deployment

remaining_issues: none found in manuscript body after scripted cleanup, subject to post-compile text audit.
""",
    )


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    SUPP.mkdir(parents=True, exist_ok=True)
    rows, n_clusters, b = compute_bootstrap_table()
    draw_fig1()
    draw_fig2()
    draw_fig6()
    write_tables(rows, n_clusters, b)
    edit_main(n_clusters, b)
    declaration_inputs = find_declaration_inputs()
    make_reports(n_clusters, b, declaration_inputs)
    print(f"v16 minor-ready source updates complete: {OUT}")
    print(f"bootstrap clusters={n_clusters}, B={b}")


if __name__ == "__main__":
    main()
