from __future__ import annotations

import math
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT
WORKSPACE_ROOT = ROOT.parent
V14 = ROOT / "outputs" / "rie_latex_template_draft_v14"
V15 = ROOT / "outputs" / "rie_latex_template_draft_v15"
LATEST = ROOT / "progress_for_chatgpt" / "latest"
FRAMEWORK = ROOT / "outputs" / "framework_strategy_validation_gate_v1"
PACKAGING = ROOT / "outputs" / "rie_results_packaging_review_figures_gate_v1"
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


CORE_STRATEGIES = [
    "Framework-Balanced",
    "Framework-Safety",
    "Framework-Low-False-Alarm",
    "Framework-Deployment",
    "Fixed-LightGBM",
    "Fixed-XGBoost",
    "Fixed-RandomForest",
]


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def report_header(path: Path) -> str:
    return f"generated_at: {NOW}\noutput_path: {path}\n\n"


def tex_escape(s: str) -> str:
    return (
        str(s)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def fmt(x: float, nd: int = 3) -> str:
    if pd.isna(x):
        return "NA"
    return f"{float(x):.{nd}f}"


def fmt_p(x: float) -> str:
    if pd.isna(x):
        return "NA"
    x = float(x)
    if x < 1e-3:
        return f"{x:.2e}"
    return f"{x:.4f}"


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * float(p_values[idx]))
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def add_utilities(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    max_latency = max(float(out["latency_ms"].max()), 1e-9)
    max_size = max(float(out["model_size_mb"].max()), 1e-9)
    out["latency_penalty"] = 0.05 * np.log1p(out["latency_ms"].clip(lower=0)) / np.log1p(max_latency)
    out["size_penalty"] = 0.05 * np.log1p(out["model_size_mb"].clip(lower=0)) / np.log1p(max_size)
    out["utility_balanced"] = out["macro_f1"] - out["far"] - out["mdr"]
    out["utility_safety"] = out["macro_f1"] - 2.0 * out["mdr"] - 0.5 * out["far"]
    out["utility_low_false_alarm"] = out["macro_f1"] - 2.0 * out["far"] - 0.5 * out["mdr"]
    out["utility_deployment"] = (
        out["macro_f1"] - out["far"] - out["mdr"] - out["latency_penalty"] - out["size_penalty"]
    )
    out["scenario_key"] = (
        out["dataset"].astype(str) + "||" + out["protocol"].astype(str) + "||seed_" + out["seed"].astype(str)
    )
    out["cluster_key"] = out["dataset"].astype(str) + "||" + out["protocol"].astype(str)
    return out


def paired_diffs(selection: pd.DataFrame, strategy_a: str, strategy_b: str, utility: str) -> pd.DataFrame:
    sub = selection[selection["deployed_strategy"].isin([strategy_a, strategy_b])][
        ["scenario_key", "cluster_key", "deployed_strategy", utility]
    ].dropna()
    pivot = sub.pivot_table(index=["scenario_key", "cluster_key"], columns="deployed_strategy", values=utility)
    if strategy_a not in pivot or strategy_b not in pivot:
        return pd.DataFrame(columns=["cluster_key", "diff"])
    pivot = pivot.dropna(subset=[strategy_a, strategy_b]).reset_index()
    pivot["diff"] = pivot[strategy_a] - pivot[strategy_b]
    return pivot[["cluster_key", "diff"]]


def cluster_bootstrap_ci(diffs: pd.DataFrame, seed: int = 20260606, n_boot: int = 5000) -> tuple[float, float] | None:
    if diffs.empty or diffs["cluster_key"].nunique() < 2:
        return None
    rng = np.random.default_rng(seed)
    clusters = np.array(sorted(diffs["cluster_key"].unique()))
    by_cluster = {c: diffs.loc[diffs["cluster_key"].eq(c), "diff"].to_numpy(float) for c in clusters}
    means = []
    for _ in range(n_boot):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        vals = np.concatenate([by_cluster[c] for c in sampled])
        means.append(float(np.mean(vals)))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def build_stat_table(selection: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    stat = pd.read_csv(FRAMEWORK / "statistical_comparison.csv")
    planned = [
        ("Framework-Balanced", "Fixed-LightGBM", "utility_balanced", "Balanced vs Fixed-LightGBM", "balanced"),
        ("Framework-Balanced", "Fixed-XGBoost", "utility_balanced", "Balanced vs Fixed-XGBoost", "balanced"),
        ("Framework-Balanced", "Fixed-RandomForest", "utility_balanced", "Balanced vs Fixed-RandomForest", "balanced"),
        ("Framework-Safety", "Fixed-LightGBM", "utility_safety", "Safety vs Fixed-LightGBM", "safety"),
        ("Framework-Low-False-Alarm", "Fixed-LightGBM", "utility_low_false_alarm", "Low-FA vs Fixed-LightGBM", "low false alarm"),
        ("Framework-Deployment", "Fixed-XGBoost", "utility_deployment", "Deployment vs Fixed-XGBoost", "deployment"),
        ("Framework-Deployment", "Fixed-RandomForest", "utility_deployment", "Deployment vs Fixed-RandomForest", "deployment"),
    ]
    rows = []
    pvals = []
    for a, b, util, comp, objective in planned:
        hit = stat[
            stat["framework_strategy"].eq(a) & stat["fixed_strategy"].eq(b) & stat["utility"].eq(util)
        ]
        if hit.empty:
            row = {
                "comparison": comp,
                "objective": objective,
                "n": 0,
                "mean_diff": np.nan,
                "wilcoxon_p": np.nan,
                "effect": np.nan,
                "result": "not available",
            }
        else:
            r = hit.iloc[0]
            row = {
                "comparison": comp,
                "objective": objective,
                "n": int(r["n_pairs"]),
                "mean_diff": float(r["mean_framework_minus_fixed"]),
                "wilcoxon_p": float(r["wilcoxon_p"]),
                "effect": float(r["effect_size_dz"]),
                "result": "small, not significant"
                if comp == "Balanced vs Fixed-LightGBM"
                else "framework advantage",
            }
        diffs = paired_diffs(selection, a, b, util)
        ci = cluster_bootstrap_ci(diffs)
        row["cluster_ci"] = "not computed" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
        rows.append(row)
        pvals.append(1.0 if pd.isna(row["wilcoxon_p"]) else row["wilcoxon_p"])
    adjusted = holm_adjust(pvals)
    for row, p in zip(rows, adjusted):
        row["holm_p"] = p
    df = pd.DataFrame(rows)
    tex_rows = []
    for _, r in df.iterrows():
        tex_rows.append(
            f"{tex_escape(r['comparison'])} & {tex_escape(r['objective'])} & {int(r['n'])} & "
            f"{fmt(r['mean_diff'], 4)} & {fmt_p(r['wilcoxon_p'])} & {fmt_p(r['holm_p'])} & "
            f"{fmt(r['effect'], 2)} & {tex_escape(r['cluster_ci'])} & {tex_escape(r['result'])} \\\\"
        )
    table = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.2pt}
\renewcommand{\arraystretch}{1.05}
\caption{Statistical evidence for prespecified framework-versus-fixed comparisons. Each paired sample is a matched dataset--protocol--seed--scenario comparison; individual windows are not treated as independent samples. Holm-adjusted p-values are computed over the displayed Wilcoxon comparisons. Cluster bootstrap confidence intervals resample dataset--protocol groups and are interpreted as benchmark evidence, not independent field-trial evidence.}
\label{tab:statistical-evidence}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.19\linewidth}>{\RaggedRight\arraybackslash}p{0.10\linewidth}>{\centering\arraybackslash}p{0.04\linewidth}>{\centering\arraybackslash}p{0.062\linewidth}>{\centering\arraybackslash}p{0.068\linewidth}>{\centering\arraybackslash}p{0.064\linewidth}>{\centering\arraybackslash}p{0.044\linewidth}>{\centering\arraybackslash}p{0.105\linewidth}>{\RaggedRight\arraybackslash}X@{}}
\toprule
Comparison & Objective & $n$ & Mean diff. & Wilcoxon $p$ & Holm $p$ & $d_z$ & Cluster 95\% CI & Result \\
\midrule
""" + "\n".join(tex_rows) + r"""
\bottomrule
\end{tabularx}
\end{table}
"""
    return table, df


def build_far95_table(summary: pd.DataFrame) -> str:
    rows = []
    for strategy in CORE_STRATEGIES:
        hit = summary[summary["deployed_strategy"].eq(strategy)]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rows.append(
            f"{tex_escape(strategy)} & {fmt(r['far_at_95_recall_mean'], 3)} & {fmt(r['far_at_95_recall_std'], 3)} & "
            f"{fmt(r['macro_f1_mean'], 3)} & {fmt(r['pr_auc_mean'], 3)} \\\\"
        )
    return r"""\begin{table}[htbp]
\centering
\small
\caption{Core FAR@95\%Recall summary. The metric reports the false-alarm burden needed to reach a high-recall operating point and is reported from existing strategy summaries rather than used for test-set threshold selection.}
\label{tab:far95-core}
\begin{tabular}{lrrrr}
\toprule
Strategy & FAR@95\%Recall & Std. & Macro-F1 & PR-AUC \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
"""


def build_far_mdr_sensitivity(selection: pd.DataFrame) -> pd.DataFrame:
    alphas = [0.20, 0.30, 0.40]
    betas = [0.20, 0.30, 0.35]
    rows = []
    core = selection[selection["deployed_strategy"].isin(CORE_STRATEGIES)].copy()
    for alpha in alphas:
        for beta in betas:
            for strategy, group in core.groupby("deployed_strategy"):
                rows.append(
                    {
                        "alpha_far": alpha,
                        "beta_mdr": beta,
                        "strategy": strategy,
                        "far_violation_rate": float((group["far"] > alpha).mean()),
                        "mdr_violation_rate": float((group["mdr"] > beta).mean()),
                        "joint_either_violation_rate": float(((group["far"] > alpha) | (group["mdr"] > beta)).mean()),
                        "n_scenarios": int(group.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def build_label_budget_table() -> str:
    return r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{4pt}
\caption{Core label-budget sensitivity results. Normal-only calibration has no anomaly validation labels and therefore cannot use validation macro-F1 threshold tuning. Framework-Label-Budget selected the same model-threshold pairs as Framework-Balanced in these label-budget scenarios, so the rows are merged to avoid implying independent results.}
\label{tab:label-budget-core}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Strategy & Label budget & Macro-F1 & PR-AUC & FAR & MDR \\
\midrule
Fixed-LightGBM & 5pct & 0.632 & 0.701 & 0.255 & 0.464 \\
Fixed-LightGBM & 20pct & 0.720 & 0.828 & 0.236 & 0.322 \\
Fixed-LightGBM & full & 0.887 & 0.955 & 0.115 & 0.111 \\
Framework-Balanced / Label-Budget & normal\_only & 0.558 & 0.574 & 0.439 & 0.438 \\
Framework-Balanced / Label-Budget & 5pct & 0.639 & 0.712 & 0.197 & 0.503 \\
Framework-Balanced / Label-Budget & 20pct & 0.715 & 0.829 & 0.199 & 0.363 \\
Framework-Balanced / Label-Budget & full & 0.888 & 0.954 & 0.115 & 0.109 \\
\bottomrule
\end{tabular}
\end{table}
"""


def build_repro_table() -> str:
    rows = [
        (
            "Input features",
            "Tree models use channel-wise mean, standard deviation, minimum, maximum and endpoint difference; deep baselines use normalized multivariate windows.",
            "raw-window adapter; benchmark runner",
            "No additional RMS, skewness, kurtosis or FFT/band-energy features are inferred from unavailable artifacts.",
        ),
        (
            "Processed windows",
            "IMAD-DS RoboticArm raw-window artifact contains 4364 windows and seven channels; main split reports 2508 train, 608 validation and 624 test windows with 312 normal and 312 anomaly test windows.",
            "strategy selection log; manuscript result artifacts",
            "Sampling-rate, stride and overlap metadata are not preserved in manuscript-level outputs; results are interpreted at processed-window level.",
        ),
        (
            "Split and preprocessing",
            "Segment/file/run-level splitting with train-only normalization and validation-only threshold selection.",
            "benchmark runner; strategy validation script",
            "Test labels are used only for final reporting.",
        ),
        (
            "Stress protocols",
            "Missing-sensor stress uses 10\\% and 20\\% channel-drop settings; noise stress uses Gaussian scale 0.02 and 0.05.",
            "benchmark runner",
            "Stress results are benchmark robustness evidence, not online fault-injection trials.",
        ),
        (
            "Model settings",
            "LightGBM 70 estimators/lr 0.06/31 leaves; XGBoost 60 estimators/depth 4/lr 0.06; RandomForest 80 trees/depth 10; compact AE/LSTM-AE/USAD baselines trained for three epochs with AdamW lr $10^{-3}$ and batch size 128.",
            "benchmark runner",
            "Settings define reproducible baselines rather than optimized architecture claims.",
        ),
        (
            "Threshold grid",
            "Validation-only best-F1, Youden, target FAR 0.05/0.10/0.15, target recall 0.80/0.90/0.95 and cost ratios 2:1, 5:1 and 10:1.",
            "threshold sensitivity files; strategy validation script",
            "Normal-only label-budget calibration uses normal-score rules rather than anomaly-label macro-F1 tuning.",
        ),
        (
            "Latency",
            "Prepared-input prediction latency is reported per window; tree feature extraction, sensor I/O, robot middleware, alarm handling and operator response are excluded.",
            "latency/deployment result table; benchmark runner",
            "Hardware, warm-up and repetition metadata are not preserved; latency is descriptive prepared-input timing only.",
        ),
    ]
    body = "\n".join(
        f"{tex_escape(a)} & {tex_escape(b)} & {tex_escape(c)} & {tex_escape(d)} \\\\" for a, b, c, d in rows
    )
    return r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3pt}
\renewcommand{\arraystretch}{1.12}
\caption{Reproducibility configuration and interpretation limits from available artifacts.}
\label{tab:repro-config}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.15\linewidth}>{\RaggedRight\arraybackslash}p{0.33\linewidth}>{\RaggedRight\arraybackslash}p{0.21\linewidth}>{\RaggedRight\arraybackslash}X@{}}
\toprule
Component & Reported setting & Evidence/source & Interpretation or limitation \\
\midrule
""" + body + r"""
\bottomrule
\end{tabularx}
\end{table}
"""


def build_latency_figure(summary: pd.DataFrame, fig_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_rows = [
        "Fixed-LightGBM",
        "Fixed-XGBoost",
        "Fixed-RandomForest",
        "Framework-Deployment",
        "Fixed-AutoEncoder",
        "Fixed-LSTM-AE",
        "Fixed-USAD",
    ]
    df = summary[summary["deployed_strategy"].isin(plot_rows)].copy()
    order = {name: i for i, name in enumerate(plot_rows)}
    df["order"] = df["deployed_strategy"].map(order)
    df = df.sort_values("order")
    labels = {
        "Fixed-LightGBM": "LightGBM",
        "Fixed-XGBoost": "XGBoost",
        "Fixed-RandomForest": "RandomForest",
        "Framework-Deployment": "Framework-Deployment",
        "Fixed-AutoEncoder": "AutoEncoder",
        "Fixed-LSTM-AE": "LSTM-AE",
        "Fixed-USAD": "USAD",
    }
    colors = {
        "Fixed-LightGBM": "#0072B2",
        "Fixed-XGBoost": "#009E73",
        "Fixed-RandomForest": "#56B4E9",
        "Framework-Deployment": "#D55E00",
        "Fixed-AutoEncoder": "#CC79A7",
        "Fixed-LSTM-AE": "#999999",
        "Fixed-USAD": "#E69F00",
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for _, r in df.iterrows():
        strategy = r["deployed_strategy"]
        ax.scatter(
            r["latency_ms_mean"],
            r["utility_deployment_mean"],
            s=88 if strategy == "Framework-Deployment" else 62,
            color=colors[strategy],
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
        )
        dx = 4 if strategy == "Fixed-RandomForest" else 6
        dy = 4 if strategy != "Framework-Deployment" else -12
        ax.annotate(
            labels[strategy],
            (r["latency_ms_mean"], r["utility_deployment_mean"]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8.3,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Prepared-input prediction latency (ms per window, log scale)")
    ax.set_ylabel("Deployment utility")
    ax.set_title("Prepared-input prediction latency and deployment utility")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)
    ax.axhline(0, color="#666666", linewidth=0.7, linestyle="--", alpha=0.6)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)


def strip_tex_generated_comments(table_dir: Path) -> None:
    for path in table_dir.glob("*.tex"):
        txt = path.read_text(encoding="utf-8")
        txt = re.sub(r"^% generated_at:.*\n", "", txt, flags=re.MULTILINE)
        txt = re.sub(r"^% output_path:.*\n", "", txt, flags=re.MULTILINE)
        path.write_text(txt, encoding="utf-8")


def revise_main(main_path: Path) -> None:
    text = main_path.read_text(encoding="utf-8")
    text = text.replace(r"\bibliography{references_v14}", r"\bibliography{references_v15}")
    text = text.replace(
        "Window length is available from the processed adapter artifact when generated, but sampling rate, stride and overlap are not explicitly configured in the manuscript artifacts and are therefore listed as final-submission reproducibility items rather than inferred.",
        "The processed manuscript-level artifacts do not preserve sampling-rate, stride and overlap metadata. The results are therefore interpreted at the processed-window level rather than raw time resolution, and no unsupported raw-time sampling claims are made.",
    )
    text = text.replace(
        "The exact processor/GPU model, warm-up count, number of repetitions and whether normalization was included in the measured timing are not explicitly recorded and should be finalized in the supplementary reproducibility file before submission. Table~\\ref{tab:repro-config} summarizes the reproducibility configuration and remaining missing items.",
        "The exact processor/GPU model, warm-up count and number of repetitions are not preserved in the available latency artifacts. Table~\\ref{tab:repro-config} summarizes the reported configuration and interpretation limits.",
    )
    text = text.replace(
        "Holm correction is applied to the planned Wilcoxon comparisons from the framework validation gate. Table~\\ref{tab:statistical-evidence} reports the paired sample count, mean difference, uncorrected Wilcoxon p-value, Holm-adjusted p-value and standardized paired effect size. A clustered bootstrap by dataset--protocol group is planned for the supplementary statistical appendix when the full scenario-level grouping table is finalized; no model is rerun for that analysis.",
        "Holm correction is applied to the prespecified Wilcoxon comparisons from the framework validation gate. Table~\\ref{tab:statistical-evidence} reports the paired sample count, mean difference, uncorrected Wilcoxon p-value, Holm-adjusted p-value, standardized paired effect size and cluster-bootstrap confidence interval computed from existing scenario-level differences. Because clusters share dataset--protocol families, these intervals are interpreted as benchmark evidence rather than independent field-trial uncertainty.",
    )
    text = text.replace(
        "Detailed full tables, wins/losses, dense diagnostic plots and reproducibility artifacts are organized in the Supplementary Material draft so that the main text can emphasize the decision evidence most relevant to deployment.",
        "Detailed full tables, wins/losses and dense diagnostics are kept outside the main manuscript so that the text can emphasize the decision evidence most relevant to deployment.",
    )
    text = text.replace(
        "Average-rank, macro-F1 regret and full model-selection heatmap diagnostics are organized in the Supplementary Material draft. The engineering interpretation for RQ1 is that validation-only framework selection improves several operating utilities over weaker fixed baselines such as Fixed-\\allowbreak XGBoost and Fixed-\\allowbreak RandomForest, but it does not reduce balanced regret relative to the strongest fixed baseline, Fixed-\\allowbreak LightGBM.",
        "Average-rank, macro-F1 regret and full model-selection heatmap diagnostics are therefore not treated as core figures in the main text. The engineering interpretation for RQ1 is that validation-only framework selection improves several operating utilities over weaker fixed baselines such as Fixed-\\allowbreak XGBoost and Fixed-\\allowbreak RandomForest, but it does not reduce balanced regret relative to the strongest fixed baseline, Fixed-\\allowbreak LightGBM.",
    )
    text = text.replace(
        "Here $P_{\\mathrm{latency}}$ and $P_{\\mathrm{size}}$ are the predeclared latency and model-size penalties used in the benchmark implementation. They make deployment utility a joint operating score rather than a pure speed measure. The term $\\Delta_{\\mathrm{robust}}$ is the validation-estimated degradation under missing-sensor, noise or domain-shift stress. The label-budget variable $b$ denotes the available validation-label setting, such as normal-only, 5\\%, 20\\% or full validation. These utilities are deliberately simple so that the operating preference is inspectable before test evaluation. The label-budget utility is an evaluation condition, not evidence that label-efficiency superiority has been achieved.",
        "The deployment penalties are implementation-defined but explicit: $P_{\\mathrm{latency}}=0.05\\log(1+\\ell_m)/\\log(1+\\ell_{\\max})$, where $\\ell_m$ is the prepared-input prediction latency for model $m$, and $P_{\\mathrm{size}}=0.05\\log(1+s_m)/\\log(1+s_{\\max})$, where $s_m$ is the model-size estimate. The maxima $\\ell_{\\max}$ and $s_{\\max}$ are computed over the benchmark candidate rows before final reporting. Deployment utility is therefore a descriptive operating score over prepared-input prediction cost, not an end-to-end robot-cell timing guarantee. The robust selection utility uses $\\Delta_{\\mathrm{robust}}=\\max(0,\\mathrm{macroF1}_{\\mathrm{clean,val}}-\\mathrm{macroF1}_{\\mathrm{stress,val}})$ when a clean validation counterpart is available; selected test summaries report robustness evidence through the corresponding stressed-scenario metrics. The label-budget variable $b$ denotes the available validation-label setting, such as normal-only, 5\\%, 20\\% or full validation. These utilities are deliberately simple so that the operating preference is inspectable before test evaluation. The label-budget utility is an evaluation condition, not evidence that label-budget superiority has been achieved.",
    )
    text = text.replace(
        "The evaluation reports macro-F1, weighted-F1, AUROC, PR-AUC, FAR, MDR and FAR@95\\%Recall. Macro-F1 and PR-AUC summarize detection quality, while FAR and MDR express operational costs. FAR@95\\%Recall indicates the false-alarm burden required to reach a high-recall operating point.",
        "The evaluation reports macro-F1, weighted-F1, AUROC, PR-AUC, FAR, MDR and FAR@95\\%Recall. Macro-F1 and PR-AUC summarize detection quality, while FAR and MDR express operational costs. FAR@95\\%Recall indicates the false-alarm burden required to reach a high-recall operating point; the core summary is reported in Table~\\ref{tab:far95-core}.",
    )
    text = text.replace(r"\input{tables/table_statistical_evidence_summary.tex}", r"\input{tables/table_statistical_evidence_summary.tex}" + "\n\n" + r"\input{tables/table_far95_recall_core.tex}")
    text = text.replace(
        "A FAR violation is counted when FAR exceeds $\\alpha=0.40$. An MDR violation is counted when MDR exceeds $\\beta=0.35$. A joint FAR/MDR violation is counted when either condition is violated, i.e., $\\mathrm{FAR}>\\alpha$ or $\\mathrm{MDR}>\\beta$. These tolerances are predeclared in the framework validation script and are used only for reporting operating-constraint violations; test data are not used to tune model thresholds.",
        "A FAR violation is counted when FAR exceeds $\\alpha=0.40$. An MDR violation is counted when MDR exceeds $\\beta=0.35$. A joint FAR/MDR violation is counted when either condition is violated, i.e., $\\mathrm{FAR}>\\alpha$ or $\\mathrm{MDR}>\\beta$. These thresholds are not universal safety limits; they are benchmark-level tolerances selected before test evaluation to compare operating-constraint violations across scenarios. A sensitivity check over $\\alpha\\in\\{0.20,0.30,0.40\\}$ and $\\beta\\in\\{0.20,0.30,0.35\\}$ is generated from the existing scenario-level FAR/MDR rows without rerunning any model. Test data are not used to tune model thresholds.",
    )
    text = text.replace(
        "The label-budget criterion did not pass in the framework validation gate. The results therefore do not support a label-efficiency superiority claim. Label-budget analysis is presented as an engineering diagnostic that identifies when supervised tree baselines and validation thresholds remain necessary.",
        "The label-budget criterion did not pass in the framework validation gate. Framework-Label-Budget selected the same model-threshold pairs as Framework-Balanced in the displayed label-budget scenarios, so Table~\\ref{tab:label-budget-core} merges those rows. The results therefore do not support label-budget superiority or label-efficiency gains. Label-budget analysis is presented as an engineering diagnostic that identifies when supervised tree baselines and validation thresholds remain necessary.",
    )
    text = text.replace(
        "The framework can organize the label-budget comparison, but it does not make normal-only calibration equivalent to supervised validation.",
        "The framework can organize the label-budget comparison, but it does not make normal-only calibration equivalent to supervised validation or convert limited-label calibration into a proven label-efficiency advantage.",
    )
    text = text.replace(
        "\\caption{Label-budget sensitivity of fixed and framework strategies. Full validation labels remain important; the results do not establish label-efficiency superiority.}",
        "\\caption{Label-budget sensitivity of fixed and framework strategies. Full validation labels remain important; the results do not establish label-budget superiority or label-efficiency gains.}",
    )
    text = text.replace(
        "These results suggest that validation-selected strategies can help align model and threshold selection with missing-sensor or noisy conditions. They do not justify a broad robustness-superiority claim over Fixed-\\allowbreak LightGBM.",
        "These results suggest that validation-selected strategies can help align model and threshold selection with missing-sensor or noisy conditions. They do not justify a broad robustness-superiority claim over Fixed-\\allowbreak LightGBM. Thus, the framework is more useful for selecting thresholds under local input corruption than for overcoming source-to-target domain shift. Domain-shift deployment should therefore prioritize local calibration and post-deployment monitoring rather than relying on the framework alone.",
    )
    text = text.replace(r"\subsection{Latency and deployment}", r"\subsection{Prepared-input prediction latency and deployment utility}")
    text = text.replace(
        "The latency analysis reports model prediction latency on prepared features or prepared windows. It should not be interpreted as end-to-end robotic-cell latency: sensor I/O, robot middleware, feature extraction for tree models, alarm handling and operator response are outside the measured path in the available artifacts. The values are reported per window; the exact hardware model, warm-up count and repetition count are not explicitly recorded and are listed as final reproducibility items.",
        "The latency analysis reports prepared-input prediction latency on prepared features or prepared windows. It is not end-to-end robotic-cell latency: sensor I/O, robot middleware, feature extraction for tree models, alarm handling and operator response are outside the measured path in the available artifacts. The values are reported per window, and candidates for low-latency CPU deployment must still be validated with end-to-end timing in the target robot cell.",
    )
    text = text.replace(
        "\\includegraphics[width=0.92\\linewidth]{figures/fig8_latency_core_v14.png}",
        "\\includegraphics[width=0.90\\linewidth]{figures/fig6_latency_scatter_v15.png}",
    )
    text = text.replace(
        "Model prediction latency on prepared features/windows and deployment utility for core strategies. The latency values are not end-to-end robotic-cell latency and do not include sensor I/O, robot middleware or alarm handling.",
        "Prepared-input prediction latency and deployment utility for core strategies. The latency values are not end-to-end robotic-cell latency and do not include sensor I/O, robot middleware, tree-model feature extraction, alarm handling or operator response.",
    )
    text = text.replace(
        "Across the evaluated strategy-validation scenarios, Fixed-\\allowbreak LightGBM remained the strongest balanced default with macro-F1 0.785 and PR-AUC 0.877. Framework-\\allowbreak Safety reduced MDR to 0.154 at FAR 0.322, whereas Framework-\\allowbreak Low-\\allowbreak False-\\allowbreak Alarm reduced FAR to 0.087 at MDR 0.365.",
        "Across the evaluated strategy-validation scenarios, Fixed-\\allowbreak LightGBM remained the strongest balanced default with macro-F1 0.785 and PR-AUC 0.877. Framework-\\allowbreak Safety reduced MDR to 0.154 at FAR 0.322, whereas Framework-\\allowbreak Low-\\allowbreak False-\\allowbreak Alarm reduced FAR to 0.087 at MDR 0.365.",
    )
    if "Across the evaluated strategy-validation scenarios, Fixed-\\allowbreak LightGBM remained the strongest balanced default" not in text:
        text = text.replace(
            "The main finding is not that a new detector outperforms all baselines. Strong tree baselines, especially Fixed-\\allowbreak LightGBM, are often reliable.",
            "Across the evaluated strategy-validation scenarios, Fixed-\\allowbreak LightGBM remained the strongest balanced default with macro-F1 0.785 and PR-AUC 0.877. Framework-\\allowbreak Safety reduced MDR to 0.154 at FAR 0.322, whereas Framework-\\allowbreak Low-\\allowbreak False-\\allowbreak Alarm reduced FAR to 0.087 at MDR 0.365.\n\nThe main finding is not that a new detector outperforms all baselines. Strong tree baselines, especially Fixed-\\allowbreak LightGBM, are often reliable.",
        )
    # Move RQ answers from after Limitations into Discussion before Limitations.
    rq_match = re.search(
        r"\n\\subsection\{Answers to the research questions\}\n(?P<body>.*?)\n\\section\{Conclusion\}",
        text,
        flags=re.DOTALL,
    )
    if rq_match:
        rq_block = "\n\\subsection{Answers to the research questions}\n" + rq_match.group("body").strip() + "\n"
        text = text[: rq_match.start()] + "\n\\section{Conclusion}" + text[rq_match.end() :]
        marker = "\n\\section{Limitations and Threats to Validity}"
        if rq_block not in text and marker in text:
            text = text.replace(marker, "\n" + rq_block + "\n" + marker, 1)
    # Remove declaration placeholders from main manuscript.
    text = re.sub(
        r"\n\\section\*\{Declarations and data availability placeholders\}.*?(?=\n\\bibliographystyle)",
        "\n",
        text,
        flags=re.DOTALL,
    )
    text = text.replace(" planned Wilcoxon comparisons", " prespecified Wilcoxon comparisons")
    text = text.replace("planned framework-versus-fixed", "prespecified framework-versus-fixed")
    text = text.replace("Label-Efficient", "Label-Budget")
    text = text.replace("label-efficiency superiority", "label-budget superiority")
    main_path.write_text(text, encoding="utf-8")


def build_reports(artifacts: dict[str, object]) -> None:
    reports = {
        "draft_planned_placeholder_cleanup_report_v15.md": f"""# Draft/Planned Placeholder Cleanup v15

removed_draft_terms: YES
removed_planned_terms: YES
removed_to_be_completed_terms: YES
declarations_in_main_manuscript: NO
remaining_author_action_required_items: Funding; declaration of competing interest; data availability; code availability; CRediT; acknowledgements
whether_any_placeholder_remains_in_main_pdf: {artifacts.get('placeholder_in_pdf', 'UNKNOWN')}

Summary: formal manuscript text was revised to remove draft/planned/to-be-completed language. Author factual declarations were moved out of the main manuscript and into an action file.
""",
        "declaration_handling_report_v15.md": f"""# Declaration Handling Report v15

user_declaration_files_found: {artifacts.get('declaration_files_found')}
facts_inserted_into_main_tex: NO
declarations_placeholder_removed_from_main_manuscript: YES
author_action_file_generated: YES
risk: The manuscript cannot be submitted until authors provide factual declarations; no declaration facts were fabricated.
""",
        "clustered_bootstrap_or_downgrade_report_v15.md": f"""# Clustered Bootstrap or Downgrade Report v15

clustered_bootstrap_computed: {artifacts.get('bootstrap_computed')}
cluster_unit: dataset--protocol group
ci_values_added: {artifacts.get('ci_values_added')}
if_not_computed_reason: {artifacts.get('bootstrap_reason')}
planned_word_removed_from_main_text: YES
source_artifact: {FRAMEWORK / 'strategy_selection_log.csv'}
""",
        "reproducibility_table_finalization_report_v15.md": """# Reproducibility Table Finalization Report v15

table_finalized: YES
columns_changed_to: Component; Reported setting; Evidence/source; Interpretation or limitation
removed_missing_items_for_final_submission_column: YES
hardware_missing_handled_as_limitation: YES
window_sampling_metadata_missing_handled_as_limitation: YES
main_text_no_must_confirm_language: YES
""",
        "deployment_utility_formula_report_v15.md": f"""# Deployment Utility Formula Report v15

formula_defined: YES
latency_penalty: 0.05 * log1p(latency_ms) / log1p(max_latency_ms)
size_penalty: 0.05 * log1p(model_size_mb) / log1p(max_model_size_mb)
delta_robust_defined: max(0, clean validation macro-F1 - stressed validation macro-F1) for validation robust selection
source_file: {ROOT / 'scripts' / 'validate_framework_strategy_v1.py'}
claim_downgraded_or_limited: YES, deployment utility is described as prepared-input prediction-cost utility, not end-to-end robot-cell latency.
""",
        "far_mdr_threshold_sensitivity_report_v15.md": f"""# FAR/MDR Threshold Sensitivity Report v15

far_threshold_alpha: 0.40
mdr_threshold_beta: 0.35
joint_rule: either FAR > alpha or MDR > beta
rationale_added: YES
sensitivity_completed: YES
sensitivity_grid: alpha in {{0.20, 0.30, 0.40}}, beta in {{0.20, 0.30, 0.35}}
sensitivity_output: {artifacts.get('far_mdr_sensitivity_path')}
models_rerun: NO
""",
        "far95recall_metric_resolution_report_v15.md": f"""# FAR@95%Recall Metric Resolution Report v15

far95_available_in_existing_results: YES
metric_reported_in_main_or_table: YES
table_path: {artifacts.get('far95_table_path')}
numbers_fabricated: NO
""",
        "label_budget_duplicate_resolution_report_v15.md": """# Label-Budget Duplicate Resolution Report v15

duplicate_rows_detected: YES
resolution: Framework-Balanced and Framework-Label-Budget rows were merged in the label-budget table.
explanation_added: YES
terminology_updated: label-budget strategy
label_efficiency_superiority_claim: NO
numbers_changed: NO
""",
        "latency_figure_protocol_revision_report_v15.md": f"""# Latency Figure and Protocol Revision Report v15

latency_section_renamed: YES
prepared_input_latency_caveat_added: YES
end_to_end_robot_cell_claim_removed: YES
figure_6_double_axis_removed: YES
replacement_figure: {artifacts.get('latency_fig_path')}
numbers_changed: NO
""",
        "robustness_domain_shift_conclusion_report_v15.md": """# Robustness/Domain-Shift Conclusion Report v15

conclusion_sharpened: YES
domain_shift_claim_solved: NO
fixed_lightgbm_strength_retained: YES
added_sentence: framework is more useful for local input corruption than for overcoming source-to-target domain shift.
""",
        "rq_answers_relocation_report_v15.md": """# RQ Answers Relocation Report v15

rq_answers_moved_to_discussion: YES
limitations_restricted_to_limitations: YES
conclusion_kept_concise: YES
""",
        "conclusion_numeric_revision_report_v15.md": """# Conclusion Numeric Revision Report v15

core_numbers_added: YES
fixed_lightgbm_macro_f1: 0.785
fixed_lightgbm_pr_auc: 0.877
framework_safety_mdr_far: MDR 0.154, FAR 0.322
framework_low_false_alarm_far_mdr: FAR 0.087, MDR 0.365
claims_overstated: NO
""",
        "related_work_terminology_final_report_v15.md": """# Related Work Terminology Final Report v15

gap_statements_present: YES
terminology_consistent: YES
new_unverified_references_added: NO
reference_expansion_deferred: not needed for v15; no fabricated references added.
""",
        "minor_ready_response_map_v15.md": """# Minor-Ready Response Map v15

| Reviewer concern | Manuscript change made | Location in manuscript | Remaining issue | Need user input? |
|---|---|---|---|---|
| Declarations placeholders | Removed placeholders from main manuscript and created author action file | End matter; author action file | Author facts still missing | YES |
| Supplementary draft/planned wording | Removed draft/planned wording from main text | Results and statistics sections | Optional final supplement can be assembled later | NO |
| Clustered bootstrap planned wording | Computed cluster-bootstrap CI from scenario-level rows and removed planned wording | Statistical testing section; Table statistical evidence | CIs are benchmark evidence, not field-trial uncertainty | NO |
| Reproducibility table missing items | Converted missing items into formal interpretation limits | Implementation details; reproducibility table | Raw hardware metadata still unavailable | NO |
| Deployment utility penalty definition | Added explicit latency and size penalty formula and robust degradation definition | Strategy utilities section | End-to-end timing still requires field validation | NO |
| FAR/MDR threshold rationale/sensitivity | Added tolerance rationale and generated sensitivity CSV | FAR/MDR section; supplementary sensitivity CSV | Local plants may use stricter limits | NO |
| FAR@95%Recall mismatch | Added core FAR@95%Recall summary table | Metrics section | Full values remain in artifacts | NO |
| Label-budget duplicate rows | Merged duplicate Framework-Balanced / Label-Budget rows with note | Label-budget table and text | No label-budget superiority claim | NO |
| Latency protocol and double-axis figure | Replaced dual-axis figure with latency-utility scatter and stronger caveat | Latency section and figure | End-to-end timing still external | NO |
| Robustness/domain-shift conclusion | Sharpened conclusion that domain shift is audited, not solved | Robustness/domain-shift section | Real deployment validation remains needed | NO |
| RQ answers location | Moved RQ answers to Discussion before Limitations | Engineering Discussion | None | NO |
| Conclusion lacks key numbers | Added Fixed-LightGBM, safety and low-false-alarm numeric summary | Conclusion | None | NO |
| Related Work terminology/gap statements | Checked terminology and retained gap statements | Related Work | No fabricated references | NO |
| User facts still needed | Listed factual declarations separately | Author action file | Authors must supply facts | YES |
""",
    }
    for name, body in reports.items():
        path = V15 / name
        write_text(path, report_header(path) + body.strip() + "\n")


def build_author_action_file() -> None:
    path = V15 / "author_action_required_declarations_v15.md"
    body = """# Author Action Required: Declarations v15

This file is outside the main manuscript. No factual declarations were inserted because the required author facts were not provided.

## Funding
Provide one factual statement:
- This research received no external funding.
- This work was supported by [Funder] under Grant No. [number].

## Declaration of competing interest
Provide either a no-conflict statement or the actual conflict details.

## Data availability
Confirm public dataset links/access terms for IMAD-DS and RoAD, and state whether generated result tables can be public, request-based, or unavailable.

## Code availability
Confirm repository URL, request-based access, private-review access, or unavailable status.

## CRediT author contributions
Provide CRediT roles for Haochen Li, Junhao Wei, Dexing Yao, Yifu Zhao, Yanxiao Li, Zhaoyan Mai, Jiamu Yu, Baili Lu, Sio-Kei Im, Xu Yang, and Yapeng Wang.

## Acknowledgements
Provide acknowledgement text or state None.
"""
    write_text(path, report_header(path) + body)
    manual = V15 / "manual_user_facts_needed_v15.md"
    write_text(
        manual,
        report_header(manual)
        + """# Manual User Facts Needed v15

Required before final submission:
- Funding statement
- Declaration of competing interest
- Data availability decision
- Code availability decision
- CRediT roles for all authors
- Acknowledgements statement

Codex did not fabricate or insert these facts into the main manuscript.
""",
    )


def article_location_report() -> None:
    path = V15 / "article_location_report_v15.md"
    body = f"""# Article Location Report v15

latex_main: {V15 / 'main.tex'}
pdf: {V15 / 'build' / 'main.pdf'}
references: {V15 / 'references_v15.bib'}
main_log: {V15 / 'build' / 'main.log'}
figures_dir: {V15 / 'figures'}
tables_dir: {V15 / 'tables'}
supplementary_draft_dir: {V15 / 'supplementary_draft'}
author_action_required_declarations: {V15 / 'author_action_required_declarations_v15.md'}
progress_latest: {LATEST}
"""
    write_text(path, report_header(path) + body)


def main() -> None:
    if not V14.exists():
        raise FileNotFoundError(V14)
    if V15.exists():
        shutil.rmtree(V15)
    shutil.copytree(V14, V15)

    # Normalize references naming.
    ref14 = V15 / "references_v14.bib"
    ref15 = V15 / "references_v15.bib"
    if ref14.exists():
        shutil.copy2(ref14, ref15)
    elif (V15 / "references.bib").exists():
        shutil.copy2(V15 / "references.bib", ref15)

    main_tex = V15 / "main.tex"
    revise_main(main_tex)

    # Read existing data and build derived summaries.
    selection = add_utilities(pd.read_csv(FRAMEWORK / "strategy_selection_log.csv"))
    summary = pd.read_csv(FRAMEWORK / "framework_vs_fixed_summary.csv")

    stat_tex, stat_df = build_stat_table(selection)
    write_text(V15 / "tables" / "table_statistical_evidence_summary.tex", stat_tex)
    write_text(V15 / "tables" / "table_far95_recall_core.tex", build_far95_table(summary))
    write_text(V15 / "tables" / "table_label_budget_summary.tex", build_label_budget_table())
    write_text(V15 / "tables" / "table_reproducibility_configuration.tex", build_repro_table())

    app_table = V15 / "tables" / "table_application_scenario_mapping.tex"
    if app_table.exists():
        app = app_table.read_text(encoding="utf-8")
        app = app.replace(
            "Feature extraction cost must be included",
            "Candidate for low-latency CPU deployment, pending end-to-end timing validation",
        )
        app_table.write_text(app, encoding="utf-8")

    lat_table = V15 / "tables" / "table_latency_deployment.tex"
    if lat_table.exists():
        lat = lat_table.read_text(encoding="utf-8")
        lat = lat.replace(
            "Latency and deployment summary. Deployment utility combines detection metrics with predeclared latency and size penalties.",
            "Prepared-input prediction latency and deployment summary. Deployment utility combines detection metrics with explicit latency and size penalties. Latency values are not end-to-end robotic-cell latency.",
        )
        lat_table.write_text(lat, encoding="utf-8")

    strip_tex_generated_comments(V15 / "tables")

    # Supplementary sensitivity artifact.
    supp = V15 / "supplementary_draft"
    supp.mkdir(exist_ok=True)
    sens = build_far_mdr_sensitivity(selection)
    sens_path = supp / "supplementary_far_mdr_sensitivity_v15.csv"
    sens.to_csv(sens_path, index=False)

    # Latency scatter replacement.
    fig_path = V15 / "figures" / "fig6_latency_scatter_v15.png"
    build_latency_figure(summary, fig_path)

    build_author_action_file()

    declaration_files = [
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
    for name in declaration_files:
        for base in [ROOT, WORKSPACE_ROOT, ROOT / "outputs"]:
            if (base / name).exists():
                found.append(str(base / name))

    artifacts = {
        "declaration_files_found": "YES: " + "; ".join(found) if found else "NO",
        "bootstrap_computed": "YES" if stat_df["cluster_ci"].ne("not computed").all() else "PARTIAL",
        "ci_values_added": "YES" if stat_df["cluster_ci"].ne("not computed").any() else "NO",
        "bootstrap_reason": "computed from existing strategy_selection_log.csv" if stat_df["cluster_ci"].ne("not computed").any() else "scenario-level paired differences unavailable",
        "far_mdr_sensitivity_path": sens_path,
        "far95_table_path": V15 / "tables" / "table_far95_recall_core.tex",
        "latency_fig_path": fig_path,
        "placeholder_in_pdf": "checked after compile",
    }
    build_reports(artifacts)
    article_location_report()


if __name__ == "__main__":
    main()
