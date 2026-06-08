from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
V15 = ROOT / "outputs" / "rie_latex_template_draft_v15"
V16 = ROOT / "outputs" / "rie_latex_template_draft_v16"
FRAMEWORK = ROOT / "outputs" / "framework_strategy_validation_gate_v1"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def save_pub(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def draw_workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.1, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    colors = {
        "data": "#E8EEF7",
        "train": "#EAF4EC",
        "val": "#FFF4D8",
        "test": "#F8E9E7",
        "edge": "#4C566A",
        "accent": "#2F5597",
    }
    bands = [
        (0.25, 4.85, 9.5, 1.0, "Leakage-safe data partitioning", colors["data"]),
        (0.25, 3.45, 3.0, 1.0, "Training role", colors["train"]),
        (3.55, 3.45, 3.1, 1.0, "Validation/calibration role", colors["val"]),
        (6.95, 3.45, 2.8, 1.0, "Test/reporting role", colors["test"]),
    ]
    for x, y, w, h, label, fc in bands:
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.025,rounding_size=0.06",
                linewidth=0.8,
                edgecolor="#B7C0CF",
                facecolor=fc,
            )
        )
        ax.text(x + 0.12, y + h - 0.18, label, fontsize=7.4, weight="bold", color="#3A4050", va="top")

    def box(x, y, w, h, title, body="", fc="white", lw=1.0):
        ax.add_patch(
            patches.FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.04,rounding_size=0.05",
                linewidth=lw,
                edgecolor=colors["edge"],
                facecolor=fc,
            )
        )
        ax.text(x + 0.12, y + h - 0.18, title, fontsize=8.2, weight="bold", va="top", color="#202530")
        if body:
            ax.text(x + 0.12, y + h - 0.50, body, fontsize=7.3, va="top", color="#303844", linespacing=1.18)

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=1.1, color="#4C566A", shrinkA=4, shrinkB=4),
        )

    box(0.55, 5.05, 2.25, 0.55, "Raw multi-sensor data", "robot-arm windows")
    box(3.15, 5.05, 2.15, 0.55, "Leakage-safe split", "segment / file / run")
    box(5.65, 5.05, 2.05, 0.55, "Train / val / test", "roles fixed upfront")
    arrow(2.82, 5.32, 3.12, 5.32)
    arrow(5.32, 5.32, 5.62, 5.32)

    box(0.55, 3.66, 2.25, 0.58, "Train-only normalization", "fit on training data")
    box(0.55, 2.35, 2.55, 0.88, "Candidate model pool", "LightGBM, XGBoost, RF\nIForest, AE, LSTM-AE, USAD")
    arrow(1.68, 3.62, 1.68, 3.25)

    box(3.75, 3.66, 2.45, 0.58, "Validation-only calibration", "threshold grid and scores")
    box(3.75, 2.35, 2.45, 0.88, "Engineering utility selection", "balanced / safety / low-FA\nrobust / deployment / label budget")
    arrow(5.0, 3.62, 5.0, 3.25)

    box(7.15, 3.66, 2.25, 0.58, "Test-only final evaluation", "FAR, MDR, PR-AUC, regret")
    box(7.15, 2.35, 2.25, 0.88, "Deployment decision", "selected model + threshold\nwith operating caveat")
    arrow(8.28, 3.62, 8.28, 3.25)

    arrow(3.10, 2.80, 3.72, 2.80)
    arrow(6.22, 2.80, 7.12, 2.80)
    arrow(6.22, 3.95, 7.12, 3.95)

    box(3.5, 0.76, 3.0, 0.68, "Model selection guidance", "scenario-specific recommendation; no test-set selection", fc="#F3F6FA")
    arrow(8.28, 2.32, 6.55, 1.10)
    arrow(5.0, 2.32, 5.0, 1.48)

    ax.text(
        0.35,
        0.25,
        "Decision workflow: model and threshold are selected only on validation/calibration data; held-out test data are used once for final reporting.",
        fontsize=7.2,
        color="#4B5563",
    )
    save_pub(fig, V16 / "figures" / "fig1_framework_workflow_nature_v16")
    plt.close(fig)


def draw_utility_heatmap() -> None:
    df = pd.read_csv(FRAMEWORK / "engineering_utility_results.csv")
    rename = {"Oracle-Best-Test": "Oracle-Best-Test-Utility"}
    df["deployed_strategy"] = df["deployed_strategy"].replace(rename)
    strategies = [
        "Fixed-LightGBM",
        "Fixed-XGBoost",
        "Fixed-RandomForest",
        "Framework-Balanced",
        "Framework-Safety",
        "Framework-Low-False-Alarm",
        "Framework-Deployment",
        "Oracle-Best-Test-Utility",
    ]
    cols = [
        ("utility_balanced_mean", "Balanced"),
        ("utility_safety_mean", "Safety"),
        ("utility_low_false_alarm_mean", "Low false alarm"),
        ("utility_deployment_mean", "Deployment"),
    ]
    mat = []
    for s in strategies:
        r = df[df["deployed_strategy"].eq(s)].iloc[0]
        mat.append([float(r[c]) for c, _ in cols])
    mat = np.array(mat)

    fig, ax = plt.subplots(figsize=(7.0, 4.35))
    vmax = max(abs(mat.min()), abs(mat.max()))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([label for _, label in cols], fontsize=8)
    ax.set_yticks(np.arange(len(strategies)))
    ax.set_yticklabels([s.replace("Framework-", "F-").replace("Fixed-", "") for s in strategies], fontsize=8)
    ax.tick_params(length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            color = "white" if abs(mat[i, j]) > 0.32 else "#202530"
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7.3, color=color)
    ax.set_title("Engineering utility by strategy and operating objective", fontsize=10, weight="bold", pad=9)
    ax.set_xlabel("Operating objective")
    ax.set_ylabel("Strategy")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.axhline(6.5, color="#4C566A", lw=0.8, ls="--")
    ax.text(
        3.55,
        7.45,
        "retrospective\nnon-deployable\nreference",
        fontsize=6.7,
        color="#555B66",
        ha="right",
        va="center",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean utility", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.tight_layout()
    save_pub(fig, V16 / "figures" / "fig2_engineering_utility_nature_v16")
    plt.close(fig)


def main() -> None:
    if V16.exists():
        shutil.rmtree(V16)
    shutil.copytree(V15, V16)
    draw_workflow()
    draw_utility_heatmap()

    contribution_table = r"""\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\caption{Contribution-to-evidence mapping. The table summarizes how each engineering contribution is supported by the protocol design, results and discussion.}
\label{tab:contribution-evidence}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.24\linewidth}>{\RaggedRight\arraybackslash}p{0.32\linewidth}>{\RaggedRight\arraybackslash}p{0.20\linewidth}>{\RaggedRight\arraybackslash}X@{}}
\toprule
Contribution & Evidence in manuscript & Main support & Evidence scope \\
\midrule
Leakage-safe and calibration-aware deployment protocol & Segment/file/run-level split, train-only preprocessing, validation-only thresholding, no random overlapping-window split and no test-set threshold tuning & Data and Protocols; Table~\ref{tab:dataset-inclusion}; protocol risk section; Fig.~\ref{fig:framework-workflow} & Evaluated under the reported split/protocol design and dataset roles \\
Cost-aware model-threshold selection for deployment-oriented anomaly diagnosis & Fixed model strategies compared with validation-selected framework strategies under balanced, safety, low-false-alarm, robust, deployment and label-budget utilities & Strategy utility definitions; Procedure~1; Table~\ref{tab:statistical-evidence}; FAR/MDR and latency results & Applies to the candidate model pool and validation/calibration quality used in this study \\
Reproducible engineering evaluation across deployment constraints & Main benchmark, FAR/MDR tradeoff, label-budget, robustness/domain-shift, latency and scenario guidance with negative findings retained & Results; application scenario mapping; limitations & Supports deployment-oriented evidence synthesis across the evaluated datasets and constraints \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write(V16 / "tables" / "table_contribution_evidence_mapping.tex", contribution_table)

    model_table = r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{3.2pt}
\renewcommand{\arraystretch}{1.08}
\caption{Candidate model settings used in the reproducible benchmark implementation. The settings define baseline configurations for model-threshold evaluation rather than optimized architecture claims.}
\label{tab:model-parameters}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.18\linewidth}>{\RaggedRight\arraybackslash}p{0.46\linewidth}>{\RaggedRight\arraybackslash}p{0.16\linewidth}>{\RaggedRight\arraybackslash}X@{}}
\toprule
Model & Main reported settings & Input representation & Training note \\
\midrule
LightGBM & 70 estimators; learning rate 0.06; 31 leaves & Window-level statistical features & Supervised tree baseline \\
XGBoost & 60 estimators; maximum depth 4; learning rate 0.06; subsampling 0.9; column subsampling 0.9; histogram tree construction & Window-level statistical features & Supervised tree baseline \\
RandomForest & 80 trees; maximum depth 10; balanced class weights & Window-level statistical features & Supervised tree baseline \\
IsolationForest & 80 trees & Window-level statistical features & Unsupervised anomaly baseline \\
AutoEncoder & Hidden dimension 64; latent dimension 24; AdamW; learning rate $10^{-3}$; three epochs; batch size 128 & Normalized multivariate windows & Reconstruction baseline; no early-stopping rule explicitly configured \\
LSTM-AE & Hidden dimension 32; latent dimension 20; AdamW; learning rate $10^{-3}$; three epochs; batch size 128 & Normalized multivariate windows & Sequence reconstruction baseline; no early-stopping rule explicitly configured \\
USAD & Hidden dimension 80; latent dimension 24; AdamW; learning rate $10^{-3}$; three epochs; batch size 128 & Normalized multivariate windows & Reconstruction baseline; no early-stopping rule explicitly configured \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write(V16 / "tables" / "table_model_parameters_v16.tex", model_table)

    main_tex = V16 / "main.tex"
    text = main_tex.read_text(encoding="utf-8")
    old_para = (
        "The implemented tree and statistical baselines use the following available settings: LightGBM uses 70 estimators, learning rate 0.06 and 31 leaves; XGBoost uses 60 estimators, maximum depth 4, learning rate 0.06, subsampling 0.9, column subsampling 0.9 and histogram tree construction; RandomForest uses 80 trees, maximum depth 10 and balanced class weights; IsolationForest uses 80 trees. The neural reconstruction baselines use compact benchmark architectures: AutoEncoder with hidden dimension 64 and latent dimension 24, LSTM-\\allowbreak AE with hidden dimension 32 and latent dimension 20, and USAD with hidden dimension 80 and latent dimension 24. The available training loop uses AdamW with learning rate $10^{-3}$, three epochs and batch size 128. No early-stopping rule is explicitly configured for these deep baselines. These settings are intended as reproducible baselines rather than optimized architecture claims."
    )
    new_para = (
        "The candidate model settings are summarized in Table~\\ref{tab:model-parameters}. These settings are reported as reproducible baseline configurations for model-threshold evaluation, not as optimized architecture claims.\n\n"
        "\\input{tables/table_model_parameters_v16.tex}"
    )
    text = text.replace(old_para, new_para)
    text = text.replace(
        "\\includegraphics[width=0.96\\linewidth]{figures/fig1_framework_workflow_v11.pdf}",
        "\\includegraphics[width=0.96\\linewidth]{figures/fig1_framework_workflow_nature_v16.pdf}",
    )
    text = text.replace(
        "\\includegraphics[width=0.92\\linewidth]{figures/fig2_engineering_utility_core_v14.png}",
        "\\includegraphics[width=0.92\\linewidth]{figures/fig2_engineering_utility_nature_v16.pdf}",
    )
    text = text.replace("\\bibliography{references_v15}", "\\bibliography{references_v16}")
    main_tex.write_text(text, encoding="utf-8")

    ref15 = V16 / "references_v15.bib"
    if ref15.exists():
        shutil.copy2(ref15, V16 / "references_v16.bib")
    report = """# v16 Table/Figure Polish Report

Table 1 caption: revised to a neutral contribution-to-evidence caption.
Table 1 final column: changed from claim-boundary language to evidence scope.
Model parameter table: added as Table~\\ref{tab:model-parameters}; pure prose model settings replaced by a short lead sentence plus table.
Figure 1: redrawn with Python/matplotlib under the Nature figure workflow; exported as PDF/SVG/PNG/TIFF.
Figure 2: redrawn as a compact utility heatmap with direct value labels; exported as PDF/SVG/PNG/TIFF.
Experiments rerun: NO.
Experimental values changed: NO.
"""
    write(V16 / "table_figure_polish_report_v16.md", report)


if __name__ == "__main__":
    main()
