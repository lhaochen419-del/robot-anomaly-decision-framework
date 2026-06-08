from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "rie_latex_template_draft_v17_official_elsarticle"
DST = ROOT / "outputs" / "rie_latex_template_draft_v18_official_elsarticle_tablefit"
NOW = datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)

    tables = DST / "tables"

    write(
        tables / "table_main_framework_strategy.tex",
        r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2pt}
\renewcommand{\arraystretch}{1.05}
\caption{Summary of fixed-model and framework strategies. Oracle-Best-Test-Utility is a theoretical upper bound and is not deployable.}
\label{tab:main-framework-strategy}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}X*{6}{>{\centering\arraybackslash}p{0.082\linewidth}}@{}}
\toprule
Strategy & Macro-F1 & PR-AUC & FAR & MDR & Avg. rank & Avg. regret \\
\midrule
Framework-Balanced & 0.773 & 0.861 & 0.174 & 0.256 & 2.675 & 0.115 \\
Framework-Safety & 0.738 & 0.858 & 0.322 & 0.154 & 6.390 & 0.136 \\
Framework-Low-False-Alarm & 0.750 & 0.859 & 0.087 & 0.365 & 6.408 & 0.127 \\
Framework-Deployment & 0.770 & 0.859 & 0.174 & 0.261 & 3.117 & 0.118 \\
Fixed-LightGBM & 0.785 & 0.877 & 0.169 & 0.240 & 3.998 & 0.110 \\
Fixed-XGBoost & 0.761 & 0.861 & 0.182 & 0.270 & 8.100 & 0.131 \\
Fixed-RandomForest & 0.744 & 0.835 & 0.189 & 0.291 & 7.554 & 0.149 \\
Oracle-Best-Test-Utility & 0.808 & 0.875 & 0.168 & 0.207 & 1.768 & 0.089 \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    write(
        tables / "table_far_mdr_constraint.tex",
        r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.05}
\caption{FAR/MDR constraint violation rates. Lower violation rates are better; safety and low-false-alarm strategies intentionally trade one error type against the other.}
\label{tab:far-mdr-constraints}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}X*{5}{>{\centering\arraybackslash}p{0.095\linewidth}}@{}}
\toprule
Strategy & FAR viol. & MDR viol. & Joint viol. & FAR & MDR \\
\midrule
Framework-Balanced & 0.080 & 0.230 & 0.290 & 0.174 & 0.256 \\
Framework-Safety & 0.370 & 0.110 & 0.480 & 0.322 & 0.154 \\
Framework-Low-False-Alarm & 0.030 & 0.420 & 0.450 & 0.087 & 0.365 \\
Framework-Deployment & 0.070 & 0.220 & 0.270 & 0.174 & 0.261 \\
Fixed-LightGBM & 0.053 & 0.179 & 0.232 & 0.169 & 0.240 \\
Fixed-XGBoost & 0.074 & 0.179 & 0.242 & 0.182 & 0.270 \\
Fixed-RandomForest & 0.095 & 0.242 & 0.337 & 0.189 & 0.291 \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    write(
        tables / "table_latency_deployment.tex",
        r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.05}
\caption{Prepared-input prediction latency and deployment summary. Deployment utility combines detection metrics with explicit latency and size penalties. Latency values are not end-to-end robotic-cell latency.}
\label{tab:latency-deployment}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}X*{5}{>{\centering\arraybackslash}p{0.092\linewidth}}@{}}
\toprule
Strategy & Latency ms & Size MB & Train s & Deploy util. & Macro-F1 \\
\midrule
Framework-Deployment & 0.050 & 0.972 & 0.158 & 0.284 & 0.770 \\
Fixed-LightGBM & 0.006 & 1.000 & 0.153 & 0.326 & 0.785 \\
Fixed-XGBoost & 0.006 & 1.000 & 0.123 & 0.259 & 0.761 \\
Fixed-RandomForest & 0.197 & 1.000 & 0.228 & 0.202 & 0.744 \\
Fixed-AutoEncoder & 0.002 & 0.453 & 0.112 & -0.368 & 0.548 \\
Fixed-LSTM-AE & 0.004 & 0.046 & 0.195 & -0.463 & 0.485 \\
Fixed-USAD & 0.004 & 0.850 & 0.156 & -0.387 & 0.548 \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    write(
        tables / "table_far95_recall_core.tex",
        r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.8pt}
\renewcommand{\arraystretch}{1.05}
\caption{Core FAR@95\%Recall summary. The metric reports the false-alarm burden needed to reach a high-recall operating point and is reported from existing strategy summaries rather than used for test-set threshold selection.}
\label{tab:far95-core}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}X*{4}{>{\centering\arraybackslash}p{0.105\linewidth}}@{}}
\toprule
Strategy & \shortstack{FAR@95\%\\Recall} & Std. & Macro-F1 & PR-AUC \\
\midrule
Framework-Balanced & 0.480 & 0.346 & 0.773 & 0.861 \\
Framework-Safety & 0.479 & 0.348 & 0.738 & 0.858 \\
Framework-Low-False-Alarm & 0.488 & 0.342 & 0.750 & 0.859 \\
Framework-Deployment & 0.485 & 0.345 & 0.770 & 0.859 \\
Fixed-LightGBM & 0.482 & 0.325 & 0.785 & 0.877 \\
Fixed-XGBoost & 0.515 & 0.309 & 0.761 & 0.861 \\
Fixed-RandomForest & 0.517 & 0.340 & 0.744 & 0.835 \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    write(
        tables / "table_label_budget_summary.tex",
        r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.6pt}
\renewcommand{\arraystretch}{1.05}
\caption{Core label-budget sensitivity results. Normal-only calibration has no anomaly validation labels and therefore cannot use validation macro-F1 threshold tuning. Framework-Label-Budget selected the same model-threshold pairs as Framework-Balanced in these label-budget scenarios, so the rows are merged to avoid implying independent results.}
\label{tab:label-budget-core}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.34\linewidth}>{\RaggedRight\arraybackslash}p{0.12\linewidth}*{4}{>{\centering\arraybackslash}p{0.09\linewidth}}@{}}
\toprule
Strategy & Label budget & Macro-F1 & PR-AUC & FAR & MDR \\
\midrule
Fixed-LightGBM & 5pct & 0.632 & 0.701 & 0.255 & 0.464 \\
Fixed-LightGBM & 20pct & 0.720 & 0.828 & 0.236 & 0.322 \\
Fixed-LightGBM & full & 0.887 & 0.955 & 0.115 & 0.111 \\
Framework-Balanced / Label-Budget & normal\_\allowbreak{}only & 0.558 & 0.574 & 0.439 & 0.438 \\
Framework-Balanced / Label-Budget & 5pct & 0.639 & 0.712 & 0.197 & 0.503 \\
Framework-Balanced / Label-Budget & 20pct & 0.715 & 0.829 & 0.199 & 0.363 \\
Framework-Balanced / Label-Budget & full & 0.888 & 0.954 & 0.115 & 0.109 \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    write(
        tables / "table_robustness_domain_shift_core.tex",
        r"""\begin{table}[htbp]
\centering
\scriptsize
\setlength{\tabcolsep}{2.4pt}
\renewcommand{\arraystretch}{1.05}
\caption{Core robustness and domain-shift results. Robustness and domain shift are reported separately because the framework audits domain-shift risk rather than solving it.}
\label{tab:robustness-domain-core}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.16\linewidth}>{\RaggedRight\arraybackslash}X*{5}{>{\centering\arraybackslash}p{0.075\linewidth}}@{}}
\toprule
Group & Strategy & Macro-F1 & PR-AUC & FAR & MDR & Utility \\
\midrule
robustness & Fixed-LightGBM & 0.834 & 0.916 & 0.171 & 0.160 & 0.503 \\
robustness & Framework-Balanced & 0.839 & 0.919 & 0.173 & 0.148 & 0.518 \\
robustness & Framework-Robust & 0.839 & 0.919 & 0.173 & 0.149 & 0.517 \\
domain shift & Fixed-LightGBM & 0.631 & 0.768 & 0.225 & 0.443 & -0.037 \\
domain shift & Framework-Balanced & 0.615 & 0.758 & 0.205 & 0.477 & -0.067 \\
domain shift & Framework-Robust & 0.615 & 0.758 & 0.205 & 0.477 & -0.067 \\
\bottomrule
\end{tabularx}
\end{table}
""",
    )

    write(
        tables / "table_statistical_evidence_summary.tex",
        r"""\begin{table}[htbp]
\centering
\tiny
\setlength{\tabcolsep}{1.8pt}
\renewcommand{\arraystretch}{1.04}
\caption{Statistical evidence for prespecified framework-versus-fixed comparisons. Each paired sample is a matched dataset--protocol--seed--scenario comparison; individual windows are not treated as independent samples. Holm-adjusted p-values are computed over the displayed Wilcoxon comparisons. Cluster bootstrap confidence intervals resample dataset--protocol groups and are interpreted as benchmark evidence, not independent field-trial evidence.}
\label{tab:statistical-evidence}
\resizebox{\linewidth}{!}{%
\begin{tabular}{llrrrrrll}
\toprule
Comparison & Objective & $n$ & Mean diff. & Wilcoxon $p$ & Holm $p$ & $d_z$ & Cluster 95\% CI & Result \\
\midrule
Balanced vs Fixed-LightGBM & balanced & 95 & 0.0020 & 0.1371 & 0.1371 & 0.02 & [-0.016, 0.018] & small, not significant \\
Balanced vs Fixed-XGBoost & balanced & 95 & 0.0686 & 2.00e-10 & 9.98e-10 & 0.60 & [0.036, 0.096] & framework advantage \\
Balanced vs Fixed-RandomForest & balanced & 95 & 0.1142 & 9.37e-11 & 5.62e-10 & 0.82 & [0.070, 0.155] & framework advantage \\
Safety vs Fixed-LightGBM & safety & 95 & 0.0706 & 4.55e-08 & 9.10e-08 & 0.35 & [0.012, 0.123] & framework advantage \\
Low-FA vs Fixed-LightGBM & low false alarm & 95 & 0.0946 & 7.01e-10 & 2.10e-09 & 0.61 & [0.065, 0.126] & framework advantage \\
Deployment vs Fixed-XGBoost & deployment & 95 & 0.0603 & 2.14e-10 & 9.98e-10 & 0.56 & [0.027, 0.087] & framework advantage \\
Deployment vs Fixed-RandomForest & deployment & 95 & 0.1175 & 5.88e-11 & 4.11e-10 & 0.85 & [0.073, 0.160] & framework advantage \\
\bottomrule
\end{tabular}}
\end{table}
""",
    )

    report = f"""generated_at: {NOW}
output_path: {DST / 'official_table_layout_adaptation_report_v18.md'}

# Official elsarticle Table Layout Adaptation Report v18

source_draft: {SRC}
output_draft: {DST}
main_text_changed: NO
experimental_values_changed: NO
figures_changed: NO
template_changed: NO

Tables adapted:
- table_main_framework_strategy.tex: switched to tabularx with wrapped strategy column.
- table_far_mdr_constraint.tex: switched to tabularx with compact numeric columns.
- table_latency_deployment.tex: switched to tabularx with compact numeric columns.
- table_far95_recall_core.tex: switched to tabularx and line-wrapped the long FAR@95%Recall header.
- table_label_budget_summary.tex: switched to tabularx and widened the label-budget column for normal_only.
- table_robustness_domain_shift_core.tex: switched to tabularx.
- table_statistical_evidence_summary.tex: resized to line width for official preprint width.

Rationale: Official elsarticle preprint width is narrower than the earlier custom geometry; these changes adapt table layout without changing manuscript text or numerical results.
"""
    write(DST / "official_table_layout_adaptation_report_v18.md", report)


if __name__ == "__main__":
    main()
