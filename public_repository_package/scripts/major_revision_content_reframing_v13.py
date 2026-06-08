from __future__ import annotations

import csv
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "rie_latex_template_draft_v12"
DST = ROOT / "outputs" / "rie_latex_template_draft_v13"
PACK = ROOT / "progress_for_chatgpt" / "latest"
STATS = ROOT / "outputs" / "framework_strategy_validation_gate_v1" / "statistical_comparison.csv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


GENERATED_AT = now()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def report_header(output_path: Path) -> str:
    return f"generated_at: {GENERATED_AT}\noutput_path: {output_path}\n\n"


def replace_between(text: str, start: str, end: str, body: str) -> str:
    i = text.index(start) + len(start)
    j = text.index(end, i)
    return text[:i] + "\n" + body.strip() + "\n" + text[j:]


def remove_figure_block(text: str, filename: str) -> str:
    pattern = re.compile(
        r"\n\\begin\{figure\}\[htbp\]\s*\\centering\s*"
        r"\\includegraphics\[[^\]]+\]\{figures/"
        + re.escape(filename)
        + r"\}.*?\\end\{figure\}\s*\n",
        re.S,
    )
    return pattern.sub("\n", text)


def holm(rows: list[dict[str, str]]) -> dict[int, float]:
    indexed = sorted(
        [(i, float(r["wilcoxon_p"])) for i, r in enumerate(rows)],
        key=lambda x: x[1],
    )
    m = len(indexed)
    raw = [0.0] * m
    for rank, (i, p) in enumerate(indexed):
        raw[rank] = min(1.0, (m - rank) * p)
    monotone = [0.0] * m
    current = 0.0
    for rank, val in enumerate(raw):
        current = max(current, val)
        monotone[rank] = min(1.0, current)
    return {indexed[rank][0]: monotone[rank] for rank in range(m)}


def fmt_p(p: float) -> str:
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


def make_stat_table() -> tuple[str, str]:
    rows = list(csv.DictReader(STATS.open(encoding="utf-8")))
    corrected = holm(rows)
    want = [
        ("Framework-Balanced", "Fixed-LightGBM"),
        ("Framework-Balanced", "Fixed-XGBoost"),
        ("Framework-Balanced", "Fixed-RandomForest"),
        ("Framework-Safety", "Fixed-LightGBM"),
        ("Framework-Low-False-Alarm", "Fixed-LightGBM"),
        ("Framework-Deployment", "Fixed-XGBoost"),
        ("Framework-Deployment", "Fixed-RandomForest"),
    ]
    selected = []
    for fw, fixed in want:
        for i, row in enumerate(rows):
            if row["framework_strategy"] == fw and row["fixed_strategy"] == fixed:
                row = dict(row)
                row["holm_p"] = corrected[i]
                selected.append(row)
                break

    lines = [
        "% generated_at: " + GENERATED_AT,
        "% output_path: " + str(DST / "tables" / "table_statistical_evidence_summary.tex"),
        r"\begin{table}[htbp]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\caption{Statistical evidence for planned framework-versus-fixed comparisons. Each paired sample is a matched dataset--protocol--seed--scenario comparison. Positive mean differences favor the framework strategy; Holm-adjusted p-values are computed over the planned Wilcoxon comparisons in the validation gate.}",
        r"\label{tab:statistical-evidence}",
        r"\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.25\linewidth}>{\RaggedRight\arraybackslash}p{0.13\linewidth}>{\centering\arraybackslash}p{0.055\linewidth}>{\centering\arraybackslash}p{0.09\linewidth}>{\centering\arraybackslash}p{0.09\linewidth}>{\centering\arraybackslash}p{0.08\linewidth}>{\RaggedRight\arraybackslash}X@{}}",
        r"\toprule",
        r"Comparison & Objective & $n$ & Mean diff. & Wilcoxon $p$ & Holm $p$ & Result \\",
        r"\midrule",
    ]
    for r in selected:
        comp = (
            r["framework_strategy"].replace("Framework-", "F-")
            + " vs "
            + r["fixed_strategy"].replace("Fixed-", "Fixed-")
        )
        obj = r["utility"].replace("utility_", "").replace("_", " ")
        mean = f'{float(r["mean_framework_minus_fixed"]):.4f}'
        wp = fmt_p(float(r["wilcoxon_p"]))
        hp = fmt_p(float(r["holm_p"]))
        eff = float(r["effect_size_dz"])
        if r["framework_strategy"] == "Framework-Balanced" and r["fixed_strategy"] == "Fixed-LightGBM":
            result = f"Small, not significant; effect $d_z={eff:.2f}$"
        elif r["framework_strategy"] == "Framework-Safety":
            result = f"Safety utility advantage; $d_z={eff:.2f}$"
        elif r["framework_strategy"] == "Framework-Low-False-Alarm":
            result = f"Low-FA utility advantage; $d_z={eff:.2f}$"
        elif r["framework_strategy"] == "Framework-Deployment":
            result = f"Deployment utility advantage; $d_z={eff:.2f}$"
        else:
            result = f"Framework advantage; $d_z={eff:.2f}$"
        lines.append(
            f"{comp} & {obj} & {r['n_pairs']} & {mean} & {wp} & {hp} & {result} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabularx}",
        r"\end{table}",
        "",
    ]
    summary = "\n".join(
        [
            f"{r['framework_strategy']} vs {r['fixed_strategy']}: n={r['n_pairs']}, "
            f"Wilcoxon p={fmt_p(float(r['wilcoxon_p']))}, Holm p={fmt_p(float(r['holm_p']))}, "
            f"effect={float(r['effect_size_dz']):.3f}"
            for r in selected
        ]
    )
    return "\n".join(lines), summary


def main() -> None:
    if DST.exists():
        shutil.rmtree(DST)
    shutil.copytree(SRC, DST)
    (DST / "build").mkdir(exist_ok=True)

    tex_path = DST / "main.tex"
    tex = tex_path.read_text(encoding="utf-8")

    tex = re.sub(
        r"\\title\{.*?\}",
        r"\\title{Leakage-safe and calibration-aware model-threshold selection for multi-sensor robotic-arm anomaly diagnosis}",
        tex,
        count=1,
    )

    abstract = r"""
Robotic-arm anomaly diagnosis requires leakage-safe evaluation, threshold calibration and FAR/MDR-aware deployment evidence rather than accuracy-only reporting. This paper evaluates a validation-only model-threshold selection framework under segment-, file- and run-level split protocols, train-only preprocessing and calibration-only threshold selection on real multi-sensor robotic-arm and industrial anomaly data. Fixed-\allowbreak LightGBM is a strong default, while Framework-\allowbreak Safety and Framework-\allowbreak Low-\allowbreak False-\allowbreak Alarm improve their corresponding utilities; the balanced-utility difference over Fixed-\allowbreak LightGBM is small and not statistically significant. The framework contributes auditable, operating-objective-specific selection rather than replacement of the strongest fixed baseline, and label-efficiency superiority is not established.
"""
    tex = replace_between(tex, r"\begin{abstract}", r"\end{abstract}", abstract)

    keywords = r"""
Robotic-arm anomaly diagnosis \sep model-threshold selection \sep leakage-safe evaluation \sep threshold calibration \sep false alarm rate \sep missed detection rate \sep deployment-oriented benchmarking
"""
    tex = replace_between(tex, r"\begin{keyword}", r"\end{keyword}", keywords)

    old_intro_tail = """This paper makes five engineering contributions. First, it consolidates robotic-arm anomaly diagnosis around leakage-safe split units, train-only preprocessing and validation-only threshold calibration. Second, it formalizes validation-only model-threshold selection through predeclared engineering utilities for balanced, safety, low-false-alarm, robust, deployment and label-budget objectives. Third, it evaluates fixed-model strategies and framework strategies under the same provenance-tracked protocols, without using test labels for model or threshold choice. Fourth, it centers the analysis on FAR, MDR, operating-constraint violations, regret and deployment latency rather than accuracy alone. Fifth, it translates the results into scenario-specific guidance for safety-critical, low-false-alarm, domain-shift, missing-sensor, label-scarce and low-latency CPU settings. These contributions are engineering contributions: the study does not claim a new anomaly detection algorithm or state-of-the-art model performance."""
    new_intro_tail = r"""
The study is organized around three research questions. \textbf{RQ1} asks whether validation-only model-threshold selection can reduce operating-cost regret compared with fixed model strategies. \textbf{RQ2} asks under which operating objectives the framework improves over Fixed-\allowbreak LightGBM and where Fixed-\allowbreak LightGBM remains the better default. \textbf{RQ3} asks how deployment conclusions change with label budget, missing sensors, domain shift and latency constraints.

This paper makes three engineering contributions. First, it provides a leakage-safe and calibration-aware deployment protocol for robotic-arm anomaly diagnosis, combining segment/file/run-level splits, train-only preprocessing, validation-only thresholding, no test-set threshold tuning and no random overlapping-window split. Second, it evaluates a cost-aware model-threshold selection framework for deployment-oriented anomaly diagnosis, comparing fixed model strategies with validation-selected framework strategies across balanced, safety, low-false-alarm, robust, deployment and label-budget utilities, with FAR/MDR, regret and latency evidence. Third, it provides a reproducible engineering evaluation across label budget, robustness, domain shift and deployment constraints, including main benchmark results, FAR/MDR tradeoffs, label-budget sensitivity, robustness/domain-shift stress, latency analysis, scenario-specific guidance and honest negative findings. The boundary is deliberately explicit: the framework is a validation-calibrated decision protocol over existing detectors and does not claim a new anomaly detection algorithm or state-of-the-art model.
"""
    tex = tex.replace(old_intro_tail, new_intro_tail.strip())

    tex = tex.replace(
        r"The credibility of such an evaluation depends on protocol design.",
        r"The credibility of such an evaluation depends on protocol design and reproducible implementation detail.",
        1,
    )

    tex = tex.replace(
        r"These issues are particularly important for robotic-arm monitoring, where recordings are structured by runs, files, loads and target conditions.",
        r"These issues are particularly important for robotic-arm monitoring, where recordings are structured by runs, files, loads and target conditions, and where domain shift should be audited rather than treated as solved by a title-level claim.",
        1,
    )

    tex = tex.replace(
        r"Threshold calibration is central to deployable anomaly detection.",
        r"Threshold calibration is central to deployable anomaly detection and is closely related to broader calibration and benchmark-reliability concerns in machine learning and time-series anomaly detection \cite{Guo2017Calibration,Wu2021TSADBenchmarksFlawed,Tatbul2018PrecisionRecallTS}.",
        1,
    )
    tex = tex.replace(
        r"Deployment-aware evaluation should consider label budgets, missing sensors, domain shift, latency, model size and the cost of false alarms or missed detections \cite{Sokolova2009PerformanceMeasures,Sculley2015TechnicalDebt,Amershi2019SE4ML}.",
        r"Deployment-aware evaluation should consider label budgets, missing sensors, domain shift, latency, model size and the cost of false alarms or missed detections \cite{Sokolova2009PerformanceMeasures,Elkan2001CostSensitive,Drummond2006CostCurves,Sculley2015TechnicalDebt,Breck2017MLTestScore,Amershi2019SE4ML}.",
        1,
    )

    dataset_input = r"\input{tables/table_dataset_inclusion_rationale.tex}"
    tex = tex.replace(
        "This conservative role assignment avoids over-claiming evidence from datasets that are useful for readiness analysis but not fully aligned with the main robotic-arm diagnostic task.\n",
        "This conservative role assignment avoids over-claiming evidence from datasets that are useful for readiness analysis but not fully aligned with the main robotic-arm diagnostic task.\n\n"
        + dataset_input
        + "\n",
        1,
    )

    implementation = r"""
\subsection{Implementation details and reproducibility}

The benchmark implementation separates raw-window and tabular segment protocols. For raw-window IMAD-DS RoboticArm experiments, tree models use window-level statistical features generated from train-only-normalized multivariate windows, while AutoEncoder, LSTM-\allowbreak AE and USAD consume normalized multivariate windows directly. Segment-level and secondary tabular datasets use numeric columns from the unified CSV adapters after excluding metadata and label fields. Tabular scaling uses a \texttt{StandardScaler} fit on the training split only; raw-window normalization uses train-only channel statistics before validation and test transformations.

The available configuration fixes five random seeds (7, 13, 23, 31 and 42), validation-only thresholding, train-only normalization, and segment/file/run-level split units. Missing-sensor and noise stress settings are implemented as validation/test corruptions for the robustness protocols. Label-budget settings include normal-only, 5\%, 20\% and full validation-label budgets. The anomaly/fault class is treated as the positive class for FAR, MDR, PR-AUC and threshold calibration.

The implemented tree and statistical baselines use the following available settings: LightGBM uses 70 estimators, learning rate 0.06 and 31 leaves; XGBoost uses 60 estimators, maximum depth 4, learning rate 0.06, subsampling 0.9, column subsampling 0.9 and histogram tree construction; RandomForest uses 80 trees, maximum depth 10 and balanced class weights; IsolationForest uses 80 trees. The neural reconstruction baselines use compact benchmark architectures: AutoEncoder with hidden dimension 64 and latent dimension 24, LSTM-\allowbreak AE with hidden dimension 32 and latent dimension 20, and USAD with hidden dimension 80 and latent dimension 24. The available training loop uses AdamW with learning rate $10^{-3}$, three epochs and batch size 128. These settings are intended as reproducible baselines rather than optimized architecture claims.

Candidate thresholds are generated on validation or calibration scores using predeclared threshold strategies, including best-F1, Youden's index, FAR-target thresholds, recall-target thresholds and missed-detection/false-alarm cost ratios. Test labels are never used for threshold selection. Latency is recorded for model prediction on prepared features or windows and reported per window. Full end-to-end latency including upstream feature extraction, robot middleware and alarm handling is not explicitly configured in the available artifacts and should be reported in supplementary reproducibility files before final deployment claims. Reproducibility artifacts include the benchmark runner, framework validation script, configuration file, command manifests, seed list, result tables and output directory map.
"""
    tex = tex.replace(
        r"\section{Benchmark Models and Framework Strategies}",
        implementation.strip() + "\n\n\\section{Benchmark Models and Framework Strategies}",
        1,
    )

    stat_protocol = r"""
Statistical comparisons use paired tests where matched scenario-level results are available. Each paired sample corresponds to one unique dataset--protocol--seed--scenario comparison when the same evaluation unit is available for both strategies. A scenario-level result is defined by the dataset, protocol, seed, metric or utility objective and strategy pair. This definition avoids treating individual windows from the same split as independent statistical samples.

The paired tests should be read as evidence summaries rather than proof of independent field trials. Results that share the same underlying test split but differ in threshold or strategy remain dependent because they are evaluated on the same held-out data. The Wilcoxon signed-rank test is emphasized because scenario-level utility differences need not be normally distributed. Paired t-tests and sign tests are retained in the analysis files. Holm correction is applied to the planned Wilcoxon comparisons from the framework validation gate; Table~\ref{tab:statistical-evidence} reports the paired sample count, uncorrected Wilcoxon p-value, Holm-adjusted p-value and standardized paired effect size where available.
"""
    tex = tex.replace(
        "Statistical comparisons use paired tests where matched scenario-level results are available. The summary includes paired t-tests, Wilcoxon signed-rank tests, sign tests, effect sizes and wins/losses. Non-significant results are reported as non-significant.",
        stat_protocol.strip(),
        1,
    )

    tex = tex.replace(
        r"\section{Results}",
        r"""\section{Results}

The Results section answers the three research questions using the existing benchmark artifacts. RQ1 is addressed by the overall utility, regret and statistical summaries. RQ2 is addressed by FAR/MDR operating constraints and the comparison with Fixed-\allowbreak LightGBM. RQ3 is addressed through label-budget, robustness/domain-shift and latency/deployment analyses. Detailed full tables and dense diagnostic plots are planned for supplementary material so that the main text can emphasize the decision evidence most relevant to deployment.""",
        1,
    )

    tex = tex.replace(
        "The overall comparison shows a useful but bounded engineering pattern.",
        "For RQ1, the overall comparison shows a useful but bounded engineering pattern.",
        1,
    )
    tex = tex.replace(
        "FAR/MDR analysis shows why a single accuracy-oriented winner is insufficient.",
        "For RQ2, FAR/MDR analysis shows why a single accuracy-oriented winner is insufficient.",
        1,
    )
    tex = tex.replace(
        "Label-budget sensitivity remains a limitation.",
        "For RQ3, label-budget sensitivity remains a limitation.",
        1,
    )

    # Move dense diagnostics out of main text while retaining their numerical claims.
    tex = remove_figure_block(tex, "fig2_average_rank.png")
    tex = remove_figure_block(tex, "fig3_macro_f1_regret.png")
    tex = remove_figure_block(tex, "fig8_strategy_selection_map.png")
    tex = tex.replace("\n\\input{tables/table_label_efficiency.tex}\n", "\n")
    tex = tex.replace("\n\\input{tables/table_robustness.tex}\n", "\n")
    tex = tex.replace("\n\\input{tables/table_model_selection_summary.tex}\n", "\n")
    tex = tex.replace(
        "The engineering interpretation is that fixed defaults and scenario-specific strategies answer different questions.",
        "Average-rank, macro-F1 regret and full model-selection heatmap diagnostics are retained in the supplementary material plan. The engineering interpretation is that fixed defaults and scenario-specific strategies answer different questions.",
        1,
    )

    discussion_start = tex.index(r"\section{Engineering Discussion}")
    limitations_start = tex.index(r"\section{Limitations and Threats to Validity}")
    new_discussion = r"""
\section{Engineering Discussion}

The main engineering distinction is between benchmark ranking and deployment decision support. A benchmark can show that Fixed-\allowbreak LightGBM is a strong model, but it does not by itself decide which threshold should be used when missed detections, nuisance alarms, calibration labels, robustness stress and CPU latency have different costs. The framework converts benchmark outputs into a pre-deployment audit: the operating objective is fixed before test evaluation, model-threshold selection is performed on validation data only and the final test metrics document the consequences of that decision.

Frequent selection of LightGBM should therefore be interpreted as an engineering finding, not as a weakness of the framework. A reliable default is valuable in a robot monitoring system, especially when it is low latency and performs well across several metrics. The framework adds value by verifying that default under leakage-safe splits, calibrating its threshold and identifying when a cost-specific strategy is justified. The safety and low-false-alarm strategies provide the clearest examples: reducing MDR is useful for safety-critical inspection workflows but increases FAR, while reducing FAR is useful for production lines sensitive to nuisance stops but increases MDR.

The offline-to-online transition remains a risk. Validation-only calibration is closer to deployment than test-set threshold tuning, but it cannot guarantee behavior under new robot programs, sensor aging, maintenance changes or operator response loops. A practical use of the framework is therefore as a deployment gate. Engineers would define an operational cost, run the leakage-safe protocol on representative calibration data, select the model-threshold pair for that cost and then review FAR, MDR, latency and robustness evidence before field testing.

This framing is aligned with a Results in Engineering contribution. The paper does not ask readers to adopt a proprietary detector. It provides a reproducible protocol, utility definitions, statistical evidence, scenario guidance and negative-result boundaries that help decide whether an anomaly monitor is ready for deployment-oriented evaluation. The framework audits domain shift and robustness risks; it does not claim to solve them.

"""
    tex = tex[:discussion_start] + new_discussion + tex[limitations_start:]

    tex = tex.replace(
        r"\textbf{Data availability:} To be completed before submission. This study uses real datasets and generated result tables; data redistribution must follow each dataset license.\\",
        r"\textbf{Data availability:} To be completed before submission. Public datasets used in this study include IMAD-DS and RoAD, subject to their original access and redistribution terms. Generated result tables and command manifests will be made available according to the authors' repository or request-based access policy.\\",
        1,
    )
    tex = tex.replace(
        r"\textbf{Code availability:} To be completed before submission. The current project scripts and command manifests are listed in the reproducibility notes.\\",
        r"\textbf{Code availability:} To be completed before submission. The analysis scripts, command manifests and configuration files should be made available through a public repository or upon reasonable request; the final access policy must be confirmed by the authors.\\",
        1,
    )

    tex = tex.replace(r"\bibliography{references_v12}", r"\bibliography{references_v13}")

    # Add a compact answer to RQs before conclusion.
    tex = tex.replace(
        r"\section{Conclusion}",
        r"""\subsection{Answers to the research questions}

RQ1 is answered positively but with boundaries: validation-only model-threshold selection reduces operating-cost regret for several objectives relative to weaker fixed strategies, but it does not dominate Fixed-\allowbreak LightGBM on balanced utility. RQ2 shows that Framework-\allowbreak Safety and Framework-\allowbreak Low-\allowbreak False-\allowbreak Alarm improve their corresponding utilities over Fixed-\allowbreak LightGBM, whereas Fixed-\allowbreak LightGBM remains the preferred default for balanced and several deployment-oriented settings. RQ3 shows that label budget, missing sensors, domain shift and latency constraints materially change the deployment recommendation; label-efficiency superiority is not established, and domain shift remains an audited risk.

\section{Conclusion}""",
        1,
    )

    write(tex_path, tex)

    # Tables.
    contribution_table = r"""% generated_at: """ + GENERATED_AT + r"""
% output_path: """ + str(DST / "tables" / "table_contribution_evidence_mapping.tex") + r"""
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.12}
\caption{Contribution-to-evidence mapping after major-revision reframing. Each contribution is tied to manuscript evidence and to a boundary that prevents over-claiming.}
\label{tab:contribution-evidence}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.24\linewidth}>{\RaggedRight\arraybackslash}p{0.32\linewidth}>{\RaggedRight\arraybackslash}p{0.20\linewidth}>{\RaggedRight\arraybackslash}X@{}}
\toprule
Claimed contribution & Evidence in manuscript & Main support & Claim boundary \\
\midrule
Leakage-safe and calibration-aware deployment protocol & Segment/file/run-level split, train-only preprocessing, validation-only thresholding, no random overlapping-window split and no test-set threshold tuning & Data and Protocols; Table~\ref{tab:dataset-inclusion}; protocol risk section; Fig.~\ref{fig:framework-workflow} & Reduces leakage risk but does not remove all dataset bias \\
Cost-aware model-threshold selection for deployment-oriented anomaly diagnosis & Fixed model strategies compared with validation-selected framework strategies under balanced, safety, low-false-alarm, robust, deployment and label-budget utilities & Strategy utility definitions; Procedure~1; Table~\ref{tab:statistical-evidence}; FAR/MDR and latency results & Depends on validation/calibration quality and candidate model pool \\
Reproducible engineering evaluation across deployment constraints & Main benchmark, FAR/MDR tradeoff, label-budget, robustness/domain-shift, latency and scenario guidance with negative findings retained & Results; application scenario mapping; limitations & Does not establish label-efficiency superiority or comprehensive LightGBM superiority \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write(DST / "tables" / "table_contribution_evidence_mapping.tex", contribution_table)

    dataset_table = r"""% generated_at: """ + GENERATED_AT + r"""
% output_path: """ + str(DST / "tables" / "table_dataset_inclusion_rationale.tex") + r"""
\begin{table}[htbp]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.10}
\caption{Dataset inclusion and exclusion rationale. The table separates primary evidence from secondary checks and readiness-only datasets to avoid overextending the claims.}
\label{tab:dataset-inclusion}
\begin{tabularx}{\linewidth}{@{}>{\RaggedRight\arraybackslash}p{0.18\linewidth}>{\RaggedRight\arraybackslash}p{0.30\linewidth}>{\RaggedRight\arraybackslash}p{0.22\linewidth}>{\RaggedRight\arraybackslash}X@{}}
\toprule
Dataset & Used for & Not used for & Rationale \\
\midrule
IMAD-DS RoboticArm & Primary binary detection; source-to-target; leave-target-weight35-out; missing sensor/noise; label budget; latency & Not applicable for the main robotic-arm claim & Multi-sensor robotic-arm data with labels and split units aligned with the task \\
IMAD-DS BrushlessMotor & Secondary industrial validation where adapter and labels support the protocol & Primary robotic-arm claim & Different asset type; useful for secondary industrial checking \\
RoAD & Secondary sanity/stress reference & Main binary claim & Confounding and artifact risks make it unsuitable as primary evidence \\
NIST UR & Readiness/exclusion note where applicable & Primary binary anomaly evidence & Label, access or protocol mismatch with the main binary task \\
KUKA & Readiness/exclusion note where applicable & Primary binary anomaly evidence & Label, access or protocol mismatch with the main binary task \\
\bottomrule
\end{tabularx}
\end{table}
"""
    write(DST / "tables" / "table_dataset_inclusion_rationale.tex", dataset_table)

    stat_table, stat_summary = make_stat_table()
    write(DST / "tables" / "table_statistical_evidence_summary.tex", stat_table)
    write(DST / "updated_statistical_evidence_table_v13.tex", stat_table)

    # References.
    src_bib = SRC / "references_v12.bib"
    if not src_bib.exists():
        src_bib = SRC / "references.bib"
    bib = src_bib.read_text(encoding="utf-8")
    additions = r"""

@article{Wu2021TSADBenchmarksFlawed,
  title={Current Time Series Anomaly Detection Benchmarks Are Flawed and Are Creating the Illusion of Progress},
  author={Wu, Renjie and Keogh, Eamonn J.},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  volume={35},
  number={3},
  pages={2421--2429},
  year={2023},
  doi={10.1109/TKDE.2021.3112126}
}

@inproceedings{Tatbul2018PrecisionRecallTS,
  title={Precision and Recall for Time Series},
  author={Tatbul, Nesime and Lee, Tae Jun and Zdonik, Stan and Alam, Mejbah and Gottschlich, Justin},
  booktitle={Advances in Neural Information Processing Systems},
  volume={31},
  pages={1924--1934},
  year={2018}
}

@inproceedings{Guo2017Calibration,
  title={On Calibration of Modern Neural Networks},
  author={Guo, Chuan and Pleiss, Geoff and Sun, Yu and Weinberger, Kilian Q.},
  booktitle={Proceedings of the 34th International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={70},
  pages={1321--1330},
  year={2017},
  publisher={PMLR},
  url={https://proceedings.mlr.press/v70/guo17a.html}
}

@inproceedings{Breck2017MLTestScore,
  title={The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction},
  author={Breck, Eric and Cai, Shanqing and Nielsen, Eric and Salib, Michael and Sculley, D.},
  booktitle={2017 IEEE International Conference on Big Data},
  year={2017},
  publisher={IEEE},
  url={https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/}
}

@article{Drummond2006CostCurves,
  title={Cost Curves: An Improved Method for Visualizing Classifier Performance},
  author={Drummond, Chris and Holte, Robert C.},
  journal={Machine Learning},
  volume={65},
  number={1},
  pages={95--130},
  year={2006},
  doi={10.1007/s10994-006-8199-5}
}

@inproceedings{Elkan2001CostSensitive,
  title={The Foundations of Cost-Sensitive Learning},
  author={Elkan, Charles},
  booktitle={Proceedings of the Seventeenth International Joint Conference on Artificial Intelligence},
  year={2001}
}
"""
    for key in [
        "Wu2021TSADBenchmarksFlawed",
        "Tatbul2018PrecisionRecallTS",
        "Guo2017Calibration",
        "Breck2017MLTestScore",
        "Drummond2006CostCurves",
        "Elkan2001CostSensitive",
    ]:
        if "{" + key + "," in bib:
            additions = re.sub(r"\n@[\s\S]*?\{" + re.escape(key) + r",[\s\S]*?\n\}\n?", "\n", additions)
    write(DST / "references_v13.bib", bib.rstrip() + "\n" + additions.strip() + "\n")
    write(DST / "references.bib", (DST / "references_v13.bib").read_text(encoding="utf-8"))

    # Reports that do not depend on compilation.
    reports = {
        "contribution_reframing_report_v13.md": """# Contribution Reframing Report v13

- Contributions compressed from five defensive items into three engineering contributions.
- Contribution 1: leakage-safe and calibration-aware deployment protocol.
- Contribution 2: cost-aware validation-only model-threshold selection.
- Contribution 3: reproducible engineering evaluation across label budget, robustness, domain shift and deployment constraints.
- Boundary retained once and clearly: decision protocol over existing detectors, not a new detector or SOTA claim.
""",
        "title_revision_report_v13.md": """# Title Revision Report v13

Selected title: Leakage-safe and calibration-aware model-threshold selection for multi-sensor robotic-arm anomaly diagnosis

Candidate titles considered:
1. Deployment-oriented model-threshold selection for multi-sensor robotic-arm anomaly diagnosis under leakage-safe protocols
2. Leakage-safe and calibration-aware model-threshold selection for multi-sensor robotic-arm anomaly diagnosis
3. A deployment-oriented engineering protocol for multi-sensor robotic-arm anomaly diagnosis with validation-only calibration

Rationale: Option B removes title-level overcommitment to solving domain shift while preserving the central claims: leakage-safe protocol, calibration-aware evaluation and model-threshold selection.
""",
        "implementation_reproducibility_report_v13.md": """# Implementation/Reproducibility Report v13

Added an Implementation details and reproducibility subsection.

Confirmed from available code/config:
- Seeds: 7, 13, 23, 31, 42.
- Split units: segment/file/run-level; no random overlapping-window split.
- Normalization: train-only raw normalizer or train-only StandardScaler.
- Threshold source: validation/calibration only.
- Raw-window tree baselines use window-level statistical features; deep reconstruction baselines use normalized multivariate windows.
- LightGBM/XGBoost/RandomForest/IsolationForest and AE/LSTM-AE/USAD baseline settings are summarized from `scripts/run_engineering_benchmark.py`.
- Threshold candidates include best-F1, Youden, target-FAR, target-recall and cost-ratio thresholds.
- Latency is recorded for model prediction on prepared inputs; full robot-middleware/feature-extraction latency remains a supplementary deployment TODO.

No experimental values were changed.
""",
        "statistical_testing_clarification_report_v13.md": f"""# Statistical Testing Clarification Report v13

Added a Statistical testing protocol paragraph clarifying:
- paired sample unit: dataset--protocol--seed--scenario comparison;
- same-test-split dependency is acknowledged;
- tests are paired evidence summaries, not independent field-trial proof;
- Wilcoxon is emphasized because scenario-level differences may be non-normal;
- paired t-test and sign test remain in analysis files;
- Holm correction is applied over planned Wilcoxon comparisons.

Updated statistical evidence table:

{stat_summary}

multiple_comparison_correction: YES
""",
        "dataset_inclusion_rationale_report_v13.md": """# Dataset Inclusion Rationale Report v13

Added `table_dataset_inclusion_rationale.tex` to Data and Leakage-Safe Protocols.

Datasets covered:
- IMAD-DS RoboticArm: primary binary robotic-arm evidence.
- IMAD-DS BrushlessMotor: secondary industrial validation.
- RoAD: secondary sanity/stress reference.
- NIST UR: readiness/exclusion note only.
- KUKA: readiness/exclusion note only.

No dataset role was inflated beyond existing evidence.
""",
        "figure_table_reduction_report_v13.md": """# Figure/Table Reduction Report v13

Figures kept in main text:
- Workflow figure.
- Engineering utility comparison.
- FAR/MDR constraint violation.
- Label-budget sensitivity.
- Robustness/domain-shift comparison.
- Latency/deployment tradeoff.

Figures moved to supplementary plan:
- Average rank.
- Macro-F1 regret.
- Strategy selection map.

Tables kept in main text:
- Contribution-evidence mapping.
- Dataset inclusion/exclusion rationale.
- Experiment design matrix.
- Statistical evidence summary with n and Holm p.
- Main framework strategy summary.
- FAR/MDR constraint table.
- Latency/deployment table.
- Application scenario mapping.

Tables moved to supplementary plan:
- Full label-budget table.
- Full robustness/domain-shift table.
- Full model-selection summary.

whether_any_numbers_changed: NO
""",
        "discussion_compression_report_v13.md": """# Discussion Compression Report v13

Discussion was compressed to focus on:
- benchmark ranking versus deployment decision support;
- why frequent LightGBM selection is an engineering finding;
- safety and low-false-alarm deployment tradeoffs;
- offline validation to online deployment risk;
- Results in Engineering fit.

Repeated defensive statements and overlap with limitations/conclusion were reduced. Negative results and Fixed-LightGBM strength were retained.
""",
        "abstract_major_revision_report_v13.md": """# Abstract Major Revision Report v13

Abstract rewritten into a concise four-sentence structure:
1. engineering problem;
2. validation-only leakage-safe framework;
3. bounded key results;
4. contribution boundary and significance.

No SOTA or new-algorithm claim was introduced.
""",
        "research_questions_report_v13.md": """# Research Questions Report v13

Added RQ1--RQ3 to the Introduction:
- RQ1: validation-only model-threshold selection and operating-cost regret.
- RQ2: objectives where the framework improves over Fixed-LightGBM versus where LightGBM remains default.
- RQ3: sensitivity to label budget, missing sensors, domain shift and latency.

Added a Results organizing sentence and an explicit `Answers to the research questions` subsection before Conclusion.
""",
        "data_code_availability_placeholder_report_v13.md": """# Data/Code Availability Placeholder Report v13

Updated declaration placeholders without fabricating repository links or policy decisions.

Data availability now identifies public dataset categories and generated result tables while leaving final access policy to authors.
Code availability now identifies scripts, command manifests and configuration files while leaving public repository or request-based access policy to authors.

No GitHub, Zenodo, OSF or DOI was invented.
""",
        "reference_expansion_report_v13.md": """# Reference Expansion Report v13

Added verified references for:
- time-series anomaly detection benchmark pitfalls;
- time-series anomaly evaluation metrics;
- calibration;
- cost-sensitive evaluation;
- production-readiness/deployment-aware ML.

Added keys:
- Wu2021TSADBenchmarksFlawed
- Tatbul2018PrecisionRecallTS
- Guo2017Calibration
- Breck2017MLTestScore
- Drummond2006CostCurves
- Elkan2001CostSensitive

No REF_NEEDED markers were introduced.
""",
    }
    for filename, body in reports.items():
        write(DST / filename, report_header(DST / filename) + body.strip() + "\n")

    response_map = """# Major Revision Response Map v13

| Reviewer concern | Manuscript change made | Location in manuscript | Remaining issue | Need user input? |
|---|---|---|---|---|
| Contribution too defensive / not strong enough | Reframed into three positive engineering contributions | Introduction; Table~\\ref{tab:contribution-evidence} | None for draft | No |
| Implementation details missing | Added implementation/reproducibility subsection with features, preprocessing, baseline settings, thresholds and latency caveat | Data/Methods before model strategies | Full supplementary config table still useful | No |
| Statistical test units unclear | Added paired sample definition and dependency caveat | Evaluation Metrics and Statistical Analysis | None for draft | No |
| Domain shift in title overpromises | Removed `under domain shift` from title | Title and abstract | None | No |
| Dataset inclusion/exclusion rationale missing | Added dataset inclusion rationale table | Data and Protocols | None | No |
| Figures too dense | Moved average rank, regret and strategy-map figures to supplementary plan | Results; supplementary plan | Final supplementary assembly pending | No |
| Too many repetitive tables | Moved full label, robustness and model-selection tables to supplementary plan | Results | Final supplementary assembly pending | No |
| Abstract too long/defensive | Rewritten into four concise sentences | Abstract | None | No |
| Research questions missing | Added RQ1--RQ3 and answer subsection | Introduction; Results/Discussion | None | No |
| Discussion repetitive | Compressed around deployment decision support | Engineering Discussion | None for draft | No |
| Declarations placeholders | Kept placeholders and did not fabricate user facts | Declarations | User facts required | Yes |
| Code/results artifact availability | Made Data/Code availability placeholders more concrete without links | Declarations | Repository policy required | Yes |
| References need stronger deployment/calibration/leakage coverage | Added verified references for benchmark pitfalls, calibration, cost-sensitive evaluation and production readiness | Related Work; references_v13.bib | Human style check before submission | No |
"""
    write(DST / "major_revision_response_map_v13.md", report_header(DST / "major_revision_response_map_v13.md") + response_map)


if __name__ == "__main__":
    main()
