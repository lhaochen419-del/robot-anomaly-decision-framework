#!/usr/bin/env python3
"""Generate RIE manuscript draft v1 from packaged real benchmark results.

This script writes a draft for review only. It does not create a cover letter,
does not claim a new algorithm, and does not add any result beyond existing
CSV/MD artifacts.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


STAGE = "RIE Manuscript Draft v1"
STATUS = "MANUSCRIPT_DRAFT_READY"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_text(path: Path, title: str, generated_at: str, output_root: Path, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- generated_at: {generated_at}",
                f"- output_path: {output_root.resolve()}",
                "- source_type: real",
                "- synthetic: NO",
                "- manuscript_stage: draft_v1_review_only",
                "",
                body.strip(),
                "",
            ]
        )
    )


def row(df: pd.DataFrame, strategy: str) -> pd.Series:
    hit = df[df["deployed_strategy"].eq(strategy)]
    if hit.empty:
        raise KeyError(strategy)
    return hit.iloc[0]


def fmt(x: object, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "TODO"


def load_inputs(input_root: Path) -> dict[str, object]:
    required = [
        "main_framework_strategy_table.csv",
        "engineering_utility_table.csv",
        "far_mdr_constraint_table.csv",
        "label_efficiency_table_final.csv",
        "robustness_table_final.csv",
        "latency_deployment_table_final.csv",
        "model_selection_summary_table.csv",
        "statistical_claims_allowed.md",
        "engineering_findings_v1.md",
        "model_selection_guidelines_manuscript_ready.md",
        "limitations_and_risks_v1.md",
        "contribution_statement_options_v1.md",
        "results_to_figures_mapping_v1.md",
        "manuscript_go_no_go_v1.md",
        "reproducibility_package_plan_v1.md",
        "final_code_index_v1.md",
        "artifact_inventory_v1.md",
    ]
    missing = [str(input_root / f) for f in required if not (input_root / f).exists()]
    if missing:
        raise FileNotFoundError("MISSING_INPUT: " + ", ".join(missing))
    if not (input_root / "figures").exists():
        raise FileNotFoundError(f"MISSING_INPUT: {input_root / 'figures'}")
    return {
        "main": pd.read_csv(input_root / "main_framework_strategy_table.csv"),
        "utility": pd.read_csv(input_root / "engineering_utility_table.csv"),
        "far": pd.read_csv(input_root / "far_mdr_constraint_table.csv"),
        "label": pd.read_csv(input_root / "label_efficiency_table_final.csv"),
        "robust": pd.read_csv(input_root / "robustness_table_final.csv"),
        "latency": pd.read_csv(input_root / "latency_deployment_table_final.csv"),
        "guidance": pd.read_csv(input_root / "model_selection_summary_table.csv"),
        "stats_claims": (input_root / "statistical_claims_allowed.md").read_text(),
        "engineering_findings": (input_root / "engineering_findings_v1.md").read_text(),
        "guidelines_md": (input_root / "model_selection_guidelines_manuscript_ready.md").read_text(),
        "limitations_md": (input_root / "limitations_and_risks_v1.md").read_text(),
        "contribution_md": (input_root / "contribution_statement_options_v1.md").read_text(),
        "mapping_md": (input_root / "results_to_figures_mapping_v1.md").read_text(),
        "go_no_go_md": (input_root / "manuscript_go_no_go_v1.md").read_text(),
        "repro_md": (input_root / "reproducibility_package_plan_v1.md").read_text(),
        "code_md": (input_root / "final_code_index_v1.md").read_text(),
        "artifact_md": (input_root / "artifact_inventory_v1.md").read_text(),
    }


def build_context(data: dict[str, object]) -> dict[str, str]:
    main = data["main"]
    util = data["utility"]
    far = data["far"]
    label = data["label"]
    robust = data["robust"]
    latency = data["latency"]

    lgbm = row(main, "Fixed-LightGBM")
    xgb = row(main, "Fixed-XGBoost")
    rf = row(main, "Fixed-RandomForest")
    fb = row(main, "Framework-Balanced")
    fbf1 = row(main, "Framework-Best-F1")
    fs = row(main, "Framework-Safety")
    flfa = row(main, "Framework-Low-False-Alarm")
    fr = row(main, "Framework-Robust")
    fd = row(main, "Framework-Deployment")
    fle = row(main, "Framework-Label-Efficient")
    oracle = row(main, "Oracle-Best-Test")

    def util_row(s: str) -> pd.Series:
        return row(util, s)

    def far_row(s: str) -> pd.Series:
        return row(far, s)

    def lat_row(s: str) -> pd.Series:
        return row(latency, s)

    full_fb = label[(label["deployed_strategy"].eq("Framework-Balanced")) & (label["label_budget"].eq("full"))]
    five_fb = label[(label["deployed_strategy"].eq("Framework-Balanced")) & (label["label_budget"].eq("5pct"))]
    normal_fb = label[(label["deployed_strategy"].eq("Framework-Balanced")) & (label["label_budget"].eq("normal_only"))]
    rb_fb = robust[(robust["deployed_strategy"].eq("Framework-Balanced")) & (robust["protocol_group"].eq("robustness"))]
    ds_fb = robust[(robust["deployed_strategy"].eq("Framework-Balanced")) & (robust["protocol_group"].eq("domain_shift"))]
    rb_lgbm = robust[(robust["deployed_strategy"].eq("Fixed-LightGBM")) & (robust["protocol_group"].eq("robustness"))]
    ds_lgbm = robust[(robust["deployed_strategy"].eq("Fixed-LightGBM")) & (robust["protocol_group"].eq("domain_shift"))]

    return {
        "lgbm_macro": fmt(lgbm["macro_f1_mean"]),
        "lgbm_pr": fmt(lgbm["pr_auc_mean"]),
        "lgbm_far": fmt(lgbm["far_mean"]),
        "lgbm_mdr": fmt(lgbm["mdr_mean"]),
        "lgbm_joint": fmt(far_row("Fixed-LightGBM")["joint_far_mdr_violation_rate"]),
        "xgb_macro": fmt(xgb["macro_f1_mean"]),
        "rf_macro": fmt(rf["macro_f1_mean"]),
        "fb_macro": fmt(fb["macro_f1_mean"]),
        "fb_pr": fmt(fb["pr_auc_mean"]),
        "fb_far": fmt(fb["far_mean"]),
        "fb_mdr": fmt(fb["mdr_mean"]),
        "fb_rank": fmt(fb["average_rank"], 2),
        "fb_regret": fmt(fb["average_regret"], 3),
        "fb_util": fmt(util_row("Framework-Balanced")["utility_balanced_mean"], 3),
        "fb_lgbm_delta": "0.0020",
        "fb_lgbm_p": "0.137",
        "fbf1_rank": fmt(fbf1["average_rank"], 2),
        "fle_rank": fmt(fle["average_rank"], 2),
        "fr_rank": fmt(fr["average_rank"], 2),
        "fd_rank": fmt(fd["average_rank"], 2),
        "lgbm_rank": fmt(lgbm["average_rank"], 2),
        "xgb_rank": fmt(xgb["average_rank"], 2),
        "rf_rank": fmt(rf["average_rank"], 2),
        "fs_far": fmt(fs["far_mean"]),
        "fs_mdr": fmt(fs["mdr_mean"]),
        "fs_util": fmt(util_row("Framework-Safety")["utility_safety_mean"], 3),
        "fs_delta": "0.0706",
        "fs_p": "4.55e-08",
        "flfa_far": fmt(flfa["far_mean"]),
        "flfa_mdr": fmt(flfa["mdr_mean"]),
        "flfa_util": fmt(util_row("Framework-Low-False-Alarm")["utility_low_false_alarm_mean"], 3),
        "flfa_delta": "0.0946",
        "flfa_p": "7.01e-10",
        "fd_util": fmt(util_row("Framework-Deployment")["utility_deployment_mean"], 3),
        "fr_util": fmt(util_row("Framework-Robust")["utility_robust_mean"], 3),
        "oracle_macro": fmt(oracle["macro_f1_mean"]),
        "oracle_status": "ORACLE_NOT_DEPLOYABLE",
        "full_fb_macro": fmt(full_fb.iloc[0]["macro_f1"]) if not full_fb.empty else "TODO",
        "five_fb_macro": fmt(five_fb.iloc[0]["macro_f1"]) if not five_fb.empty else "TODO",
        "normal_fb_macro": fmt(normal_fb.iloc[0]["macro_f1"]) if not normal_fb.empty else "TODO",
        "rob_fb_macro": fmt(rb_fb.iloc[0]["macro_f1"]) if not rb_fb.empty else "TODO",
        "rob_fb_util": fmt(rb_fb.iloc[0]["utility_robust"]) if not rb_fb.empty else "TODO",
        "ds_fb_macro": fmt(ds_fb.iloc[0]["macro_f1"]) if not ds_fb.empty else "TODO",
        "rob_lgbm_macro": fmt(rb_lgbm.iloc[0]["macro_f1"]) if not rb_lgbm.empty else "TODO",
        "ds_lgbm_macro": fmt(ds_lgbm.iloc[0]["macro_f1"]) if not ds_lgbm.empty else "TODO",
        "lgbm_latency": fmt(lat_row("Fixed-LightGBM")["latency_ms_mean"], 4),
        "xgb_latency": fmt(lat_row("Fixed-XGBoost")["latency_ms_mean"], 4),
        "rf_latency": fmt(lat_row("Fixed-RandomForest")["latency_ms_mean"], 4),
        "fd_latency": fmt(lat_row("Framework-Deployment")["latency_ms_mean"], 4),
        "ae_latency": fmt(lat_row("Fixed-AutoEncoder")["latency_ms_mean"], 4),
    }


def manuscript(ctx: dict[str, str], generated_at: str, output_root: Path) -> str:
    return f"""
# Manuscript Draft v1: Review-only, not a final submission manuscript

- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- source_type: real
- synthetic: NO
- cover letter generated: NO
- final submission abstract generated: NO
- new algorithm claim: NO
- SOTA claim: NO

## Terminology ledger

| Canonical term | Definition used in this draft | Notes |
| --- | --- | --- |
| Framework strategy | A validation-selected choice of candidate model and threshold strategy under a stated engineering utility | Not a new anomaly detection algorithm |
| Fixed model strategy | A deployment policy that uses one model family across all evaluated protocols | Includes Fixed-LightGBM, Fixed-XGBoost, Fixed-RandomForest and others |
| FAR | False alarm rate | Lower is better |
| MDR | Missed detection rate | Lower is better |
| Oracle-Best-Test | The best test-set model per scenario | Theoretical upper bound only; not deployable |
| Leakage-safe split | Split by segment/file/run-level units before windowing or evaluation | Avoids random overlapping-window leakage |

## One-sentence argument

In robotic-arm multi-sensor anomaly diagnosis, we show that a leakage-safe and calibration-aware engineering framework can align model and threshold selection with operational constraints, supported by real benchmark results across fixed tree, statistical and deep baselines, with the boundary that Fixed-LightGBM remains a strong default and the framework is not a new anomaly detection algorithm.

## Title candidates

1. **Preferred:** Leakage-safe and calibration-aware multi-sensor anomaly diagnosis for robotic arms under domain shift
2. Engineering evaluation of model-selection strategies for robotic-arm anomaly detection under operational constraints
3. Calibration-aware benchmark protocols for multi-sensor robotic-arm anomaly diagnosis

## DRAFT Abstract (not final submission abstract)

Multi-sensor robotic-arm monitoring requires anomaly detection systems that remain credible under changing operating conditions, missing sensors and deployment constraints. Accuracy-only evaluation is insufficient for this setting because false alarms and missed detections carry different operational costs, and because random overlapping-window splits or test-set threshold tuning can make benchmark results appear more reliable than they are. This draft reports a leakage-safe and calibration-aware engineering framework for evaluating robotic-arm anomaly diagnosis systems. The framework uses segment-, file- or run-level splits, train-only normalization and validation-only threshold calibration, and compares fixed model strategies with validation-selected framework strategies across real multi-sensor robotic-arm and industrial datasets. The experiments include LightGBM, XGBoost, RandomForest, IsolationForest, AutoEncoder, LSTM-AE and USAD baselines, and evaluate macro-F1, PR-AUC, FAR, MDR, FAR@95%Recall, label budget, domain shift, missing sensors and latency.

The results show that Fixed-LightGBM is a strong default, with mean macro-F1 {ctx['lgbm_macro']}, PR-AUC {ctx['lgbm_pr']}, FAR {ctx['lgbm_far']} and MDR {ctx['lgbm_mdr']} across the strategy validation scenarios. The framework does not comprehensively outperform Fixed-LightGBM. Instead, its value is scenario-specific: Framework-Safety reduced mean MDR to {ctx['fs_mdr']} at the cost of a higher FAR of {ctx['fs_far']}, while Framework-Low-False-Alarm reduced mean FAR to {ctx['flfa_far']} at the cost of a higher MDR of {ctx['flfa_mdr']}. Paired tests indicated significant improvements over Fixed-LightGBM for the safety and low-false-alarm utility functions, but not for balanced, robust or deployment utilities. Label-efficiency superiority was not established. These results support a Results in Engineering style contribution: a reproducible evaluation and decision framework for choosing models and thresholds under explicit engineering constraints, rather than a new anomaly detection algorithm.

## Keywords

Robotic-arm anomaly detection; multi-sensor monitoring; false alarm rate; missed detection rate; threshold calibration; domain shift; engineering benchmark.

## 1. Introduction

Industrial robots and robotic arms are increasingly monitored through multiple sensor streams, including motion, current, vibration, torque and acoustic channels [REF_NEEDED]. These signals can support early detection of collisions, abnormal loads, speed-related faults or degradation, but the monitoring task is not equivalent to ordinary static classification. In practical deployments, a false alarm may interrupt production or trigger unnecessary inspection, while a missed detection may leave a safety or maintenance risk unresolved. A useful evaluation must therefore make false alarm rate (FAR) and missed detection rate (MDR) visible rather than treating all errors as equivalent.

Benchmarking robotic anomaly detectors is also vulnerable to protocol error. Randomly splitting overlapping windows can leak nearly identical temporal segments across train and test sets. Tuning thresholds on test labels can similarly overstate deployable performance. These issues are especially important for robotic data because conditions such as payload, speed, target position and sensor availability can shift the feature distribution even when the underlying task remains anomaly diagnosis. A credible evaluation should split by segment, file or run unit, fit normalization only on training data and calibrate thresholds only on validation or calibration data.

Many anomaly-detection studies compare a small set of models under a single operating point [REF_NEEDED]. Such comparisons can be useful, but they do not answer an engineering deployment question: which model and threshold should be used when the cost of missed detection, false alarm, latency, label scarcity or missing sensors changes? A single fixed model may be the best default in one setting and a poor choice in another. Conversely, a framework that selects among validated candidate models should not be presented as a new detector; its value lies in making model and threshold choice auditable.

This study evaluates a leakage-safe and calibration-aware engineering framework for robotic-arm multi-sensor anomaly diagnosis. The framework compares fixed single-model strategies with validation-selected framework strategies using real benchmark outputs. It explicitly reports FAR, MDR, PR-AUC, regret, average rank, label-budget sensitivity, robustness and latency. The contribution is an engineering diagnostic framework and evidence-based model-selection guideline. It is not a new anomaly detection algorithm and does not claim state-of-the-art performance.

## 2. Related Work

### 2.1 Robotic and industrial anomaly diagnosis

Robotic anomaly diagnosis has been studied for collision detection, fault monitoring, tool-condition tracking and predictive maintenance [REF_NEEDED]. These applications share a deployment constraint: an alarm is not merely a label prediction but an operational decision. Prior work has reported promising results with model-specific pipelines, but many evaluations still depend on dataset-specific splits or accuracy-oriented summaries [REF_NEEDED]. The present work focuses on the evaluation and decision protocol rather than proposing another detector.

### 2.2 Multi-sensor time-series anomaly detection

Multi-sensor time-series methods often exploit correlations among channels, temporal reconstruction error or learned latent representations [REF_NEEDED]. These methods can be effective when abnormal behavior changes the temporal pattern, but they can also be sensitive to condition shift, window construction and threshold selection. This draft therefore treats deep time-series models as benchmark candidates, not as privileged methods.

### 2.3 Tree-based and statistical baselines

Tree-based models, including LightGBM, XGBoost and RandomForest, remain strong baselines for tabular or engineered sensor-window features [REF_NEEDED]. In our results, Fixed-LightGBM remains a particularly strong default. This observation is not hidden or reframed as a weakness of prior work. Instead, it motivates the paper's engineering question: when are fixed tree baselines sufficient, and when does validation-selected strategy selection add value?

### 2.4 Deep reconstruction-based anomaly detection

AutoEncoder, LSTM-AE and USAD-style reconstruction methods are commonly used for unsupervised or weakly supervised anomaly detection [REF_NEEDED]. Their appeal is reduced dependence on anomaly labels. However, the completed benchmark indicated that their detection metrics can be unstable under FAR/MDR constraints. This motivates reporting operating-point behavior rather than only reconstruction quality or aggregate discrimination.

### 2.5 Threshold calibration and FAR/MDR evaluation

Threshold selection is a central part of deployable anomaly detection [REF_NEEDED]. A detector with good ranking metrics can still be unsuitable if its threshold produces excessive false alarms or missed detections. This study uses validation-only threshold calibration and reports sensitivity through FAR, MDR and FAR@95%Recall. Test labels are used only for final evaluation.

### 2.6 Deployment-aware engineering evaluation

Deployment-oriented evaluation should include label budget, missing sensors, domain shift, latency and model size [REF_NEEDED]. These factors change the preferred model family. The framework in this paper formalizes that choice at the strategy level while retaining fixed models as transparent baselines.

## 3. Data and Leakage-Safe Protocols

The primary dataset for the engineering benchmark is IMAD-DS RoboticArm in raw-window form. The benchmark also includes IMAD-DS RoboticArm segment-level features for granularity comparison and IMAD-DS BrushlessMotor as a secondary industrial anomaly dataset. RoAD is retained only as a secondary sanity or stress reference because previous audits identified artifact and confounding risks. NIST UR and KUKA are treated as readiness or optional sources when labels or data availability do not support the same binary anomaly protocols.

All experimental protocols are designed to avoid leakage. Window-level random splitting is not used. Instead, split units are segment, file or run identifiers wherever available. Normalization is fit on training data only. Model thresholds and framework strategy choices are selected using validation or calibration data only. Test labels are not used to choose a model, threshold, score weight or operating point.

The protocol set includes main binary classification, source-to-target domain shift, leave-target-weight35-out stress testing, missing-sensor settings, noise stress, label-efficiency settings and latency benchmarking. Each result row carries provenance fields including dataset, protocol, seed, method, train/validation/test counts, test normal/anomaly counts, generated time and output path. No synthetic result is used in the manuscript evidence.

[INSERT TABLE: dataset and protocol inventory, if prepared for the final manuscript]

## 4. Benchmark Models and Framework Strategies

The fixed model strategies use one model family across all evaluated scenarios. The fixed strategies are Fixed-LightGBM, Fixed-XGBoost, Fixed-RandomForest, Fixed-IsolationForest, Fixed-AutoEncoder, Fixed-LSTM-AE and Fixed-USAD. These models are treated as benchmark candidates. They are not components of a proposed new algorithm.

The framework strategies select a candidate model and threshold using validation or calibration data. Framework-Best-F1 selects the validation macro-F1 optimum. Framework-Balanced maximizes macro-F1 minus FAR and MDR. Framework-Safety increases the penalty on missed detections. Framework-Low-False-Alarm increases the penalty on false alarms. Framework-Robust considers clean performance together with validation robustness degradation. Framework-Deployment includes latency and model-size penalties. Framework-Label-Efficient evaluates normal-only, 5%, 20% and full validation-label budgets.

Oracle-Best-Test is included only as a theoretical upper bound. It uses the best test result per scenario and is therefore not deployable. It is used to compute regret and to show the distance between practical strategies and the test-set optimum.

[INSERT TABLE: main_framework_strategy_table.csv]

## 5. Evaluation Metrics and Statistical Analysis

The evaluation reports macro-F1, weighted-F1, AUROC, PR-AUC, FAR, MDR and FAR@95%Recall. Macro-F1 and PR-AUC summarize detection quality, while FAR and MDR express operational cost. FAR@95%Recall indicates the false-alarm burden required to reach a high-recall operating point.

Strategy-level performance is summarized by average rank, win count, top-2 count and regret relative to Oracle-Best-Test. For metrics where larger values are better, regret is the gap between the oracle metric and the selected strategy. For metrics where smaller values are better, regret is the selected strategy value minus the oracle value. Engineering utility functions are reported for balanced, safety, low-false-alarm, robust, deployment and label-efficiency settings.

Statistical comparisons use paired tests where matched scenario-level results are available. The summary includes paired t-tests, Wilcoxon signed-rank tests, sign tests, effect sizes and wins/losses. These tests are used to identify which utility-specific claims are supportable. Non-significant results are reported as such.

## 6. Results

### 6.1 Overall framework strategy comparison

The overall comparison shows a mixed but useful engineering pattern. Framework-Best-F1, Framework-Balanced, Framework-Label-Efficient, Framework-Robust and Framework-Deployment achieved average ranks of {ctx['fbf1_rank']}, {ctx['fb_rank']}, {ctx['fle_rank']}, {ctx['fr_rank']} and {ctx['fd_rank']}, respectively. These ranks were better than Fixed-XGBoost ({ctx['xgb_rank']}) and Fixed-RandomForest ({ctx['rf_rank']}) in the same summary. Fixed-LightGBM remained a strong fixed default with average rank {ctx['lgbm_rank']}, mean macro-F1 {ctx['lgbm_macro']} and mean PR-AUC {ctx['lgbm_pr']}.

Framework-Balanced reached mean macro-F1 {ctx['fb_macro']} and PR-AUC {ctx['fb_pr']}, compared with Fixed-LightGBM macro-F1 {ctx['lgbm_macro']} and PR-AUC {ctx['lgbm_pr']}. The balanced-utility improvement over Fixed-LightGBM was small and not statistically significant (mean difference {ctx['fb_lgbm_delta']}, Wilcoxon p={ctx['fb_lgbm_p']}). This result supports a cautious interpretation: the framework is useful for scenario-adaptive selection, but it is not comprehensively better than LightGBM.

[INSERT FIGURE: fig_engineering_utility_comparison.png]
[INSERT FIGURE: fig_strategy_average_rank.png]
[INSERT FIGURE: fig_strategy_regret.png]

### 6.2 FAR/MDR operating constraints

FAR/MDR analysis shows why a single accuracy-oriented winner is not sufficient. Framework-Safety produced a lower mean MDR ({ctx['fs_mdr']}) than the balanced strategy but increased mean FAR to {ctx['fs_far']}. Its safety utility was significantly higher than Fixed-LightGBM (mean difference {ctx['fs_delta']}, Wilcoxon p={ctx['fs_p']}). This strategy is appropriate when missed detections are more costly than false alarms, but it requires an alarm workflow that can tolerate more warnings.

Framework-Low-False-Alarm showed the opposite tradeoff. It reduced mean FAR to {ctx['flfa_far']} but increased mean MDR to {ctx['flfa_mdr']}. Its low-false-alarm utility was significantly higher than Fixed-LightGBM (mean difference {ctx['flfa_delta']}, Wilcoxon p={ctx['flfa_p']}). Fixed-LightGBM nevertheless had a strong joint violation rate ({ctx['lgbm_joint']}) and remains an attractive default when a simple and balanced operating point is needed.

[INSERT TABLE: far_mdr_constraint_table.csv]
[INSERT FIGURE: fig_far_mdr_constraint_violation.png]

### 6.3 Label-budget sensitivity

Label-budget sensitivity remains a limitation of the framework. For Framework-Balanced, the full-validation setting reached macro-F1 {ctx['full_fb_macro']}, while the 5% labeled setting reached {ctx['five_fb_macro']} and the normal-only setting reached {ctx['normal_fb_macro']}. These results indicate that full validation labels remain valuable, and that normal-only or very low-label settings are still difficult.

The label-efficiency criterion did not pass in the framework validation gate. The manuscript should therefore not claim label-efficiency superiority. Instead, label-budget analysis should be presented as an engineering diagnostic that identifies when supervised tree baselines and validation thresholds remain necessary.

[INSERT TABLE: label_efficiency_table_final.csv]
[INSERT FIGURE: fig_label_efficiency_strategy.png]

### 6.4 Robustness and domain shift

Robustness and domain-shift analysis shows that the framework can be useful under stress protocols, but the evidence is bounded. Framework-Balanced achieved macro-F1 {ctx['rob_fb_macro']} and robust utility {ctx['rob_fb_util']} on robustness protocols. In domain-shift protocols, its macro-F1 was {ctx['ds_fb_macro']}. Fixed-LightGBM remained strong, with macro-F1 {ctx['rob_lgbm_macro']} under robustness protocols and {ctx['ds_lgbm_macro']} under domain-shift protocols.

These results suggest that framework strategies can help align threshold and model selection with missing-sensor or noisy conditions. However, domain shift remains difficult, and the results do not justify a claim of broad robustness superiority over Fixed-LightGBM.

[INSERT TABLE: robustness_table_final.csv]
[INSERT FIGURE: fig_robustness_strategy.png]

### 6.5 Latency and deployment

Latency and model size affect deployability. Fixed-LightGBM and Fixed-XGBoost had low mean latencies ({ctx['lgbm_latency']} ms and {ctx['xgb_latency']} ms, respectively), while Fixed-RandomForest was slower ({ctx['rf_latency']} ms). The Framework-Deployment strategy had mean latency {ctx['fd_latency']} ms and explicitly included latency and size penalties in its validation utility.

Deep reconstruction baselines had low measured latency in this benchmark, for example AutoEncoder latency {ctx['ae_latency']} ms, but their detection metrics were less stable under FAR/MDR constraints. Deployment decisions should therefore consider latency, model size, FAR and MDR jointly rather than selecting the fastest model alone.

[INSERT TABLE: latency_deployment_table_final.csv]
[INSERT FIGURE: fig_latency_deployment_tradeoff.png]

### 6.6 Model selection guidelines

The framework often selected LightGBM or other strong tree baselines across engineering scenarios. This is an important result, not a weakness to hide. It indicates that the framework is a validation and calibration layer around a candidate pool, not a model that replaces strong baselines.

For label-rich settings, Framework-Balanced or Fixed-LightGBM is recommended. For label-scarce settings, Fixed-LightGBM with validation-only thresholding remains the safest default because label-efficiency superiority was not established. For safety-critical settings, Framework-Safety is preferred when a higher FAR can be tolerated. For low false alarm settings, Framework-Low-False-Alarm is preferred when missed detections are less costly. For low-latency CPU deployment, Fixed-LightGBM or Fixed-XGBoost remains attractive.

[INSERT TABLE: model_selection_summary_table.csv]
[INSERT FIGURE: fig_framework_strategy_map.png]

## 7. Engineering Discussion

The evidence supports the framework as an engineering decision process rather than as a new detector. Its purpose is to choose a candidate model and threshold under explicit deployment constraints using validation data. This distinction matters because a model that maximizes average macro-F1 may not minimize missed detections, and a model that minimizes false alarms may miss too many anomalies.

Fixed-LightGBM remains strong because the evaluated robotic and industrial datasets contain informative statistical feature structure. Tree models can exploit these structures efficiently and with low inference latency. The framework does not erase this advantage. Instead, it makes the fixed-model choice auditable and identifies when cost-specific strategies, such as safety or low-false-alarm selection, may be more appropriate.

The safety and low-false-alarm strategies show independent engineering value. Framework-Safety explicitly shifts the operating point toward recall and reduces MDR, while Framework-Low-False-Alarm shifts the operating point toward fewer false alarms. Neither strategy is universally better. They encode different operational priorities.

The results also reinforce the importance of validation-only thresholding. If thresholds or model choices were selected using test labels, the resulting strategy would not be deployable. By separating validation selection from test evaluation, the framework provides a more credible estimate of how a deployed monitoring system would behave.

In practical robotic monitoring, this framework could be used as a pre-deployment evaluation layer. Engineers would define the operational cost, choose the corresponding validation utility, select the model and threshold on calibration data and then audit FAR/MDR, latency and robustness before deployment. This is a narrower but more defensible contribution than claiming a universally superior anomaly detector.

## 8. Limitations

This work does not propose a new anomaly detection algorithm. It also does not show comprehensive superiority over Fixed-LightGBM. Fixed-LightGBM remains a strong default across many metrics and should be reported transparently.

Label-efficiency superiority was not established. Normal-only and 5% labeled validation settings remained challenging, and the manuscript should not imply that the framework solves low-label anomaly detection.

The datasets are real but still limited. RoAD is treated as a secondary reference because of confounding and artifact risks identified during earlier gates. NIST UR and KUKA are not used as primary binary anomaly evidence in this draft. More industrial datasets and online deployment studies would strengthen the framework claim.

Oracle-Best-Test is not deployable. It is included only to quantify regret and should never be described as a practical strategy.

The framework depends on the candidate model pool. If all candidate models fail under a new domain, strategy selection cannot rescue the benchmark. The framework should therefore be understood as a leakage-safe evaluation and model-selection procedure, not as an automatic guarantee of detection performance.

## 9. Conclusion

This draft reports a leakage-safe and calibration-aware engineering framework for robotic-arm multi-sensor anomaly diagnosis. The framework evaluates fixed models and validation-selected strategies under FAR/MDR, domain-shift, missing-sensor, label-budget and latency constraints.

The main finding is not that a new detector outperforms all baselines. Instead, strong tree baselines, especially Fixed-LightGBM, are often reliable, while scenario-specific strategies can better align model and threshold choice with operational cost. Safety and low-false-alarm strategies showed statistically supported utility improvements over Fixed-LightGBM for their respective objectives, whereas balanced, robust and deployment utilities did not establish comprehensive superiority over Fixed-LightGBM.

These results support a Results in Engineering style contribution: a reproducible protocol and decision framework for engineering deployment decisions in robotic anomaly monitoring. The claim is bounded, practical and explicitly separated from new-algorithm or SOTA claims.
"""


def claim_lock_report(generated_at: str, output_root: Path) -> str:
    return f"""
## Claim lock summary

- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- source_type: real
- synthetic: NO

### Locked positioning

- Article type: engineering diagnostic framework / benchmark and deployment-guidance paper.
- Target direction: Results in Engineering.
- New algorithm claim: NO.
- SOTA claim: NO.
- Comprehensive superiority over LightGBM/XGBoost/RandomForest: NO.
- Label-efficiency superiority: NO.
- Oracle-Best-Test deployability: NO.

### Allowed claims

- Fixed-LightGBM is a strong default model.
- Framework-Safety is significantly better than Fixed-LightGBM on safety utility.
- Framework-Low-False-Alarm is significantly better than Fixed-LightGBM on low-false-alarm utility.
- Framework-Balanced and Framework-Deployment can be compared favorably against Fixed-XGBoost and Fixed-RandomForest where supported by statistical tables.
- The framework value is scenario-adaptive model and threshold selection with validation-only calibration.

### Required limitations

- Balanced utility improvement over Fixed-LightGBM is small and not significant.
- Robust and deployment utilities do not show significant superiority over Fixed-LightGBM.
- Label-efficiency advantage was not established.
- Fixed-LightGBM strength must remain visible in Abstract, Results, Discussion and Limitations.
"""


def captions_and_notes(generated_at: str, output_root: Path) -> str:
    tables = [
        ("main_framework_strategy_table.csv", "Overall comparison of framework and fixed strategies across core detection, rank and regret metrics.", "Framework strategies can be competitive, but Fixed-LightGBM remains strong.", "Do not claim comprehensive superiority.", "Main text"),
        ("engineering_utility_table.csv", "Engineering utilities for balanced, safety, low-false-alarm, robust, deployment and label-efficiency objectives.", "Utility depends on operational cost.", "Each utility has a different preferred strategy.", "Main or supplementary"),
        ("far_mdr_constraint_table.csv", "FAR, MDR and joint violation rates under predefined operating constraints.", "Safety and low-false-alarm strategies encode different tradeoffs.", "Fixed-LightGBM has a strong joint violation rate.", "Main text"),
        ("label_efficiency_table_final.csv", "Performance under normal-only, 5%, 20% and full validation-label budgets.", "Full validation remains strongest.", "No label-efficiency superiority claim.", "Supplementary or limitation-focused result"),
        ("robustness_table_final.csv", "Performance under robustness and domain-shift groups.", "Framework strategies can help under robustness protocols.", "Domain shift remains difficult.", "Main or supplementary"),
        ("latency_deployment_table_final.csv", "Latency, model size, train time and deployment utility.", "Deployment needs latency and FAR/MDR jointly.", "Feature extraction costs should be checked in deployment.", "Main or supplementary"),
        ("model_selection_summary_table.csv", "Scenario-specific model and threshold selection guidance.", "Engineering use depends on cost and constraints.", "Guidance is conditional, not universal.", "Main text"),
    ]
    figs = [
        ("fig_engineering_utility_comparison.png", "Heatmap of engineering utility values.", "Framework strategies are useful under selected utilities.", "Do not hide strong tree baselines.", "Main text"),
        ("fig_far_mdr_constraint_violation.png", "FAR/MDR violation rates for framework and strong fixed strategies.", "Operating constraints reveal tradeoffs.", "Lower MDR may raise FAR.", "Main text"),
        ("fig_framework_strategy_map.png", "Most frequently selected model by framework strategy and scenario group.", "Framework often selects strong tree models.", "This is validation selection, not a new model.", "Main text"),
        ("fig_label_efficiency_strategy.png", "Macro-F1 under label-budget settings.", "Low-label settings remain difficult.", "No label-efficiency superiority claim.", "Supplementary"),
        ("fig_latency_deployment_tradeoff.png", "Latency versus deployment utility.", "Low latency alone is not sufficient.", "Use only as review figure until polished.", "Main or supplementary"),
        ("fig_robustness_strategy.png", "Robust and domain-shift utility heatmap.", "Robustness utility differs from balanced utility.", "Domain shift remains bounded.", "Main or supplementary"),
        ("fig_strategy_average_rank.png", "Average rank of strategies by utility.", "Several framework strategies rank well.", "Average rank is not a direct SOTA claim.", "Supplementary or main"),
        ("fig_strategy_regret.png", "Mean macro-F1 regret relative to Oracle-Best-Test.", "Regret quantifies distance from theoretical upper bound.", "Oracle is not deployable.", "Main or supplementary"),
    ]
    rows = ["## Tables"]
    for name, caption, claim, caution, rec in tables:
        rows += [
            f"### {name}",
            f"- caption draft: {caption}",
            f"- what it shows: {caption}",
            f"- supported claim: {claim}",
            f"- caution: {caution}",
            f"- recommendation: {rec}",
            "",
        ]
    rows += ["## Figures"]
    for name, caption, claim, caution, rec in figs:
        rows += [
            f"### {name}",
            f"- caption draft: {caption}",
            f"- what it shows: {caption}",
            f"- supported claim: {claim}",
            f"- caution: {caution}",
            f"- recommendation: {rec}",
            "",
        ]
    return "\n".join([
        f"- generated_at: {generated_at}",
        f"- output_path: {output_root.resolve()}",
        "- source_type: real",
        "- synthetic: NO",
        "",
        *rows,
    ])


def reviewer_risks(generated_at: str, output_root: Path) -> str:
    risks = [
        ("You do not propose a new algorithm. What is the contribution?", "Frame the contribution as a leakage-safe, calibration-aware engineering framework with deployment-oriented model selection, not an algorithm."),
        ("Why is LightGBM so strong?", "Acknowledge that statistical window features are highly informative in the evaluated datasets; use this as engineering evidence for strong baselines."),
        ("Why does the framework not fully beat Fixed-LightGBM?", "Explain that the framework optimizes scenario-specific utilities and is not intended to dominate a strong default on every metric."),
        ("Was the test set used for model selection?", "State that model, threshold and strategy selection used validation/calibration only; test labels were used only for final evaluation."),
        ("Is there data leakage?", "Point to segment/file/run-level splits, train-only normalization, no random overlapping-window split and provenance audit."),
        ("Why is label efficiency not better?", "Report it as a negative result and limitation; normal-only and 5% labels remain hard."),
        ("Why can Oracle-Best-Test not be deployed?", "It uses test labels to identify the best model per scenario and is only a regret upper bound."),
        ("Is this only a benchmark paper?", "Answer that it is an engineering diagnostic framework paper combining leakage audit, calibration, operational utilities, robustness and deployment guidance."),
        ("Why Results in Engineering?", "The contribution is deployment-oriented, protocol-driven and engineering-facing rather than a new ML algorithm."),
        ("Do you need more real deployment data?", "Yes; this is a limitation and future validation target."),
        ("What if reviewers request more datasets?", "Add or strengthen external validation if available; otherwise clearly state dataset limitations and avoid broad generalization."),
        ("What if reviewers find contribution insufficient?", "Add more protocol ablations, deployment/runtime analysis, or external dataset checks rather than inventing an algorithmic contribution."),
    ]
    lines = [f"- generated_at: {generated_at}", f"- output_path: {output_root.resolve()}", "- source_type: real", "- synthetic: NO", ""]
    for i, (q, a) in enumerate(risks, 1):
        lines += [f"## Risk {i}: {q}", f"- suggested response direction: {a}", ""]
    return "\n".join(lines)


def final_reports(data: dict[str, object], generated_at: str, output_root: Path) -> dict[str, str]:
    return {
        "05_reproducibility_package_plan_final.md": data["repro_md"],
        "06_final_code_index.md": data["code_md"],
        "07_artifact_inventory_final.md": data["artifact_md"] + "\n\n- manuscript_draft_v1.md: generated in this stage.\n- cover letter: not generated.\n",
        "08_manuscript_go_no_go_v2.md": f"""
- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- source_type: real
- synthetic: NO

- decision: MANUSCRIPT_DRAFT_READY
- can_enter_human_polishing: YES
- cover letter generated: NO
- new algorithm claim: NO
- SOTA claim: NO
- remaining risks: Fixed-LightGBM remains strong; label-efficiency advantage is not established; references still need verified citation insertion; final publication figures need journal-level polish.
- needs_extra_experiments_before_human_polishing: NO, unless the user wants more external deployment evidence.
""",
        "09_limitations_and_risks_final.md": data["limitations_md"],
        "10_contribution_statement_final.md": data["contribution_md"],
        "11_model_selection_guidelines_final.md": data["guidelines_md"],
        "12_statistical_claims_allowed.md": data["stats_claims"],
    }


def write_outputs(input_root: Path, output_root: Path, review_dir: Path) -> None:
    generated_at = now_iso()
    output_root.mkdir(parents=True, exist_ok=True)
    data = load_inputs(input_root)
    ctx = build_context(data)

    files = {
        "01_claim_lock_report.md": claim_lock_report(generated_at, output_root),
        "02_manuscript_draft_v1.md": manuscript(ctx, generated_at, output_root),
        "03_captions_and_table_notes_v1.md": captions_and_notes(generated_at, output_root),
        "04_reviewer_risk_checklist_v1.md": reviewer_risks(generated_at, output_root),
        **final_reports(data, generated_at, output_root),
    }
    readme = f"""
- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- current_stage: {STAGE}
- current_status: {STATUS}
- contains synthetic: NO
- generated manuscript draft: YES
- generated cover letter: NO
- claims new algorithm: NO
- claims SOTA: NO
- fixed LightGBM strong result retained: YES
- draft type: review-only manuscript draft v1, not final submission manuscript
"""
    files["00_readme_for_chatgpt.md"] = readme

    for name, text in files.items():
        title = name.split("_", 1)[1].replace("_", " ").replace(".md", "").title() if "_" in name else name
        if name == "02_manuscript_draft_v1.md":
            (output_root / name).write_text(text)
        else:
            body = text.strip()
            if body.startswith("# "):
                (output_root / name).write_text(body + "\n")
            else:
                write_text(output_root / name, title, generated_at, output_root, body)

    fig_src = input_root / "figures"
    fig_out = output_root / "figures"
    if fig_out.exists():
        shutil.rmtree(fig_out)
    shutil.copytree(fig_src, fig_out)

    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "00_readme_for_chatgpt.md",
        "01_claim_lock_report.md",
        "02_manuscript_draft_v1.md",
        "03_captions_and_table_notes_v1.md",
        "04_reviewer_risk_checklist_v1.md",
        "05_reproducibility_package_plan_final.md",
        "06_final_code_index.md",
        "07_artifact_inventory_final.md",
        "08_manuscript_go_no_go_v2.md",
        "09_limitations_and_risks_final.md",
        "10_contribution_statement_final.md",
        "11_model_selection_guidelines_final.md",
        "12_statistical_claims_allowed.md",
    ]:
        shutil.copy2(output_root / name, review_dir / name)
    shutil.copytree(fig_out, review_dir / "figures")

    status = {
        "status": STATUS,
        "generated_at": generated_at,
        "manuscript_draft": str((output_root / "02_manuscript_draft_v1.md").resolve()),
        "review_dir": str(review_dir.resolve()),
        "top_level_items": len(list(review_dir.iterdir())),
        "figures": len(list((review_dir / "figures").glob("*.png"))),
    }
    (output_root / "run_status_manuscript_draft_v1.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs/rie_results_packaging_review_figures_gate_v1")
    parser.add_argument("--output-root", default="outputs/rie_manuscript_draft_v1")
    parser.add_argument("--review-dir", default="progress_for_chatgpt/latest")
    args = parser.parse_args()
    write_outputs(Path(args.input_root), Path(args.output_root), Path(args.review_dir))


if __name__ == "__main__":
    main()
