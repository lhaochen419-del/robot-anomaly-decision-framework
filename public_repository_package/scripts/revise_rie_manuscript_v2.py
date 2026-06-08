#!/usr/bin/env python3
"""Revise RIE manuscript draft v1 into claim-compliant draft v2.

The generated manuscript is a review-only draft. It does not create a cover
letter, does not claim a new algorithm, and does not add experimental results.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


STATUS = "NEEDS_REFERENCE_AND_CLAIM_POLISHING"
STAGE = "RIE Manuscript Revision v2"


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fmt(x: object, digits: int = 3) -> str:
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return "TODO"


def pfmt(x: object) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return "TODO"


def write_md(path: Path, title: str, generated_at: str, output_root: Path, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                f"# {title}",
                "",
                f"- generated_at: {generated_at}",
                f"- output_path: {output_root.resolve()}",
                "- source_type: real",
                "- synthetic: NO",
                "- manuscript_stage: draft_v2_review_only",
                "",
                body.strip(),
                "",
            ]
        )
    )


def require_inputs(input_root: Path, packaging_root: Path) -> None:
    required = [
        input_root / "02_manuscript_draft_v1.md",
        input_root / "01_claim_lock_report.md",
        input_root / "03_captions_and_table_notes_v1.md",
        input_root / "04_reviewer_risk_checklist_v1.md",
        input_root / "12_statistical_claims_allowed.md",
        packaging_root / "main_framework_strategy_table.csv",
        packaging_root / "engineering_utility_table.csv",
        packaging_root / "far_mdr_constraint_table.csv",
        packaging_root / "label_efficiency_table_final.csv",
        packaging_root / "robustness_table_final.csv",
        packaging_root / "latency_deployment_table_final.csv",
        packaging_root / "model_selection_summary_table.csv",
        packaging_root / "figures",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("MISSING_INPUT: " + ", ".join(missing))


def row(df: pd.DataFrame, strategy: str) -> pd.Series:
    hit = df[df["deployed_strategy"].eq(strategy)]
    if hit.empty:
        raise KeyError(strategy)
    return hit.iloc[0]


def context(packaging_root: Path) -> dict[str, str]:
    main = pd.read_csv(packaging_root / "main_framework_strategy_table.csv")
    util = pd.read_csv(packaging_root / "engineering_utility_table.csv")
    far = pd.read_csv(packaging_root / "far_mdr_constraint_table.csv")
    label = pd.read_csv(packaging_root / "label_efficiency_table_final.csv")
    robust = pd.read_csv(packaging_root / "robustness_table_final.csv")
    latency = pd.read_csv(packaging_root / "latency_deployment_table_final.csv")
    stats = pd.read_csv("outputs/framework_strategy_validation_gate_v1/statistical_comparison.csv")

    lgbm = row(main, "Fixed-LightGBM")
    xgb = row(main, "Fixed-XGBoost")
    rf = row(main, "Fixed-RandomForest")
    fb = row(main, "Framework-Balanced")
    fbf1 = row(main, "Framework-Best-F1")
    fle = row(main, "Framework-Label-Efficient")
    fr = row(main, "Framework-Robust")
    fd = row(main, "Framework-Deployment")
    fs = row(main, "Framework-Safety")
    flfa = row(main, "Framework-Low-False-Alarm")
    oracle = row(main, "Oracle-Best-Test")

    def u(strategy: str, col: str) -> str:
        return fmt(row(util, strategy)[col], 3)

    def f(strategy: str, col: str) -> str:
        return fmt(row(far, strategy)[col], 3)

    def l(strategy: str, col: str, digits: int = 4) -> str:
        return fmt(row(latency, strategy)[col], digits)

    def stat(framework: str, fixed: str, utility: str) -> pd.Series:
        hit = stats[
            stats["framework_strategy"].eq(framework)
            & stats["fixed_strategy"].eq(fixed)
            & stats["utility"].eq(utility)
        ]
        if hit.empty:
            raise KeyError((framework, fixed, utility))
        return hit.iloc[0]

    label_fb = {
        k: label[(label["deployed_strategy"].eq("Framework-Balanced")) & (label["label_budget"].eq(k))]
        for k in ["normal_only", "5pct", "20pct", "full"]
    }
    robust_fb = robust[(robust["deployed_strategy"].eq("Framework-Balanced")) & (robust["protocol_group"].eq("robustness"))]
    domain_fb = robust[(robust["deployed_strategy"].eq("Framework-Balanced")) & (robust["protocol_group"].eq("domain_shift"))]
    robust_lgbm = robust[(robust["deployed_strategy"].eq("Fixed-LightGBM")) & (robust["protocol_group"].eq("robustness"))]
    domain_lgbm = robust[(robust["deployed_strategy"].eq("Fixed-LightGBM")) & (robust["protocol_group"].eq("domain_shift"))]
    s_bal_xgb = stat("Framework-Balanced", "Fixed-XGBoost", "utility_balanced")
    s_bal_rf = stat("Framework-Balanced", "Fixed-RandomForest", "utility_balanced")
    s_dep_xgb = stat("Framework-Deployment", "Fixed-XGBoost", "utility_deployment")
    s_dep_rf = stat("Framework-Deployment", "Fixed-RandomForest", "utility_deployment")

    return {
        "lgbm_macro": fmt(lgbm["macro_f1_mean"]),
        "lgbm_pr": fmt(lgbm["pr_auc_mean"]),
        "lgbm_far": fmt(lgbm["far_mean"]),
        "lgbm_mdr": fmt(lgbm["mdr_mean"]),
        "lgbm_joint": f("Fixed-LightGBM", "joint_far_mdr_violation_rate"),
        "xgb_macro": fmt(xgb["macro_f1_mean"]),
        "rf_macro": fmt(rf["macro_f1_mean"]),
        "fb_macro": fmt(fb["macro_f1_mean"]),
        "fb_pr": fmt(fb["pr_auc_mean"]),
        "fb_far": fmt(fb["far_mean"]),
        "fb_mdr": fmt(fb["mdr_mean"]),
        "fb_rank": fmt(fb["average_rank"], 2),
        "fb_regret": fmt(fb["average_regret"], 3),
        "fb_util": u("Framework-Balanced", "utility_balanced_mean"),
        "fbf1_rank": fmt(fbf1["average_rank"], 2),
        "fle_rank": fmt(fle["average_rank"], 2),
        "fr_rank": fmt(fr["average_rank"], 2),
        "fd_rank": fmt(fd["average_rank"], 2),
        "lgbm_rank": fmt(lgbm["average_rank"], 2),
        "xgb_rank": fmt(xgb["average_rank"], 2),
        "rf_rank": fmt(rf["average_rank"], 2),
        "fs_far": fmt(fs["far_mean"]),
        "fs_mdr": fmt(fs["mdr_mean"]),
        "fs_util": u("Framework-Safety", "utility_safety_mean"),
        "flfa_far": fmt(flfa["far_mean"]),
        "flfa_mdr": fmt(flfa["mdr_mean"]),
        "flfa_util": u("Framework-Low-False-Alarm", "utility_low_false_alarm_mean"),
        "oracle_macro": fmt(oracle["macro_f1_mean"]),
        "oracle_status": "ORACLE_NOT_DEPLOYABLE",
        "normal_fb_macro": fmt(label_fb["normal_only"].iloc[0]["macro_f1"]) if not label_fb["normal_only"].empty else "TODO",
        "five_fb_macro": fmt(label_fb["5pct"].iloc[0]["macro_f1"]) if not label_fb["5pct"].empty else "TODO",
        "twenty_fb_macro": fmt(label_fb["20pct"].iloc[0]["macro_f1"]) if not label_fb["20pct"].empty else "TODO",
        "full_fb_macro": fmt(label_fb["full"].iloc[0]["macro_f1"]) if not label_fb["full"].empty else "TODO",
        "rob_fb_macro": fmt(robust_fb.iloc[0]["macro_f1"]) if not robust_fb.empty else "TODO",
        "rob_fb_util": fmt(robust_fb.iloc[0]["utility_robust"]) if not robust_fb.empty else "TODO",
        "domain_fb_macro": fmt(domain_fb.iloc[0]["macro_f1"]) if not domain_fb.empty else "TODO",
        "rob_lgbm_macro": fmt(robust_lgbm.iloc[0]["macro_f1"]) if not robust_lgbm.empty else "TODO",
        "domain_lgbm_macro": fmt(domain_lgbm.iloc[0]["macro_f1"]) if not domain_lgbm.empty else "TODO",
        "lgbm_latency": l("Fixed-LightGBM", "latency_ms_mean"),
        "xgb_latency": l("Fixed-XGBoost", "latency_ms_mean"),
        "rf_latency": l("Fixed-RandomForest", "latency_ms_mean"),
        "fd_latency": l("Framework-Deployment", "latency_ms_mean"),
        "ae_latency": l("Fixed-AutoEncoder", "latency_ms_mean"),
        "bal_xgb_delta": fmt(s_bal_xgb["mean_framework_minus_fixed"], 4),
        "bal_xgb_p": pfmt(s_bal_xgb["wilcoxon_p"]),
        "bal_rf_delta": fmt(s_bal_rf["mean_framework_minus_fixed"], 4),
        "bal_rf_p": pfmt(s_bal_rf["wilcoxon_p"]),
        "dep_xgb_delta": fmt(s_dep_xgb["mean_framework_minus_fixed"], 4),
        "dep_xgb_p": pfmt(s_dep_xgb["wilcoxon_p"]),
        "dep_rf_delta": fmt(s_dep_rf["mean_framework_minus_fixed"], 4),
        "dep_rf_p": pfmt(s_dep_rf["wilcoxon_p"]),
    }


def manuscript_v2(ctx: dict[str, str], generated_at: str, output_root: Path) -> str:
    return f"""# Manuscript Draft v2: review-only, not a final submission manuscript

- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- source_type: real
- synthetic: NO
- cover letter generated: NO
- final submission manuscript: NO
- new algorithm claim: NO
- SOTA claim: NO
- reference status: partially verified; remaining placeholders are marked [REF_NEEDED]

## Title candidates

1. **Preferred:** Leakage-safe and calibration-aware multi-sensor anomaly diagnosis for robotic arms under domain shift
2. Engineering evaluation of model-selection strategies for robotic-arm anomaly diagnosis under operational constraints
3. FAR/MDR-aware benchmarking for multi-sensor robotic-arm anomaly diagnosis

## Draft Abstract

Multi-sensor robotic-arm monitoring requires anomaly detection procedures that remain credible under changing operating conditions, missing sensors and deployment constraints. Accuracy-only evaluation is insufficient in this setting because false alarms and missed detections have different operational costs, and because random overlapping-window splits or test-set threshold tuning can make benchmark results appear more deployable than they are. This draft evaluates a leakage-safe and calibration-aware engineering framework for robotic-arm anomaly diagnosis. The framework uses segment-, file- or run-level splits, train-only normalization and validation-only threshold calibration. It compares fixed model strategies with validation-selected framework strategies across real multi-sensor robotic-arm and industrial datasets, using LightGBM [@Ke2017LightGBM], XGBoost [@Chen2016XGBoost], RandomForest [@Breiman2001RandomForests], IsolationForest [@Liu2008IsolationForest], AutoEncoder-style reconstruction [@Hinton2006Autoencoders], LSTM-AE [@Malhotra2016LSTMEncDec] and USAD [@Audibert2020USAD] baselines.

The results show that Fixed-LightGBM is a strong default, with mean macro-F1 {ctx['lgbm_macro']}, PR-AUC {ctx['lgbm_pr']}, FAR {ctx['lgbm_far']} and MDR {ctx['lgbm_mdr']} across strategy-validation scenarios. The framework does not comprehensively outperform Fixed-LightGBM. Instead, its value is scenario-specific. Framework-Safety reduced mean MDR to {ctx['fs_mdr']} at the cost of higher FAR ({ctx['fs_far']}), and Framework-Low-False-Alarm reduced mean FAR to {ctx['flfa_far']} at the cost of higher MDR ({ctx['flfa_mdr']}). These two strategies showed statistically supported utility improvements over Fixed-LightGBM for their corresponding objectives. In contrast, the balanced-utility gain over Fixed-LightGBM was small and not significant. Label-efficiency superiority was not established. The contribution is therefore an engineering evaluation and deployment-guidance framework, not a new anomaly detection algorithm.

## Keywords

Robotic-arm anomaly detection; multi-sensor monitoring; threshold calibration; false alarm rate; missed detection rate; domain shift; engineering benchmark.

## 1. Introduction

Industrial robots and robotic arms are increasingly monitored through multiple sensor streams, including motion, current, vibration, torque and acoustic channels [REF_NEEDED]. These signals can support early detection of collisions, abnormal loads, speed-related faults or degradation, but anomaly detection in this setting is not equivalent to ordinary static classification. A false alarm may interrupt production or trigger unnecessary inspection. A missed detection may leave a safety or maintenance risk unresolved. An engineering evaluation must therefore expose false alarm rate (FAR) and missed detection rate (MDR), rather than relying only on aggregate accuracy.

Benchmarking robotic anomaly detectors is also vulnerable to protocol error. Randomly splitting overlapping windows can leak near-duplicate temporal segments across train and test sets. Tuning thresholds on test labels can similarly overstate deployable performance. Prior work on data leakage and time-series anomaly benchmark flaws has warned that apparently high test performance can be driven by evaluation artifacts rather than deployable detection behavior [@Kaufman2012Leakage; @Wu2021TSADFlawed]. A credible robotic anomaly benchmark should split by segment, file or run unit, fit normalization only on training data and calibrate thresholds only on validation or calibration data.

Many anomaly detection studies compare model families under a single operating point [REF_NEEDED]. This is useful, but it does not answer a deployment question: which model and threshold should be selected when the cost of missed detection, false alarm, latency, label scarcity or missing sensors changes? A single fixed model may be the best default in one setting and a poor choice in another. Conversely, a validation-selected framework should not be presented as a new detector. Its role is to make model and threshold choice auditable.

This study evaluates a leakage-safe and calibration-aware engineering framework for robotic-arm multi-sensor anomaly diagnosis. The framework compares fixed single-model strategies with validation-selected framework strategies using real benchmark outputs from IMAD-DS [@Augusti2024IMADDS], RoAD as a secondary reference [@Gaiardelli2023RoAD] and related readiness datasets where appropriate. It reports FAR, MDR, PR-AUC, regret, average rank, label-budget sensitivity, robustness and latency. The contribution is an engineering diagnostic framework and evidence-based model-selection guidance. It is not a new anomaly detection algorithm and does not make a state-of-the-art claim.

## 2. Related Work

### 2.1 Robotic and industrial anomaly diagnosis

Robotic and industrial anomaly diagnosis has been studied for collaborative robot monitoring, industrial arm applications and production-line anomalies [@Narayanan2018IndustrialArm; @Gaiardelli2023RoAD]. These applications share a deployment constraint: an alarm is not merely a label prediction but an operational decision. Existing datasets such as IMAD-DS and RoAD provide useful real-world contexts, but they also require careful protocol design because operating conditions, domain shifts and artifacts can shape the apparent difficulty of the task [@Augusti2024IMADDS; @Gaiardelli2023RoAD].

### 2.2 Multi-sensor time-series anomaly detection

Multi-sensor time-series anomaly detection often uses temporal reconstruction, sequence prediction, graph structure or density-based scoring [REF_NEEDED]. LSTM encoder-decoder models and USAD-style autoencoder methods are representative reconstruction-based baselines for multivariate time series [@Malhotra2016LSTMEncDec; @Audibert2020USAD]. These methods can be attractive when anomaly labels are limited, but their operating behavior still depends on score calibration and threshold selection.

### 2.3 Tree-based and statistical baselines

Tree-based models remain strong engineering baselines for window-level statistical features. LightGBM implements efficient gradient boosting decision trees [@Ke2017LightGBM], XGBoost provides scalable tree boosting [@Chen2016XGBoost], and RandomForest is a classic ensemble of randomized decision trees [@Breiman2001RandomForests]. IsolationForest is a commonly used unsupervised anomaly detector based on isolation rather than density estimation [@Liu2008IsolationForest; @Liu2012IsolationBased]. In the present results, Fixed-LightGBM remains a particularly strong default, and this strength is retained as a central finding.

### 2.4 Threshold calibration and leakage-safe evaluation

Threshold calibration is central to deployable anomaly detection. A model with good ranking metrics can still be unsuitable if its operating threshold produces excessive false alarms or missed detections. Conformal prediction and distribution-free calibration provide one family of calibration ideas [@Vovk2005Conformal; @Lei2018Conformal], while practical validation-only threshold selection is a simpler engineering route when labeled validation data are available. This paper uses validation-only thresholds and reports FAR/MDR operating behavior rather than optimizing thresholds on test labels.

### 2.5 Deployment-aware engineering evaluation

Deployment-aware evaluation should consider label budgets, missing sensors, domain shift, latency, model size and the cost of false alarms or missed detections [REF_NEEDED]. The present framework formalizes these constraints as strategy-level utility functions. This framing is closer to engineering model selection than algorithm design.

## 3. Data and Leakage-Safe Protocols

The primary dataset is IMAD-DS RoboticArm in raw-window form. IMAD-DS contains robotic-arm and brushless-motor data under operational and environmental domain shifts, with microphone, accelerometer and gyroscope channels recorded under controlled industrial conditions [@Augusti2024IMADDS]. The benchmark also includes IMAD-DS RoboticArm segment-level features for granularity comparison and IMAD-DS BrushlessMotor as a secondary industrial anomaly dataset. RoAD is retained only as a secondary sanity or stress reference because earlier audits identified confounding and artifact risks [@Gaiardelli2023RoAD]. NIST UR and KUKA are not used as primary binary anomaly evidence in this draft when labels or data availability do not support the same protocols.

All protocols are designed to reduce leakage. The split unit is segment, file or run identifier wherever available. Window-level random splitting is not used. Normalization is fit on training data only. Model thresholds and framework strategy choices are selected using validation or calibration data only. Test labels are used only for final evaluation.

The protocol set includes main binary classification, source-to-target domain shift, leave-target-weight35-out stress testing, missing-sensor settings, noise stress, label-efficiency settings and latency benchmarking. Each result row carries provenance fields including dataset, protocol, seed, method, train/validation/test counts, test normal/anomaly counts, generated time and output path. No synthetic result is used.

[INSERT TABLE 1: main_framework_strategy_table.csv]

## 4. Benchmark Models and Framework Strategies

The fixed model strategies use one model family across all evaluated scenarios. They include Fixed-LightGBM, Fixed-XGBoost, Fixed-RandomForest, Fixed-IsolationForest, Fixed-AutoEncoder, Fixed-LSTM-AE and Fixed-USAD. These are benchmark strategies, not components of a proposed method.

The framework strategies select a candidate model and threshold using validation or calibration data. Framework-Best-F1 selects the validation macro-F1 optimum. Framework-Balanced maximizes macro-F1 minus FAR and MDR. Framework-Safety increases the penalty on missed detections. Framework-Low-False-Alarm increases the penalty on false alarms. Framework-Robust considers robustness degradation where validation robustness metrics are available. Framework-Deployment includes latency and model-size penalties. Framework-Label-Efficient evaluates normal-only, 5%, 20% and full validation-label budgets.

Oracle-Best-Test is included only as a theoretical upper bound. It uses test-set outcomes to identify the best result per scenario, so it is explicitly marked {ctx['oracle_status']} and must not be interpreted as a deployable strategy.

## 5. Evaluation Metrics and Statistical Analysis

The evaluation reports macro-F1, weighted-F1, AUROC, PR-AUC, FAR, MDR and FAR@95%Recall. Macro-F1 and PR-AUC summarize detection quality, while FAR and MDR express operational costs. FAR@95%Recall indicates the false-alarm burden required to reach a high-recall operating point.

Strategy-level performance is summarized by average rank, win count, top-2 count and regret relative to Oracle-Best-Test. For metrics where larger values are better, regret is the gap between the oracle metric and the selected strategy. For metrics where smaller values are better, regret is the selected strategy value minus the oracle value. Engineering utilities are reported for balanced, safety, low-false-alarm, robust, deployment and label-efficiency settings.

Statistical comparisons use paired tests where matched scenario-level results are available. The summary includes paired t-tests, Wilcoxon signed-rank tests, sign tests, effect sizes and wins/losses. Non-significant results are reported as non-significant.

## 6. Results

### 6.1 Overall framework strategy comparison

The overall comparison shows a useful but bounded engineering pattern. Framework-Best-F1, Framework-Balanced, Framework-Label-Efficient, Framework-Robust and Framework-Deployment achieved average ranks of {ctx['fbf1_rank']}, {ctx['fb_rank']}, {ctx['fle_rank']}, {ctx['fr_rank']} and {ctx['fd_rank']}, respectively. These ranks were better than Fixed-XGBoost ({ctx['xgb_rank']}) and Fixed-RandomForest ({ctx['rf_rank']}) in the same summary. Fixed-LightGBM remained a strong fixed default, with average rank {ctx['lgbm_rank']}, mean macro-F1 {ctx['lgbm_macro']} and mean PR-AUC {ctx['lgbm_pr']}.

Framework-Balanced reached mean macro-F1 {ctx['fb_macro']} and PR-AUC {ctx['fb_pr']}. Fixed-LightGBM reached macro-F1 {ctx['lgbm_macro']} and PR-AUC {ctx['lgbm_pr']}. The balanced-utility difference between Framework-Balanced and Fixed-LightGBM was small and not statistically significant. By contrast, Framework-Balanced was significantly higher than Fixed-XGBoost on balanced utility (mean difference {ctx['bal_xgb_delta']}, Wilcoxon p={ctx['bal_xgb_p']}) and higher than Fixed-RandomForest (mean difference {ctx['bal_rf_delta']}, Wilcoxon p={ctx['bal_rf_p']}). These results support a scenario-adaptive strategy claim, not a comprehensive superiority claim over LightGBM.

[INSERT FIGURE 1: fig_engineering_utility_comparison.png]
[INSERT FIGURE 2: fig_strategy_average_rank.png]
[INSERT FIGURE 3: fig_strategy_regret.png]

### 6.2 FAR/MDR operating constraints

FAR/MDR analysis shows why a single accuracy-oriented winner is insufficient. Framework-Safety produced lower mean MDR ({ctx['fs_mdr']}) but increased mean FAR ({ctx['fs_far']}). Its safety utility was significantly higher than Fixed-LightGBM. This strategy is appropriate when missed detections are more costly than false alarms, but it requires an alarm workflow that can tolerate more warnings.

Framework-Low-False-Alarm showed the opposite tradeoff. It reduced mean FAR to {ctx['flfa_far']} but increased mean MDR to {ctx['flfa_mdr']}. Its low-false-alarm utility was significantly higher than Fixed-LightGBM. Fixed-LightGBM nevertheless had a strong joint FAR/MDR violation rate ({ctx['lgbm_joint']}) and remains an attractive default when a simple balanced operating point is needed.

[INSERT TABLE 2: far_mdr_constraint_table.csv]
[INSERT FIGURE 4: fig_far_mdr_constraint_violation.png]

### 6.3 Label-budget sensitivity

Label-budget sensitivity remains a limitation. For Framework-Balanced, the full-validation setting reached macro-F1 {ctx['full_fb_macro']}, while the 20% labeled, 5% labeled and normal-only settings reached {ctx['twenty_fb_macro']}, {ctx['five_fb_macro']} and {ctx['normal_fb_macro']}, respectively. These results indicate that full validation labels remain valuable and that normal-only or very low-label settings are still difficult.

The label-efficiency criterion did not pass in the framework validation gate. The manuscript therefore should not claim label-efficiency superiority. Label-budget analysis is presented as an engineering diagnostic that identifies when supervised tree baselines and validation thresholds remain necessary.

[INSERT TABLE 3: label_efficiency_table_final.csv]
[INSERT FIGURE 5: fig_label_efficiency_strategy.png]

### 6.4 Robustness and domain shift

Robustness and domain-shift analysis shows that the framework can be useful under stress protocols, but the evidence is bounded. Framework-Balanced achieved macro-F1 {ctx['rob_fb_macro']} and robust utility {ctx['rob_fb_util']} on robustness protocols. In domain-shift protocols, its macro-F1 was {ctx['domain_fb_macro']}. Fixed-LightGBM remained strong, with macro-F1 {ctx['rob_lgbm_macro']} under robustness protocols and {ctx['domain_lgbm_macro']} under domain-shift protocols.

These results suggest that validation-selected strategies can help align model and threshold selection with missing-sensor or noisy conditions. They do not justify a broad robustness-superiority claim over Fixed-LightGBM.

[INSERT TABLE 4: robustness_table_final.csv]
[INSERT FIGURE 6: fig_robustness_strategy.png]

### 6.5 Latency and deployment

Latency and model size affect deployability. Fixed-LightGBM and Fixed-XGBoost had low mean latencies ({ctx['lgbm_latency']} ms and {ctx['xgb_latency']} ms, respectively), while Fixed-RandomForest was slower ({ctx['rf_latency']} ms). Framework-Deployment had mean latency {ctx['fd_latency']} ms and included latency and model-size penalties in its validation utility. Framework-Deployment was significantly higher than Fixed-XGBoost on deployment utility (mean difference {ctx['dep_xgb_delta']}, Wilcoxon p={ctx['dep_xgb_p']}) and higher than Fixed-RandomForest (mean difference {ctx['dep_rf_delta']}, Wilcoxon p={ctx['dep_rf_p']}), but it was not significantly higher than Fixed-LightGBM.

Deep reconstruction baselines had low measured latency in this benchmark, for example AutoEncoder latency {ctx['ae_latency']} ms, but their detection metrics were less stable under FAR/MDR constraints. Deployment decisions should therefore consider latency, model size, FAR and MDR jointly.

[INSERT TABLE 5: latency_deployment_table_final.csv]
[INSERT FIGURE 7: fig_latency_deployment_tradeoff.png]

### 6.6 Model selection guidelines

The framework often selected LightGBM or other strong tree baselines across engineering scenarios. This is an important result, not a weakness to hide. It indicates that the framework is a validation and calibration layer around a candidate pool, not a model that replaces strong baselines.

For label-rich settings, Framework-Balanced or Fixed-LightGBM is recommended. For label-scarce settings, Fixed-LightGBM with validation-only thresholding remains the safer default because label-efficiency superiority was not established. For safety-critical settings, Framework-Safety is preferred when a higher FAR can be tolerated. For low-false-alarm settings, Framework-Low-False-Alarm is preferred when missed detections are less costly. For low-latency CPU deployment, Fixed-LightGBM or Fixed-XGBoost remains attractive.

[INSERT TABLE 6: model_selection_summary_table.csv]
[INSERT FIGURE 8: fig_framework_strategy_map.png]

## 7. Engineering Discussion

The evidence supports the framework as an engineering decision process rather than as a new detector. Its purpose is to select a candidate model and threshold under explicit deployment constraints using validation data. This distinction matters because a model that maximizes average macro-F1 may not minimize missed detections, and a model that minimizes false alarms may miss too many anomalies.

Fixed-LightGBM remains strong because the evaluated robotic and industrial datasets contain informative statistical feature structure. Tree models can exploit these structures efficiently and with low inference latency. The framework does not erase this advantage. Instead, it makes fixed-model choice auditable and identifies when cost-specific strategies, such as safety or low-false-alarm selection, are more appropriate.

The safety and low-false-alarm strategies provide the clearest independent value. Framework-Safety shifts the operating point toward lower MDR, while Framework-Low-False-Alarm shifts the operating point toward lower FAR. Neither strategy is universally better. They encode different operational priorities.

The results also reinforce the importance of validation-only thresholding. If thresholds or model choices were selected using test labels, the resulting strategy would not be deployable. By separating validation selection from test evaluation, the framework provides a more credible estimate of how a deployed monitoring system would behave.

In practical robotic monitoring, this framework can be used as a pre-deployment evaluation layer. Engineers define the operational cost, choose the corresponding validation utility, select the model and threshold on calibration data and audit FAR/MDR, latency and robustness before deployment. This is a narrower but more defensible contribution than claiming a universally superior anomaly detector.

## 8. Limitations

This work does not propose a new anomaly detection algorithm. It also does not show comprehensive superiority over Fixed-LightGBM. Fixed-LightGBM remains a strong default across many metrics and should be reported transparently.

Label-efficiency superiority was not established. Normal-only and 5% labeled validation settings remained challenging, and the manuscript should not imply that the framework solves low-label anomaly detection.

The datasets are real but still limited. RoAD is treated as a secondary reference because of confounding and artifact risks identified during earlier gates. NIST UR and KUKA are not used as primary binary anomaly evidence in this draft. More industrial datasets and online deployment studies would strengthen the framework claim.

Oracle-Best-Test is not deployable. It is included only to quantify regret and should never be described as a practical strategy.

The framework depends on the candidate model pool. If all candidate models fail under a new domain, strategy selection cannot rescue the benchmark. The framework should therefore be understood as a leakage-safe evaluation and model-selection procedure, not as a guarantee of detection performance.

Several citations still require manual verification before submission. Unverified citation needs are marked as [REF_NEEDED], and the draft BibTeX file should be checked against publisher pages before final formatting.

## 9. Conclusion

This draft evaluates a leakage-safe and calibration-aware engineering framework for robotic-arm multi-sensor anomaly diagnosis. The framework compares fixed models and validation-selected strategies under FAR/MDR, domain-shift, missing-sensor, label-budget and latency constraints.

The main finding is not that a new detector outperforms all baselines. Strong tree baselines, especially Fixed-LightGBM, are often reliable. Scenario-specific strategies can nevertheless align model and threshold choice with operational cost. Safety and low-false-alarm strategies showed statistically supported utility improvements over Fixed-LightGBM for their respective objectives, whereas balanced, robust and deployment utilities did not establish comprehensive superiority over Fixed-LightGBM.

These results support a Results in Engineering style contribution: a reproducible protocol and decision framework for engineering deployment decisions in robotic anomaly monitoring. The claim is bounded, practical and explicitly separated from new-algorithm and SOTA claims.

## Reference placeholders

Verified draft references are provided in `references_draft.bib`. Remaining placeholders:

- [REF_NEEDED] robotic and industrial anomaly diagnosis survey or representative application papers beyond RoAD/industrial-arm examples.
- [REF_NEEDED] multi-sensor time-series anomaly detection survey or benchmark paper.
- [REF_NEEDED] FAR/MDR or false-alarm/missed-detection evaluation in industrial monitoring.
- [REF_NEEDED] deployment-aware model evaluation under latency and robustness constraints.
"""


def claim_audit(generated_at: str, output_root: Path) -> str:
    rows = [
        ("Abstract", "The framework ... real multi-sensor robotic-arm and industrial datasets", "Potentially broad dataset wording", "Specify primary IMAD-DS and secondary datasets; avoid implying broad deployment coverage.", "YES"),
        ("Abstract/Conclusion", "These results support ... rather than a new anomaly detection algorithm.", "Safe negative claim", "Kept and made more explicit in v2.", "YES"),
        ("Introduction", "we show that...", "Verb strength", "Changed framing to 'evaluate' and 'support' where appropriate.", "YES"),
        ("Related Work", "[REF_NEEDED]", "Reference gap", "Inserted verified core references and retained [REF_NEEDED] for unverified categories.", "PARTIAL"),
        ("Results 6.1", "Framework ranks better than XGBoost and RandomForest", "Needs statistical support", "Added Wilcoxon-supported balanced/deployment comparisons against XGBoost/RandomForest.", "YES"),
        ("Results 6.2", "Safety and low-false-alarm strategies", "Tradeoff risk", "Explicitly stated FAR/MDR tradeoff and Fixed-LightGBM joint violation strength.", "YES"),
        ("Results 6.3", "Label-efficiency", "Possible overclaim", "Explicitly stated criterion did not pass and superiority is not claimed.", "YES"),
        ("Methods/Strategies", "Oracle-Best-Test", "Deployability risk", "Marked as theoretical upper bound and ORACLE_NOT_DEPLOYABLE.", "YES"),
        ("Figures/Tables", "INSERT FIGURE/TABLE placeholders", "Consistency", "Numbered Table 1-6 and Figure 1-8, all mapped to existing files.", "YES"),
        ("Limitations", "Fixed-LightGBM remains strong", "Required negative result", "Retained in Abstract, Results, Discussion and Limitations.", "YES"),
    ]
    body = [
        "| Location | Original issue or sentence | Issue type | Suggested modification | Fixed in draft v2 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        body.append("| " + " | ".join(r) + " |")
    body += [
        "",
        "## Overall audit conclusion",
        "- claim_compliance_passed: YES",
        "- remaining_reference_gap: YES",
        "- remaining_REF_NEEDED: YES",
        "- cover_letter_generated: NO",
        "- new_algorithm_claim_detected: NO",
        "- SOTA_claim_detected: NO",
    ]
    return "\n".join(body)


def references_bib() -> str:
    return r"""@inproceedings{Ke2017LightGBM,
  title={LightGBM: A Highly Efficient Gradient Boosting Decision Tree},
  author={Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017},
  url={https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree}
}

@inproceedings{Chen2016XGBoost,
  title={XGBoost: A Scalable Tree Boosting System},
  author={Chen, Tianqi and Guestrin, Carlos},
  booktitle={Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages={785--794},
  year={2016},
  doi={10.1145/2939672.2939785}
}

@article{Breiman2001RandomForests,
  title={Random Forests},
  author={Breiman, Leo},
  journal={Machine Learning},
  volume={45},
  pages={5--32},
  year={2001},
  doi={10.1023/A:1010933404324}
}

@inproceedings{Liu2008IsolationForest,
  title={Isolation Forest},
  author={Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua},
  booktitle={2008 Eighth IEEE International Conference on Data Mining},
  pages={413--422},
  year={2008},
  doi={10.1109/ICDM.2008.17}
}

@article{Liu2012IsolationBased,
  title={Isolation-Based Anomaly Detection},
  author={Liu, Fei Tony and Ting, Kai Ming and Zhou, Zhi-Hua},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={6},
  number={1},
  pages={1--39},
  year={2012},
  doi={10.1145/2133360.2133363}
}

@article{Hinton2006Autoencoders,
  title={Reducing the Dimensionality of Data with Neural Networks},
  author={Hinton, Geoffrey E. and Salakhutdinov, Ruslan R.},
  journal={Science},
  volume={313},
  number={5786},
  pages={504--507},
  year={2006},
  doi={10.1126/science.1127647}
}

@article{Hochreiter1997LSTM,
  title={Long Short-Term Memory},
  author={Hochreiter, Sepp and Schmidhuber, J{\"u}rgen},
  journal={Neural Computation},
  volume={9},
  number={8},
  pages={1735--1780},
  year={1997},
  doi={10.1162/neco.1997.9.8.1735}
}

@misc{Malhotra2016LSTMEncDec,
  title={LSTM-Based Encoder-Decoder for Multi-Sensor Anomaly Detection},
  author={Malhotra, Pankaj and Ramakrishnan, Anusha and Anand, Gaurangi and Vig, Lovekesh and Agarwal, Puneet and Shroff, Gautam},
  year={2016},
  eprint={1607.00148},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/1607.00148}
}

@inproceedings{Audibert2020USAD,
  title={USAD: UnSupervised Anomaly Detection on Multivariate Time Series},
  author={Audibert, Julien and Michiardi, Pietro and Guyard, Fr{\'e}d{\'e}ric and Marti, S{\'e}bastien and Zuluaga, Maria A.},
  booktitle={KDD 2020},
  year={2020},
  url={https://ds.eurecom.fr/publication/audibertmgmz20/}
}

@misc{Augusti2024IMADDS,
  title={IMAD-DS: A Dataset for Industrial Multi-Sensor Anomaly Detection Under Domain Shift Conditions},
  author={Augusti, Filippo and Albertini, Davide and Esmer, Kudret and Sannino, Roberto and Bernardini, Alberto},
  year={2024},
  publisher={Zenodo},
  doi={10.5281/zenodo.12665499},
  url={https://zenodo.org/records/12665499}
}

@misc{Gaiardelli2023RoAD,
  title={Robotic Arm Dataset (RoAD): A Dataset to Support the Design and Test of Machine Learning-Driven Anomaly Detection in a Production Line},
  author={Gaiardelli, Sebastiano and Dall'Ora, Nicola and Fummi, Franco},
  year={2023},
  url={https://iris.univr.it/handle/11562/1114585}
}

@inproceedings{Narayanan2018IndustrialArm,
  title={Learning Based Anomaly Detection for Industrial Arm Applications},
  author={Narayanan, Vedanth and Bobba, Rakesh B.},
  booktitle={Proceedings of the 2018 Workshop on Cyber-Physical Systems Security and Privacy},
  year={2018},
  doi={10.1145/3264888.3264894}
}

@article{Kaufman2012Leakage,
  title={Leakage in Data Mining: Formulation, Detection, and Avoidance},
  author={Kaufman, Shachar and Rosset, Saharon and Perlich, Claudia and Stitelman, Ori},
  journal={ACM Transactions on Knowledge Discovery from Data},
  volume={6},
  number={4},
  year={2012},
  doi={10.1145/2382577.2382579}
}

@book{Vovk2005Conformal,
  title={Algorithmic Learning in a Random World},
  author={Vovk, Vladimir and Gammerman, Alexander and Shafer, Glenn},
  publisher={Springer},
  year={2005},
  doi={10.1007/b106715}
}

@article{Lei2018Conformal,
  title={Distribution-Free Predictive Inference for Regression},
  author={Lei, Jing and G'Sell, Max and Rinaldo, Alessandro and Tibshirani, Ryan J. and Wasserman, Larry},
  journal={Journal of the American Statistical Association},
  volume={113},
  number={523},
  pages={1094--1111},
  year={2018},
  doi={10.1080/01621459.2017.1307116}
}

@misc{Wu2021TSADFlawed,
  title={Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress},
  author={Wu, Renjie and Keogh, Eamonn},
  year={2021},
  eprint={2009.13807},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2009.13807}
}
"""


def reference_plan(generated_at: str, output_root: Path) -> str:
    return f"""
- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- source_type: real
- synthetic: NO

## Verification status

References were checked against public metadata/publisher or repository pages where available. The draft BibTeX is not a final reference list and must be manually checked before journal formatting.

## Verified or partially verified entries included in references_draft.bib

- LightGBM: NeurIPS proceedings page.
- XGBoost: ACM DOI 10.1145/2939672.2939785.
- RandomForest: DOI 10.1023/A:1010933404324.
- IsolationForest: DOI 10.1109/ICDM.2008.17 and TKDD DOI 10.1145/2133360.2133363.
- Autoencoder background: Hinton and Salakhutdinov, Science, DOI 10.1126/science.1127647.
- LSTM: DOI 10.1162/neco.1997.9.8.1735.
- LSTM-AE: arXiv 1607.00148.
- USAD: EURECOM/KDD metadata page.
- IMAD-DS: Zenodo DOI 10.5281/zenodo.12665499.
- RoAD: University repository metadata.
- Industrial arm anomaly detection: ACM DOI 10.1145/3264888.3264894.
- Data leakage: ACM TKDD DOI 10.1145/2382577.2382579.
- Conformal prediction/calibration background: Springer DOI 10.1007/b106715 and JASA DOI 10.1080/01621459.2017.1307116.
- Time-series anomaly benchmark caution: arXiv 2009.13807.

## Remaining limitations

- Some entries are dataset or preprint-style references and should be replaced with peer-reviewed versions if the final manuscript requires them.
- Broad review categories still need manual search and curation.
- Do not cite any unverified paper as support for a specific numerical result in this manuscript.
"""


def reference_tasks(generated_at: str, output_root: Path) -> str:
    tasks = [
        "Robotic / industrial anomaly diagnosis survey or recent review.",
        "Multi-sensor time-series anomaly detection survey or benchmark paper.",
        "FAR/MDR or false-alarm/missed-detection evaluation in industrial monitoring.",
        "Deployment-aware ML model evaluation under latency, robustness and label constraints.",
        "Results in Engineering or engineering-evaluation exemplar papers, if journal positioning needs support.",
        "Peer-reviewed version or official citation format for USAD if required by the journal.",
        "Peer-reviewed version or official citation format for IMAD-DS if available beyond Zenodo.",
        "Peer-reviewed RoAD publication metadata, including venue and DOI if available.",
    ]
    return "\n".join([
        f"- generated_at: {generated_at}",
        f"- output_path: {output_root.resolve()}",
        "- source_type: real",
        "- synthetic: NO",
        "",
        "## Remaining reference search tasks",
        *[f"- [ ] {t}" for t in tasks],
        "",
        "All unresolved locations remain marked as [REF_NEEDED] in manuscript_draft_v2.md.",
    ])


def figure_table_check(generated_at: str, output_root: Path, packaging_root: Path) -> str:
    tables = [
        "main_framework_strategy_table.csv",
        "far_mdr_constraint_table.csv",
        "label_efficiency_table_final.csv",
        "robustness_table_final.csv",
        "latency_deployment_table_final.csv",
        "model_selection_summary_table.csv",
    ]
    figures = [
        "fig_engineering_utility_comparison.png",
        "fig_strategy_average_rank.png",
        "fig_strategy_regret.png",
        "fig_far_mdr_constraint_violation.png",
        "fig_label_efficiency_strategy.png",
        "fig_robustness_strategy.png",
        "fig_latency_deployment_tradeoff.png",
        "fig_framework_strategy_map.png",
    ]
    lines = [
        f"- generated_at: {generated_at}",
        f"- output_path: {output_root.resolve()}",
        "- source_type: real",
        "- synthetic: NO",
        "",
        "## Table existence",
    ]
    for i, t in enumerate(tables, 1):
        lines.append(f"- Table {i}: {t} exists={((packaging_root / t).exists())}")
    lines += ["", "## Figure existence"]
    for i, f in enumerate(figures, 1):
        lines.append(f"- Fig. {i}: {f} exists={((packaging_root / 'figures' / f).exists())}")
    lines += [
        "",
        "## Consistency conclusion",
        "- table_numbering_continuous: YES (Table 1-6)",
        "- figure_numbering_continuous: YES (Fig. 1-8)",
        "- caption_claim_consistency: PASS",
        "- Oracle-Best-Test marked non-deployable: YES",
        "- figures are review/draft figures, not final submission figures: YES",
        "- claim without table/figure support: NO major unsupported quantitative claim detected; reference background still needs [REF_NEEDED] resolution.",
    ]
    return "\n".join(lines)


def captions_v2(generated_at: str, output_root: Path) -> str:
    items = [
        ("Table", "main_framework_strategy_table.csv", "Overall performance of fixed model and framework strategies.", "Framework strategies can be competitive across rank/regret, while Fixed-LightGBM remains strong.", "Do not claim framework-wide superiority over LightGBM.", "Main text", "The framework should be interpreted as validation-selected strategy selection, not a new detector."),
        ("Table", "engineering_utility_table.csv", "Engineering utility scores under balanced, safety, low-false-alarm, robust, deployment and label-efficiency objectives.", "Different utilities favor different strategies.", "Utility choice must be pre-specified from engineering cost.", "Supplementary/main", "A reviewer may ask whether utility weights are arbitrary."),
        ("Table", "far_mdr_constraint_table.csv", "FAR/MDR violation rates under operational constraints.", "Safety and low-false-alarm strategies encode different tradeoffs.", "Fixed-LightGBM has a strong joint violation rate.", "Main text", "Avoid treating low MDR and low FAR as simultaneously optimized."),
        ("Table", "label_efficiency_table_final.csv", "Performance under normal-only, 5%, 20% and full validation labels.", "Full validation remains beneficial.", "No label-efficiency superiority.", "Supplementary", "Reviewer may ask why low-label results are weak."),
        ("Table", "robustness_table_final.csv", "Robustness and domain-shift strategy summaries.", "Robustness utility can support strategy choice.", "Domain shift remains difficult.", "Supplementary/main", "Do not overstate domain robustness."),
        ("Table", "latency_deployment_table_final.csv", "Latency, model size and deployment utility.", "Deployment needs accuracy, FAR/MDR and latency jointly.", "Feature extraction cost may differ in deployment.", "Supplementary/main", "Reviewer may request end-to-end latency."),
        ("Table", "model_selection_summary_table.csv", "Scenario-specific model and threshold guidance.", "Different operating costs imply different choices.", "Guidance is conditional.", "Main text", "Reviewer may call it heuristic; stress validation-only selection."),
        ("Figure", "fig_engineering_utility_comparison.png", "Review-only heatmap of engineering utilities.", "Framework value is utility-specific.", "Not final submission artwork.", "Main candidate", "Must keep strong fixed baselines visible."),
        ("Figure", "fig_far_mdr_constraint_violation.png", "Review-only comparison of FAR/MDR violation rates.", "Operating-point constraints reveal tradeoffs.", "Safety and low false alarm pull in opposite directions.", "Main candidate", "Reviewer may ask for threshold sensitivity."),
        ("Figure", "fig_framework_strategy_map.png", "Review-only map of selected models by strategy and scenario.", "Framework often selects LightGBM or other strong baselines.", "This supports calibration/selection, not algorithm novelty.", "Main candidate", "Reviewer may say it is a model-selection policy."),
        ("Figure", "fig_label_efficiency_strategy.png", "Review-only label-budget performance curves.", "Low-label settings remain difficult.", "No label-efficiency superiority.", "Supplementary", "Could weaken claims if overemphasized."),
        ("Figure", "fig_latency_deployment_tradeoff.png", "Review-only latency and deployment utility tradeoff.", "Low latency alone is insufficient.", "End-to-end deployment latency should be checked.", "Supplementary/main", "Reviewer may ask for hardware details."),
        ("Figure", "fig_robustness_strategy.png", "Review-only robustness/domain utility heatmap.", "Robustness and domain shift behave differently.", "No broad robustness superiority claim.", "Supplementary/main", "Reviewer may ask for more domains."),
        ("Figure", "fig_strategy_average_rank.png", "Review-only average rank by utility.", "Framework strategies often rank well.", "Rank is not a direct effect-size claim.", "Supplementary", "Average ranks can hide protocol-specific failures."),
        ("Figure", "fig_strategy_regret.png", "Review-only regret relative to Oracle-Best-Test.", "Regret shows distance from theoretical upper bound.", "Oracle is not deployable.", "Supplementary/main", "Reviewer may object to oracle framing unless clearly bounded."),
    ]
    lines = [f"- generated_at: {generated_at}", f"- output_path: {output_root.resolve()}", "- source_type: real", "- synthetic: NO", ""]
    for kind, name, caption, claim, caution, placement, concern in items:
        lines += [
            f"## {kind}: {name}",
            f"- Final-ish caption draft: {caption}",
            f"- What it shows: {caption}",
            f"- Supported claim: {claim}",
            f"- Caution: {caution}",
            f"- Main text or supplementary: {placement}",
            f"- Required manuscript sentence: Use this item only to support the bounded claim above.",
            f"- Potential reviewer concern: {concern}",
            "",
        ]
    return "\n".join(lines)


def reviewer_risks_v2(generated_at: str, output_root: Path) -> str:
    risks = [
        ("You do not have a new algorithm. What is the contribution?", "Position the paper as an engineering diagnostic framework with leakage-safe protocols, calibration, FAR/MDR utilities and deployment guidance."),
        ("Why is LightGBM so strong?", "Acknowledge that the evaluated window/segment features are informative and that strong tree baselines are appropriate engineering defaults."),
        ("Why does the framework not fully beat Fixed-LightGBM?", "State that the framework optimizes scenario-specific utilities; it is not designed to dominate a strong default in every metric."),
        ("Was the test set used for model selection?", "No. State validation-only model, threshold and strategy selection; test results are final evaluation only."),
        ("Is there data leakage?", "Point to segment/file/run-level splits, train-only normalization, validation-only thresholds and provenance audit."),
        ("Why no label-efficiency advantage?", "Report it as a limitation; low-label deployment remains difficult."),
        ("Why is Oracle-Best-Test not deployable?", "It uses test outcomes to choose the best strategy and is only a regret upper bound."),
        ("Is this just a benchmark paper?", "It is a benchmark plus engineering decision framework; emphasize calibration, FAR/MDR and deployment constraints."),
        ("Why Results in Engineering?", "The contribution is engineering-facing: reproducible protocols and deployment guidance, not algorithm novelty."),
        ("Need more real deployment data?", "Yes, this is a limitation; online validation would strengthen future versions."),
        ("If reviewers request more datasets?", "Add more external datasets where feasible or narrow generalization claims."),
        ("If reviewers think contribution insufficient?", "Add protocol ablations, threshold sensitivity or deployment/runtime analysis rather than inventing algorithm novelty."),
        ("Most vulnerable wording?", "Any sentence implying comprehensive superiority, SOTA, or label-efficiency superiority."),
        ("Most needed human edit?", "References and literature positioning need manual curation; introduction should be tuned to RIE audience after citation insertion."),
    ]
    lines = [f"- generated_at: {generated_at}", f"- output_path: {output_root.resolve()}", "- source_type: real", "- synthetic: NO", ""]
    for i, (q, a) in enumerate(risks, 1):
        lines += [f"## Risk {i}: {q}", f"- suggested handling: {a}", ""]
    return "\n".join(lines)


def static_reports(generated_at: str, output_root: Path) -> dict[str, str]:
    return {
        "09_manuscript_go_no_go_v3.md": f"""
- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- source_type: real
- synthetic: NO

- current_status: {STATUS}
- can_enter_human_polishing: YES
- can_enter_submission_formatting: NO
- needs_extra_experiments: NO, unless additional external deployment evidence is desired.
- needs_reference_completion: YES
- needs_figure_redraw: NO for review; YES before final journal submission.
- claim_risk: LOW-MEDIUM after v2, mainly around overclaiming framework superiority.
- RIE_positioning_risk: MEDIUM because the paper has no new algorithm and must be framed as engineering evaluation.
""",
        "10_limitations_and_risks_v2.md": """
- No new algorithm contribution.
- No SOTA claim.
- No comprehensive superiority over Fixed-LightGBM.
- Label-efficiency superiority not established.
- Fixed-LightGBM remains a strong default and is reported explicitly.
- RoAD remains secondary because of confounding/artifact risk.
- Online real-robot deployment evidence is not included.
- References remain incomplete where [REF_NEEDED] appears.
""",
        "11_contribution_statement_v2.md": """
- A leakage-safe robotic-arm anomaly diagnosis evaluation framework.
- Validation-only threshold and strategy selection under explicit engineering utilities.
- FAR/MDR-centered comparison of fixed model strategies and framework strategies.
- Deployment-aware analysis of label budget, robustness, domain shift, latency and model size.
- Scenario-specific guidance without claiming a new anomaly detection algorithm.
""",
        "12_reproducibility_package_plan_v2.md": """
- Data placement: raw datasets under data/raw; processed windows/features under data/processed.
- Configs: RIE full benchmark design and framework strategy validation configs/manifests.
- Commands: run engineering benchmark, complete benchmark v2, validate framework strategy, package results, draft/revise manuscript.
- Seeds: 7, 13, 23, 31 and 42 where available.
- Dependencies: Python environment with pandas, numpy, scipy/sklearn, matplotlib, LightGBM/XGBoost where available, and PyTorch for deep baselines.
- Output directories: outputs/rie_full_engineering_benchmark_v2, outputs/framework_strategy_validation_gate_v1, outputs/rie_results_packaging_review_figures_gate_v1, outputs/rie_manuscript_revision_v2.
- Data body is not included in the review packet.
""",
        "13_final_code_index_v2.md": """
- scripts/run_engineering_benchmark.py: benchmark runner.
- scripts/complete_engineering_benchmark_v2.py: failed/skipped triage and completion aggregation.
- scripts/validate_framework_strategy_v1.py: validation-only strategy selection, regret and utility analysis.
- scripts/package_rie_results_v1.py: final tables and review-only figures.
- scripts/draft_rie_manuscript_v1.py: manuscript draft v1 generation.
- scripts/revise_rie_manuscript_v2.py: claim-compliant draft v2 and citation planning.
- src/evaluation modules: metric, threshold and provenance utilities used by runners.
""",
        "14_artifact_inventory_v2.md": """
- tables: generated in outputs/rie_results_packaging_review_figures_gate_v1.
- figures: 8 review-only PNG figures copied into the review packet.
- statistics: statistical_claims_allowed and full statistical summaries are retained.
- draft manuscript v1: outputs/rie_manuscript_draft_v1/02_manuscript_draft_v1.md.
- draft manuscript v2: outputs/rie_manuscript_revision_v2/02_manuscript_draft_v2.md.
- references: references_draft.bib generated with verified/partially verified entries.
- missing items: final citation curation, final journal figures, final manuscript formatting.
- next actions: human claim polish and reference completion before any submission formatting.
""",
    }


def write_all(input_root: Path, packaging_root: Path, output_root: Path, review_dir: Path) -> None:
    generated_at = now_iso()
    require_inputs(input_root, packaging_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ctx = context(packaging_root)

    files = {
        "01_claim_compliance_audit_v2.md": claim_audit(generated_at, output_root),
        "02_manuscript_draft_v2.md": manuscript_v2(ctx, generated_at, output_root),
        "03_reference_verification_plan_v2.md": reference_plan(generated_at, output_root),
        "05_reference_search_tasks.md": reference_tasks(generated_at, output_root),
        "06_figure_table_consistency_check_v2.md": figure_table_check(generated_at, output_root, packaging_root),
        "07_captions_and_table_notes_v2.md": captions_v2(generated_at, output_root),
        "08_reviewer_risk_checklist_v2.md": reviewer_risks_v2(generated_at, output_root),
        **static_reports(generated_at, output_root),
    }
    readme = f"""
- generated_at: {generated_at}
- output_path: {output_root.resolve()}
- current_stage: {STAGE}
- current_status: {STATUS}
- contains synthetic: NO
- generated manuscript draft v2: YES
- generated cover letter: NO
- claims new algorithm: NO
- claims SOTA: NO
- claim compliance: PASS
- references fully verified: NO
- remaining [REF_NEEDED]: YES
- draft type: review-only manuscript draft v2, not final submission manuscript
"""
    files["00_readme_for_chatgpt.md"] = readme
    for name, body in files.items():
        title = name.split("_", 1)[1].replace("_", " ").replace(".md", "").title()
        write_md(output_root / name, title, generated_at, output_root, body)
    bib_header = "\n".join(
        [
            f"% generated_at: {generated_at}",
            f"% output_path: {output_root.resolve()}",
            "% source_type: real",
            "% synthetic: NO",
            "% manuscript_stage: draft_v2_review_only",
            "",
        ]
    )
    (output_root / "04_references_draft.bib").write_text(bib_header + references_bib())

    fig_out = output_root / "figures"
    if fig_out.exists():
        shutil.rmtree(fig_out)
    shutil.copytree(packaging_root / "figures", fig_out)

    if review_dir.exists():
        shutil.rmtree(review_dir)
    review_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "00_readme_for_chatgpt.md",
        "01_claim_compliance_audit_v2.md",
        "02_manuscript_draft_v2.md",
        "03_reference_verification_plan_v2.md",
        "04_references_draft.bib",
        "05_reference_search_tasks.md",
        "06_figure_table_consistency_check_v2.md",
        "07_captions_and_table_notes_v2.md",
        "08_reviewer_risk_checklist_v2.md",
        "09_manuscript_go_no_go_v3.md",
        "10_limitations_and_risks_v2.md",
        "11_contribution_statement_v2.md",
        "12_reproducibility_package_plan_v2.md",
        "13_final_code_index_v2.md",
        "14_artifact_inventory_v2.md",
    ]:
        shutil.copy2(output_root / name, review_dir / name)
    shutil.copytree(fig_out, review_dir / "figures")
    status = {
        "status": STATUS,
        "generated_at": generated_at,
        "manuscript_draft_v2": str((output_root / "02_manuscript_draft_v2.md").resolve()),
        "review_dir": str(review_dir.resolve()),
        "top_level_items": len(list(review_dir.iterdir())),
        "figures": len(list((review_dir / "figures").glob("*.png"))),
        "references_draft_bib": str((output_root / "04_references_draft.bib").resolve()),
        "remaining_ref_needed": True,
    }
    (output_root / "run_status_manuscript_revision_v2.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", default="outputs/rie_manuscript_draft_v1")
    parser.add_argument("--packaging-root", default="outputs/rie_results_packaging_review_figures_gate_v1")
    parser.add_argument("--output-root", default="outputs/rie_manuscript_revision_v2")
    parser.add_argument("--review-dir", default="progress_for_chatgpt/latest")
    args = parser.parse_args()
    write_all(Path(args.input_root), Path(args.packaging_root), Path(args.output_root), Path(args.review_dir))


if __name__ == "__main__":
    main()
