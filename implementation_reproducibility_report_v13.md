generated_at: 2026-06-06T07:08:00+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v13/implementation_reproducibility_report_v13.md

# Implementation/Reproducibility Report v13

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
