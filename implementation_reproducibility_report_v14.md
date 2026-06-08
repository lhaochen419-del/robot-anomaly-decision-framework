# Implementation Reproducibility Report v14

generated_at: 2026-06-06T07:51:37+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v14/implementation_reproducibility_report_v14.md

- section_expanded: YES
- feature_details_added: mean, std, min, max and endpoint-difference for tree raw-window features; no unconfirmed FFT/RMS/skew/kurtosis inferred
- raw_window_counts_added: 4364 windows, 7 channels; representative train/val/test 2508/608/624 with 312 normal and 312 anomaly test windows
- stress_protocol_added: random zeroed channel drop 10/20 percent; Gaussian noise 0.02/0.05 times channel std
- hyperparameters_added: LightGBM/XGBoost/RF/IsolationForest/AE/LSTM-AE/USAD settings from run_engineering_benchmark.py
- threshold_candidates_added: best-F1, Youden, FAR targets, recall targets and cost ratios
- latency_limits_added: prepared-feature/window prediction latency only; not end-to-end robot-cell latency
- missing_items: sampling rate, stride/overlap and exact hardware timing protocol remain reproducibility checklist items
