# Model Selection Guidelines Final v2

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

- Label-rich deployment: start with LightGBM/XGBoost/RandomForest and validate FAR/MDR thresholds.
- Label-scarce or normal-only deployment: use unsupervised baselines and report empirical FAR; supervised trees are not applicable without anomaly labels.
- Domain-shift deployment: select by source-to-target and leave-target-weight35-out rankings, not main-binary ranking.
- Missing sensor/noise deployment: select by robustness protocol degradation.
- Low-latency CPU deployment: prefer the fastest calibrated model that satisfies FAR/MDR constraints.
- Historical failed references should remain appendix/negative evidence, not a new-method claim.
