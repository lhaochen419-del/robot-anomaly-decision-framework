# Engineering Evidence Synthesis v2

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

- Core IMAD-DS RoboticArm raw-window protocols completed across five seeds.
- Strong tree/statistical baselines, deep reconstruction baselines, and historical references are separated by role.
- Engineering value comes from leakage-safe splits, validation-only calibration, FAR/MDR tradeoff, domain-shift, label-efficiency, robustness, and latency evidence rather than new algorithm claims.
- RoAD remains secondary and high-risk due to prior confounding concerns.
