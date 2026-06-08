# FAR/MDR Constraint Analysis

- dataset: real IMAD-DS benchmark outputs
- source_type: external_real
- protocol: framework_strategy_validation_gate_v1
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T14:24:59+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1

- FAR constraint: FAR <= 0.4
- MDR constraint: MDR <= 0.35
| deployed_strategy | n_scenarios | far_violation_rate | mdr_violation_rate | joint_far_mdr_violation_rate |
| --- | --- | --- | --- | --- |
| Fixed-LightGBM | 95 | 0.05263157894736842 | 0.17894736842105263 | 0.23157894736842105 |
| Fixed-XGBoost | 95 | 0.07368421052631578 | 0.17894736842105263 | 0.24210526315789474 |
| Oracle-Best-Test | 100 | 0.08 | 0.21 | 0.25 |
| Framework-Deployment | 100 | 0.07 | 0.22 | 0.27 |
| Framework-Best-F1 | 100 | 0.08 | 0.22 | 0.28 |
| Framework-Robust | 100 | 0.09 | 0.21 | 0.28 |
| Framework-Balanced | 100 | 0.08 | 0.23 | 0.29 |
| Framework-Label-Efficient | 100 | 0.08 | 0.23 | 0.29 |
| Fixed-RandomForest | 95 | 0.09473684210526316 | 0.24210526315789474 | 0.3368421052631579 |
| Framework-Low-False-Alarm | 100 | 0.03 | 0.42 | 0.45 |
| Framework-Safety | 100 | 0.37 | 0.11 | 0.48 |
| Fixed-AutoEncoder | 60 | 0.85 | 0.6333333333333333 | 1.0 |
| Fixed-IsolationForest | 100 | 0.8 | 0.52 | 1.0 |
| Fixed-LSTM-AE | 60 | 0.0 | 1.0 | 1.0 |
| Fixed-USAD | 60 | 0.8 | 0.7 | 1.0 |
