# Rank and Win Summary

- dataset: real IMAD-DS benchmark outputs
- source_type: external_real
- protocol: framework_strategy_validation_gate_v1
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T14:24:59+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1

| deployed_strategy | utility | average_rank | win_count | top2_count | n |
| --- | --- | --- | --- | --- | --- |
| Oracle-Best-Test | utility_balanced | 1.1 | 98 | 98 | 100 |
| Framework-Best-F1 | utility_balanced | 2.57 | 30 | 49 | 100 |
| Framework-Balanced | utility_balanced | 2.58 | 30 | 49 | 100 |
| Framework-Label-Efficient | utility_balanced | 2.58 | 30 | 49 | 100 |
| Framework-Robust | utility_balanced | 2.81 | 28 | 46 | 100 |
| Framework-Deployment | utility_balanced | 3.01 | 27 | 45 | 100 |
| Fixed-LightGBM | utility_balanced | 3.957894736842105 | 18 | 36 | 95 |
| Framework-Low-False-Alarm | utility_balanced | 6.88 | 10 | 19 | 100 |
| Framework-Safety | utility_balanced | 7.0 | 15 | 21 | 100 |
| Fixed-RandomForest | utility_balanced | 7.494736842105263 | 17 | 25 | 95 |
| Fixed-XGBoost | utility_balanced | 8.042105263157895 | 9 | 16 | 95 |
| Fixed-USAD | utility_balanced | 11.15 | 3 | 4 | 60 |
| Fixed-AutoEncoder | utility_balanced | 11.45 | 1 | 2 | 60 |
| Fixed-IsolationForest | utility_balanced | 12.37 | 2 | 5 | 100 |
| Fixed-LSTM-AE | utility_balanced | 13.366666666666667 | 1 | 1 | 60 |
| Oracle-Best-Test | utility_deployment | 1.27 | 92 | 96 | 100 |
| Framework-Best-F1 | utility_deployment | 2.56 | 31 | 49 | 100 |
| Framework-Balanced | utility_deployment | 2.57 | 31 | 49 | 100 |
| Framework-Label-Efficient | utility_deployment | 2.57 | 31 | 49 | 100 |
| Framework-Robust | utility_deployment | 2.88 | 29 | 46 | 100 |
| Framework-Deployment | utility_deployment | 2.98 | 29 | 46 | 100 |
| Fixed-LightGBM | utility_deployment | 3.9263157894736844 | 19 | 37 | 95 |
| Framework-Low-False-Alarm | utility_deployment | 6.85 | 11 | 18 | 100 |
| Framework-Safety | utility_deployment | 7.07 | 15 | 19 | 100 |
| Fixed-RandomForest | utility_deployment | 7.831578947368421 | 13 | 19 | 95 |
| Fixed-XGBoost | utility_deployment | 8.042105263157895 | 11 | 15 | 95 |
| Fixed-AutoEncoder | utility_deployment | 10.95 | 3 | 4 | 60 |
| Fixed-USAD | utility_deployment | 11.75 | 1 | 3 | 60 |
| Fixed-IsolationForest | utility_deployment | 12.17 | 3 | 5 | 100 |
| Fixed-LSTM-AE | utility_deployment | 13.25 | 1 | 1 | 60 |
| Oracle-Best-Test | utility_label_efficiency | 1.1 | 98 | 98 | 100 |
| Framework-Best-F1 | utility_label_efficiency | 2.57 | 30 | 49 | 100 |
| Framework-Balanced | utility_label_efficiency | 2.58 | 30 | 49 | 100 |
| Framework-Label-Efficient | utility_label_efficiency | 2.58 | 30 | 49 | 100 |
| Framework-Robust | utility_label_efficiency | 2.81 | 28 | 46 | 100 |
| Framework-Deployment | utility_label_efficiency | 3.01 | 27 | 45 | 100 |
| Fixed-LightGBM | utility_label_efficiency | 3.957894736842105 | 18 | 36 | 95 |
| Framework-Low-False-Alarm | utility_label_efficiency | 6.88 | 10 | 19 | 100 |
| Framework-Safety | utility_label_efficiency | 7.0 | 15 | 21 | 100 |
| Fixed-RandomForest | utility_label_efficiency | 7.494736842105263 | 17 | 25 | 95 |
| Fixed-XGBoost | utility_label_efficiency | 8.042105263157895 | 9 | 16 | 95 |
| Fixed-USAD | utility_label_efficiency | 11.15 | 3 | 4 | 60 |
| Fixed-AutoEncoder | utility_label_efficiency | 11.45 | 1 | 2 | 60 |
| Fixed-IsolationForest | utility_label_efficiency | 12.37 | 2 | 5 | 100 |
| Fixed-LSTM-AE | utility_label_efficiency | 13.366666666666667 | 1 | 1 | 60 |
| Framework-Low-False-Alarm | utility_low_false_alarm | 2.42 | 60 | 72 | 100 |
| Framework-Balanced | utility_low_false_alarm | 2.84 | 16 | 47 | 100 |
| Framework-Label-Efficient | utility_low_false_alarm | 2.84 | 16 | 47 | 100 |
| Framework-Best-F1 | utility_low_false_alarm | 2.93 | 15 | 45 | 100 |
| Oracle-Best-Test | utility_low_false_alarm | 3.04 | 42 | 74 | 100 |
| Framework-Robust | utility_low_false_alarm | 3.15 | 16 | 43 | 100 |
| Framework-Deployment | utility_low_false_alarm | 3.23 | 14 | 43 | 100 |
| Fixed-LightGBM | utility_low_false_alarm | 4.126315789473685 | 9 | 35 | 95 |
| Fixed-RandomForest | utility_low_false_alarm | 7.242105263157895 | 13 | 19 | 95 |
| Fixed-XGBoost | utility_low_false_alarm | 7.747368421052632 | 6 | 13 | 95 |
| Framework-Safety | utility_low_false_alarm | 7.84 | 12 | 22 | 100 |
| Fixed-LSTM-AE | utility_low_false_alarm | 10.433333333333334 | 0 | 5 | 60 |
| Fixed-USAD | utility_low_false_alarm | 12.55 | 0 | 1 | 60 |
| Fixed-IsolationForest | utility_low_false_alarm | 12.86 | 0 | 0 | 100 |
| Fixed-AutoEncoder | utility_low_false_alarm | 13.533333333333333 | 0 | 0 | 60 |
| Oracle-Best-Test | utility_robust | 1.1 | 98 | 98 | 100 |
| Framework-Best-F1 | utility_robust | 2.57 | 30 | 49 | 100 |
| Framework-Balanced | utility_robust | 2.58 | 30 | 49 | 100 |
| Framework-Label-Efficient | utility_robust | 2.58 | 30 | 49 | 100 |
| Framework-Robust | utility_robust | 2.81 | 28 | 46 | 100 |
| Framework-Deployment | utility_robust | 3.01 | 27 | 45 | 100 |
| Fixed-LightGBM | utility_robust | 3.957894736842105 | 18 | 36 | 95 |
| Framework-Low-False-Alarm | utility_robust | 6.88 | 10 | 19 | 100 |
| Framework-Safety | utility_robust | 7.0 | 15 | 21 | 100 |
| Fixed-RandomForest | utility_robust | 7.494736842105263 | 17 | 25 | 95 |
| Fixed-XGBoost | utility_robust | 8.042105263157895 | 9 | 16 | 95 |
| Fixed-USAD | utility_robust | 11.15 | 3 | 4 | 60 |
| Fixed-AutoEncoder | utility_robust | 11.45 | 1 | 2 | 60 |
| Fixed-IsolationForest | utility_robust | 12.37 | 2 | 5 | 100 |
| Fixed-LSTM-AE | utility_robust | 13.366666666666667 | 1 | 1 | 60 |
| Framework-Safety | utility_safety | 2.43 | 57 | 66 | 100 |
| Framework-Best-F1 | utility_safety | 2.79 | 24 | 52 | 100 |
| Framework-Balanced | utility_safety | 2.9 | 23 | 50 | 100 |
| Framework-Label-Efficient | utility_safety | 2.9 | 23 | 50 | 100 |
| Framework-Robust | utility_safety | 2.96 | 24 | 50 | 100 |
| Oracle-Best-Test | utility_safety | 3.0 | 44 | 71 | 100 |
| Framework-Deployment | utility_safety | 3.46 | 17 | 44 | 100 |
| Fixed-LightGBM | utility_safety | 4.063157894736842 | 14 | 41 | 95 |
| Fixed-RandomForest | utility_safety | 7.768421052631579 | 16 | 21 | 95 |
| Framework-Low-False-Alarm | utility_safety | 8.54 | 5 | 11 | 100 |
| Fixed-XGBoost | utility_safety | 8.68421052631579 | 6 | 10 | 95 |
| Fixed-AutoEncoder | utility_safety | 10.416666666666666 | 2 | 6 | 60 |
| Fixed-USAD | utility_safety | 11.033333333333333 | 1 | 6 | 60 |
| Fixed-IsolationForest | utility_safety | 11.33 | 5 | 6 | 100 |
| Fixed-LSTM-AE | utility_safety | 14.25 | 0 | 0 | 60 |
