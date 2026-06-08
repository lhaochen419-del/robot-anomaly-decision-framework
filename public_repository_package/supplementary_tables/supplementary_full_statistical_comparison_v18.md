# Statistical Evidence Summary

- generated_at: 2026-06-05T15:45:06+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_results_packaging_review_figures_gate_v1
- source_type: real
- synthetic: NO

## Statistical comparisons
| framework_strategy | fixed_strategy | utility | n_pairs | mean_framework_minus_fixed | paired_t_p | wilcoxon_p | sign_test_p | effect_size_dz | wins | losses | generated_at | output_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Framework-Balanced | Fixed-LightGBM | utility_balanced | 95 | 0.00202291 | 0.843102 | 0.137147 | 0.168638 | 0.0203632 | 17 | 9 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Balanced | Fixed-RandomForest | utility_balanced | 95 | 0.114197 | 3.94628e-12 | 9.37403e-11 | 8.90985e-07 | 0.81628 | 60 | 17 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Balanced | Fixed-XGBoost | utility_balanced | 95 | 0.0685505 | 7.30781e-08 | 1.99623e-10 | 4.26054e-10 | 0.599569 | 72 | 15 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Deployment | Fixed-LightGBM | utility_deployment | 95 | -0.00618051 | 0.561922 | 0.585027 | 0.557197 | -0.0597177 | 15 | 11 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Deployment | Fixed-RandomForest | utility_deployment | 95 | 0.117534 | 8.8229e-13 | 5.87575e-11 | 5.2044e-07 | 0.848033 | 63 | 18 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Deployment | Fixed-XGBoost | utility_deployment | 95 | 0.0603098 | 4.48356e-07 | 2.14306e-10 | 1.93819e-09 | 0.556627 | 69 | 15 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Low-False-Alarm | Fixed-LightGBM | utility_low_false_alarm | 95 | 0.094577 | 5.11661e-08 | 7.00941e-10 | 3.91812e-12 | 0.607851 | 72 | 11 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Low-False-Alarm | Fixed-RandomForest | utility_low_false_alarm | 95 | 0.201589 | 3.59433e-13 | 9.74188e-12 | 4.26054e-10 | 0.86699 | 72 | 15 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Low-False-Alarm | Fixed-XGBoost | utility_low_false_alarm | 95 | 0.159072 | 1.16272e-15 | 1.04615e-12 | 1.38385e-16 | 0.987316 | 82 | 8 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Robust | Fixed-LightGBM | utility_robust | 95 | -0.00312393 | 0.775923 | 0.416386 | 0.361595 | -0.0292872 | 18 | 12 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Robust | Fixed-RandomForest | utility_robust | 95 | 0.10905 | 8.39867e-11 | 7.69489e-10 | 4.71324e-06 | 0.750757 | 58 | 18 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Robust | Fixed-XGBoost | utility_robust | 95 | 0.0634036 | 1.54134e-06 | 2.95397e-09 | 8.35092e-09 | 0.526505 | 70 | 17 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Safety | Fixed-LightGBM | utility_safety | 95 | 0.0705768 | 0.000945952 | 4.54912e-08 | 1.41147e-08 | 0.350281 | 65 | 15 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Safety | Fixed-RandomForest | utility_safety | 95 | 0.223531 | 3.01166e-15 | 1.31942e-12 | 7.65884e-12 | 0.96738 | 66 | 9 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
| Framework-Safety | Fixed-XGBoost | utility_safety | 95 | 0.160194 | 3.27466e-13 | 3.58826e-13 | 3.09049e-19 | 0.868953 | 86 | 6 | 2026-06-05T14:24:59+00:00 | /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1 |
