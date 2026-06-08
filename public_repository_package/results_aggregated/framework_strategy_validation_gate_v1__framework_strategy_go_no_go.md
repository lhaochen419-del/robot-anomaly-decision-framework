# Framework Strategy GO/NO-GO

- dataset: real IMAD-DS benchmark outputs
- source_type: external_real
- protocol: framework_strategy_validation_gate_v1
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T14:24:59+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1

## Decision: FRAMEWORK_STRATEGY_GO
| criterion | passed | value |
| --- | --- | --- |
| at_least_5_valid_protocols | True | 20 |
| framework_regret_lower_than_two_strong_fixed | True | 2 |
| framework_top2_in_three_utilities | False | 0 |
| far_mdr_violation_better_than_two_strong_fixed | False | 1 |
| label_efficiency_advantage | False | False |
| robustness_advantage | True | True |
| deployment_advantage | True | True |
| validation_only_selection_no_test_leakage | True | True |
| different_models_selected_across_scenarios | True | 6 |
