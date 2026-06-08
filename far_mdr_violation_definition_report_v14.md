# FAR/MDR Violation Definition Report v14

generated_at: 2026-06-06T07:51:37+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v14/far_mdr_violation_definition_report_v14.md

- far_threshold_alpha: 0.40
- mdr_threshold_beta: 0.35
- joint_violation_rule: FAR > alpha OR MDR > beta
- source_file: scripts/validate_framework_strategy_v1.py lines defining FAR_LIMIT/MDR_LIMIT and joint_far_mdr_violation_rate
- test_data_used_to_tune_thresholds: NO
