# Abstract and RQ Alignment Report v14

generated_at: 2026-06-06T08:08:32+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/progress_for_chatgpt/latest/02_abstract_rq_alignment_report_v14.md


- abstract_includes_key_numbers: YES
- fixed_lightgbm_numbers: macro-F1 0.785; PR-AUC 0.877; FAR 0.169; MDR 0.240
- framework_safety_numbers: MDR 0.154; FAR 0.322
- framework_low_false_alarm_numbers: FAR 0.087; MDR 0.365
- balanced_over_lightgbm_not_significant: YES
- sota_claim: NO
- new_algorithm_claim: NO


- rq1_rewritten: YES
- rq1_alignment: Now asks comparison in utility/regret relative to strongest fixed baseline, not guaranteed regret reduction.
- rq1_answer_added: Framework improves over weaker fixed baselines in several utilities but does not reduce balanced regret over Fixed-LightGBM.
- rq2_answer_retained: Safety and low-false-alarm improve their target utilities; balanced/robust/deployment do not establish broad LightGBM superiority.
- rq3_answer_retained: Label budget, missing sensors, domain shift and latency affect recommendations.
