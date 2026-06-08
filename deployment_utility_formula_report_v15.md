generated_at: 2026-06-07T03:37:40+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v15/deployment_utility_formula_report_v15.md

# Deployment Utility Formula Report v15

formula_defined: YES
latency_penalty: 0.05 * log1p(latency_ms) / log1p(max_latency_ms)
size_penalty: 0.05 * log1p(model_size_mb) / log1p(max_model_size_mb)
delta_robust_defined: max(0, clean validation macro-F1 - stressed validation macro-F1) for validation robust selection
source_file: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/scripts/validate_framework_strategy_v1.py
claim_downgraded_or_limited: YES, deployment utility is described as prepared-input prediction-cost utility, not end-to-end robot-cell latency.
