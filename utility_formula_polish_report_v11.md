# Utility Formula Polish Report v11

generated_at: 2026-06-06 14:25:39 Asia/Macau
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v11/utility_formula_polish_report_v11.md

status: COMPLETED
utility_definitions_polished: YES

Variables clarified:
- M: candidate model pool.
- m: a candidate model.
- tau: operating threshold.
- T_m: candidate threshold set for model m.
- D_val: validation/calibration split for selection.
- D_test: held-out test split for final reporting.
- U_k: utility for engineering objective k.
- P_latency and P_size: predeclared deployment penalties.
- Delta_robust: validation-estimated robustness degradation.
- b: validation-label budget.

Interpretation added:
- Utility weights are engineering preferences fixed before test reporting.
- Deployment utility is a joint operating score, not a pure latency ranking.
- Robust utility depends on validation robustness degradation.
- Label-efficient utility does not prove label-efficiency superiority.
