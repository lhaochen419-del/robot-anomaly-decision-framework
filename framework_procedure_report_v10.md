# Framework Procedure Report v10

generated_at: 2026-06-06 14:06:18 
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v10/framework_procedure_report_v10.md

Status: COMPLETED.

Added item: `Procedure 1. Leakage-safe calibration-aware model and threshold selection`.

Procedure contents:
1. Split by segment/file/run unit.
2. Fit normalization on training data only.
3. Train candidate models on training data.
4. Generate validation scores.
5. Calibrate candidate thresholds on validation/calibration data.
6. Compute engineering utility for each model-threshold pair.
7. Select the best validation utility pair.
8. Evaluate once on held-out test data.
9. Report FAR, MDR, PR-AUC, regret, latency and deployment guidance.

Input/output clarity:
- Inputs include multi-sensor robotic-arm data, split units, candidate model pool, engineering objective, validation/calibration split and test split.
- Outputs include selected model, selected threshold, operating strategy, final test metrics and deployment caution.

Claim boundary:
- The manuscript states this is an evaluation and decision workflow, not a new detector.
