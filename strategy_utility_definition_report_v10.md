# Strategy Utility Definition Report v10

generated_at: 2026-06-06 14:06:18 
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v10/strategy_utility_definition_report_v10.md

Status: COMPLETED.

Added section: `Strategy utilities and validation-only selection`.

Definitions included:
- Candidate model set M = {LightGBM, XGBoost, RandomForest, IsolationForest, AutoEncoder, LSTM-AE, USAD}.
- Validation/calibration split D_val and held-out test split D_test.
- Selection rule: model-threshold pair is selected by maximizing predeclared utility on validation/calibration data only.
- Test split is used once for final reporting.

Utility definitions included:
- Balanced: macro-F1 - FAR - MDR.
- Safety: macro-F1 - 2*MDR - 0.5*FAR.
- Low false alarm: macro-F1 - 2*FAR - 0.5*MDR.
- Deployment: macro-F1 - FAR - MDR - latency penalty - size penalty.
- Robust: detection utility adjusted by validation robustness degradation.
- Label-efficient: macro-F1 - FAR - MDR under the declared label budget.

Claim boundary:
- The text explicitly states that these are predeclared engineering preferences.
- The framework is described as a model-threshold decision rule, not a new anomaly detector.
- No test-set model or threshold selection was introduced.
