# Workflow Figure Redraw Report v11

generated_at: 2026-06-06 14:25:39 Asia/Macau
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v11/workflow_figure_redraw_report_v11.md

status: COMPLETED
figure_redrawn: YES
backend: Python/matplotlib
png_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v11/figures/fig1_framework_workflow_v11.png
pdf_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v11/figures/fig1_framework_workflow_v11.pdf
inserted_in_latex: YES, PDF version used

Figure contract:
- Core conclusion: the framework separates training, validation/calibration selection and held-out test reporting for auditable deployment decisions.
- Archetype: schematic-led composite.
- Evidence chain: raw data -> leakage-safe split -> train-only normalization -> candidate model pool -> validation-only calibration -> utility selection -> test-only evaluation -> guidance/deployment decision.

Design changes:
- Reduced text density compared with v10 workflow.
- Separated training/validation/test roles visually.
- Used restrained colors and direct labels.
- Explicitly states that candidate models are baselines, not a proposed detector.

No experimental data were changed.
