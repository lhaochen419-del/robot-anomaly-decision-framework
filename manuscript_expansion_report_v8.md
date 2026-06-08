# Manuscript Expansion Report v8

- generated_at: 2026-06-06 13:27:47 Asia/Macau
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v8/manuscript_expansion_report_v8.md

## Summary
- source_draft: outputs/rie_latex_template_draft_v7/main.tex
- target_draft: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v8/main.tex
- experiments_rerun: NO
- experimental_results_changed: NO
- cover_letter_generated: NO
- final_submission_package_generated: NO

## Expanded Sections
- Introduction: added engineering motivation and four bounded framework contributions.
- Data and Leakage-Safe Protocols: expanded dataset role assignment, primary/secondary evidence boundaries, split unit, train-only normalization, validation-only thresholding, no random overlapping-window split, no synthetic results, and provenance tracking.
- Benchmark Models and Framework Strategies: clarified fixed-model strategies, framework input/output, validation-only selection, calibrated thresholds, utility functions, and Oracle-Best-Test as non-deployable.
- Evaluation Metrics and Statistical Analysis: added statistical evidence summary table and reinforced FAR/MDR, regret, rank, utility, and paired-test rationale.
- Results: added engineering interpretation for framework vs fixed defaults, FAR/MDR tradeoffs, label-budget limits, robustness/domain shift, latency/deployment, and model-selection guidance.
- Discussion: expanded practical deployment workflow, operational cost selection, calibration decision layer, and distinction from a pure benchmark paper.
- Limitations: expanded no-new-algorithm, no comprehensive LightGBM superiority, label-efficiency limitation, dataset limits, utility-weight risk, and declaration placeholders.

## Inserted Assets
- workflow_figure_inserted: YES
- experiment_design_table_inserted: YES
- statistical_evidence_table_inserted: YES

## Editing Note
- apply_patch was attempted first, but the workspace path triggered a bwrap sandbox failure. The same targeted replacements were then applied with a controlled Python text-replacement script. No experimental result values were altered.
