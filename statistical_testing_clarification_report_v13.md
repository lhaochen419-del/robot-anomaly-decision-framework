generated_at: 2026-06-06T07:08:00+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v13/statistical_testing_clarification_report_v13.md

# Statistical Testing Clarification Report v13

Added a Statistical testing protocol paragraph clarifying:
- paired sample unit: dataset--protocol--seed--scenario comparison;
- same-test-split dependency is acknowledged;
- tests are paired evidence summaries, not independent field-trial proof;
- Wilcoxon is emphasized because scenario-level differences may be non-normal;
- paired t-test and sign test remain in analysis files;
- Holm correction is applied over planned Wilcoxon comparisons.

Updated statistical evidence table:

Framework-Balanced vs Fixed-LightGBM: n=95, Wilcoxon p=0.1371, Holm p=0.4114, effect=0.020
Framework-Balanced vs Fixed-XGBoost: n=95, Wilcoxon p=2.00e-10, Holm p=1.80e-09, effect=0.600
Framework-Balanced vs Fixed-RandomForest: n=95, Wilcoxon p=9.37e-11, Holm p=9.37e-10, effect=0.816
Framework-Safety vs Fixed-LightGBM: n=95, Wilcoxon p=4.55e-08, Holm p=1.82e-07, effect=0.350
Framework-Low-False-Alarm vs Fixed-LightGBM: n=95, Wilcoxon p=7.01e-10, Holm p=4.91e-09, effect=0.608
Framework-Deployment vs Fixed-XGBoost: n=95, Wilcoxon p=2.14e-10, Holm p=1.80e-09, effect=0.557
Framework-Deployment vs Fixed-RandomForest: n=95, Wilcoxon p=5.88e-11, Holm p=6.46e-10, effect=0.848

multiple_comparison_correction: YES
