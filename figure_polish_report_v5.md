generated_at: 2026-06-06 11:43:23 Asia/Macau
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v5/figure_polish_report_v5.md

# Figure Polish Report v5

generated_at: 2026-06-06 11:20:28 
output_path: outputs/rie_latex_template_draft_v5/figure_polish_report_v5.md

## Backend And Contract
- backend: Python / matplotlib
- figure_archetype: quantitative grid / engineering comparison panels
- core_conclusion: The framework is a validation/calibration strategy layer for engineering constraints; it does not replace strong tree baselines.
- evidence_chain: utility, rank, regret, FAR/MDR, label budget, robustness/domain shift, latency and strategy-selection map.
- review_risk: avoid overstating framework superiority; keep Fixed-LightGBM and Oracle-Best-Test visible.

## Generated Figures
1. fig1_engineering_utility_comparison.png
2. fig2_average_rank.png
3. fig3_macro_f1_regret.png
4. fig4_far_mdr_violation.png
5. fig5_label_efficiency.png
6. fig6_robustness_domain_shift.png
7. fig7_latency_deployment_tradeoff.png
8. fig8_strategy_selection_map.png

## Changes Relative To Review Figures
- Removed `Review-only` wording from figure titles and captions.
- Reduced label crowding in latency/deployment by using a log latency axis and direct labels only for plotted strategies.
- Preserved Fixed-LightGBM, XGBoost and RandomForest in comparison figures.
- Kept Oracle-Best-Test as a theoretical non-deployable upper bound where relevant.
- Did not alter any experimental values.
