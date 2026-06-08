# Statistical Claims Allowed

- generated_at: 2026-06-05T15:45:06+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_results_packaging_review_figures_gate_v1
- source_type: real
- synthetic: NO

Allowed cautious claims:
- Framework-Safety vs Fixed-LightGBM on utility_safety: framework mean is higher by 0.0706; Wilcoxon p=4.5491155011565136e-08; significant=True; wins/losses=65/15.
- Framework-Balanced vs Fixed-LightGBM on utility_balanced: framework mean is higher by 0.0020; Wilcoxon p=0.1371466519263771; significant=False; wins/losses=17/9.
- Framework-Low-False-Alarm vs Fixed-LightGBM on utility_low_false_alarm: framework mean is higher by 0.0946; Wilcoxon p=7.009413895072905e-10; significant=True; wins/losses=72/11.
- Framework-Robust vs Fixed-LightGBM on utility_robust: framework mean is lower by -0.0031; Wilcoxon p=0.4163858883429933; significant=False; wins/losses=18/12.
- Framework-Deployment vs Fixed-LightGBM on utility_deployment: framework mean is lower by -0.0062; Wilcoxon p=0.5850270925103997; significant=False; wins/losses=15/11.

Disallowed claims:
- Do not claim the framework comprehensively outperforms Fixed-LightGBM.
- Do not claim a new model or algorithm contribution.
- Do not claim label-efficiency superiority; Gate v1 did not support that criterion.
- Do not describe Oracle-Best-Test as deployable; it is only a theoretical upper bound.
