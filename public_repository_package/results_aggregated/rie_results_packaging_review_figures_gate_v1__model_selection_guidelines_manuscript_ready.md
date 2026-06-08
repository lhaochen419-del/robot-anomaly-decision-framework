# Model Selection Guidelines Manuscript Ready

- generated_at: 2026-06-05T15:45:06+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_results_packaging_review_figures_gate_v1
- source_type: real
- synthetic: NO

| scenario | recommended_strategy | recommended_reason | risk |
| --- | --- | --- | --- |
| label-rich | Framework-Balanced or Fixed-LightGBM | Full-validation rows show the framework slightly improves macro-F1/MDR while Fixed-LightGBM remains a strong simple default. | Do not claim broad dominance; differences are small. |
| label-scarce | Fixed-LightGBM with validation-only threshold, or Framework-Balanced when multiple candidates are available | Label-efficiency advantage was not proven, but LightGBM remains the strongest stable supervised baseline. | Normal-only and 5% settings remain weak; avoid overclaiming. |
| safety-critical | Framework-Safety | Validation utility explicitly penalizes missed detections; test MDR is lower than fixed strong baselines. | FAR increases substantially, so alarm workflow must tolerate more warnings. |
| low false alarm | Framework-Low-False-Alarm | Lowest mean FAR among deployable strategies. | MDR increases; not suitable where missed detections dominate. |
| domain shift | Fixed-LightGBM or cost-specific framework strategy | Fixed-LightGBM remains strong in source-to-target style summaries; framework choice depends on FAR/MDR cost. | Framework does not uniformly beat LightGBM under domain shift. |
| missing sensor | Framework-Robust or Framework-Balanced | Robustness summaries show framework strategies preserve high macro-F1/PR-AUC under robustness protocols. | Needs deployment-time validation for exact missing-sensor pattern. |
| low latency CPU | Fixed-LightGBM or Fixed-XGBoost | Tree baselines have very low latency and strong accuracy in completed benchmark outputs. | Feature extraction cost must be included in real deployment. |

- Deep reconstruction models are not recommended as default choices in this benchmark unless labels are unavailable and tree features cannot be computed.
- Tree models are recommended as strong engineering baselines, not as a new proposed algorithm.
