# Model Selection Guidelines V3

- generated_at: 2026-06-06T02:24:29+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_manuscript_completion_v3
- source_type: real
- synthetic: NO

- Label-rich: Framework-Balanced or Fixed-LightGBM.
- Label-scarce: Fixed-LightGBM with validation-only thresholding remains safest; no label-efficiency superiority claim.
- Safety-critical: Framework-Safety when lower MDR is worth higher FAR.
- Low false alarm: Framework-Low-False-Alarm when lower FAR is worth higher MDR.
- Domain shift: Fixed-LightGBM or validation-selected framework strategy, with stress-protocol validation.
- Missing sensor: Framework-Robust or Framework-Balanced, validated for the expected missing-channel pattern.
- Low-latency CPU: Fixed-LightGBM or Fixed-XGBoost.
