# Completion Denominator Audit

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

| group | total | completed | failed | skipped | completion_rate | historical_not_rerun |
| --- | --- | --- | --- | --- | --- | --- |
| all_jobs | 1305 | 745 | 380 | 180 | 0.5708812260536399 |  |
| core_required | 745 | 745 | 0 | 0 | 1.0 |  |
| optional_secondary | 150 | 0 | 0 | 150 | 0.0 |  |
| historical_reference | 275 | 0 | 245 | 30 | 0.0 | 275.0 |

- Core denominator excludes historical failed routes that are intentionally not rerun.
- Core denominator excludes supervised tree models under normal-only calibration because they are not scientifically applicable.
- Core denominator excludes RoAD optional secondary protocols because RoAD is not the main RIE evidence source.
