# Failed / Skipped Triage

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

## Triage Counts
| triage_category | count |
| --- | --- |
| CORE_REQUIRED_COMPLETED | 745 |
| HISTORICAL_REFERENCE_NOT_RERUN | 275 |
| OPTIONAL_SECONDARY_INVALID | 150 |
| SKIPPED_WITH_VALID_REASON | 135 |

## Interpretation
- CETRA_full and COIL_full are historical failed algorithm routes and are not rerun as RIE core baselines.
- Supervised tree models under normal-only calibration are not applicable because no anomaly labels are available.
- RoAD remains secondary stress evidence; invalid secondary RoAD tabular splits do not block IMAD-DS core evidence.
- Failed jobs are retained in CSV and not deleted.
