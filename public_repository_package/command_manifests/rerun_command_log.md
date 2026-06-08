# Rerun Command Log

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

- rerun_required: NO.
- reason: after v2 triage, no true core-required missing jobs remain.
- v1 command retained: `python scripts/run_engineering_benchmark.py --manifest outputs/rie_full_engineering_design_gate_v1/08_full_experiment_command_manifest.csv --output-root outputs/rie_full_engineering_benchmark_v1 --run-mode full --no-figures --no-resume`.
