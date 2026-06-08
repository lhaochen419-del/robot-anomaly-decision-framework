# Code Index and Commands

- dataset: real IMAD-DS benchmark outputs
- source_type: external_real
- protocol: framework_strategy_validation_gate_v1
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T14:24:59+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1

- `scripts/run_engineering_benchmark.py`: patched to emit validation metrics (`val_*`).
- `scripts/validate_framework_strategy_v1.py`: framework strategy validation and packet generator.
- validation-run command: `python scripts/run_engineering_benchmark.py --manifest outputs/framework_strategy_validation_gate_v1/strategy_validation_manifest.csv --output-root outputs/framework_strategy_validation_gate_v1/validation_runs --run-mode full --no-figures --no-resume`.
- strategy command: `python scripts/validate_framework_strategy_v1.py --input-root outputs/framework_strategy_validation_gate_v1/validation_runs --output-root outputs/framework_strategy_validation_gate_v1 --no-figures`.
