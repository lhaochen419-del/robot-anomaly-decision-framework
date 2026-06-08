# Strategy Definitions

- dataset: real IMAD-DS benchmark outputs
- source_type: external_real
- protocol: framework_strategy_validation_gate_v1
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T14:24:59+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1

- Framework-Best-F1: select model and threshold strategy maximizing `val_macro_f1`.
- Framework-Balanced: maximize `val_macro_f1 - val_far - val_mdr`.
- Framework-Safety: maximize `val_macro_f1 - 2*val_mdr - 0.5*val_far`.
- Framework-Low-False-Alarm: maximize `val_macro_f1 - 2*val_far - 0.5*val_mdr`.
- Framework-Deployment: maximize balanced validation utility minus latency and size penalties.
- latency_penalty: `0.05 * log1p(latency_ms) / log1p(max_latency_ms)` using pre-run latency metadata.
- size_penalty: `0.05 * log1p(model_size_mb) / log1p(max_model_size_mb)`.
- Framework-Robust: validation balanced utility minus validation degradation from matching clean protocol.
- Framework-Label-Efficient: validation balanced utility minus a predeclared label-budget penalty.
