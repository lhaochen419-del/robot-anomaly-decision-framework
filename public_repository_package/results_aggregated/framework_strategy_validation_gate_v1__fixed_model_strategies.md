# Fixed Model Strategies

- dataset: real IMAD-DS benchmark outputs
- source_type: external_real
- protocol: framework_strategy_validation_gate_v1
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T14:24:59+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/framework_strategy_validation_gate_v1

- Fixed-LightGBM, Fixed-XGBoost, Fixed-RandomForest, Fixed-IsolationForest, Fixed-AutoEncoder, Fixed-LSTM-AE, Fixed-USAD.
- Fixed strategies keep the model fixed and choose threshold strategy using validation macro-F1.
- Oracle-Best-Test is included only as ORACLE_NOT_DEPLOYABLE theoretical upper bound.
- Supervised tree models are not available in normal-only label budget rows.
