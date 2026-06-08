# Latency / Deployment Analysis v2

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

| method | model_size_mb_mean | train_time_sec_mean | inference_time_sec_mean | cpu_latency_ms_mean | gpu_latency_ms_mean |
| --- | --- | --- | --- | --- | --- |
| AutoEncoder | 0.453216552734375 | 0.1122815660142805 | 0.0015012619376647684 |  | 0.0024297771125493667 |
| CIRFL_v3_reference | 0.666 | 0.0 | 0.4827891042048577 | 0.8137480283328045 |  |
| IsolationForest | 0.05 | 0.2407737656717654 | 0.0081250448012724 | 0.02731401235952834 |  |
| LSTM-AE | 0.0460014343261718 | 0.19866716742205118 | 0.0027165876585058383 |  | 0.004408144684117136 |
| LightGBM | 1.0 | 0.15173780169141918 | 0.0016193895343396547 | 0.0064284241915944776 |  |
| RandomForest | 1.0 | 0.22528271529774524 | 0.051267850736009014 | 0.1886969224378224 |  |
| USAD | 0.850128173828125 | 0.15841229480186783 | 0.0022008096682838633 |  | 0.0035782861078430517 |
| XGBoost | 1.0 | 0.12140404724908106 | 0.0013968102707478073 | 0.00573452996633056 |  |
| condition_decoupled_residual_energy | 0.001 | 0.0 | 0.3597117746190634 | 0.6064362988086617 |  |
| raw_residual_energy | 0.001 | 0.0 | 0.04760124098393131 | 0.08040842702790392 |  |
- Tree models are CPU-first for reproducibility; deep baselines use GPU where CUDA is available.
- Low-latency deployment should be chosen from the latency table after applying the required FAR/MDR operating point.
