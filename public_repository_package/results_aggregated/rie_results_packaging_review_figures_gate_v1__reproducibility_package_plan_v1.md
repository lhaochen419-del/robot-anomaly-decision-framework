# Reproducibility Package Plan v1

- generated_at: 2026-06-05T15:45:06+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_results_packaging_review_figures_gate_v1
- source_type: real
- synthetic: NO

- Data placement: keep original datasets under data/raw and processed windows under data/processed.
- Configs: preserve RIE benchmark and framework strategy configs/manifests.
- Commands: include benchmark runner, completion script, strategy validation, and packaging commands.
- Seeds: benchmark used 7, 13, 23, 31, 42 where available.
- Dependencies: Python environment with pandas, numpy, scipy/sklearn, matplotlib, LightGBM/XGBoost where available, PyTorch for deep baselines.
- Output directories: outputs/rie_full_engineering_benchmark_v2 and outputs/framework_strategy_validation_gate_v1.
- Data body is not included in the review packet.
