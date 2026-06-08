# Engineering Findings v1

- generated_at: 2026-06-05T15:45:06+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_results_packaging_review_figures_gate_v1
- source_type: real
- synthetic: NO

- Strong fixed models: Fixed-LightGBM remains the strongest simple default; Fixed-XGBoost and Fixed-RandomForest are also competitive.
- Fixed-LightGBM mean macro-F1=0.785, PR-AUC=0.877, FAR=0.169, MDR=0.240.
- Framework-Balanced mean macro-F1=0.773, FAR=0.174, MDR=0.256.
- Framework value is scenario adaptation, not highest single-model accuracy.
- Safety strategy reduces MDR to 0.154 but raises FAR to 0.322.
- Low-false-alarm strategy reduces FAR to 0.087 but raises MDR to 0.365.
- FAR/MDR are more engineering-relevant than accuracy because false alarms and missed detections create different operational costs.
- Label-efficiency advantage is not established and must be reported as a limitation.
