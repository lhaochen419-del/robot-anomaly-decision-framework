# Major Revision Response Map v14

generated_at: 2026-06-06T07:51:37+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v14/major_revision_response_map_v14.md

| Reviewer concern | Manuscript change made | Location in manuscript | Remaining issue | Need user input? |
|---|---|---|---|---|
| Abstract lacks key numbers | Added Fixed-LightGBM, Framework-Safety and Framework-Low-False-Alarm values | Abstract | None | No |
| RQ1 wording inconsistent with regret results | Rewrote RQ1 and RQ answers to emphasize bounded utility/regret comparison | Introduction; Answers to RQs | None | No |
| Implementation details missing | Expanded feature, split, stress, model, threshold and latency details | Implementation details and reproducibility | Hardware and stride/sampling details still need final confirmation | Yes for final metadata |
| Statistical paired unit unclear | Defined dataset-protocol-seed-scenario paired unit | Evaluation metrics/statistics | None | No |
| n=95 may have hierarchical dependence | Added hierarchy/dependence caveat; windows not independent | Statistical testing protocol | Clustered bootstrap planned in supplement | No model rerun |
| Planned supplementary wording unacceptable | Replaced planned wording and created supplementary draft workspace | Results; supplementary_draft | Final supplement not packaged | Yes before submission |
| Oracle name confusing | Renamed/clarified Oracle-Best-Test-Utility | Framework strategy; captions | None | No |
| FAR/MDR violation threshold undefined | Added alpha=0.40, beta=0.35, either-condition joint rule | FAR/MDR subsection | None | No |
| Normal-only calibration unclear | Clarified no anomaly validation labels and no macro-F1 tuning | Implementation; Label-budget results | None | No |
| Robustness/domain shift numeric table needed | Added concise core table | Robustness/domain-shift results | None | No |
| Latency protocol unclear | Clarified prepared-feature/window prediction latency, not end-to-end | Implementation; Latency subsection; caption | Hardware details need final confirmation | Yes for final metadata |
| Figures too dense | Replaced Figure 2 and Figure 8 with simplified core plots | Results figures | Final visual polishing optional | No |
| Declarations placeholders unacceptable for final submission | Converted to author-action placeholders | Declarations section | User facts missing | Yes |
| Related Work needs gap statements and stronger references | Added gap statements; retained verified v13 reference expansion | Related Work | Optional human reference style check | Optional |
| Terminology inconsistent | Replaced label-efficient manuscript term; clarified oracle and latency terms | Throughout manuscript | Internal CSV names may differ | No |
