generated_at: 2026-06-07T08:46:43+00:00
output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_latex_template_draft_v16/minor_ready_response_map_v16.md

# Minor-ready Response Map v16

| Reviewer concern | Manuscript change made | Location in manuscript | Remaining issue | Need user input? |
|---|---|---|---|---|
| Remove Results in Engineering self-positioning. | Removed journal-fit claims and replaced with objective engineering-protocol wording. | Discussion, Limitations, Conclusion | None in manuscript body. | No |
| Declarations placeholders. | Removed placeholder declarations from main manuscript; created author action file. | Main manuscript; author_action_required_declarations_v16.md | Real declarations still missing. | Yes |
| Supplementary draft/planned language. | Removed planned/draft language from main manuscript. | Statistical/Results text | None. | No |
| Window count difference. | Added split accounting explanation and table. | Implementation details; Table split accounting | Full provenance remains in source artifacts. | No |
| Low-latency CPU deployment overclaim. | Replaced with low-latency candidate selection and prepared-input latency wording. | Introduction, Results, scenario table | End-to-end timing requires future author/engineering validation. | Yes |
| Figure 1 too dense. | Rebuilt simplified eight-block workflow. | Figure 1 | None. | No |
| Figure 2 readability. | Rebuilt compact desirability heatmap with abbreviated labels and separate oracle row. | Figure 2 | None. | No |
| Table 6 too wide. | Compressed to comparison, objective, mean diff., Holm p, cluster CI and result. | Statistical evidence table | Full details retained in supplementary analysis files. | No |
| Bootstrap cluster count/B/CI method missing. | Added 19 dataset--protocol clusters, B=5000, percentile 95% CI, replacement resampling. | Statistical testing protocol | None. | No |
| FAR/MDR sensitivity result. | Added qualitative summary across alpha/beta tolerance grid. | FAR/MDR results | None. | No |
| FAR@95%Recall interpretation. | Added engineering interpretation of high false-alarm burden at high recall. | Evaluation/results text | None. | No |
| Deep baselines limited training limitation. | Retained model-parameter table and baseline-configuration boundary. | Implementation details; limitations | Further tuning would be future work, not added here. | No |
| Label-budget duplicate rows. | Kept merged Framework-Balanced / Label-Budget explanation. | Label-budget table and text | None. | No |
| Figure 6 dual y-axis. | Rebuilt as direct-label scatter plot, no dual axis. | Figure 6 | None. | No |
| Robustness/domain-shift sharper conclusion. | Stated framework helps local corruption thresholding more than domain shift. | Robustness/domain-shift results | None. | No |
| RQ answers location. | RQ answers remain in Engineering Discussion before Limitations. | Discussion | None. | No |
| Conclusion needs core numbers. | Core Fixed-LightGBM, Safety and Low-FA numbers retained. | Conclusion | None. | No |
| Terminology consistency. | Standardized label-budget, Oracle-Best-Test-Utility, prepared-input latency and domain-shift wording. | Throughout | None. | No |
