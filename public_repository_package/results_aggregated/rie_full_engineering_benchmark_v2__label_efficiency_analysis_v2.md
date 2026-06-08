# Label Efficiency Analysis v2

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

| dataset | protocol | method | n | macro_f1_mean | pr_auc_mean | far_mean | mdr_mean | train_time_sec_mean | cpu_latency_ms_mean | gpu_latency_ms_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LightGBM | 5 | 0.7081855007773219 | 0.8279930722299221 | 0.3923076923076923 | 0.1833333333333333 | 0.20103345902170985 | 0.0027588653700569 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | XGBoost | 5 | 0.6719744147320893 | 0.8257169155997328 | 0.4653846153846154 | 0.1589743589743589 | 0.16900614020414642 | 0.00258307629384286 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | RandomForest | 5 | 0.6058283174381165 | 0.7448433484718049 | 0.6089743589743589 | 0.12628205128205122 | 0.27999212800059464 | 0.09357260196105548 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | USAD | 5 | 0.3741392624782565 | 0.565069359734236 | 0.9551282051282051 | 0.02499999999999996 | 0.1597408069996163 |  | 0.00334747657269376 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | 5 | 0.3735608110470935 | 0.5526872937800904 | 0.9557692307692308 | 0.026282051282051237 | 0.10907029279042031 |  | 0.00234720512078354 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | 5 | 0.34121329039742837 | 0.5541035289640487 | 0.9923076923076923 | 0.0025641025641025598 | 0.19172524677123873 |  | 0.004329249331464899 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | raw_residual_energy | 5 | 0.34121329039742837 | 0.554149000067178 | 0.9923076923076923 | 0.0025641025641025598 | 0.0 | 0.07697209265853204 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | 5 | 0.3366042734364905 | 0.5055828153367805 | 0.9955128205128204 | 0.00705128205128204 | 0.2947365584317595 | 0.01603728850545262 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | condition_decoupled_residual_energy | 5 | 0.33658749151637907 | 0.5574020938008457 | 0.9967948717948719 | 0.0012820512820512799 | 0.0 | 0.5759287246828302 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | 5 | 0.3357340118494097 | 0.5292439796798023 | 0.9974358974358974 | 0.0019230769230769197 | 0.0 | 0.762356054791524 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | XGBoost | 5 | 0.45540516112810375 | 0.6817527836218389 | 0.8153846153846154 | 0.07179487179487176 | 0.1688044904032722 | 0.0023558458241705405 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | LightGBM | 5 | 0.38778328121869565 | 0.7009952316837986 | 0.9326923076923077 | 0.0397435897435897 | 0.20254852760117498 | 0.0027671394771179598 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | USAD | 5 | 0.3741392624782565 | 0.565069359734236 | 0.9551282051282051 | 0.02499999999999996 | 0.15856324876658615 |  | 0.0033227936140834205 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | AutoEncoder | 5 | 0.3735608110470935 | 0.5526872937800904 | 0.9557692307692308 | 0.026282051282051237 | 0.11096801359672095 |  | 0.00242111441016622 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | LSTM-AE | 5 | 0.34121329039742837 | 0.5541035289640487 | 0.9923076923076923 | 0.0025641025641025598 | 0.1943897184450179 |  | 0.00448688978138256 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | raw_residual_energy | 5 | 0.34121329039742837 | 0.554149000067178 | 0.9923076923076923 | 0.0025641025641025598 | 0.0 | 0.07579265869795698 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | IsolationForest | 5 | 0.3366042734364905 | 0.5055828153367805 | 0.9955128205128204 | 0.00705128205128204 | 0.2862970922142267 | 0.01566089194966474 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | condition_decoupled_residual_energy | 5 | 0.33658749151637907 | 0.5574020938008457 | 0.9967948717948719 | 0.0012820512820512799 | 0.0 | 0.5741138204869527 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | CIRFL_v3_reference | 5 | 0.3357340118494097 | 0.5292439796798023 | 0.9974358974358974 | 0.0019230769230769197 | 0.0 | 0.7917625391154358 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_5pct | RandomForest | 5 | 0.3333333333333333 | 0.6486907578055225 | 1.0 | 0.0 | 0.2706142354058102 | 0.12488809141079678 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | LightGBM | 5 | 0.8170078967516711 | 0.9197293476861945 | 0.26217948717948714 | 0.09935897435897431 | 0.20680465579498555 | 0.00280207817783 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | XGBoost | 5 | 0.7892305703034491 | 0.9011113737957561 | 0.29166666666666663 | 0.12564102564102558 | 0.17823870277497914 | 0.00246452531054946 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | RandomForest | 5 | 0.7335658109903861 | 0.8621472238788328 | 0.3897435897435897 | 0.12628205128205122 | 0.2864173147827387 | 0.08943146251392761 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | USAD | 5 | 0.3741392624782565 | 0.565069359734236 | 0.9551282051282051 | 0.02499999999999996 | 0.1563537547830492 |  | 0.0032552079527447195 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | AutoEncoder | 5 | 0.3735608110470935 | 0.5526872937800904 | 0.9557692307692308 | 0.026282051282051237 | 0.1096445621922612 |  | 0.0022994858460524403 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | LSTM-AE | 5 | 0.34121329039742837 | 0.5541035289640487 | 0.9923076923076923 | 0.0025641025641025598 | 0.19477859062608327 |  | 0.004296814418361981 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | raw_residual_energy | 5 | 0.34121329039742837 | 0.554149000067178 | 0.9923076923076923 | 0.0025641025641025598 | 0.0 | 0.07627874841161354 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | IsolationForest | 5 | 0.3366042734364905 | 0.5055828153367805 | 0.9955128205128204 | 0.00705128205128204 | 0.30022393059916797 | 0.016437578447855577 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | condition_decoupled_residual_energy | 5 | 0.33658749151637907 | 0.5574020938008457 | 0.9967948717948719 | 0.0012820512820512799 | 0.0 | 0.577198078248423 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_full | CIRFL_v3_reference | 5 | 0.3357340118494097 | 0.5292439796798023 | 0.9974358974358974 | 0.0019230769230769197 | 0.0 | 0.8129410259500265 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_normal_only | condition_decoupled_residual_energy | 5 | 0.44522090213615834 | 0.5574020938008457 | 0.0557692307692307 | 0.8698717948717949 | 0.0 | 0.5779806000273078 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_normal_only | LSTM-AE | 5 | 0.42984413708182395 | 0.5541035289640487 | 0.059615384615384556 | 0.885897435897436 | 0.19793035639449952 |  | 0.00450957013526928 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_normal_only | raw_residual_energy | 5 | 0.42869949228143867 | 0.554149000067178 | 0.059615384615384556 | 0.8871794871794872 | 0.0 | 0.07725189863823542 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_normal_only | CIRFL_v3_reference | 5 | 0.4183851999425989 | 0.5292439796798023 | 0.05192307692307687 | 0.9012820512820513 | 0.0 | 0.7912782401082298 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_normal_only | USAD | 5 | 0.4004104726451221 | 0.565069359734236 | 0.04807692307692303 | 0.9217948717948717 | 0.15713401560205964 |  | 0.00346099516788778 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_normal_only | AutoEncoder | 5 | 0.39489514947635096 | 0.5526872937800904 | 0.051923076923076905 | 0.9262820512820513 | 0.10994960663374506 |  | 0.0023809945560059795 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_normal_only | IsolationForest | 5 | 0.3787504100407605 | 0.5055828153367805 | 0.062179487179487104 | 0.9403846153846154 | 0.2917859392240644 | 0.01527929614381622 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_label_efficiency | LightGBM | 5 | 0.9448424270074047 | 0.9903832720783482 | 0.051282051282051225 | 0.058974358974358945 | 0.09105050356592977 | 0.00923565911272392 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_label_efficiency | RandomForest | 5 | 0.9421944649439817 | 0.986933056853767 | 0.051282051282051246 | 0.06410256410256406 | 0.16232447137590494 | 0.23011073980552071 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_label_efficiency | XGBoost | 5 | 0.9100952477372999 | 0.9775018584985595 | 0.10256410256410253 | 0.07692307692307687 | 0.045197421172633714 | 0.00808803478064822 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_label_efficiency | IsolationForest | 5 | 0.46249338945079677 | 0.5138002973245387 | 0.8487179487179487 | 0.0538461538461538 | 0.1638105717953294 | 0.04251873201260768 |  |

- Normal-only rows exclude supervised trees by design; they remain evaluated at 5%, 20%, and full validation budgets.
