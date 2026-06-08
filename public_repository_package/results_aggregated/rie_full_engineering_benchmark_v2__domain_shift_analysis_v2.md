# Domain Shift Analysis v2

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
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | 5 | 0.9768983916848178 | 0.999903440441415 | 0.04102564102564102 | 0.0051282051282051195 | 0.045937304990366065 | 0.008626836978902019 |  |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | 5 | 0.9525046169091947 | 0.9873143596730255 | 0.07435897435897433 | 0.020512820512820478 | 0.07767115063033991 | 0.0087730270117903 |  |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | 5 | 0.8845024112212798 | 0.9986868552769177 | 0.0 | 0.2230769230769231 | 0.1426925552077591 | 0.2513381165977663 |  |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | 5 | 0.3389080459770114 | 0.37388985206630526 | 0.9948717948717949 | 0.0 | 0.16508690759073943 | 0.033533852547407095 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | LightGBM | 5 | 0.5227004202184072 | 0.6670001521887223 | 0.09230769230769227 | 0.7538461538461538 | 0.20321117178536946 | 0.00533513651671218 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | XGBoost | 5 | 0.46200421349011933 | 0.6351578681068425 | 0.06923076923076918 | 0.8384615384615385 | 0.17387632001191372 | 0.004283470573285779 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | RandomForest | 5 | 0.4315781179915703 | 0.5942973844502004 | 0.08589743589743586 | 0.8666666666666668 | 0.29200386060401795 | 0.17982814669943387 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | raw_residual_energy | 5 | 0.34810581126229045 | 0.434196451351781 | 0.9833333333333334 | 0.01410256410256408 | 0.0 | 0.14869781212511063 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | IsolationForest | 5 | 0.34802814042075697 | 0.5435677160276742 | 0.9820512820512821 | 0.01794871794871794 | 0.2835367125924676 | 0.02871365137159438 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | LSTM-AE | 5 | 0.34778170010256615 | 0.43364358327501246 | 0.9833333333333334 | 0.015384615384615358 | 0.19604335119947786 |  | 0.00535005887146462 |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | USAD | 5 | 0.3362912491294376 | 0.5942738613264235 | 0.9961538461538462 | 0.0051282051282051195 | 0.1633293153950944 |  | 0.004268389938470779 |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | CIRFL_v3_reference | 5 | 0.3361666143837205 | 0.4209673387222505 | 0.9974358974358974 | 0.0 | 0.0 | 1.4535011897514312 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | AutoEncoder | 5 | 0.3333333333333333 | 0.5938247025390861 | 1.0 | 0.0 | 0.10884448396973308 |  | 0.0029151718197867405 |
| IMAD-DS RoboticArm raw-window | imadds_raw_leave_target_weight35_out | condition_decoupled_residual_energy | 5 | 0.3333333333333333 | 0.4269461194658728 | 1.0 | 0.0 | 0.0 | 1.1091113403941004 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | LightGBM | 5 | 0.6665187866847749 | 0.8135943877416262 | 0.3875 | 0.275 | 0.1976944498252123 | 0.00191538901758728 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | XGBoost | 5 | 0.6305786764250987 | 0.804148027521423 | 0.45 | 0.2823275862068965 | 0.1666862067999318 | 0.00147872741418438 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | RandomForest | 5 | 0.6187959378719128 | 0.7918906004811138 | 0.5064655172413792 | 0.2353448275862069 | 0.27948598442599176 | 0.06246864893533897 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | IsolationForest | 5 | 0.349016525322641 | 0.4880007421146213 | 0.9758620689655173 | 0.027586206896551717 | 0.2864348764065653 | 0.01074568489425938 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | LSTM-AE | 5 | 0.3456281216875218 | 0.579947186036504 | 0.988362068965517 | 0.0021551724137931 | 0.19893462858162816 |  | 0.00414500945821752 |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | raw_residual_energy | 5 | 0.3456281216875218 | 0.5801565823255757 | 0.988362068965517 | 0.0021551724137931 | 0.0 | 0.0510207487015326 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | condition_decoupled_residual_energy | 5 | 0.34406286918948303 | 0.5636290146208334 | 0.9896551724137932 | 0.00258620689655172 | 0.0 | 0.3999639083962518 |  |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | AutoEncoder | 5 | 0.3342876256934366 | 0.5082353784987456 | 0.9991379310344828 | 0.0 | 0.10317766261287029 |  | 0.0024029797400700595 |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | USAD | 5 | 0.3333333333333333 | 0.529600658718064 | 1.0 | 0.0 | 0.15344337581191206 |  | 0.00311594508118634 |
| IMAD-DS RoboticArm raw-window | imadds_raw_source_to_target | CIRFL_v3_reference | 5 | 0.33275682534285445 | 0.5131517189029322 | 1.0 | 0.00258620689655172 | 0.0 | 0.5053170575812491 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_leave_target_weight35_out | LightGBM | 5 | 0.41052867111826463 | 0.5053846459450131 | 0.17435897435897435 | 0.8358974358974359 | 0.08640655197668816 | 0.0172517023598536 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_leave_target_weight35_out | IsolationForest | 5 | 0.3333333333333333 | 0.831794261106863 | 1.0 | 0.0 | 0.1655285609653219 | 0.06903273543008621 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_leave_target_weight35_out | RandomForest | 5 | 0.3212021361492311 | 0.4569719796273037 | 0.07179487179487176 | 0.9948717948717949 | 0.14364822739735242 | 0.47264938983015525 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_leave_target_weight35_out | XGBoost | 5 | 0.2945283526514116 | 0.4492402499746776 | 0.19487179487179487 | 0.9897435897435898 | 0.046229852992109896 | 0.01617706177803944 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_source_to_target | XGBoost | 5 | 0.6419806152673301 | 0.84379803707269 | 0.4534482758620689 | 0.2362068965517241 | 0.04855767653789364 | 0.006096897514312119 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_source_to_target | LightGBM | 5 | 0.6281469979023917 | 0.8676360165474912 | 0.4793103448275862 | 0.2327586206896551 | 0.08474390418268737 | 0.00607213276390238 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_source_to_target | RandomForest | 5 | 0.5632851608564895 | 0.8160098298741714 | 0.5379310344827586 | 0.3224137931034482 | 0.14227028200402853 | 0.19519711043765958 |  |
| IMAD-DS RoboticArm segment-level | imadds_segment_source_to_target | IsolationForest | 5 | 0.3333333333333333 | 0.46530643200791905 | 1.0 | 0.0 | 0.16668740739114582 | 0.026568759469604398 |  |
