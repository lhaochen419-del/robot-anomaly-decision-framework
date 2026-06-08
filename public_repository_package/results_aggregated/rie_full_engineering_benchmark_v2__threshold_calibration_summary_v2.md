# Threshold Calibration Summary v2

- dataset: real IMAD-DS/RoAD data only
- source_type: external_real / real_reference
- protocol: rie_full_engineering_benchmark_completion_v2
- seed: see CSV rows
- method: see CSV rows
- n_train/n_val/n_test: see CSV rows
- n_test_normal/n_test_anomaly: see CSV rows
- generated_at: 2026-06-05T11:44:38+00:00
- output_path: /home/zyf/文档/跨工况工业机器人异常检测/robot_cirfl/outputs/rie_full_engineering_benchmark_v2

| dataset | protocol | method | strategy | macro_f1 | far | mdr | pr_auc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | best_f1 | 0.5699959744141958 | 0.692 | 0.08800000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | cost_md10_fp1 | 0.5399111588822569 | 0.764 | 0.012 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | cost_md2_fp1 | 0.5683788388871431 | 0.72 | 0.04 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | cost_md5_fp1 | 0.5399111588822569 | 0.764 | 0.012 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | target_far_0.05 | 0.3339445722195048 | 0.016 | 0.9960000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | target_far_0.10 | 0.3328587983016892 | 0.04 | 0.992 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | target_far_0.15 | 0.33218806843182486 | 0.06000000000000001 | 0.9879999999999999 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | target_recall_0.80 | 0.5806549697319138 | 0.632 | 0.156 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | target_recall_0.90 | 0.5699959744141958 | 0.692 | 0.08800000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | target_recall_0.95 | 0.5473132701678516 | 0.74 | 0.048 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | IsolationForest | youden_j | 0.5699959744141958 | 0.692 | 0.08800000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | best_f1 | 0.9136708610486396 | 0.096 | 0.076 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | cost_md10_fp1 | 0.8714052708682933 | 0.2 | 0.052000000000000005 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | cost_md2_fp1 | 0.90771199751334 | 0.124 | 0.06 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | cost_md5_fp1 | 0.90771199751334 | 0.124 | 0.06 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | target_far_0.05 | 0.9198680555662255 | 0.044000000000000004 | 0.11600000000000002 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | target_far_0.10 | 0.9178536810795839 | 0.076 | 0.08800000000000001 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | target_far_0.15 | 0.9158603004985993 | 0.092 | 0.076 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | target_recall_0.80 | 0.9276773236510472 | 0.012 | 0.132 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | target_recall_0.90 | 0.9157146148640554 | 0.064 | 0.10400000000000001 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | target_recall_0.95 | 0.8915058783506989 | 0.156 | 0.06 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | LightGBM | youden_j | 0.9257934072067572 | 0.05600000000000001 | 0.092 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | best_f1 | 0.9639518419973369 | 0.044 | 0.028000000000000004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | cost_md10_fp1 | 0.9496365390764542 | 0.096 | 0.004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | cost_md2_fp1 | 0.9639518419973369 | 0.044 | 0.028000000000000004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | cost_md5_fp1 | 0.9659315812614636 | 0.05600000000000001 | 0.012 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | target_far_0.05 | 0.9619331837835103 | 0.02 | 0.05600000000000001 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | target_far_0.10 | 0.9679919967987194 | 0.032 | 0.032 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | target_far_0.15 | 0.9679438234052846 | 0.052000000000000005 | 0.012 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | target_recall_0.80 | 0.9355772494579602 | 0.004 | 0.124 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | target_recall_0.90 | 0.9477885986857256 | 0.02 | 0.084 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | target_recall_0.95 | 0.9519358293466075 | 0.044 | 0.052000000000000005 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | RandomForest | youden_j | 0.9639518419973369 | 0.044 | 0.028000000000000004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | best_f1 | 0.919754408296766 | 0.068 | 0.092 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | cost_md10_fp1 | 0.7862863907413965 | 0.372 | 0.032 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | cost_md2_fp1 | 0.9237656281693303 | 0.068 | 0.084 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | cost_md5_fp1 | 0.9076870283083334 | 0.10800000000000001 | 0.076 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | target_far_0.05 | 0.9337774669027903 | 0.028000000000000004 | 0.10400000000000001 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | target_far_0.10 | 0.9137682505709526 | 0.088 | 0.08399999999999999 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | target_far_0.15 | 0.9157311409902782 | 0.092 | 0.076 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | target_recall_0.80 | 0.9254696450697573 | 0.004 | 0.144 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | target_recall_0.90 | 0.9197399504036922 | 0.039999999999999994 | 0.12 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | target_recall_0.95 | 0.886517731031292 | 0.172 | 0.048 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_main_binary_if_valid | XGBoost | youden_j | 0.9217931185176964 | 0.052000000000000005 | 0.10400000000000001 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | best_f1 | 0.5699959744141958 | 0.692 | 0.08800000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | cost_md10_fp1 | 0.5399111588822569 | 0.764 | 0.012 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | cost_md2_fp1 | 0.5683788388871431 | 0.72 | 0.04 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | cost_md5_fp1 | 0.5399111588822569 | 0.764 | 0.012 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | target_far_0.05 | 0.3339445722195048 | 0.016 | 0.9960000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | target_far_0.10 | 0.3328587983016892 | 0.04 | 0.992 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | target_far_0.15 | 0.33218806843182486 | 0.06000000000000001 | 0.9879999999999999 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | target_recall_0.80 | 0.5806549697319138 | 0.632 | 0.156 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | target_recall_0.90 | 0.5699959744141958 | 0.692 | 0.08800000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | target_recall_0.95 | 0.5473132701678516 | 0.74 | 0.048 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | IsolationForest | youden_j | 0.5699959744141958 | 0.692 | 0.08800000000000001 | 0.48423018383673544 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | best_f1 | 0.9136708610486396 | 0.096 | 0.076 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | cost_md10_fp1 | 0.8714052708682933 | 0.2 | 0.052000000000000005 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | cost_md2_fp1 | 0.90771199751334 | 0.124 | 0.06 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | cost_md5_fp1 | 0.90771199751334 | 0.124 | 0.06 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | target_far_0.05 | 0.9198680555662255 | 0.044000000000000004 | 0.11600000000000002 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | target_far_0.10 | 0.9178536810795839 | 0.076 | 0.08800000000000001 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | target_far_0.15 | 0.9158603004985993 | 0.092 | 0.076 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | target_recall_0.80 | 0.9276773236510472 | 0.012 | 0.132 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | target_recall_0.90 | 0.9157146148640554 | 0.064 | 0.10400000000000001 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | target_recall_0.95 | 0.8915058783506989 | 0.156 | 0.06 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | LightGBM | youden_j | 0.9257934072067572 | 0.05600000000000001 | 0.092 | 0.976298802813776 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | best_f1 | 0.9639518419973369 | 0.044 | 0.028000000000000004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | cost_md10_fp1 | 0.9496365390764542 | 0.096 | 0.004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | cost_md2_fp1 | 0.9639518419973369 | 0.044 | 0.028000000000000004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | cost_md5_fp1 | 0.9659315812614636 | 0.05600000000000001 | 0.012 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | target_far_0.05 | 0.9619331837835103 | 0.02 | 0.05600000000000001 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | target_far_0.10 | 0.9679919967987194 | 0.032 | 0.032 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | target_far_0.15 | 0.9679438234052846 | 0.052000000000000005 | 0.012 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | target_recall_0.80 | 0.9355772494579602 | 0.004 | 0.124 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | target_recall_0.90 | 0.9477885986857256 | 0.02 | 0.084 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | target_recall_0.95 | 0.9519358293466075 | 0.044 | 0.052000000000000005 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | RandomForest | youden_j | 0.9639518419973369 | 0.044 | 0.028000000000000004 | 0.9963599223337273 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | best_f1 | 0.919754408296766 | 0.068 | 0.092 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | cost_md10_fp1 | 0.7862863907413965 | 0.372 | 0.032 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | cost_md2_fp1 | 0.9237656281693303 | 0.068 | 0.084 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | cost_md5_fp1 | 0.9076870283083334 | 0.10800000000000001 | 0.076 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | target_far_0.05 | 0.9337774669027903 | 0.028000000000000004 | 0.10400000000000001 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | target_far_0.10 | 0.9137682505709526 | 0.088 | 0.08399999999999999 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | target_far_0.15 | 0.9157311409902782 | 0.092 | 0.076 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | target_recall_0.80 | 0.9254696450697573 | 0.004 | 0.144 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | target_recall_0.90 | 0.9197399504036922 | 0.039999999999999994 | 0.12 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | target_recall_0.95 | 0.886517731031292 | 0.172 | 0.048 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_missing_sensor_if_meaningful | XGBoost | youden_j | 0.9217931185176964 | 0.052000000000000005 | 0.10400000000000001 | 0.9814681170848768 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | best_f1 | 0.33890804597701146 | 0.9948717948717949 | 0.0 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | cost_md10_fp1 | 0.3333333333333333 | 1.0 | 0.0 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | cost_md2_fp1 | 0.33890804597701146 | 0.9948717948717949 | 0.0 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | cost_md5_fp1 | 0.33890804597701146 | 0.9948717948717949 | 0.0 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | target_far_0.05 | 0.35913190379279175 | 0.7974358974358975 | 0.44102564102564096 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | target_far_0.10 | 0.38675108330507835 | 0.8076923076923077 | 0.3538461538461538 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | target_far_0.15 | 0.40928540487541937 | 0.8538461538461538 | 0.1974358974358974 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | target_recall_0.80 | 0.3411206896551724 | 0.9923076923076923 | 0.0025641025641025598 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | target_recall_0.90 | 0.33890804597701146 | 0.9948717948717949 | 0.0 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | target_recall_0.95 | 0.33890804597701146 | 0.9948717948717949 | 0.0 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | IsolationForest | youden_j | 0.33890804597701146 | 0.9948717948717949 | 0.0 | 0.3738898520663052 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | best_f1 | 0.9525046169091947 | 0.07435897435897432 | 0.020512820512820478 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | cost_md10_fp1 | 0.7376536291122524 | 0.4256410256410256 | 0.007692307692307681 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | cost_md2_fp1 | 0.858111125386527 | 0.2282051282051282 | 0.010256410256410239 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | cost_md5_fp1 | 0.8397389750910023 | 0.26666666666666666 | 0.007692307692307681 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | target_far_0.05 | 0.9525046169091947 | 0.07435897435897432 | 0.020512820512820478 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | target_far_0.10 | 0.9219653801617651 | 0.14615384615384613 | 0.007692307692307681 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | target_far_0.15 | 0.9150153839512972 | 0.15897435897435894 | 0.007692307692307681 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | target_recall_0.80 | 0.9499654506775308 | 0.06410256410256406 | 0.03589743589743586 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | target_recall_0.90 | 0.9333875342468222 | 0.12051282051282049 | 0.010256410256410239 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | target_recall_0.95 | 0.858111125386527 | 0.2282051282051282 | 0.010256410256410239 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | LightGBM | youden_j | 0.9525046169091947 | 0.07435897435897432 | 0.020512820512820478 | 0.9873143596730255 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | best_f1 | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | cost_md10_fp1 | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | cost_md2_fp1 | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | cost_md5_fp1 | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | target_far_0.05 | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | target_far_0.10 | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | target_far_0.15 | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | target_recall_0.80 | 0.8806529362393931 | 0.0 | 0.23076923076923075 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | target_recall_0.90 | 0.8806529362393931 | 0.0 | 0.23076923076923075 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | target_recall_0.95 | 0.8806529362393931 | 0.0 | 0.23076923076923075 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | RandomForest | youden_j | 0.8845024112212798 | 0.0 | 0.2230769230769231 | 0.9986868552769177 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | best_f1 | 0.9768983916848178 | 0.04102564102564102 | 0.0051282051282051195 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | cost_md10_fp1 | 0.8322899877583257 | 0.2846153846153846 | 0.0025641025641025598 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | cost_md2_fp1 | 0.9433766839574869 | 0.10512820512820512 | 0.0051282051282051195 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | cost_md5_fp1 | 0.9433766839574869 | 0.10512820512820512 | 0.0051282051282051195 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | target_far_0.05 | 0.9820379928244188 | 0.0282051282051282 | 0.007692307692307679 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | target_far_0.10 | 0.9433766839574869 | 0.10512820512820512 | 0.0051282051282051195 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | target_far_0.15 | 0.9433766839574869 | 0.10512820512820512 | 0.0051282051282051195 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | target_recall_0.80 | 0.9871723156797856 | 0.01282051282051282 | 0.0128205128205128 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | target_recall_0.90 | 0.9871723156797856 | 0.01282051282051282 | 0.0128205128205128 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | target_recall_0.95 | 0.9768983916848178 | 0.04102564102564102 | 0.0051282051282051195 | 0.999903440441415 |
| IMAD-DS BrushlessMotor | brushless_source_to_target_if_valid | XGBoost | youden_j | 0.9768983916848178 | 0.04102564102564102 | 0.0051282051282051195 | 0.999903440441415 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | best_f1 | 0.3735608110470935 | 0.9557692307692308 | 0.026282051282051244 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | cost_md10_fp1 | 0.34728869029613413 | 0.9852564102564102 | 0.008333333333333321 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | cost_md2_fp1 | 0.36765877984338285 | 0.9628205128205127 | 0.020512820512820478 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | cost_md5_fp1 | 0.34728869029613413 | 0.9852564102564102 | 0.008333333333333321 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | target_far_0.05 | 0.3933326598537631 | 0.05064102564102561 | 0.9282051282051281 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | target_far_0.10 | 0.4198762340030185 | 0.09615384615384612 | 0.8826923076923077 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | target_far_0.15 | 0.44677412729779553 | 0.1423076923076923 | 0.8314102564102563 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | target_recall_0.80 | 0.5057913074654424 | 0.7301282051282051 | 0.18397435897435893 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | target_recall_0.90 | 0.4454397803016322 | 0.8532051282051283 | 0.0961538461538461 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | target_recall_0.95 | 0.4090060541907211 | 0.9121794871794873 | 0.0493589743589743 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | AutoEncoder | youden_j | 0.5573040293007777 | 0.4685897435897436 | 0.4147435897435897 | 0.5526872937800904 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | best_f1 | 0.3357340118494097 | 0.9974358974358974 | 0.0019230769230769197 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | cost_md10_fp1 | 0.3357340118494097 | 0.9974358974358974 | 0.0019230769230769197 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | cost_md2_fp1 | 0.3357340118494097 | 0.9974358974358974 | 0.0019230769230769197 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | cost_md5_fp1 | 0.3357340118494097 | 0.9974358974358974 | 0.0019230769230769197 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | target_far_0.05 | 0.41360657637246734 | 0.0461538461538461 | 0.9083333333333334 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | target_far_0.10 | 0.44434316366507104 | 0.10961538461538456 | 0.8493589743589745 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | target_far_0.15 | 0.47092672856828416 | 0.1705128205128205 | 0.7865384615384615 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | target_recall_0.80 | 0.42713081449917656 | 0.8243589743589744 | 0.2147435897435897 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | target_recall_0.90 | 0.3617836770851144 | 0.9487179487179487 | 0.09487179487179483 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | target_recall_0.95 | 0.3429252706992455 | 0.9801282051282051 | 0.0493589743589743 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | CIRFL_v3_reference | youden_j | 0.4652514712144704 | 0.1987179487179487 | 0.7711538461538462 | 0.5292439796798023 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | best_f1 | 0.3366042734364906 | 0.9955128205128204 | 0.00705128205128204 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | cost_md10_fp1 | 0.33375629592551137 | 0.9993589743589745 | 0.0012820512820512799 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | cost_md2_fp1 | 0.33603870834876887 | 0.9961538461538462 | 0.0064102564102564 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | cost_md5_fp1 | 0.33375629592551137 | 0.9993589743589745 | 0.0012820512820512799 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | target_far_0.05 | 0.3764652845278517 | 0.0589743589743589 | 0.9435897435897436 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | target_far_0.10 | 0.40997630060901147 | 0.10897435897435892 | 0.8903846153846153 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | target_far_0.15 | 0.4394303109857017 | 0.1551282051282051 | 0.8346153846153846 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | target_recall_0.80 | 0.46156182927824807 | 0.7878205128205128 | 0.1923076923076923 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | target_recall_0.90 | 0.40263156619489904 | 0.9051282051282051 | 0.09102564102564098 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | target_recall_0.95 | 0.3568856446805472 | 0.9679487179487178 | 0.042307692307692255 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | IsolationForest | youden_j | 0.49646322381766106 | 0.48910256410256403 | 0.4980769230769231 | 0.5055828153367805 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | best_f1 | 0.3412132903974284 | 0.9923076923076923 | 0.0025641025641025598 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | cost_md10_fp1 | 0.3412132903974284 | 0.9923076923076923 | 0.0025641025641025598 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | cost_md2_fp1 | 0.3412132903974284 | 0.9923076923076923 | 0.0025641025641025598 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | cost_md5_fp1 | 0.3412132903974284 | 0.9923076923076923 | 0.0025641025641025598 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | target_far_0.05 | 0.42706064557597456 | 0.05769230769230762 | 0.8897435897435898 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | target_far_0.10 | 0.4789213819715096 | 0.10961538461538459 | 0.8057692307692307 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | target_far_0.15 | 0.4948503948454433 | 0.15448717948717944 | 0.7621794871794871 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | target_recall_0.80 | 0.44688506136620604 | 0.8038461538461539 | 0.20192307692307687 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | target_recall_0.90 | 0.384475506163835 | 0.9211538461538462 | 0.10448717948717942 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | target_recall_0.95 | 0.35689397055340005 | 0.9692307692307691 | 0.03782051282051278 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LSTM-AE | youden_j | 0.5009914953886943 | 0.21538461538461534 | 0.7153846153846154 | 0.5541035289640487 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LightGBM | best_f1 | 0.7081855007773219 | 0.3923076923076923 | 0.1833333333333333 | 0.8279930722299224 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LightGBM | cost_md10_fp1 | 0.346628759360928 | 0.9871794871794872 | 0.0025641025641025598 | 0.8279930722299224 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LightGBM | cost_md2_fp1 | 0.6883392930859366 | 0.4608974358974359 | 0.14230769230769225 | 0.8279930722299224 |
| IMAD-DS RoboticArm raw-window | imadds_raw_label_efficiency_20pct | LightGBM | cost_md5_fp1 | 0.41016178717054197 | 0.9198717948717949 | 0.01410256410256408 | 0.8279930722299224 |

- threshold_source: validation_only for all RUN_OK rows.
- score direction source: validation_only.
- test labels were used only for final metric computation.
- normal-only calibration does not train supervised tree models.
