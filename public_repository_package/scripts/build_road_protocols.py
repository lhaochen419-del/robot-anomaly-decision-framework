from __future__ import annotations

import argparse
import sys
from pathlib import Path
from copy import deepcopy

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_real_data import prepare_one
from src.datasets import build_datasets_from_config
from src.utils.markdown import dataframe_to_markdown

OUT = ROOT / "outputs/road_protocols"

PROTOCOLS = [
    {
        "protocol": "road_binary_main",
        "family": "binary_main",
        "train_runs": [0, 1, 2, 3, 4, 9, 12, 13],
        "val_runs": [5, 10],
        "test_runs": [6, 7, 11, 14],
        "notes": "Run-level split. Test contains independent normal runs and a velocity anomaly run; velocity is represented in train by run 13.",
    },
    {
        "protocol": "scenario_holdout_collision",
        "family": "scenario_holdout",
        "holdout_scenario": "collision",
        "train_runs": [0, 1, 2, 3, 4, 12, 13],
        "val_runs": [5, 14],
        "test_runs": [6, 10],
        "notes": "Collision anomaly held out from training. Test normal comes from independent normal run 6 and normal segments inside run 10.",
    },
    {
        "protocol": "scenario_holdout_weight",
        "family": "scenario_holdout",
        "holdout_scenario": "weight",
        "train_runs": [0, 1, 2, 3, 4, 9, 10, 13],
        "val_runs": [5, 14],
        "test_runs": [6, 11, 12],
        "notes": "Weight anomaly held out from training. Test normal comes from independent normal runs 6 and 11.",
    },
    {
        "protocol": "scenario_holdout_velocity",
        "family": "scenario_holdout",
        "holdout_scenario": "velocity",
        "train_runs": [0, 1, 2, 3, 4, 9, 12],
        "val_runs": [5, 10],
        "test_runs": [6, 11, 14],
        "notes": "Velocity anomaly held out from training. Test normal comes from independent normal runs 6 and 11.",
    },
    {
        "protocol": "condition_holdout_collision_condition",
        "family": "condition_holdout",
        "train_runs": [0, 1, 2, 3, 4, 5, 12, 13],
        "val_runs": [6, 14],
        "test_runs": [9, 10],
        "train_conditions": [0, 3, 4],
        "val_conditions": [0, 4],
        "test_conditions": [1],
        "notes": "Condition 1 collision holdout is metric-valid because test has normal and anomaly, but RoAD condition-label confounding remains high.",
    },
]


def base_config() -> dict:
    return {
        "project": {"name": "robot_cirfl_real_gate_v2_road", "stage": "real_gate_v2", "output_dir": "outputs/real_gate_v2", "real_data_available": True},
        "seeds": [7, 13, 23],
        "device": "auto",
        "data": {
            "source": "road",
            "source_type": "real",
            "dataset_name": "RoAD",
            "path": "data/processed/road/road_unified.csv",
            "window_size": 128,
            "stride": 64,
            "feature_cols": [],
            "label_mode": "binary",
            "label_col": "fault_label",
            "condition_col": "condition_id",
            "run_col": "run_id",
            "trajectory_col": "trajectory_id",
            "fault_episode_col": "fault_episode_id",
            "time_col": "t",
            "normalization": {"method": "median_iqr", "clip_quantiles": [0.5, 99.5]},
            "metadata": {"dataset_name": "RoAD", "robot_platform": "collaborative robotic arm in a production line", "sampling_frequency": "not specified by adapter", "source_url": "https://gitlab.com/AlessioMascolini/roaddataset", "split_unit": "run_id / recording"},
        },
        "model": {
            "name": "CIRFL_v2",
            "hidden_dim": 64,
            "condition_dim": 16,
            "residual_dim": 48,
            "n_relation_atoms": 8,
            "dropout": 0.05,
            "num_classes": 2,
            "max_conditions": 16,
            "min_scale": 0.05,
            "log_scale_clip": 5.0,
            "energy_clip": 100000.0,
            "atom_temperature": 0.7,
            "loss_weights": {
                "residual_nll": 1.0,
                "prototype_ce": 0.55,
                "condition_invariance": 0.20,
                "field_consistency": 0.03,
                "source_sparsity": 0.005,
                "source_smoothness": 0.01,
                "atom_diversity": 0.10,
                "atom_usage_balance": 0.03,
                "atom_assignment_entropy": 0.02,
                "zh_zc_covariance": 0.08,
                "zh_zc_orthogonality": 0.03,
            },
        },
        "training": {"epochs": 5, "overfit_epochs": 8, "batch_size": 64, "learning_rate": 0.001, "weight_decay": 0.0001, "grad_clip": 5.0, "threshold_strategy": "target_recall_0.90", "class_balanced_loss": True, "grl_max": 1.0},
        "baselines": {"require_real_booster": True, "epochs": 3, "batch_size": 64, "learning_rate": 0.001, "methods": ["random_forest", "xgboost", "lightgbm", "isolation_forest", "autoencoder", "lstm_ae", "usad"]},
        "latency": {"warmup": 10, "repeats": 50},
        "review_packet": {"output_dir": "progress_for_chatgpt/latest"},
    }


def config_for_protocol(proto: dict) -> dict:
    cfg = base_config()
    cfg["project"]["name"] = f"robot_cirfl_real_gate_v2_{proto['protocol']}"
    cfg["data"]["split"] = {"train_runs": proto["train_runs"], "val_runs": proto["val_runs"], "test_runs": proto["test_runs"]}
    cfg["data"]["protocol_name"] = proto["protocol"]
    cfg["data"]["protocol_family"] = proto["family"]
    if "train_conditions" in proto:
        cfg["data"]["cross_condition"] = {"train_conditions": proto["train_conditions"], "val_conditions": proto["val_conditions"], "test_conditions": proto["test_conditions"]}
    else:
        cfg["data"]["cross_condition"] = {"train_conditions": [], "val_conditions": [], "test_conditions": []}
    return cfg


def count_split(cfg: dict, split_mode: str = "main") -> dict:
    train, val, test, _, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
    def counts(ds, prefix):
        labels = ds.labels()
        return {f"n_{prefix}_windows": len(ds), f"n_{prefix}_normal": int((labels == 0).sum()), f"n_{prefix}_anomaly": int((labels > 0).sum())}
    row = {}
    row.update(counts(train, "train"))
    row.update(counts(val, "val"))
    row.update(counts(test, "test"))
    row["can_compute_metrics"] = row["n_test_normal"] > 0 and row["n_test_anomaly"] > 0 and row["n_val_normal"] > 0 and row["n_val_anomaly"] > 0
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-configs", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not (ROOT / "data/processed/road/road_unified.csv").exists():
        prepare_one("road")
    rows = []
    configs = {}
    for proto in PROTOCOLS:
        cfg = config_for_protocol(proto)
        split_mode = "cross_condition" if proto["family"] == "condition_holdout" else "main"
        counts = count_split(cfg, split_mode=split_mode)
        row = {**proto, **counts}
        row["has_label_condition_confounding"] = proto["family"] in {"condition_holdout", "scenario_holdout"}
        row["protocol_valid"] = bool(counts["can_compute_metrics"])
        if proto["family"] == "condition_holdout" and not counts["can_compute_metrics"]:
            row["protocol_valid"] = False
        rows.append(row)
        configs[proto["protocol"]] = cfg
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "protocol_validity.csv", index=False)
    lines = ["# RoAD Gate v2 Protocol Validity Report", "", dataframe_to_markdown(df), "", "Protocols with no normal or no anomaly in validation/test are invalid and must not be used for gate metrics."]
    (OUT / "protocol_validity_report.md").write_text("\n".join(lines), encoding="utf-8")
    if args.write_configs:
        cfg_dir = ROOT / "configs"
        # Main single-protocol config.
        with (cfg_dir / "real_gate_v2_road_binary.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(configs["road_binary_main"], f, sort_keys=False)
        scenario_cfg = base_config()
        scenario_cfg["project"]["name"] = "robot_cirfl_real_gate_v2_road_scenario_holdout"
        scenario_cfg["project"]["output_dir"] = "outputs/real_gate_v2"
        scenario_cfg["protocols"] = [p for p in PROTOCOLS if p["family"] == "scenario_holdout"]
        with (cfg_dir / "real_gate_v2_road_scenario_holdout.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(scenario_cfg, f, sort_keys=False)
        cond_cfg = configs["condition_holdout_collision_condition"]
        with (cfg_dir / "real_gate_v2_road_condition_holdout.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(cond_cfg, f, sort_keys=False)
        combined = base_config()
        combined["project"]["name"] = "robot_cirfl_real_gate_v2_road_all_protocols"
        combined["project"]["output_dir"] = "outputs/real_gate_v2"
        combined["protocols"] = PROTOCOLS
        with (cfg_dir / "real_gate_v2_road_all.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(combined, f, sort_keys=False)
    print(f"Wrote protocol validity to {OUT}")


if __name__ == "__main__":
    main()
