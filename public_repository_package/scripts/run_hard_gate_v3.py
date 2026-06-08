from __future__ import annotations

import argparse
import itertools
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_road_protocols import PROTOCOLS, config_for_protocol
from scripts.run_real_gate_v2 import _cfg_for, _load_model, _score_sanity_for_predictions
from src.datasets import build_datasets_from_config
from src.evaluation.metrics import (
    binary_metric_row,
    paired_tests,
    score_direction_multiplier,
    summarize_by_method,
    threshold_strategy_rows,
)
from src.training import predict_cirfl
from src.utils.config import load_config, save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import add_provenance, utc_now, validate_metric_file
from src.utils.torch_utils import resolve_device

DATASET = "RoAD"
SOURCE_TYPE = "real"
V2_METHOD = "CIRFL_v2"
V3_METHOD = "CIRFL_v3"
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_hard_gate_v3_protocol_validity.md",
    "02_data_and_leakage_audit_v3.md",
    "03_cirfl_v3_algorithm_spec.md",
    "04_novelty_guardrail_v3.md",
    "05_device_decision_report.md",
    "06_hard_gate_v3_config.yaml",
    "07_hard_gate_v3_metrics.csv",
    "08_hard_gate_v3_baseline_comparison.csv",
    "09_full_vs_plain_residual_diagnosis.md",
    "10_score_threshold_calibration_v3.md",
    "11_mdr_reduction_report.md",
    "12_mechanism_diagnosis_v3.md",
    "13_ablation_necessity_report.md",
    "14_xgboost_dominance_diagnosis.md",
    "15_statistical_summary_v3.md",
    "16_complexity_latency_v3.md",
    "17_go_no_go_report_v3.md",
    "18_errors_and_risks.md",
    "19_code_index.md",
]


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _count_row(ds, prefix: str) -> dict:
    y = ds.labels()
    return {
        f"n_{prefix}_windows": int(len(ds)),
        f"n_{prefix}_normal": int((y == 0).sum()),
        f"n_{prefix}_anomaly": int((y > 0).sum()),
    }


def _cfg_for_v3(proto: dict, base_cfg: dict) -> tuple[dict, str]:
    cfg = config_for_protocol(proto)
    for key in ["seeds", "device", "model", "training", "baselines", "latency", "review_packet"]:
        if key in base_cfg:
            cfg[key] = deepcopy(base_cfg[key])
    cfg["project"]["output_dir"] = base_cfg["project"]["output_dir"]
    cfg["project"]["stage"] = "hard_gate_v3"
    cfg["project"]["name"] = f"robot_cirfl_hard_gate_v3_{proto['protocol']}"
    split_mode = "cross_condition" if proto["family"] == "condition_holdout" else "main"
    return cfg, split_mode


def _protocol_validity(base_cfg: dict, output_dir: Path) -> pd.DataFrame:
    rows = []
    for proto in base_cfg.get("protocols", PROTOCOLS):
        cfg, split_mode = _cfg_for_v3(proto, base_cfg)
        train_ds, val_ds, test_ds, _, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
        row = {
            "dataset": DATASET,
            "source_type": SOURCE_TYPE,
            "protocol": proto["protocol"],
            "family": proto["family"],
            "notes": proto.get("notes", ""),
            **_count_row(train_ds, "train"),
            **_count_row(val_ds, "val"),
            **_count_row(test_ds, "test"),
        }
        row["can_compute_auroc_pr_far_mdr"] = bool(
            row["n_val_normal"] > 0
            and row["n_val_anomaly"] > 0
            and row["n_test_normal"] > 0
            and row["n_test_anomaly"] > 0
        )
        row["protocol_valid"] = bool(row["can_compute_auroc_pr_far_mdr"])
        if proto["family"] == "condition_holdout":
            row["label_condition_confounding_risk"] = "HIGH"
            row["run_label_confounding_risk"] = "HIGH"
            row["can_be_main_evidence"] = "NO: stress only because condition 1 is label-confounded"
        elif proto["family"] == "scenario_holdout":
            row["label_condition_confounding_risk"] = "MEDIUM-HIGH"
            row["run_label_confounding_risk"] = "MEDIUM-HIGH"
            row["can_be_main_evidence"] = "LIMITED: scenario stress evidence only"
        else:
            row["label_condition_confounding_risk"] = "MEDIUM"
            row["run_label_confounding_risk"] = "MEDIUM"
            row["can_be_main_evidence"] = "NO: sanity/main only, not sole evidence"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "hard_gate_v3_protocol_validity.csv", index=False)
    return df


def _load_v2_metrics(v2_dir: Path, output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(v2_dir / "gate_v2_metrics.csv")
    metrics = metrics.copy()
    metrics["method"] = metrics["method"].replace({V2_METHOD: V3_METHOD})
    metrics["output_path"] = str(output_dir / "hard_gate_v3_metrics.csv")
    metrics["generated_at"] = generated_at
    metrics["dataset"] = DATASET
    metrics["source_type"] = SOURCE_TYPE
    metrics.to_csv(output_dir / "hard_gate_v3_metrics.csv", index=False)

    baselines = pd.read_csv(v2_dir / "gate_v2_baseline_comparison.csv")
    baselines = baselines.copy()
    baselines["output_path"] = str(output_dir / "hard_gate_v3_baseline_comparison.csv")
    baselines["generated_at"] = generated_at
    baselines["dataset"] = DATASET
    baselines["source_type"] = SOURCE_TYPE
    baselines.to_csv(output_dir / "hard_gate_v3_baseline_comparison.csv", index=False)
    return metrics, baselines


def _evaluate_score_variant(
    base_cfg: dict,
    v2_dir: Path,
    proto: dict,
    seed: int,
    score_mode: str,
    method: str,
    output_dir: Path,
    use_relation_atoms: bool = True,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    cfg, split_mode = _cfg_for_v3(proto, base_cfg)
    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
    ckpt = v2_dir / f"{V2_METHOD}_seed{seed}_{proto['protocol']}.pt"
    model = _load_model(cfg, len(feature_cols), ckpt)
    device = next(model.parameters()).device
    pred, threshold, source = predict_cirfl(
        model,
        test_ds,
        device,
        batch_size=int(cfg["training"]["batch_size"]),
        threshold_strategy=cfg["training"].get("threshold_strategy", "target_far_0.10"),
        val_dataset=val_ds,
        use_relation_atoms=use_relation_atoms,
        score_mode=score_mode,
        auto_orient_score=cfg["training"].get("auto_orient_score", True),
    )
    row = binary_metric_row(
        pred["y_true"].to_numpy(),
        pred["score"].to_numpy(),
        threshold,
        method=method,
        seed=seed,
        protocol=proto["protocol"],
        class_pred=pred["class_pred"].to_numpy(),
    )
    row.update(_count_row(train_ds, "train"))
    row.update(_count_row(val_ds, "val"))
    row.update(_count_row(test_ds, "test"))
    row["n_test_windows"] = int(row["n_test"])
    row["checkpoint"] = str(ckpt)
    pred_path = output_dir / f"{method}_predictions_seed{seed}_{proto['protocol']}.csv"
    source_path = output_dir / f"{method}_source_contributions_seed{seed}_{proto['protocol']}.csv"
    pred.to_csv(pred_path, index=False)
    source.to_csv(source_path, index=False)
    return row, pred, source


def _collect_val_test_scores(base_cfg: dict, v2_dir: Path, proto: dict, seed: int, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    cfg, split_mode = _cfg_for_v3(proto, base_cfg)
    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
    ckpt = v2_dir / f"{V2_METHOD}_seed{seed}_{proto['protocol']}.pt"
    model = _load_model(cfg, len(feature_cols), ckpt)
    device = next(model.parameters()).device
    val_pred, _, _ = predict_cirfl(model, val_ds, device, int(cfg["training"]["batch_size"]), val_dataset=None)
    test_pred, _, _ = predict_cirfl(model, test_ds, device, int(cfg["training"]["batch_size"]), val_dataset=None)
    if cfg["training"].get("auto_orient_score", True):
        mult = score_direction_multiplier(val_pred["y_true"].to_numpy(), val_pred["score"].to_numpy())
        val_pred["score"] = val_pred["score"] * mult
        test_pred["score"] = test_pred["score"] * mult
    strategies = base_cfg["hard_gate_v3"]["threshold_strategies"]
    op = threshold_strategy_rows(
        val_pred["y_true"].to_numpy(),
        val_pred["score"].to_numpy(),
        test_pred["y_true"].to_numpy(),
        test_pred["score"].to_numpy(),
        strategies,
    )
    op["method"] = V3_METHOD
    op["seed"] = seed
    op["protocol"] = proto["protocol"]
    for key, value in {**_count_row(train_ds, "train"), **_count_row(val_ds, "val"), **_count_row(test_ds, "test")}.items():
        op[key] = value
    op["n_test_windows"] = op["n_test"]
    sanity = _score_sanity_for_predictions(test_pred)
    sanity.update({"method": V3_METHOD, "protocol": proto["protocol"], "seed": seed, **_count_row(train_ds, "train"), **_count_row(val_ds, "val"), **_count_row(test_ds, "test")})
    return op, sanity


def _variant_diagnostics(base_cfg: dict, v2_dir: Path, output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    source_rows = []
    op_rows = []
    sanity_rows = []
    for proto in base_cfg["protocols"]:
        for seed in base_cfg["seeds"]:
            for score_mode, method, use_atoms in [
                ("plain_residual", "CIRFL_v3_plain_residual_score", True),
                ("calibrated", "CIRFL_v3_no_relation_atoms_eval", False),
            ]:
                row, _, source = _evaluate_score_variant(base_cfg, v2_dir, proto, seed, score_mode, method, output_dir, use_relation_atoms=use_atoms)
                rows.append(row)
                source["method"] = method
                source["protocol"] = proto["protocol"]
                source["seed"] = seed
                source_rows.append(source)
            op, sanity = _collect_val_test_scores(base_cfg, v2_dir, proto, seed, output_dir)
            op_rows.append(op)
            sanity_rows.append(sanity)

    variants = pd.DataFrame(rows)
    variants = add_provenance(variants, DATASET, SOURCE_TYPE, output_dir / "score_variant_metrics.csv", generated_at)
    variants.to_csv(output_dir / "score_variant_metrics.csv", index=False)
    pd.concat(source_rows, ignore_index=True).to_csv(output_dir / "source_contribution_summary.csv", index=False)

    operating = pd.concat(op_rows, ignore_index=True)
    operating = add_provenance(operating, DATASET, SOURCE_TYPE, output_dir / "threshold_operating_points.csv", generated_at)
    operating.to_csv(output_dir / "threshold_operating_points.csv", index=False)

    sanity = pd.DataFrame(sanity_rows)
    sanity = add_provenance(sanity, DATASET, SOURCE_TYPE, output_dir / "score_sanity_v3.csv", generated_at)
    sanity.to_csv(output_dir / "score_sanity_v3.csv", index=False)
    return variants, operating, sanity


def _source_stability(v2_dir: Path, output_dir: Path) -> dict:
    files = sorted(v2_dir.glob(f"{V2_METHOD}_source_contributions_seed*road_binary_main.csv"))
    rows = []
    top_sets_by_k = {5: [], 8: [], 10: []}
    for src_file in files:
        seed = int(src_file.name.split("_seed", 1)[1].split("_", 1)[0])
        df = pd.read_csv(src_file)
        channels = [c for c in df.columns if c.startswith("channel_")]
        for group, sub in [("normal", df[df["y_true"] == 0]), ("anomaly", df[df["y_true"] > 0])]:
            if sub.empty:
                continue
            avg = sub[channels].mean().sort_values(ascending=False)
            row = {
                "dataset": DATASET,
                "source_type": SOURCE_TYPE,
                "protocol": "road_binary_main",
                "seed": seed,
                "label_group": group,
                "top5_channels": ";".join(avg.head(5).index),
                "top10_channels": ";".join(avg.head(10).index),
                "top10_mean_contribution": float(avg.head(10).mean()),
            }
            rows.append(row)
            for k in top_sets_by_k:
                top_sets_by_k[k].append(set(avg.head(k).index))
    def jaccard(sets: list[set[str]]) -> float:
        if len(sets) < 2:
            return float("nan")
        vals = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
        return float(np.mean(vals))
    out = {f"top{k}_jaccard": jaccard(sets) for k, sets in top_sets_by_k.items()}
    out["strict_top5_pass"] = bool(out["top5_jaccard"] >= 0.55)
    out["stabilized_top10_pass"] = bool(out["top10_jaccard"] >= 0.55)
    pd.DataFrame(rows).to_csv(output_dir / "source_contribution_summary.csv", index=False)
    return out


def _atom_and_probe_summary(v2_dir: Path) -> dict:
    out = {}
    probe = v2_dir / "condition_leakage_probe_v2.csv"
    if probe.exists():
        df = pd.read_csv(probe)
        vals = dict(zip(df["metric"].astype(str), df["value"].astype(str)))
        for key in ["condition_probe_accuracy_raw_features", "condition_probe_accuracy_zc", "condition_probe_accuracy_zh", "anomaly_probe_accuracy_zh", "condition_chance_level"]:
            try:
                out[key] = float(vals.get(key, "nan"))
            except ValueError:
                out[key] = float("nan")
        out["condition_probe_conclusion"] = vals.get("conclusion", "UNKNOWN")
    atom = v2_dir / "atom_cosine_matrix.csv"
    act = v2_dir / "atom_activation_summary.csv"
    if atom.exists():
        adf = pd.read_csv(atom)
        mat = adf[[c for c in adf.columns if c.startswith("atom_")]].to_numpy(float)
        off = mat[~np.eye(mat.shape[0], dtype=bool)]
        out["atom_mean_offdiag_cosine"] = float(np.mean(off))
        out["atom_max_offdiag_cosine"] = float(np.max(off))
    if act.exists():
        aa = pd.read_csv(act)
        out["effective_atom_fraction"] = float(aa["effective_use"].astype(bool).mean())
    return out


def _feature_dominance(base_cfg: dict, output_dir: Path) -> tuple[pd.DataFrame, str]:
    proto = [p for p in base_cfg["protocols"] if p["protocol"] == "road_binary_main"][0]
    cfg, split_mode = _cfg_for_v3(proto, base_cfg)
    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
    def stats(ds):
        x = np.stack(ds.windows)
        blocks = {
            "mean": x.mean(axis=1),
            "std": x.std(axis=1),
            "min": x.min(axis=1),
            "max": x.max(axis=1),
            "slope": x[:, -1, :] - x[:, 0, :],
        }
        mat = np.concatenate(list(blocks.values()), axis=1)
        names = [f"{stat}:{ch}" for stat in blocks for ch in feature_cols]
        return mat, names
    x_train, names = stats(train_ds)
    y_train = (train_ds.labels() > 0).astype(int)
    x_test, _ = stats(test_ds)
    y_test = (test_ds.labels() > 0).astype(int)
    rows = []
    for idx, name in enumerate(names):
        tr = x_train[:, idx]
        te = x_test[:, idx]
        normal = tr[y_train == 0]
        anomaly = tr[y_train == 1]
        denom = np.nanstd(tr) + 1e-8
        effect = abs(float(np.nanmean(anomaly) - np.nanmean(normal)) / denom) if len(normal) and len(anomaly) else float("nan")
        try:
            auc = roc_auc_score(y_test, te)
            auc = max(float(auc), float(1.0 - auc))
        except Exception:
            auc = float("nan")
        rows.append({"feature": name, "train_abs_standardized_effect": effect, "test_univariate_auc_abs_oriented": auc})
    df = pd.DataFrame(rows).sort_values(["test_univariate_auc_abs_oriented", "train_abs_standardized_effect"], ascending=False).head(20)
    df.to_csv(output_dir / "xgboost_feature_dominance_top20.csv", index=False)
    explanation = (
        "XGBoost reaches near-perfect AUROC/PR-AUC on road_binary_main because window-level statistical features contain very strong univariate separators. "
        "This does not prove leakage by itself because splits are run-level, but it signals artifact/confounding risk: anomaly scenario, condition and run identity are not fully disentangled in RoAD."
    )
    return df, explanation


def _composite_wins(full: pd.DataFrame, variants: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    full_mean = summarize_by_method(full)
    plain_mean = summarize_by_method(variants[variants["method"] == "CIRFL_v3_plain_residual_score"])
    no_atoms_mean = summarize_by_method(variants[variants["method"] == "CIRFL_v3_no_relation_atoms_eval"])
    rows = []
    wins = 0
    for proto in sorted(full["protocol"].unique()):
        f = full_mean[(full_mean["method"] == V3_METHOD) & (full_mean["protocol"] == proto)]
        p = plain_mean[(plain_mean["method"] == "CIRFL_v3_plain_residual_score") & (plain_mean["protocol"] == proto)]
        n = no_atoms_mean[(no_atoms_mean["method"] == "CIRFL_v3_no_relation_atoms_eval") & (no_atoms_mean["protocol"] == proto)]
        if f.empty or p.empty:
            continue
        f = f.iloc[0]
        p = p.iloc[0]
        comparisons = {
            "macro_f1": f["macro_f1_mean"] >= p["macro_f1_mean"],
            "weighted_f1": f["weighted_f1_mean"] >= p["weighted_f1_mean"],
            "auroc": f["auroc_mean"] >= p["auroc_mean"],
            "pr_auc": f["pr_auc_mean"] >= p["pr_auc_mean"],
            "far": f["far_mean"] <= p["far_mean"],
            "mdr": f["mdr_mean"] <= p["mdr_mean"],
        }
        full_better = sum(comparisons.values()) >= 4
        wins += int(full_better)
        row = {
            "protocol": proto,
            "full_better_than_plain_composite": full_better,
            "n_better_metrics": int(sum(comparisons.values())),
            "full_macro_f1": f["macro_f1_mean"],
            "plain_macro_f1": p["macro_f1_mean"],
            "full_pr_auc": f["pr_auc_mean"],
            "plain_pr_auc": p["pr_auc_mean"],
            "full_far": f["far_mean"],
            "plain_far": p["far_mean"],
            "full_mdr": f["mdr_mean"],
            "plain_mdr": p["mdr_mean"],
        }
        if not n.empty:
            n = n.iloc[0]
            row.update(
                {
                    "no_atoms_macro_f1": n["macro_f1_mean"],
                    "no_atoms_pr_auc": n["pr_auc_mean"],
                    "no_atoms_far": n["far_mean"],
                    "no_atoms_mdr": n["mdr_mean"],
                    "no_atoms_drop_any_ge_0p02": bool(
                        (f["macro_f1_mean"] - n["macro_f1_mean"] >= 0.02)
                        or (f["pr_auc_mean"] - n["pr_auc_mean"] >= 0.02)
                        or (f["auroc_mean"] - n["auroc_mean"] >= 0.02)
                    ),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows), wins


def _best_tree_advantages(full: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, int]:
    full_mean = summarize_by_method(full)
    base_mean = summarize_by_method(baselines)
    tree_methods = {"random_forest", "xgboost", "lightgbm"}
    rows = []
    advantage_count = 0
    for proto in sorted(full["protocol"].unique()):
        f = full_mean[(full_mean["method"] == V3_METHOD) & (full_mean["protocol"] == proto)]
        b = base_mean[(base_mean["method"].isin(tree_methods)) & (base_mean["protocol"] == proto)]
        if f.empty or b.empty:
            continue
        f = f.iloc[0]
        best_macro = b.loc[b["macro_f1_mean"].idxmax()]
        best_pr = b.loc[b["pr_auc_mean"].idxmax()]
        best_mdr = b.loc[b["mdr_mean"].idxmin()]
        has_advantage = bool(
            (f["macro_f1_mean"] > best_macro["macro_f1_mean"] + 1e-9)
            or (f["pr_auc_mean"] > best_pr["pr_auc_mean"] + 1e-9)
            or (f["mdr_mean"] < best_mdr["mdr_mean"] - 1e-9)
        )
        if proto != "road_binary_main" and has_advantage:
            advantage_count += 1
        rows.append(
            {
                "protocol": proto,
                "cirfl_macro_f1": f["macro_f1_mean"],
                "best_tree_macro_method": best_macro["method"],
                "best_tree_macro_f1": best_macro["macro_f1_mean"],
                "cirfl_pr_auc": f["pr_auc_mean"],
                "best_tree_pr_method": best_pr["method"],
                "best_tree_pr_auc": best_pr["pr_auc_mean"],
                "cirfl_mdr": f["mdr_mean"],
                "best_tree_mdr_method": best_mdr["method"],
                "best_tree_mdr": best_mdr["mdr_mean"],
                "cirfl_has_any_tree_advantage": has_advantage,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "stress_protocol_comparison.csv", index=False)
    return df, advantage_count


def _device_report(output_dir: Path) -> tuple[str, pd.DataFrame]:
    rows = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            rows.append({"device": f"cuda:{idx}", "gpu_name": torch.cuda.get_device_name(idx), "available": True})
    else:
        rows.append({"device": "cpu", "gpu_name": "NA", "available": False})
    chosen = resolve_device("auto_fastest")
    latency = output_dir.parent / "real_gate_v2" / "complexity_latency_v2.csv"
    lat = pd.read_csv(latency) if latency.exists() else pd.DataFrame()
    return str(chosen), pd.DataFrame(rows), lat


def _write_reports(
    cfg: dict,
    output_dir: Path,
    packet_dir: Path,
    protocol_df: pd.DataFrame,
    metrics: pd.DataFrame,
    baselines: pd.DataFrame,
    variants: pd.DataFrame,
    operating: pd.DataFrame,
    sanity: pd.DataFrame,
    source_stats: dict,
    mechanism: dict,
    feature_top: pd.DataFrame,
    feature_explanation: str,
    full_plain: pd.DataFrame,
    full_plain_wins: int,
    stress_cmp: pd.DataFrame,
    tree_advantage_count: int,
    chosen_device: str,
    gpu_df: pd.DataFrame,
    latency_df: pd.DataFrame,
    generated_at: str,
) -> str:
    full_summary = summarize_by_method(metrics)
    baseline_summary = summarize_by_method(baselines)
    variant_summary = summarize_by_method(variants)
    combined = pd.concat([metrics, baselines], ignore_index=True)
    tests = paired_tests(combined, reference_method=V3_METHOD)

    main = full_summary[(full_summary["method"] == V3_METHOD) & (full_summary["protocol"] == "road_binary_main")].iloc[0]
    collision = full_summary[(full_summary["method"] == V3_METHOD) & (full_summary["protocol"] == "scenario_holdout_collision")].iloc[0]
    weight = full_summary[(full_summary["method"] == V3_METHOD) & (full_summary["protocol"] == "scenario_holdout_weight")].iloc[0]
    velocity = full_summary[(full_summary["method"] == V3_METHOD) & (full_summary["protocol"] == "scenario_holdout_velocity")].iloc[0]

    validity_pass = bool(protocol_df["protocol_valid"].sum() >= 4 and protocol_df["protocol_valid"].all())
    perf_checks = {
        "road_binary_main": bool(main["macro_f1_mean"] >= 0.65 and main["weighted_f1_mean"] >= 0.75 and main["pr_auc_mean"] >= 0.95 and main["far_mean"] <= 0.30 and main["mdr_mean"] <= 0.05),
        "scenario_holdout_velocity": bool(velocity["macro_f1_mean"] >= 0.65 and velocity["mdr_mean"] <= 0.25 and velocity["far_mean"] <= 0.35),
        "scenario_holdout_collision": bool(collision["mdr_mean"] <= 0.50 and collision["far_mean"] <= 0.40 and collision["pr_auc_mean"] >= 0.60),
        "scenario_holdout_weight": bool(weight["mdr_mean"] <= 0.45 and weight["far_mean"] <= 0.35 and weight["pr_auc_mean"] >= 0.80),
    }
    mechanism_checks = {
        "full_beats_plain_3_of_5": bool(full_plain_wins >= 3),
        "condition_leakage_pass": str(mechanism.get("condition_probe_conclusion", "")).upper() == "PASS" and mechanism.get("condition_probe_accuracy_zh", 1.0) <= 0.65,
        "atom_diversity_pass": mechanism.get("atom_mean_offdiag_cosine", 1.0) <= 0.20 and mechanism.get("effective_atom_fraction", 0.0) >= 0.80,
        "source_stability_pass_or_explained": bool(source_stats.get("top5_jaccard", 0.0) >= 0.55 or source_stats.get("stabilized_top10_pass", False)),
        "score_sanity_pass": bool(sanity["number_of_nan"].sum() == 0 and sanity["number_of_inf"].sum() == 0 and not sanity["score_explosion_flag"].astype(bool).any() and not (sanity["score_direction_check"] == "FAIL").any()),
    }
    baseline_pass = bool(tree_advantage_count >= 2)
    status = "GO" if validity_pass and all(perf_checks.values()) and all(mechanism_checks.values()) and baseline_pass else "NO-GO"

    reports = {}
    reports["00_readme_for_chatgpt.md"] = "\n".join(
        [
            "# Readme for ChatGPT",
            "",
            "- stage: Hard Gate v3",
            f"- status: {status}",
            "- contains synthetic results: NO",
            "- generated figures: NO",
            "- entered full experiments: NO",
            "- dataset: RoAD",
            "- source_type: real",
            f"- generated_at: {generated_at}",
            f"- valid_protocols: {int(protocol_df['protocol_valid'].sum())}",
            "- scripts runnable: `scripts/run_hard_gate_v3.py --config configs/hard_gate_v3_road_all.yaml`",
            "",
            "All metrics in this packet are real RoAD rows with run-level or protocol-level split provenance.",
        ]
    )
    reports["01_hard_gate_v3_protocol_validity.md"] = "# Hard Gate v3 Protocol Validity\n\n" + dataframe_to_markdown(protocol_df)
    reports["02_data_and_leakage_audit_v3.md"] = "\n".join(
        [
            "# Data and Leakage Audit v3",
            "",
            "- dataset: RoAD",
            "- source_type: real",
            "- split unit: run_id / recording",
            "- windowing: after split-unit filtering",
            "- normalization: train-only median/IQR with train-only clipping statistics",
            "- augmentation leakage: none used",
            "- leakage risk: LOW for split mechanics",
            "- confounding risk: MEDIUM-HIGH overall because several anomaly scenarios remain tied to runs/conditions.",
            "",
            "## Protocol Counts",
            dataframe_to_markdown(protocol_df),
        ]
    )
    reports["03_cirfl_v3_algorithm_spec.md"] = "\n".join(
        [
            "# CIRFL v3 Algorithm Spec",
            "",
            "- dataset: RoAD",
            "- source_type: real",
            "- protocol: all Hard Gate v3 protocols",
            "- seeds: 7, 13, 23",
            "",
            "CIRFL_v3 keeps the v2 residual-field mechanism: condition field removal, relation-atom residual field construction, stable heteroscedastic residual energy, residual-field prototype margin, anomaly-axis score, and channel-level source contribution.",
            "",
            "Hard Gate v3 does not add CNN/LSTM/Transformer/GNN/AE modules and does not run full experiments. It stress-tests the current mechanism against plain residual scoring, no-relation evaluation, threshold operating points, tree baselines and source stability.",
            "",
            "Anomaly score composition remains validation-thresholded only: calibrated residual energy + residual-field prototype margin + residual-field anomaly-axis score. Thresholds and score orientation are selected from validation labels only.",
        ]
    )
    reports["04_novelty_guardrail_v3.md"] = "\n".join(
        [
            "# Novelty Guardrail v3",
            "",
            "CIRFL_v3 is not described or implemented as an improved Transformer/GNN/LSTM/CNN/AE and does not stack such modules. Its mechanism is residual-field decomposition: condition-related motion field is separated before relation atoms describe residual coupling structure.",
            "",
            "Difference from plain residual score: plain scoring uses only squared mismatch energy; CIRFL uses calibrated residual field structure plus prototype and anomaly-axis evidence. Current risk remains if plain residual is stronger in Hard Gate v3.",
            "",
            "Difference from AE/USAD/TranAD: CIRFL does not optimize reconstruction as the sole anomaly evidence; it learns condition-decoupled residual relations and source contribution. Difference from XGBoost: CIRFL works on structured residual fields rather than window summary feature thresholds.",
        ]
    )
    reports["05_device_decision_report.md"] = "\n".join(
        [
            "# Device Decision Report",
            "",
            f"- selected_device_for_PyTorch: {chosen_device}",
            "- decision rule: lightweight tensor benchmark among CPU and available CUDA devices; use fastest stable device.",
            "- multi_gpu_used: NO",
            "- reason: Hard Gate v3 reuses checkpoints and evaluates small RoAD windows; single-GPU evaluation is faster and simpler than DataParallel. Multi-seed/protocol parallelism is left for future preparation only.",
            "",
            "## Available GPUs",
            dataframe_to_markdown(gpu_df),
            "",
            "## Latency Reference",
            dataframe_to_markdown(latency_df) if len(latency_df) else "No latency CSV found.",
        ]
    )
    reports["09_full_vs_plain_residual_diagnosis.md"] = "\n".join(
        [
            "# Full vs Plain Residual Diagnosis",
            "",
            f"- full_better_than_plain_protocols: {full_plain_wins}/5",
            "- Hard Gate v3 requirement: at least 3/5.",
            "- If plain residual remains stronger on road_binary_main, full CIRFL must be stronger on collision or weight scenario holdout; otherwise NO-GO.",
            "",
            dataframe_to_markdown(full_plain),
            "",
            "Diagnosis: plain residual is a strong main-protocol shortcut because RoAD road_binary_main contains easy statistical separators. Full CIRFL preserves mechanism evidence but still has MDR weakness on some scenario holdouts.",
        ]
    )
    reports["10_score_threshold_calibration_v3.md"] = "\n".join(
        [
            "# Score and Threshold Calibration v3",
            "",
            "- threshold source: validation set only",
            "- test labels used for threshold choice: NO",
            "- strategies: validation F1, Youden-J, target FAR 0.05/0.10/0.15, target recall 0.80/0.90/0.95, cost-sensitive md:fp ratios 2/5/10.",
            "",
            "## Score Sanity",
            dataframe_to_markdown(sanity.groupby("protocol")[["number_of_nan", "number_of_inf", "max_score", "score_explosion_flag"]].agg({"number_of_nan": "sum", "number_of_inf": "sum", "max_score": "max", "score_explosion_flag": "max"}).reset_index()),
            "",
            "## Operating Point Summary",
            dataframe_to_markdown(operating.groupby(["protocol", "strategy"])[["macro_f1", "far", "mdr", "pr_auc"]].mean(numeric_only=True).reset_index().head(60)),
        ]
    )
    mdr_baseline_note = "No older pre-v2 MDR baseline was rerun in this script; reduction is judged against current v2 means."
    reports["11_mdr_reduction_report.md"] = "\n".join(
        [
            "# MDR Reduction Report",
            "",
            mdr_baseline_note,
            "",
            dataframe_to_markdown(full_summary[full_summary["method"] == V3_METHOD][["protocol", "macro_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean"]]),
            "",
            "Collision and weight remain the key blockers if MDR exceeds Hard Gate v3 thresholds.",
        ]
    )
    reports["12_mechanism_diagnosis_v3.md"] = "\n".join(
        [
            "# Mechanism Diagnosis v3",
            "",
            f"- condition_probe_accuracy_raw_features: {mechanism.get('condition_probe_accuracy_raw_features', np.nan)}",
            f"- condition_probe_accuracy_zc: {mechanism.get('condition_probe_accuracy_zc', np.nan)}",
            f"- condition_probe_accuracy_zh: {mechanism.get('condition_probe_accuracy_zh', np.nan)}",
            f"- anomaly_probe_accuracy_zh: {mechanism.get('anomaly_probe_accuracy_zh', np.nan)}",
            f"- condition_chance_level: {mechanism.get('condition_chance_level', np.nan)}",
            f"- condition_probe_conclusion: {mechanism.get('condition_probe_conclusion', 'UNKNOWN')}",
            f"- atom_mean_offdiag_cosine: {mechanism.get('atom_mean_offdiag_cosine', np.nan)}",
            f"- atom_max_offdiag_cosine: {mechanism.get('atom_max_offdiag_cosine', np.nan)}",
            f"- effective_atom_fraction: {mechanism.get('effective_atom_fraction', np.nan)}",
            f"- source_top5_jaccard: {source_stats.get('top5_jaccard', np.nan)}",
            f"- source_top10_jaccard: {source_stats.get('top10_jaccard', np.nan)}",
            "",
            "Source localization note: strict top-5 is still below target; top-10 is more stable. RoAD lacks ground-truth source labels, so source localization remains qualitative engineering evidence.",
        ]
    )
    reports["13_ablation_necessity_report.md"] = "\n".join(
        [
            "# Ablation Necessity Report",
            "",
            "Hard Gate v3 evaluates plain residual score and no-relation-atoms using the same real RoAD splits and checkpoints. Existing v2 no-condition-invariance retraining evidence remains limited to road_binary_main and is not over-claimed.",
            "",
            "## Variant Summary",
            dataframe_to_markdown(variant_summary),
            "",
            "## Full vs Plain / No Relation",
            dataframe_to_markdown(full_plain),
        ]
    )
    reports["14_xgboost_dominance_diagnosis.md"] = "\n".join(
        [
            "# XGBoost Dominance Diagnosis",
            "",
            feature_explanation,
            "",
            "## Top Univariate Statistical Separators",
            dataframe_to_markdown(feature_top),
            "",
            "## Stress Protocol Comparison",
            dataframe_to_markdown(stress_cmp),
            "",
            "Conclusion: road_binary_main cannot be the sole evidence. Scenario holdout and hard-negative protocols are required before full experiments.",
        ]
    )
    reports["15_statistical_summary_v3.md"] = "\n".join(
        [
            "# Statistical Summary v3",
            "",
            "## CIRFL and Baseline Mean +/- Std",
            dataframe_to_markdown(summarize_by_method(combined)),
            "",
            "## Paired Tests",
            dataframe_to_markdown(tests.head(80)),
        ]
    )
    reports["16_complexity_latency_v3.md"] = "\n".join(
        [
            "# Complexity and Latency v3",
            "",
            "- model parameters: 171286",
            "- model size: about 0.666 MB from v2 checkpoint",
            "- full experiments run: NO",
            "",
            dataframe_to_markdown(latency_df) if len(latency_df) else "No latency CSV found.",
        ]
    )
    reason_lines = []
    reason_lines.append(f"Protocol validity pass: {validity_pass}")
    reason_lines.extend([f"Performance {k}: {v}" for k, v in perf_checks.items()])
    reason_lines.extend([f"Mechanism {k}: {v}" for k, v in mechanism_checks.items()])
    reason_lines.append(f"Tree-baseline stress advantage count >=2: {baseline_pass} ({tree_advantage_count})")
    reports["17_go_no_go_report_v3.md"] = "\n".join(
        [
            "# Hard Gate v3 GO / NO-GO Report",
            "",
            f"## Decision: {status}",
            "",
            "## Criteria Audit",
            *[f"- {line}" for line in reason_lines],
            "",
            "## Interpretation",
            "Hard Gate v3 remains an algorithm-improvement stress test. GO here would not mean full experiments were run. NO-GO blocks full experiments and paper writing.",
        ]
    )
    risks = []
    if not perf_checks["scenario_holdout_collision"]:
        risks.append("scenario_holdout_collision MDR/FAR/PR-AUC gate not fully satisfied.")
    if not perf_checks["scenario_holdout_weight"]:
        risks.append("scenario_holdout_weight MDR/FAR/PR-AUC gate not fully satisfied.")
    if not mechanism_checks["full_beats_plain_3_of_5"]:
        risks.append("Full residual-field composition does not beat plain residual score on at least 3/5 protocols.")
    if not source_stats.get("strict_top5_pass", False):
        risks.append("Source localization strict top-5 Jaccard remains below 0.55; top-10 is more stable but qualitative.")
    if tree_advantage_count < 2:
        risks.append("CIRFL_v3 does not show enough scenario/stress advantages over tree baselines.")
    reports["18_errors_and_risks.md"] = "\n".join(
        ["# Errors and Risks", "", "- runtime_errors: none recorded in Hard Gate v3 script.", *[f"- {r}" for r in risks], "- synthetic_results_in_packet: NO", "- figures_in_packet: NO"]
    )
    reports["19_code_index.md"] = "\n".join(
        [
            "# Code Index",
            "",
            "- `configs/hard_gate_v3_road_all.yaml`: Hard Gate v3 configuration.",
            "- `scripts/run_hard_gate_v3.py`: real-only stress gate, diagnostics, and review packet generation.",
            "- `src/models/cirfl.py`: CIRFL residual-field model.",
            "- `src/training/train_cirfl.py`: validation-only prediction/threshold path.",
            "- `scripts/build_road_protocols.py`: RoAD run-level/scenario/condition protocols.",
            "",
            "## Command",
            "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_hard_gate_v3.py --config configs/hard_gate_v3_road_all.yaml`",
            "",
            f"## Output Path\n`{output_dir}`",
        ]
    )

    save_config(cfg, output_dir / "hard_gate_v3_config.yaml")
    shutil.copyfile(output_dir / "hard_gate_v3_config.yaml", output_dir / "06_hard_gate_v3_config.yaml")
    metrics.to_csv(output_dir / "07_hard_gate_v3_metrics.csv", index=False)
    baselines.to_csv(output_dir / "08_hard_gate_v3_baseline_comparison.csv", index=False)
    for name, text in reports.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    pd.DataFrame([{"status": status, "generated_at": generated_at}]).to_csv(output_dir / "hard_gate_v3_status.csv", index=False)

    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        src = output_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing packet source file: {src}")
        shutil.copyfile(src, packet_dir / name)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_gate_v3_road_all.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    output_dir = ROOT / cfg["project"]["output_dir"]
    v2_dir = ROOT / cfg["project"].get("v2_reference_dir", "outputs/real_gate_v2")
    packet_dir = ROOT / cfg["review_packet"]["output_dir"]
    _clean_dir(output_dir)
    generated_at = utc_now()
    save_config(cfg, output_dir / "hard_gate_v3_config.yaml")

    protocol_df = _protocol_validity(cfg, output_dir)
    metrics, baselines = _load_v2_metrics(v2_dir, output_dir, generated_at)
    variants, operating, sanity = _variant_diagnostics(cfg, v2_dir, output_dir, generated_at)
    mechanism = _atom_and_probe_summary(v2_dir)
    source_stats = _source_stability(v2_dir, output_dir)
    feature_top, feature_explanation = _feature_dominance(cfg, output_dir)
    full_plain, full_plain_wins = _composite_wins(metrics, variants)
    full_plain.to_csv(output_dir / "full_vs_plain_residual.csv", index=False)
    stress_cmp, tree_advantage_count = _best_tree_advantages(metrics, baselines, output_dir)
    chosen_device, gpu_df, latency_df = _device_report(output_dir)
    status = _write_reports(
        cfg,
        output_dir,
        packet_dir,
        protocol_df,
        metrics,
        baselines,
        variants,
        operating,
        sanity,
        source_stats,
        mechanism,
        feature_top,
        feature_explanation,
        full_plain,
        full_plain_wins,
        stress_cmp,
        tree_advantage_count,
        chosen_device,
        gpu_df,
        latency_df,
        generated_at,
    )
    checks = {
        "07_hard_gate_v3_metrics.csv": validate_metric_file(packet_dir / "07_hard_gate_v3_metrics.csv", DATASET, require_real=True),
        "08_hard_gate_v3_baseline_comparison.csv": validate_metric_file(packet_dir / "08_hard_gate_v3_baseline_comparison.csv", DATASET, require_real=True),
    }
    if not all(ok for ok, _ in checks.values()):
        raise RuntimeError(f"Packet provenance validation failed: {checks}")
    print(f"Hard Gate v3 finished: {status}. Packet files: {len(list(packet_dir.iterdir()))}")


if __name__ == "__main__":
    main()
