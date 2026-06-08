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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_road_protocols import PROTOCOLS, config_for_protocol
from scripts.run_hard_gate_v3 import _atom_and_probe_summary, _feature_dominance
from scripts.run_real_gate_v2 import _load_model
from src.datasets import build_datasets_from_config
from src.evaluation.metrics import binary_metric_row, choose_threshold, paired_tests, summarize_by_method
from src.evaluation.residual_dominant import (
    evaluate_rd_row,
    fit_residual_dominant_composer,
    score_with_component,
)
from src.evaluation.score_composition import _fit_orientation_scale, apply_composer, collect_score_components, transform_components
from src.evaluation.thresholding import dual_level_metrics, select_threshold_from_validation, threshold_operating_points
from src.utils.config import load_config, save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import add_provenance, utc_now, validate_metric_file
from src.utils.torch_utils import resolve_device

DATASET = "RoAD"
SOURCE_TYPE = "real"
V2_METHOD = "CIRFL_v2"
V5_METHOD = "CIRFL_v5_CIRFL_RD"
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_protocol_validity_v5.md",
    "02_data_leakage_and_external_readiness_v5.md",
    "03_cirfl_v5_algorithm_spec.md",
    "04_novelty_guardrail_v5.md",
    "05_device_decision_report_v5.md",
    "06_hard_gate_v5_config.yaml",
    "07_hard_gate_v5_metrics.csv",
    "08_hard_gate_v5_baseline_comparison.csv",
    "09_best_path_recovery_v5.md",
    "10_score_design_and_selection_v5.md",
    "11_mdr_gate_v5.md",
    "12_mechanism_necessity_v5.md",
    "13_ablation_matrix_v5.csv",
    "14_xgboost_and_hard_negative_v5.md",
    "15_source_localization_v5.md",
    "16_statistical_summary_v5.md",
    "17_complexity_latency_v5.md",
    "18_go_no_go_report_v5.md",
    "19_errors_risks_and_code_index.md",
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


def _cfg_for_v5(proto: dict, base_cfg: dict) -> tuple[dict, str]:
    cfg = config_for_protocol(proto)
    for key in ["seeds", "device", "model", "training", "baselines", "latency", "review_packet"]:
        if key in base_cfg:
            cfg[key] = deepcopy(base_cfg[key])
    cfg["project"]["output_dir"] = base_cfg["project"]["output_dir"]
    cfg["project"]["stage"] = "hard_gate_v5"
    cfg["project"]["name"] = f"robot_cirfl_hard_gate_v5_{proto['protocol']}"
    split_mode = "cross_condition" if proto["family"] == "condition_holdout" else "main"
    return cfg, split_mode


def _protocol_validity(base_cfg: dict, output_dir: Path) -> pd.DataFrame:
    rows = []
    for proto in base_cfg.get("protocols", PROTOCOLS):
        cfg, split_mode = _cfg_for_v5(proto, base_cfg)
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
        row["can_compute_auroc_pr_far_mdr"] = bool(row["n_val_normal"] > 0 and row["n_val_anomaly"] > 0 and row["n_test_normal"] > 0 and row["n_test_anomaly"] > 0)
        row["protocol_valid"] = bool(row["can_compute_auroc_pr_far_mdr"])
        if proto["family"] == "condition_holdout":
            row["label_condition_confounding_risk"] = "HIGH"
            row["run_label_confounding_risk"] = "HIGH"
            row["can_be_main_evidence"] = "NO: high-confounding stress only"
        elif proto["family"] == "scenario_holdout":
            row["label_condition_confounding_risk"] = "MEDIUM-HIGH"
            row["run_label_confounding_risk"] = "MEDIUM-HIGH"
            row["can_be_main_evidence"] = "LIMITED: scenario stress evidence"
        else:
            row["label_condition_confounding_risk"] = "MEDIUM"
            row["run_label_confounding_risk"] = "MEDIUM"
            row["can_be_main_evidence"] = "NO: sanity/main only"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "protocol_validity_v5.csv", index=False)
    return df


def _far_limit(base_cfg: dict, protocol: str) -> float:
    return float(base_cfg.get("hard_gate_v5", {}).get("far_limits", {}).get(protocol, 0.40))


def _score_sanity(y: np.ndarray, scores: np.ndarray, protocol: str, seed: int) -> dict:
    yb = (np.asarray(y) > 0).astype(int)
    scores = np.asarray(scores, dtype=float)
    normal = scores[yb == 0]
    anomaly = scores[yb == 1]
    row = {
        "protocol": protocol,
        "seed": seed,
        "normal_score_mean": float(np.mean(normal)) if len(normal) else float("nan"),
        "normal_score_p95": float(np.percentile(normal, 95)) if len(normal) else float("nan"),
        "anomaly_score_mean": float(np.mean(anomaly)) if len(anomaly) else float("nan"),
        "anomaly_score_p05": float(np.percentile(anomaly, 5)) if len(anomaly) else float("nan"),
        "max_score": float(np.nanmax(scores)) if len(scores) else float("nan"),
        "min_score": float(np.nanmin(scores)) if len(scores) else float("nan"),
        "number_of_nan": int(np.isnan(scores).sum()),
        "number_of_inf": int(np.isinf(scores).sum()),
        "score_explosion_flag": bool(np.nanmax(np.abs(scores)) > 1e6) if len(scores) else True,
    }
    if len(np.unique(yb)) > 1:
        auc = roc_auc_score(yb, scores)
        inv_auc = roc_auc_score(yb, -scores)
        row["score_auroc"] = float(auc)
        row["inverted_score_auroc"] = float(inv_auc)
        row["score_direction_check"] = "PASS" if auc >= inv_auc else "FAIL"
    else:
        row["score_auroc"] = float("nan")
        row["inverted_score_auroc"] = float("nan")
        row["score_direction_check"] = "UNDEFINED"
    return row


def _component_row(method: str, seed: int, protocol: str, val_df: pd.DataFrame, test_df: pd.DataFrame, component: str, strategies: list[str], far_limit: float) -> dict:
    _, test_score, thr, strategy = score_with_component(val_df, test_df, component, strategies, far_limit, objective_mode="mdr_control")
    row = binary_metric_row(test_df["y_true"].to_numpy(), test_score, thr, method=method, seed=seed, protocol=protocol, class_pred=test_df.get("class_pred", pd.Series(dtype=int)).to_numpy() if "class_pred" in test_df else None)
    row["threshold_strategy"] = strategy
    row["threshold_source"] = "validation"
    row["score_weight_source"] = f"validation_component::{component}"
    return row


def _fixed_component_row(method: str, seed: int, protocol: str, val_df: pd.DataFrame, test_df: pd.DataFrame, components: list[str], weights: list[float], strategies: list[str], far_limit: float) -> dict:
    orientation, center, scale = _fit_orientation_scale(val_df, components)
    x_val = transform_components(val_df, names=components, orientation=orientation, center=center, scale=scale)
    x_test = transform_components(test_df, names=components, orientation=orientation, center=center, scale=scale)
    w = np.asarray(weights, dtype=float)
    w = w / max(w.sum(), 1e-12)
    val_score = x_val @ w
    test_score = x_test @ w
    selected = select_threshold_from_validation(val_df["y_true"].to_numpy(), val_score, strategies, far_limit=far_limit, objective_mode="mdr_control")
    row = binary_metric_row(test_df["y_true"].to_numpy(), test_score, selected.threshold, method=method, seed=seed, protocol=protocol, class_pred=test_df.get("class_pred", pd.Series(dtype=int)).to_numpy() if "class_pred" in test_df else None)
    row["threshold_strategy"] = selected.strategy
    row["threshold_source"] = "validation"
    row["score_weight_source"] = "validation_fixed_components"
    for name, weight in zip(components, w):
        row[f"w_{name}"] = float(weight)
    return row


def _load_baselines(v2_dir: Path, output_dir: Path, generated_at: str) -> pd.DataFrame:
    baselines = pd.read_csv(v2_dir / "gate_v2_baseline_comparison.csv").copy()
    baselines["dataset"] = DATASET
    baselines["source_type"] = SOURCE_TYPE
    baselines["output_path"] = str(output_dir / "hard_gate_v5_baseline_comparison.csv")
    baselines["generated_at"] = generated_at
    baselines.to_csv(output_dir / "hard_gate_v5_baseline_comparison.csv", index=False)
    return baselines


def _evaluate_v5(base_cfg: dict, v2_dir: Path, output_dir: Path, generated_at: str):
    strategies = base_cfg["hard_gate_v5"]["threshold_strategies"]
    metric_rows = []
    ablation_rows = []
    selection_rows = []
    component_rows = []
    operating_rows = []
    sanity_rows = []
    source_rows = []
    timing_rows = []
    chosen_device = resolve_device(base_cfg.get("device", "auto_fastest"))
    for proto in base_cfg["protocols"]:
        protocol = proto["protocol"]
        far_limit = _far_limit(base_cfg, protocol)
        cfg, split_mode = _cfg_for_v5(proto, base_cfg)
        train_ds, val_ds, test_ds, feature_cols, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
        counts = {**_count_row(train_ds, "train"), **_count_row(val_ds, "val"), **_count_row(test_ds, "test")}
        for seed in base_cfg["seeds"]:
            print(f"[HardGateV5] evaluating protocol={protocol} seed={seed}", flush=True)
            ckpt = v2_dir / f"{V2_METHOD}_seed{seed}_{protocol}.pt"
            model = _load_model(cfg, len(feature_cols), ckpt)
            device = next(model.parameters()).device
            start = time.perf_counter()
            collect_source_detail = protocol == "road_binary_main"
            val_df, val_source = collect_score_components(model, val_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=True, collect_sources=collect_source_detail)
            test_df, test_source = collect_score_components(model, test_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=True, collect_sources=collect_source_detail)
            no_rel_val, _ = collect_score_components(model, val_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=False, collect_sources=False)
            no_rel_test, _ = collect_score_components(model, test_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=False, collect_sources=False)
            inference_time = time.perf_counter() - start
            fit = fit_residual_dominant_composer(val_df, strategies, far_limit, objective_mode="mdr_control", allow_source=True)
            row, val_score, test_score = evaluate_rd_row(V5_METHOD, seed, protocol, val_df, test_df, fit)
            row.update(counts)
            row["n_test_windows"] = row["n_test"]
            row["checkpoint"] = str(ckpt)
            row["device"] = str(device)
            row["inference_time_sec"] = float(inference_time)
            metric_rows.append(row)
            audit = fit.selection_audit.copy()
            audit["protocol"] = protocol
            audit["seed"] = seed
            selection_rows.append(audit)
            sanity = _score_sanity(test_df["y_true"].to_numpy(), test_score, protocol, seed)
            sanity.update(counts)
            sanity_rows.append(sanity)
            op = threshold_operating_points(val_df["y_true"].to_numpy(), val_score, test_df["y_true"].to_numpy(), test_score, strategies, V5_METHOD, seed, protocol)
            op = op.assign(**counts)
            warn_thr = select_threshold_from_validation(val_df["y_true"].to_numpy(), val_score, ["target_recall_0.90", "target_recall_0.95"], far_limit=1.0).threshold
            alarm_thr = select_threshold_from_validation(val_df["y_true"].to_numpy(), val_score, ["target_far_0.10", "target_far_0.15"], far_limit=far_limit).threshold
            dual = dual_level_metrics(test_df["y_true"].to_numpy(), test_score, warn_thr, alarm_thr)
            for key, value in dual.items():
                op[key] = value
            operating_rows.append(op)

            for component in ["plain_residual", "relation_mismatch", "calibrated_energy", "source_concentration", "raw_window_energy", "prototype_margin", "anomaly_axis"]:
                c_row = _component_row(f"CIRFL_v5_component_{component}", seed, protocol, val_df, test_df, component, strategies, far_limit)
                c_row.update(counts)
                component_rows.append(c_row)

            ablation_specs = [
                ("CIRFL_v5_raw_residual_energy", val_df, test_df, ["raw_window_energy"], [1.0]),
                ("CIRFL_v5_condition_decoupled_residual_without_relation_atoms", no_rel_val, no_rel_test, ["plain_residual"], [1.0]),
                ("CIRFL_v5_relation_atoms_without_condition_decoupling_proxy", val_df, test_df, ["raw_window_energy", "relation_mismatch"], [0.70, 0.30]),
                ("CIRFL_v5_calibrated_residual_without_source_regularization", val_df, test_df, ["plain_residual", "relation_mismatch", "calibrated_energy"], [0.70, 0.20, 0.10]),
                ("CIRFL_v5_no_calibrated_energy", val_df, test_df, ["plain_residual", "relation_mismatch"], [0.75, 0.25]),
                ("CIRFL_v5_plain_residual_previous", val_df, test_df, ["plain_residual"], [1.0]),
                ("CIRFL_v5_relation_mismatch_only", val_df, test_df, ["relation_mismatch"], [1.0]),
            ]
            for method, a_val, a_test, names, weights in ablation_specs:
                try:
                    a_row = _fixed_component_row(method, seed, protocol, a_val, a_test, names, weights, strategies, far_limit)
                    a_row.update(counts)
                    a_row["n_test_windows"] = a_row["n_test"]
                    ablation_rows.append(a_row)
                except Exception as exc:
                    ablation_rows.append({"method": method, "seed": seed, "protocol": protocol, "error": str(exc), **counts})

            if protocol == "road_binary_main" and len(test_source):
                val_channels = [c for c in val_source.columns if c.startswith("channel_")]
                med = val_source[val_channels].median(axis=0)
                iqr = val_source[val_channels].quantile(0.75) - val_source[val_channels].quantile(0.25)
                iqr = iqr.replace(0, 1.0)
                test_norm = ((test_source[val_channels] - med) / iqr).clip(lower=0.0, upper=20.0)
                test_norm["y_true"] = test_source["y_true"].to_numpy()
                for group, sub in [("normal", test_norm[test_norm["y_true"] == 0]), ("anomaly", test_norm[test_norm["y_true"] > 0])]:
                    if len(sub):
                        avg = sub[val_channels].mean().sort_values(ascending=False)
                        source_rows.append({"dataset": DATASET, "source_type": SOURCE_TYPE, "protocol": protocol, "seed": seed, "label_group": group, "top5_channels": ";".join(avg.head(5).index), "top10_channels": ";".join(avg.head(10).index), "top10_mean_contribution": float(avg.head(10).mean())})
            timing_rows.append({"protocol": protocol, "seed": seed, "device": str(device), "inference_time_sec": float(inference_time), "n_val_windows": len(val_ds), "n_test_windows": len(test_ds)})
            print(f"[HardGateV5] finished protocol={protocol} seed={seed}", flush=True)
    metrics = pd.DataFrame(metric_rows)
    ablations = pd.DataFrame(ablation_rows)
    selection = pd.concat(selection_rows, ignore_index=True) if selection_rows else pd.DataFrame()
    components = pd.DataFrame(component_rows)
    operating = pd.concat(operating_rows, ignore_index=True) if operating_rows else pd.DataFrame()
    sanity = pd.DataFrame(sanity_rows)
    source = pd.DataFrame(source_rows)
    timing = pd.DataFrame(timing_rows)
    outputs = [
        (metrics, "hard_gate_v5_metrics.csv"),
        (ablations, "ablation_matrix_v5.csv"),
        (components, "score_component_diagnostics_v5.csv"),
        (operating, "operating_points_v5.csv"),
        (sanity, "score_sanity_v5.csv"),
    ]
    for df, name in outputs:
        if len(df):
            prov = add_provenance(df, DATASET, SOURCE_TYPE, output_dir / name, generated_at)
            prov.to_csv(output_dir / name, index=False)
            if name == "hard_gate_v5_metrics.csv":
                metrics = prov
            elif name == "ablation_matrix_v5.csv":
                ablations = prov
            elif name == "score_component_diagnostics_v5.csv":
                components = prov
            elif name == "operating_points_v5.csv":
                operating = prov
            elif name == "score_sanity_v5.csv":
                sanity = prov
    selection.to_csv(output_dir / "score_selection_audit_v5.csv", index=False)
    source.to_csv(output_dir / "source_stability_v5.csv", index=False)
    timing.to_csv(output_dir / "device_timing_v5.csv", index=False)
    return metrics, ablations, selection, components, operating, sanity, source, {"timing": timing, "selected_device": str(chosen_device)}


def _jaccard_from_source(source: pd.DataFrame) -> dict:
    out = {"top5_jaccard_all_groups": float("nan"), "top10_jaccard_all_groups": float("nan"), "top5_jaccard_anomaly": float("nan"), "top10_jaccard_anomaly": float("nan")}
    for group_name, sub in [("all_groups", source), ("anomaly", source[source.get("label_group", "") == "anomaly"] if len(source) else source)]:
        for k in [5, 10]:
            sets = [set(text.split(";")) for text in sub.get(f"top{k}_channels", pd.Series(dtype=str)).dropna().astype(str)]
            if len(sets) > 1:
                vals = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
                out[f"top{k}_jaccard_{group_name}"] = float(np.mean(vals))
    out["source_stability_pass"] = bool(out.get("top5_jaccard_anomaly", 0.0) >= 0.45 or out.get("top10_jaccard_anomaly", 0.0) >= 0.60)
    return out


def _composite_better(full_row: pd.Series, other_row: pd.Series) -> tuple[bool, int]:
    checks = [
        full_row["macro_f1_mean"] >= other_row["macro_f1_mean"],
        full_row["weighted_f1_mean"] >= other_row["weighted_f1_mean"],
        full_row["auroc_mean"] >= other_row["auroc_mean"],
        full_row["pr_auc_mean"] >= other_row["pr_auc_mean"],
        full_row["far_mean"] <= other_row["far_mean"],
        full_row["mdr_mean"] <= other_row["mdr_mean"],
    ]
    return bool(sum(checks) >= 4), int(sum(checks))


def _mechanism_necessity(metrics: pd.DataFrame, ablations: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    full = summarize_by_method(metrics)
    abl = summarize_by_method(ablations.dropna(subset=["macro_f1"]))
    rows = []
    raw_wins = 0
    no_rel_wins = 0
    core_drop_methods = set()
    for protocol in sorted(metrics["protocol"].unique()):
        f = full[(full["method"] == V5_METHOD) & (full["protocol"] == protocol)]
        if f.empty:
            continue
        f = f.iloc[0]
        for method in sorted(abl["method"].unique()):
            a = abl[(abl["method"] == method) & (abl["protocol"] == protocol)]
            if a.empty:
                continue
            a = a.iloc[0]
            better, n = _composite_better(f, a)
            macro_drop = f["macro_f1_mean"] - a["macro_f1_mean"]
            pr_drop = f["pr_auc_mean"] - a["pr_auc_mean"]
            mdr_worse = (a["mdr_mean"] - f["mdr_mean"]) / max(f["mdr_mean"], 1e-6)
            obvious = bool(macro_drop >= 0.03 or pr_drop >= 0.03 or mdr_worse >= 0.15)
            if obvious:
                core_drop_methods.add(method)
            if method == "CIRFL_v5_raw_residual_energy" and better:
                raw_wins += 1
            if method == "CIRFL_v5_condition_decoupled_residual_without_relation_atoms" and better:
                no_rel_wins += 1
            rows.append({"protocol": protocol, "comparison": method, "cirfl_v5_better_composite": better, "n_better_metrics": n, "macro_f1_drop": macro_drop, "pr_auc_drop": pr_drop, "mdr_relative_worse": mdr_worse, "obvious_drop": obvious})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "mechanism_necessity_v5.csv", index=False)
    stats = {"v5_beats_raw_protocols": raw_wins, "v5_beats_no_relation_protocols": no_rel_wins, "core_ablation_drop_count": len(core_drop_methods), "core_ablation_drop_methods": sorted(core_drop_methods)}
    return df, stats


def _full_vs_references(metrics: pd.DataFrame, v3_dir: Path, v4_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, dict]:
    frames = []
    if (v3_dir / "07_hard_gate_v3_metrics.csv").exists():
        frames.append(pd.read_csv(v3_dir / "07_hard_gate_v3_metrics.csv"))
    elif (v3_dir / "hard_gate_v3_metrics.csv").exists():
        frames.append(pd.read_csv(v3_dir / "hard_gate_v3_metrics.csv"))
    if (v3_dir / "score_variant_metrics.csv").exists():
        frames.append(pd.read_csv(v3_dir / "score_variant_metrics.csv"))
    if (v4_dir / "07_hard_gate_v4_metrics.csv").exists():
        frames.append(pd.read_csv(v4_dir / "07_hard_gate_v4_metrics.csv"))
    elif (v4_dir / "hard_gate_v4_metrics.csv").exists():
        frames.append(pd.read_csv(v4_dir / "hard_gate_v4_metrics.csv"))
    if (v4_dir / "mechanism_necessity_matrix_v4.csv").exists():
        frames.append(pd.read_csv(v4_dir / "mechanism_necessity_matrix_v4.csv"))
    frames.append(metrics)
    combined = pd.concat(frames, ignore_index=True)
    summary = summarize_by_method(combined.dropna(subset=["macro_f1"]))
    summary.to_csv(output_dir / "best_path_recovery_v5.csv", index=False)
    score_defs = pd.DataFrame(
        [
            {"score_path": "CIRFL_v3 original score", "definition": "checkpoint native score: calibrated residual + prototype margin + anomaly axis", "main_detection": "reference only"},
            {"score_path": "CIRFL_v4 full composition", "definition": "validation-learned broad component composition; failed due unstable component weights", "main_detection": "failure control"},
            {"score_path": "CIRFL_v5 / CIRFL-RD", "definition": "residual-dominant fixed-grid composition of condition-decoupled residual, relation mismatch, and calibrated energy; source optional by validation guard", "main_detection": "candidate"},
            {"score_path": "raw residual energy", "definition": "raw window energy proxy without condition decoupling or relation field", "main_detection": "ablation"},
            {"score_path": "condition-decoupled residual without relation atoms", "definition": "condition field removed but relation atoms disabled at evaluation", "main_detection": "ablation"},
        ]
    )
    score_defs.to_csv(output_dir / "score_definition_matrix_v5.csv", index=False)
    v5 = summary[summary["method"] == V5_METHOD]
    v3 = summary[summary["method"] == "CIRFL_v3"]
    stats = {"v5_not_clearly_worse_than_v3": False, "road_main_not_v4_level": False}
    try:
        v5_main = v5[v5["protocol"] == "road_binary_main"].iloc[0]
        if not v3.empty:
            v3_main = v3[v3["protocol"] == "road_binary_main"].iloc[0]
            stats["v5_not_clearly_worse_than_v3"] = bool(v5_main["macro_f1_mean"] >= v3_main["macro_f1_mean"] - 0.05 and v5_main["pr_auc_mean"] >= v3_main["pr_auc_mean"] - 0.05)
        stats["road_main_not_v4_level"] = bool(v5_main["macro_f1_mean"] > 0.526 and v5_main["pr_auc_mean"] > 0.544 and v5_main["far_mean"] < 0.489)
    except Exception:
        pass
    return summary, stats


def _tree_advantages(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, int]:
    full = summarize_by_method(metrics)
    base = summarize_by_method(baselines)
    tree_methods = {"random_forest", "xgboost", "lightgbm"}
    rows = []
    count = 0
    for protocol in sorted(metrics["protocol"].unique()):
        f = full[(full["method"] == V5_METHOD) & (full["protocol"] == protocol)]
        b = base[(base["method"].isin(tree_methods)) & (base["protocol"] == protocol)]
        if f.empty or b.empty:
            continue
        f = f.iloc[0]
        best_macro = b.loc[b["macro_f1_mean"].idxmax()]
        best_pr = b.loc[b["pr_auc_mean"].idxmax()]
        best_mdr = b.loc[b["mdr_mean"].idxmin()]
        adv = bool((f["macro_f1_mean"] > best_macro["macro_f1_mean"]) or (f["pr_auc_mean"] > best_pr["pr_auc_mean"]) or (f["mdr_mean"] < best_mdr["mdr_mean"]))
        if protocol != "road_binary_main" and adv:
            count += 1
        rows.append({"protocol": protocol, "cirfl_macro_f1": f["macro_f1_mean"], "best_tree_macro_method": best_macro["method"], "best_tree_macro_f1": best_macro["macro_f1_mean"], "cirfl_pr_auc": f["pr_auc_mean"], "best_tree_pr_method": best_pr["method"], "best_tree_pr_auc": best_pr["pr_auc_mean"], "cirfl_mdr": f["mdr_mean"], "best_tree_mdr_method": best_mdr["method"], "best_tree_mdr": best_mdr["mdr_mean"], "cirfl_has_any_tree_advantage": adv})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "stress_protocol_comparison_v5.csv", index=False)
    return df, count


def _protocol_winners(metrics: pd.DataFrame, baselines: pd.DataFrame, ablations: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    combined = pd.concat([metrics, baselines, ablations], ignore_index=True)
    summary = summarize_by_method(combined.dropna(subset=["macro_f1"]))
    rows = []
    for protocol in sorted(summary["protocol"].unique()):
        sub = summary[summary["protocol"] == protocol]
        for metric, ascending in [("macro_f1_mean", False), ("pr_auc_mean", False), ("mdr_mean", True), ("far_mean", True)]:
            winner = sub.sort_values(metric, ascending=ascending).iloc[0]
            rows.append({"protocol": protocol, "metric": metric, "winner": winner["method"], "value": winner[metric]})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "winner_summary_v5.csv", index=False)
    return df


def _hard_negative_stress(base_cfg: dict, output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, str]:
    proto = [p for p in base_cfg["protocols"] if p["protocol"] == "road_binary_main"][0]
    cfg, split_mode = _cfg_for_v5(proto, base_cfg)
    train_ds, val_ds, test_ds, _, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
    # Use train-only window statistics for artifact stress. Full flattened windows are too high-dimensional for gate diagnostics.
    stat_dim = int(train_ds.windows[0].shape[1]) * 5
    x_train = train_ds.to_flat_features()[:, :stat_dim]
    y_train = (train_ds.labels() > 0).astype(int)
    x_val = val_ds.to_flat_features()[:, :stat_dim]
    x_test = test_ds.to_flat_features()[:, :stat_dim]
    y_val = val_ds.labels()
    y_test = test_ds.labels()
    effect = np.zeros(x_train.shape[1], dtype=float)
    for j in range(x_train.shape[1]):
        col = x_train[:, j]
        normal = col[y_train == 0]
        anomaly = col[y_train == 1]
        denom = np.nanstd(col) + 1e-8
        effect[j] = abs(float(np.nanmean(anomaly) - np.nanmean(normal)) / denom) if len(normal) and len(anomaly) else 0.0
    top = np.argsort(-effect)
    stress_masks = {
        "all_flat_features": np.ones(x_train.shape[1], dtype=bool),
        "remove_top5_train_univariate": np.ones(x_train.shape[1], dtype=bool),
        "remove_top10_train_univariate": np.ones(x_train.shape[1], dtype=bool),
    }
    stress_masks["remove_top5_train_univariate"][top[:5]] = False
    stress_masks["remove_top10_train_univariate"][top[:10]] = False
    rows = []
    errors = []
    for seed in base_cfg["seeds"]:
        available_methods = ["random_forest"]
        try:
            from xgboost import XGBClassifier
            available_methods.append("xgboost")
        except Exception as exc:
            errors.append(f"xgboost stress unavailable: {exc}")
        try:
            from lightgbm import LGBMClassifier
            available_methods.append("lightgbm")
        except Exception as exc:
            errors.append(f"lightgbm stress unavailable: {exc}")
        for stress, mask in stress_masks.items():
            for method in available_methods:
                try:
                    if method == "random_forest":
                        model = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=seed, n_jobs=1, class_weight="balanced")
                    elif method == "xgboost":
                        model = XGBClassifier(n_estimators=40, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, objective="binary:logistic", eval_metric="logloss", random_state=seed, n_jobs=1)
                    elif method == "lightgbm":
                        model = LGBMClassifier(n_estimators=40, max_depth=4, learning_rate=0.08, random_state=seed, n_jobs=1, verbose=-1)
                    else:
                        raise ValueError(method)
                    model.fit(x_train[:, mask], y_train)
                    if hasattr(model, "predict_proba"):
                        score_val = model.predict_proba(x_val[:, mask])[:, 1]
                        score_test = model.predict_proba(x_test[:, mask])[:, 1]
                    else:
                        score_val = model.predict(x_val[:, mask])
                        score_test = model.predict(x_test[:, mask])
                    threshold = choose_threshold(y_val, score_val)
                    row = binary_metric_row(y_test, score_test, threshold, method=f"{method}::{stress}", seed=seed, protocol="road_binary_main_channel_removal_stress")
                    row.update(_count_row(train_ds, "train"))
                    row.update(_count_row(val_ds, "val"))
                    row.update(_count_row(test_ds, "test"))
                    row["n_test_windows"] = row["n_test"]
                    row["stress_type"] = stress
                    row["removed_feature_count"] = int((~mask).sum())
                    rows.append(row)
                except Exception as exc:
                    errors.append(f"{method} {stress} failed seed {seed}: {exc}")
    df = pd.DataFrame(rows)
    if len(df):
        df = add_provenance(df, DATASET, SOURCE_TYPE, output_dir / "hard_negative_stress_v5.csv", generated_at)
    df.to_csv(output_dir / "hard_negative_stress_v5.csv", index=False)
    explanation = "Channel-removal stress removes top train-only statistical separators from mean/std/min/max/slope features. It diagnoses tree artifact sensitivity; CIRFL_v5 itself was not retrained with channel removal in this algorithm-improvement gate."
    if errors:
        explanation += " Errors: " + " | ".join(errors[:6])
    return df, explanation


def _external_readiness(output_dir: Path) -> tuple[pd.DataFrame, str]:
    rows = []
    datasets = [
        ("IMAD-DS robotic arm subset", ROOT / "data/raw/imadds", ROOT / "src/datasets/adapters/imadds.py", ROOT / "configs/datasets/imadds_robotic_arm.yaml"),
        ("NIST UR robot health/degradation", ROOT / "data/raw/nist_ur", ROOT / "src/datasets/adapters/nist_ur.py", ROOT / "configs/datasets/nist_ur.yaml"),
        ("KUKA torque/collision", ROOT / "data/raw/kuka_torque", ROOT / "src/datasets/adapters/kuka_torque.py", ROOT / "configs/datasets/kuka_torque.yaml"),
    ]
    for name, raw_dir, adapter, cfg in datasets:
        raw_files = sorted(raw_dir.glob("*")) if raw_dir.exists() else []
        rows.append({"dataset": name, "source_type": "external_adapter_status", "raw_dir": str(raw_dir), "raw_files_present": int(len(raw_files)), "adapter_exists": bool(adapter.exists()), "config_exists": bool(cfg.exists()), "status": "READY_FOR_DRY_RUN" if raw_files and adapter.exists() and cfg.exists() else "NEED_DATA"})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "external_data_readiness_v5.csv", index=False)
    lines = ["External dataset preparation does not run full experiments. No synthetic substitute is used.", "", dataframe_to_markdown(df)]
    return df, "\n".join(lines)


def _device_report(extra: dict) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    rows = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            rows.append({"device": f"cuda:{idx}", "gpu_name": torch.cuda.get_device_name(idx), "available": True, "assigned_task": "single-GPU v5 evaluation" if str(extra["selected_device"]) == f"cuda:{idx}" else "idle/reserved for seed/protocol parallelism"})
    else:
        rows.append({"device": "cpu", "gpu_name": "NA", "available": False, "assigned_task": "all evaluation"})
    latency_path = ROOT / "outputs/real_gate_v2/complexity_latency_v2.csv"
    latency = pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame()
    return str(extra["selected_device"]), pd.DataFrame(rows), latency


def _write_reports(cfg, output_dir, packet_dir, protocol_df, metrics, baselines, ablations, selection, components, operating, sanity, source_df, source_stats, mechanism_matrix, mechanism_stats, best_summary, recovery_stats, stress_cmp, tree_adv_count, feature_top, feature_explanation, hard_stress, hard_stress_expl, external_df, external_text, chosen_device, gpu_df, latency_df, winners, generated_at):
    full_summary = summarize_by_method(metrics)
    base_summary = summarize_by_method(baselines)
    ablation_summary = summarize_by_method(ablations.dropna(subset=["macro_f1"]))
    tests = paired_tests(pd.concat([metrics, baselines, ablations], ignore_index=True).dropna(subset=["macro_f1"]), reference_method=V5_METHOD)
    def srow(protocol: str) -> pd.Series:
        return full_summary[(full_summary["method"] == V5_METHOD) & (full_summary["protocol"] == protocol)].iloc[0]
    main = srow("road_binary_main")
    collision = srow("scenario_holdout_collision")
    weight = srow("scenario_holdout_weight")
    velocity = srow("scenario_holdout_velocity")
    v4_ref = cfg["hard_gate_v5"].get("v4_reference_mdr", {})
    collision_drop_v4 = (float(v4_ref.get("scenario_holdout_collision", collision["mdr_mean"])) - collision["mdr_mean"]) / max(float(v4_ref.get("scenario_holdout_collision", collision["mdr_mean"])), 1e-9)
    weight_drop_v4 = (float(v4_ref.get("scenario_holdout_weight", weight["mdr_mean"])) - weight["mdr_mean"]) / max(float(v4_ref.get("scenario_holdout_weight", weight["mdr_mean"])), 1e-9)
    validity_pass = bool(protocol_df["protocol_valid"].sum() >= 4 and protocol_df["protocol_valid"].all())
    recovery_pass = bool(recovery_stats["v5_not_clearly_worse_than_v3"] and recovery_stats["road_main_not_v4_level"])
    perf = {
        "road_binary_main": bool(main["macro_f1_mean"] >= 0.65 and main["weighted_f1_mean"] >= 0.75 and main["pr_auc_mean"] >= 0.95 and main["far_mean"] <= 0.30 and main["mdr_mean"] <= 0.05),
        "scenario_holdout_velocity": bool(velocity["macro_f1_mean"] >= 0.65 and velocity["mdr_mean"] <= 0.25 and velocity["far_mean"] <= 0.35),
        "scenario_holdout_collision": bool((collision["mdr_mean"] <= 0.75 or collision_drop_v4 >= 0.20) and collision["far_mean"] <= 0.40 and collision["pr_auc_mean"] >= 0.60),
        "scenario_holdout_weight": bool((weight["mdr_mean"] <= 0.60 or weight_drop_v4 >= 0.20) and weight["far_mean"] <= 0.35 and weight["pr_auc_mean"] >= 0.80),
    }
    mechanism = _atom_and_probe_summary(ROOT / cfg["project"].get("v2_reference_dir", "outputs/real_gate_v2"))
    mechanism_checks = {
        "condition_leakage_pass": str(mechanism.get("condition_probe_conclusion", "")).upper() == "PASS" and mechanism.get("condition_probe_accuracy_zh", 1.0) <= 0.65,
        "atom_diversity_pass": mechanism.get("atom_mean_offdiag_cosine", 1.0) <= 0.20 and mechanism.get("effective_atom_fraction", 0.0) >= 0.80,
        "source_stability_pass_or_explained": bool(source_stats.get("source_stability_pass", False)),
        "score_sanity_pass": bool(sanity["number_of_nan"].sum() == 0 and sanity["number_of_inf"].sum() == 0 and not sanity["score_explosion_flag"].astype(bool).any() and not (sanity["score_direction_check"] == "FAIL").any()),
    }
    mechanism_necessity_pass = bool(mechanism_stats["v5_beats_raw_protocols"] >= 3 and mechanism_stats["v5_beats_no_relation_protocols"] >= 2 and mechanism_stats["core_ablation_drop_count"] >= 2)
    tree_pass = bool(tree_adv_count >= 2)
    status = "GO" if validity_pass and recovery_pass and all(perf.values()) and mechanism_necessity_pass and all(mechanism_checks.values()) and tree_pass else "NO-GO"
    reports = {}
    provenance = f"- dataset: RoAD\n- source_type: real\n- protocols: {', '.join(protocol_df['protocol'].astype(str))}\n- seeds: {', '.join(map(str, cfg['seeds']))}\n- generated_at: {generated_at}"
    reports["00_readme_for_chatgpt.md"] = "\n".join(["# Readme for ChatGPT", "", "- current_stage: Hard Gate v5", f"- status: {status}", "- contains synthetic results: NO", "- generated images/figures: NO", "- entered full experiments: NO", provenance, f"- valid_protocols: {int(protocol_df['protocol_valid'].sum())}", "- runnable command: `/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_hard_gate_v5.py --config configs/hard_gate_v5_road_all.yaml`"])
    reports["01_protocol_validity_v5.md"] = "# Protocol Validity v5\n\n" + provenance + "\n\n" + dataframe_to_markdown(protocol_df)
    reports["02_data_leakage_and_external_readiness_v5.md"] = "\n".join(["# Data Leakage and External Readiness v5", "", provenance, "", "- RoAD split unit: run_id / recording", "- windowing: after split-unit filtering", "- normalization: train-only median/IQR with train-only clipping statistics", "- augmentation leakage: none", "- leakage risk: LOW for split mechanics", "- confounding risk: MEDIUM-HIGH because label, run and condition are not fully disentangled in RoAD", "", "## External Data Readiness", external_text])
    reports["03_cirfl_v5_algorithm_spec.md"] = "\n".join(["# CIRFL-RD / CIRFL v5 Algorithm Spec", "", provenance, "", "CIRFL_v5 is a residual-dominant detector. The main anomaly score is selected from a fixed validation-only grid over condition-decoupled residual energy, relation mismatch energy, and calibrated residual energy. Source concentration is added only if validation improves FAR/MDR tradeoff without non-degradation violations.", "", "Prototype margin and anomaly-axis scores are disabled for primary detection in v5 because v4 showed unstable score composition. They remain auxiliary diagnostics only. Source localization remains based on calibrated residual energy contributions."])
    reports["04_novelty_guardrail_v5.md"] = "\n".join(["# Novelty Guardrail v5", "", provenance, "", "CIRFL_v5 is not an improved/hybrid/stacked Transformer, GNN, LSTM, CNN or AE. It redefines the usable path as condition-decoupled residual-field detection, not a larger architecture.", "", "It is not plain residual renamed: v5 keeps train-time condition decoupling, relation atom residual field construction, calibrated residual scale, validation-only residual-dominant composition, and source contribution. If the gate shows it is only equivalent to plain residual, the report marks NO-GO rather than masking the failure."])
    reports["05_device_decision_report_v5.md"] = "\n".join(["# Device Decision Report v5", "", provenance, "", f"- selected_device_for_PyTorch: {chosen_device}", "- multi_gpu_used: NO", "- reason: checkpoint-based v5 evaluation is small; single GPU was selected by auto benchmark and avoids DataParallel overhead. cuda:1/cuda:2 remain available for future seed/protocol parallelism.", "", "## GPUs", dataframe_to_markdown(gpu_df), "", "## Latency Reference", dataframe_to_markdown(latency_df) if len(latency_df) else "No latency reference CSV found."])
    reports["09_best_path_recovery_v5.md"] = "\n".join(["# Best-Path Recovery v5", "", provenance, "", f"- v5_not_clearly_worse_than_v3: {recovery_stats['v5_not_clearly_worse_than_v3']}", f"- road_main_not_v4_level: {recovery_stats['road_main_not_v4_level']}", "", "## Candidate Summary", dataframe_to_markdown(best_summary[best_summary["method"].astype(str).str.contains("CIRFL_v3|CIRFL_v4|CIRFL_v5|plain_residual", regex=True)].head(80))])
    reports["10_score_design_and_selection_v5.md"] = "\n".join(["# Score Design and Selection v5", "", provenance, "", "- score_weight_source: validation fixed grid", "- threshold_source: validation only", "- prototype_margin_in_primary_score: NO", "- anomaly_axis_in_primary_score: NO", "- degenerate single-component selection rejected: YES", "", "## Selection Audit", dataframe_to_markdown(selection.head(80)), "", "## Component Diagnostics", dataframe_to_markdown(components.groupby(["protocol", "method"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index().head(80))])
    reports["11_mdr_gate_v5.md"] = "\n".join(["# MDR Gate v5", "", provenance, "", f"- collision_mdr_relative_drop_vs_v4: {collision_drop_v4:.6f}", f"- weight_mdr_relative_drop_vs_v4: {weight_drop_v4:.6f}", "", "## CIRFL v5 Summary", dataframe_to_markdown(full_summary[full_summary["method"] == V5_METHOD][["protocol", "macro_f1_mean", "weighted_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean"]]), "", "## Operating Points", dataframe_to_markdown(operating.groupby(["protocol", "strategy"])[["macro_f1", "far", "mdr", "pr_auc", "warning_recall", "warning_far", "alarm_far", "alarm_mdr"]].mean(numeric_only=True).reset_index().head(100))])
    reports["12_mechanism_necessity_v5.md"] = "\n".join(["# Mechanism Necessity v5", "", provenance, "", f"- v5_beats_raw_residual_protocols: {mechanism_stats['v5_beats_raw_protocols']}/5", f"- v5_beats_conditioned_no_relation_protocols: {mechanism_stats['v5_beats_no_relation_protocols']}/5", f"- core_ablation_drop_count: {mechanism_stats['core_ablation_drop_count']}", f"- core_ablation_drop_methods: {', '.join(mechanism_stats['core_ablation_drop_methods'])}", "", dataframe_to_markdown(mechanism_matrix.head(120))])
    reports["14_xgboost_and_hard_negative_v5.md"] = "\n".join(["# XGBoost Artifact and Hard Negative v5", "", provenance, "", feature_explanation, "", "## Top Univariate Separators", dataframe_to_markdown(feature_top), "", "## Tree Stress vs CIRFL", dataframe_to_markdown(stress_cmp), "", "## Channel-Removal Stress", hard_stress_expl, "", dataframe_to_markdown(hard_stress.groupby(["method", "stress_type"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index().head(80)) if len(hard_stress) else "No channel-removal rows generated."])
    reports["15_source_localization_v5.md"] = "\n".join(["# Source Localization v5", "", provenance, "", f"- top5_jaccard_anomaly: {source_stats.get('top5_jaccard_anomaly', np.nan)}", f"- top10_jaccard_anomaly: {source_stats.get('top10_jaccard_anomaly', np.nan)}", f"- source_stability_pass: {source_stats.get('source_stability_pass', False)}", "- RoAD source ground-truth labels: unavailable, so localization remains engineering qualitative evidence if top-k stability is insufficient.", "", dataframe_to_markdown(source_df)])
    reports["16_statistical_summary_v5.md"] = "\n".join(["# Statistical Summary v5", "", provenance, "", "## Mean +/- Std", dataframe_to_markdown(summarize_by_method(pd.concat([metrics, baselines], ignore_index=True))), "", "## Winners", dataframe_to_markdown(winners), "", "## Paired Tests", dataframe_to_markdown(tests.head(120))])
    reports["17_complexity_latency_v5.md"] = "\n".join(["# Complexity and Latency v5", "", provenance, "", "- parameters: 171286", "- model_size_mb: about 0.666", "- full experiments run: NO", "", dataframe_to_markdown(latency_df) if len(latency_df) else "No latency reference CSV found."])
    criteria = [f"Protocol validity pass: {validity_pass}", f"Recovery pass: {recovery_pass}"] + [f"Performance {k}: {v}" for k, v in perf.items()] + [f"Mechanism necessity pass: {mechanism_necessity_pass}"] + [f"Mechanism diagnostic {k}: {v}" for k, v in mechanism_checks.items()] + [f"Tree/stress advantage pass: {tree_pass} ({tree_adv_count})"]
    reports["18_go_no_go_report_v5.md"] = "\n".join(["# Hard Gate v5 GO / NO-GO Report", "", provenance, "", f"## Decision: {status}", "", "## Criteria Audit", *[f"- {x}" for x in criteria], "", "NO-GO blocks full experiments, manuscript writing and abstract generation."])
    risks = []
    if not recovery_pass:
        risks.append("CIRFL_v5 did not recover v3 main-path strength; consider rollback or repositioning.")
    if not perf["scenario_holdout_collision"]:
        risks.append("scenario_holdout_collision MDR/FAR/PR gate remains unresolved.")
    if not perf["scenario_holdout_weight"]:
        risks.append("scenario_holdout_weight MDR/FAR/PR gate remains unresolved.")
    if not mechanism_necessity_pass:
        risks.append("Mechanism necessity is not proven against raw/no-relation residual ablations.")
    reports["19_errors_risks_and_code_index.md"] = "\n".join(["# Errors, Risks and Code Index", "", provenance, "", "## Risks", *[f"- {r}" for r in risks], "- synthetic_results_in_packet: NO", "- figures_in_packet: NO", "", "## Code Index", "- `src/evaluation/residual_dominant.py`: fixed-grid residual-dominant v5 scorer.", "- `scripts/run_hard_gate_v5.py`: real-only Hard Gate v5 runner and packet builder.", "- `configs/hard_gate_v5_road_all.yaml`: v5 config.", "- `src/models/cirfl.py`: existing residual-field components used; no large module added.", "", "## Command", "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_hard_gate_v5.py --config configs/hard_gate_v5_road_all.yaml`"])
    save_config(cfg, output_dir / "hard_gate_v5_config.yaml")
    shutil.copyfile(output_dir / "hard_gate_v5_config.yaml", output_dir / "06_hard_gate_v5_config.yaml")
    metrics.to_csv(output_dir / "07_hard_gate_v5_metrics.csv", index=False)
    baselines.to_csv(output_dir / "08_hard_gate_v5_baseline_comparison.csv", index=False)
    ablations.to_csv(output_dir / "13_ablation_matrix_v5.csv", index=False)
    for name, text in reports.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        shutil.copyfile(output_dir / name, packet_dir / name)
    return status, {"collision_drop_v4": collision_drop_v4, "weight_drop_v4": weight_drop_v4}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_gate_v5_road_all.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    output_dir = ROOT / cfg["project"]["output_dir"]
    v2_dir = ROOT / cfg["project"].get("v2_reference_dir", "outputs/real_gate_v2")
    v3_dir = ROOT / cfg["project"].get("v3_reference_dir", "outputs/hard_gate_v3")
    v4_dir = ROOT / "outputs/hard_gate_v4"
    packet_dir = ROOT / cfg["review_packet"]["output_dir"]
    _clean_dir(output_dir)
    generated_at = utc_now()
    save_config(cfg, output_dir / "hard_gate_v5_config.yaml")
    protocol_df = _protocol_validity(cfg, output_dir)
    baselines = _load_baselines(v2_dir, output_dir, generated_at)
    metrics, ablations, selection, components, operating, sanity, source_df, extra = _evaluate_v5(cfg, v2_dir, output_dir, generated_at)
    best_summary, recovery_stats = _full_vs_references(metrics, v3_dir, v4_dir, output_dir)
    mechanism_matrix, mechanism_stats = _mechanism_necessity(metrics, ablations, output_dir)
    stress_cmp, tree_adv_count = _tree_advantages(metrics, baselines, output_dir)
    winners = _protocol_winners(metrics, baselines, ablations, output_dir)
    feature_top, feature_explanation = _feature_dominance(cfg, output_dir)
    hard_stress, hard_stress_expl = _hard_negative_stress(cfg, output_dir, generated_at)
    external_df, external_text = _external_readiness(output_dir)
    source_stats = _jaccard_from_source(source_df)
    chosen_device, gpu_df, latency_df = _device_report(extra)
    status, _ = _write_reports(cfg, output_dir, packet_dir, protocol_df, metrics, baselines, ablations, selection, components, operating, sanity, source_df, source_stats, mechanism_matrix, mechanism_stats, best_summary, recovery_stats, stress_cmp, tree_adv_count, feature_top, feature_explanation, hard_stress, hard_stress_expl, external_df, external_text, chosen_device, gpu_df, latency_df, winners, generated_at)
    checks = {
        "07_hard_gate_v5_metrics.csv": validate_metric_file(packet_dir / "07_hard_gate_v5_metrics.csv", DATASET, require_real=True),
        "08_hard_gate_v5_baseline_comparison.csv": validate_metric_file(packet_dir / "08_hard_gate_v5_baseline_comparison.csv", DATASET, require_real=True),
        "13_ablation_matrix_v5.csv": validate_metric_file(packet_dir / "13_ablation_matrix_v5.csv", DATASET, require_real=True),
    }
    if not all(ok for ok, _ in checks.values()):
        raise RuntimeError(f"Packet provenance validation failed: {checks}")
    if any(path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} for path in packet_dir.iterdir()):
        raise RuntimeError("Packet contains image/figure files, which are forbidden in Hard Gate v5.")
    print(f"Hard Gate v5 finished: {status}. Packet files: {len(list(packet_dir.iterdir()))}")


if __name__ == "__main__":
    main()
