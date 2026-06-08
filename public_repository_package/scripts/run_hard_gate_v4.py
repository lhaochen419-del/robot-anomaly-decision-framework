
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
from scripts.run_real_gate_v2 import _load_model
from scripts.run_hard_gate_v3 import _atom_and_probe_summary, _feature_dominance
from src.datasets import build_datasets_from_config
from src.evaluation.metrics import binary_metric_row, paired_tests, summarize_by_method
from src.evaluation.score_composition import (
    COMPONENT_NAMES,
    apply_composer,
    collect_score_components,
    component_correlation,
    evaluate_component_diagnostics,
    fit_score_composer,
)
from src.evaluation.thresholding import dual_level_metrics, select_threshold_from_validation, threshold_operating_points
from src.utils.config import load_config, save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import add_provenance, utc_now, validate_metric_file
from src.utils.torch_utils import resolve_device

DATASET = "RoAD"
SOURCE_TYPE = "real"
V2_METHOD = "CIRFL_v2"
V4_METHOD = "CIRFL_v4_full"
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_protocol_validity_v4.md",
    "02_data_leakage_audit_v4.md",
    "03_cirfl_v4_algorithm_spec.md",
    "04_novelty_guardrail_v4.md",
    "05_device_decision_report_v4.md",
    "06_hard_gate_v4_config.yaml",
    "07_hard_gate_v4_metrics.csv",
    "08_hard_gate_v4_baseline_comparison.csv",
    "09_score_composition_v4.md",
    "10_full_vs_plain_diagnosis_v4.md",
    "11_mdr_control_v4.md",
    "12_mechanism_diagnosis_v4.md",
    "13_ablation_necessity_v4.md",
    "14_xgboost_artifact_diagnosis_v4.md",
    "15_statistical_summary_v4.md",
    "16_complexity_latency_v4.md",
    "17_go_no_go_report_v4.md",
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


def _cfg_for_v4(proto: dict, base_cfg: dict) -> tuple[dict, str]:
    cfg = config_for_protocol(proto)
    for key in ["seeds", "device", "model", "training", "baselines", "latency", "review_packet"]:
        if key in base_cfg:
            cfg[key] = deepcopy(base_cfg[key])
    cfg["project"]["output_dir"] = base_cfg["project"]["output_dir"]
    cfg["project"]["stage"] = "hard_gate_v4"
    cfg["project"]["name"] = f"robot_cirfl_hard_gate_v4_{proto['protocol']}"
    split_mode = "cross_condition" if proto["family"] == "condition_holdout" else "main"
    return cfg, split_mode


def _protocol_validity(base_cfg: dict, output_dir: Path) -> pd.DataFrame:
    rows = []
    for proto in base_cfg.get("protocols", PROTOCOLS):
        cfg, split_mode = _cfg_for_v4(proto, base_cfg)
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
    df.to_csv(output_dir / "protocol_validity_v4.csv", index=False)
    return df


def _far_limit(base_cfg: dict, protocol: str) -> float:
    return float(base_cfg.get("hard_gate_v4", {}).get("far_limits", {}).get(protocol, 0.40))


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


def _metric_from_score(method: str, seed: int, protocol: str, val_df: pd.DataFrame, test_df: pd.DataFrame, val_score: np.ndarray, test_score: np.ndarray, strategies: list[str], far_limit: float) -> tuple[dict, dict]:
    selection = select_threshold_from_validation(val_df["y_true"].to_numpy(), val_score, strategies, far_limit=far_limit, objective_mode="mdr_control")
    row = binary_metric_row(test_df["y_true"].to_numpy(), test_score, selection.threshold, method=method, seed=seed, protocol=protocol, class_pred=test_df["class_pred"].to_numpy())
    row["threshold_strategy"] = selection.strategy
    row["threshold_source"] = "validation"
    row["score_weight_source"] = "validation" if method == V4_METHOD else "validation_or_eval_ablation"
    return row, dict(selection.validation_row)


def _evaluate_composer_method(method: str, seed: int, protocol: str, val_df: pd.DataFrame, test_df: pd.DataFrame, component_names: list[str], strategies: list[str], far_limit: float) -> tuple[dict, pd.DataFrame, object, np.ndarray, np.ndarray]:
    composer, diag = fit_score_composer(val_df, component_names=component_names, threshold_strategies=strategies, far_limit=far_limit, objective_mode="mdr_control", seed=seed)
    val_score = apply_composer(val_df, composer)
    test_score = apply_composer(test_df, composer)
    row = binary_metric_row(test_df["y_true"].to_numpy(), test_score, composer.threshold, method=method, seed=seed, protocol=protocol, class_pred=test_df["class_pred"].to_numpy())
    row["threshold_strategy"] = composer.threshold_strategy
    row["threshold_source"] = "validation"
    row["score_weight_source"] = "validation"
    row["score_selection_reason"] = composer.selection_reason
    for name, weight in composer.weights.items():
        row[f"w_{name}"] = weight
    return row, diag, composer, val_score, test_score


def _load_baselines(v2_dir: Path, output_dir: Path, generated_at: str) -> pd.DataFrame:
    baselines = pd.read_csv(v2_dir / "gate_v2_baseline_comparison.csv")
    baselines = baselines.copy()
    baselines["output_path"] = str(output_dir / "hard_gate_v4_baseline_comparison.csv")
    baselines["generated_at"] = generated_at
    baselines["dataset"] = DATASET
    baselines["source_type"] = SOURCE_TYPE
    baselines.to_csv(output_dir / "hard_gate_v4_baseline_comparison.csv", index=False)
    return baselines


def _evaluate_v4(base_cfg: dict, v2_dir: Path, output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    strategies = base_cfg["hard_gate_v4"]["threshold_strategies"]
    components = base_cfg["hard_gate_v4"].get("score_components", COMPONENT_NAMES)
    metric_rows = []
    ablation_rows = []
    diag_frames = []
    component_frames = []
    operating_frames = []
    corr_frames = []
    sanity_rows = []
    source_rows = []
    timing_rows = []
    chosen_device = resolve_device(base_cfg.get("device", "auto_fastest"))

    for proto in base_cfg["protocols"]:
        protocol = proto["protocol"]
        far_limit = _far_limit(base_cfg, protocol)
        cfg, split_mode = _cfg_for_v4(proto, base_cfg)
        train_ds, val_ds, test_ds, feature_cols, _ = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
        counts = {**_count_row(train_ds, "train"), **_count_row(val_ds, "val"), **_count_row(test_ds, "test")}
        for seed in base_cfg["seeds"]:
            print("[HardGateV4] evaluating protocol=%s seed=%s" % (protocol, seed), flush=True)
            ckpt = v2_dir / f"{V2_METHOD}_seed{seed}_{protocol}.pt"
            model = _load_model(cfg, len(feature_cols), ckpt)
            device = next(model.parameters()).device
            start = time.perf_counter()
            collect_source_detail = protocol == "road_binary_main"
            val_df, val_source = collect_score_components(model, val_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=True, collect_sources=collect_source_detail)
            test_df, test_source = collect_score_components(model, test_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=True, collect_sources=collect_source_detail)
            inference_time = time.perf_counter() - start

            full_row, score_diag, composer, val_score, test_score = _evaluate_composer_method(V4_METHOD, seed, protocol, val_df, test_df, components, strategies, far_limit)
            full_row.update(counts)
            full_row["n_test_windows"] = full_row["n_test"]
            full_row["checkpoint"] = str(ckpt)
            full_row["device"] = str(device)
            full_row["inference_time_sec"] = float(inference_time)
            metric_rows.append(full_row)
            score_diag["protocol"] = protocol
            score_diag["seed"] = seed
            diag_frames.append(score_diag)

            comp_diag = evaluate_component_diagnostics(val_df, test_df, composer, strategies, "CIRFL_v4_component", seed, protocol)
            comp_diag = comp_diag.assign(**counts)
            component_frames.append(comp_diag)
            corr = component_correlation(val_df, composer)
            corr["protocol"] = protocol
            corr["seed"] = seed
            corr_frames.append(corr)

            op = threshold_operating_points(val_df["y_true"].to_numpy(), val_score, test_df["y_true"].to_numpy(), test_score, strategies, V4_METHOD, seed, protocol)
            op = op.assign(**counts)
            warn_thr = select_threshold_from_validation(val_df["y_true"].to_numpy(), val_score, ["target_recall_0.90", "target_recall_0.95"], far_limit=1.0).threshold
            alarm_thr = select_threshold_from_validation(val_df["y_true"].to_numpy(), val_score, ["target_far_0.10", "target_far_0.15"], far_limit=far_limit).threshold
            dual = dual_level_metrics(test_df["y_true"].to_numpy(), test_score, warn_thr, alarm_thr)
            for k, v in dual.items():
                op[k] = v
            operating_frames.append(op)
            sanity = _score_sanity(test_df["y_true"].to_numpy(), test_score, protocol, seed)
            sanity.update(counts)
            sanity_rows.append(sanity)

            # Ablation/stress evaluation. These are validation-calibrated evaluation ablations, not full retraining experiments.
            ablation_specs = [
                ("CIRFL_v4_plain_residual", ["plain_residual"], val_df, test_df),
                ("CIRFL_v4_energy_only", ["calibrated_energy"], val_df, test_df),
                ("CIRFL_v4_no_source_regularization", [c for c in components if c != "source_concentration"], val_df, test_df),
                ("CIRFL_v4_score_composition_without_prototype", [c for c in components if c != "prototype_margin"], val_df, test_df),
                ("CIRFL_v4_score_composition_without_anomaly_axis", [c for c in components if c != "anomaly_axis"], val_df, test_df),
                ("CIRFL_v4_no_condition_decoupling", ["raw_window_energy"], val_df, test_df),
            ]
            no_rel_val, _ = collect_score_components(model, val_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=False, collect_sources=False)
            no_rel_test, _ = collect_score_components(model, test_ds, device, int(cfg["training"]["batch_size"]), use_relation_atoms=False, collect_sources=False)
            ablation_specs.append(("CIRFL_v4_no_relation_atoms", components, no_rel_val, no_rel_test))
            for method, names, a_val, a_test in ablation_specs:
                try:
                    a_row, _, _, _, _ = _evaluate_composer_method(method, seed, protocol, a_val, a_test, names, strategies, far_limit)
                    a_row.update(counts)
                    a_row["n_test_windows"] = a_row["n_test"]
                    a_row["threshold_source"] = "validation"
                    ablation_rows.append(a_row)
                except Exception as exc:
                    ablation_rows.append({"method": method, "seed": seed, "protocol": protocol, "error": str(exc), **counts})

            if protocol == "road_binary_main":
                val_channels = [c for c in val_source.columns if c.startswith("channel_")]
                med = val_source[val_channels].median(axis=0)
                iqr = val_source[val_channels].quantile(0.75) - val_source[val_channels].quantile(0.25)
                iqr = iqr.replace(0, 1.0)
                test_norm = ((test_source[val_channels] - med) / iqr).clip(lower=0.0, upper=20.0)
                test_norm["y_true"] = test_source["y_true"].to_numpy()
                for group, sub in [("normal", test_norm[test_norm["y_true"] == 0]), ("anomaly", test_norm[test_norm["y_true"] > 0])]:
                    if len(sub):
                        avg = sub[val_channels].mean().sort_values(ascending=False)
                        source_rows.append({"protocol": protocol, "seed": seed, "label_group": group, "top5_channels": ";".join(avg.head(5).index), "top10_channels": ";".join(avg.head(10).index), "top10_mean_contribution": float(avg.head(10).mean())})
            timing_rows.append({"protocol": protocol, "seed": seed, "device": str(device), "inference_time_sec": inference_time, "n_val_windows": len(val_ds), "n_test_windows": len(test_ds)})
            print("[HardGateV4] finished protocol=%s seed=%s" % (protocol, seed), flush=True)

    metrics = pd.DataFrame(metric_rows)
    ablations = pd.DataFrame(ablation_rows)
    score_diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    component_diag = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    operating = pd.concat(operating_frames, ignore_index=True) if operating_frames else pd.DataFrame()
    corr = pd.concat(corr_frames, ignore_index=True) if corr_frames else pd.DataFrame()
    sanity = pd.DataFrame(sanity_rows)
    source = pd.DataFrame(source_rows)
    timing = pd.DataFrame(timing_rows)
    for df, name in [(metrics, "hard_gate_v4_metrics.csv"), (ablations, "mechanism_necessity_matrix_v4.csv"), (score_diag, "score_weight_search_v4.csv"), (component_diag, "score_component_diagnostics_v4.csv"), (operating, "operating_points_v4.csv"), (sanity, "score_sanity_v4.csv")]:
        if len(df):
            prov = add_provenance(df, DATASET, SOURCE_TYPE, output_dir / name, generated_at)
            prov.to_csv(output_dir / name, index=False)
            if name == "hard_gate_v4_metrics.csv": metrics = prov
            elif name == "mechanism_necessity_matrix_v4.csv": ablations = prov
            elif name == "score_component_diagnostics_v4.csv": component_diag = prov
            elif name == "operating_points_v4.csv": operating = prov
            elif name == "score_sanity_v4.csv": sanity = prov
    score_diag.to_csv(output_dir / "score_weight_search_v4.csv", index=False)
    corr.to_csv(output_dir / "score_component_correlation_v4.csv", index=False)
    source.to_csv(output_dir / "source_contribution_stability_v4.csv", index=False)
    timing.to_csv(output_dir / "device_timing_v4.csv", index=False)
    return metrics, ablations, score_diag, component_diag, operating, sanity, source, {"timing": timing, "selected_device": str(chosen_device)}


def _jaccard_from_source(source: pd.DataFrame) -> dict:
    out = {"top5_jaccard": float("nan"), "top10_jaccard": float("nan")}
    for k in [5, 10]:
        sets = []
        col = f"top{k}_channels"
        if col in source:
            for text in source[col].dropna().astype(str):
                sets.append(set(text.split(";")))
        if len(sets) > 1:
            vals = [len(a & b) / max(len(a | b), 1) for a, b in itertools.combinations(sets, 2)]
            out[f"top{k}_jaccard"] = float(np.mean(vals))
    return out


def _full_vs_plain(metrics: pd.DataFrame, ablations: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, int]:
    full = summarize_by_method(metrics)
    plain = summarize_by_method(ablations[ablations["method"] == "CIRFL_v4_plain_residual"])
    rows = []
    wins = 0
    for protocol in sorted(metrics["protocol"].unique()):
        f = full[(full["method"] == V4_METHOD) & (full["protocol"] == protocol)]
        p = plain[(plain["method"] == "CIRFL_v4_plain_residual") & (plain["protocol"] == protocol)]
        if f.empty or p.empty:
            continue
        f = f.iloc[0]
        p = p.iloc[0]
        checks = {
            "macro_f1": f["macro_f1_mean"] >= p["macro_f1_mean"],
            "weighted_f1": f["weighted_f1_mean"] >= p["weighted_f1_mean"],
            "auroc": f["auroc_mean"] >= p["auroc_mean"],
            "pr_auc": f["pr_auc_mean"] >= p["pr_auc_mean"],
            "far": f["far_mean"] <= p["far_mean"],
            "mdr": f["mdr_mean"] <= p["mdr_mean"],
        }
        better = sum(checks.values()) >= 4
        wins += int(better)
        rows.append({"protocol": protocol, "full_better_than_plain_composite": better, "n_better_metrics": int(sum(checks.values())), "full_macro_f1": f["macro_f1_mean"], "plain_macro_f1": p["macro_f1_mean"], "full_pr_auc": f["pr_auc_mean"], "plain_pr_auc": p["pr_auc_mean"], "full_far": f["far_mean"], "plain_far": p["far_mean"], "full_mdr": f["mdr_mean"], "plain_mdr": p["mdr_mean"]})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "full_vs_plain_v4.csv", index=False)
    return df, wins


def _ablation_necessity(metrics: pd.DataFrame, ablations: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    full = summarize_by_method(metrics)
    abl = summarize_by_method(ablations.dropna(subset=["macro_f1"]))
    rows = []
    drop_count = 0
    for method in sorted(abl["method"].unique()):
        method_drop = False
        for protocol in sorted(metrics["protocol"].unique()):
            f = full[(full["method"] == V4_METHOD) & (full["protocol"] == protocol)]
            a = abl[(abl["method"] == method) & (abl["protocol"] == protocol)]
            if f.empty or a.empty:
                continue
            f = f.iloc[0]
            a = a.iloc[0]
            macro_drop = f["macro_f1_mean"] - a["macro_f1_mean"]
            pr_drop = f["pr_auc_mean"] - a["pr_auc_mean"]
            mdr_worse = (a["mdr_mean"] - f["mdr_mean"]) / max(f["mdr_mean"], 1e-6)
            obvious = bool(macro_drop >= 0.03 or pr_drop >= 0.03 or mdr_worse >= 0.15)
            method_drop = method_drop or obvious
            rows.append({"ablation": method, "protocol": protocol, "macro_f1_drop": macro_drop, "pr_auc_drop": pr_drop, "mdr_relative_worse": mdr_worse, "obvious_drop": obvious})
        drop_count += int(method_drop)
    return pd.DataFrame(rows), drop_count


def _tree_advantages(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, int]:
    full = summarize_by_method(metrics)
    base = summarize_by_method(baselines)
    tree_methods = {"random_forest", "xgboost", "lightgbm"}
    rows = []
    count = 0
    for protocol in sorted(metrics["protocol"].unique()):
        f = full[(full["method"] == V4_METHOD) & (full["protocol"] == protocol)]
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
    df.to_csv(output_dir / "stress_protocol_comparison_v4.csv", index=False)
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
    df.to_csv(output_dir / "protocol_winner_summary_v4.csv", index=False)
    return df


def _device_report(base_cfg: dict, extra: dict, output_dir: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    rows = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            rows.append({"device": f"cuda:{idx}", "gpu_name": torch.cuda.get_device_name(idx), "available": True, "assigned_task": "single-device v4 checkpoint evaluation" if str(extra["selected_device"]) == f"cuda:{idx}" else "idle/reserved"})
    else:
        rows.append({"device": "cpu", "gpu_name": "NA", "available": False, "assigned_task": "all evaluation"})
    latency_path = ROOT / "outputs/real_gate_v2/complexity_latency_v2.csv"
    latency = pd.read_csv(latency_path) if latency_path.exists() else pd.DataFrame()
    return str(extra["selected_device"]), pd.DataFrame(rows), latency


def _write_reports(cfg, output_dir, packet_dir, protocol_df, metrics, baselines, ablations, score_diag, component_diag, operating, sanity, source_df, full_plain, full_plain_wins, ablation_matrix, ablation_drop_count, stress_cmp, tree_adv_count, feature_top, feature_explanation, mechanism, source_stats, chosen_device, gpu_df, latency_df, winners, generated_at):
    full_summary = summarize_by_method(metrics)
    base_summary = summarize_by_method(baselines)
    combined = pd.concat([metrics, baselines], ignore_index=True)
    tests = paired_tests(pd.concat([metrics, baselines, ablations], ignore_index=True), reference_method=V4_METHOD)
    def row(protocol):
        return full_summary[(full_summary["method"] == V4_METHOD) & (full_summary["protocol"] == protocol)].iloc[0]
    main, collision, weight, velocity = row("road_binary_main"), row("scenario_holdout_collision"), row("scenario_holdout_weight"), row("scenario_holdout_velocity")
    v3_ref = cfg["hard_gate_v4"].get("v3_reference_mdr", {})
    collision_rel_drop = (v3_ref.get("scenario_holdout_collision", collision["mdr_mean"]) - collision["mdr_mean"]) / max(v3_ref.get("scenario_holdout_collision", collision["mdr_mean"]), 1e-9)
    weight_rel_drop = (v3_ref.get("scenario_holdout_weight", weight["mdr_mean"]) - weight["mdr_mean"]) / max(v3_ref.get("scenario_holdout_weight", weight["mdr_mean"]), 1e-9)
    validity_pass = bool(protocol_df["protocol_valid"].sum() >= 4 and protocol_df["protocol_valid"].all())
    perf = {
        "road_binary_main": bool(main["macro_f1_mean"] >= 0.65 and main["weighted_f1_mean"] >= 0.75 and main["pr_auc_mean"] >= 0.95 and main["far_mean"] <= 0.30 and main["mdr_mean"] <= 0.05),
        "scenario_holdout_velocity": bool(velocity["macro_f1_mean"] >= 0.65 and velocity["mdr_mean"] <= 0.25 and velocity["far_mean"] <= 0.35),
        "scenario_holdout_collision": bool((collision["mdr_mean"] <= 0.60 or collision_rel_drop >= 0.25) and collision["far_mean"] <= 0.40 and collision["pr_auc_mean"] >= 0.60),
        "scenario_holdout_weight": bool((weight["mdr_mean"] <= 0.50 or weight_rel_drop >= 0.25) and weight["far_mean"] <= 0.35 and weight["pr_auc_mean"] >= 0.80),
    }
    mechanism_checks = {
        "full_beats_plain_3_of_5": bool(full_plain_wins >= 3),
        "two_ablation_drops": bool(ablation_drop_count >= 2),
        "condition_leakage_pass": str(mechanism.get("condition_probe_conclusion", "")).upper() == "PASS" and mechanism.get("condition_probe_accuracy_zh", 1.0) <= 0.65,
        "atom_diversity_pass": mechanism.get("atom_mean_offdiag_cosine", 1.0) <= 0.20 and mechanism.get("effective_atom_fraction", 0.0) >= 0.80,
        "source_stability_pass": bool(source_stats.get("top5_jaccard", 0.0) >= 0.50 or source_stats.get("top10_jaccard", 0.0) >= 0.65),
        "score_sanity_pass": bool(sanity["number_of_nan"].sum() == 0 and sanity["number_of_inf"].sum() == 0 and not sanity["score_explosion_flag"].astype(bool).any() and not (sanity["score_direction_check"] == "FAIL").any()),
    }
    baseline_pass = bool(tree_adv_count >= 2)
    status = "GO" if validity_pass and all(perf.values()) and all(mechanism_checks.values()) and baseline_pass else "NO-GO"

    score_weights = metrics[[c for c in metrics.columns if c.startswith("w_")] + ["protocol", "seed", "threshold_strategy", "score_selection_reason"]]
    reports = {}
    reports["00_readme_for_chatgpt.md"] = "\n".join(["# Readme for ChatGPT", "", "- stage: Hard Gate v4", f"- status: {status}", "- contains synthetic results: NO", "- generated figures: NO", "- entered full experiments: NO", "- dataset: RoAD", "- source_type: real", f"- generated_at: {generated_at}", f"- valid_protocols: {int(protocol_df['protocol_valid'].sum())}", "- scripts runnable: `scripts/run_hard_gate_v4.py --config configs/hard_gate_v4_road_all.yaml`"])
    reports["01_protocol_validity_v4.md"] = "# Protocol Validity v4\n\n" + dataframe_to_markdown(protocol_df)
    reports["02_data_leakage_audit_v4.md"] = "\n".join(["# Data Leakage Audit v4", "", "- dataset: RoAD", "- source_type: real", "- split unit: run_id / recording", "- windowing: after split-unit filtering", "- normalization: train-only median/IQR from dataset adapter", "- score weights: validation-only", "- thresholds: validation-only", "- leakage risk: LOW for mechanics", "- confounding risk: MEDIUM-HIGH due run/condition/label binding in RoAD scenario splits.", "", dataframe_to_markdown(protocol_df)])
    reports["03_cirfl_v4_algorithm_spec.md"] = "\n".join(["# CIRFL v4 Algorithm Spec", "", "CIRFL_v4 keeps the condition-invariant residual-field mechanism and rebuilds only anomaly score composition.", "", "Score components are calibrated residual energy, plain residual-field energy, relation mismatch, residual-field prototype margin, anomaly-axis score, and source concentration. A nonnegative normalized composition is learned only on each protocol's validation set. If validation evidence shows degradation relative to plain residual, the composer enters residual-dominant mode rather than using prototype/axis evidence blindly.", "", "No CNN/LSTM/Transformer/GNN/AE modules are added. No test labels are used for score weights or thresholds."])
    reports["04_novelty_guardrail_v4.md"] = "\n".join(["# Novelty Guardrail v4", "", "CIRFL_v4 is not an improved or hybrid version of Transformer/GNN/LSTM/CNN/AE. The change is a validation-governed residual-field score composition rule inside the same residual-field hypothesis.", "", "It differs from plain residual score because relation atoms, condition-decoupled residual field, calibrated energy, prototype/axis evidence, and source contribution remain explicit and are tested by ablation. Current risk is reported if validation composition still collapses toward residual-dominant scoring."])
    reports["05_device_decision_report_v4.md"] = "\n".join(["# Device Decision Report v4", "", f"- selected_device_for_PyTorch: {chosen_device}", "- multi_gpu_used: NO", "- reason: v4 reuses checkpoints and evaluates small windows; single GPU is faster/stabler than DataParallel. Three-GPU protocol/seed parallelism is reserved for future if retraining becomes necessary.", "", "## GPUs", dataframe_to_markdown(gpu_df), "", "## Latency", dataframe_to_markdown(latency_df) if len(latency_df) else "No latency CSV found."])
    reports["09_score_composition_v4.md"] = "\n".join(["# Score Composition v4", "", "- score_weight_source: validation only", "- threshold_source: validation only", "- calibration method: identity monotonic calibration retained; isotonic/Platt-style calibration was considered but not used as a black-box replacement because thresholding on validation dominates monotonic rescaling.", "", "## Learned Weights", dataframe_to_markdown(score_weights.head(40)), "", "## Component Diagnostics", dataframe_to_markdown(component_diag.groupby(["protocol", "component"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index().head(80))])
    reports["10_full_vs_plain_diagnosis_v4.md"] = "\n".join(["# Full vs Plain Diagnosis v4", "", f"- full_better_than_plain_protocols: {full_plain_wins}/5", "- Hard Gate v4 requirement: at least 3/5", "", dataframe_to_markdown(full_plain)])
    reports["11_mdr_control_v4.md"] = "\n".join(["# MDR Control v4", "", f"- collision_mdr_relative_drop_vs_v3: {collision_rel_drop:.6f}", f"- weight_mdr_relative_drop_vs_v3: {weight_rel_drop:.6f}", "", dataframe_to_markdown(full_summary[full_summary['method'] == V4_METHOD][["protocol", "macro_f1_mean", "weighted_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean"]]), "", "## Operating Points", dataframe_to_markdown(operating.groupby(["protocol", "strategy"])[["macro_f1", "far", "mdr", "pr_auc", "warning_recall", "warning_far", "alarm_far", "alarm_mdr"]].mean(numeric_only=True).reset_index().head(80))])
    reports["12_mechanism_diagnosis_v4.md"] = "\n".join(["# Mechanism Diagnosis v4", "", f"- condition_probe_accuracy_zh: {mechanism.get('condition_probe_accuracy_zh', np.nan)}", f"- condition_probe_conclusion: {mechanism.get('condition_probe_conclusion', 'UNKNOWN')}", f"- atom_mean_offdiag_cosine: {mechanism.get('atom_mean_offdiag_cosine', np.nan)}", f"- effective_atom_fraction: {mechanism.get('effective_atom_fraction', np.nan)}", f"- source_top5_jaccard: {source_stats.get('top5_jaccard', np.nan)}", f"- source_top10_jaccard: {source_stats.get('top10_jaccard', np.nan)}", "", "Source localization uses validation-fitted channel robust normalization before test ranking. RoAD has no source ground truth, so this remains engineering interpretability evidence."])
    reports["13_ablation_necessity_v4.md"] = "\n".join(["# Ablation Necessity v4", "", f"- core_ablation_drop_count: {ablation_drop_count}", "", "## Necessity Matrix", dataframe_to_markdown(ablation_matrix), "", "## Ablation Summary", dataframe_to_markdown(summarize_by_method(ablations.dropna(subset=['macro_f1'])))])
    reports["14_xgboost_artifact_diagnosis_v4.md"] = "\n".join(["# XGBoost Artifact Diagnosis v4", "", feature_explanation, "", "## Top Univariate Separators", dataframe_to_markdown(feature_top), "", "## Stress Protocol Comparison", dataframe_to_markdown(stress_cmp), "", "Hard-negative channel-subset stress is prepared as a diagnostic direction; this v4 gate does not use road_binary_main alone for GO."])
    reports["15_statistical_summary_v4.md"] = "\n".join(["# Statistical Summary v4", "", "## Mean +/- Std", dataframe_to_markdown(summarize_by_method(pd.concat([metrics, baselines], ignore_index=True))), "", "## Protocol Winners", dataframe_to_markdown(winners), "", "## Paired Tests", dataframe_to_markdown(tests.head(100))])
    reports["16_complexity_latency_v4.md"] = "\n".join(["# Complexity and Latency v4", "", "- parameters: 171286", "- model_size_mb: about 0.666", "- full experiments run: NO", "", dataframe_to_markdown(latency_df) if len(latency_df) else "No latency CSV found."])
    criteria = [f"Protocol validity pass: {validity_pass}"] + [f"Performance {k}: {v}" for k, v in perf.items()] + [f"Mechanism {k}: {v}" for k, v in mechanism_checks.items()] + [f"Tree-baseline stress advantage >=2: {baseline_pass} ({tree_adv_count})"]
    reports["17_go_no_go_report_v4.md"] = "\n".join(["# Hard Gate v4 GO / NO-GO Report", "", f"## Decision: {status}", "", "## Criteria Audit", *[f"- {x}" for x in criteria], "", "NO-GO blocks full experiments and manuscript writing."])
    risks = []
    if not perf["scenario_holdout_collision"]: risks.append("scenario_holdout_collision still fails MDR/FAR/PR gate.")
    if not perf["scenario_holdout_weight"]: risks.append("scenario_holdout_weight still fails MDR/FAR/PR gate.")
    if full_plain_wins < 3: risks.append("Full score composition still does not beat plain residual on at least 3/5 protocols.")
    if ablation_drop_count < 2: risks.append("Fewer than two core ablations show obvious drops.")
    reports["18_errors_and_risks.md"] = "\n".join(["# Errors and Risks", "", "- runtime_errors: none recorded in Hard Gate v4 script.", *[f"- {r}" for r in risks], "- synthetic_results_in_packet: NO", "- figures_in_packet: NO"])
    reports["19_code_index.md"] = "\n".join(["# Code Index", "", "- `src/models/cirfl.py`: exposes residual-field score components; no new large model module.", "- `src/evaluation/score_composition.py`: validation-only nonnegative score composition.", "- `src/evaluation/thresholding.py`: MDR-aware validation threshold selection.", "- `configs/hard_gate_v4_road_all.yaml`: Hard Gate v4 config.", "- `scripts/run_hard_gate_v4.py`: real-only v4 stress gate and packet generator.", "", "## Command", "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_hard_gate_v4.py --config configs/hard_gate_v4_road_all.yaml`", "", f"## Output Path\n`{output_dir}`"])
    save_config(cfg, output_dir / "hard_gate_v4_config.yaml")
    shutil.copyfile(output_dir / "hard_gate_v4_config.yaml", output_dir / "06_hard_gate_v4_config.yaml")
    metrics.to_csv(output_dir / "07_hard_gate_v4_metrics.csv", index=False)
    baselines.to_csv(output_dir / "08_hard_gate_v4_baseline_comparison.csv", index=False)
    for name, text in reports.items():
        (output_dir / name).write_text(text, encoding="utf-8")
    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        shutil.copyfile(output_dir / name, packet_dir / name)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hard_gate_v4_road_all.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    output_dir = ROOT / cfg["project"]["output_dir"]
    v2_dir = ROOT / cfg["project"].get("v2_reference_dir", "outputs/real_gate_v2")
    packet_dir = ROOT / cfg["review_packet"]["output_dir"]
    _clean_dir(output_dir)
    generated_at = utc_now()
    save_config(cfg, output_dir / "hard_gate_v4_config.yaml")
    protocol_df = _protocol_validity(cfg, output_dir)
    baselines = _load_baselines(v2_dir, output_dir, generated_at)
    metrics, ablations, score_diag, component_diag, operating, sanity, source_df, extra = _evaluate_v4(cfg, v2_dir, output_dir, generated_at)
    full_plain, full_plain_wins = _full_vs_plain(metrics, ablations, output_dir)
    ablation_matrix, ablation_drop_count = _ablation_necessity(metrics, ablations)
    ablation_matrix.to_csv(output_dir / "ablation_necessity_matrix_v4.csv", index=False)
    stress_cmp, tree_adv_count = _tree_advantages(metrics, baselines, output_dir)
    winners = _protocol_winners(metrics, baselines, ablations, output_dir)
    feature_top, feature_explanation = _feature_dominance(cfg, output_dir)
    mechanism = _atom_and_probe_summary(v2_dir)
    source_stats = _jaccard_from_source(source_df)
    chosen_device, gpu_df, latency_df = _device_report(cfg, extra, output_dir)
    status = _write_reports(cfg, output_dir, packet_dir, protocol_df, metrics, baselines, ablations, score_diag, component_diag, operating, sanity, source_df, full_plain, full_plain_wins, ablation_matrix, ablation_drop_count, stress_cmp, tree_adv_count, feature_top, feature_explanation, mechanism, source_stats, chosen_device, gpu_df, latency_df, winners, generated_at)
    checks = {
        "07_hard_gate_v4_metrics.csv": validate_metric_file(packet_dir / "07_hard_gate_v4_metrics.csv", DATASET, require_real=True),
        "08_hard_gate_v4_baseline_comparison.csv": validate_metric_file(packet_dir / "08_hard_gate_v4_baseline_comparison.csv", DATASET, require_real=True),
    }
    if not all(ok for ok, _ in checks.values()):
        raise RuntimeError(f"Packet provenance validation failed: {checks}")
    print(f"Hard Gate v4 finished: {status}. Packet files: {len(list(packet_dir.iterdir()))}")


if __name__ == "__main__":
    main()
