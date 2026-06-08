from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.adapters.imadds_raw import build_imadds_raw_windows, load_imadds_attributes
from src.datasets.imadds_raw_window import load_raw_windows
from src.evaluation.metrics import binary_metric_row, score_direction_multiplier, summarize_by_method, paired_tests
from src.evaluation.cetra_metrics import source_stability
from src.models.cetra import CETRA
from src.utils.config import save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import utc_now
from src.utils.torch_utils import resolve_device

SEEDS = [7, 13, 23]
STAGE = "CETRA Gate v1"
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_old_direction_stop_report.md",
    "02_cetra_algorithm_spec.md",
    "03_cetra_novelty_guardrail.md",
    "04_cetra_protocol_validity.md",
    "05_cetra_gate_config.yaml",
    "06_cetra_gate_metrics.csv",
    "07_cetra_baseline_comparison.csv",
    "08_cetra_ablation_report.md",
    "09_cetra_conformal_calibration.md",
    "10_cetra_source_localization.md",
    "11_cetra_robustness.md",
    "12_cetra_statistical_summary.md",
    "13_cetra_vs_tree_analysis.md",
    "14_device_decision_report.md",
    "15_complexity_latency_cetra.md",
    "16_cetra_go_no_go_report.md",
    "17_errors_and_risks.md",
    "18_code_index.md",
    "19_next_tasks_for_codex.md",
]


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _prov(protocol: str, seed: Any, output_dir: Path, generated_at: str) -> str:
    return "\n".join([
        "- dataset: IMAD-DS RoboticArm raw windows + RoAD frozen reference",
        "- source_type: external_real / real_reference",
        f"- protocol: {protocol}",
        f"- seed: {seed}",
        "- n_train/n_val/n_test: see CSV columns",
        "- n_test_normal/n_test_anomaly: see CSV columns",
        f"- generated_at: {generated_at}",
        f"- output_path: {output_dir}",
    ])


def _write_text(path: Path, title: str, provenance: str, body: list[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", provenance, "", *body]), encoding="utf-8")


def _segment_table(meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["segment_uid", "split_stage", "split_label", "domain_label", "domain_shift_op", "domain_shift_env", "condition_name", "condition_id", "fault_label", "fault_label_name"]
    return meta[cols].drop_duplicates("segment_uid").reset_index(drop=True)


def _split_segments(seg: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    a: list[str] = []
    b: list[str] = []
    c: list[str] = []
    for _, group in seg.groupby(["fault_label", "domain_label"], dropna=False):
        ids = group["segment_uid"].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_a = max(1, int(round(ratios[0] * n))) if n >= 3 else max(0, n - 2)
        n_b = max(1, int(round(ratios[1] * n))) if n - n_a >= 2 else max(0, n - n_a - 1)
        a.extend(ids[:n_a]); b.extend(ids[n_a:n_a+n_b]); c.extend(ids[n_a+n_b:])
    return set(a), set(b), set(c)


def _build_splits(meta: pd.DataFrame, seed: int) -> dict[str, dict[str, set[str]]]:
    seg = _segment_table(meta)
    test_seg = seg[seg["split_stage"] == "test"].copy()
    train_main = set(seg[(seg["split_stage"] == "train") & (seg["fault_label"] == 0)]["segment_uid"].astype(str))
    calib, val, test = _split_segments(test_seg, seed, (0.34, 0.33, 0.33))
    splits = {"imadds_raw_main_binary": {"train": train_main, "calib": calib, "val": val, "test": test}}
    source_train = set(seg[(seg["split_stage"] == "train") & (seg["domain_label"] == "source") & (seg["fault_label"] == 0)]["segment_uid"].astype(str))
    source_test = test_seg[test_seg["domain_label"] == "source"].copy()
    target_test = test_seg[test_seg["domain_label"] == "target"].copy()
    calib_s, val_s, _ = _split_segments(source_test, seed, (0.50, 0.50, 0.0))
    splits["imadds_raw_source_to_target"] = {"train": source_train, "calib": calib_s, "val": val_s, "test": set(target_test["segment_uid"].astype(str))}
    held = test_seg[(test_seg["domain_label"] == "target") & (test_seg["domain_shift_op"] == "weight35")].copy()
    rest = test_seg.drop(index=held.index).copy()
    calib_r, val_r, _ = _split_segments(rest, seed, (0.50, 0.50, 0.0))
    splits["imadds_raw_leave_target_weight35_out"] = {"train": train_main, "calib": calib_r, "val": val_r, "test": set(held["segment_uid"].astype(str))}
    splits["imadds_raw_sensor_missing_10"] = splits["imadds_raw_main_binary"].copy()
    splits["imadds_raw_sensor_missing_20"] = splits["imadds_raw_main_binary"].copy()
    return splits


def _idx(meta: pd.DataFrame, segs: set[str]) -> np.ndarray:
    return np.where(meta["segment_uid"].astype(str).isin(segs).to_numpy())[0]


def _validity(meta: pd.DataFrame, output_dir: Path, generated_at: str) -> pd.DataFrame:
    rows = []
    splits = _build_splits(meta, 7)
    for protocol, parts in splits.items():
        train_idx, val_idx, test_idx = _idx(meta, parts["train"]), _idx(meta, parts["val"]), _idx(meta, parts["test"])
        yv = meta.iloc[val_idx]["fault_label"].to_numpy(int) if len(val_idx) else np.array([])
        yt = meta.iloc[test_idx]["fault_label"].to_numpy(int) if len(test_idx) else np.array([])
        valid = len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0 and len(np.unique(yv)) == 2 and len(np.unique(yt)) == 2
        rows.append({
            "dataset": "IMAD-DS RoboticArm raw windows",
            "source_type": "external_real",
            "protocol": protocol,
            "validity": "VALID" if valid else "INVALID",
            "split_unit": "segment_uid/file_id; windows inherit segment split",
            "n_train_windows": int(len(train_idx)),
            "n_val_windows": int(len(val_idx)),
            "n_test_windows": int(len(test_idx)),
            "n_test_normal": int((yt == 0).sum()) if len(yt) else 0,
            "n_test_anomaly": int((yt == 1).sum()) if len(yt) else 0,
            "leakage_risk": "LOW-MEDIUM_PILOT" if valid else "INVALID",
            "threshold_source": "validation/calibration only",
            "transport_fit_source": "train normal only",
            "generated_at": generated_at,
            "output_path": str(output_dir / "cetra_protocol_validity_v1.csv"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "cetra_protocol_validity_v1.csv", index=False)
    return df


def _mask_missing(x: np.ndarray, rate: float, seed: int) -> np.ndarray:
    if rate <= 0:
        return x
    out = x.copy()
    rng = np.random.default_rng(seed)
    n_channels = out.shape[-1]
    n_drop = max(1, int(round(n_channels * rate)))
    drop = rng.choice(n_channels, size=min(n_drop, n_channels), replace=False)
    out[:, :, drop] = 0.0
    return out


def _fit_score_variant(method: str, x_train: np.ndarray, cond_train: np.ndarray, x_val: np.ndarray, cond_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, cond_test: np.ndarray, y_test: np.ndarray, channel_names: list[str], seed: int, protocol: str, output_dir: Path, generated_at: str, missing_rate: float = 0.0) -> tuple[dict, pd.DataFrame, dict]:
    use_transport = method != "CETRA_no_condition_transport"
    include_cross = method not in {"CETRA_no_cross_sensor_relations", "CETRA_univariate_event_only"}
    include_lag = method != "CETRA_no_lag_relations" and method != "CETRA_univariate_event_only"
    univariate = method == "CETRA_univariate_event_only"
    conformal = method != "CETRA_no_conformal_calibration"
    model = CETRA(use_condition_transport=use_transport, include_cross_sensor=include_cross, include_lag=include_lag, univariate_only=univariate, threshold_strategy="best_f1")
    t0 = time.perf_counter()
    model.fit(x_train, cond_train, channel_names=channel_names)
    train_sec = time.perf_counter() - t0
    x_val_eval = _mask_missing(x_val, missing_rate, seed + 101)
    x_test_eval = _mask_missing(x_test, missing_rate, seed + 202)
    t1 = time.perf_counter()
    s_val, contrib_val, notes_val, _ = model.score(x_val_eval, cond_val)
    s_test, contrib_test, notes_test, _ = model.score(x_test_eval, cond_test)
    # Score orientation is selected from validation only.
    mult = score_direction_multiplier(y_val, s_val)
    s_val = s_val * mult
    s_test = s_test * mult
    model.calibrate(y_val, s_val, conformal=conformal)
    p_test = model.p_values(s_test)
    infer_sec = time.perf_counter() - t1
    row = binary_metric_row(y_test, s_test, model.threshold(), method, seed, protocol)
    normal_calib = int((y_val == 0).sum())
    pred = (s_test >= model.threshold()).astype(int)
    row.update({
        "dataset": "IMAD-DS RoboticArm raw windows",
        "source_type": "external_real",
        "stage": "CETRA_GATE_V1",
        "status": "RUN_OK",
        "n_train_windows": int(len(x_train)),
        "n_val_windows": int(len(x_val)),
        "n_test_windows": int(len(x_test)),
        "model_size_mb": float(model.automaton_size_mb()),
        "automaton_size_mb": float(model.automaton_size_mb()),
        "cpu_latency_ms": float(infer_sec / max(len(x_test), 1) * 1000),
        "gpu_latency_ms": np.nan,
        "train_time_sec": float(train_sec),
        "inference_time_sec": float(infer_sec),
        "calibration_set_size": normal_calib,
        "threshold_source": "validation_only" if conformal else "train_normal_quantile_no_conformal_ablation",
        "conformal_coverage_normal": float(np.mean(p_test[y_test == 0] > 0.05)) if np.any(y_test == 0) else np.nan,
        "missing_channel_rate": float(missing_rate),
        "generated_at": generated_at,
        "output_path": str(output_dir),
    })
    source_rows = []
    names = channel_names or [f"sensor_{i}" for i in range(contrib_test.shape[1])]
    for i in range(len(x_test)):
        order = np.argsort(-contrib_test[i])
        top = [names[j] for j in order[: min(3, len(order))]]
        source_rows.append({
            "dataset": "IMAD-DS RoboticArm raw windows",
            "source_type": "external_real",
            "protocol": protocol,
            "seed": seed,
            "method": method,
            "window_rank": i,
            "y_true": int(y_test[i]),
            "predicted_label": int(pred[i]),
            "anomaly_score": float(s_test[i]),
            "conformal_p_value": float(p_test[i]) if np.isfinite(p_test[i]) else np.nan,
            "top_sensors": ";".join(top),
            "top_event_transition": notes_test[i],
            "max_sensor_contribution": float(np.max(contrib_test[i])),
            "contribution_entropy": float(_entropy(contrib_test[i])),
            "generated_at": generated_at,
            "output_path": str(output_dir / "cetra_source_localization_v1.csv"),
        })
    diag = {
        "threshold": float(model.threshold()),
        "score_val_mean": float(np.mean(s_val)),
        "score_test_mean": float(np.mean(s_test)),
        "p_value_test_mean": float(np.nanmean(p_test)),
        "normal_calibration_size": normal_calib,
        "automaton_size_mb": float(model.automaton_size_mb()),
    }
    # Save full models only; review packet excludes checkpoints.
    if method == "CETRA_full" and protocol in {"imadds_raw_main_binary", "imadds_raw_source_to_target", "imadds_raw_leave_target_weight35_out"}:
        model.save(output_dir / "automata" / f"{protocol}_seed{seed}_cetra_full.pkl")
    return row, pd.DataFrame(source_rows), diag


def _entropy(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = np.maximum(v, 0)
    s = v.sum()
    if s <= 0:
        return 0.0
    p = v / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _run_cetra(data, cfg: dict, output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core_protocols = cfg["protocols"]["core"]
    robustness_protocols = cfg["protocols"].get("robustness", [])
    missing_rates = cfg["protocols"].get("missing_channel_rates", {})
    methods = [
        "CETRA_full",
        "CETRA_no_condition_transport",
        "CETRA_no_cross_sensor_relations",
        "CETRA_no_lag_relations",
        "CETRA_no_conformal_calibration",
        "CETRA_univariate_event_only",
    ]
    rows = []
    source_rows = []
    calibration_rows = []
    for seed in SEEDS:
        splits = _build_splits(data.meta, seed)
        for protocol in core_protocols + robustness_protocols:
            base_protocol = "imadds_raw_main_binary" if protocol in robustness_protocols else protocol
            parts = splits[base_protocol]
            train_idx, val_idx, test_idx = _idx(data.meta, parts["train"]), _idx(data.meta, parts["val"]), _idx(data.meta, parts["test"])
            train_meta, val_meta, test_meta = data.meta.iloc[train_idx].reset_index(drop=True), data.meta.iloc[val_idx].reset_index(drop=True), data.meta.iloc[test_idx].reset_index(drop=True)
            normal_train = train_meta["fault_label"].to_numpy(int) == 0
            x_train = data.x[train_idx][normal_train]
            cond_train = train_meta.loc[normal_train, "condition_name"].astype(str).to_numpy()
            x_val, x_test = data.x[val_idx], data.x[test_idx]
            cond_val, cond_test = val_meta["condition_name"].astype(str).to_numpy(), test_meta["condition_name"].astype(str).to_numpy()
            y_val, y_test = val_meta["fault_label"].to_numpy(int), test_meta["fault_label"].to_numpy(int)
            if len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                continue
            missing_rate = float(missing_rates.get(protocol, 0.0))
            run_methods = ["CETRA_full"] if protocol in robustness_protocols else methods
            for method in run_methods:
                row, src_df, diag = _fit_score_variant(method, x_train, cond_train, x_val, cond_val, y_val, x_test, cond_test, y_test, data.channel_names, seed, protocol, output_dir, generated_at, missing_rate=missing_rate)
                rows.append(row)
                if method == "CETRA_full":
                    source_rows.append(src_df)
                calibration_rows.append({"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "protocol": protocol, "seed": seed, "method": method, **diag, "threshold_source": row["threshold_source"], "n_train_windows": row["n_train_windows"], "n_val_windows": row["n_val_windows"], "n_test_windows": row["n_test_windows"], "n_test_normal": row["n_test_normal"], "n_test_anomaly": row["n_test_anomaly"], "generated_at": generated_at, "output_path": str(output_dir / "cetra_calibration_v1.csv")})
    metrics = pd.DataFrame(rows)
    source = pd.concat(source_rows, ignore_index=True) if source_rows else pd.DataFrame()
    calib = pd.DataFrame(calibration_rows)
    stability = source_stability(source) if len(source) else pd.DataFrame()
    metrics.to_csv(output_dir / "cetra_gate_metrics_all_methods.csv", index=False)
    metrics[metrics["method"] == "CETRA_full"].to_csv(output_dir / "06_cetra_gate_metrics.csv", index=False)
    source.to_csv(output_dir / "cetra_source_localization_v1.csv", index=False)
    calib.to_csv(output_dir / "cetra_calibration_v1.csv", index=False)
    stability.to_csv(output_dir / "cetra_source_stability_v1.csv", index=False)
    return metrics, source, calib, stability


def _baseline_comparison(output_dir: Path, generated_at: str) -> pd.DataFrame:
    frames = []
    for p in [ROOT / "outputs/raw_window_external_gate_v3/07_raw_window_gate_metrics.csv", ROOT / "outputs/raw_window_external_gate_v3/08_raw_window_baseline_comparison.csv"]:
        if p.exists():
            df = pd.read_csv(p)
            df["comparison_source"] = p.as_posix()
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["generated_at"] = generated_at
    out["output_path"] = str(output_dir / "07_cetra_baseline_comparison.csv")
    out.to_csv(output_dir / "07_cetra_baseline_comparison.csv", index=False)
    return out


def _winner_summary(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path, generated_at: str) -> pd.DataFrame:
    comp = pd.concat([metrics[metrics["method"] == "CETRA_full"], baselines], ignore_index=True)
    comp = comp[comp.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
    rows = []
    for protocol, group in comp.groupby("protocol"):
        for metric, maximize in [("macro_f1", True), ("pr_auc", True), ("mdr", False), ("far", False)]:
            good = group.dropna(subset=[metric])
            if good.empty:
                continue
            idx = good[metric].idxmax() if maximize else good[metric].idxmin()
            item = good.loc[idx]
            rows.append({"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "protocol": protocol, "metric": metric, "winner": item["method"], "winner_value": item[metric], "seed": item.get("seed"), "generated_at": generated_at, "output_path": str(output_dir / "cetra_protocol_winners_v1.csv")})
    winners = pd.DataFrame(rows)
    winners.to_csv(output_dir / "cetra_protocol_winners_v1.csv", index=False)
    return winners


def _decision(metrics: pd.DataFrame, baselines: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    core = ["imadds_raw_main_binary", "imadds_raw_source_to_target", "imadds_raw_leave_target_weight35_out"]
    full = metrics[metrics["method"] == "CETRA_full"]
    info: dict[str, Any] = {}
    beats = {"raw_residual_energy": 0, "condition_decoupled_residual_energy": 0, "source_concentration_residual": 0}
    deep_beats = 0
    tree_close = 0
    tree_wins = 0
    for protocol in core:
        f = full[full["protocol"] == protocol]
        if f.empty:
            continue
        fm = f[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
        bgroup = baselines[baselines["protocol"] == protocol]
        for ref in beats:
            r = bgroup[bgroup["method"] == ref]
            if len(r):
                rm = r[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
                if fm["macro_f1"] >= rm["macro_f1"] + 0.03 or fm["pr_auc"] >= rm["pr_auc"] + 0.03 or (fm["mdr"] <= rm["mdr"] - 0.05 and fm["far"] <= rm["far"] + 0.10):
                    beats[ref] += 1
        deep = bgroup[bgroup["method"].isin(["AutoEncoder", "LSTM-AE", "USAD"])]
        if len(deep):
            dm = deep.groupby("method")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            if fm["macro_f1"] >= dm["macro_f1"].max() + 0.03 or fm["pr_auc"] >= dm["pr_auc"].max() + 0.03 or fm["mdr"] <= dm["mdr"].min() - 0.05:
                deep_beats += 1
        tree = bgroup[bgroup["method"].isin(["RandomForest", "XGBoost", "LightGBM"])]
        if len(tree):
            tm = tree.groupby("method")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            best_macro, best_prauc = tm["macro_f1"].max(), tm["pr_auc"].max()
            if fm["macro_f1"] >= best_macro - 0.05 or fm["pr_auc"] >= best_prauc - 0.05:
                tree_close += 1
            if best_macro >= fm["macro_f1"] + 0.03 and best_prauc >= fm["pr_auc"] + 0.03:
                tree_wins += 1
    # Mechanism ablation: full must beat ablations in at least one metric/protocol.
    ablation_support = 0
    for ablation in ["CETRA_no_condition_transport", "CETRA_no_cross_sensor_relations", "CETRA_no_conformal_calibration", "CETRA_univariate_event_only"]:
        for protocol in core:
            f = full[full["protocol"] == protocol]
            a = metrics[(metrics["method"] == ablation) & (metrics["protocol"] == protocol)]
            if len(f) and len(a):
                fm = f[["macro_f1", "pr_auc", "mdr"]].mean(numeric_only=True)
                am = a[["macro_f1", "pr_auc", "mdr"]].mean(numeric_only=True)
                if fm["macro_f1"] >= am["macro_f1"] + 0.03 or fm["pr_auc"] >= am["pr_auc"] + 0.03 or fm["mdr"] <= am["mdr"] * 0.85:
                    ablation_support += 1
                    break
    info.update({"beats_residual_refs": beats, "deep_beats_protocols": deep_beats, "tree_close_protocols": tree_close, "tree_wins_protocols": tree_wins, "ablation_support_count": ablation_support})
    if all(v >= 2 for v in beats.values()) and deep_beats >= 1 and ablation_support >= 3:
        if tree_close >= 1:
            return "CETRA_GO", info
        return "CETRA_PROMISING_NEEDS_REFINEMENT", info
    if tree_wins >= 2:
        return "CETRA_NO_GO_TREE_DOMINATES", info
    if ablation_support < 2:
        return "CETRA_NO_GO_MECHANISM_FAIL", info
    return "CETRA_PROMISING_NEEDS_REFINEMENT", info


def _write_packet(cfg: dict, output_dir: Path, packet_dir: Path, generated_at: str, status: str, decision_info: dict[str, Any], data, validity: pd.DataFrame, metrics: pd.DataFrame, baselines: pd.DataFrame, winners: pd.DataFrame, source: pd.DataFrame, calib: pd.DataFrame, stability: pd.DataFrame, device: str) -> None:
    prov = _prov("cetra_gate_v1", "7,13,23", output_dir, generated_at)
    save_config(cfg, output_dir / "05_cetra_gate_config.yaml")
    metrics[metrics["method"] == "CETRA_full"].to_csv(output_dir / "06_cetra_gate_metrics.csv", index=False)
    baselines.to_csv(output_dir / "07_cetra_baseline_comparison.csv", index=False)
    _write_text(output_dir / "00_readme_for_chatgpt.md", "Readme for ChatGPT", prov, [f"- current_stage: {STAGE}", f"- current_status: {status}", "- contains synthetic results: NO", "- generated images/figures: NO", "- entered full experiments: NO", "- CIRFL/residual-source tuning stopped: YES"])
    _write_text(output_dir / "01_old_direction_stop_report.md", "Old Direction Stop Report", prov, ["- CIRFL full mechanism stopped: YES.", "- residual-source route stopped: YES.", "- Raw-window External Gate v3 was REDESIGN_REQUIRED because residual/source candidates did not beat raw residual or deep baselines.", "- Direction was TREE_BASELINE_DOMINATES because LightGBM/XGBoost dominated IMAD-DS raw-window protocols.", "- No more RoAD-only or CIRFL_v4/v5 tuning is performed.", "- Historical references retained: CIRFL_v3_reference, raw_residual_energy, condition_decoupled_residual_energy, source_concentration_residual, LightGBM/XGBoost/RandomForest, AE/LSTM-AE/USAD."])
    _write_text(output_dir / "02_cetra_algorithm_spec.md", "CETRA Algorithm Spec", prov, ["- Core hypothesis: normal operation preserves a condition-normalized event-transition law even when raw amplitudes shift across load/noise/domain conditions.", "- Event alphabet: train-only per-channel amplitude quantile state, local slope direction, and local energy state form interpretable event tokens.", "- Condition transport: train-only empirical quantile transport maps condition-specific event distributions into a shared reference event-rank space.", "- Relational automaton: train-normal within-sensor transitions, cross-sensor co-events, and lagged cross-sensor transitions define the event-transport law.", "- Conformal risk scoring: nonconformity is transition-law violation; validation/calibration only selects thresholds and p-values.", "- Source localization: channel contributions and rare transitions are emitted from automaton violations, not from tree feature importance."])
    _write_text(output_dir / "03_cetra_novelty_guardrail.md", "CETRA Novelty Guardrail", prov, ["- CETRA is not an improved CNN/LSTM/Transformer/GNN/AE; it uses no neural sequence module.", "- CETRA is not a stacked hybrid; event alphabet, condition transport, automaton, and conformal scoring all serve one object: condition-normalized event-transition law.", "- CETRA is not residual detector renaming; it scores discrete event-law violations rather than raw/reconstruction residual magnitude.", "- CETRA does not include XGBoost/LightGBM/RandomForest in its algorithm.", "- If ablations do not support event transport or relational laws, the report marks CETRA_NO_GO rather than packaging weak components."])
    _write_text(output_dir / "04_cetra_protocol_validity.md", "CETRA Protocol Validity", prov, [dataframe_to_markdown(validity)])
    ablation = _ablation_text(metrics)
    _write_text(output_dir / "08_cetra_ablation_report.md", "CETRA Ablation Report", prov, ablation)
    _write_text(output_dir / "09_cetra_conformal_calibration.md", "CETRA Conformal Calibration", prov, [dataframe_to_markdown(calib[calib["method"].isin(["CETRA_full", "CETRA_no_conformal_calibration"])].head(100)) if len(calib) else "NOT_RUN", "- thresholds use validation/calibration only; no test threshold selection."])
    loc_body = ["## Source stability", dataframe_to_markdown(stability) if len(stability) else "NOT_AVAILABLE", "", "## Source examples", dataframe_to_markdown(source.head(80)) if len(source) else "NOT_AVAILABLE", "- Degeneracy check: source localization is non-degenerate if more than one top sensor appears across anomaly windows and seed-level Jaccard is finite."]
    _write_text(output_dir / "10_cetra_source_localization.md", "CETRA Source Localization", prov, loc_body)
    robustness = metrics[metrics["protocol"].isin(["imadds_raw_sensor_missing_10", "imadds_raw_sensor_missing_20"])]
    _write_text(output_dir / "11_cetra_robustness.md", "CETRA Robustness", prov, [dataframe_to_markdown(robustness.groupby(["protocol", "method"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index()) if len(robustness) else "NOT_RUN"])
    comp = pd.concat([metrics[metrics["method"] == "CETRA_full"], baselines], ignore_index=True)
    comp = comp[comp.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
    summary = summarize_by_method(comp) if len(comp) else pd.DataFrame()
    tests = paired_tests(comp, reference_method="CETRA_full") if len(comp) else pd.DataFrame()
    _write_text(output_dir / "12_cetra_statistical_summary.md", "CETRA Statistical Summary", prov, ["## Mean +/- std", dataframe_to_markdown(summary) if len(summary) else "NOT_RUN", "", "## Paired tests", dataframe_to_markdown(tests.head(80)) if len(tests) else "NOT_ENOUGH_RUNS", "", "## Protocol winners", dataframe_to_markdown(winners) if len(winners) else "NO_WINNERS"])
    _write_text(output_dir / "13_cetra_vs_tree_analysis.md", "CETRA vs Tree Analysis", prov, [_tree_text(metrics, baselines), f"- tree_close_protocols: {decision_info.get('tree_close_protocols')}", f"- tree_wins_protocols: {decision_info.get('tree_wins_protocols')}"])
    gpu_rows = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_rows.append({"device": f"cuda:{i}", "gpu_name": torch.cuda.get_device_name(i), "available": True, "assigned_task": "deep baselines from Raw-window Gate v3 reference" if i == 0 else "available for future parallelism; CETRA itself used CPU", "memory_allocated_mb": round(torch.cuda.memory_allocated(i)/(1024*1024),3)})
    else:
        gpu_rows.append({"device": "cpu", "gpu_name": "NA", "available": False, "assigned_task": "CETRA CPU", "memory_allocated_mb": 0})
    _write_text(output_dir / "14_device_decision_report.md", "Device Decision Report", prov, ["- CETRA device: CPU/NumPy; automaton scoring is small and stable on CPU.", f"- automatic device visible for deep baselines/reference: {device}", "- DataParallel used: NO.", "- multi-GPU used: NO for CETRA; three RTX 2080Ti remain available for future baseline parallelism.", dataframe_to_markdown(pd.DataFrame(gpu_rows))])
    lat = metrics.groupby("method")[["model_size_mb", "automaton_size_mb", "train_time_sec", "inference_time_sec", "cpu_latency_ms"]].mean(numeric_only=True).reset_index()
    _write_text(output_dir / "15_complexity_latency_cetra.md", "Complexity and Latency CETRA", prov, [dataframe_to_markdown(lat)])
    _write_text(output_dir / "16_cetra_go_no_go_report.md", "CETRA GO / NO-GO Report", prov, [f"## Decision: {status}", f"- valid IMAD-DS raw-window protocols: {int((validity['validity'] == 'VALID').sum())}", "- synthetic substitute used: NO", "- threshold/calibration uses validation only: YES", f"- beats residual references: {decision_info.get('beats_residual_refs')}", f"- deep baseline beat protocols: {decision_info.get('deep_beats_protocols')}", f"- tree close protocols: {decision_info.get('tree_close_protocols')}", f"- ablation support count: {decision_info.get('ablation_support_count')}", "- full experiments/manuscript remain blocked unless status is CETRA_GO and user explicitly starts full-experiment design."])
    _write_text(output_dir / "17_errors_and_risks.md", "Errors and Risks", prov, ["- no_full_experiments: YES", "- no_figures_generated: YES", "- no_synthetic_results: YES", "- CETRA is a new gate prototype and not yet validated beyond IMAD-DS raw-window pilot.", "- If tree models dominate, data protocol may be statistically easy or CETRA event law is insufficient.", "- Raw-window sample cap remains pilot-scale; all test segments retained.", "- apply_patch was blocked by sandbox bwrap loopback error; controlled local Python writes were used."])
    _write_text(output_dir / "18_code_index.md", "Code Index", prov, ["- `src/models/cetra_event_alphabet.py`: train-only event token alphabet.", "- `src/models/cetra_transport.py`: empirical condition quantile transport.", "- `src/models/cetra_automaton.py`: within/cross/lag relational automaton.", "- `src/models/cetra_conformal.py`: validation-only conformal calibrator.", "- `src/models/cetra.py`: CETRA wrapper with save/load and source scoring.", "- `src/evaluation/cetra_metrics.py`: source stability utility.", "- `configs/cetra_gate_v1.yaml`: CETRA gate config.", "- `scripts/run_cetra_gate_v1.py`: CETRA Gate v1 runner.", "", "## Commands", "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_cetra_gate_v1.py --config configs/cetra_gate_v1.yaml`"]) 
    next_line = "If CETRA_GO: design full experiments only. If PROMISING: small focused CETRA refinement. If NO-GO: stop CETRA and change research problem/algorithm."
    if status == "CETRA_NO_GO_TREE_DOMINATES":
        next_line = "Tree/statistical baselines still dominate; design a harder protocol/new data or replace the algorithmic direction."
    elif status == "CETRA_NO_GO_MECHANISM_FAIL":
        next_line = "Ablations do not support CETRA mechanism; stop CETRA or redesign core event-law assumptions."
    _write_text(output_dir / "19_next_tasks_for_codex.md", "Next Tasks for Codex", prov, [next_line])
    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        shutil.copyfile(output_dir / name, packet_dir / name)
    files = list(packet_dir.iterdir())
    if len(files) > 20:
        raise RuntimeError("packet exceeds 20 files")
    if any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} for p in files):
        raise RuntimeError("packet contains forbidden image")


def _ablation_text(metrics: pd.DataFrame) -> list[str]:
    rows = []
    for method in ["CETRA_full", "CETRA_no_condition_transport", "CETRA_no_cross_sensor_relations", "CETRA_no_lag_relations", "CETRA_no_conformal_calibration", "CETRA_univariate_event_only"]:
        sub = metrics[metrics["method"] == method]
        if len(sub):
            item = sub.groupby("protocol")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index()
            item.insert(0, "method", method)
            rows.append(item)
    if rows:
        return [dataframe_to_markdown(pd.concat(rows, ignore_index=True)), "- `CETRA_no_source_localization_regularization` is not applicable: CETRA has no trained source regularizer; source localization is derived directly from automaton violations."]
    return ["NOT_RUN"]


def _tree_text(metrics: pd.DataFrame, baselines: pd.DataFrame) -> str:
    full = metrics[metrics["method"] == "CETRA_full"]
    tree = baselines[baselines["method"].isin(["RandomForest", "XGBoost", "LightGBM"])]
    if full.empty or tree.empty:
        return "Tree comparison unavailable."
    fsum = full.groupby("protocol")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index()
    tsum = tree.groupby(["protocol", "method"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index()
    return dataframe_to_markdown(pd.concat([fsum.assign(method="CETRA_full"), tsum], ignore_index=True).head(80))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cetra_gate_v1.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = ROOT / cfg["project"]["output_dir"]
    packet_dir = ROOT / cfg["review_packet"]["output_dir"]
    _clean_dir(output_dir)
    generated_at = utc_now()
    raw_cfg_path = ROOT / cfg["raw_windows"]["config"]
    raw_cfg = yaml.safe_load(raw_cfg_path.read_text(encoding="utf-8"))["dataset"]
    npz_path = ROOT / raw_cfg["processed_npz"]
    meta_path = ROOT / raw_cfg["metadata_csv"]
    if not npz_path.exists() or not meta_path.exists():
        build_imadds_raw_windows(ROOT / raw_cfg["raw_dir"], npz_path, meta_path, segment_length=int(raw_cfg.get("segment_length",512)), window_size=int(raw_cfg.get("window_size",128)), stride=int(raw_cfg.get("stride",128)), max_train_source_segments=raw_cfg.get("max_train_source_segments",600), max_train_target_segments=raw_cfg.get("max_train_target_segments"), seed=int(raw_cfg.get("seed",7)))
    data = load_raw_windows(npz_path, meta_path)
    validity = _validity(data.meta, output_dir, generated_at)
    metrics, source, calib, stability = _run_cetra(data, cfg, output_dir, generated_at)
    baselines = _baseline_comparison(output_dir, generated_at)
    winners = _winner_summary(metrics, baselines, output_dir, generated_at)
    status, info = _decision(metrics, baselines)
    device = str(resolve_device(cfg.get("device", "auto_fastest")))
    _write_packet(cfg, output_dir, packet_dir, generated_at, status, info, data, validity, metrics, baselines, winners, source, calib, stability, device)
    print(f"CETRA Gate v1 finished: {status}; packet_files={len(list(packet_dir.iterdir()))}")


if __name__ == "__main__":
    main()
