from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.adapters.imadds_raw import build_imadds_raw_windows
from src.datasets.imadds_raw_window import load_raw_windows, window_stat_features
from src.evaluation.coil_metrics import coil_source_stability
from src.evaluation.metrics import binary_metric_row, choose_threshold, paired_tests, score_direction_multiplier, summarize_by_method
from src.models.coil import COIL
from src.models.coil_source_localization import entropy
from src.utils.config import save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import utc_now
from src.utils.torch_utils import resolve_device

STAGE = "COIL Gate v1"
SEEDS = [7, 13, 23]
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_old_direction_stop_report_v2.md",
    "02_coil_algorithm_spec.md",
    "03_coil_novelty_guardrail.md",
    "04_evidence_atom_audit.md",
    "05_coil_protocol_validity.md",
    "06_coil_gate_config.yaml",
    "07_coil_gate_metrics.csv",
    "08_coil_baseline_comparison.csv",
    "09_coil_statistical_summary.md",
    "10_coil_mechanism_necessity.md",
    "11_coil_vs_tree_analysis.md",
    "12_label_efficiency_gate.md",
    "13_robustness_gate.md",
    "14_source_evidence_localization.md",
    "15_device_decision_report.md",
    "16_complexity_latency_coil.md",
    "17_secondary_dataset_readiness.md",
    "18_go_no_go_report_coil_v1.md",
    "19_code_index_and_next_tasks.md",
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
        "- dataset: IMAD-DS RoboticArm raw windows + real RoAD/CETRA references",
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
        a.extend(ids[:n_a])
        b.extend(ids[n_a:n_a + n_b])
        c.extend(ids[n_a + n_b:])
    return set(a), set(b), set(c)


def _build_splits(meta: pd.DataFrame, seed: int) -> dict[str, dict[str, set[str]]]:
    seg = _segment_table(meta)
    test_seg = seg[seg["split_stage"] == "test"].copy()
    train_main = set(seg[(seg["split_stage"] == "train") & (seg["fault_label"] == 0)]["segment_uid"].astype(str))
    calib, val, test = _split_segments(test_seg, seed, (0.34, 0.33, 0.33))
    splits: dict[str, dict[str, set[str]]] = {"imadds_raw_main_binary": {"train": train_main, "calib": calib, "val": val, "test": test}}
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
    splits["imadds_raw_noise_mild"] = splits["imadds_raw_main_binary"].copy()
    splits["imadds_raw_noise_moderate"] = splits["imadds_raw_main_binary"].copy()
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
            "atom_fit_source": "train normal only",
            "lattice_fit_source": "train normal + calibration labels; no test",
            "threshold_source": "validation only",
            "generated_at": generated_at,
            "output_path": str(output_dir / "coil_protocol_validity_v1.csv"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "coil_protocol_validity_v1.csv", index=False)
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


def _add_noise(x: np.ndarray, std: float, seed: int) -> np.ndarray:
    if std <= 0:
        return x
    rng = np.random.default_rng(seed)
    scale = np.nanstd(x, axis=(0, 1), keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (x + rng.normal(0.0, std, size=x.shape).astype(np.float32) * scale).astype(np.float32)


def _cap_fit_arrays(x: np.ndarray, cond: np.ndarray, y: np.ndarray | None, max_n: int, seed: int):
    if max_n <= 0 or len(x) <= max_n:
        return (x, cond, y) if y is not None else (x, cond)
    rng = np.random.default_rng(seed)
    if y is None:
        idx = rng.choice(len(x), size=max_n, replace=False)
    else:
        y_arr = np.asarray(y)
        chunks = []
        for lab in sorted(set(y_arr.tolist())):
            lab_idx = np.where(y_arr == lab)[0]
            n_lab = max(1, int(round(max_n * len(lab_idx) / max(len(y_arr), 1))))
            chunks.append(rng.choice(lab_idx, size=min(n_lab, len(lab_idx)), replace=False))
        idx = np.concatenate(chunks) if chunks else np.arange(min(max_n, len(x)))
        if len(idx) > max_n:
            idx = rng.choice(idx, size=max_n, replace=False)
    idx = np.sort(idx)
    return (x[idx], cond[idx], y[idx]) if y is not None else (x[idx], cond[idx])


def _variant_params(method: str, cfg: dict) -> dict[str, Any]:
    base = dict(
        n_bins=int(cfg["coil"].get("n_bins", 7)),
        max_pairs=int(cfg["coil"].get("max_pairs", 80)),
        top_k=int(cfg["coil"].get("top_k", 12)),
        shrinkage_strength=float(cfg["coil"].get("shrinkage_strength", 20.0)),
        threshold_strategy=str(cfg["coil"].get("threshold_strategy", "best_f1")),
    )
    base.update(use_condition_orthogonalization=True, use_pairwise_coevidence=True, use_empirical_bayes=True, use_conformal_calibration=True, raw_evidence_only=False, normal_only_calibration=False)
    if method == "COIL_no_condition_orthogonalization":
        base["use_condition_orthogonalization"] = False
    elif method in {"COIL_univariate_only", "COIL_no_pairwise_coevidence"}:
        base["use_pairwise_coevidence"] = False
    elif method == "COIL_no_empirical_bayes":
        base["use_empirical_bayes"] = False
    elif method == "COIL_no_conformal_calibration":
        base["use_conformal_calibration"] = False
    elif method == "COIL_raw_evidence_only":
        base["raw_evidence_only"] = True
        base["use_pairwise_coevidence"] = False
    elif method == "COIL_normal_only_calibration":
        base["normal_only_calibration"] = True
    return base


def _fit_score_variant(method: str, x_train: np.ndarray, cond_train: np.ndarray, x_calib: np.ndarray, cond_calib: np.ndarray, y_calib: np.ndarray, x_val: np.ndarray, cond_val: np.ndarray, y_val: np.ndarray, x_test: np.ndarray, cond_test: np.ndarray, y_test: np.ndarray, channel_names: list[str], seed: int, protocol: str, cfg: dict, output_dir: Path, generated_at: str, missing_rate: float = 0.0, noise_std: float = 0.0, label_budget_fraction: float | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = COIL(**_variant_params(method, cfg))
    t0 = time.perf_counter()
    model.fit(x_train, cond_train, x_calib, cond_calib, y_calib, channel_names=channel_names, label_budget_fraction=label_budget_fraction, seed=seed)
    train_sec = time.perf_counter() - t0
    x_val_eval = _add_noise(_mask_missing(x_val, missing_rate, seed + 11), noise_std, seed + 31)
    x_test_eval = _add_noise(_mask_missing(x_test, missing_rate, seed + 17), noise_std, seed + 37)
    t1 = time.perf_counter()
    s_val, contrib_val, _, _ = model.score(x_val_eval, cond_val)
    s_test, contrib_test, notes_test, pair_df = model.score(x_test_eval, cond_test)
    mult = score_direction_multiplier(y_val, s_val)
    s_val = s_val * mult
    s_test = s_test * mult
    normal_only_threshold = method in {"COIL_no_conformal_calibration", "COIL_normal_only_calibration"}
    model.calibrate(y_val, s_val, use_labels=not normal_only_threshold, normal_only=normal_only_threshold)
    p_test = model.p_values(s_test)
    infer_sec = time.perf_counter() - t1
    thr = model.threshold()
    pred = (s_test >= thr).astype(int)
    row = binary_metric_row(y_test, s_test, thr, method, seed, protocol)
    row.update({
        "dataset": "IMAD-DS RoboticArm raw windows",
        "source_type": "external_real",
        "stage": "COIL_GATE_V1",
        "status": "RUN_OK",
        "n_train_windows": int(len(x_train)),
        "n_calib_windows": int(len(x_calib)),
        "n_val_windows": int(len(x_val)),
        "n_test_windows": int(len(x_test)),
        "model_size_mb": float(model.lattice_size_mb()),
        "lattice_size_mb": float(model.lattice_size_mb()),
        "cpu_latency_ms": float(infer_sec / max(len(x_test), 1) * 1000),
        "gpu_latency_ms": np.nan,
        "train_time_sec": float(train_sec),
        "inference_time_sec": float(infer_sec),
        "threshold_source": model.calibrator.threshold_source_,
        "calibration_set_size": int((y_val == 0).sum()),
        "conformal_p_mean": float(np.nanmean(p_test)),
        "missing_channel_rate": float(missing_rate),
        "noise_std": float(noise_std),
        "label_budget_fraction": float(label_budget_fraction) if label_budget_fraction is not None else 1.0,
        "generated_at": generated_at,
        "output_path": str(output_dir),
    })
    source_df = _source_rows(model, contrib_test, notes_test, pair_df, y_test, pred, s_test, p_test, protocol, seed, method, channel_names, output_dir, generated_at)
    atom_summary = model.atom_summary()
    atom_summary["method"] = method
    atom_summary["protocol"] = protocol
    atom_summary["seed"] = seed
    atom_summary["generated_at"] = generated_at
    atom_summary["output_path"] = str(output_dir / "evidence_atom_summary.csv")
    support = model.condition_support_report()
    support["method"] = method
    support["protocol"] = protocol
    support["seed"] = seed
    support["generated_at"] = generated_at
    support["output_path"] = str(output_dir / "condition_support_report.csv")
    lattice = model.lattice_fit_report()
    lattice["method"] = method
    lattice["protocol"] = protocol
    lattice["seed"] = seed
    lattice["generated_at"] = generated_at
    lattice["output_path"] = str(output_dir / "evidence_lattice_fit_report.csv")
    return row, source_df, atom_summary, support, lattice


def _source_rows(model: COIL, contrib_test: np.ndarray, notes_test: list[str], pair_df: pd.DataFrame, y_test: np.ndarray, pred: np.ndarray, scores: np.ndarray, p_values: np.ndarray, protocol: str, seed: int, method: str, channel_names: list[str], output_dir: Path, generated_at: str) -> pd.DataFrame:
    rows = []
    names = channel_names or [f"sensor_{i}" for i in range(contrib_test.shape[1])]
    atom_names = model.atomizer.atom_info_frame()["atom_name"].astype(str).to_list()
    pair_lookup: dict[int, str] = {}
    if pair_df is not None and len(pair_df):
        for w, group in pair_df.groupby("window_index"):
            item = group.sort_values("pair_contribution", ascending=False).iloc[0]
            li, ri = int(item["left_atom"]), int(item["right_atom"])
            pair_lookup[int(w)] = f"{atom_names[li]} <-> {atom_names[ri]}"
    for i in range(len(y_test)):
        order = np.argsort(-contrib_test[i])
        top = [names[j] for j in order[: min(3, len(order))]]
        rows.append({
            "dataset": "IMAD-DS RoboticArm raw windows",
            "source_type": "external_real",
            "protocol": protocol,
            "seed": seed,
            "method": method,
            "window_rank": i,
            "y_true": int(y_test[i]),
            "predicted_label": int(pred[i]),
            "anomaly_score": float(scores[i]),
            "conformal_p_value": float(p_values[i]) if np.isfinite(p_values[i]) else np.nan,
            "top_sensors": ";".join(top),
            "top_evidence_atom": notes_test[i] if i < len(notes_test) else "NA",
            "top_pairwise_evidence": pair_lookup.get(i, "NA"),
            "max_sensor_contribution": float(np.max(contrib_test[i])),
            "contribution_entropy": float(entropy(contrib_test[i])),
            "generated_at": generated_at,
            "output_path": str(output_dir / "source_evidence_localization_v1.csv"),
        })
    return pd.DataFrame(rows)


def _run_coil(data, cfg: dict, output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    methods = [
        "COIL_full",
        "COIL_no_condition_orthogonalization",
        "COIL_univariate_only",
        "COIL_no_empirical_bayes",
        "COIL_no_conformal_calibration",
        "COIL_raw_evidence_only",
        "COIL_normal_only_calibration",
    ]
    rows = []
    source_frames = []
    atom_frames = []
    support_frames = []
    lattice_frames = []
    for seed in SEEDS:
        splits = _build_splits(data.meta, seed)
        for protocol in cfg["protocols"]["core"] + cfg["protocols"].get("robustness", []):
            base_protocol = "imadds_raw_main_binary" if protocol in cfg["protocols"].get("robustness", []) else protocol
            parts = splits[base_protocol]
            train_idx, calib_idx, val_idx, test_idx = _idx(data.meta, parts["train"]), _idx(data.meta, parts["calib"]), _idx(data.meta, parts["val"]), _idx(data.meta, parts["test"])
            train_meta = data.meta.iloc[train_idx].reset_index(drop=True)
            calib_meta = data.meta.iloc[calib_idx].reset_index(drop=True)
            val_meta = data.meta.iloc[val_idx].reset_index(drop=True)
            test_meta = data.meta.iloc[test_idx].reset_index(drop=True)
            normal_train = train_meta["fault_label"].to_numpy(int) == 0
            x_train = data.x[train_idx][normal_train]
            cond_train = train_meta.loc[normal_train, "condition_name"].astype(str).to_numpy()
            x_calib, x_val, x_test = data.x[calib_idx], data.x[val_idx], data.x[test_idx]
            cond_calib = calib_meta["condition_name"].astype(str).to_numpy()
            cond_val = val_meta["condition_name"].astype(str).to_numpy()
            cond_test = test_meta["condition_name"].astype(str).to_numpy()
            y_calib = calib_meta["fault_label"].to_numpy(int)
            y_val = val_meta["fault_label"].to_numpy(int)
            y_test = test_meta["fault_label"].to_numpy(int)
            if len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
                continue
            train_cap = int(cfg.get("coil", {}).get("fit_train_cap", 0))
            calib_cap = int(cfg.get("coil", {}).get("fit_calib_cap", 0))
            x_train_fit, cond_train_fit = _cap_fit_arrays(x_train, cond_train, None, train_cap, seed + 901)
            x_calib_fit, cond_calib_fit, y_calib_fit = _cap_fit_arrays(x_calib, cond_calib, y_calib, calib_cap, seed + 902)
            missing = float(cfg["protocols"].get("missing_channel_rates", {}).get(protocol, 0.0))
            noise = float(cfg["protocols"].get("noise_std", {}).get(protocol, 0.0))
            run_methods = ["COIL_full"] if protocol in cfg["protocols"].get("robustness", []) else methods
            for method in run_methods:
                row, src, atom, support, lattice = _fit_score_variant(method, x_train_fit, cond_train_fit, x_calib_fit, cond_calib_fit, y_calib_fit, x_val, cond_val, y_val, x_test, cond_test, y_test, data.channel_names, seed, protocol, cfg, output_dir, generated_at, missing_rate=missing, noise_std=noise)
                rows.append(row)
                if method == "COIL_full":
                    source_frames.append(src)
                    atom_frames.append(atom)
                    support_frames.append(support)
                    lattice_frames.append(lattice)
    metrics = pd.DataFrame(rows)
    source = pd.concat(source_frames, ignore_index=True) if source_frames else pd.DataFrame()
    atom_summary = pd.concat(atom_frames, ignore_index=True) if atom_frames else pd.DataFrame()
    support = pd.concat(support_frames, ignore_index=True) if support_frames else pd.DataFrame()
    lattice = pd.concat(lattice_frames, ignore_index=True) if lattice_frames else pd.DataFrame()
    stability = coil_source_stability(source) if len(source) else pd.DataFrame()
    metrics.to_csv(output_dir / "coil_gate_metrics_all_methods.csv", index=False)
    metrics[metrics["method"] == "COIL_full"].to_csv(output_dir / "07_coil_gate_metrics.csv", index=False)
    source.to_csv(output_dir / "source_evidence_localization_v1.csv", index=False)
    atom_summary.to_csv(output_dir / "evidence_atom_summary.csv", index=False)
    support.to_csv(output_dir / "condition_support_report.csv", index=False)
    lattice.to_csv(output_dir / "evidence_lattice_fit_report.csv", index=False)
    stability.to_csv(output_dir / "source_evidence_stability.csv", index=False)
    return metrics, source, atom_summary, support, lattice, stability


def _baseline_comparison(output_dir: Path, generated_at: str) -> pd.DataFrame:
    frames = []
    paths = [
        ROOT / "outputs/cetra_gate_v1/06_cetra_gate_metrics.csv",
        ROOT / "outputs/cetra_gate_v1/07_cetra_baseline_comparison.csv",
        ROOT / "outputs/raw_window_external_gate_v3/07_raw_window_gate_metrics.csv",
        ROOT / "outputs/raw_window_external_gate_v3/08_raw_window_baseline_comparison.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            df["comparison_source"] = p.as_posix()
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["method", "seed", "protocol"], keep="first")
    out["generated_at"] = generated_at
    out["output_path"] = str(output_dir / "08_coil_baseline_comparison.csv")
    out.to_csv(output_dir / "08_coil_baseline_comparison.csv", index=False)
    return out


def _budget_mask(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 0.999:
        return np.ones(len(y), dtype=bool)
    rng = np.random.default_rng(seed)
    keep = np.zeros(len(y), dtype=bool)
    for lab in [0, 1]:
        idx = np.where(y == lab)[0]
        if len(idx) == 0:
            continue
        n = max(1, int(round(len(idx) * frac)))
        keep[rng.choice(idx, size=min(n, len(idx)), replace=False)] = True
    return keep


def _run_label_efficiency(data, cfg: dict, output_dir: Path, generated_at: str) -> pd.DataFrame:
    rows = []
    budgets = cfg.get("label_efficiency", {}).get("budgets", {"normal_only": 0.0, "labeled_5pct": 0.05, "labeled_20pct": 0.2, "full_validation": 1.0})
    for seed in SEEDS:
        splits = _build_splits(data.meta, seed)
        for protocol in ["imadds_raw_main_binary"]:
            parts = splits[protocol]
            train_idx, calib_idx, val_idx, test_idx = _idx(data.meta, parts["train"]), _idx(data.meta, parts["calib"]), _idx(data.meta, parts["val"]), _idx(data.meta, parts["test"])
            train_meta = data.meta.iloc[train_idx].reset_index(drop=True)
            calib_meta = data.meta.iloc[calib_idx].reset_index(drop=True)
            val_meta = data.meta.iloc[val_idx].reset_index(drop=True)
            test_meta = data.meta.iloc[test_idx].reset_index(drop=True)
            normal_train = train_meta["fault_label"].to_numpy(int) == 0
            x_train = data.x[train_idx][normal_train]
            cond_train = train_meta.loc[normal_train, "condition_name"].astype(str).to_numpy()
            x_calib, x_val, x_test = data.x[calib_idx], data.x[val_idx], data.x[test_idx]
            cond_calib = calib_meta["condition_name"].astype(str).to_numpy()
            cond_val = val_meta["condition_name"].astype(str).to_numpy()
            cond_test = test_meta["condition_name"].astype(str).to_numpy()
            y_calib = calib_meta["fault_label"].to_numpy(int)
            y_val = val_meta["fault_label"].to_numpy(int)
            y_test = test_meta["fault_label"].to_numpy(int)
            train_cap = int(cfg.get("coil", {}).get("fit_train_cap", 0))
            calib_cap = int(cfg.get("coil", {}).get("fit_calib_cap", 0))
            x_train_fit, cond_train_fit = _cap_fit_arrays(x_train, cond_train, None, train_cap, seed + 1901)
            x_calib_fit, cond_calib_fit, y_calib_fit = _cap_fit_arrays(x_calib, cond_calib, y_calib, calib_cap, seed + 1902)
            for name, frac in budgets.items():
                frac = float(frac)
                normal_only = name == "normal_only"
                method = f"COIL_full::{name}"
                params = _variant_params("COIL_full", cfg)
                params["normal_only_calibration"] = normal_only
                model = COIL(**params)
                model.fit(x_train_fit, cond_train_fit, x_calib_fit, cond_calib_fit, y_calib_fit, channel_names=data.channel_names, label_budget_fraction=None if normal_only else frac, seed=seed)
                s_val, _, _, _ = model.score(x_val, cond_val)
                s_test, _, _, _ = model.score(x_test, cond_test)
                if not normal_only:
                    val_keep = _budget_mask(y_val, frac, seed + 500)
                    mult = score_direction_multiplier(y_val[val_keep], s_val[val_keep]) if len(np.unique(y_val[val_keep])) == 2 else score_direction_multiplier(y_val, s_val)
                    s_val_o = s_val * mult
                    s_test_o = s_test * mult
                    y_thr = y_val[val_keep] if len(np.unique(y_val[val_keep])) == 2 else y_val
                    s_thr = s_val_o[val_keep] if len(np.unique(y_val[val_keep])) == 2 else s_val_o
                    model.calibrator.fit(y_thr, s_thr, use_labels=True, normal_only=False)
                else:
                    mult = 1.0
                    s_val_o = s_val
                    s_test_o = s_test
                    model.calibrator.fit(y_val, s_val_o, use_labels=False, normal_only=True)
                row = binary_metric_row(y_test, s_test_o, model.threshold(), method, seed, protocol)
                row.update({"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "label_budget": name, "label_budget_fraction": frac, "n_train_windows": len(x_train), "n_val_windows": len(x_val), "n_test_windows": len(x_test), "threshold_source": model.calibrator.threshold_source_, "generated_at": generated_at, "output_path": str(output_dir / "label_budget_comparison.csv")})
                rows.append(row)
                if name != "normal_only" and cfg.get("label_efficiency", {}).get("run_tree_budget_baselines", True):
                    rows.extend(_tree_budget_rows(data, train_idx, calib_idx, val_idx, test_idx, y_calib, y_val, y_test, frac, name, seed, protocol, output_dir, generated_at))
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "label_budget_comparison.csv", index=False)
    return out


def _tree_budget_rows(data, train_idx: np.ndarray, calib_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray, y_calib: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, frac: float, budget_name: str, seed: int, protocol: str, output_dir: Path, generated_at: str) -> list[dict]:
    rows = []
    x_train_norm = data.x[train_idx]
    y_train_norm = np.zeros(len(train_idx), dtype=int)
    keep_calib = _budget_mask(y_calib, frac, seed + 700)
    keep_val = _budget_mask(y_val, frac, seed + 800)
    if len(np.unique(y_calib[keep_calib])) < 2 or len(np.unique(y_val[keep_val])) < 2:
        return rows
    x_fit = np.concatenate([window_stat_features(x_train_norm), window_stat_features(data.x[calib_idx][keep_calib])], axis=0)
    y_fit = np.concatenate([y_train_norm, y_calib[keep_calib]], axis=0)
    x_val = window_stat_features(data.x[val_idx])
    x_test = window_stat_features(data.x[test_idx])
    scaler = StandardScaler().fit(x_fit)
    x_fit_s, x_val_s, x_test_s = scaler.transform(x_fit), scaler.transform(x_val), scaler.transform(x_test)
    models: list[tuple[str, Any]] = [("RandomForest", RandomForestClassifier(n_estimators=80, max_depth=8, random_state=seed, n_jobs=1, class_weight="balanced"))]
    if XGBClassifier is not None:
        models.append(("XGBoost", XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", tree_method="hist", random_state=seed, n_jobs=1)))
    if LGBMClassifier is not None:
        models.append(("LightGBM", LGBMClassifier(n_estimators=80, max_depth=5, learning_rate=0.06, random_state=seed, n_jobs=1, verbose=-1)))
    for name, model in models:
        try:
            model.fit(x_fit_s, y_fit)
            if hasattr(model, "predict_proba"):
                s_val = model.predict_proba(x_val_s)[:, 1]
                s_test = model.predict_proba(x_test_s)[:, 1]
            else:
                s_val = model.decision_function(x_val_s)
                s_test = model.decision_function(x_test_s)
            thr = choose_threshold(y_val[keep_val], s_val[keep_val], strategy="best_f1")
            row = binary_metric_row(y_test, s_test, thr, f"{name}::{budget_name}", seed, protocol)
            row.update({"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "label_budget": budget_name, "label_budget_fraction": frac, "n_train_windows": int(len(x_fit_s)), "n_val_windows": int(len(x_val)), "n_test_windows": int(len(x_test)), "threshold_source": "validation_budget_only", "generated_at": generated_at, "output_path": str(output_dir / "label_budget_comparison.csv")})
            rows.append(row)
        except Exception as exc:
            rows.append({"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "protocol": protocol, "seed": seed, "method": f"{name}::{budget_name}", "status": f"SKIPPED_WITH_REASON: {exc}", "label_budget": budget_name, "label_budget_fraction": frac, "n_train_windows": int(len(x_fit_s)), "n_val_windows": int(len(x_val)), "n_test_windows": int(len(x_test)), "n_test_normal": int((y_test == 0).sum()), "n_test_anomaly": int((y_test == 1).sum()), "generated_at": generated_at, "output_path": str(output_dir / "label_budget_comparison.csv")})
    return rows


def _winner_summary(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path, generated_at: str) -> pd.DataFrame:
    comp = pd.concat([metrics[metrics["method"] == "COIL_full"], baselines], ignore_index=True)
    comp = comp[comp.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
    rows = []
    for protocol, group in comp.groupby("protocol"):
        for metric, maximize in [("macro_f1", True), ("pr_auc", True), ("mdr", False), ("far", False)]:
            good = group.dropna(subset=[metric])
            if good.empty:
                continue
            idx = good[metric].idxmax() if maximize else good[metric].idxmin()
            item = good.loc[idx]
            rows.append({"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "protocol": protocol, "metric": metric, "winner": item["method"], "winner_value": item[metric], "seed": item.get("seed"), "generated_at": generated_at, "output_path": str(output_dir / "coil_gate_v1_protocol_winners.csv")})
    winners = pd.DataFrame(rows)
    winners.to_csv(output_dir / "coil_gate_v1_protocol_winners.csv", index=False)
    return winners


def _decision(metrics: pd.DataFrame, baselines: pd.DataFrame, label_df: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    core = ["imadds_raw_main_binary", "imadds_raw_source_to_target", "imadds_raw_leave_target_weight35_out"]
    full = metrics[metrics["method"] == "COIL_full"]
    info: dict[str, Any] = {}
    main_conditions = 0
    deep_beats = 0
    tree_close = 0
    tree_wins = 0
    for protocol in core:
        f = full[full["protocol"] == protocol]
        if f.empty:
            continue
        fm = f[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
        if protocol == "imadds_raw_main_binary" and (fm["macro_f1"] >= 0.75 or fm["pr_auc"] >= 0.85):
            main_conditions += 1
        if protocol == "imadds_raw_source_to_target" and (fm["macro_f1"] >= 0.65 or fm["pr_auc"] >= 0.78):
            main_conditions += 1
        if protocol == "imadds_raw_leave_target_weight35_out" and (fm["macro_f1"] >= 0.50 or fm["pr_auc"] >= 0.65):
            main_conditions += 1
        bgroup = baselines[baselines["protocol"] == protocol]
        deep = bgroup[bgroup["method"].isin(["AutoEncoder", "LSTM-AE", "USAD"])]
        if len(deep):
            dm = deep.groupby("method")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            if fm["macro_f1"] >= dm["macro_f1"].max() + 0.03 or fm["pr_auc"] >= dm["pr_auc"].max() + 0.03 or fm["mdr"] <= dm["mdr"].min() - 0.05:
                deep_beats += 1
        tree = bgroup[bgroup["method"].isin(["RandomForest", "XGBoost", "LightGBM"])]
        if len(tree):
            tm = tree.groupby("method")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            best_macro, best_pr = tm["macro_f1"].max(), tm["pr_auc"].max()
            if fm["macro_f1"] >= best_macro - 0.05 or fm["pr_auc"] >= best_pr - 0.05:
                tree_close += 1
            if best_macro >= fm["macro_f1"] + 0.03 and best_pr >= fm["pr_auc"] + 0.03:
                tree_wins += 1
    mechanism_support = 0
    for ablation in ["COIL_no_condition_orthogonalization", "COIL_univariate_only", "COIL_no_empirical_bayes", "COIL_no_conformal_calibration", "COIL_raw_evidence_only"]:
        for protocol in core:
            f = full[full["protocol"] == protocol]
            a = metrics[(metrics["method"] == ablation) & (metrics["protocol"] == protocol)]
            if len(f) and len(a):
                fm = f[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
                am = a[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
                if fm["macro_f1"] >= am["macro_f1"] + 0.03 or fm["pr_auc"] >= am["pr_auc"] + 0.03 or (fm["far"] <= am["far"] * 0.90 and fm["mdr"] <= am["mdr"] + 0.05):
                    mechanism_support += 1
                    break
    label_advantage = _label_advantage(label_df)
    robustness_ok = _robustness_ok(metrics)
    info.update({"main_performance_conditions": main_conditions, "deep_beats_protocols": deep_beats, "tree_close_protocols": tree_close, "tree_wins_protocols": tree_wins, "mechanism_support_count": mechanism_support, "label_efficiency_advantage": label_advantage, "robustness_ok": robustness_ok})
    if main_conditions >= 2 and mechanism_support >= 3 and tree_close >= 1:
        return "COIL_GO", info
    if label_advantage and mechanism_support >= 2:
        return "COIL_PROMISING_FOR_LABEL_EFFICIENCY", info
    if mechanism_support < 2:
        return "COIL_NO_GO_MECHANISM_FAIL", info
    if tree_wins >= 2:
        return "COIL_NO_GO_TREE_DOMINATES", info
    return "COIL_PROMISING_NEEDS_REFINEMENT", info


def _label_advantage(label_df: pd.DataFrame) -> bool:
    if label_df.empty:
        return False
    coil = label_df[label_df["method"].astype(str).str.startswith("COIL_full")]
    trees = label_df[label_df["method"].astype(str).str.startswith(("LightGBM", "XGBoost", "RandomForest"))]
    for budget in ["normal_only", "labeled_5pct"]:
        c = coil[coil["label_budget"] == budget]
        t = trees[trees["label_budget"] == budget]
        if not c.empty and not t.empty:
            cm = c.groupby("protocol")[["macro_f1", "pr_auc"]].mean(numeric_only=True)
            tm = t.groupby("protocol")[["macro_f1", "pr_auc"]].max(numeric_only=True)
            common = cm.index.intersection(tm.index)
            for protocol in common:
                if cm.loc[protocol, "macro_f1"] >= tm.loc[protocol, "macro_f1"] + 0.03 or cm.loc[protocol, "pr_auc"] >= tm.loc[protocol, "pr_auc"] + 0.03:
                    return True
    return False


def _robustness_ok(metrics: pd.DataFrame) -> bool:
    main = metrics[(metrics["method"] == "COIL_full") & (metrics["protocol"] == "imadds_raw_main_binary")]
    miss = metrics[(metrics["method"] == "COIL_full") & (metrics["protocol"] == "imadds_raw_sensor_missing_10")]
    if main.empty or miss.empty:
        return False
    mm = main[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
    ms = miss[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
    drop = max(mm["macro_f1"] - ms["macro_f1"], mm["pr_auc"] - ms["pr_auc"])
    return bool(drop <= 0.07 and not (ms["far"] > mm["far"] and ms["mdr"] > mm["mdr"]))


def _write_packet(cfg: dict, output_dir: Path, packet_dir: Path, generated_at: str, status: str, info: dict[str, Any], data, validity: pd.DataFrame, metrics: pd.DataFrame, baselines: pd.DataFrame, winners: pd.DataFrame, label_df: pd.DataFrame, source: pd.DataFrame, atom_summary: pd.DataFrame, support: pd.DataFrame, lattice: pd.DataFrame, stability: pd.DataFrame, device: str) -> None:
    prov = _prov("coil_gate_v1", "7,13,23", output_dir, generated_at)
    save_config(cfg, output_dir / "06_coil_gate_config.yaml")
    metrics.to_csv(output_dir / "07_coil_gate_metrics.csv", index=False)
    baselines.to_csv(output_dir / "08_coil_baseline_comparison.csv", index=False)
    _write_text(output_dir / "00_readme_for_chatgpt.md", "Readme for ChatGPT", prov, [f"- current_stage: {STAGE}", f"- current_status: {status}", "- contains synthetic results: NO", "- generated images/figures: NO", "- entered full experiments: NO", "- CIRFL/residual-source/CETRA tuning stopped: YES"])
    _write_text(output_dir / "01_old_direction_stop_report_v2.md", "Old Direction Stop Report v2", prov, ["- CIRFL full mechanism stopped: YES.", "- residual-source route stopped: YES.", "- CETRA/event-grammar route stopped: YES.", "- CETRA Gate v1 was CETRA_NO_GO_TREE_DOMINATES because LightGBM/XGBoost/RandomForest dominated core raw-window protocols.", "- Micro-tuning CETRA is not pursued; it did not establish a new main algorithm path.", "- Historical references retained: CIRFL_v3_reference, raw_residual_energy, condition_decoupled_residual_energy, source_concentration_residual, CETRA_full, LightGBM/XGBoost/RandomForest, AE/LSTM-AE/USAD."])
    _write_text(output_dir / "02_coil_algorithm_spec.md", "COIL Algorithm Spec", prov, ["- Core hypothesis: anomalies form sparse evidence atoms in a condition-orthogonal evidence lattice after train-only condition normalization.", "- Evidence atoms: train-only rank-amplitude, local-change, energy-shape, stability, extremeness, and sparse pairwise co-evidence atoms.", "- Condition orthogonalization: per-condition robust centering with global empirical-Bayes shrinkage; no neural domain adaptation and no test distribution fitting.", "- Evidence lattice: univariate and sparse pairwise bin risks with empirical-Bayes smoothing and uncertainty shrinkage, not recursive tree splitting.", "- Score: monotone abnormal evidence mass from high-risk lattice atoms plus conformal p-value and validation-only threshold.", "- Source localization: sensor, atom, and pairwise evidence contributions are emitted directly from lattice contributions."])
    _write_text(output_dir / "03_coil_novelty_guardrail.md", "COIL Novelty Guardrail", prov, ["- COIL is not an improved Transformer/GNN/LSTM/CNN/AE and does not use deep modules as its core.", "- COIL is not LightGBM/XGBoost/RandomForest wrapping: it does not do recursive split search or tree ensembles.", "- COIL is not residual detector renaming and is not CETRA/event grammar continuation.", "- The unified object is the condition-orthogonal evidence lattice; atoms, orthogonalization, empirical-Bayes lattice, conformal calibration, and source localization all serve that object.", "- If ablations fail, the gate report marks NO-GO rather than packaging weak components."])
    _write_text(output_dir / "04_evidence_atom_audit.md", "Evidence Atom Audit", prov, ["## Train-only atom summary", dataframe_to_markdown(atom_summary.head(80)) if len(atom_summary) else "NOT_RUN", "", "## Condition support", dataframe_to_markdown(support.head(80)) if len(support) else "NOT_RUN", "", "- leakage_check: atom statistics and condition orthogonalization were fit on train-normal only; lattice used train-normal + calibration labels; threshold used validation only."])
    _write_text(output_dir / "05_coil_protocol_validity.md", "COIL Protocol Validity", prov, [dataframe_to_markdown(validity)])
    _write_text(output_dir / "09_coil_statistical_summary.md", "COIL Statistical Summary", prov, ["## Mean +/- std", dataframe_to_markdown(summarize_by_method(pd.concat([metrics[metrics["method"] == "COIL_full"], baselines], ignore_index=True)).head(120)), "", "## Paired tests", dataframe_to_markdown(paired_tests(pd.concat([metrics[metrics["method"] == "COIL_full"], baselines], ignore_index=True), reference_method="COIL_full").head(80)), "", "## Protocol winners", dataframe_to_markdown(winners)])
    _write_text(output_dir / "10_coil_mechanism_necessity.md", "COIL Mechanism Necessity", prov, [_ablation_table(metrics), f"- mechanism_support_count: {info.get('mechanism_support_count')}"])
    _write_text(output_dir / "11_coil_vs_tree_analysis.md", "COIL vs Tree Analysis", prov, [_tree_table(metrics, baselines), f"- tree_close_protocols: {info.get('tree_close_protocols')}", f"- tree_wins_protocols: {info.get('tree_wins_protocols')}"])
    _write_text(output_dir / "12_label_efficiency_gate.md", "Label Efficiency Gate", prov, [dataframe_to_markdown(label_df.groupby(["method", "label_budget", "protocol"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index().head(120)) if len(label_df) else "NOT_RUN", f"- label_efficiency_advantage: {info.get('label_efficiency_advantage')}"])
    rob = metrics[metrics["protocol"].isin(cfg["protocols"].get("robustness", []))]
    _write_text(output_dir / "13_robustness_gate.md", "Robustness Gate", prov, [dataframe_to_markdown(rob.groupby(["protocol", "method"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index()) if len(rob) else "NOT_RUN", f"- robustness_ok: {info.get('robustness_ok')}"])
    _write_text(output_dir / "14_source_evidence_localization.md", "Source Evidence Localization", prov, ["## Source stability", dataframe_to_markdown(stability) if len(stability) else "NOT_AVAILABLE", "", "## Source examples", dataframe_to_markdown(source.head(80)) if len(source) else "NOT_AVAILABLE", "- non_degenerate_check: source evidence is non-degenerate if multiple sensors/atoms appear and seed-level Jaccard is finite."])
    gpu_rows = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_rows.append({"device": f"cuda:{i}", "gpu_name": torch.cuda.get_device_name(i), "available": True, "assigned_task": "deep baselines from prior raw-window reference" if i == 0 else "available; COIL used CPU", "memory_allocated_mb": round(torch.cuda.memory_allocated(i)/(1024*1024), 3)})
    else:
        gpu_rows.append({"device": "cpu", "gpu_name": "NA", "available": False, "assigned_task": "COIL CPU", "memory_allocated_mb": 0})
    _write_text(output_dir / "15_device_decision_report.md", "Device Decision Report", prov, ["- COIL device: CPU/NumPy/pandas; evidence extraction and lattice scoring are small and deterministic on CPU.", f"- automatic device visible for deep baselines/reference: {device}", "- DataParallel used: NO.", "- multi-GPU used: NO for COIL; three RTX 2080Ti remain available for future baseline parallelism.", dataframe_to_markdown(pd.DataFrame(gpu_rows))])
    lat = metrics.groupby("method")[["model_size_mb", "lattice_size_mb", "train_time_sec", "inference_time_sec", "cpu_latency_ms"]].mean(numeric_only=True).reset_index()
    _write_text(output_dir / "16_complexity_latency_coil.md", "Complexity and Latency COIL", prov, [dataframe_to_markdown(lat.head(80))])
    readiness = ROOT / "outputs/raw_window_external_gate_v3/14_brushless_nist_kuka_readiness_v3.md"
    readiness_text = readiness.read_text(encoding="utf-8") if readiness.exists() else "BrushlessMotor READY from previous gate; NIST UR READY for health/degradation feasibility; KUKA optional missing."
    _write_text(output_dir / "17_secondary_dataset_readiness.md", "Secondary Dataset Readiness", prov, [readiness_text[:4000]])
    _write_text(output_dir / "18_go_no_go_report_coil_v1.md", "COIL GO / NO-GO Report", prov, [f"## Decision: {status}", f"- valid protocols: {int((validity['validity'] == 'VALID').sum())}", "- synthetic substitute used: NO", "- threshold/atom/lattice fitting uses no test: YES", f"- main_performance_conditions: {info.get('main_performance_conditions')}", f"- deep_beats_protocols: {info.get('deep_beats_protocols')}", f"- tree_close_protocols: {info.get('tree_close_protocols')}", f"- mechanism_support_count: {info.get('mechanism_support_count')}", f"- label_efficiency_advantage: {info.get('label_efficiency_advantage')}", f"- robustness_ok: {info.get('robustness_ok')}", "- full experiments/manuscript remain blocked unless status is COIL_GO and user explicitly starts full-experiment design."])
    next_line = "Stop COIL or design a harder protocol/new data; tree/deep baselines still dominate."
    if status == "COIL_PROMISING_FOR_LABEL_EFFICIENCY":
        next_line = "Keep COIL only for a narrow label-efficiency refinement gate; do not enter full experiments yet."
    elif status == "COIL_PROMISING_NEEDS_REFINEMENT":
        next_line = "Run one small refinement gate focused on the failing mechanism or robustness condition."
    elif status == "COIL_GO":
        next_line = "Prepare full experiment design only; do not write manuscript yet."
    _write_text(output_dir / "19_code_index_and_next_tasks.md", "Code Index and Next Tasks", prov, ["- `src/models/coil_evidence_atoms.py`: train-only evidence atoms.", "- `src/models/coil_condition_orthogonalizer.py`: condition-orthogonal transform.", "- `src/models/coil_lattice.py`: empirical-Bayes evidence lattice.", "- `src/models/coil_conformal.py`: validation-only conformal calibration.", "- `src/models/coil_source_localization.py`: source evidence stability.", "- `src/models/coil.py`: COIL wrapper.", "- `src/evaluation/coil_metrics.py`: COIL metrics helpers.", "- `configs/coil_gate_v1_imadds_raw.yaml`: gate config.", "- `scripts/run_coil_gate_v1.py`: gate runner.", "", "## Command", "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_coil_gate_v1.py --config configs/coil_gate_v1_imadds_raw.yaml`", "", f"## Next", next_line])
    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        shutil.copyfile(output_dir / name, packet_dir / name)
    files = list(packet_dir.iterdir())
    if len(files) > 20:
        raise RuntimeError("packet exceeds 20 files")
    if any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} for p in files):
        raise RuntimeError("packet contains forbidden image")


def _ablation_table(metrics: pd.DataFrame) -> str:
    methods = ["COIL_full", "COIL_no_condition_orthogonalization", "COIL_univariate_only", "COIL_no_pairwise_coevidence", "COIL_no_empirical_bayes", "COIL_no_conformal_calibration", "COIL_raw_evidence_only", "COIL_normal_only_calibration"]
    sub = metrics[metrics["method"].isin(methods)]
    if sub.empty:
        return "NOT_RUN"
    return dataframe_to_markdown(sub.groupby(["method", "protocol"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index().head(120))


def _tree_table(metrics: pd.DataFrame, baselines: pd.DataFrame) -> str:
    full = metrics[metrics["method"] == "COIL_full"]
    tree = baselines[baselines["method"].isin(["RandomForest", "XGBoost", "LightGBM"])]
    if full.empty or tree.empty:
        return "Tree comparison unavailable."
    fsum = full.groupby("protocol")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index().assign(method="COIL_full")
    tsum = tree.groupby(["protocol", "method"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index()
    return dataframe_to_markdown(pd.concat([fsum, tsum], ignore_index=True).head(100))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/coil_gate_v1_imadds_raw.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    output_dir = ROOT / cfg["project"]["output_dir"]
    packet_dir = ROOT / cfg["review_packet"]["output_dir"]
    _clean_dir(output_dir)
    generated_at = utc_now()
    raw_cfg = yaml.safe_load((ROOT / cfg["raw_windows"]["config"]).read_text(encoding="utf-8"))["dataset"]
    npz_path = ROOT / raw_cfg["processed_npz"]
    meta_path = ROOT / raw_cfg["metadata_csv"]
    if not npz_path.exists() or not meta_path.exists():
        build_imadds_raw_windows(ROOT / raw_cfg["raw_dir"], npz_path, meta_path, segment_length=int(raw_cfg.get("segment_length", 512)), window_size=int(raw_cfg.get("window_size", 128)), stride=int(raw_cfg.get("stride", 128)), max_train_source_segments=raw_cfg.get("max_train_source_segments", 600), max_train_target_segments=raw_cfg.get("max_train_target_segments"), seed=int(raw_cfg.get("seed", 7)))
    data = load_raw_windows(npz_path, meta_path)
    validity = _validity(data.meta, output_dir, generated_at)
    metrics, source, atom_summary, support, lattice, stability = _run_coil(data, cfg, output_dir, generated_at)
    baselines = _baseline_comparison(output_dir, generated_at)
    label_df = _run_label_efficiency(data, cfg, output_dir, generated_at)
    winners = _winner_summary(metrics, baselines, output_dir, generated_at)
    status, info = _decision(metrics, baselines, label_df)
    device = str(resolve_device(cfg.get("device", "auto_fastest")))
    _write_packet(cfg, output_dir, packet_dir, generated_at, status, info, data, validity, metrics, baselines, winners, label_df, source, atom_summary, support, lattice, stability, device)
    print(f"COIL Gate v1 finished: {status}; packet_files={len(list(packet_dir.iterdir()))}")


if __name__ == "__main__":
    main()

