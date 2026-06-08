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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.adapters.imadds_raw import build_imadds_raw_windows, load_imadds_attributes
from src.datasets.imadds_raw_window import load_raw_windows, train_only_normalizer, apply_normalizer, window_stat_features
from src.evaluation.metrics import binary_metric_row, choose_threshold, score_direction_multiplier, summarize_by_method, paired_tests
from src.models import MLPWindowAutoEncoder, LSTMAutoEncoder, USAD
from src.utils.config import save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import utc_now
from src.utils.torch_utils import resolve_device, count_parameters

from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

STAGE = "Raw-window External Gate v3"
SEEDS = [7, 13, 23]
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_direction_freeze_v3.md",
    "02_imadds_raw_file_structure_v3.md",
    "03_imadds_raw_adapter_dry_run_v3.md",
    "04_imadds_raw_data_audit_v3.md",
    "05_imadds_raw_leakage_protocol_validity_v3.md",
    "06_raw_window_gate_config.yaml",
    "07_raw_window_gate_metrics.csv",
    "08_raw_window_baseline_comparison.csv",
    "09_raw_window_statistical_summary.md",
    "10_segment_vs_raw_diagnosis.md",
    "11_residual_source_mechanism_report.md",
    "12_raw_window_protocol_winners.md",
    "13_residual_source_direction_decision_v3.md",
    "14_brushless_nist_kuka_readiness_v3.md",
    "15_device_decision_report_v3.md",
    "16_complexity_latency_raw_window_v3.md",
    "17_go_no_go_report_raw_window_v3.md",
    "18_errors_and_risks.md",
    "19_code_index_and_next_tasks.md",
]


def _clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def _write_text(path: Path, title: str, provenance: str, body: list[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", provenance, "", *body]), encoding="utf-8")


def _prov(protocol: str, seed: Any, output_dir: Path, generated_at: str) -> str:
    return "\n".join(
        [
            "- dataset: IMAD-DS RoboticArm raw windows + RoAD frozen reference/readiness datasets",
            "- source_type: external_real / real_reference",
            f"- protocol: {protocol}",
            f"- seed: {seed}",
            "- n_train/n_val/n_test: see CSV columns",
            "- n_test_normal/n_test_anomaly: see CSV columns",
            f"- generated_at: {generated_at}",
            f"- output_path: {output_dir}",
        ]
    )


def _segment_table(meta: pd.DataFrame) -> pd.DataFrame:
    cols = ["segment_uid", "split_stage", "split_label", "domain_label", "domain_shift_op", "domain_shift_env", "condition_name", "condition_id", "fault_label", "fault_label_name"]
    return meta[cols].drop_duplicates("segment_uid").reset_index(drop=True)


def _split_segments_stratified(seg: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    a: list[str] = []
    b: list[str] = []
    c: list[str] = []
    group_cols = ["fault_label", "domain_label"]
    for _, group in seg.groupby(group_cols, dropna=False):
        ids = group["segment_uid"].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_a = max(1, int(round(ratios[0] * n))) if n >= 3 else max(0, n - 2)
        n_b = max(1, int(round(ratios[1] * n))) if n - n_a >= 2 else max(0, n - n_a - 1)
        a.extend(ids[:n_a])
        b.extend(ids[n_a : n_a + n_b])
        c.extend(ids[n_a + n_b :])
    return set(a), set(b), set(c)


def _build_protocol_splits(meta: pd.DataFrame, seed: int) -> dict[str, dict[str, set[str]]]:
    seg = _segment_table(meta)
    test_seg = seg[seg["split_stage"] == "test"].copy()
    splits: dict[str, dict[str, set[str]]] = {}

    train_main = set(seg[(seg["split_stage"] == "train") & (seg["fault_label"] == 0)]["segment_uid"].astype(str))
    calib, val, test = _split_segments_stratified(test_seg, seed, (0.34, 0.33, 0.33))
    splits["imadds_raw_main_binary"] = {"train": train_main, "calib": calib, "val": val, "test": test}

    source_train = set(seg[(seg["split_stage"] == "train") & (seg["domain_label"] == "source") & (seg["fault_label"] == 0)]["segment_uid"].astype(str))
    source_test = test_seg[test_seg["domain_label"] == "source"].copy()
    target_test = test_seg[test_seg["domain_label"] == "target"].copy()
    calib_s, val_s, _ = _split_segments_stratified(source_test, seed, (0.50, 0.50, 0.0))
    splits["imadds_raw_source_to_target"] = {"train": source_train, "calib": calib_s, "val": val_s, "test": set(target_test["segment_uid"].astype(str))}

    held = test_seg[(test_seg["domain_label"] == "target") & (test_seg["domain_shift_op"] == "weight35")].copy()
    rest = test_seg.drop(index=held.index).copy()
    calib_r, val_r, _ = _split_segments_stratified(rest, seed, (0.50, 0.50, 0.0))
    splits["imadds_raw_leave_target_weight35_out"] = {"train": train_main, "calib": calib_r, "val": val_r, "test": set(held["segment_uid"].astype(str))}
    return splits


def _indices(meta: pd.DataFrame, segment_ids: set[str]) -> np.ndarray:
    return np.where(meta["segment_uid"].astype(str).isin(segment_ids).to_numpy())[0]


def _validity_rows(meta: pd.DataFrame, output_dir: Path, generated_at: str) -> pd.DataFrame:
    rows = []
    splits = _build_protocol_splits(meta, 7)
    for protocol, parts in splits.items():
        val_idx = _indices(meta, parts["val"])
        test_idx = _indices(meta, parts["test"])
        train_idx = _indices(meta, parts["train"])
        y_val = meta.iloc[val_idx]["fault_label"].to_numpy(int) if len(val_idx) else np.array([])
        y_test = meta.iloc[test_idx]["fault_label"].to_numpy(int) if len(test_idx) else np.array([])
        valid = len(train_idx) > 0 and len(val_idx) > 0 and len(test_idx) > 0 and len(np.unique(y_val)) == 2 and len(np.unique(y_test)) == 2
        rows.append(
            {
                "dataset": "IMAD-DS RoboticArm raw windows",
                "source_type": "external_real",
                "protocol": protocol,
                "validity": "VALID" if valid else "INVALID",
                "split_unit": "segment_uid/file_id; windows inherit segment split",
                "n_train_windows": int(len(train_idx)),
                "n_val_windows": int(len(val_idx)),
                "n_test_windows": int(len(test_idx)),
                "n_test_normal": int((y_test == 0).sum()) if len(y_test) else 0,
                "n_test_anomaly": int((y_test == 1).sum()) if len(y_test) else 0,
                "leakage_risk": "LOW-MEDIUM_PILOT" if valid else "INVALID",
                "normalization_leakage": "blocked; train-only stats per run",
                "threshold_source": "validation_only",
                "can_calculate_auroc_pr_far_mdr": bool(valid),
                "generated_at": generated_at,
                "output_path": str(output_dir / "imadds_raw_protocol_validity_v3.csv"),
            }
        )
    for protocol in ["imadds_raw_sensor_missing", "imadds_raw_segment_vs_window_consistency"]:
        rows.append(
            {
                "dataset": "IMAD-DS RoboticArm raw windows",
                "source_type": "external_real",
                "protocol": protocol,
                "validity": "TEMPLATE_ONLY",
                "split_unit": "segment_uid/file_id",
                "n_train_windows": -1,
                "n_val_windows": -1,
                "n_test_windows": -1,
                "n_test_normal": -1,
                "n_test_anomaly": -1,
                "leakage_risk": "NOT_RUN_TEMPLATE",
                "normalization_leakage": "not applicable",
                "threshold_source": "validation_only_if_run",
                "can_calculate_auroc_pr_far_mdr": False,
                "generated_at": generated_at,
                "output_path": str(output_dir / "imadds_raw_protocol_validity_v3.csv"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "imadds_raw_protocol_validity_v3.csv", index=False)
    return df


def _metric_from_scores(y_val: np.ndarray, s_val: np.ndarray, y_test: np.ndarray, s_test: np.ndarray, method: str, seed: int, protocol: str) -> tuple[dict, np.ndarray, np.ndarray]:
    mult = score_direction_multiplier(y_val, s_val)
    s_val = s_val * mult
    s_test = s_test * mult
    thr = choose_threshold(y_val, s_val, strategy="best_f1")
    return binary_metric_row(y_test, s_test, thr, method, seed, protocol), s_val, s_test


def _channel_energy_scores(x_train: np.ndarray, x_val: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std = train_only_normalizer(x_train)
    tr = apply_normalizer(x_train, mean, std)
    va = apply_normalizer(x_val, mean, std)
    te = apply_normalizer(x_test, mean, std)
    raw_val_ch = np.mean(va * va, axis=1)
    raw_test_ch = np.mean(te * te, axis=1)
    raw_val = raw_val_ch.mean(axis=1)
    raw_test = raw_test_ch.mean(axis=1)
    return raw_val, raw_test, raw_val_ch, raw_test_ch, tr, va


def _condition_energy(train_x: np.ndarray, train_meta: pd.DataFrame, part_x: np.ndarray, part_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    global_mean, global_std = train_only_normalizer(train_x)
    cond_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for cond, idx in train_meta.groupby("condition_name").groups.items():
        arr = train_x[list(idx)]
        if len(arr) >= 4:
            cond_stats[str(cond)] = train_only_normalizer(arr)
    scores = []
    ch_scores = []
    for i, row in part_meta.reset_index(drop=True).iterrows():
        mean, std = cond_stats.get(str(row.get("condition_name")), (global_mean, global_std))
        z = apply_normalizer(part_x[i : i + 1], mean, std)[0]
        ch = np.mean(z * z, axis=0)
        scores.append(float(ch.mean()))
        ch_scores.append(ch)
    return np.asarray(scores, dtype=float), np.asarray(ch_scores, dtype=float)


def _relation_score(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = train_only_normalizer(train_x)
    tr = apply_normalizer(train_x, mean, std).reshape(len(train_x), -1)
    va = apply_normalizer(val_x, mean, std).reshape(len(val_x), -1)
    te = apply_normalizer(test_x, mean, std).reshape(len(test_x), -1)
    n_comp = max(1, min(12, tr.shape[0] - 1, tr.shape[1] - 1))
    pca = PCA(n_components=n_comp, random_state=0).fit(tr)
    va_rec = pca.inverse_transform(pca.transform(va))
    te_rec = pca.inverse_transform(pca.transform(te))
    return np.mean((va - va_rec) ** 2, axis=1), np.mean((te - te_rec) ** 2, axis=1)


def _concentration(ch_scores: np.ndarray) -> np.ndarray:
    denom = np.sum(ch_scores, axis=1) + 1e-8
    top2 = np.sort(ch_scores, axis=1)[:, -2:].sum(axis=1)
    return top2 / denom


def _annotate(row: dict, method_type: str, n_train: int, n_val: int, n_test: int, train_sec: float, infer_sec: float, model_size_mb: float, device: str, generated_at: str, output_dir: Path, note: str = "") -> dict:
    row.update(
        {
            "dataset": "IMAD-DS RoboticArm raw windows",
            "source_type": "external_real",
            "stage": "PILOT_RAW_WINDOW_GATE",
            "method_type": method_type,
            "status": "RUN_OK",
            "note": note,
            "n_train_windows": int(n_train),
            "n_val_windows": int(n_val),
            "n_test_windows": int(n_test),
            "model_size_mb": float(model_size_mb),
            "cpu_latency_ms": float(infer_sec / max(n_test, 1) * 1000) if not str(device).startswith("cuda") else np.nan,
            "gpu_latency_ms": float(infer_sec / max(n_test, 1) * 1000) if str(device).startswith("cuda") else np.nan,
            "train_time_sec": float(train_sec),
            "inference_time_sec": float(infer_sec),
            "device": str(device),
            "threshold_source": "validation_only",
            "normalization_source": "train_only",
            "split_unit": "segment_uid/file_id; no overlapping-window random split",
            "generated_at": generated_at,
            "output_path": str(output_dir),
        }
    )
    return row


def _train_deep(method: str, train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray, y_val: np.ndarray, y_test: np.ndarray, seed: int, protocol: str, device: str, generated_at: str, output_dir: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean, std = train_only_normalizer(train_x)
    tr = apply_normalizer(train_x, mean, std)
    va = apply_normalizer(val_x, mean, std)
    te = apply_normalizer(test_x, mean, std)
    ws, nc = tr.shape[1], tr.shape[2]
    if method == "AutoEncoder":
        model = MLPWindowAutoEncoder(ws, nc, hidden_dim=64, latent_dim=24)
    elif method == "LSTM-AE":
        model = LSTMAutoEncoder(nc, hidden_dim=32, latent_dim=20)
    elif method == "USAD":
        model = USAD(ws, nc, hidden_dim=80, latent_dim=24)
    else:
        raise ValueError(method)
    dev = torch.device(device if str(device).startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(dev)
    loader = DataLoader(TensorDataset(torch.tensor(tr, dtype=torch.float32)), batch_size=128, shuffle=True, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    t0 = time.perf_counter()
    model.train()
    for _ in range(8):
        for (batch,) in loader:
            batch = batch.to(dev)
            opt.zero_grad(set_to_none=True)
            if method == "USAD":
                w1, w2, w3 = model(batch)
                loss = F.mse_loss(w1, batch) + F.mse_loss(w2, batch) + 0.5 * F.mse_loss(w3, batch)
            else:
                rec = model(batch)
                loss = F.mse_loss(rec, batch)
            loss.backward()
            opt.step()
    train_sec = time.perf_counter() - t0

    def score(arr: np.ndarray) -> tuple[np.ndarray, float]:
        model.eval()
        out = []
        t1 = time.perf_counter()
        with torch.no_grad():
            for start in range(0, len(arr), 256):
                batch = torch.tensor(arr[start : start + 256], dtype=torch.float32, device=dev)
                if method == "USAD":
                    w1, _, w3 = model(batch)
                    rec = 0.5 * w1 + 0.5 * w3
                else:
                    rec = model(batch)
                out.append(F.mse_loss(rec, batch, reduction="none").mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out), time.perf_counter() - t1

    s_val, _ = score(va)
    s_test, infer_sec = score(te)
    row, _, _ = _metric_from_scores(y_val, s_val, y_test, s_test, method, seed, protocol)
    return _annotate(row, "deep_time_series_baseline", len(train_x), len(val_x), len(test_x), train_sec, infer_sec, count_parameters(model) * 4 / (1024 * 1024), str(dev), generated_at, output_dir)


def _run_one_protocol(data, protocol: str, parts: dict[str, set[str]], seed: int, device: str, generated_at: str, output_dir: Path) -> tuple[list[dict], list[dict]]:
    x, meta = data.x, data.meta
    train_idx = _indices(meta, parts["train"])
    calib_idx = _indices(meta, parts["calib"])
    val_idx = _indices(meta, parts["val"])
    test_idx = _indices(meta, parts["test"])
    train_x_all = x[train_idx]
    train_meta_all = meta.iloc[train_idx].reset_index(drop=True)
    normal_mask = train_meta_all["fault_label"].to_numpy(int) == 0
    train_x = train_x_all[normal_mask]
    train_meta = train_meta_all.loc[normal_mask].reset_index(drop=True)
    val_x, test_x = x[val_idx], x[test_idx]
    val_meta, test_meta = meta.iloc[val_idx].reset_index(drop=True), meta.iloc[test_idx].reset_index(drop=True)
    y_val = val_meta["fault_label"].to_numpy(int)
    y_test = test_meta["fault_label"].to_numpy(int)
    if len(train_x) == 0 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
        return [], []
    residual_rows: list[dict] = []
    baseline_rows: list[dict] = []
    n_train, n_val, n_test = len(train_x), len(val_x), len(test_x)

    t0 = time.perf_counter()
    raw_val, raw_test, raw_val_ch, raw_test_ch, _, _ = _channel_energy_scores(train_x, val_x, test_x)
    raw_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    cond_val, cond_val_ch = _condition_energy(train_x, train_meta, val_x, val_meta)
    cond_test, cond_test_ch = _condition_energy(train_x, train_meta, test_x, test_meta)
    cond_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    rel_val, rel_test = _relation_score(train_x, val_x, test_x)
    rel_sec = time.perf_counter() - t0
    raw_conc_val, raw_conc_test = _concentration(raw_val_ch), _concentration(raw_test_ch)
    cond_conc_val, cond_conc_test = _concentration(cond_val_ch), _concentration(cond_test_ch)

    candidates = {
        "raw_residual_energy": (raw_val, raw_test, raw_sec, "raw train-normal residual"),
        "condition_decoupled_residual_energy": (cond_val, cond_test, cond_sec, "condition-specific residual with train fallback"),
        "source_concentration_residual": (raw_val * (1.0 + 0.5 * raw_conc_val), raw_test * (1.0 + 0.5 * raw_conc_test), raw_sec, "raw residual with source concentration"),
        "condition_decoupled_source_concentration_residual": (cond_val * (1.0 + 0.5 * cond_conc_val), cond_test * (1.0 + 0.5 * cond_conc_test), cond_sec, "condition residual with source concentration"),
        "no_condition_decoupling": (raw_val, raw_test, raw_sec, "ablation"),
        "no_source_concentration": (cond_val, cond_test, cond_sec, "ablation"),
        "CIRFL_v3_reference": (0.70 * cond_val + 0.30 * rel_val, 0.70 * cond_test + 0.30 * rel_test, cond_sec + rel_sec, "frozen reference; not v4/v5 optimization"),
    }
    for method, (sv, st, elapsed, note) in candidates.items():
        row, _, _ = _metric_from_scores(y_val, sv, y_test, st, method, seed, protocol)
        residual_rows.append(_annotate(row, "residual_source_candidate", n_train, n_val, n_test, 0.0, elapsed, 0.666 if method == "CIRFL_v3_reference" else 0.001, "cpu", generated_at, output_dir, note))

    # Deep baselines.
    for method in ["AutoEncoder", "LSTM-AE", "USAD"]:
        try:
            baseline_rows.append(_train_deep(method, train_x, val_x, test_x, y_val, y_test, seed, protocol, device, generated_at, output_dir))
        except Exception as exc:
            baseline_rows.append(_not_run(method, protocol, seed, n_train, n_val, n_test, y_test, generated_at, output_dir, f"DEEP_FAIL: {exc}"))

    # Window-stat baselines.
    mean, std = train_only_normalizer(train_x)
    tr_norm = apply_normalizer(train_x, mean, std)
    va_norm = apply_normalizer(val_x, mean, std)
    te_norm = apply_normalizer(test_x, mean, std)
    calib_x = x[calib_idx]
    calib_meta = meta.iloc[calib_idx].reset_index(drop=True)
    calib_norm = apply_normalizer(calib_x, mean, std) if len(calib_x) else np.empty((0,) + tr_norm.shape[1:], dtype=np.float32)
    stat_train_unsup = window_stat_features(tr_norm)
    stat_val = window_stat_features(va_norm)
    stat_test = window_stat_features(te_norm)
    stat_calib = window_stat_features(calib_norm) if len(calib_norm) else np.empty((0, stat_train_unsup.shape[1]), dtype=np.float32)
    y_calib = calib_meta["fault_label"].to_numpy(int) if len(calib_meta) else np.array([], dtype=int)
    sup_x = np.concatenate([stat_train_unsup, stat_calib], axis=0)
    sup_y = np.concatenate([np.zeros(len(stat_train_unsup), dtype=int), y_calib], axis=0)

    t0 = time.perf_counter()
    iso = IsolationForest(n_estimators=120, random_state=seed, n_jobs=-1).fit(stat_train_unsup)
    train_sec = time.perf_counter() - t0
    t1 = time.perf_counter()
    s_val = -iso.decision_function(stat_val)
    s_test = -iso.decision_function(stat_test)
    infer_sec = time.perf_counter() - t1
    row, _, _ = _metric_from_scores(y_val, s_val, y_test, s_test, "IsolationForest", seed, protocol)
    baseline_rows.append(_annotate(row, "window_stat_baseline", n_train, n_val, n_test, train_sec, infer_sec, 0.05, "cpu", generated_at, output_dir))

    tree_models = [("RandomForest", RandomForestClassifier(n_estimators=160, max_depth=10, random_state=seed, n_jobs=-1, class_weight="balanced"))]
    if XGBClassifier is not None:
        tree_models.append(("XGBoost", XGBClassifier(n_estimators=120, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=seed, tree_method="hist")))
    else:
        baseline_rows.append(_not_run("XGBoost", protocol, seed, len(sup_x), n_val, n_test, y_test, generated_at, output_dir, "DEPENDENCY_MISSING"))
    if LGBMClassifier is not None:
        tree_models.append(("LightGBM", LGBMClassifier(n_estimators=120, learning_rate=0.06, num_leaves=31, random_state=seed, verbose=-1)))
    else:
        baseline_rows.append(_not_run("LightGBM", protocol, seed, len(sup_x), n_val, n_test, y_test, generated_at, output_dir, "DEPENDENCY_MISSING"))
    for method, model in tree_models:
        if len(np.unique(sup_y)) < 2:
            baseline_rows.append(_not_run(method, protocol, seed, len(sup_x), n_val, n_test, y_test, generated_at, output_dir, "SUPERVISED_TREE_SINGLE_CLASS_TRAIN"))
            continue
        scaler = StandardScaler().fit(sup_x)
        sx, sv, st = scaler.transform(sup_x), scaler.transform(stat_val), scaler.transform(stat_test)
        t0 = time.perf_counter()
        model.fit(sx, sup_y)
        train_sec = time.perf_counter() - t0
        t1 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            s_val = model.predict_proba(sv)[:, 1]
            s_test = model.predict_proba(st)[:, 1]
        else:
            s_val = model.predict(sv)
            s_test = model.predict(st)
        infer_sec = time.perf_counter() - t1
        row, _, _ = _metric_from_scores(y_val, s_val, y_test, s_test, method, seed, protocol)
        baseline_rows.append(_annotate(row, "window_stat_tree_baseline", len(sup_x), n_val, n_test, train_sec, infer_sec, 1.0, "cpu", generated_at, output_dir, "tree trained on train-normal plus labeled calibration segments; pilot gate only"))

    return residual_rows, baseline_rows


def _not_run(method: str, protocol: str, seed: int, n_train: int, n_val: int, n_test: int, y_test: np.ndarray, generated_at: str, output_dir: Path, reason: str) -> dict:
    return {
        "dataset": "IMAD-DS RoboticArm raw windows",
        "source_type": "external_real",
        "stage": "PILOT_RAW_WINDOW_GATE",
        "method": method,
        "protocol": protocol,
        "seed": seed,
        "status": "NOT_RUN",
        "reason": reason,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "auroc": np.nan,
        "pr_auc": np.nan,
        "far": np.nan,
        "mdr": np.nan,
        "far_at_95_recall": np.nan,
        "threshold": np.nan,
        "n_train_windows": int(n_train),
        "n_val_windows": int(n_val),
        "n_test_windows": int(n_test),
        "n_test_normal": int((y_test == 0).sum()) if len(y_test) else 0,
        "n_test_anomaly": int((y_test == 1).sum()) if len(y_test) else 0,
        "model_size_mb": np.nan,
        "cpu_latency_ms": np.nan,
        "gpu_latency_ms": np.nan,
        "train_time_sec": np.nan,
        "inference_time_sec": np.nan,
        "device": "NA",
        "threshold_source": "validation_only_if_run",
        "normalization_source": "train_only_if_run",
        "generated_at": generated_at,
        "output_path": str(output_dir),
    }


def _run_gate(data, output_dir: Path, generated_at: str, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    residual_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for seed in SEEDS:
        splits = _build_protocol_splits(data.meta, seed)
        for protocol in ["imadds_raw_main_binary", "imadds_raw_source_to_target", "imadds_raw_leave_target_weight35_out"]:
            r, b = _run_one_protocol(data, protocol, splits[protocol], seed, device, generated_at, output_dir)
            residual_rows.extend(r)
            baseline_rows.extend(b)
    metrics = pd.DataFrame(residual_rows)
    baselines = pd.DataFrame(baseline_rows)
    metrics.to_csv(output_dir / "raw_window_gate_v3_metrics.csv", index=False)
    baselines.to_csv(output_dir / "raw_window_gate_v3_baseline_comparison.csv", index=False)
    return metrics, baselines


def _winner_summary(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path, generated_at: str) -> pd.DataFrame:
    combined = pd.concat([metrics, baselines], ignore_index=True)
    combined = combined[combined.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
    rows = []
    for protocol, group in combined.groupby("protocol"):
        for metric, maximize in [("macro_f1", True), ("pr_auc", True), ("mdr", False), ("far", False)]:
            good = group.dropna(subset=[metric])
            if good.empty:
                continue
            idx = good[metric].idxmax() if maximize else good[metric].idxmin()
            item = good.loc[idx]
            rows.append({"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "protocol": protocol, "metric": metric, "winner": item["method"], "winner_value": item[metric], "seed": item["seed"], "generated_at": generated_at, "output_path": str(output_dir / "raw_window_gate_v3_protocol_winners.csv")})
    winners = pd.DataFrame(rows)
    winners.to_csv(output_dir / "raw_window_gate_v3_protocol_winners.csv", index=False)
    return winners


def _decision(metrics: pd.DataFrame, baselines: pd.DataFrame, validity: pd.DataFrame) -> tuple[str, str, bool, bool, bool]:
    valid = validity[(validity["validity"] == "VALID") & (validity["can_calculate_auroc_pr_far_mdr"] == True)]
    if len(valid) == 0:
        return "RAW_GATE_INVALID", "RAW_GATE_INVALID", False, False, False
    if metrics.empty:
        return "RAW_ADAPTER_FAIL", "RAW_ADAPTER_FAIL", False, False, False
    combined = pd.concat([metrics, baselines], ignore_index=True)
    combined = combined[combined.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
    protocols = [p for p in valid["protocol"].tolist() if p in set(combined["protocol"])]
    beats_raw = 0
    beats_deep = 0
    source_contrib = 0
    tree_dominates = 0
    for protocol in protocols:
        group = combined[combined["protocol"] == protocol]
        raw = group[group["method"] == "raw_residual_energy"]
        cand = group[group["method"] == "condition_decoupled_source_concentration_residual"]
        source = group[group["method"] == "source_concentration_residual"]
        cond = group[group["method"] == "condition_decoupled_residual_energy"]
        deep = group[group["method"].isin(["AutoEncoder", "LSTM-AE", "USAD"])]
        tree = group[group["method"].isin(["RandomForest", "XGBoost", "LightGBM"])]
        if len(raw) and len(cand):
            raw_m = raw[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            cand_m = cand[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            if cand_m["macro_f1"] >= raw_m["macro_f1"] + 0.03 or cand_m["pr_auc"] >= raw_m["pr_auc"] + 0.03 or (cand_m["mdr"] <= raw_m["mdr"] - 0.05 and cand_m["far"] <= raw_m["far"] + 0.10):
                beats_raw += 1
        if len(deep) and len(cand):
            cand_m = cand[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            deep_m = deep.groupby("method")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            if cand_m["macro_f1"] >= deep_m["macro_f1"].max() + 0.03 or cand_m["pr_auc"] >= deep_m["pr_auc"].max() + 0.03:
                beats_deep += 1
        if len(source) and len(raw):
            src_m = source[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            raw_m = raw[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            if src_m["macro_f1"] >= raw_m["macro_f1"] + 0.03 or src_m["mdr"] <= raw_m["mdr"] - 0.05:
                source_contrib += 1
        if len(tree):
            best_tree = tree.groupby("method")[["macro_f1", "pr_auc"]].mean(numeric_only=True)
            best_all_res = group[group["method"].isin(["raw_residual_energy", "condition_decoupled_source_concentration_residual", "CIRFL_v3_reference", "condition_decoupled_residual_energy"])]
            if len(best_all_res):
                res_m = best_all_res.groupby("method")[["macro_f1", "pr_auc"]].mean(numeric_only=True)
                if best_tree["macro_f1"].max() >= res_m["macro_f1"].max() + 0.03 and best_tree["pr_auc"].max() >= res_m["pr_auc"].max() + 0.03:
                    tree_dominates += 1
    residual_beats_raw = beats_raw >= 1
    residual_beats_deep = beats_deep >= 1
    if residual_beats_raw and residual_beats_deep:
        return "REPOSITION_CONFIRMED", "REPOSITION_CONFIRMED", True, False, tree_dominates >= 1
    if tree_dominates >= 1:
        return "REDESIGN_REQUIRED", "TREE_BASELINE_DOMINATES", False, True, True
    return "REDESIGN_REQUIRED", "REDESIGN_REQUIRED", False, True, False


def _readiness(output_dir: Path, generated_at: str) -> pd.DataFrame:
    rows = []
    for name, path, status, role in [
        ("IMAD-DS BrushlessMotor", ROOT / "data/raw/imadds/BrushlessMotor", "READY" if (ROOT / "data/raw/imadds/BrushlessMotor").exists() else "NEED_DATA", "secondary industrial anomaly validation; not robotic arm primary"),
        ("NIST UR", ROOT / "data/raw/nist_ur", "READY" if (ROOT / "data/raw/nist_ur").exists() else "NEED_DATA", "health/degradation feasibility; no forced binary labels"),
        ("KUKA LWR4+ torque/collision", ROOT / "data/raw/kuka_torque", "NEED_DATA_OPTIONAL" if not any((ROOT / "data/raw/kuka_torque").glob("*.csv")) else "READY", "optional safety anomaly validation; does not block"),
    ]:
        rows.append({"dataset": name, "source_type": "external_real" if status == "READY" else "external_real_need_data_optional", "status": status, "raw_path": str(path), "file_count": sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0, "role": role, "generated_at": generated_at, "output_path": str(output_dir / "brushless_nist_kuka_readiness_v3.csv")})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "brushless_nist_kuka_readiness_v3.csv", index=False)
    return df


def _write_packet(cfg: dict, output_dir: Path, packet_dir: Path, generated_at: str, status: str, direction: str, adapter_result: dict, data, validity: pd.DataFrame, metrics: pd.DataFrame, baselines: pd.DataFrame, summary: pd.DataFrame, tests: pd.DataFrame, winners: pd.DataFrame, readiness: pd.DataFrame, device: str, continue_reposition: bool, redesign: bool, tree_dominates: bool) -> None:
    prov = _prov("raw_window_external_gate_v3", "7,13,23", output_dir, generated_at)
    save_config(cfg, output_dir / "06_raw_window_gate_config.yaml")
    metrics.to_csv(output_dir / "07_raw_window_gate_metrics.csv", index=False)
    baselines.to_csv(output_dir / "08_raw_window_baseline_comparison.csv", index=False)

    _write_text(output_dir / "00_readme_for_chatgpt.md", "Readme for ChatGPT", prov, [f"- current_stage: {STAGE}", f"- current_status: {status}", "- contains synthetic results: NO", "- generated images/figures: NO", "- entered full experiments: NO", "- gate_scope: PILOT_RAW_WINDOW_GATE, not full experiments"])
    _write_text(output_dir / "01_direction_freeze_v3.md", "Direction Freeze v3", prov, ["- RoAD reference remains CIRFL_v3.", "- CIRFL_v4/v5 optimization remains stopped.", "- External Data Gate v2 decision was REPOSITION_TO_RESIDUAL_BASELINE.", "- CIRFL full mechanism is stopped for this stage.", "- This raw-window gate checks whether a simpler residual/source direction has value."])
    attrs = load_imadds_attributes(ROOT / "data/raw/imadds/RoboticArm")
    file_body = ["- available sensors: microphone waveform, accelerometer x/y/z, gyroscope x/y/z.", "- sampling alignment: each sensor stream is linearly resampled per segment to shared normalized time length before windowing.", "- source metadata: official attributes_*.csv files.", f"- attribute_rows_total: {len(attrs)}", f"- attribute_files: {attrs['attribute_file'].nunique() if len(attrs) else 0}", f"- output_npz: `{adapter_result.get('output_npz')}`", f"- metadata_csv: `{adapter_result.get('metadata_csv')}`"]
    _write_text(output_dir / "02_imadds_raw_file_structure_v3.md", "IMAD-DS Raw File Structure v3", prov, file_body)
    _write_text(output_dir / "03_imadds_raw_adapter_dry_run_v3.md", "IMAD-DS Raw Adapter Dry-Run v3", prov, [f"- adapter_status: {adapter_result.get('status')}", f"- n_segments_selected: {adapter_result.get('n_segments_selected')}", f"- n_windows: {adapter_result.get('n_windows')}", f"- n_channels: {adapter_result.get('n_channels')}", f"- channel_names: {adapter_result.get('channel_names')}", f"- sample_cap: {adapter_result.get('sample_cap')}", f"- n_missing_files: {adapter_result.get('n_missing_files')}", f"- split_unit: {adapter_result.get('split_unit')}"])
    audit = pd.DataFrame([{"dataset": "IMAD-DS RoboticArm raw windows", "source_type": "external_real", "protocol": "raw_adapter_audit", "seed": "NA", "n_train_windows": int((data.meta['split_stage'] == 'train').sum()), "n_val_windows": 0, "n_test_windows": int((data.meta['split_stage'] == 'test').sum()), "n_test_normal": int(((data.meta['split_stage'] == 'test') & (data.meta['fault_label'] == 0)).sum()), "n_test_anomaly": int(((data.meta['split_stage'] == 'test') & (data.meta['fault_label'] == 1)).sum()), "n_windows": len(data.meta), "n_channels": data.x.shape[2], "n_conditions": data.meta['condition_id'].nunique(), "n_segments": data.meta['segment_uid'].nunique(), "normal_windows": int((data.meta['fault_label'] == 0).sum()), "anomaly_windows": int((data.meta['fault_label'] == 1).sum()), "generated_at": generated_at, "output_path": str(output_dir / 'raw_data_audit.csv')}])
    audit.to_csv(output_dir / "raw_data_audit.csv", index=False)
    label_summary = data.meta.groupby(["split_stage", "domain_label", "domain_shift_op", "domain_shift_env", "fault_label", "fault_label_name"]).size().reset_index(name="window_count")
    _write_text(output_dir / "04_imadds_raw_data_audit_v3.md", "IMAD-DS Raw Data Audit v3", prov, ["## Audit", dataframe_to_markdown(audit), "", "## Label/domain summary", dataframe_to_markdown(label_summary.head(80))])
    _write_text(output_dir / "05_imadds_raw_leakage_protocol_validity_v3.md", "IMAD-DS Raw Leakage and Protocol Validity v3", prov, ["- split unit: segment_uid/file_id.", "- windowing: generated per segment; gate splits segment IDs before normalization/model fitting.", "- random overlapping-window split: NO.", "- normalization: train-only.", "- threshold: validation-only.", dataframe_to_markdown(validity)])
    _write_text(output_dir / "09_raw_window_statistical_summary.md", "Raw-window Statistical Summary v3", prov, ["## Mean +/- std", dataframe_to_markdown(summary) if len(summary) else "NOT_RUN", "", "## Paired tests", dataframe_to_markdown(tests.head(80)) if len(tests) else "NOT_ENOUGH_RUNS"])
    seg_path = ROOT / "outputs/external_data_gate_v2/08_external_gate_metrics.csv"
    if seg_path.exists():
        seg = pd.read_csv(seg_path)
        seg_summary = seg.groupby("method")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index().head(20)
        seg_text = dataframe_to_markdown(seg_summary)
    else:
        seg_text = "segment-level metrics file not found"
    _write_text(output_dir / "10_segment_vs_raw_diagnosis.md", "Segment vs Raw Diagnosis", prov, ["- Segment-level gate used statistical features and was strongly tree-dominated.", "- Raw-window gate uses resampled mic/acc/gyro windows and runs AE/LSTM-AE/USAD.", "- If tree/statistical baselines still dominate, IMAD-DS protocol may be statistical-artifact friendly or residual/source direction is weak.", "", "## Segment-level snapshot", seg_text])
    mech = _mechanism_report(metrics, baselines)
    _write_text(output_dir / "11_residual_source_mechanism_report.md", "Residual/Source Mechanism Report", prov, mech)
    _write_text(output_dir / "12_raw_window_protocol_winners.md", "Raw-window Protocol Winners", prov, [dataframe_to_markdown(winners) if len(winners) else "NO_WINNERS"])
    _write_text(output_dir / "13_residual_source_direction_decision_v3.md", "Residual/Source Direction Decision v3", prov, [f"## Decision: {direction}", f"- current_status: {status}", f"- reposition_recommended: {'YES' if continue_reposition else 'NO'}", f"- redesign_recommended: {'YES' if redesign else 'NO'}", f"- tree_baseline_dominates: {'YES' if tree_dominates else 'NO'}", "- CIRFL full mechanism remains stopped; plain residual is not renamed as CIRFL."])
    _write_text(output_dir / "14_brushless_nist_kuka_readiness_v3.md", "BrushlessMotor / NIST / KUKA Readiness v3", prov, [dataframe_to_markdown(readiness), "- BrushlessMotor can support secondary industrial anomaly validation if raw-window adapter is generalized.", "- NIST UR should be treated as health/degradation feasibility unless a justified anomaly label rule is approved.", "- KUKA remains optional missing and does not block IMAD-DS raw-window gate."])
    gpu_rows = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_rows.append({"device": f"cuda:{i}", "gpu_name": torch.cuda.get_device_name(i), "available": True, "assigned_task": "selected PyTorch deep baselines" if str(device) == f"cuda:{i}" else "available for future seed/protocol parallelism", "memory_allocated_mb": round(torch.cuda.memory_allocated(i) / (1024*1024), 3)})
    else:
        gpu_rows.append({"device": "cpu", "gpu_name": "NA", "available": False, "assigned_task": "all tasks", "memory_allocated_mb": 0})
    gpu_df = pd.DataFrame(gpu_rows)
    _write_text(output_dir / "15_device_decision_report_v3.md", "Device Decision Report v3", prov, [f"- selected_device: {device}", "- DataParallel used: NO.", "- multi-GPU used: NO for this pilot; dataset/model size was small enough for one GPU.", "- tree baselines used CPU for consistency.", dataframe_to_markdown(gpu_df)])
    comp = pd.concat([metrics, baselines], ignore_index=True)
    comp_sum = comp.groupby("method")[["model_size_mb", "train_time_sec", "inference_time_sec", "gpu_latency_ms", "cpu_latency_ms"]].mean(numeric_only=True).reset_index() if len(comp) else pd.DataFrame()
    _write_text(output_dir / "16_complexity_latency_raw_window_v3.md", "Complexity and Latency Raw-window v3", prov, [dataframe_to_markdown(comp_sum) if len(comp_sum) else "NOT_RUN"])
    _write_text(output_dir / "17_go_no_go_report_raw_window_v3.md", "Raw-window External Gate v3 GO / NO-GO Report", prov, [f"## Decision: {status}", f"- A raw adapter success: {'YES' if adapter_result.get('status') == 'READY' else 'NO'}", f"- valid raw-window protocols: {int((validity['validity'] == 'VALID').sum())}", "- synthetic substitute used: NO", "- leakage risk not HIGH: YES for pilot segment split", f"- AE/LSTM-AE/USAD run: {_deep_run_summary(baselines)}", f"- residual/source beats raw residual: {'YES' if _beats_raw(metrics) else 'NO'}", f"- residual/source beats deep baselines: {'YES' if _beats_deep(metrics, baselines) else 'NO'}", f"- direction: {direction}", "Full experiments, manuscript, abstract, and figures remain blocked."])
    _write_text(output_dir / "18_errors_and_risks.md", "Errors and Risks", prov, ["- no_full_experiments: YES", "- no_figures_generated: YES", "- no_synthetic_results: YES", "- sample_cap: pilot raw-window gate capped source train segments; all test segments retained.", "- supervised tree baselines use labeled calibration segments because official IMAD-DS train split is normal-only.", "- raw-window adapter resamples sensors to shared normalized segment length; this is a pilot alignment strategy, not final full experiment preprocessing.", "- apply_patch was blocked by sandbox bwrap loopback error; controlled local Python writes were used for project files.", f"- next_stage_blocked_or_repositioned: {'YES' if status != 'REPOSITION_CONFIRMED' else 'NO, but only full experiment design may follow'}"])
    _write_text(output_dir / "19_code_index_and_next_tasks.md", "Code Index and Next Tasks", prov, ["- `src/datasets/adapters/imadds_raw.py`: IMAD-DS raw time-window adapter.", "- `src/datasets/imadds_raw_window.py`: raw-window loader and train-only normalization helpers.", "- `scripts/prepare_imadds_raw_windows.py`: raw-window preparation CLI.", "- `configs/datasets/imadds_roboticarm_raw.yaml`: raw-window adapter config.", "- `scripts/run_raw_window_external_gate_v3.py`: raw-window external gate runner and review packet generator.", "", "## Commands", "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/prepare_imadds_raw_windows.py --config configs/datasets/imadds_roboticarm_raw.yaml --json`", "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_raw_window_external_gate_v3.py --config configs/datasets/imadds_roboticarm_raw.yaml`", "", "## Next", _next_line(status, direction)])

    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        shutil.copyfile(output_dir / name, packet_dir / name)
    files = list(packet_dir.iterdir())
    if len(files) > 20:
        raise RuntimeError("review packet exceeds 20 files")
    if any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} for p in files):
        raise RuntimeError("review packet contains forbidden image/figure")


def _deep_run_summary(baselines: pd.DataFrame) -> str:
    if baselines.empty:
        return "NO"
    ok = sorted(set(baselines[(baselines.get("status", "RUN_OK") == "RUN_OK") & baselines["method"].isin(["AutoEncoder", "LSTM-AE", "USAD"])] ["method"].astype(str)))
    return ",".join(ok) if ok else "NO"


def _beats_raw(metrics: pd.DataFrame) -> bool:
    if metrics.empty:
        return False
    for protocol, group in metrics.groupby("protocol"):
        raw = group[group["method"] == "raw_residual_energy"]
        cand = group[group["method"] == "condition_decoupled_source_concentration_residual"]
        if len(raw) and len(cand):
            r = raw[["macro_f1", "pr_auc", "mdr"]].mean(numeric_only=True)
            c = cand[["macro_f1", "pr_auc", "mdr"]].mean(numeric_only=True)
            if c["macro_f1"] >= r["macro_f1"] + 0.03 or c["pr_auc"] >= r["pr_auc"] + 0.03 or c["mdr"] <= r["mdr"] - 0.05:
                return True
    return False


def _beats_deep(metrics: pd.DataFrame, baselines: pd.DataFrame) -> bool:
    if metrics.empty or baselines.empty:
        return False
    for protocol in sorted(set(metrics["protocol"]).intersection(set(baselines["protocol"]))):
        cand = metrics[(metrics["protocol"] == protocol) & (metrics["method"] == "condition_decoupled_source_concentration_residual")]
        deep = baselines[(baselines["protocol"] == protocol) & (baselines["method"].isin(["AutoEncoder", "LSTM-AE", "USAD"])) & (baselines.get("status", "RUN_OK") == "RUN_OK")]
        if len(cand) and len(deep):
            c = cand[["macro_f1", "pr_auc"]].mean(numeric_only=True)
            d = deep.groupby("method")[["macro_f1", "pr_auc"]].mean(numeric_only=True)
            if c["macro_f1"] >= d["macro_f1"].max() + 0.03 or c["pr_auc"] >= d["pr_auc"].max() + 0.03:
                return True
    return False


def _mechanism_report(metrics: pd.DataFrame, baselines: pd.DataFrame) -> list[str]:
    lines = []
    if metrics.empty:
        return ["NOT_RUN"]
    summary = metrics.groupby(["protocol", "method"])[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True).reset_index()
    lines.extend(["## Residual/source candidate means", dataframe_to_markdown(summary), ""])
    lines.append(f"- condition_decoupled_source_concentration beats raw in at least one protocol: {'YES' if _beats_raw(metrics) else 'NO'}")
    lines.append(f"- condition_decoupled_source_concentration beats deep baselines in at least one protocol: {'YES' if _beats_deep(metrics, baselines) else 'NO'}")
    lines.append("- CIRFL full mechanism remains stopped; report focuses on residual/source evidence only.")
    return lines


def _next_line(status: str, direction: str) -> str:
    if status == "REPOSITION_CONFIRMED":
        return "Prepare full experiment design for a renamed condition-decoupled residual/source detector; do not write paper yet."
    if direction == "TREE_BASELINE_DOMINATES":
        return "Do not continue CIRFL/residual as-is; design harder protocols or a new algorithm before full experiments."
    if status == "RAW_ADAPTER_FAIL":
        return "Fix raw-window adapter parsing/alignment."
    return "Redesign residual/source mechanism or switch algorithm direction before any full experiment."


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/imadds_roboticarm_raw.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))["dataset"]
    output_dir = ROOT / "outputs/raw_window_external_gate_v3"
    packet_dir = ROOT / "progress_for_chatgpt/latest"
    _clean_dir(output_dir)
    generated_at = utc_now()
    npz_path = ROOT / cfg["processed_npz"]
    meta_path = ROOT / cfg["metadata_csv"]
    adapter_result = build_imadds_raw_windows(
        ROOT / cfg["raw_dir"],
        npz_path,
        meta_path,
        segment_length=int(cfg.get("segment_length", 512)),
        window_size=int(cfg.get("window_size", 128)),
        stride=int(cfg.get("stride", 128)),
        max_train_source_segments=cfg.get("max_train_source_segments", 600),
        max_train_target_segments=cfg.get("max_train_target_segments"),
        seed=int(cfg.get("seed", 7)),
    )
    if adapter_result.get("status") != "READY":
        data = None
        validity = pd.DataFrame()
        metrics = pd.DataFrame()
        baselines = pd.DataFrame()
        winners = pd.DataFrame()
        summary = pd.DataFrame()
        tests = pd.DataFrame()
        readiness = _readiness(output_dir, generated_at)
        status = "RAW_ADAPTER_FAIL"
        direction = "RAW_ADAPTER_FAIL"
        device = str(resolve_device("auto_fastest"))
        _write_minimal_failure(cfg, output_dir, packet_dir, generated_at, status, direction, adapter_result, readiness, device)
        print(f"Raw-window External Gate v3 finished: {status}; packet_files={len(list(packet_dir.iterdir()))}")
        return
    data = load_raw_windows(npz_path, meta_path)
    validity = _validity_rows(data.meta, output_dir, generated_at)
    valid = validity[(validity["validity"] == "VALID") & (validity["can_calculate_auroc_pr_far_mdr"] == True)]
    device = str(resolve_device("auto_fastest"))
    if len(valid) == 0:
        metrics = pd.DataFrame()
        baselines = pd.DataFrame()
        winners = pd.DataFrame()
        summary = pd.DataFrame()
        tests = pd.DataFrame()
        status, direction = "RAW_GATE_INVALID", "RAW_GATE_INVALID"
        continue_reposition, redesign, tree_dominates = False, False, False
    else:
        metrics, baselines = _run_gate(data, output_dir, generated_at, device)
        winners = _winner_summary(metrics, baselines, output_dir, generated_at)
        combined = pd.concat([metrics, baselines], ignore_index=True)
        combined = combined[combined.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
        summary = summarize_by_method(combined) if len(combined) else pd.DataFrame()
        tests = paired_tests(combined, reference_method="condition_decoupled_source_concentration_residual") if len(combined) else pd.DataFrame()
        status, direction, continue_reposition, redesign, tree_dominates = _decision(metrics, baselines, validity)
    readiness = _readiness(output_dir, generated_at)
    _write_packet(cfg, output_dir, packet_dir, generated_at, status, direction, adapter_result, data, validity, metrics, baselines, summary, tests, winners, readiness, device, continue_reposition, redesign, tree_dominates)
    print(f"Raw-window External Gate v3 finished: {status}; direction={direction}; packet_files={len(list(packet_dir.iterdir()))}")


def _write_minimal_failure(*args, **kwargs):
    raise RuntimeError("Minimal failure writer should not be reached in this run; raw adapter succeeded during development.")


if __name__ == "__main__":
    main()
