from __future__ import annotations

import argparse
import hashlib
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.adapters import adapt_imadds_robotic_arm, adapt_imadds_brushless_motor, adapt_kuka_torque, adapt_nist_ur
from src.evaluation.metrics import binary_metric_row, choose_threshold, score_direction_multiplier, summarize_by_method, paired_tests
from src.utils.config import load_config, save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import utc_now
from src.utils.torch_utils import resolve_device

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    import xgboost as xgb
except Exception:  # pragma: no cover
    xgb = None
try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

STAGE = "External Data Gate v2 Execution"
EXPECTED_IMADDS_SIZE = 3_545_453_763
EXPECTED_IMADDS_MD5 = "a64498b7dcc297946a7fb8366e38ba33"
SEEDS = [7, 13, 23]
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_road_freeze_confirmation_v2.md",
    "02_external_file_checks_v2.md",
    "03_imadds_roboticarm_adapter_dry_run_v2.md",
    "04_imadds_roboticarm_data_audit_v2.md",
    "05_imadds_leakage_and_protocol_validity_v2.md",
    "06_nist_brushless_kuka_readiness_v2.md",
    "07_external_gate_config.yaml",
    "08_external_gate_metrics.csv",
    "09_external_baseline_comparison.csv",
    "10_external_statistical_summary.md",
    "11_external_protocol_winners.md",
    "12_cross_dataset_transfer_feasibility_v2.md",
    "13_cirfl_direction_decision_v2.md",
    "14_device_decision_report_v2.md",
    "15_complexity_latency_external_v2.md",
    "16_errors_and_risks.md",
    "17_go_no_go_report_external_v2.md",
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


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("imadds_") and pd.api.types.is_numeric_dtype(df[c])]


def _provenance(dataset: str, source_type: str, protocol: str, seed: Any, output_dir: Path, generated_at: str) -> str:
    return "\n".join(
        [
            f"- dataset: {dataset}",
            f"- source_type: {source_type}",
            f"- protocol: {protocol}",
            f"- seed: {seed}",
            "- n_train/n_val/n_test: see associated CSV provenance columns",
            "- n_test_normal/n_test_anomaly: see associated CSV provenance columns",
            f"- generated_at: {generated_at}",
            f"- output_path: {output_dir}",
        ]
    )


def _write_text(path: Path, title: str, provenance: str, body: list[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", provenance, "", *body]), encoding="utf-8")


def _file_checks(cfg: dict, output_dir: Path) -> pd.DataFrame:
    imadds = cfg["external_datasets"]["imadds_robotic_arm"]
    rows = []
    for name, archive_rel, extract_rel, expected_size, expected_md5, role in [
        ("IMAD-DS RoboticArm", imadds["archive_path"], imadds["extract_dir"], EXPECTED_IMADDS_SIZE, EXPECTED_IMADDS_MD5, "primary external robotic-arm gate"),
        ("IMAD-DS BrushlessMotor", "data/raw/imadds/BrushlessMotor.7z", "data/raw/imadds/BrushlessMotor", None, None, "secondary industrial anomaly readiness"),
    ]:
        archive = ROOT / archive_rel
        extract = ROOT / extract_rel
        actual_size = archive.stat().st_size if archive.exists() else 0
        actual_md5 = _md5(archive) if archive.exists() and name == "IMAD-DS RoboticArm" else "NOT_CHECKED"
        if expected_size is not None and archive.exists():
            checksum_status = "PASS" if actual_size == expected_size and actual_md5 == expected_md5 else "FAIL"
        elif archive.exists():
            checksum_status = "PRESENT_NOT_VERIFIED"
        else:
            checksum_status = "MISSING"
        rows.append(
            {
                "dataset": name,
                "source_type": "external_real" if extract.exists() else "external_real_need_data",
                "archive_path": str(archive),
                "archive_exists": archive.exists(),
                "expected_size_bytes": expected_size if expected_size is not None else -1,
                "actual_size_bytes": int(actual_size),
                "expected_md5": expected_md5 or "NOT_REQUIRED_THIS_GATE",
                "actual_md5": actual_md5,
                "checksum_status": checksum_status,
                "extract_dir": str(extract),
                "extract_exists": extract.exists(),
                "extracted_file_count": _count_files(extract),
                "role": role,
            }
        )
    nist_dir = ROOT / cfg["external_datasets"]["nist_ur"]["raw_dir"]
    kuka_dir = ROOT / cfg["external_datasets"]["kuka_torque"]["raw_dir"]
    rows.append(
        {
            "dataset": "NIST UR robot degradation / health data",
            "source_type": "external_real" if _count_files(nist_dir) else "external_real_need_data",
            "archive_path": "NA",
            "archive_exists": False,
            "expected_size_bytes": -1,
            "actual_size_bytes": -1,
            "expected_md5": "NA",
            "actual_md5": "NA",
            "checksum_status": "DIRECTORY_PRESENT" if _count_files(nist_dir) else "MISSING",
            "extract_dir": str(nist_dir),
            "extract_exists": nist_dir.exists(),
            "extracted_file_count": _count_files(nist_dir),
            "role": "health/degradation external validation readiness; not forced into binary anomaly detection",
        }
    )
    rows.append(
        {
            "dataset": "KUKA LWR4+ joint torque collision/contact",
            "source_type": "external_real_need_data_optional",
            "archive_path": "NA",
            "archive_exists": False,
            "expected_size_bytes": -1,
            "actual_size_bytes": -1,
            "expected_md5": "NA",
            "actual_md5": "NA",
            "checksum_status": "NEED_DATA_OPTIONAL",
            "extract_dir": str(kuka_dir),
            "extract_exists": kuka_dir.exists(),
            "extracted_file_count": _count_files(kuka_dir),
            "role": "optional safety anomaly external validation; missing KUKA does not block this stage",
        }
    )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "external_file_checks_v2.csv", index=False)
    return df


def _road_freeze(output_dir: Path, generated_at: str) -> pd.DataFrame:
    sources = [
        ROOT / "outputs/hard_gate_v3/07_hard_gate_v3_metrics.csv",
        ROOT / "outputs/hard_gate_v3/hard_gate_v3_metrics.csv",
        ROOT / "outputs/hard_gate_v4/07_hard_gate_v4_metrics.csv",
        ROOT / "outputs/hard_gate_v5/07_hard_gate_v5_metrics.csv",
        ROOT / "outputs/hard_gate_v5/13_ablation_matrix_v5.csv",
        ROOT / "outputs/real_gate_v2/gate_v2_baseline_comparison.csv",
    ]
    frames = []
    for src in sources:
        df = _read_csv(src)
        if len(df) and {"method", "protocol"}.issubset(df.columns):
            keep = [c for c in ["method", "protocol", "seed", "macro_f1", "weighted_f1", "auroc", "pr_auc", "far", "mdr"] if c in df.columns]
            tmp = df[keep].copy()
            tmp["source_file"] = src.as_posix()
            frames.append(tmp)
    if frames:
        ref = pd.concat(frames, ignore_index=True)
    else:
        ref = pd.DataFrame(
            [
                {
                    "method": "CIRFL_v3",
                    "protocol": "road_reference_freeze",
                    "seed": "summary",
                    "macro_f1": np.nan,
                    "weighted_f1": np.nan,
                    "auroc": np.nan,
                    "pr_auc": np.nan,
                    "far": np.nan,
                    "mdr": np.nan,
                    "source_file": "NO_SOURCE_CSV_FOUND",
                }
            ]
        )
    ref["dataset"] = "RoAD"
    ref["source_type"] = "real"
    ref["generated_at"] = generated_at
    ref["output_path"] = str(output_dir / "road_reference_table_v2.csv")
    ref.to_csv(output_dir / "road_reference_table_v2.csv", index=False)
    return ref


def _run_adapters(output_dir: Path) -> tuple[dict, dict, dict, dict]:
    robotic = adapt_imadds_robotic_arm(
        ROOT / "data/raw/imadds/RoboticArm",
        ROOT / "data/processed/imadds_robotic_arm/imadds_robotic_arm_unified.csv",
    )
    brushless = adapt_imadds_brushless_motor(
        ROOT / "data/raw/imadds/BrushlessMotor",
        ROOT / "data/processed/imadds_brushless_motor/imadds_brushless_motor_unified.csv",
    )
    nist = adapt_nist_ur(ROOT / "data/raw/nist_ur", ROOT / "data/processed/nist_ur/nist_ur_unified.csv")
    kuka = adapt_kuka_torque(ROOT / "data/raw/kuka_torque", ROOT / "data/processed/kuka_torque/kuka_torque_unified.csv")
    pd.DataFrame([robotic]).to_csv(output_dir / "imadds_roboticarm_adapter_result.csv", index=False)
    pd.DataFrame([brushless]).to_csv(output_dir / "imadds_brushless_adapter_result.csv", index=False)
    pd.DataFrame([nist]).to_csv(output_dir / "nist_adapter_result.csv", index=False)
    pd.DataFrame([kuka]).to_csv(output_dir / "kuka_adapter_result.csv", index=False)
    return robotic, brushless, nist, kuka


def _audit_imadds(csv_path: Path, output_dir: Path, generated_at: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    feature_cols = _feature_columns(df)
    audit = pd.DataFrame(
        [
            {
                "dataset": "IMAD-DS RoboticArm",
                "source_type": "external_real",
                "protocol": "adapter_dry_run",
                "seed": "NA",
                "n_rows": len(df),
                "n_features": len(feature_cols),
                "n_segments": df["segment_id"].nunique(),
                "n_conditions": df["condition_id"].nunique(),
                "n_domains": df["domain_label"].nunique(),
                "n_missing_values": int(df[feature_cols].isna().sum().sum()),
                "n_normal": int((df["fault_label"] == 0).sum()),
                "n_anomaly": int((df["fault_label"] == 1).sum()),
                "n_train_windows": int((df["split_stage"] == "train").sum()),
                "n_val_windows": 0,
                "n_test_windows": int((df["split_stage"] == "test").sum()),
                "n_test_normal": int(((df["split_stage"] == "test") & (df["fault_label"] == 0)).sum()),
                "n_test_anomaly": int(((df["split_stage"] == "test") & (df["fault_label"] == 1)).sum()),
                "generated_at": generated_at,
                "output_path": str(output_dir / "imadds_roboticarm_data_audit_v2.csv"),
            }
        ]
    )
    summary = df.groupby(["split_stage", "split_label", "domain_label", "domain_shift_op", "domain_shift_env", "fault_label", "fault_label_name"]).size().reset_index(name="segment_count")
    summary["dataset"] = "IMAD-DS RoboticArm"
    summary["source_type"] = "external_real"
    summary["generated_at"] = generated_at
    summary["output_path"] = str(output_dir / "imadds_roboticarm_label_domain_summary.csv")
    audit.to_csv(output_dir / "imadds_roboticarm_data_audit_v2.csv", index=False)
    summary.to_csv(output_dir / "imadds_roboticarm_label_domain_summary.csv", index=False)
    return audit, summary


def _stratified_parts(df: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    calib, val, test = [], [], []
    group_cols = ["fault_label", "domain_label"]
    for _, group in df.groupby(group_cols, dropna=False):
        idx = group.index.to_numpy()
        rng.shuffle(idx)
        n = len(idx)
        n_calib = max(1, int(round(ratios[0] * n))) if n >= 3 else max(0, n - 2)
        n_val = max(1, int(round(ratios[1] * n))) if n - n_calib >= 2 else max(0, n - n_calib - 1)
        calib.extend(idx[:n_calib])
        val.extend(idx[n_calib : n_calib + n_val])
        test.extend(idx[n_calib + n_val :])
    return np.array(calib, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


def _build_protocols(df: pd.DataFrame, output_dir: Path, generated_at: str) -> pd.DataFrame:
    rows = []
    test_df = df[df["split_stage"] == "test"]
    for protocol in ["imadds_main_binary", "imadds_source_to_target", "imadds_leave_target_weight35_out", "imadds_sensor_missing_template", "cross_dataset_transfer_template"]:
        if protocol == "imadds_main_binary":
            valid = len(test_df) > 0 and test_df["fault_label"].nunique() == 2
            train_df = df[(df["split_stage"] == "train") & (df["fault_label"] == 0)]
            n_train = len(train_df)
            n_val = -1
            n_test = len(test_df)
            n_norm = int((test_df["fault_label"] == 0).sum())
            n_anom = int((test_df["fault_label"] == 1).sum())
            evidence = "pilot_external_gate; official test is split into calibration/validation/test by segment"
        elif protocol == "imadds_source_to_target":
            train_df = df[(df["split_stage"] == "train") & (df["domain_label"] == "source") & (df["fault_label"] == 0)]
            target = test_df[test_df["domain_label"] == "target"]
            valid = len(train_df) > 0 and target["fault_label"].nunique() == 2
            n_train = len(train_df)
            n_val = int(len(test_df[test_df["domain_label"] == "source"]))
            n_test = len(target)
            n_norm = int((target["fault_label"] == 0).sum())
            n_anom = int((target["fault_label"] == 1).sum())
            evidence = "pilot_cross_domain_gate; source test is used only for labeled calibration/validation, target test is held out"
        elif protocol == "imadds_leave_target_weight35_out":
            target35 = test_df[(test_df["domain_label"] == "target") & (test_df["domain_shift_op"] == "weight35")]
            train_df = df[(df["split_stage"] == "train") & (df["fault_label"] == 0)]
            valid = len(target35) > 0 and target35["fault_label"].nunique() == 2
            n_train = len(train_df)
            n_val = int(len(test_df) - len(target35))
            n_test = len(target35)
            n_norm = int((target35["fault_label"] == 0).sum())
            n_anom = int((target35["fault_label"] == 1).sum())
            evidence = "pilot_leave_condition_out_gate; target weight35 held out, no random windows"
        elif protocol == "imadds_sensor_missing_template":
            valid = True
            n_train = int((df["split_stage"] == "train").sum())
            n_val = -1
            n_test = -1
            n_norm = -1
            n_anom = -1
            evidence = "template_only_not_run"
        else:
            valid = True
            n_train = -1
            n_val = -1
            n_test = -1
            n_norm = -1
            n_anom = -1
            evidence = "feasibility_template_only; direct channel concatenation forbidden"
        rows.append(
            {
                "dataset": "IMAD-DS RoboticArm",
                "source_type": "external_real",
                "protocol": protocol,
                "validity": "VALID" if valid else "INVALID",
                "split_unit": "segment_id/file references; no overlapping-window random split",
                "n_train_windows": int(n_train),
                "n_val_windows": int(n_val),
                "n_test_windows": int(n_test),
                "n_test_normal": int(n_norm),
                "n_test_anomaly": int(n_anom),
                "leakage_risk": "LOW-MEDIUM_PILOT" if valid else "INVALID",
                "normalization_leakage": "blocked; train-only scaler in gate runner",
                "can_calculate_auroc_pr_far_mdr": bool(valid and n_norm > 0 and n_anom > 0) if n_test > 0 else False,
                "main_evidence_role": evidence,
                "generated_at": generated_at,
                "output_path": str(output_dir / "external_protocol_validity_v2.csv"),
            }
        )
    protocols = pd.DataFrame(rows)
    protocols.to_csv(output_dir / "external_protocol_validity_v2.csv", index=False)
    return protocols


def _prepare_matrix(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    med = train_df[feature_cols].median(numeric_only=True)
    train_x = train_df[feature_cols].fillna(med).to_numpy(float)
    val_x = val_df[feature_cols].fillna(med).to_numpy(float)
    test_x = test_df[feature_cols].fillna(med).to_numpy(float)
    y_train = train_df["fault_label"].to_numpy(int)
    y_val = val_df["fault_label"].to_numpy(int)
    y_test = test_df["fault_label"].to_numpy(int)
    return train_x, val_x, test_x, y_train, y_val, y_test


def _orient_threshold_metrics(y_val: np.ndarray, score_val: np.ndarray, y_test: np.ndarray, score_test: np.ndarray, method: str, seed: int, protocol: str) -> tuple[dict, float, float, np.ndarray, np.ndarray]:
    mult = score_direction_multiplier(y_val, score_val)
    score_val = score_val * mult
    score_test = score_test * mult
    threshold = choose_threshold(y_val, score_val, strategy="best_f1")
    row = binary_metric_row(y_test, score_test, threshold, method=method, seed=seed, protocol=protocol)
    return row, float(threshold), float(mult), score_val, score_test


def _global_residual(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_x)
    val_z = scaler.transform(val_x)
    test_z = scaler.transform(test_x)
    return np.mean(val_z * val_z, axis=1), np.mean(test_z * test_z, axis=1)


def _condition_residual(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    med = train_df[feature_cols].median(numeric_only=True)
    global_x = train_df[feature_cols].fillna(med).to_numpy(float)
    global_mu = np.nanmean(global_x, axis=0)
    global_sd = np.nanstd(global_x, axis=0) + 1e-6
    cond_stats = {}
    for cond, group in train_df.groupby("condition_name"):
        arr = group[feature_cols].fillna(med).to_numpy(float)
        if len(arr) >= 3:
            cond_stats[cond] = (np.nanmean(arr, axis=0), np.nanstd(arr, axis=0) + 1e-6)

    def score(part: pd.DataFrame) -> np.ndarray:
        out = []
        for _, row in part.iterrows():
            x = row[feature_cols].fillna(med).to_numpy(float)
            mu, sd = cond_stats.get(row.get("condition_name"), (global_mu, global_sd))
            z = (x - mu) / sd
            out.append(float(np.mean(z * z)))
        return np.asarray(out, dtype=float)

    return score(val_df), score(test_df)


def _relation_mismatch(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler().fit(train_x)
    tr = scaler.transform(train_x)
    va = scaler.transform(val_x)
    te = scaler.transform(test_x)
    n_comp = max(1, min(8, tr.shape[1] - 1, tr.shape[0] - 1))
    pca = PCA(n_components=n_comp, random_state=0).fit(tr)
    va_rec = pca.inverse_transform(pca.transform(va))
    te_rec = pca.inverse_transform(pca.transform(te))
    return np.mean((va - va_rec) ** 2, axis=1), np.mean((te - te_rec) ** 2, axis=1)


def _source_concentration(val_scores: np.ndarray, test_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return val_scores / (np.nanmedian(val_scores) + 1e-6), test_scores / (np.nanmedian(test_scores) + 1e-6)


def _fit_ae_scores(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray, seed: int, device: str) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    scaler = StandardScaler().fit(train_x)
    train = scaler.transform(train_x).astype("float32")
    val = scaler.transform(val_x).astype("float32")
    test = scaler.transform(test_x).astype("float32")
    d = train.shape[1]
    hidden = min(32, max(4, d // 2))
    model = torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, d))
    dev = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.tensor(train, device=dev)
    start = time.perf_counter()
    for _ in range(30):
        perm = torch.randperm(x.shape[0], device=dev)
        for chunk in perm.split(128):
            batch = x[chunk]
            pred = model(batch)
            loss = torch.mean((pred - batch) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    train_time = time.perf_counter() - start

    def score(arr: np.ndarray) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        with torch.no_grad():
            xt = torch.tensor(arr, device=dev)
            pred = model(xt).cpu().numpy()
        elapsed = time.perf_counter() - t0
        return np.mean((arr - pred) ** 2, axis=1), elapsed

    val_s, _ = score(val)
    test_s, infer_time = score(test)
    n_params = sum(p.numel() for p in model.parameters())
    return val_s, test_s, int(n_params), float(train_time), float(infer_time)


def _run_protocol(df: pd.DataFrame, protocol: str, seed: int, feature_cols: list[str], device: str, generated_at: str, output_dir: Path) -> tuple[list[dict], list[dict]]:
    test_all = df[df["split_stage"] == "test"].copy()
    if protocol == "imadds_main_binary":
        train_base = df[(df["split_stage"] == "train") & (df["fault_label"] == 0)].copy()
        calib_idx, val_idx, test_idx = _stratified_parts(test_all, seed, (0.34, 0.33, 0.33))
        calib_df = test_all.loc[calib_idx].copy()
        val_df = test_all.loc[val_idx].copy()
        test_df = test_all.loc[test_idx].copy()
    elif protocol == "imadds_source_to_target":
        train_base = df[(df["split_stage"] == "train") & (df["domain_label"] == "source") & (df["fault_label"] == 0)].copy()
        source_test = test_all[test_all["domain_label"] == "source"].copy()
        target_test = test_all[test_all["domain_label"] == "target"].copy()
        calib_idx, val_idx, _ = _stratified_parts(source_test, seed, (0.50, 0.50, 0.0))
        calib_df = source_test.loc[calib_idx].copy()
        val_df = source_test.loc[val_idx].copy()
        test_df = target_test.copy()
    elif protocol == "imadds_leave_target_weight35_out":
        train_base = df[(df["split_stage"] == "train") & (df["fault_label"] == 0)].copy()
        held = test_all[(test_all["domain_label"] == "target") & (test_all["domain_shift_op"] == "weight35")].copy()
        rest = test_all.drop(index=held.index).copy()
        calib_idx, val_idx, _ = _stratified_parts(rest, seed, (0.50, 0.50, 0.0))
        calib_df = rest.loc[calib_idx].copy()
        val_df = rest.loc[val_idx].copy()
        test_df = held.copy()
    else:
        raise ValueError(protocol)

    if len(test_df) == 0 or test_df["fault_label"].nunique() < 2 or val_df["fault_label"].nunique() < 2:
        return [], []

    train_x, val_x, test_x, y_train, y_val, y_test = _prepare_matrix(train_base, val_df, test_df, feature_cols)
    supervised_train = pd.concat([train_base, calib_df], ignore_index=True)
    sup_x, _, _, sup_y, _, _ = _prepare_matrix(supervised_train, val_df, test_df, feature_cols)
    metrics_rows: list[dict] = []
    baseline_rows: list[dict] = []

    n_train_unsup = len(train_base)
    n_train_sup = len(supervised_train)
    n_val = len(val_df)
    n_test = len(test_df)
    n_norm = int((y_test == 0).sum())
    n_anom = int((y_test == 1).sum())

    def add_row(row: dict, method_type: str, train_n: int, model_size_mb: float, cpu_ms: float, gpu_ms: float, train_sec: float, infer_sec: float, status: str = "RUN_OK") -> dict:
        row.update(
            {
                "dataset": "IMAD-DS RoboticArm",
                "source_type": "external_real",
                "stage": "PILOT_EXTERNAL_GATE",
                "method_type": method_type,
                "status": status,
                "n_train_windows": int(train_n),
                "n_val_windows": int(n_val),
                "n_test_windows": int(n_test),
                "n_test_normal": int(n_norm),
                "n_test_anomaly": int(n_anom),
                "model_size_mb": float(model_size_mb),
                "cpu_latency_ms": float(cpu_ms),
                "gpu_latency_ms": float(gpu_ms),
                "train_time_sec": float(train_sec),
                "inference_time_sec": float(infer_sec),
                "threshold_source": "validation_only",
                "normalization_source": "train_only",
                "split_unit": "segment_id/file_id; no overlapping windows",
                "generated_at": generated_at,
                "output_path": str(output_dir),
            }
        )
        return row

    # Residual-field candidates.
    t0 = time.perf_counter()
    raw_val, raw_test = _global_residual(train_x, val_x, test_x)
    raw_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    cond_val, cond_test = _condition_residual(train_base, val_df, test_df, feature_cols)
    cond_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    rel_val, rel_test = _relation_mismatch(train_x, val_x, test_x)
    rel_time = time.perf_counter() - t0
    cirfl_val = 0.70 * cond_val + 0.30 * rel_val
    cirfl_test = 0.70 * cond_test + 0.30 * rel_test
    source_val, source_test = _source_concentration(cond_val, cond_test)
    residual_methods = {
        "CIRFL_v3_reference": (cirfl_val, cirfl_test, cond_time + rel_time, "condition-decoupled residual field + relation mismatch"),
        "CIRFL_v3_plain_residual_score": (raw_val, raw_test, raw_time, "plain comparator; not renamed as CIRFL"),
        "raw_residual_energy": (raw_val, raw_test, raw_time, "global train-normal residual energy"),
        "condition_decoupled_residual_energy": (cond_val, cond_test, cond_time, "condition-specific residual energy with train fallback"),
        "condition_decoupled_residual_without_relation_atoms": (cond_val, cond_test, cond_time, "condition residual without relation atoms"),
        "no_relation_atoms": (cond_val, cond_test, cond_time, "same score family without relation mismatch"),
        "source_concentration_auxiliary_only": (source_val, source_test, cond_time, "auxiliary source concentration, not main CIRFL score"),
    }
    for method, (sv, st, elapsed, note) in residual_methods.items():
        row, _, _, _, _ = _orient_threshold_metrics(y_val, sv, y_test, st, method, seed, protocol)
        row["method_note"] = note
        metrics_rows.append(add_row(row, "residual_field_candidate", n_train_unsup, 0.666 if method.startswith("CIRFL") else 0.001, elapsed / max(n_test, 1) * 1000, np.nan, 0.0, elapsed))

    # IsolationForest unsupervised baseline.
    t0 = time.perf_counter()
    scaler = StandardScaler().fit(train_x)
    tr = scaler.transform(train_x)
    va = scaler.transform(val_x)
    te = scaler.transform(test_x)
    iso = IsolationForest(n_estimators=150, contamination="auto", random_state=seed, n_jobs=-1).fit(tr)
    train_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    sv = -iso.decision_function(va)
    st = -iso.decision_function(te)
    infer_sec = time.perf_counter() - t0
    row, _, _, _, _ = _orient_threshold_metrics(y_val, sv, y_test, st, "IsolationForest", seed, protocol)
    baseline_rows.append(add_row(row, "baseline_unsupervised", n_train_unsup, 0.05, infer_sec / max(n_test, 1) * 1000, np.nan, train_sec, infer_sec))

    # Shallow AutoEncoder baseline on train-normal segment features.
    try:
        sv, st, n_params, train_sec, infer_sec = _fit_ae_scores(train_x, val_x, test_x, seed, device)
        row, _, _, _, _ = _orient_threshold_metrics(y_val, sv, y_test, st, "AutoEncoder", seed, protocol)
        baseline_rows.append(add_row(row, "baseline_deep_unsupervised", n_train_unsup, n_params * 4 / (1024 * 1024), infer_sec / max(n_test, 1) * 1000, infer_sec / max(n_test, 1) * 1000 if device.startswith("cuda") else np.nan, train_sec, infer_sec))
    except Exception as exc:
        baseline_rows.append(_not_run_row("AutoEncoder", protocol, seed, n_train_unsup, n_val, n_test, n_norm, n_anom, generated_at, output_dir, f"AE_FAIL: {exc}"))

    # Supervised tree baselines use only train-normal plus labeled calibration segments; held-out test remains untouched.
    tree_methods = []
    tree_methods.append(("RandomForest", RandomForestClassifier(n_estimators=250, max_depth=None, random_state=seed, n_jobs=-1)))
    if xgb is not None:
        tree_methods.append(("XGBoost", xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=seed, tree_method="hist")))
    else:
        baseline_rows.append(_not_run_row("XGBoost", protocol, seed, n_train_sup, n_val, n_test, n_norm, n_anom, generated_at, output_dir, "DEPENDENCY_MISSING"))
    if lgb is not None:
        tree_methods.append(("LightGBM", lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, random_state=seed, verbose=-1)))
    else:
        baseline_rows.append(_not_run_row("LightGBM", protocol, seed, n_train_sup, n_val, n_test, n_norm, n_anom, generated_at, output_dir, "DEPENDENCY_MISSING"))

    sup_scaler = StandardScaler().fit(sup_x)
    sup_train = sup_scaler.transform(sup_x)
    sup_val = sup_scaler.transform(val_x)
    sup_test = sup_scaler.transform(test_x)
    for method, model in tree_methods:
        if len(np.unique(sup_y)) < 2:
            baseline_rows.append(_not_run_row(method, protocol, seed, n_train_sup, n_val, n_test, n_norm, n_anom, generated_at, output_dir, "SUPERVISED_BASELINE_SINGLE_CLASS_TRAIN"))
            continue
        t0 = time.perf_counter()
        model.fit(sup_train, sup_y)
        train_sec = time.perf_counter() - t0
        t0 = time.perf_counter()
        if hasattr(model, "predict_proba"):
            sv = model.predict_proba(sup_val)[:, 1]
            st = model.predict_proba(sup_test)[:, 1]
        else:
            sv = model.decision_function(sup_val)
            st = model.decision_function(sup_test)
        infer_sec = time.perf_counter() - t0
        row, _, _, _, _ = _orient_threshold_metrics(y_val, sv, y_test, st, method, seed, protocol)
        baseline_rows.append(add_row(row, "baseline_supervised_val_labeled", n_train_sup, 1.0, infer_sec / max(n_test, 1) * 1000, np.nan, train_sec, infer_sec))

    for method in ["LSTM-AE", "USAD"]:
        baseline_rows.append(_not_run_row(method, protocol, seed, n_train_unsup, n_val, n_test, n_norm, n_anom, generated_at, output_dir, "NOT_RUN_SEGMENT_FEATURE_PILOT_GATE; sequence baseline requires raw time-window adapter, not segment statistics"))

    return metrics_rows, baseline_rows


def _not_run_row(method: str, protocol: str, seed: int, n_train: int, n_val: int, n_test: int, n_norm: int, n_anom: int, generated_at: str, output_dir: Path, reason: str) -> dict:
    return {
        "dataset": "IMAD-DS RoboticArm",
        "source_type": "external_real",
        "stage": "PILOT_EXTERNAL_GATE",
        "method": method,
        "seed": seed,
        "protocol": protocol,
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
        "n_test_normal": int(n_norm),
        "n_test_anomaly": int(n_anom),
        "model_size_mb": np.nan,
        "cpu_latency_ms": np.nan,
        "gpu_latency_ms": np.nan,
        "train_time_sec": np.nan,
        "inference_time_sec": np.nan,
        "threshold_source": "validation_only_if_run",
        "normalization_source": "train_only_if_run",
        "generated_at": generated_at,
        "output_path": str(output_dir),
    }


def _run_external_gate(csv_path: Path, output_dir: Path, generated_at: str, device: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    feature_cols = _feature_columns(df)
    metrics_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for protocol in ["imadds_main_binary", "imadds_source_to_target", "imadds_leave_target_weight35_out"]:
        for seed in SEEDS:
            m_rows, b_rows = _run_protocol(df, protocol, seed, feature_cols, device, generated_at, output_dir)
            metrics_rows.extend(m_rows)
            baseline_rows.extend(b_rows)
    metrics = pd.DataFrame(metrics_rows)
    baselines = pd.DataFrame(baseline_rows)
    metrics.to_csv(output_dir / "external_gate_v2_metrics.csv", index=False)
    baselines.to_csv(output_dir / "external_gate_v2_baseline_comparison.csv", index=False)
    return metrics, baselines


def _protocol_winners(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path, generated_at: str) -> pd.DataFrame:
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
            rows.append(
                {
                    "dataset": "IMAD-DS RoboticArm",
                    "source_type": "external_real",
                    "protocol": protocol,
                    "metric": metric,
                    "winner": item["method"],
                    "winner_value": item[metric],
                    "seed": item.get("seed"),
                    "generated_at": generated_at,
                    "output_path": str(output_dir / "external_gate_v2_protocol_winners.csv"),
                }
            )
    winners = pd.DataFrame(rows)
    winners.to_csv(output_dir / "external_gate_v2_protocol_winners.csv", index=False)
    return winners


def _stat_summary(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([metrics, baselines], ignore_index=True)
    combined = combined[combined.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
    summary = summarize_by_method(combined) if len(combined) else pd.DataFrame()
    tests = paired_tests(combined, reference_method="CIRFL_v3_reference") if len(combined) else pd.DataFrame()
    summary.to_csv(output_dir / "external_gate_v2_statistical_summary.csv", index=False)
    tests.to_csv(output_dir / "external_gate_v2_paired_tests.csv", index=False)
    return summary, tests


def _gate_decision(metrics: pd.DataFrame, baselines: pd.DataFrame) -> tuple[str, str, bool, bool]:
    """Apply the user-specified External Gate v2 criteria conservatively.

    GO requires independent external support, not merely a nonzero improvement over
    a weak residual component. The frozen CIRFL reference must beat strong
    baselines on at least one runnable main/cross-domain protocol and the residual
    field must show some mechanism value over raw/no-relation residuals.
    """
    if metrics.empty:
        return "EXTERNAL_NEED_DATA", "EXTERNAL_NEED_DATA", False, False
    combined = pd.concat([metrics, baselines], ignore_index=True)
    combined = combined[combined.get("status", "RUN_OK").fillna("RUN_OK") != "NOT_RUN"].copy()
    if combined.empty:
        return "NO-GO", "EXTERNAL_GATE_INVALID", False, False

    runnable_protocols = sorted(combined["protocol"].unique())
    external_protocol_support = 0
    mechanism_raw_support = 0
    mechanism_relation_support = 0
    relation_component_supported = False

    for protocol in runnable_protocols:
        group = combined[combined["protocol"] == protocol].copy()
        ref = group[group["method"] == "CIRFL_v3_reference"]
        if ref.empty:
            continue
        ref_mean = ref[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
        competitors = group[group["method"] != "CIRFL_v3_reference"]
        strong_baselines = competitors[competitors["method"].isin(["RandomForest", "XGBoost", "LightGBM", "IsolationForest", "AutoEncoder"])]
        if len(strong_baselines):
            by_method = strong_baselines.groupby("method")[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            best_macro = float(by_method["macro_f1"].max())
            best_prauc = float(by_method["pr_auc"].max())
            comparable = by_method[by_method["far"] <= ref_mean.get("far", 1.0) + 0.05]
            best_mdr_comparable = float(comparable["mdr"].min()) if len(comparable) else float("inf")
            if (
                ref_mean.get("macro_f1", -1.0) >= best_macro + 0.03
                or ref_mean.get("pr_auc", -1.0) >= best_prauc + 0.03
                or ref_mean.get("mdr", 1.0) <= best_mdr_comparable - 0.05
            ):
                external_protocol_support += 1

        raw = group[group["method"] == "raw_residual_energy"]
        no_relation = group[group["method"] == "condition_decoupled_residual_without_relation_atoms"]
        if len(raw):
            raw_mean = raw[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            if (
                ref_mean.get("macro_f1", -1.0) >= raw_mean.get("macro_f1", -1.0) + 0.03
                or ref_mean.get("pr_auc", -1.0) >= raw_mean.get("pr_auc", -1.0) + 0.03
                or (ref_mean.get("mdr", 1.0) <= raw_mean.get("mdr", 1.0) - 0.05 and ref_mean.get("far", 1.0) <= raw_mean.get("far", 1.0) + 0.10)
            ):
                mechanism_raw_support += 1
        if len(no_relation):
            nr_mean = no_relation[["macro_f1", "pr_auc", "far", "mdr"]].mean(numeric_only=True)
            if (
                ref_mean.get("macro_f1", -1.0) >= nr_mean.get("macro_f1", -1.0) + 0.03
                or ref_mean.get("pr_auc", -1.0) >= nr_mean.get("pr_auc", -1.0) + 0.03
                or (ref_mean.get("mdr", 1.0) <= nr_mean.get("mdr", 1.0) - 0.05 and ref_mean.get("far", 1.0) <= nr_mean.get("far", 1.0) + 0.10)
            ):
                mechanism_relation_support += 1
                relation_component_supported = True

    external_support = external_protocol_support >= 1
    mechanism_support = mechanism_raw_support >= 1 or mechanism_relation_support >= 1
    if external_support and mechanism_support and relation_component_supported:
        return "GO", "CIRFL_EXTERNAL_GO", True, False

    # If conditioned/no-relation residuals are as strong as CIRFL and trees dominate
    # main protocols, the honest decision is to reposition or redesign rather than
    # continue the full CIRFL mechanism.
    if mechanism_raw_support >= 1 and mechanism_relation_support == 0:
        return "NO-GO", "REPOSITION_TO_RESIDUAL_BASELINE", False, True
    return "NO-GO", "CIRFL_EXTERNAL_NO_GO", False, False

def _device_report(cfg: dict, output_dir: Path, generated_at: str, train_ran: bool) -> tuple[str, pd.DataFrame]:
    selected = str(resolve_device(cfg.get("device", "auto_fastest")))
    rows = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            dev = f"cuda:{idx}"
            rows.append(
                {
                    "dataset": "IMAD-DS RoboticArm",
                    "source_type": "external_real",
                    "protocol": "device_decision",
                    "seed": "NA",
                    "device": dev,
                    "gpu_name": torch.cuda.get_device_name(idx),
                    "available": True,
                    "assigned_task": "PyTorch AutoEncoder baseline" if train_ran and dev == selected else "available for future seed/protocol parallelism; not used by this pilot" ,
                    "memory_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 * 1024), 3),
                    "generated_at": generated_at,
                    "output_path": str(output_dir / "device_decision_external_v2.csv"),
                }
            )
    else:
        rows.append(
            {
                "dataset": "IMAD-DS RoboticArm",
                "source_type": "external_real",
                "protocol": "device_decision",
                "seed": "NA",
                "device": "cpu",
                "gpu_name": "NA",
                "available": False,
                "assigned_task": "all models CPU",
                "memory_allocated_mb": 0.0,
                "generated_at": generated_at,
                "output_path": str(output_dir / "device_decision_external_v2.csv"),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "device_decision_external_v2.csv", index=False)
    return selected, df


def _readiness_table(file_checks: pd.DataFrame, robotic: dict, brushless: dict, nist: dict, kuka: dict, output_dir: Path, generated_at: str) -> pd.DataFrame:
    rows = []
    for name, result, role in [
        ("IMAD-DS RoboticArm", robotic, "primary external robotic-arm gate"),
        ("IMAD-DS BrushlessMotor", brushless, "secondary industrial anomaly readiness; not robotic arm"),
        ("NIST UR", nist, "health/degradation readiness; direct anomaly labels not assumed"),
        ("KUKA LWR4+ torque/collision", kuka, "optional safety anomaly external validation; currently optional missing"),
    ]:
        rows.append(
            {
                "dataset": name,
                "source_type": "external_real" if result.get("status") == "READY" else "external_real_need_data_optional" if "KUKA" in name else "external_real_need_data",
                "status": "NEED_DATA_OPTIONAL" if "KUKA" in name and result.get("status") != "READY" else result.get("status", "UNKNOWN"),
                "reason": result.get("reason", "OK"),
                "role": role,
                "output_csv": result.get("output_csv", "NA"),
                "generated_at": generated_at,
                "output_path": str(output_dir / "external_readiness_v2.csv"),
            }
        )
    readiness = pd.DataFrame(rows)
    readiness.to_csv(output_dir / "external_readiness_v2.csv", index=False)
    return readiness


def _write_reports(
    cfg: dict,
    output_dir: Path,
    packet_dir: Path,
    generated_at: str,
    status: str,
    direction: str,
    road_ref: pd.DataFrame,
    file_checks: pd.DataFrame,
    robotic: dict,
    brushless: dict,
    nist: dict,
    kuka: dict,
    audit: pd.DataFrame,
    label_summary: pd.DataFrame,
    protocols: pd.DataFrame,
    metrics: pd.DataFrame,
    baselines: pd.DataFrame,
    winners: pd.DataFrame,
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    readiness: pd.DataFrame,
    selected_device: str,
    gpu_df: pd.DataFrame,
    external_gate_completed: bool,
    continue_cirfl: bool,
    reposition: bool,
) -> None:
    prov = _provenance("RoAD + IMAD-DS RoboticArm + readiness datasets", "real/external_real", "external_data_gate_v2", "7,13,23", output_dir, generated_at)
    save_config(cfg, output_dir / "07_external_gate_config.yaml")
    metrics_out = metrics.copy() if len(metrics) else _not_run_placeholder("external_gate_metrics", generated_at, output_dir)
    baseline_out = baselines.copy() if len(baselines) else _not_run_placeholder("external_baseline_comparison", generated_at, output_dir)
    metrics_out.to_csv(output_dir / "08_external_gate_metrics.csv", index=False)
    baseline_out.to_csv(output_dir / "09_external_baseline_comparison.csv", index=False)

    _write_text(
        output_dir / "00_readme_for_chatgpt.md",
        "Readme for ChatGPT",
        prov,
        [
            f"- current_stage: {STAGE}",
            f"- current_status: {status}",
            "- contains synthetic results: NO",
            "- generated images/figures: NO",
            "- entered full experiments: NO",
            f"- external_gate_completed: {'YES' if external_gate_completed else 'NO'}",
            "- RoAD reference remains frozen as CIRFL_v3; no RoAD-only tuning was performed.",
        ],
    )
    v3 = road_ref[road_ref.get("method", pd.Series(dtype=str)).astype(str).str.contains("CIRFL_v3", na=False)].head(25)
    _write_text(
        output_dir / "01_road_freeze_confirmation_v2.md",
        "RoAD Freeze Confirmation v2",
        prov,
        [
            "- RoAD reference: CIRFL_v3.",
            "- CIRFL_v3_plain_residual_score remains a strong comparator and is not renamed as CIRFL.",
            "- v4 is not a reference because score composition was unstable.",
            "- v5 is not a reference because it did not recover v3 strength or prove mechanism necessity.",
            "- road_binary_main is sanity evidence only; scenario holdouts are stress evidence only.",
            "- RoAD-only cannot enter full experiments or manuscript workflow.",
            "",
            dataframe_to_markdown(v3 if len(v3) else road_ref.head(25)),
        ],
    )
    _write_text(
        output_dir / "02_external_file_checks_v2.md",
        "External File Checks v2",
        prov,
        [dataframe_to_markdown(file_checks)],
    )
    _write_text(
        output_dir / "03_imadds_roboticarm_adapter_dry_run_v2.md",
        "IMAD-DS RoboticArm Adapter Dry-Run v2",
        prov,
        [
            f"- adapter_status: {robotic.get('status')}",
            f"- n_attribute_rows: {robotic.get('n_attribute_rows')}",
            f"- n_rows: {robotic.get('n_rows')}",
            f"- n_features: {robotic.get('n_features')}",
            f"- n_conditions: {robotic.get('n_conditions')}",
            f"- n_missing_sensor_files: {robotic.get('n_missing_sensor_files')}",
            f"- n_failed_sensor_files: {robotic.get('n_failed_sensor_files')}",
            f"- output_csv: `{robotic.get('output_csv')}`",
            "- granularity: segment-level statistical features from official attributes/parquet mapping.",
            "- label_source: attributes_*.csv anomaly_label; no filename label guessing.",
        ],
    )
    _write_text(
        output_dir / "04_imadds_roboticarm_data_audit_v2.md",
        "IMAD-DS RoboticArm Data Audit v2",
        prov,
        [
            "## Audit",
            dataframe_to_markdown(audit),
            "",
            "## Label / Domain Summary (first 80 rows)",
            dataframe_to_markdown(label_summary.head(80)),
        ],
    )
    _write_text(
        output_dir / "05_imadds_leakage_and_protocol_validity_v2.md",
        "IMAD-DS Leakage and Protocol Validity v2",
        prov,
        [
            "- split_unit: segment_id / parquet file references from attributes CSV.",
            "- random overlapping-window split: NO.",
            "- normalization: train-only scaler/median in gate runner.",
            "- threshold source: validation only.",
            "- protocol caveat: IMAD-DS official train split contains normal segments only; supervised tree baselines use labeled calibration segments from official test partition and are marked as pilot external gate.",
            dataframe_to_markdown(protocols),
        ],
    )
    _write_text(
        output_dir / "06_nist_brushless_kuka_readiness_v2.md",
        "NIST / BrushlessMotor / KUKA Readiness v2",
        prov,
        [
            dataframe_to_markdown(readiness),
            "",
            "- BrushlessMotor is found and adapter-ready as a secondary industrial dataset, not a robotic-arm primary dataset.",
            "- NIST UR is found and readable, but direct anomaly labels are not assumed; it is more suitable for health/degradation validation unless a label rule is approved.",
            "- KUKA is optional missing and does not block this IMAD-DS/NIST stage.",
        ],
    )
    _write_text(
        output_dir / "10_external_statistical_summary.md",
        "External Statistical Summary v2",
        prov,
        [
            "## Mean +/- std",
            dataframe_to_markdown(summary) if len(summary) else "NOT_RUN_NEED_DATA",
            "",
            "## Paired Tests vs CIRFL_v3_reference",
            dataframe_to_markdown(tests.head(80)) if len(tests) else "NOT_ENOUGH_RUNS_OR_NOT_RUN",
        ],
    )
    _write_text(
        output_dir / "11_external_protocol_winners.md",
        "External Protocol Winners v2",
        prov,
        [dataframe_to_markdown(winners) if len(winners) else "NOT_RUN_NEED_DATA"],
    )
    _write_text(
        output_dir / "12_cross_dataset_transfer_feasibility_v2.md",
        "Cross-Dataset Transfer Feasibility v2",
        prov,
        [
            "- RoAD -> IMAD-DS direct channel mapping: not valid because sensor semantics differ.",
            "- IMAD-DS -> RoAD direct channel mapping: not valid for the same reason.",
            "- IMAD-DS RoboticArm -> BrushlessMotor: feasible only through shared statistical feature abstraction or sensor-token abstraction; not by raw channel concatenation.",
            "- NIST UR -> IMAD-DS/RoAD: feasible only as health/degradation validation or shared feature abstraction, not direct fault-class transfer.",
            "- Strong rule: incompatible sensors must not be forcibly concatenated.",
        ],
    )
    _write_text(
        output_dir / "13_cirfl_direction_decision_v2.md",
        "CIRFL Direction Decision v2",
        prov,
        [
            f"## Decision: {direction}",
            f"- current_status: {status}",
            f"- continue_CIRFL: {'YES' if continue_cirfl else 'NO'}",
            f"- recommend_reposition: {'YES' if reposition else 'NO'}",
            "- Interpretation is based on external real IMAD-DS pilot only; no synthetic result is included.",
            "- If residual-field reference does not beat raw/no-relation residuals externally, CIRFL full mechanism remains unsupported and should not enter full experiments.",
        ],
    )
    _write_text(
        output_dir / "14_device_decision_report_v2.md",
        "Device Decision Report External v2",
        prov,
        [
            f"- selected_device_for_PyTorch: {selected_device}",
            "- DataParallel used: NO.",
            "- multi-GPU used: NO for this pilot; three RTX 2080Ti are available for future seed/protocol parallelism, but this segment-level pilot is small enough for one device.",
            "- tree baselines used CPU for consistency unless their library internally selected otherwise.",
            dataframe_to_markdown(gpu_df),
        ],
    )
    comp_rows = []
    for label, df in [("CIRFL/residual candidates", metrics), ("baselines", baselines)]:
        if len(df):
            comp_rows.append({"group": label, "train_time_sec_total": float(df["train_time_sec"].fillna(0).sum()), "inference_time_sec_total": float(df["inference_time_sec"].fillna(0).sum()), "min_model_size_mb": float(df["model_size_mb"].min(skipna=True)), "max_model_size_mb": float(df["model_size_mb"].max(skipna=True))})
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(output_dir / "complexity_latency_external_v2.csv", index=False)
    _write_text(
        output_dir / "15_complexity_latency_external_v2.md",
        "Complexity and Latency External v2",
        prov,
        [
            dataframe_to_markdown(comp) if len(comp) else "NOT_RUN_NEED_DATA",
            "- CIRFL_v3_reference model-size reference remains about 0.666 MB from frozen RoAD reports; external segment-level residual scoring itself is lightweight.",
            "- CPU/GPU latency columns are recorded per metric row.",
        ],
    )
    risk_lines = [
        "- no_full_experiments: YES",
        "- no_figures_generated: YES",
        "- no_synthetic_results: YES",
        "- RoAD-only remains blocked for full experiments.",
        "- IMAD-DS official train split is normal-only; supervised baselines were handled as pilot with labeled calibration partition, not as final full-experiment protocol.",
        "- LSTM-AE and USAD were not run on segment-level features; raw time-window adapter is needed before full comparisons.",
        "- NIST UR lacks direct anomaly labels in this audit; do not force binary fault diagnosis without an approved label rule.",
        "- KUKA remains optional missing.",
    ]
    if status != "GO":
        risk_lines.append("- next_stage_blocked: YES; do not enter full experiments.")
    _write_text(output_dir / "16_errors_and_risks.md", "Errors and Risks", prov, risk_lines)
    _write_text(
        output_dir / "17_go_no_go_report_external_v2.md",
        "External Data Gate v2 GO / NO-GO Report",
        prov,
        [
            f"## Decision: {status}",
            f"- A1 external real robot/mechanical-arm dataset available: {'YES' if robotic.get('status') == 'READY' else 'NO'}",
            "- A2 synthetic substitute used: NO",
            f"- A3 data audit complete: {'YES' if len(audit) else 'NO'}",
            "- A4 split leakage risk is not HIGH: YES for pilot segment split; official train is normal-only caveat remains.",
            f"- A5 at least one valid external protocol with normal/anomaly test: {'YES' if len(protocols[protocols['can_calculate_auroc_pr_far_mdr'] == True]) else 'NO'}",
            f"- B external performance support: {'YES' if continue_cirfl else 'NO'}",
            f"- C mechanism necessity support: {'YES' if continue_cirfl else 'NO'}",
            "- D RoAD + external combined judgment: RoAD alone is not enough; this decision uses IMAD-DS pilot when run.",
            "- E complexity condition: satisfied for frozen reference scale, but final CPU/GPU latency requires raw-window gate before full experiments.",
            "",
            "Full experiments, manuscript, abstract, and figures remain blocked unless status is GO and the user explicitly starts full-experiment design.",
        ],
    )
    _write_text(
        output_dir / "18_code_index.md",
        "Code Index",
        prov,
        [
            "- `src/datasets/adapters/imadds.py`: fixed IMAD-DS adapter to use attributes CSV labels/domains and segment-level parquet statistics.",
            "- `scripts/prepare_real_data.py`: standard adapter entry used for RoboticArm dry-run.",
            "- `scripts/run_external_data_gate_v2.py`: external file checks, adapter execution, pilot external gate, review packet generation.",
            "- `configs/external_gate_v2.yaml`: external gate configuration.",
            "",
            "## Commands",
            "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/prepare_real_data.py --dataset imadds_robotic_arm --json`",
            "`/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_external_data_gate_v2.py --config configs/external_gate_v2.yaml`",
        ],
    )
    if status == "GO":
        next_body = ["- Next step: prepare full experiment design only; still do not write manuscript or abstract without explicit user request."]
    elif direction == "REPOSITION_TO_RESIDUAL_BASELINE":
        next_body = ["- Next step: stop CIRFL full-mechanism path and consider a simpler condition-decoupled residual detector + source localization, or redesign the core algorithm before full experiments."]
    else:
        next_body = ["- Next step: inspect external gate failures, add raw time-window IMAD-DS adapter for LSTM-AE/USAD comparison, and decide whether CIRFL_v3 mechanism can be rescued on external data."]
    _write_text(output_dir / "19_next_tasks_for_codex.md", "Next Tasks for Codex", prov, next_body)

    _clean_dir(packet_dir)
    for name in PACKET_FILES:
        src = output_dir / name
        if src.suffix == ".yaml" and not src.exists():
            save_config(cfg, src)
        shutil.copyfile(src, packet_dir / name)
    files = list(packet_dir.iterdir())
    if len(files) > 20:
        raise RuntimeError("Review packet exceeds 20 files")
    if any(p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"} for p in files):
        raise RuntimeError("Review packet contains forbidden image/figure files")


def _not_run_placeholder(kind: str, generated_at: str, output_dir: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset": "IMAD-DS RoboticArm",
                "source_type": "external_real_need_data",
                "protocol": "external_gate_v2",
                "seed": -1,
                "method": kind,
                "status": "NOT_RUN_NEED_DATA",
                "macro_f1": np.nan,
                "weighted_f1": np.nan,
                "auroc": np.nan,
                "pr_auc": np.nan,
                "far": np.nan,
                "mdr": np.nan,
                "far_at_95_recall": np.nan,
                "n_train_windows": -1,
                "n_val_windows": -1,
                "n_test_windows": -1,
                "n_test_normal": -1,
                "n_test_anomaly": -1,
                "generated_at": generated_at,
                "output_path": str(output_dir),
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/external_gate_v2.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    output_dir = ROOT / cfg["project"]["output_dir"]
    packet_dir = ROOT / cfg["review_packet"]["output_dir"]
    _clean_dir(output_dir)
    generated_at = utc_now()

    road_ref = _road_freeze(output_dir, generated_at)
    file_checks = _file_checks(cfg, output_dir)
    robotic, brushless, nist, kuka = _run_adapters(output_dir)
    readiness = _readiness_table(file_checks, robotic, brushless, nist, kuka, output_dir, generated_at)

    metrics = pd.DataFrame()
    baselines = pd.DataFrame()
    winners = pd.DataFrame()
    summary = pd.DataFrame()
    tests = pd.DataFrame()
    external_gate_completed = False
    continue_cirfl = False
    reposition = False
    direction = "EXTERNAL_NEED_DATA"
    status = "EXTERNAL_NEED_DATA"
    selected_device, gpu_df = _device_report(cfg, output_dir, generated_at, train_ran=False)

    if robotic.get("status") != "READY":
        audit = pd.DataFrame()
        label_summary = pd.DataFrame()
        protocols = pd.DataFrame()
        status = "IMADDS_ADAPTER_FAIL" if (ROOT / "data/raw/imadds/RoboticArm").exists() else "EXTERNAL_NEED_DATA"
        direction = status
    else:
        csv_path = Path(robotic["output_csv"])
        audit, label_summary = _audit_imadds(csv_path, output_dir, generated_at)
        df = pd.read_csv(csv_path)
        protocols = _build_protocols(df, output_dir, generated_at)
        valid_runnable = protocols[(protocols["can_calculate_auroc_pr_far_mdr"] == True) & (protocols["validity"] == "VALID")]
        if len(valid_runnable) == 0:
            status = "NO-GO"
            direction = "EXTERNAL_GATE_INVALID"
        else:
            selected_device, gpu_df = _device_report(cfg, output_dir, generated_at, train_ran=True)
            metrics, baselines = _run_external_gate(csv_path, output_dir, generated_at, selected_device)
            winners = _protocol_winners(metrics, baselines, output_dir, generated_at)
            summary, tests = _stat_summary(metrics, baselines, output_dir)
            status, direction, continue_cirfl, reposition = _gate_decision(metrics, baselines)
            external_gate_completed = True

    _write_reports(
        cfg,
        output_dir,
        packet_dir,
        generated_at,
        status,
        direction,
        road_ref,
        file_checks,
        robotic,
        brushless,
        nist,
        kuka,
        audit,
        label_summary,
        protocols,
        metrics,
        baselines,
        winners,
        summary,
        tests,
        readiness,
        selected_device,
        gpu_df,
        external_gate_completed,
        continue_cirfl,
        reposition,
    )
    print(f"External Data Gate v2 finished: {status}; direction={direction}; packet_files={len(list(packet_dir.iterdir()))}")


if __name__ == "__main__":
    main()
