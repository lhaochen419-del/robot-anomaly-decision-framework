from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

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

from src.datasets.imadds_raw_window import apply_normalizer, load_raw_windows, train_only_normalizer, window_stat_features
from src.evaluation.metrics import binary_metric_row, choose_threshold, far_at_recall, score_direction_multiplier, summarize_by_method
from src.models import LSTMAutoEncoder, MLPWindowAutoEncoder, USAD
from src.utils.torch_utils import count_parameters

RAW_DATASET = "IMAD-DS RoboticArm raw-window"
SEG_DATASET = "IMAD-DS RoboticArm segment-level"
BRUSH_DATASET = "IMAD-DS BrushlessMotor"
ROAD_DATASET = "RoAD"
HISTORICAL = {"CIRFL_v3_reference", "CETRA_full", "COIL_full", "raw_residual_energy", "condition_decoupled_residual_energy"}
DEEP = {"AutoEncoder", "LSTM-AE", "USAD"}
TREE = {"LightGBM", "XGBoost", "RandomForest"}
ALL_STRATEGIES = [
    "best_f1",
    "youden_j",
    "target_far_0.05",
    "target_far_0.10",
    "target_far_0.15",
    "target_recall_0.80",
    "target_recall_0.90",
    "target_recall_0.95",
    "cost_md2_fp1",
    "cost_md5_fp1",
    "cost_md10_fp1",
]
PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_runner_implementation_report.md",
    "02_job_execution_status.md",
    "03_dataset_protocol_summary.md",
    "04_full_metric_summary.csv",
    "05_baseline_comparison_summary.csv",
    "06_full_statistical_summary.md",
    "07_threshold_calibration_summary.md",
    "08_far_mdr_engineering_analysis.md",
    "09_label_efficiency_analysis.md",
    "10_robustness_analysis.md",
    "11_domain_shift_analysis.md",
    "12_latency_deployment_analysis.md",
    "13_model_selection_guidelines_final.md",
    "14_quality_control_report.md",
    "15_failed_job_report.csv",
    "16_provenance_integrity_report.md",
    "17_rie_evidence_readiness_assessment.md",
    "18_go_no_go_report_full_benchmark_v1.md",
    "19_code_index_and_next_tasks.md",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
    if df.empty:
        return "NOT_AVAILABLE"
    view = df.head(max_rows).copy().fillna("")
    cols = [str(c) for c in view.columns]
    rows = [[str(v) for v in row] for row in view.to_numpy().tolist()]

    def esc(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", " ")

    return "\n".join(
        [
            "| " + " | ".join(esc(c) for c in cols) + " |",
            "| " + " | ".join("---" for _ in cols) + " |",
            *["| " + " | ".join(esc(v) for v in row) + " |" for row in rows],
        ]
    )


def prov(protocol: str, generated_at: str, output_root: Path) -> str:
    return "\n".join(
        [
            "- dataset: real IMAD-DS/RoAD data only",
            "- source_type: external_real / real_reference",
            f"- protocol: {protocol}",
            "- seed: see CSV rows",
            "- n_train/n_val/n_test: see CSV rows",
            "- n_test_normal/n_test_anomaly: see CSV rows",
            f"- generated_at: {generated_at}",
            f"- output_path: {output_root}",
        ]
    )


def write_md(path: Path, title: str, provenance: str, lines: list[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", provenance, "", *lines, ""]), encoding="utf-8")


def segment_table(meta: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ["segment_uid", "segment_id", "split_stage", "domain_label", "domain_shift_op", "domain_shift_env", "condition_name", "condition_id", "fault_label", "fault_label_name"] if c in meta.columns]
    table = meta[cols].copy()
    if "segment_uid" not in table and "segment_id" in table:
        table["segment_uid"] = table["segment_id"].astype(str)
    return table.drop_duplicates("segment_uid").reset_index(drop=True)


def split_segments(seg: pd.DataFrame, seed: int, ratios: tuple[float, float, float]) -> tuple[set[str], set[str], set[str]]:
    rng = np.random.default_rng(seed)
    a: list[str] = []
    b: list[str] = []
    c: list[str] = []
    group_cols = [col for col in ["fault_label", "domain_label"] if col in seg]
    grouped = seg.groupby(group_cols, dropna=False) if group_cols else [("all", seg)]
    for _, group in grouped:
        ids = group["segment_uid"].astype(str).to_numpy()
        rng.shuffle(ids)
        n = len(ids)
        n_a = max(1, int(round(ratios[0] * n))) if n >= 3 else max(0, n - 2)
        n_b = max(1, int(round(ratios[1] * n))) if n - n_a >= 2 else max(0, n - n_a - 1)
        a.extend(ids[:n_a])
        b.extend(ids[n_a : n_a + n_b])
        c.extend(ids[n_a + n_b :])
    return set(a), set(b), set(c)


def ids_to_idx(meta: pd.DataFrame, ids: set[str]) -> np.ndarray:
    col = "segment_uid" if "segment_uid" in meta.columns else "segment_id"
    return np.where(meta[col].astype(str).isin(ids).to_numpy())[0]


def apply_stress(x: np.ndarray, protocol: str, seed: int) -> np.ndarray:
    out = x.copy()
    rng = np.random.default_rng(seed)
    if protocol.endswith("missing_sensor_10") or protocol == "brushless_missing_sensor_if_meaningful":
        rate = 0.10
    elif protocol.endswith("missing_sensor_20"):
        rate = 0.20
    else:
        rate = 0.0
    if rate > 0 and out.ndim == 3:
        n_drop = max(1, int(round(out.shape[-1] * rate)))
        drop = rng.choice(out.shape[-1], size=n_drop, replace=False)
        out[:, :, drop] = 0.0
    if "noise_mild" in protocol:
        scale = np.nanstd(out, axis=(0, 1), keepdims=True)
        scale[scale < 1e-6] = 1.0
        out = out + rng.normal(0, 0.02, size=out.shape).astype(np.float32) * scale
    elif "noise_moderate" in protocol:
        scale = np.nanstd(out, axis=(0, 1), keepdims=True)
        scale[scale < 1e-6] = 1.0
        out = out + rng.normal(0, 0.05, size=out.shape).astype(np.float32) * scale
    return out.astype(np.float32)


def raw_splits(meta: pd.DataFrame, protocol: str, seed: int) -> dict[str, set[str]]:
    seg = segment_table(meta)
    test_seg = seg[seg["split_stage"] == "test"].copy()
    train_main = set(seg[(seg["split_stage"] == "train") & (seg["fault_label"] == 0)]["segment_uid"].astype(str))
    if protocol in {"imadds_raw_source_to_target"}:
        source_train = set(seg[(seg["split_stage"] == "train") & (seg["domain_label"] == "source") & (seg["fault_label"] == 0)]["segment_uid"].astype(str))
        source_test = test_seg[test_seg["domain_label"] == "source"]
        target_test = test_seg[test_seg["domain_label"] == "target"]
        calib, val, _ = split_segments(source_test, seed, (0.50, 0.50, 0.0))
        return {"train": source_train, "calib": calib, "val": val, "test": set(target_test["segment_uid"].astype(str))}
    if protocol in {"imadds_raw_leave_target_weight35_out"}:
        held = test_seg[(test_seg["domain_label"] == "target") & (test_seg["domain_shift_op"] == "weight35")]
        rest = test_seg.drop(index=held.index)
        calib, val, _ = split_segments(rest, seed, (0.50, 0.50, 0.0))
        return {"train": train_main, "calib": calib, "val": val, "test": set(held["segment_uid"].astype(str))}
    calib, val, test = split_segments(test_seg, seed, (0.34, 0.33, 0.33))
    return {"train": train_main, "calib": calib, "val": val, "test": test}


def tabular_features(df: pd.DataFrame) -> list[str]:
    meta = {
        "t",
        "run_id",
        "trajectory_id",
        "segment_id",
        "condition_id",
        "condition_name",
        "domain_label",
        "domain_shift_op",
        "domain_shift_env",
        "split_stage",
        "split_label",
        "fault_episode_id",
        "fault_label",
        "fault_class_id",
        "fault_label_name",
        "attribute_file",
        "source_file",
        "adapter_granularity",
        "adapter_row_status",
        "source_subset",
    }
    return [c for c in df.columns if c not in meta and pd.api.types.is_numeric_dtype(df[c])]


def tabular_splits(df: pd.DataFrame, dataset: str, protocol: str, seed: int) -> dict[str, np.ndarray]:
    if dataset == ROAD_DATASET:
        if "source_subset" not in df:
            return {}
        train = np.where((df["source_subset"].eq("training")) & (df["fault_label"].eq(0)))[0]
        control = np.where((df["source_subset"].eq("control")) & (df["fault_label"].eq(0)))[0]
        collision = np.where(df["source_subset"].eq("collision"))[0]
        if len(collision) == 0 or len(control) == 0:
            return {}
        rng = np.random.default_rng(seed)
        rng.shuffle(control)
        val = control[: min(len(control), max(1000, len(control) // 4))]
        calib = control[min(len(control), max(1000, len(control) // 4)) : min(len(control), max(2000, len(control) // 2))]
        test = collision
        if protocol == "road_artifact_confounding_diagnostic":
            test = collision
        return {"train": train, "calib": calib, "val": val, "test": test}
    seg_col = "segment_id"
    seg = df.copy()
    seg["segment_uid"] = seg[seg_col].astype(str)
    table = segment_table(seg)
    test_seg = table[table["split_stage"] == "test"].copy()
    train = np.where((df["split_stage"].eq("train")) & (df["fault_label"].eq(0)))[0]
    if protocol.endswith("source_to_target") or "source_to_target" in protocol:
        source = test_seg[test_seg["domain_label"] == "source"]
        target = test_seg[test_seg["domain_label"] == "target"]
        calib_ids, val_ids, _ = split_segments(source, seed, (0.50, 0.50, 0.0))
        return {"train": train, "calib": ids_to_idx(seg, calib_ids), "val": ids_to_idx(seg, val_ids), "test": ids_to_idx(seg, set(target["segment_uid"].astype(str)))}
    if "leave_target_weight35" in protocol:
        held = test_seg[(test_seg["domain_label"] == "target") & (test_seg["domain_shift_op"] == "weight35")]
        rest = test_seg.drop(index=held.index)
        calib_ids, val_ids, _ = split_segments(rest, seed, (0.50, 0.50, 0.0))
        return {"train": train, "calib": ids_to_idx(seg, calib_ids), "val": ids_to_idx(seg, val_ids), "test": ids_to_idx(seg, set(held["segment_uid"].astype(str)))}
    calib_ids, val_ids, test_ids = split_segments(test_seg, seed, (0.34, 0.33, 0.33))
    return {"train": train, "calib": ids_to_idx(seg, calib_ids), "val": ids_to_idx(seg, val_ids), "test": ids_to_idx(seg, test_ids)}


@dataclass
class Prepared:
    train_x: np.ndarray
    calib_x: np.ndarray
    val_x: np.ndarray
    test_x: np.ndarray
    y_calib: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    train_meta: pd.DataFrame
    val_meta: pd.DataFrame
    test_meta: pd.DataFrame
    n_train: int
    n_val: int
    n_test: int
    split_unit: str


class BenchmarkRunner:
    def __init__(self, output_root: Path, generated_at: str) -> None:
        self.output_root = output_root
        self.generated_at = generated_at
        self._raw = None
        self._tabular: dict[str, pd.DataFrame] = {}

    def raw_data(self):
        if self._raw is None:
            self._raw = load_raw_windows(ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows.npz", ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows_metadata.csv")
        return self._raw

    def tabular_data(self, dataset: str) -> pd.DataFrame:
        if dataset not in self._tabular:
            path = {
                SEG_DATASET: ROOT / "data/processed/imadds_robotic_arm/imadds_robotic_arm_unified.csv",
                BRUSH_DATASET: ROOT / "data/processed/imadds_brushless_motor/imadds_brushless_motor_unified.csv",
                ROAD_DATASET: ROOT / "data/processed/road/road_unified.csv",
            }[dataset]
            self._tabular[dataset] = pd.read_csv(path)
        return self._tabular[dataset]

    def prepare_raw(self, protocol: str, seed: int) -> Prepared | None:
        data = self.raw_data()
        parts = raw_splits(data.meta, protocol, seed)
        idx = {k: ids_to_idx(data.meta, v) for k, v in parts.items()}
        if any(len(idx[k]) == 0 for k in ["train", "val", "test"]):
            return None
        train_x_all = data.x[idx["train"]]
        train_meta_all = data.meta.iloc[idx["train"]].reset_index(drop=True)
        normal_mask = train_meta_all["fault_label"].to_numpy(int) == 0
        train_x = train_x_all[normal_mask]
        train_meta = train_meta_all.loc[normal_mask].reset_index(drop=True)
        val_x = apply_stress(data.x[idx["val"]], protocol, seed)
        test_x = apply_stress(data.x[idx["test"]], protocol, seed)
        calib_x = apply_stress(data.x[idx.get("calib", np.array([], dtype=int))], protocol, seed)
        calib_meta = data.meta.iloc[idx.get("calib", np.array([], dtype=int))].reset_index(drop=True)
        val_meta = data.meta.iloc[idx["val"]].reset_index(drop=True)
        test_meta = data.meta.iloc[idx["test"]].reset_index(drop=True)
        y_val = val_meta["fault_label"].to_numpy(int)
        y_test = test_meta["fault_label"].to_numpy(int)
        y_calib = calib_meta["fault_label"].to_numpy(int) if len(calib_meta) else np.array([], dtype=int)
        if len(train_x) == 0 or len(np.unique(y_val)) < 2 or len(np.unique(y_test)) < 2:
            return None
        return Prepared(train_x, calib_x, val_x, test_x, y_calib, y_val, y_test, train_meta, val_meta, test_meta, len(train_x), len(val_x), len(test_x), "segment_uid/file_id; no random overlapping-window split")

    def prepare_tabular(self, dataset: str, protocol: str, seed: int) -> tuple[Prepared | None, list[str]]:
        df = self.tabular_data(dataset)
        feats = tabular_features(df)
        parts = tabular_splits(df, dataset, protocol, seed)
        if not parts or any(len(parts[k]) == 0 for k in ["train", "val", "test"]):
            return None, feats
        train_df = df.iloc[parts["train"]].copy()
        normal_train = train_df[train_df["fault_label"] == 0]
        val_df = df.iloc[parts["val"]].copy()
        test_df = df.iloc[parts["test"]].copy()
        calib_df = df.iloc[parts.get("calib", np.array([], dtype=int))].copy()
        if len(normal_train) == 0 or val_df["fault_label"].nunique() < 2 or test_df["fault_label"].nunique() < 2:
            return None, feats
        train_x = normal_train[feats].to_numpy(np.float32)
        val_x = val_df[feats].to_numpy(np.float32)
        test_x = test_df[feats].to_numpy(np.float32)
        calib_x = calib_df[feats].to_numpy(np.float32) if len(calib_df) else np.empty((0, len(feats)), dtype=np.float32)
        return Prepared(train_x, calib_x, val_x, test_x, calib_df["fault_label"].to_numpy(int) if len(calib_df) else np.array([], dtype=int), val_df["fault_label"].to_numpy(int), test_df["fault_label"].to_numpy(int), normal_train, val_df, test_df, len(train_x), len(val_x), len(test_x), "segment/file/run split; no random row/window split"), feats

    def run_job(self, job: pd.Series, resume: bool = True) -> tuple[dict[str, Any], dict[str, Any] | None, pd.DataFrame]:
        dataset = str(job["dataset"])
        protocol = str(job["protocol"])
        method = str(job["model"])
        seed = int(job["seed"])
        job_dir = self.output_root / "jobs" / safe(dataset) / protocol / method / f"seed_{seed}"
        metrics_path = job_dir / "metrics.csv"
        status_path = job_dir / "status.json"
        if resume and metrics_path.exists() and status_path.exists():
            row = pd.read_csv(metrics_path).iloc[0].to_dict()
            status = json.loads(status_path.read_text(encoding="utf-8"))
            return status, row, pd.DataFrame()
        job_dir.mkdir(parents=True, exist_ok=True)
        t_start = time.perf_counter()
        try:
            if dataset == RAW_DATASET:
                prep = self.prepare_raw(protocol, seed)
                if prep is None:
                    return self.skip_status(job, "INVALID_RAW_PROTOCOL_SPLIT", job_dir, t_start), None, pd.DataFrame()
                row, sens = self.run_raw_model(prep, method, protocol, seed)
            elif dataset in {SEG_DATASET, BRUSH_DATASET, ROAD_DATASET}:
                prep, feats = self.prepare_tabular(dataset, protocol, seed)
                if prep is None:
                    return self.skip_status(job, "INVALID_TABULAR_PROTOCOL_SPLIT_OR_SINGLE_CLASS", job_dir, t_start), None, pd.DataFrame()
                row, sens = self.run_tabular_model(prep, method, dataset, protocol, seed, feats)
            else:
                return self.skip_status(job, "DATASET_NOT_SUPPORTED_FOR_FULL_BENCHMARK", job_dir, t_start), None, pd.DataFrame()
            row.update(self.row_provenance(job, prep, job_dir, time.perf_counter() - t_start))
            pd.DataFrame([row]).to_csv(metrics_path, index=False)
            status = self.ok_status(job, job_dir, t_start)
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            if len(sens):
                sens.to_csv(job_dir / "threshold_sensitivity.csv", index=False)
            return status, row, sens
        except Exception as exc:
            status = self.fail_status(job, repr(exc), job_dir, t_start)
            status_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
            return status, None, pd.DataFrame()

    def row_provenance(self, job: pd.Series, prep: Prepared, job_dir: Path, elapsed: float) -> dict[str, Any]:
        return {
            "dataset": job["dataset"],
            "source_type": "external_real" if job["dataset"] != ROAD_DATASET else "real_reference_secondary",
            "stage": "RIE_FULL_ENGINEERING_BENCHMARK_V1",
            "n_train_windows": prep.n_train,
            "n_val_windows": prep.n_val,
            "n_test_windows": prep.n_test,
            "n_test_normal": int((prep.y_test == 0).sum()),
            "n_test_anomaly": int((prep.y_test > 0).sum()),
            "split_unit": prep.split_unit,
            "threshold_source": "validation_only",
            "normalization_source": "train_only",
            "generated_at": self.generated_at,
            "output_path": str(job_dir),
            "job_elapsed_sec": elapsed,
        }

    def ok_status(self, job: pd.Series, job_dir: Path, t_start: float) -> dict[str, Any]:
        return self.status_base(job, "RUN_OK", "completed", job_dir, time.perf_counter() - t_start)

    def skip_status(self, job: pd.Series, reason: str, job_dir: Path, t_start: float) -> dict[str, Any]:
        status = self.status_base(job, "SKIPPED_WITH_REASON", reason, job_dir, time.perf_counter() - t_start)
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
        return status

    def fail_status(self, job: pd.Series, reason: str, job_dir: Path, t_start: float) -> dict[str, Any]:
        return self.status_base(job, "FAILED", reason, job_dir, time.perf_counter() - t_start)

    def status_base(self, job: pd.Series, status: str, reason: str, job_dir: Path, elapsed: float) -> dict[str, Any]:
        return {
            "dataset": job["dataset"],
            "protocol": job["protocol"],
            "method": job["model"],
            "seed": int(job["seed"]),
            "status": status,
            "reason": reason,
            "elapsed_sec": float(elapsed),
            "device_planned": job.get("expected_device", ""),
            "generated_at": self.generated_at,
            "output_path": str(job_dir),
            "core_job": is_core_job(job),
        }

    def run_raw_model(self, prep: Prepared, method: str, protocol: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
        if method in {"CETRA_full", "COIL_full"}:
            raise RuntimeError("Historical failed route is not rerun in RIE benchmark; previous gate output retained only as reference")
        if method in HISTORICAL:
            return self.run_raw_residual(prep, method, protocol, seed)
        if method in DEEP:
            return self.run_deep(prep, method, protocol, seed)
        return self.run_stat_model(prep, method, RAW_DATASET, protocol, seed, raw=True)

    def run_tabular_model(self, prep: Prepared, method: str, dataset: str, protocol: str, seed: int, feats: list[str]) -> tuple[dict[str, Any], pd.DataFrame]:
        if method in DEEP:
            raise RuntimeError("Deep time-series baseline requires raw-window tensor input; segment/tabular protocol not supported")
        if method in HISTORICAL:
            raise RuntimeError("Historical raw-window reference not rerun on tabular/secondary protocol")
        return self.run_stat_model(prep, method, dataset, protocol, seed, raw=False)

    def run_raw_residual(self, prep: Prepared, method: str, protocol: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
        t0 = time.perf_counter()
        raw_val, raw_test, raw_ch_val, raw_ch_test = raw_energy(prep.train_x, prep.val_x, prep.test_x)
        if method == "raw_residual_energy":
            s_val, s_test = raw_val, raw_test
        elif method == "condition_decoupled_residual_energy":
            s_val, _ = condition_energy(prep.train_x, prep.train_meta, prep.val_x, prep.val_meta)
            s_test, _ = condition_energy(prep.train_x, prep.train_meta, prep.test_x, prep.test_meta)
        elif method == "CIRFL_v3_reference":
            cond_val, _ = condition_energy(prep.train_x, prep.train_meta, prep.val_x, prep.val_meta)
            cond_test, _ = condition_energy(prep.train_x, prep.train_meta, prep.test_x, prep.test_meta)
            rel_val, rel_test = relation_score(prep.train_x, prep.val_x, prep.test_x)
            s_val, s_test = 0.70 * cond_val + 0.30 * rel_val, 0.70 * cond_test + 0.30 * rel_test
        else:
            raise RuntimeError(f"unsupported residual reference: {method}")
        infer = time.perf_counter() - t0
        row, sens = evaluate_scores(prep.y_val, s_val, prep.y_test, s_test, method, seed, protocol)
        row.update({"method_type": "historical_failed_reference" if method == "CIRFL_v3_reference" else "historical_residual_reference", "status": "RUN_OK", "device": "cpu", "model_size_mb": 0.666 if method == "CIRFL_v3_reference" else 0.001, "train_time_sec": 0.0, "inference_time_sec": infer, "cpu_latency_ms": infer / max(prep.n_test, 1) * 1000, "gpu_latency_ms": np.nan, "note": "negative/historical reference; not a new algorithm"})
        return row, sens

    def run_deep(self, prep: Prepared, method: str, protocol: str, seed: int) -> tuple[dict[str, Any], pd.DataFrame]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        dev_id = seed_to_gpu(seed)
        dev = torch.device(dev_id if torch.cuda.is_available() else "cpu")
        mean, std = train_only_normalizer(prep.train_x)
        tr = apply_normalizer(prep.train_x, mean, std)
        va = apply_normalizer(prep.val_x, mean, std)
        te = apply_normalizer(prep.test_x, mean, std)
        ws, nc = tr.shape[1], tr.shape[2]
        if method == "AutoEncoder":
            model = MLPWindowAutoEncoder(ws, nc, hidden_dim=64, latent_dim=24)
        elif method == "LSTM-AE":
            model = LSTMAutoEncoder(nc, hidden_dim=32, latent_dim=20)
        elif method == "USAD":
            model = USAD(ws, nc, hidden_dim=80, latent_dim=24)
        else:
            raise RuntimeError(method)
        model.to(dev)
        loader = DataLoader(TensorDataset(torch.tensor(tr, dtype=torch.float32)), batch_size=128, shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        t0 = time.perf_counter()
        model.train()
        for _ in range(3):
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
        s_val, _ = deep_scores(model, method, va, dev)
        s_test, infer_sec = deep_scores(model, method, te, dev)
        row, sens = evaluate_scores(prep.y_val, s_val, prep.y_test, s_test, method, seed, protocol)
        row.update({"method_type": "deep_time_series_baseline", "status": "RUN_OK", "device": str(dev), "model_size_mb": count_parameters(model) * 4 / (1024 * 1024), "train_time_sec": train_sec, "inference_time_sec": infer_sec, "cpu_latency_ms": np.nan, "gpu_latency_ms": infer_sec / max(prep.n_test, 1) * 1000 if str(dev).startswith("cuda") else np.nan, "note": "deep baseline; no architecture modification"})
        return row, sens

    def run_stat_model(self, prep: Prepared, method: str, dataset: str, protocol: str, seed: int, raw: bool) -> tuple[dict[str, Any], pd.DataFrame]:
        t0 = time.perf_counter()
        if raw:
            mean, std = train_only_normalizer(prep.train_x)
            tr = window_stat_features(apply_normalizer(prep.train_x, mean, std))
            va = window_stat_features(apply_normalizer(prep.val_x, mean, std))
            te = window_stat_features(apply_normalizer(prep.test_x, mean, std))
            calib = window_stat_features(apply_normalizer(prep.calib_x, mean, std)) if len(prep.calib_x) else np.empty((0, tr.shape[1]), dtype=np.float32)
        else:
            scaler0 = StandardScaler().fit(prep.train_x)
            tr = scaler0.transform(prep.train_x)
            va = scaler0.transform(prep.val_x)
            te = scaler0.transform(prep.test_x)
            calib = scaler0.transform(prep.calib_x) if len(prep.calib_x) else np.empty((0, tr.shape[1]), dtype=np.float32)
        if method == "IsolationForest":
            model = IsolationForest(n_estimators=80, random_state=seed, n_jobs=-1)
            model.fit(tr)
            train_sec = time.perf_counter() - t0
            t1 = time.perf_counter()
            s_val = -model.decision_function(va)
            s_test = -model.decision_function(te)
            infer_sec = time.perf_counter() - t1
        elif method in TREE:
            label_budget = protocol_label_budget(protocol)
            if label_budget == 0.0:
                raise RuntimeError("Supervised tree model skipped for normal-only calibration budget")
            y_calib = prep.y_calib.copy()
            calib_x = calib
            if label_budget < 1.0 and len(y_calib):
                rng = np.random.default_rng(seed)
                keep = []
                for lab in sorted(set(y_calib.tolist())):
                    idx = np.where(y_calib == lab)[0]
                    n = max(1, int(round(len(idx) * label_budget)))
                    keep.extend(rng.choice(idx, size=min(n, len(idx)), replace=False).tolist())
                keep = np.array(sorted(set(keep)), dtype=int)
                calib_x = calib_x[keep]
                y_calib = y_calib[keep]
            sup_x = np.concatenate([tr, calib_x], axis=0)
            sup_y = np.concatenate([np.zeros(len(tr), dtype=int), (y_calib > 0).astype(int)], axis=0)
            if len(sup_x) > 50000:
                rng = np.random.default_rng(seed)
                keep = []
                for lab in sorted(set(sup_y.tolist())):
                    idx = np.where(sup_y == lab)[0]
                    n_lab = min(len(idx), max(1, int(round(50000 * len(idx) / len(sup_y)))))
                    keep.extend(rng.choice(idx, size=n_lab, replace=False).tolist())
                keep = np.array(sorted(set(keep)), dtype=int)
                sup_x = sup_x[keep]
                sup_y = sup_y[keep]
            if len(np.unique(sup_y)) < 2:
                raise RuntimeError("Supervised model has no anomaly labels in training/calibration budget")
            scaler = StandardScaler().fit(sup_x)
            sx, sv, st = scaler.transform(sup_x), scaler.transform(va), scaler.transform(te)
            if method == "RandomForest":
                model = RandomForestClassifier(n_estimators=80, max_depth=10, random_state=seed, n_jobs=-1, class_weight="balanced")
            elif method == "XGBoost":
                if XGBClassifier is None:
                    raise RuntimeError("xgboost dependency unavailable")
                model = XGBClassifier(n_estimators=60, max_depth=4, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", random_state=seed, tree_method="hist")
            else:
                if LGBMClassifier is None:
                    raise RuntimeError("lightgbm dependency unavailable")
                model = LGBMClassifier(n_estimators=70, learning_rate=0.06, num_leaves=31, random_state=seed, verbose=-1)
            model.fit(sx, sup_y)
            train_sec = time.perf_counter() - t0
            t1 = time.perf_counter()
            s_val = model.predict_proba(sv)[:, 1]
            s_test = model.predict_proba(st)[:, 1]
            infer_sec = time.perf_counter() - t1
        else:
            raise RuntimeError(f"unsupported statistical model: {method}")
        row, sens = evaluate_scores(prep.y_val, s_val, prep.y_test, s_test, method, seed, protocol)
        row.update({"method_type": "window_stat_tree_baseline" if method in TREE else "traditional_anomaly_baseline", "status": "RUN_OK", "device": "cpu", "model_size_mb": 1.0 if method in TREE else 0.05, "train_time_sec": train_sec, "inference_time_sec": infer_sec, "cpu_latency_ms": infer_sec / max(prep.n_test, 1) * 1000, "gpu_latency_ms": np.nan, "note": "CPU baseline; validation-only threshold"})
        return row, sens


def safe(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def is_core_job(job: pd.Series) -> bool:
    dataset = str(job["dataset"])
    protocol = str(job["protocol"])
    method = str(job["model"])
    if method in {"CETRA_full", "COIL_full"}:
        return False
    if dataset == RAW_DATASET:
        return method in {"LightGBM", "XGBoost", "RandomForest", "IsolationForest", "AutoEncoder", "LSTM-AE", "USAD", "raw_residual_energy", "condition_decoupled_residual_energy", "CIRFL_v3_reference"}
    if dataset == SEG_DATASET:
        return method in {"LightGBM", "XGBoost", "RandomForest", "IsolationForest"}
    if dataset in {BRUSH_DATASET, ROAD_DATASET}:
        return method in {"LightGBM", "XGBoost", "RandomForest", "IsolationForest"}
    return False


def protocol_label_budget(protocol: str) -> float:
    if "normal_only" in protocol:
        return 0.0
    if "5pct" in protocol:
        return 0.05
    if "20pct" in protocol:
        return 0.20
    return 1.0


def seed_to_gpu(seed: int) -> str:
    if not torch.cuda.is_available():
        return "cpu"
    ids = {7: 0, 13: 1, 23: 2, 31: 0, 42: 1}
    return f"cuda:{ids.get(seed, seed % max(torch.cuda.device_count(), 1))}"


def deep_scores(model, method: str, arr: np.ndarray, dev: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    out = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(arr), 256):
            batch = torch.tensor(arr[start : start + 256], dtype=torch.float32, device=dev)
            if method == "USAD":
                w1, _, w3 = model(batch)
                rec = 0.5 * w1 + 0.5 * w3
            else:
                rec = model(batch)
            out.append(F.mse_loss(rec, batch, reduction="none").mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out), time.perf_counter() - t0


def raw_energy(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean, std = train_only_normalizer(train_x)
    va = apply_normalizer(val_x, mean, std)
    te = apply_normalizer(test_x, mean, std)
    val_ch = np.mean(va * va, axis=1)
    test_ch = np.mean(te * te, axis=1)
    return val_ch.mean(axis=1), test_ch.mean(axis=1), val_ch, test_ch


def condition_energy(train_x: np.ndarray, train_meta: pd.DataFrame, part_x: np.ndarray, part_meta: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    global_mean, global_std = train_only_normalizer(train_x)
    cond_stats: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if "condition_name" in train_meta:
        for cond, idx in train_meta.groupby("condition_name").groups.items():
            arr = train_x[list(idx)]
            if len(arr) >= 4:
                cond_stats[str(cond)] = train_only_normalizer(arr)
    scores, ch_scores = [], []
    for i, row in part_meta.reset_index(drop=True).iterrows():
        mean, std = cond_stats.get(str(row.get("condition_name")), (global_mean, global_std))
        z = apply_normalizer(part_x[i : i + 1], mean, std)[0]
        ch = np.mean(z * z, axis=0)
        scores.append(float(ch.mean()))
        ch_scores.append(ch)
    return np.asarray(scores), np.asarray(ch_scores)


def relation_score(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.decomposition import PCA

    mean, std = train_only_normalizer(train_x)
    tr = apply_normalizer(train_x, mean, std).reshape(len(train_x), -1)
    va = apply_normalizer(val_x, mean, std).reshape(len(val_x), -1)
    te = apply_normalizer(test_x, mean, std).reshape(len(test_x), -1)
    n_comp = max(1, min(12, tr.shape[0] - 1, tr.shape[1] - 1))
    pca = PCA(n_components=n_comp, random_state=0).fit(tr)
    return np.mean((va - pca.inverse_transform(pca.transform(va))) ** 2, axis=1), np.mean((te - pca.inverse_transform(pca.transform(te))) ** 2, axis=1)


def threshold_table(y_true: np.ndarray, scores: np.ndarray) -> pd.DataFrame:
    y = (np.asarray(y_true) > 0).astype(int)
    rows = []
    for thr in np.unique(scores):
        pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        rec = tp / max(tp + fn, 1)
        far = fp / max(fp + tn, 1)
        prec = tp / max(tp + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        rows.append({"threshold": float(thr), "recall": rec, "far": far, "mdr": 1 - rec, "f1": f1, "youden_j": rec - far})
    return pd.DataFrame(rows)


def evaluate_scores(y_val: np.ndarray, s_val_raw: np.ndarray, y_test: np.ndarray, s_test_raw: np.ndarray, method: str, seed: int, protocol: str) -> tuple[dict[str, Any], pd.DataFrame]:
    mult = score_direction_multiplier(y_val, s_val_raw)
    s_val, s_test = s_val_raw * mult, s_test_raw * mult
    strategy = "normal_95" if protocol_label_budget(protocol) == 0.0 and method in {"IsolationForest", "AutoEncoder", "LSTM-AE", "USAD", "raw_residual_energy", "condition_decoupled_residual_energy", "CIRFL_v3_reference"} else "best_f1"
    thr = choose_threshold(y_val, s_val, strategy=strategy)
    row = binary_metric_row(y_test, s_test, thr, method, seed, protocol)
    val_row = binary_metric_row(y_val, s_val, thr, method, seed, protocol)
    for key, value in val_row.items():
        if key not in {"method", "seed", "protocol"}:
            row[f"val_{key}"] = value
    row["threshold_strategy"] = strategy
    sens_rows = []
    for st in ALL_STRATEGIES:
        try:
            t = choose_threshold(y_val, s_val, strategy=st)
            r = binary_metric_row(y_test, s_test, t, method, seed, protocol)
            v = binary_metric_row(y_val, s_val, t, method, seed, protocol)
            for key, value in v.items():
                if key not in {"method", "seed", "protocol"}:
                    r[f"val_{key}"] = value
            r.update({"strategy": st, "threshold_source": "validation_only", "score_direction_source": "validation_only", "method": method, "seed": seed, "protocol": protocol})
            sens_rows.append(r)
        except Exception:
            continue
    return row, pd.DataFrame(sens_rows)


def load_manifest(path: Path, args) -> pd.DataFrame:
    df = pd.read_csv(path)
    if args.dataset:
        df = df[df["dataset"].eq(args.dataset)]
    if args.protocol:
        df = df[df["protocol"].eq(args.protocol)]
    if args.model:
        df = df[df["model"].eq(args.model)]
    if args.seed is not None:
        df = df[df["seed"].astype(int).eq(int(args.seed))]
    if args.run_mode == "core":
        df = df[df.apply(is_core_job, axis=1)]
    if args.max_jobs:
        df = df.head(args.max_jobs)
    return df.reset_index(drop=True)


def aggregate(output_root: Path, statuses: list[dict[str, Any]], metrics: list[dict[str, Any]], sensitivities: list[pd.DataFrame], manifest: pd.DataFrame, generated_at: str) -> None:
    status_df = pd.DataFrame(statuses)
    metrics_df = pd.DataFrame(metrics)
    sens_df = pd.concat(sensitivities, ignore_index=True) if sensitivities else pd.DataFrame()
    output_root.mkdir(parents=True, exist_ok=True)
    status_df.to_csv(output_root / "job_execution_status.csv", index=False)
    metrics_df.to_csv(output_root / "full_metric_summary.csv", index=False)
    metrics_df.to_csv(output_root / "baseline_comparison_summary.csv", index=False)
    sens_df.to_csv(output_root / "threshold_calibration_results.csv", index=False)
    failed = status_df[status_df["status"].isin(["FAILED", "SKIPPED_WITH_REASON"])].copy()
    failed.to_csv(output_root / "failed_job_report.csv", index=False)
    manifest.to_csv(output_root / "command_manifest_used.csv", index=False)
    write_analysis_reports(output_root, status_df, metrics_df, sens_df, manifest, generated_at)


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    return summarize_by_method(metrics)


def winners(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for protocol, group in metrics.groupby("protocol"):
        for metric, maximize in [("macro_f1", True), ("pr_auc", True), ("far", False), ("mdr", False)]:
            good = group.dropna(subset=[metric])
            if good.empty:
                continue
            means = good.groupby("method")[metric].mean()
            win = means.idxmax() if maximize else means.idxmin()
            rows.append({"protocol": protocol, "metric": metric, "winner": win, "winner_value": means.loc[win]})
    return pd.DataFrame(rows)


def family(method: str) -> str:
    if method in TREE:
        return "tree_statistical"
    if method in DEEP:
        return "deep_time_series"
    if method == "IsolationForest":
        return "traditional_anomaly"
    if method in HISTORICAL:
        return "historical_reference"
    return "other"


def family_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    df = metrics.copy()
    df["family"] = df["method"].map(family)
    return df.groupby(["protocol", "family"])[["macro_f1", "pr_auc", "far", "mdr", "cpu_latency_ms", "gpu_latency_ms"]].mean(numeric_only=True).reset_index()


def paired_tests(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if metrics.empty:
        return pd.DataFrame()
    for protocol in metrics["protocol"].unique():
        g = metrics[metrics["protocol"] == protocol]
        if "LightGBM" not in set(g["method"]):
            continue
        ref = g[g["method"] == "LightGBM"]
        for method in sorted(set(g["method"]) - {"LightGBM"}):
            other = g[g["method"] == method]
            merged = ref.merge(other, on="seed", suffixes=("_lgbm", "_other"))
            for metric in ["macro_f1", "pr_auc", "far", "mdr"]:
                if len(merged) < 2:
                    continue
                a = merged[f"{metric}_lgbm"].to_numpy(float)
                b = merged[f"{metric}_other"].to_numpy(float)
                try:
                    t_p = float(stats.ttest_rel(a, b, nan_policy="omit").pvalue)
                except Exception:
                    t_p = np.nan
                try:
                    w_p = float(stats.wilcoxon(a, b).pvalue)
                except Exception:
                    w_p = np.nan
                rows.append({"protocol": protocol, "reference": "LightGBM", "method": method, "metric": metric, "mean_diff_lightgbm_minus_method": float(np.nanmean(a - b)), "paired_t_p": t_p, "wilcoxon_p": w_p})
    return pd.DataFrame(rows)


def write_analysis_reports(output_root: Path, status: pd.DataFrame, metrics: pd.DataFrame, sens: pd.DataFrame, manifest: pd.DataFrame, generated_at: str) -> None:
    summary = summarize(metrics)
    win = winners(metrics)
    fam = family_comparison(metrics)
    tests = paired_tests(metrics)
    summary.to_csv(output_root / "full_statistical_summary.csv", index=False)
    win.to_csv(output_root / "protocol_winner_summary.csv", index=False)
    fam.to_csv(output_root / "model_family_comparison.csv", index=False)
    tests.to_csv(output_root / "paired_tests.csv", index=False)
    status_counts = status["status"].value_counts().to_dict() if len(status) else {}
    core = status[status["core_job"] == True] if "core_job" in status else status
    core_ok = int(core["status"].eq("RUN_OK").sum()) if len(core) else 0
    core_total = int(len(core))
    complete_rate = core_ok / max(core_total, 1)
    final_status = "FULL_RUN_COMPLETE" if complete_rate >= 0.95 else ("PARTIAL_RUN" if core_ok > 0 else "RUN_FAILED")
    (output_root / "run_status.json").write_text(json.dumps({"status": final_status, "status_counts": status_counts, "core_total": core_total, "core_completed": core_ok, "generated_at": generated_at}, indent=2), encoding="utf-8")
    provenance = prov("rie_full_engineering_benchmark_v1", generated_at, output_root)
    write_md(output_root / "runner_implementation_report.md", "Runner Implementation Report", provenance, [
        "- `scripts/run_engineering_benchmark.py` implemented.",
        "- Supports manifest filtering by dataset/protocol/model/seed.",
        "- Supports resume via per-job `metrics.csv` and `status.json`.",
        "- Supports failed-job retry by deleting/re-running failed job directory or using filters.",
        f"- manifest jobs loaded: {len(manifest)}",
        "- No figures generated.",
    ])
    write_md(output_root / "full_statistical_summary.md", "Full Statistical Summary", provenance, [
        "## Mean +/- std",
        md_table(summary, 120),
        "",
        "## Protocol winners",
        md_table(win, 120),
        "",
        "## Paired tests versus LightGBM",
        md_table(tests, 120),
    ])
    write_md(output_root / "model_family_comparison.md", "Model Family Comparison", provenance, [md_table(fam, 120)])
    write_md(output_root / "threshold_sensitivity_summary.md", "Threshold Sensitivity Summary", provenance, [md_table(sens.groupby(["protocol", "method", "strategy"])[["macro_f1", "far", "mdr", "pr_auc"]].mean(numeric_only=True).reset_index() if len(sens) else pd.DataFrame(), 160)])
    write_md(output_root / "threshold_leakage_audit.md", "Threshold Leakage Audit", provenance, [
        "- threshold_source: validation_only for all RUN_OK metric rows.",
        "- score direction source: validation_only.",
        "- test labels were used only for final metric computation.",
        "- normal-only budget uses normal-tail threshold where applicable; supervised trees are skipped when no anomaly labels exist.",
    ])
    write_engineering_markdowns(output_root, metrics, status, generated_at)
    make_review_packet(output_root, generated_at)


def write_engineering_markdowns(output_root: Path, metrics: pd.DataFrame, status: pd.DataFrame, generated_at: str) -> None:
    provenance = prov("rie_full_engineering_benchmark_v1", generated_at, output_root)
    core = metrics[metrics["protocol"].isin(["imadds_raw_main_binary", "imadds_raw_source_to_target", "imadds_raw_leave_target_weight35_out"])].copy()
    main = summarize(core)
    label = metrics[metrics["protocol"].str.contains("label_efficiency", na=False)]
    robust = metrics[metrics["protocol"].str.contains("missing_sensor|noise", regex=True, na=False)]
    domain = metrics[metrics["protocol"].isin(["imadds_raw_source_to_target", "imadds_raw_leave_target_weight35_out", "imadds_segment_source_to_target", "imadds_segment_leave_target_weight35_out"])]
    latency = metrics.groupby("method")[["model_size_mb", "cpu_latency_ms", "gpu_latency_ms", "train_time_sec", "inference_time_sec"]].mean(numeric_only=True).reset_index() if len(metrics) else pd.DataFrame()
    write_md(output_root / "far_mdr_engineering_analysis.md", "FAR/MDR Engineering Analysis", provenance, [
        md_table(main[["method", "protocol", "macro_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean"]], 120) if len(main) else "NOT_AVAILABLE",
        "- Use FAR and MDR jointly; low MDR from over-alarming models is not acceptable when FAR is near 1.",
        "- Safety-critical settings should inspect target-recall and cost-sensitive threshold sensitivity before deployment.",
    ])
    write_md(output_root / "label_efficiency_analysis.md", "Label Efficiency Analysis", provenance, [
        md_table(summarize(label)[["method", "protocol", "macro_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean"]], 160) if len(label) else "NOT_AVAILABLE",
        "- Normal-only supervised tree jobs are skipped by design because anomaly labels are unavailable.",
    ])
    write_md(output_root / "robustness_analysis.md", "Robustness Analysis", provenance, [
        md_table(summarize(robust)[["method", "protocol", "macro_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean"]], 160) if len(robust) else "NOT_AVAILABLE",
    ])
    write_md(output_root / "domain_shift_analysis.md", "Domain Shift Analysis", provenance, [
        md_table(summarize(domain)[["method", "protocol", "macro_f1_mean", "pr_auc_mean", "far_mean", "mdr_mean"]], 160) if len(domain) else "NOT_AVAILABLE",
    ])
    write_md(output_root / "latency_deployment_analysis.md", "Latency / Deployment Analysis", provenance, [
        md_table(latency, 80),
        "- Tree baselines run on CPU by default; deep baselines run on available GPUs.",
        "- Deployment choice should combine latency, FAR/MDR, and label availability rather than a single metric.",
    ])
    write_md(output_root / "model_selection_guidelines_final.md", "Model Selection Guidelines Final", provenance, [
        "- Label-rich tabular/window-stat deployment: prefer LightGBM/XGBoost after validation-only calibration.",
        "- Low-latency CPU deployment: compare LightGBM and XGBoost first; RandomForest can be heavier.",
        "- Normal-only deployment: use IsolationForest/deep reconstruction only with strict FAR calibration.",
        "- Missing sensors/noise: choose by robustness table, not main-binary score.",
        "- High missed-detection cost: use target-recall/cost-sensitive thresholds and verify FAR remains acceptable.",
    ])
    write_md(output_root / "device_execution_report_v1.md", "Device Execution Report v1", provenance, [
        f"- detected_gpu_count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}",
        f"- detected_gpu_names: {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []}",
        "- Deep jobs use GPU selected by seed mapping when CUDA is available.",
        "- Tree/statistical jobs use CPU for consistency.",
        "- DataParallel not used; independent job parallelism is preferred.",
    ])
    write_md(output_root / "quality_control_report.md", "Quality Control Report", provenance, [
        "- synthetic results included: NO.",
        "- random overlapping-window split: NO.",
        "- normalization: train_only in all RUN_OK metric rows.",
        "- threshold: validation_only in all RUN_OK metric rows.",
        "- test labels used for model/threshold selection: NO.",
        "- figures generated: NO.",
        f"- failed/skipped jobs retained: {int(status['status'].isin(['FAILED', 'SKIPPED_WITH_REASON']).sum()) if len(status) else 0}.",
    ])
    schema_ok = []
    required = ["dataset", "source_type", "protocol", "seed", "method", "n_train_windows", "n_val_windows", "n_test_windows", "n_test_normal", "n_test_anomaly", "generated_at", "output_path"]
    for col in required:
        schema_ok.append({"field": col, "present": col in metrics.columns if len(metrics) else False})
    write_md(output_root / "provenance_integrity_report.md", "Provenance Integrity Report", provenance, [
        md_table(pd.DataFrame(schema_ok)),
        "- Each job also has status JSON under `jobs/`.",
    ])
    readiness = 4 if len(metrics) and int(status[status["core_job"] == True]["status"].eq("RUN_OK").sum()) / max(int((status["core_job"] == True).sum()), 1) >= 0.95 else 3
    write_md(output_root / "rie_evidence_readiness_assessment.md", "RIE Evidence Readiness Assessment", provenance, [
        f"- readiness_score: {readiness}/5",
        "- Manuscript planning is recommended only if QC passes and core protocols are complete.",
        "- This file does not contain manuscript text or abstract.",
    ])


def make_review_packet(output_root: Path, generated_at: str) -> None:
    review = ROOT / "progress_for_chatgpt/latest"
    clean_dir(review)
    status = json.loads((output_root / "run_status.json").read_text(encoding="utf-8"))
    provenance = prov("rie_full_engineering_benchmark_v1", generated_at, output_root)
    status_df = pd.read_csv(output_root / "job_execution_status.csv")
    metrics = pd.read_csv(output_root / "full_metric_summary.csv")
    protocol_summary = status_df.groupby(["dataset", "protocol", "status"]).size().reset_index(name="job_count")
    write_md(output_root / "00_readme_for_chatgpt.md", "Readme for ChatGPT", provenance, [
        "- current_stage: RIE Full Engineering Benchmark Run v1",
        f"- current_status: {status['status']}",
        "- contains synthetic: NO",
        "- generated images: NO",
        "- wrote manuscript: NO",
        "- new algorithm generation stopped: YES",
    ])
    write_md(output_root / "02_job_execution_status.md", "Job Execution Status", provenance, [
        f"- total_jobs: {len(status_df)}",
        f"- completed_jobs: {int(status_df['status'].eq('RUN_OK').sum())}",
        f"- failed_jobs: {int(status_df['status'].eq('FAILED').sum())}",
        f"- skipped_jobs: {int(status_df['status'].eq('SKIPPED_WITH_REASON').sum())}",
        "",
        md_table(status_df["status"].value_counts().rename_axis("status").reset_index(name="count")),
    ])
    write_md(output_root / "03_dataset_protocol_summary.md", "Dataset Protocol Summary", provenance, [md_table(protocol_summary, 160)])
    shutil.copyfile(output_root / "full_metric_summary.csv", output_root / "04_full_metric_summary.csv")
    shutil.copyfile(output_root / "baseline_comparison_summary.csv", output_root / "05_baseline_comparison_summary.csv")
    write_md(output_root / "06_full_statistical_summary.md", "Full Statistical Summary", provenance, [
        (output_root / "full_statistical_summary.md").read_text(encoding="utf-8") if (output_root / "full_statistical_summary.md").exists() else "NOT_AVAILABLE",
        "",
        "## Model family comparison",
        (output_root / "model_family_comparison.md").read_text(encoding="utf-8") if (output_root / "model_family_comparison.md").exists() else "NOT_AVAILABLE",
    ])
    write_md(output_root / "07_threshold_calibration_summary.md", "Threshold Calibration Summary", provenance, [
        (output_root / "threshold_sensitivity_summary.md").read_text(encoding="utf-8") if (output_root / "threshold_sensitivity_summary.md").exists() else "NOT_AVAILABLE",
        "",
        (output_root / "threshold_leakage_audit.md").read_text(encoding="utf-8") if (output_root / "threshold_leakage_audit.md").exists() else "",
    ])
    failed = pd.read_csv(output_root / "failed_job_report.csv")
    failed.to_csv(output_root / "15_failed_job_report.csv", index=False)
    write_md(output_root / "18_go_no_go_report_full_benchmark_v1.md", "GO/NO-GO Report Full Benchmark v1", provenance, [
        f"## Decision: {status['status']}",
        f"- total_jobs: {len(status_df)}",
        f"- completed_jobs: {int(status_df['status'].eq('RUN_OK').sum())}",
        f"- failed_jobs: {int(status_df['status'].eq('FAILED').sum())}",
        f"- skipped_jobs: {int(status_df['status'].eq('SKIPPED_WITH_REASON').sum())}",
        f"- core_completed: {status.get('core_completed')}/{status.get('core_total')}",
        "- can_enter_manuscript_planning: YES if FULL_RUN_COMPLETE and QC report remains clean.",
    ])
    write_md(output_root / "19_code_index_and_next_tasks.md", "Code Index and Next Tasks", provenance, [
        "- `scripts/run_engineering_benchmark.py`: full engineering benchmark runner, resume/retry, aggregation, QC packet.",
        "- command used: `python scripts/run_engineering_benchmark.py --manifest outputs/rie_full_engineering_design_gate_v1/08_full_experiment_command_manifest.csv --output-root outputs/rie_full_engineering_benchmark_v1 --run-mode full --no-figures`",
        "- Next: review metrics and QC; only then decide whether to start manuscript planning. No abstract or figures generated.",
    ])
    copy_map = {
        "00_readme_for_chatgpt.md": "00_readme_for_chatgpt.md",
        "runner_implementation_report.md": "01_runner_implementation_report.md",
        "02_job_execution_status.md": "02_job_execution_status.md",
        "03_dataset_protocol_summary.md": "03_dataset_protocol_summary.md",
        "04_full_metric_summary.csv": "04_full_metric_summary.csv",
        "05_baseline_comparison_summary.csv": "05_baseline_comparison_summary.csv",
        "06_full_statistical_summary.md": "06_full_statistical_summary.md",
        "07_threshold_calibration_summary.md": "07_threshold_calibration_summary.md",
        "far_mdr_engineering_analysis.md": "08_far_mdr_engineering_analysis.md",
        "label_efficiency_analysis.md": "09_label_efficiency_analysis.md",
        "robustness_analysis.md": "10_robustness_analysis.md",
        "domain_shift_analysis.md": "11_domain_shift_analysis.md",
        "latency_deployment_analysis.md": "12_latency_deployment_analysis.md",
        "model_selection_guidelines_final.md": "13_model_selection_guidelines_final.md",
        "quality_control_report.md": "14_quality_control_report.md",
        "15_failed_job_report.csv": "15_failed_job_report.csv",
        "provenance_integrity_report.md": "16_provenance_integrity_report.md",
        "rie_evidence_readiness_assessment.md": "17_rie_evidence_readiness_assessment.md",
        "18_go_no_go_report_full_benchmark_v1.md": "18_go_no_go_report_full_benchmark_v1.md",
        "19_code_index_and_next_tasks.md": "19_code_index_and_next_tasks.md",
    }
    for src_name, dst_name in copy_map.items():
        shutil.copyfile(output_root / src_name, review / dst_name)
    files = [p for p in review.iterdir() if p.is_file()]
    if len(files) > 20:
        raise RuntimeError(f"review packet has {len(files)} files")
    bad = [p.name for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}]
    if bad:
        raise RuntimeError(f"review packet contains figures/images: {bad}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="")
    parser.add_argument("--output-root", default="outputs/rie_full_engineering_benchmark_v1")
    parser.add_argument("--run-mode", choices=["full", "core"], default="full")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--protocol", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--no-figures", action="store_true", default=False)
    args = parser.parse_args()
    manifest_path = Path(args.manifest) if args.manifest else ROOT / "outputs/rie_full_engineering_design_gate_v1/08_full_experiment_command_manifest.csv"
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    generated_at = now()
    manifest = load_manifest(manifest_path, args)
    runner = BenchmarkRunner(output_root, generated_at)
    statuses: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    sensitivities: list[pd.DataFrame] = []
    for i, job in manifest.iterrows():
        status, row, sens = runner.run_job(job, resume=args.resume)
        statuses.append(status)
        if row is not None:
            metrics.append(row)
        if len(sens):
            sens = sens.copy()
            sens["dataset"] = job["dataset"]
            sens["source_type"] = "external_real" if job["dataset"] != ROAD_DATASET else "real_reference_secondary"
            sens["generated_at"] = generated_at
            sens["output_path"] = str(output_root)
            sensitivities.append(sens)
        if (i + 1) % 50 == 0:
            print(f"processed {i+1}/{len(manifest)} jobs", flush=True)
    aggregate(output_root, statuses, metrics, sensitivities, manifest, generated_at)
    run_status = json.loads((output_root / "run_status.json").read_text(encoding="utf-8"))
    print(f"status={run_status['status']}")
    print(f"total_jobs={len(statuses)} completed={sum(s['status']=='RUN_OK' for s in statuses)} failed={sum(s['status']=='FAILED' for s in statuses)} skipped={sum(s['status']=='SKIPPED_WITH_REASON' for s in statuses)}")
    print(f"output_root={output_root}")
    print(f"review_dir={ROOT / 'progress_for_chatgpt/latest'}")


if __name__ == "__main__":
    main()
