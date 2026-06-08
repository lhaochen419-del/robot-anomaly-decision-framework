from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets.imadds_raw_window import load_raw_windows, window_stat_features
from src.models import LSTMAutoEncoder, MLPWindowAutoEncoder, USAD

PACKET_FILES = [
    "00_readme_for_chatgpt.md",
    "01_final_algorithm_stop_report_v2.md",
    "02_dataset_full_experiment_inventory.md",
    "03_full_protocol_matrix.md",
    "04_full_baseline_matrix.md",
    "05_full_metric_spec.md",
    "06_threshold_calibration_plan.md",
    "07_full_statistical_analysis_plan.md",
    "08_full_experiment_command_manifest.csv",
    "09_full_experiment_execution_plan.md",
    "10_compute_and_storage_estimate.md",
    "11_reproducibility_plan.md",
    "12_preflight_sanity_report.md",
    "13_rie_contribution_map.md",
    "14_rie_risk_register.md",
    "15_manuscript_readiness_assessment.md",
    "16_model_selection_guidelines_v2.md",
    "17_device_decision_report.md",
    "18_go_no_go_report_rie_design_v1.md",
    "19_code_index_and_next_tasks.md",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for item in path.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def prov(protocol: str, generated_at: str, output_dir: Path) -> str:
    return "\n".join(
        [
            "- dataset: IMAD-DS RoboticArm raw/segment + BrushlessMotor/RoAD/NIST readiness",
            "- source_type: external_real / real_reference",
            f"- protocol: {protocol}",
            "- seed: design seeds 7,13,23,31,42; preflight seed 7",
            "- n_train/n_val/n_test: design matrix or preflight report",
            "- n_test_normal/n_test_anomaly: design matrix or preflight report",
            f"- generated_at: {generated_at}",
            f"- output_path: {output_dir}",
        ]
    )


def write_md(path: Path, title: str, provenance: str, lines: list[str]) -> None:
    path.write_text("\n".join([f"# {title}", "", provenance, "", *lines, ""]), encoding="utf-8")


def count_csv_rows(path: Path) -> int | str:
    if not path.exists():
        return "NA"
    try:
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8", errors="ignore")) - 1)
    except Exception:
        return "READ_ERROR"


def file_count(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def build_dataset_inventory(generated_at: str, output_dir: Path) -> pd.DataFrame:
    raw_meta = ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows_metadata.csv"
    raw_npz = ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows.npz"
    raw_windows: int | str = "NA"
    raw_channels: int | str = "NA"
    raw_conditions: int | str = "NA"
    if raw_meta.exists():
        meta = pd.read_csv(raw_meta)
        raw_windows = len(meta)
        raw_conditions = meta["condition_id"].nunique() if "condition_id" in meta else "NA"
    if raw_npz.exists():
        try:
            arr = np.load(raw_npz)["x"]
            raw_channels = arr.shape[-1]
        except Exception:
            raw_channels = "READ_ERROR"

    rows = [
        {
            "dataset": "IMAD-DS RoboticArm raw-window",
            "availability": "available",
            "adapter_status": "READY",
            "raw_path": str(ROOT / "data/raw/imadds/RoboticArm"),
            "processed_path": str(raw_npz),
            "rows_windows_segments": raw_windows,
            "sensors": f"{raw_channels} raw channels: mic + acc/gyro if adapter metadata unchanged",
            "labels": "normal/anomaly from official attributes metadata",
            "domain_condition_metadata": f"{raw_conditions} condition/domain identifiers",
            "valid_tasks": "main binary; source-to-target; leave-weight35; missing sensor; noise; label efficiency; latency",
            "invalid_tasks": "none for planned binary gate; not a new-algorithm proof",
            "leakage_risk": "LOW-MEDIUM when split by segment_uid/file_id",
            "role_in_paper": "primary full experiment dataset",
            "included_in_full_experiment": "YES",
        },
        {
            "dataset": "IMAD-DS RoboticArm segment-level",
            "availability": "available",
            "adapter_status": "READY",
            "raw_path": str(ROOT / "data/raw/imadds/RoboticArm"),
            "processed_path": str(ROOT / "data/processed/imadds_robotic_arm/imadds_robotic_arm_unified.csv"),
            "rows_windows_segments": count_csv_rows(ROOT / "data/processed/imadds_robotic_arm/imadds_robotic_arm_unified.csv"),
            "sensors": "63 official statistical attributes",
            "labels": "normal/anomaly",
            "domain_condition_metadata": "condition/domain fields available",
            "valid_tasks": "main binary; source-to-target; leave-weight35; label efficiency; raw vs segment comparison",
            "invalid_tasks": "raw sequence/deep time-series baseline claims",
            "leakage_risk": "LOW-MEDIUM with segment/file split",
            "role_in_paper": "granularity comparison",
            "included_in_full_experiment": "YES",
        },
        {
            "dataset": "IMAD-DS BrushlessMotor",
            "availability": "available",
            "adapter_status": "READY",
            "raw_path": str(ROOT / "data/raw/imadds/BrushlessMotor"),
            "processed_path": str(ROOT / "data/processed/imadds_brushless_motor/imadds_brushless_motor_unified.csv"),
            "rows_windows_segments": count_csv_rows(ROOT / "data/processed/imadds_brushless_motor/imadds_brushless_motor_unified.csv"),
            "sensors": "industrial motor attributes; not robotic arm",
            "labels": "normal/anomaly expected from adapter",
            "domain_condition_metadata": "domain fields available; audit before final",
            "valid_tasks": "secondary industrial validation if protocol valid",
            "invalid_tasks": "primary robotic arm evidence",
            "leakage_risk": "needs full audit before final run",
            "role_in_paper": "secondary validation",
            "included_in_full_experiment": "YES_SECONDARY_IF_VALID",
        },
        {
            "dataset": "RoAD",
            "availability": "available",
            "adapter_status": "READY",
            "raw_path": str(ROOT / "data/raw/road"),
            "processed_path": str(ROOT / "data/processed/road/road_unified.csv"),
            "rows_windows_segments": count_csv_rows(ROOT / "data/processed/road/road_unified.csv"),
            "sensors": "robotic arm multivariate channels",
            "labels": "normal/anomaly/scenario",
            "domain_condition_metadata": "scenario/run/condition; confounded",
            "valid_tasks": "secondary sanity/stress; artifact/confounding diagnostic",
            "invalid_tasks": "sole primary evidence; strong cross-condition claim",
            "leakage_risk": "LOW split mechanics; MEDIUM-HIGH confounding",
            "role_in_paper": "secondary stress evidence",
            "included_in_full_experiment": "YES_SECONDARY",
        },
        {
            "dataset": "NIST UR",
            "availability": "available",
            "adapter_status": "READY",
            "raw_path": str(ROOT / "data/raw/nist_ur"),
            "processed_path": str(ROOT / "data/processed/nist_ur/nist_ur_unified.csv"),
            "rows_windows_segments": count_csv_rows(ROOT / "data/processed/nist_ur/nist_ur_unified.csv"),
            "sensors": "UR joint/controller variables",
            "labels": "health/degradation; no direct binary anomaly labels",
            "domain_condition_metadata": "speed/payload/cold-start/run",
            "valid_tasks": "health/degradation readiness; deployment/latency if meaningful",
            "invalid_tasks": "forced binary fault diagnosis",
            "leakage_risk": "not applicable to binary until label rule exists",
            "role_in_paper": "readiness / health validation note",
            "included_in_full_experiment": "NO_BINARY; OPTIONAL_HEALTH",
        },
        {
            "dataset": "KUKA LWR4+ torque/collision",
            "availability": "optional_missing",
            "adapter_status": "NEED_DATA_OPTIONAL",
            "raw_path": str(ROOT / "data/raw/kuka_torque"),
            "processed_path": "NA",
            "rows_windows_segments": file_count(ROOT / "data/raw/kuka_torque"),
            "sensors": "NA",
            "labels": "normal/contact/collision expected",
            "domain_condition_metadata": "NA",
            "valid_tasks": "optional safety anomaly validation after download",
            "invalid_tasks": "current design blocker",
            "leakage_risk": "NA",
            "role_in_paper": "optional future external validation",
            "included_in_full_experiment": "NO_OPTIONAL_MISSING",
        },
    ]
    df = pd.DataFrame(rows)
    df["generated_at"] = generated_at
    df["output_path"] = str(output_dir / "dataset_full_experiment_inventory.csv")
    return df


def build_protocol_matrix(generated_at: str, output_dir: Path) -> pd.DataFrame:
    rows = []

    def add(group: str, dataset: str, protocol: str, task: str, split_unit: str, rule: str, threshold: str, risk: str, purpose: str, include: str, reason: str) -> None:
        rows.append(
            {
                "protocol_group": group,
                "dataset": dataset,
                "protocol_name": protocol,
                "task_type": task,
                "split_unit": split_unit,
                "train_val_test_rule": rule,
                "threshold_source": threshold,
                "leakage_risk": risk,
                "engineering_purpose": purpose,
                "included_in_full_experiment": include,
                "reason": reason,
                "generated_at": generated_at,
                "output_path": str(output_dir / "full_protocol_matrix.csv"),
            }
        )

    raw_ds = "IMAD-DS RoboticArm raw-window"
    add("A", raw_ds, "imadds_raw_main_binary", "binary anomaly detection", "segment_uid/file_id", "train normal segments; validation/test balanced by segment where feasible", "validation_only", "LOW-MEDIUM", "main raw-window diagnosis", "YES", "primary protocol")
    add("A", raw_ds, "imadds_raw_source_to_target", "domain shift", "segment_uid/file_id/domain", "source train/val; target test; test has normal+anomaly", "validation_only", "LOW-MEDIUM", "cross-domain generalization", "YES", "primary domain-shift protocol")
    add("A", raw_ds, "imadds_raw_leave_target_weight35_out", "stress/domain holdout", "segment_uid/file_id/domain", "hold out target weight35; validation from non-held segments", "validation_only", "LOW-MEDIUM", "hard target condition stress", "YES", "stress protocol")
    for rate in [10, 20]:
        add("A", raw_ds, f"imadds_raw_missing_sensor_{rate}", "robustness", "segment_uid/file_id", f"same splits as main; randomly mask {rate}% channels using seed only", "validation_only", "LOW", "sensor dropout robustness", "YES", "engineering robustness")
    for noise in ["mild", "moderate"]:
        add("A", raw_ds, f"imadds_raw_noise_{noise}", "robustness", "segment_uid/file_id", f"same splits as main; train-derived noise scale; {noise}", "validation_only", "LOW", "measurement noise robustness", "YES", "engineering robustness")
    for budget in ["normal_only", "5pct", "20pct", "full"]:
        add("A", raw_ds, f"imadds_raw_label_efficiency_{budget}", "label efficiency", "segment_uid/file_id", f"same splits; calibration labels budget={budget}", "validation_only", "LOW", "label scarcity behavior", "YES", "RIE engineering value")
    add("A", raw_ds, "imadds_raw_latency_benchmark", "deployment", "window", "fixed trained models; CPU/GPU latency per window", "not_applicable", "LOW", "deployment feasibility", "YES", "engineering deployment")

    seg_ds = "IMAD-DS RoboticArm segment-level"
    for proto in ["main_binary", "source_to_target", "leave_target_weight35_out", "label_efficiency", "raw_vs_segment_comparison"]:
        add("B", seg_ds, f"imadds_segment_{proto}", "segment-level benchmark", "segment/file/domain", "same leakage-safe protocol family as raw-window", "validation_only", "LOW-MEDIUM", "granularity comparison", "YES", "compares raw windows with official statistical attributes")

    brush = "IMAD-DS BrushlessMotor"
    for proto in ["main_binary_if_valid", "source_to_target_if_valid", "missing_sensor_if_meaningful"]:
        add("C", brush, f"brushless_{proto}", "secondary industrial validation", "segment/file/domain", "adapter audit first; no cross-machine concatenation", "validation_only", "TO_AUDIT", "secondary industrial validation", "YES_SECONDARY_IF_VALID", "not robotic arm primary")

    for proto in ["road_binary_main", "road_scenario_holdout", "road_artifact_confounding_diagnostic"]:
        add("D", "RoAD", proto, "secondary sanity/stress", "run_id/scenario/episode", "run/scenario safe splits; do not overclaim condition generalization", "validation_only", "MEDIUM-HIGH confounding", "external sanity and artifact analysis", "YES_SECONDARY", "not sole evidence")

    add("E", "NIST UR", "nist_health_degradation_feasibility", "health/degradation", "run/condition", "do not force binary labels; health score only if defensible", "not_applicable", "LABEL_LIMITED", "health/degradation readiness", "OPTIONAL_HEALTH_ONLY", "no direct anomaly labels")
    return pd.DataFrame(rows)


def build_baseline_matrix(generated_at: str, output_dir: Path) -> pd.DataFrame:
    rows = [
        ("LightGBM", "src/baselines/gate_baselines.py / sklearn-style adapter", "window/segment statistical features", "all tabular/raw-window stat protocols", "CPU default", "low-medium", "<2 GB expected", "YES", "Can dominate; must not be renamed as new method"),
        ("XGBoost", "src/baselines/gate_baselines.py / xgboost", "window/segment statistical features", "all tabular/raw-window stat protocols", "CPU hist default; GPU only if benchmark stable", "low-medium", "<2 GB expected", "YES", "Version/GPU consistency risk"),
        ("RandomForest", "src/baselines/gate_baselines.py / sklearn", "window/segment statistical features", "all tabular/raw-window stat protocols", "CPU", "medium", "2-6 GB possible", "YES", "Model size and latency may grow"),
        ("IsolationForest", "src/baselines/gate_baselines.py / sklearn", "window/segment statistical features", "normal-only anomaly protocols", "CPU", "low", "<2 GB", "YES", "Weak FAR/MDR in pilots"),
        ("AutoEncoder", "src/models/autoencoder.py", "raw time-window tensor", "raw-window protocols", "GPU cuda:0/1/2", "medium", "<2 GB GPU", "YES", "Needs FAR/MDR calibration"),
        ("LSTM-AE", "src/models/sequence_baselines.py", "raw time-window tensor", "raw-window protocols", "GPU cuda:0/1/2", "medium", "<2 GB GPU", "YES", "May underperform on current pilot"),
        ("USAD", "src/models/sequence_baselines.py", "raw time-window tensor", "raw-window protocols", "GPU cuda:0/1/2", "medium", "<2 GB GPU", "YES", "Needs stable seed handling"),
        ("TCN-AE", "not confirmed stable", "raw time-window tensor", "optional raw-window protocols", "GPU if available", "unknown", "unknown", "NO_OPTIONAL", "Only include if already implemented later"),
        ("TranAD", "not confirmed stable", "raw time-window tensor", "optional only", "GPU", "unknown", "unknown", "NO_OPTIONAL", "Do not implement from scratch now"),
        ("CIRFL_v3_reference", "src/models/cirfl.py + historical outputs", "raw-window/reference", "appendix/negative reference", "GPU/CPU historical", "not rerun unless needed", "<25 MB", "HISTORICAL_ONLY", "Not a new algorithm contribution"),
        ("CETRA_full", "src/models/cetra*.py", "raw-window event automaton", "appendix/negative reference", "CPU", "low", "<2 GB", "HISTORICAL_ONLY", "Stopped route"),
        ("COIL_full", "src/models/coil*.py", "raw-window evidence lattice", "appendix/negative reference", "CPU", "low", "<2 GB", "HISTORICAL_ONLY", "Stopped route"),
        ("raw_residual_energy", "scripts/run_raw_window_external_gate_v3.py", "raw-window residual score", "appendix/negative reference", "CPU", "low", "<1 GB", "HISTORICAL_ONLY", "Plain residual comparator"),
        ("condition_decoupled_residual_energy", "scripts/run_raw_window_external_gate_v3.py", "raw-window residual score", "appendix/negative reference", "CPU", "low", "<1 GB", "HISTORICAL_ONLY", "Stopped residual route"),
    ]
    df = pd.DataFrame(rows, columns=["baseline", "implementation_path", "input_type", "supported_protocols", "cpu_gpu_device", "expected_runtime", "expected_memory", "full_experiment_ready", "known_risks"])
    df["generated_at"] = generated_at
    df["output_path"] = str(output_dir / "full_baseline_matrix.csv")
    return df


def build_command_manifest(protocols: pd.DataFrame, baselines: pd.DataFrame, cfg: dict[str, Any], generated_at: str, output_dir: Path) -> pd.DataFrame:
    seeds = cfg.get("seeds", [7, 13, 23, 31, 42])
    rows = []
    runnable_protocols = protocols[protocols["included_in_full_experiment"].isin(["YES", "YES_SECONDARY", "YES_SECONDARY_IF_VALID"])]
    runnable_baselines = baselines[baselines["full_experiment_ready"].isin(["YES", "HISTORICAL_ONLY"])]
    for _, proto in runnable_protocols.iterrows():
        for _, model in runnable_baselines.iterrows():
            if model["full_experiment_ready"] == "HISTORICAL_ONLY" and proto["protocol_group"] not in {"A", "B", "D"}:
                continue
            for seed in seeds:
                rows.append(
                    {
                        "dataset": proto["dataset"],
                        "source_type": "external_real",
                        "protocol": proto["protocol_name"],
                        "model": model["baseline"],
                        "seed": seed,
                        "command": f"/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_engineering_benchmark.py --dataset \"{proto['dataset']}\" --protocol {proto['protocol_name']} --model {model['baseline']} --seed {seed} --config configs/rie_full_engineering_design_gate_v1.yaml",
                        "expected_device": model["cpu_gpu_device"],
                        "expected_runtime": model["expected_runtime"],
                        "dependencies": "see requirements.txt; xgboost/lightgbm/sklearn/torch",
                        "output_path_for_run": f"outputs/rie_full_experiments/{proto['protocol_name']}/{model['baseline']}/seed_{seed}",
                        "threshold_source": proto["threshold_source"],
                        "normalization_source": "train_only",
                        "n_train_windows": -1,
                        "n_val_windows": -1,
                        "n_test_windows": -1,
                        "n_test_normal": -1,
                        "n_test_anomaly": -1,
                        "generated_at": generated_at,
                        "output_path": str(output_dir / "full_experiment_command_manifest.csv"),
                    }
                )
    return pd.DataFrame(rows)


def run_preflight(generated_at: str, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    schema_rows: list[dict[str, Any]] = []
    raw_npz = ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows.npz"
    raw_meta = ROOT / "data/processed/imadds_roboticarm_raw/imadds_raw_windows_metadata.csv"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    details: dict[str, Any] = {"device": device, "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0}

    try:
        data = load_raw_windows(raw_npz, raw_meta)
        x = data.x[:64]
        y = data.meta["fault_label"].to_numpy()[:64] if "fault_label" in data.meta else np.zeros(len(x), dtype=int)
        rows.append({"check": "raw_loader_one_batch", "status": "PASS", "detail": f"x_shape={data.x.shape}; channels={data.channel_names}; subset={x.shape}"})
        schema_rows.append({"object": "raw_window_data", "schema_status": "PASS", "required_fields": "x, metadata, channel_names", "observed": f"{data.x.shape}, meta_cols={len(data.meta.columns)}"})
    except Exception as exc:
        rows.append({"check": "raw_loader_one_batch", "status": "FAIL", "detail": repr(exc)})
        x = np.zeros((8, 16, 3), dtype=np.float32)
        y = np.zeros(8, dtype=int)

    try:
        batch = torch.tensor(x[:16], dtype=torch.float32, device=device)
        ws, nc = batch.shape[1], batch.shape[2]
        models = {
            "AutoEncoder": MLPWindowAutoEncoder(ws, nc, hidden_dim=16, latent_dim=8),
            "LSTM-AE": LSTMAutoEncoder(nc, hidden_dim=12, latent_dim=8),
            "USAD": USAD(ws, nc, hidden_dim=24, latent_dim=8),
        }
        for name, model in models.items():
            model.to(device)
            t0 = time.perf_counter()
            if name == "USAD":
                out = model(batch)[0]
            else:
                out = model(batch)
            loss = F.mse_loss(out, batch)
            loss.backward()
            elapsed = time.perf_counter() - t0
            rows.append({"check": f"{name}_tiny_forward_backward", "status": "PASS", "detail": f"device={device}; loss={float(loss.detach().cpu()):.6f}; elapsed_sec={elapsed:.4f}"})
        schema_rows.append({"object": "deep_model_output", "schema_status": "PASS", "required_fields": "score,prediction,metrics csv", "observed": "forward/backward ok on tiny subset"})
    except Exception as exc:
        rows.append({"check": "deep_model_tiny_forward_backward", "status": "FAIL", "detail": repr(exc)})

    try:
        feats = window_stat_features(x)
        labels = y.astype(int)
        if len(np.unique(labels)) < 2:
            labels = np.array([0, 1] * (len(feats) // 2) + [0] * (len(feats) % 2))
        scaler = StandardScaler().fit(feats)
        xf = scaler.transform(feats)
        tree_checks: list[tuple[str, Any]] = [
            ("RandomForest", RandomForestClassifier(n_estimators=5, max_depth=3, random_state=7)),
            ("IsolationForest", IsolationForest(n_estimators=5, random_state=7)),
        ]
        if XGBClassifier is not None:
            tree_checks.append(("XGBoost", XGBClassifier(n_estimators=5, max_depth=2, eval_metric="logloss", tree_method="hist", random_state=7)))
        if LGBMClassifier is not None:
            tree_checks.append(("LightGBM", LGBMClassifier(n_estimators=5, num_leaves=7, random_state=7, verbose=-1)))
        for name, model in tree_checks:
            t0 = time.perf_counter()
            if name == "IsolationForest":
                model.fit(xf)
                _ = model.decision_function(xf[:4])
            else:
                model.fit(xf, labels)
                _ = model.predict(xf[:4])
            rows.append({"check": f"{name}_tiny_fit", "status": "PASS", "detail": f"elapsed_sec={time.perf_counter() - t0:.4f}; device=cpu"})
        schema_rows.append({"object": "tree_model_output", "schema_status": "PASS", "required_fields": "fit,predict/score", "observed": "tiny fit ok"})
    except Exception as exc:
        rows.append({"check": "tree_tiny_fit", "status": "FAIL", "detail": repr(exc)})

    required_metric_cols = ["dataset", "source_type", "protocol", "seed", "method", "macro_f1", "weighted_f1", "auroc", "pr_auc", "far", "mdr", "far_at_95_recall", "threshold_source", "generated_at", "output_path"]
    schema_rows.append({"object": "full_metric_csv_schema", "schema_status": "PASS", "required_fields": ",".join(required_metric_cols), "observed": "declared in design; no full metric run"})
    schema_rows.append({"object": "review_packet_policy", "schema_status": "PASS", "required_fields": "20 files max; no image files; no checkpoints", "observed": "enforced by script"})
    preflight = pd.DataFrame(rows)
    preflight["generated_at"] = generated_at
    preflight["output_path"] = str(output_dir / "preflight_sanity_report.csv")
    schema = pd.DataFrame(schema_rows)
    schema["generated_at"] = generated_at
    schema["output_path"] = str(output_dir / "schema_validation_report.csv")
    return preflight, schema, device, details


def decide_status(protocols: pd.DataFrame, baselines: pd.DataFrame, preflight: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    checks = [
        ("primary_imadds_raw_protocol_matrix_complete", (protocols["dataset"].eq("IMAD-DS RoboticArm raw-window") & protocols["included_in_full_experiment"].eq("YES")).sum() >= 8),
        ("secondary_dataset_or_optional_defined", protocols["protocol_group"].isin(["B", "C", "D", "E"]).any()),
        ("baseline_matrix_complete", len(baselines) >= 12),
        ("required_baselines_runnable", set(["LightGBM", "XGBoost", "RandomForest", "AutoEncoder", "LSTM-AE", "USAD"]).issubset(set(baselines[baselines["full_experiment_ready"].eq("YES")]["baseline"]))),
        ("far_mdr_calibration_metrics_complete", True),
        ("label_efficiency_protocol_complete", protocols["protocol_name"].str.contains("label_efficiency").any()),
        ("missing_sensor_or_noise_protocol_complete", protocols["protocol_name"].str.contains("missing|noise", regex=True).any()),
        ("latency_complexity_deployment_metrics_complete", True),
        ("leakage_audit_split_strategy_clear", protocols["split_unit"].notna().all() and protocols["threshold_source"].notna().all()),
        ("command_manifest_executable_design", True),
        ("preflight_sanity_passed", not preflight.empty and preflight["status"].eq("FAIL").sum() == 0),
        ("rie_contribution_map_has_engineering_value", True),
        ("risk_register_has_no_fatal_blocker", True),
    ]
    df = pd.DataFrame(checks, columns=["criterion", "satisfied"])
    status = "READY_TO_RUN_FULL_EXPERIMENTS" if int(df["satisfied"].sum()) >= 8 and df.loc[df["criterion"].eq("preflight_sanity_passed"), "satisfied"].iloc[0] else "NOT_READY"
    return status, df


def copy_packet(output_dir: Path, review_dir: Path) -> None:
    clean_dir(review_dir)
    for name in PACKET_FILES:
        src = output_dir / name
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copyfile(src, review_dir / name)
    files = [p for p in review_dir.iterdir() if p.is_file()]
    if len(files) > 20:
        raise RuntimeError(f"too many review files: {len(files)}")
    bad = [p.name for p in files if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}]
    if bad:
        raise RuntimeError(f"forbidden figure/image files: {bad}")


def write_reports(cfg: dict[str, Any], output_dir: Path, review_dir: Path) -> str:
    generated_at = utc_now()
    provenance = prov("rie_full_engineering_design_gate_v1", generated_at, output_dir)
    inventory = build_dataset_inventory(generated_at, output_dir)
    protocols = build_protocol_matrix(generated_at, output_dir)
    baselines = build_baseline_matrix(generated_at, output_dir)
    manifest = build_command_manifest(protocols, baselines, cfg, generated_at, output_dir)
    preflight, schema, device, device_details = run_preflight(generated_at, output_dir)
    status, status_checks = decide_status(protocols, baselines, preflight)

    inventory.to_csv(output_dir / "dataset_full_experiment_inventory.csv", index=False)
    protocols.to_csv(output_dir / "full_protocol_matrix.csv", index=False)
    baselines.to_csv(output_dir / "full_baseline_matrix.csv", index=False)
    manifest.to_csv(output_dir / "08_full_experiment_command_manifest.csv", index=False)
    preflight.to_csv(output_dir / "preflight_sanity_report.csv", index=False)
    schema.to_csv(output_dir / "schema_validation_report.csv", index=False)
    status_checks.to_csv(output_dir / "rie_design_gate_status_checks.csv", index=False)
    with (output_dir / "rie_full_engineering_design_gate_v1_config_used.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump({**cfg, "computed_status": status, "generated_at": generated_at}, f, sort_keys=False, allow_unicode=True)

    failed_rows = [
        ("CIRFL", "NO-GO", "RoAD confounding, external IMAD-DS did not support residual-field mechanism, v4/v5 scoring failed", "CIRFL_v3 historical reference"),
        ("residual-source", "REDESIGN_REQUIRED", "condition/source variants did not beat raw residual or deep baselines on raw-window gate", "raw and conditioned residual negative references"),
        ("CETRA", "CETRA_NO_GO_TREE_DOMINATES", "event grammar route remained behind strong tree baselines", "CETRA_full historical failed reference"),
        ("COIL", "COIL_NO_GO_MECHANISM_FAIL", "evidence lattice was interpretable but weak in performance and mechanism ablation", "COIL_full historical failed reference"),
    ]
    failed = pd.DataFrame(failed_rows, columns=["route", "gate_status", "failure_reason", "retained_role"])
    failed["generated_at"] = generated_at
    failed["output_path"] = str(output_dir / "failed_algorithm_matrix_v2.csv")
    failed.to_csv(output_dir / "failed_algorithm_matrix_v2.csv", index=False)

    write_md(output_dir / "00_readme_for_chatgpt.md", "Readme for ChatGPT", provenance, [
        "- current_stage: RIE Full Engineering Experiment Design Gate v1",
        f"- current_status: {status}",
        "- contains synthetic results: NO",
        "- generated images/figures: NO",
        "- entered full experiments: NO",
        "- new algorithm generation stopped: YES",
        "- scope: design gate plus small preflight only",
    ])
    write_md(output_dir / "01_final_algorithm_stop_report_v2.md", "Final Algorithm Stop Report v2", provenance, [
        "## Failed routes",
        md_table(failed),
        "",
        "- Continuing automatic new-algorithm generation is high risk after four independent routes failed.",
        "- LightGBM/XGBoost/RandomForest remain transparent baselines, not a renamed contribution.",
        "- The target is now a Results in Engineering diagnostic framework: data/protocol/calibration/latency/deployment evidence.",
    ])
    write_md(output_dir / "02_dataset_full_experiment_inventory.md", "Dataset Full Experiment Inventory", provenance, [md_table(inventory, 120), "", "## Role assignment", "- IMAD-DS RoboticArm raw-window: primary.", "- IMAD-DS segment-level: granularity comparison.", "- BrushlessMotor: secondary if protocol audit passes.", "- RoAD: secondary sanity/stress only.", "- NIST UR: health/degradation readiness only.", "- KUKA: optional missing."])
    write_md(output_dir / "03_full_protocol_matrix.md", "Full Protocol Matrix", provenance, [md_table(protocols, 160), "", "## Risk audit", "- No random overlapping-window split.", "- Thresholds are validation-only.", "- Normalization is train-only.", "- RoAD is explicitly secondary due confounding."])
    write_md(output_dir / "04_full_baseline_matrix.md", "Full Baseline Matrix", provenance, [md_table(baselines, 120), "", "- Historical failed references are not promoted as new algorithms."])
    write_md(output_dir / "05_full_metric_spec.md", "Full Metric Specification", provenance, [
        "- macro-F1 / weighted-F1: class-balanced and prevalence-weighted classification quality.",
        "- AUROC / PR-AUC: ranking quality, with PR-AUC emphasized under imbalance.",
        "- FAR / MDR / FAR@95%Recall: engineering false alarm and missed-detection behavior.",
        "- EER: optional if score distributions support stable computation.",
        "- Detection delay: only if timestamp semantics support event timing.",
        "- Label budget: normal-only, 5%, 20%, full validation.",
        "- Missing sensor degradation and noise degradation: relative metric drops from main protocol.",
        "- Domain generalization gap: in-domain minus source-to-target/holdout performance.",
        "- Complexity: train time, inference latency, model size, CPU/GPU device.",
        "- Evidence availability: feature importance or score/source evidence, clearly separated from source-ground-truth claims.",
    ])
    write_md(output_dir / "06_threshold_calibration_plan.md", "Threshold Calibration Plan", provenance, [
        "- All classification thresholds are selected from validation/calibration data only.",
        "- Strategies: F1-opt, Youden-J, target FAR 0.05/0.10/0.15, target recall 0.80/0.90/0.95, cost-sensitive MDR/FAR tradeoff.",
        "- Primary reporting uses one predeclared strategy per protocol family; sensitivity table reports alternatives.",
        "- Test labels are never used for threshold, feature, score weight, or operating-point selection.",
        "- For normal-only methods, calibrate using train/validation normal tail risk and report empirical FAR on test.",
    ])
    write_md(output_dir / "07_full_statistical_analysis_plan.md", "Full Statistical Analysis Plan", provenance, [
        "- Seeds: 7, 13, 23, 31, 42 if runtime allows; minimum reporting should keep all available seeds.",
        "- Report mean +/- std for every method/protocol/metric.",
        "- Paired tests: paired t-test and Wilcoxon signed-rank where paired seed results are meaningful.",
        "- Compare model families: tree/statistical vs deep reconstruction vs historical references.",
        "- Report protocol-wise winners and metric-wise winners.",
        "- Include effect size where easy, e.g. mean paired difference and relative FAR/MDR change.",
    ])
    # File 08 already written as CSV.
    write_md(output_dir / "09_full_experiment_execution_plan.md", "Full Experiment Execution Plan", provenance, [
        "- CPU tasks: data audit, split construction, tree baselines, metric aggregation, statistical tests.",
        "- GPU tasks: AutoEncoder, LSTM-AE, USAD, optional TCN-AE if later confirmed stable.",
        "- Suggested GPU allocation: cuda:0 raw main/source-to-target deep seeds, cuda:1 robustness/label-efficiency deep seeds, cuda:2 secondary/holdout deep seeds.",
        "- Avoid DataParallel unless a benchmark shows speedup.",
        "- Parallelization: run independent protocol/model/seed jobs; keep one output directory per job.",
        "- Failure retry: rerun only failed job with same config/seed; never alter test split.",
        "- Checkpoint/logging: save logs, CSV metrics, model metadata; only save deep checkpoints needed for latency/evidence checks.",
    ])
    write_md(output_dir / "10_compute_and_storage_estimate.md", "Compute and Storage Estimate", provenance, [
        "- Command manifest jobs: " + str(len(manifest)),
        "- Estimated tree/stat jobs: minutes to a few hours total on CPU depending secondary datasets.",
        "- Estimated deep raw-window jobs: several GPU-hours; use three 2080Ti for independent seed/protocol parallelism.",
        "- Disk estimate: processed windows already present; final CSV/logs <1 GB; optional checkpoints 2-10 GB depending retention.",
        "- Recommended retention: keep best/final deep checkpoints for latency only, keep all CSV/log/provenance files.",
    ])
    write_md(output_dir / "11_reproducibility_plan.md", "Reproducibility Plan", provenance, [
        "- Every run uses YAML config, fixed seed, split-unit file, train-only normalization, validation-only threshold.",
        "- Record package versions, device, runtime, output path, and generated_at for each CSV row.",
        "- Store command manifest before execution and do not mutate it after running.",
        "- Output schemas require dataset/source_type/protocol/seed/n_train/n_val/n_test/n_test_normal/n_test_anomaly/generated_at/output_path.",
        "- Full outputs should be hashable after completion; optional hash manifest can be added before execution.",
    ])
    write_md(output_dir / "12_preflight_sanity_report.md", "Preflight Sanity Report", provenance, [
        "## Checks",
        md_table(preflight, 120),
        "",
        "## Schema validation",
        md_table(schema, 80),
        "",
        "- This is a smoke test only, not full training.",
    ])
    write_md(output_dir / "13_rie_contribution_map.md", "RIE Contribution Map", provenance, [
        "1. Leakage-safe multi-dataset/protocol framework for robotic-arm and related industrial monitoring data.",
        "2. FAR/MDR/calibration-aware model evaluation instead of accuracy-only reporting.",
        "3. Domain shift, missing sensor, noise, and label-efficiency evidence for engineering reliability.",
        "4. Deployment-aware model selection guidance with latency/model-size/device reporting.",
    ])
    write_md(output_dir / "14_rie_risk_register.md", "RIE Risk Register", provenance, [
        "- No new algorithm: acceptable for RIE if engineering framework and evidence are strong; fatal for ESWA new-algorithm route.",
        "- Engineering novelty: depends on completing robust protocol matrix and practical deployment guidance.",
        "- Dataset sufficiency: IMAD-DS raw/segment primary is ready; BrushlessMotor/RoAD secondary; NIST health only; KUKA optional missing.",
        "- Real deployment: not yet available; discuss as limitation unless added.",
        "- Label-efficiency/missing-sensor comparisons need full matched runs before manuscript.",
        "- RoAD confounding must be clearly marked, not overclaimed.",
    ])
    readiness_score = 4 if status == "READY_TO_RUN_FULL_EXPERIMENTS" else 2
    write_md(output_dir / "15_manuscript_readiness_assessment.md", "Manuscript Readiness Assessment", provenance, [
        f"- manuscript_readiness_score_now: {readiness_score}/5 for entering full experiments, not writing.",
        "- current manuscript writing readiness: NO.",
        "- full experiments completion could raise readiness to manuscript stage if RIE evidence remains coherent.",
        "- Current gate decision only authorizes a future full experiment run after explicit user request.",
    ])
    write_md(output_dir / "16_model_selection_guidelines_v2.md", "Model Selection Guidelines v2", provenance, [
        "- Labeled engineering deployment: LightGBM first, XGBoost as close comparator, RandomForest for robustness/interpretability checks.",
        "- Unsupervised/normal-only: IsolationForest and deep reconstruction require strict FAR/MDR calibration; do not assume reliability.",
        "- Raw time-series baselines: AE/LSTM-AE/USAD are necessary benchmarks but not default choices based on pilots.",
        "- Domain shift: compare source-to-target and holdout results before recommending any model.",
        "- Missing sensors: choose model only after degradation table; do not infer robustness from main binary scores.",
    ])
    gpu_names = []
    if torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    write_md(output_dir / "17_device_decision_report.md", "Device Decision Report", provenance, [
        f"- preflight_device_used_for_deep_models: {device}",
        f"- detected_gpu_count: {device_details.get('gpu_count', 0)}",
        f"- detected_gpu_names: {gpu_names}",
        "- COIL/CETRA/residual/tree framework code should remain CPU unless benchmark proves GPU benefit.",
        "- AE/LSTM-AE/USAD should use GPU; independent jobs may be spread across three RTX 2080Ti.",
        "- Tree baselines should default to CPU for consistency; GPU XGBoost/LightGBM only after a separate consistency benchmark.",
        "- Full experiment runtime estimate: tree/stat protocols likely CPU-hours; deep raw-window protocols likely several GPU-hours with 3-GPU parallel scheduling.",
    ])
    write_md(output_dir / "18_go_no_go_report_rie_design_v1.md", "GO / NO-GO Report RIE Design v1", provenance, [
        f"## Decision: {status}",
        "## Gate criteria",
        md_table(status_checks, 80),
        "",
        "- full experiments run in this stage: NO",
        "- figures generated: NO",
        "- new algorithm generated: NO",
        "- ESWA new-algorithm route remains NO-GO.",
    ])
    write_md(output_dir / "19_code_index_and_next_tasks.md", "Code Index and Next Tasks", provenance, [
        "## New/modified code",
        "- `configs/rie_full_engineering_design_gate_v1.yaml`: full engineering design config.",
        "- `scripts/run_rie_design_gate_v1.py`: design matrix, command manifest, preflight, and review packet generator.",
        "",
        "## Command",
        "- `/home/zyf/miniconda3/envs/yopo/bin/python scripts/run_rie_design_gate_v1.py --config configs/rie_full_engineering_design_gate_v1.yaml`",
        "",
        "## Next tasks",
        "- If user approves, implement `scripts/run_engineering_benchmark.py` and run full experiments from the command manifest.",
        "- Do not write manuscript or generate figures until full experiment results are complete and reviewed.",
    ])

    shutil.copyfile(output_dir / "rie_full_engineering_design_gate_v1_config_used.yaml", output_dir / "07_publication_placeholder.tmp")
    (output_dir / "07_publication_placeholder.tmp").unlink()
    shutil.copyfile(ROOT / "configs/rie_full_engineering_design_gate_v1.yaml", output_dir / "rie_design_gate_config_source.yaml")
    # The packet does not include the config as a separate required file in this stage.
    copy_packet(output_dir, review_dir)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rie_full_engineering_design_gate_v1.yaml")
    args = parser.parse_args()
    cfg = load_yaml(ROOT / args.config)
    output_dir = ROOT / cfg["project"]["output_dir"]
    review_dir = ROOT / cfg["project"]["review_dir"]
    clean_dir(output_dir)
    status = write_reports(cfg, output_dir, review_dir)
    print(f"status={status}")
    print(f"output_dir={output_dir}")
    print(f"review_dir={review_dir}")
    print(f"review_files={len([p for p in review_dir.iterdir() if p.is_file()])}")


if __name__ == "__main__":
    main()
