from __future__ import annotations

import argparse
import math
import shutil
import sys
from pathlib import Path
from copy import deepcopy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_road_protocols import PROTOCOLS, config_for_protocol
from scripts.make_review_packet import make_review_packet
from scripts.prepare_real_data import prepare_one
from src.baselines import run_all_baselines
from src.datasets import audit_dataset, build_datasets_from_config, write_audit_reports
from src.evaluation import measure_cirfl_latency
from src.evaluation.metrics import binary_metric_row, choose_threshold, paired_tests, score_direction_multiplier, summarize_by_method, threshold_strategy_rows
from src.models import CIRFL
from src.training import predict_cirfl, train_cirfl_once
from src.utils.config import load_config, save_config
from src.utils.logging import setup_logger
from src.utils.markdown import dataframe_to_markdown
from src.utils.provenance import add_provenance, utc_now
from src.utils.torch_utils import resolve_device

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 8, "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False})

DATASET = "RoAD"
SOURCE_TYPE = "real"


def _clean_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in output_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir() and item.name in {"tmp"}:
            shutil.rmtree(item)


def _cfg_for(proto: dict, base: dict) -> tuple[dict, str]:
    cfg = config_for_protocol(proto)
    # Preserve configurable top-level settings from the selected v2 config.
    for key in ["seeds", "device", "model", "training", "baselines", "latency", "review_packet"]:
        if key in base:
            cfg[key] = deepcopy(base[key])
    cfg["project"]["output_dir"] = base["project"].get("output_dir", "outputs/real_gate_v2")
    split_mode = "cross_condition" if proto["family"] == "condition_holdout" else "main"
    return cfg, split_mode


def _load_model(cfg: dict, n_channels: int, ckpt: Path) -> CIRFL:
    device = resolve_device(cfg.get("device", "auto"))
    mc = cfg["model"]
    model = CIRFL(
        n_channels=n_channels,
        window_size=cfg["data"]["window_size"],
        num_classes=2,
        hidden_dim=mc["hidden_dim"],
        condition_dim=mc["condition_dim"],
        residual_dim=mc["residual_dim"],
        n_relation_atoms=mc["n_relation_atoms"],
        max_conditions=mc["max_conditions"],
        dropout=mc["dropout"],
        min_scale=mc.get("min_scale", 0.05),
        log_scale_clip=mc.get("log_scale_clip", 5.0),
        energy_clip=mc.get("energy_clip", 100000.0),
        atom_temperature=mc.get("atom_temperature", 0.7),
        atom_usage_floor=mc.get("atom_usage_floor", 0.0),
        residual_score_weight=mc.get("residual_score_weight", 1.0),
        prototype_score_weight=mc.get("prototype_score_weight", 0.0),
        axis_score_weight=mc.get("axis_score_weight", 0.0),
        score_margin=mc.get("score_margin", 0.5),
    ).to(device)
    payload = torch.load(ckpt, map_location=device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _count_row(ds, prefix: str) -> dict:
    y = ds.labels()
    return {f"n_{prefix}_windows": len(ds), f"n_{prefix}_normal": int((y == 0).sum()), f"n_{prefix}_anomaly": int((y > 0).sum())}


def _add_counts(row: dict, train_ds, val_ds, test_ds) -> dict:
    row.update(_count_row(train_ds, "train"))
    row.update(_count_row(val_ds, "val"))
    row.update(_count_row(test_ds, "test"))
    return row


def _score_sanity_for_predictions(pred: pd.DataFrame) -> dict:
    y = pred["y_true"].to_numpy()
    s = pred["score"].to_numpy(float)
    normal = s[y == 0]
    anomaly = s[y > 0]
    out = {
        "normal_score_mean": float(np.mean(normal)) if len(normal) else float("nan"),
        "normal_score_median": float(np.median(normal)) if len(normal) else float("nan"),
        "normal_score_p95": float(np.percentile(normal, 95)) if len(normal) else float("nan"),
        "normal_score_p99": float(np.percentile(normal, 99)) if len(normal) else float("nan"),
        "anomaly_score_mean": float(np.mean(anomaly)) if len(anomaly) else float("nan"),
        "anomaly_score_median": float(np.median(anomaly)) if len(anomaly) else float("nan"),
        "anomaly_score_p05": float(np.percentile(anomaly, 5)) if len(anomaly) else float("nan"),
        "anomaly_score_p50": float(np.percentile(anomaly, 50)) if len(anomaly) else float("nan"),
        "anomaly_score_p95": float(np.percentile(anomaly, 95)) if len(anomaly) else float("nan"),
        "max_score": float(np.nanmax(s)) if len(s) else float("nan"),
        "min_score": float(np.nanmin(s)) if len(s) else float("nan"),
        "number_of_nan": int(np.isnan(s).sum()),
        "number_of_inf": int(np.isinf(s).sum()),
        "score_explosion_flag": bool(np.nanmax(s) > 1e6) if len(s) else True,
    }
    if len(np.unique(y > 0)) > 1:
        auc = roc_auc_score((y > 0).astype(int), s)
        inv_auc = roc_auc_score((y > 0).astype(int), -s)
        out["score_auroc"] = float(auc)
        out["inverted_score_auroc"] = float(inv_auc)
        out["score_direction_check"] = "PASS" if auc >= inv_auc else "FAIL"
    else:
        out["score_auroc"] = float("nan")
        out["inverted_score_auroc"] = float("nan")
        out["score_direction_check"] = "UNDEFINED"
    return out


def _threshold_and_score_report(protocol_rows: list[dict], output_dir: Path) -> None:
    df = pd.DataFrame(protocol_rows)
    df.to_csv(output_dir / "threshold_score_sanity.csv", index=False)
    lines = ["# Threshold and Score Sanity", "", "All thresholds are selected on validation scores only. No test score is used to choose thresholds.", "", dataframe_to_markdown(df)]
    (output_dir / "threshold_score_sanity.md").write_text("\n".join(lines), encoding="utf-8")


def _collect_embeddings(model: CIRFL, ds, cfg: dict) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=0)
    zc, zh, y, cond, atom_weights, source = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device), batch["condition"].to(device), grl_scale=0.0)
            zc.append(out["z_c"].cpu().numpy())
            zh.append(out["z_h"].cpu().numpy())
            atom_weights.append(out["atom_weights"].cpu().numpy())
            source.append(out["source_channel"].cpu().numpy())
            y.append(batch["y"].numpy())
            cond.append(batch["condition"].numpy())
    return {"z_c": np.vstack(zc), "z_h": np.vstack(zh), "y": np.concatenate(y), "condition": np.concatenate(cond), "atom_weights": np.vstack(atom_weights), "source_channel": np.vstack(source)}


def _probe_acc(xtr, ytr, xte, yte) -> float:
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    clf.fit(xtr, ytr)
    return float(accuracy_score(yte, clf.predict(xte)))


def _mechanism_reports(cfg: dict, output_dir: Path, main_proto: dict) -> dict:
    proto_cfg, split_mode = _cfg_for(main_proto, cfg)
    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets_from_config(proto_cfg, ROOT, split_mode=split_mode)
    ckpt = output_dir / f"CIRFL_v2_seed{cfg['seeds'][0]}_{main_proto['protocol']}.pt"
    model = _load_model(proto_cfg, len(feature_cols), ckpt)
    train_emb = _collect_embeddings(model, train_ds, proto_cfg)
    test_emb = _collect_embeddings(model, test_ds, proto_cfg)
    raw_acc = _probe_acc(train_ds.to_flat_features(), train_ds.conditions(), test_ds.to_flat_features(), test_ds.conditions())
    zc_acc = _probe_acc(train_emb["z_c"], train_emb["condition"], test_emb["z_c"], test_emb["condition"])
    zh_acc = _probe_acc(train_emb["z_h"], train_emb["condition"], test_emb["z_h"], test_emb["condition"])
    anom_acc = _probe_acc(train_emb["z_h"], (train_emb["y"] > 0).astype(int), test_emb["z_h"], (test_emb["y"] > 0).astype(int))
    chance = float(max(np.bincount(test_emb["condition"])) / len(test_emb["condition"]))
    probe_df = pd.DataFrame([
        {"metric": "condition_probe_accuracy_raw_features", "value": raw_acc},
        {"metric": "condition_probe_accuracy_zc", "value": zc_acc},
        {"metric": "condition_probe_accuracy_zh", "value": zh_acc},
        {"metric": "anomaly_probe_accuracy_zh", "value": anom_acc},
        {"metric": "condition_chance_level", "value": chance},
    ])
    probe_conclusion = "PASS" if (not math.isnan(zh_acc) and zh_acc <= 0.65 and (math.isnan(raw_acc) or zh_acc < raw_acc)) else "FAIL"
    probe_df.loc[len(probe_df)] = {"metric": "conclusion", "value": probe_conclusion}
    probe_df.to_csv(output_dir / "condition_leakage_probe_v2.csv", index=False)

    atoms = model.relation_atoms().detach().cpu().reshape(model.n_relation_atoms, -1).numpy()
    norm = atoms / np.maximum(np.linalg.norm(atoms, axis=1, keepdims=True), 1e-12)
    cos = norm @ norm.T
    cos_df = pd.DataFrame(cos, columns=[f"atom_{i}" for i in range(cos.shape[0])])
    cos_df.insert(0, "atom", [f"atom_{i}" for i in range(cos.shape[0])])
    cos_df.to_csv(output_dir / "atom_cosine_matrix.csv", index=False)
    weights = test_emb["atom_weights"]
    act = pd.DataFrame({"atom": np.arange(weights.shape[1]), "mean_weight": weights.mean(axis=0), "std_weight": weights.std(axis=0), "max_weight": weights.max(axis=0), "effective_use": weights.mean(axis=0) > 0.01})
    act.to_csv(output_dir / "atom_activation_summary.csv", index=False)
    off = cos[~np.eye(cos.shape[0], dtype=bool)]
    atom_stats = {"mean_offdiag_cosine": float(off.mean()), "max_offdiag_cosine": float(off.max()), "effective_atom_fraction": float(act["effective_use"].mean())}

    source_rows = []
    top_sets = []
    for src_file in sorted(output_dir.glob("CIRFL_v2_source_contributions_seed*road_binary_main.csv")):
        sdf = pd.read_csv(src_file)
        ch_cols = [c for c in sdf.columns if c.startswith("channel_")]
        for label_name, sub in [("normal", sdf[sdf["y_true"] == 0]), ("anomaly", sdf[sdf["y_true"] > 0])]:
            if len(sub):
                top = sub[ch_cols].mean().sort_values(ascending=False).head(5)
                top_sets.append(set(top.index))
                source_rows.append({"seed_file": src_file.name, "label_group": label_name, "top_channels": ";".join(top.index), "top_values": ";".join(f"{v:.4g}" for v in top.values)})
    source_df = pd.DataFrame(source_rows)
    source_df.to_csv(output_dir / "source_localization_v2.csv", index=False)
    jacc = float("nan")
    if len(top_sets) > 1:
        vals = [len(top_sets[i] & top_sets[j]) / max(len(top_sets[i] | top_sets[j]), 1) for i in range(len(top_sets)) for j in range(i + 1, len(top_sets))]
        jacc = float(np.mean(vals))

    proto_text = "Binary anomaly detection is the main task. Multiclass fault diagnosis remains auxiliary because RoAD label 5 has only 2 windows and run/condition-label confounding is high."
    lines = [
        "# Mechanism Diagnosis v2",
        "",
        "## Condition Leakage Probe",
        dataframe_to_markdown(probe_df),
        "",
        "## Atom Diversity",
        f"- mean_offdiag_cosine: {atom_stats['mean_offdiag_cosine']:.6f}",
        f"- max_offdiag_cosine: {atom_stats['max_offdiag_cosine']:.6f}",
        f"- effective_atom_fraction: {atom_stats['effective_atom_fraction']:.6f}",
        "",
        "## Prototype / Fault Diagnosis Feasibility",
        proto_text,
        "",
        "## Source Localization Stability",
        dataframe_to_markdown(source_df),
        f"- top-channel Jaccard stability: {jacc:.6f}",
    ]
    (output_dir / "mechanism_diagnosis_v2.md").write_text("\n".join(lines), encoding="utf-8")
    (output_dir / "atom_diversity_report.md").write_text("# Atom Diversity Report\n\n" + dataframe_to_markdown(pd.DataFrame([atom_stats])) + "\n\n" + dataframe_to_markdown(act), encoding="utf-8")
    (output_dir / "fault_diagnosis_feasibility.md").write_text("# Fault Diagnosis Feasibility\n\n" + proto_text, encoding="utf-8")
    (output_dir / "binary_vs_multiclass_decision.md").write_text("# Binary vs Multiclass Decision\n\nGate v2 uses binary anomaly detection as the primary task. Multiclass prototype diagnosis is auxiliary because RoAD labels and conditions are highly confounded and label 5 is too rare for main-table multiclass evaluation.", encoding="utf-8")
    return {**atom_stats, "condition_probe_accuracy_zh": zh_acc, "condition_probe_conclusion": probe_conclusion, "source_jaccard": jacc}


def _make_figures(output_dir: Path) -> None:
    metrics = output_dir / "gate_v2_metrics.csv"
    bases = output_dir / "gate_v2_baseline_comparison.csv"
    if metrics.exists() and bases.exists():
        df = pd.concat([pd.read_csv(metrics), pd.read_csv(bases)], ignore_index=True)
        valid = df[df["protocol"].isin(["road_binary_main", "scenario_holdout_collision", "scenario_holdout_weight", "scenario_holdout_velocity"])]
        summary = valid.groupby("method")[["macro_f1", "pr_auc", "mdr", "far"]].mean(numeric_only=True).sort_values("pr_auc", ascending=False)
        fig, ax = plt.subplots(figsize=(5.8, 3.0), dpi=180)
        x = np.arange(len(summary))
        ax.bar(x - 0.2, summary["macro_f1"], width=0.2, label="macro-F1", color="#72B7B2")
        ax.bar(x, summary["pr_auc"], width=0.2, label="PR-AUC", color="#F58518")
        ax.bar(x + 0.2, 1 - summary["mdr"], width=0.2, label="1-MDR", color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title("Real Gate v2 valid protocols")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fig_gate_v2_comparison.png", dpi=300)
        plt.close(fig)
    pred = output_dir / "CIRFL_v2_predictions_seed7_road_binary_main.csv"
    if pred.exists():
        p = pd.read_csv(pred)
        fig, ax = plt.subplots(figsize=(3.6, 2.5), dpi=180)
        ax.hist(p[p["y_true"] == 0]["score"], bins=30, alpha=0.75, label="normal", color="#4C78A8")
        ax.hist(p[p["y_true"] > 0]["score"], bins=30, alpha=0.65, label="anomaly", color="#E45756")
        ax.set_xlabel("CIRFL_v2 score")
        ax.set_ylabel("windows")
        ax.set_title("Road binary main score distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fig_score_distribution_v2.png", dpi=300)
        plt.close(fig)


def _write_summary(metrics: pd.DataFrame, baselines: pd.DataFrame, output_dir: Path) -> None:
    combined = pd.concat([metrics, baselines], ignore_index=True)
    summary = summarize_by_method(combined)
    tests = paired_tests(combined, reference_method="CIRFL_v2") if len(combined) else pd.DataFrame()
    lines = ["# Statistical Summary v2", "", "## Mean +/- std", dataframe_to_markdown(summary)]
    if len(tests):
        lines.extend(["", "## Paired tests", dataframe_to_markdown(tests)])
    (output_dir / "statistical_summary_v2.md").write_text("\n".join(lines), encoding="utf-8")


def _write_latency(cfg: dict, output_dir: Path, n_channels: int) -> None:
    ckpt = output_dir / f"CIRFL_v2_seed{cfg['seeds'][0]}_road_binary_main.pt"
    rows = [measure_cirfl_latency(cfg, n_channels, checkpoint_path=ckpt, device_name="cpu")]
    if torch.cuda.is_available():
        rows.append(measure_cirfl_latency(cfg, n_channels, checkpoint_path=ckpt, device_name="cuda"))
    else:
        rows.append({**rows[0], "device": "cuda", "latency_ms_per_window": float("nan"), "cuda_available": False, "gpu_name": "NA"})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "complexity_latency_v2.csv", index=False)
    (output_dir / "complexity_latency_v2.md").write_text("# Complexity and Latency v2\n\n" + dataframe_to_markdown(df), encoding="utf-8")


def _minority_recall_ok(main: pd.DataFrame) -> tuple[bool, float]:
    recalls = []
    for text in main.get("per_class_recall", pd.Series(dtype=str)).fillna(""):
        for part in str(text).split(";"):
            if part.startswith("1:"):
                try:
                    recalls.append(float(part.split(":", 1)[1]))
                except ValueError:
                    pass
    val = float(np.mean(recalls)) if recalls else float("nan")
    return (not math.isnan(val) and val >= 0.5), val


def _decide(metrics: pd.DataFrame, baselines: pd.DataFrame, mechanism: dict, score_sanity: pd.DataFrame) -> tuple[str, list[str]]:
    reasons = []
    main = metrics[metrics["protocol"] == "road_binary_main"]
    base_main = baselines[baselines["protocol"] == "road_binary_main"]
    if main.empty or base_main.empty:
        return "NO-GO", ["Missing road_binary_main metrics or baselines."]
    cm = main[["pr_auc", "auroc", "macro_f1", "weighted_f1", "far", "mdr"]].mean(numeric_only=True)
    main_checks = {
        "macro_f1>=0.35": cm["macro_f1"] >= 0.35,
        "weighted_f1>=0.35": cm["weighted_f1"] >= 0.35,
        "auroc>=0.85": cm["auroc"] >= 0.85,
        "pr_auc>=0.83": cm["pr_auc"] >= 0.83,
        "mdr<=0.12": cm["mdr"] <= 0.12,
        "far<=0.40": cm["far"] <= 0.40,
    }
    for name, ok in main_checks.items():
        if not ok:
            reasons.append(f"Main RoAD criterion failed: {name} (means: {cm.to_dict()}).")
    minority_ok, minority_recall = _minority_recall_ok(main)
    if not minority_ok:
        reasons.append(f"Minority anomaly prototype recall criterion failed: mean recall={minority_recall}.")
    if mechanism.get("condition_probe_accuracy_zh", 1.0) > 0.65:
        reasons.append("Condition leakage criterion failed: z_h condition probe accuracy > 0.65.")
    if mechanism.get("mean_offdiag_cosine", 1.0) > 0.20:
        reasons.append("Relation atom cosine criterion failed: mean off-diagonal cosine > 0.20.")
    if score_sanity["number_of_nan"].sum() > 0 or score_sanity["number_of_inf"].sum() > 0 or score_sanity["score_explosion_flag"].any():
        reasons.append("Numerical stability criterion failed: NaN/Inf or max_score > 1e6.")
    if (score_sanity["score_direction_check"] == "FAIL").any():
        reasons.append("Score direction check failed on at least one valid protocol.")
    return ("GO" if not reasons else "NO-GO"), reasons


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_gate_v2_road_all.yaml")
    args = parser.parse_args()
    base_cfg = load_config(ROOT / args.config)
    output_dir = ROOT / base_cfg["project"].get("output_dir", "outputs/real_gate_v2")
    _clean_output(output_dir)
    logger = setup_logger(output_dir / "real_gate_v2.log")
    save_config(base_cfg, output_dir / "gate_v2_config.yaml")
    prepare_one("road")

    valid_protocols = [p for p in base_cfg.get("protocols", PROTOCOLS)]
    protocol_rows = []
    metric_rows, baseline_frames, score_rows = [], [], []
    errors = []
    generated_at = utc_now()

    for proto in valid_protocols:
        proto_name = proto["protocol"]
        cfg, split_mode = _cfg_for(proto, base_cfg)
        train_ds, val_ds, test_ds, feature_cols, frame = build_datasets_from_config(cfg, ROOT, split_mode=split_mode)
        counts = {"protocol": proto_name, "family": proto["family"], **_count_row(train_ds, "train"), **_count_row(val_ds, "val"), **_count_row(test_ds, "test")}
        counts["can_compute_metrics"] = counts["n_val_normal"] > 0 and counts["n_val_anomaly"] > 0 and counts["n_test_normal"] > 0 and counts["n_test_anomaly"] > 0
        counts["protocol_valid"] = bool(counts["can_compute_metrics"])
        protocol_rows.append(counts)
        if not counts["protocol_valid"]:
            errors.append(f"Protocol {proto_name} invalid and skipped: missing normal/anomaly in validation or test.")
            continue
        logger.info("Running protocol %s", proto_name)
        if proto_name == "road_binary_main":
            audit = audit_dataset(frame, feature_cols, cfg)
            write_audit_reports(audit, output_dir)
        for seed in base_cfg["seeds"]:
            row = train_cirfl_once(cfg, ROOT, output_dir, seed=seed, protocol=proto_name, variant="CIRFL_v2", split_mode_override=split_mode)
            row = _add_counts(row, train_ds, val_ds, test_ds)
            metric_rows.append(row)
        try:
            bdf, berr = run_all_baselines(cfg, ROOT, output_dir, protocol=proto_name, require_real_booster=True, split_mode_override=split_mode)
            bdf["n_train_windows"] = len(train_ds)
            bdf["n_val_windows"] = len(val_ds)
            bdf["n_test_windows"] = len(test_ds)
            bdf["n_test_normal"] = int((test_ds.labels() == 0).sum())
            bdf["n_test_anomaly"] = int((test_ds.labels() > 0).sum())
            baseline_frames.append(bdf)
            errors.extend(berr)
        except Exception as exc:
            errors.append(f"Baseline failed for {proto_name}: {exc}")
        # Score sanity and threshold diagnosis for seed 7.
        ckpt = output_dir / f"CIRFL_v2_seed{base_cfg['seeds'][0]}_{proto_name}.pt"
        model = _load_model(cfg, len(feature_cols), ckpt)
        device = next(model.parameters()).device
        val_pred, _, _ = predict_cirfl(model, val_ds, device, cfg["training"]["batch_size"], val_dataset=None)
        test_pred, _, _ = predict_cirfl(model, test_ds, device, cfg["training"]["batch_size"], val_dataset=None)
        if cfg["training"].get("auto_orient_score", True):
            mult = score_direction_multiplier(val_pred["y_true"].to_numpy(), val_pred["score"].to_numpy())
            val_pred["score"] = val_pred["score"] * mult
            test_pred["score"] = test_pred["score"] * mult
        strategies = ["validation_f1", "validation_youden_j", "target_far_0.05", "target_far_0.10", "target_far_0.15", "target_recall_0.90", "target_recall_0.95", "cost_md5_fp1"]
        thr_df = threshold_strategy_rows(val_pred["y_true"].to_numpy(), val_pred["score"].to_numpy(), test_pred["y_true"].to_numpy(), test_pred["score"].to_numpy(), strategies)
        thr_df["protocol"] = proto_name
        thr_df.to_csv(output_dir / f"threshold_calibration_{proto_name}.csv", index=False)
        sanity = _score_sanity_for_predictions(test_pred)
        sanity.update({"protocol": proto_name, "seed": base_cfg["seeds"][0]})
        score_rows.append(sanity)

    metrics = pd.DataFrame(metric_rows)
    baselines = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    metrics = add_provenance(metrics, DATASET, SOURCE_TYPE, output_dir / "gate_v2_metrics.csv", generated_at) if len(metrics) else metrics
    baselines = add_provenance(baselines, DATASET, SOURCE_TYPE, output_dir / "gate_v2_baseline_comparison.csv", generated_at) if len(baselines) else baselines
    metrics.to_csv(output_dir / "gate_v2_metrics.csv", index=False)
    baselines.to_csv(output_dir / "gate_v2_baseline_comparison.csv", index=False)

    protocol_df = pd.DataFrame(protocol_rows)
    protocol_df.to_csv(output_dir / "protocol_validity_v2.csv", index=False)
    (output_dir / "protocol_validity_report_v2.md").write_text("# Protocol Validity Report v2\n\n" + dataframe_to_markdown(protocol_df), encoding="utf-8")

    score_df = pd.DataFrame(score_rows)
    _threshold_and_score_report(score_rows, output_dir)
    mechanism = _mechanism_reports(base_cfg, output_dir, valid_protocols[0])

    # Ablations on road_binary_main only.
    ablation_rows = []
    main_cfg, main_split = _cfg_for(valid_protocols[0], base_cfg)
    train_ds, val_ds, test_ds, _, _ = build_datasets_from_config(main_cfg, ROOT, split_mode=main_split)
    for name, use_ci, use_atoms, score_mode in [("CIRFL_v2_no_condition_invariance", False, True, "calibrated"), ("CIRFL_v2_no_relation_atoms", True, False, "calibrated"), ("CIRFL_v2_plain_residual_score", True, True, "plain_residual")]:
        try:
            row = train_cirfl_once(main_cfg, ROOT, output_dir, seed=base_cfg["seeds"][0], protocol="road_binary_main", variant=name, use_condition_invariance=use_ci, use_relation_atoms=use_atoms, score_mode=score_mode, split_mode_override=main_split)
            row = _add_counts(row, train_ds, val_ds, test_ds)
            ablation_rows.append(row)
        except Exception as exc:
            errors.append(f"Ablation {name} failed: {exc}")
    ablation = pd.DataFrame(ablation_rows)
    if len(ablation):
        ablation = add_provenance(ablation, DATASET, SOURCE_TYPE, output_dir / "ablation_v2.csv", generated_at)
    ablation.to_csv(output_dir / "ablation_v2.csv", index=False)
    (output_dir / "ablation_v2.md").write_text("# Ablation v2\n\n" + dataframe_to_markdown(ablation), encoding="utf-8")

    _write_summary(metrics, baselines, output_dir)
    _write_latency(base_cfg, output_dir, 86)

    status, reasons = _decide(metrics, baselines, mechanism, score_df)
    lines = ["# Real Gate v2 GO / NO-GO Report", "", f"## Decision: {status}", "", "## Reasons"]
    lines.extend([f"- {r}" for r in reasons] or ["- All gate criteria passed."])
    lines.extend(["", "## Protocol Validity", dataframe_to_markdown(protocol_df), "", "## Synthetic Result Inclusion", "- This packet and gate decision contain no synthetic metric rows."])
    (output_dir / "go_no_go_report_v2.md").write_text("\n".join(lines), encoding="utf-8")

    if errors:
        (output_dir / "errors_and_risks.md").write_text("# Errors and Risks\n\n" + "\n".join(f"- {e}" for e in errors), encoding="utf-8")
    else:
        (output_dir / "errors_and_risks.md").write_text("# Errors and Risks\n\n- No runtime errors recorded.\n- RoAD condition-label and run-label confounding remains a major protocol risk.\n- Full experiments remain blocked unless Gate v2 is GO.\n", encoding="utf-8")

    packet_cfg = deepcopy(base_cfg)
    packet_cfg["project"]["computed_status"] = status
    packet_cfg["project"]["output_dir"] = str(output_dir.relative_to(ROOT))
    make_review_packet(packet_cfg, ROOT)
    logger.info("Real Gate v2 finished: %s", status)


if __name__ == "__main__":
    main()
