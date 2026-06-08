from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_review_packet import make_review_packet
from scripts.prepare_real_data import prepare_one
from src.baselines import run_all_baselines
from src.datasets import audit_dataset, build_datasets_from_config, write_audit_reports
from src.evaluation import measure_cirfl_latency
from src.evaluation.metrics import summarize_by_method, write_summary_markdown
from src.training import train_cirfl_once
from src.utils.config import load_config, save_config
from src.utils.logging import setup_logger
from src.utils.markdown import dataframe_to_markdown

mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 8, "axes.spines.right": False, "axes.spines.top": False, "legend.frameon": False})


def _booster_status() -> dict:
    status = {"xgboost": importlib.util.find_spec("xgboost") is not None, "lightgbm": importlib.util.find_spec("lightgbm") is not None}
    return status


def _write_acquisition(output_dir: Path, acquisition: dict, booster: dict, imadds_status: dict | None = None) -> None:
    lines = ["# Real Data Acquisition", ""]
    lines.extend([
        "## RoAD",
        f"- Status: {acquisition.get('status')}",
        f"- Official source: {acquisition.get('official_source')}",
        f"- Output CSV: {acquisition.get('output_csv')}",
        f"- Rows: {acquisition.get('n_rows', 'NA')}",
        f"- Recordings: {acquisition.get('n_recordings', 'NA')}",
        "- Version: GitLab package commit installed by pip; exact package metadata recorded by pip environment.",
        "",
        "## IMAD-DS Robotic Arm",
    ])
    if imadds_status:
        lines.extend([f"- Status: {imadds_status.get('status')}", f"- Reason: {imadds_status.get('reason', 'NA')}", f"- Output CSV: {imadds_status.get('output_csv', 'NA')}"])
    else:
        lines.append("- Status: NEED_DATA; RoboticArm.7z is not extracted locally.")
    lines.extend([
        "",
        "## NIST UR",
        "- Status: NEED_DATA unless user places files under data/raw/nist_ur/.",
        "",
        "## KUKA Torque",
        "- Status: NEED_DATA unless user places Zenodo CSV files under data/raw/kuka_torque/.",
        "",
        "## Booster Dependencies",
        f"- XGBoost available: {booster['xgboost']}",
        f"- LightGBM available: {booster['lightgbm']}",
    ])
    (output_dir / "real_data_acquisition.md").write_text("\n".join(lines), encoding="utf-8")


def _make_figures(output_dir: Path) -> None:
    pred_path = output_dir / "CIRFL_predictions_seed7_main.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        fig, ax = plt.subplots(figsize=(3.5, 2.4), dpi=180)
        ax.hist(pred[pred["y_true"] == 0]["score"], bins=25, alpha=0.75, color="#4C78A8", label="normal")
        ax.hist(pred[pred["y_true"] > 0]["score"], bins=25, alpha=0.65, color="#E45756", label="fault")
        ax.set_xlabel("validation-calibrated score")
        ax.set_ylabel("windows")
        ax.set_title("RoAD score distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fig_score_distribution_real.png", dpi=300)
        plt.close(fig)
    comp = output_dir / "real_baseline_comparison.csv"
    pilot = output_dir / "real_pilot_metrics.csv"
    if comp.exists() and pilot.exists() and comp.stat().st_size > 5 and pilot.stat().st_size > 5:
        df = pd.concat([pd.read_csv(comp), pd.read_csv(pilot)], ignore_index=True)
        main = df[df["protocol"] == "main"]
        if len(main):
            summary = main.groupby("method")[["macro_f1", "pr_auc"]].mean().sort_values("pr_auc", ascending=False)
            fig, ax = plt.subplots(figsize=(5.4, 2.8), dpi=180)
            x = range(len(summary))
            ax.bar([i - 0.18 for i in x], summary["macro_f1"], width=0.36, color="#72B7B2", label="macro-F1")
            ax.bar([i + 0.18 for i in x], summary["pr_auc"], width=0.36, color="#F58518", label="PR-AUC")
            ax.set_xticks(list(x))
            ax.set_xticklabels(summary.index, rotation=35, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("mean metric")
            ax.set_title("RoAD real gate comparison")
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / "fig_real_gate_comparison.png", dpi=300)
            plt.close(fig)


def _write_latency(config: dict, output_dir: Path, n_channels: int) -> None:
    ckpt = output_dir / f"CIRFL_seed{config['seeds'][0]}_main.pt"
    rows = [measure_cirfl_latency(config, n_channels, checkpoint_path=ckpt, device_name="cpu")]
    import torch
    if torch.cuda.is_available():
        rows.append(measure_cirfl_latency(config, n_channels, checkpoint_path=ckpt, device_name="cuda"))
    else:
        rows.append({"device": "cuda", "parameters": rows[0]["parameters"], "model_size_mb": rows[0]["model_size_mb"], "latency_ms_per_window": float("nan"), "python": rows[0]["python"], "platform": rows[0]["platform"], "torch": rows[0]["torch"], "cuda_available": False, "gpu_name": "NA"})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "latency.csv", index=False)
    lines = ["# Complexity and Latency", "", dataframe_to_markdown(df)]
    (output_dir / "complexity_latency.md").write_text("\n".join(lines), encoding="utf-8")


def _gate_decision(config: dict, audit: dict, metrics: pd.DataFrame, baseline: pd.DataFrame, booster: dict) -> tuple[str, str]:
    if audit["leakage_risk"] == "HIGH":
        return "NO-GO", "High leakage risk."
    if not (booster["xgboost"] or booster["lightgbm"]):
        return "NO-GO", "XGBoost/LightGBM baseline missing."
    if metrics.empty or baseline.empty:
        return "NEED_DATA", "Real metrics are missing."
    main_c = metrics[(metrics["method"] == "CIRFL") & (metrics["protocol"] == "main")]
    main_b = baseline[baseline["protocol"] == "main"]
    cross_c = metrics[(metrics["method"] == "CIRFL") & (metrics["protocol"] == "cross_condition")]
    cross_b = baseline[baseline["protocol"] == "cross_condition"]
    if main_c.empty or cross_c.empty or main_b.empty or cross_b.empty:
        return "NO-GO", "Main or cross-condition real protocol did not complete."
    cm = main_c[["auroc", "pr_auc", "macro_f1", "weighted_f1", "far", "mdr"]].mean()
    bm = main_b.groupby("method")[["auroc", "pr_auc", "macro_f1", "weighted_f1", "far", "mdr"]].mean()
    best_pr = bm["pr_auc"].max()
    best_auc = bm["auroc"].max()
    best_f1 = bm["macro_f1"].max()
    best_wf1 = bm["weighted_f1"].max()
    best_far = bm["far"].min()
    wins = int(cm["auroc"] >= best_auc + 0.03) + int(cm["pr_auc"] >= best_pr + 0.03) + int(cm["macro_f1"] >= best_f1 + 0.03) + int(cm["far"] <= best_far * 0.85)
    main_ok = ((cm["auroc"] >= 0.93 or cm["auroc"] >= best_auc + 0.03) and (cm["pr_auc"] >= 0.90 or cm["pr_auc"] >= best_pr + 0.03) and (cm["macro_f1"] >= 0.80 or cm["macro_f1"] >= best_f1 + 0.03) and (cm["weighted_f1"] >= 0.85 or cm["weighted_f1"] >= best_wf1 + 0.03) and (cm["far"] <= 0.15 or cm["far"] <= best_far * 0.85) and wins >= 3)
    if not main_ok:
        return "NO-GO", "CIRFL did not satisfy Real Gate v1 main-metric thresholds against strongest baseline."
    return "GO", "Real Gate v1 main metrics passed; verify ablation and cross-condition details before full experiment preparation."


def _write_go_no_go(output_dir: Path, status: str, reason: str, audit: dict, booster: dict) -> None:
    lines = [
        "# Real Gate v1 GO / NO-GO Report",
        "",
        f"## Decision: {status}",
        f"- Reason: {reason}",
        "",
        "## Criteria Check",
        f"- Real robot dataset ran: {'YES' if status != 'NEED_DATA' else 'NO'}",
        "- Real cross-condition protocol attempted: YES for RoAD condition holdout; IMAD-DS remains NEED_DATA unless files are provided.",
        f"- Leakage risk: {audit['leakage_risk']}",
        f"- XGBoost available: {booster['xgboost']}",
        f"- LightGBM available: {booster['lightgbm']}",
        "- Synthetic results are excluded from GO evidence.",
        "- Full experiments and manuscript writing remain blocked unless status is GO.",
    ]
    (output_dir / "go_no_go_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_gate_road.yaml")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "real_gate.log")
    save_config(config, output_dir / "real_gate_config_used.yaml")

    logger.info("Preparing RoAD data")
    acquisition = prepare_one("road")
    imadds_status = prepare_one("imadds_robotic_arm")
    booster = _booster_status()
    _write_acquisition(output_dir, acquisition, booster, imadds_status)
    if acquisition.get("status") != "READY":
        _write_go_no_go(output_dir, "NEED_DATA", "RoAD data is not available.", {"leakage_risk": "UNKNOWN"}, booster)
        make_review_packet(config, ROOT)
        return

    logger.info("Building datasets and audit")
    train_ds, val_ds, test_ds, feature_cols, frame = build_datasets_from_config(config, ROOT, split_mode="main")
    audit = audit_dataset(frame, feature_cols, config)
    write_audit_reports(audit, output_dir)

    errors = []
    pilot_path = output_dir / "real_pilot_metrics.csv"
    if pilot_path.exists() and pilot_path.stat().st_size > 5:
        logger.info("Reusing existing CIRFL real pilot metrics")
        cirfl_df = pd.read_csv(pilot_path)
    else:
        cirfl_rows = []
        logger.info("Training CIRFL on RoAD main and cross-condition protocols")
        for protocol in ["main", "cross_condition"]:
            for seed in config["seeds"]:
                cirfl_rows.append(train_cirfl_once(config, ROOT, output_dir, seed=seed, protocol=protocol, variant="CIRFL"))
        cirfl_df = pd.DataFrame(cirfl_rows)
        cirfl_df.to_csv(pilot_path, index=False)

    logger.info("Running real baselines with true XGBoost/LightGBM requirement")
    baseline_frames = []
    for protocol in ["main", "cross_condition"]:
        try:
            df, err = run_all_baselines(config, ROOT, output_dir, protocol=protocol, require_real_booster=True)
            baseline_frames.append(df)
            errors.extend(err)
        except Exception as exc:
            errors.append(f"baseline protocol {protocol} failed: {exc}")
    baseline_df = pd.concat(baseline_frames, ignore_index=True) if baseline_frames else pd.DataFrame()
    if len(baseline_df):
        baseline_df.to_csv(output_dir / "real_baseline_comparison.csv", index=False)
    else:
        (output_dir / "real_baseline_comparison.csv").write_text("method,seed,protocol,macro_f1,weighted_f1,auroc,pr_auc,far,mdr,far_at_95_recall\n", encoding="utf-8")

    ablation_path = output_dir / "ablation_real_gate.csv"
    if ablation_path.exists() and ablation_path.stat().st_size > 5:
        logger.info("Reusing existing real ablation metrics")
        ablation_df = pd.read_csv(ablation_path)
    else:
        logger.info("Running real ablation preview on RoAD cross-condition")
        ablation_rows = []
        for name, use_ci, use_atoms, score_mode in [
            ("CIRFL_no_condition_invariance", False, True, "calibrated"),
            ("CIRFL_no_relation_atoms", True, False, "calibrated"),
            ("CIRFL_plain_residual_score", True, True, "plain_residual"),
        ]:
            try:
                ablation_rows.append(train_cirfl_once(config, ROOT, output_dir, seed=config["seeds"][0], protocol="cross_condition", variant=name, use_condition_invariance=use_ci, use_relation_atoms=use_atoms, score_mode=score_mode))
            except Exception as exc:
                errors.append(f"ablation {name} failed: {exc}")
        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(ablation_path, index=False)
    (output_dir / "ablation_real_gate.md").write_text("# Real-Gate Ablation\n\n" + dataframe_to_markdown(ablation_df), encoding="utf-8")

    logger.info("Writing statistics, latency, figures")
    if len(cirfl_df) and len(baseline_df):
        write_summary_markdown(pd.concat([cirfl_df, baseline_df], ignore_index=True), output_dir / "statistical_summary.md")
    _write_latency(config, output_dir, len(feature_cols))
    try:
        _make_figures(output_dir)
    except Exception as exc:
        errors.append(f"figure generation failed: {exc}")
    if errors:
        (output_dir / "errors_and_risks.md").write_text("# Errors and Risks\n\n" + "\n".join(f"- {e}" for e in errors), encoding="utf-8")
    else:
        (output_dir / "errors_and_risks.md").write_text("# Errors and Risks\n\n- No runtime errors recorded.\n- IMAD-DS/NIST/KUKA remain NEED_DATA unless raw files are supplied.\n", encoding="utf-8")

    status, reason = _gate_decision(config, audit, cirfl_df, baseline_df, booster)
    _write_go_no_go(output_dir, status, reason, audit, booster)
    config["project"]["real_data_available"] = status == "GO"
    config["project"]["computed_status"] = status
    make_review_packet(config, ROOT)
    logger.info("Real gate finished with status: %s (%s)", status, reason)


if __name__ == "__main__":
    main()
