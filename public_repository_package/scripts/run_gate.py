from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_review_packet import make_review_packet
from src.baselines import run_all_baselines
from src.datasets import audit_dataset, build_datasets_from_config, write_audit_reports
from src.evaluation import measure_cirfl_latency
from src.evaluation.metrics import binary_metric_row, write_summary_markdown
from src.models import CIRFL
from src.training import predict_cirfl, tiny_overfit_check, train_cirfl_once
from src.utils.config import load_config, save_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.logging import setup_logger
from src.utils.torch_utils import resolve_device


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def _load_cirfl_for_seed(config: dict, feature_cols: list[str], checkpoint: Path) -> CIRFL:
    model_cfg = config["model"]
    device = resolve_device(config.get("device", "auto"))
    model = CIRFL(
        n_channels=len(feature_cols),
        window_size=config["data"]["window_size"],
        num_classes=model_cfg["num_classes"],
        hidden_dim=model_cfg["hidden_dim"],
        condition_dim=model_cfg["condition_dim"],
        residual_dim=model_cfg["residual_dim"],
        n_relation_atoms=model_cfg["n_relation_atoms"],
        max_conditions=model_cfg["max_conditions"],
        dropout=model_cfg["dropout"],
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"])
    return model


def _write_ablation_report(df: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Ablation Preview", "", "Synthetic sanity only; not real gate evidence.", "", dataframe_to_markdown(df)]
    (output_dir / "ablation_preview.md").write_text("\n".join(lines), encoding="utf-8")


def _write_robustness_report(df: pd.DataFrame, output_dir: Path) -> None:
    lines = ["# Robustness Preview", "", "Synthetic sanity only; not real gate evidence.", "", dataframe_to_markdown(df)]
    (output_dir / "robustness_preview.md").write_text("\n".join(lines), encoding="utf-8")


def _write_latency_report(row: dict, output_dir: Path) -> None:
    lines = [
        "# Complexity and Latency",
        "",
        f"- Parameters: {row['parameters']}",
        f"- Model size MB: {row['model_size_mb']:.4f}",
        f"- Device: {row['device']}",
        f"- Latency ms/window: {row['latency_ms_per_window']:.4f}",
        f"- CUDA available: {row['cuda_available']}",
        f"- GPU: {row['gpu_name']}",
        f"- Platform: {row['platform']}",
        f"- Torch: {row['torch']}",
    ]
    (output_dir / "complexity_latency.md").write_text("\n".join(lines), encoding="utf-8")


def _write_errors(errors: list[str], output_dir: Path) -> None:
    lines = ["# Errors and Risks", ""]
    if errors:
        lines.extend([f"- {e}" for e in errors])
    else:
        lines.append("- No runtime errors recorded during synthetic gate sanity.")
    lines.extend(
        [
            "- Real robot/manipulator data is not present; formal gate must remain NEED_DATA / NO-GO.",
            "- Synthetic sanity metrics are not paper evidence.",
        ]
    )
    (output_dir / "errors_and_risks.md").write_text("\n".join(lines), encoding="utf-8")


def _make_figures(output_dir: Path) -> None:
    pred_path = output_dir / "CIRFL_predictions_seed7_main.csv"
    if pred_path.exists():
        pred = pd.read_csv(pred_path)
        fig, ax = plt.subplots(figsize=(3.4, 2.4), dpi=180)
        normal = pred[pred["y_true"] == 0]["score"]
        abnormal = pred[pred["y_true"] > 0]["score"]
        ax.hist(normal, bins=18, alpha=0.75, label="normal", color="#4C78A8")
        ax.hist(abnormal, bins=18, alpha=0.65, label="fault", color="#E45756")
        ax.set_xlabel("calibrated residual score")
        ax.set_ylabel("windows")
        ax.set_title("Synthetic score distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fig_pilot_score_distribution.png", dpi=300)
        plt.close(fig)

    comp_path = output_dir / "baseline_comparison.csv"
    pilot_path = output_dir / "pilot_metrics.csv"
    if comp_path.exists() and pilot_path.exists():
        base = pd.read_csv(comp_path)
        cirfl = pd.read_csv(pilot_path)
        merged = pd.concat([base, cirfl], ignore_index=True)
        main = merged[merged["protocol"] == "main"]
        summary = main.groupby("method")[["macro_f1", "pr_auc"]].mean().sort_values("pr_auc", ascending=False)
        fig, ax = plt.subplots(figsize=(5.4, 2.8), dpi=180)
        x = range(len(summary))
        ax.bar([i - 0.18 for i in x], summary["macro_f1"], width=0.36, color="#72B7B2", label="macro-F1")
        ax.bar([i + 0.18 for i in x], summary["pr_auc"], width=0.36, color="#F58518", label="PR-AUC")
        ax.set_xticks(list(x))
        ax.set_xticklabels(summary.index, rotation=35, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("mean metric")
        ax.set_title("Synthetic gate comparison")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "fig_gate_comparison.png", dpi=300)
        plt.close(fig)


def _run_robustness(config: dict, output_dir: Path, feature_cols: list[str]) -> pd.DataFrame:
    seed = config["seeds"][0]
    device = resolve_device(config.get("device", "auto"))
    train_ds, val_ds, test_ds, _, _ = build_datasets_from_config(config, ROOT, split_mode="main")
    model = _load_cirfl_for_seed(config, feature_cols, output_dir / f"CIRFL_seed{seed}_main.pt")
    rows = []
    for name, noise, missing in [
        ("clean", 0.0, 0.0),
        ("gaussian_noise", config["robustness"]["gaussian_noise_std"], 0.0),
        ("missing_channel", 0.0, config["robustness"]["missing_channel_prob"]),
    ]:
        pred, threshold, _ = predict_cirfl(
            model,
            test_ds,
            device,
            batch_size=config["training"]["batch_size"],
            val_dataset=val_ds,
            noise_std=noise,
            missing_channel_prob=missing,
        )
        row = binary_metric_row(pred["y_true"].to_numpy(), pred["score"].to_numpy(), threshold, "CIRFL", seed, name, pred["class_pred"].to_numpy())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "robustness_preview.csv", index=False)
    _write_robustness_report(df, output_dir)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_config.yaml")
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    output_dir = ROOT / config["project"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "gate.log")
    save_config(config, output_dir / "gate_config_used.yaml")

    logger.info("Building datasets and running audit")
    train_ds, val_ds, test_ds, feature_cols, frame = build_datasets_from_config(config, ROOT, split_mode="main")
    audit = audit_dataset(frame, feature_cols, config)
    write_audit_reports(audit, output_dir)

    logger.info("Running tiny overfit check")
    tiny_overfit_check(config, ROOT, output_dir, config["seeds"][0])

    logger.info("Training CIRFL seeds for main and cross-condition protocols")
    cirfl_rows = []
    for protocol in ["main", "cross_condition"]:
        for seed in config["seeds"]:
            cirfl_rows.append(train_cirfl_once(config, ROOT, output_dir, seed=seed, protocol=protocol, variant="CIRFL"))
    cirfl_df = pd.DataFrame(cirfl_rows)
    cirfl_df.to_csv(output_dir / "pilot_metrics.csv", index=False)

    logger.info("Running gate baselines")
    baseline_frames = []
    errors = []
    for protocol in ["main", "cross_condition"]:
        df, err = run_all_baselines(config, ROOT, output_dir, protocol=protocol)
        baseline_frames.append(df)
        errors.extend(err)
    baseline_df = pd.concat(baseline_frames, ignore_index=True)
    baseline_df.to_csv(output_dir / "baseline_comparison.csv", index=False)

    logger.info("Running ablation preview on first seed")
    seed = config["seeds"][0]
    ablation_rows = []
    ablation_specs = [
        ("CIRFL_no_condition_invariance", False, True, "calibrated"),
        ("CIRFL_no_relation_atoms", True, False, "calibrated"),
        ("CIRFL_plain_residual_score", True, True, "plain_residual"),
    ]
    for name, use_ci, use_atoms, score_mode in ablation_specs:
        ablation_rows.append(
            train_cirfl_once(
                config,
                ROOT,
                output_dir,
                seed=seed,
                protocol="cross_condition",
                variant=name,
                use_condition_invariance=use_ci,
                use_relation_atoms=use_atoms,
                score_mode=score_mode,
            )
        )
    ablation_df = pd.DataFrame(ablation_rows)
    ablation_df.to_csv(output_dir / "ablation_preview.csv", index=False)
    _write_ablation_report(ablation_df, output_dir)

    logger.info("Running robustness preview")
    _run_robustness(config, output_dir, feature_cols)

    logger.info("Measuring latency")
    latency = measure_cirfl_latency(config, len(feature_cols), checkpoint_path=output_dir / f"CIRFL_seed{config['seeds'][0]}_main.pt")
    pd.DataFrame([latency]).to_csv(output_dir / "latency.csv", index=False)
    _write_latency_report(latency, output_dir)

    logger.info("Writing statistical summary and figures")
    combined_metrics = pd.concat([cirfl_df, baseline_df], ignore_index=True)
    write_summary_markdown(combined_metrics, output_dir / "statistical_summary.md")
    _make_figures(output_dir)
    _write_errors(errors, output_dir)

    logger.info("Creating review packet")
    review_dir = make_review_packet(config, ROOT)
    n_files = len([p for p in review_dir.iterdir() if p.is_file()])
    logger.info("Review packet ready: %s (%s files)", review_dir, n_files)
    if not config["project"].get("real_data_available"):
        logger.info("Final status: NEED_DATA / NO-GO. Real robot pilot is required before full experiments.")


if __name__ == "__main__":
    main()
