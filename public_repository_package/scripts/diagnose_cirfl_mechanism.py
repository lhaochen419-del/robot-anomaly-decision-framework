from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.datasets import build_datasets_from_config
from src.evaluation.metrics import threshold_strategy_rows
from src.models import CIRFL
from src.training import predict_cirfl
from src.utils.config import load_config
from src.utils.markdown import dataframe_to_markdown
from src.utils.torch_utils import resolve_device


def _model_from_checkpoint(config: dict, n_channels: int, checkpoint: Path) -> CIRFL:
    model_cfg = config["model"]
    device = resolve_device(config.get("device", "auto"))
    payload = torch.load(checkpoint, map_location=device)
    num_classes = payload.get("metadata", {}).get("num_classes", model_cfg["num_classes"])
    model = CIRFL(
        n_channels=n_channels,
        window_size=config["data"]["window_size"],
        num_classes=num_classes,
        hidden_dim=model_cfg["hidden_dim"],
        condition_dim=model_cfg["condition_dim"],
        residual_dim=model_cfg["residual_dim"],
        n_relation_atoms=model_cfg["n_relation_atoms"],
        max_conditions=model_cfg["max_conditions"],
        dropout=model_cfg["dropout"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _collect_embeddings(model: CIRFL, ds, config: dict) -> dict[str, np.ndarray]:
    device = next(model.parameters()).device
    loader = DataLoader(ds, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)
    zc, zh, y, cond, atom_weights, logits = [], [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device), batch["condition"].to(device), grl_scale=0.0)
            zc.append(out["z_c"].cpu().numpy())
            zh.append(out["z_h"].cpu().numpy())
            atom_weights.append(out["atom_weights"].cpu().numpy())
            logits.append(out["class_logits"].cpu().numpy())
            y.append(batch["y"].numpy())
            cond.append(batch["condition"].numpy())
    return {"z_c": np.vstack(zc), "z_h": np.vstack(zh), "y": np.concatenate(y), "condition": np.concatenate(cond), "atom_weights": np.vstack(atom_weights), "class_logits": np.vstack(logits)}


def _probe_report(train: dict, test: dict) -> tuple[pd.DataFrame, str]:
    rows = []
    chance = max(np.bincount(test["condition"])) / len(test["condition"])
    specs = [
        ("condition_probe_accuracy_zc", train["z_c"], train["condition"], test["z_c"], test["condition"]),
        ("condition_probe_accuracy_zh", train["z_h"], train["condition"], test["z_h"], test["condition"]),
        ("anomaly_probe_accuracy_zh", train["z_h"], (train["y"] > 0).astype(int), test["z_h"], (test["y"] > 0).astype(int)),
    ]
    vals = {}
    for name, xtr, ytr, xte, yte in specs:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(xtr, ytr)
        acc = accuracy_score(yte, clf.predict(xte))
        vals[name] = acc
        rows.append({"probe": name, "accuracy": float(acc)})
    rows.append({"probe": "condition_chance_level", "accuracy": float(chance)})
    conclusion = "PASS" if vals["condition_probe_accuracy_zc"] > chance and vals["condition_probe_accuracy_zh"] <= max(chance + 0.10, 0.55) and vals["anomaly_probe_accuracy_zh"] > 0.60 else "FAIL"
    rows.append({"probe": "conclusion", "accuracy": conclusion})
    return pd.DataFrame(rows), conclusion


def _tree_diagnosis(train_ds, test_ds, feature_cols: list[str], output_dir: Path) -> pd.DataFrame:
    x_train = train_ds.to_flat_features()
    y_train = train_ds.labels()
    x_test = test_ds.to_flat_features()
    y_test = test_ds.labels()
    rf = RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=7, n_jobs=1)
    rf.fit(x_train, y_train)
    acc = accuracy_score(y_test, rf.predict(x_test))
    importances = rf.feature_importances_
    names = []
    stats = ["mean", "std", "min", "max", "slope"]
    n_ch = len(feature_cols)
    for stat in stats:
        names.extend([f"{stat}:{c}" for c in feature_cols])
    for t in range(train_ds.window_size):
        names.extend([f"raw_t{t}:{c}" for c in feature_cols])
    if len(names) < len(importances):
        names.extend([f"feature_{i}" for i in range(len(names), len(importances))])
    df = pd.DataFrame({"feature": names[: len(importances)], "importance": importances})
    df["family"] = df["feature"].str.split(":").str[0].str.replace(r"raw_t.*", "raw", regex=True)
    df = df.sort_values("importance", ascending=False)
    df.to_csv(output_dir / "tree_feature_importance.csv", index=False)
    condition_label = pd.crosstab(train_ds.conditions(), (train_ds.labels() > 0).astype(int), normalize="index")
    lines = [
        "# Synthetic Dataset Diagnosis",
        "",
        f"- RandomForest test accuracy on flattened statistical/raw features: {acc:.4f}",
        "- Top feature families indicate whether simple point statistics dominate the synthetic task.",
        "",
        "## Top 20 Features",
        dataframe_to_markdown(df.head(20)),
        "",
        "## Train condition vs anomaly-label distribution",
        dataframe_to_markdown(condition_label.reset_index()),
        "",
        "## Conclusion",
        "If top features are mostly single-channel mean/std/max/slope and condition-label association is strong, the synthetic generator is not a good proof of residual-field mechanisms. It remains a code sanity dataset only.",
    ]
    (output_dir / "synthetic_dataset_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def _prototype_report(test: dict, output_dir: Path) -> str:
    y = test["y"]
    pred = test["class_logits"].argmax(axis=1)
    labels = sorted(np.unique(y).tolist())
    cm = confusion_matrix(y, pred, labels=labels)
    recalls = recall_score(y, pred, labels=labels, average=None, zero_division=0)
    dist_proxy = -test["class_logits"]
    rows = []
    for cls in labels:
        mask = y == cls
        if mask.any():
            rows.append({"class": cls, "n": int(mask.sum()), "recall": float(recalls[labels.index(cls)]), "mean_own_distance_proxy": float(dist_proxy[mask, cls].mean()) if cls < dist_proxy.shape[1] else float("nan"), "min_distance_proxy_mean": float(dist_proxy[mask].min(axis=1).mean())})
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "prototype_distance_statistics.csv", index=False)
    cm_df = pd.DataFrame(cm, index=[f"true_{x}" for x in labels], columns=[f"pred_{x}" for x in labels])
    cm_df.to_csv(output_dir / "fault_confusion_matrix.csv")
    zero = df[df["recall"] == 0]["class"].tolist()
    lines = ["## Prototype Diagnosis", "", "### Confusion matrix", dataframe_to_markdown(cm_df.reset_index()), "", "### Per-class distance/recall", dataframe_to_markdown(df), "", f"Classes with zero recall: {zero}", "Class-balanced prototype loss is recommended if minority classes remain at zero recall on real data."]
    return "\n".join(lines)


def _relation_report(model: CIRFL, emb: dict, output_dir: Path, ablation_csv: Path) -> str:
    weights = emb["atom_weights"]
    activation = pd.DataFrame({"atom": np.arange(weights.shape[1]), "mean_weight": weights.mean(axis=0), "std_weight": weights.std(axis=0), "max_weight": weights.max(axis=0)})
    activation.to_csv(output_dir / "relation_atom_activation_summary.csv", index=False)
    atoms = model.relation_atoms().detach().cpu().reshape(model.n_relation_atoms, -1).numpy()
    norm = atoms / np.maximum(np.linalg.norm(atoms, axis=1, keepdims=True), 1e-12)
    sim = norm @ norm.T
    diversity = {"mean_offdiag_cosine": float((sim.sum() - np.trace(sim)) / max(sim.size - len(sim), 1)), "max_offdiag_cosine": float((sim - np.eye(len(sim))).max())}
    ablation_text = "No ablation CSV found."
    if ablation_csv.exists():
        ablation_text = dataframe_to_markdown(pd.read_csv(ablation_csv))
    return "\n".join(["## Relation Atom Utility", "", "### Activation summary", dataframe_to_markdown(activation), "", f"Atom diversity: {diversity}", "", "### Ablation", ablation_text])


def _source_report(output_dir: Path, gate_dir: Path) -> str:
    files = sorted(gate_dir.glob("CIRFL_source_contributions_seed*_main.csv"))
    if not files:
        return "## Source Localization Sanity\n\nNo source contribution files found."
    rows = []
    top_sets = []
    for path in files:
        df = pd.read_csv(path)
        ch_cols = [c for c in df.columns if c.startswith("channel_")]
        for label_name, subset in [("normal", df[df["y_true"] == 0]), ("anomaly", df[df["y_true"] > 0])]:
            if len(subset):
                means = subset[ch_cols].mean().sort_values(ascending=False)
                top = means.head(5)
                top_sets.append(set(top.index))
                rows.append({"seed_file": path.name, "label_group": label_name, "top_channels": ";".join(top.index), "top_values": ";".join(f"{v:.4f}" for v in top.values)})
    stability = float("nan")
    if len(top_sets) > 1:
        vals = []
        for i in range(len(top_sets)):
            for j in range(i + 1, len(top_sets)):
                vals.append(len(top_sets[i] & top_sets[j]) / max(len(top_sets[i] | top_sets[j]), 1))
        stability = float(np.mean(vals))
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "source_localization_examples.csv", index=False)
    return "\n".join(["## Source Localization Sanity", "", dataframe_to_markdown(out), "", f"Top-channel Jaccard stability across seed/group summaries: {stability:.4f}"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gate_config.yaml")
    parser.add_argument("--gate-output", default="outputs/gate_sanity")
    parser.add_argument("--output-dir", default="outputs/mechanism_diagnosis")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    output_dir = ROOT / args.output_dir
    gate_dir = ROOT / args.gate_output
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, feature_cols, _ = build_datasets_from_config(config, ROOT, split_mode="main")
    ckpt = gate_dir / f"CIRFL_seed{config['seeds'][0]}_main.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing synthetic checkpoint: {ckpt}")
    model = _model_from_checkpoint(config, len(feature_cols), ckpt)

    val_pred, _, _ = predict_cirfl(model, val_ds, next(model.parameters()).device, config["training"]["batch_size"], val_dataset=None)
    test_pred, _, _ = predict_cirfl(model, test_ds, next(model.parameters()).device, config["training"]["batch_size"], val_dataset=None)
    strategies = ["validation_f1", "validation_youden_j", "target_far_0.05", "target_far_0.10", "target_far_0.15", "target_recall_0.90", "target_recall_0.95", "cost_md5_fp1"]
    threshold_df = threshold_strategy_rows(val_pred["y_true"].to_numpy(), val_pred["score"].to_numpy(), test_pred["y_true"].to_numpy(), test_pred["score"].to_numpy(), strategies)
    threshold_df.to_csv(output_dir / "threshold_calibration.csv", index=False)
    (output_dir / "threshold_calibration.md").write_text("# Threshold Calibration Diagnosis\n\nAll thresholds are selected on validation scores only and evaluated on test scores.\n\n" + dataframe_to_markdown(threshold_df), encoding="utf-8")

    train_emb = _collect_embeddings(model, train_ds, config)
    test_emb = _collect_embeddings(model, test_ds, config)
    probe_df, probe_conclusion = _probe_report(train_emb, test_emb)
    probe_df.to_csv(output_dir / "condition_leakage_probe.csv", index=False)
    _tree_diagnosis(train_ds, test_ds, feature_cols, output_dir)
    proto_text = _prototype_report(test_emb, output_dir)
    rel_text = _relation_report(model, test_emb, output_dir, gate_dir / "ablation_preview.csv")
    source_text = _source_report(output_dir, gate_dir)

    lines = [
        "# CIRFL Mechanism Diagnosis",
        "",
        "## Condition Leakage Probe",
        dataframe_to_markdown(probe_df),
        f"Conclusion: {probe_conclusion}",
        "",
        "## Tree Baseline Dominance Diagnosis",
        "See `tree_feature_importance.csv` and `synthetic_dataset_diagnosis.md`. The synthetic task is not used as paper evidence.",
        "",
        proto_text,
        "",
        rel_text,
        "",
        source_text,
    ]
    (output_dir / "mechanism_diagnosis.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote mechanism diagnosis to {output_dir}")


if __name__ == "__main__":
    main()
