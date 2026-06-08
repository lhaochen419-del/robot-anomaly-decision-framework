from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_real_data import prepare_one
from src.datasets.timeseries import WindowedTimeSeriesDataset, infer_feature_cols, load_timeseries_frame
from src.utils.markdown import dataframe_to_markdown

ROAD_CSV = ROOT / "data/processed/road/road_unified.csv"
OUT = ROOT / "outputs/road_feasibility"


def _window_df(df: pd.DataFrame, window_size: int, stride: int) -> pd.DataFrame:
    data_cfg = {
        "label_col": "fault_label",
        "condition_col": "condition_id",
        "run_col": "run_id",
        "trajectory_col": "trajectory_id",
        "fault_episode_col": "fault_episode_id",
        "time_col": "t",
        "feature_cols": [],
    }
    feats = infer_feature_cols(df, data_cfg)
    ds = WindowedTimeSeriesDataset(df, feats, "fault_label", "condition_id", "run_id", "trajectory_id", "fault_episode_id", window_size, stride, label_mode="multiclass")
    return pd.DataFrame([r.__dict__ for r in ds.records])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not ROAD_CSV.exists():
        prepare_one("road")
    df = load_timeseries_frame(ROAD_CSV)
    wdf = _window_df(df, args.window_size, args.stride)
    row_label = df.groupby("fault_label").size().reset_index(name="row_count")
    win_label = wdf.groupby("label").size().reset_index(name="window_count").rename(columns={"label": "fault_label"})
    label_feas = row_label.merge(win_label, on="fault_label", how="outer").fillna(0)
    label_feas["binary_allowed"] = label_feas["fault_label"].gt(0)
    label_feas["multiclass_main_allowed"] = label_feas["window_count"] >= 100
    label_feas["recommendation"] = label_feas.apply(lambda r: "normal" if r["fault_label"] == 0 else ("auxiliary_multiclass_ok" if r["multiclass_main_allowed"] else "rare_exclude_or_merge"), axis=1)
    label_feas.to_csv(OUT / "road_label_condition_feasibility.csv", index=False)

    run_mat = pd.crosstab(df["run_id"], df["fault_label"])
    run_bin = pd.crosstab(df["run_id"], df["fault_label"].gt(0).map({False: "normal_rows", True: "anomaly_rows"}))
    run_report = run_mat.join(run_bin, how="outer").fillna(0).reset_index()
    run_report.to_csv(OUT / "road_run_label_matrix.csv", index=False)

    cond_mat = pd.crosstab(df["condition_id"], df["fault_label"])
    cond_bin = pd.crosstab(df["condition_id"], df["fault_label"].gt(0).map({False: "normal_rows", True: "anomaly_rows"}))
    cond_report = cond_mat.join(cond_bin, how="outer").fillna(0).reset_index()
    cond_report.to_csv(OUT / "road_condition_label_matrix.csv", index=False)

    episode_sizes = df[df["fault_episode_id"] >= 0].groupby(["fault_episode_id", "fault_label", "run_id", "condition_id"]).size().reset_index(name="row_count")
    episode_sizes.to_csv(OUT / "road_fault_episode_sizes.csv", index=False)

    condition_confounded = []
    for cid, group in df.groupby("condition_id"):
        labels = set(group["fault_label"].unique())
        has_normal = 0 in labels
        has_anom = any(x > 0 for x in labels)
        risk = "HIGH" if not (has_normal and has_anom) else "MEDIUM"
        condition_confounded.append({"condition_id": cid, "labels": sorted(labels), "has_normal": has_normal, "has_anomaly": has_anom, "confounding_risk": risk})
    cond_conf = pd.DataFrame(condition_confounded)

    run_confounded = []
    for rid, group in df.groupby("run_id"):
        labels = set(group["fault_label"].unique())
        has_normal = 0 in labels
        has_anom = any(x > 0 for x in labels)
        risk = "HIGH" if not (has_normal and has_anom) else "MEDIUM"
        run_confounded.append({"run_id": rid, "condition_id": int(group["condition_id"].iloc[0]), "labels": sorted(labels), "has_normal": has_normal, "has_anomaly": has_anom, "confounding_risk": risk})
    run_conf = pd.DataFrame(run_confounded)

    lines = [
        "# RoAD Label / Run / Condition Feasibility Report",
        "",
        "## Label Feasibility",
        dataframe_to_markdown(label_feas),
        "",
        "## Condition Confounding",
        dataframe_to_markdown(cond_conf),
        "",
        "## Run Confounding",
        dataframe_to_markdown(run_conf),
        "",
        "## Decisions",
        "- Main task for Gate v2 is binary anomaly detection.",
        "- Multiclass fault diagnosis is auxiliary only; labels with <100 windows cannot be main multiclass categories.",
        "- Conditions 0, 3, and 4 are label-confounded because they contain only normal or only anomaly rows; pure condition-holdout must be treated as high-risk unless test has both normal and anomaly.",
        "- Scenario holdout is preferred over pure condition holdout for RoAD v2.",
    ]
    (OUT / "label_feasibility_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote RoAD feasibility reports to {OUT}")


if __name__ == "__main__":
    main()
