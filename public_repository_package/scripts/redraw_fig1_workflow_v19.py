from __future__ import annotations

from pathlib import Path
import textwrap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "rie_latex_template_draft_v19_abstract_fig_layout" / "figures"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "font.size": 8.5,
        "axes.linewidth": 0.8,
    }
)


COLORS = {
    "data": "#E8F1FA",
    "split": "#F5F5F5",
    "train": "#EAF4EA",
    "val": "#FFF2D8",
    "test": "#FDECEC",
    "deploy": "#EFEAF7",
    "edge": "#4A5568",
    "arrow": "#3B4351",
}


def wrapped(text: str, width: int = 24) -> str:
    lines: list[str] = []
    for part in text.split("\n"):
        lines.extend(textwrap.wrap(part, width=width) if part else [""])
    return "\n".join(lines)


def box(ax, xy, wh, text, fc, title=None, wrap=25):
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.035",
        linewidth=1.15,
        edgecolor=COLORS["edge"],
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(patch)
    if title:
        ax.text(
            x + w / 2,
            y + h - 0.055,
            title,
            ha="center",
            va="top",
            fontsize=9.2,
            fontweight="bold",
            color="#1F2937",
            zorder=3,
        )
        body_y = y + h / 2 - 0.02
    else:
        body_y = y + h / 2
    ax.text(
        x + w / 2,
        body_y,
        wrapped(text, wrap),
        ha="center",
        va="center",
        fontsize=8.2,
        color="#253040",
        linespacing=1.18,
        zorder=3,
    )
    return patch


def arrow(ax, start, end, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.1,
            color=COLORS["arrow"],
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
            shrinkA=4,
            shrinkB=4,
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(13.2, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        "Leakage-safe and calibration-aware model-threshold decision workflow",
        ha="center",
        va="top",
        fontsize=13.2,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.5,
        0.925,
        "Selection uses validation/calibration data only; held-out test data are used once for final reporting.",
        ha="center",
        va="top",
        fontsize=9.4,
        color="#4B5563",
    )

    # Source and split layer.
    box(
        ax,
        (0.055, 0.765),
        (0.23, 0.115),
        "Raw multi-sensor robotic-arm data\nmotion, current, vibration, torque and related channels",
        COLORS["data"],
        title="Input data",
        wrap=28,
    )
    box(
        ax,
        (0.37, 0.765),
        (0.24, 0.115),
        "Leakage-safe split by segment, file or run unit\nno random overlapping-window split",
        COLORS["split"],
        title="Protocol split",
        wrap=28,
    )
    box(
        ax,
        (0.70, 0.765),
        (0.235, 0.115),
        "Train, validation/calibration and held-out test roles remain separate",
        COLORS["split"],
        title="Data roles",
        wrap=28,
    )
    arrow(ax, (0.285, 0.822), (0.37, 0.822))
    arrow(ax, (0.61, 0.822), (0.70, 0.822))

    # Swimlane labels.
    for y, label, color in [
        (0.61, "Training role", COLORS["train"]),
        (0.39, "Validation/calibration role", COLORS["val"]),
        (0.17, "Test and deployment role", COLORS["test"]),
    ]:
        ax.text(
            0.055,
            y + 0.08,
            label,
            ha="left",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            color="#374151",
        )
        ax.plot([0.05, 0.95], [y + 0.14, y + 0.14], color=color, lw=7, alpha=0.35, solid_capstyle="round")

    # Training lane.
    box(
        ax,
        (0.13, 0.575),
        (0.22, 0.125),
        "Fit preprocessing on training split only\ntrain-only normalization",
        COLORS["train"],
        title="Preprocessing",
        wrap=27,
    )
    box(
        ax,
        (0.43, 0.575),
        (0.39, 0.125),
        "LightGBM, XGBoost, RandomForest, IsolationForest,\nAutoEncoder, LSTM-AE and USAD",
        COLORS["train"],
        title="Candidate model pool",
        wrap=36,
    )
    arrow(ax, (0.35, 0.637), (0.43, 0.637))

    # Validation lane.
    box(
        ax,
        (0.13, 0.355),
        (0.23, 0.13),
        "Generate validation scores\ncalibrate candidate thresholds on validation/calibration data",
        COLORS["val"],
        title="Threshold calibration",
        wrap=27,
    )
    box(
        ax,
        (0.44, 0.355),
        (0.34, 0.13),
        "Balanced, safety, low false alarm, robustness,\ndeployment and label-budget utilities",
        COLORS["val"],
        title="Engineering utility selection",
        wrap=34,
    )
    arrow(ax, (0.62, 0.575), (0.62, 0.485))
    arrow(ax, (0.36, 0.42), (0.44, 0.42))

    # Test and deployment lane.
    box(
        ax,
        (0.13, 0.14),
        (0.24, 0.13),
        "Evaluate selected model-threshold pair once on held-out test data",
        COLORS["test"],
        title="Test-only final evaluation",
        wrap=28,
    )
    box(
        ax,
        (0.45, 0.14),
        (0.22, 0.13),
        "Report FAR, MDR, PR-AUC, regret and prepared-input latency",
        COLORS["test"],
        title="Final metrics",
        wrap=25,
    )
    box(
        ax,
        (0.75, 0.14),
        (0.18, 0.13),
        "Model-selection guidance\nand deployment decision",
        COLORS["deploy"],
        title="Engineering output",
        wrap=24,
    )
    arrow(ax, (0.61, 0.355), (0.31, 0.27), rad=0.06)
    arrow(ax, (0.37, 0.205), (0.45, 0.205))
    arrow(ax, (0.67, 0.205), (0.75, 0.205))

    ax.text(
        0.5,
        0.055,
        "Decision layer around existing detectors; no test-set threshold selection and no new anomaly score.",
        ha="center",
        va="center",
        fontsize=8.8,
        color="#4B5563",
        style="italic",
    )

    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"fig1_framework_workflow_v19.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
