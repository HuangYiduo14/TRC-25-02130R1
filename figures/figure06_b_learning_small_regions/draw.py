"""Draw Figure 6(b): learning curves for small regions."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FOLDER = Path(__file__).resolve().parent
observations = pd.read_csv(FOLDER / "raw_data/learning.csv",
                           float_precision="round_trip")
fractions = [0.1, 0.25, 0.5, 0.7, 1.0]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.labelsize": 9, "legend.fontsize": 8,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})
fig, ax = plt.subplots(figsize=(3.35, 3.05))
fig.subplots_adjust(left=0.19, right=0.98, bottom=0.16, top=0.85)

# Every family/fraction has the same four complete, finite regional results.
# Faint marks show the raw observations; solid markers and whiskers show
# their median and interquartile range. The two families are offset slightly.
families = [
    ("ICNN", -0.035, "o", "#0072B2"),
    ("XGBOOST", 0.035, "s", "#D55E00"),
]
for family, offset, marker, color in families:
    previous = None
    for fraction in fractions:
        rows = observations.loc[
            (observations["family"] == family)
            & (observations["fraction"] == fraction)
        ].sort_values("region_key")
        values = rows["TEST_WMAPE_percent"].to_numpy()
        x = fraction + offset

        # Fixed spacing makes each of the four regional observations visible.
        for ordinal, value in enumerate(values):
            point_x = x + (ordinal - 1.5) * 0.007
            ax.scatter(point_x, value, s=13, marker=marker, color=color,
                       alpha=0.36, linewidths=0, zorder=2)

        lower, median, upper = np.quantile(values, [0.25, 0.5, 0.75])
        ax.errorbar(x, median, yerr=[[median - lower], [upper - median]],
                    fmt=marker, markersize=5, markerfacecolor=color,
                    markeredgecolor=color, color=color, capsize=3,
                    elinewidth=1, linewidth=0)
        if previous is not None:
            ax.plot([previous[0], x], [previous[1], median],
                    color=color, linewidth=0.9, alpha=0.7)
        previous = (x, median)

ax.set_xlim(0.015, 1.075)
ax.set_xticks(fractions, ["10", "25", "50", "70", "100"])
ax.set_xlabel("Training fraction (%)", labelpad=2)
ax.set_ylabel("TEST cost WMAPE (%)")
upper_limit = observations["TEST_WMAPE_percent"].max() * 1.12
ax.set_ylim(-0.025 * upper_limit, upper_limit)

ax.grid(axis="y", alpha=0.17)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=8)
legend = [
    Line2D([0], [0], marker="o", linestyle="None", color="#0072B2", label="ICNN"),
    Line2D([0], [0], marker="s", linestyle="None", color="#D55E00", label="XGBoost"),
]
ax.legend(handles=legend, loc="lower center", bbox_to_anchor=(0.5, 1.02),
          ncol=2, frameon=False, borderaxespad=0, columnspacing=1.0)

# Export this one panel as a vector PDF and a 600-dpi PNG.
fig.savefig(FOLDER / "figure.pdf")
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
