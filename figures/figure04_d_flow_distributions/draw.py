"""Figure 4(d): backbone flow distributions. Run this file to redraw this panel."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "raw_data"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "pdf.fonttype": 42,
    "axes.unicode_minus": False, "savefig.facecolor": "white",
})

from matplotlib.patches import Rectangle

flows = pd.read_csv(DATA / "flows.csv", float_precision="round_trip")
families = ["P reference", "FFTT", "Power4", "ICNN", "XGBoost"]
labels = {"P reference": "FNE-CFW", "FFTT": "H-FFTT", "Power4": "H-POWER4",
          "ICNN": "H-ICNN", "XGBoost": "H-XGB"}
styles = {
    "P reference": dict(color="#202020", linewidth=2.0, linestyle=(0, (4, 2.5)), zorder=5),
    "ICNN": dict(color="#0072B2", linewidth=1.85, linestyle="-", zorder=4),
    "FFTT": dict(color="#BC5090", linewidth=1.10, linestyle=(0, (2, 1.5)), zorder=3),
    "Power4": dict(color="#009E73", linewidth=1.10, linestyle=(0, (5, 1.6, 1, 1.6)), zorder=3),
    "XGBoost": dict(color="#D89000", linewidth=1.10, linestyle=(0, (1, 1.3)), zorder=3),
}

def draw_ecdf(ax, width_scale=1.0):
    """Each curve uses every link, including zeros; the zoom only changes axes."""
    handles = {}
    for family in families:
        x = np.sort(flows[family].to_numpy())
        fraction = np.arange(1, len(x) + 1, dtype=float) / len(x)
        style = dict(styles[family])
        style["linewidth"] *= width_scale
        handles[family] = ax.step(x, fraction, where="post", label=labels[family], **style)[0]
    return handles

plt.rcParams["font.size"] = 8
fig, ax = plt.subplots(figsize=(3.60, 3.25))
fig.subplots_adjust(left=.18, right=.97, bottom=.17, top=.82)
handles = draw_ecdf(ax)
ax.set_xlabel("Flow (thousand tons/year)")
ax.set_ylabel("Empirical cumulative fraction")
ax.grid(alpha=.18, linewidth=.45)
ax.set_xlim(0, 250)
ax.set_ylim(.294, .341)
ax.set_xticks([0, 50, 100, 150, 200, 250])
ax.set_yticks([.30, .31, .32, .33, .34])
ax.text(.025, .97, "Low-flow detail", transform=ax.transAxes, fontsize=7.0, va="top")

legend_order = ["P reference", "ICNN", "FFTT", "Power4", "XGBoost"]
legend = fig.legend([handles[name] for name in legend_order], [labels[name] for name in legend_order],
                    loc="upper center", bbox_to_anchor=(.55, .98), ncol=3, frameon=False,
                    fontsize=7.1, columnspacing=1.0, handlelength=2.7, handletextpad=.5)
for text in legend.get_texts():
    if text.get_text() in ["FNE-CFW", "H-ICNN"]:
        text.set_fontweight("bold")

# The inset shows the full empirical distribution, using the same observations.
inset = fig.add_axes([.64, .205, .31, .23], facecolor="white")
draw_ecdf(inset, width_scale=.57)
inset.set_xscale("symlog", linthresh=100)
inset.set_xlim(0, 1.025 * flows[families].to_numpy().max())
inset.set_ylim(0, 1.03)
inset.set_xticks([0, 1000, 100000], labels=["0", "1k", "100k"])
inset.set_yticks([0, .5, 1])
inset.tick_params(labelsize=5.5, length=2, pad=1)
inset.text(.045, .92, "Full range", transform=inset.transAxes, fontsize=5.7, va="top")
inset.add_patch(Rectangle((0, .294), 250, .341 - .294,
                           facecolor="none", edgecolor="#666666", linewidth=.7))
for spine in inset.spines.values():
    spine.set_linewidth(.6)

# The PDF keeps lines and text as vectors; the PNG is exported at 600 dpi.
fig.savefig(FOLDER / "figure.png", dpi=600)
fig.savefig(FOLDER / "figure.pdf", dpi=600)
plt.close(fig)
