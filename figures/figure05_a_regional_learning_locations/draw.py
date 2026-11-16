"""Draw Figure 5(a): representative regional learning locations."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "raw_data"
COORDINATES = ["tail_longitude", "tail_latitude", "head_longitude", "head_latitude"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.unicode_minus": False,
    "pdf.fonttype": 42, "savefig.facecolor": "white",
})

# All coordinates belong to this panel's frozen geographic view.
backbone = pd.read_csv(DATA / "backbone.csv", float_precision="round_trip")
locations = pd.read_csv(DATA / "locations.csv", float_precision="round_trip")
segments = backbone[COORDINATES].to_numpy().reshape(-1, 2, 2)
points = locations[["longitude", "latitude"]].to_numpy()

fig, ax = plt.subplots(figsize=(3.35, 2.95))
ax.set_facecolor("#F6F8FA")
ax.add_collection(LineCollection(
    segments, colors="#B4BEC7", linewidths=0.32, alpha=0.72, zorder=1,
))
ax.scatter(points[:, 0], points[:, 1], marker="o", c="#0072B2", s=45,
           label="Region Location", zorder=3)

# Keep the original labels and east/west label alignment.
for point, label in zip(points, locations["label"]):
    east = point[0] > -80
    ax.annotate(label.split()[-1], point, xytext=(-4 if east else 4, 4),
                ha="right" if east else "left", textcoords="offset points",
                fontsize=7)

ax.set_xlim(-125, -66)
ax.set_ylim(24, 50)
ax.set_aspect(1 / np.cos(np.deg2rad(points[:, 1].mean())))
ax.set_axis_off()
handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Line2D([0], [0], color="#9CA9B4", linewidth=1.1))
labels.insert(0, "Retained physical backbone")
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.035),
          ncol=2, fontsize=6.5, frameon=False, borderaxespad=0)
fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.19)

# Export this one panel as a vector PDF and a 600-dpi PNG.
fig.savefig(FOLDER / "figure.pdf")
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
