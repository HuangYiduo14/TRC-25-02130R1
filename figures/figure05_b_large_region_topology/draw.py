"""Draw Figure 5(b): the large regional topology."""
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

geometry = pd.read_csv(DATA / "geometry.csv", float_precision="round_trip")
segments = geometry[COORDINATES].to_numpy().reshape(-1, 2, 2)

fig, ax = plt.subplots(figsize=(3.35, 2.95))
ax.set_facecolor("#F6F8FA")

# The role selects the symbol; duplicated endpoints mark a graph node.
for role, rows in geometry.groupby("geometry_role", sort=True):
    coordinates = rows[COORDINATES].to_numpy().reshape(-1, 2, 2)
    if role == "PHYSICAL_LINK":
        ax.add_collection(LineCollection(
            coordinates, colors="#678EAA", linewidths=0.8,
        ))
    else:
        points = coordinates.mean(axis=1)
        marker, color, size, label = {
            "CENTROID": ("*", "#222222", 30, "Centroid"),
            "PORTAL": ("o", "#D55E00", 12, "Boundary node"),
        }[role]
        ax.scatter(points[:, 0], points[:, 1], marker=marker, c=color,
                   s=size, label=label, zorder=3)

# Match the frozen local extent, including its eight-percent margin.
coordinates = segments.reshape(-1, 2)
lower = coordinates.min(axis=0)
upper = coordinates.max(axis=0)
padding = np.maximum((upper - lower) * 0.08, 0.015)
ax.set_xlim(max(-125, lower[0] - padding[0]), min(-66, upper[0] + padding[0]))
ax.set_ylim(max(24, lower[1] - padding[1]), min(50, upper[1] + padding[1]))
ax.set_aspect(1 / np.cos(np.deg2rad(segments[:, :, 1].mean())))
ax.set_axis_off()

handles, labels = ax.get_legend_handles_labels()
handles.insert(0, Line2D([0], [0], color="#678EAA", linewidth=1.1))
labels.insert(0, "Physical regional links")
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.035),
          ncol=2, fontsize=6.5, frameon=False, borderaxespad=0)
fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.19)

# Export this one panel as a vector PDF and a 600-dpi PNG.
fig.savefig(FOLDER / "figure.pdf")
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
