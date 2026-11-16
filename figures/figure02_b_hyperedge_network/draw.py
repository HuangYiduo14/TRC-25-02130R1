"""Figure 2(b): hyperedge network. Run this file to redraw this panel."""
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

import csv
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

def read_rows(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

nodes = read_rows("nodes.csv")
edges = read_rows("edges.csv")
regions = read_rows("regions.csv")
positions = {node["node_id"]: (float(node["x"]), float(node["y"])) for node in nodes}

plt.rcParams["font.size"] = 16.8
fig, ax = plt.subplots(figsize=(6.6, 6.2))
fig.subplots_adjust(left=.015, right=.985, bottom=.24, top=.98)
ax.set(xlim=(-.1, 10.1), ylim=(0, 7.3))
ax.set_aspect("equal")
ax.axis("off")

# A shaded rounded box marks the regional boundary.
for region in regions:
    ax.add_patch(FancyBboxPatch(
        (float(region["xmin"]), float(region["ymin"])),
        float(region["width"]), float(region["height"]),
        boxstyle="round,pad=0.03,rounding_size=.15",
        facecolor="#f6f2fb", edgecolor="#8a6ca8", linewidth=1.4,
        linestyle="--", zorder=0,
    ))

edge_colors = {"backbone": "#202020", "regional_physical": "#2674ad", "regional_shortcut": "#bf4928"}
for edge in edges:
    ax.add_patch(FancyArrowPatch(
        positions[edge["tail"]], positions[edge["head"]],
        arrowstyle="-|>", mutation_scale=16, shrinkA=8, shrinkB=8,
        connectionstyle="arc3,rad=" + edge["curvature"],
        linewidth=2.4 if edge["role"] == "backbone" else 1.6,
        color=edge_colors[edge["role"]], zorder=1,
    ))

for node in nodes:
    x, y = positions[node["node_id"]]
    role = node["role"]
    marker = "*" if role.startswith("centroid") else "o"
    size = 220 if marker == "*" else 80 if role == "portal" else 35
    face = ("#e5ad30" if marker == "*" else "white" if role == "portal"
            else "#2674ad" if role == "interior" else "#202020")
    ax.scatter(x, y, s=size, marker=marker, c=face,
               edgecolors="#202020", linewidths=1.1, zorder=3)
    ax.annotate(node["label"], (x, y),
                xytext=(float(node["label_dx"]), float(node["label_dy"])),
                textcoords="offset points", ha=node["label_ha"], fontsize=16.8, zorder=4)
ax.text(5, .05, "Region r", ha="center", va="top", color="#765393", fontsize=16.8)

# Keep the shared physical/hyperedge legend used in the manuscript.
legend = [
    Line2D([0], [0], color="#202020", lw=2.4, label="Retained backbone"),
    Line2D([0], [0], color="#2674ad", lw=1.6, label="Regional physical arc"),
    Line2D([0], [0], color="#bf4928", lw=1.6, label="Regional shortcut"),
    Line2D([0], [0], marker="o", ls="", mfc="white", mec="#202020", ms=8, label="Portal"),
    Line2D([0], [0], marker="*", ls="", mfc="#e5ad30", mec="#202020", ms=12, label="OD centroid terminal"),
]
fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .015),
           ncol=2, frameon=False, fontsize=13)

# The PDF keeps lines and text as vectors; the PNG is exported at 600 dpi.
fig.savefig(FOLDER / "figure.pdf", dpi=600)
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
