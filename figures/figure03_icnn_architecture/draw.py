"""Figure 3: anchored ICNN architecture. Run this file to redraw this panel."""
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
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def read_rows(filename):
    with (DATA / filename).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

nodes = read_rows("nodes.csv")
arrows = read_rows("arrows.csv")
plt.rcParams["font.size"] = 12
fig, ax = plt.subplots(figsize=(13.5, 7.0))
fig.subplots_adjust(left=.012, right=.988, bottom=.025, top=.99)
ax.set(xlim=(-.05, 13.2), ylim=(-.05, 7.0))
ax.axis("off")

# Draw connections first so the boxes cover their ends.
for arrow in arrows:
    start = (float(arrow["x1"]), float(arrow["y1"]))
    end = (float(arrow["x2"]), float(arrow["y2"]))
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                                lw=1.4, color="#475467", linestyle=arrow["style"], zorder=1))
    if arrow["label"]:
        ax.text(float(arrow["label_x"]), float(arrow["label_y"]), arrow["label"],
                ha="center", va="center", fontsize=16.6, color="#344054",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5}, zorder=3)

for node in nodes:
    x, y = float(node["x"]), float(node["y"])
    width, height = float(node["width"]), float(node["height"])
    ax.add_patch(FancyBboxPatch((x - width / 2, y - height / 2), width, height,
                                boxstyle="round,pad=0.015,rounding_size=0.07",
                                lw=1.2, ec="#475467", fc=node["fill"], zorder=2))
    ax.text(x, y, node["label"], ha="center", va="center",
            fontsize=float(node["fontsize"]), linespacing=1.4, zorder=3)

# The PDF keeps lines and text as vectors; the PNG is exported at 600 dpi.
fig.savefig(FOLDER / "figure.pdf", dpi=600)
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
