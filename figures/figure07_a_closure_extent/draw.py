"""Draw Figure 7a: closure extent.

Run this file in Python, or open it in your preferred Python editor and run it.
It reads only the CSV files in raw_data beside this file and writes one panel
as figure.pdf (vector) and figure.png (600 dpi).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "raw_data"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 7.5,
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})


def line_segments(table):
    """Return the two geographic endpoints of every plotted link."""
    columns = [
        "tail_longitude", "tail_latitude",
        "head_longitude", "head_latitude",
    ]
    return table[columns].to_numpy(float).reshape(-1, 2, 2)


figure, axis = plt.subplots(figsize=(1.25, 3.35))
figure.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.36)

# The gray lines give the backbone context; blue lines show the closure.
backbone = pd.read_csv(DATA / "backbone_context.csv", float_precision="round_trip")
closure = pd.read_csv(DATA / "closure.csv", float_precision="round_trip")

axis.add_collection(LineCollection(
    line_segments(backbone), colors="#b8c1ca", linewidths=0.23, alpha=0.65,
))
axis.add_collection(LineCollection(
    line_segments(closure), colors="#0072B2", linewidths=0.8,
))

# Retain the geographic viewport, aspect ratio, and margins of the manuscript.
axis.set_xlim(-83, -75)
axis.set_ylim(24, 40)
axis.set_aspect(1 / np.cos(np.deg2rad(37)))
axis.set_axis_off()

figure.savefig(FOLDER / "figure.pdf")
figure.savefig(FOLDER / "figure.png", dpi=600)
plt.close(figure)
