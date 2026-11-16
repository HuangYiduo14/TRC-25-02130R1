"""Draw Figure 7e: regional change.

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


def compact_flow(value):
    """Format flow ticks, using k for thousands of the stated flow unit."""
    return f"{value / 1000:.2g}k" if abs(value) >= 1000 else f"{value:.2g}"


def add_colorbar(lines, vertical_position, label, ticks):
    """Place this panel's horizontal flow scale below the map."""
    color_axis = axis.inset_axes([0.06, vertical_position, 0.88, 0.034])
    colorbar = figure.colorbar(lines, cax=color_axis, orientation="horizontal")
    flow_ticks = np.sign(ticks) * (10 ** np.abs(ticks) - 1)
    colorbar.set_ticks(ticks, labels=[compact_flow(value) for value in flow_ticks])
    colorbar.ax.tick_params(labelsize=7, pad=1, length=2)
    labels = colorbar.ax.get_xticklabels()
    labels[0].set_ha("left")
    labels[-1].set_ha("right")
    colorbar.set_label(label, fontsize=7.5, labelpad=1)
    # Keep the color scale vector and cover PDF viewer seams between its cells.
    colorbar.solids.set_rasterized(False)
    colorbar.solids.set_edgecolor("face")

# Raw flows are in thousand tons/year. Positive change means CLOSED > OPEN.
flows = pd.read_csv(DATA / "regional.csv", float_precision="round_trip")
flow_change = flows["flow_closed"].to_numpy() - flows["flow_open"].to_numpy()
signed_log_change = np.sign(flow_change) * np.log10(1 + np.abs(flow_change))

# Preserve the symmetric full-network change scale of the manuscript.
color_limit = np.log10(1 + 16392.54194915416)
lines = LineCollection(
    line_segments(flows),
    array=signed_log_change,
    cmap="PiYG",
    norm=Normalize(-color_limit, color_limit),
    linewidths=0.34,
    alpha=0.9,
)
axis.add_collection(lines)
add_colorbar(lines, -0.13, "Regional", np.array([-color_limit, 0, color_limit]))

# Retain the geographic viewport, aspect ratio, and margins of the manuscript.
axis.set_xlim(-83, -75)
axis.set_ylim(24, 40)
axis.set_aspect(1 / np.cos(np.deg2rad(37)))
axis.set_axis_off()

figure.savefig(FOLDER / "figure.pdf")
figure.savefig(FOLDER / "figure.png", dpi=600)
plt.close(figure)
