"""Figure 4(a): UE reference backbone flows. Run this file to redraw this panel."""
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

from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, SymLogNorm, LinearSegmentedColormap

COORDINATES = ["tail_longitude", "tail_latitude", "head_longitude", "head_latitude"]

def segments(frame):
    return frame[COORDINATES].to_numpy(dtype=float).reshape(-1, 2, 2)

def map_axes(ax, bounds):
    ax.set_xlim(bounds[:2])
    ax.set_ylim(bounds[2:])
    ax.set_aspect(1 / np.cos(np.deg2rad((bounds[2] + bounds[3]) / 2)))
    ax.set_axis_off()

plt.rcParams["font.size"] = 7.5

# Fixed v6 color scale, shared originally with the ICNN map: thousand tons/year.
FLOW_MAXIMUM = 76274.9237174365
LINE_WIDTH_MULTIPLIER = 2.2

def draw_flows(ax, filename):
    frame = pd.read_csv(DATA / filename, float_precision="round_trip")
    flows = frame["reference_flow"].to_numpy()
    order = np.argsort(flows, kind="stable")
    intensity = np.log10(1 + flows[order]) / np.log10(1 + FLOW_MAXIMUM)
    colors = plt.get_cmap("Blues")(intensity)
    colors[:, 3] = .35 + .65 * intensity
    colors[intensity == 0, :3] = (183 / 255, 193 / 255, 202 / 255)
    widths = (.15 + .75 * intensity) * LINE_WIDTH_MULTIPLIER
    ax.add_collection(LineCollection(segments(frame)[order], colors=colors,
                                     linewidths=widths, capstyle="round"))

fig = plt.figure(figsize=(3.35, 2.75))
ax = fig.add_axes([.015, .30, .97, .69])
map_axes(ax, (-126, -66, 24, 50))
draw_flows(ax, "conus.csv")

# Alaska and Hawaii are insets of this same panel.
for filename, label, rectangle, bounds in [
    ("alaska.csv", "AK", [.03, .195, .22, .225], [-169, -129, 50, 73]),
    ("hawaii.csv", "HI", [.275, .195, .155, .15], [-161, -154, 18, 23]),
]:
    inset = fig.add_axes(rectangle, facecolor="white")
    map_axes(inset, bounds)
    draw_flows(inset, filename)
    inset.text(.02, .98, label, fontsize=5.5, ha="left", va="top",
               transform=inset.transAxes, color="#555555")

color_axis = fig.add_axes([.09, .118, .82, .035])
bar = fig.colorbar(ScalarMappable(norm=Normalize(0, np.log10(1 + FLOW_MAXIMUM)), cmap="Blues"),
                   cax=color_axis, orientation="horizontal")
bar.solids.set_rasterized(False)
# Adjacent vector color cells overlap at their edges to avoid PDF hairlines.
bar.solids.set_edgecolor("face")
bar.set_ticks(np.log10(1 + np.array([0, 100, 1000, 10000, FLOW_MAXIMUM])),
              labels=["0", "100", "1k", "10k", f"{FLOW_MAXIMUM / 1000:.1f}k"])
bar.set_label("Flow (thousand tons/year; log scale)", fontsize=7.3, labelpad=3)
bar.ax.tick_params(labelsize=6.6, pad=1, length=2)
bar.ax.get_xticklabels()[0].set_ha("left")
bar.ax.get_xticklabels()[-1].set_ha("right")
bar.outline.set_linewidth(.4)

# The PDF keeps lines and text as vectors; the PNG is exported at 600 dpi.
fig.savefig(FOLDER / "figure.pdf", dpi=600)
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
