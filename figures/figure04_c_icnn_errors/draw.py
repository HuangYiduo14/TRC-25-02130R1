"""Figure 4(c): ICNN differences on key links. Run this file to redraw this panel."""
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

background = pd.read_csv(DATA / "backbone.csv", float_precision="round_trip")
links = pd.read_csv(DATA / "key_differences.csv", float_precision="round_trip")
values = links["signed_percent_difference"].to_numpy()
absolute = np.abs(values)
link_segments = segments(links)

difference_colors = LinearSegmentedColormap.from_list("BackboneDifference", [
    "#053061", "#2166ac", "#4393c3", "#92c5de", "#bfc4c8",
    "#f4a582", "#d6604d", "#b2182b", "#67001f",
])
normalization = SymLogNorm(linthresh=.1, linscale=1, base=10, vmin=-100, vmax=100)
fig = plt.figure(figsize=(3.35, 2.75), dpi=600)
ax = fig.add_axes([.015, .245, .97, .745])
map_axes(ax, (-126, -66, 24, 50))
ax.add_collection(LineCollection(segments(background), colors="#bbc5cf", linewidths=.16, alpha=.65))

# Stronger differences are drawn on top for the link strokes.
order = np.argsort(absolute, kind="stable")
magnitude = np.log10(1 + absolute[order]) / np.log10(101)
colors = difference_colors(normalization(values[order]))
colors[:, 3] = .70 + .30 * magnitude
ax.add_collection(LineCollection(link_segments[order], colors=colors,
                                 linewidths=.95 + 2.25 * magnitude, capstyle="round"))

# Circle FACE area is 6 + 4*|difference| square points. No jitter is applied.
# Matplotlib's circular marker uses s = 4*face_area/pi. Small circles are last.
centers = link_segments.mean(axis=1)
circle_sizes = 4 * (6 + 4 * absolute) / np.pi
circle_order = np.argsort(-absolute, kind="stable")
circle_colors = difference_colors(normalization(values))
circle_colors[:, 3] = .70 + .30 * np.log10(1 + absolute) / np.log10(101)
ax.scatter(centers[circle_order, 0], centers[circle_order, 1], s=circle_sizes[circle_order],
           c=circle_colors[circle_order], marker="o", edgecolors="none", linewidths=0, zorder=5)
levels = [0, 1, 5, 15]
legend = [ax.scatter([], [], s=4 * (6 + 4 * level) / np.pi, marker="o",
                      facecolors="#76818b", edgecolors="none", linewidths=0) for level in levels]
fig.text(.5, .27, "Circle area: |difference| (visible minimum at 0%)",
         ha="center", va="center", fontsize=6.1)
fig.legend(legend, [f"{level}%" for level in levels], loc="center", bbox_to_anchor=(.5, .216),
           ncol=4, frameon=False, fontsize=6.1, columnspacing=1.4, handletextpad=.45, borderaxespad=0)

color_axis = fig.add_axes([.09, .118, .82, .035])
bar = fig.colorbar(ScalarMappable(norm=normalization, cmap=difference_colors),
                   cax=color_axis, orientation="horizontal")
bar.solids.set_rasterized(False)
bar.solids.set_edgecolor("face")
bar.set_ticks([-100, -10, -1, -.1, 0, .1, 1, 10, 100],
              labels=["-100", "-10", "-1", "-.1", "0", ".1", "1", "10", "100"])
bar.set_label("Bounded difference from FNE-CFW (%)", fontsize=7.3, labelpad=3)
bar.ax.tick_params(labelsize=6.6, pad=1, length=2)
bar.ax.get_xticklabels()[0].set_ha("left")
bar.ax.get_xticklabels()[-1].set_ha("right")
bar.outline.set_linewidth(.4)

# The PDF keeps lines and text as vectors; the PNG is exported at 600 dpi.
fig.savefig(FOLDER / "figure.pdf", dpi=600)
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
