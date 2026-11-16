"""Draw only supplement Figure S2(a) from this folder's data.

Run this file in Python. It writes figure.pdf and figure.png beside draw.py.
Dependencies: NumPy and Matplotlib. Coordinates are longitude/latitude degrees.
"""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Rectangle


FOLDER = Path(__file__).resolve().parent
DATA = FOLDER / "raw_data"
LONGITUDE_LIMITS = (-125.0, -66.0)
LATITUDE_LIMITS = (24.0, 50.0)


def endpoint_segments(data):
    """Turn the two geographic endpoints of each link into a line segment."""
    tails = np.column_stack((data["tail_longitude"], data["tail_latitude"]))
    heads = np.column_stack((data["head_longitude"], data["head_latitude"]))
    return np.stack((tails, heads), axis=1)


# These files contain only links included in this panel, in their original order.
with np.load(DATA / "regional_links.npz") as data:
    regional_segments = endpoint_segments(data)
    region_index = data["region_index"]

with np.load(DATA / "backbone_links.npz") as data:
    backbone_segments = endpoint_segments(data)

# Region indices select the original hue. Colors distinguish membership within
# this resolution; the same color does not identify a region in another panel.
hues = np.mod(region_index * 0.61803398875, 0.999999)
regional_colors = hsv_to_rgb(
    np.column_stack((hues, np.full(len(hues), 0.72), np.full(len(hues), 0.77)))
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    }
)
figure, axes = plt.subplots(figsize=(4.35, 2.5))
figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)

# Common backbone first, then colored regional links in the foreground.
axes.add_collection(
    LineCollection(
        backbone_segments,
        colors="#C9CED2",
        linewidths=0.2,
        alpha=0.85,
        zorder=1,
    )
)
axes.add_collection(
    LineCollection(
        regional_segments,
        colors=regional_colors,
        linewidths=0.32,
        alpha=0.95,
        zorder=2,
    )
)

# The outline marks the window used by the accompanying Chicago panel.
chicago_west, chicago_east = -88.7, -87.15
chicago_south, chicago_north = 41.2, 42.5
axes.add_patch(
    Rectangle(
        (chicago_west, chicago_south),
        chicago_east - chicago_west,
        chicago_north - chicago_south,
        fill=False,
        edgecolor="#151515",
        linewidth=0.65,
        zorder=3,
    )
)

axes.set_xlim(LONGITUDE_LIMITS)
axes.set_ylim(LATITUDE_LIMITS)
middle_latitude = np.mean(LATITUDE_LIMITS)
axes.set_aspect(1 / np.cos(np.deg2rad(middle_latitude)))
axes.set_axis_off()

# The PDF retains vector lines. The PNG is 600 dpi (2610 by 1500 pixels).
# Figure and panel captions are supplied by the manuscript, not embedded here.
figure.savefig(FOLDER / "figure.pdf", dpi=600)
figure.savefig(FOLDER / "figure.png", dpi=600)
plt.close(figure)
