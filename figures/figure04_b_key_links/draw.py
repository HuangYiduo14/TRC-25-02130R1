"""Figure 4(b): 200 fixed key links. Run this file to redraw this panel."""
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
key_links = pd.read_csv(DATA / "key_links.csv", float_precision="round_trip")
fig, ax = plt.subplots(figsize=(3.35, 2.95))
fig.subplots_adjust(left=.015, right=.985, top=.985, bottom=.04)
ax.set_facecolor("#F6F8FA")
ax.add_collection(LineCollection(segments(background), colors="#AEBBC5",
                                 linewidths=.28, alpha=.65, zorder=1))
key_segments = segments(key_links)
ax.add_collection(LineCollection(key_segments, colors="#0072B2", linewidths=.9))
midpoints = key_segments.mean(axis=1)
ax.scatter(midpoints[:, 0], midpoints[:, 1], s=8, color="#0072B2", linewidths=0)
ax.set_xlim(-125, -66)
ax.set_ylim(24, 50)
ax.set_aspect(1 / np.cos(np.deg2rad(37)))
ax.set_axis_off()

# The PDF keeps lines and text as vectors; the PNG is exported at 600 dpi.
fig.savefig(FOLDER / "figure.pdf", dpi=600)
fig.savefig(FOLDER / "figure.png", dpi=600)
plt.close(fig)
