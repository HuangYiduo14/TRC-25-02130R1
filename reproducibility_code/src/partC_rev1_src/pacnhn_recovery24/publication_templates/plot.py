"""Reproduce this scientific figure using only CSV files beside this script.

No repository, network, parquet, array archive, model or external geometry input
is used. Missing panels are recorded as dispositions, never drawn as data.
"""
from pathlib import Path
import hashlib
import json
import math
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.ticker import MaxNLocator, FuncFormatter

ROOT = Path(__file__).resolve().parent
READS = {}
PALETTE = {"ICNN": "#0072B2", "XGBOOST": "#E69F00", "POWER4": "#009E73",
           "FFTT": "#CC79A7", "CFW": "#6A3D9A", "FW": "#777777", "MSA": "#A65628",
           "REFERENCE": "#222222"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8,
    "pdf.fonttype": 42, "savefig.facecolor": "white"})


# EMBED_LOCAL_LOGICAL_CSV_SUPPORT_HERE


def read(name):
    return read_logical_csv(ROOT, name, READS)


def number(frame, column):
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def boolean(values):
    return values.astype(str).str.lower().isin(["true", "1", "1.0"])


def color(label):
    label = str(label).upper()
    for key in ["XGBOOST", "POWER4", "ICNN", "FFTT", "REFERENCE", "CFW", "MSA", "FW"]:
        if key in label:
            return PALETTE[key]
    return "#4C677D"


def groups(frame, name):
    if not name:
        return [("All observations", frame)]
    return list(frame.groupby(name, sort=True, dropna=False))


def suite_linestyle(label):
    """Keep a family's color while distinguishing frozen resolution suites."""
    for suite, style in [("s02", ":"), ("s03", "-"), ("s04", "--"),
                         ("s05", "-."), ("s06", (0, (5, 1, 1, 1, 1, 1)))]:
        if suite in str(label).lower():
            return style
    return "-"


def axis_style(ax, spec):
    ax.set_title(spec["title"], loc="left", pad=10)
    ax.set_xlabel(spec.get("xlabel", "")); ax.set_ylabel(spec.get("ylabel", ""))
    ax.spines[["right", "top"]].set_visible(False)
    ax.grid(axis="y", alpha=.16)
    # Setting a linear scale after categorical ticks can reset their formatter.
    for axis in ["x", "y"]:
        scale = spec.get(axis+"scale", "")
        if scale == "symlog":
            getattr(ax, "set_"+axis+"scale")("symlog", linthresh=float(spec.get(axis+"linthresh", "") or 1))
        elif scale == "log":
            getattr(ax, "set_"+axis+"scale")("log")


def geometry(frame):
    coords = frame[["tail_longitude", "tail_latitude", "head_longitude", "head_latitude"]].apply(pd.to_numeric, errors="coerce").to_numpy()
    keep = np.isfinite(coords).all(axis=1)
    # Crop only drawing. Scientific statistics retain the complete native B.
    keep &= (coords[:, [0, 2]].max(axis=1) >= -125) & (coords[:, [0, 2]].min(axis=1) <= -66)
    keep &= (coords[:, [1, 3]].max(axis=1) >= 24) & (coords[:, [1, 3]].min(axis=1) <= 50)
    return coords[keep].reshape(-1, 2, 2), keep


def eligible(frame, spec):
    kind = spec["kind"]
    if kind in {"map", "region_map", "flow_map", "location_map"}:
        _, keep = geometry(frame)
        if spec.get("y"):
            keep &= np.isfinite(number(frame, spec["y"]))
        return frame.loc[keep].copy()
    required = [spec.get("x", ""), spec.get("y", "")]
    required = [v for v in required if v]
    if kind in {"bar", "gain_loss", "points"}:
        required = [spec["y"]]
    keep = np.ones(len(frame), dtype=bool)
    for col in required:
        keep &= np.isfinite(number(frame, col))
    if kind not in {"bar", "gain_loss", "points"} and spec.get("xscale") == "log" and spec.get("x"):
        keep &= number(frame, spec["x"]) > 0
    if spec.get("yscale") == "log" and spec.get("y"):
        keep &= number(frame, spec["y"]) > 0
    selected = frame.loc[keep].copy()
    for col in required:
        selected[col] = pd.to_numeric(selected[col], errors="raise")
    return selected


def draw(ax, frame, spec):
    kind, x, y, group = [spec.get(k, "") for k in ["kind", "x", "y", "group"]]
    axis_style(ax, spec)
    if kind in {"region_map", "location_map"}:
        ax.set_facecolor("#F6F8FA")
        # Point rows repeat their actual coordinates at both endpoints.
        for role, subset in frame.groupby("geometry_role", sort=True):
            seg, _ = geometry(subset)
            if role == "PHYSICAL_LINK":
                ax.add_collection(LineCollection(seg, colors="#678EAA", linewidths=.8, rasterized=True))
            else:
                points = seg.mean(axis=1)
                styles = {"PORTAL": ("o", "#D55E00", 30), "CENTROID": ("*", "#222222", 70),
                          "REGION_LOCATION": ("o", "#0072B2", 45)}
                if role not in styles:
                    raise ValueError("Unknown geometry role: "+str(role))
                marker, shade, size = styles[role]
                ax.scatter(points[:,0], points[:,1], marker=marker, c=shade, s=size,
                           label=str(role).replace("_", " ").title(), zorder=3)
                if "label" in subset:
                    for point, label in zip(points, subset.label.fillna("")):
                        if label: ax.annotate(str(label), point, xytext=(4,4), textcoords="offset points", fontsize=8)
        coords, _ = geometry(frame)
        if kind == "location_map":
            ax.set_xlim(-125,-66); ax.set_ylim(24,50)
        else:
            values=coords.reshape(-1,2); lo=values.min(axis=0); hi=values.max(axis=0)
            pad=np.maximum((hi-lo)*.08,.015)
            ax.set_xlim(max(-125,lo[0]-pad[0]),min(-66,hi[0]+pad[0]))
            ax.set_ylim(max(24,lo[1]-pad[1]),min(50,hi[1]+pad[1]))
        ax.set_aspect(1/np.cos(np.deg2rad(float(coords[:,:,1].mean()))))
        ax.set_axis_off(); ax.legend(loc="lower left", frameon=False)
    elif kind == "flow_map":
        ax.set_facecolor("#F6F8FA")
        for j, (role, subset) in enumerate(frame.groupby("edge_kind", sort=True)):
            seg, _ = geometry(subset); raw=number(subset,y)
            values=np.log10(1+np.abs(raw)); maximum=float(values.max()) if len(values) else 0.
            lines=LineCollection(seg, array=values, cmap="Blues" if role=="BACKBONE" else "Oranges",
                norm=Normalize(0,max(maximum,1e-12)), linewidths=.75 if role=="BACKBONE" else .45,
                linestyles=np.where(raw<0,"dashed","solid").tolist(), rasterized=True)
            ax.add_collection(lines)
            bar=ax.figure.colorbar(lines,ax=ax,orientation="horizontal",pad=.025,fraction=.04,shrink=.75)
            ticks=np.linspace(0,maximum,4) if maximum else [0]
            bar.set_ticks(ticks,labels=[f"{10**v-1:,.0f}" for v in ticks])
            bar.set_label(str(role).title()+" "+spec.get("colorbar",y)+" (log1p scale)",fontsize=8)
            bar.ax.tick_params(labelsize=8)
        ax.set_xlim(-125,-66); ax.set_ylim(24,50);ax.set_aspect(1/np.cos(np.deg2rad(37)))
        ax.set_axis_off()
    elif kind == "map":
        seg, _ = geometry(frame)
        ax.set_facecolor("#F6F8FA")
        if y:
            values = number(frame, y)
            radius = float(np.max(np.abs(values)))
            radius = radius if radius > 0 else 1.
            if spec.get("symmetric") == "true":
                norm = TwoSlopeNorm(vmin=-radius, vcenter=0, vmax=radius); cmap = "RdBu_r"
            else:
                norm = Normalize(vmin=0, vmax=radius); cmap = "viridis"
            lines = LineCollection(seg, array=values, cmap=cmap, norm=norm, linewidths=.65, rasterized=True)
            ax.add_collection(lines)
            bar = ax.figure.colorbar(lines, ax=ax, orientation="horizontal", pad=.025, fraction=.045, shrink=.8)
            bar.set_label(spec.get("colorbar", y), fontsize=9); bar.ax.tick_params(labelsize=8)
            bar.set_ticks(np.linspace(-radius if spec.get("symmetric") == "true" else 0, radius, 5))
            bar.ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        else:
            ax.add_collection(LineCollection(seg, colors="#0072B2", linewidths=.9, rasterized=True))
            if spec.get("point_markers") == "true":
                midpoints = seg.mean(axis=1)
                ax.scatter(midpoints[:, 0], midpoints[:, 1], s=8, color="#0072B2", linewidths=0)
        ax.set_xlim(-125, -66); ax.set_ylim(24, 50)
        ax.set_aspect(1/np.cos(np.deg2rad(37)))
        ax.set_axis_off()
    elif kind == "ecdf":
        for label, subset in groups(frame, group):
            vals = np.sort(number(subset, x)); n = len(vals)
            ax.step(vals, np.arange(1, n+1)/n, where="post", label=f"{label} (n={n:,})", color=color(label))
        ax.set_ylim(0, 1.02); ax.legend(loc="best", frameon=False)
    elif kind == "scatter":
        for label, subset in groups(frame, group):
            ax.scatter(number(subset, x), number(subset, y), s=11, alpha=.65, label=f"{label} (n={len(subset):,})", color=color(label), rasterized=True)
        limit = max(float(np.max(number(frame, x))), float(np.max(number(frame, y))))
        ax.plot([0, limit], [0, limit], color="#333333", linestyle="--", linewidth=.8, label="Identity")
        ax.legend(loc="best", frameon=False)
    elif kind == "iqr":
        for group_index, (label, subset) in enumerate(groups(frame,group)):
            subset=subset.sort_values(x);xx=number(subset,x);yy=number(subset,y)
            low=number(subset,"q25_WMAPE");high=number(subset,"q75_WMAPE")
            if not np.isfinite(low).all() or not np.isfinite(high).all() or np.any(low>yy) or np.any(high<yy):
                raise ValueError("Invalid declared IQR")
            counts=number(subset,"region_seed_rows")
            if not np.isfinite(counts).all() or np.any(counts<1) or np.any(counts!=np.floor(counts)):
                raise ValueError("Finite plotted summaries need positive integer region counts")
            counts=counts.astype(int)
            # Independent available-case summaries: do not bridge missing fractions
            # or imply a paired learning curve when observed regions differ.
            ax.errorbar(xx,yy,yerr=np.vstack((yy-low,high-yy)),fmt="o",linestyle="none",
                        capsize=3,label=f"{label} (available-region summaries)",color=color(label))
            for px,py,n in zip(xx,yy,counts):
                ax.annotate(f"n={n}",(px,py),xytext=(8,7 if group_index%2==0 else -13),
                            textcoords="offset points",ha="left",fontsize=8,color=color(label))
        ax.set_xticks([.1,.25,.5,.7,1.],["10","25","50","70","100"])
        ax.legend(loc="best",frameon=False)
    elif kind in {"line", "median_curve"}:
        for label, subset in groups(frame, group):
            if kind == "median_curve":
                summary = subset.groupby(x)[y].agg(["median", "count"]).sort_index()
                xx, yy = summary.index.to_numpy(float), summary["median"].to_numpy(float)
                nn = summary["count"].to_numpy(int)
                caption = f"{label} (n/ordinal {nn.min()}–{nn.max()})"
            else:
                subset = subset.sort_values(x)
                xx, yy = number(subset, x), number(subset, y); caption = str(label)
            ax.plot(xx, yy, label=caption, color=color(label), linewidth=1.3,
                linestyle=(suite_linestyle(label) if kind == "line" else
                           "--" if "VALID" in str(label).upper() else "-"))
        threshold = spec.get("threshold", "")
        if threshold:
            ax.axhline(float(threshold), color="#777777", linestyle=":", linewidth=.8, label="Requested tolerance")
        ax.legend(loc="best", frameon=False)
    elif kind == "points":
        categories = sorted(frame[x].astype(str).unique())
        position = {v: i for i, v in enumerate(categories)}
        for j, (label, subset) in enumerate(groups(frame, group)):
            # No interpolation through absent resolutions.
            offset = (j-(len(groups(frame, group))-1)/2)*.09
            ax.scatter([position[str(v)]+offset for v in subset[x]], number(subset, y), color=color(label), label=str(label), s=35)
        ax.set_xticks(range(len(categories)), categories); ax.legend(loc="best", frameon=False)
    elif kind == "bar":
        labels = frame[x].astype(str).tolist(); values = number(frame, y)
        yy = np.arange(len(frame))
        ax.barh(yy, values, color=[color(v) for v in labels], height=.65)
        ax.set_yticks(yy, [textwrap.fill(v, 24) for v in labels], fontsize=9); ax.invert_yaxis()
        ax.set_xlabel(spec.get("xlabel", "")); ax.set_ylabel("")
        if "status" in frame:
            censored = frame.status.astype(str).str.upper().eq("TIME_LIMIT").to_numpy()
            if censored.any():
                ax.scatter(values[censored], yy[censored], marker=">", color="#111111", s=36, zorder=3, label="TIME_LIMIT: completion time censored")
                ax.legend(loc="best", frameon=False, fontsize=8)
    elif kind == "gain_loss":
        rows = []
        for label, subset in groups(frame, group):
            values = number(subset, y)
            rows.append((str(label), float(values[values>0].sum()), float(-values[values<0].sum())))
        yy = np.arange(len(rows))
        ax.barh(yy-.18, [r[1] for r in rows], height=.33, color="#D55E00", label="Positive link-flow change")
        ax.barh(yy+.18, [r[2] for r in rows], height=.33, color="#0072B2", label="Magnitude of negative change")
        ax.set_yticks(yy, [textwrap.fill(r[0], 24) for r in rows], fontsize=9)
        ax.set_ylim(len(rows)-.5, -2)
        ax.legend(loc="upper right", frameon=False)
    else:
        raise ValueError("Unsupported declared panel kind: "+kind)


def main():
    spec = read("asset.csv").iloc[0].to_dict()
    panels = read("panels.csv").fillna("")
    sources = {}
    for filename in sorted(panels.source.unique()):
        sources[filename] = read(filename)
    # These are underlying local observations retained with derived plotting data.
    for name in csv_logical_names(ROOT, "observations_", READS):
        if name not in sources:
            read(name)
    selected = []; statuses = []
    for row in panels.to_dict("records"):
        data = sources[row["source"]]
        if row.get("filter_column"):
            data = data.loc[data[row["filter_column"]].astype(str).eq(str(row["filter_value"]))]
        valid = eligible(data, row)
        disposition = "RENDERED" if len(valid) else "UNAVAILABLE_NO_FINITE_OBSERVATIONS"
        statuses.append(dict(panel_id=row["panel_id"], title=row["title"], source=row["source"],
            input_rows=len(data), plotted_rows=len(valid), excluded_nonfinite_or_display_rows=len(data)-len(valid), status=disposition,
            note=row.get("note", "")))
        if len(valid):
            selected.append((row, valid))
    outputs = []
    for row, data in selected:
        height = max(6.5, .36*len(data)+2.3) if row["kind"] == "bar" else 6.5
        fig, ax = plt.subplots(figsize=(8.8, height))
        draw(ax, data, row)
        fig.subplots_adjust(left=.31 if row["kind"] in {"bar", "gain_loss"} else .12,
                            right=.96, bottom=.2, top=.91)
        note = textwrap.fill(row.get("note", ""), 112, break_long_words=False)
        if note.count("\n") > 4:
            raise ValueError("Panel note exceeds readable five-line area")
        fig.text(.035, .035, note, va="bottom", fontsize=9, linespacing=1.25)
        stem = "panel_"+row["panel_id"]
        for extension in ["png", "pdf"]:
            filename = stem+"."+extension
            options = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
            fig.savefig(ROOT/filename, dpi=200, **options); outputs.append(filename)
        plt.close(fig)
    # The combined figure is an additional convenience; manuscript panels retain
    # their independent page scale instead of shrinking a many-panel composite.
    if selected:
        cols = 2 if len(selected) > 1 else 1
        rows = math.ceil(len(selected)/cols)
        fig, axes = plt.subplots(rows, cols, figsize=(8.8*cols, 6.5*rows), squeeze=False)
        for ax, (row, data) in zip(axes.flat, selected):
            draw(ax, data, row)
        for ax in list(axes.flat)[len(selected):]:
            ax.remove()
        fig.suptitle(spec["title"], x=.03, ha="left", fontsize=15)
        fig.subplots_adjust(left=.16, right=.96, bottom=.09, top=.93, hspace=.4, wspace=.45)
        fig.text(.025, .02, textwrap.fill(spec["caption"], 110*cols), fontsize=10, va="bottom")
        for extension in ["png", "pdf"]:
            filename = spec["asset_id"]+"."+extension
            options = {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {}
            fig.savefig(ROOT/filename, dpi=160, **options); outputs.append(filename)
        plt.close(fig)
    pd.DataFrame(statuses).to_csv(ROOT/"panel_status.csv", index=False)
    pd.DataFrame(list(READS.values())).sort_values("file").to_csv(ROOT/"read_csv_hashes.csv", index=False)
    result = {"status": "RENDERED" if selected else "UNAVAILABLE_NO_RENDERABLE_OBSERVATIONS",
        "available_panels": len(selected), "declared_panels": len(panels), "outputs": outputs,
        "visual_qa": "PENDING_ACTUAL_PANEL_INSPECTION" if selected else "NOT_APPLICABLE_NO_RENDER"}
    (ROOT/"RENDER_RESULT.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
