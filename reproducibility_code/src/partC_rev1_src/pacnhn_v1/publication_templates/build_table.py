"""Reproduce this table from the CSV files beside this script only."""
from pathlib import Path
import csv
import hashlib
import json
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parent
READS = {}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})


# EMBED_LOCAL_LOGICAL_CSV_SUPPORT_HERE


def read(name):
    return read_logical_csv(ROOT, name, READS, dtype=str)


def escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_\allowbreak{}", "/": r"/\allowbreak{}",
        "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
        "\u2013": "--", "\u2014": "---"}
    # Break literal tokens before escaping, never inside a generated TeX command.
    # A zero-width penalty permits wrapping a SHA/identifier without changing its
    # visible characters. Existing underscore/slash opportunities reset the run.
    parts = []
    run_length = 0
    for c in str(value):
        if c.isspace() or c in "_/":
            run_length = 0
        else:
            if run_length == 12:
                parts.append(r"\allowbreak{}")
                run_length = 0
            run_length += 1
        parts.append(replacements.get(c, c))
    return "".join(parts)


def wrapped(value, width, renderer, font):
    """Greedy display wrapping in pixels, including hard breaks in long IDs.

    Data/CSV strings are untouched. Explicit newlines are respected; ordinary
    inter-word whitespace is rendered as a space. No character is elided.
    """
    if not math.isfinite(width) or width <= 0:
        raise ValueError("Invalid text wrapping width")
    def fits(text):
        return renderer.get_text_width_height_descent(text, font, ismath=False)[0] <= width
    lines = []
    for paragraph in str(value).split("\n"):
        current = ""
        for word in paragraph.split():
            candidate = current+" "+word if current else word
            if fits(candidate):
                current = candidate
                continue
            if current:
                lines.append(current); current = ""
            while word and not fits(word):
                # Binary search a visible prefix; never truncate an identifier.
                lo, hi = 1, len(word)
                if not fits(word[:1]):
                    raise ValueError("A single glyph exceeds the usable cell width")
                while lo < hi:
                    mid = (lo+hi+1)//2
                    if fits(word[:mid]): lo = mid
                    else: hi = mid-1
                lines.append(word[:lo]); word = word[lo:]
            current = word
        lines.append(current)
    return "\n".join(lines)


def assert_text_inside_cells(fig, table, tolerance_pixels=1.0):
    """Reject a clipped header/body before publishing any page image."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for key, cell in table.get_celld().items():
        text = cell.get_text()
        if not text.get_text():
            continue
        outer = cell.get_window_extent(renderer)
        inner = text.get_window_extent(renderer)
        if (inner.x0 < outer.x0-tolerance_pixels or inner.x1 > outer.x1+tolerance_pixels
                or inner.y0 < outer.y0-tolerance_pixels or inner.y1 > outer.y1+tolerance_pixels):
            raise ValueError(f"Rendered table text exceeds cell {key}: {text.get_text()!r}")


def tex_table(frame, columns, spec):
    # The main manuscript loads these standard packages. Repeated captions use
    # no new caption/label, avoiding duplicate hyperlink destinations.
    total = sum(float(v) for v in columns.width)
    # Landscape letter with 0.7-inch margins: 9.6 inches total, including
    # the 8 pt of padding per column. Keep a small rounding allowance.
    available = 9.5 - 8*len(columns)/72.27
    widths = [available*float(v)/total for v in columns.width]
    fmt = "".join(r">{\raggedright\arraybackslash}p{"+f"{w:.3f}"+"in}" for w in widths)
    header = " & ".join(r"\textbf{"+escape(v)+"}" for v in columns.label) + r" \\"
    lines = [r"\begin{landscape}", r"\begingroup\small", r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.18}", r"\begin{longtable}{"+fmt+"}",
        r"\caption{"+escape(spec["title"])+r"}\label{tab:"+spec["asset_id"]+r"}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{"+str(len(columns))+r"}{l}{\small "+escape(spec["title"])+r" (continued)}\\",
        r"\toprule", header, r"\midrule", r"\endhead", r"\midrule",
        r"\multicolumn{"+str(len(columns))+r"}{r}{\small Continued on next page}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot"]
    for row in frame[columns.column].itertuples(index=False, name=None):
        lines.append(" & ".join(escape(v) for v in row)+r" \\")
    lines.extend([r"\end{longtable}", r"\noindent\footnotesize "+escape(spec["caption"]),
                  r"\par\endgroup", r"\end{landscape}"])
    (ROOT/"table.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    spec = read("asset.csv").iloc[0].to_dict()
    columns = read("columns.csv")
    data = read("table.csv")
    # Keep the exact underlying observations in the reproducibility read audit.
    for name in csv_logical_names(ROOT, "observations_", READS):
        read(name)
    if not len(data) or not len(columns) or set(columns.column)-set(data):
        raise ValueError("Table must contain explicit observations or unavailable dispositions")
    weights = columns.width.astype(float).to_numpy()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid table column widths")
    weights /= weights.sum()
    dpi = 180
    renderer = RendererAgg(round(13.2*dpi), round(8.6*dpi), dpi)
    normal = FontProperties(family="DejaVu Sans", size=9, weight="normal")
    bold = FontProperties(family="DejaVu Sans", size=9, weight="bold")
    # Cell.PAD is .1 per side. Use .78, leaving a further two-percent margin.
    widths = 13.2*dpi*.95*weights*.78
    formatted = [[wrapped(v, width, renderer, normal) for v, width in zip(row, widths)]
                 for row in data[columns.column].itertuples(index=False, name=None)]
    heights = [max(1, max(cell.count("\n")+1 for cell in row)) for row in formatted]
    headers = [wrapped(x, width, renderer, bold) for x, width in zip(columns.label, widths)]
    header_lines = max(x.count("\n")+1 for x in headers)
    header_height = .035*header_lines+.018
    # Split by actual cell heights, preserving every data row at 9 pt.
    chunks = []; start = 0; occupied = 0
    for i, count in enumerate(heights):
        height = .025*count+.013
        if height+header_height > .96:
            raise ValueError("A table row exceeds one page; shorten display text and retain full observations")
        if i > start and occupied+height+header_height > .96:
            chunks.append((start, i)); start = i; occupied = 0
        occupied += height
    chunks.append((start, len(data)))
    caption = wrapped(spec["caption"], 13.2*dpi*.94, renderer, normal)
    if caption.count("\n") > 4:
        raise ValueError("Table caption exceeds its readable five-line area")
    outputs = []
    with PdfPages(ROOT/"table.pdf", metadata={"Title": spec["title"], "CreationDate": None, "ModDate": None}) as pdf:
        for page, (a, b) in enumerate(chunks, 1):
            fig = plt.figure(figsize=(13.2, 8.6), dpi=dpi); ax = fig.add_axes([.025, .17, .95, .69]); ax.axis("off")
            fig.text(.03, .94, spec["title"], fontsize=13, weight="bold", va="top")
            if len(chunks) > 1:
                fig.text(.97, .94, f"Page {page}/{len(chunks)}", ha="right", va="top", fontsize=9)
            cells = formatted[a:b]
            table = ax.table(cellText=cells, colLabels=headers,
                colWidths=weights, loc="upper center", cellLoc="left", colLoc="left")
            table.auto_set_font_size(False); table.set_fontsize(9)
            for (row, col), cell in table.get_celld().items():
                cell.set_edgecolor("#ccd1d8"); cell.set_linewidth(.45)
                if row == 0:
                    cell.set_facecolor("#e6edf4"); cell.set_text_props(weight="bold")
                    cell.set_height(header_height)
                else:
                    cell.set_facecolor("#f6f8fa" if row % 2 == 0 else "white")
                    cell.set_height(.025*heights[a+row-1]+.013)
            fig.text(.03, .12, caption, va="top", fontsize=9, linespacing=1.3)
            fig.text(.03, .025, "N/A = unavailable or undefined; the status and note give the reason. No missing measurement is replaced by zero.", fontsize=8)
            filename = "table.png" if page == 1 else f"table_page_{page:02d}.png"
            assert_text_inside_cells(fig, table)
            fig.savefig(ROOT/filename, dpi=dpi, facecolor="white")
            pdf.savefig(fig); plt.close(fig); outputs.append(filename)
    tex_table(data, columns, spec)
    pd.DataFrame(list(READS.values())).sort_values("file").to_csv(ROOT/"read_csv_hashes.csv", index=False)
    result = {"status": "RENDERED", "table_rows": len(data), "pages": len(chunks),
        "outputs": ["table.pdf", "table.tex"]+outputs, "visual_qa": "PENDING_ACTUAL_PAGE_INSPECTION"}
    (ROOT/"RENDER_RESULT.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
