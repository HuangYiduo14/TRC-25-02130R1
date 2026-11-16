"""Tables and individual map panels for the measured I-95 FFTT closure."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


POLICY = "fftt_const_approximation"
CONUS_BOUNDS = (-124.77, -66.95, 24.52, 49.38)
PNG_DPI = 400
MIN_FONT_POINTS = 8.0
# Current elsarticle preprint/11 pt layout inherits article's 345 pt width.
ASSUMED_MANUSCRIPT_TEXT_WIDTH_INCHES = 345.0 / 72.27
MANUSCRIPT_WIDTH_FRACTION = 0.98

plt.rcParams.update(
    {
        "font.size": MIN_FONT_POINTS,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 10,
    }
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    source_width = float(fig.get_figwidth())
    source_height = float(fig.get_figheight())
    target_width = ASSUMED_MANUSCRIPT_TEXT_WIDTH_INCHES * MANUSCRIPT_WIDTH_FRACTION
    scale = target_width / source_width
    fig.set_size_inches(target_width, source_height * scale, forward=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.canvas.draw()
    sizes: list[float] = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_text():
            continue
        if float(text.get_fontsize()) < MIN_FONT_POINTS:
            text.set_fontsize(MIN_FONT_POINTS)
        sizes.append(float(text.get_fontsize()))
    minimum_font = min(sizes) if sizes else MIN_FONT_POINTS
    fig.savefig(stem.with_suffix(".pdf"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    _write_json(
        stem.with_name(f"{stem.name}_render_metadata.json"),
        {
            "figure_stem": stem.name,
            "png_dpi": PNG_DPI,
            "minimum_rendered_font_points": minimum_font,
            "figure_width_inches": float(fig.get_figwidth()),
            "figure_height_inches": float(fig.get_figheight()),
            "original_figure_width_inches": source_width,
            "original_figure_height_inches": source_height,
            "manuscript_width_fraction": MANUSCRIPT_WIDTH_FRACTION,
            "assumed_manuscript_text_width_inches": ASSUMED_MANUSCRIPT_TEXT_WIDTH_INCHES,
            "estimated_manuscript_minimum_font_points": minimum_font,
            "pdf_vector_output": True,
            "pdf_rasterized_layer_dpi": PNG_DPI,
            "dense_line_layers_rasterized_inside_pdf": True,
        },
    )
    plt.close(fig)


def _map_axes(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(CONUS_BOUNDS[0], CONUS_BOUNDS[1])
    ax.set_ylim(CONUS_BOUNDS[2], CONUS_BOUNDS[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title(title)
    ax.grid(True, color="0.91", linewidth=0.4, zorder=0)


def _update_map_extent_table(
    numerical_root: Path, stem_counts: dict[str, int]
) -> None:
    path = numerical_root / "appendix" / "tables" / "partC_rev1_map_extent_validation.csv"
    prior = pd.read_csv(path) if path.exists() else pd.DataFrame()
    stems = list(stem_counts)
    if not prior.empty:
        prior = prior.loc[~prior["map_id"].astype(str).isin(stems)].copy()
    rows = pd.DataFrame(
        [
            {
                "map_id": stem,
                "view": "mainland",
                "longitude_min": CONUS_BOUNDS[0],
                "longitude_max": CONUS_BOUNDS[1],
                "latitude_min": CONUS_BOUNDS[2],
                "latitude_max": CONUS_BOUNDS[3],
                "plot_crs": "EPSG:4326",
                "source_crs": "EPSG:4269",
                "plotted_feature_count": int(stem_counts[stem]),
                "inside_mainland_bounds": True,
                "exact_mainland_view": True,
            }
            for stem in stems
        ]
    )
    result = pd.concat([prior, rows], ignore_index=True)
    result.sort_values("map_id").to_csv(path, index=False, lineterminator="\n")


def _terminal_node_uid(terminal_uid: Any) -> str:
    return str(terminal_uid).rsplit("|", 1)[-1]


def _flow_style(
    values: pd.Series,
    *,
    reference_values: pd.Series | None = None,
    width_min: float,
    width_max: float,
    alpha_min: float,
    alpha_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.maximum(pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy(), 0.0)
    logged = np.log1p(value)
    if reference_values is None:
        reference_logged = logged
    else:
        reference_logged = np.log1p(
            np.maximum(
                pd.to_numeric(reference_values, errors="coerce")
                .fillna(0.0)
                .to_numpy(),
                0.0,
            )
        )
    positive = reference_logged[reference_logged > 0]
    if len(positive):
        lower = float(np.quantile(positive, 0.05))
        upper = float(np.quantile(positive, 0.99))
    else:
        lower, upper = 0.0, 1.0
    if upper <= lower:
        upper = lower + 1.0
    normalized = np.clip((logged - lower) / (upper - lower), 0.0, 1.0)
    return (
        width_min + (width_max - width_min) * normalized,
        alpha_min + (alpha_max - alpha_min) * normalized,
    )


def _geometry_segments(frame) -> tuple[list[np.ndarray], np.ndarray]:
    segments: list[np.ndarray] = []
    positions: list[int] = []
    for pos, geometry in enumerate(frame.geometry):
        if geometry is None or geometry.is_empty:
            continue
        if geometry.geom_type == "LineString":
            segments.append(np.asarray(geometry.coords, dtype=np.float64))
            positions.append(pos)
        elif geometry.geom_type == "MultiLineString":
            for part in geometry.geoms:
                segments.append(np.asarray(part.coords, dtype=np.float64))
                positions.append(pos)
    return segments, np.asarray(positions, dtype=np.int64)


def _add_lines(
    ax: plt.Axes,
    frame,
    *,
    color: str | pd.Series,
    linewidth: float | pd.Series,
    alpha: float | pd.Series,
    zorder: int,
) -> None:
    segments, positions = _geometry_segments(frame)
    if not segments:
        return
    if isinstance(linewidth, pd.Series):
        widths = pd.to_numeric(linewidth, errors="coerce").fillna(0.1).to_numpy()[positions]
    else:
        widths = np.full(len(segments), float(linewidth), dtype=np.float64)
    if isinstance(alpha, pd.Series):
        alphas = pd.to_numeric(alpha, errors="coerce").fillna(0.2).to_numpy()[positions]
    else:
        alphas = np.full(len(segments), float(alpha), dtype=np.float64)
    if isinstance(color, pd.Series):
        base_colors = color.astype(str).to_numpy()[positions]
    else:
        base_colors = np.full(len(segments), str(color), dtype=object)
    colors = np.asarray([to_rgba(value) for value in base_colors], dtype=np.float64)
    colors[:, 3] = alphas
    collection = LineCollection(
        segments,
        colors=colors,
        linewidths=widths,
        zorder=zorder,
        rasterized=True,
    )
    ax.add_collection(collection)


def _flow_map_data(flow: pd.DataFrame, backbone_geo, node_geo):
    import geopandas as gpd
    from shapely.geometry import LineString, box

    bounds_geometry = box(
        CONUS_BOUNDS[0], CONUS_BOUNDS[2], CONUS_BOUNDS[1], CONUS_BOUNDS[3]
    )
    backbone_flow = flow.loc[
        flow["abstract_arc_kind"].astype(str).eq("backbone")
        & pd.to_numeric(flow["flow"], errors="coerce").fillna(0.0).gt(1e-12)
    ].copy()
    backbone_flow["directed_arc_id"] = backbone_flow["directed_arc_id"].astype(str)
    backbone = backbone_geo.merge(
        backbone_flow[
            [
                "abstract_arc_idx",
                "directed_arc_id",
                "flow",
                "cost",
            ]
        ],
        on="directed_arc_id",
        how="inner",
        validate="one_to_one",
    )
    backbone = backbone.loc[backbone.geometry.intersects(bounds_geometry)].copy()
    backbone["network_kind"] = "backbone"
    backbone["source_node_uid"] = backbone["from_node_uid"].astype(str)
    backbone["target_node_uid"] = backbone["to_node_uid"].astype(str)

    nodes = node_geo[["partC_rev1_node_uid", "geometry"]].copy()
    nodes["partC_rev1_node_uid"] = nodes["partC_rev1_node_uid"].astype(str)
    point_by_uid = dict(zip(nodes["partC_rev1_node_uid"], nodes.geometry))
    regional_flow = flow.loc[
        flow["abstract_arc_kind"].astype(str).eq("regional_transfer")
        & pd.to_numeric(flow["flow"], errors="coerce").fillna(0.0).gt(1e-12)
    ].copy()
    regional_flow["source_node_uid"] = regional_flow["source_terminal_uid"].map(
        _terminal_node_uid
    )
    regional_flow["target_node_uid"] = regional_flow["target_terminal_uid"].map(
        _terminal_node_uid
    )
    regional_geometry = []
    for row in regional_flow.itertuples(index=False):
        source = point_by_uid.get(str(row.source_node_uid))
        target = point_by_uid.get(str(row.target_node_uid))
        if source is None or target is None:
            regional_geometry.append(None)
        else:
            regional_geometry.append(LineString([source, target]))
    regional = gpd.GeoDataFrame(
        regional_flow,
        geometry=regional_geometry,
        crs=node_geo.crs,
    )
    regional = regional.loc[
        regional.geometry.notna() & regional.geometry.intersects(bounds_geometry)
    ].copy()
    regional["network_kind"] = "regional"
    return backbone, regional


def _styled_flow_layers(
    backbone,
    regional,
    *,
    backbone_reference: pd.Series | None = None,
    regional_reference: pd.Series | None = None,
):
    backbone_width, _ = _flow_style(
        backbone["flow"],
        reference_values=backbone_reference,
        width_min=0.05,
        width_max=1.20,
        alpha_min=0.08,
        alpha_max=0.78,
    )
    regional_width, _ = _flow_style(
        regional["flow"],
        reference_values=regional_reference,
        width_min=0.02,
        width_max=1.05,
        alpha_min=0.02,
        alpha_max=0.52,
    )
    backbone = backbone.copy()
    regional = regional.copy()
    backbone["line_color"] = "#1565C0"
    backbone["line_width_points"] = backbone_width
    backbone["line_alpha"] = 0.5
    regional["line_color"] = "#EF6C00"
    regional["line_width_points"] = regional_width
    regional["line_alpha"] = 0.5
    return backbone, regional


def _export_map_csv(path: Path, frames: list[pd.DataFrame], figure_stem: str) -> None:
    parts: list[pd.DataFrame] = []
    columns = [
        "abstract_arc_idx",
        "directed_arc_id",
        "partC_rev1_link_uid",
        "region_uuid",
        "region_leg_id",
        "source_node_uid",
        "target_node_uid",
        "network_kind",
        "flow",
        "cost",
        "flow_change",
        "geometry_wkt",
        "figure_stem",
        "plot_crs",
    ]
    for frame in frames:
        data = pd.DataFrame(frame.drop(columns="geometry", errors="ignore")).copy()
        data["geometry_wkt"] = frame.geometry.to_wkt()
        data["figure_stem"] = figure_stem
        data["plot_crs"] = "EPSG:4326"
        for column in columns:
            if column not in data:
                data[column] = ""
        parts.append(data[columns])
    pd.concat(parts, ignore_index=True).to_csv(path, index=False, lineterminator="\n")


def _plot_hybrid_flow(
    *,
    backbone,
    regional,
    context,
    figure_stem: str,
    title: str,
    figure_dir: Path,
    plot_dir: Path,
    backbone_reference: pd.Series,
    regional_reference: pd.Series,
) -> None:
    backbone, regional = _styled_flow_layers(
        backbone,
        regional,
        backbone_reference=backbone_reference,
        regional_reference=regional_reference,
    )
    _export_map_csv(plot_dir / f"{figure_stem}.csv", [backbone, regional], figure_stem)
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    _add_lines(
        ax,
        context,
        color="0.78",
        linewidth=0.025,
        alpha=0.22,
        zorder=1,
    )
    _add_lines(
        ax,
        regional,
        color=regional["line_color"],
        linewidth=regional["line_width_points"],
        alpha=regional["line_alpha"],
        zorder=2,
    )
    _add_lines(
        ax,
        backbone,
        color=backbone["line_color"],
        linewidth=backbone["line_width_points"],
        alpha=backbone["line_alpha"],
        zorder=3,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#1565C0", linewidth=1.8, label="Backbone"),
            Line2D(
                [0],
                [0],
                color="#EF6C00",
                linewidth=1.8,
                label="Regional $i\\to j$",
            ),
        ],
        loc="lower left",
        frameon=True,
        handlelength=1.5,
        labelspacing=0.3,
        borderpad=0.3,
    )
    _map_axes(ax, title)
    fig.tight_layout()
    _save_figure(fig, figure_dir / figure_stem)


def _scenario_metrics(
    *,
    open_flow: pd.DataFrame,
    closed_flow: pd.DataFrame,
    open_summary: pd.Series,
    closed_summary: pd.Series,
    selected_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    columns = [
        "abstract_arc_idx",
        "directed_arc_id",
        "partC_rev1_link_uid",
        "abstract_arc_kind",
        "region_uuid",
        "region_leg_id",
        "length_miles",
        "flow",
        "cost",
    ]
    merged = open_flow[columns].merge(
        closed_flow[columns],
        on=[
            "abstract_arc_idx",
            "directed_arc_id",
            "partC_rev1_link_uid",
            "abstract_arc_kind",
            "region_uuid",
            "region_leg_id",
            "length_miles",
        ],
        suffixes=("_open", "_closed"),
        validate="one_to_one",
    )
    merged["flow_change"] = merged["flow_closed"] - merged["flow_open"]
    merged["absolute_flow_change"] = merged["flow_change"].abs()
    merged["is_i95_closed_arc"] = merged["directed_arc_id"].astype(str).isin(selected_ids)

    rows = []
    for state, summary, frame in (
        ("open", open_summary, open_flow),
        ("fully_closed", closed_summary, closed_flow),
    ):
        flow = pd.to_numeric(frame["flow"], errors="coerce").fillna(0.0)
        cost = pd.to_numeric(frame["cost"], errors="coerce").fillna(0.0)
        length = pd.to_numeric(frame["length_miles"], errors="coerce").fillna(0.0)
        i95 = frame["directed_arc_id"].astype(str).isin(selected_ids)
        backbone = frame["abstract_arc_kind"].astype(str).eq("backbone")
        regional = frame["abstract_arc_kind"].astype(str).eq("regional_transfer")
        rows.append(
            {
                "state": state,
                "policy": POLICY,
                "closure_fraction": 0.0 if state == "open" else 1.0,
                "global_runtime_sec": float(summary["global_runtime_sec"]),
                "iteration_count": int(summary["iteration_count"]),
                "final_gap_or_residual": float(summary["final_gap_or_residual"]),
                "stop_reason": str(summary["stop_reason"]),
                "assigned_demand": float(summary["assigned_demand"]),
                "abstract_unreachable_demand": float(
                    summary["abstract_unreachable_demand"]
                ),
                "total_arc_flow": float(flow.sum()),
                "total_system_flow_minutes": float(np.dot(flow, cost)),
                "backbone_arc_flow": float(flow.loc[backbone].sum()),
                "backbone_system_flow_minutes": float(
                    np.dot(flow.loc[backbone], cost.loc[backbone])
                ),
                "backbone_flow_ton_miles": float(
                    np.dot(flow.loc[backbone], length.loc[backbone])
                ),
                "regional_transfer_arc_flow": float(flow.loc[regional].sum()),
                "i95_arc_flow": float(flow.loc[i95].sum()),
                "i95_flow_minutes": float(np.dot(flow.loc[i95], cost.loc[i95])),
            }
        )
    scenario = pd.DataFrame(rows)

    open_unreachable = float(
        scenario.loc[scenario["state"].eq("open"), "abstract_unreachable_demand"].iloc[0]
    )
    closed_unreachable = float(
        scenario.loc[
            scenario["state"].eq("fully_closed"), "abstract_unreachable_demand"
        ].iloc[0]
    )
    comparisons: list[dict[str, Any]] = []

    def add(metric: str, value: float, unit: str, formula: str) -> None:
        comparisons.append(
            {
                "metric": metric,
                "value": value,
                "unit": unit,
                "formula_or_definition": formula,
            }
        )

    add(
        "additional_unreachable_demand",
        closed_unreachable - open_unreachable,
        "thousand tons/year",
        "closed abstract unreachable demand - open abstract unreachable demand",
    )
    for kind, group in merged.groupby("abstract_arc_kind", sort=True):
        denom = max(float(group["flow_open"].abs().sum()), 1e-12)
        add(
            f"{kind}_flow_WMAPE_closed_vs_open",
            float(group["absolute_flow_change"].sum()) / denom,
            "dimensionless",
            "sum(abs(closed-open))/sum(abs(open))",
        )
        add(
            f"{kind}_flow_L1_change",
            float(group["absolute_flow_change"].sum()),
            "thousand tons/year across arcs",
            "sum(abs(closed-open))",
        )
        add(
            f"{kind}_net_flow_change",
            float(group["flow_change"].sum()),
            "thousand tons/year across arcs",
            "sum(closed-open)",
        )
    selected = merged.loc[merged["is_i95_closed_arc"]]
    open_i95 = float(selected["flow_open"].sum())
    closed_i95 = float(selected["flow_closed"].sum())
    add("open_i95_arc_flow", open_i95, "thousand tons/year across arcs", "sum open flow on audited class-11 I-95 arcs")
    add("closed_i95_arc_flow", closed_i95, "thousand tons/year across arcs", "sum fully-closed flow on audited class-11 I-95 arcs")
    add(
        "i95_arc_flow_reduction_fraction",
        1.0 - closed_i95 / max(open_i95, 1e-12),
        "dimensionless",
        "1 - closed I-95 arc flow/open I-95 arc flow",
    )
    add(
        "total_system_flow_minutes_change",
        float(
            scenario.loc[
                scenario["state"].eq("fully_closed"), "total_system_flow_minutes"
            ].iloc[0]
            - scenario.loc[
                scenario["state"].eq("open"), "total_system_flow_minutes"
            ].iloc[0]
        ),
        "flow-minutes",
        "closed sum(flow*cost) - open sum(flow*cost)",
    )
    for column, metric_name, unit in (
        (
            "backbone_system_flow_minutes",
            "backbone_system_flow_minutes_change",
            "flow-minutes",
        ),
        (
            "backbone_flow_ton_miles",
            "backbone_flow_ton_miles_change",
            "thousand ton-miles/year",
        ),
        (
            "backbone_arc_flow",
            "backbone_arc_flow_change",
            "thousand tons/year across arcs",
        ),
    ):
        open_value = float(scenario.loc[scenario["state"].eq("open"), column].iloc[0])
        closed_value = float(
            scenario.loc[scenario["state"].eq("fully_closed"), column].iloc[0]
        )
        add(
            metric_name,
            closed_value - open_value,
            unit,
            f"fully closed {column} - open {column}",
        )
    comparison = pd.DataFrame(comparisons)
    return scenario, comparison, merged


def build_i95_report(
    *,
    recovery_root: str | Path,
    scenario_root: str | Path,
    policy_gpkg: str | Path,
    partition_gpkg: str | Path,
    numerical_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """Build the completed Section 5.5 tables, data, and four Figure 8 panels."""

    import geopandas as gpd
    import pyogrio
    from shapely.geometry import box

    recovery = Path(recovery_root).resolve()
    scenario = Path(scenario_root).resolve()
    numerical = Path(numerical_root).resolve()
    output = Path(output_root).resolve()
    section = numerical / "sec5_5"
    for child in ("source_extracts", "plot_data", "figures", "tables"):
        (section / child).mkdir(parents=True, exist_ok=True)
    for obsolete in (
        section / "figures" / "partC_rev1_figure8_i95_placeholder.pdf",
        section / "figures" / "partC_rev1_figure8_i95_placeholder.png",
        section
        / "figures"
        / "partC_rev1_figure8_i95_placeholder_render_metadata.json",
        section
        / "plot_data"
        / "partC_rev1_i95_baseline_corridor_plot_data.csv",
        section
        / "partC_rev1_source_extracts"
        / "partC_rev1_i95_selected_links.csv",
        section
        / "partC_rev1_source_extracts"
        / "partC_rev1_i95_selected_links.geojson",
        section
        / "partC_rev1_source_extracts"
        / "partC_rev1_i95_scenario_config.json",
    ):
        if obsolete.exists():
            obsolete.unlink()

    run_metadata = json.loads(
        (scenario / "partC_rev1_i95_fftt_closure_run_metadata.json").read_text(
            encoding="utf-8"
        )
    )
    audit = run_metadata["selection_audit"]
    selected_path = Path(audit["selected_csv"])
    excluded_path = Path(audit["excluded_csv"])
    for source in (
        selected_path,
        excluded_path,
        Path(audit["audit_json"]),
        scenario / "partC_rev1_i95_fftt_closure_run_metadata.json",
        scenario / "partC_rev1_i95_full_closure_fftt.log",
        scenario / "partC_rev1_i95_full_closure_fftt_progress.jsonl",
    ):
        shutil.copy2(source, section / "source_extracts" / source.name)

    open_flow_path = recovery / f"partC_rev1_abstract_global_flow_{POLICY}.parquet"
    closed_flow_path = Path(run_metadata["flow_path"])
    open_summary_path = recovery / "partC_rev1_abstract_global_summary.csv"
    closed_summary_path = Path(run_metadata["summary_csv"])
    open_history_path = (
        recovery / f"partC_rev1_abstract_global_history_{POLICY}.csv"
    )
    closed_history_path = Path(run_metadata["history_path"])
    open_flow = pd.read_parquet(open_flow_path)
    closed_flow = pd.read_parquet(closed_flow_path)
    open_summary = pd.read_csv(open_summary_path).loc[
        lambda frame: frame["policy"].eq(POLICY)
    ].iloc[0]
    closed_summary = pd.read_csv(closed_summary_path).loc[
        lambda frame: frame["policy"].eq(POLICY)
    ].iloc[0]
    selected = pd.read_csv(selected_path)
    selected_ids = set(selected["directed_arc_id"].astype(str))
    scenario_table, comparison, merged = _scenario_metrics(
        open_flow=open_flow,
        closed_flow=closed_flow,
        open_summary=open_summary,
        closed_summary=closed_summary,
        selected_ids=selected_ids,
    )
    scenario_table.insert(
        1,
        "scenario_label",
        "scenario demonstration; not observed-event validation",
    )
    scenario_table["affected_class_code"] = 11
    scenario_table["closed_directed_arc_count"] = np.where(
        scenario_table["state"].eq("fully_closed"),
        int(audit["selected_class11_directed_arc_count"]),
        0,
    )
    scenario_table["closed_unique_link_miles"] = np.where(
        scenario_table["state"].eq("fully_closed"),
        float(audit["selected_unique_miles_by_physical_link"]),
        0.0,
    )
    scenario_table["prohibitive_fftt_min"] = np.where(
        scenario_table["state"].eq("fully_closed"), 1.0e12, np.nan
    )
    scenario_table.to_csv(
        section / "tables" / "partC_rev1_i95_scenario_table.csv",
        index=False,
        lineterminator="\n",
    )
    comparison.to_csv(
        section / "tables" / "partC_rev1_i95_scenario_comparison.csv",
        index=False,
        lineterminator="\n",
    )
    flow_by_kind = (
        merged.groupby("abstract_arc_kind", as_index=False)
        .agg(
            arc_count=("abstract_arc_idx", "size"),
            open_flow=("flow_open", "sum"),
            fully_closed_flow=("flow_closed", "sum"),
            net_flow_change=("flow_change", "sum"),
            absolute_flow_change=("absolute_flow_change", "sum"),
        )
        .sort_values("abstract_arc_kind")
    )
    flow_by_kind["wmape_closed_vs_open"] = (
        flow_by_kind["absolute_flow_change"]
        / flow_by_kind["open_flow"].abs().clip(lower=1e-12)
    )
    flow_by_kind.to_csv(
        section / "tables" / "partC_rev1_i95_flow_change_by_arc_kind.csv",
        index=False,
        lineterminator="\n",
    )
    histories = pd.concat(
        [
            pd.read_csv(open_history_path).assign(state="open"),
            pd.read_csv(closed_history_path).assign(state="fully_closed"),
        ],
        ignore_index=True,
    )
    histories.to_csv(
        section / "tables" / "partC_rev1_i95_open_closed_convergence.csv",
        index=False,
        lineterminator="\n",
    )

    policy_columns = [
        "directed_arc_id",
        "partC_rev1_link_uid",
        "from_node_uid",
        "to_node_uid",
        "road_name",
        "sign_route",
        "route_type",
        "route_number",
        "class_code",
        "length_miles",
    ]
    policy_links = pyogrio.read_dataframe(
        policy_gpkg,
        layer="partC_rev1_recovery_backbone_policy_flows",
        columns=policy_columns,
    ).to_crs(4326)
    policy_links["directed_arc_id"] = policy_links["directed_arc_id"].astype(str)
    context_box = box(
        CONUS_BOUNDS[0], CONUS_BOUNDS[2], CONUS_BOUNDS[1], CONUS_BOUNDS[3]
    )
    context = policy_links.loc[policy_links.geometry.intersects(context_box)].copy()
    nodes = pyogrio.read_dataframe(
        partition_gpkg,
        layer="partC_rev1_backbone_partition_nodes",
        columns=["partC_rev1_node_uid"],
    ).to_crs(4326)

    diversion = merged.loc[merged["abstract_arc_kind"].eq("backbone")].copy()
    diversion["directed_arc_id"] = diversion["directed_arc_id"].astype(str)
    diversion = diversion.merge(
        pd.DataFrame(policy_links.drop(columns="geometry"))[
            [
                "directed_arc_id",
                "road_name",
                "sign_route",
                "route_type",
                "route_number",
                "class_code",
                "length_miles",
            ]
        ],
        on="directed_arc_id",
        how="left",
        validate="one_to_one",
    )
    top_gains = diversion.nlargest(25, "flow_change").assign(change_rank_group="gain")
    top_losses = diversion.nsmallest(25, "flow_change").assign(
        change_rank_group="loss"
    )
    pd.concat([top_gains, top_losses], ignore_index=True).to_csv(
        section / "tables" / "partC_rev1_i95_top_backbone_diversions.csv",
        index=False,
        lineterminator="\n",
    )
    regional_diversion = (
        merged.loc[merged["abstract_arc_kind"].eq("regional_transfer")]
        .groupby("region_uuid", as_index=False)
        .agg(
            open_flow=("flow_open", "sum"),
            fully_closed_flow=("flow_closed", "sum"),
            net_flow_change=("flow_change", "sum"),
            absolute_flow_change=("absolute_flow_change", "sum"),
        )
        .sort_values("absolute_flow_change", ascending=False)
    )
    regional_diversion.head(50).to_csv(
        section / "tables" / "partC_rev1_i95_top_regional_flow_changes.csv",
        index=False,
        lineterminator="\n",
    )

    # Figure 8(a): exact selected population and the two excluded class-21 arcs.
    stem_a = "partC_rev1_fig8_suba_i95_class11_closure"
    selected_geo = policy_links.loc[
        policy_links["directed_arc_id"].isin(selected_ids)
    ].copy()
    excluded_ids = set(pd.read_csv(excluded_path)["directed_arc_id"].astype(str))
    excluded_geo = policy_links.loc[
        policy_links["directed_arc_id"].isin(excluded_ids)
    ].copy()
    selected_plot = pd.DataFrame(selected_geo.drop(columns="geometry"))
    selected_plot["selection_status"] = "closed: exact I-95 and class 11"
    selected_plot["line_color"] = "#C62828"
    selected_plot["line_width_points"] = 1.05
    selected_plot["line_alpha"] = 0.90
    selected_plot["geometry_wkt"] = selected_geo.geometry.to_wkt()
    selected_plot["figure_stem"] = stem_a
    selected_plot["plot_crs"] = "EPSG:4326"
    excluded_plot = pd.DataFrame(excluded_geo.drop(columns="geometry"))
    excluded_plot["selection_status"] = "excluded: class 21 nonmainline/ramp"
    excluded_plot["line_color"] = "#6A1B9A"
    excluded_plot["line_width_points"] = 1.35
    excluded_plot["line_alpha"] = 1.0
    excluded_plot["geometry_wkt"] = excluded_geo.geometry.to_wkt()
    excluded_plot["figure_stem"] = stem_a
    excluded_plot["plot_crs"] = "EPSG:4326"
    pd.concat([selected_plot, excluded_plot], ignore_index=True).to_csv(
        section / "plot_data" / f"{stem_a}.csv",
        index=False,
        lineterminator="\n",
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    _add_lines(
        ax, context, color="0.78", linewidth=0.025, alpha=0.25, zorder=1
    )
    _add_lines(
        ax, selected_geo, color="#C62828", linewidth=1.05, alpha=0.90, zorder=3
    )
    _add_lines(
        ax, excluded_geo, color="#6A1B9A", linewidth=1.35, alpha=1.0, zorder=4
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#C62828", linewidth=2.0, label="Closed class 11"),
            Line2D(
                [0],
                [0],
                color="#6A1B9A",
                linewidth=2.0,
                label="Excluded class 21",
            ),
        ],
        loc="lower left",
        frameon=True,
        handlelength=1.5,
        labelspacing=0.3,
        borderpad=0.3,
    )
    _map_axes(ax, "Audited class-11 I-95 closure")
    fig.tight_layout()
    _save_figure(fig, section / "figures" / stem_a)

    open_backbone, open_regional = _flow_map_data(open_flow, policy_links, nodes)
    closed_backbone, closed_regional = _flow_map_data(closed_flow, policy_links, nodes)
    shared_backbone_reference = pd.concat(
        [open_backbone["flow"], closed_backbone["flow"]], ignore_index=True
    )
    shared_regional_reference = pd.concat(
        [open_regional["flow"], closed_regional["flow"]], ignore_index=True
    )
    stem_b = "partC_rev1_fig8_subb_open_hybrid_flow"
    stem_c = "partC_rev1_fig8_subc_closed_hybrid_flow"
    _plot_hybrid_flow(
        backbone=open_backbone,
        regional=open_regional,
        context=context,
        figure_stem=stem_b,
        title="Open abstract/hybrid flow",
        figure_dir=section / "figures",
        plot_dir=section / "plot_data",
        backbone_reference=shared_backbone_reference,
        regional_reference=shared_regional_reference,
    )
    _plot_hybrid_flow(
        backbone=closed_backbone,
        regional=closed_regional,
        context=context,
        figure_stem=stem_c,
        title="Full-closure abstract/hybrid flow",
        figure_dir=section / "figures",
        plot_dir=section / "plot_data",
        backbone_reference=shared_backbone_reference,
        regional_reference=shared_regional_reference,
    )

    # Figure 8(d): signed changes, retaining backbone/regional color families.
    stem_d = "partC_rev1_fig8_subd_hybrid_flow_change"
    change_backbone = policy_links.merge(
        merged.loc[
            merged["abstract_arc_kind"].eq("backbone")
            & merged["absolute_flow_change"].gt(1e-12),
            [
                "abstract_arc_idx",
                "directed_arc_id",
                "region_uuid",
                "region_leg_id",
                "flow_open",
                "flow_closed",
                "flow_change",
                "absolute_flow_change",
            ],
        ].assign(directed_arc_id=lambda frame: frame["directed_arc_id"].astype(str)),
        on="directed_arc_id",
        how="inner",
        validate="one_to_one",
    )
    change_backbone = change_backbone.loc[
        change_backbone.geometry.intersects(context_box)
    ].copy()
    regional_template = closed_flow.copy()
    absolute_change_by_idx = merged.set_index("abstract_arc_idx")[
        "absolute_flow_change"
    ]
    regional_template["flow"] = regional_template["abstract_arc_idx"].map(
        absolute_change_by_idx
    )
    regional_template["cost"] = 0.0
    _, change_regional = _flow_map_data(regional_template, policy_links, nodes)
    regional_changes = merged.set_index("abstract_arc_idx")["flow_change"]
    change_regional["flow_change"] = change_regional["abstract_arc_idx"].map(
        regional_changes
    )
    change_regional["absolute_flow_change"] = change_regional["flow"]
    change_backbone_width, _ = _flow_style(
        change_backbone["absolute_flow_change"],
        width_min=0.07,
        width_max=1.45,
        alpha_min=0.12,
        alpha_max=0.85,
    )
    change_regional_width, _ = _flow_style(
        change_regional["absolute_flow_change"],
        width_min=0.11,
        width_max=1.80,
        alpha_min=0.16,
        alpha_max=0.88,
    )
    change_backbone["network_kind"] = "backbone"
    change_regional["network_kind"] = "regional"
    change_backbone["source_node_uid"] = change_backbone["from_node_uid"].astype(str)
    change_backbone["target_node_uid"] = change_backbone["to_node_uid"].astype(str)
    change_backbone["flow"] = change_backbone["absolute_flow_change"]
    change_backbone["cost"] = 0.0
    change_backbone["line_color"] = np.where(
        change_backbone["flow_change"].ge(0), "#0D47A1", "#64B5F6"
    )
    change_backbone["line_width_points"] = change_backbone_width
    change_backbone["line_alpha"] = 0.5
    change_regional["line_color"] = np.where(
        change_regional["flow_change"].ge(0), "#E65100", "#FFB74D"
    )
    change_regional["line_width_points"] = change_regional_width
    change_regional["line_alpha"] = 0.5
    _export_map_csv(
        section / "plot_data" / f"{stem_d}.csv",
        [change_backbone, change_regional],
        stem_d,
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    _add_lines(
        ax, context, color="0.78", linewidth=0.025, alpha=0.22, zorder=1
    )
    _add_lines(
        ax,
        change_regional,
        color=change_regional["line_color"],
        linewidth=change_regional["line_width_points"],
        alpha=change_regional["line_alpha"],
        zorder=2,
    )
    _add_lines(
        ax,
        change_backbone,
        color=change_backbone["line_color"],
        linewidth=change_backbone["line_width_points"],
        alpha=change_backbone["line_alpha"],
        zorder=3,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#0D47A1", linewidth=1.8, label="Backbone +"),
            Line2D([0], [0], color="#64B5F6", linewidth=1.8, label="Backbone -"),
            Line2D([0], [0], color="#E65100", linewidth=1.8, label="Regional +"),
            Line2D([0], [0], color="#FFB74D", linewidth=1.8, label="Regional -"),
        ],
        loc="lower left",
        frameon=True,
        ncol=2,
        handlelength=1.5,
        labelspacing=0.3,
        borderpad=0.3,
    )
    _map_axes(ax, "Flow change: full I-95 closure")
    fig.tight_layout()
    _save_figure(fig, section / "figures" / stem_d)

    _update_map_extent_table(
        numerical,
        {
            stem_a: len(context) + len(selected_geo) + len(excluded_geo),
            stem_b: len(context) + len(open_backbone) + len(open_regional),
            stem_c: len(context) + len(closed_backbone) + len(closed_regional),
            stem_d: len(context) + len(change_backbone) + len(change_regional),
        },
    )
    report = {
        "status": "complete",
        "states_compared": ["open", "fully_closed"],
        "capacity_loss_levels": [0.0, 1.0],
        "policy": POLICY,
        "selected_class11_directed_arc_count": audit[
            "selected_class11_directed_arc_count"
        ],
        "excluded_class21_directed_arc_count": audit[
            "excluded_nonclass11_directed_arc_count"
        ],
        "scenario_table": str(
            section / "tables" / "partC_rev1_i95_scenario_table.csv"
        ),
        "comparison_table": str(
            section / "tables" / "partC_rev1_i95_scenario_comparison.csv"
        ),
        "figure_stems": [stem_a, stem_b, stem_c, stem_d],
        "same_stem_csvs": [
            str(section / "plot_data" / f"{stem}.csv")
            for stem in (stem_a, stem_b, stem_c, stem_d)
        ],
        "open_flow_path": str(open_flow_path),
        "closed_flow_path": str(closed_flow_path),
    }
    _write_json(output / "partC_rev1_i95_completed_report.json", report)
    _write_json(section / "partC_rev1_i95_completed_report.json", report)
    return report


__all__ = ["build_i95_report"]
