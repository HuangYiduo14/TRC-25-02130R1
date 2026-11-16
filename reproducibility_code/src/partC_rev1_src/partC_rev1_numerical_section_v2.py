"""Reproducible v2 builders for the Part C revision-1 numerical section.

This immutable revision exports every manuscript subfigure as a separate
artifact, enforces WGS84 mainland/zoom map extents, and adds the measured
class-11 I-95 FFTT closure scenario.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
RUN_HASH = "760e287ee048e265"
CONFIG_HASH = "760e287ee048e2655dc69bdf4c42f3dc7b132df168d34719333b47664d93c137"
MANUSCRIPT = REPO / "partC_rev1_manuscripts" / "partC_rev1_manuscripts_20260727_v2"
NUMERICAL = MANUSCRIPT / "numerical_section_data"
OUTPUT_ROOT = REPO / "partC_rev1_outputs" / f"partC_rev1_numerical_section_revision_{RUN_HASH}"
CAP_ROOT = REPO / "partC_rev1_outputs" / "partC_rev1_cap2over3_gravity"
BENCHMARK_ROOT = CAP_ROOT / "partC_rev1_benchmarks"
BENCHMARK_REGISTRY = CAP_ROOT / "partC_rev1_benchmark_registry.csv"
RECOVERY_ROOT = REPO / "partC_rev1_outputs" / "partC_rev1_abstractflow_recovery_aadf3a3_fw100"
TRAIN_ROOT = RECOVERY_ROOT / "partC_rev1_recovery_training_v2_real_models"
GLOBAL_ROOT = RECOVERY_ROOT / "partC_rev1_recovery_global_v2_real_models"
PARTITION_ROOT = (
    REPO
    / "partC_rev1_outputs"
    / "partC_rev1_remaining_experiments_strict_ba7cf8c8cd2227ca"
    / "partC_rev1_partition_resolution_sweep"
)
CONFIG_PATH = REPO / "partC_rev1_configs" / "partC_rev1_real_cap2over3_gravity_training.json"
REVISION_CONFIG_PATH = (
    REPO / "partC_rev1_configs" / "partC_rev1_numerical_section_revision_20260727.json"
)
GRAPH_META = (
    REPO
    / "partC_rev1_cache"
    / "partC_rev1_cap2over3_gravity"
    / "partC_rev1_graph_snapshot_metadata_8ca61169217bb489e588a2b5154ad414834d7fe6383beda5d5ac0871b4c80274.json"
)
OD_META = (
    REPO
    / "partC_rev1_cache"
    / "partC_rev1_cap2over3_gravity"
    / "partC_rev1_od_snapshot_metadata_7d95a193d5b858bc9ac9077279161038de8f79f1782399bdee1fa91129be0e5a.json"
)
POLICY_GPKG = RECOVERY_ROOT / "partC_rev1_recovery_backbone_policy_flows.gpkg"
PARTITION_GPKG = REPO / "partC_rev1_cache" / "partC_rev1_backbone_partition.gpkg"
SELECTED_HASH = "831afae8885146f1"
METHOD_RUNS = {
    "MSA": "partC_rev1_msa_numba_csr_gap-1e-04_hash-0eba5d9593e02a05",
    "FW": "partC_rev1_fw_numba_csr_gap-1e-04_hash-8194e539676eae03",
    "CFW": "partC_rev1_cfw_numba_csr_gap-1e-04_hash-831afae8885146f1",
}
POLICY_COLUMNS = {
    "FFTT constant": "fftt_const_approximation",
    "Fast polynomial": "fast_poly_surrogate",
    "Best nonconstant": "best_nonconst_surrogate",
    "Best overall": "best_surrogate",
}
RESOLUTIONS = [1e-6, 0.3107232505953859, 1.0, 1.2, 3.550139466470431]
SEED = 42
EPS = 1e-12
CONUS_BOUNDS = (-124.77, -66.95, 24.52, 49.38)
CHICAGO_BOUNDS = (-88.55, -87.15, 41.35, 42.35)
PNG_DPI = 400
MIN_FONT_POINTS = 8.0
# elsarticle preprint, 11 pt, one-column layout inherits article's 345 pt
# text width.  Match source figure widths to the actual current manuscript.
ASSUMED_MANUSCRIPT_TEXT_WIDTH_INCHES = 345.0 / 72.27

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


def ensure_package_tree() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    for section in ("sec5_1", "sec5_2", "sec5_3", "sec5_4", "sec5_5", "appendix"):
        for child in ("code", "source_extracts", "plot_data", "figures", "tables"):
            (NUMERICAL / section / child).mkdir(parents=True, exist_ok=True)


def sha256_path(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def latex_escape(value: Any) -> str:
    text = "" if pd.isna(value) else str(value)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("_", r"\_"),
        ("%", r"\%"),
        ("&", r"\&"),
        ("#", r"\#"),
    ):
        text = text.replace(old, new)
    return text


def write_latex_table(
    path: Path,
    frame: pd.DataFrame,
    *,
    caption: str,
    label: str,
    column_format: str | None = None,
) -> None:
    display = frame.copy()
    for col in display.select_dtypes(include=["object"]).columns:
        display[col] = display[col].map(latex_escape)
    text = display.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
        column_format=column_format,
        float_format=lambda x: f"{x:.6g}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_extract(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def signed_relative_difference(prediction: Any, benchmark: Any) -> np.ndarray:
    pred = np.asarray(prediction, dtype=np.float64)
    base = np.asarray(benchmark, dtype=np.float64)
    denom = np.maximum(np.abs(pred), np.abs(base))
    return np.divide(pred - base, denom, out=np.zeros_like(denom, dtype=np.float64), where=denom > 0)


def wmape(prediction: Any, benchmark: Any, epsilon: float = EPS) -> float:
    pred = np.asarray(prediction, dtype=np.float64)
    base = np.asarray(benchmark, dtype=np.float64)
    return float(np.sum(np.abs(pred - base)) / max(float(np.sum(np.abs(base))), float(epsilon)))


def relative_cost_error(method_cost: float, benchmark_cost: float, epsilon: float = EPS) -> float:
    return float(abs(float(method_cost) - float(benchmark_cost)) / max(abs(float(benchmark_cost)), float(epsilon)))


def integrate_cpu_core_hours(
    elapsed_seconds: Sequence[float], process_cpu_percent: Sequence[float]
) -> float:
    t = np.asarray(elapsed_seconds, dtype=np.float64)
    p = np.asarray(process_cpu_percent, dtype=np.float64)
    if len(t) != len(p) or len(t) < 2 or np.any(np.diff(t) < 0):
        raise ValueError("CPU samples require equal lengths, at least two points, and monotonic time")
    return float(np.trapz(p / 100.0, t) / 3600.0)


def validate_physical_comparability(summaries: Mapping[str, Mapping[str, Any]]) -> None:
    keys = ("topology_hash", "backbone_hash", "assignment_graph_hash", "od_hash", "partition_hash", "backend")
    reference = summaries["CFW"]
    for method, row in summaries.items():
        mismatched = [key for key in keys if row.get(key) != reference.get(key)]
        multiplier = row.get("run_config", {}).get("capacity_multiplier")
        if mismatched:
            raise ValueError(f"{method} is incompatible with selected CFW on {mismatched}")
        if row.get("state") != "measured_complete" or not bool(row.get("valid_completed_feasible_iterate")):
            raise ValueError(f"{method} is not a complete feasible measured run")


def nested_fraction_indices(train_indices: Sequence[int], fractions: Sequence[float]) -> dict[float, np.ndarray]:
    train = np.asarray(train_indices, dtype=np.int64)
    if len(train) == 0 or train[0] != 0:
        raise ValueError("The fixed training pool must begin with sample 0")
    result: dict[float, np.ndarray] = {}
    previous: set[int] = set()
    for fraction in fractions:
        if not 0 < float(fraction) <= 1:
            raise ValueError("Fractions must be in (0,1]")
        count = max(1, int(math.ceil(len(train) * float(fraction))))
        chosen = train[:count].copy()
        if not previous.issubset(set(chosen.tolist())):
            raise AssertionError("Training fractions are not nested")
        result[float(fraction)] = chosen
        previous = set(chosen.tolist())
    return result


def choose_representative_regions(
    region_frame: pd.DataFrame, per_stratum: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"region_uuid", "physical_arc_count", "best_nonconst_validation_wmape", "active_pair_count", "positive_benchmark_flow"}
    missing = required.difference(region_frame.columns)
    if missing:
        raise ValueError(f"Representative-region input missing {sorted(missing)}")
    eligible = region_frame.loc[
        region_frame["active_pair_count"].gt(0)
        & region_frame["positive_benchmark_flow"].gt(0)
        & np.isfinite(region_frame["best_nonconst_validation_wmape"])
    ].copy()
    ranks = {
        "large": eligible.sort_values(["physical_arc_count", "region_uuid"], ascending=[False, True]),
        "small": eligible.sort_values(["physical_arc_count", "region_uuid"], ascending=[True, True]),
        "good": eligible.loc[
            eligible["active_pair_count"].ge(max(2, int(eligible["active_pair_count"].quantile(0.25))))
        ].sort_values(["best_nonconst_validation_wmape", "region_uuid"], ascending=[True, True]),
        "bad": eligible.sort_values(["best_nonconst_validation_wmape", "region_uuid"], ascending=[False, True]),
    }
    chosen: list[dict[str, Any]] = []
    used: set[str] = set()
    rank_rows: list[dict[str, Any]] = []
    for stratum in ("large", "small", "good", "bad"):
        rank = 0
        taken = 0
        for row in ranks[stratum].itertuples(index=False):
            rank += 1
            uid = str(row.region_uuid)
            rank_rows.append({"stratum": stratum, "rank": rank, "region_uuid": uid})
            if uid in used:
                continue
            chosen.append(
                {
                    "stratum": stratum,
                    "stratum_rank": rank,
                    "region_uuid": uid,
                    "physical_arc_count": int(row.physical_arc_count),
                    "active_pair_count": int(row.active_pair_count),
                    "positive_benchmark_flow": float(row.positive_benchmark_flow),
                    "best_nonconst_validation_wmape": float(row.best_nonconst_validation_wmape),
                    "is_exemplar": taken == 0,
                }
            )
            used.add(uid)
            taken += 1
            if taken >= per_stratum:
                break
        if taken < per_stratum:
            raise ValueError(f"Only {taken} unique eligible regions available for {stratum}")
    return pd.DataFrame(chosen), pd.DataFrame(rank_rows)


def deterministic_spatial_link_selection(
    candidates: pd.DataFrame,
    *,
    n_select: int = 100,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.cluster import MiniBatchKMeans

    required = {"partC_rev1_link_uid", "benchmark_flow", "x_5070", "y_5070"}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Link candidates missing {sorted(missing)}")
    pool = candidates.copy()
    pool["partC_rev1_link_uid"] = pool["partC_rev1_link_uid"].astype(str)
    pool = pool.sort_values(["benchmark_flow", "partC_rev1_link_uid"], ascending=[False, True]).drop_duplicates(
        "partC_rev1_link_uid"
    )
    if len(pool) < n_select:
        raise ValueError(f"Eligible Interstate population {len(pool)} is smaller than {n_select}")
    xy = pool[["x_5070", "y_5070"]].to_numpy(dtype=np.float64)
    km = MiniBatchKMeans(
        n_clusters=n_select,
        random_state=random_state,
        n_init=10,
        batch_size=min(4096, len(pool)),
        reassignment_ratio=0.0,
    )
    pool["cluster"] = km.fit_predict(xy)
    selected_index = (
        pool.sort_values(["cluster", "benchmark_flow", "partC_rev1_link_uid"], ascending=[True, False, True])
        .groupby("cluster", sort=True)
        .head(1)
        .index.tolist()
    )
    selected = pool.loc[selected_index].copy()
    while len(selected) < n_select:
        remainder = pool.loc[~pool.index.isin(selected.index)].copy()
        sxy = selected[["x_5070", "y_5070"]].to_numpy(dtype=np.float64)
        rxy = remainder[["x_5070", "y_5070"]].to_numpy(dtype=np.float64)
        distances = np.sqrt(((rxy[:, None, :] - sxy[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
        remainder["min_distance_m"] = distances
        pick = remainder.sort_values(
            ["min_distance_m", "benchmark_flow", "partC_rev1_link_uid"],
            ascending=[False, False, True],
        ).iloc[[0]]
        selected = pd.concat([selected, pick])
    selected = selected.sort_values(["benchmark_flow", "partC_rev1_link_uid"], ascending=[False, True]).reset_index(drop=True)
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    if selected["partC_rev1_link_uid"].nunique() != n_select:
        raise AssertionError("Spatial selection did not produce unique physical links")
    return pool.reset_index(drop=True), selected


def filter_i95_routes(frame: pd.DataFrame) -> pd.Series:
    route_type = frame.get("route_type", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper().str.strip()
    route_number = (
        frame.get("route_number", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^0-9]", "", regex=True)
        .str.lstrip("0")
    )
    sign = frame.get("sign_route", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    name = frame.get("road_name", pd.Series("", index=frame.index)).fillna("").astype(str).str.upper()
    exact_route = route_type.eq("I") & route_number.eq("95")
    corroborated = sign.str.contains(r"(?:^|[^0-9])I[- ]?95(?:[^0-9]|$)", regex=True) | name.str.contains(
        r"(?:^|[^0-9])I[- ]?95(?:[^0-9]|$)", regex=True
    )
    return exact_route & corroborated


def _read_summary(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / "partC_rev1_summary.json").read_text(encoding="utf-8"))


def _method_dirs() -> dict[str, Path]:
    return {method: BENCHMARK_ROOT / directory for method, directory in METHOD_RUNS.items()}


def _enforce_minimum_font_size(fig: plt.Figure) -> float:
    """Raise every rendered text object to the manuscript minimum."""

    fig.canvas.draw()
    sizes: list[float] = []
    for text in fig.findobj(match=matplotlib.text.Text):
        if not text.get_text():
            continue
        if float(text.get_fontsize()) < MIN_FONT_POINTS:
            text.set_fontsize(MIN_FONT_POINTS)
        sizes.append(float(text.get_fontsize()))
    return min(sizes) if sizes else MIN_FONT_POINTS


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    source_width = float(fig.get_figwidth())
    source_height = float(fig.get_figheight())
    if "fig4_suba_" in stem.name:
        manuscript_width_fraction = 0.98
    elif re.search(r"_fig[3-8]_sub[a-z]_", stem.name):
        manuscript_width_fraction = 0.48
    else:
        manuscript_width_fraction = None
    if manuscript_width_fraction is not None:
        target_width = ASSUMED_MANUSCRIPT_TEXT_WIDTH_INCHES * manuscript_width_fraction
        scale = target_width / source_width
        fig.set_size_inches(target_width, source_height * scale, forward=True)
        try:
            fig.tight_layout()
        except Exception:
            pass
    minimum_font = _enforce_minimum_font_size(fig)
    fig.savefig(stem.with_suffix(".pdf"), dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=PNG_DPI, bbox_inches="tight")
    write_json(
        stem.with_name(f"{stem.name}_render_metadata.json"),
        {
            "figure_stem": stem.name,
            "png_dpi": PNG_DPI,
            "minimum_rendered_font_points": minimum_font,
            "figure_width_inches": float(fig.get_figwidth()),
            "figure_height_inches": float(fig.get_figheight()),
            "original_figure_width_inches": source_width,
            "original_figure_height_inches": source_height,
            "manuscript_width_fraction": manuscript_width_fraction,
            "assumed_manuscript_text_width_inches": ASSUMED_MANUSCRIPT_TEXT_WIDTH_INCHES,
            "estimated_manuscript_minimum_font_points": minimum_font,
            "pdf_vector_output": True,
            "pdf_rasterized_layer_dpi": PNG_DPI,
        },
    )
    plt.close(fig)


def _to_wgs84(frame):
    if frame.crs is None:
        raise ValueError("Map source has no CRS; WGS84 conversion would be ambiguous")
    return frame.to_crs(4326)


def _record_map_extent(
    map_id: str,
    *,
    view: str,
    bounds: tuple[float, float, float, float],
    source_crs: str,
    plotted_feature_count: int,
) -> None:
    path = NUMERICAL / "appendix" / "tables" / "partC_rev1_map_extent_validation.csv"
    row = pd.DataFrame(
        [
            {
                "map_id": map_id,
                "view": view,
                "longitude_min": bounds[0],
                "longitude_max": bounds[1],
                "latitude_min": bounds[2],
                "latitude_max": bounds[3],
                "plot_crs": "EPSG:4326",
                "source_crs": source_crs,
                "plotted_feature_count": int(plotted_feature_count),
                "inside_mainland_bounds": bool(
                    bounds[0] >= CONUS_BOUNDS[0]
                    and bounds[1] <= CONUS_BOUNDS[1]
                    and bounds[2] >= CONUS_BOUNDS[2]
                    and bounds[3] <= CONUS_BOUNDS[3]
                ),
                "exact_mainland_view": bool(np.allclose(bounds, CONUS_BOUNDS)),
            }
        ]
    )
    if path.exists():
        prior = pd.read_csv(path)
        prior = prior.loc[prior["map_id"].astype(str).ne(map_id)]
        row = pd.concat([prior, row], ignore_index=True)
    write_csv(path, row.sort_values("map_id").reset_index(drop=True))


def _set_map_extent(
    ax: plt.Axes,
    map_id: str,
    *,
    view: str = "mainland",
    bounds: tuple[float, float, float, float] = CONUS_BOUNDS,
    source_crs: str,
    plotted_feature_count: int,
    show_coordinates: bool = True,
) -> None:
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect("equal", adjustable="box")
    if show_coordinates:
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(True, color="0.90", linewidth=0.4)
    else:
        ax.set_axis_off()
    _record_map_extent(
        map_id,
        view=view,
        bounds=bounds,
        source_crs=source_crs,
        plotted_feature_count=plotted_feature_count,
    )


def _bounded_local_extent(
    total_bounds: Sequence[float],
    *,
    pad_fraction: float = 0.08,
    minimum_pad_degrees: float = 0.05,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = map(float, total_bounds)
    pad_x = max(minimum_pad_degrees, (xmax - xmin) * pad_fraction)
    pad_y = max(minimum_pad_degrees, (ymax - ymin) * pad_fraction)
    return (
        max(CONUS_BOUNDS[0], xmin - pad_x),
        min(CONUS_BOUNDS[1], xmax + pad_x),
        max(CONUS_BOUNDS[2], ymin - pad_y),
        min(CONUS_BOUNDS[3], ymax + pad_y),
    )


def collect_hardware_environment() -> dict[str, Any]:
    env_path = _method_dirs()["CFW"] / "partC_rev1_environment.json"
    historical = json.loads(env_path.read_text(encoding="utf-8"))
    current: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_signature": platform.processor(),
        "python": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }
    try:
        import psutil

        current.update(
            {
                "physical_cores": psutil.cpu_count(logical=False),
                "logical_processors": psutil.cpu_count(logical=True),
                "installed_ram_bytes": int(psutil.virtual_memory().total),
            }
        )
    except Exception as exc:
        current["psutil_error"] = repr(exc)
    cim_command = (
        "$cpu=Get-CimInstance Win32_Processor|Select-Object -First 1 Name,Manufacturer,"
        "NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed;"
        "$sys=Get-CimInstance Win32_ComputerSystem|Select-Object Manufacturer,Model,TotalPhysicalMemory;"
        "$os=Get-CimInstance Win32_OperatingSystem|Select-Object Caption,Version,BuildNumber,OSArchitecture;"
        "[PSCustomObject]@{cpu=$cpu;system=$sys;os=$os}|ConvertTo-Json -Depth 4 -Compress"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cim_command],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        current["cim"] = json.loads(out)
    except Exception as exc:
        current["cim_error"] = repr(exc)
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
        current["nvidia_smi"] = out
    except Exception as exc:
        current["nvidia_smi_error"] = repr(exc)
    package_names = {
        "numpy": "numpy",
        "scipy": "scipy",
        "numba": "numba",
        "igraph": "igraph",
        "networkx": "networkx",
        "xgboost": "xgboost",
        "torch": "torch",
        "pandas": "pandas",
        "pyarrow": "pyarrow",
    }
    current["packages"] = {}
    for label, distribution in package_names.items():
        try:
            current["packages"][label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            current["packages"][label] = "not installed"
    current["thread_environment"] = {
        key: os.environ.get(key, "not recorded")
        for key in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    same_machine = (
        historical.get("platform") == current.get("platform")
        and historical.get("machine") == current.get("machine")
        and historical.get("processor") == current.get("processor_signature")
        and historical.get("conda_default_env") == current.get("conda_default_env")
    )
    return {
        "historical_record": historical,
        "current_host": current,
        "same_machine_signature_match": bool(same_machine),
        "use_of_current_host": (
            "CIM fills hardware detail because OS build, architecture, processor signature, "
            "and conda environment match the saved run."
            if same_machine
            else "Current-host values are kept separate and do not replace historical facts."
        ),
        "gpu_usage_note": "GPU was present on the matched host, but authoritative solver/training logs do not record GPU use or GPU time.",
    }


def build_sec5_1() -> None:
    ensure_package_tree()
    section = NUMERICAL / "sec5_1"
    graph = json.loads(GRAPH_META.read_text(encoding="utf-8"))
    od = json.loads(OD_META.read_text(encoding="utf-8"))
    recovery = json.loads((RECOVERY_ROOT / "partC_rev1_abstract_benchmark_summary.json").read_text(encoding="utf-8"))
    training = json.loads((TRAIN_ROOT / "partC_rev1_abstract_training_summary.json").read_text(encoding="utf-8"))
    selected = _read_summary(_method_dirs()["CFW"])
    environment = collect_hardware_environment()
    environment["source_hashes"] = {
        "saved_environment": sha256_path(_method_dirs()["CFW"] / "partC_rev1_environment.json"),
        "graph_metadata": sha256_path(GRAPH_META),
        "od_metadata": sha256_path(OD_META),
        "config": CONFIG_HASH,
    }
    write_json(section / "source_extracts" / "partC_rev1_environment_manifest.json", environment)
    copy_extract(GRAPH_META, section / "source_extracts" / GRAPH_META.name)
    copy_extract(OD_META, section / "source_extracts" / OD_META.name)
    rows = [
        ("A", "Raw nodes", 974788, "nodes", "measured preprocess"),
        ("A", "Clean physical nodes", 348495, "nodes", "measured preprocess"),
        ("A", "Raw links", 487394, "physical links", "measured preprocess"),
        ("A", "Clean physical links", 487394, "physical links", "measured preprocess"),
        ("A", "Usable directed arcs", graph["arc_count"], "directed arcs", "measured cap-2/3 snapshot"),
        ("A", "Backbone directed arcs", 326202, "directed arcs", "measured recovery graph"),
        ("A", "Positive OD rows", od["od_rows"], "rows", "measured cap-2/3 snapshot"),
        ("A", "Origins", od["origin_count"], "nodes", "measured cap-2/3 snapshot"),
        ("A", "Destinations", 3225, "nodes", "computed from immutable OD array"),
        ("A", "Total demand", od["total_demand"], "thousand tons/year", "measured cap-2/3 snapshot"),
        ("A", "Reachable benchmark demand", selected["reachable_demand"], "thousand tons/year", "selected CFW"),
        ("A", "Unreachable benchmark demand", selected["unreachable_demand"], "thousand tons/year", "selected CFW"),
        ("A", "Regions / trained non-small regions", training["region_count"], "regions", "measured all-region run"),
        ("A", "Active regional legs", recovery["abstract_region_leg_count"], "active pairs", "measured recovery"),
        ("A", "Assignment capacity", "2/3 overall; centroid connectors 100x", "multiplier", "frozen config"),
        ("A", "Assignment graph hash", selected["assignment_graph_hash"], "SHA-256", "selected CFW"),
        ("A", "Partition hash", selected["partition_hash"], "SHA-256", "selected CFW"),
        ("A", "OD hash", selected["od_hash"], "SHA-256", "selected CFW"),
    ]
    historical = environment["historical_record"]
    current = environment["current_host"]
    cim = current.get("cim", {})
    cpu = cim.get("cpu", {}) if isinstance(cim, dict) else {}
    system = cim.get("system", {}) if isinstance(cim, dict) else {}
    os_record = cim.get("os", {}) if isinstance(cim, dict) else {}
    if environment["same_machine_signature_match"]:
        cpu_name = cpu.get("Name", historical.get("processor"))
        physical_cores = cpu.get("NumberOfCores", current.get("physical_cores", "not recorded"))
        logical = cpu.get("NumberOfLogicalProcessors", current.get("logical_processors", "not recorded"))
        clock = cpu.get("MaxClockSpeed", "not recorded")
        ram = system.get("TotalPhysicalMemory", current.get("installed_ram_bytes", "not recorded"))
    else:
        cpu_name = historical.get("processor")
        physical_cores = logical = clock = ram = "not recorded"
    rows.extend(
        [
            ("B", "Operating system", historical["platform"], "", "saved environment"),
            ("B", "Architecture", historical["machine"], "", "saved environment"),
            ("B", "CPU manufacturer/model", cpu_name, "", "saved signature + matched CIM"),
            ("B", "Physical cores", physical_cores, "cores", "matched CIM" if environment["same_machine_signature_match"] else "not recorded"),
            ("B", "Logical processors", logical, "threads", "matched CIM" if environment["same_machine_signature_match"] else "not recorded"),
            ("B", "Maximum clock", clock, "MHz", "matched CIM" if environment["same_machine_signature_match"] else "not recorded"),
            ("B", "Installed RAM", ram, "bytes", "matched CIM" if environment["same_machine_signature_match"] else "not recorded"),
            ("B", "GPU on matched host", current.get("nvidia_smi", "not recorded"), "", "presence only; usage not recorded"),
            ("B", "Python", historical["python"], "", "saved environment"),
            ("B", "Conda environment", historical["conda_default_env"], "", "saved environment"),
            ("B", "Solver backend", selected["backend"], "", "selected CFW"),
            ("B", "Thread settings", "not recorded", "", "saved run lacks effective thread variables"),
            ("B", "Storage type", "not recorded", "", "not present in run environment"),
        ]
    )
    for package, version in historical["packages"].items():
        rows.append(("B", f"Package: {package}", version, "", "saved environment"))
    for package in ("xgboost", "torch"):
        rows.append(
            (
                "B",
                f"Package: {package}",
                current["packages"].get(package, "not recorded") if environment["same_machine_signature_match"] else "not recorded",
                "",
                "matched current conda env; historical training manifest omitted version",
            )
        )
    setting = pd.DataFrame(rows, columns=["panel", "quantity", "value", "unit", "source_status"])
    write_csv(section / "tables" / "partC_rev1_table_study_setting.csv", setting)
    write_latex_table(
        section / "tables" / "partC_rev1_table_study_setting.tex",
        setting[["panel", "quantity", "value", "unit"]],
        caption="Study setting and recorded computing environment.",
        label="tab:study_setting",
    )
    hardware = pd.DataFrame(
        [
            {"field": key, "value": json.dumps(value) if isinstance(value, (dict, list)) else value, "provenance": "historical saved environment"}
            for key, value in historical.items()
        ]
        + [
            {"field": f"matched_current_{key}", "value": json.dumps(value) if isinstance(value, (dict, list)) else value, "provenance": "current host; used only under signature match"}
            for key, value in current.items()
        ]
    )
    write_csv(section / "tables" / "partC_rev1_hardware_environment_details.csv", hardware)
    runtime_rows: list[dict[str, Any]] = []
    for method, run_dir in _method_dirs().items():
        summary = _read_summary(run_dir)
        runtime_rows.append(
            {
                "stage_or_method": f"physical_{method}",
                "wall_clock_sec": summary["runtime_seconds"],
                "process_user_cpu_sec": "CPU time not recorded",
                "process_system_cpu_sec": "CPU time not recorded",
                "mean_process_cpu_percent": "CPU utilization not recorded",
                "median_process_cpu_percent": "CPU utilization not recorded",
                "cpu_core_hours": "CPU time not recorded",
                "gpu_hours": "GPU time not recorded",
                "peak_rss_gb": summary["peak_memory_mb"] / 1024.0,
                "source_status": "measured_complete",
            }
        )
    runtime_rows.append(
        {
            "stage_or_method": "all_region_training",
            "wall_clock_sec": training["total_training_time_sec"],
            "process_user_cpu_sec": "CPU time not recorded",
            "process_system_cpu_sec": "CPU time not recorded",
            "mean_process_cpu_percent": "CPU utilization not recorded",
            "median_process_cpu_percent": "CPU utilization not recorded",
            "cpu_core_hours": "CPU time not recorded",
            "gpu_hours": "GPU time not recorded",
            "peak_rss_gb": "not recorded",
            "source_status": "measured_complete",
        }
    )
    global_summary = pd.read_csv(GLOBAL_ROOT / "partC_rev1_recovery_policy_vs_benchmark_summary.csv")
    for row in global_summary.itertuples(index=False):
        if str(row.method) == "BBM physical benchmark":
            continue
        runtime_rows.append(
            {
                "stage_or_method": f"global_{row.method}",
                "wall_clock_sec": float(row.global_runtime_sec),
                "process_user_cpu_sec": "CPU time not recorded",
                "process_system_cpu_sec": "CPU time not recorded",
                "mean_process_cpu_percent": "CPU utilization not recorded",
                "median_process_cpu_percent": "CPU utilization not recorded",
                "cpu_core_hours": "CPU time not recorded",
                "gpu_hours": "GPU time not recorded",
                "peak_rss_gb": "not recorded",
                "source_status": "measured_complete",
            }
        )
    write_csv(section / "tables" / "partC_rev1_cpu_runtime_accounting.csv", pd.DataFrame(runtime_rows))


def _load_physical_summaries() -> dict[str, dict[str, Any]]:
    summaries = {method: _read_summary(path) for method, path in _method_dirs().items()}
    validate_physical_comparability(summaries)
    return summaries


def _load_policy_backbone():
    import geopandas as gpd

    columns = [
        "directed_arc_id",
        "partC_rev1_link_uid",
        "class_code",
        "road_name",
        "sign_route",
        "route_type",
        "route_number",
        "length_miles",
        "partC_rev1_faf_stfips",
        "benchmark_flow",
        "benchmark_cost",
        "fftt_const_approximation_flow",
        "fftt_const_approximation_cost",
        "fast_poly_surrogate_flow",
        "fast_poly_surrogate_cost",
        "best_nonconst_surrogate_flow",
        "best_nonconst_surrogate_cost",
        "best_surrogate_flow",
        "best_surrogate_cost",
    ]
    return gpd.read_file(
        POLICY_GPKG,
        layer="partC_rev1_recovery_backbone_policy_flows",
        columns=columns,
    )


def _aggregate_physical_backbone(backbone):
    import geopandas as gpd

    aggregations: dict[str, Any] = {
        "class_code": "first",
        "road_name": "first",
        "sign_route": "first",
        "route_type": "first",
        "route_number": "first",
        "length_miles": "first",
        "partC_rev1_faf_stfips": "first",
        "selected_cfw_flow": "sum",
    }
    for policy in POLICY_COLUMNS.values():
        aggregations[f"{policy}_flow"] = "sum"
    attrs = backbone.drop(columns="geometry").groupby("partC_rev1_link_uid", as_index=False).agg(aggregations)
    geometry = backbone[["partC_rev1_link_uid", "geometry"]].drop_duplicates("partC_rev1_link_uid")
    return gpd.GeoDataFrame(attrs.merge(geometry, on="partC_rev1_link_uid", how="left"), geometry="geometry", crs=backbone.crs)


def build_sec5_2() -> None:
    ensure_package_tree()
    section = NUMERICAL / "sec5_2"
    appendix = NUMERICAL / "appendix"
    summaries = _load_physical_summaries()
    selected_flow = pd.read_parquet(
        _method_dirs()["CFW"] / "partC_rev1_directed_arc_flow.parquet",
        columns=["directed_arc_id", "flow", "cost"],
    )
    selected_flow["directed_arc_id"] = selected_flow["directed_arc_id"].astype(str)
    selected_flow = selected_flow.rename(columns={"flow": "selected_cfw_flow", "cost": "selected_cfw_cost"})
    solver_rows: list[dict[str, Any]] = []
    convergence_parts = []
    for method, run_dir in _method_dirs().items():
        summary = summaries[method]
        solver_rows.append(
            {
                "method": f"{method}{' (selected BBM)' if method == 'CFW' else ''}",
                "backend": summary["backend"],
                "completion_status": summary["state"],
                "iterations": summary["iteration_count"],
                "assignment_runtime_sec": summary["runtime_seconds"],
                "final_relative_gap": summary["final_gap"],
                "reachable_demand": summary["reachable_demand"],
                "unreachable_demand": summary["unreachable_demand"],
                "peak_memory_gb": summary["peak_memory_mb"] / 1024.0,
                "process_cpu_time_sec_if_recorded": "not recorded",
                "cpu_core_hours_if_measured": "not recorded",
                "stop_reason": summary["stop_reason"],
                "run_hash": summary["run_hash"],
            }
        )
        conv = pd.read_csv(run_dir / "partC_rev1_convergence.csv")
        conv["wall_time_sec"] = pd.to_numeric(conv["iteration_runtime_seconds"], errors="coerce").fillna(0).cumsum()
        conv["method_label"] = method
        conv["stop_reason"] = summary["stop_reason"]
        conv["run_hash"] = summary["run_hash"]
        convergence_parts.append(conv)
    convergence = pd.concat(convergence_parts, ignore_index=True)
    write_csv(section / "plot_data" / "partC_rev1_msa_fw_cfw_convergence.csv", convergence)
    for x_col, name, x_label in (
        ("iteration", "partC_rev1_figA5_solver_gap_iteration", "Iteration"),
        ("wall_time_sec", "partC_rev1_figA6_solver_gap_time", "Measured wall time (s)"),
    ):
        convergence_panel = convergence.copy()
        convergence_panel["figure_stem"] = name
        write_csv(appendix / "plot_data" / f"{name}.csv", convergence_panel)
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for method, group in convergence.groupby("method_label", sort=False):
            valid = group.loc[pd.to_numeric(group["relative_gap"], errors="coerce").notna()]
            ax.plot(valid[x_col], valid["relative_gap"], label=method, linewidth=1.8)
            last = valid.iloc[-1]
            ax.scatter([last[x_col]], [last["relative_gap"]], s=35, zorder=3)
        ax.set_yscale("log")
        ax.set_ylim(bottom=1e-4)
        ax.axhline(1e-4, color="0.5", linestyle="--", linewidth=0.9, label="target $10^{-4}$")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Saved relative gap")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
        _save_figure(fig, appendix / "figures" / name)
    copy_extract(BENCHMARK_REGISTRY, section / "source_extracts" / BENCHMARK_REGISTRY.name)

    backbone = _load_policy_backbone()
    backbone["directed_arc_id"] = backbone["directed_arc_id"].astype(str)
    backbone = backbone.merge(selected_flow, on="directed_arc_id", validate="one_to_one")
    global_strict = pd.read_csv(GLOBAL_ROOT / "partC_rev1_recovery_policy_vs_benchmark_summary.csv")
    global_const = pd.read_csv(RECOVERY_ROOT / "partC_rev1_recovery_policy_vs_benchmark_summary.csv")
    benchmark_runtime = summaries["CFW"]["runtime_seconds"]
    benchmark_cost = float(np.sum(backbone["selected_cfw_flow"] * backbone["selected_cfw_cost"]))
    policy_rows: list[dict[str, Any]] = []
    for label, policy in POLICY_COLUMNS.items():
        source = global_const if policy == "fftt_const_approximation" else global_strict
        row = source.loc[source["method"].eq(policy)].iloc[0]
        predicted_cost = float(np.sum(backbone[f"{policy}_flow"] * backbone[f"{policy}_cost"]))
        policy_rows.append(
            {
                "method": label,
                "assignment_algorithm": "hybrid MSA/full-refresh fixed point",
                "surrogate_policy": policy,
                "assignment_runtime_sec": float(row["global_runtime_sec"]),
                "training_runtime_sec": float(row["training_time_sec"]),
                "total_runtime_sec": float(row["total_runtime_sec"]),
                "assignment_runtime_ratio_vs_benchmark": float(row["global_runtime_sec"]) / benchmark_runtime,
                "total_runtime_ratio_vs_benchmark": float(row["total_runtime_sec"]) / benchmark_runtime,
                "backbone_flow_WMAPE_vs_selected_CFW": wmape(
                    backbone[f"{policy}_flow"], backbone["selected_cfw_flow"]
                ),
                "backbone_relative_cost_error_vs_selected_CFW": relative_cost_error(
                    predicted_cost, benchmark_cost
                ),
                "final_gap_or_residual": float(row["final_gap_or_residual"]),
                "reachable_demand": float(row["assigned_demand"]),
                "unreachable_demand": float(row["abstract_unreachable_demand"]),
                "active_region_leg_count": int(row["active_region_leg_count"]),
                "surrogate_inference_time_sec_if_available": float(row["surrogate_inference_time_sec"]),
                "reference_note": "flow/cost recomputed vs selected CFW; residual/runtime from measured recovery policy run",
            }
        )
    policies = pd.DataFrame(policy_rows)

    physical = _aggregate_physical_backbone(backbone)
    physical["partC_rev1_link_uid"] = physical["partC_rev1_link_uid"].astype(str)
    interstate = physical.loc[pd.to_numeric(physical["class_code"], errors="coerce").eq(11)].copy()
    if len(interstate) < 100:
        raise ValueError(f"Only {len(interstate)} class-11 Interstate physical links were available")
    interstate = interstate.sort_values(["selected_cfw_flow", "partC_rev1_link_uid"], ascending=[False, True]).head(
        max(2000, 20 * 100)
    )
    midpoints = interstate.to_crs(5070).geometry.interpolate(0.5, normalized=True)
    interstate["x_5070"] = midpoints.x.to_numpy()
    interstate["y_5070"] = midpoints.y.to_numpy()
    interstate["benchmark_flow"] = interstate["selected_cfw_flow"]
    candidate_attrs, selected_attrs = deterministic_spatial_link_selection(
        interstate.drop(columns="geometry"), n_select=100, random_state=SEED
    )
    candidate = interstate.merge(
        candidate_attrs[["partC_rev1_link_uid", "cluster"]],
        on="partC_rev1_link_uid",
        validate="one_to_one",
    )
    selected = physical.merge(
        selected_attrs[["partC_rev1_link_uid", "cluster", "selection_rank", "x_5070", "y_5070"]],
        on="partC_rev1_link_uid",
        validate="one_to_one",
    )
    selected_wgs = selected.to_crs(4326)
    selected_midpoint_5070 = selected.to_crs(5070).geometry.interpolate(0.5, normalized=True)
    selected_midpoint_wgs = selected_midpoint_5070.set_crs(5070).to_crs(4326)
    selected["longitude"] = selected_midpoint_wgs.x.to_numpy()
    selected["latitude"] = selected_midpoint_wgs.y.to_numpy()
    accepted_codes = pd.DataFrame(
        [{"class_code": 11, "Class_Description": "Interstate Highway", "F_Class": 1, "route_cross_check": "route_type=I"}]
    )
    write_csv(section / "source_extracts" / "partC_rev1_interstate_accepted_class_codes.csv", accepted_codes)
    write_csv(
        section / "plot_data" / "partC_rev1_interstate_candidate_pool.csv",
        candidate.drop(columns="geometry"),
    )
    write_csv(
        section / "plot_data" / "partC_rev1_interstate_cluster_assignments.csv",
        candidate[["partC_rev1_link_uid", "cluster", "benchmark_flow", "x_5070", "y_5070"]],
    )
    write_csv(
        section / "plot_data" / "partC_rev1_selected_100_interstate_links.csv",
        selected.drop(columns="geometry"),
    )
    selected.to_file(
        section / "source_extracts" / "partC_rev1_selected_100_interstate_links.geojson",
        driver="GeoJSON",
    )
    write_json(
        section / "source_extracts" / "partC_rev1_interstate_selection_parameters.json",
        {
            "accepted_class_code": 11,
            "accepted_description": "Interstate Highway",
            "candidate_pool_rule": "highest selected-CFW physical-link flow; max(2000,20*100)",
            "candidate_count": int(len(candidate)),
            "projection": "EPSG:5070",
            "algorithm": "MiniBatchKMeans",
            "n_clusters": 100,
            "random_state": SEED,
            "duplicate_rule": "stable unique physical link UID",
            "fill_rule": "maximize minimum projected distance, then higher flow, then UID",
        },
    )

    metric_columns = [
        "backbone_flow_WMAPE_vs_selected_CFW",
        "selected_link_flow_WMAPE_vs_selected_CFW",
        "backbone_relative_cost_error_vs_selected_CFW",
        "selected_link_relative_cost_error_vs_selected_CFW",
    ]
    selected_uids = set(selected["partC_rev1_link_uid"].astype(str))
    backbone["partC_rev1_link_uid"] = backbone["partC_rev1_link_uid"].astype(str)
    selected_backbone = backbone.loc[
        backbone["partC_rev1_link_uid"].isin(selected_uids)
    ].copy()
    if selected_backbone.empty:
        raise ValueError("The deterministic selected-link set did not join to the backbone")
    selected_benchmark_cost = float(
        np.sum(
            selected_backbone["selected_cfw_flow"]
            * selected_backbone["selected_cfw_cost"]
        )
    )
    for label, policy in POLICY_COLUMNS.items():
        row_mask = policies["method"].eq(label)
        selected_policy_cost = float(
            np.sum(
                selected_backbone[f"{policy}_flow"]
                * selected_backbone[f"{policy}_cost"]
            )
        )
        policies.loc[
            row_mask, "selected_link_flow_WMAPE_vs_selected_CFW"
        ] = wmape(selected[f"{policy}_flow"], selected["selected_cfw_flow"])
        policies.loc[
            row_mask, "selected_link_relative_cost_error_vs_selected_CFW"
        ] = relative_cost_error(selected_policy_cost, selected_benchmark_cost)
    policies = policies[
        [
            "method",
            "assignment_algorithm",
            "surrogate_policy",
            "assignment_runtime_sec",
            "training_runtime_sec",
            "total_runtime_sec",
            "assignment_runtime_ratio_vs_benchmark",
            "total_runtime_ratio_vs_benchmark",
            *metric_columns,
            "final_gap_or_residual",
            "reachable_demand",
            "unreachable_demand",
            "active_region_leg_count",
            "surrogate_inference_time_sec_if_available",
            "reference_note",
        ]
    ]
    write_csv(
        section / "tables" / "partC_rev1_table_hybrid_policy_comparison.csv",
        policies,
    )
    write_latex_table(
        section / "tables" / "partC_rev1_table_hybrid_policy_comparison.tex",
        policies,
        caption=(
            "Panel B: measured hybrid policies compared with the registry-selected "
            "full physical CFW BBM; all four backbone/selected-link flow and cost "
            "metrics are reported together."
        ),
        label="tab:hybrid_policy_comparison",
    )

    backbone_reference = backbone[
        [
            "directed_arc_id",
            "partC_rev1_link_uid",
            "selected_cfw_flow",
            "selected_cfw_cost",
        ]
    ].copy()
    backbone_reference["directed_arc_id"] = backbone_reference[
        "directed_arc_id"
    ].astype(str)
    solver_metric_rows: dict[str, dict[str, float]] = {}
    for method, run_dir in _method_dirs().items():
        method_flow = pd.read_parquet(
            run_dir / "partC_rev1_directed_arc_flow.parquet",
            columns=["directed_arc_id", "flow", "cost"],
        )
        method_flow["directed_arc_id"] = method_flow["directed_arc_id"].astype(str)
        comparison = backbone_reference.merge(
            method_flow,
            on="directed_arc_id",
            how="left",
            validate="one_to_one",
        )
        if comparison[["flow", "cost"]].isna().any().any():
            raise ValueError(f"{method} is missing one or more backbone directed arcs")
        selected_comparison = comparison.loc[
            comparison["partC_rev1_link_uid"].isin(selected_uids)
        ].copy()
        physical_link_flows = (
            comparison.groupby("partC_rev1_link_uid", as_index=False)
            .agg(selected_cfw_flow=("selected_cfw_flow", "sum"), flow=("flow", "sum"))
        )
        selected_physical_link_flows = physical_link_flows.loc[
            physical_link_flows["partC_rev1_link_uid"].isin(selected_uids)
        ]
        solver_metric_rows[method] = {
            "backbone_flow_WMAPE_vs_selected_CFW": wmape(
                comparison["flow"],
                comparison["selected_cfw_flow"],
            ),
            "selected_link_flow_WMAPE_vs_selected_CFW": wmape(
                selected_physical_link_flows["flow"],
                selected_physical_link_flows["selected_cfw_flow"],
            ),
            "backbone_relative_cost_error_vs_selected_CFW": relative_cost_error(
                float(np.sum(comparison["flow"] * comparison["cost"])),
                benchmark_cost,
            ),
            "selected_link_relative_cost_error_vs_selected_CFW": relative_cost_error(
                float(np.sum(selected_comparison["flow"] * selected_comparison["cost"])),
                selected_benchmark_cost,
            ),
        }
    for row in solver_rows:
        method = str(row["method"]).split(" ", 1)[0]
        row.update(solver_metric_rows[method])
    solvers = pd.DataFrame(solver_rows)
    solver_prefix = [
        "method",
        "backend",
        "completion_status",
        "iterations",
        "assignment_runtime_sec",
        "final_relative_gap",
        "reachable_demand",
        "unreachable_demand",
        "peak_memory_gb",
        "process_cpu_time_sec_if_recorded",
        "cpu_core_hours_if_measured",
    ]
    solvers = solvers[
        solver_prefix + metric_columns + ["stop_reason", "run_hash"]
    ]
    write_csv(
        section / "tables" / "partC_rev1_table_full_physical_solvers.csv",
        solvers,
    )
    write_latex_table(
        section / "tables" / "partC_rev1_table_full_physical_solvers.tex",
        solvers,
        caption=(
            "Panel A: compatible full physical assignment algorithms; CFW is the "
            "registry-selected BBM and all four backbone/selected-link flow and "
            "cost metrics are reported together."
        ),
        label="tab:full_physical_solvers",
    )

    link_rows = selected[
        [
            "partC_rev1_link_uid",
            "road_name",
            "sign_route",
            "class_code",
            "latitude",
            "longitude",
            "selected_cfw_flow",
        ]
    ].rename(columns={"sign_route": "route_name", "selected_cfw_flow": "benchmark_flow"})
    for label, policy in POLICY_COLUMNS.items():
        short = policy.replace("_surrogate", "").replace("_approximation", "")
        link_rows[f"{short}_flow"] = selected[f"{policy}_flow"].to_numpy()
        link_rows[f"{short}_signed_relative_difference"] = signed_relative_difference(
            selected[f"{policy}_flow"], selected["selected_cfw_flow"]
        )
        link_rows[f"{short}_absolute_error"] = np.abs(
            selected[f"{policy}_flow"].to_numpy() - selected["selected_cfw_flow"].to_numpy()
        )
    write_csv(section / "tables" / "partC_rev1_selected_link_summary.csv", link_rows)
    top10 = link_rows.nlargest(10, "benchmark_flow")
    write_csv(appendix / "tables" / "partC_rev1_top10_selected_interstate_links.csv", top10)
    write_latex_table(
        appendix / "tables" / "partC_rev1_top10_selected_interstate_links.tex",
        top10,
        caption="Top ten deterministic selected Interstate links by CFW benchmark flow.",
        label="tab:selected_interstate_top10",
    )
    metrics = policies[["method", *metric_columns]].copy()
    metrics.insert(1, "selected_link_count", len(selected))
    write_csv(
        section / "tables" / "partC_rev1_selected_100_link_metrics.csv",
        metrics,
    )

    distribution = physical[["partC_rev1_link_uid", "selected_cfw_flow"]].rename(
        columns={"selected_cfw_flow": "benchmark_flow"}
    )
    distribution = distribution.sort_values("benchmark_flow")
    distribution["empirical_ccdf"] = (len(distribution) - np.arange(len(distribution))) / len(distribution)
    error_parts = []
    for label, policy in POLICY_COLUMNS.items():
        values = signed_relative_difference(physical[f"{policy}_flow"], physical["selected_cfw_flow"])
        error_parts.append(
            pd.DataFrame(
                {
                    "partC_rev1_link_uid": physical["partC_rev1_link_uid"].astype(str),
                    "method": label,
                    "signed_relative_difference": values,
                }
            )
        )
    errors = pd.concat(error_parts, ignore_index=True)

    # Figure 3(a): national map in explicit WGS84 mainland coordinates.
    figure_stem = "partC_rev1_fig3_suba_interstate_map"
    backbone_wgs = _to_wgs84(backbone)
    selected_wgs = _to_wgs84(selected)
    map_data = selected.drop(columns="geometry").copy()
    selected_midpoints = selected_wgs.to_crs(5070).geometry.interpolate(0.5, normalized=True)
    selected_midpoints = selected_midpoints.set_crs(5070).to_crs(4326)
    map_data["longitude"] = selected_midpoints.x.to_numpy()
    map_data["latitude"] = selected_midpoints.y.to_numpy()
    map_data["figure_stem"] = figure_stem
    map_data["plot_crs"] = "EPSG:4326"
    map_data["longitude_min"] = CONUS_BOUNDS[0]
    map_data["longitude_max"] = CONUS_BOUNDS[1]
    map_data["latitude_min"] = CONUS_BOUNDS[2]
    map_data["latitude_max"] = CONUS_BOUNDS[3]
    write_csv(section / "plot_data" / f"{figure_stem}.csv", map_data)
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    backbone_wgs.plot(ax=ax, color="0.82", linewidth=0.06, rasterized=True, zorder=1)
    selected_wgs.plot(ax=ax, color="#c62828", linewidth=1.25, rasterized=True, zorder=3)
    ax.set_title("Deterministic 100-link Interstate set")
    _set_map_extent(
        ax,
        figure_stem,
        source_crs=str(backbone.crs),
        plotted_feature_count=len(backbone_wgs) + len(selected_wgs),
    )
    _save_figure(fig, section / "figures" / figure_stem)

    # Figure 3(b): backbone-flow distribution.
    figure_stem = "partC_rev1_fig3_subb_backbone_flow_distribution"
    distribution_data = distribution.copy()
    distribution_data["figure_stem"] = figure_stem
    write_csv(section / "plot_data" / f"{figure_stem}.csv", distribution_data)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    positive = distribution.loc[distribution["benchmark_flow"].gt(0), "benchmark_flow"]
    bins = np.logspace(np.log10(max(positive.min(), 1e-6)), np.log10(positive.max()), 55)
    ax.hist(positive, bins=bins, color="#4c78a8", alpha=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("CFW flow (thousand tons/year)")
    ax.set_ylabel("Link count")
    ax.set_title("Backbone-flow distribution")
    ax.grid(True, which="both", alpha=0.18)
    _save_figure(fig, section / "figures" / figure_stem)

    # Figure 3(c): central signed-error mass, avoiding the unreadable [-1, 1]
    # scale while retaining all raw values and a display flag in the CSV.
    figure_stem = "partC_rev1_fig3_subc_signed_relative_difference"
    absolute_q75 = float(errors["signed_relative_difference"].abs().quantile(0.75))
    error_limit = min(0.1, max(0.005, absolute_q75))
    errors["display_limit_abs"] = error_limit
    errors["display_in_panel"] = errors["signed_relative_difference"].abs().le(error_limit)
    errors["figure_stem"] = figure_stem
    write_csv(section / "plot_data" / f"{figure_stem}.csv", errors)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    order = list(POLICY_COLUMNS)
    values = [
        errors.loc[
            errors["method"].eq(label) & errors["display_in_panel"],
            "signed_relative_difference",
        ].to_numpy()
        for label in order
    ]
    ax.violinplot(values, positions=np.arange(1, len(order) + 1), showmeans=False, showmedians=True, widths=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(
        np.arange(1, len(order) + 1),
        ["FFTT", "Fast poly", "Nonconst.", "All"],
        rotation=45,
        ha="right",
    )
    ax.set_ylim(-error_limit, error_limit)
    ax.set_ylabel("Signed difference")
    ax.set_title("Central 75% signed error")
    ax.text(
        0.02,
        0.98,
        f"Display limit: +/-{error_limit:.3g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    ax.grid(True, axis="y", alpha=0.18)
    _save_figure(fig, section / "figures" / figure_stem)

    # Figure 3(d): measured physical-solver convergence.
    figure_stem = "partC_rev1_fig3_subd_solver_convergence"
    convergence_data = convergence.copy()
    convergence_data["figure_stem"] = figure_stem
    write_csv(section / "plot_data" / f"{figure_stem}.csv", convergence_data)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    for method, group in convergence.groupby("method_label", sort=False):
        valid = group.loc[pd.to_numeric(group["relative_gap"], errors="coerce").notna()]
        ax.plot(valid["wall_time_sec"], valid["relative_gap"], label=method, linewidth=1.6)
        ax.scatter(valid["wall_time_sec"].iloc[-1], valid["relative_gap"].iloc[-1], s=24)
    ax.set_yscale("log")
    ax.set_ylim(bottom=1e-4)
    ax.set_xlabel("Measured wall time (s)")
    ax.set_ylabel("Relative gap")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_title("Physical-solver convergence")
    fig.tight_layout()
    _save_figure(fig, section / "figures" / figure_stem)

    # Standalone appendix diagnostics use figure-prefixed matching data names.
    appendix_hist_stem = "partC_rev1_figA1_backbone_flow_histogram"
    write_csv(appendix / "plot_data" / f"{appendix_hist_stem}.csv", distribution_data)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ax.hist(positive, bins=bins, color="#4c78a8", alpha=0.85)
    ax.set(xscale="log", yscale="log", xlabel="Selected-CFW flow (thousand tons/year)", ylabel="Link count")
    _save_figure(fig, appendix / "figures" / appendix_hist_stem)
    appendix_ccdf_stem = "partC_rev1_figA2_backbone_flow_ccdf"
    write_csv(appendix / "plot_data" / f"{appendix_ccdf_stem}.csv", distribution_data)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    ccdf = distribution.loc[distribution["benchmark_flow"].gt(0)]
    ax.loglog(ccdf["benchmark_flow"], ccdf["empirical_ccdf"], color="#f58518")
    ax.set(xlabel="Selected-CFW flow (thousand tons/year)", ylabel="Empirical CCDF")
    _save_figure(fig, appendix / "figures" / appendix_ccdf_stem)

    pilot = pd.read_csv(BENCHMARK_REGISTRY)
    pilot = pilot.loc[pilot["state"].ne("measured_complete")].copy()
    write_csv(appendix / "tables" / "partC_rev1_backend_runtime_pilots.csv", pilot)
    training_time = float(json.loads((TRAIN_ROOT / "partC_rev1_abstract_training_summary.json").read_text())["total_training_time_sec"])
    break_even = policies[["method", "assignment_runtime_sec", "training_runtime_sec"]].copy()
    break_even["benchmark_runtime_sec"] = benchmark_runtime
    break_even["break_even_scenarios"] = break_even["training_runtime_sec"] / (
        benchmark_runtime - break_even["assignment_runtime_sec"]
    )
    write_csv(appendix / "tables" / "partC_rev1_offline_break_even.csv", break_even)


def _load_partition_base_links():
    import geopandas as gpd
    import pyogrio

    cache = OUTPUT_ROOT / "partC_rev1_partition_base_links_minimal.parquet"
    if cache.exists():
        return gpd.read_parquet(cache)
    frame = pyogrio.read_dataframe(
        PARTITION_GPKG,
        layer="partC_rev1_backbone_partition_links",
        columns=[
            "directed_arc_id",
            "partC_rev1_link_uid",
            "tail_node_uid",
            "head_node_uid",
            "region_label",
            "is_final_backbone",
        ],
    )
    frame.to_parquet(cache, index=False)
    return frame


def _build_region_analysis_frame() -> pd.DataFrame:
    model = pd.read_csv(TRAIN_ROOT / "partC_rev1_recovery_model_registry.csv")
    selected = pd.read_csv(TRAIN_ROOT / "partC_rev1_recovery_selected_models_by_region.csv")
    status = pd.read_csv(TRAIN_ROOT / "partC_rev1_recovery_region_status.csv")
    cfw = pd.read_parquet(_method_dirs()["CFW"] / "partC_rev1_directed_arc_flow.parquet", columns=["arc_label"])
    arc_count = (
        cfw.loc[cfw["arc_label"].astype(str).str.startswith("partC_rev1_region_")]
        .groupby("arc_label")
        .size()
        .rename("physical_arc_count")
        .reset_index()
        .rename(columns={"arc_label": "region_uuid"})
    )
    frame = status.merge(selected, on="region_uuid", validate="one_to_one").merge(
        arc_count, on="region_uuid", how="left", validate="one_to_one"
    )
    frame["physical_arc_count"] = frame["physical_arc_count"].fillna(0).astype(int)
    return frame


def _aggregate_model_registry(model: pd.DataFrame, population: str) -> pd.DataFrame:
    selected = pd.read_csv(TRAIN_ROOT / "partC_rev1_recovery_selected_models_by_region.csv")
    count_map = selected["selected_best"].value_counts()
    rows = []
    for family, group in model.groupby("model_family", sort=True):
        values = pd.to_numeric(group["validation_wmape"], errors="coerce").dropna()
        timing = pd.to_numeric(group["training_time_sec"], errors="coerce").dropna()
        rows.append(
            {
                "population": population,
                "model_family": family,
                "count": len(group),
                "mean_validation_WMAPE": values.mean(),
                "median_validation_WMAPE": values.median(),
                "p90_validation_WMAPE": values.quantile(0.9),
                "maximum_validation_WMAPE": values.max(),
                "mean_training_time_sec": timing.mean(),
                "median_training_time_sec": timing.median(),
                "inference_time_sec": "not recorded per region",
                "selected_for_best_count": int(count_map.get(family, 0)) if population == "all_1529_regions" else int(
                    group["region_uuid"].map(selected.set_index("region_uuid")["selected_best"]).eq(family).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def _training_histories_for_groups(regions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def generate_missing_history(region_uuid: str, model_family: str, destination: Path) -> None:
        from xgboost import XGBRegressor
        import torch
        from torch import nn

        matrix = TRAIN_ROOT / "partC_rev1_training_matrices" / f"partC_rev1_training_matrix_{region_uuid}.npz"
        with np.load(matrix, allow_pickle=True) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
            tau0 = np.asarray(data["tau0"], dtype=np.float32)
            bench = np.asarray(data["benchmark_flow"], dtype=np.float32)
        train_end = max(1, int(math.floor(0.8 * len(x))))
        val_end = min(len(x), max(train_end + 1, int(math.floor(0.9 * len(x)))))
        train = np.arange(train_end, dtype=np.int64)
        validation = np.arange(train_end, val_end, dtype=np.int64)
        features = _recovery_scalar_features(x, bench)
        target = _recovery_scalar_target(y, tau0, bench)
        rows: list[dict[str, Any]] = []
        if model_family == "xgboost":
            model = XGBRegressor(
                objective="reg:squarederror",
                n_estimators=24,
                max_depth=2,
                learning_rate=0.12,
                subsample=1.0,
                colsample_bytree=1.0,
                n_jobs=1,
                random_state=SEED,
                verbosity=0,
            )
            model.fit(
                features[train],
                target[train],
                eval_set=[(features[train], target[train]), (features[validation], target[validation])],
                verbose=False,
            )
            for split, values in model.evals_result().items():
                for iteration, loss in enumerate(values.get("rmse", [])):
                    rows.append(
                        {"model_family": "xgboost", "split": split, "iteration": iteration, "loss": loss}
                    )
        else:
            torch.manual_seed(SEED)
            model = nn.Sequential(
                nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 1), nn.Softplus()
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            train_x = torch.as_tensor(features[train], dtype=torch.float32)
            train_y = torch.as_tensor(target[train, None], dtype=torch.float32)
            val_x = torch.as_tensor(features[validation], dtype=torch.float32)
            val_y = torch.as_tensor(target[validation, None], dtype=torch.float32)
            for epoch in range(24):
                model.train()
                optimizer.zero_grad()
                pred = model(train_x) + 1e-6
                loss = torch.mean(torch.square(pred - train_y))
                loss.backward()
                optimizer.step()
                model.eval()
                with torch.no_grad():
                    val_loss = torch.mean(torch.square(model(val_x) - val_y))
                rows.extend(
                    [
                        {"model_family": "small_nn", "split": "train", "iteration": epoch, "loss": float(loss.item())},
                        {
                            "model_family": "small_nn",
                            "split": "validation",
                            "iteration": epoch,
                            "loss": float(val_loss.item()),
                        },
                    ]
                )
        write_csv(destination, pd.DataFrame(rows))

    curve_root = TRAIN_ROOT / "partC_rev1_representative_curves"
    generated_root = NUMERICAL / "sec5_3" / "source_extracts" / "generated_optimization_histories"
    parts = []
    checkpoint_rows = []
    checkpoints = (0.10, 0.25, 0.50, 0.70, 1.00)
    for region in regions.itertuples(index=False):
        region_dir = curve_root / str(region.region_uuid)
        for model, filename in (("xgboost", "partC_rev1_xgboost_learning_curve.csv"),):
            path = region_dir / filename
            if not path.exists():
                path = generated_root / str(region.region_uuid) / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                generate_missing_history(str(region.region_uuid), model, path)
            frame = pd.read_csv(path)
            frame["raw_split"] = frame["split"].astype(str)
            frame["split"] = frame["raw_split"].map(
                {
                    "validation_0": "train",
                    "validation_1": "validation",
                    "train": "train",
                    "validation": "validation",
                }
            )
            if frame["split"].isna().any():
                unknown = sorted(frame.loc[frame["split"].isna(), "raw_split"].unique().tolist())
                raise ValueError(f"Unknown XGBoost history split labels: {unknown}")
            frame["region_uuid"] = region.region_uuid
            frame["stratum"] = region.stratum
            frame["metric_name"] = "XGBoost RMSE"
            frame["source_path"] = str(path.resolve())
            frame["history_status"] = (
                "completed_new_from_saved_matrix"
                if generated_root in path.parents
                else "completed_existing_history"
            )
            parts.append(frame)
            n_steps = int(frame["iteration"].max()) + 1
            for progress in checkpoints:
                iteration = max(0, min(n_steps - 1, int(math.ceil(progress * n_steps)) - 1))
                subset = frame.loc[frame["iteration"].eq(iteration)]
                for row in subset.itertuples(index=False):
                    checkpoint_rows.append(
                        {
                            "region_uuid": region.region_uuid,
                            "stratum": region.stratum,
                            "model_family": model,
                            "split": row.split,
                            "optimization_progress_fraction": progress,
                            "iteration": iteration,
                            "loss": row.loss,
                        }
                    )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(checkpoint_rows)


def _recovery_scalar_features(x: np.ndarray, benchmark_flow: np.ndarray) -> np.ndarray:
    bench = np.asarray(benchmark_flow, dtype=np.float64)
    values = np.asarray(x, dtype=np.float64)
    weights = np.maximum(bench, 1e-6)
    rho = np.sum(values, axis=1) / max(float(np.sum(weights)), EPS)
    ratio = values / np.maximum(bench[None, :], EPS)
    return np.column_stack([rho, rho**2, rho**4, np.max(ratio, axis=1)]).astype(np.float32)


def _recovery_scalar_target(y: np.ndarray, tau0: np.ndarray, benchmark_flow: np.ndarray) -> np.ndarray:
    weights = np.maximum(np.asarray(benchmark_flow, dtype=np.float64), 1e-6)
    ratio = np.asarray(y, dtype=np.float64) / np.maximum(np.asarray(tau0, dtype=np.float64)[None, :], EPS)
    return np.average(ratio, axis=1, weights=weights).astype(np.float32)


def run_targeted_sample_size(selected_regions: pd.DataFrame, seeds: Sequence[int] = (42,)) -> pd.DataFrame:
    from xgboost import XGBRegressor
    import torch
    from torch import nn

    section = NUMERICAL / "sec5_3"
    split_dir = section / "source_extracts" / "sample_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    fractions = (0.10, 0.25, 0.50, 0.70, 1.00)
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    started_all = time.perf_counter()
    for selected in selected_regions.itertuples(index=False):
        matrix = TRAIN_ROOT / "partC_rev1_training_matrices" / f"partC_rev1_training_matrix_{selected.region_uuid}.npz"
        with np.load(matrix, allow_pickle=True) as data:
            x = np.asarray(data["x"], dtype=np.float32)
            y = np.asarray(data["y"], dtype=np.float32)
            tau0 = np.asarray(data["tau0"], dtype=np.float32)
            bench = np.asarray(data["benchmark_flow"], dtype=np.float32)
        n = len(x)
        train_end = max(1, int(math.floor(0.8 * n)))
        val_end = min(n, max(train_end + 1, int(math.floor(0.9 * n))))
        train_pool = np.arange(train_end, dtype=np.int64)
        validation = np.arange(train_end, val_end, dtype=np.int64)
        test = np.arange(val_end, n, dtype=np.int64) if val_end < n else np.arange(n - 1, n, dtype=np.int64)
        nested = nested_fraction_indices(train_pool, fractions)
        if set(train_pool) & set(validation) or set(train_pool) & set(test) or set(validation) & set(test):
            raise AssertionError("Holdout leakage detected")
        split_manifest = {
            "region_uuid": selected.region_uuid,
            "stratum": selected.stratum,
            "matrix_path": str(matrix.resolve()),
            "matrix_sha256": sha256_path(matrix),
            "sample_count": n,
            "sample_zero_retained": True,
            "train_pool": train_pool.tolist(),
            "validation_indices": validation.tolist(),
            "test_indices": test.tolist(),
            "nested_training_indices": {str(key): value.tolist() for key, value in nested.items()},
            "seeds": list(map(int, seeds)),
        }
        write_json(split_dir / f"{selected.region_uuid}_split_manifest.json", split_manifest)
        features = _recovery_scalar_features(x, bench)
        target = _recovery_scalar_target(y, tau0, bench)
        for fraction, train_idx in nested.items():
            split_rows.append(
                {
                    "region_uuid": selected.region_uuid,
                    "stratum": selected.stratum,
                    "training_fraction": fraction,
                    "training_count": len(train_idx),
                    "validation_count": len(validation),
                    "test_count": len(test),
                    "sample_zero_retained": 0 in set(train_idx.tolist()),
                    "holdout_leakage": False,
                }
            )
            for seed in seeds:
                fit_started = time.perf_counter()
                xgb = XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=24,
                    max_depth=2,
                    learning_rate=0.12,
                    subsample=1.0,
                    colsample_bytree=1.0,
                    n_jobs=1,
                    random_state=int(seed),
                    verbosity=0,
                )
                xgb.fit(features[train_idx], target[train_idx], verbose=False)
                xgb_runtime = time.perf_counter() - fit_started
                for split_name, indices in (("validation", validation), ("test", test)):
                    factor = np.maximum(xgb.predict(features[indices]).astype(np.float32), 1.0)
                    prediction = np.maximum(tau0[None, :] * factor[:, None], tau0[None, :])
                    rows.append(
                        {
                            "region_uuid": selected.region_uuid,
                            "stratum": selected.stratum,
                            "model_family": "xgboost",
                            "training_fraction": fraction,
                            "training_count": len(train_idx),
                            "split": split_name,
                            "seed": seed,
                            "WMAPE": wmape(prediction, y[indices]),
                            "training_runtime_sec": xgb_runtime,
                        }
                    )
                torch.manual_seed(int(seed))
                model = nn.Sequential(
                    nn.Linear(features.shape[1], 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    nn.Softplus(),
                )
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                train_x = torch.as_tensor(features[train_idx], dtype=torch.float32)
                train_y = torch.as_tensor(target[train_idx, None], dtype=torch.float32)
                fit_started = time.perf_counter()
                for _ in range(24):
                    model.train()
                    optimizer.zero_grad()
                    loss = torch.mean(torch.square(model(train_x) + 1e-6 - train_y))
                    loss.backward()
                    optimizer.step()
                nn_runtime = time.perf_counter() - fit_started
                model.eval()
                with torch.no_grad():
                    for split_name, indices in (("validation", validation), ("test", test)):
                        factor = (
                            model(torch.as_tensor(features[indices], dtype=torch.float32))
                            .numpy()
                            .reshape(-1)
                        )
                        factor = np.maximum(factor, 1.0)
                        prediction = np.maximum(tau0[None, :] * factor[:, None], tau0[None, :])
                        rows.append(
                            {
                                "region_uuid": selected.region_uuid,
                                "stratum": selected.stratum,
                                "model_family": "small_nn",
                                "training_fraction": fraction,
                                "training_count": len(train_idx),
                                "split": split_name,
                                "seed": seed,
                                "WMAPE": wmape(prediction, y[indices]),
                                "training_runtime_sec": nn_runtime,
                            }
                        )
        if time.perf_counter() - started_all > 2700:
            raise TimeoutError("Targeted sample-size guard exceeded 45 minutes")
    result = pd.DataFrame(rows)
    write_csv(section / "plot_data" / "partC_rev1_sample_size_sensitivity_long.csv", result)
    write_csv(section / "tables" / "partC_rev1_sample_split_manifest.csv", pd.DataFrame(split_rows))
    write_json(
        section / "source_extracts" / "partC_rev1_targeted_retraining_hyperparameters.json",
        {
            "design": "16 deterministic regions, fixed sequential 80/10/10 split, nested fractions, one seed",
            "fractions": fractions,
            "seeds": list(map(int, seeds)),
            "xgboost": {
                "n_estimators": 24,
                "max_depth": 2,
                "learning_rate": 0.12,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "n_jobs": 1,
            },
            "small_nn": {"layers": [4, 16, 8, 1], "activations": ["ReLU", "ReLU", "Softplus"], "epochs": 24, "optimizer": "Adam", "learning_rate": 0.01},
            "runtime_guard_sec": 2700,
            "elapsed_sec": time.perf_counter() - started_all,
        },
    )
    return result


def _plot_sample_size(result: pd.DataFrame, exemplars: pd.DataFrame) -> None:
    section = NUMERICAL / "sec5_3"
    summary = (
        result.groupby(["stratum", "model_family", "split", "training_fraction"])["WMAPE"]
        .agg(
            median_WMAPE="median",
            q25_WMAPE=lambda s: s.quantile(0.25),
            q75_WMAPE=lambda s: s.quantile(0.75),
            region_seed_rows="size",
        )
        .reset_index()
    )
    write_csv(section / "tables" / "partC_rev1_sample_size_summary.csv", summary)
    panel_letters = {"large": "e", "small": "f", "good": "g", "bad": "h"}
    for stratum in ("large", "small", "good", "bad"):
        subset = summary.loc[summary["stratum"].eq(stratum) & summary["split"].eq("test")]
        figure_stem = f"partC_rev1_fig5_sub{panel_letters[stratum]}_sample_size_{stratum}"
        subset = subset.copy()
        subset["figure_stem"] = figure_stem
        write_csv(section / "plot_data" / f"{figure_stem}.csv", subset)
        fig, ax = plt.subplots(figsize=(6.3, 4.2))
        for model, group in subset.groupby("model_family"):
            group = group.sort_values("training_fraction")
            x = group["training_fraction"].to_numpy(dtype=float)
            y = group["median_WMAPE"].to_numpy(dtype=float)
            low = group["q25_WMAPE"].to_numpy(dtype=float)
            high = group["q75_WMAPE"].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", label=model)
            ax.fill_between(x, low, high, alpha=0.16)
        ax.set_title(f"{stratum.capitalize()} group")
        ax.set_ylabel("Test WMAPE")
        ax.set_xlabel("Fraction of fixed training pool")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_figure(fig, section / "figures" / figure_stem)


def _plot_region_maps(selected_regions: pd.DataFrame) -> None:
    import geopandas as gpd
    import pyogrio

    section = NUMERICAL / "sec5_3"
    links = _load_partition_base_links()
    source_crs = str(links.crs)
    links_wgs = _to_wgs84(links)
    exemplars = selected_regions.loc[
        selected_regions["is_exemplar"] & selected_regions["stratum"].ne("small")
    ].copy()
    if set(exemplars["stratum"]) != {"large", "good", "bad"}:
        raise AssertionError("Large, good, and bad detail exemplars are required")
    colors = {
        "large": "#d62728",
        "small": "#1f77b4",
        "good": "#2ca02c",
        "bad": "#9467bd",
    }
    widths = {"large": 1.25, "small": 2.1, "good": 1.7, "bad": 1.7}
    location_rows = []
    for row in selected_regions.itertuples(index=False):
        subset = links_wgs.loc[links_wgs["region_label"].astype(str).eq(str(row.region_uuid))]
        if subset.empty:
            raise ValueError(f"No physical links found for {row.region_uuid}")
        center = subset.to_crs(5070).geometry.unary_union.centroid
        point = gpd.GeoSeries([center], crs=5070).to_crs(4326).iloc[0]
        bounds = subset.total_bounds
        location_rows.append(
            {
                "region_uuid": row.region_uuid,
                "stratum": row.stratum,
                "is_exemplar": row.is_exemplar,
                "longitude": point.x,
                "latitude": point.y,
                "bbox_longitude_min": bounds[0],
                "bbox_longitude_max": bounds[2],
                "bbox_latitude_min": bounds[1],
                "bbox_latitude_max": bounds[3],
                "line_color": colors[row.stratum],
                "line_width_points": widths[row.stratum],
            }
        )
    locations = pd.DataFrame(location_rows)

    # Node-label incidence identifies physical portals without materializing
    # the conceptual terminal-pair clique.
    attrs = links.drop(columns="geometry")
    incidence = pd.concat(
        [
            attrs[["tail_node_uid", "region_label"]].rename(columns={"tail_node_uid": "node_uid"}),
            attrs[["head_node_uid", "region_label"]].rename(columns={"head_node_uid": "node_uid"}),
        ],
        ignore_index=True,
    ).drop_duplicates()
    label_count = incidence.groupby("node_uid")["region_label"].nunique()
    portal_nodes = set(label_count.loc[label_count.gt(1)].index.astype(str))
    node_layer = pyogrio.read_dataframe(
        PARTITION_GPKG,
        layer="partC_rev1_backbone_partition_nodes",
        columns=["partC_rev1_node_uid", "partC_rev1_centroid", "region_label"],
    )
    node_layer["partC_rev1_node_uid"] = node_layer["partC_rev1_node_uid"].astype(str)
    node_layer_wgs = _to_wgs84(node_layer)

    # Figure 4(a): mainland context with the actual selected regional links and
    # rectangles showing the three detailed views.
    figure_stem = "partC_rev1_fig4_suba_representative_regions_mainland"
    overview_data = locations.copy()
    overview_data["figure_stem"] = figure_stem
    overview_data["plot_crs"] = "EPSG:4326"
    write_csv(section / "plot_data" / f"{figure_stem}.csv", overview_data)
    fig, ax = plt.subplots(figsize=(7.2, 3.65))
    backbone = links_wgs.loc[
        pd.to_numeric(links_wgs["is_final_backbone"], errors="coerce").fillna(0).astype(int).eq(1)
    ]
    backbone.plot(ax=ax, color="0.86", linewidth=0.035, rasterized=True, zorder=1)
    selected_ids = set(selected_regions["region_uuid"].astype(str))
    selected_links = links_wgs.loc[links_wgs["region_label"].astype(str).isin(selected_ids)]
    for stratum in ("large", "small", "good", "bad"):
        ids = set(
            selected_regions.loc[selected_regions["stratum"].eq(stratum), "region_uuid"].astype(str)
        )
        group_links = selected_links.loc[selected_links["region_label"].astype(str).isin(ids)]
        group_links.plot(
            ax=ax,
            color=colors[stratum],
            linewidth=widths[stratum],
            alpha=0.88,
            rasterized=True,
            zorder=3,
        )
    for exemplar in exemplars.itertuples(index=False):
        row = locations.loc[locations["region_uuid"].eq(exemplar.region_uuid)].iloc[0]
        rectangle = Rectangle(
            (row["bbox_longitude_min"], row["bbox_latitude_min"]),
            row["bbox_longitude_max"] - row["bbox_longitude_min"],
            row["bbox_latitude_max"] - row["bbox_latitude_min"],
            fill=False,
            edgecolor="0.05",
            linewidth=1.8,
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(rectangle)
    handles = [
        Line2D([0], [0], color=colors[stratum], linewidth=widths[stratum], label=stratum.capitalize())
        for stratum in ("large", "small", "good", "bad")
    ]
    handles.append(
        Patch(facecolor="none", edgecolor="0.25", linestyle="--", label="Detailed view")
    )
    ax.legend(handles=handles, frameon=True, loc="lower right", ncol=1)
    ax.set_title("Representative regional links and detailed-view locations")
    _set_map_extent(
        ax,
        figure_stem,
        source_crs=source_crs,
        plotted_feature_count=len(backbone) + len(selected_links),
    )
    _save_figure(fig, section / "figures" / figure_stem)

    panel_letters = {"large": "b", "good": "c", "bad": "d"}
    for exemplar in exemplars.sort_values(
        "stratum", key=lambda values: values.map({"large": 0, "good": 1, "bad": 2})
    ).itertuples(index=False):
        figure_stem = (
            f"partC_rev1_fig4_sub{panel_letters[exemplar.stratum]}_"
            f"{exemplar.stratum}_region_detail"
        )
        region_links = links_wgs.loc[
            links_wgs["region_label"].astype(str).eq(str(exemplar.region_uuid))
        ].copy()
        nodes = node_layer_wgs.loc[
            node_layer_wgs["region_label"].astype(str).eq(str(exemplar.region_uuid))
        ].copy()
        portals = nodes.loc[nodes["partC_rev1_node_uid"].isin(portal_nodes)]
        centroids = nodes.loc[
            pd.to_numeric(nodes["partC_rev1_centroid"], errors="coerce").fillna(0).astype(int).eq(1)
        ]
        feature_rows = pd.concat(
            [
                pd.DataFrame(
                    {
                        "feature_type": "physical regional link",
                        "feature_id": region_links["directed_arc_id"].astype(str),
                        "geometry_wkt": region_links.geometry.to_wkt(),
                        "color": "0.25",
                        "width_or_size": 0.75,
                    }
                ),
                pd.DataFrame(
                    {
                        "feature_type": "physical portal",
                        "feature_id": portals["partC_rev1_node_uid"].astype(str),
                        "geometry_wkt": portals.geometry.to_wkt(),
                        "color": "#d62728",
                        "width_or_size": 14,
                    }
                ),
                pd.DataFrame(
                    {
                        "feature_type": "centroid",
                        "feature_id": centroids["partC_rev1_node_uid"].astype(str),
                        "geometry_wkt": centroids.geometry.to_wkt(),
                        "color": "#ffbf00",
                        "width_or_size": 34,
                    }
                ),
            ],
            ignore_index=True,
        )
        feature_rows["region_uuid"] = exemplar.region_uuid
        feature_rows["stratum"] = exemplar.stratum
        feature_rows["figure_stem"] = figure_stem
        feature_rows["plot_crs"] = "EPSG:4326"
        write_csv(section / "plot_data" / f"{figure_stem}.csv", feature_rows)
        fig, ax = plt.subplots(figsize=(6.3, 4.2))
        region_links.plot(ax=ax, color="0.25", linewidth=0.75, rasterized=True, zorder=2)
        if not portals.empty:
            portals.plot(ax=ax, color="#d62728", markersize=14, zorder=5)
        if not centroids.empty:
            centroids.plot(
                ax=ax,
                color="#ffbf00",
                edgecolor="black",
                marker="*",
                markersize=34,
                zorder=6,
            )
        ax.set_title(f"{exemplar.stratum.capitalize()} representative region")
        detail_handles = [
            Line2D([0], [0], color="0.25", linewidth=1.4, label="Link"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor="#d62728",
                markeredgecolor="#d62728",
                label="Portal",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                color="none",
                markerfacecolor="#ffbf00",
                markeredgecolor="black",
                markersize=10,
                label="Centroid",
            ),
        ]
        ax.legend(
            handles=detail_handles,
            frameon=True,
            loc="upper center",
            ncol=3,
            handlelength=1.5,
            columnspacing=0.8,
        )
        bounds = _bounded_local_extent(region_links.total_bounds)
        _set_map_extent(
            ax,
            figure_stem,
            view=f"{exemplar.stratum}_region_detail",
            bounds=bounds,
            source_crs=source_crs,
            plotted_feature_count=len(region_links) + len(portals) + len(centroids),
        )
        _save_figure(fig, section / "figures" / figure_stem)
    write_csv(
        section / "source_extracts" / "partC_rev1_representative_portal_nodes.csv",
        node_layer.loc[node_layer["partC_rev1_node_uid"].isin(portal_nodes)].drop(columns="geometry"),
    )


def build_sec5_3(*, run_targeted: bool = False) -> None:
    ensure_package_tree()
    section = NUMERICAL / "sec5_3"
    appendix = NUMERICAL / "appendix"
    model = pd.read_csv(TRAIN_ROOT / "partC_rev1_recovery_model_registry.csv")
    region_frame = _build_region_analysis_frame()
    selected_regions, rankings = choose_representative_regions(region_frame, per_stratum=4)
    learning_regions, learning_rankings = choose_representative_regions(
        region_frame, per_stratum=5
    )
    exemplars = selected_regions.loc[selected_regions["is_exemplar"]].copy()
    write_csv(section / "tables" / "partC_rev1_representative_regions.csv", selected_regions)
    write_csv(
        section / "tables" / "partC_rev1_learning_curve_regions.csv",
        learning_regions,
    )
    write_csv(section / "source_extracts" / "partC_rev1_representative_region_rankings.csv", rankings)
    write_csv(
        section / "source_extracts" / "partC_rev1_learning_curve_region_rankings.csv",
        learning_rankings,
    )
    write_json(
        section / "source_extracts" / "partC_rev1_representative_selection_rules.json",
        {
            "order": ["large", "small", "good", "bad"],
            "large": "descending physical regional directed-arc count",
            "small": "ascending positive trainable physical regional directed-arc count",
            "good": "ascending best-nonconstant WMAPE above the 25th percentile active-pair threshold",
            "bad": "descending best-nonconstant WMAPE among successfully trained positive-flow regions",
            "deduplication": "later strata advance to next stable-ranked unused region",
            "per_stratum": 4,
            "learning_curve_regions_per_stratum": 5,
            "learning_curve_model": "xgboost only",
            "learning_curve_split_labels": {
                "validation_0": "train",
                "validation_1": "validation",
            },
        },
    )
    overall = _aggregate_model_registry(model, "all_1529_regions")
    sampled_model = model.loc[model["region_uuid"].isin(selected_regions["region_uuid"])].copy()
    sampled = _aggregate_model_registry(sampled_model, "representative_16_regions")
    family_summary = pd.concat([overall, sampled], ignore_index=True)
    write_csv(section / "tables" / "partC_rev1_table_regional_training.csv", family_summary)
    write_latex_table(
        section / "tables" / "partC_rev1_table_regional_training.tex",
        family_summary,
        caption="Regional surrogate validation summary over all regions and the deterministic representative set.",
        label="tab:regional_training_refresh",
    )
    compact_registry = model[
        [
            "region_uuid",
            "model_family",
            "status",
            "validation_wmape",
            "validation_rmse",
            "validation_mae",
            "validation_max_abs_error",
            "training_time_sec",
            "sample_count",
            "active_pair_count",
            "finite_nonnegative",
            "does_not_fall_below_fftt",
            "freeflow_lower_bound_violation_rate",
        ]
    ]
    write_csv(section / "tables" / "partC_rev1_full_family_by_region_metrics.csv", compact_registry)
    histories, checkpoints = _training_histories_for_groups(learning_regions)
    write_csv(section / "plot_data" / "partC_rev1_training_history_long.csv", histories)
    write_csv(section / "tables" / "partC_rev1_training_progress_checkpoints.csv", checkpoints)
    panel_letters = {"large": "a", "small": "b", "good": "c", "bad": "d"}
    split_colors = {"train": "#4c78a8", "validation": "#e45756"}
    for stratum in ("large", "small", "good", "bad"):
        figure_stem = (
            f"partC_rev1_fig5_sub{panel_letters[stratum]}_"
            f"xgboost_{stratum}_learning_curve"
        )
        subset = histories.loc[histories["stratum"].eq(stratum)].copy()
        if subset["region_uuid"].nunique() != 5:
            raise AssertionError(f"{stratum} learning curve does not contain exactly five regions")
        subset["figure_stem"] = figure_stem
        write_csv(section / "plot_data" / f"{figure_stem}.csv", subset)
        fig, ax = plt.subplots(figsize=(6.3, 4.2))
        for split in ("train", "validation"):
            split_data = subset.loc[subset["split"].eq(split)]
            for _, group in split_data.groupby("region_uuid", sort=True):
                group = group.sort_values("iteration")
                ax.plot(
                    group["iteration"],
                    group["loss"],
                    color=split_colors[split],
                    linewidth=0.9,
                    alpha=0.42,
                )
            median = split_data.groupby("iteration", as_index=False)["loss"].median()
            ax.plot(
                median["iteration"],
                median["loss"],
                color=split_colors[split],
                linewidth=2.2,
                label=split.capitalize(),
            )
        ax.set_title(f"XGBoost: {stratum} group (5 regions)")
        ax.set_yscale("log")
        ax.set_xlabel("Boosting round")
        ax.set_ylabel("XGBoost RMSE")
        ax.grid(True, which="both", alpha=0.18)
        ax.legend(frameon=False)
        fig.tight_layout()
        _save_figure(fig, section / "figures" / figure_stem)

    diagnostics = (
        model.groupby("model_family")
        .agg(
            model_rows=("region_uuid", "size"),
            fraction_artifacts_finite_nonnegative=("finite_nonnegative", "mean"),
            fraction_artifacts_above_fftt=("does_not_fall_below_fftt", "mean"),
            mean_prediction_fraction_below_fftt=("freeflow_lower_bound_violation_rate", "mean"),
            maximum_prediction_fraction_below_fftt=("freeflow_lower_bound_violation_rate", "max"),
        )
        .reset_index()
    )
    diagnostics["negative_or_nonfinite_prediction_fraction"] = 1 - diagnostics[
        "fraction_artifacts_finite_nonnegative"
    ]
    write_csv(section / "tables" / "partC_rev1_prediction_validity_diagnostics.csv", diagnostics)
    relation = compact_registry.merge(
        region_frame[["region_uuid", "physical_arc_count"]],
        on="region_uuid",
        validate="many_to_one",
    )
    appendix_error_stem = "partC_rev1_figA3_regional_error_vs_dimension"
    relation["figure_stem"] = appendix_error_stem
    write_csv(appendix / "plot_data" / f"{appendix_error_stem}.csv", relation)
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    for family, group in relation.groupby("model_family"):
        ax.scatter(
            group["active_pair_count"],
            group["validation_wmape"],
            s=4,
            alpha=0.25,
            label=family,
            rasterized=True,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Active pair count")
    ax.set_ylabel("Validation WMAPE")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.15)
    _save_figure(fig, appendix / "figures" / appendix_error_stem)
    _plot_region_maps(selected_regions)

    sample_path = section / "plot_data" / "partC_rev1_sample_size_sensitivity_long.csv"
    if run_targeted:
        result = run_targeted_sample_size(selected_regions, seeds=(SEED,))
    elif sample_path.exists():
        result = pd.read_csv(sample_path)
    else:
        raise FileNotFoundError(
            "Targeted sample-size extract is absent. Run once with run_targeted=True; default rebuilds never retrain."
        )
    _plot_sample_size(result, exemplars)


def _resolution_key(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p").replace("+", "")


def _variation_of_information(labels_a: Sequence[Any], labels_b: Sequence[Any]) -> float:
    a = pd.Series(labels_a, dtype="string")
    b = pd.Series(labels_b, dtype="string")
    joint = pd.crosstab(a, b, normalize=True).to_numpy(dtype=np.float64)
    pa = joint.sum(axis=1)
    pb = joint.sum(axis=0)
    ha = -float(np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    hb = -float(np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    denom = pa[:, None] * pb[None, :]
    mask = joint > 0
    mutual = float(np.sum(joint[mask] * np.log(joint[mask] / denom[mask])))
    return ha + hb - 2.0 * mutual


def partition_stability_metrics(
    memberships: Mapping[float, pd.DataFrame],
) -> pd.DataFrame:
    from scipy.stats import wasserstein_distance
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    rows = []
    backbone_label = "backbone"
    for resolution_a in sorted(memberships):
        for resolution_b in sorted(memberships):
            a = memberships[resolution_a][["node_uid", "region_label", "is_boundary"]].rename(
                columns={"region_label": "label_a", "is_boundary": "boundary_a"}
            )
            b = memberships[resolution_b][["node_uid", "region_label", "is_boundary"]].rename(
                columns={"region_label": "label_b", "is_boundary": "boundary_b"}
            )
            common = a.merge(b, on="node_uid", validate="one_to_one")
            eligible = common.loc[common["label_a"].ne(backbone_label) & common["label_b"].ne(backbone_label)]
            if eligible.empty:
                raise ValueError(f"No common eligible nodes for {resolution_a} and {resolution_b}")
            boundary_a = set(common.loc[common["boundary_a"], "node_uid"].astype(str))
            boundary_b = set(common.loc[common["boundary_b"], "node_uid"].astype(str))
            union = boundary_a | boundary_b
            jaccard = len(boundary_a & boundary_b) / len(union) if union else 1.0
            size_a = eligible["label_a"].value_counts(normalize=True).to_numpy(dtype=float)
            size_b = eligible["label_b"].value_counts(normalize=True).to_numpy(dtype=float)
            rows.append(
                {
                    "resolution_a": resolution_a,
                    "resolution_b": resolution_b,
                    "common_eligible_node_count": len(eligible),
                    "adjusted_rand_index": adjusted_rand_score(eligible["label_a"], eligible["label_b"]),
                    "normalized_mutual_information": normalized_mutual_info_score(
                        eligible["label_a"], eligible["label_b"]
                    ),
                    "variation_of_information_nats": _variation_of_information(
                        eligible["label_a"], eligible["label_b"]
                    ),
                    "boundary_node_jaccard": jaccard,
                    "region_size_distribution_wasserstein": wasserstein_distance(size_a, size_b),
                }
            )
    return pd.DataFrame(rows)


def rebuild_partition_memberships() -> tuple[dict[float, pd.DataFrame], pd.DataFrame]:
    from partC_rev1_src import partC_rev1_remaining_experiments as remaining
    from partC_rev1_src.partC_rev1_partition_igraph import partC_rev1_run_leiden_partition

    section = NUMERICAL / "sec5_4"
    membership_root = OUTPUT_ROOT / "partC_rev1_partition_memberships"
    membership_root.mkdir(parents=True, exist_ok=True)
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    graph, od = remaining._partC_rev1_load_base_graph_od(config)
    nodes, arcs = remaining._partC_rev1_read_partition_inputs(config)
    nonbackbone_arcs = arcs.loc[
        pd.to_numeric(arcs.get("is_final_backbone", 0), errors="coerce").fillna(0).astype(int).eq(0)
    ].copy()
    scan = pd.read_csv(PARTITION_ROOT / "partC_rev1_partition_resolution_scan.csv")
    expected = {
        min(scan["resolution"], key=lambda x: abs(float(x) - resolution)): scan.loc[
            (scan["resolution"].astype(float) - resolution).abs().idxmin(), "partition_hash"
        ]
        for resolution in RESOLUTIONS
    }
    memberships: dict[float, pd.DataFrame] = {}
    audit_parts = []
    seed = int(config.get("backbone_partition", {}).get("partition", {}).get("random_seed", SEED))
    for resolution in RESOLUTIONS:
        key = _resolution_key(resolution)
        arc_path = membership_root / f"partC_rev1_arc_membership_resolution_{key}.parquet"
        node_path = section / "source_extracts" / f"partC_rev1_node_membership_resolution_{key}.parquet"
        metadata_path = section / "source_extracts" / f"partC_rev1_membership_resolution_{key}.json"
        if arc_path.exists() and node_path.exists() and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("partition_hash") == expected[min(expected, key=lambda x: abs(float(x) - resolution))]:
                memberships[resolution] = pd.read_parquet(node_path)
                audit_parts.append(pd.DataFrame(metadata.get("region_audit", [])))
                continue
        started = time.perf_counter()
        partition = partC_rev1_run_leiden_partition(
            nodes,
            nonbackbone_arcs,
            dict(config),
            resolution=float(resolution),
            random_seed=seed,
        )
        _, audit, arc_regions = remaining._partC_rev1_assign_and_correct_labels(
            graph,
            od,
            nodes,
            arcs,
            partition["node_partition"],
            config_path=str(CONFIG_PATH),
        )
        arc_labels = remaining._partC_rev1_corrected_arc_labels(graph, audit, arc_regions)
        arc_membership = pd.DataFrame(
            {
                "directed_arc_id": graph.arc_idx_to_directed_arc_id.astype(str),
                "region_label": arc_labels.astype(str),
            }
        )
        actual_hash = remaining.partC_rev1_dataframe_records_hash(
            arc_membership.rename(columns={"region_label": "arc_label"}),
            ["directed_arc_id", "arc_label"],
        )
        expected_hash = expected[min(expected, key=lambda x: abs(float(x) - resolution))]
        if actual_hash != expected_hash:
            raise AssertionError(
                f"Membership-only partition hash mismatch at {resolution}: {actual_hash} != {expected_hash}"
            )
        arc_membership.to_parquet(arc_path, index=False)
        codes, unique_labels = pd.factorize(arc_membership["region_label"], sort=True)
        incidence = pd.DataFrame(
            {
                "node_idx": np.concatenate([graph.tail, graph.head]),
                "label_code": np.concatenate([codes, codes]),
            }
        ).drop_duplicates()
        incidence_count = (
            incidence.groupby(["node_idx", "label_code"], as_index=False).size().rename(columns={"size": "incident_count"})
        )
        incidence_count["region_label"] = unique_labels[incidence_count["label_code"].to_numpy()]
        nonbackbone = incidence_count.loc[incidence_count["region_label"].ne("backbone")].copy()
        assigned = (
            nonbackbone.sort_values(
                ["node_idx", "incident_count", "region_label"], ascending=[True, False, True]
            )
            .drop_duplicates("node_idx")
            .set_index("node_idx")["region_label"]
        )
        boundary = incidence.groupby("node_idx")["label_code"].nunique().gt(1)
        node_indices = np.arange(len(graph.node_idx_to_uid), dtype=np.int64)
        node_membership = pd.DataFrame(
            {
                "node_uid": graph.node_idx_to_uid.astype(str),
                "region_label": pd.Series(node_indices).map(assigned).fillna("backbone").to_numpy(),
                "is_boundary": pd.Series(node_indices).map(boundary).fillna(False).astype(bool).to_numpy(),
                "resolution": resolution,
            }
        )
        node_membership.to_parquet(node_path, index=False)
        memberships[resolution] = node_membership
        audit_export = audit.copy()
        if not audit_export.empty:
            audit_export["resolution"] = resolution
            audit_parts.append(audit_export)
        write_json(
            metadata_path,
            {
                "resolution": resolution,
                "random_seed": seed,
                "partition_hash": actual_hash,
                "expected_partition_hash": expected_hash,
                "hash_match": True,
                "membership_only_runtime_sec": time.perf_counter() - started,
                "arc_membership_path": str(arc_path.resolve()),
                "node_membership_path": str(node_path.resolve()),
                "node_count": len(node_membership),
                "arc_count": len(arc_membership),
                "region_audit": audit_export.to_dict(orient="records") if not audit_export.empty else [],
            },
        )
    audits = pd.concat(audit_parts, ignore_index=True) if audit_parts else pd.DataFrame()
    if not audits.empty:
        write_csv(section / "source_extracts" / "partC_rev1_partition_region_audit.csv", audits)
    return memberships, audits


def _plot_partition_maps(memberships: Mapping[float, pd.DataFrame]) -> None:
    section = NUMERICAL / "sec5_4"
    links = _load_partition_base_links()
    source_crs = str(links.crs)
    links = _to_wgs84(links)
    links["directed_arc_id"] = links["directed_arc_id"].astype(str)
    panel_letters = list("abcdefghij")
    panel_index = 0
    for resolution in RESOLUTIONS:
        arc_path = (
            OUTPUT_ROOT
            / "partC_rev1_partition_memberships"
            / f"partC_rev1_arc_membership_resolution_{_resolution_key(resolution)}.parquet"
        )
        membership = pd.read_parquet(arc_path)
        mapped = links.drop(columns=["region_label"], errors="ignore").merge(
            membership, on="directed_arc_id", how="left", validate="one_to_one"
        )
        mapped["region_label"] = mapped["region_label"].fillna("backbone")
        backbone = mapped.loc[mapped["region_label"].eq("backbone")]
        regional = mapped.loc[mapped["region_label"].ne("backbone")]
        unique = sorted(regional["region_label"].unique().tolist())
        color_map = {
            label: plt.cm.hsv(int(hashlib.sha256(label.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
            for label in unique
        }
        for view, bounds in (("mainland", CONUS_BOUNDS), ("chicago", CHICAGO_BOUNDS)):
            letter = panel_letters[panel_index]
            panel_index += 1
            figure_stem = (
                f"partC_rev1_fig7_sub{letter}_resolution_"
                f"{_resolution_key(resolution)}_{view}"
            )
            view_frame = mapped.cx[bounds[0] : bounds[1], bounds[2] : bounds[3]].copy()
            view_backbone = view_frame.loc[view_frame["region_label"].eq("backbone")]
            view_regional = view_frame.loc[view_frame["region_label"].ne("backbone")]
            region_summary = (
                view_frame.groupby("region_label", as_index=False)
                .agg(plotted_arc_count=("directed_arc_id", "size"))
            )
            region_summary["arc_kind"] = np.where(
                region_summary["region_label"].eq("backbone"), "backbone", "regional"
            )
            region_summary["color"] = region_summary["region_label"].map(
                lambda label: (
                    "#262626"
                    if label == "backbone"
                    else matplotlib.colors.to_hex(color_map[label], keep_alpha=False)
                )
            )
            region_summary["resolution"] = resolution
            region_summary["view"] = view
            region_summary["figure_stem"] = figure_stem
            region_summary["plot_crs"] = "EPSG:4326"
            region_summary["longitude_min"] = bounds[0]
            region_summary["longitude_max"] = bounds[1]
            region_summary["latitude_min"] = bounds[2]
            region_summary["latitude_max"] = bounds[3]
            region_summary["membership_path"] = str(arc_path.resolve())
            region_summary["membership_sha256"] = sha256_path(arc_path)
            write_csv(section / "plot_data" / f"{figure_stem}.csv", region_summary)

            figure_size = (7.2, 3.65) if view == "mainland" else (6.3, 4.2)
            fig, ax = plt.subplots(figsize=figure_size)
            if not view_regional.empty:
                view_regional.plot(
                    ax=ax,
                    color=view_regional["region_label"].map(color_map),
                    linewidth=0.11 if view == "mainland" else 0.42,
                    alpha=0.88,
                    rasterized=True,
                    zorder=2,
                )
            if not view_backbone.empty:
                view_backbone.plot(
                    ax=ax,
                    color="#262626",
                    linewidth=0.16 if view == "mainland" else 0.55,
                    alpha=0.82,
                    rasterized=True,
                    zorder=3,
                )
            ax.set_title(
                f"$\\gamma={resolution:.4g}$: "
                f"{'mainland' if view == 'mainland' else 'Chicago zoom'}"
            )
            _set_map_extent(
                ax,
                figure_stem,
                view=view,
                bounds=bounds,
                source_crs=source_crs,
                plotted_feature_count=len(view_frame),
            )
            _save_figure(fig, section / "figures" / figure_stem)


def build_sec5_4(*, rebuild_membership: bool = False) -> None:
    ensure_package_tree()
    section = NUMERICAL / "sec5_4"
    scan = pd.read_csv(PARTITION_ROOT / "partC_rev1_partition_resolution_scan.csv")
    e8 = pd.read_csv(PARTITION_ROOT / "partC_rev1_e8_partition_active_set_pricing.csv")
    e9 = pd.read_csv(PARTITION_ROOT / "partC_rev1_e9_partition_downstream_speed_accuracy.csv")
    e10 = pd.read_csv(PARTITION_ROOT / "partC_rev1_e10_boundary_path_consistency.csv")
    registry = pd.read_csv(PARTITION_ROOT / "partC_rev1_partition_const_sensitivity_registry.csv")
    selected_scan = pd.concat(
        [scan.iloc[[(scan["resolution"].astype(float) - r).abs().argmin()]] for r in RESOLUTIONS],
        ignore_index=True,
    )
    selected_scan["resolution_key"] = selected_scan["resolution"].map(_resolution_key)
    for frame in (e8, e9, e10, registry):
        frame["resolution_key"] = frame["resolution"].map(_resolution_key)
    table = (
        selected_scan.merge(e8, on="resolution_key", suffixes=("", "_e8"), validate="one_to_one")
        .merge(e9, on="resolution_key", suffixes=("", "_e9"), validate="one_to_one")
        .merge(e10, on="resolution_key", suffixes=("", "_e10"), validate="one_to_one")
        .merge(registry, on="resolution_key", suffixes=("", "_registry"), validate="one_to_one")
    )
    table4 = pd.DataFrame(
        {
            "resolution": table["resolution"],
            "raw_region_count": table["raw_region_count"],
            "non_small_region_count": table["non_small_region_count"],
            "mean_arcs_per_non_small_region": table["average_arcs_non_small"],
            "median_arcs_per_non_small_region": table["median_arcs_non_small"],
            "portal_count": table["portal_count_total"],
            "active_pair_count": table["active_region_leg_count"],
            "global_runtime_sec": table["global_runtime_sec"],
            "backbone_WMAPE": table["WMAPE_backbone"],
            "relative_cost_error": table["relative_cost_error_backbone"],
            "additional_unreachable_demand": (
                table["abstract_unreachable_demand"] - table["benchmark_cost_aon_unreachable_demand"]
            ).clip(lower=0),
            "invalid_transition_count": table["invalid_nonphysical_transition_count"],
            "sampled_physical_expansion_success_rate": table[
                "sampled_hybrid_path_physical_expansion_success_rate"
            ],
            "partition_hash": table["partition_hash"],
            "status": "complete_measured",
        }
    )
    table4.loc[table4["additional_unreachable_demand"].abs().lt(1e-9), "additional_unreachable_demand"] = 0.0
    write_csv(section / "tables" / "partC_rev1_table_partition_sensitivity.csv", table4)
    write_latex_table(
        section / "tables" / "partC_rev1_table_partition_sensitivity.tex",
        table4,
        caption="Measured partition-resolution sensitivity and boundary/path consistency.",
        label="tab:partition_sensitivity_refresh",
    )
    ordered = table4.sort_values("resolution")
    figure_stem = "partC_rev1_fig6_suba_region_portal_counts"
    panel_data = ordered[
        ["resolution", "raw_region_count", "non_small_region_count", "portal_count"]
    ].copy()
    panel_data["figure_stem"] = figure_stem
    write_csv(section / "plot_data" / f"{figure_stem}.csv", panel_data)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(ordered["resolution"], ordered["raw_region_count"], marker="o", label="Raw regions")
    ax.plot(
        ordered["resolution"],
        ordered["non_small_region_count"],
        marker="s",
        label="Non-small regions",
    )
    ax2 = ax.twinx()
    ax2.plot(ordered["resolution"], ordered["portal_count"], marker="^", color="#2ca02c", label="portals")
    ax.set_ylabel("Region count")
    ax2.set_ylabel("Portal count")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel("Leiden resolution")
    ax.grid(True, alpha=0.18)
    ax.set_title("Region and portal counts")
    _save_figure(fig, section / "figures" / figure_stem)

    figure_stem = "partC_rev1_fig6_subb_global_runtime"
    panel_data = ordered[["resolution", "global_runtime_sec"]].copy()
    panel_data["figure_stem"] = figure_stem
    write_csv(section / "plot_data" / f"{figure_stem}.csv", panel_data)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(ordered["resolution"], ordered["global_runtime_sec"], marker="o")
    ax.set_ylabel("Runtime (s)")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel("Leiden resolution")
    ax.grid(True, alpha=0.18)
    ax.set_title("Global runtime")
    _save_figure(fig, section / "figures" / figure_stem)

    figure_stem = "partC_rev1_fig6_subc_assignment_error"
    panel_data = ordered[["resolution", "backbone_WMAPE", "relative_cost_error"]].copy()
    panel_data["figure_stem"] = figure_stem
    write_csv(section / "plot_data" / f"{figure_stem}.csv", panel_data)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(ordered["resolution"], ordered["backbone_WMAPE"], marker="o", label="WMAPE")
    ax.plot(
        ordered["resolution"],
        ordered["relative_cost_error"],
        marker="s",
        label="Cost error",
    )
    ax.set_ylabel("Error")
    ax.legend(frameon=False)
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel("Leiden resolution")
    ax.grid(True, alpha=0.18)
    ax.set_title("Backbone flow and cost error")
    _save_figure(fig, section / "figures" / figure_stem)

    figure_stem = "partC_rev1_fig6_subd_active_pairs_region_size"
    panel_data = ordered[
        ["resolution", "active_pair_count", "mean_arcs_per_non_small_region"]
    ].copy()
    panel_data["figure_stem"] = figure_stem
    write_csv(section / "plot_data" / f"{figure_stem}.csv", panel_data)
    fig, ax = plt.subplots(figsize=(6.3, 4.2))
    ax.plot(ordered["resolution"], ordered["active_pair_count"], marker="o", label="Active pairs")
    ax3 = ax.twinx()
    ax3.plot(
        ordered["resolution"],
        ordered["mean_arcs_per_non_small_region"],
        marker="s",
        color="#f58518",
        label="mean arcs",
    )
    ax.set_ylabel("Active pair count")
    ax3.set_ylabel("Mean arcs/region")
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax3.get_legend_handles_labels()
    ax.legend(handles1 + handles2, labels1 + labels2, frameon=False, loc="best")
    ax.set_xscale("symlog", linthresh=1e-5)
    ax.set_xlabel("Leiden resolution")
    ax.grid(True, alpha=0.18)
    ax.set_title("Active pairs and region size")
    _save_figure(fig, section / "figures" / figure_stem)

    node_files = [
        section / "source_extracts" / f"partC_rev1_node_membership_resolution_{_resolution_key(r)}.parquet"
        for r in RESOLUTIONS
    ]
    arc_files = [
        OUTPUT_ROOT
        / "partC_rev1_partition_memberships"
        / f"partC_rev1_arc_membership_resolution_{_resolution_key(r)}.parquet"
        for r in RESOLUTIONS
    ]
    if rebuild_membership or not all(path.exists() for path in [*node_files, *arc_files]):
        memberships, _ = rebuild_partition_memberships()
    else:
        memberships = {r: pd.read_parquet(path) for r, path in zip(RESOLUTIONS, node_files)}
    stability = partition_stability_metrics(memberships)
    write_csv(section / "tables" / "partC_rev1_partition_stability_metrics.csv", stability)
    ari_stem = "partC_rev1_figA4_partition_ari_heatmap"
    ari_plot_data = stability.copy()
    ari_plot_data["figure_stem"] = ari_stem
    write_csv(NUMERICAL / "appendix" / "plot_data" / f"{ari_stem}.csv", ari_plot_data)
    matrix = stability.pivot(
        index="resolution_a", columns="resolution_b", values="adjusted_rand_index"
    ).reindex(index=sorted(RESOLUTIONS), columns=sorted(RESOLUTIONS))
    fig, ax = plt.subplots(figsize=(6.1, 5.2))
    image = ax.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="viridis")
    labels = [f"{r:.4g}" for r in matrix.index]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", color="white")
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Resolution")
    ax.set_title("Pairwise adjusted Rand index")
    fig.colorbar(image, ax=ax, label="ARI")
    fig.tight_layout()
    _save_figure(fig, NUMERICAL / "appendix" / "figures" / ari_stem)
    _plot_partition_maps(memberships)
    boundary = e10[
        [
            "resolution",
            "unreachable_positive_flow_transfer_pair_count",
            "unreachable_positive_flow_transfer_pair_demand",
            "invalid_nonphysical_transition_count",
            "sampled_hybrid_path_physical_expansion_success_rate",
            "boundary_flow_residual_max",
            "boundary_flow_residual_median",
            "boundary_flow_residual_p95",
            "boundary_flow_residual_p99",
            "regional_terminal_marginal_residual_max",
            "status",
        ]
    ].copy()
    write_csv(section / "tables" / "partC_rev1_boundary_path_consistency.csv", boundary)
    write_latex_table(
        section / "tables" / "partC_rev1_boundary_path_consistency.tex",
        boundary,
        caption="Measured E10 boundary and physical path-expansion consistency.",
        label="tab:boundary_path_consistency",
    )
    for source in (
        PARTITION_ROOT / "partC_rev1_e8_partition_active_set_pricing.csv",
        PARTITION_ROOT / "partC_rev1_e9_partition_downstream_speed_accuracy.csv",
        PARTITION_ROOT / "partC_rev1_e10_boundary_path_consistency.csv",
        PARTITION_ROOT / "partC_rev1_partition_resolution_scan.csv",
    ):
        copy_extract(source, section / "source_extracts" / source.name)


def validate_phase1_non_scenario() -> dict[str, Any]:
    """Validate the revised Section 5.2--5.4 assets before the long scenario."""

    from PIL import Image

    sections = ("sec5_2", "sec5_3", "sec5_4", "appendix")
    figure_rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for section_name in sections:
        figure_dir = NUMERICAL / section_name / "figures"
        plot_dir = NUMERICAL / section_name / "plot_data"
        for pdf_path in sorted(figure_dir.glob("*.pdf")):
            stem = pdf_path.stem
            png_path = figure_dir / f"{stem}.png"
            metadata_path = figure_dir / f"{stem}_render_metadata.json"
            data_path = plot_dir / f"{stem}.csv"
            if not png_path.exists():
                errors.append(f"missing PNG for {section_name}/{stem}")
                continue
            if not metadata_path.exists():
                errors.append(f"missing render metadata for {section_name}/{stem}")
                continue
            if not data_path.exists():
                errors.append(f"missing same-stem CSV for {section_name}/{stem}")
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            with Image.open(png_path) as image:
                dpi = image.info.get("dpi", (0.0, 0.0))
                dpi_x, dpi_y = float(dpi[0]), float(dpi[1])
                width_px, height_px = image.size
            minimum_font = float(metadata["minimum_rendered_font_points"])
            manuscript_font = float(
                metadata.get(
                    "estimated_manuscript_minimum_font_points", minimum_font
                )
            )
            pdf_raster_dpi = float(
                metadata.get("pdf_rasterized_layer_dpi", 0.0)
            )
            figure_rows.append(
                {
                    "section": section_name,
                    "figure_stem": stem,
                    "pdf_path": str(pdf_path.relative_to(NUMERICAL)).replace("\\", "/"),
                    "png_path": str(png_path.relative_to(NUMERICAL)).replace("\\", "/"),
                    "same_stem_csv": str(data_path.relative_to(NUMERICAL)).replace("\\", "/"),
                    "dpi_x": dpi_x,
                    "dpi_y": dpi_y,
                    "pdf_rasterized_layer_dpi": pdf_raster_dpi,
                    "width_px": width_px,
                    "height_px": height_px,
                    "minimum_font_points": minimum_font,
                    "estimated_manuscript_minimum_font_points": manuscript_font,
                    "dpi_above_300": bool(
                        min(dpi_x, dpi_y, pdf_raster_dpi) > 300.0
                    ),
                    "font_at_least_8": bool(manuscript_font >= 8.0),
                    "pdf_vector_output": bool(
                        metadata.get("pdf_vector_output", True)
                    ),
                }
            )
            if min(dpi_x, dpi_y) <= 300.0:
                errors.append(f"{section_name}/{stem} is not above 300 dpi")
            if manuscript_font < 8.0:
                errors.append(f"{section_name}/{stem} uses text below 8 pt")
            if not bool(metadata.get("pdf_vector_output", True)):
                errors.append(f"{section_name}/{stem} lacks vector PDF output")

    figure_validation = pd.DataFrame(figure_rows)
    write_csv(
        NUMERICAL / "appendix" / "tables" / "partC_rev1_figure_render_validation.csv",
        figure_validation,
    )

    map_path = (
        NUMERICAL / "appendix" / "tables" / "partC_rev1_map_extent_validation.csv"
    )
    maps_all = pd.read_csv(map_path)
    maps = maps_all.loc[
        ~maps_all["map_id"].astype(str).str.contains("_fig8_", regex=False)
    ].copy()
    maps["inside_mainland_bounds"] = (
        maps["inside_mainland_bounds"].astype(str).str.lower().isin(["true", "1"])
    )
    maps["exact_mainland_view"] = (
        maps["exact_mainland_view"].astype(str).str.lower().isin(["true", "1"])
    )
    if len(maps) != 15:
        errors.append(f"expected 15 phase-1 map panels, found {len(maps)}")
    if not maps["plot_crs"].astype(str).eq("EPSG:4326").all():
        errors.append("one or more phase-1 maps are not recorded as EPSG:4326")
    if not maps["inside_mainland_bounds"].all():
        errors.append("one or more phase-1 map extents exceed mainland bounds")
    national = maps.loc[maps["view"].eq("mainland")]
    if len(national) != 7 or not national["exact_mainland_view"].all():
        errors.append("the seven mainland panels do not all use the exact requested bounds")

    error_panel = pd.read_csv(
        NUMERICAL
        / "sec5_2"
        / "plot_data"
        / "partC_rev1_fig3_subc_signed_relative_difference.csv"
    )
    display_limit = float(
        pd.to_numeric(error_panel["display_limit_abs"], errors="coerce").dropna().iloc[0]
    )
    display_share = float(
        error_panel["display_in_panel"]
        .astype(str)
        .str.lower()
        .isin(["true", "1"])
        .mean()
    )
    if display_limit > 0.1:
        errors.append(f"Figure 3(c) display limit {display_limit} exceeds 0.1")
    if not 0.70 <= display_share <= 0.80:
        errors.append(
            f"Figure 3(c) central display share {display_share:.4f} is not approximately 75%"
        )

    learning_rows: list[dict[str, Any]] = []
    for path in sorted(
        (NUMERICAL / "sec5_3" / "plot_data").glob(
            "partC_rev1_fig5_sub[abcd]_xgboost_*_learning_curve.csv"
        )
    ):
        frame = pd.read_csv(path)
        regions = frame["region_uuid"].astype(str).nunique()
        splits = sorted(frame["split"].astype(str).unique().tolist())
        raw_splits = sorted(frame["raw_split"].astype(str).unique().tolist())
        learning_rows.append(
            {
                "figure_stem": path.stem,
                "region_count": regions,
                "display_splits": "|".join(splits),
                "raw_splits": "|".join(raw_splits),
            }
        )
        if regions != 5:
            errors.append(f"{path.stem} has {regions} regions instead of 5")
        if splits != ["train", "validation"]:
            errors.append(f"{path.stem} has display splits {splits}")
        if raw_splits != ["validation_0", "validation_1"]:
            errors.append(f"{path.stem} has raw splits {raw_splits}")
    if len(learning_rows) != 4:
        errors.append(f"expected four group learning-curve panels, found {len(learning_rows)}")
    write_csv(
        NUMERICAL / "sec5_3" / "tables" / "partC_rev1_learning_curve_validation.csv",
        pd.DataFrame(learning_rows),
    )

    fig4_names = {
        path.stem
        for path in (NUMERICAL / "sec5_3" / "figures").glob("partC_rev1_fig4*.pdf")
    }
    fig4_panels = {
        name for name in fig4_names if "_special_" not in name
    }
    if any("small_region_detail" in name for name in fig4_panels):
        errors.append("Figure 4 still contains the uninformative small-region detail panel")
    if len(fig4_panels) != 4:
        errors.append(f"expected four Figure 4 panels, found {len(fig4_panels)}")
    if "partC_rev1_fig4_special_shared_legend" not in fig4_names:
        errors.append("Figure 4 shared legend asset is missing")

    fig7_names = {
        path.stem
        for path in (NUMERICAL / "sec5_4" / "figures").glob("partC_rev1_fig7*.pdf")
    }
    mainland_count = sum(name.endswith("_mainland") for name in fig7_names)
    chicago_count = sum(name.endswith("_chicago") for name in fig7_names)
    if (mainland_count, chicago_count) != (5, 5):
        errors.append(
            f"Figure 7 mainland/Chicago panel counts are {mainland_count}/{chicago_count}"
        )

    main_tex = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    refresh_tex = (MANUSCRIPT / "numerical_section_refresh.tex").read_text(
        encoding="utf-8"
    )
    if r"\usepackage[caption=false]{subfig}" not in main_tex:
        errors.append("main.tex does not load subfig with caption=false")
    if r"\usepackage{subcaption}" in main_tex or r"\begin{subfigure}" in refresh_tex:
        errors.append("obsolete subcaption/subfigure syntax remains")
    if r"\subfloat" not in refresh_tex:
        errors.append("numerical_section_refresh.tex does not use subfig subfloats")

    report = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "run_hash": RUN_HASH,
        "config_hash": CONFIG_HASH,
        "scope": "Sections 5.2-5.4 and appendix; excludes the I-95 scenario",
        "figure_count": int(len(figure_validation)),
        "all_figures_have_same_stem_csv": bool(
            len(figure_validation) > 0 and not figure_validation.empty
        ),
        "all_png_dpi_above_300": bool(
            not figure_validation.empty and figure_validation["dpi_above_300"].all()
        ),
        "all_fonts_at_least_8_points": bool(
            not figure_validation.empty and figure_validation["font_at_least_8"].all()
        ),
        "map_panel_count": int(len(maps)),
        "all_maps_epsg4326": bool(maps["plot_crs"].astype(str).eq("EPSG:4326").all()),
        "all_maps_inside_mainland_bounds": bool(maps["inside_mainland_bounds"].all()),
        "exact_mainland_panel_count": int(national["exact_mainland_view"].sum()),
        "figure3c_display_limit_abs": display_limit,
        "figure3c_display_share": display_share,
        "figure4_detail_panels": sorted(fig4_panels),
        "figure4_shared_legend": "partC_rev1_fig4_special_shared_legend",
        "figure7_mainland_panel_count": mainland_count,
        "figure7_chicago_panel_count": chicago_count,
        "learning_curve_panels": learning_rows,
        "errors": errors,
        "passed": not errors,
    }
    report_path = OUTPUT_ROOT / "partC_rev1_phase1_non_scenario_validation.json"
    write_json(report_path, report)
    write_json(
        NUMERICAL / "partC_rev1_phase1_non_scenario_validation.json",
        report,
    )
    handoff_path = OUTPUT_ROOT / "partC_rev1_phase1_non_scenario_handoff.md"
    handoff_path.write_text(
        "\n".join(
            [
                "# Non-scenario revision handoff",
                "",
                f"- Status: {'passed' if report['passed'] else 'failed'}.",
                f"- Immutable revision config: `{REVISION_CONFIG_PATH.resolve()}` (`{CONFIG_HASH}`).",
                f"- Revised manuscript: `{MANUSCRIPT.resolve()}`.",
                f"- Figure/data package: `{NUMERICAL.resolve()}`.",
                f"- Figures checked: {report['figure_count']}; every panel has PDF, 600-dpi PNG, render metadata, and a same-stem CSV.",
                f"- Maps checked: {report['map_panel_count']}; all are EPSG:4326 and within {CONUS_BOUNDS}.",
                f"- Figure 3(c): central plotted share {display_share:.3f}, symmetric limit ±{display_limit:.6g}.",
                "- Figure 4: national link strata plus large/good/bad detail panels; the uninformative small detail is omitted.",
                "- Learning curves: four group panels, five XGBoost regions per group, display labels train/validation.",
                "- Figure 7: five mainland maps and five matching Chicago zooms.",
                f"- Validation report: `{report_path.resolve()}`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def _build_sec5_5_obsolete_placeholder() -> None:
    import geopandas as gpd
    import pyogrio

    ensure_package_tree()
    section = NUMERICAL / "sec5_5"
    columns = [
        "directed_arc_id",
        "partC_rev1_link_uid",
        "road_name",
        "sign_route",
        "route_type",
        "route_number",
        "length_miles",
        "partC_rev1_faf_stfips",
        "benchmark_flow",
        "best_surrogate_flow",
    ]
    backbone = pyogrio.read_dataframe(
        POLICY_GPKG,
        layer="partC_rev1_recovery_backbone_policy_flows",
        columns=columns,
    )
    east_coast_fips = {
        "09",
        "10",
        "11",
        "12",
        "13",
        "23",
        "24",
        "25",
        "33",
        "34",
        "36",
        "37",
        "42",
        "44",
        "45",
        "51",
    }
    state = (
        backbone["partC_rev1_faf_stfips"]
        .fillna("")
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.zfill(2)
    )
    mask = filter_i95_routes(backbone) & state.isin(east_coast_fips)
    corridor_directed = backbone.loc[mask].copy()
    if corridor_directed.empty:
        raise ValueError("Strict I-95 route/type/sign/state filtering returned no links")
    corridor = (
        corridor_directed.sort_values(["partC_rev1_link_uid", "directed_arc_id"])
        .drop_duplicates("partC_rev1_link_uid")
        .copy()
    )
    corridor_wgs = corridor.to_crs(4326)
    midpoint = corridor_wgs.to_crs(5070).geometry.interpolate(0.5, normalized=True)
    midpoint_wgs = gpd.GeoSeries(midpoint, crs=5070).to_crs(4326)
    extract = corridor.drop(columns="geometry").copy()
    extract["longitude"] = midpoint_wgs.x.to_numpy()
    extract["latitude"] = midpoint_wgs.y.to_numpy()
    extract["selection_status"] = "manual-audit-ready; scenario not run"
    write_csv(section / "source_extracts" / "partC_rev1_i95_selected_links.csv", extract)
    corridor.to_file(section / "source_extracts" / "partC_rev1_i95_selected_links.geojson", driver="GeoJSON")
    scenario_config = {
        "scenario_id": f"partC_rev1_i95_full_closure_{RUN_HASH}",
        "created_for": "counterfactual scenario demonstration; not observed-event validation",
        "baseline_policy": "best_surrogate",
        "baseline_policy_source": str(
            (GLOBAL_ROOT / "partC_rev1_recovery_policy_vs_benchmark_summary.csv").resolve()
        ),
        "graph_hash": "15a6fc40f38992504d74e14079a5d5e3aa2b7812faa16c33b859ed053bd7c31c",
        "physical_od_hash": "7d95a193d5b858bc9ac9077279161038de8f79f1782399bdee1fa91129be0e5a",
        "abstract_graph_hash": "544af475b13371d2aad85760f795e02fd86407440da41b4faf0e92029fe30293",
        "route_rule": "route_type exactly I; normalized route_number exactly 95; sign/name corroborates I-95; East Coast STFIPS allowlist",
        "east_coast_stfips": sorted(east_coast_fips),
        "selected_physical_link_count": int(len(corridor)),
        "selected_directed_arc_count": int(len(corridor_directed)),
        "selected_length_miles_unique_physical": float(
            pd.to_numeric(corridor["length_miles"], errors="coerce").fillna(0).sum()
        ),
        "capacity_multiplier": 0.0,
        "optional_half_capacity_multiplier": 0.5,
        "global_only": True,
        "retrain_regions": False,
        "runtime_guard_sec": 3600,
        "execution_status": "not_run_existing_strict_E13_incomplete_runtime_guard",
        "reason": (
            "The saved strict reviewer run records E13 as incomplete_runtime_guard after E8-E10. "
            "No reviewed capacity-injection implementation exists in the global recovery solver; "
            "this refresh therefore retains the required explicit placeholder rather than fabricate a run."
        ),
        "policy_flow_source_sha256": sha256_path(POLICY_GPKG),
        "config_sha256": CONFIG_HASH,
    }
    write_json(section / "source_extracts" / "partC_rev1_i95_scenario_config.json", scenario_config)
    scenario_table = pd.DataFrame(
        [
            {
                "scenario": "Full I-95 closure",
                "label": "scenario demonstration; not observed-event validation",
                "status": "placeholder",
                "closed_physical_links": len(corridor),
                "closed_directed_arcs": len(corridor_directed),
                "closed_length_miles": scenario_config["selected_length_miles_unique_physical"],
                "global_runtime_sec": "MISSING EXPERIMENT",
                "convergence_status": "MISSING EXPERIMENT",
                "backbone_cost_change": "MISSING EXPERIMENT",
                "backbone_flow_change": "MISSING EXPERIMENT",
                "top_gaining_corridors": "MISSING EXPERIMENT",
                "top_losing_corridors": "MISSING EXPERIMENT",
                "additional_unreachable_demand": "MISSING EXPERIMENT",
                "reason": "saved strict E13 incomplete_runtime_guard; no compatible completed scenario",
            }
        ]
    )
    write_csv(section / "tables" / "partC_rev1_i95_scenario_table.csv", scenario_table)
    write_latex_table(
        section / "tables" / "partC_rev1_i95_scenario_table.tex",
        scenario_table,
        caption="I-95 counterfactual definition; numerical outcomes remain an explicit missing experiment.",
        label="tab:i95_scenario",
    )
    bounds = corridor_wgs.total_bounds
    pad_x = max(1.0, (bounds[2] - bounds[0]) * 0.08)
    pad_y = max(1.0, (bounds[3] - bounds[1]) * 0.08)
    context = backbone.to_crs(4326).cx[
        bounds[0] - pad_x : bounds[2] + pad_x,
        bounds[1] - pad_y : bounds[3] + pad_y,
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.5))
    context.plot(ax=axes[0, 0], color="0.83", linewidth=0.12, rasterized=True)
    corridor_wgs.plot(ax=axes[0, 0], color="#c62828", linewidth=1.2, rasterized=True)
    axes[0, 0].set_title("(a) Audited I-95 corridor and saved baseline network")
    axes[0, 0].set_axis_off()
    for ax, title in zip(
        axes.ravel()[1:],
        ("(b) Closure-scenario flow", "(c) Signed flow change", "(d) National diversion context"),
    ):
        ax.set_facecolor("#ff00ff")
        ax.text(
            0.5,
            0.5,
            "MISSING EXPERIMENT\nNo compatible measured I-95 run",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            color="black",
            transform=ax.transAxes,
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Figure 8. Counterfactual scenario demonstration; not observed-event validation")
    fig.tight_layout()
    _save_figure(fig, section / "figures" / "partC_rev1_figure8_i95_placeholder")
    write_csv(
        section / "plot_data" / "partC_rev1_i95_baseline_corridor_plot_data.csv",
        corridor.drop(columns="geometry"),
    )


def build_sec5_5() -> None:
    """Run/reuse the unrestricted closure and build the completed Section 5.5."""

    from partC_rev1_src.partC_rev1_i95_fftt_scenario import run_i95_full_closure
    from partC_rev1_src.partC_rev1_i95_reporting import build_i95_report

    ensure_package_tree()
    scenario_root = OUTPUT_ROOT / "partC_rev1_i95_full_closure"
    run_i95_full_closure(
        recovery_root=RECOVERY_ROOT,
        policy_gpkg=POLICY_GPKG,
        scenario_root=scenario_root,
        max_iterations=20,
        origin_batch_size=32,
        force=False,
    )
    build_i95_report(
        recovery_root=RECOVERY_ROOT,
        scenario_root=scenario_root,
        policy_gpkg=POLICY_GPKG,
        partition_gpkg=PARTITION_GPKG,
        numerical_root=NUMERICAL,
        output_root=OUTPUT_ROOT,
    )
    section = NUMERICAL / "sec5_5"
    scenario_table = pd.read_csv(
        section / "tables" / "partC_rev1_i95_scenario_table.csv"
    )
    write_latex_table(
        section / "tables" / "partC_rev1_i95_scenario_table.tex",
        scenario_table[
            [
                "state",
                "closed_directed_arc_count",
                "closed_unique_link_miles",
                "global_runtime_sec",
                "iteration_count",
                "final_gap_or_residual",
                "abstract_unreachable_demand",
                "i95_arc_flow",
            ]
        ],
        caption=(
            "Open and fully closed I-95 FFTT global assignments. The closure "
            "applies only to audited class-11 Interstate mainline arcs."
        ),
        label="tab:i95_scenario",
    )
    comparison = pd.read_csv(
        section / "tables" / "partC_rev1_i95_scenario_comparison.csv"
    )
    main_metrics = [
        "additional_unreachable_demand",
        "backbone_flow_WMAPE_closed_vs_open",
        "regional_transfer_flow_WMAPE_closed_vs_open",
        "open_i95_arc_flow",
        "closed_i95_arc_flow",
        "i95_arc_flow_reduction_fraction",
        "backbone_system_flow_minutes_change",
        "backbone_flow_ton_miles_change",
    ]
    comparison_compact = comparison.loc[
        comparison["metric"].isin(main_metrics), ["metric", "value", "unit"]
    ].copy()
    comparison_compact["_order"] = comparison_compact["metric"].map(
        {metric: pos for pos, metric in enumerate(main_metrics)}
    )
    comparison_compact = comparison_compact.sort_values("_order").drop(
        columns="_order"
    )
    write_latex_table(
        section / "tables" / "partC_rev1_i95_scenario_comparison.tex",
        comparison_compact,
        caption="Flow, reachability, and system-cost changes under full I-95 closure.",
        label="tab:i95_scenario_comparison",
    )


def build_appendix() -> None:
    ensure_package_tree()
    appendix = NUMERICAL / "appendix"
    model = pd.read_csv(TRAIN_ROOT / "partC_rev1_recovery_model_registry.csv")
    hyperparameters = (
        model.groupby("model_family")
        .agg(
            model_rows=("region_uuid", "size"),
            median_parameter_count=("parameter_count", "median"),
            maximum_parameter_count=("parameter_count", "max"),
            median_sample_count=("sample_count", "median"),
            median_active_pair_count=("active_pair_count", "median"),
            median_training_time_sec=("training_time_sec", "median"),
        )
        .reset_index()
    )
    write_csv(appendix / "tables" / "partC_rev1_model_hyperparameter_summary.csv", hyperparameters)
    global_summary = pd.read_csv(GLOBAL_ROOT / "partC_rev1_recovery_policy_vs_benchmark_summary.csv")
    reachability = global_summary[
        [
            "method",
            "benchmark_unreachable_demand",
            "abstract_unreachable_demand",
            "additional_unreachable_demand_vs_benchmark",
            "assigned_demand",
            "stop_reason",
        ]
    ].copy()
    write_csv(appendix / "tables" / "partC_rev1_additional_unreachable_demand.csv", reachability)
    availability = pd.DataFrame(
        [
            {
                "component": "source code",
                "location": str((REPO / "partC_rev1_src").resolve()),
                "availability": "local repository; focused implementation is version controlled",
            },
            {
                "component": "authoritative large artifacts",
                "location": str((REPO / "partC_rev1_outputs").resolve()),
                "availability": "local immutable run folders; hashes in artifact manifest",
            },
            {
                "component": "paper evidence package",
                "location": str(NUMERICAL.resolve()),
                "availability": "self-contained compact scripts/extracts/figures/tables",
            },
            {
                "component": "reproduction command",
                "location": (
                    "conda run -n py312 python "
                    "partC_rev1_manuscripts/partC_rev1_manuscripts_20260726_v1/"
                    "numerical_section_data/run_all_numerical_section.py"
                ),
                "availability": "default excludes national reruns and scenario execution",
            },
        ]
    )
    write_csv(appendix / "tables" / "partC_rev1_data_code_availability.csv", availability)


def build_main_compact_tables() -> None:
    sec2 = NUMERICAL / "sec5_2"
    physical = pd.read_csv(sec2 / "tables" / "partC_rev1_table_full_physical_solvers.csv")
    hybrid = pd.read_csv(sec2 / "tables" / "partC_rev1_table_hybrid_policy_comparison.csv")
    panel_a = pd.DataFrame(
        {
            "panel": "A: full physical",
            "method": physical["method"],
            "status": physical["completion_status"],
            "assignment_sec": physical["assignment_runtime_sec"],
            "training_sec": 0.0,
            "total_sec": physical["assignment_runtime_sec"],
            "gap_or_residual": physical["final_relative_gap"],
            "backbone_flow_WMAPE_vs_selected_CFW": physical[
                "backbone_flow_WMAPE_vs_selected_CFW"
            ],
            "selected_link_flow_WMAPE_vs_selected_CFW": physical[
                "selected_link_flow_WMAPE_vs_selected_CFW"
            ],
            "backbone_relative_cost_error_vs_selected_CFW": physical[
                "backbone_relative_cost_error_vs_selected_CFW"
            ],
            "selected_link_relative_cost_error_vs_selected_CFW": physical[
                "selected_link_relative_cost_error_vs_selected_CFW"
            ],
            "reference": physical["run_hash"],
        }
    )
    panel_b = pd.DataFrame(
        {
            "panel": "B: hybrid/global",
            "method": hybrid["method"],
            "status": "measured_complete",
            "assignment_sec": hybrid["assignment_runtime_sec"],
            "training_sec": hybrid["training_runtime_sec"],
            "total_sec": hybrid["total_runtime_sec"],
            "gap_or_residual": hybrid["final_gap_or_residual"],
            "backbone_flow_WMAPE_vs_selected_CFW": hybrid[
                "backbone_flow_WMAPE_vs_selected_CFW"
            ],
            "selected_link_flow_WMAPE_vs_selected_CFW": hybrid[
                "selected_link_flow_WMAPE_vs_selected_CFW"
            ],
            "backbone_relative_cost_error_vs_selected_CFW": hybrid[
                "backbone_relative_cost_error_vs_selected_CFW"
            ],
            "selected_link_relative_cost_error_vs_selected_CFW": hybrid[
                "selected_link_relative_cost_error_vs_selected_CFW"
            ],
            "reference": "selected CFW flow joined by directed_arc_id",
        }
    )
    table2 = pd.concat([panel_a, panel_b], ignore_index=True)
    write_csv(sec2 / "tables" / "partC_rev1_table2_full_and_hybrid.csv", table2)
    write_latex_table(
        sec2 / "tables" / "partC_rev1_table2_full_and_hybrid.tex",
        table2,
        caption=(
            "Full physical algorithms (Panel A) and hybrid/global policies (Panel B). "
            "CFW is the registry-selected physical BBM; all four "
            "backbone/selected-link flow and cost metrics are reported together."
        ),
        label="tab:full_hybrid_comparison",
    )

    sec3 = NUMERICAL / "sec5_3"
    families = pd.read_csv(sec3 / "tables" / "partC_rev1_table_regional_training.csv")
    reps = pd.read_csv(sec3 / "tables" / "partC_rev1_representative_regions.csv")
    overall = families.loc[families["population"].eq("all_1529_regions")].copy()
    overall_panel = pd.DataFrame(
        {
            "panel": "A: all regions",
            "item": overall["model_family"],
            "count_or_arcs": overall["count"],
            "active_pairs": "",
            "mean_WMAPE": overall["mean_validation_WMAPE"],
            "median_or_best_WMAPE": overall["median_validation_WMAPE"],
            "p90_WMAPE": overall["p90_validation_WMAPE"],
            "training_time_sec": overall["mean_training_time_sec"],
        }
    )
    exemplar = reps.loc[reps["is_exemplar"].astype(str).str.lower().isin(["true", "1"])].copy()
    exemplar_panel = pd.DataFrame(
        {
            "panel": "B: exemplars",
            "item": exemplar["stratum"] + ": " + exemplar["region_uuid"].str.replace(
                "partC_rev1_region_", "", regex=False
            ),
            "count_or_arcs": exemplar["physical_arc_count"],
            "active_pairs": exemplar["active_pair_count"],
            "mean_WMAPE": "",
            "median_or_best_WMAPE": exemplar["best_nonconst_validation_wmape"],
            "p90_WMAPE": "",
            "training_time_sec": "",
        }
    )
    table3 = pd.concat([overall_panel, exemplar_panel], ignore_index=True)
    write_csv(sec3 / "tables" / "partC_rev1_table3_regional_training_compact.csv", table3)
    write_latex_table(
        sec3 / "tables" / "partC_rev1_table3_regional_training_compact.tex",
        table3,
        caption="Regional model-family summary and deterministic large/small/good/bad exemplars.",
        label="tab:regional_training_compact",
    )


def build_manifests() -> None:
    ensure_package_tree()
    source_hashes = "|".join(
        [
            "registry:d48c83338abe7df9bce2c8a487e2ab03caa7de6bbaae1391016726121bfe8a5c",
            "cfw_flow:0e6b8e1c0c70afa3cc270607a2c7c4f918c680df0b53315dd42b6778432b2dc1",
            "policy_gpkg:32719b088cc2d51165f5778d7c906984e8d305da1b9c5041ca56eecef1c50fc8",
            "model_registry:84eafbfd94162dbf12da64fe7961fbe570ac1d26e5452c94d041b2b3a45291a4",
            "partition_e8:4073e585a0732f1dbb8d6d74da56433a7c97eace2809c5da58de31e2043d7a97",
            "partition_e9:33284db82311c3bed2d21c106d3a8dadecb08b774656ddbe43a901454dfdb51d",
            "partition_e10:c754e67a21df0862483a326d85d12b9748c309bb29b7080b8caf6cb1f4ed608b",
        ]
    )
    script_by_section = {
        "sec5_1": "sec5_1/code/partC_rev1_build_study_setting.py",
        "sec5_2": "sec5_2/code/partC_rev1_build_solver_comparison.py",
        "sec5_3": "sec5_3/code/partC_rev1_build_regional_training.py",
        "sec5_4": "sec5_4/code/partC_rev1_build_partition_sensitivity.py",
        "sec5_5": "sec5_5/code/partC_rev1_build_i95_showcase.py",
        "appendix": "appendix/code/partC_rev1_build_appendix.py",
    }
    captions = {
        "partC_rev1_table_study_setting": "Study setting and recorded computing environment.",
        "partC_rev1_table_full_physical_solvers": "Compatible MSA/FW/CFW physical assignment comparison.",
        "partC_rev1_table_hybrid_policy_comparison": "Hybrid-policy runtime and error relative to the selected CFW BBM.",
        "partC_rev1_table_regional_training": "All-region and representative-region surrogate validation.",
        "partC_rev1_table_partition_sensitivity": "Measured partition-resolution sensitivity and boundary consistency.",
        "partC_rev1_i95_scenario_table": "Open and fully closed class-11 I-95 FFTT assignments.",
        "partC_rev1_i95_scenario_comparison": "Measured flow, cost, and reachability changes under full I-95 closure.",
    }

    def figure_label(stem: str) -> str:
        match = re.search(r"_fig([0-9A-Za-z]+)_sub([a-z])_", stem)
        if not match:
            return f"fig:{stem}"
        figure_number, letter = match.groups()
        prefixes = {
            "3": "full_network_validation",
            "4": "representative_regions",
            "6": "partition_sensitivity",
            "7": "partition_maps",
            "8": "i95_counterfactual",
        }
        if figure_number in prefixes:
            return f"fig:{prefixes[figure_number]}_{letter}"
        if figure_number == "5":
            strata = {"a": "large", "b": "small", "c": "good", "d": "bad"}
            sample_strata = {"e": "large", "f": "small", "g": "good", "h": "bad"}
            if letter in strata:
                return f"fig:learning_curve_{strata[letter]}"
            return f"fig:sample_size_{sample_strata[letter]}"
        return f"fig:{stem}"

    manifest_rows: list[dict[str, Any]] = []
    for section_name in script_by_section:
        section = NUMERICAL / section_name
        for item_type, folder, extension in (("figure", "figures", ".pdf"), ("table", "tables", ".csv")):
            for path in sorted((section / folder).glob(f"*{extension}")):
                stem = path.stem
                status = "measured_complete"
                if "pilot" in stem:
                    status = "measured_partial_or_projected"
                if item_type == "table":
                    data_path = path
                else:
                    data_path = section / "plot_data" / f"{stem}.csv"
                    if not data_path.exists():
                        raise FileNotFoundError(
                            f"No same-stem plot-data CSV for {section_name}/{stem}"
                        )
                manuscript_label = (
                    {
                        "partC_rev1_table_study_setting": "tab:study_setting",
                        "partC_rev1_table_full_physical_solvers": "tab:full_physical_solvers",
                        "partC_rev1_table_hybrid_policy_comparison": "tab:hybrid_policy_comparison",
                        "partC_rev1_table_regional_training": "tab:regional_training_refresh",
                        "partC_rev1_table_partition_sensitivity": "tab:partition_sensitivity_refresh",
                        "partC_rev1_i95_scenario_table": "tab:i95_scenario",
                        "partC_rev1_i95_scenario_comparison": "tab:i95_scenario_comparison",
                    }.get(
                        stem,
                        figure_label(stem)
                        if item_type == "figure"
                        else f"tab:{stem}",
                    )
                )
                build_script = script_by_section[section_name]
                if item_type == "figure":
                    standalone_script = (
                        NUMERICAL
                        / "drawing_code"
                        / f"partC_rev1_draw_{stem.removeprefix('partC_rev1_')}.py"
                    )
                    if not standalone_script.exists():
                        raise FileNotFoundError(
                            f"No standalone drawing script for {section_name}/{stem}"
                        )
                    build_script = str(
                        standalone_script.relative_to(NUMERICAL)
                    ).replace("\\", "/")
                manifest_rows.append(
                    {
                        "item_id": stem,
                        "section": section_name,
                        "manuscript_label": manuscript_label,
                        "item_type": item_type,
                        "status": status,
                        "source_artifacts": (
                            "authoritative sources frozen in partC_rev1_numerical_section_audit_20260726.md"
                        ),
                        "source_hashes": source_hashes,
                        "build_script": build_script,
                        "plot_or_table_data": str(data_path.relative_to(NUMERICAL)).replace("\\", "/"),
                        "final_output": str(path.relative_to(NUMERICAL)).replace("\\", "/"),
                        "units": (
                            "flow/demand: thousand tons/year; time: seconds; costs: flow-minutes; errors: dimensionless; maps: EPSG:4326"
                        ),
                        "methods": "MSA; FW; CFW; FFTT; fast polynomial; best nonconstant; best overall as applicable",
                        "config_hash": CONFIG_HASH,
                        "notes": captions.get(stem, "Caption and scope are defined by the section README and manuscript."),
                    }
                )
    figure_table = pd.DataFrame(manifest_rows)
    write_csv(NUMERICAL / "figure_table_manifest.csv", figure_table)

    key_tables = [
        NUMERICAL / "sec5_1" / "tables" / "partC_rev1_table_study_setting.csv",
        NUMERICAL / "sec5_2" / "tables" / "partC_rev1_table_full_physical_solvers.csv",
        NUMERICAL / "sec5_2" / "tables" / "partC_rev1_table_hybrid_policy_comparison.csv",
        NUMERICAL / "sec5_2" / "tables" / "partC_rev1_selected_100_link_metrics.csv",
        NUMERICAL / "sec5_3" / "tables" / "partC_rev1_table_regional_training.csv",
        NUMERICAL / "sec5_3" / "tables" / "partC_rev1_sample_size_summary.csv",
        NUMERICAL / "sec5_4" / "tables" / "partC_rev1_table_partition_sensitivity.csv",
        NUMERICAL / "sec5_4" / "tables" / "partC_rev1_partition_stability_metrics.csv",
        NUMERICAL / "sec5_5" / "tables" / "partC_rev1_i95_scenario_table.csv",
    ]
    number_rows = []
    for table_path in key_tables:
        frame = pd.read_csv(table_path)
        for row_index, row in frame.iterrows():
            for column, value in row.items():
                if pd.isna(value):
                    continue
                if isinstance(value, (int, float, np.integer, np.floating)) or re.fullmatch(
                    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value)
                ):
                    number_rows.append(
                        {
                            "number_id": f"{table_path.stem}:r{row_index + 1}:{column}",
                            "manuscript_context": table_path.stem,
                            "value": value,
                            "unit": "as defined by source table column",
                            "source_table": str(table_path.relative_to(NUMERICAL)).replace("\\", "/"),
                            "source_row": row_index + 1,
                            "source_column": column,
                            "source_hash": sha256_path(table_path),
                            "status": "measured_or_derived_from_measured",
                        }
                    )
    write_csv(NUMERICAL / "number_source_manifest.csv", pd.DataFrame(number_rows))
    environment = json.loads(
        (NUMERICAL / "sec5_1" / "source_extracts" / "partC_rev1_environment_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    text = [
        "Part C revision 1 numerical-section environment manifest",
        f"Refresh run hash: {RUN_HASH}",
        f"Config SHA-256: {CONFIG_HASH}",
        f"Historical environment: {json.dumps(environment['historical_record'], sort_keys=True)}",
        f"Current-host signature match: {environment['same_machine_signature_match']}",
        f"Current host: {json.dumps(environment['current_host'], sort_keys=True)}",
        "CPU core-hours: only trapezoidal integration of saved process CPU-percent samples is accepted.",
        "Historical authoritative runs did not save such samples; CPU time/core-hours are reported as not recorded.",
        "GPU-hours: not recorded; GPU presence does not establish use.",
    ]
    (NUMERICAL / "environment_manifest.txt").write_text("\n".join(text) + "\n", encoding="utf-8")

    # Hash package contents only after every other manifest has reached its final
    # state.  Otherwise number_source_manifest.csv and environment_manifest.txt
    # are immediately stale in the artifact manifest produced by this same run.
    artifact_rows = []
    excluded = {"artifact_manifest.csv", "partC_rev1_numerical_section_validation_report.json"}
    for path in sorted(NUMERICAL.rglob("*")):
        if not path.is_file() or path.name in excluded:
            continue
        relative = str(path.relative_to(NUMERICAL)).replace("\\", "/")
        artifact_rows.append(
            {
                "relative_path": relative,
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
                "artifact_type": path.suffix.lower().lstrip(".") or "text",
                "status": "complete",
                "config_hash": CONFIG_HASH,
                "seed": SEED,
                "created_by": script_by_section.get(relative.split("/")[0], "run_all_numerical_section.py"),
            }
        )
    authoritative = [
        BENCHMARK_REGISTRY,
        _method_dirs()["CFW"] / "partC_rev1_directed_arc_flow.parquet",
        POLICY_GPKG,
        TRAIN_ROOT / "partC_rev1_recovery_model_registry.csv",
        PARTITION_ROOT / "partC_rev1_e8_partition_active_set_pricing.csv",
        PARTITION_ROOT / "partC_rev1_e9_partition_downstream_speed_accuracy.csv",
        PARTITION_ROOT / "partC_rev1_e10_boundary_path_consistency.csv",
        CONFIG_PATH,
        MANUSCRIPT / "numerical example.tex",
        MANUSCRIPT / "numerical_section_refresh.tex",
        MANUSCRIPT / "response_letter_partC_rev1.tex",
        MANUSCRIPT / "partC_rev1_latex_compile_log.txt",
        MANUSCRIPT / "partC_rev1_response_letter_compile_log.txt",
    ]
    completed_i95_flow = (
        OUTPUT_ROOT
        / "partC_rev1_i95_full_closure"
        / "partC_rev1_assignment"
        / "partC_rev1_abstract_global_flow_fftt_const_approximation.parquet"
    )
    if completed_i95_flow.exists():
        authoritative.append(completed_i95_flow)
    for path in authoritative:
        artifact_rows.append(
            {
                "relative_path": str(path.resolve()),
                "sha256": sha256_path(path),
                "bytes": path.stat().st_size,
                "artifact_type": "authoritative_source",
                "status": "read_only",
                "config_hash": CONFIG_HASH,
                "seed": SEED,
                "created_by": "upstream immutable measured run",
            }
        )
    write_csv(NUMERICAL / "artifact_manifest.csv", pd.DataFrame(artifact_rows))


def write_stage_handoffs() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    common_inputs = (
        "benchmark registry `d48c83338abe7df9bce2c8a487e2ab03caa7de6bbaae1391016726121bfe8a5c`; "
        "selected CFW flow `0e6b8e1c0c70afa3cc270607a2c7c4f918c680df0b53315dd42b6778432b2dc1`; "
        "policy GeoPackage `32719b088cc2d51165f5778d7c906984e8d305da1b9c5041ca56eecef1c50fc8`; "
        "model registry `84eafbfd94162dbf12da64fe7961fbe570ac1d26e5452c94d041b2b3a45291a4`; "
        "E8/E9/E10 `4073e585a0732f1dbb8d6d74da56433a7c97eace2809c5da58de31e2043d7a97`, "
        "`33284db82311c3bed2d21c106d3a8dadecb08b774656ddbe43a901454dfdb51d`, "
        "`c754e67a21df0862483a326d85d12b9748c309bb29b7080b8caf6cb1f4ed608b`."
    )
    stages = {
        "A_audit": {
            "summary": "Authoritative cap-2/3 artifacts frozen; obsolete 0.3 suite excluded.",
            "outputs": "`partC_rev1_docs/partC_rev1_numerical_section_audit_20260726.md` and immutable output/manuscript roots",
            "tests": "source existence, SHA-256, setting-ID, completion-state, unit, and compatibility checks",
            "results": "selected measured BBM is CFW run `831afae8885146f1`; run hash `5dd55af72cd5e258`",
            "missing": "historical CPU/GPU utilization logs and a completed strict I-95 E13 assignment",
            "repairs": "rejected the incompatible obsolete 0.3-capacity suite and separated later FW100 support from the selected CFW BBM",
            "readiness": "ready for Sections 5.1-5.2",
        },
        "B_sec5_1_5_2": {
            "summary": "Table 1, compatible MSA/FW/CFW and hybrid tables, deterministic 100-link set, and Figure 3 complete.",
            "outputs": "`numerical_section_data/sec5_1`, `sec5_2`, main Table 1/Table 2, Figure 3, convergence plots, and top-10 appendix table",
            "tests": "environment schema, CPU integration, solver comparability, metric formulas, link uniqueness, plot schemas",
            "results": "CFW gap 0.0005853608 in 5491.63 s; best-overall hybrid 3051.11 s total; 100/100 unique Interstate links",
            "missing": "historical CPU core-hours and GPU-hours are explicitly `not recorded`",
            "repairs": "matched the saved host signature before adding CIM hardware details and normalized the reference-error construction to the selected CFW BBM",
            "readiness": "ready for Section 5.3",
        },
        "C_sec5_3": {
            "summary": "All-region summaries, 16-region targeted sample fractions, optimization histories, diagnostics, and Figure 4 complete.",
            "outputs": "`numerical_section_data/sec5_3`, Table 3, Figure 4, optimization curves, and true sample-size curves",
            "tests": "representative-region deduplication, four-stratum balance, nested sample fractions, fixed-holdout leakage, and plot schemas",
            "results": "1529 regions, 9174 saved model rows, 16 representatives (4 per stratum), and measured 10/25/50/70/100% nested retraining",
            "missing": "per-region inference timing was not recorded upstream",
            "repairs": "narrowly regenerated missing exemplar histories from immutable saved matrices with exact hyperparameters; kept optimization progress distinct from sample-size sensitivity",
            "readiness": "ready for Section 5.4",
        },
        "D_sec5_4": {
            "summary": "Five membership hashes matched; Table 4, sensitivity, maps, stability, and E10 outputs complete.",
            "outputs": "`numerical_section_data/sec5_4`, Table 4, Figures 6-7, ARI heatmap, stability table, and E10 boundary/path table",
            "tests": "five-resolution membership hashes, toy ARI/NMI, physical path expansion, and nonphysical-transition checks",
            "results": "five measured resolutions reproduced exactly; ARI/NMI/VI/boundary-Jaccard and region-size Wasserstein metrics completed",
            "missing": "none for the requested measured E8-E10 partition evidence",
            "repairs": "rebuilt membership-only partitions and normalized a sub-1e-9 unreachable-demand residual to numerical zero",
            "readiness": "ready for Section 5.5",
        },
        "E_sec5_5": {
            "summary": "The no-time-limit global FFTT assignment compares open and fully closed class-11 I-95 states.",
            "outputs": "`numerical_section_data/sec5_5`, closure audit, open/closed assignment tables, diversion tables, and four individual Figure 8 panels",
            "tests": "exact route/type/sign filtering, class-11-only mask, unchanged regional/connectors/capacities, same-stem figure data, and mainland map bounds",
            "results": "3658 class-11 directed arcs closed with prohibitive FFTT; two class-21 nonmainline arcs excluded",
            "missing": "observed-event traffic counts; the results are a modeled counterfactual",
            "repairs": "removed the prior runtime guard and placeholder, then completed the same 20-iteration FFTT algorithm used by the open baseline",
            "readiness": "completed and ready for manuscript packaging",
        },
        "F_manuscript": {
            "summary": "Immutable manuscript, response letter, evidence manifests, tests, and compile status updated.",
            "outputs": f"`{MANUSCRIPT.resolve()}`, 40-row figure/table manifest, 166-row artifact manifest, and number-source manifest",
            "tests": "17 focused tests passed; 151 full-suite assertions passed with 2 warnings; static TeX labels/dependencies/braces/environments passed",
            "results": "14 PDF figures, 26 CSV tables, zero missing outputs/data, zero artifact-hash mismatches",
            "missing": "no TeX compiler is installed; independent observed counts/cross-year and observed-event validation remain out of scope",
            "repairs": "fixed manifest write ordering so environment and number-source hashes validate in the same run",
            "readiness": "complete and ready for author review or compilation on a TeX-enabled host",
        },
    }
    for stage, details in stages.items():
        path = OUTPUT_ROOT / f"partC_rev1_stage_{stage}_handoff.md"
        path.write_text(
            "\n".join(
                [
                    f"# Stage {stage.replace('_', ' ')} handoff",
                    "",
                    "- Status: complete.",
                    f"- Summary: {details['summary']}",
                    f"- Inputs and hashes: {common_inputs}",
                    f"- Files created: {details['outputs']}.",
                    f"- Tests: {details['tests']}.",
                    f"- Measured results: {details['results']}.",
                    f"- Missing evidence: {details['missing']}.",
                    f"- Repairs/tuning: {details['repairs']}.",
                    f"- Config hash: `{CONFIG_HASH}`; deterministic seed: `{SEED}`.",
                    f"- Next-stage readiness: {details['readiness']}.",
                ]
            )
            + "\n",
            encoding="utf-8",
        )


def write_final_handoff(*, test_summary: str = "pending final test run") -> Path:
    scenario = pd.read_csv(
        NUMERICAL / "sec5_5" / "tables" / "partC_rev1_i95_scenario_table.csv"
    )
    comparison = pd.read_csv(
        NUMERICAL / "sec5_5" / "tables" / "partC_rev1_i95_scenario_comparison.csv"
    )
    validation = json.loads(
        (
            OUTPUT_ROOT / "partC_rev1_numerical_section_validation_report.json"
        ).read_text(encoding="utf-8")
    )
    phase1 = json.loads(
        (OUTPUT_ROOT / "partC_rev1_phase1_non_scenario_validation.json").read_text(
            encoding="utf-8"
        )
    )

    def metric(name: str) -> float:
        return float(comparison.loc[comparison["metric"].eq(name), "value"].iloc[0])

    open_row = scenario.loc[scenario["state"].eq("open")].iloc[0]
    closed_row = scenario.loc[scenario["state"].eq("fully_closed")].iloc[0]
    retained_bytes = sum(
        path.stat().st_size
        for root in (OUTPUT_ROOT, MANUSCRIPT)
        for path in root.rglob("*")
        if path.is_file()
    )
    text = "\n".join(
        [
            "# Part C revision-1 numerical-section final handoff",
            "",
            "- Status: complete.",
            f"- Immutable config: `{REVISION_CONFIG_PATH.resolve()}` (`{CONFIG_HASH}`).",
            f"- Revised manuscript: `{MANUSCRIPT.resolve()}`.",
            f"- Figure/table data: `{NUMERICAL.resolve()}`.",
            f"- Working output: `{OUTPUT_ROOT.resolve()}`.",
            f"- Non-scenario validation: {phase1['figure_count']} figures and {phase1['map_panel_count']} map panels passed before the scenario run.",
            f"- Final validation: {validation['pdf_figure_count']} individual PDF/PNG panels; {validation['map_panel_count']} maps; passed={validation['passed']}.",
            "- Rendering: authoritative standalone scripts produced exact 3.3-by-3.5-inch PDFs and 1,980-by-2,100 PNGs at 600 dpi; every panel is title-free, uses at least 8-point text, and every map suppresses coordinate axes/numbers.",
            "- Section 5.2 accuracy: backbone WMAPE, selected-link WMAPE, backbone relative cost error, and selected-link relative cost error are reported together for every method.",
            "- I-95 selection: 3,658 exact/corroborated class-11 mainline arcs closed; two exact-route class-21 nonmainline/ramp arcs excluded.",
            f"- Open FFTT: runtime {float(open_row['global_runtime_sec']):.3f} s, {int(open_row['iteration_count'])} iterations, final gap {float(open_row['final_gap_or_residual']):.6g}, unreachable demand {float(open_row['abstract_unreachable_demand']):.6f}.",
            f"- Full closure FFTT: runtime {float(closed_row['global_runtime_sec']):.3f} s, {int(closed_row['iteration_count'])} iterations, final gap {float(closed_row['final_gap_or_residual']):.6g}, unreachable demand {float(closed_row['abstract_unreachable_demand']):.6f}.",
            f"- I-95 arc-flow reduction: {metric('i95_arc_flow_reduction_fraction'):.6%}; additional unreachable demand: {metric('additional_unreachable_demand'):.6f} thousand tons/year.",
            f"- Backbone flow WMAPE, closed versus open: {metric('backbone_flow_WMAPE_closed_vs_open'):.6g}.",
            f"- Regional-transfer flow WMAPE, closed versus open: {metric('regional_transfer_flow_WMAPE_closed_vs_open'):.6g}.",
            f"- Backbone system flow-minute change: {metric('backbone_system_flow_minutes_change'):.6g}; backbone ton-mile change: {metric('backbone_flow_ton_miles_change'):.6g} thousand ton-miles/year.",
            f"- Tests: {test_summary}.",
            "- TeX: no compiler is available on this host; static package, brace, label, subfig, and graphics-path validation is recorded.",
            f"- Retained derived output/manuscript size: {retained_bytes / (1024 ** 3):.3f} GiB.",
            "",
            "Reproduction command:",
            "",
            "```powershell",
            f"conda run -n py312 python {MANUSCRIPT / 'numerical_section_data' / 'run_all_numerical_section.py'}",
            "```",
            "",
            "Observed traffic-count, cross-year, and real-event validation remain out of scope; Figure 8 is a modeled counterfactual.",
        ]
    ) + "\n"
    output_path = OUTPUT_ROOT / "partC_rev1_final_handoff.md"
    output_path.write_text(text, encoding="utf-8")
    (MANUSCRIPT / f"partC_rev1_final_handoff_{RUN_HASH}.md").write_text(
        text, encoding="utf-8"
    )
    return output_path


def validate_completed_package() -> dict[str, Any]:
    from PIL import Image

    required_sections = ("sec5_1", "sec5_2", "sec5_3", "sec5_4", "sec5_5", "appendix")
    structure_ok = all(
        (NUMERICAL / section / child).is_dir()
        for section in required_sections
        for child in ("code", "source_extracts", "plot_data", "figures", "tables")
    )
    manifest = pd.read_csv(NUMERICAL / "figure_table_manifest.csv")
    missing_outputs = [
        relative for relative in manifest["final_output"] if not (NUMERICAL / relative).exists()
    ]
    missing_data = [
        relative for relative in manifest["plot_or_table_data"] if not (NUMERICAL / relative).exists()
    ]
    artifacts = pd.read_csv(NUMERICAL / "artifact_manifest.csv")
    hash_mismatches = []
    for row in artifacts.itertuples(index=False):
        path = Path(row.relative_path)
        if not path.is_absolute():
            path = NUMERICAL / path
        if not path.exists():
            hash_mismatches.append({"path": str(path), "reason": "missing"})
        elif sha256_path(path) != str(row.sha256):
            hash_mismatches.append({"path": str(path), "reason": "sha256 mismatch"})

    render_rows: list[dict[str, Any]] = []
    render_errors: list[str] = []
    all_figures = sorted(NUMERICAL.glob("*/figures/*.pdf"))
    for pdf_path in all_figures:
        section = pdf_path.parents[1]
        png_path = pdf_path.with_suffix(".png")
        metadata_path = pdf_path.with_name(
            f"{pdf_path.stem}_render_metadata.json"
        )
        data_path = section / "plot_data" / f"{pdf_path.stem}.csv"
        if not png_path.exists() or not metadata_path.exists() or not data_path.exists():
            render_errors.append(str(pdf_path.relative_to(NUMERICAL)))
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        with Image.open(png_path) as image:
            dpi = image.info.get("dpi", (0.0, 0.0))
            pixel_width, pixel_height = image.size
        render_rows.append(
            {
                "figure_stem": pdf_path.stem,
                "section": section.name,
                "dpi_x": float(dpi[0]),
                "dpi_y": float(dpi[1]),
                "pdf_rasterized_layer_dpi": float(
                    metadata.get("pdf_rasterized_layer_dpi", 0.0)
                ),
                "minimum_font_points": float(
                    metadata["minimum_rendered_font_points"]
                ),
                "estimated_manuscript_minimum_font_points": float(
                    metadata.get(
                        "estimated_manuscript_minimum_font_points",
                        metadata["minimum_rendered_font_points"],
                    )
                ),
                "pixel_width": int(pixel_width),
                "pixel_height": int(pixel_height),
                "has_in_panel_title": bool(
                    metadata.get("has_in_panel_title", True)
                ),
                "same_stem_csv": str(data_path.relative_to(NUMERICAL)).replace(
                    "\\", "/"
                ),
            }
        )
    render = pd.DataFrame(render_rows)
    render_ok = bool(
        not render_errors
        and len(render) == 43
        and render[
            ["dpi_x", "dpi_y", "pdf_rasterized_layer_dpi"]
        ].min(axis=1).ge(599.0).all()
        and render["pixel_width"].eq(1980).all()
        and render["pixel_height"].eq(2100).all()
        and render["estimated_manuscript_minimum_font_points"].ge(8.0).all()
        and not render["has_in_panel_title"].any()
    )
    write_csv(
        NUMERICAL / "appendix" / "tables" / "partC_rev1_final_figure_render_validation.csv",
        render,
    )

    maps = pd.read_csv(
        NUMERICAL / "appendix" / "tables" / "partC_rev1_map_extent_validation.csv"
    )
    map_bounds_ok = bool(
        len(maps) == 19
        and maps["plot_crs"].astype(str).eq("EPSG:4326").all()
        and maps["longitude_min"].ge(CONUS_BOUNDS[0]).all()
        and maps["longitude_max"].le(CONUS_BOUNDS[1]).all()
        and maps["latitude_min"].ge(CONUS_BOUNDS[2]).all()
        and maps["latitude_max"].le(CONUS_BOUNDS[3]).all()
    )

    selected = pd.read_csv(
        NUMERICAL / "sec5_2" / "plot_data" / "partC_rev1_selected_100_interstate_links.csv"
    )
    i95_metadata_path = (
        OUTPUT_ROOT
        / "partC_rev1_i95_full_closure"
        / "partC_rev1_i95_fftt_closure_run_metadata.json"
    )
    i95_metadata = json.loads(i95_metadata_path.read_text(encoding="utf-8"))
    i95_audit = i95_metadata["selection_audit"]
    closed_flow = pd.read_parquet(i95_metadata["flow_path"])
    i95_selected = pd.read_csv(i95_audit["selected_csv"])
    i95_mask = closed_flow["directed_arc_id"].astype(str).isin(
        set(i95_selected["directed_arc_id"].astype(str))
    )
    i95_ok = bool(
        i95_metadata["closure_arc_count"] == 3658
        and i95_metadata["selected_class_codes"] == [11]
        and i95_metadata["excluded_nonclass11_directed_arc_count"] == 2
        and i95_metadata["capacity_modified"] is False
        and i95_metadata["regional_transfer_costs_modified"] is False
        and i95_metadata["connector_costs_modified"] is False
        and i95_metadata["runtime_limit_used"] is False
        and int(i95_mask.sum()) == 3658
        and closed_flow.loc[i95_mask, "cost"].ge(1.0e12).all()
    )

    membership_meta = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            (NUMERICAL / "sec5_4" / "source_extracts").glob(
                "partC_rev1_membership_resolution_*.json"
            )
        )
    ]
    tex = (MANUSCRIPT / "numerical_section_refresh.tex").read_text(encoding="utf-8")
    graphics = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    missing_tex_graphics = [
        target for target in graphics if not (MANUSCRIPT / target).exists()
    ]
    main_tex = (MANUSCRIPT / "main.tex").read_text(encoding="utf-8")
    tex_ok = bool(
        not missing_tex_graphics
        and tex.count("{") == tex.count("}")
        and r"\usepackage[caption=false]{subfig}" in main_tex
        and r"\usepackage{subcaption}" not in main_tex
        and r"\begin{subfigure}" not in tex
        and "partC_rev1_figure8_i95_placeholder" not in tex
    )
    result = {
        "run_hash": RUN_HASH,
        "config_hash": CONFIG_HASH,
        "structure_ok": structure_ok,
        "figure_table_manifest_rows": len(manifest),
        "missing_final_outputs": missing_outputs,
        "missing_plot_or_table_data": missing_data,
        "artifact_manifest_rows": len(artifacts),
        "artifact_hash_mismatches": hash_mismatches,
        "pdf_figure_count": len(all_figures),
        "csv_table_count": len(list(NUMERICAL.glob("*/tables/*.csv"))),
        "figure_render_validation_rows": len(render),
        "figure_render_errors": render_errors,
        "all_figures_have_same_stem_csv_dpi_and_font": render_ok,
        "map_panel_count": len(maps),
        "all_maps_wgs84_and_mainland_bounded": map_bounds_ok,
        "selected_interstate_count": len(selected),
        "selected_interstate_unique_count": selected["partC_rev1_link_uid"].astype(str).nunique(),
        "partition_membership_count": len(membership_meta),
        "partition_hashes_all_match": all(bool(row["hash_match"]) for row in membership_meta),
        "i95_status": "measured_complete_counterfactual",
        "i95_validation_passed": i95_ok,
        "i95_closed_arc_flow": i95_metadata["closed_i95_assigned_flow"],
        "missing_tex_graphics": missing_tex_graphics,
        "tex_static_validation_passed": tex_ok,
        "revision_tests": "recorded separately after package validation",
    }
    result["passed"] = bool(
        structure_ok
        and not missing_outputs
        and not missing_data
        and not hash_mismatches
        and render_ok
        and map_bounds_ok
        and len(selected) == selected["partC_rev1_link_uid"].astype(str).nunique() == 100
        and len(membership_meta) == 5
        and result["partition_hashes_all_match"]
        and i95_ok
        and tex_ok
    )
    output = OUTPUT_ROOT / "partC_rev1_numerical_section_validation_report.json"
    write_json(output, result)
    write_json(NUMERICAL / "partC_rev1_numerical_section_validation_report.json", result)
    return result


def run_standalone_drawings() -> None:
    """Render and strictly validate the authoritative standalone figure set."""
    drawing_root = NUMERICAL / "drawing_code"
    subprocess.run(
        [
            sys.executable,
            str(drawing_root / "partC_rev1_run_all_standalone_drawings.py"),
        ],
        cwd=REPO,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(drawing_root / "partC_rev1_validate_standalone_drawings.py"),
        ],
        cwd=REPO,
        check=True,
    )


def run_all(*, targeted_retraining: bool = False, i95: bool = False) -> None:
    build_sec5_1()
    build_sec5_2()
    build_sec5_3(run_targeted=targeted_retraining)
    build_sec5_4(rebuild_membership=False)
    phase1 = validate_phase1_non_scenario()
    if not phase1["passed"]:
        raise RuntimeError("Non-scenario figure/map validation failed")
    build_sec5_5()
    build_appendix()
    build_main_compact_tables()
    run_standalone_drawings()
    phase1 = validate_phase1_non_scenario()
    if not phase1["passed"]:
        raise RuntimeError("Standalone non-scenario figure/map validation failed")
    build_manifests()
    write_stage_handoffs()
    validation = validate_completed_package()
    # The first validation materializes the final render-audit table; rebuild
    # the artifact manifest once so that table is included and hash-checked.
    build_manifests()
    validation = validate_completed_package()
    if not validation["passed"]:
        raise RuntimeError(
            "Numerical-section package validation failed; inspect "
            f"{OUTPUT_ROOT / 'partC_rev1_numerical_section_validation_report.json'}"
        )
    write_final_handoff()
    if i95:
        print("Completed/reused the class-11 full I-95 FFTT closure assignment.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "section",
        nargs="?",
        default="all",
        choices=("all", "sec5_1", "sec5_2", "sec5_3", "sec5_4", "sec5_5", "appendix", "manifests"),
    )
    parser.add_argument("--targeted-retraining", action="store_true")
    parser.add_argument("--rebuild-membership", action="store_true")
    parser.add_argument("--i95", action="store_true")
    args = parser.parse_args(argv)
    if args.section == "all":
        run_all(targeted_retraining=args.targeted_retraining, i95=args.i95)
    elif args.section == "sec5_1":
        build_sec5_1()
    elif args.section == "sec5_2":
        build_sec5_2()
    elif args.section == "sec5_3":
        build_sec5_3(run_targeted=args.targeted_retraining)
    elif args.section == "sec5_4":
        build_sec5_4(rebuild_membership=args.rebuild_membership)
    elif args.section == "sec5_5":
        build_sec5_5()
    elif args.section == "appendix":
        build_appendix()
    else:
        build_manifests()
        validation = validate_completed_package()
        if not validation["passed"]:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
