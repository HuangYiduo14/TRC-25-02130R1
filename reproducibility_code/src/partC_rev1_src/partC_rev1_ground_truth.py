"""Ground-truth assignment flow discovery, normalization, and link matching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .partC_rev1_io import partC_rev1_read_table, partC_rev1_write_table
from .partC_rev1_paths import partC_rev1_output_file, partC_rev1_resolve_path
from .partC_rev1_schema import partC_rev1_find_column
from .partC_rev1_units import partC_rev1_convert_flow_units


_PARTC_REV1_LOG = logging.getLogger("partC_rev1")


def _partC_rev1_gt_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("ground_truth", {})


def _partC_rev1_flow_candidates(config: dict[str, Any]) -> list[str]:
    gt = _partC_rev1_gt_cfg(config)
    candidates = list(
        gt.get(
            "preferred_flow_col_candidates",
            [
                gt.get("default_total_flow_col", "TOT Tons_17 All"),
                "TOT_Tons_17_All",
                "TOT Tons_2017 All",
                "TOT_Tons_2017_All",
                "flow",
                "Flow",
            ],
        )
    )
    return [str(value) for value in candidates if value]


def _partC_rev1_id_candidates(config: dict[str, Any]) -> list[str]:
    gt = _partC_rev1_gt_cfg(config)
    return [str(value) for value in gt.get("id_col_candidates", [gt.get("link_id_col", "ID"), "id", "LinkID", "link_id"]) if value]


def _partC_rev1_candidate_report(config: dict[str, Any], candidates: list[dict[str, Any]]) -> pd.DataFrame:
    report = pd.DataFrame(candidates)
    if report.empty:
        report = pd.DataFrame(columns=["path", "id_column", "flow_column", "status"])
    partC_rev1_write_table(report, partC_rev1_output_file(config, "partC_rev1_ground_truth_candidates.csv"), index=False)
    return report


def partC_rev1_find_ground_truth_flow_file(config: dict[str, Any]) -> str | None:
    inputs = config.get("inputs", {})
    configured = str(inputs.get("ground_truth_flow_path", "") or "").strip()
    if configured:
        path = partC_rev1_resolve_path(configured)
        if not path.exists():
            raise FileNotFoundError(f"Configured ground-truth flow file not found: {path}")
        _PARTC_REV1_LOG.info("Using configured ground-truth flow file: %s", path)
        return str(path)

    flow_dir_value = str(inputs.get("ground_truth_flow_dir", "") or "").strip()
    if not flow_dir_value:
        _partC_rev1_candidate_report(config, [])
        return None
    flow_dir = partC_rev1_resolve_path(flow_dir_value)
    if not flow_dir.exists():
        _partC_rev1_candidate_report(config, [])
        return None

    gt = _partC_rev1_gt_cfg(config)
    csv_files = sorted(flow_dir.glob("*.csv"))
    recommended = str(gt.get("recommended_file", "") or "")
    if recommended and (flow_dir / recommended).exists():
        _PARTC_REV1_LOG.info("Using recommended ground-truth flow file: %s", flow_dir / recommended)
        return str(flow_dir / recommended)
    candidate_names = set(map(str, gt.get("candidate_files", [])))
    filtered = [path for path in csv_files if not candidate_names or path.name in candidate_names]
    if len(filtered) == 1:
        return str(filtered[0])
    if not filtered:
        _partC_rev1_candidate_report(config, [])
        return None
    candidates = [{"path": str(path), "id_column": "", "flow_column": "", "status": "multiple_candidate_files"} for path in filtered]
    _partC_rev1_candidate_report(config, candidates)
    raise ValueError("Multiple ground-truth candidate files found; set inputs.ground_truth_flow_path explicitly")


def _partC_rev1_detect_columns(df: pd.DataFrame, config: dict[str, Any]) -> tuple[str, str]:
    id_col = partC_rev1_find_column(df, _partC_rev1_id_candidates(config)[0], candidates=_partC_rev1_id_candidates(config)[1:])
    flow_candidates = _partC_rev1_flow_candidates(config)
    matches = [column for column in flow_candidates if partC_rev1_find_column(df, column, required=False) is not None]
    if matches:
        flow_col = partC_rev1_find_column(df, matches[0])
    else:
        flow_col = partC_rev1_find_column(df, flow_candidates[0], candidates=flow_candidates[1:])
    return id_col, flow_col


def _partC_rev1_unit_from_flow_column(column: str, config: dict[str, Any]) -> str:
    text = column.lower()
    if "trip" in text:
        return "daily_truck_trips"
    if "ton" in text:
        return str(_partC_rev1_gt_cfg(config).get("preferred_flow_unit", _partC_rev1_gt_cfg(config).get("flow_units", "annual_ktons")))
    return str(_partC_rev1_gt_cfg(config).get("preferred_flow_unit", "annual_ktons"))


def partC_rev1_read_ground_truth_flow(config: dict[str, Any]) -> pd.DataFrame | None:
    path_text = partC_rev1_find_ground_truth_flow_file(config)
    if path_text is None:
        _PARTC_REV1_LOG.info("No ground-truth flow file found or configured")
        return None
    path = Path(path_text)
    _PARTC_REV1_LOG.info("Reading ground-truth flow from %s", path)
    df = partC_rev1_read_table(path)
    _PARTC_REV1_LOG.info("Read ground-truth raw rows=%s columns=%s", len(df), len(df.columns))
    id_col, flow_col = _partC_rev1_detect_columns(df, config)
    _partC_rev1_candidate_report(
        config,
        [{"path": str(path), "id_column": id_col, "flow_column": flow_col, "status": "selected"}],
    )
    source_unit = _partC_rev1_unit_from_flow_column(flow_col, config)
    target_unit = str(_partC_rev1_gt_cfg(config).get("preferred_flow_unit", _partC_rev1_gt_cfg(config).get("flow_units", "annual_ktons")))
    flow = pd.to_numeric(df[flow_col], errors="coerce").fillna(0.0)
    normalized_flow = partC_rev1_convert_flow_units(flow, source_unit, target_unit, config)
    return pd.DataFrame(
        {
            "ID": df[id_col].astype("string").str.strip(),
            "flow": normalized_flow,
            "flow_unit": target_unit,
            "source_file": str(path),
            "source_column": flow_col,
        }
    )


def partC_rev1_match_ground_truth_to_clean_links(
    clean_links_df: pd.DataFrame,
    gt_df: pd.DataFrame | None,
    config: dict[str, Any],
) -> pd.DataFrame:
    out = clean_links_df.copy()
    gt_cfg = _partC_rev1_gt_cfg(config)
    clean_col = str(gt_cfg.get("ground_truth_clean_col", "partC_rev1_ground_truth_flow"))
    if gt_df is None or gt_df.empty:
        _PARTC_REV1_LOG.info("Skipping ground-truth match because no ground-truth rows are available")
        out[clean_col] = pd.NA
        report = pd.DataFrame(
            [{"clean_link_rows": len(out), "ground_truth_rows": 0, "matched_link_rows": 0, "match_rate": 0.0}]
        )
        partC_rev1_write_table(report, partC_rev1_output_file(config, "partC_rev1_ground_truth_match_report.csv"), index=False)
        return out

    gt = gt_df.copy()
    gt["ID"] = gt["ID"].astype("string").str.strip()
    gt = gt.rename(columns={"ID": "_partC_rev1_gt_id", "flow": "_partC_rev1_gt_flow"})
    merged = out.merge(gt[["_partC_rev1_gt_id", "_partC_rev1_gt_flow"]], left_on="physical_link_id", right_on="_partC_rev1_gt_id", how="left")
    matched = merged["_partC_rev1_gt_flow"].notna()
    if bool(gt_cfg.get("missing_ground_truth_flow_default_zero", "missing_ground_truth_flow_value" in gt_cfg)):
        merged[clean_col] = merged["_partC_rev1_gt_flow"].fillna(float(gt_cfg.get("missing_ground_truth_flow_value", 0.0)))
    else:
        merged[clean_col] = merged["_partC_rev1_gt_flow"]
    report = pd.DataFrame(
        [
            {
                "clean_link_rows": len(out),
                "ground_truth_rows": len(gt),
                "matched_link_rows": int(matched.sum()),
                "missing_link_rows": int((~matched).sum()),
                "match_rate": float(matched.mean()) if len(matched) else 0.0,
            }
        ]
    )
    partC_rev1_write_table(report, partC_rev1_output_file(config, "partC_rev1_ground_truth_match_report.csv"), index=False)
    _PARTC_REV1_LOG.info(
        "Ground-truth match complete: matched=%s missing=%s match_rate=%.6f",
        int(matched.sum()),
        int((~matched).sum()),
        float(matched.mean()) if len(matched) else 0.0,
    )
    return merged.drop(columns=["_partC_rev1_gt_id", "_partC_rev1_gt_flow"])
