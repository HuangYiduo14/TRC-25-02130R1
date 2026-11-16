"""Strict full-I-95 FFTT closure on the saved abstract/hybrid graph.

Only exact, corroborated Interstate 95 arcs with FAF class code 11 receive
the prohibitive free-flow travel time.  Regional-transfer and connector arcs
are left unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .partC_rev1_abstract_flow import (
    PartCRev1AbstractGraphBundle,
    partC_rev1_load_saved_abstract_graph_bundle,
    partC_rev1_run_abstract_global_assignment,
)
from .partC_rev1_progress_logging import PartCRev1ProgressReporter


POLICY = "fftt_const_approximation"
INTERSTATE_MAINLINE_CLASS = 11
PROHIBITIVE_FFTT_MIN = 1.0e12


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _strict_i95_mask(frame: pd.DataFrame) -> pd.Series:
    route_type = (
        frame["route_type"].fillna("").astype(str).str.strip().str.upper()
    )
    route_number = (
        frame["route_number"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(r"[^0-9]", "", regex=True)
        .str.lstrip("0")
    )
    sign = frame["sign_route"].fillna("").astype(str).str.upper()
    name = frame["road_name"].fillna("").astype(str).str.upper()
    corroborated = sign.str.contains(
        r"(?:^|[^0-9])I[- ]?95(?:[^0-9]|$)", regex=True
    ) | name.str.contains(r"(?:^|[^0-9])I[- ]?95(?:[^0-9]|$)", regex=True)
    return route_type.eq("I") & route_number.eq("95") & corroborated


def audit_i95_mainline(policy_gpkg: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Audit and persist the exact class-11 closure population."""

    import pyogrio

    policy_path = Path(policy_gpkg).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "directed_arc_id",
        "partC_rev1_link_uid",
        "physical_link_id",
        "from_node_uid",
        "to_node_uid",
        "length_miles",
        "class_code",
        "road_name",
        "sign_route",
        "route_type",
        "route_number",
        "partC_rev1_faf_stfips",
        "partC_rev1_fftt_min",
    ]
    links = pyogrio.read_dataframe(
        policy_path,
        layer="partC_rev1_recovery_backbone_policy_flows",
        columns=columns,
        read_geometry=False,
    )
    exact_i95 = _strict_i95_mask(links)
    class_code = pd.to_numeric(links["class_code"], errors="coerce")
    selected_mask = exact_i95 & class_code.eq(INTERSTATE_MAINLINE_CLASS)
    selected = links.loc[selected_mask].copy()
    excluded = links.loc[exact_i95 & ~class_code.eq(INTERSTATE_MAINLINE_CLASS)].copy()
    if selected.empty:
        raise RuntimeError("Strict I-95 class-11 audit selected no arcs")
    if not pd.to_numeric(selected["class_code"], errors="coerce").eq(11).all():
        raise AssertionError("Closure population contains non-class-11 arcs")
    if selected["directed_arc_id"].astype(str).duplicated().any():
        raise AssertionError("Closure population contains duplicate directed_arc_id values")

    selected["closure_population"] = True
    selected["closure_rule"] = (
        "route_type=I; route_number=95; sign/name corroborated; class_code=11"
    )
    selected["closure_fftt_min"] = PROHIBITIVE_FFTT_MIN
    excluded["closure_population"] = False
    excluded["exclusion_reason"] = "not Interstate mainline class_code 11"
    selected_path = out_dir / "partC_rev1_i95_class11_selected_directed_arcs.csv"
    excluded_path = out_dir / "partC_rev1_i95_exact_route_excluded_nonclass11.csv"
    selected.to_csv(selected_path, index=False, lineterminator="\n")
    excluded.to_csv(excluded_path, index=False, lineterminator="\n")

    selected_physical_key = (
        selected["physical_link_id"]
        .where(selected["physical_link_id"].notna(), selected["partC_rev1_link_uid"])
        .astype(str)
    )
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy_gpkg": str(policy_path),
        "policy_gpkg_sha256": _sha256(policy_path),
        "strict_exact_i95_directed_arc_count_before_class_filter": int(exact_i95.sum()),
        "selected_class11_directed_arc_count": int(len(selected)),
        "selected_unique_physical_link_count": int(selected_physical_key.nunique()),
        "selected_unique_link_uid_count": int(
            selected["partC_rev1_link_uid"].astype(str).nunique()
        ),
        "selected_unique_miles_by_physical_link": float(
            selected.assign(_physical_key=selected_physical_key)
            .sort_values(["_physical_key", "directed_arc_id"])
            .drop_duplicates("_physical_key")["length_miles"]
            .pipe(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .sum()
        ),
        "selected_class_codes": sorted(
            pd.to_numeric(selected["class_code"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "excluded_nonclass11_directed_arc_count": int(len(excluded)),
        "excluded_class_codes": sorted(
            pd.to_numeric(excluded["class_code"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        ),
        "ramps_or_nonmainline_affected": False,
        "selected_csv": str(selected_path),
        "excluded_csv": str(excluded_path),
    }
    audit_path = out_dir / "partC_rev1_i95_class11_selection_audit.json"
    _write_json(audit_path, audit)
    audit["audit_json"] = str(audit_path)
    return audit


def _closed_bundle(
    base_bundle: PartCRev1AbstractGraphBundle,
    selected_directed_arc_ids: set[str],
) -> tuple[PartCRev1AbstractGraphBundle, np.ndarray, str]:
    arc_table = base_bundle.arc_table.copy()
    directed_ids = arc_table["directed_arc_id"].astype(str)
    kind = arc_table["abstract_arc_kind"].astype(str)
    closure_mask = directed_ids.isin(selected_directed_arc_ids).to_numpy()
    missing = selected_directed_arc_ids.difference(
        set(directed_ids.loc[closure_mask].tolist())
    )
    if missing:
        raise RuntimeError(
            f"{len(missing)} audited I-95 directed arcs are absent from the abstract graph"
        )
    if not kind.loc[closure_mask].eq("backbone").all():
        raise AssertionError("Closure mask includes a non-backbone abstract arc")
    if int(closure_mask.sum()) != len(selected_directed_arc_ids):
        raise AssertionError("Closure mask cardinality does not match the audited population")

    free_time = np.asarray(base_bundle.graph.free_time, dtype=np.float64).copy()
    unchanged = free_time.copy()
    free_time[closure_mask] = PROHIBITIVE_FFTT_MIN
    if not np.array_equal(
        free_time[~closure_mask],
        unchanged[~closure_mask],
        equal_nan=True,
    ):
        raise AssertionError("A non-I-95 arc free-flow time changed")
    arc_table.loc[closure_mask, "partC_rev1_fftt_min"] = PROHIBITIVE_FFTT_MIN
    arc_table["i95_full_closure"] = closure_mask

    hash_payload = {
        "base_abstract_graph_hash": base_bundle.graph.graph_hash,
        "closure_policy": POLICY,
        "selected_directed_arc_ids": sorted(selected_directed_arc_ids),
        "prohibitive_fftt_min": PROHIBITIVE_FFTT_MIN,
        "capacity_unchanged": True,
    }
    closure_hash = hashlib.sha256(
        json.dumps(hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    metadata = dict(base_bundle.graph.metadata)
    metadata.update(
        {
            "base_abstract_graph_hash": base_bundle.graph.graph_hash,
            "assignment_graph_hash": closure_hash,
            "i95_full_closure": True,
            "i95_closure_arc_count": int(closure_mask.sum()),
            "i95_prohibitive_fftt_min": PROHIBITIVE_FFTT_MIN,
            "i95_capacity_unchanged": True,
        }
    )
    graph = replace(
        base_bundle.graph,
        graph_hash=closure_hash,
        free_time=free_time,
        metadata=metadata,
    )
    bundle_metadata = dict(base_bundle.metadata)
    bundle_metadata.update(metadata)
    return (
        replace(
            base_bundle,
            graph=graph,
            arc_table=arc_table,
            metadata=bundle_metadata,
        ),
        closure_mask,
        closure_hash,
    )


def run_i95_full_closure(
    *,
    recovery_root: str | Path,
    policy_gpkg: str | Path,
    scenario_root: str | Path,
    max_iterations: int = 20,
    origin_batch_size: int = 32,
    force: bool = False,
) -> dict[str, Any]:
    """Run or reuse the unrestricted 20-iteration class-11 full closure."""

    root = Path(scenario_root).resolve()
    source_dir = root / "partC_rev1_source_extracts"
    assignment_dir = root / "partC_rev1_assignment"
    source_dir.mkdir(parents=True, exist_ok=True)
    assignment_dir.mkdir(parents=True, exist_ok=True)
    summary_json = assignment_dir / "partC_rev1_abstract_global_summary.json"
    flow_path = (
        assignment_dir
        / f"partC_rev1_abstract_global_flow_{POLICY}.parquet"
    )
    run_metadata_path = root / "partC_rev1_i95_fftt_closure_run_metadata.json"
    if not force and summary_json.exists() and flow_path.exists() and run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        old_root = Path(run_metadata["flow_path"]).resolve().parent.parent

        def rebase(value: Any) -> Any:
            if isinstance(value, dict):
                return {key: rebase(item) for key, item in value.items()}
            if isinstance(value, list):
                return [rebase(item) for item in value]
            if isinstance(value, str):
                try:
                    relative = Path(value).resolve().relative_to(old_root)
                except (OSError, ValueError):
                    return value
                if relative.parts and relative.parts[0] in {
                    "assignment",
                    "partC_rev1_assignment",
                }:
                    return str(assignment_dir.joinpath(*relative.parts[1:]))
                if relative.parts and relative.parts[0] in {
                    "source_extracts",
                    "partC_rev1_source_extracts",
                }:
                    return str(source_dir.joinpath(*relative.parts[1:]))
                return str(root / relative)
            return value

        run_metadata = rebase(run_metadata)
        _write_json(run_metadata_path, run_metadata)
        summary = rebase(json.loads(summary_json.read_text(encoding="utf-8")))
        _write_json(summary_json, summary)
        summary_csv = assignment_dir / "partC_rev1_abstract_global_summary.csv"
        if summary_csv.exists():
            summary_frame = pd.read_csv(summary_csv)
            for column in ("flow_path", "history_path"):
                if column in summary_frame:
                    summary_frame[column] = summary_frame[column].map(rebase)
            summary_frame.to_csv(summary_csv, index=False, lineterminator="\n")
        return run_metadata

    audit = audit_i95_mainline(policy_gpkg, source_dir)
    selected = pd.read_csv(audit["selected_csv"])
    selected_ids = set(selected["directed_arc_id"].astype(str))
    base_bundle = partC_rev1_load_saved_abstract_graph_bundle(recovery_root)
    closed_bundle, closure_mask, closure_hash = _closed_bundle(
        base_bundle, selected_ids
    )
    reporter = PartCRev1ProgressReporter(
        stage="partC_rev1_i95_full_closure_fftt",
        output_dir=root,
        heartbeat_seconds=15.0,
    )
    reporter.emit(
        "closure_audit_complete",
        closure_arc_count=int(closure_mask.sum()),
        class_codes=audit["selected_class_codes"],
        excluded_nonclass11_arc_count=audit[
            "excluded_nonclass11_directed_arc_count"
        ],
        no_wall_time_limit=True,
        prohibitive_fftt_min=PROHIBITIVE_FFTT_MIN,
    )
    try:
        result = partC_rev1_run_abstract_global_assignment(
            closed_bundle,
            policies=[POLICY],
            output_dir=assignment_dir,
            max_iterations=max_iterations,
            origin_batch_size=origin_batch_size,
            backend="numba_csr",
            reporter=reporter,
            training_summary_path=None,
            strict_nonconst_models=False,
            disk_budget_gb=None,
            max_wall_hours=None,
        )
        reporter.emit(
            "closure_assignment_complete",
            summary_csv=result["summary_csv"],
            flow_path=result["flow_paths"][POLICY],
        )
    finally:
        reporter.close()

    flow = pd.read_parquet(flow_path)
    closed_selected_flow = float(
        pd.to_numeric(
            flow.loc[flow["directed_arc_id"].astype(str).isin(selected_ids), "flow"],
            errors="coerce",
        )
        .fillna(0.0)
        .sum()
    )
    run_metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "I-95 fully closed; open and fully closed states only",
        "policy": POLICY,
        "base_recovery_root": str(Path(recovery_root).resolve()),
        "base_abstract_graph_hash": base_bundle.graph.graph_hash,
        "closed_abstract_graph_hash": closure_hash,
        "closure_arc_count": int(closure_mask.sum()),
        "selected_class_codes": audit["selected_class_codes"],
        "excluded_nonclass11_directed_arc_count": audit[
            "excluded_nonclass11_directed_arc_count"
        ],
        "prohibitive_fftt_min": PROHIBITIVE_FFTT_MIN,
        "capacity_modified": False,
        "regional_transfer_costs_modified": False,
        "connector_costs_modified": False,
        "max_iterations": int(max_iterations),
        "origin_batch_size": int(origin_batch_size),
        "max_wall_hours": None,
        "runtime_limit_used": False,
        "summary_json": str(summary_json),
        "summary_csv": str(assignment_dir / "partC_rev1_abstract_global_summary.csv"),
        "flow_path": str(flow_path),
        "history_path": str(
            assignment_dir
            / f"partC_rev1_abstract_global_history_{POLICY}.csv"
        ),
        "closed_i95_assigned_flow": closed_selected_flow,
        "selection_audit": audit,
    }
    _write_json(run_metadata_path, run_metadata)
    return run_metadata


__all__ = [
    "INTERSTATE_MAINLINE_CLASS",
    "POLICY",
    "PROHIBITIVE_FFTT_MIN",
    "audit_i95_mainline",
    "run_i95_full_closure",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-root", required=True)
    parser.add_argument("--policy-gpkg", required=True)
    parser.add_argument("--scenario-root", required=True)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--origin-batch-size", type=int, default=32)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    metadata = run_i95_full_closure(
        recovery_root=args.recovery_root,
        policy_gpkg=args.policy_gpkg,
        scenario_root=args.scenario_root,
        max_iterations=args.max_iterations,
        origin_batch_size=args.origin_batch_size,
        force=args.force,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
