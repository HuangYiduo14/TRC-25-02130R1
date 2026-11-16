"""Read-only bindings for the accepted Phase 3/4 inputs used by the pilot."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from partC_rev1_src.phase4.inputs import load_or_build_aggregated_od, load_phase4_graph
from partC_rev1_src.phase4.models import AggregatedOD, SuiteTraceMap
from partC_rev1_src.phase4.tracing import load_suite_trace_maps


@dataclass(frozen=True, slots=True)
class PhysicalState:
    benchmark_id: str
    current_flow: np.ndarray
    aon_flow: np.ndarray
    current_cost_s: np.ndarray
    used_capacity: np.ndarray


@dataclass(frozen=True, slots=True)
class PilotBindings:
    execution: dict[str, Any]
    sources: dict[str, Any]
    graph: Any
    od: AggregatedOD
    trace_maps: tuple[SuiteTraceMap, ...]
    low_physical: PhysicalState
    high_physical: PhysicalState
    phase3_root: Path


def _physical_state(
    binding: dict[str, Any], capacity_factor: float, *, load_aon: bool = True
) -> PhysicalState:
    with np.load(binding["physical_arrays"], allow_pickle=False) as data:
        current_key = "current_physical_flow"
        aon_key = "refreshed_aon_physical_flow"
        cost_key = "current_cost" if "current_cost" in data.files else "final_cost"
        clean = data["clean_prepared_capacity"].astype(np.float64, copy=True)
        used = data["used_capacity"].astype(np.float64, copy=True)
        if not np.array_equal(used, clean * float(capacity_factor)):
            raise ValueError(
                f"{binding['benchmark_id']} capacity factor is not exactly {capacity_factor}"
            )
        return PhysicalState(
            benchmark_id=str(binding["benchmark_id"]),
            current_flow=data[current_key].astype(np.float64, copy=True),
            aon_flow=(
                data[aon_key].astype(np.float64, copy=True)
                if load_aon
                else np.empty(0, dtype=np.float64)
            ),
            current_cost_s=data[cost_key].astype(np.float64, copy=True) * 60.0,
            used_capacity=used,
        )


def load_bindings(
    execution_config: str | Path,
    *,
    verify_suite_digests: bool = True,
    suite_ids: list[str] | tuple[str, ...] | None = None,
    load_aon: bool = True,
) -> PilotBindings:
    config_path = Path(execution_config).resolve()
    execution = json.loads(config_path.read_text(encoding="utf-8"))
    project_root = Path(execution["project_root"]).resolve()
    installed_goal = execution.get("installed_authority_goal_relative_path")
    if installed_goal:
        authority_root = (project_root / str(installed_goal)).resolve().parent
    else:
        authority_root = (
            project_root
            / "partC_doc_icnn"
            / "review_gates"
            / "phase_05"
            / "backbone_compilation_pilot"
            / "phase5_pilot_authority_20260903"
        )
    explicit_source = execution.get("inherited_source_bindings_relative_path")
    inherited_sources = authority_root / "config" / "inherited_source_bindings.json"
    source_path = (
        (project_root / str(explicit_source)).resolve()
        if explicit_source
        else (
            inherited_sources
            if inherited_sources.is_file()
            else authority_root / "config" / "source_bindings.json"
        )
    )
    if not source_path.is_file() or not source_path.is_relative_to(project_root):
        raise FileNotFoundError(
            "Frozen Phase 5 source bindings are absent or outside the project root"
        )
    sources = json.loads(source_path.read_text(encoding="utf-8"))
    selected_suites = list(suite_ids or execution["suite_ids"])
    if not selected_suites or not set(selected_suites).issubset(set(execution["suite_ids"])):
        raise ValueError("Requested suite subset is empty or outside the execution contract")
    if (
        Path(sources["project_root"]).resolve() != project_root
        or sources["branch"] != execution["expected_branch"]
        or list(sources["suite_ids"]) != list(execution["suite_ids"])
        or list(sources["excluded_suite_ids"]) != list(execution["excluded_suite_ids"])
        or sources["low_benchmark"]["benchmark_id"]
        != execution["primary_benchmark"]["benchmark_id"]
        or int(sources["low_benchmark"]["state_iteration"])
        != int(execution["primary_benchmark"]["state_iteration"])
        or sources["high_benchmark"]["benchmark_id"]
        != execution["heldout_benchmark"]["benchmark_id"]
        or int(sources["high_benchmark"]["state_iteration"])
        != int(execution["heldout_benchmark"]["state_iteration"])
    ):
        raise ValueError("Execution configuration and frozen source bindings disagree")
    phase4_config = project_root / "partC_doc_icnn" / "phase_4_work" / "phase4_dual_accuracy_benchmark_execution_config.json"
    phase4 = json.loads(phase4_config.read_text(encoding="utf-8"))
    graph = load_phase4_graph(phase4)
    graph_source = sources["graph"]
    capacity_payload = hashlib.sha256(
        np.ascontiguousarray(graph.capacity, dtype=np.float64).tobytes(order="C")
    ).hexdigest()
    if (
        graph.graph_hash != graph_source["identity"]
        or len(graph.node_idx_to_uid) != int(graph_source["node_count"])
        or len(graph.arc_idx_to_directed_arc_id) != int(graph_source["arc_count"])
        or capacity_payload != graph_source["clean_capacity_payload_sha256"]
    ):
        raise ValueError("Frozen Phase 5 graph binding does not match the loaded graph")
    od = load_or_build_aggregated_od(
        phase4,
        output_root=Path(execution["output_root"]),
        existing_root=Path(phase4["input_versions"]["accepted_aggregated_od_root"]),
    )
    trace_maps = tuple(
        load_suite_trace_maps(
            phase4,
            graph,
            suite_ids=selected_suites,
            verify_directory_digests=verify_suite_digests,
        )
    )
    factor = float(execution["capacity_factor"])
    low = _physical_state(sources["low_benchmark"], factor, load_aon=load_aon)
    high = _physical_state(sources["high_benchmark"], factor, load_aon=load_aon)
    arc_count = int(sources["graph"]["arc_count"])
    for state in (low, high):
        for value in (state.current_flow, state.current_cost_s, state.used_capacity):
            if value.shape != (arc_count,) or np.any(~np.isfinite(value)):
                raise ValueError(f"{state.benchmark_id} physical arrays are invalid")
        if np.any(state.current_flow < 0.0):
            raise ValueError(f"{state.benchmark_id} contains negative flow")
        if load_aon:
            if (
                state.aon_flow.shape != (arc_count,)
                or np.any(~np.isfinite(state.aon_flow))
                or np.any(state.aon_flow < 0.0)
            ):
                raise ValueError(f"{state.benchmark_id} AON flow is invalid")
        elif state.aon_flow.size != 0:
            raise RuntimeError("R0 no-AON binding unexpectedly loaded AON flow")
    if not np.array_equal(low.used_capacity, high.used_capacity):
        raise ValueError("Low/high accepted used-capacity arrays differ")
    if not np.array_equal(low.used_capacity, graph.capacity * factor):
        raise ValueError(
            "Accepted used capacity differs from graph capacity times configured factor"
        )
    for state in (low, high):
        reconstructed_cost_s = (
            graph.free_time
            * 60.0
            * (
                1.0
                + graph.alpha
                * np.power(state.current_flow / state.used_capacity, graph.beta)
            )
        )
        if not np.allclose(
            state.current_cost_s,
            reconstructed_cost_s,
            rtol=2.0e-15,
            atol=1.0e-7,
        ):
            raise ValueError(
                f"{state.benchmark_id} current cost fails the canonical seconds BPR identity"
            )
    return PilotBindings(
        execution=execution,
        sources=sources,
        graph=graph,
        od=od,
        trace_maps=trace_maps,
        low_physical=low,
        high_physical=high,
        phase3_root=Path(sources["phase3_root"]),
    )


def load_suite_trace_arrays(
    sources: dict[str, Any],
    suite_id: str,
    tier: str,
    *,
    trace_map: SuiteTraceMap,
) -> dict[str, np.ndarray]:
    if tier not in {"low_benchmark", "high_benchmark"}:
        raise ValueError("Trace tier must be low_benchmark or high_benchmark")
    path = Path(str(sources[tier]["suite_trace_template"]).format(suite=suite_id))
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    expected_dense = {
        "bundle_flow": trace_map.bundle_count,
        "bacn_internal_arc_flow": trace_map.bacn_count,
        "bacn_internal_traversal_flow": trace_map.bacn_count,
        "terminal_connector_flow": trace_map.terminal_count,
        "racn_interface_flow": trace_map.racn_count,
        "repeated_region_entry_flow": trace_map.region_count,
    }
    required: set[str] = set()
    for prefix in ("current__", "aon__"):
        for field, count in expected_dense.items():
            name = prefix + field
            required.add(name)
            value = arrays.get(name)
            if (
                value is None
                or value.shape != (count,)
                or not np.issubdtype(value.dtype, np.floating)
                or np.any(~np.isfinite(value))
                or np.any(value < -1.0e-8)
            ):
                raise ValueError(f"{suite_id} {tier} trace field {name} is invalid")
        for stem in ("coarse_regional_pair_flow", "fine_regional_pair_flow"):
            key_name = prefix + stem + "__keys"
            value_name = prefix + stem + "__values"
            required.update((key_name, value_name))
            keys = arrays.get(key_name)
            values = arrays.get(value_name)
            if (
                keys is None
                or values is None
                or keys.dtype != np.uint64
                or not np.issubdtype(values.dtype, np.floating)
                or keys.ndim != 1
                or keys.shape != values.shape
                or np.any(~np.isfinite(values))
                or np.any(values < -1.0e-8)
                or (len(keys) > 1 and np.any(keys[1:] <= keys[:-1]))
            ):
                raise ValueError(
                    f"{suite_id} {tier} sparse trace field {stem} is invalid"
                )
            if stem == "coarse_regional_pair_flow" and len(keys):
                entry = (keys >> np.uint64(32)).astype(np.int64)
                exit_ = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
                if (
                    np.any(entry >= trace_map.terminal_count)
                    or np.any(exit_ >= trace_map.terminal_count)
                ):
                    raise ValueError(
                        f"{suite_id} {tier} coarse trace key contains an invalid terminal"
                    )
            if stem == "fine_regional_pair_flow" and len(keys):
                node_mask = np.uint64((1 << 19) - 1)
                exit_node = (keys & node_mask).astype(np.int64)
                entry_node = ((keys >> np.uint64(19)) & node_mask).astype(
                    np.int64
                )
                region = (keys >> np.uint64(38)).astype(np.int64)
                node_count = len(trace_map.node_to_bacn_local)
                if (
                    np.any(entry_node >= node_count)
                    or np.any(exit_node >= node_count)
                    or np.any(region >= trace_map.region_count)
                ):
                    raise ValueError(
                        f"{suite_id} {tier} fine trace key contains an invalid node or region"
                    )
    missing = sorted(required - arrays.keys())
    if missing:
        raise ValueError(f"{suite_id} {tier} trace is missing required fields: {missing}")
    return arrays
