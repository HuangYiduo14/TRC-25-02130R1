"""Phase 4 MSA, ordinary FW, and safeguarded CFW with identical trace updates."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import psutil

from partC_rev1_src.partC_rev1_assignment import partC_rev1_cfw_alpha
from partC_rev1_src.partC_rev1_bpr import PartCRev1LineSearchResult, partC_rev1_fw_line_search
from partC_rev1_src.partC_rev1_graph_snapshot import PartCRev1GraphSnapshot
from partC_rev1_src.partC_rev1_shortest_paths import partC_rev1_arc_tie_rank

from .aon import AONExecutionConfig, run_exact_aon
from .checkpoint import Phase4RestartState, save_phase4_checkpoint
from .costs import (
    RefreshedStateMetrics,
    phase4_bpr_cost,
    phase4_bpr_derivative,
    phase4_bpr_objective,
    phase4_scenario_capacity,
    refreshed_state_metrics,
)
from .models import (
    AONTraceResult,
    AggregatedOD,
    SuiteLinearTrace,
    SuiteTraceMap,
    linear_combine_suite_traces,
)
from .tracing import pack_trace_maps


METHODS = {"MSA", "FW", "SAFEGUARDED_CFW"}


@dataclass(frozen=True, slots=True)
class Phase4SolverConfig:
    method: str
    capacity_multiplier: float
    max_iterations: int
    minimum_iterations: int = 1
    gap_target: float = 0.0
    worker_count: int = 1
    max_origins: int | None = None
    tie_tolerance: float = 1e-12
    line_search_derivative_tolerance: float = 1e-8
    line_search_interval_tolerance: float = 1e-10
    line_search_max_iterations: int = 80
    descent_tolerance: float = 1e-9
    denominator_tolerance: float = 1e-12
    cfw_delta: float = 1e-4
    gap_epsilon: float = 1e-12
    max_runtime_seconds: float | None = None
    rss_hard_bytes: int = 24 * 1024**3
    checkpoint_directory: Path | None = None
    checkpoint_iterations: tuple[int, ...] = ()
    checkpoint_compressed: bool = False
    final_checkpoint: bool = False
    iteration_record_path: Path | None = None

    def normalized_method(self) -> str:
        value = self.method.upper()
        if value == "CFW":
            value = "SAFEGUARDED_CFW"
        if value not in METHODS:
            raise ValueError(f"Unsupported Phase 4 assignment method: {self.method}")
        return value

    def validate(self) -> None:
        self.normalized_method()
        if self.max_iterations <= 0:
            raise ValueError("Phase 4 solver requires at least one refreshed update iteration")
        if self.minimum_iterations <= 0 or self.minimum_iterations > self.max_iterations:
            raise ValueError("minimum_iterations must be in [1, max_iterations]")
        if self.worker_count <= 0:
            raise ValueError("worker_count must be positive")
        if self.max_origins is not None and self.max_origins <= 0:
            raise ValueError("max_origins must be positive when specified")
        if self.gap_target < 0.0 or not np.isfinite(self.gap_target):
            raise ValueError("gap_target must be finite and nonnegative")
        if self.rss_hard_bytes <= 0:
            raise ValueError("rss_hard_bytes must be positive")
        if any(value <= 0 or value > self.max_iterations for value in self.checkpoint_iterations):
            raise ValueError("checkpoint_iterations must reference completed update iterations")


@dataclass(slots=True)
class Phase4IterationRecord:
    iteration: int
    evaluated_state_iteration: int
    method: str
    capacity_multiplier: float
    objective_before: float
    objective_after: float
    tstt: float
    sptt: float
    absolute_gap: float
    relative_gap_sptt: float
    relative_gap_tstt: float
    step_size: float
    cfw_alpha: float | None
    cfw_numerator: float | None
    cfw_denominator: float | None
    cfw_ratio: float | None
    cfw_reset: bool
    fallback_reason: str | None
    line_search: dict[str, Any]
    reachable_demand: float
    unreachable_demand: float
    flow_l1_change: float
    cost_update_seconds: float
    aon_tree_load_trace_seconds: float
    line_search_seconds: float
    state_update_seconds: float
    checkpoint_seconds: float
    iteration_wall_seconds: float
    rss_bytes: int
    trace_mixture_coefficients: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Phase4SolverResult:
    method: str
    capacity_multiplier: float
    used_capacity: np.ndarray
    flow: np.ndarray
    cost: np.ndarray
    suite_traces: dict[str, SuiteLinearTrace]
    latest_aon: AONTraceResult
    histories: list[Phase4IterationRecord]
    final_metrics: RefreshedStateMetrics
    stop_reason: str
    runtime_seconds: float
    peak_rss_bytes: int
    initialization_seconds: float
    final_refresh_seconds: float
    ever_active_regional_pair_keys: dict[str, np.ndarray]
    ever_active_bundle_masks: dict[str, np.ndarray]
    checkpoint_receipts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def _rss_bytes() -> int:
    process = psutil.Process()
    total = int(process.memory_info().rss)
    for child in process.children(recursive=True):
        try:
            total += int(child.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def _copy_trace(trace: SuiteLinearTrace) -> SuiteLinearTrace:
    return linear_combine_suite_traces([(1.0, trace)])


def _copy_traces(
    source: dict[str, SuiteLinearTrace],
    trace_maps: Sequence[SuiteTraceMap],
) -> dict[str, SuiteLinearTrace]:
    return {item.suite_id: _copy_trace(source[item.suite_id]) for item in trace_maps}


def _combine_traces(
    left_weight: float,
    left: dict[str, SuiteLinearTrace],
    right_weight: float,
    right: dict[str, SuiteLinearTrace],
    trace_maps: Sequence[SuiteTraceMap],
) -> dict[str, SuiteLinearTrace]:
    return {
        item.suite_id: linear_combine_suite_traces(
            [
                (float(left_weight), left[item.suite_id]),
                (float(right_weight), right[item.suite_id]),
            ]
        )
        for item in trace_maps
    }


def _line_result_dict(result: PartCRev1LineSearchResult | None) -> dict[str, Any]:
    if result is None:
        return {}
    return {
        "eta": float(result.eta),
        "derivative_at_zero": float(result.derivative_at_zero),
        "derivative_at_one": float(result.derivative_at_one),
        "iterations": int(result.iterations),
        "stop_reason": str(result.stop_reason),
    }


def _run_line_search(
    graph: PartCRev1GraphSnapshot,
    used_capacity: np.ndarray,
    flow: np.ndarray,
    direction: np.ndarray,
    cfg: Phase4SolverConfig,
) -> PartCRev1LineSearchResult:
    return partC_rev1_fw_line_search(
        flow,
        direction,
        graph.free_time,
        used_capacity,
        graph.alpha,
        graph.beta,
        derivative_tol=cfg.line_search_derivative_tolerance,
        interval_tol=cfg.line_search_interval_tolerance,
        max_iterations=cfg.line_search_max_iterations,
    )


def _validate_initial_aon(
    initial: AONTraceResult,
    trace_maps: Sequence[SuiteTraceMap],
    *,
    expected_origins: int,
    expected_demand: float,
) -> None:
    initial.validate(trace_maps)
    if initial.processed_origins != expected_origins:
        raise ValueError("Initial AON scope differs from the requested deterministic origin scope")
    if not np.isclose(
        initial.reachable_demand + initial.unreachable_demand,
        expected_demand,
        rtol=1e-12,
        atol=1e-8,
    ):
        raise RuntimeError("Initial AON does not classify all accepted OD demand")


def solve_assignment(
    graph: PartCRev1GraphSnapshot,
    od: AggregatedOD,
    *,
    trace_maps: Sequence[SuiteTraceMap],
    config: Phase4SolverConfig,
    initial_aon: AONTraceResult | None = None,
    restart_state: Phase4RestartState | None = None,
    run_metadata: dict[str, Any] | None = None,
) -> Phase4SolverResult:
    """Solve one bounded Phase 4 pilot while preserving physical/trace mixtures."""

    cfg = config
    cfg.validate()
    method = cfg.normalized_method()
    started = time.perf_counter()
    peak_rss = _rss_bytes()
    used_capacity = phase4_scenario_capacity(graph, cfg.capacity_multiplier)
    aon_config = AONExecutionConfig(
        worker_count=cfg.worker_count,
        max_origins=cfg.max_origins,
        tie_tolerance=cfg.tie_tolerance,
    )
    # Both arrays are immutable for a solver run.  Constructing them once
    # preserves exact shortest-path and trace semantics while removing
    # repeated setup from every refreshed AON.
    arc_tie_rank = np.ascontiguousarray(partC_rev1_arc_tie_rank(graph), dtype=np.int64)
    packed_trace_maps = pack_trace_maps(trace_maps)
    expected_origins = min(
        len(od.origin_group_node_idx),
        cfg.max_origins if cfg.max_origins is not None else len(od.origin_group_node_idx),
    )
    selected_end = int(od.origin_offsets[expected_origins])
    expected_selected_demand = float(od.demand[:selected_end].sum(dtype=np.float64))

    initialization_started = time.perf_counter()
    if restart_state is None:
        if initial_aon is None:
            initial_aon = run_exact_aon(
                graph,
                od,
                graph.free_time,
                trace_maps=trace_maps,
                config=aon_config,
                arc_tie_rank=arc_tie_rank,
                packed_trace_maps=packed_trace_maps,
            )
        _validate_initial_aon(
            initial_aon,
            trace_maps,
            expected_origins=expected_origins,
            expected_demand=expected_selected_demand,
        )
        flow = np.ascontiguousarray(initial_aon.physical_flow.copy())
        current_traces = _copy_traces(initial_aon.suite_traces, trace_maps)
        previous_sight_flow: np.ndarray | None = None
        previous_sight_traces: dict[str, SuiteLinearTrace] | None = None
        previous_direction: np.ndarray | None = None
        previous_step: float | None = None
        next_iteration = 1
        prior_history: list[dict[str, Any]] = []
    else:
        if initial_aon is not None:
            raise ValueError("Specify either an initial AON or a restart state, not both")
        flow = np.ascontiguousarray(np.asarray(restart_state.physical_flow, dtype=np.float64))
        current_traces = _copy_traces(restart_state.suite_traces, trace_maps)
        previous_sight_flow = (
            np.ascontiguousarray(restart_state.previous_sight_flow, dtype=np.float64)
            if restart_state.previous_sight_flow is not None
            else None
        )
        previous_sight_traces = (
            _copy_traces(restart_state.previous_sight_traces, trace_maps)
            if restart_state.previous_sight_traces is not None
            else None
        )
        previous_direction = (
            np.ascontiguousarray(restart_state.previous_direction, dtype=np.float64)
            if restart_state.previous_direction is not None
            else None
        )
        previous_step = restart_state.previous_step_size
        next_iteration = int(restart_state.next_iteration)
        prior_history = list(restart_state.history)
    initialization_seconds = time.perf_counter() - initialization_started

    if flow.shape != graph.free_time.shape or np.any(~np.isfinite(flow)) or np.any(flow < -1e-8):
        raise ValueError("Initial/restart physical flow is invalid")
    for item in trace_maps:
        current_traces[item.suite_id].validate(item)
    if next_iteration <= 0 or next_iteration > cfg.max_iterations + 1:
        raise ValueError("Restart next-iteration index is outside the configured run")

    histories: list[Phase4IterationRecord] = []
    checkpoint_receipts: list[dict[str, Any]] = []
    ever_active_pairs = {
        item.suite_id: current_traces[item.suite_id].coarse_regional_pair_flow.keys.copy()
        for item in trace_maps
    }
    ever_active_bundles = {
        item.suite_id: current_traces[item.suite_id].bundle_flow > 0.0 for item in trace_maps
    }
    stop_reason = "max_iterations"
    prefetched_aon: AONTraceResult | None = None

    for iteration in range(next_iteration, cfg.max_iterations + 1):
        iteration_started = time.perf_counter()
        cost_started = time.perf_counter()
        cost = phase4_bpr_cost(graph, flow, used_capacity)
        cost_seconds = time.perf_counter() - cost_started

        aon_started = time.perf_counter()
        if prefetched_aon is None:
            aon = run_exact_aon(
                graph,
                od,
                cost,
                trace_maps=trace_maps,
                config=aon_config,
                arc_tie_rank=arc_tie_rank,
                packed_trace_maps=packed_trace_maps,
            )
        else:
            aon = prefetched_aon
            prefetched_aon = None
        aon_seconds = time.perf_counter() - aon_started
        if not np.isclose(
            aon.reachable_demand + aon.unreachable_demand,
            expected_selected_demand,
            rtol=1e-12,
            atol=1e-8,
        ):
            raise RuntimeError("A Phase 4 iteration did not classify all accepted OD demand")
        for item in trace_maps:
            suite_id = item.suite_id
            ever_active_pairs[suite_id] = np.union1d(
                ever_active_pairs[suite_id],
                aon.suite_traces[suite_id].coarse_regional_pair_flow.keys,
            )
            ever_active_bundles[suite_id] |= aon.suite_traces[suite_id].bundle_flow > 0.0
        metrics = refreshed_state_metrics(
            graph,
            flow,
            used_capacity,
            sptt=aon.sptt,
            epsilon=cfg.gap_epsilon,
        )

        y = aon.physical_flow
        fw_direction = y - flow
        sight_flow = y
        sight_traces = aon.suite_traces
        cfw_alpha: float | None = None
        cfw_meta: dict[str, float | str] = {}
        cfw_reset = False
        fallback_reason: str | None = None

        if method == "SAFEGUARDED_CFW":
            if (
                previous_sight_flow is None
                or previous_sight_traces is None
                or previous_direction is None
                or previous_step is None
                or previous_step >= 1.0 - 1e-8
            ):
                cfw_alpha = 0.0
                cfw_reset = True
                fallback_reason = "cfw_initial_or_unretained_fw_reset"
            else:
                hessian = phase4_bpr_derivative(graph, flow, used_capacity)
                cfw_alpha, cfw_meta = partC_rev1_cfw_alpha(
                    previous_sight_flow,
                    flow,
                    fw_direction,
                    previous_step,
                    previous_direction,
                    hessian,
                    denominator_tol=cfg.denominator_tolerance,
                    delta_cfw=cfg.cfw_delta,
                )
                if (
                    not np.isfinite(cfw_alpha)
                    or cfw_meta.get("reason") == "denominator_tol"
                    or np.any(~np.isfinite(hessian))
                ):
                    cfw_alpha = 0.0
                    cfw_reset = True
                    fallback_reason = "cfw_invalid_coefficient_fw_reset"
                elif cfw_alpha > 0.0:
                    sight_flow = cfw_alpha * previous_sight_flow + (1.0 - cfw_alpha) * y
                    sight_traces = _combine_traces(
                        cfw_alpha,
                        previous_sight_traces,
                        1.0 - cfw_alpha,
                        aon.suite_traces,
                        trace_maps,
                    )
                direction_test = sight_flow - flow
                descent = float(np.dot(cost, direction_test))
                if not np.isfinite(descent) or descent >= -cfg.descent_tolerance:
                    sight_flow = y
                    sight_traces = aon.suite_traces
                    cfw_alpha = 0.0
                    cfw_reset = True
                    fallback_reason = "cfw_non_descent_fw_reset"

        direction = sight_flow - flow
        line_started = time.perf_counter()
        line_result: PartCRev1LineSearchResult | None = None
        if method == "MSA":
            # x^0 is the feasible free-flow AON; the first refreshed update is
            # k=1 and therefore uses gamma_1=1/(1+1)=1/2.
            step_size = 1.0 / float(iteration + 1)
        else:
            try:
                line_result = _run_line_search(graph, used_capacity, flow, direction, cfg)
                step_size = float(line_result.eta)
            except (FloatingPointError, OverflowError, ValueError) as exc:
                if method != "SAFEGUARDED_CFW" or np.shares_memory(sight_flow, y):
                    raise
                sight_flow = y
                sight_traces = aon.suite_traces
                direction = fw_direction
                cfw_alpha = 0.0
                cfw_reset = True
                fallback_reason = f"cfw_line_search_exception_fw_reset:{type(exc).__name__}"
                line_result = _run_line_search(graph, used_capacity, flow, direction, cfg)
                step_size = float(line_result.eta)
            if method == "SAFEGUARDED_CFW" and line_result.stop_reason == "boundary_zero" and cfw_alpha:
                sight_flow = y
                sight_traces = aon.suite_traces
                direction = fw_direction
                cfw_alpha = 0.0
                cfw_reset = True
                fallback_reason = "cfw_zero_step_fw_reset"
                line_result = _run_line_search(graph, used_capacity, flow, direction, cfg)
                step_size = float(line_result.eta)
        line_seconds = time.perf_counter() - line_started
        if not np.isfinite(step_size) or step_size < 0.0 or step_size > 1.0:
            raise RuntimeError("Assignment line search produced an invalid convex coefficient")

        update_started = time.perf_counter()
        new_flow = (1.0 - step_size) * flow + step_size * sight_flow
        new_flow[np.abs(new_flow) <= 1e-13] = 0.0
        if np.any(~np.isfinite(new_flow)) or np.any(new_flow < -1e-8):
            raise RuntimeError("Assignment update produced invalid or negative physical flow")
        new_flow = np.maximum(new_flow, 0.0)
        new_traces = _combine_traces(
            1.0 - step_size,
            current_traces,
            step_size,
            sight_traces,
            trace_maps,
        )
        objective_after = phase4_bpr_objective(graph, new_flow, used_capacity)
        if method != "MSA" and objective_after > metrics.objective + 1e-9 * max(abs(metrics.objective), 1.0):
            if method != "SAFEGUARDED_CFW" or cfw_alpha == 0.0:
                raise RuntimeError("FW line search increased the Beckmann objective")
            sight_flow = y
            sight_traces = aon.suite_traces
            direction = fw_direction
            cfw_alpha = 0.0
            cfw_reset = True
            fallback_reason = "cfw_objective_increase_fw_reset"
            line_result = _run_line_search(graph, used_capacity, flow, direction, cfg)
            step_size = float(line_result.eta)
            new_flow = (1.0 - step_size) * flow + step_size * sight_flow
            new_flow[np.abs(new_flow) <= 1e-13] = 0.0
            new_traces = _combine_traces(
                1.0 - step_size,
                current_traces,
                step_size,
                sight_traces,
                trace_maps,
            )
            objective_after = phase4_bpr_objective(graph, new_flow, used_capacity)
            if objective_after > metrics.objective + 1e-9 * max(abs(metrics.objective), 1.0):
                raise RuntimeError("Ordinary-FW reset still increased the Beckmann objective")
        flow_l1_change = float(np.sum(np.abs(new_flow - flow), dtype=np.float64))
        update_seconds = time.perf_counter() - update_started

        previous_sight_flow = np.ascontiguousarray(sight_flow.copy())
        previous_sight_traces = _copy_traces(sight_traces, trace_maps)
        previous_direction = np.ascontiguousarray(direction.copy())
        previous_step = float(step_size)
        flow = np.ascontiguousarray(new_flow)
        current_traces = new_traces

        checkpoint_seconds = 0.0
        if iteration in set(cfg.checkpoint_iterations):
            checkpoint_started = time.perf_counter()
            checkpoint_cost = phase4_bpr_cost(graph, flow, used_capacity)
            prefetched_aon = run_exact_aon(
                graph,
                od,
                checkpoint_cost,
                trace_maps=trace_maps,
                config=aon_config,
                arc_tie_rank=arc_tie_rank,
                packed_trace_maps=packed_trace_maps,
            )
            state = Phase4RestartState(
                next_iteration=iteration + 1,
                physical_flow=flow,
                suite_traces=current_traces,
                previous_sight_flow=previous_sight_flow,
                previous_sight_traces=previous_sight_traces,
                previous_direction=previous_direction,
                previous_step_size=previous_step,
                history=prior_history + [item.to_dict() for item in histories],
                metadata={"method": method, "capacity_multiplier": cfg.capacity_multiplier},
            )
            if cfg.checkpoint_directory is None:
                raise ValueError("checkpoint_iterations requires checkpoint_directory")
            receipt = save_phase4_checkpoint(
                Path(cfg.checkpoint_directory) / f"iteration_{iteration:04d}",
                state,
                trace_maps=trace_maps,
                latest_aon=prefetched_aon,
                run_metadata=run_metadata or {},
                compressed=cfg.checkpoint_compressed,
            )
            checkpoint_receipts.append(receipt)
            checkpoint_seconds = time.perf_counter() - checkpoint_started

        rss = _rss_bytes()
        peak_rss = max(peak_rss, rss)
        if rss >= cfg.rss_hard_bytes:
            raise MemoryError(f"Phase 4 process-tree RSS reached the hard limit: {rss}")
        record = Phase4IterationRecord(
            iteration=iteration,
            evaluated_state_iteration=iteration - 1,
            method=method,
            capacity_multiplier=float(cfg.capacity_multiplier),
            objective_before=float(metrics.objective),
            objective_after=float(objective_after),
            tstt=float(metrics.tstt),
            sptt=float(metrics.sptt),
            absolute_gap=float(metrics.absolute_gap),
            relative_gap_sptt=float(metrics.relative_gap_sptt),
            relative_gap_tstt=float(metrics.relative_gap_tstt),
            step_size=float(step_size),
            cfw_alpha=float(cfw_alpha) if cfw_alpha is not None else None,
            cfw_numerator=(
                float(cfw_meta["numerator"]) if "numerator" in cfw_meta else None
            ),
            cfw_denominator=(
                float(cfw_meta["denominator"]) if "denominator" in cfw_meta else None
            ),
            cfw_ratio=float(cfw_meta["ratio"]) if "ratio" in cfw_meta else None,
            cfw_reset=bool(cfw_reset),
            fallback_reason=fallback_reason,
            line_search=_line_result_dict(line_result),
            reachable_demand=float(aon.reachable_demand),
            unreachable_demand=float(aon.unreachable_demand),
            flow_l1_change=flow_l1_change,
            cost_update_seconds=float(cost_seconds),
            aon_tree_load_trace_seconds=float(aon_seconds),
            line_search_seconds=float(line_seconds),
            state_update_seconds=float(update_seconds),
            checkpoint_seconds=float(checkpoint_seconds),
            iteration_wall_seconds=float(time.perf_counter() - iteration_started),
            rss_bytes=int(rss),
            trace_mixture_coefficients={
                "current": float(1.0 - step_size),
                "search_point": float(step_size),
                "retained_sight_within_search_point": float(cfw_alpha or 0.0),
                "new_aon_within_search_point": float(1.0 - (cfw_alpha or 0.0)),
            },
        )
        histories.append(record)
        if cfg.iteration_record_path is not None:
            record_path = Path(cfg.iteration_record_path)
            record_path.parent.mkdir(parents=True, exist_ok=True)
            with record_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record.to_dict(), sort_keys=True, allow_nan=False))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

        completed = iteration
        if completed >= cfg.minimum_iterations and cfg.gap_target > 0.0:
            # This record's refreshed gap is for x^(iteration-1). A stop is
            # accepted only on the next refresh/final refresh, never stale.
            if prefetched_aon is not None:
                prefetched_metrics = refreshed_state_metrics(
                    graph,
                    flow,
                    used_capacity,
                    sptt=prefetched_aon.sptt,
                    epsilon=cfg.gap_epsilon,
                )
                if prefetched_metrics.relative_gap_sptt <= cfg.gap_target:
                    stop_reason = "refreshed_gap_target_met"
                    break
        elapsed = time.perf_counter() - started
        if cfg.max_runtime_seconds is not None and elapsed >= cfg.max_runtime_seconds:
            stop_reason = "runtime_limit_after_minimum" if completed >= cfg.minimum_iterations else "runtime_limit_before_minimum"
            break

    final_refresh_started = time.perf_counter()
    final_cost = phase4_bpr_cost(graph, flow, used_capacity)
    if prefetched_aon is None:
        final_aon = run_exact_aon(
            graph,
            od,
            final_cost,
            trace_maps=trace_maps,
            config=aon_config,
            arc_tie_rank=arc_tie_rank,
            packed_trace_maps=packed_trace_maps,
        )
    else:
        final_aon = prefetched_aon
    final_refresh_seconds = time.perf_counter() - final_refresh_started
    final_metrics = refreshed_state_metrics(
        graph,
        flow,
        used_capacity,
        sptt=final_aon.sptt,
        epsilon=cfg.gap_epsilon,
    )
    if not np.isclose(
        final_aon.reachable_demand + final_aon.unreachable_demand,
        expected_selected_demand,
        rtol=1e-12,
        atol=1e-8,
    ):
        raise RuntimeError("Final refreshed state does not classify all accepted OD demand")
    for item in trace_maps:
        suite_id = item.suite_id
        ever_active_pairs[suite_id] = np.union1d(
            ever_active_pairs[suite_id],
            final_aon.suite_traces[suite_id].coarse_regional_pair_flow.keys,
        )
        ever_active_bundles[suite_id] |= final_aon.suite_traces[suite_id].bundle_flow > 0.0

    if cfg.final_checkpoint:
        if cfg.checkpoint_directory is None:
            raise ValueError("final_checkpoint requires checkpoint_directory")
        completed_iterations = histories[-1].iteration if histories else next_iteration - 1
        state = Phase4RestartState(
            next_iteration=completed_iterations + 1,
            physical_flow=flow,
            suite_traces=current_traces,
            previous_sight_flow=previous_sight_flow,
            previous_sight_traces=previous_sight_traces,
            previous_direction=previous_direction,
            previous_step_size=previous_step,
            history=prior_history + [item.to_dict() for item in histories],
            metadata={"method": method, "capacity_multiplier": cfg.capacity_multiplier},
        )
        receipt = save_phase4_checkpoint(
            Path(cfg.checkpoint_directory) / f"final_iteration_{completed_iterations:04d}",
            state,
            trace_maps=trace_maps,
            latest_aon=final_aon,
            run_metadata=run_metadata or {},
            compressed=cfg.checkpoint_compressed,
        )
        checkpoint_receipts.append(receipt)

    peak_rss = max(peak_rss, _rss_bytes())
    reset_count = sum(int(item.cfw_reset) for item in histories)
    return Phase4SolverResult(
        method=method,
        capacity_multiplier=float(cfg.capacity_multiplier),
        used_capacity=used_capacity,
        flow=flow,
        cost=final_cost,
        suite_traces=current_traces,
        latest_aon=final_aon,
        histories=histories,
        final_metrics=final_metrics,
        stop_reason=stop_reason,
        runtime_seconds=float(time.perf_counter() - started),
        peak_rss_bytes=int(peak_rss),
        initialization_seconds=float(initialization_seconds),
        final_refresh_seconds=float(final_refresh_seconds),
        ever_active_regional_pair_keys=ever_active_pairs,
        ever_active_bundle_masks=ever_active_bundles,
        checkpoint_receipts=checkpoint_receipts,
        metadata={
            "schema_id": "phase4.solver_result.v1",
            "graph_hash": graph.graph_hash,
            "od_hash": od.od_hash,
            "suite_ids": [item.suite_id for item in trace_maps],
            "processed_origins": expected_origins,
            "total_origins": len(od.origin_group_node_idx),
            "accepted_selected_demand": expected_selected_demand,
            "final_reachable_demand": float(final_aon.reachable_demand),
            "final_unreachable_demand": float(final_aon.unreachable_demand),
            "unreachable_demand_is_explicit_not_silently_dropped": True,
            "iteration_convention": (
                "x^0 is the feasible free-flow AON; refreshed update k starts at 1; "
                "MSA gamma_k=1/(k+1), so gamma_1=1/2"
            ),
            "cfw_formula": (
                "dbar=(1-tau_previous)*d_cfw_previous; "
                "alpha=clip(<dbar,H,d_fw>/<dbar,H,(d_fw-dbar)>,0,1-delta); "
                "sight=alpha*sight_previous+(1-alpha)*y; invalid/non-descent/failed "
                "line search resets to ordinary FW"
            ),
            "cfw_reset_count": reset_count,
            "physical_and_trace_coefficients_identical": True,
            "prepared_capacity_identity": "graph.capacity",
            "used_capacity_rule": "capacity_multiplier * graph.capacity exactly once",
            "capacity_multiplier": float(cfg.capacity_multiplier),
            "gap_denominator_recommendation_candidate": "SPTT",
            "gap_epsilon": float(cfg.gap_epsilon),
            "prior_restart_history_records": len(prior_history),
        },
    )
