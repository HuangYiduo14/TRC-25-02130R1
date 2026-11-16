"""Detailed-regional FW / optional safeguarded CFW labels, immutable sample roles and bounded append-only runs.

No historical label/model reuse or graph discovery is performed. Every cost label
is a fresh shortest-path vector at the recorded feasible detailed-regional flow.
The FW gap bounds potential error, not coordinatewise marginal-cost error.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import copy
import json
import math
from pathlib import Path
import time

import numpy as np

from partC_rev1_src.pacnhn_v1.control import array_hash, read_json, save_npz, sha, write_json
from partC_rev1_src.pacnhn_v1.oracle import Network, Demand, aon, conservation
from partC_rev1_src.pacnhn_v1.solver import PolynomialObjective, certificate, line_search, partC_rev1_cfw_alpha

ROLES = {"TRAIN", "VALID", "TEST"}


class RegionalInvariantError(ArithmeticError):
    """A region-wide topology/identity failure; dependent labels must stop."""


def _numeric(value, name, *, shape=None, nonnegative=True):
    raw = np.asarray(value)
    if not np.isrealobj(raw):
        raise ValueError(name+": complex values are invalid")
    a = np.asarray(value, np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all() or (nonnegative and np.any(a < 0)):
        raise ValueError(name+": invalid native shape/value")
    return a


@dataclass
class RegionGraph:
    network: Network
    base: np.ndarray
    k4: np.ndarray
    pair_source: np.ndarray
    pair_target: np.ndarray
    pair_keys: np.ndarray
    baseline_open: np.ndarray
    baseline_i95: np.ndarray
    c0: np.ndarray | None
    graph_sha256: str
    identity: str
    support_identity: str
    source_path: str

    @property
    def m(self):
        return len(self.pair_source)

    def demand(self, u):
        q = _numeric(u, "regional demand", shape=(self.m,))
        order = np.lexsort((self.pair_target, self.pair_source))
        origins, starts = np.unique(self.pair_source[order], return_index=True)
        offsets = np.r_[starts, self.m].astype(np.int64)
        return Demand(origins, offsets, self.pair_target[order], q[order], np.ones(self.m, bool)), order


def load_region_graph(path, *, expected_sha256=None, expected_identity=None):
    path = Path(path).resolve(strict=True)
    digest = sha(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Detailed-regional graph file hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        fields = ("tail", "head", "node_ids", "edge_ids", "base", "k4", "pair_source", "pair_target",
                  "pair_keys", "baseline_open", "baseline_i95")
        a = {k: z[k].copy() for k in fields}
        c0 = z["c0"].copy() if "c0" in z.files else None
    n, edges, m = len(a["node_ids"]), len(a["tail"]), len(a["pair_source"])
    if not n or not edges or not m:
        raise ValueError("An active regional label task requires nodes, arcs and pairs")
    for key in ("tail", "head", "pair_source", "pair_target"):
        expected = (edges,) if key in {"tail", "head"} else (m,)
        if a[key].dtype != np.dtype("int64") or a[key].shape != expected or np.any(a[key] < 0) or np.any(a[key] >= n):
            raise ValueError(key+": require in-range local int64 endpoints")
    if a["node_ids"].shape != (n,) or a["edge_ids"].shape != (edges,):
        raise ValueError("Invalid node/edge identity dimensions")
    if a["node_ids"].dtype.kind not in "US" or a["edge_ids"].dtype.kind not in "US":
        raise ValueError("Node/edge identities must be non-object strings")
    if a["pair_keys"].dtype != np.dtype("uint64") or a["pair_keys"].shape != (m,) or len(np.unique(a["pair_keys"])) != m:
        raise ValueError("Pair keys must be unique ordered uint64 identities")
    if len(np.unique(np.column_stack([a["pair_source"], a["pair_target"]]), axis=0)) != m:
        raise ValueError("Duplicate endpoint pairs in active support")
    for key in ("base", "k4"):
        a[key] = _numeric(a[key], key, shape=(edges,))
    for key in ("baseline_open", "baseline_i95"):
        a[key] = _numeric(a[key], key, shape=(m,))
    if c0 is not None:
        c0 = _numeric(c0, "c0", shape=(m,))
    # Underlying physical vertices, including centroid vertices, permit transit.
    graph = Network(a["tail"], a["head"], np.zeros(n, bool), a["edge_ids"], a["node_ids"])
    support = array_hash({k: a[k] for k in ("pair_source", "pair_target", "pair_keys")})
    identity = array_hash(a)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError("Detailed-regional graph/support identity mismatch")
    return RegionGraph(graph, a["base"], a["k4"], a["pair_source"], a["pair_target"], a["pair_keys"],
        a["baseline_open"], a["baseline_i95"], c0, digest, identity, support, str(path))


def _load(region, od, cost, guard):
    result = aon(region.network, od, cost, workers=1, batch_origins=16, hash_capacity=16, guard=guard)
    if not np.isfinite(result["distances"]).all():
        raise RegionalInvariantError("UNREACHABLE_ACTIVE_PAIR: zero-demand pairs also require directed paths")
    if abs(result["reachable_demand"]-od.total) > 1e-10*max(1., od.total):
        raise ArithmeticError("Regional served demand changed; renormalization prohibited")
    return result


def solve_label(region, u, *, max_seconds, max_iterations, relative_gap_tolerance=1e-6,
                deadline_epoch=None, guard=None, method="FW", cfw_reset_on_bound=False,
                iterate_selection="LAST_CERTIFIED"):
    """Bounded local FW/CFW, including exact zero-demand labels and fresh final evidence.

    LAST_CERTIFIED preserves the existing output policy. BEST_CERTIFIED_RG
    retains the complete feasible state with the smallest finite certified RG,
    without changing the search trajectory, termination rule or quality thresholds.
    A zero-SPTT certificate with arithmetic-zero gap is handled explicitly.
    Selection uses certificates already freshly computed at the retained state;
    no historical scalar gap is imported and no extra AON runs after the deadline.
    If a later iteration times out, only a fully certified state is returned.
    Without a completed initial certificate, TimeoutError is propagated instead
    of assigning the previous state's gap to a new/unexamined flow.
    """
    if not isinstance(region, RegionGraph):
        raise TypeError("Explicit authenticated RegionGraph required")
    if (not math.isfinite(max_seconds) or max_seconds <= 0 or not isinstance(max_iterations, int)
            or max_iterations < 1 or not math.isfinite(relative_gap_tolerance) or relative_gap_tolerance <= 0):
        raise ValueError("Explicit positive local wall, update and accuracy budgets required")
    if method not in {"FW", "CFW"}:
        raise ValueError("Explicit local method must be FW or CFW")
    if not isinstance(cfw_reset_on_bound,bool):
        raise ValueError("cfw_reset_on_bound requires an explicit boolean")
    if iterate_selection not in {"LAST_CERTIFIED", "BEST_CERTIFIED_RG"}:
        raise ValueError("Explicit iterate_selection must be LAST_CERTIFIED or BEST_CERTIFIED_RG")
    q = _numeric(u, "u", shape=(region.m,)).copy()
    started = time.time(); mono_started = time.perf_counter()
    end = min(started+max_seconds, deadline_epoch if deadline_epoch is not None else started+max_seconds)
    if not math.isfinite(end):
        raise ValueError("Finite original deadline required")
    def check():
        if time.time() >= end or time.perf_counter()-mono_started >= max_seconds:
            raise TimeoutError("Regional label wall/deadline limit")
        if guard is not None:
            guard()
    objective = PolynomialObjective(region.base, region.k4)
    od, order = region.demand(q)
    balance = od.balance(len(region.network.node_ids))
    history = []; latest = None; latest_flow = None; latest_tau = None; latest_cost = None
    c0 = None; flow = None; iteration = 0; accepted_updates = 0; reason = "RUNNING"
    previous_sight = previous_direction = previous_step = None
    cfw_evaluations = cfw_accepted = cfw_resets = cfw_bound_resets = 0
    direction_seconds = line_search_seconds = 0.
    best_snapshot = None; best_score = math.inf; finite_certified_states = 0
    snapshot_seconds = 0.; snapshot_updates = 0
    try:
        check()
        initial = _load(region, od, region.base, check)
        c0 = np.empty(region.m, np.float64); c0[order] = initial["distances"]
        if region.c0 is not None and not np.allclose(c0, region.c0, rtol=1e-12, atol=1e-10):
            raise RegionalInvariantError("Supplied c0 differs from this exact detailed-local graph")
        flow = initial["flow"].copy()
        while True:
            check(); t = time.perf_counter()
            value, cost = objective.value_cost(flow)
            if not np.isfinite(value) or not np.isfinite(cost).all() or np.any(cost < 0):
                raise FloatingPointError("Invalid detailed-regional polynomial state")
            linear = _load(region, od, cost, check)
            tau = np.empty(region.m, np.float64); tau[order] = linear["distances"]
            cert = certificate(flow, cost, linear, value, has_potential=True)
            residual = conservation(region.network, flow, balance, od.total)
            if np.min(tau-c0) < -1e-10*max(1., float(c0.max(initial=0.))):
                raise ArithmeticError("Detailed nonnegative congestion produced below-freeflow label")
            # Only the immediately preceding tau is needed for the exported
            # stability statistic. Keep O(pair_count), not O(iterations*pairs),
            # transient marginal-cost vectors; exported history is unchanged.
            previous_tau = latest_tau
            latest = dict(cert, **residual); latest_flow = flow.copy(); latest_tau = tau.copy(); latest_cost = cost.copy()
            stability = None if previous_tau is None else float(np.max(np.abs(tau-previous_tau)))
            history.append(dict(iteration=iteration, elapsed_seconds=time.perf_counter()-mono_started, certificate=latest,
                                max_cost_change_since_previous_certificate=stability))
            if iterate_selection == "BEST_CERTIFIED_RG":
                # Conservation above is a hard gate. Copy all state-dependent
                # outputs together, after fresh ordinary-AON certification.
                snapshot_started = time.perf_counter()
                candidate_rg = cert["RG_SPTT"]
                finite_rg = candidate_rg is not None and math.isfinite(candidate_rg)
                zero_denominator = (candidate_rg is None and cert["SPTT"] == 0 and
                    math.isfinite(cert["raw_signed_gap"]) and
                    abs(cert["raw_signed_gap"]) <= cert["arithmetic_allowance"])
                finite_certified_states += int(finite_rg)
                score = float(candidate_rg) if finite_rg else 0. if zero_denominator else math.inf
                if (finite_rg or zero_denominator) and (best_snapshot is None or score < best_score):
                    vectors = {"flow": latest_flow.copy(), "tau": latest_tau.copy(),
                               "edge_cost": latest_cost.copy()}
                    for vector in vectors.values():
                        vector.setflags(write=False)
                    best_snapshot = dict(**vectors, certificate=copy.deepcopy(latest),
                        iteration=iteration, last_cost_change=stability,
                        selection_basis="FINITE_RG" if finite_rg else "ZERO_SPTT_ARITHMETIC_ZERO_GAP")
                    best_score = score; snapshot_updates += 1
                snapshot_seconds += time.perf_counter()-snapshot_started
            elapsed_refresh = time.perf_counter()-t
            rg = cert["RG_SPTT"]
            numerically_zero = abs(cert["raw_signed_gap"]) <= cert["arithmetic_allowance"]
            if (rg is not None and rg <= relative_gap_tolerance) or (rg is None and numerically_zero):
                reason = "CONVERGED"; break
            if iteration >= max_iterations:
                reason = "ITERATION_LIMIT"; break
            if time.time()+max(.01, 1.5*elapsed_refresh) >= end:
                reason = "TIME_LIMIT"; break
            # Same safeguarded conjugate FW construction as the national
            # separable-polynomial solver. All search points are convex mixtures
            # of feasible AON/search flows for this exact unchanged demand.
            direction_started = time.perf_counter()
            sight = linear["flow"]; beta = 0.; cfw_meta = {}
            if method == "CFW" and previous_sight is not None and previous_step < 1-1e-8:
                hessian = objective.hessian_diag(flow)
                if not np.isfinite(hessian).all() or np.any(hessian < 0):
                    raise FloatingPointError("Invalid local polynomial curvature")
                cfw_evaluations += 1
                beta, cfw_meta = partC_rev1_cfw_alpha(previous_sight, flow, sight-flow,
                    previous_step, previous_direction, hessian, denominator_tol=1e-12, delta_cfw=1e-4)
                finite_meta = all(not isinstance(v, (float, np.floating)) or math.isfinite(v)
                                  for v in cfw_meta.values())
                if not math.isfinite(beta) or not 0 <= beta <= 1-1e-4 or not finite_meta:
                    beta = 0.; cfw_meta = {"reset": "invalid_or_nonfinite_coefficient"}; cfw_resets += 1
                elif cfw_reset_on_bound and (beta >= 1-1e-4 or
                        isinstance(cfw_meta.get("ratio"),(int,float,np.floating)) and cfw_meta["ratio"] > 1-1e-4):
                    # Fixed safeguard at the helper's existing coefficient cap;
                    # no data-dependent threshold or altered acceptance tolerance.
                    cfw_meta.update(reset="upper_coefficient_bound", beta_before_reset=float(beta),
                                    coefficient_upper_bound=1-1e-4)
                    beta = 0.; sight = linear["flow"]
                    cfw_resets += 1; cfw_bound_resets += 1
                    # Preserve the attempted reset reason even if line search
                    # times out. It does not claim an accepted flow update.
                    history[-1].update(cfw_beta=beta,cfw_details=cfw_meta.copy())
                elif beta > 0:
                    sight = beta*previous_sight+(1-beta)*linear["flow"]
                    conservation(region.network, sight, balance, od.total)
                    if float(cost @ (sight-flow)) >= -1e-9:
                        beta = 0.; sight = linear["flow"]
                        cfw_meta["reset"] = "non_descent"; cfw_resets += 1
            direction_seconds += time.perf_counter()-direction_started
            search_started = time.perf_counter()
            try:
                alpha, _, search = line_search(objective, flow, sight, value, cost, check)
                if alpha == 0 and beta > 0:
                    beta = 0.; sight = linear["flow"]
                    cfw_meta["reset"] = "zero_step"; cfw_resets += 1
                    alpha, _, search = line_search(objective, flow, sight, value, cost, check)
            finally:
                # Nested detail includes interrupted searches; it is never added
                # to complete_label_elapsed_seconds as a separate wall interval.
                line_search_seconds += time.perf_counter()-search_started
            history[-1].update(step=alpha, cfw_beta=beta, cfw_details=cfw_meta, line_search=search)
            if alpha == 0:
                reason = "STAGNATION"; break
            trial = (1-alpha)*flow+alpha*sight
            conservation(region.network, trial, balance, od.total)
            trial_value, _ = objective.value_cost(trial)
            if trial_value > value+1e-10*max(1., abs(value)):
                raise ArithmeticError("Regional accepted "+method+" objective increased")
            previous_sight = sight; previous_direction = sight-flow; previous_step = alpha
            cfw_accepted += int(beta > 0)
            flow = trial; iteration += 1; accepted_updates = iteration
    except TimeoutError:
        reason = "TIME_LIMIT"
        if latest is None:
            raise
    last_certificate = copy.deepcopy(latest)
    last_iteration = history[-1]["iteration"]
    selected_iteration = last_iteration
    selected_cost_change = history[-1]["max_cost_change_since_previous_certificate"]
    selection_basis = "LAST_FULLY_CERTIFIED_STATE"
    if iterate_selection == "BEST_CERTIFIED_RG":
        if best_snapshot is None:
            raise FloatingPointError("No finite conserved certificate eligible for BEST_CERTIFIED_RG")
        latest = copy.deepcopy(best_snapshot["certificate"])
        latest_flow = best_snapshot["flow"].copy()
        latest_tau = best_snapshot["tau"].copy()
        latest_cost = best_snapshot["edge_cost"].copy()
        selected_iteration = best_snapshot["iteration"]
        selected_cost_change = best_snapshot["last_cost_change"]
        selection_basis = best_snapshot["selection_basis"]
    rg = latest["RG_SPTT"]
    zero_gap = abs(latest["raw_signed_gap"]) <= latest["arithmetic_allowance"]
    quality = "A" if ((rg is not None and rg <= 1e-6) or (rg is None and zero_gap)) else (
        "B" if rg is not None and rg <= 1e-5 else "C" if rg is not None and rg <= 1e-4 else "QUARANTINE")
    gap_upper = max(0., latest["raw_signed_gap"])+latest["arithmetic_allowance"]
    upper = float(latest["value"])
    lower = max(0., upper-gap_upper, float(c0 @ q))
    if lower > upper+latest["arithmetic_allowance"]:
        raise ArithmeticError("Invalid certified potential interval")
    record = dict(status=reason, quality=quality, admitted=quality in {"A", "B"},
        graph_identity=region.identity, graph_sha256=region.graph_sha256, support_identity=region.support_identity,
        input_identity=array_hash({"u": q}), source_graph_path=region.source_path, pair_count=region.m,
        input_total=float(q.sum()), zero_input=bool(np.all(q == 0)),
        solver=("LOCAL_FW_DETAILED_POLYNOMIAL" if method == "FW" else
                "LOCAL_SAFEGUARDED_CFW_BOUND_RESET_DETAILED_POLYNOMIAL_V1" if cfw_reset_on_bound else
                "LOCAL_SAFEGUARDED_CFW_DETAILED_POLYNOMIAL_V1"),
        solver_method=method, initialization="COLD_BASE_COST_AON_SAME_INPUT",
        solver_source_sha256=sha(Path(__file__)),
        cfw_direction_evaluations=cfw_evaluations, cfw_accepted_updates=cfw_accepted, cfw_reset_count=cfw_resets,
        cfw_reset_on_bound=cfw_reset_on_bound, cfw_bound_reset_active=method == "CFW" and cfw_reset_on_bound,
        cfw_bound_reset_count=cfw_bound_resets, cfw_coefficient_upper_bound=1-1e-4,
        nested_direction_seconds=direction_seconds, nested_line_search_seconds=line_search_seconds,
        accepted_updates=accepted_updates, reported_state_iteration=selected_iteration,
        iterate_selection=iterate_selection, selected_state_iteration=selected_iteration,
        selected_state_hash=latest["state_hash"], selection_basis=selection_basis,
        last_certified_iteration=last_iteration, last_certified_state_hash=last_certificate["state_hash"],
        last_certified_certificate=last_certificate,
        last_certified_cost_change=history[-1]["max_cost_change_since_previous_certificate"],
        finite_certified_selection_candidates=finite_certified_states if iterate_selection == "BEST_CERTIFIED_RG" else None,
        best_state_snapshot_updates=snapshot_updates, nested_snapshot_seconds=snapshot_seconds,
        additional_retained_state_vector_bytes=(sum(best_snapshot[k].nbytes for k in ("flow", "tau", "edge_cost"))
            if best_snapshot is not None else 0),
        certificate=latest, fresh_certificate=True, potential_lower=lower, potential_upper=upper,
        potential_error_bound=gap_upper, marginal_cost_error_bound_available=False,
        marginal_cost_interpretation="Shortest-path costs at the recorded approximate regional UE; FW gap does not bound each coordinate error",
        last_cost_change=selected_cost_change,
        complete_label_elapsed_seconds=time.perf_counter()-mono_started, complete_label_start_epoch=started,
        deadline_epoch=end, relative_gap_target=relative_gap_tolerance,
        max_iterations=max_iterations, max_seconds=max_seconds,
        history=[{k: v for k, v in row.items() if k != "tau"} for row in history])
    return dict(record=record, u=q, tau=latest_tau, c0=c0, flow=latest_flow, edge_cost=latest_cost,
                pair_keys=region.pair_keys.copy())


def sample_requests(region, *, role, count, seed, start_index=0, existing_input_ids=(), reference_scale=None,
                    log_amplitude_bounds=(.05, 4.)):
    """Deterministic, prefix-stable unique inputs with roles fixed before labels.

    Only TRAIN includes exact zero/open/closed/half-mixture anchors. VALID/TEST use
    independent role seeds. Call with the union of existing IDs from ALL roles.
    Unexcited baseline coordinates use the explicitly recorded median-positive
    baseline sampling scale, or a supplied positive reference scale.
    """
    if role not in ROLES or not isinstance(count, int) or count < 1 or not isinstance(start_index, int) or start_index < 0:
        raise ValueError("Explicit nonempty role/count/index required")
    lo, hi = map(float, log_amplitude_bounds)
    if not 0 < lo < hi or not math.isfinite(hi):
        raise ValueError("Invalid positive log-amplitude interval")
    baseline = np.maximum(region.baseline_open, region.baseline_i95)
    if reference_scale is None:
        positive = baseline[baseline > 0]
        if not len(positive):
            raise ValueError("All-zero baselines require an explicit positive sampling reference_scale")
        reference = np.where(baseline > 0, baseline, float(np.median(positive)))
        scale_policy = "MAX_OPEN_I95_WITH_MEDIAN_POSITIVE_BASELINE_FOR_UNEXCITED_COORDINATES"
    else:
        reference = _numeric(reference_scale, "reference_scale", shape=(region.m,))
        if np.any(reference <= 0):
            raise ValueError("Sampling reference scales must be strictly positive")
        scale_policy = "EXPLICIT_FROZEN_REFERENCE_SCALE"
    existing = set(existing_input_ids); requests = []; index = start_index
    maximum_candidates = start_index+max(100, count*100)
    while len(requests) < count and index < maximum_candidates:
        key = f"PACNHN_SAMPLE_V1|{region.identity}|{role}|{int(seed)}|{index}"
        derived_seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little")
        rng = np.random.default_rng(derived_seed)
        if role == "TRAIN" and index < 4:
            anchors = [np.zeros(region.m), region.baseline_open, region.baseline_i95,
                       .5*(region.baseline_open+region.baseline_i95)]
            q = anchors[index].copy(); family = ("ZERO", "OPEN", "I95", "HALF_MIXTURE")[index]
        else:
            mixture = rng.uniform()
            q = (mixture*region.baseline_open+(1-mixture)*region.baseline_i95)*math.exp(rng.uniform(math.log(lo), math.log(hi)))
            mode = index % 3
            if mode == 0:
                q = q*rng.lognormal(0., .35, region.m); family = "LOG_AMPLITUDE_DENSE_PERTURBATION"
            elif mode == 1:
                choose = rng.random(region.m) < .15
                choose[int(rng.integers(region.m))] = True
                q[choose] = reference[choose]*np.exp(rng.uniform(math.log(lo), math.log(hi), int(choose.sum())))
                family = "LOG_AMPLITUDE_SPARSE_PERTURBATION"
            else:
                choose = rng.random(region.m) < .15
                q[choose] = 0.; family = "LOG_AMPLITUDE_SPARSE_ZERO"
            # Explore frozen active coordinates even when both final baselines are zero.
            missing = (baseline == 0) & (rng.random(region.m) < .2)
            q[missing] = reference[missing]*np.exp(rng.uniform(math.log(lo), math.log(hi), int(missing.sum())))
        q = np.ascontiguousarray(q, dtype=np.float64); q[q == 0] = 0.0
        input_id = array_hash({"u": q})
        if input_id not in existing:
            sample_id = hashlib.sha256(f"{region.identity}|{role}|{input_id}".encode()).hexdigest()
            requests.append(dict(sample_id=sample_id, input_id=input_id, role=role, sample_index=index,
                family=family, seed=int(seed), derived_seed=derived_seed, graph_identity=region.identity,
                support_identity=region.support_identity, sampling_scale_policy=scale_policy,
                sampling_reference_hash=array_hash({"scale": reference}), log_amplitude_bounds=[lo, hi], u=q))
            existing.add(input_id)
        index += 1
    if len(requests) != count:
        raise RuntimeError("Unique nonempty sampling budget exhausted")
    return requests


def generate_labels(config, region_run, *, requests, guard=None):
    """Execute an explicit finite request list; preserve attempts and blind TEST files.

    config requires phase_id/run_id/region_id/graph_path/graph_sha256/deadline_epoch,
    per_label_seconds/max_iterations/relative_gap_tolerance. Old labels are reused
    only from this same authenticated run and exact graph, role and input identity.
    """
    required = {"phase_id", "run_id", "region_id", "graph_path", "graph_sha256", "deadline_epoch",
                "per_label_seconds", "max_iterations", "relative_gap_tolerance"}
    if required-config.keys() or config.get("phase_id") != "PACNHN_V1" or not config["run_id"] or not config["region_id"]:
        raise ValueError("Incomplete explicit regional execution configuration")
    solver_method = config.get("solver_method", "FW")
    retry_solver_method = config.get("retry_solver_method", solver_method)
    if solver_method not in {"FW", "CFW"} or retry_solver_method not in {"FW", "CFW"}:
        raise ValueError("Explicit local label solver policy must use FW/CFW")
    first_iterations = config["max_iterations"]
    retry_iterations = config.get("retry_max_iterations", first_iterations)
    first_bound_reset = config.get("cfw_reset_on_bound", False)
    retry_bound_reset = config.get("retry_cfw_reset_on_bound", first_bound_reset)
    max_attempts = config.get("max_attempts_per_input")
    first_selection = config.get("iterate_selection", "LAST_CERTIFIED")
    retry_selection = config.get("retry_iterate_selection", first_selection)
    if any(type(v) is not str or v not in {"LAST_CERTIFIED", "BEST_CERTIFIED_RG"} for v in (first_selection, retry_selection)):
        raise ValueError("Explicit label iterate selection must be LAST_CERTIFIED or BEST_CERTIFIED_RG")
    if any(type(value) is not int or value <= 0 for value in (first_iterations, retry_iterations)):
        raise ValueError("Label iteration caps must be positive integers")
    if any(type(value) is not bool for value in (first_bound_reset, retry_bound_reset)):
        raise ValueError("Label bound-reset flags must be explicit booleans")
    if max_attempts is not None and (type(max_attempts) is not int or max_attempts <= 0):
        raise ValueError("Global maximum attempts must be a positive integer")
    policy = dict(first_attempt=solver_method, retry_attempts=retry_solver_method,
        first_max_iterations=first_iterations, retry_max_iterations=retry_iterations,
        first_cfw_reset_on_bound=first_bound_reset, retry_cfw_reset_on_bound=retry_bound_reset,
        first_iterate_selection=first_selection, retry_iterate_selection=retry_selection,
        max_attempts_per_input=max_attempts, attempt_scope="GLOBAL_ALL_PRIOR_ATTEMPT_SLOTS",
        prior_attempts_preserved=True, checkpoint_continuation=False)
    requests = list(requests)
    if not requests:
        raise ValueError("No-op regional label execution is forbidden")
    if len({r["input_id"] for r in requests}) != len(requests):
        raise ValueError("Duplicate inputs in one request list are not additional unique labels")
    region = load_region_graph(config["graph_path"], expected_sha256=config["graph_sha256"])
    root = Path(region_run); root.mkdir(parents=True, exist_ok=True)
    identity = dict(phase_id=config["phase_id"], run_id=config["run_id"], region_id=config["region_id"],
                    graph_identity=region.identity, graph_sha256=region.graph_sha256, support_identity=region.support_identity)
    identity_path = root/"REGION_IDENTITY.json"
    if identity_path.exists():
        if read_json(identity_path) != identity:
            raise ValueError("Existing regional run identity changed")
    else:
        write_json(identity_path, identity)
    index_path = root/"REQUEST_INDEX.json"
    index = read_json(index_path) if index_path.exists() else {}
    rows = []; attempted = 0; reused = 0; attempt_limited = 0; stop = "COMPLETE_REQUEST_LIST"
    # Freeze every requested role/input BEFORE opening any label for this batch.
    for request in requests:
        role = request["role"]; sample_id = request["sample_id"]
        q = _numeric(request["u"], "frozen request", shape=(region.m,)).copy(); q[q == 0] = 0.
        input_id = array_hash({"u": q})
        canonical = hashlib.sha256(f"{region.identity}|{role}|{input_id}".encode()).hexdigest()
        if role not in ROLES or sample_id != canonical or request["input_id"] != input_id:
            raise ValueError("Request role/input/sample identity mismatch")
        if request["graph_identity"] != region.identity or request["support_identity"] != region.support_identity:
            raise ValueError("Request graph/support identity mismatch")
        old = index.get(input_id)
        if old is not None and (old["role"] != role or old["sample_id"] != sample_id):
            raise ValueError("Immutable sample role would change or leak across cohorts")
        folder = root/"requests"/role/sample_id
        if old is None:
            metadata = {k: v for k, v in request.items() if k != "u"}
            recovered = folder.exists()
            if recovered:
                # Recover a fully written request after interruption before the index
                # flush; never replace its bytes or infer a different role.
                prior = read_json(folder/"REQUEST.json")
                if {k: v for k, v in prior.items() if k != "input_file_sha256"} != metadata:
                    raise ValueError("Orphan request has incompatible immutable metadata")
                if sha(folder/"input.npz") != prior["input_file_sha256"]:
                    raise ValueError("Orphan request input bytes changed")
                with np.load(folder/"input.npz", allow_pickle=False) as z:
                    if array_hash({"u": z["u"]}) != input_id:
                        raise ValueError("Orphan request native input identity mismatch")
                metadata = prior
            else:
                folder.mkdir(parents=True, exist_ok=False)
                save_npz(folder/"input.npz", u=q)
                metadata["input_file_sha256"] = sha(folder/"input.npz")
                write_json(folder/"REQUEST.json", metadata)
            index[input_id] = dict(sample_id=sample_id, role=role, input_file=str(folder/"input.npz"),
                request_metadata=str(folder/"REQUEST.json"), input_file_sha256=metadata["input_file_sha256"],
                request_metadata_sha256=sha(folder/"REQUEST.json"), selected_label=None, recovered_completed_request=recovered)
    write_json(index_path, index)
    freeze_dir = root/"request_versions"; freeze_dir.mkdir(exist_ok=True)
    freeze_id = len(list(freeze_dir.glob("v*.json")))+1
    frozen_path = freeze_dir/f"v{freeze_id:06d}.json"
    if frozen_path.exists():
        raise FileExistsError("Request version collision")
    write_json(frozen_path, dict(identity=identity, requests={k: {f: v for f, v in row.items() if f != "selected_label"}
        for k, row in index.items()}, recorded_epoch=time.time()))
    for request in requests:
        entry = index[request["input_id"]]
        if sha(entry["input_file"]) != entry["input_file_sha256"] or sha(entry["request_metadata"]) != entry["request_metadata_sha256"]:
            raise ValueError("Frozen input/role artifact changed")
        if entry["selected_label"] is not None:
            label = entry["selected_label"]
            if sha(label["path"]) != label["sha256"] or sha(label["receipt"]) != label["receipt_sha256"]:
                raise ValueError("Prior admitted label artifact changed")
            rows.append(dict(sample_id=entry["sample_id"], role=entry["role"], status="REUSED_EXACT_SAME_RUN_LABEL", admitted=True))
            reused += 1; continue
        if time.time() >= config["deadline_epoch"]:
            stop = "TIME_LIMIT"; break
        if guard is not None:
            guard()
        cohort = "blind_TEST" if entry["role"] == "TEST" else entry["role"]
        sample_root = root/"labels"/cohort/entry["sample_id"]
        sample_root.mkdir(parents=True, exist_ok=True)
        attempt = len(list(sample_root.glob("attempt_*")))+1
        if max_attempts is not None and attempt > max_attempts:
            rows.append(dict(sample_id=entry["sample_id"], role=entry["role"],
                status="ATTEMPT_LIMIT", admitted=False, existing_attempt_slots=attempt-1,
                solver_policy=policy))
            attempt_limited += 1
            continue
        destination = sample_root/f"attempt_{attempt:06d}"
        destination.mkdir(exist_ok=False)
        started = time.time(); attempted += 1
        resolved_method = solver_method if attempt == 1 else retry_solver_method
        resolved_iterations = first_iterations if attempt == 1 else retry_iterations
        resolved_bound_reset = first_bound_reset if attempt == 1 else retry_bound_reset
        resolved_selection = first_selection if attempt == 1 else retry_selection
        attempt_policy = dict(policy, resolved_method=resolved_method,
            resolved_max_iterations=resolved_iterations, resolved_cfw_reset_on_bound=resolved_bound_reset,
            attempt_number_including_prior_failures=attempt, resolved_iterate_selection=resolved_selection)
        try:
            with np.load(entry["input_file"], allow_pickle=False) as z:
                q = z["u"].copy()
            result = solve_label(region, q, max_seconds=config["per_label_seconds"],
                max_iterations=resolved_iterations, relative_gap_tolerance=config["relative_gap_tolerance"],
                deadline_epoch=config["deadline_epoch"], guard=guard, method=resolved_method,
                cfw_reset_on_bound=resolved_bound_reset, iterate_selection=resolved_selection)
            save_npz(destination/"label.npz", **{k: v for k, v in result.items() if k != "record"})
            record = dict(result["record"], sample_id=entry["sample_id"], role=entry["role"], attempt=attempt,
                solver_policy=attempt_policy,
                input_file_sha256=entry["input_file_sha256"], request_version_sha256=sha(frozen_path),
                label_sha256=sha(destination/"label.npz"))
            write_json(destination/"RESULT.json", record)
            if record["admitted"]:
                entry["selected_label"] = dict(path=str(destination/"label.npz"), sha256=record["label_sha256"],
                    receipt=str(destination/"RESULT.json"), receipt_sha256=sha(destination/"RESULT.json"))
            row = dict(sample_id=entry["sample_id"], role=entry["role"], attempt=attempt,
                status=record["status"], admitted=record["admitted"], quality=record["quality"],
                RG_SPTT=record["certificate"]["RG_SPTT"], solver=record["solver"], solver_method=resolved_method,
                solver_policy=attempt_policy, max_iterations=resolved_iterations,
                cfw_reset_on_bound=resolved_bound_reset,
                complete_label_elapsed_seconds=record["complete_label_elapsed_seconds"], receipt=str(destination/"RESULT.json"))
        except Exception as exc:
            row = dict(sample_id=entry["sample_id"], role=entry["role"], attempt=attempt,
                status="TIME_LIMIT" if isinstance(exc, TimeoutError) else "FAILED", admitted=False,
                error_type=type(exc).__name__, error=str(exc), solver_method=resolved_method,
                solver_policy=attempt_policy, max_iterations=resolved_iterations,
                cfw_reset_on_bound=resolved_bound_reset,
                started_epoch=started, elapsed_seconds=time.time()-started)
            write_json(destination/"FAILED.json", row)
            if isinstance(exc, ArithmeticError) and not isinstance(exc, FloatingPointError):
                stop = "BLOCKED_REGIONAL_INVARIANT"
        rows.append(row)
        with (root/"ATTEMPTS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
        write_json(index_path, index)
        if stop == "BLOCKED_REGIONAL_INVARIANT":
            break
    if stop == "COMPLETE_REQUEST_LIST" and attempt_limited == len(requests):
        stop = "ATTEMPT_LIMIT_NO_EXECUTION"
    summary = dict(identity=identity, status=stop, request_count=len(requests), attempted=attempted, reused=reused,
        attempt_limited=attempt_limited,
        admitted_current_batch=sum(bool(row["admitted"]) for row in rows), unattempted=len(requests)-len(rows),
        total_unique_frozen_inputs=len(index), admitted_by_role={role: sum(row["role"] == role and row["selected_label"] is not None
        for row in index.values()) for role in sorted(ROLES)}, rows=rows,
        request_version=str(frozen_path), request_version_sha256=sha(frozen_path),
        test_storage="labels/blind_TEST; training loader refuses TEST", deadline_epoch=config["deadline_epoch"],
        solver_policy=policy)
    write_json(root/"SUMMARY.json", summary)
    return summary


def load_admitted_dataset(region_run, *, role):
    """TRAIN/VALID only. Final TEST requires a separate selection-frozen evaluator."""
    if role not in {"TRAIN", "VALID"}:
        raise PermissionError("Blind TEST is unavailable through the training dataset loader")
    root = Path(region_run); identity = read_json(root/"REGION_IDENTITY.json")
    index = read_json(root/"REQUEST_INDEX.json"); inputs = []; costs = []; records = []; c0 = None
    for input_id, row in index.items():
        selected = row["selected_label"]
        if row["role"] != role or selected is None:
            continue
        if sha(selected["path"]) != selected["sha256"] or sha(selected["receipt"]) != selected["receipt_sha256"]:
            raise ValueError("Admitted label/receipt bytes changed")
        record = read_json(selected["receipt"])
        if (not record["admitted"] or record["role"] != role or record["input_identity"] != input_id
                or record["graph_identity"] != identity["graph_identity"] or record["support_identity"] != identity["support_identity"]):
            raise ValueError("Admitted label role/input/graph evidence mismatch")
        with np.load(selected["path"], allow_pickle=False) as z:
            q, tau, current_c0 = z["u"].copy(), z["tau"].copy(), z["c0"].copy()
        if array_hash({"u": q}) != input_id or not np.isfinite(tau).all() or np.any(tau < 0):
            raise ValueError("Invalid current native label")
        if c0 is not None and not np.array_equal(c0, current_c0):
            raise ValueError("Free-flow baseline changed within one regional dataset")
        c0 = current_c0; inputs.append(q); costs.append(tau); records.append(record)
    if not inputs:
        raise ValueError("No admitted labels in requested role")
    u, tau = np.stack(inputs), np.stack(costs)
    return dict(u=u, tau=tau, c0=c0, records=records, role=role, identity=identity,
        dataset_identity=array_hash({"u": u, "tau": tau, "c0": c0}),
        unique_rows=len(u), train_cost_mean=float(tau.mean()) if role == "TRAIN" else None)
