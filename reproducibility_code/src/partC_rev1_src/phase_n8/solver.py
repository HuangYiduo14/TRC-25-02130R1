"""Feasible ordinary FW with consistent composite search and signed certificates."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np

from .common import array_hash, read_json, write_json, read_npz, save_npz, sha, ResourceLimit
from .objective import UnsafeCost
from .oracle import aon, conservation


def certificate(flow, cost, linear, objective, state_identity):
    tstt = float(flow@cost)
    sptt = float(linear["sptt"])
    raw = tstt-sptt
    allowance = 1e-10*max(1., abs(tstt), abs(sptt))
    if abs(linear["loaded_sptt"]-sptt) > allowance or raw < -allowance:
        raise ArithmeticError("INVALID_ORACLE: materially negative signed gap or inconsistent SPTT")
    rg = raw/sptt if sptt > 0 else None
    return dict(state_hash=state_identity, objective=objective, TSTT=tstt, SPTT=sptt,
                SPTT_loaded=float(linear["loaded_sptt"]), raw_signed_gap=raw,
                RG_SPTT=rg, arithmetic_allowance=allowance,
                roundoff_negative=raw < 0, fresh=True,
                zero_SPTT_absolute_criterion=allowance if sptt == 0 else None)


def meets_tolerance(cert, tol):
    return (cert["RG_SPTT"] <= tol if cert["RG_SPTT"] is not None
            else abs(cert["raw_signed_gap"]) <= cert["zero_SPTT_absolute_criterion"])


def line_search(objective, flow, sight, value, cost, guard=None):
    direction = sight-flow
    derivative0 = float(cost@direction)
    if derivative0 >= 0 or np.count_nonzero(direction) == 0:
        return 0., value, cost, dict(calls=0, derivative0=derivative0, derivative1=None)
    if guard:
        guard()
    v1, c1, _, _ = objective.evaluate(sight)
    derivative1 = float(c1@direction)
    if derivative1 <= 0:
        return 1., v1, c1, dict(calls=1, derivative0=derivative0, derivative1=derivative1)
    lo, hi = 0., 1.
    calls = 1
    for _ in range(64):
        if guard:
            guard()
        alpha = .5*(lo+hi)
        trial = (1-alpha)*flow+alpha*sight
        trial_value, trial_cost, _, _ = objective.evaluate(trial)
        calls += 1
        derivative = float(trial_cost@direction)
        if derivative > 0:
            hi = alpha
        else:
            lo = alpha
        if hi-lo <= 1e-12:
            break
    if trial_value > value+1e-10*max(1., abs(value)):
        raise ArithmeticError("Composite line search materially increased the fixed objective")
    return alpha, trial_value, trial_cost, dict(calls=calls, derivative0=derivative0,
                                                derivative1=derivative1, final_derivative=derivative,
                                                alpha_bracket_width=hi-lo)


def save_checkpoint(root, flow, iteration, identity, clocks, cert, *, suffix="certified"):
    p = Path(root)/"checkpoints"/(f"state_{iteration:07d}_"+suffix)
    p.mkdir(parents=True, exist_ok=False)
    state_hash = array_hash({"flow": flow})
    if cert is not None and cert["state_hash"] != state_hash:
        raise ValueError("Stale certificate cannot be attached to a different state")
    save_npz(p/"flow.npz", dict(flow=flow))
    meta = dict(identity=identity, clocks=clocks, accepted_iteration=iteration,
                state_hash=state_hash, flow_file_sha256=sha(p/"flow.npz"), certificate=cert,
                solver="ORDINARY_FW", original_clock_preserved=True)
    write_json(p/"STATE.json", meta)
    write_json(Path(root)/"LATEST_CHECKPOINT.json", dict(path=str(p), metadata_sha256=sha(p/"STATE.json")))
    return str(p)


def reopen_checkpoint(root, identity, clocks):
    pointer = read_json(Path(root)/"LATEST_CHECKPOINT.json")
    p = Path(pointer["path"])
    if sha(p/"STATE.json") != pointer["metadata_sha256"]:
        raise ValueError("Checkpoint metadata authentication failed")
    meta = read_json(p/"STATE.json")
    if meta["identity"] != identity or meta["clocks"] != clocks:
        raise ValueError("Checkpoint source identity or original clock changed")
    if sha(p/"flow.npz") != meta["flow_file_sha256"]:
        raise ValueError("Checkpoint flow authentication failed")
    flow = read_npz(p/"flow.npz")["flow"]
    if array_hash({"flow": flow}) != meta["state_hash"]:
        raise ValueError("Checkpoint state identity mismatch")
    cert = meta["certificate"]
    if cert is not None and cert["state_hash"] != meta["state_hash"]:
        raise ValueError("Checkpoint contains a stale certificate")
    return flow, meta


def solve(graph, od, objective, balance, root, *, identity, clocks, ctx,
          tol=.001, micro_iteration_limit=None, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    deadline = clocks["run_deadline_epoch"]
    start = clocks["run_start_epoch"]
    history = []
    stages = []
    latest_cert = None
    flow = None
    iteration = 0
    last_valid_path = None
    stop = "CODE_BLOCKED"
    last_refresh = 0.
    checkpoint_seconds = 0.
    last_search = 0.

    def guard():
        ctx.guard(deadline=deadline)

    try:
        if od.total <= 0:
            raise ValueError("Zero represented demand is not a successful scientific run")
        if resume:
            flow, meta = reopen_checkpoint(root, identity, clocks)
            iteration = meta["accepted_iteration"]
            latest_cert = meta["certificate"]
        else:
            value, cost, _, _ = objective.evaluate(np.zeros(len(graph.tail)))
            t = time.perf_counter()
            linear = aon(graph, od, cost, guard=guard)
            stages.append(dict(stage="FEASIBLE_AON_INITIALIZATION", seconds=time.perf_counter()-t))
            flow = linear["flow"]
            conservation(graph, flow, balance, od.total)
            if abs(linear["reachable_demand"]-od.total) > 1e-10*max(1., od.total):
                raise ArithmeticError("INVALID_ORACLE: frozen represented demand is not all reachable")
        while True:
            guard()
            t = time.perf_counter()
            value, cost, _, diagnostics = objective.evaluate(flow)
            eval_seconds = time.perf_counter()-t
            t = time.perf_counter()
            linear = aon(graph, od, cost, guard=guard)
            last_refresh = time.perf_counter()-t
            if abs(linear["reachable_demand"]-od.total) > 1e-10*max(1., od.total):
                raise ArithmeticError("INVALID_ORACLE: represented coverage changed")
            state_hash = array_hash({"flow": flow})
            latest_cert = certificate(flow, cost, linear, value, state_hash)
            residual = conservation(graph, flow, balance, od.total)
            row = dict(iteration=iteration, elapsed_seconds=time.time()-start, **latest_cert,
                       **residual, **diagnostics, value_gradient_seconds=eval_seconds,
                       aon_seconds=last_refresh, previous_line_search_seconds=last_search)
            history.append(row)
            t = time.perf_counter()
            last_valid_path = save_checkpoint(root, flow, iteration, identity, clocks, latest_cert,
                                               suffix="certified" if not resume else "resume_certified")
            checkpoint_seconds = time.perf_counter()-t
            resume = False
            row["checkpoint_seconds"] = checkpoint_seconds
            write_json(root/"HISTORY.json", history)
            write_json(root/"PROGRESS.json", dict(iteration=iteration, RG_SPTT=latest_cert["RG_SPTT"],
                elapsed_seconds=time.time()-start, state_hash=state_hash, deadline_epoch=deadline))
            print(f"FW {root.name} iter={iteration} RG={latest_cert['RG_SPTT']} elapsed={time.time()-start:.2f}s", flush=True)
            if meets_tolerance(latest_cert, tol):
                stop = "CONVERGED" if np.all(od.serve) else "CONVERGED_REPRESENTED_DEMAND_ONLY"
                break
            if micro_iteration_limit is not None and iteration >= micro_iteration_limit:
                stop = "PILOT_ITERATION_BOUND"
                break
            reserve = max(120., 2*(last_refresh+checkpoint_seconds))
            forecast = last_refresh+max(last_search, 2*eval_seconds)+checkpoint_seconds
            if time.time()+reserve+forecast >= deadline:
                stop = "TIME_LIMIT"
                break
            direction = linear["flow"]-flow
            if np.count_nonzero(direction) == 0:
                stop = "STAGNATION"
                break
            t = time.perf_counter()
            try:
                alpha, new_value, _, search = line_search(objective, flow, linear["flow"], value, cost, guard)
            except UnsafeCost as exc:
                # Retain the complete violating trial, even though it is not accepted.
                exc.diagnostic.update(accepted_state_hash=state_hash, phase="LINE_SEARCH")
                raise
            last_search = time.perf_counter()-t
            row.update(alpha=float(alpha), line_search=search, line_search_seconds=last_search)
            if alpha == 0:
                stop = "STAGNATION"
                break
            trial = (1-alpha)*flow+alpha*linear["flow"]
            conservation(graph, trial, balance, od.total)
            if new_value > value+1e-10*max(1., abs(value)):
                raise ArithmeticError("Ordinary FW objective increase")
            flow = trial
            iteration += 1
            # Persist an accepted yet uncertified state separately. The previous
            # certificate keeps its own state identity; it never certifies this one.
            save_checkpoint(root, flow, iteration, identity, clocks, None, suffix="accepted")
            write_json(root/"HISTORY.json", history)
    except UnsafeCost as exc:
        stop = exc.reason
        if exc.flow is not None:
            save_npz(root/"offending_query.npz", {"flow": exc.flow})
            exc.diagnostic["offending_query_sha256"] = sha(root/"offending_query.npz")
        write_json(root/"SAFETY_STOP.json", exc.diagnostic)
        if flow is not None:
            save_npz(root/"state_at_stop.npz", {"flow": flow})
    except TimeoutError as exc:
        stop = "TIME_LIMIT"
        write_json(root/"DEADLINE_STOP.json", dict(error=str(exc)))
    except ResourceLimit as exc:
        stop = "RESOURCE_LIMIT"
        write_json(root/"RESOURCE_STOP.json", dict(error=str(exc)))
    state_hash = None if flow is None else array_hash({"flow": flow})
    fresh = latest_cert is not None and latest_cert["state_hash"] == state_hash
    final = dict(stop_reason=stop, accepted_iterations=iteration, elapsed_seconds=time.time()-start,
                 state_hash=state_hash, final_certificate=latest_cert if fresh else None,
                 last_valid_certificate=latest_cert, final_certificate_fresh=fresh,
                 final_certificate_status="FRESH_UNCHANGED_STATE" if fresh else "FINAL_CERTIFICATE_MISSING",
                 last_certified_checkpoint=last_valid_path, represented_demand=od.total,
                 identity=identity, clocks=clocks, initialization="AON_AT_FIXED_SCENARIO_ZERO_INPUT_COSTS",
                 nominal_tolerance=tol, solver="FW", micro_iteration_limit=micro_iteration_limit,
                 history_rows=len(history), CFW="NOT_IMPLEMENTED_OPTIONAL",
                 no_cost_clipping=True, no_flow_clipping=True)
    if flow is not None:
        save_npz(root/"final_flow.npz", {"flow": flow})
        final["final_flow_sha256"] = sha(root/"final_flow.npz")
    write_json(root/"HISTORY.json", history)
    write_json(root/"STAGES.json", stages)
    write_json(root/"RESULT.json", final)
    return final
