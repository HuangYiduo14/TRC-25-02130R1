"""One explicit bounded task per supervised child; no default scientific task."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np
import pandas as pd

from .common import Context, read_json, write_json, sha, code_identity, save_npz, array_hash


def main():
    p = argparse.ArgumentParser()
    for name in ["config", "run-id", "task", "job-id"]:
        p.add_argument("--"+name, required=True)
    p.add_argument("--job-start", type=float, required=True)
    p.add_argument("--job-deadline", type=float, required=True)
    p.add_argument("--suite")
    p.add_argument("--scenario")
    p.add_argument("--micro", action="store_true")
    args = p.parse_args()
    steps = dict(unit_tests="NATIVE_TESTS", build="BUILD", native_tests="NATIVE_TESTS", pilot="PILOT", timing="TIMING", reconcile_self_od="BUILD")
    if args.task not in steps:
        raise ValueError("Unknown explicit task")
    ctx = Context(args.config, steps[args.task], args.run_id)
    from .native import Sources, build_coverage, reopen_demand, load_registry
    from .objective import Composite, UnsafeCost
    from .oracle import Demand
    from .solver import solve
    try:
        if args.task == "unit_tests":
            import pytest
            code = pytest.main([str(ctx.repo/"partC_rev1_src/phase_n8/tests"), "-q", "-p", "no:cacheprovider",
                "--basetemp="+str(ctx.root/"validation"/args.job_id)])
            ctx.finish(args.job_id, dict(status="PASS" if code == 0 else "FAIL", pytest_exit_code=int(code)))
            return int(code)
        sources = Sources(ctx)
        if args.task == "build":
            original, uids, reachable, _ = sources.original_od()
            result = build_coverage(ctx, sources, args.suite, original, uids, reachable)
            ctx.finish(args.job_id, dict(status="PASS", coverage=result))
        elif args.task == "reconcile_self_od":
            from .validation import reconcile_self_od
            ctx.finish(args.job_id, reconcile_self_od(ctx, sources))
        elif args.task == "native_tests":
            from .validation import native_tests
            result = native_tests(ctx, sources)
            ctx.finish(args.job_id, result)
        elif args.task == "timing":
            from .timing_audit import audit
            ctx.finish(args.job_id, audit(ctx, sources))
        elif args.task == "pilot":
            matrix = read_json(ctx.root/"SCENARIO_DEFINITION_FREEZE.json")
            matches = [r for r in matrix["scenarios"] if r["scenario_id"] == args.scenario]
            if len(matches) != 1:
                raise ValueError("Scenario not in the frozen full matrix")
            sc = matches[0]
            s = sources.suite(sc["suite"])
            original, uids, _, _ = sources.original_od()
            od, balance, coverage = reopen_demand(ctx, s, original)
            entries, registry = load_registry(sources, s, sc["registry_id"])
            if registry["identity"] != sc["registry_identity"]:
                raise ValueError("Scenario registry differs from pre-outcome freeze")
            if args.micro:
                sample_path = sources.base/"phase_05/fidelity_0906/run_001/sampling/frozen_od_sample.csv"
                sources.bind(sample_path, "BOUND_HISTORICAL_456_OD_SAMPLE", "7dcb9d2fad9d159fd299c66326ca5e44a4b7eef8fcff9d3a21e8955f801f28fd")
                sample = pd.read_csv(sample_path, float_precision="round_trip").sort_values("stable_od_key", kind="stable").head(32)
                maps = __import__("partC_rev1_src.phase_n8.common", fromlist=["read_npz"]).read_npz(ctx.root/"graphs"/s["suite"]/"mapping.npz")
                rows = sample.accepted_od_row.to_numpy(np.int64)
                origin_groups = np.searchsorted(original["origin_offsets"][1:], rows, side="right")
                if (not np.array_equal(original["origin_group_node_idx"][origin_groups], sample.origin_idx.to_numpy(np.int64))
                    or not np.array_equal(original["destination_node_idx"][rows], sample.destination_idx.to_numpy(np.int64))
                    or not np.array_equal(original["demand"][rows], sample.demand.to_numpy(float))):
                    raise ValueError("Frozen sample differs from exact accepted original OD arrays")
                groups, count = np.unique(sample.origin_idx.to_numpy(np.int64), return_counts=True)
                # Freeze first-32 membership; sort only for origin loading.
                sample = sample.sort_values(["origin_idx", "destination_idx"], kind="stable")
                sampled_dest = maps["destination_map"][sample.destination_idx.to_numpy(np.int64)].copy()
                is_self = sample.origin_idx.to_numpy(np.int64) == sample.destination_idx.to_numpy(np.int64)
                sampled_dest[is_self] = maps["origin_map"][sample.origin_idx.to_numpy(np.int64)[is_self]]
                serve = od.serve[sample.accepted_od_row.to_numpy(np.int64)]
                od = Demand(maps["origin_map"][groups], np.r_[0, np.cumsum(count)], sampled_dest,
                            sample.demand.to_numpy(float), serve)
                balance = od.balance(len(s["graph"].node_ids))
                sample.to_csv(ctx.root/"tables/micro_32_frozen_rows.csv", index=False)
            objective = Composite(s, entries, registry, guard=lambda: ctx.guard(deadline=args.job_deadline))
            ch, code_members = code_identity(ctx.repo)
            identity = dict(graph=s["graph"].identity, demand=od.identity, objective=objective.identity,
                            support=s["support_identity"], registry=registry["identity"], code=ch,
                            scenario=args.scenario, solver="FW", tolerance=.001)
            clocks = dict(R2_start_epoch=ctx.c["start_epoch"], R2_deadline_epoch=ctx.c["deadline_epoch"],
                          run_start_epoch=args.job_start, run_deadline_epoch=args.job_deadline)
            root = ctx.root/"runs"/args.job_id
            root.mkdir(exist_ok=False)
            write_json(root/"PRE_RUN_FREEZE.json", dict(identity=identity, clocks=clocks,
                 coverage=coverage, registry=registry, code_members=code_members, job_id=args.job_id,
                 task_type="MICRO_PILOT" if args.micro else "FULL_REPRESENTED_DEMAND_PILOT",
                 micro_results_are_not_national_forecast=args.micro))
            # Complete coupled cost/gradient/HVP timing at a native fixed reference.
            # Safety is reported per queried case; no source weights are changed.
            t = time.perf_counter()
            try:
                fixed = objective.evaluate(s["fixed_x45"], np.ones(len(s["graph"].tail)))
                write_json(root/"FIXED_X45_INFERENCE.json", dict(elapsed_seconds=time.perf_counter()-t,
                           objective=fixed[0], diagnostic=fixed[3], negative_coordinates=0,
                           source_state_role="MAPPED_CURRENT_X45_NOT_A_FEASIBLE_WARM_START"))
                load_seconds=time.time()-args.job_start
                solver_started=time.time()
                result = solve(s["graph"], od, objective, balance, root, identity=identity, clocks=clocks,
                               ctx=ctx, tol=.001, micro_iteration_limit=20 if args.micro else None)
                result['solver_elapsed_seconds']=time.time()-solver_started
                result['pre_run_input_load_seconds']=load_seconds
            except UnsafeCost as exc:
                save_npz(root/'offending_query.npz',dict(flow=exc.flow))
                exc.diagnostic.update(phase='FIXED_X45_NATIVE_ADMISSION',offending_query_sha256=sha(root/'offending_query.npz'))
                write_json(root/'SAFETY_STOP.json',exc.diagnostic)
                result=dict(stop_reason=exc.reason,accepted_iterations=0,elapsed_seconds=time.time()-args.job_start,
                  state_hash=None,final_certificate=None,last_valid_certificate=None,final_certificate_fresh=False,
                  final_certificate_status='NO_VALID_ACCEPTED_SOLVER_STATE',identity=identity,clocks=clocks,
                  initialization='NATIVE_FIXED_X45_ADMISSION_STOP_BEFORE_INITIAL_AON',represented_demand=od.total,
                  nominal_tolerance=.001,solver='FW',solver_elapsed_seconds=0.,
                  pre_run_input_load_seconds=time.time()-args.job_start)
            write_json(root/"INFERENCE_TIMINGS.json", objective.timings)
            result.update(scenario=args.scenario, suite=sc["suite"], job_id=args.job_id,
                          coverage=coverage, r3_official_run=False)
            write_json(root/"RESULT.json", result)
            ctx.finish(args.job_id, dict(status="EXECUTED", run=result))
    except Exception as exc:
        trace = traceback.format_exc()
        print(trace, flush=True)
        write_json(ctx.root/f"receipts/{args.job_id}_FAILURE.json", dict(type=type(exc).__name__,
                    error=str(exc), traceback=trace, original_clock_preserved=True))
        write_json(ctx.root/f"receipts/{args.job_id}_SOURCE_BINDINGS.json", list(ctx.bindings.values()))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
