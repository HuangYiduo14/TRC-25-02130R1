"""Read-only reconciliation of accepted physical timing boundaries."""
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_json, write_json, sha, atomic_bytes


def audit(ctx, sources):
    p4 = sources.p4
    high = Path(p4["source_run_root"])
    low = Path(p4["output_root"])
    final = high/"finalization/user_directed_iteration_0200_v1"
    ref = sources.authority/"references/physical_timing"
    paths = dict(update=final/"tables/authoritative_update_history.csv",
        history=low/"comparison/source_refreshed_state_history.csv",
        chunks=final/"tables/completed_chunk_registry.csv",
        metrics=final/"reports/final_metrics.json",
        registry=low/"benchmark_registry.json",
        recovery=low/"reports/RECOVERY_AUDIT.json",
        refresh=low/"resources/independent_refresh_resource.json",
        resource=ctx.repo/"partC_doc_icnn/handoffs/phase_04/step_physical_reference_production_24h/phase4_prod_run001_iter200_waiting_review_seal_v2_20260902/summary_data/validation/resource_usage.json",
        config=high/"config/phase4_physical_reference_production_execution_config.json",
        orchestration=high/"status/orchestration_state.json",
        source_code=ctx.repo/"partC_rev1_src/phase4/production_finalize.py")
    copies = dict(update="high_authoritative_update_history.csv", history="source_refreshed_state_history.csv",
                  resource="high_resource_usage.json", refresh="independent_refresh_resource.json",
                  registry="benchmark_registry.json", recovery="RECOVERY_AUDIT.json")
    for name, path in paths.items():
        sources.bind(path, "PHYSICAL_TIMING_"+name.upper(), sha(ref/copies[name]) if name in copies else None)
    h = pd.read_csv(paths["update"], float_precision="round_trip")
    state = pd.read_csv(paths["history"], float_precision="round_trip")
    chunks = pd.read_csv(paths["chunks"])
    resource = read_json(paths["resource"])
    refresh = read_json(paths["refresh"])
    config = read_json(paths["config"])
    metrics = read_json(paths["metrics"])
    if not np.array_equal(h.iteration.to_numpy(int), np.arange(1,201)):
        raise ValueError("Accepted authoritative update sequence differs")
    if not h.authoritative_after_interruption_filter.all():
        raise ValueError("Unfiltered interrupted history used")
    t33=float(h.loc[h.iteration<=33,'iteration_wall_seconds'].sum())
    t45=float(h.loc[h.iteration<=45,'iteration_wall_seconds'].sum())
    t200=float(h.iteration_wall_seconds.sum())
    for actual,expected in [(t33,2530.9787666001357),(t45,3353.9471747006755),(t200,14614.275030500488)]:
        if abs(actual-expected)>1e-8:
            raise ValueError("Accepted physical timing arithmetic does not reconcile")
    if abs(t200-resource['solver_update_seconds'])>1e-8 or refresh['solver_updates_performed']!=0:
        raise ValueError("Timing boundaries inconsistent")
    rows=[]
    for label,it,t,rg in [('physical_mu030_low_rg1e3_CURRENT_x45',45,t45,.00069493318901541054),
                          ('physical_mu030_high_iter0200_CURRENT',200,t200,.00044683233333997485),
                          ('x33_scalar_history_diagnostic_only',33,t33,.0009952932645325787)]:
        sh=state.loc[state.state_iteration.eq(it)].iloc[0]
        if abs(float(sh.wall_seconds_at_refresh)-t)>1e-8:
            raise ValueError("Refreshed history cumulative update clock differs")
        rows.append(dict(benchmark_id=label,checkpoint_iteration=it,RG_SPTT=rg,
            update_wall_seconds=t,standalone_assignment_end_to_end_seconds=None,
            broader_campaign_elapsed_seconds=resource['campaign_elapsed_seconds_to_independent_refresh'] if it==200 else None,
            later_refresh_only_seconds=refresh['wall_seconds'] if it==45 else None,
            source_path=str(paths['update']),source_sha256=sha(paths['update']),
            source_field='sum(iteration_wall_seconds) through update '+str(it),
            worker_count=int(config['worker_count']),origin_count=p4['input_versions']['origin_count'],
            historical_cpu_model=None,historical_physical_memory_bytes=None,
            attempt_peak_rss_bytes=resource['attempt_peak_rss_bytes'],
            exact_standalone_clock_known=False,matched_hardware_speedup_supported=False,
            confidence='EXACT_ACCEPTED_HISTORY_AND_RECEIPT',
            checkpoint_available=it!=33,role='SCALAR_DIAGNOSTIC_ONLY' if it==33 else 'ACCEPTED_CURRENT_REFERENCE'))
    frame=pd.DataFrame(rows)
    frame.to_csv(ctx.root/'tables/physical_benchmark_timing.csv',index=False)
    # Preserve selected compact native evidence, including the chunk registry
    # schema/count without duplicating its full serialized checkpoint contents.
    evidence=dict(paths={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},
        authoritative_updates=len(h),completed_chunk_registry_rows=len(chunks),
        chunk_registry_columns=chunks.columns.tolist(),update_history_workers=config['worker_count'],
        exact_x45_standalone_end_to_end_unknown=True,source_hardware_identity_not_in_targeted_metadata=True,
        historical_timings_are_contextual=True,physical_solver_updates_executed_in_R2=0,
        source_clock_rule='Cumulative iteration_wall_seconds; includes work inside each recorded iteration, including checkpoint/refresh. Not a pure kernel clock.',
        interruptions='Authoritative history filters interrupted attempts; inter-chunk waits/restarts and setup/finalization are not guaranteed included.',
        later_x45_refresh_is_separate=True,high_gap=float(metrics['relative_gap_sptt']),
        lower_first_crossing_33_not_selected_without_checkpoint=True)
    write_json(ctx.root/'validation/PHYSICAL_TIMING_EVIDENCE.json',evidence)
    report=f'''# Physical benchmark timing audit

The accepted native physical history and receipts record the following wall clocks. No physical solve or new physical AON was executed in N8R2.

| Reference | Cumulative update wall time | Broader elapsed | Later refresh only |
|---|---:|---:|---:|
| CURRENT x45, RG 0.00069493318901541054 | {t45:.12f} s ({t45/60:.3f} min) | Unknown standalone end-to-end | {refresh['wall_seconds']:.12f} s |
| CURRENT x200, RG 0.00044683233333997485 | {t200:.12f} s ({t200/3600:.3f} h) | {resource['campaign_elapsed_seconds_to_independent_refresh']:.12f} s ({resource['campaign_elapsed_seconds_to_independent_refresh']/3600:.3f} h) to independent refresh | Separate record |
| x33 scalar-history crossing | {t33:.12f} s | Unknown | No complete qualifying checkpoint survives |

## Meaning and comparison limits

The update clock sums `iteration_wall_seconds` from the interruption-filtered authoritative 200-update history. The source finalizer builds `wall_seconds_at_refresh` from that cumulative clock. Work inside the iterations includes recorded checkpoint and refresh work. Initialization, pauses, inter-chunk restart overhead, and later finalization are not guaranteed included. x45 was recovered from its saved CURRENT state; it was not a newly timed standalone solve. Adding its later 138-second refresh does not reconstruct the missing standalone end-to-end boundary.

The high receipt separately records the 17,724-second broader elapsed clock. Native configuration records {config['worker_count']} workers and {p4['input_versions']['origin_count']:,} origins. Its attempt peak RSS was {resource['attempt_peak_rss_bytes']:,} bytes; this is an attempt-process record, not an inferred process-tree peak. Historical CPU model and installed-memory identity were not established by the targeted metadata audit. The present N8 runtime uses one numeric worker with nested BLAS threads set to one. These physical timings are contextual; a matched-hardware speedup is not established.

The primary contextual comparison is physical x45 versus a surrogate run freshly reaching RG <= 0.001. x200 did not reach the original 0.0001 goal. The 138-second measurement is certificate refresh only and must never be used as a complete physical assignment denominator.

## Native evidence

- Update history: `{paths['update']}`; SHA-256 `{sha(paths['update'])}`.
- Refreshed state history: `{paths['history']}`; SHA-256 `{sha(paths['history'])}`.
- Broader elapsed receipt: `{paths['resource']}`; SHA-256 `{sha(paths['resource'])}`.
- x45 refresh receipt: `{paths['refresh']}`; SHA-256 `{sha(paths['refresh'])}`.
- All targeted native paths, hashes, fields, chunk count, and boundaries are in `validation/PHYSICAL_TIMING_EVIDENCE.json` and `tables/physical_benchmark_timing.csv`.
'''
    atomic_bytes(ctx.root/'PHYSICAL_TIMING_AUDIT.md',report.encode())
    return dict(status='PASS_READ_ONLY_TIMING_RECONCILIATION',x45_update_seconds=t45,
                x200_update_seconds=t200,x200_broader_elapsed_seconds=resource['campaign_elapsed_seconds_to_independent_refresh'],
                x45_refresh_only_seconds=refresh['wall_seconds'],physical_solver_updates=0)
