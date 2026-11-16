"""Two bounded workers, serialized publication, baseline-before-enrichment gates."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd

from .common import read_json, write_json, write_frame, event, Monitor, digest, can_admit, elapsed, tree_bytes
from .worker import baseline_region
from .registry import region_dir
from .validation import baseline_audit


def representatives(regions, count=8):
    active=regions.loc[regions.m_active.gt(0)].copy().sort_values(["m_active","region_id"])
    chosen=[]
    def add(rid):
        if rid not in chosen: chosen.append(rid)
    for quantile in [0.,.5,.95,1.]: add(active.iloc[min(len(active)-1,int(quantile*(len(active)-1)))].region_id)
    for col in ["physical_arcs","n_full","low_exposure"]:
        add(active.sort_values([col,"region_id"],ascending=[False,True]).iloc[0].region_id)
    # Empty active support has a separate skip check. Fill all positive choices deterministically.
    for row in active.sort_values("region_id").itertuples():
        if len(chosen)>=count: break
        add(row.region_id)
    return chosen[:count]


def run_jobs(c, jobs, stage, fn, *, max_workers=2):
    from .resume_policy import enabled
    if enabled(c):
        from .resume_campaign import run_scoped_jobs
        return run_scoped_jobs(c,jobs,stage,fn,max_workers=max_workers)
    if max_workers!=2: raise ValueError("Worker increase requires measured authorization evidence")
    out=Path(c["output_root"]); results=[]; queue=iter(jobs); exhausted=False
    with Monitor(c,stage) as monitor, ThreadPoolExecutor(max_workers=2,thread_name_prefix="regional") as executor:
        running={}
        while running or not exhausted:
            monitor.check()
            while len(running)<2 and not exhausted:
                try: job=next(queue)
                except StopIteration: exhausted=True; break
                if not can_admit(c,600):
                    exhausted=True
                    event(out/"control/deferrals.jsonl",dict(stage=stage,job=job,reason="ORIGINAL_DEADLINE_ADMISSION"))
                    break
                running[executor.submit(fn,job)]=job
            if not running: break
            done,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                job=running.pop(future)
                try: result=future.result()
                except BaseException as exc:
                    record=dict(stage=stage,job=job,error=repr(exc),traceback=traceback.format_exc())
                    event(out/"control/integrity_failures.jsonl",record)
                    write_json(out/"control/STOP_SCIENCE.json",record,new=False)
                    raise
                results.append(result)
                event(out/"control/dispatch_history.jsonl",dict(stage=stage,job=job,result=result))
                print(stage,len(results),"/",len(jobs),job,result.get("status"),flush=True)
    return results


def startup(c):
    out=Path(c["output_root"])
    read_json(out/"samples/BASELINE_PLAN_FROZEN.json")
    regions=pd.read_parquet(out/"registry/regions.parquet")
    chosen=representatives(regions)
    inactive=regions.loc[regions.m_active.eq(0)].sort_values("region_id").iloc[0].region_id
    selection=dict(region_ids=chosen,inactive_region_id=inactive,selection="LOW_AND_STRUCTURE_ONLY",roles=[0,1,31,32,33,34,35,36,37,38,39,48],workers=2,
        native_threads=1,publication="SINGLE_PROCESS_SERIALIZED_ATOMIC_WRITER",original_start=c["task_started_unix"])
    write_json(out/"validation/P2_selection.json",selection)
    empty=baseline_region(c,inactive)
    if empty["optimizer_calls"]!=0: raise ValueError("Inactive optimizer used")
    started=time.perf_counter()
    rows=run_jobs(c,chosen,"P2_STARTUP",lambda rid:baseline_region(c,rid,roles=selection["roles"]))
    # Mandatory hard regions remain in the same campaign; retry only missing planned startup inputs.
    retry=[r["region_id"] for r in rows if r["deferred"] or r["hierarchy"]["status"]!="PASS"]
    if retry:
        again=run_jobs(c,retry,"P2_STARTUP_COLD_RETRY",lambda rid:baseline_region(c,rid,roles=selection["roles"],retry=True))
        rows=[r for r in rows if r["region_id"] not in retry]+again
    from .storage import Store
    from .common import read_npz
    reopened=0
    for rid in chosen:
        rd=region_dir(c,rid); d=read_json(rd/"descriptor.json"); s=Store(rd,d,read_npz(rd/"local_graph.npz"))
        plan=pd.read_parquet(rd/"baseline_plan.parquet")
        required=set(plan.loc[plan.role_index.isin(selection["roles"]),"input_hash"])
        if not required.issubset(s.hashes()): raise ValueError("Startup required labels remain quarantined")
        for ih in sorted(required): s.get(ih,recertify=True); reopened+=1
        if read_json(rd/"baseline_status.json")["hierarchy"]["status"]!="PASS": raise ValueError("Startup hierarchy incomplete")
    seconds=time.perf_counter()-started
    result=dict(status="PASS",regions=len(chosen),reopened_unique_labels=reopened,inactive_optimizer_calls=0,
        wall_seconds=seconds,observed_labels_per_second=reopened/max(seconds,1e-9),
        conservative_baseline_seconds=2*75276/max(reopened/max(seconds,1e-9),1e-9),
        artifacts_reused_by_baseline=True,workers=2,added_output_bytes=tree_bytes(out),
        all_later_steps_remain_current_campaign=True)
    write_json(out/"validation/P2_PASS.json",result)
    return result


def baseline_campaign(c):
    from .resume_policy import enabled
    if enabled(c):
        from .resume_campaign import remaining_baseline
        return remaining_baseline(c)
    out=Path(c["output_root"]); read_json(out/"validation/P2_PASS.json")
    regions=pd.read_parquet(out/"registry/regions.parquet")
    active=regions.loc[regions.m_active.gt(0)].copy()
    active["work_estimate"]=active.m_active*active.physical_arcs
    jobs=active.sort_values(["work_estimate","region_id"]).region_id.tolist()
    rows=run_jobs(c,jobs,"P3_BASELINE",lambda rid:baseline_region(c,rid))
    retry=[r["region_id"] for r in rows if r["status"]!="BASELINE_LABELS_COMPLETE"]
    if retry:
        again=run_jobs(c,retry,"P3_BASELINE_COLD_RETRY",lambda rid:baseline_region(c,rid,retry=True))
        rows=[r for r in rows if r["region_id"] not in retry]+again
    complete=[r["region_id"] for r in rows if r["status"]=="BASELINE_LABELS_COMPLETE"]
    audits=run_jobs(c,complete,"P4_BASELINE_REOPEN",lambda rid:baseline_audit(c,rid))
    empty=[baseline_audit(c,rid) for rid in regions.loc[regions.m_active.eq(0),"region_id"]]
    result=dict(status="BASELINE_LABEL_AUDITS_PASS" if len(complete)==len(active) and all(r["status"]=="PASS" for r in audits+empty) else "BASELINE_PARTIAL",
        active_regions=len(active),complete_regions=len(complete),inactive_regions=len(empty),
        missing_regions=sorted(set(jobs)-set(complete)),baseline_eligible_for_enrichment=False,
        pending_gate="PRECISION_AND_FULL_RUNTIME_LOADER_CHECKS")
    write_frame(out/"validation/baseline_region_audits.parquet",pd.DataFrame(audits+empty))
    write_json(out/"validation/BASELINE_LABEL_AUDITS.json",result)
    return result
