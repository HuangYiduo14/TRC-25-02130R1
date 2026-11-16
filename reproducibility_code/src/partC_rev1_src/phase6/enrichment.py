"""Fresh-seed proportional rounds with a durable fair queue and original clock."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import sqlite3
import time
import math
import json
from functools import lru_cache
import numpy as np
import pandas as pd

from .common import (read_json,read_npz,write_json,write_npz,write_frame,event,sha,digest,Monitor,
                     elapsed,can_admit,tree_bytes,WRITE_LOCK)
from .registry import region_dir,save_snapshot_npz
from .storage import Store
from .sampling import extra_candidate
from .worker import solve_sample
from .resume_policy import enabled,require_readiness


def connect(c):
    db=sqlite3.connect(Path(c["output_root"])/"control/enrichment.sqlite",timeout=30)
    db.row_factory=sqlite3.Row
    return db


class KnownInputs:
    def __init__(self,db,owner=None): self.db=db; self.owner=owner
    def __contains__(self,ih):
        row=self.db.execute("SELECT region_id,round_index,absolute_slot FROM known_inputs WHERE input_hash=?",(ih,)).fetchone()
        return row is not None and (self.owner is None or tuple(row)!=self.owner)


def initialize(c):
    root=Path(c["output_root"])
    if enabled(c):require_readiness(root)
    else:
        gate=read_json(root/"validation/BASELINE_COMPLETE.json")
        if not gate["baseline_complete"] or gate["status"]!="PASS": raise ValueError("Every mandatory baseline must pass before enrichment")
    path=root/"control/enrichment.sqlite"
    fresh=not path.exists()
    with connect(c) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS known_inputs (input_hash TEXT PRIMARY KEY, region_id TEXT NOT NULL, kind TEXT NOT NULL, round_index INTEGER NOT NULL, absolute_slot INTEGER NOT NULL)")
        db.execute("CREATE TABLE IF NOT EXISTS waves (round_index INTEGER NOT NULL, wave TEXT NOT NULL, region_id TEXT NOT NULL, quota INTEGER NOT NULL, next_slot INTEGER NOT NULL DEFAULT 0, completed INTEGER NOT NULL DEFAULT 0, quarantined INTEGER NOT NULL DEFAULT 0, degenerate INTEGER NOT NULL DEFAULT 0, tie TEXT NOT NULL, PRIMARY KEY(round_index,wave,region_id))")
        db.execute("CREATE TABLE IF NOT EXISTS completed_batches (round_index INTEGER, wave TEXT, region_id TEXT, start_slot INTEGER, result_path TEXT, PRIMARY KEY(round_index,wave,region_id,start_slot))")
        if fresh:
            plan=pd.read_parquet(root/"samples/baseline_plan.parquet").drop_duplicates("input_hash")
            db.executemany("INSERT INTO known_inputs VALUES (?,?,?,?,?)",[(r.input_hash,r.region_id,"BASELINE",0,int(r.role_index)) for r in plan.itertuples()])
            db.executemany("INSERT INTO meta VALUES (?,?)",[("run_id",c["run_id"]),("original_start",str(c["task_started_unix"])),("state","ACTIVE")])
        else:
            identity=dict(db.execute("SELECT key,value FROM meta"))
            if identity["run_id"]!=c["run_id"] or identity["original_start"]!=str(c["task_started_unix"]): raise ValueError("Resume cannot reset original clock")
    return fresh


def absolute_slot(round_index,G,wave,local_slot):
    # Absolute position counts the 32 original training roles, not aliases as new inputs.
    offset=32 if round_index==1 else (round_index-1)*G
    return offset+(0 if wave=="A" else 32)+local_slot


def batch(c,rid,round_index,wave,start,count):
    if count<1 or count>4: raise ValueError("Microbatch limit")
    rd=region_dir(c,rid); d=read_json(rd/"descriptor.json")
    if not d["m_active"]: raise ValueError("Inactive region cannot enter enrichment")
    out=rd/"enrichment"/f"round_{round_index:06d}"/("wave_"+wave)
    stem=f"batch_{start:09d}"
    result_path=out/(stem+"_result.json")
    if result_path.exists(): return read_json(result_path)
    a=read_npz(rd/"local_graph.npz"); low=read_npz(rd/"support_map.npz")["LOW_ACTIVE"]
    scales=read_npz(rd/"scales.npz"); pairs=pd.read_parquet(rd/"active_pairs.parquet")
    frozen=read_json(rd/"BASELINE_PLAN_FROZEN.json"); protected=set(frozen["heldout_hashes"])
    store=Store(rd,d,a); plan_path=out/(stem+"_plan.json"); inputs_path=out/(stem+"_inputs.npz")
    with connect(c) as db:
        if plan_path.exists():
            plan=read_json(plan_path)
            if plan["model_hash"]!=d["model_hash"] or sha(inputs_path)!=plan["inputs_sha256"]: raise ValueError("Changed resumed enrichment plan")
            inputs=read_npz(inputs_path)["x"]
        else:
            rows=[]; xs=[]; ledger=[]
            for local in range(start,start+count):
                slot=absolute_slot(round_index,d["G_target"],wave,local)
                x,record,attempts=extra_candidate(d,low,pairs.source_terminal_index.to_numpy(),pairs.target_terminal_index.to_numpy(),
                    scales,round_index,slot,KnownInputs(db,(rid,round_index,slot)),protected)
                ledger.extend(attempts)
                row={**record,"local_wave_slot":local,"absolute_slot":slot,"input_index":-1 if x is None else len(xs)}
                if x is not None:
                    xs.append(x)
                    # Reserve the new native hash before another desired slot is drawn.
                    with WRITE_LOCK:
                        db.execute("INSERT OR IGNORE INTO known_inputs VALUES (?,?,?,?,?)",(record["input_hash"],rid,"EXTRA",round_index,slot))
                        owner=db.execute("SELECT region_id,round_index,absolute_slot FROM known_inputs WHERE input_hash=?",(record["input_hash"],)).fetchone()
                        if tuple(owner)!=(rid,round_index,slot): raise ValueError("Input reservation belongs to another role")
                        db.commit()
                rows.append(row)
            inputs=np.asarray(xs,dtype=np.float64).reshape((-1,d["m_active"]))
            save_snapshot_npz(inputs_path,x=inputs)
            plan=dict(region_id=rid,model_hash=d["model_hash"],support_hash=d["support_hash"],round_index=round_index,
                wave=wave,start_slot=start,count=count,rows=rows,redraw_ledger=ledger,inputs_sha256=sha(inputs_path),
                heldout_binding_sha256=sha(rd/"BASELINE_PLAN_FROZEN.json"),generator="PCG64",numpy_version=np.__version__)
            write_json(plan_path,plan)
    started=time.perf_counter(); results=[]; warm=None
    before_bytes=sum(p.stat().st_size for p in [plan_path,inputs_path])
    for row in plan["rows"]:
        if row["input_index"]<0:
            results.append({**row,"result":"NO_UNIQUE_INPUT"}); continue
        x=inputs[row["input_index"]]
        meta={"split":"TRAIN","family":row["family"],"round_index":round_index,"absolute_slot":row["absolute_slot"],
              "seed":row["seed"],"redraw":row["redraw"],"wave":wave,"baseline":False,
              "support_hash":d["support_hash"],"plan_sha256":sha(plan_path),
              "effective_finalization_reserve_seconds":c["resources"]["sealing_reserve_seconds"]}
        r,disposition=solve_sample(c,store,x,meta,1e-5,scales,warm)
        if disposition=="COLD_RETRY_REQUIRED" and not enabled(c): r,disposition=solve_sample(c,store,x,meta,1e-5,scales,retry=True)
        accepted=r is not None and store.exists(row["input_hash"])
        if accepted: warm=r
        results.append({**row,"result":"CERTIFIED" if accepted else disposition,
                        "quality":r.quality if r is not None else None,"seconds":r.seconds if r is not None else 0.})
    record=dict(region_id=rid,round_index=round_index,wave=wave,start_slot=start,count=count,
        status="PASS" if all(r["result"] in {"CERTIFIED","NO_UNIQUE_INPUT"} for r in results) else "PARTIAL",
        completed=sum(r["result"]=="CERTIFIED" for r in results),
        quarantined=sum(r["result"] not in {"CERTIFIED","NO_UNIQUE_INPUT","DEFERRED_DEADLINE"} for r in results),
        deferred=sum(r["result"]=="DEFERRED_DEADLINE" for r in results),
        degenerate=sum(r["result"]=="NO_UNIQUE_INPUT" for r in results),rows=results,
        seconds=time.perf_counter()-started,result_path=str(result_path),plan_bytes=before_bytes,
        original_clock=c["task_started_unix"])
    write_json(result_path,record)
    return record


@lru_cache(maxsize=1)
def baseline_reopen_seconds(root_string):
    root=Path(root_string)
    records=[json.loads(line) for line in (root/"control/resource_history.jsonl").read_text().splitlines()]
    audit=[r for r in records if r["stage"]=="P4_BASELINE_REOPEN"]
    measured=(max(r["unix"] for r in audit)-min(r["unix"] for r in audit)) if len(audit)>1 else 300.
    return max(300.,measured)


@lru_cache(maxsize=1)
def region_reopen_costs(root_string):
    frame=pd.read_parquet(Path(root_string)/"validation/reopen_costs.parquet")
    return {r.region_id:(r.conservative_per_label_seconds,r.setup_seconds) for r in frame.itertuples()}


def forecast_finalization(c,total_labels,added_bytes,extra_counts=None):
    # Includes fresh source verification, inventory, summary compilation and compact seal.
    if extra_counts is None:
        label_cost=baseline_reopen_seconds(c["output_root"])/75276*total_labels
    else:
        rates=region_reopen_costs(c["output_root"])
        label_cost=sum(rate*(49+extra_counts.get(rid,0))+setup for rid,(rate,setup) in rates.items())
        label_cost=max(label_cost,baseline_reopen_seconds(c["output_root"])/75276*total_labels)
    return label_cost+1800.+added_bytes/(20*2**20)


def run_enrichment(c):
    if enabled(c):
        from .resume_enrichment import run
        return run(c)
    root=Path(c["output_root"]); initialize(c)
    regions=pd.read_parquet(root/"registry/regions.parquet"); active=regions.loc[regions.m_active.gt(0)]
    descriptors={r.region_id:r for r in active.itertuples()}
    end_reason=None; rounds=[]; batch_count=0; initial=time.time()
    with Monitor(c,"E_PROPORTIONAL_ROUNDS") as monitor,connect(c) as db,ThreadPoolExecutor(max_workers=2,thread_name_prefix="enrichment") as executor:
        current=db.execute("SELECT MAX(round_index) FROM waves").fetchone()[0] or 1
        for round_index in range(int(current),1000000000):
            if end_reason: break
            for wave in ["A","B"]:
                with WRITE_LOCK:
                    for r in active.itertuples():
                        quota=(r.G_target-32 if round_index==1 else r.G_target)
                        size=32 if wave=="A" else quota-32
                        db.execute("INSERT OR IGNORE INTO waves(round_index,wave,region_id,quota,tie) VALUES (?,?,?,?,?)",
                            (round_index,wave,r.region_id,size,digest([r.region_id,round_index,wave])))
                    db.commit()
                running={}; running_regions=set(); last_status=0.
                while True:
                    monitor.check()
                    done_labels=db.execute("SELECT COALESCE(SUM(completed),0) FROM waves").fetchone()[0]
                    extra_counts={row[0]:row[1] for row in db.execute("SELECT region_id,SUM(completed) FROM waves GROUP BY region_id")}
                    forecast=forecast_finalization(c,75276+done_labels,monitor.last["added_output_bytes"],extra_counts)
                    guard=max(5400.,1.5*forecast)
                    dynamic={**c,"resources":{**c["resources"],"sealing_reserve_seconds":guard}}
                    storage_left=c["resources"]["added_disk_bytes"]-2*2**30-monitor.last["added_output_bytes"]
                    admission=(elapsed(c)+1200<86400-guard and storage_left>32*2**20)
                    if not admission and end_reason is None:
                        end_reason="ORIGINAL_TIME_WITH_MEASURED_FINALIZATION_RESERVE" if elapsed(c)+1200>=86400-guard else "STORAGE_WITH_FINALIZATION_RESERVE"
                        event(root/"control/enrichment_admission.jsonl",dict(reason=end_reason,elapsed=elapsed(c),reserve=guard,
                              forecast_finalize_seconds=forecast,total_labels=75276+done_labels,added_bytes=monitor.last["added_output_bytes"]))
                    while admission and len(running)<2:
                        candidates=db.execute("SELECT * FROM waves WHERE round_index=? AND wave=? AND next_slot<quota ORDER BY CAST(completed AS REAL)/MAX(quota,1),tie",
                                              (round_index,wave)).fetchall()
                        chosen=next((row for row in candidates if row["region_id"] not in running_regions),None)
                        if chosen is None: break
                        rid=chosen["region_id"]; start=chosen["next_slot"]; count=min(4,chosen["quota"]-start)
                        future=executor.submit(batch,dynamic,rid,round_index,wave,start,count)
                        running[future]=(rid,start,count); running_regions.add(rid)
                        event(root/"control/enrichment_fair_dispatch.jsonl",dict(round_index=round_index,wave=wave,region_id=rid,
                            start_slot=start,count=count,completed=chosen["completed"],quota=chosen["quota"],
                            fraction=chosen["completed"]/chosen["quota"],stable_tie=chosen["tie"],reserve=guard))
                    if time.time()-last_status>=30:
                        status=dict(status="EXECUTING",round_index=round_index,wave=wave,completed_new_labels=done_labels,
                            running_regions=sorted(running_regions),elapsed_seconds=elapsed(c),reserved_finalization_seconds=guard,
                            forecast_finalization_seconds=forecast,added_output_bytes=monitor.last["added_output_bytes"],end_reason=end_reason)
                        write_json(root/"control/enrichment_status.json",status,new=False); last_status=time.time()
                    if not running: break
                    completed,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
                    for future in completed:
                        rid,start,count=running.pop(future); running_regions.remove(rid)
                        try: result=future.result()
                        except BaseException as exc:
                            write_json(root/"control/STOP_SCIENCE.json",dict(stage="ENRICHMENT",region_id=rid,error=repr(exc)),new=False)
                            raise
                        with WRITE_LOCK:
                            db.execute("INSERT INTO completed_batches VALUES (?,?,?,?,?)",(round_index,wave,rid,start,result["result_path"]))
                            db.execute("UPDATE waves SET next_slot=?,completed=completed+?,quarantined=quarantined+?,degenerate=degenerate+? WHERE round_index=? AND wave=? AND region_id=?",
                                (start+count,result["completed"],result["quarantined"],result["degenerate"],round_index,wave,rid))
                            db.commit()
                        batch_count+=1
                        if batch_count%100==0: print("E",round_index,wave,"batches",batch_count,"new labels",done_labels+result["completed"],flush=True)
                        if result["deferred"]: end_reason="ORIGINAL_TIME_ADMISSION_DEFERRED"
                wave_rows=[dict(row) for row in db.execute("SELECT * FROM waves WHERE round_index=? AND wave=? ORDER BY region_id",(round_index,wave))]
                pending=sum(r["quota"]-r["next_slot"] for r in wave_rows)
                rec=dict(round_index=round_index,wave=wave,target=sum(r["quota"] for r in wave_rows),
                    completed=sum(r["completed"] for r in wave_rows),quarantined=sum(r["quarantined"] for r in wave_rows),
                    degenerate=sum(r["degenerate"] for r in wave_rows),unattempted=pending,
                    status="PARTIAL_AT_RESOURCE_BOUND" if pending or end_reason else "ALL_SLOTS_TERMINAL")
                event(root/"control/enrichment_wave_results.jsonl",rec); rounds.append(rec)
                if end_reason: break
                if pending: raise ValueError("Earlier wave has waiting tasks")
            if end_reason: break
        allrows=[dict(row) for row in db.execute("SELECT * FROM waves ORDER BY round_index,wave,region_id")]
        write_frame(root/"samples/enrichment_quotas.parquet",pd.DataFrame(allrows))
        result=dict(status="ENRICHMENT_PARTIAL_AT_RESOURCE_BOUND",reason=end_reason,workers=2,
            new_unique_certified=sum(r["completed"] for r in allrows),quarantined_slots=sum(r["quarantined"] for r in allrows),
            no_unique_slots=sum(r["degenerate"] for r in allrows),attempted_rounds=max(r["round_index"] for r in allrows),
            wall_seconds=time.time()-initial,baseline_complete_preserved=True,original_start=c["task_started_unix"],
            finalization_reserve_used=guard,round_and_wave_results=rounds)
        write_json(root/"samples/ENRICHMENT_RESULT.json",result)
        db.execute("INSERT OR REPLACE INTO meta VALUES ('state','SEALED_FOR_FINAL_AUDIT')"); db.commit()
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return result
