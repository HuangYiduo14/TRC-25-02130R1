"""Healthy-subset fair waves, explicit terminal skips and measured original-clock admission."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import time
import json
import pandas as pd

from .common import read_json,write_json,write_frame,event,Monitor,digest,elapsed,WRITE_LOCK
from .enrichment import initialize,connect,batch
from .resume_policy import ledger,READY,QUARANTINED,LocalRegionError,SharedIntegrityError,FailureThreshold,require_readiness


def ensure_skip_columns(db):
    columns={row[1] for row in db.execute("PRAGMA table_info(waves)")}
    for name,kind in [("skipped","INTEGER NOT NULL DEFAULT 0"),("skip_reason","TEXT")]:
        if name not in columns:db.execute(f"ALTER TABLE waves ADD COLUMN {name} {kind}")
    db.execute("CREATE TABLE IF NOT EXISTS optional_stops (region_id TEXT PRIMARY KEY, reason TEXT NOT NULL)")
    db.commit()


def terminal_skips(db,quarantined):
    changes=[]
    reasons={rid:"QUARANTINED_SKIP" for rid in quarantined}
    for row in db.execute("SELECT region_id,reason FROM optional_stops"):
        reasons.setdefault(row[0],row[1])
    for rid,reason in reasons.items():
        rows=db.execute("SELECT round_index,wave,quota-next_slot FROM waves WHERE region_id=? AND next_slot<quota",(rid,)).fetchall()
        db.execute("UPDATE waves SET skipped=skipped+quota-next_slot,next_slot=quota,skip_reason=? WHERE region_id=? AND next_slot<quota",(reason,rid))
        changes.extend(dict(region_id=rid,round_index=row[0],wave=row[1],skipped=row[2],reason=reason) for row in rows)
    db.commit()
    return changes


def finalization_forecast(c,rates,extra_counts,total_extra,batches):
    # Measured native reopens plus a conservative source-observed file inventory cost.
    local=sum(float(r.conservative_per_label_seconds)*(49+extra_counts.get(rid,0))+float(r.setup_seconds)
              for rid,r in rates.items())
    calibration=Path(c["output_root"])/"validation/inventory_cost_calibration.json"
    seconds_per_file=read_json(calibration)["conservative_parallel_seconds_per_file"] if calibration.exists() else .012
    inventory=seconds_per_file*(171759+3*total_extra+2*batches)
    return dict(total_seconds=local+inventory+1200.,native_reopen_seconds=local,inventory_seconds=inventory,
                source_report_archive_overhead_seconds=1200.,seconds_per_inventory_file=seconds_per_file)


def run(c):
    root=Path(c["output_root"]);initialize(c);gate=require_readiness(root);book=ledger(str(root))
    regions=pd.read_parquet(root/"registry/regions.parquet");active=regions.loc[regions.m_active.gt(0)]
    costs=pd.read_parquet(root/"validation/reopen_costs.parquet")
    rates={r.region_id:r for r in costs.itertuples()}
    forecasts={rid:float(r.sample_solve_forecast_seconds) for rid,r in rates.items()}
    initial=time.time();end_reason=None;batch_count=0;wave_results=[];guard=5400.
    with Monitor(c,"E_PROPORTIONAL_ROUNDS") as monitor,connect(c) as db,ThreadPoolExecutor(max_workers=2,thread_name_prefix="resume_extra") as executor:
        ensure_skip_columns(db)
        row=db.execute("SELECT MAX(round_index) FROM waves").fetchone();current=int(row[0] or 1)
        try:
            for round_index in range(current,1000000000):
                for wave in ["A","B"]:
                    with WRITE_LOCK:
                        for r in active.itertuples():
                            quota=int(r.G_target)-32 if round_index==1 else int(r.G_target)
                            size=32 if wave=="A" else quota-32
                            db.execute("INSERT OR IGNORE INTO waves(round_index,wave,region_id,quota,tie) VALUES (?,?,?,?,?)",
                                (round_index,wave,r.region_id,size,digest([r.region_id,round_index,wave])))
                        db.commit()
                    running={};busy=set();last_status=0.
                    while True:
                        monitor.check()
                        if not book.within_limit:raise FailureThreshold("79 failed active regions")
                        for rec in terminal_skips(db,book.failed):event(root/"control/enrichment_skips.jsonl",rec)
                        optional_stopped={row[0] for row in db.execute("SELECT region_id FROM optional_stops")}
                        if len(optional_stopped|book.failed)==1566:
                            end_reason="NO_HEALTHY_REGION_REMAINS_ELIGIBLE_FOR_OPTIONAL_SOLVES"
                        counts={row[0]:int(row[1]) for row in db.execute("SELECT region_id,SUM(completed) FROM waves GROUP BY region_id")}
                        total=sum(counts.values());forecast=finalization_forecast(c,rates,counts,total,batch_count)
                        guard=max(5400.,1.5*forecast["total_seconds"])
                        dynamic={**c,"resources":{**c["resources"],"sealing_reserve_seconds":guard}}
                        seconds_left=86400-guard-elapsed(c)
                        storage_left=c["resources"]["added_disk_bytes"]-2*2**30-monitor.last["added_output_bytes"]
                        candidates=db.execute("SELECT * FROM waves WHERE round_index=? AND wave=? AND next_slot<quota ORDER BY CAST(next_slot AS REAL)/MAX(quota,1),tie",
                                              (round_index,wave)).fetchall()
                        while len(running)<2 and not end_reason:
                            eligible=[]
                            for row in candidates:
                                rid=row["region_id"]
                                if rid in busy or book.states[rid]["state"]!=READY:continue
                                count=min(4,row["quota"]-row["next_slot"])
                                sample_forecast=forecasts[rid]
                                # Shrink the final microbatch; admit any useful remaining input that fits.
                                while count and (2*(sample_forecast*count+rates[rid].setup_seconds)>=seconds_left or
                                                 count*rates[rid].sample_added_bytes_forecast>storage_left):count-=1
                                if count:eligible.append((row,count))
                            if not eligible:break
                            row,count=eligible[0];rid=row["region_id"];start=int(row["next_slot"])
                            cc={**dynamic,"sample_admission_forecast_seconds":forecasts[rid]}
                            future=executor.submit(batch,cc,rid,round_index,wave,start,count)
                            running[future]=(rid,start,count);busy.add(rid)
                            storage_left-=count*rates[rid].sample_added_bytes_forecast
                            event(root/"control/enrichment_fair_dispatch.jsonl",dict(region_id=rid,round_index=round_index,wave=wave,
                                start_slot=start,count=count,quota=row["quota"],opportunity_fraction=start/max(1,row["quota"]),
                                sample_forecast_seconds=forecasts[rid],finalization_reserve_seconds=guard))
                        if time.time()-last_status>=30:
                            write_json(root/"control/enrichment_status.json",dict(status="EXECUTING",round_index=round_index,wave=wave,
                                completed_new_labels=total,running_regions=sorted(busy),failed_regions=len(book.failed),
                                elapsed_seconds=elapsed(c),reserved_finalization_seconds=guard,forecast=forecast,
                                remaining_science_seconds=seconds_left,added_output_bytes=monitor.last["added_output_bytes"]),new=False)
                            last_status=time.time()
                        if not running:
                            if candidates and end_reason is None:
                                end_reason="ORIGINAL_TIME_WITH_MEASURED_FINALIZATION_RESERVE" if seconds_left<=2*max(forecasts.values())+1 else "NO_REMAINING_MICROBATCH_FITS_TIME_OR_STORAGE"
                            break
                        done,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
                        for future in done:
                            rid,start,count=running.pop(future);busy.remove(rid)
                            try:result=future.result()
                            except LocalRegionError as exc:
                                book.quarantine(c,rid,exc,"E_PROPORTIONAL_ROUNDS")
                                for rec in terminal_skips(db,book.failed):event(root/"control/enrichment_skips.jsonl",rec)
                                continue
                            except Exception as exc:
                                from .resume_campaign import control_stop
                                control_stop(c,"E_PROPORTIONAL_ROUNDS",exc)
                                raise SharedIntegrityError("Unknown enrichment error scope: "+rid) from exc
                            with WRITE_LOCK:
                                inserted=db.execute("INSERT OR IGNORE INTO completed_batches VALUES (?,?,?,?,?)",(round_index,wave,rid,start,result["result_path"])).rowcount
                                if inserted:
                                    db.execute("UPDATE waves SET next_slot=?,completed=completed+?,quarantined=quarantined+?,degenerate=degenerate+? WHERE round_index=? AND wave=? AND region_id=?",
                                        (start+count,result["completed"],result["quarantined"],result["degenerate"],round_index,wave,rid))
                                rejected=db.execute("SELECT SUM(quarantined) FROM waves WHERE region_id=?",(rid,)).fetchone()[0] or 0
                                if rejected>=4:
                                    db.execute("INSERT OR IGNORE INTO optional_stops VALUES (?,?)",(rid,"OPTIONAL_ACCURACY_MISSES_PRESERVE_HEALTHY_BASELINE"))
                                db.commit()
                            # Includes plan, catalog, publication and native solve costs, without HIGH tuning.
                            forecasts[rid]=min(300.,max(.05,.5*forecasts[rid],2*float(result["seconds"])/max(1,count)))
                            batch_count+=1
                            if batch_count%100==0:print("E",round_index,wave,"batches",batch_count,"certified",total+result["completed"],"failed",len(book.failed),flush=True)
                            if result["deferred"]:end_reason="ORIGINAL_RESERVE_REACHED_DURING_BATCH"
                    wave_rows=[dict(row) for row in db.execute("SELECT * FROM waves WHERE round_index=? AND wave=? ORDER BY region_id",(round_index,wave))]
                    pending=sum(r["quota"]-r["next_slot"] for r in wave_rows)
                    rec=dict(round_index=round_index,wave=wave,target=sum(r["quota"] for r in wave_rows),
                        completed=sum(r["completed"] for r in wave_rows),rejected_optional=sum(r["quarantined"] for r in wave_rows),
                        skipped=sum(r["skipped"] for r in wave_rows),degenerate=sum(r["degenerate"] for r in wave_rows),unattempted=pending,
                        status="PARTIAL_AT_RESOURCE_BOUND" if pending or end_reason else "ALL_SLOTS_TERMINAL")
                    event(root/"control/enrichment_wave_results.jsonl",rec);wave_results.append(rec)
                    if end_reason:break
                    if pending:raise SharedIntegrityError("Earlier wave has unresolved waiting tasks")
                if end_reason:break
        except BaseException as exc:
            from .resume_campaign import control_stop
            control_stop(c,"E_PROPORTIONAL_ROUNDS",exc)
            raise
        finally:
            allrows=[dict(row) for row in db.execute("SELECT * FROM waves ORDER BY round_index,wave,region_id")]
            write_frame(root/"samples/enrichment_quotas.parquet",pd.DataFrame(allrows))
            result=dict(status="ENRICHMENT_PARTIAL_AT_RESOURCE_BOUND",reason=end_reason,workers=2,
                new_unique_certified=sum(r["completed"] for r in allrows),rejected_optional_slots=sum(r["quarantined"] for r in allrows),
                explicit_skipped_slots=sum(r["skipped"] for r in allrows),no_unique_slots=sum(r["degenerate"] for r in allrows),
                attempted_rounds=max((r["round_index"] for r in allrows),default=0),wall_seconds=time.time()-initial,
                healthy_baseline_preserved=True,original_start=c["task_started_unix"],finalization_reserve_used=guard,
                round_and_wave_results=wave_results,**book.summary())
            write_json(root/"samples/ENRICHMENT_RESULT.json",result)
            db.execute("INSERT OR REPLACE INTO meta VALUES ('state','FROZEN_FOR_CUMULATIVE_REOPEN')");db.commit()
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return result
