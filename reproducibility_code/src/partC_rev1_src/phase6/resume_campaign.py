"""Automatic R1-R5 continuation with frozen denominators and regional isolation."""
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import json
import time
import traceback
import numpy as np
import pandas as pd

from .common import (read_json,read_npz,write_json,write_frame,event,Monitor,can_admit,sha,digest,arrays_hash,WRITE_LOCK)
from .registry import region_dir
from .resume_policy import (ledger,READY,QUARANTINED,LocalRegionError,SharedIntegrityError,
                            FailureThreshold,UnavailableRegion,source_mismatches,comparison_ready)


def control_stop(c,stage,exc):
    root=Path(c["output_root"])
    record=dict(stage=stage,error_type=type(exc).__name__,error=repr(exc),traceback=traceback.format_exc(),
        authority_sha256=c["authority_zip_sha256"],original_start=c["task_started_unix"])
    event(root/"control/integrity_incidents.jsonl",record)
    write_json(root/"control/STOP_SCIENCE.json",record,new=False)


def run_scoped_jobs(c,jobs,stage,fn,*,max_workers=2):
    if max_workers!=2:raise SharedIntegrityError("Worker expansion requires a measured speed/RSS record")
    jobs=list(jobs)
    if len(jobs)!=len(set(jobs)):raise SharedIntegrityError("Concurrent duplicate region jobs")
    root=Path(c["output_root"]);book=ledger(str(root));rows=[];queue=iter(jobs);exhausted=False
    with Monitor(c,stage) as monitor,ThreadPoolExecutor(max_workers=2,thread_name_prefix="resume_region") as pool:
        running={}
        try:
            while running or not exhausted:
                monitor.check()
                if not book.within_limit:raise FailureThreshold("Frozen active failure fraction reached 5%")
                while not exhausted and len(running)<2:
                    try:rid=next(queue)
                    except StopIteration:exhausted=True;break
                    if rid not in book.active:raise SharedIntegrityError("Unknown/inactive regional dispatch")
                    if book.states[rid]["state"]==QUARANTINED:
                        event(root/"control/dispatch_history.jsonl",dict(stage=stage,region_id=rid,status="QUARANTINED_SKIP"));continue
                    if not can_admit(c,300 if stage=="R2_RECONCILE" else 600):
                        book.set(rid,"RESOURCE_DEFERRED","ORIGINAL_DEADLINE_ADMISSION")
                        exhausted=True;break
                    before=book.states[rid]["state"]
                    book.set(rid,"RUNNING",stage)
                    running[pool.submit(fn,rid)]=(rid,before)
                if not running:break
                completed,_=wait(running,timeout=30,return_when=FIRST_COMPLETED)
                for future in completed:
                    rid,before=running.pop(future)
                    try:result=future.result()
                    except LocalRegionError as exc:
                        result=book.quarantine(c,rid,exc,stage)
                    except Exception as exc:
                        control_stop(c,stage,exc)
                        raise SharedIntegrityError("Unresolved source/code scope at "+stage+" / "+rid) from exc
                    else:
                        status=result.get("status")
                        if before==READY and stage=="R4_PRECISION":
                            book.set(rid,READY,"HEALTHY_BASELINE_PRESERVED_AFTER_OPTIONAL_PRECISION",precision_status=status)
                        elif status=="DEFERRED_DEADLINE" or result.get("resource_deferred"):
                            book.set(rid,"RESOURCE_DEFERRED",stage)
                        elif stage=="R4_BASELINE_AUDIT" and status=="PASS":
                            book.set(rid,READY,"MANDATORY_LABELS_AND_LOCAL_AUDITS_VERIFIED")
                        elif before==READY and stage in {"R4_PRECISION","R5_CUMULATIVE_REOPEN"}:
                            book.set(rid,READY,"HEALTHY_BASELINE_PRESERVED_AFTER_"+stage)
                        else:
                            book.set(rid,"PENDING","CURRENT_AUDITS_PENDING",last_stage=stage,last_status=status)
                    rows.append(result)
                    event(root/"control/dispatch_history.jsonl",dict(stage=stage,region_id=rid,result=result))
                    if len(rows)%50==0 or len(rows)==len(jobs):print(stage,len(rows),"/",len(jobs),"failed",len(book.failed),flush=True)
        except BaseException as exc:
            # Only the controller owns a global stop. Operator interrupts are propagated,
            # never counted as regional scientific failures or silently continued.
            control_stop(c,stage,exc)
            for future in running:future.cancel()
            raise
    return rows


def verify_sources(c,filename):
    root=Path(c["output_root"]);frame=pd.read_parquet(root/"validation/source_manifest.parquet");rows=[]
    for r in frame.itertuples():
        p=Path(r.absolute_path);okay=p.stat().st_size==r.size_bytes and sha(p)==r.sha256
        rows.append(dict(path=r.absolute_path,sha256=r.sha256,bytes=r.size_bytes,unchanged=okay))
    write_frame(root/"validation"/filename,pd.DataFrame(rows))
    if not all(r["unchanged"] for r in rows):raise SharedIntegrityError("Authoritative source corruption")
    return dict(status="PASS",files=len(rows),manifest_path=str(root/"validation"/filename))


def known_witness(c):
    from .worker import baseline_region
    from .preservation import reopen_existing_region
    root=Path(c["output_root"]);read_json(root/"validation/COPY_COMPLETE.json")
    rid="partC_p2_region_9567843a01be0d75fd9b"
    summary,_=reopen_existing_region(c,rid)
    result=baseline_region(c,rid,roles=[],generate_missing=False)
    comparison=result["hierarchy"]
    if not comparison_ready(comparison) or comparison["mapped_comparison"]!="NOT_APPLICABLE_SOURCE_PATH_MISMATCH":
        raise SharedIntegrityError("Known review witness did not resolve under the exact local model")
    review=read_json(Path(c["authority_root"])/"config/review_decision.json")
    if review["reviewed_handoff_sha256"]!=sha(c["source_handoff_path"]):raise SharedIntegrityError("Wrong reviewed blocker")
    record=dict(status="VERIFIED_RESOLVED",validation_policy_id=c["validation_policy_id"],
        governance_policy_id=c["governance_policy_id"],authority_sha256=c["authority_zip_sha256"],
        reviewed_handoff_sha256=review["reviewed_handoff_sha256"],reviewed_head=review["reviewed_head"],
        local_model_changed=False,source_path_rule_changed=False,old_handoff_unchanged=True,
        known_warning_region_ids=sorted(source_mismatches(c["parent_output_root"])),
        witness_local_reopen=summary,witness_comparison=comparison,new_optimizer_solves=0,
        execution_head=c.get("execution_head"),config_sha256=c.get("execution_config_sha256"),
        implementation_sha256={p.name:sha(p) for p in Path(__file__).parent.glob("*.py")})
    latest=root/"validation/review_resolution.json"
    if latest.exists():
        old=read_json(latest);old_version=root/"validation/review_resolutions"/(digest(old)+".json")
        if not old_version.exists():write_json(old_version,old)
    version=root/"validation/review_resolutions"/(digest(record)+".json")
    if not version.exists():write_json(version,record)
    write_json(root/"validation/review_resolution.json",record,new=False)
    return record


def graph_identity(rd,d):
    a=read_npz(rd/"local_graph.npz")
    keys=["tail","head","indptr","t0","capacity","alpha","beta","blocked_through","global_node_index","global_arc_index"]
    if arrays_hash({k:a[k] for k in keys})!=d["graph_hash"]:raise SharedIntegrityError("Regional graph/cost/path-rule hash changed")
    return a


def reconcile(c):
    from .preservation import reopen_existing_region
    from .worker import baseline_region
    root=Path(c["output_root"]);read_json(root/"validation/review_resolution.json")
    registry=read_json(root/"registry/REGISTRY_COMPLETE.json")
    if registry["source_registry_hash"]!=c["expected_interface"]["required_active_registry_hash"]:
        raise SharedIntegrityError("Frozen active support hash changed")
    regions=pd.read_parquet(root/"registry/regions.parquet")
    verify_sources(c,"R2_source_invariance.parquet")
    def one(rid):
        rd=region_dir(c,rid);d=read_json(rd/"descriptor.json");graph_identity(rd,d)
        saved=rd/"RESUME_SOURCE_REOPEN.json"
        if saved.exists():return read_json(saved)
        result,labels=reopen_existing_region(c,rid)
        comparison=baseline_region(c,rid,generate_missing=False)
        result.update(baseline_status=comparison["status"],comparison=comparison["hierarchy"],
            region_id=rid,status="PRESERVED_NATIVE_REOPEN_PASS",new_optimizer_solves=0)
        write_frame(rd/"resume_source_label_scalars.parquet",pd.DataFrame(labels))
        write_json(saved,result)
        return result
    active=regions.loc[regions.m_active.gt(0)].region_id.tolist()
    rows=run_scoped_jobs(c,active,"R2_RECONCILE",one)
    for rid in regions.loc[regions.m_active.eq(0),"region_id"]:
        result,_=reopen_existing_region(c,rid);rows.append(result)
    native=[r for r in rows if r.get("status")!="QUARANTINED_LOCAL"]
    result=dict(status="PASS",regions_accounted=len(rows),active_regions=1566,no_active_regions=281,
        reused_published_coarse_labels=sum(r.get("labels",0) for r in native),
        reused_fixed_fine_labels=sum(r.get("fixed_fine_labels",0) for r in native),
        live_missing_baseline_unique=sum(r.get("missing_baseline_unique",0) for r in native),
        failed_count=len(ledger(str(root)).failed),new_optimizer_solves=0,source_expectation_not_used_to_drop_live_labels=True)
    write_json(root/"validation/R2_RECONCILE_RESULT.json",result)
    region_tables(c,"R2")
    return result


def region_tables(c,tag):
    root=Path(c["output_root"]);regions=pd.read_parquet(root/"registry/regions.parquet");book=ledger(str(root))
    warnings=source_mismatches(c["parent_output_root"]);rows=[];comparisons=[];missing_rows=[]
    for d in regions.to_dict("records"):
        rid=d["region_id"];rd=region_dir(c,rid)
        row={k:d[k] for k in ["region_id","m_active","D_full","physical_nodes","physical_arcs","low_exposure","model_hash","support_hash"]}
        if not d["m_active"]:
            row.update(state="NO_ACTIVE_LOW_PAIRS",planned_unique=0,completed_unique=0,missing_unique=0,required_diagnostics_ready=True,reused=0,new=0)
        else:
            plan=pd.read_parquet(rd/"baseline_plan.parquet");required=set(plan.input_hash)
            receipts={p.stem:p for p in (rd/"labels/receipts").glob("*.json")}
            existing=set(receipts);missing=required-existing
            source=read_json(rd/"RESUME_SOURCE_REOPEN.json") if (rd/"RESUME_SOURCE_REOPEN.json").exists() else {}
            state=book.states[rid]
            row.update(state=state["state"],reason=state["reason"],planned_unique=len(required),completed_unique=len(required&existing),
                missing_unique=len(missing),required_diagnostics_ready=(rd/"BASELINE_AUDIT_PASS.json").exists(),
                reused=source.get("labels",0),new=len(existing)-source.get("labels",0),incident_path=state.get("incident_path"))
            row["updated_unix"]=state.get("updated_unix")
            if tag=="FINAL":
                grades={"A":0,"B":0,"C":0};fallbacks=0
                for receipt_path in receipts.values():
                    receipt=read_json(receipt_path);grades[receipt["scalars"]["quality"]]+=1
                    fallbacks+=bool(receipt["metadata"].get("finite_tolerance_fallback"))
                row.update(quality_A=grades["A"],quality_B=grades["B"],quality_C=grades["C"],explicit_fallback_labels=fallbacks)
            status=read_json(rd/"baseline_status.json") if (rd/"baseline_status.json").exists() else {}
            h=status.get("hierarchy",{})
            scalar={k:v for k,v in h.items() if not isinstance(v,(dict,list))}
            scalar.update(region_id=rid,known_source_warning=rid in warnings,premise_evidence=json.dumps(h.get("premise_evidence"),sort_keys=True))
            if not scalar.get("mapped_premise"):
                scalar.update(mapped_premise="SOURCE_PATH_MISMATCH" if rid in warnings else "NOT_ESTABLISHED",
                    mapped_comparison="NOT_APPLICABLE_SOURCE_PATH_MISMATCH" if rid in warnings else "NOT_ESTABLISHED_NONBLOCKING",
                    same_model_coarse_fixed="PENDING",policy_decision="PENDING")
            comparisons.append(scalar)
            for ih in sorted(missing):
                slots=plan.loc[plan.input_hash.eq(ih)]
                missing_rows.append(dict(region_id=rid,input_hash=ih,model_hash=d["model_hash"],support_hash=d["support_hash"],
                    state=state["state"],role_indices=json.dumps([int(x) for x in slots.role_index]),
                    families=json.dumps(slots.family.tolist()),native_input_path=str(rd/"baseline_inputs.npz"),
                    input_index=int(slots.iloc[0].input_index),baseline_plan_path=str(rd/"baseline_plan.parquet")))
        rows.append(row)
    frames={"region_status":pd.DataFrame(rows),"benchmark_comparison_by_region":pd.DataFrame(comparisons),"missing_mandatory_inputs":pd.DataFrame(missing_rows)}
    for name,frame in frames.items():
        path=root/"validation"/(name+"_"+tag+".parquet");write_frame(path,frame)
        frame.to_csv(root/"validation"/(name+"_"+tag+".csv"),index=False)
    record=dict(tag=tag,region_status_path=str(root/"validation"/("region_status_"+tag+".parquet")),
        region_status_sha256=sha(root/"validation"/("region_status_"+tag+".parquet")),
        missing_mandatory_unique=len(missing_rows),**book.summary())
    write_json(root/"validation/REGION_TABLES_LATEST.json",record,new=False)
    return record


def remaining_baseline(c):
    from .worker import baseline_region
    root=Path(c["output_root"]);read_json(root/"validation/R2_RECONCILE_RESULT.json")
    regions=pd.read_parquet(root/"registry/regions.parquet");active=regions.loc[regions.m_active.gt(0)].copy()
    active["work_estimate"]=active.m_active*active.physical_arcs
    jobs=[]
    for rid in active.sort_values(["work_estimate","region_id"]).region_id:
        rd=region_dir(c,rid);p=rd/"baseline_status.json"
        if not p.exists() or read_json(p)["status"]!="BASELINE_LABELS_COMPLETE":jobs.append(rid)
    rows=run_scoped_jobs(c,jobs,"R3_HEALTHY_BASELINE",lambda rid:baseline_region(c,rid))
    record=dict(status="HEALTHY_BASELINE_DISPATCH_FINISHED",attempted_regions=len(rows),scheduled_missing_regions=len(jobs),
        bulk_failed_retry_performed=False,**region_tables(c,"R3"))
    write_json(root/"validation/R3_BASELINE_RESULT.json",record)
    return record


def healthy_gate(c):
    from .validation import baseline_audit
    from .precision import select_precision,precision_region,real_restriction_probe,native_balance_audit
    from .runtime import validate_topology,RegionalDataset
    from .storage import Store
    root=Path(c["output_root"]);book=ledger(str(root));regions=pd.read_parquet(root/"registry/regions.parquet")
    jobs=[]
    for rid in regions.loc[regions.m_active.gt(0),"region_id"]:
        p=region_dir(c,rid)/"baseline_status.json"
        if rid not in book.failed and p.exists() and read_json(p)["status"]=="BASELINE_LABELS_COMPLETE":jobs.append(rid)
    # The R2 process already re-certified every reused label; publication certifies new labels.
    audits=run_scoped_jobs(c,jobs,"R4_BASELINE_AUDIT",lambda rid:baseline_audit(c,rid,fresh=False))
    inactive=[baseline_audit(c,rid) for rid in regions.loc[regions.m_active.eq(0),"region_id"]]
    summary=book.summary(inactive_valid=len(inactive)==281 and all(r["status"]=="PASS" for r in inactive))
    if summary["pending_active_count"]:
        result=dict(status="PARTIAL_PENDING_HEALTHY_BASELINE",**summary)
        write_json(root/"validation/R4_RESULT.json",result)
        return result
    healthy=regions.loc[regions.region_id.isin(summary["healthy_ids"])]
    chosen=select_precision(root,healthy)
    write_json(root/"validation/precision_selection.json",dict(regions=chosen,basis="FROZEN_LOW_AND_STRUCTURE",quarantined_exclusions=sorted(book.failed)))
    precision=run_scoped_jobs(c,chosen,"R4_PRECISION",lambda rid:precision_region(c,rid))
    # Preserve failed nominations and deterministically replace them for a shared implementation audit.
    replacements=[]
    if len([r for r in precision if r["status"] in {"PASS","PRECISION_TARGET_NOT_REACHED"}])<12 and book.within_limit:
        pool=[r for r in select_precision(root,regions.loc[~regions.region_id.isin(set(chosen)|book.failed)]) if r not in chosen]
        needed=12-len([r for r in precision if r["status"] in {"PASS","PRECISION_TARGET_NOT_REACHED"}])
        replacements=run_scoped_jobs(c,pool[:needed],"R4_PRECISION",lambda rid:precision_region(c,rid));precision+=replacements
    healthy=regions.loc[regions.region_id.isin(book.summary()["healthy_ids"])]
    restriction=real_restriction_probe(c,healthy)
    with Monitor(c,"R4_GLOBAL_INTERFACE"):
        balances=native_balance_audit(c,regions)
        runtime=validate_topology(c)
    precision_done=sum(r["status"] in {"PASS","PRECISION_TARGET_NOT_REACHED"} for r in precision)>=12
    global_ok=precision_done and restriction["status"]=="PASS" and runtime["status"]=="PASS" and balances["status"]=="PASS"
    tables=region_tables(c,"R4")
    record=dict(status="PASS" if global_ok else "PARTIAL_REQUIRED_DIAGNOSTIC",**book.summary(global_integrity=global_ok),
        no_active_ids=regions.loc[regions.m_active.eq(0),"region_id"].tolist(),
        region_status_path=tables["region_status_path"],region_status_sha256=tables["region_status_sha256"],
        review_resolution_sha256=sha(root/"validation/review_resolution.json"),precision_regions_completed=sum(r.get("inputs",0)>0 for r in precision),
        precision_target_notes=precision,precision_replacements=replacements,restriction=restriction,runtime=runtime,
        baseline_all_regions_complete=book.summary(global_integrity=global_ok)["all_regions_dataset_complete"],
        phase7_training_authorized=False,topology_unchanged_by_quarantine=True)
    write_json(root/"validation/BASELINE_SCOPE_READINESS.json",record)
    if not record["healthy_scope_complete"]:
        write_json(root/"validation/R4_RESULT.json",record);return record
    costs=[];views=[]
    for rid in record["healthy_ids"]:
        rd=region_dir(c,rid);d=read_json(rd/"descriptor.json");before=time.perf_counter()
        store=Store(rd,d,read_npz(rd/"local_graph.npz"));setup=time.perf_counter()-before
        plan=pd.read_parquet(rd/"baseline_plan.parquet");times=[];solve_times=[];payload_sizes=[]
        for role in [1,31,45]:
            before=time.perf_counter();r,receipt=store.get(plan.loc[plan.role_index.eq(role),"input_hash"].iloc[0],recertify=True);times.append(time.perf_counter()-before)
            solve_times.append(r.seconds);payload_sizes.append(receipt["payload_bytes"])
        costs.append(dict(region_id=rid,setup_seconds=setup,conservative_per_label_seconds=2*max(times)+.002,
            sample_reopen_max_seconds=max(times),m_active=d["m_active"],physical_arcs=d["physical_arcs"],
            sample_solve_forecast_seconds=min(300.,max(.1,2*max(solve_times)+.05)),
            sample_added_bytes_forecast=max(256*1024,4*max(payload_sizes))))
        for grades in [("A","B"),("A","B","C")]:
            count=sum(1 for _ in RegionalDataset(root,rid,grades=grades).iter_labels(normalized=True))
            views.append(dict(region_id=rid,view="".join(grades),training_unique_inputs=count))
    for rid in record["quarantined_ids"]:
        try:RegionalDataset(root,rid)
        except UnavailableRegion:pass
        else:raise SharedIntegrityError("Quarantined region escaped default loader")
    write_frame(root/"validation/reopen_costs.parquet",pd.DataFrame(costs))
    write_frame(root/"validation/baseline_loader_views.parquet",pd.DataFrame(views))
    inventory_calibration(c)
    write_json(root/"dataset/availability.json",dict(status="HEALTHY_SUBSET_READY",healthy_ids=record["healthy_ids"],
        unavailable_region_ids=record["quarantined_ids"],default_grades=["A","B"],opt_in_grades=["A","B","C"],
        baseline_readiness_sha256=sha(root/"validation/BASELINE_SCOPE_READINESS.json"),topology_retains_all_80316_edges=True))
    write_json(root/"validation/R4_RESULT.json",record)
    return record


def inventory_calibration(c):
    from .artifacts import describe
    root=Path(c["output_root"]);copy_record=read_json(root/"validation/COPY_COMPLETE.json")
    paths=[]
    for shard in copy_record["manifest_shards"]:
        frame=pd.read_parquet(shard["path"])
        selected=frame.loc[frame.relative_path.str.contains("/payloads/")].iloc[::97]
        paths.extend(Path(p) for p in selected.destination_path)
    paths=paths[:512]
    if len(paths)<64:raise SharedIntegrityError("Insufficient native inventory calibration samples")
    def inspect(p):return describe(p),sha(p)
    started=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(inspect,paths))
    seconds=time.perf_counter()-started
    record=dict(status="PASS",sampled_files=len(paths),wall_seconds=seconds,parallel_readers=4,
        conservative_parallel_seconds_per_file=max(.004,2*seconds/len(paths)),
        basis="DETERMINISTIC_PRESERVED_NATIVE_PAYLOAD_SCHEMA_AND_HASH_READS")
    write_json(root/"validation/inventory_cost_calibration.json",record)
    return record


def cumulative_reopen(c):
    from .preservation import reopen_existing_region
    from .runtime import RegionalDataset,validate_topology
    root=Path(c["output_root"]);regions=pd.read_parquet(root/"registry/regions.parquet");book=ledger(str(root))
    rows=[];shards=[]
    gate_path=root/"validation/BASELINE_SCOPE_READINESS.json"
    gate=read_json(gate_path) if gate_path.exists() else {}
    default_scope=bool(gate.get("healthy_scope_complete") and gate.get("global_integrity_pass") and not (root/"control/STOP_SCIENCE.json").exists())
    with Monitor(c,"R5_CUMULATIVE_REOPEN"):
        for i,rid in enumerate(regions.region_id):
            rd=region_dir(c,rid);d=read_json(rd/"descriptor.json")
            if rid in book.failed:
                rows.append(dict(region_id=rid,status=QUARANTINED,default_loader_available=False));continue
            try:result,labels=reopen_existing_region(c,rid)
            except LocalRegionError as exc:
                rows.append(book.quarantine(c,rid,exc,"R5_CUMULATIVE_REOPEN"));continue
            for label in labels:
                label["phase7_usable"]=bool(default_scope and book.states[rid]["state"]==READY and label["quality"] in {"A","B"})
            path=root/"validation/cumulative_labels"/(rid.removeprefix("partC_p2_region_")+".parquet")
            if labels:write_frame(path,pd.DataFrame(labels));shards.append(dict(path=str(path),sha256=sha(path),rows=len(labels)))
            result["default_loader_available"]=default_scope and rid in book.active and book.states[rid]["state"]==READY
            result["default_loader_rejected"]=not result["default_loader_available"]
            if result["default_loader_available"]:
                # Validate ordinary access only after its actual manifest gate exists.
                if (root/"validation/BASELINE_SCOPE_READINESS.json").exists():
                    result["default_AB_TRAIN_rows"]=sum(1 for _ in RegionalDataset(root,rid).iter_labels())
            rows.append(result)
            if (i+1)%100==0:print("R5 REOPEN",i+1,"/1847",flush=True)
        original=verify_sources(c,"R5_source_invariance.parquet")
        topology=validate_topology(c,output_name="R5_RUNTIME_PASS.json")
    write_frame(root/"validation/cumulative_region_reopens.parquet",pd.DataFrame(rows))
    from .resume_preservation import verify_copy_lineage
    lineage=verify_copy_lineage(c)
    record=dict(status="PASS",regions=len(rows),eligible_labels=sum(s["rows"] for s in shards),
        scalar_shards=shards,source_invariance=original,topology=topology,copy_lineage=lineage,fresh_process_pid=__import__("os").getpid())
    write_json(root/"validation/R5_REOPEN_RESULT.json",record)
    region_tables(c,"R5")
    if gate.get("healthy_scope_complete"):
        current=book.summary(global_integrity=bool(gate.get("global_integrity_pass") and not (root/"control/STOP_SCIENCE.json").exists()))
        write_json(root/"dataset/availability_after_R5.json",dict(**current,default_grades=["A","B"],opt_in_grades=["A","B","C"],
            topology_retains_all_80316_edges=True,source_reopen_sha256=sha(root/"validation/R5_REOPEN_RESULT.json")))
        old=root/"dataset/availability.json"
        if old.exists() and not (root/"dataset/availability_at_baseline.json").exists():write_json(root/"dataset/availability_at_baseline.json",read_json(old))
        write_json(old,read_json(root/"dataset/availability_after_R5.json"),new=False)
    return record
