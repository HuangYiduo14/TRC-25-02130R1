"""Prescribed cold/warm precision and real zero-embedding restriction audits."""
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .common import (read_json,read_npz,write_json,write_npz,write_frame,digest,arrays_hash,input_hash,Monitor)
from .registry import region_dir,make_choices
from .storage import Store
from .worker import solve_sample
from .campaign import run_jobs
from .runtime import validate_topology,RegionalDataset
from .resume_policy import enabled,LocalRegionError


def select_precision(root,regions):
    selection=root/"validation/P2_selection.json"
    if not selection.exists():selection=root/"history/parent_run/validation/P2_selection.json"
    allowed=set(regions.loc[regions.m_active.gt(0),"region_id"])
    chosen=[rid for rid in read_json(selection)["region_ids"] if rid in allowed]
    if len(chosen)>=12:return chosen[:12]
    active=regions.loc[regions.m_active.gt(0)]
    extra=[]
    for col in ["m_active","physical_arcs","n_full","low_exposure"]:
        ordered=active.sort_values([col,"region_id"])
        for q in [.25,.75,.99]: extra.append(ordered.iloc[int(q*(len(ordered)-1))].region_id)
    for rid in extra+active.sort_values("region_id").region_id.tolist():
        if rid not in chosen: chosen.append(rid)
        if len(chosen)==12: break
    return chosen


def precision_region(c,rid):
    rd=region_dir(c,rid); d=read_json(rd/"descriptor.json"); a=read_npz(rd/"local_graph.npz")
    if enabled(c) and (rd/"PRECISION_AUDIT.json").exists():return read_json(rd/"PRECISION_AUDIT.json")
    base=Store(rd,d,a); cold=Store(rd,d,a,namespace="precision_cold"); warm=Store(rd,d,a,namespace="precision_warm")
    scales=read_npz(rd/"scales.npz"); plan=pd.read_parquet(rd/"baseline_plan.parquet")
    xs=read_npz(rd/"baseline_inputs.npz")["x"]; rows=[]
    for role in [1,31]:
        row=plan.loc[plan.role_index.eq(role)].iloc[0]; x=xs[int(row.input_index)]
        original,_=base.get(row.input_hash,recertify=True)
        init=warm.oracle.import_fixed_warm(base.oracle,original,np.arange(d["m_active"],dtype=np.int64),x)
        cr,cs=solve_sample(c,cold,x,dict(split="DIAGNOSTIC",role="PRECISION_COLD",role_index=role),1e-7,scales)
        if cs=="COLD_RETRY_REQUIRED" and not enabled(c): cr,cs=solve_sample(c,cold,x,dict(split="DIAGNOSTIC",role="PRECISION_COLD",role_index=role),1e-7,scales,retry=True)
        wr,ws=solve_sample(c,warm,x,dict(split="DIAGNOSTIC",role="PRECISION_WARM",role_index=role),1e-7,scales,init)
        if ws=="COLD_RETRY_REQUIRED" and not enabled(c): wr,ws=solve_sample(c,warm,x,dict(split="DIAGNOSTIC",role="PRECISION_WARM_COLD_RETRY",role_index=role),1e-7,scales,retry=True)
        if cr is None or wr is None: return dict(region_id=rid,status="DEFERRED_DEADLINE")
        if enabled(c) and (cr.quality not in {"A","B","C"} or wr.quality not in {"A","B","C"}):
            return dict(region_id=rid,status="OPTIONAL_PRECISION_UNCERTIFIED",cold_gap=cr.relative_gap,warm_gap=wr.relative_gap,
                healthy_baseline_preserved=True,not_counted_as_completed_precision=True)
        allowance=cr.roundoff_allowance+wr.roundoff_allowance+original.roundoff_allowance
        interval=max(cr.value_lower,wr.value_lower,original.value_lower)<=min(cr.value_upper,wr.value_upper,original.value_upper)+allowance
        if not interval: raise LocalRegionError("COLD_WARM_INTERVAL_CONTRADICTION",evidence=dict(region_id=rid,input_hash=row.input_hash,allowance=allowance))
        rows.append(dict(region_id=rid,role_index=role,input_hash=row.input_hash,
            original_gap=original.relative_gap,cold_gap=cr.relative_gap,warm_gap=wr.relative_gap,
            target=1e-7,target_reached=max(cr.relative_gap,wr.relative_gap)<=1e-7,
            interval_compatible=interval,roundoff_allowance=allowance,
            value_cold_warm_difference=abs(cr.value_upper-wr.value_upper),
            tau_cold_warm_max_absolute=float(np.max(np.abs(cr.tau-wr.tau))),
            tau_cold_original_max_absolute=float(np.max(np.abs(cr.tau-original.tau))),
            normalized_gradient_cold_warm_max_absolute=float(np.max(np.abs(scales["input_scale"]*(cr.tau-wr.tau)/float(scales["Vscale"])))),
            cold_seconds=cr.seconds,warm_seconds=wr.seconds,
            gradient_error_not_bounded_by_global_value_gap=True))
    write_frame(rd/"precision_audit.parquet",pd.DataFrame(rows))
    result=dict(region_id=rid,status="PASS" if all(r["target_reached"] for r in rows) else "PRECISION_TARGET_NOT_REACHED",
        inputs=len(rows),target_reached=sum(r["target_reached"] for r in rows),all_intervals_compatible=True)
    write_json(rd/"PRECISION_AUDIT.json",result)
    return result


def real_restriction_probe(c,regions):
    root=Path(c["output_root"])
    if enabled(c) and (root/"validation/REAL_ZERO_EMBEDDING_PASS.json").exists():return read_json(root/"validation/REAL_ZERO_EMBEDDING_PASS.json")
    candidates=regions.loc[regions.m_active.gt(0)&regions.D_full.gt(regions.m_active)].sort_values(["D_full","m_active","region_id"])
    rid=candidates.iloc[0].region_id; rd=region_dir(c,rid); d=read_json(rd/"descriptor.json")
    a=read_npz(rd/"local_graph.npz"); active=Store(rd,d,a)
    maps=read_npz(rd/"support_map.npz"); low=maps["LOW_ACTIVE"]
    ih=input_hash(d["model_hash"],d["support_hash"],low); ar,_=active.get(ih,recertify=True)
    full=pd.read_parquet(rd/"full_pair_audit.parquet"); fm=read_npz(rd/"fine_members.npz")
    members={int(k):fm["local_nodes"][fm["offsets"][j]:fm["offsets"][j+1]] for j,k in enumerate(fm["terminal_indices"])}
    fa=dict(a); fa.update(make_choices(full.source_terminal_index.to_numpy(),full.target_terminal_index.to_numpy(),members))
    fd={**d,"model_class":"FULL_ZERO_EMBEDDING_DIAGNOSTIC",
        "model_hash":digest({"active_model":d["model_hash"],"full_zero_embedding_probe":True})}
    fs=Store(rd,fd,fa,namespace="full_zero_embedding_probe")
    x=np.zeros(d["D_full"]); x[maps["active_indices"]]=low
    init=fs.oracle.import_fixed_warm(active.oracle,ar,maps["active_indices"],x)
    r,status=solve_sample(c,fs,x,dict(split="DIAGNOSTIC",role="REAL_EXACT_ZERO_OMITTED_MODEL_RESTRICTION"),1e-7,warm=init)
    if status=="COLD_RETRY_REQUIRED" and not enabled(c): r,status=solve_sample(c,fs,x,dict(split="DIAGNOSTIC",role="REAL_ZERO_OMITTED"),1e-7,retry=True)
    if r is None: return dict(status="DEFERRED_DEADLINE")
    allowance=r.roundoff_allowance+ar.roundoff_allowance
    okay=max(r.value_lower,ar.value_lower)<=min(r.value_upper,ar.value_upper)+allowance
    if not okay: raise ValueError("Real zero-embedding identity failed")
    write_npz(rd/"full_zero_embedding_probe/definition.npz",**fa,native_input=x)
    write_json(rd/"full_zero_embedding_probe/descriptor.json",fd)
    record=dict(status="PASS",region_id=rid,full_dimension=d["D_full"],active_dimension=d["m_active"],
        exact_zero_omitted=bool(np.all(x[maps["full_to_active"]<0]==0)),
        full_lower=r.value_lower,full_upper=r.value_upper,active_lower=ar.value_lower,active_upper=ar.value_upper,
        interval_compatible=okay,production_runtime_edges_changed=False,full_model_training_labels_published=False,
        previous_positive_demand_labels_reused=False)
    write_json(root/"validation/REAL_ZERO_EMBEDDING_PASS.json",record)
    return record


def native_balance_audit(c,regions):
    rows=[]
    for d in regions.to_dict("records"):
        rd=region_dir(c,d["region_id"]); a=read_npz(rd/"local_graph.npz"); z=read_npz(rd/"LOW_mapped_physical.npz")["z"]
        fine=pd.read_parquet(rd/"LOW_native_fine_trace.parquet")
        n=len(a["global_node_index"])
        balance=np.bincount(a["tail"],weights=z,minlength=n)-np.bincount(a["head"],weights=z,minlength=n)
        src=np.searchsorted(a["global_node_index"],fine.source_global_node.to_numpy(np.int64))
        dst=np.searchsorted(a["global_node_index"],fine.target_global_node.to_numpy(np.int64))
        rhs=np.bincount(src,weights=fine.amount,minlength=n)-np.bincount(dst,weights=fine.amount,minlength=n)
        error=float(np.max(np.abs(balance-rhs),initial=0.))
        allowance=1e-8+1e-11*max(float(np.max(np.abs(balance),initial=0.)),float(np.max(z,initial=0.)))
        rows.append(dict(region_id=d["region_id"],max_node_balance_error=error,allowance=allowance,passed=error<=allowance))
    write_frame(Path(c["output_root"])/"validation/native_LOW_node_balances.parquet",pd.DataFrame(rows))
    if not all(r["passed"] for r in rows): raise ValueError("Native LOW regional physical/fine balance failed")
    return dict(status="PASS",regions=len(rows),maximum_error=max(r["max_node_balance_error"] for r in rows))


def baseline_gate(c):
    if enabled(c):
        from .resume_campaign import healthy_gate
        return healthy_gate(c)
    root=Path(c["output_root"])
    prior=read_json(root/"validation/BASELINE_LABEL_AUDITS.json")
    if prior["status"]!="BASELINE_LABEL_AUDITS_PASS": raise ValueError("Missing mandatory baseline prevents enrichment")
    regions=pd.read_parquet(root/"registry/regions.parquet")
    chosen=select_precision(root,regions)
    write_json(root/"validation/precision_selection.json",dict(regions=chosen,basis="FROZEN_LOW_AND_STRUCTURE",target=1e-7))
    precision=run_jobs(c,chosen,"P4_PRECISION",lambda rid:precision_region(c,rid))
    restriction=real_restriction_probe(c,regions)
    with Monitor(c,"P5_BASELINE_INTERFACE"):
        balances=native_balance_audit(c,regions)
        runtime=validate_topology(c)
        views=[]; reopen_costs=[]
        for d in regions.to_dict("records"):
            if not d["m_active"]: continue
            rd=region_dir(c,d["region_id"]); native_desc=read_json(rd/"descriptor.json")
            before=time.perf_counter()
            measured_store=Store(rd,native_desc,read_npz(rd/"local_graph.npz"))
            setup_seconds=time.perf_counter()-before
            plan=pd.read_parquet(rd/"baseline_plan.parquet")
            times=[]
            for role in [1,31,45]:
                ih=plan.loc[plan.role_index.eq(role),"input_hash"].iloc[0]
                before=time.perf_counter(); measured_store.get(ih,recertify=True)
                times.append(time.perf_counter()-before)
            reopen_costs.append(dict(region_id=d["region_id"],m_active=d["m_active"],physical_arcs=d["physical_arcs"],
                setup_seconds=setup_seconds,probe_count=len(times),sample_reopen_max_seconds=max(times),
                sample_reopen_mean_seconds=float(np.mean(times)),conservative_per_label_seconds=2*max(times)+.002))
            for grades in [("A","B"),("A","B","C")]:
                loader=RegionalDataset(root,d["region_id"],grades=grades,split="TRAIN",allow_incomplete=True)
                count=0
                for label in loader.iter_labels(normalized=True):
                    if not np.array_equal(label["input"],label["input"]*1): raise ValueError("Dummy CPU identity failed")
                    count+=1
                views.append(dict(region_id=d["region_id"],view="".join(grades),training_unique_inputs=count))
        write_frame(root/"validation/baseline_loader_views.parquet",pd.DataFrame(views))
        write_frame(root/"validation/reopen_costs.parquet",pd.DataFrame(reopen_costs))
    allprecision=len(precision)==12 and all(r["status"]=="PASS" for r in precision)
    record=dict(status="PASS" if allprecision and restriction["status"]=="PASS" else "PRECISION_INCOMPLETE",
        baseline_complete=allprecision and restriction["status"]=="PASS",precision_regions=len(precision),
        precision_inputs=sum(r.get("inputs",0) for r in precision),
        all_active_mandatory_labels=True,all_inactive_skip_records=True,
        runtime=runtime,native_node_balances=balances,untrained_loader_views=True,
        enrichment_authorized_by_existing_charter=allprecision and restriction["status"]=="PASS")
    write_json(root/"validation/BASELINE_COMPLETE.json",record)
    return record
