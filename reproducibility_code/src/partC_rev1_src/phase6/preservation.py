"""Fresh-process inventory and recertification of all existing work after a gate stop."""
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .common import (read_json,read_npz,write_json,write_frame,sha,input_hash,Monitor,event)
from .registry import region_dir
from .storage import Store
from .oracle import Oracle
from .runtime import validate_topology,RegionalDataset
from .validation import pair_check,cycle_check
from .sampling import rng_for
from .resume_policy import enabled


def reopen_existing_region(c,rid):
    rd=region_dir(c,rid); d=read_json(rd/"descriptor.json")
    a=read_npz(rd/"local_graph.npz"); started=time.perf_counter()
    if not d["m_active"]:
        forbidden=[name for name in ["labels","fixed_fine","scales.npz","baseline_inputs.npz"] if (rd/name).exists()]
        if forbidden: raise ValueError("Inactive optimizer/data artifacts: "+rid)
        return dict(region_id=rid,m_active=0,status="NO_ACTIVE_LOW_PAIRS",labels=0,fixed_fine_labels=0,
                    baseline_unique_required=0,missing_baseline_unique=0,model_hash=d["model_hash"],seconds=time.perf_counter()-started),[]
    plan=pd.read_parquet(rd/"baseline_plan.parquet"); xs=read_npz(rd/"baseline_inputs.npz")["x"]
    required=set(plan.input_hash); labels=[]; records=[]; fixed_count=0; hashes=set()
    if (rd/"labels").exists():
        store=Store(rd,d,a); hashes=store.hashes(); scales=read_npz(rd/"scales.npz")
        for ih in sorted(hashes):
            r,receipt=store.get(ih,recertify=True)
            if ih not in required and not enabled(c): raise ValueError("Unexpected nonbaseline published label")
            if ih not in required and (receipt["metadata"].get("baseline",True) or receipt["metadata"].get("split")!="TRAIN"):
                raise ValueError("Unplanned nontraining label")
            if r.quality not in {"A","B","C"}: raise ValueError("Uncertified label published")
            matching=plan.loc[plan.input_hash.eq(ih)]
            if not all(np.array_equal(xs[int(j)],r.demand) for j in matching.input_index): raise ValueError("Published input differs from frozen plan")
            payload=read_npz(store.root/receipt["payload_path"])
            expected_gradient=scales["input_scale"]*(r.tau-scales["c0"])/float(scales["Vscale"])
            if not np.array_equal(payload["normalized_input"],r.demand/scales["input_scale"]) or not np.array_equal(payload["normalized_gradient"],expected_gradient):
                raise ValueError("Normalized data differs from native labels")
            pair,source,target=store.oracle.pool_arrays(); selected=r.path_ids
            keys=np.column_stack([pair[selected],source[selected],target[selected]])
            unique,inv=np.unique(keys,axis=0,return_inverse=True)
            amounts=np.bincount(inv,weights=r.weights,minlength=len(unique))
            for key,col in [("fine_pair",0),("fine_source",1),("fine_target",2)]:
                if not np.array_equal(payload[key],unique[:,col]): raise ValueError("Saved fine endpoint allocation changed")
            if not np.array_equal(payload["fine_amount"],amounts): raise ValueError("Saved fine flow changed")
            labels.append(r)
            records.append(dict(region_id=rid,input_hash=ih,model_hash=d["model_hash"],support_hash=d["support_hash"],
                m_active=d["m_active"],owner_split=matching.iloc[0].owner_split if len(matching) else receipt["metadata"]["split"],role_count=len(matching),
                phase7_usable=False,preserved_under_exact_restricted_model=True,
                baseline=ih in required,origin_run_id=receipt["metadata"].get("run_id"),
                native_payload_bytes=receipt["payload_bytes"],**r.scalars()))
    else:
        Oracle(a) # Registry-only LMO check; no solve and no output directory is created.
    diagnostics=0
    if enabled(c):
        for namespace in ["precision_cold","precision_warm","full_zero_embedding_probe"]:
            if not (rd/namespace/"catalog_index.json").exists():continue
            da=a;dd=d
            if namespace=="full_zero_embedding_probe":
                dd=read_json(rd/namespace/"descriptor.json");da=read_npz(rd/namespace/"definition.npz");da.pop("native_input")
            diagnostic=Store(rd,dd,da,namespace=namespace)
            for ih in sorted(diagnostic.hashes()):
                r,_=diagnostic.get(ih,recertify=True)
                if r.quality not in {"A","B","C"}:raise ValueError("Invalid published diagnostic")
                diagnostics+=1
    if (rd/"fixed_fine/descriptor.json").exists():
        fd=read_json(rd/"fixed_fine/descriptor.json"); fa=read_npz(rd/"fixed_fine/definition.npz")
        native=fa.pop("native_demand"); fine_to_coarse=fa.pop("fine_to_coarse")
        fs=Store(rd,fd,fa,namespace="fixed_fine")
        for ih in sorted(fs.hashes()):
            r,_=fs.get(ih,recertify=True)
            if not np.array_equal(r.demand,native): raise ValueError("Fixed-fine input changed")
            fixed_count+=1
    rng,_=rng_for(rid,0,"PRESERVATION_CROSS_CHECKS",0,support_hash=d["support_hash"])
    pc=0; cc=0
    for _ in range(32 if len(labels)>1 else 0):
        i,j=rng.choice(len(labels),2,replace=False)
        if not pair_check(labels[int(i)],labels[int(j)])["passed"]: raise ValueError("Preserved support-plane/monotonicity failure")
        pc+=1
    for j in range(8 if len(labels)>=3 else 0):
        indices=rng.choice(len(labels),min(len(labels),3+j%4),replace=False)
        if not cycle_check([labels[int(i)] for i in indices])["passed"]: raise ValueError("Preserved cyclic monotonicity failure")
        cc+=1
    rejected=False
    if not enabled(c):
        try: RegionalDataset(c["output_root"],rid)
        except ValueError as exc:
            if "Dataset blocked" not in str(exc): raise
            rejected=True
        if not rejected: raise ValueError("Blocked data escaped the default loader gate")
    result=dict(region_id=rid,m_active=d["m_active"],status="EXISTING_CERTIFICATES_PASS" if enabled(c) else "EXISTING_CERTIFICATES_PASS_DATASET_BLOCKED",
        labels=len(records),fixed_fine_labels=fixed_count,baseline_unique_required=len(required),missing_baseline_unique=len(required-hashes),
        diagnostic_labels=diagnostics,
        mandatory_role_count=len(plan),pair_checks=pc,cycle_checks=cc,model_hash=d["model_hash"],
        default_loader_rejected=True if not enabled(c) else None,seconds=time.perf_counter()-started)
    return result,records


def preserve(c):
    root=Path(c["output_root"]); blocker=read_json(root/"validation/BLOCKING_DIAGNOSIS.json")
    if blocker["status"]!="BLOCKED_WITH_EXACT_INTEGRITY_DEFECT": raise ValueError("Expected exact gate stop")
    regions=pd.read_parquet(root/"registry/regions.parquet")
    rows=[]; labels=[]; source_rows=[]
    with Monitor(c,"P6_EXISTING_WORK_FRESH_REOPEN") as monitor:
        for i,rid in enumerate(regions.region_id):
            monitor.check(); result,records=reopen_existing_region(c,rid)
            rows.append(result); labels.extend(records)
            if (i+1)%100==0: print("PRESERVATION",i+1,"/1847 labels",len(labels),flush=True)
        manifest=pd.read_parquet(root/"validation/source_manifest.parquet")
        for row in manifest.itertuples():
            p=Path(row.absolute_path); okay=p.stat().st_size==row.size_bytes and sha(p)==row.sha256
            source_rows.append(dict(absolute_path=row.absolute_path,size_bytes=row.size_bytes,sha256=row.sha256,unchanged=okay))
        if not all(r["unchanged"] for r in source_rows): raise ValueError("Authoritative source modified")
        topology=validate_topology(c,output_name="P6_PRESERVED_RUNTIME_PASS.json")
    statuses=pd.DataFrame(rows); scalar=pd.DataFrame(labels)
    out=root/"validation/preservation"
    write_frame(out/"all_region_status.parquet",statuses)
    write_frame(out/"all_existing_label_scalars.parquet",scalar)
    write_frame(out/"fresh_source_invariance.parquet",pd.DataFrame(source_rows))
    record=dict(status="FRESH_EXISTING_WORK_VERIFIED_DATASET_BLOCKED",regions=len(rows),active_regions=int(regions.m_active.gt(0).sum()),
        inactive_regions=int(regions.m_active.eq(0).sum()),published_coarse_labels=len(scalar),
        fixed_fine_labels=int(statuses.fixed_fine_labels.sum()),
        regions_with_all_mandatory_unique_labels=int(((statuses.m_active>0)&statuses.missing_baseline_unique.eq(0)).sum()),
        regions_with_missing_labels=int(((statuses.m_active>0)&statuses.missing_baseline_unique.gt(0)).sum()),
        missing_baseline_unique_inputs=int(statuses.missing_baseline_unique.sum()),
        quality_counts=scalar.quality.value_counts().to_dict(),
        default_AB_quality_only=int(scalar.quality.isin(["A","B"]).sum()),default_dataset_loader_enabled=False,
        all_published_path_fine_arc_certificates_reopened=True,all_native_normalized_labels_rechecked=True,
        authoritative_source_files_rehashed=len(source_rows),all_authoritative_sources_unchanged=True,
        active_runtime=topology,new_scientific_labels_generated_by_reopen=0,
        no_positive_omitted_demand_labels_migrated=True,no_existing_label_bytes_rewritten=True,
        fresh_process_pid=__import__("os").getpid(),campaign_status="BLOCKED_WITH_EXACT_INTEGRITY_DEFECT")
    write_json(out/"PRESERVATION_PASS.json",record)
    return record
