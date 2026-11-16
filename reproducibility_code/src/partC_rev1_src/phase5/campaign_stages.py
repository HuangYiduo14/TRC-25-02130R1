"""Authorized C0-C4 stages; each stage uses complete dependency-bound inputs."""
from __future__ import annotations

import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile

import numpy as np
import pandas as pd

from .bundle_oracle import exact_bundle_oracle, member_potential, series_power_terms
from .campaign_compile import Population, PilotGridReuse, compile_request
from .campaign_data import audit_charges, emit_suite, export_database, suite_inputs, PRIMARY
from .campaign_io import (binding_hash, file_record, frame_write, git, publish_receipt,
                         sha, verify_receipt, write_json)
from .campaign_runtime import choose_split, stable_id


def source_audit(ctx):
    cfg=ctx.config; source=Path(cfg["source_pilot_output_root"])
    if sha(cfg["authority_zip_path"])!=cfg["authority_zip_sha256"] or sha(cfg["source_pilot_zip"])!=cfg["source_pilot_sha256"]:
        raise ValueError("Controlling authority/source ZIP mismatch")
    root=Path(cfg["repository_root"])
    if git(root,"branch","--show-current")!=cfg["branch"]:raise ValueError("Wrong live branch")
    git(root,"merge-base","--is-ancestor",cfg["required_ancestor"],"HEAD")
    bindings=[];hash_cache={}
    with zipfile.ZipFile(cfg["source_pilot_zip"]) as z:
        manifest=pd.read_csv(io.BytesIO(z.read("external_artifact_index/full_output_MANIFEST.csv")))
        for ordinal,r in enumerate(manifest.itertuples(index=False)):
            p=(source/r.relative_path).resolve()
            if not p.is_relative_to(source.resolve()) or not p.is_file():raise ValueError("Missing pilot source: "+str(p))
            if p.stat().st_size!=int(r.size_bytes) or sha(p)!=r.sha256:raise ValueError("Changed pilot source: "+str(p))
            bindings.append({"role":"PILOT_SOURCE","absolute_path":str(p),"size_bytes":p.stat().st_size,
                             "mtime_ns":p.stat().st_mtime_ns,"sha256":r.sha256})
            hash_cache[str(p)]=r.sha256
            if ordinal%32==0:ctx.mark(f"authenticate source {ordinal}/{len(manifest)}")
        for name in ("experiments.py","bundle_oracle.py","runtime_functions.py","scc_reachability.py","topology_reduction.py"):
            candidates=[n for n in z.namelist() if n.endswith("partC_rev1_src/phase5/"+name) and n.startswith("code_snapshot/")]
            if len(candidates)!=1 or hashlib.sha256(z.read(candidates[0])).hexdigest()!=sha(root/"partC_rev1_src/phase5"/name):
                raise ValueError("Validated source helper changed: "+name)
    # Authenticate each accepted scientific input, including complete suite directories.
    external=pd.read_csv(source/"manifests/external_artifact_bindings.csv")
    for r in external.itertuples(index=False):
        p=Path(r.absolute_path);ctx.mark("authenticate accepted "+str(r.role))
        if p.is_dir():
            digest_object=hashlib.sha256()
            for f in sorted(v for v in p.rglob("*") if v.is_file()):
                value=hash_cache.get(str(f))
                if value is None:value=sha(f);hash_cache[str(f)]=value
                digest_object.update(f.relative_to(p).as_posix().encode()+b"\0"+str(f.stat().st_size).encode()+b"\0"+value.encode()+b"\n")
            digest=digest_object.hexdigest()
        else:
            digest=hash_cache.get(str(p))
            if digest is None:digest=sha(p);hash_cache[str(p)]=digest
        if digest!=str(r.sha256_or_directory_digest):raise ValueError("Accepted external binding changed: "+str(p))
        if p.is_file():
            bindings.append({"role":r.role,"absolute_path":str(p),"size_bytes":p.stat().st_size,
                             "mtime_ns":p.stat().st_mtime_ns,"sha256":digest})
        else:
            for f in sorted(p.rglob("*")):
                if f.is_file():bindings.append({"role":r.role,"absolute_path":str(f),"size_bytes":f.stat().st_size,
                                              "mtime_ns":f.stat().st_mtime_ns,"sha256":hash_cache[str(f)]})
    frame_write(ctx.output/"validation/source_bindings.parquet",pd.DataFrame(bindings).drop_duplicates("absolute_path"))
    write_json(ctx.output/"reports/source_and_reuse_audit.json",{"status":"PASS","pilot_manifest_entries":len(manifest),
        "accepted_external_bindings":len(external),"source_helpers_byte_identical":True,
        "reuse_policy":"Full registries, charges and SCC reused; training grids require exact source/algorithm/low/domain/grid/weight identity; every fit receives new independent probes",
        "old_pilot_relaunched":False,"old_products_edited":False,"source_head":cfg["required_ancestor"]})


def freeze_preflight(pop):
    pools={"EXACT_MACRO":[],"HETEROGENEOUS_MACRO":[]}
    for k,e in pop.experiments.items():
        pools["EXACT_MACRO" if series_power_terms(e.primitive_bundles) is not None else "HETEROGENEOUS_MACRO"].append(k)
    selected={};forced={}
    def rank(k):return hashlib.sha256(("P5_CAMPAIGN_C0\0"+k).encode()).hexdigest()
    for name,keys in pools.items():
        stable=sorted(keys,key=rank)
        mandatory=set()
        if keys:
            mandatory.add(max(keys,key=lambda k:(pop.experiments[k].chain_length,rank(k))))
            mandatory.add(max(keys,key=lambda k:(sum(len(np.unique(s.free_time_s)) for s in pop.experiments[k].primitive_bundles),rank(k))))
            zero=[k for k in stable if pop.experiments[k].low_flow==0]
            if zero:mandatory.add(zero[0])
        forced[name]=mandatory;selected[name]=set(stable[:384])|mandatory
    # Include actual counterpart request directions before freezing the sample.
    all_selected=set.union(*selected.values())
    active=pop.chains.loc[~pop.chains.protected_or_blocked]
    groups=set(active.loc[active.calibration_request_key.isin(all_selected)&active.reciprocal_group_id.ne(""),"reciprocal_group_id"])
    # Only force counterparts of the declared extreme cases, then fill paired groups
    # in deterministic rank while staying within both class caps.
    extremes=set.union(*forced.values())
    pairs=active.loc[active.reciprocal_group_id.ne("")].groupby(["suite_short_id","reciprocal_group_id"])
    poolkind={k:n for n,ks in pools.items() for k in ks}
    for _,g in pairs:
        ks=set(g.calibration_request_key)
        if ks&extremes:
            for k in ks:forced[poolkind[k]].add(k)
    for name,keys in pools.items():
        mandatory=forced[name]
        if len(mandatory)>384:raise ValueError("Preflight extreme/counterpart set exceeds declared cap")
        selected[name]=set(mandatory)
    # Fill with complete direction pairs, choosing each object's source occurrence.
    groupkeys={key:set(g.calibration_request_key) for key,g in pairs}
    for k in sorted(poolkind,key=rank):
        ex=pop.experiments[k];row=pop.chain_rows[ex.occurrence_id]
        pair=groupkeys.get((row.suite_short_id,row.reciprocal_group_id),{k}) if row.reciprocal_group_id else {k}
        if all(len(selected[name]|{p for p in pair if poolkind[p]==name})<=384 for name in pools):
            for p in pair:selected[poolkind[p]].add(p)
    # Counterpart coverage is explicit, with the full deterministic partner list.
    used=set()
    for r in active.itertuples(index=False):used.update((r.suite_short_id,p) for p in json.loads(r.primitive_bundle_ids_json))
    controls=pop.bundles.loc[~pop.bundles.exact_homogeneous_closed_form].copy()
    controls=controls.loc[[(s,b) not in used for s,b in zip(controls.suite_short_id,controls.bundle_id)]]
    controls=controls.drop_duplicates("canonical_primitive_key")
    controls["rank"]=controls.canonical_primitive_key.map(rank)
    controlrows=controls.sort_values("rank").head(256)
    mandatory_controls=[]
    if len(controls):
        mandatory_controls.append(controls.sort_values(["positive_activation_threshold_count","rank"],ascending=[False,True]).iloc[0])
        zero=controls.loc[controls.low_current_flow.eq(0)]
        if len(zero):mandatory_controls.append(zero.sort_values("rank").iloc[0])
    if mandatory_controls:
        controlrows=pd.concat([pd.DataFrame(mandatory_controls),controlrows]).drop_duplicates("canonical_primitive_key").head(256)
    rows=[]
    for name,keys in selected.items():
        for k in sorted(keys,key=rank):
            ex=pop.experiments[k]
            rows.append({"object_id":k,"stratum":name,"suite_id":ex.suite_id,"occurrence_id":ex.occurrence_id,
                "selection_reason":"DECLARED_EXTREME_OR_COUNTERPART" if k in forced[name] else "STABLE_SHA256_RANK","rank":rank(k)})
    for r in controlrows.itertuples(index=False):
        rows.append({"object_id":r.canonical_primitive_key,"stratum":"EXPANDED_PRIMITIVE","suite_id":r.suite_short_id,
            "occurrence_id":r.bundle_id,"selection_reason":"STABLE_SHA256_PLUS_EXTREMES","rank":r.rank})
    return pd.DataFrame(rows)


def compile_shards(ctx,pop,experiments,namespace,reuse,cache):
    keys=sorted(k for k in experiments if k not in cache)
    receipts=[]
    for lo in range(0,len(keys),500):
        chunk=keys[lo:lo+500]
        token=binding_hash(chunk)[:16]
        base=ctx.output/"compilation"/namespace/f"shard_{lo//500:04d}_{token}"
        receipt=base/"receipt.json"
        dep=binding_hash({"campaign_dependency":ctx.dependency_hash,"keys":chunk,"namespace":namespace})
        if receipt.exists():
            verify_receipt(ctx.output,receipt,dep)
            for r in pd.read_parquet(base/"models.parquet").to_dict("records"):cache[r["request_id"]]=r
            receipts.append(receipt);continue
        started=time.time();result=[];probes=[];certs=[]
        for ordinal,k in enumerate(chunk):
            if time.time()-started>ctx.config["resources"]["initial_shard_limit_seconds"]:
                raise TimeoutError("Measured shard exceeded initial 90-minute bound; progress and ETA diagnosis required")
            row,p,c=compile_request(experiments[k],reuse)
            result.append(row);probes.extend(p);certs.extend(c);cache[k]=row
            if ordinal%32==0:ctx.mark(f"{namespace} shard {lo//500} object {ordinal+1}/{len(chunk)}")
        paths=[]
        for name,frame in (("models",pd.DataFrame(result)),("probes",pd.DataFrame(probes)),("certificates",pd.DataFrame(certs))):
            path=base/(name+".parquet");frame_write(path,frame);paths.append(path)
        publish_receipt(ctx.output,receipt,namespace,dep,paths,started,object_count=len(chunk))
        receipts.append(receipt)
        ctx.mark(f"{namespace} shard {lo//500} committed {lo+len(chunk)}/{len(keys)}")
    return receipts


def compiled_cache(output):
    result={}
    for p in sorted((output/"compilation").rglob("receipt.json")):
        verify_receipt(output,p)
        for r in pd.read_parquet(p.parent/"models.parquet").to_dict("records"):
            if r["request_id"] in result and any(result[r["request_id"]][k]!=r[k] for k in ("k4","tau_s","status","validation_grid_hash")):
                raise ValueError("Conflicting compiled request cache")
            result[r["request_id"]]=r
    return result


def expanded_preflight(ctx,pop,sample):
    from .bundle_oracle import BundleSpec
    from .campaign_runtime import pack_runtime
    from .campaign_io import npz_write
    import sqlite3
    records=[];rows=[]
    controls=sample.loc[sample.stratum.eq("EXPANDED_PRIMITIVE")]
    for suite,selected in controls.groupby("suite_id"):
        ctx.mark("C0 observed member expansion and native IO "+suite)
        b,t,frames,low,high=suite_inputs(ctx.config,suite)
        members=frames["bundle_membership"].copy()
        members=members.loc[members.bundle_id.isin(selected.occurrence_id)].sort_values(["bundle_id","directed_arc_id"])
        members["arc_idx"]=pd.Index(b.graph.arc_idx_to_directed_arc_id.astype(str)).get_indexer(members.directed_arc_id.astype(str))
        selected_index=selected.set_index("occurrence_id")
        for primitive,g in members.groupby("bundle_id"):
            a=g.arc_idx.to_numpy(int);ids=tuple(g.directed_arc_id.astype(str));current=b.low_physical.current_flow[a]
            spec=BundleSpec(ids,b.graph.free_time[a]*60,b.low_physical.used_capacity[a],b.graph.alpha[a],b.graph.beta[a])
            q=float(current.sum());source_flow=pop.bundle_rows[(suite,primitive)].low_current_flow
            if not np.isclose(q,source_flow,rtol=1e-10,atol=1e-8):raise ValueError("C0 member CURRENT sum closure")
            exact=exact_bundle_oracle(spec,q)
            actual=float(member_potential(spec,current).sum());optimal=float(member_potential(spec,exact.member_allocation).sum())
            if not np.isclose(optimal,exact.potential_flow_seconds,rtol=1e-12,atol=1e-7) or actual<optimal-1e-7-1e-10*max(1,optimal):
                raise ArithmeticError("Explicit expansion optimal-value certificate failed")
            records.append({"primitive_expression":selected_index.loc[primitive,"object_id"],"suite_id":suite,"bundle_id":primitive,
                "member_count":len(a),"aggregate_CURRENT_flow":q,"observed_member_potential":actual,
                "locally_minimized_diagnostic_potential":optimal,"observed_minus_minimized_potential":actual-optimal,
                "current_allocation_forced_to_optimum":False,"certified":exact.certified,
                "flow_residual":exact.allocation_residual,"kkt_s":exact.kkt_residual_s,"gap_per_flow_s":exact.fw_gap_per_flow_s})
            for j,arc in enumerate(a):
                rows.append({"suite_id":suite,"primitive_id":primitive,"link_id":stable_id("p5_member_",suite,primitive,ids[j]),
                    "tail_node_id":str(g.iloc[j].tail_bacn_id),"head_node_id":str(g.iloc[j].head_bacn_id),
                    "runtime_class":"EXACT_PHYSICAL_MEMBER_BPR","beta":float(spec.beta[j]),"tau_s":float(spec.free_time_s[j]),
                    "k4":float(spec.free_time_s[j]*spec.alpha[j]/spec.capacity[j]**4),"capacity":float(spec.capacity[j]),"alpha":float(spec.alpha[j]),
                    "fixed_internal_H_s":0.,"tail_half_charge_s":2.,"head_half_charge_s":5.,"structural_fixed_s":3.,
                    "domain_Q":np.inf,"member_arc_idx":int(arc),"low_current_flow":float(current[j]),"physical_arc_id":ids[j]})
    frame=pd.DataFrame(rows);nodes=pd.DataFrame({"node_id":sorted(set(frame.tail_node_id)|set(frame.head_node_id))})
    runtime=pack_runtime(frame,nodes);q=frame.low_current_flow.to_numpy(float)
    (_,potential,_),_=runtime.evaluate(q)
    physical=frame.tau_s.to_numpy()*q+frame.k4.to_numpy()*q**5/5
    if not np.allclose(potential,physical+7*q,rtol=1e-12,atol=1e-7):raise ValueError("C0 own-flow endpoint charge multiplication")
    path=ctx.output/"preflight/observed_member_multiedges.parquet";frame_write(path,frame)
    npz_write(ctx.output/"preflight/native_runtime_arrays.npz",runtime.arrays)
    with sqlite3.connect(ctx.output/"preflight/native_multiedge_probe.sqlite") as db:
        frame.to_sql("runtime_links",db,index=False,if_exists="fail")
        db.execute("CREATE UNIQUE INDEX native_identity ON runtime_links(suite_id,link_id)")
        restored=pd.read_sql_query("SELECT link_id,low_current_flow FROM runtime_links",db)
        if not np.array_equal(restored.link_id,frame.link_id) or not np.array_equal(restored.low_current_flow,frame.low_current_flow):raise ValueError("C0 native multiedge roundtrip")
    if len(pd.read_parquet(path))!=len(frame):raise ValueError("C0 Parquet multiedge reopen")
    frame_write(ctx.output/"preflight/expanded_equivalence.parquet",pd.DataFrame(records))
    return len(records),len(frame)


def c0(ctx):
    source_audit(ctx)
    ctx.mark("C0 full existing and new tests")
    root=Path(ctx.config["repository_root"])
    tests=sorted(str(p.relative_to(root)) for p in (root/"tests").glob("test_phase5*.py"))
    tests += [str(Path(ctx.config["authority_root"])/"tests/test_expanded_member_reference.py")]
    log=ctx.output/"validation/full_tests.log";junit=ctx.output/"validation/full_tests.xml"
    command=[ctx.config["python_executable"],"-m","pytest","-q",*tests,"--junitxml",str(junit)]
    with log.open("w",encoding="utf-8") as f:
        done=subprocess.run(command,cwd=root,stdout=f,stderr=subprocess.STDOUT,text=True)
    if done.returncode:raise RuntimeError("C0 focused/full test gate failed; see "+str(log))
    write_json(ctx.output/"validation/test_command.json",{"command":command,"exit_code":done.returncode,"log_sha256":sha(log)})
    pop=Population(Path(ctx.config["source_pilot_output_root"]),ctx.config["suites"])
    sample=freeze_preflight(pop);frame_write(ctx.output/"preflight/frozen_objects.parquet",sample)
    reused=PilotGridReuse(pop.source);cache={}
    ex={k:pop.experiments[k] for k in sample.loc[sample.stratum.ne("EXPANDED_PRIMITIVE"),"object_id"]}
    receipts=compile_shards(ctx,pop,ex,"C0_preflight",reused,cache)
    control_count,member_count=expanded_preflight(ctx,pop,sample)
    allcharges=[];witnesses=[];witness_arcs=[];reports=[]
    for suite in ctx.config["suites"]:
        ctx.mark("C0 charge/physical ownership "+suite)
        charge,report,witness,arcs=audit_charges(ctx.config,suite)
        allcharges.append(charge);reports.append(report);witnesses.append(witness);witness_arcs.append(arcs)
    charges=pd.concat(allcharges,ignore_index=True)
    frame_write(ctx.output/"registries/charges.parquet",charges)
    frame_write(ctx.output/"registries/racn_ownership.parquet",pd.concat(witnesses,ignore_index=True))
    frame_write(ctx.output/"registries/racn_witness_arcs.parquet",pd.concat(witness_arcs,ignore_index=True))
    frame_write(ctx.output/"preflight/charge_top50_outliers.parquet",charges.loc[charges.model_variant.eq(PRIMARY)].sort_values(["chi_s","E_L"],ascending=False).head(50))
    frame_write(ctx.output/"preflight/charge_audit_summary.csv",pd.DataFrame(reports))
    originalpaths=pd.read_parquet(pop.source/"path_validation/path_cost_sample.parquet")
    if len(originalpaths)!=600:raise ValueError("Full 600 local witness records are unavailable")
    worst=originalpaths.sort_values("absolute_relative_cost_error",ascending=False).head(20)
    frame_write(ctx.output/"preflight/legacy_worst20_paths.parquet",worst)
    costs=pd.DataFrame(cache.values());hetero=costs.loc[~costs.is_exact]
    rates={"heterogeneous_mean_seconds":float(hetero.elapsed_seconds.mean()),"heterogeneous_p95_seconds":float(hetero.elapsed_seconds.quantile(.95)),
        "exact_mean_seconds":float(costs.loc[costs.is_exact,"elapsed_seconds"].mean())}
    counts=pop.counts();remaining=counts["heterogeneous_macro_requests"]-len(hetero)
    # Upper forecast includes a full four-child split attempt for every heterogeneous
    # request, actual C0 source IO/audit time, C3-C7 IO/verification plus uncertainty.
    science_est=remaining*max(rates["heterogeneous_mean_seconds"]*2,rates["heterogeneous_p95_seconds"])*5
    elapsed=time.time()-ctx.config["task_started_unix"]
    upper=science_est+7200
    budget=ctx.config["resources"]["base_wall_seconds"]-ctx.config["resources"]["sealing_reserve_seconds"]-elapsed
    proceed=upper<budget
    decision={"decision":"PROCEED_TECHNICALLY" if proceed else "BLOCK_TECHNICAL","technical_status":"PASS" if proceed else "FAIL",
        "universe":counts,"preflight_objects":len(sample),"strata":sample.stratum.value_counts().to_dict(),
        "stratum_deficits":{k:max(0,n-int(sample.stratum.eq(k).sum())) for k,n in [("EXACT_MACRO",384),("HETEROGENEOUS_MACRO",384),("EXPANDED_PRIMITIVE",256)]},
        "measured_rates":rates,"remaining_serial_forecast_seconds":[science_est/5+1800,upper],
        "two_worker_projection":"NOT_ADMITTED_NO_MEASURED_1.3X_THROUGHPUT; serial execution retained",
        "forecast_RSS_bytes":[4*2**30,12*2**30],"forecast_added_disk_bytes":[5*2**30,20*2**30],
        "numerical_checks":"PASS","multiedge_checks":"PASS_FULL_TESTS_PLUS_REAL_EXPANSIONS","ownership_checks":"PASS",
        "explicit_control_bundles":control_count,"observed_native_member_edges":member_count,
        "charge_outliers_reproduced":50,"worst_legacy_paths_preserved":20,"model_readiness":"MODEL_LIMITATIONS_REMAIN",
        "unresolved_model_findings":["Large fixed B charge/path errors","Unknown counterfactual R/interface state loss","No global new-flow liftability or national speedup established"],
        "automatic_continuation_authorized":True,"elapsed_whole_task_seconds":elapsed,"science_seconds_available_before_reserve":budget}
    write_json(ctx.output/"reports/preflight_decision.json",decision)
    if not proceed:raise RuntimeError("C0 measured conservative forecast does not fit protected base envelope")
    return [ctx.output/"reports/preflight_decision.json",ctx.output/"preflight/frozen_objects.parquet",ctx.output/"validation/source_bindings.parquet",
        ctx.output/"validation/full_tests.log",ctx.output/"validation/full_tests.xml",ctx.output/"registries/charges.parquet",
        ctx.output/"registries/racn_ownership.parquet",ctx.output/"registries/racn_witness_arcs.parquet",*receipts]


def c1(ctx):
    pop=Population(Path(ctx.config["source_pilot_output_root"]),ctx.config["suites"])
    cache=compiled_cache(ctx.output)
    exact={k:e for k,e in pop.experiments.items() if series_power_terms(e.primitive_bundles) is not None}
    receipts=compile_shards(ctx,pop,exact,"C1_exact",None,cache)
    frame_write(ctx.output/"registries/original_corridors.parquet",pop.chains)
    requests=pd.DataFrame([{"request_id":k,"expression_id":e.dynamic_expression_key,"low_current_flow":e.low_flow,
        "low_aon_domain_flow":e.low_aon_flow,"exact_eligible":k in exact,"source_occurrence_id":e.occurrence_id,
        "source_suite":e.suite_id} for k,e in sorted(pop.experiments.items())])
    frame_write(ctx.output/"registries/full_request_population.parquet",requests)
    write_json(ctx.output/"reports/full_population.json",pop.counts())
    return [ctx.output/"registries/original_corridors.parquet",ctx.output/"registries/full_request_population.parquet",ctx.output/"reports/full_population.json",*receipts]


def c2(ctx):
    pop=Population(Path(ctx.config["source_pilot_output_root"]),ctx.config["suites"])
    cache=compiled_cache(ctx.output);reuse=PilotGridReuse(pop.source)
    receipts=compile_shards(ctx,pop,pop.experiments,"C2_parent",reuse,cache)
    chains=pop.chains.copy();chains["atomic_group"]=np.where(chains.reciprocal_group_id.ne(""),chains.reciprocal_group_id,chains.chain_id)
    plans=[];children={};childrows={}
    for (suite,group),frame in chains.groupby(["suite_short_id","atomic_group"],sort=True):
        directions=list(frame.itertuples(index=False));parent_good=not frame.protected_or_blocked.any()
        parent_good=parent_good and all(cache[r.calibration_request_key]["status"]=="ACCEPT" for r in directions)
        reason="ALL_PARENT_DIRECTIONS_PASS" if parent_good else "PARENT_REJECTED"
        action="ACCEPT_PARENT" if parent_good else "RETAIN";cut=None;ids=[];saving=0
        if not parent_good and not frame.protected_or_blocked.any():
            payload=[]
            for r in directions:
                primitives=json.loads(r.primitive_bundle_ids_json)
                payload.append({"internal_nodes":json.loads(r.internal_bacn_ids_json),
                    "member_counts":[pop.bundle_rows[(suite,p)].member_count for p in primitives]})
            cut=choose_split(payload)
            saving=max(0,len(payload[0]["internal_nodes"])-1)
            if cut and saving>0:
                action="PENDING_ONE_SPLIT";reason="DETERMINISTIC_SINGLE_CUT"
                for r in directions:
                    primitive=json.loads(r.primitive_bundle_ids_json);internal=json.loads(r.internal_bacn_ids_json);j=internal.index(cut)+1
                    for ordinal,part,nodes,tail,head in ((0,primitive[:j],internal[:j-1],r.retained_tail_bacn_id,cut),
                        (1,primitive[j:],internal[j:],cut,r.retained_head_bacn_id)):
                        cid=stable_id("p5_split_child_",suite,r.chain_id,cut,ordinal)
                        ex=pop.experiment(suite,part,cid);children.setdefault(ex.canonical_signature,ex);ids.append(cid)
                        row=r._asdict();row.update(chain_id=cid,parent_chain_id=r.chain_id,primitive_bundle_ids_json=json.dumps(part),
                            internal_bacn_ids_json=json.dumps(nodes),primitive_link_count=len(part),internal_node_count=len(nodes),
                            retained_tail_bacn_id=tail,retained_head_bacn_id=head,calibration_request_key=ex.canonical_signature,
                            split_role="LEFT" if ordinal==0 else "RIGHT",retained_split_node=cut)
                        childrows[cid]=row
            else:reason="NO_POSITIVE_NODE_SAVING_ONE_SPLIT"
        if frame.protected_or_blocked.any():reason="SOURCE_TOPOLOGY_OR_CURRENT_SAFETY_VETO"
        plans.append({"suite_id":suite,"group_id":group,"parent_ids_json":json.dumps(frame.chain_id.tolist()),
            "action":action,"reason":reason,"split_node":cut or "","potential_nodes_removed":saving,
            "child_ids_json":json.dumps(ids),"direction_count":len(directions),"split_attempt_count":int(action=="PENDING_ONE_SPLIT")})
    receipts+=compile_shards(ctx,pop,children,"C2_children",reuse,cache)
    accepted=[];child_actions=[]
    for p in plans:
        if p["action"]=="PENDING_ONE_SPLIT":
            ids=json.loads(p["child_ids_json"])
            passed=all(cache[childrows[c]["calibration_request_key"]]["status"]=="ACCEPT" for c in ids)
            p["action"]="ACCEPT_SPLIT" if passed else "RETAIN"
            p["reason"]="ALL_CHILDREN_PASS_AND_POSITIVE_NODE_SAVING" if passed else "ONE_SPLIT_HAS_REJECTED_CHILD_RETAIN_ORIGINAL"
            for cid in ids:
                row=childrows[cid];row["model_status"]=cache[row["calibration_request_key"]]["status"];row["group_action"]=p["action"]
                child_actions.append(row)
                if passed:accepted.append(row)
        elif p["action"]=="ACCEPT_PARENT":
            for cid in json.loads(p["parent_ids_json"]):
                row=pop.chain_rows[cid]._asdict();row.update(parent_chain_id=cid,split_role="PARENT",retained_split_node="")
                accepted.append(row)
    if not accepted:raise ValueError("Zero actual reductions despite full campaign")
    models=pd.DataFrame(cache.values()).sort_values("request_id")
    paths=[]
    for filename,data in (("calibration_requests.parquet",models),("corridor_decisions.parquet",pd.DataFrame(plans)),
        ("split_decisions.parquet",pd.DataFrame(child_actions)),("accepted_macro_occurrences.parquet",pd.DataFrame(accepted)),
        ("fit_rejections.parquet",models.loc[models.status.ne("ACCEPT")])):
        path=ctx.output/"registries"/filename;frame_write(path,data);paths.append(path)
    return paths+receipts


def c3(ctx):
    pop=Population(Path(ctx.config["source_pilot_output_root"]),ctx.config["suites"])
    compiled=compiled_cache(ctx.output)
    accepted=pd.read_parquet(ctx.output/"registries/accepted_macro_occurrences.parquet")
    charges=pd.read_parquet(ctx.output/"registries/charges.parquet")
    counts=[];models=[];ownership=[];interfaces=[];paths=[]
    for suite in ctx.config["suites"]:
        counts.append(emit_suite(ctx,pop,suite,accepted,compiled,charges))
        base=ctx.output/"suites"/suite
        models.append(pd.read_parquet(base/"dynamic_models.parquet"));ownership.append(pd.read_parquet(base/"physical_link_ownership.parquet"))
        interfaces.append(json.loads((base/"regional_interface_contract.json").read_text()))
        paths.extend(p for p in base.iterdir() if p.is_file())
    model=pd.concat(models,ignore_index=True).drop_duplicates("model_id")
    frame_write(ctx.output/"registries/dynamic_models.parquet",model)
    frame_write(ctx.output/"registries/physical_link_ownership.parquet",pd.concat(ownership,ignore_index=True))
    write_json(ctx.output/"registries/regional_interface_contract.json",{"suites":interfaces,"phase6_authorized":False})
    frame_write(ctx.output/"reports/actual_topology_counts.csv",pd.DataFrame([{k:v for k,v in r.items() if k not in ("classes","incidence")} for r in counts]))
    write_json(ctx.output/"validation/complete_coverage.json",{"status":"PASS","suites":counts,"root_free_coverage":1.,"proxy_count":0,
        "emitted_links":sum(r["actual_total_links"] for r in counts),"dynamic_models":len(model)})
    return paths+[ctx.output/p for p in ("registries/dynamic_models.parquet","registries/physical_link_ownership.parquet",
        "registries/regional_interface_contract.json","reports/actual_topology_counts.csv","validation/complete_coverage.json")]


def c4(ctx):
    database=export_database(ctx)
    # Fresh-process reopen is also performed by the C6 CLI validation entrypoint.
    from .campaign_validation import validate_serialization
    result=validate_serialization(ctx.config)
    write_json(ctx.output/"validation/serialization_roundtrip.json",result)
    return [database,ctx.output/"validation/serialization_roundtrip.json"]
