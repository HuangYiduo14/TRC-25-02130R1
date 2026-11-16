"""Complete per-suite runtime products, physical ownership and observed flows."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd

from .bacn_charges import internal_exposure, reconstruct_bacn_visit_semantics
from .inputs import load_bindings, load_suite_trace_arrays
from .pilot import _suite_frames
from .scc_reachability import load_reachable_od_supply_cache
from .topology_reduction import remaining_incidence_audit
from .campaign_io import (array_schema, binding_hash, frame_write, npz_write,
                          sha, write_json)
from .campaign_runtime import pack_runtime, stable_id

PRIMARY = "LOW_MEAN_STATIC_INCIDENCE_V1"
STRUCTURAL = "STRUCTURAL_FFTT_SENSITIVITY_V1"


def load_accepted(config, suite):
    # This read-only loader requires the accepted OD cache to already exist.
    old = json.loads(Path(config["source_loader_config"]).read_text(encoding="utf-8"))
    p4 = json.loads((Path(config["repository_root"])/"partC_doc_icnn/phase_4_work/phase4_dual_accuracy_benchmark_execution_config.json").read_text())
    if not Path(p4["input_versions"]["accepted_aggregated_od_root"]).is_dir():
        raise FileNotFoundError("Accepted OD cache absent: rebuilding is unauthorized")
    if Path(old["output_root"]).resolve() != Path(config["source_pilot_output_root"]).resolve():
        raise ValueError("Source-loader output identity changed")
    return load_bindings(config["source_loader_config"], verify_suite_digests=False, suite_ids=[suite])


def suite_inputs(config,suite):
    b = load_accepted(config,suite)
    trace = b.trace_maps[0]
    frames = _suite_frames(trace.root)
    low = load_suite_trace_arrays(b.sources,suite,"low_benchmark",trace_map=trace)
    high = load_suite_trace_arrays(b.sources,suite,"high_benchmark",trace_map=trace)
    return b,trace,frames,low,high


def connector_flows(trace,frames,arrays,prefix="current__"):
    keys=arrays[prefix+"coarse_regional_pair_flow__keys"]
    values=arrays[prefix+"coarse_regional_pair_flow__values"]
    enter=np.bincount((keys>>np.uint64(32)).astype(np.int64),weights=values,minlength=trace.terminal_count)
    leave=np.bincount((keys&np.uint64(0xffffffff)).astype(np.int64),weights=values,minlength=trace.terminal_count)
    local={v:i for i,v in enumerate(trace.terminal_ids)}
    result={}
    for r in frames["terminal_connectors"].itertuples(index=False):
        if r.direction_role == "parent_to_region":
            result[str(r.link_id)]=float(enter[local[str(r.head_node_id)]])
        elif r.direction_role == "region_to_parent":
            result[str(r.link_id)]=float(leave[local[str(r.tail_node_id)]])
        else:
            raise ValueError("Unknown connector direction: "+str(r.direction_role))
    return result


def audit_charges(config,suite):
    b,t,f,low,high=suite_inputs(config,suite)
    source=Path(config["source_pilot_output_root"])
    supply=load_reachable_od_supply_cache(source/"cache/reachable_od_supply_v2.npz",
        expected_graph_identity=b.graph.graph_hash,expected_od_identity=b.od.od_hash,
        expected_node_count=len(b.graph.node_idx_to_uid),expected_row_count=len(b.od.demand))
    visits=[]
    for label,arrays in ((b.low_physical.benchmark_id,low),(b.high_physical.benchmark_id,high)):
        visits.append(reconstruct_bacn_visit_semantics(trace_map=t,bundle_summary=f["bundle_summary"],
            terminals=f["consolidated_terminals"],trace_arrays=arrays,benchmark_id=label,
            supply=supply,static_incidence_only=True))
    vL=visits[0].incidence_visit_flow.to_numpy(float)
    vH=visits[1].incidence_visit_flow.to_numpy(float)
    eL=internal_exposure(t,b.low_physical.current_flow,b.low_physical.current_cost_s)
    eH=internal_exposure(t,b.high_physical.current_flow,b.high_physical.current_cost_s)
    arc_index=pd.Index(b.graph.arc_idx_to_directed_arc_id.astype(str))
    suppressed=f["suppressed_backbone_arcs"]
    indices=arc_index.get_indexer(suppressed.directed_arc_id.astype(str))
    owners=pd.Index(t.bacn_ids).get_indexer(suppressed.bacn_id.astype(str))
    if (indices<0).any() or (owners<0).any():raise ValueError("Suppressed physical membership missing")
    direct=np.bincount(owners,weights=b.low_physical.current_flow[indices]*b.low_physical.current_cost_s[indices],minlength=t.bacn_count)
    if not np.allclose(direct,eL,rtol=1e-12,atol=1e-7):raise ValueError("Direct B exposure disagrees with physical partition")
    seed=f["structural_internal_charges"].set_index("bacn_id").loc[list(t.bacn_ids)].structural_charge_s.to_numpy(float)
    chi=np.divide(eL,vL,out=seed.copy(),where=vL>0)
    if not np.isfinite(chi).all() or (chi<0).any() or (eL<0).any():raise ValueError("Invalid unclipped E/V charge")
    previous=pd.read_parquet(source/f"suite_products/{suite}/fixed_charge_candidates.parquet").set_index("bacn_id").loc[list(t.bacn_ids)]
    for actual,column in ((chi,"recommended_charge_seconds"),(eL,"low_internal_exposure_flow_seconds"),(vL,"low_visit_flow")):
        if not np.allclose(actual,previous[column],rtol=1e-12,atol=1e-7):raise ValueError("V4 charge reproduction failed: "+column)
    rows=[]
    for variant,estimate in ((PRIMARY,chi),(STRUCTURAL,seed)):
        rows.append(pd.DataFrame({"suite_id":suite,"bacn_id":t.bacn_ids,"model_variant":variant,
            "E_L":eL,"V_L":vL,"chi_s":estimate,"structural_seed_s":seed,
            "unvisited":vL==0,"zero_denominator_positive_exposure":(vL==0)&(eL>0),
            "fallback_used":(vL==0)&(variant==PRIMARY),"E_H_evaluation_only":eH,
            "V_H_evaluation_only":vH,"high_residual_evaluation_only":eH-vH*estimate,
            "clipped":False,"high_calibrated":False,
            "charge_source":"accepted CURRENT x45 physical exposure / static half incidence"}))
    charges=pd.concat(rows,ignore_index=True)
    components=f["acnhn_nodes"].set_index("node_id").physical_weak_component_id
    charges["physical_component_id"]=charges.bacn_id.map(components)
    charges["exposure_quantile"]=pd.qcut(charges.E_L.rank(method="min"),10,labels=False,duplicates="drop")
    charges["visit_quantile"]=pd.qcut(charges.V_L.rank(method="min"),10,labels=False,duplicates="drop")
    report={"suite_id":suite,"status":"PASS","bacn_count":t.bacn_count,
        "low_exposure":float(eL.sum()),"low_static_visits":float(vL.sum()),
        "unvisited_nodes":int((vL==0).sum()),"positive_exposure_zero_denominator_nodes":int(((vL==0)&(eL>0)).sum()),
        "unrepresentable_exposure":float(eL[vL==0].sum()),"direct_exposure_max_abs_error":float(np.abs(direct-eL).max()),
        "low_weighted_charge_residual":float((vL*chi-eL).sum()),
        "zero_R_policy":"PROVISIONAL_ZERO_NO_SEPARATE_EXPOSURE","high_used_for_calibration":False}
    # Every available restricted witness gets an explicit physical partition ledger.
    wr=[]; ar=[]
    physical_node_index=pd.Index(b.graph.node_idx_to_uid.astype(str))
    for i,r in enumerate(f["racn_restricted_witnesses"].itertuples(index=False)):
        arcs=np.asarray(json.loads(r.physical_arc_indices_json),dtype=np.int64)
        categories=(t.arc_region_local[arcs]>=0).astype(int)+(t.arc_to_bundle_local[arcs]>=0)+(t.arc_internal_bacn_local[arcs]>=0)
        if not len(arcs) or np.any(categories!=1):raise ValueError("Restricted witness physical partition mismatch")
        witness=stable_id("p5_R_witness_",suite,str(r.source_node_uid),str(r.target_node_uid),i)
        endpoint_local=physical_node_index.get_indexer([str(r.source_node_uid),str(r.target_node_uid)])
        if np.any(endpoint_local<0):raise ValueError("R witness endpoint missing from physical graph")
        racn_local=t.node_to_racn_local[endpoint_local]
        counts={"REGION":int((t.arc_region_local[arcs]>=0).sum()),"INTER_B":int((t.arc_to_bundle_local[arcs]>=0).sum()),"SUPPRESSED_B":int((t.arc_internal_bacn_local[arcs]>=0).sum())}
        direct_time=float((b.graph.free_time[arcs]*60).sum())
        if not np.isclose(direct_time,float(r.restricted_fftt_s),rtol=1e-10,atol=1e-7):raise ValueError("Restricted witness time unit mismatch")
        wr.append({"suite_id":suite,"witness_id":witness,"source_node_uid":str(r.source_node_uid),
            "target_node_uid":str(r.target_node_uid),"physical_arc_count":len(arcs),
            "source_racn_id":t.racn_ids[racn_local[0]] if racn_local[0]>=0 else None,
            "target_racn_id":t.racn_ids[racn_local[1]] if racn_local[1]>=0 else None,
            "region_owned_count":counts["REGION"],"backbone_owned_count":counts["INTER_B"],"B_owned_count":counts["SUPPRESSED_B"],
            "overlap_count":0,"gap_count":0,"restricted_fftt_s":direct_time,
            "separately_observed_omitted_time_s":None,"candidate_R_charge_s":0.,
            "physical_ownership":"COMPLETE_PARTITION","interface_state_loss":"UNRESOLVED_COUNTERFACTUAL_STATE",
            "reason":"Every observed arc already has a physical cost owner; no separate transition exposure is identified"})
        for position,a in enumerate(arcs):
            role="REGION" if t.arc_region_local[a]>=0 else "INTER_B" if t.arc_to_bundle_local[a]>=0 else "SUPPRESSED_B"
            ar.append({"suite_id":suite,"witness_id":witness,"position":position,"arc_idx":int(a),
                "physical_arc_id":str(b.graph.arc_idx_to_directed_arc_id[a]),"ownership_role":role})
    return charges,report,pd.DataFrame(wr),pd.DataFrame(ar)


def model_record(model_id,expression,runtime_class,tau,k,domain=np.inf,**extra):
    return {"model_id":model_id,"expression_id":expression,"runtime_class":runtime_class,
        "tau_s":float(tau),"k4":float(k),"beta":4.,"domain_Q":float(domain),
        "is_exact":runtime_class!="FIT_POWER4","oracle_version":"V4_ACTIVATION_RELATIVE_CERTIFIED_OFFLINE",
        "parameter_digest":stable_id("",float(tau),float(k),4.),**extra}


def emit_suite(ctx,pop,suite,accepted,compiled,charges,*,baseline=False):
    b,t,f,low,high=suite_inputs(ctx.config,suite)
    graph=b.graph
    charge=charges.loc[charges.suite_id.eq(suite)]
    primary=charge.loc[charge.model_variant.eq(PRIMARY)].set_index("bacn_id").chi_s.to_dict()
    structural=charge.loc[charge.model_variant.eq(STRUCTURAL)].set_index("bacn_id").chi_s.to_dict()
    original=f["acnhn_static_links"].set_index("link_id")
    protection=pd.read_parquet(Path(ctx.config["source_pilot_output_root"])/f"suite_products/{suite}/bacn_protection_census.parquet")
    chosen=accepted.loc[accepted.suite_short_id.eq(suite)].copy()
    if baseline:chosen=chosen.iloc[:0]
    chosen["protected_or_blocked"]=False
    _,incidence=remaining_incidence_audit(suite,chosen,f["acnhn_static_links"],f["bundle_summary"],
        protection.loc[protection.protection_reasons_json.str.contains("OD_OR_CENTROID_ROLE"),"bacn_id"])
    if incidence["status"]!="PASS":raise ValueError("Final static incidence does not close")
    removed=set();owner={};sequence=[];model_rows={};links=[]
    low_reference=[];high_reference=[];aon_reference=[]
    def add(link_id,tail,head,role,model,H=0.,structuralH=0.,primitive="",member="",arc=-1,flow=(0.,0.,0.),corridor=""):
        if tail==head:raise ValueError("Emitted self endpoint")
        model_rows[model["model_id"]]=model
        links.append({"suite_id":suite,"link_id":link_id,"tail_node_id":tail,"head_node_id":head,
            "role":role,"dynamic_model_id":model["model_id"],"runtime_class":model["runtime_class"],
            "fixed_internal_H_s":H,"tail_half_charge_s":primary.get(tail,0.)/2.,"head_half_charge_s":primary.get(head,0.)/2.,
            "structural_fixed_s":structuralH+(structural.get(tail,0.)+structural.get(head,0.))/2.,
            "primitive_id":primitive,"member_arc_id":member,"member_arc_idx":arc,
            "domain_id":model["model_id"]+":domain","model_variant":PRIMARY,"provenance":corridor or primitive or "ACCEPTED_TERMINAL_CONNECTOR",
            "tau_s":model["tau_s"],"k4":model["k4"],"beta":model["beta"],"domain_Q":model["domain_Q"]})
        low_reference.append(flow[0]);high_reference.append(flow[1]);aon_reference.append(flow[2])
    for r in chosen.itertuples(index=False):
        ids=json.loads(r.primitive_bundle_ids_json);internal=json.loads(r.internal_bacn_ids_json)
        removed.update(internal);c=compiled[r.calibration_request_key]
        if c["status"]!="ACCEPT":raise ValueError("Rejected model reached overlay")
        m=model_record(c["model_id"],c["expression_id"],c["runtime_class"],c["tau_s"],c["k4"],c["domain_Q"],calibration_request_id=c["request_id"])
        ex=pop.experiment(suite,ids,r.chain_id)
        add(r.chain_id,r.retained_tail_bacn_id,r.retained_head_bacn_id,"MACRO",m,
            sum(primary[n] for n in internal),sum(structural[n] for n in internal),
            flow=(ex.low_flow,ex.high_flow,ex.low_aon_flow),corridor=r.chain_id)
        for j,p in enumerate(ids):
            if p in owner:raise ValueError("Duplicate inter-B owner")
            owner[p]=(r.chain_id,j)
            sequence.append({"suite_id":suite,"corridor_id":r.chain_id,"group_id":r.reciprocal_group_id,
                "sequence_position":j,"primitive_id":p,"internal_node_after":internal[j] if j<len(internal) else "",
                "split_role":getattr(r,"split_role","PARENT"),"model_id":m["model_id"]})
    if removed & set(protection.loc[protection.protected,"bacn_id"]):raise ValueError("Protected B node removed")
    bundle_frame=pop.by_suite[suite]
    bundle_flow_low=dict(zip(t.bundle_ids,low["current__bundle_flow"]))
    bundle_flow_high=dict(zip(t.bundle_ids,high["current__bundle_flow"]))
    bundle_flow_aon=dict(zip(t.bundle_ids,low["aon__bundle_flow"]))
    membership=f["bundle_membership"].copy()
    membership["arc_idx"]=pd.Index(graph.arc_idx_to_directed_arc_id.astype(str)).get_indexer(membership.directed_arc_id.astype(str))
    if (membership.arc_idx<0).any():raise ValueError("Unmapped physical member")
    groups=membership.groupby("bundle_id",sort=False).indices
    exact_primitive=expanded=0
    for ordinal,r in enumerate(bundle_frame.itertuples()):
        primitive=str(r.Index)
        if primitive in owner:continue
        st=original.loc[primitive];tail=str(st.tail_node_id);head=str(st.head_node_id)
        key=str(r.canonical_primitive_key);exact=pop.exact[key]
        if exact is not None:
            tau,terms=exact;k=sum(v for _,v in terms)
            mid=stable_id("p5_exact_primitive_",key)
            m=model_record(mid,key,"EXACT_PRIMITIVE_BPR",tau,k)
            add(primitive,tail,head,"EXACT_PRIMITIVE",m,primitive=primitive,
                flow=(bundle_flow_low[primitive],bundle_flow_high[primitive],bundle_flow_aon[primitive]))
            owner[primitive]=(primitive,0);exact_primitive+=1
        else:
            rows=membership.iloc[groups[primitive]]
            for member in rows.itertuples(index=False):
                a=int(member.arc_idx);arcid=str(member.directed_arc_id)
                tau=float(graph.free_time[a]*60.);cap=float(b.low_physical.used_capacity[a]);alpha=float(graph.alpha[a]);beta=float(graph.beta[a])
                if beta!=4.:raise ValueError("Unexpected physical beta")
                mid=stable_id("p5_physical_model_",arcid,tau,cap,alpha,beta)
                m=model_record(mid,stable_id("p5_member_expression_",tau,cap,alpha,beta),"EXACT_PHYSICAL_MEMBER_BPR",tau,tau*alpha/cap**4,
                    t0_s=tau,capacity=cap,alpha=alpha,physical_beta=beta)
                lid=stable_id("p5_member_",suite,primitive,arcid)
                add(lid,tail,head,"PHYSICAL_MEMBER",m,primitive=primitive,member=arcid,arc=a,
                    flow=(b.low_physical.current_flow[a],b.high_physical.current_flow[a],b.low_physical.aon_flow[a]))
            owner[primitive]=("EXPANDED_PHYSICAL_MEMBERS",0);expanded+=1
        if ordinal%2048==0:ctx.mark(f"{suite} {'baseline' if baseline else 'reduced'} primitive {ordinal}/{len(bundle_frame)}")
    for arrays,state in ((low,b.low_physical),(high,b.high_physical)):
        actual=np.bincount(t.arc_to_bundle_local[t.arc_to_bundle_local>=0],weights=state.current_flow[t.arc_to_bundle_local>=0],minlength=t.bundle_count)
        if not np.allclose(actual,arrays["current__bundle_flow"],rtol=1e-10,atol=1e-8):raise ValueError("CURRENT member-to-bundle closure failed")
    lf=connector_flows(t,f,low);hf=connector_flows(t,f,high);af=connector_flows(t,f,low,"aon__")
    for r in f["terminal_connectors"].itertuples(index=False):
        tau=float(r.direct_base_fftt_s)
        mid=stable_id("p5_connector_model_",tau)
        add(str(r.link_id),str(r.tail_node_id),str(r.head_node_id),"TERMINAL_CONNECTOR",
            model_record(mid,mid,"STATIC_CONNECTOR",tau,0.),flow=(lf[str(r.link_id)],hf[str(r.link_id)],af[str(r.link_id)]))
    frame=pd.DataFrame(links)
    frame["low_current_flow"]=low_reference;frame["high_current_flow"]=high_reference;frame["low_aon_domain_flow"]=aon_reference
    frame=frame.sort_values("link_id",kind="stable").reset_index(drop=True)
    nodes=f["acnhn_nodes"].copy();nodes["retained"]=~nodes.node_id.isin(removed)
    nodes["suite_id"]=suite;nodes["node_kind"]=nodes.node_type
    nodes["original_parent_id"]=nodes.parent_object_id.where(nodes.parent_object_id.notna(),nodes.node_id)
    nodes["protected_roles"]=nodes.node_id.map(protection.set_index("bacn_id").protection_reasons_json).fillna("UNCHANGED_NON_B_INTERFACE")
    nodes["membership_fingerprint"]=t.directory_digest
    retained=nodes.loc[nodes.retained].sort_values("node_id").reset_index(drop=True)
    runtime=pack_runtime(frame,retained)
    frame["tail_local_idx"]=runtime.arrays["tail"];frame["head_local_idx"]=runtime.arrays["head"]
    base=ctx.output/("baseline" if baseline else "suites")/suite
    frame_write(base/"runtime_links.parquet",frame)
    frame_write(base/"suite_nodes.parquet",nodes)
    frame_write(base/"retained_nodes.parquet",retained)
    frame_write(base/"dynamic_models.parquet",pd.DataFrame(model_rows.values()))
    frame_write(base/"macro_sequences.parquet",pd.DataFrame(sequence,columns=["suite_id","corridor_id","group_id","sequence_position","primitive_id","internal_node_after","split_role","model_id"]))
    npz_write(base/"runtime_arrays.npz",runtime.arrays)
    write_json(base/"runtime_arrays.schema.json",array_schema(base/"runtime_arrays.npz",runtime.arrays,
        binding_hash(frame.link_id.tolist()),binding_hash(retained.node_id.tolist())))
    refs=[]
    for label,col,benchmark,role in (("low_current","low_current_flow",b.low_physical.benchmark_id,"CURRENT"),
        ("high_current","high_current_flow",b.high_physical.benchmark_id,"CURRENT"),("low_aon_domain","low_aon_domain_flow",b.low_physical.benchmark_id,"AON_DOMAIN_EVIDENCE_ONLY")):
        q=frame[col].to_numpy(float);npz_write(base/f"reference_{label}.npz",{"flow":q})
        refs.append(pd.DataFrame({"suite_id":suite,"link_id":frame.link_id,"source_id":benchmark,"state_role":role,
            "flow":q,"physical_member_id":frame.member_arc_id,
            "projection_rule":np.select([frame.role.eq("PHYSICAL_MEMBER"),frame.role.eq("MACRO"),frame.role.eq("TERMINAL_CONNECTOR")],
                ["OBSERVED_PHYSICAL_MEMBER","COMMON_CHAIN_AGGREGATE" if role=="CURRENT" else "MAX_LOW_AON_DOMAIN_EVIDENCE","DIRECTIONAL_TERMINAL_INCIDENCE"],default="OBSERVED_BUNDLE_AGGREGATE"),
            "oracle_disaggregated":False,"observed_state":(frame.role.ne("MACRO") | (role=="CURRENT"))}))
    frame_write(base/"reference_mapping_ledger.parquet",pd.concat(refs,ignore_index=True))
    if not baseline:
        n=len(graph.tail);kind=np.full(n,"",dtype=object);ownerid=np.full(n,"",dtype=object);final=np.full(n,"",dtype=object);seq=np.full(n,-1,dtype=np.int32)
        for mask,role,ids,index in ((t.arc_region_local>=0,"REGION",t.region_ids,t.arc_region_local),(t.arc_internal_bacn_local>=0,"SUPPRESSED_B",t.bacn_ids,t.arc_internal_bacn_local)):
            kind[mask]=role;ownerid[mask]=np.asarray(ids,dtype=object)[index[mask]]
        mask=t.arc_to_bundle_local>=0;idx=t.arc_to_bundle_local[mask]
        bundleids=np.asarray(t.bundle_ids,dtype=object)[idx];kind[mask]="INTER_B";ownerid[mask]=bundleids
        final[mask]=[owner[v][0] for v in bundleids];seq[mask]=[owner[v][1] for v in bundleids]
        expandedlinks=frame.loc[frame.role.eq("PHYSICAL_MEMBER")]
        final[expandedlinks.member_arc_idx.to_numpy(int)]=expandedlinks.link_id.to_numpy()
        if np.any(kind=="") or np.any(final[mask]=="EXPANDED_PHYSICAL_MEMBERS") or np.any(final[mask]==""):
            raise ValueError("Incomplete physical ownership")
        paramhash=hashlib.sha256(np.column_stack([graph.free_time*60,b.low_physical.used_capacity,graph.alpha,graph.beta]).tobytes()).hexdigest()
        ledger=pd.DataFrame({"suite_id":suite,"physical_arc_id":graph.arc_idx_to_directed_arc_id.astype(str),
            "physical_arc_idx":np.arange(n),"ownership_role":kind,"original_owner_id":ownerid,"final_link_id":final,
            "sequence_position":seq,"original_parameter_payload_sha256":paramhash,"owner_count":1})
        frame_write(base/"physical_link_ownership.parquet",ledger)
        interface_files=[p for p in t.root.glob("*.parquet") if any(word in p.stem for word in ("terminal","regional_pair","support","fiber","centroid","racn"))]
        write_json(base/"regional_interface_contract.json",{"suite_id":suite,"phase6_authorized":False,
            "conceptual_regional_pair_count":t.conceptual_regional_pair_count,"terminal_count":t.terminal_count,
            "regional_source_target_roles_unchanged":True,"source_directory_digest":t.directory_digest,
            "files":[{"path":str(p),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(interface_files)]})
    counts={"suite_id":suite,"baseline":baseline,"original_B_nodes":t.bacn_count,"retained_B_nodes":t.bacn_count-len(removed),
        "original_nodes":len(nodes),"retained_nodes":len(retained),"original_bundles":t.bundle_count,
        "actual_backbone_links":int(frame.role.ne("TERMINAL_CONNECTOR").sum()),"terminal_connectors":len(f["terminal_connectors"]),
        "actual_total_links":len(frame),"accepted_macro_links":len(chosen),"retained_exact_primitives":exact_primitive,
        "expanded_primitive_bundles":expanded,"physical_member_links":int(frame.role.eq("PHYSICAL_MEMBER").sum()),
        "physical_interB_members":len(membership),"zero_current_member_links":int((frame.role.eq("PHYSICAL_MEMBER")&frame.low_current_flow.eq(0)).sum()),
        "root_free_coverage":1.,"proxy_count":0,"classes":frame.runtime_class.value_counts().to_dict(),"incidence":incidence}
    write_json(base/"counts.json",counts)
    return counts


def export_database(ctx):
    dest=ctx.output/"database/phase5_compilation_core.sqlite"
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError("Never overwrite a campaign database")
    registry=ctx.output/"registries"
    table_files={"dynamic_models":"dynamic_models.parquet","calibration_requests":"calibration_requests.parquet",
        "original_corridors":"original_corridors.parquet",
        "corridor_decisions":"corridor_decisions.parquet","split_decisions":"split_decisions.parquet",
        "fit_rejections":"fit_rejections.parquet","physical_link_ownership":"physical_link_ownership.parquet",
        "bacn_charges":"charges.parquet","racn_ownership":"racn_ownership.parquet"}
    with sqlite3.connect(dest) as db:
        db.execute("PRAGMA foreign_keys=ON")
        for table,file in table_files.items():
            ctx.mark("SQLite "+table)
            frame=pd.read_parquet(registry/file)
            frame.to_sql(table,db,index=False,if_exists="fail",chunksize=10000)
        for i,suite in enumerate(ctx.config["suites"]):
            base=ctx.output/"suites"/suite
            for table,file in (("runtime_links","runtime_links.parquet"),("suite_nodes","suite_nodes.parquet"),
                               ("reference_flows","reference_mapping_ledger.parquet"),("macro_sequences","macro_sequences.parquet")):
                pd.read_parquet(base/file).to_sql(table,db,index=False,if_exists="fail" if i==0 else "append",chunksize=10000)
        for table,columns in (("runtime_links","suite_id,link_id"),("suite_nodes","suite_short_id,node_id"),
                              ("dynamic_models","model_id"),("physical_link_ownership","suite_id,physical_arc_id"),
                              ("bacn_charges","suite_id,bacn_id,model_variant")):
            db.execute(f"CREATE UNIQUE INDEX idx_{table}_id ON {table} ({columns})")
        db.commit()
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok":raise ValueError("SQLite integrity failed")
        missing=db.execute("SELECT COUNT(*) FROM runtime_links l LEFT JOIN dynamic_models m ON l.dynamic_model_id=m.model_id WHERE m.model_id IS NULL").fetchone()[0]
        if missing:raise ValueError("Missing dynamic model foreign reference")
        for endpoint in ("tail_node_id","head_node_id"):
            if db.execute(f"SELECT COUNT(*) FROM runtime_links l LEFT JOIN suite_nodes n ON l.suite_id=n.suite_short_id AND l.{endpoint}=n.node_id AND n.retained=1 WHERE n.node_id IS NULL").fetchone()[0]:
                raise ValueError("Missing retained node foreign reference")
    return dest
