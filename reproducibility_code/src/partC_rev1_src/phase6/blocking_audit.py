"""Read-only diagnosis of a mandatory LOW comparison defect; no new labels."""
from pathlib import Path
import time
import numpy as np
import pandas as pd

from partC_rev1_src.phase3a.regional_pairs import build_region_support,LazyRegionalPairOracle
from .common import (read_json,read_npz,write_json,write_npz,write_frame,sha,Monitor,event,input_hash)
from .registry import region_dir
from .storage import Store

FAILED_REGION="partC_p2_region_9567843a01be0d75fd9b"


def original_oracle_certificate(c):
    root=Path(c["output_root"]); rd=region_dir(c,FAILED_REGION)
    d=read_json(rd/"descriptor.json"); a=read_npz(rd/"local_graph.npz")
    s=Store(rd,d,a); low=read_npz(rd/"support_map.npz")["LOW_ACTIVE"]
    ih=input_hash(d["model_hash"],d["support_hash"],low); label,receipt=s.get(ih,recertify=True)
    fm=read_npz(rd/"fine_members.npz"); pairs=pd.read_parquet(rd/"active_pairs.parquet")
    terms=pd.read_parquet(root/"registry/coarse_terminals.parquet")
    terms=terms.loc[terms.region_id.eq(FAILED_REGION)]
    terminal_names=terms.set_index("terminal_index").terminal_id.to_dict()
    members={terminal_names[int(k)]:tuple(int(v) for v in a["global_node_index"][fm["local_nodes"][fm["offsets"][j]:fm["offsets"][j+1]]]) for j,k in enumerate(fm["terminal_indices"])}
    cost=a["t0"]*(1+a["alpha"]*(label.z/a["capacity"])**a["beta"])
    U=float(np.sum(a["t0"]*label.z*(1+a["alpha"]/(a["beta"]+1)*(label.z/a["capacity"])**a["beta"])))
    T=float(label.z@cost)
    support=build_region_support(region_id=FAILED_REGION,tail=a["global_node_index"][a["tail"]],
        head=a["global_node_index"][a["head"]],fftt_s=cost,arc_ids=a["physical_arc_id"],region_arc_mask=np.ones(len(cost),bool))
    oracle=LazyRegionalPairOracle(suite_id="s03_diagnostic",candidate_id="geo_gc_1000m",support=support,
        terminals=terms,fine_members_global=members,physical_arc_ids=a["physical_arc_id"],
        centroid_global_nodes=frozenset(int(v) for v in a["global_node_index"][a["blocked_through"]]),
        cost_version="independent_saved_current_cost")
    bysource={source:{p.target_terminal_id:p for p in oracle.query(source)} for source in sorted(set(pairs.source_terminal_id))}
    tau=np.array([bysource[r.source_terminal_id][r.target_terminal_id].fftt_s for r in pairs.itertuples()])
    if not np.allclose(tau,label.tau,rtol=2e-12,atol=1e-8): raise ValueError("Independent original-oracle pricing differs")
    S=float(tau@low); lower=U-T+S
    mapped=read_npz(rd/"LOW_mapped_physical.npz")["z"]
    mappedU=float(np.sum(a["t0"]*mapped*(1+a["alpha"]/(a["beta"]+1)*(mapped/a["capacity"])**a["beta"])))
    allowance=3e-10*max(1.,abs(T),abs(S),abs(mappedU))
    oracle.centroid_global_nodes=frozenset(); oracle.cache.clear()
    unrestricted={source:{p.target_terminal_id:p for p in oracle.query(source)} for source in sorted(set(pairs.source_terminal_id))}
    comparison=[]
    for j,row in enumerate(pairs.itertuples()):
        route=unrestricted[row.source_terminal_id][row.target_terminal_id]
        transit=[int(a["head"][k]) for k in route.arc_indices[:-1] if a["blocked_through"][a["head"][k]]]
        if route.fftt_s+1e-8<tau[j]:
            comparison.append(dict(pair_id=row.pair_id,source_terminal_id=row.source_terminal_id,target_terminal_id=row.target_terminal_id,
                LOW_CURRENT=row.LOW_CURRENT,restricted_cost=float(tau[j]),unrestricted_diagnostic_cost=float(route.fftt_s),
                unrestricted_arc_ids=list(route.arc_ids),prohibited_transit_node_uids=[str(a["physical_node_id"][v]) for v in transit],
                diagnostic_path_not_added_to_model=True))
    out=root/"validation/mandatory_hierarchy_defect"
    write_npz(out/"independent_certificate_arrays.npz",native_input=low,certified_z=label.z,
              independent_cost=cost,independent_tau=tau,native_mapped_z=mapped)
    write_json(out/"unrestricted_pricing_witnesses.json",comparison)
    record=dict(status="CONFIRMED_MANDATORY_HIERARCHY_DEFECT",region_id=FAILED_REGION,
        model_hash=d["model_hash"],support_hash=d["support_hash"],native_LOW_input_hash=ih,
        independent_pricing="ORIGINAL_PHASE3_LAZY_REGIONAL_PAIR_ORACLE",original_pricing_code_sha256=sha(Path(__file__).parents[1]/"phase3a/regional_pairs.py"),
        label_receipt_sha256=sha(s.root/"receipts"/(ih+".json")),independent_U=U,independent_T=T,independent_S=S,
        independent_lower_bound=lower,mapped_physical_potential=mappedU,
        lower_exceeds_mapped_by=lower-mappedU,roundoff_allowance=allowance,
        outside_LOW_exposure=d["outside_LOW_exposure"],centroid_restriction_preserved=True,
        lower_bound_independently_confirmed=lower>mappedU+allowance,
        cheaper_unrestricted_diagnostic_paths=len(comparison),new_labels_generated=0,
        original_model_semantics_not_changed=True)
    write_json(out/"INDEPENDENT_CERTIFICATE.json",record)
    if not record["lower_bound_independently_confirmed"]: raise ValueError("Recorded failure did not reproduce")
    return record


def population_source_flow_audit(c):
    root=Path(c["output_root"]); regions=pd.read_parquet(root/"registry/regions.parquet")
    balances=[]; centroids=[]
    for d in regions.to_dict("records"):
        rd=region_dir(c,d["region_id"]); a=read_npz(rd/"local_graph.npz"); z=read_npz(rd/"LOW_mapped_physical.npz")["z"]
        fine=pd.read_parquet(rd/"LOW_native_fine_trace.parquet"); n=len(a["global_node_index"])
        outgoing=np.bincount(a["tail"],weights=z,minlength=n); incoming=np.bincount(a["head"],weights=z,minlength=n)
        src=np.searchsorted(a["global_node_index"],fine.source_global_node.to_numpy(np.int64))
        dst=np.searchsorted(a["global_node_index"],fine.target_global_node.to_numpy(np.int64))
        origin=np.bincount(src,weights=fine.amount,minlength=n); destination=np.bincount(dst,weights=fine.amount,minlength=n)
        error=float(np.max(np.abs(outgoing-incoming-origin+destination),initial=0.))
        allowance=1e-8+1e-11*max(float(outgoing.max(initial=0.)),float(incoming.max(initial=0.)))
        balances.append(dict(region_id=d["region_id"],m_active=d["m_active"],max_node_balance_error=error,allowance=allowance,passed=error<=allowance))
        for node in np.flatnonzero(a["blocked_through"]):
            excess_out=float(outgoing[node]-origin[node]); excess_in=float(incoming[node]-destination[node])
            centroids.append(dict(region_id=d["region_id"],m_active=d["m_active"],physical_node_id=str(a["physical_node_id"][node]),
                local_node_index=int(node),native_outgoing=float(outgoing[node]),native_incoming=float(incoming[node]),
                fine_source_amount=float(origin[node]),fine_target_amount=float(destination[node]),
                excess_outgoing=excess_out,excess_incoming=excess_in,roundoff_allowance=allowance,
                positive_nonendpoint_transit=min(excess_out,excess_in)>allowance))
    b=pd.DataFrame(balances); cen=pd.DataFrame(centroids)
    out=root/"validation/mandatory_hierarchy_defect"
    write_frame(out/"all_region_native_node_balances.parquet",b)
    write_frame(out/"centroid_flow_and_endpoint_ledger.parquet",cen)
    affected=cen.loc[cen.positive_nonendpoint_transit]
    result=dict(status="SOURCE_PATH_CONTRACT_MISMATCH",regions=len(regions),all_native_node_balances_pass=bool(b.passed.all()),
        maximum_node_balance_error=float(b.max_node_balance_error.max()),centroid_physical_nodes=len(cen),
        positive_nonendpoint_transit_centroids=len(affected),affected_regions=affected.region_id.nunique(),
        affected_active_regions=affected.loc[affected.m_active.gt(0)].region_id.nunique(),
        summed_node_transit_exposure=float(affected.excess_outgoing.sum()),
        node_transit_exposure_is_not_deduplicated_OD=True,
        necessary_simple_path_flow_test=True,not_a_reconstruction_of_historical_path_mixtures=True,
        original_arrays_modified=False,local_optimizer_calls=0)
    write_json(out/"POPULATION_SOURCE_FLOW_AUDIT.json",result)
    return result


def run_blocking_audit(c):
    with Monitor(c,"P3_MANDATORY_HIERARCHY_DIAGNOSIS"):
        certificate=original_oracle_certificate(c)
        source=population_source_flow_audit(c)
    record=dict(status="BLOCKED_WITH_EXACT_INTEGRITY_DEFECT",review_state="WAITING_FOR_EXTERNAL_REVIEW",
        certificate=certificate,source_audit=source,baseline_complete=False,enrichment_performed=False,
        reason="Mandatory mapped-LOW potential comparison contradicts an independently repriced regional lower bound under the inherited centroid nontransit path contract",
        engineering_fix_authorized=False,theory_or_source_contract_review_required=True,
        preserve_existing_certified_labels_conditionally_under_their_exact_model_hashes=True,
        no_model_reinterpretation_or_source_rerun_performed=True)
    write_json(Path(c["output_root"])/"validation/BLOCKING_DIAGNOSIS.json",record)
    return record
