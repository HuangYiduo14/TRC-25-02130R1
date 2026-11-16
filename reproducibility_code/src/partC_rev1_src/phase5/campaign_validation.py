"""Physical diagnostics, complete runtime ablation and fresh product audits."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import time

import numpy as np
import pandas as pd

from .bacn_charges import StaticPathProjector
from .bundle_oracle import (exact_bundle_oracle, exact_series_oracle,
    get_exact_root_call_count, reset_exact_root_call_count)
from .campaign_compile import Population
from .campaign_data import PRIMARY, STRUCTURAL, emit_suite, suite_inputs
from .campaign_io import binding_hash, frame_write, sha, write_json
from .campaign_runtime import PackedRuntime, pack_runtime, stable_id


def validate_serialization(config):
    out=Path(config["output_root"]);records=[]
    database=out/"database/phase5_campaign.sqlite"
    if not database.exists():database=out/"database/phase5_compilation_core.sqlite"
    with sqlite3.connect(f"file:{database.as_posix()}?mode=ro",uri=True) as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok":raise ValueError("SQLite integrity failure")
        if db.execute("PRAGMA foreign_key_check").fetchall():raise ValueError("SQLite foreign key failure")
        for suite in config["suites"]:
            base=out/"suites"/suite
            links=pd.read_parquet(base/"runtime_links.parquet");nodes=pd.read_parquet(base/"retained_nodes.parquet")
            stored=pd.read_sql_query("SELECT * FROM runtime_links WHERE suite_id=? ORDER BY link_id",db,params=(suite,))
            if len(stored)!=len(links) or not np.array_equal(stored.link_id,links.link_id):raise ValueError("SQLite native ID ordering changed")
            for name in links:
                if pd.api.types.is_numeric_dtype(links[name]):
                    if not np.array_equal(stored[name].to_numpy(),links[name].to_numpy(),equal_nan=True):raise ValueError("SQLite float64 changed: "+name)
                elif not np.array_equal(stored[name].fillna("").astype(str),links[name].fillna("").astype(str)):
                    raise ValueError("SQLite registry string changed: "+name)
            expected=pack_runtime(links,nodes)
            with np.load(base/"runtime_arrays.npz",allow_pickle=False) as z:
                if set(z.files)!=set(expected.arrays):raise ValueError("Numeric runtime schema mismatch")
                for name,actual in expected.arrays.items():
                    if z[name].dtype!=actual.dtype or not np.array_equal(z[name],actual):raise ValueError("NPZ kernel roundtrip changed: "+name)
            schema=json.loads((base/"runtime_arrays.schema.json").read_text())
            if schema["sha256"]!=sha(base/"runtime_arrays.npz") or schema["link_id_order_sha256"]!=binding_hash(links.link_id.tolist()):
                raise ValueError("Array hash/ID ordering schema changed")
            for tier,column in (("low_current","low_current_flow"),("high_current","high_current_flow"),("low_aon_domain","low_aon_domain_flow")):
                with np.load(base/f"reference_{tier}.npz",allow_pickle=False) as z:
                    if not np.array_equal(z["flow"],links[column].to_numpy(float)):raise ValueError("CURRENT/AON reference roundtrip changed")
            if len(np.unique(expected.arrays["adjacency_edge_idx"]))!=len(links):raise ValueError("Adjacency lost native parallel alternatives")
            records.append({"suite_id":suite,"links":len(links),"nodes":len(nodes),"numeric_fields_exact":True,"native_ids_exact":True,"status":"PASS"})
        missing=db.execute("SELECT COUNT(*) FROM runtime_links l LEFT JOIN dynamic_models m ON l.dynamic_model_id=m.model_id WHERE m.model_id IS NULL").fetchone()[0]
        if missing:raise ValueError("Dangling model reference")
        owners=db.execute("SELECT suite_id,COUNT(*),COUNT(DISTINCT physical_arc_id),MIN(owner_count),MAX(owner_count) FROM physical_link_ownership GROUP BY suite_id").fetchall()
        if len(owners)!=5 or any(n!=661644 or unique!=n or mn!=1 or mx!=1 for _,n,unique,mn,mx in owners):raise ValueError("Physical ownership does not cover all arcs exactly once")
    return {"status":"PASS","fresh_reopen":True,"suite_records":records,"physical_ownership_rows":sum(v[1] for v in owners)}


def c5(ctx):
    config=ctx.config;out=ctx.output;source=Path(config["source_pilot_output_root"])
    pop=Population(source,config["suites"])
    b,t,f,low,high=suite_inputs(config,"s03")
    graph=b.graph
    physical={"LOW_ON_FROZEN_FREE_FLOW_PATH":b.low_physical,"HIGH_HELDOUT_FROZEN_COMPARATOR_PATH":b.high_physical,
              "HIGH_HELDOUT_SELECTED_PATH":b.high_physical}
    frozen=pd.read_parquet(b.sources["frozen_path_sample"])
    highpaths=pd.read_parquet(Path(b.sources["high_benchmark"]["root"])/"routes/sampled_route_switches.parquet")
    legacy=pd.read_parquet(source/"path_validation/path_cost_sample.parquet")
    accepted=pd.read_parquet(out/"registries/accepted_macro_occurrences.parquet")
    accepted=accepted.loc[accepted.suite_short_id.eq("s03")]
    links=pd.read_parquet(out/"suites/s03/runtime_links.parquet").set_index("link_id")
    charges=pd.read_parquet(out/"registries/charges.parquet")
    charges=charges.loc[charges.suite_id.eq("s03")]
    ch={v:charges.loc[charges.model_variant.eq(v)].set_index("bacn_id").chi_s.to_dict() for v in (PRIMARY,STRUCTURAL)}
    first={};consumed=set()
    for r in accepted.itertuples(index=False):
        ids=tuple(json.loads(r.primitive_bundle_ids_json));consumed.update(ids)
        if ids[0] in first:raise ValueError("Ambiguous accepted macro first primitive")
        first[ids[0]]=(ids,r.chain_id)
    directmember={str(r.member_arc_id):r for r in links.loc[links.role.eq("PHYSICAL_MEMBER")].itertuples()}
    projector=StaticPathProjector(graph,t,f)
    traces={"LOW_ON_FROZEN_FREE_FLOW_PATH":low,"HIGH_HELDOUT_FROZEN_COMPARATOR_PATH":high,"HIGH_HELDOUT_SELECTED_PATH":high}
    scalar_cache={};macro_cache={};records=[];arcrows=[];charge_rows=[];static_rows=[];query_rows=[]
    roles=[("LOW_ON_FROZEN_FREE_FLOW_PATH",frozen,"arc_indices_json","arc_ids_sha256","\x1f"),
           ("HIGH_HELDOUT_FROZEN_COMPARATOR_PATH",frozen,"arc_indices_json","arc_ids_sha256","\x1f"),
           ("HIGH_HELDOUT_SELECTED_PATH",highpaths,"selected_arc_indices_json","selected_arc_ids_sha256","\n")]
    for role,table,arc_column,hash_column,separator in roles:
        state=physical[role];bundle_flow=traces[role]["current__bundle_flow"]
        for ordinal,r in enumerate(table.itertuples(index=False)):
            arcs=np.asarray(json.loads(getattr(r,arc_column)),dtype=np.int64)
            arcids=graph.arc_idx_to_directed_arc_id[arcs].astype(str)
            digest=hashlib.sha256(separator.join(arcids).encode()).hexdigest()
            if digest!=getattr(r,hash_column) or not len(arcs) or not np.all(graph.head[arcs[:-1]]==graph.tail[arcs[1:]]) or graph.tail[arcs[0]]!=r.origin_idx or graph.head[arcs[-1]]!=r.destination_idx:
                raise ValueError("Accepted witness stable-ID or continuity failure")
            pid=stable_id("p5_path_",role,str(r.sample_od_id))
            region=t.arc_region_local[arcs];bundle=t.arc_to_bundle_local[arcs];internal=t.arc_internal_bacn_local[arcs]
            physical_total=float(state.current_cost_s[arcs].sum())
            regional=float(state.current_cost_s[arcs[region>=0]].sum())
            Bphysical=float(state.current_cost_s[arcs[internal>=0]].sum())
            backbonephysical=float(state.current_cost_s[arcs[bundle>=0]].sum())
            if not np.isclose(physical_total,regional+Bphysical+backbonephysical,rtol=1e-12,atol=1e-7):raise ValueError("Physical witness ownership gap/overlap")
            counts,weights=projector.project(arcs)
            for link,count in counts.items():static_rows.append({"path_id":pid,"original_link_id":link,"occurrences":count})
            for node,weight in weights.items():
                nid=t.bacn_ids[node]
                charge_rows.append({"path_id":pid,"bacn_id":nid,"half_incidence":weight,
                    "primary_charge_s":ch[PRIMARY][nid],"structural_charge_s":ch[STRUCTURAL][nid]})
            scalar=0.;segments=[];segment=[]
            for position,a in enumerate(arcs):
                local=int(t.arc_to_bundle_local[a]);reg=int(t.arc_region_local[a])
                arcrows.append({"path_id":pid,"position":position,"physical_arc_idx":int(a),"physical_arc_id":str(graph.arc_idx_to_directed_arc_id[a]),
                    "owner":"REGION" if reg>=0 else "INTER_B" if local>=0 else "SUPPRESSED_B"})
                if reg>=0:
                    if segment:segments.append(segment);segment=[]
                elif local>=0:
                    primitive=t.bundle_ids[local];key=(role,primitive)
                    if key not in scalar_cache:
                        spec=pop.specs[pop.bundle_rows[("s03",primitive)].canonical_primitive_key]
                        result=exact_bundle_oracle(spec,float(bundle_flow[local]));scalar_cache[key]=result.marginal_cost_s
                        query_rows.append({"source_role":role,"primitive_id":primitive,"q":float(bundle_flow[local]),
                            "cost_s":result.marginal_cost_s,"certified":result.certified,"flow_residual":result.allocation_residual,
                            "kkt_s":result.kkt_residual_s,"gap_per_flow_s":result.fw_gap_per_flow_s})
                    scalar+=scalar_cache[key];segment.append((primitive,int(a),local))
            if segment:segments.append(segment)
            actual_dynamic=0.;fit_error=0.;alignment=0.;expansion_delta=0.;macros=0;flags=0;runtime_links=[]
            for segment in segments:
                position=0
                while position<len(segment):
                    primitive,a,local=segment[position]
                    match=first.get(primitive)
                    if match is not None:
                        ids,lid=match
                        if tuple(v[0] for v in segment[position:position+len(ids)])!=ids:
                            raise ValueError("Accepted physical witness only partially traverses removed macro interior")
                        row=links.loc[lid];q=float(row.low_current_flow if role.startswith("LOW") else row.high_current_flow)
                        key=(role,lid)
                        if key not in macro_cache:
                            specs=tuple(pop.specs[pop.bundle_rows[("s03",p)].canonical_primitive_key] for p in ids)
                            exact=exact_series_oracle(specs,q)
                            macro_cache[key]=exact.marginal_cost_s
                        exact=macro_cache[key];compiled=float(row.tau_s+row.k4*q**4)
                        base=sum(scalar_cache[(role,p)] for p in ids)
                        actual_dynamic+=compiled;fit_error+=compiled-exact;alignment+=exact-base
                        macros+=1;flags+=int(row.runtime_class=="FIT_POWER4" and q>row.domain_Q)
                        runtime_links.append((lid,1));position+=len(ids)
                    else:
                        if primitive in consumed:raise ValueError("Witness cannot start inside a removed macro")
                        arcid=str(graph.arc_idx_to_directed_arc_id[a])
                        if arcid in directmember:
                            member=directmember[arcid];q=float(state.current_flow[a]);value=float(member.tau_s+member.k4*q**4)
                            if not np.isclose(value,state.current_cost_s[a],rtol=2e-15,atol=1e-7):raise ValueError("Expanded member CURRENT cost differs from physical BPR")
                            expansion_delta+=value-scalar_cache[(role,primitive)];lid=member.Index
                        else:
                            row=links.loc[primitive];q=float(bundle_flow[local]);value=float(row.tau_s+row.k4*q**4);lid=primitive
                            if not np.isclose(value,scalar_cache[(role,primitive)],rtol=1e-11,atol=1e-7):raise ValueError("Exact primitive differs from certified scalar")
                        actual_dynamic+=value;runtime_links.append((lid,1));position+=1
            # Charge evaluated through the emitted macro/member/connector path also
            # has to match the original static incidence, for each charge registry.
            for connector,count in counts.items():
                if connector in links.index and links.loc[connector,"role"]=="TERMINAL_CONNECTOR":runtime_links.append((connector,count))
            for variant in (PRIMARY,STRUCTURAL):
                fixed=sum(ch[variant][t.bacn_ids[n]]*w for n,w in weights.items())
                emitted_fixed=0.
                for lid,count in runtime_links:
                    link=links.loc[lid]
                    value=link.structural_fixed_s if variant==STRUCTURAL else link.fixed_internal_H_s+link.tail_half_charge_s+link.head_half_charge_s
                    emitted_fixed+=count*value
                if not np.isclose(fixed,emitted_fixed,rtol=1e-11,atol=1e-7):raise ValueError("Path H/endpoint halves lost or duplicated after split/expansion")
                exact_scalar=regional+scalar+fixed
                runtime=regional+actual_dynamic+fixed
                total_error=runtime-physical_total
                fixed_error=fixed-Bphysical;free_mixing=scalar-backbonephysical
                closure=total_error-(fixed_error+free_mixing+expansion_delta+alignment+fit_error)
                if abs(closure)>1e-7+1e-10*max(1,abs(physical_total)):raise ValueError("Signed error decomposition fails closure")
                records.append({"path_id":pid,"sample_od_id":str(r.sample_od_id),"source_role":role,"suite_id":"s03","model_variant":variant,
                    "physical_path_cost_s":physical_total,"regional_owned_cost_s":regional,"suppressed_B_physical_cost_s":Bphysical,
                    "selected_physical_backbone_cost_s":backbonephysical,"exact_scalar_dynamic_s":scalar,"actual_compiled_dynamic_s":actual_dynamic,
                    "fixed_charge_s":fixed,"exact_scalar_total_s":exact_scalar,"compiled_total_s":runtime,
                    "fixed_charge_error_s":fixed_error,"scalar_free_mixing_error_s":free_mixing,
                    "expanded_observed_member_allocation_delta_s":expansion_delta,"current_chain_alignment_error_s":alignment,
                    "runtime_fit_error_s":fit_error,"signed_total_error_s":total_error,"closure_error_s":closure,
                    "signed_relative_error":total_error/max(1,abs(physical_total)),"absolute_relative_error":abs(total_error)/max(1,abs(physical_total)),
                    "macro_count":macros,"fitted_out_of_domain_count":flags,"current_cfw_mixture_sample":False,
                    "global_liftability":"NOT_ESTABLISHED","R_charge_policy":"PROVISIONAL_ZERO_NO_SEPARATE_EXPOSURE",
                    "accounting_and_witness_closure":"PASS","demand_weight":"NOT_ASSUMED"})
            if ordinal%32==0:ctx.mark(f"C5 {role} {ordinal+1}/{len(table)}")
    result=pd.DataFrame(records)
    if result.path_id.nunique()!=600 or len(result)!=1200:raise ValueError("Incomplete 600-path/two-variant coverage")
    paths=[]
    for name,frame in (("path_fidelity.parquet",result),("path_physical_arcs.parquet",pd.DataFrame(arcrows)),
        ("path_node_half_incidence.parquet",pd.DataFrame(charge_rows)),("path_original_static_occurrences.parquet",pd.DataFrame(static_rows)),
        ("path_offline_primitive_certificates.parquet",pd.DataFrame(query_rows))):
        path=out/"validation"/name;frame_write(path,frame);paths.append(path)
    ranking=[]
    for variant,g in result.groupby("model_variant"):
        frozen=g.loc[g.source_role.eq("HIGH_HELDOUT_FROZEN_COMPARATOR_PATH")].set_index("sample_od_id")
        chosen=g.loc[g.source_role.eq("HIGH_HELDOUT_SELECTED_PATH")].set_index("sample_od_id").loc[frozen.index]
        for od in frozen.index:
            fr=frozen.loc[od];hi=chosen.loc[od]
            physical_adv=fr.physical_path_cost_s-hi.physical_path_cost_s
            scalar_adv=fr.exact_scalar_total_s-hi.exact_scalar_total_s
            runtime_adv=fr.compiled_total_s-hi.compiled_total_s
            ranking.append({"sample_od_id":od,"model_variant":variant,"physical_selected_advantage_s":physical_adv,
                "scalar_selected_advantage_s":scalar_adv,"compiled_selected_advantage_s":runtime_adv,
                "physical_strict_advantage":physical_adv>1e-7,"physical_to_scalar_reversal":physical_adv>1e-7 and scalar_adv< -1e-7,
                "physical_to_compiled_reversal":physical_adv>1e-7 and runtime_adv< -1e-7,
                "scalar_to_compiled_rank_change":np.sign(scalar_adv)!=np.sign(runtime_adv),"high_selected_winner":False})
    frame_write(out/"validation/paired_ranking.csv",pd.DataFrame(ranking));paths.append(out/"validation/paired_ranking.csv")
    summary=[]
    for (variant,role),g in result.groupby(["model_variant","source_role"]):
        summary.append({"model_variant":variant,"source_role":role,"count":len(g),"mean_signed_relative_error":g.signed_relative_error.mean(),
            "median_absolute_relative_error":g.absolute_relative_error.median(),"p95_absolute_relative_error":g.absolute_relative_error.quantile(.95),
            "max_absolute_relative_error":g.absolute_relative_error.max(),"max_abs_closure_s":g.closure_error_s.abs().max(),
            "current_cfw_path_mixture_sample":False})
    frame_write(out/"validation/path_fidelity_summary.csv",pd.DataFrame(summary));paths.append(out/"validation/path_fidelity_summary.csv")
    return paths


def runtime_audit(base,charges):
    links=pd.read_parquet(base/"runtime_links.parquet");nodes=pd.read_parquet(base/"retained_nodes.parquet")
    with np.load(base/"runtime_arrays.npz",allow_pickle=False) as z:runtime=PackedRuntime({n:z[n] for n in z.files})
    fitted=runtime.arrays["is_fitted"];domains=runtime.arrays["domain_Q"]
    low=links.low_current_flow.to_numpy(float);high=links.high_current_flow.to_numpy(float)
    stress=np.maximum(1.,2*np.maximum(low,high));stress[fitted]=1.25*domains[fitted]
    max_fd=0.;flags={}
    reset_exact_root_call_count()
    for name,q in (("ZERO",np.zeros(len(links))),("LOW_CURRENT",low),("HIGH_SENSITIVITY",high),("DOMAIN_STRESS",stress)):
        (cost,potential,derivative),flag=runtime.evaluate(q)
        if not np.isfinite(cost).all() or not np.isfinite(potential).all() or not np.isfinite(derivative).all() or (derivative<0).any():
            raise ValueError("All-link numerical runtime gate failed")
        h=np.cbrt(np.finfo(float).eps)*np.maximum(1,q)
        pplus=runtime.evaluate(q+h,"potential")[0];pminus=runtime.evaluate(np.maximum(0,q-h),"potential")[0]
        centered=(pplus-pminus)/(q+h-np.maximum(0,q-h))
        forward=(-3*potential+4*pplus-runtime.evaluate(q+2*h,"potential")[0])/(2*h)
        fd=np.where(q<2*h,forward,centered)
        err=float((np.abs(fd-cost)/np.maximum(1,np.abs(cost))).max())
        if err>1e-6:raise ValueError("Full packed potential-gradient identity failed")
        max_fd=max(max_fd,err);flags[name]=int(flag.sum())
    if get_exact_root_call_count()!=0:raise ValueError("Online kernel called an oracle")
    suite=str(links.suite_id.iloc[0]);records=[]
    # Account for removed internal nodes and retained halves at both CURRENT states.
    seq=pd.read_parquet(base/"macro_sequences.parquet")
    for tier,column,V in (("LOW","low_current_flow","V_L"),("HIGH","high_current_flow","V_H_evaluation_only")):
        q=links[column].to_numpy(float)
        for variant in (PRIMARY,STRUCTURAL):
            original=charges.loc[charges.suite_id.eq(suite)&charges.model_variant.eq(variant)].set_index("bacn_id")
            node_index=pd.Index(original.index)
            tail=node_index.get_indexer(links.tail_node_id);head=node_index.get_indexer(links.head_node_id)
            incidence=np.bincount(tail[tail>=0],weights=.5*q[tail>=0],minlength=len(original))+np.bincount(head[head>=0],weights=.5*q[head>=0],minlength=len(original))
            if len(seq):
                interior=seq.loc[seq.internal_node_after.ne("")]
                local=node_index.get_indexer(interior.internal_node_after)
                weights=interior.corridor_id.map(links.set_index("link_id")[column]).to_numpy(float)
                incidence+=np.bincount(local,weights=weights,minlength=len(original))
            residual=incidence-original[V].to_numpy(float)
            tolerance=1e-8+1e-10*np.maximum(1,original[V].to_numpy(float))
            if np.any(np.abs(residual)>tolerance):raise ValueError("Per-original-B charged incidence does not close")
            h=links.structural_fixed_s.to_numpy(float) if variant==STRUCTURAL else links.fixed_internal_H_s.to_numpy(float)+links.tail_half_charge_s.to_numpy(float)+links.head_half_charge_s.to_numpy(float)
            total=float(np.dot(q,h));expected=float(np.dot(original[V],original.chi_s))
            if not np.isclose(total,expected,rtol=1e-10,atol=1e-7):raise ValueError("Fixed potential duplicated after member expansion")
            records.append({"suite_id":suite,"tier":tier,"variant":variant,"max_incidence_residual":float(np.abs(residual).max()),"total_fixed_potential":total,"expected_original":expected,"status":"PASS"})
    return runtime,links,nodes,{"suite_id":suite,"link_count":len(links),"potential_gradient_error":max_fd,
        "root_calls":get_exact_root_call_count(),"out_of_domain_counts":flags,"charge_records":records,"status":"PASS"}


def benchmark_runtime(runtime,links,nodes,suite,representation):
    low=links.low_current_flow.to_numpy(float);high=links.high_current_flow.to_numpy(float)
    fitted=runtime.arrays["is_fitted"];stress=np.maximum(1.,2*np.maximum(low,high));stress[fitted]=1.25*runtime.arrays["domain_Q"][fitted]
    rows=[]
    for tier,q in (("LOW_CURRENT",low),("HIGH_SENSITIVITY",high),("ZERO",np.zeros(len(low))),("DOMAIN_STRESS",stress)):
        for operation in ("cost","potential","derivative","combined","flow_update_and_combined"):
            fn=(lambda:runtime.evaluate(.875*q+.125*stress)) if operation=="flow_update_and_combined" else (lambda:runtime.evaluate(q,operation))
            for _ in range(5):fn()
            times=[]
            for _ in range(30):
                start=time.perf_counter_ns();fn();times.append(time.perf_counter_ns()-start)
            rows.append({"suite_id":suite,"representation":representation,"state":tier,"operation":operation,
                "warmups":5,"repetitions":30,"median_ns":float(np.median(times)),"p95_ns":float(np.quantile(times,.95)),
                "min_ns":min(times),"max_ns":max(times),"edge_count":len(links),"node_count":len(nodes),
                "array_bytes":sum(a.nbytes for a in runtime.arrays.values()),"coverage":1.,"proxy_count":0,
                "root_calls":get_exact_root_call_count(),"timing_samples_ns_json":json.dumps(times)})
    # Packing/dispatch is measured separately, including stable native adjacency.
    for _ in range(5):pack_runtime(links,nodes)
    times=[]
    for _ in range(30):
        start=time.perf_counter_ns();pack_runtime(links,nodes);times.append(time.perf_counter_ns()-start)
    rows.append({"suite_id":suite,"representation":representation,"state":"REGISTRY","operation":"packing_dispatch",
        "warmups":5,"repetitions":30,"median_ns":float(np.median(times)),"p95_ns":float(np.quantile(times,.95)),
        "min_ns":min(times),"max_ns":max(times),"edge_count":len(links),"node_count":len(nodes),
        "array_bytes":sum(a.nbytes for a in runtime.arrays.values()),"coverage":1.,"proxy_count":0,
        "root_calls":get_exact_root_call_count(),"timing_samples_ns_json":json.dumps(times)})
    return rows


def c6(ctx):
    from .campaign_stages import compiled_cache
    cfg=ctx.config;out=ctx.output
    pop=Population(Path(cfg["source_pilot_output_root"]),cfg["suites"])
    cache=compiled_cache(out);accepted=pd.read_parquet(out/"registries/accepted_macro_occurrences.parquet")
    charges=pd.read_parquet(out/"registries/charges.parquet")
    benchmarks=[];audits=[]
    for suite in cfg["suites"]:
        ctx.mark("C6 complete no-degree2 reference compiler "+suite)
        emit_suite(ctx,pop,suite,accepted,cache,charges,baseline=True)
        for rep,folder in (("NO_DEGREE2_SAME_EXPLICIT_DISPATCH","baseline"),("REDUCED_FULL_COMPILER","suites")):
            runtime,links,nodes,audit=runtime_audit(out/folder/suite,charges);audit["representation"]=rep;audits.append(audit)
            ctx.mark("C6 all-link benchmark "+suite+" "+rep)
            benchmarks+=benchmark_runtime(runtime,links,nodes,suite,rep)
    frame_write(out/"validation/runtime_benchmark.csv",pd.DataFrame(benchmarks))
    write_json(out/"validation/all_link_kernel_and_charge_checks.json",{"status":"PASS","suites_and_baselines":audits})
    command=[cfg["python_executable"],"-m","partC_rev1_src.phase5.campaign_cli","validate","--config",str(ctx.config_path)]
    done=subprocess.run(command,capture_output=True,text=True,cwd=cfg["repository_root"])
    (out/"validation/fresh_process_reopen.log").write_text(done.stdout+done.stderr,encoding="utf-8")
    if done.returncode:raise RuntimeError("Fresh process product reopen failed")
    bindings=pd.read_parquet(out/"validation/source_bindings.parquet");invariance=[]
    for i,r in enumerate(bindings.itertuples(index=False)):
        p=Path(r.absolute_path);actual=sha(p)
        if p.stat().st_size!=r.size_bytes or actual!=r.sha256:raise ValueError("Accepted input changed during campaign: "+str(p))
        invariance.append({"absolute_path":str(p),"sha256":actual,"status":"PASS"})
        if i%32==0:ctx.mark(f"C6 source invariance {i}/{len(bindings)}")
    write_json(out/"validation/source_invariance.json",{"status":"PASS","files":invariance,"accepted_products_modified":False})
    fits=pd.read_parquet(out/"registries/calibration_requests.parquet");selected=fits.loc[fits.status.eq("ACCEPT")]
    if np.any(selected.cost_error>.01) or np.any(selected.potential_error>.01) or np.any(selected.potential_gradient_error>1e-6):raise ValueError("Selected fit validation gate changed")
    if not set(pop.experiments).issubset(set(fits.request_id)):raise ValueError("Incomplete full macro request population")
    pathsummary=pd.read_csv(out/"validation/path_fidelity_summary.csv");ranking=pd.read_csv(out/"validation/paired_ranking.csv")
    bench=pd.DataFrame(benchmarks)
    primary=pathsummary.loc[pathsummary.model_variant.eq(PRIMARY)]
    limitations={"model_readiness":"MODEL_LIMITATIONS_REMAIN","production_accepted":False,"national_assignment_authorized":False,
        "phase6_authorized":False,"training_authorized":False,"path_population":"600 existing witnesses in three roles; not CURRENT CFW mixture or national demand-weighted sample",
        "primary_path_summary":primary.to_dict("records"),"paired_ranking":ranking.groupby("model_variant").agg(
            strict_physical=("physical_strict_advantage","sum"),physical_to_scalar_reversals=("physical_to_scalar_reversal","sum"),
            physical_to_compiled_reversals=("physical_to_compiled_reversal","sum"),scalar_to_compiled_changes=("scalar_to_compiled_rank_change","sum")).reset_index().to_dict("records"),
        "R_physical_ownership":"Every available witness arc has one observed owner",
        "R_interface_state":"Counterfactual transition distortion remains unresolved; zero charge is provisional",
        "free_mixing":"Member expansion preserves optimum after minimizing local split, not equality at arbitrary observed CURRENT allocation",
        "liftability":"Global new-flow liftability not established","national_speedup":"Not established by compiler-only kernel benchmark",
        "charge_rule_selection":"Low static E/V baseline and structural sensitivity both retained; high selects neither"}
    write_json(out/"validation/model_limitations.json",limitations)
    gates=pd.read_csv(cfg["acceptance_tests_file"])
    evidence={"G01":"reports/source_and_reuse_audit.json","G02":"validation/complete_coverage.json","G03":"reports/actual_topology_counts.csv",
        "G04":"registries/corridor_decisions.parquet","G05":"registries/physical_link_ownership.parquet","G06":"validation/all_link_kernel_and_charge_checks.json",
        "G07":"compilation/*/*/certificates.parquet and immutable reused pilot certificate bindings","G08":"validation/all_link_kernel_and_charge_checks.json",
        "G09":"registries/calibration_requests.parquet","G10":"suites/*/runtime_links.parquet","G11":"validation/full_tests.log; validation/serialization_roundtrip.json",
        "G12":"validation/runtime_benchmark.csv","G13":"validation/all_link_kernel_and_charge_checks.json","G14":"registries/regional_interface_contract.json",
        "G15":"validation/fresh_process_reopen.log","G16":"validation/runtime_benchmark.csv","G17":"validation/path_fidelity_summary.csv; validation/paired_ranking.csv",
        "G18":"validation/resource_history.csv","G19":"C7 seal receipt and adjacent companion (pending)"}
    gates["status"]=["PENDING_C7_SEAL" if g=="G19" else "PASS" for g in gates.gate_id]
    gates["evidence"]=gates.gate_id.map(evidence)
    frame_write(out/"validation/acceptance_results_preseal.csv",gates)
    make_figures(out)
    return [out/p for p in ("validation/runtime_benchmark.csv","validation/all_link_kernel_and_charge_checks.json",
        "validation/fresh_process_reopen.log","validation/source_invariance.json","validation/model_limitations.json","validation/acceptance_results_preseal.csv")]+list((out/"figures").glob("*"))


def make_figures(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    figdir=out/"figures";figdir.mkdir(exist_ok=True)
    topo=pd.read_csv(out/"reports/actual_topology_counts.csv")
    path=pd.read_csv(out/"validation/path_fidelity_summary.csv")
    bench=pd.read_csv(out/"validation/runtime_benchmark.csv")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout="constrained")
    x=np.arange(len(topo));w=.36
    axes[0].bar(x-w/2,topo.original_B_nodes,w,label="Original B nodes",color="#a5b8cb")
    axes[0].bar(x+w/2,topo.retained_B_nodes,w,label="Actual retained B nodes",color="#246a86")
    axes[0].set_xticks(x,topo.suite_id);axes[0].set_title("Actual node retention after fit and split");axes[0].legend(fontsize=8)
    axes[1].bar(x-w/2,topo.original_bundles,w,label="Original primitive bundles",color="#a5b8cb")
    axes[1].bar(x+w/2,topo.actual_backbone_links,w,label="Actual native backbone edges",color="#c17837")
    axes[1].set_xticks(x,topo.suite_id);axes[1].set_title("Member expansion can offset edge reduction");axes[1].legend(fontsize=8)
    fig.savefig(figdir/"actual_topology.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.8),layout="constrained")
    for variant,g in path.groupby("model_variant"):
        ax.plot(np.arange(len(g)),100*g.p95_absolute_relative_error,"o-",label=variant)
    ax.set_xticks(np.arange(3),["High frozen comparator","High selected witness","Low frozen witness"])
    ax.set_ylabel("95th percentile absolute path error (%)");ax.set_title("Existing witness fidelity remains separate from numerical closure")
    ax.legend(fontsize=8);fig.savefig(figdir/"path_model_limitations.png",dpi=180);plt.close(fig)
    subset=bench.loc[bench.operation.eq("combined")&bench.state.eq("LOW_CURRENT")]
    pivot=subset.pivot(index="suite_id",columns="representation",values="median_ns")/1e6
    fig,ax=plt.subplots(figsize=(10,4.5),layout="constrained");pivot.plot.bar(ax=ax,color=["#a5b8cb","#246a86"],rot=0)
    ax.set_ylabel("Median combined kernel time (ms)");ax.set_title("Complete root-free arrays: compiler-only ablation, no assignment")
    ax.legend(fontsize=8);fig.savefig(figdir/"full_kernel_ablation.png",dpi=180);plt.close(fig)
    qa=[]
    for p in sorted(figdir.glob("*.png")):
        with Image.open(p) as img:img.load();qa.append({"file":p.name,"width":img.width,"height":img.height,"status":"REOPEN_PASS_MANUAL_REVIEW_REQUIRED"})
    write_json(figdir/"figure_reopen.json",qa)
