"""Portable physical assignment, selected consolidation and common P topology.

Uses the project's numerical APIs. Historical approvals, campaign clocks and
original workstation output directories are deliberately absent from this CLI.
All real work requires the supplied explicit configuration and fresh inputs.
"""
from __future__ import annotations
import argparse
from dataclasses import fields, asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import psutil

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'src'))
sys.path.insert(0,str(PACKAGE/'scripts'))
from partC_rev1_src.partC_rev1_graph_snapshot import partC_rev1_load_graph_snapshot
from partC_rev1_src.partC_rev1_od_snapshot import partC_rev1_load_od_snapshot
from partC_rev1_src.phase4.inputs import load_or_build_aggregated_od
from partC_rev1_src.phase4.models import SuiteTraceMap
from partC_rev1_src.phase4.solver import Phase4SolverConfig, solve_assignment
from partC_rev1_src.phase4.checkpoint import _trace_arrays, load_phase4_checkpoint
from partC_rev1_src.phase5.bundle_oracle import BundleSpec, series_power_terms
from partC_rev1_src.phase5.experiments import MacroExperiment, dynamic_expression_key, calibration_request_key
from partC_rev1_src.phase5.campaign_compile import compile_request
from partC_rev1_src.pacnhn_v1.topology import assemble_common, partition_arrays, interface_registry, runtime_identities
from partC_rev1_src.pacnhn_v1.control import array_hash


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    def default(v):
        if isinstance(v,(np.integer,np.floating)):return v.item()
        if isinstance(v,Path):return str(v)
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=default)+'\n',encoding='utf-8')
def arrays(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def save(path,**a):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('xb') as f:np.savez_compressed(f,**a)
def parquet(path,frame):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    if Path(path).exists():raise FileExistsError(path)
    frame.to_parquet(path,index=False,compression='zstd')


class Run:
    def __init__(self,path,step):
        self.path=Path(path).resolve(strict=True);self.c=read(self.path);self.started=time.monotonic();self.peak=0
        if 'REPRODUCTION_RUN_ID' in os.environ:self.c['run_id']=os.environ['REPRODUCTION_RUN_ID']
        for name in ('phase_id','run_id','tasks','input_versions','expected_outputs','acceptance_tests','resource_budget'):
            if not self.c.get(name):raise ValueError('Nonempty execution field required: '+name)
        if step not in self.c['tasks']:raise ValueError('Step absent from explicit tasks')
        self.step=step;self.root=self.pathof(self.c['output_root']);self.root.mkdir(parents=True,exist_ok=True)
        self.plan=self.pathof(self.c['input_versions']['frozen_plan'])
        manifest=read(self.plan/'PLAN_MANIFEST.json')
        for r in manifest['members']:
            if sha(self.plan/r['name'])!=r['sha256']:raise ValueError('Frozen topology metadata changed: '+r['name'])
        self.mapping=arrays(self.plan/'trace_mapping.npz');self.metadata=manifest
        receipt=read(self.pathof(self.c['input_versions']['upstream_receipt']))
        gp=receipt['graph_paths'];op=receipt['od_paths']
        self.graph=partC_rev1_load_graph_snapshot(self.pathof(gp['npz']),self.pathof(gp['metadata']))
        self.input_config={'input_versions':{'od_npz':str(self.pathof(op['npz'])),
            'od_metadata_json':str(self.pathof(op['metadata']))}}
        od=partC_rev1_load_od_snapshot(self.input_config['input_versions']['od_npz'],self.input_config['input_versions']['od_metadata_json'])
        self.input_config['input_versions'].update(od_identity=od.od_hash,od_row_count=len(od.demand),
            origin_count=len(od.origin_group_node_idx),total_demand=float(od.demand.sum()))
        for name,current in [('physical_node_ids',self.graph.node_idx_to_uid.astype(str)),
                             ('physical_arc_ids',self.graph.arc_idx_to_directed_arc_id.astype(str))]:
            if not np.array_equal(self.mapping[name],current):
                raise ValueError('FAF cleaned stable-ID order differs from frozen plan '+name+'. Use the documented input editions; do not fill missing IDs.')
        if len(self.graph.tail)==0 or len(od.demand)==0:raise ValueError('Empty input is a failure')
        self.guard()
    def pathof(self,p):
        p=Path(p);return p if p.is_absolute() else PACKAGE/p
    def frame(self,name):return pd.read_parquet(self.plan/(name+'.parquet'))
    def guard(self):
        b=self.c['resource_budget'];p=psutil.Process();rss=p.memory_info().rss
        for child in p.children(recursive=True):
            try:rss+=child.memory_info().rss
            except psutil.NoSuchProcess:pass
        self.peak=max(self.peak,rss)
        if rss>int(b['rss_bytes']):raise MemoryError('Combined process tree RSS budget exceeded')
        if time.monotonic()-self.started>float(b['max_wall_seconds']):raise TimeoutError('Explicit stage wall budget exceeded')
        if psutil.virtual_memory().available<int(b['minimum_available_bytes']):raise MemoryError('Available RAM guard exceeded')
    def finish(self,result):
        self.guard();result.update(phase_id=self.c['phase_id'],step_id=self.step,run_id=self.c['run_id'],
            config_sha256=sha(self.path),wall_seconds=time.monotonic()-self.started,peak_rss_bytes=self.peak,
            fresh_graph_identity=self.graph.graph_hash,frozen_plan_sha256=sha(self.plan/'PLAN_MANIFEST.json'))
        write(self.root/(self.step+'_STATUS.json'),result)
        print(json.dumps(result,default=str))
    def trace(self):
        m=self.mapping;kwargs={f.name:m[f.name] for f in fields(SuiteTraceMap) if f.name in m}
        for name in ('terminal_ids','region_ids','bundle_ids','bacn_ids','racn_ids'):kwargs[name]=tuple(m[name].astype(str))
        kwargs.update(suite_id='s03',phase2_suite_id=self.metadata['source_phase2_suite'],
            candidate_id=self.metadata['source_candidate'],root=self.plan,directory_digest=sha(self.plan/'PLAN_MANIFEST.json'),
            conceptual_regional_pair_count=self.metadata['conceptual_regional_pair_count'])
        t=SuiteTraceMap(**kwargs);t.validate(node_count=len(self.graph.node_idx_to_uid),arc_count=len(self.graph.tail));return t


def export_physical(r,label,result):
    dst=r.root/'physical'/label;dst.mkdir(parents=True,exist_ok=True)
    a=dict(current_physical_flow=result.flow,current_cost_minutes=result.cost,
           used_capacity=result.used_capacity,aon_physical_flow=result.latest_aon.physical_flow)
    _trace_arrays('current',result.suite_traces['s03'],a)
    _trace_arrays('aon',result.latest_aon.suite_traces['s03'],a)
    save(dst/'state.npz',**a)
    metrics=asdict(result.final_metrics)
    write(dst/'RESULT.json',dict(status='PASS',role=label,CURRENT='Safeguarded-CFW convex mixture',
        AON='Auxiliary shortest path at CURRENT costs; domain evidence only',
        state_iteration=int(result.checkpoint_receipts[-1]['directory'].split('_')[-1]),
        metrics=metrics,stop_reason=result.stop_reason,checkpoint=result.checkpoint_receipts[-1]['directory'],
        reachable_demand=result.latest_aon.reachable_demand,unreachable_demand=result.latest_aon.unreachable_demand))


def physical(r):
    od=load_or_build_aggregated_od(r.input_config,output_root=r.root)
    t=r.trace();c=r.c['physical'];out=r.root/'physical'
    if out.exists():raise FileExistsError('Fresh physical output root required: '+str(out))
    common=dict(method='SAFEGUARDED_CFW',capacity_multiplier=float(c['capacity_multiplier']),
        max_iterations=int(c['high_iterations']),worker_count=int(c['workers']),
        rss_hard_bytes=int(r.c['resource_budget']['rss_bytes']),checkpoint_compressed=True,
        final_checkpoint=True,iteration_record_path=out/'iterations.jsonl',
        max_runtime_seconds=float(c['max_seconds_per_solve']))
    low=solve_assignment(r.graph,od,trace_maps=[t],config=Phase4SolverConfig(**common,
        gap_target=float(c['low_rg_sptt']),checkpoint_directory=out/'low_checkpoints'),
        run_metadata={'phase_id':r.c['phase_id'],'step_id':'physical','run_id':r.c['run_id']})
    export_physical(r,'low',low)
    if low.final_metrics.relative_gap_sptt>float(c['low_rg_sptt']):
        raise RuntimeError('LOW fully refreshed gap criterion not met; partial state preserved')
    low_iteration=int(Path(low.checkpoint_receipts[-1]['directory']).name.rsplit('_',1)[-1])
    if low_iteration>=int(c['high_iterations']):
        high=low
    else:
        state=load_phase4_checkpoint(low.checkpoint_receipts[-1]['directory'],trace_maps=[t])
        high=solve_assignment(r.graph,od,trace_maps=[t],config=Phase4SolverConfig(**common,
            gap_target=0.,checkpoint_directory=out/'high_checkpoints'),restart_state=state,
            run_metadata={'phase_id':r.c['phase_id'],'step_id':'physical','run_id':r.c['run_id']})
    export_physical(r,'high',high)
    iteration=int(Path(high.checkpoint_receipts[-1]['directory']).name.rsplit('_',1)[-1])
    if iteration!=int(c['high_iterations']):raise RuntimeError('HIGH iteration target not reached; partial state preserved')
    r.finish(dict(status='PASS',low_iteration=low_iteration,high_iteration=iteration,
        low_rg_sptt=low.final_metrics.relative_gap_sptt,high_rg_sptt=high.final_metrics.relative_gap_sptt,
        high_1e4_target_claimed=False,meaningful_output_count=len(high.flow)))


def static_visits(mapping,bundles,current):
    """Phase5 static half-incidence denominator; OD supply is not a charged link."""
    bi=pd.Index(mapping['bacn_ids']);bundles=bundles.set_index('bundle_id').loc[list(mapping['bundle_ids'])]
    tail=bi.get_indexer(bundles.tail_bacn_id);head=bi.get_indexer(bundles.head_bacn_id)
    if min(tail.min(),head.min())<0:raise ValueError('Bundle B endpoint missing')
    q=current['current__s03__bundle_flow'];n=len(bi)
    v=np.bincount(tail,weights=q,minlength=n)+np.bincount(head,weights=q,minlength=n)
    k=current['current__s03__coarse_regional_pair_keys'];f=current['current__s03__coarse_regional_pair_values']
    parent=mapping['terminal_parent_bacn_local']
    for ti in ((k>>np.uint64(32)).astype(np.int64),(k&np.uint64(0xffffffff)).astype(np.int64)):
        b=parent[ti];valid=b>=0;v+=np.bincount(b[valid],weights=f[valid],minlength=n)
    return v/2.


def consolidate(r):
    out=r.root/'consolidation'
    if out.exists():raise FileExistsError('Fresh consolidation root required')
    out.mkdir()
    low=arrays(r.root/'physical/low/state.npz');high=arrays(r.root/'physical/high/state.npz');m=r.mapping;g=r.graph
    t=r.trace();be=r.frame('bundle_endpoints');membership=r.frame('bundle_membership')
    ai=pd.Index(g.arc_idx_to_directed_arc_id.astype(str)).get_indexer(membership.directed_arc_id.astype(str))
    if np.any(ai<0):raise ValueError('Primitive physical member missing')
    membership=membership.assign(arc_idx=ai)
    specs={}
    for bid,f in membership.groupby('bundle_id',sort=False):
        idx=f.arc_idx.to_numpy(np.int64)
        specs[str(bid)]=BundleSpec(tuple(f.directed_arc_id.astype(str)),g.free_time[idx]*60,
            low['used_capacity'][idx],g.alpha[idx],g.beta[idx])
    # Regenerate structural median and LOW E/V charge from the fresh CURRENT state.
    internal=m['arc_internal_bacn_local'];valid=internal>=0;n=len(m['bacn_ids'])
    E=np.bincount(internal[valid],weights=low['current_physical_flow'][valid]*low['current_cost_minutes'][valid]*60,minlength=n)
    seed=pd.DataFrame(dict(bacn_local=internal[valid],fftt=g.free_time[valid]*60)).groupby('bacn_local').fftt.median().reindex(np.arange(n),fill_value=0).to_numpy()
    V=static_visits(m,be,low);chi=np.divide(E,V,out=seed.copy(),where=V>0)
    if np.any(~np.isfinite(chi)) or np.any(chi<0):raise ValueError('Invalid unclipped LOW E/V charge')
    charges=pd.DataFrame(dict(suite_id='s03',bacn_id=m['bacn_ids'],model_variant='LOW_MEAN_STATIC_INCIDENCE_V1',
        E_L=E,V_L=V,chi_s=chi,structural_seed_s=seed,fallback_used=V==0,high_calibrated=False,clipped=False))
    parquet(out/'charges.parquet',charges);chi_by=dict(zip(charges.bacn_id,chi));seed_by=dict(zip(charges.bacn_id,seed))
    seq=r.frame('macro_sequences').sort_values(['corridor_id','sequence_position'])
    macro={str(cid):f for cid,f in seq.groupby('corridor_id',sort=False)}
    bundle_index={v:i for i,v in enumerate(m['bundle_ids'])}
    qL=low['current__s03__bundle_flow'];qA=low['aon__s03__bundle_flow'];qH=high['current__s03__bundle_flow']
    rows=[];fits=[];fitted_count=0
    for ordinal,row in enumerate(r.frame('backbone_plan').itertuples(index=False)):
        if ordinal%64==0:r.guard()
        if row.role=='PHYSICAL_MEMBER':
            a=int(row.member_arc_idx)
            if str(g.arc_idx_to_directed_arc_id[a])!=str(row.member_arc_id):raise ValueError('Physical member ID/index mismatch')
            tau=float(g.free_time[a]*60);k=tau*float(g.alpha[a])/float(low['used_capacity'][a])**4
            dynamic_id='physical:'+str(row.member_arc_id);domain=np.inf;H=SH=0.
        else:
            if row.role=='MACRO':
                f=macro[str(row.link_id)];ids=f.primitive_id.astype(str).tolist()
                internal_ids=f.internal_node_after.astype(str).tolist();H=sum(chi_by.get(v,0.) for v in internal_ids if v)
                SH=sum(seed_by.get(v,0.) for v in internal_ids if v)
            elif row.role=='EXACT_PRIMITIVE':ids=[str(row.primitive_id)];H=SH=0.
            else:raise ValueError('Unknown selected backbone role: '+str(row.role))
            selected=tuple(specs[v] for v in ids);exact=series_power_terms(selected)
            ii=np.asarray([bundle_index[v] for v in ids]);lo=qL[ii];hi=qH[ii]
            if np.ptp(lo)>1e-8+1e-10*max(1.,np.max(np.abs(lo))):
                raise ValueError('Fresh LOW CURRENT chain flow is inconsistent: '+str(row.link_id))
            if exact is not None:
                tau,terms=exact
                if any(beta!=4. for beta,_ in terms):raise ValueError('Unexpected exponent')
                k=sum(v for _,v in terms);domain=np.inf;dynamic_id=dynamic_expression_key(selected)
            else:
                # AON enters only the LOW-only domain grid. HIGH never selects or calibrates.
                request=calibration_request_key(selected,float(lo.min()),float(qA[ii].max()))
                ex=MacroExperiment(request,str(row.link_id),'s03',selected,float(lo.min()),float(qA[ii].max()),
                    float(hi.min()),len(selected),False,dynamic_expression_key=dynamic_expression_key(selected),calibration_request_key=request)
                fit,probes,certificates=compile_request(ex)
                # Persist fit evidence before any failure. A fresh rejected plan requires explicit review.
                write(out/'fit_evidence'/f'{row.link_id}.json',dict(fit=fit,probes=probes,certificates=certificates))
                if fit['status']!='ACCEPT':raise RuntimeError('Frozen selected fit fails fresh certificate: '+str(row.link_id))
                tau=fit['tau_s'];k=fit['k4'];domain=fit['domain_Q'];dynamic_id=fit['model_id'];fits.append(fit);fitted_count+=1
        record=dict(link_id=row.link_id,tail_node_id=row.tail_node_id,head_node_id=row.head_node_id,
            role=row.role,runtime_class='FIT_POWER4' if np.isfinite(domain) else row.runtime_class,
            dynamic_model_id=dynamic_id,tau_s=tau,k4=k,beta=4.,domain_Q=domain,
            fixed_internal_H_s=H,tail_half_charge_s=chi_by.get(row.tail_node_id,0.)/2,
            head_half_charge_s=chi_by.get(row.head_node_id,0.)/2,
            structural_fixed_s=SH+(seed_by.get(row.tail_node_id,0.)+seed_by.get(row.head_node_id,0.))/2)
        rows.append(record)
    links=pd.DataFrame(rows)
    if len(links)!=r.metadata['frozen_backbone_coordinate_count'] or not links.link_id.is_unique:raise ValueError('Frozen B coordinate closure fails')
    if not np.isfinite(links[['tau_s','k4','fixed_internal_H_s','tail_half_charge_s','head_half_charge_s']].to_numpy()).all():raise ValueError('Nonfinite runtime coefficient')
    parquet(out/'runtime_links.parquet',links)
    if fits:parquet(out/'fit_summary.parquet',pd.DataFrame(fits))
    r.finish(dict(status='PASS',backbone_count=len(links),fit_count=fitted_count,charge_count=len(charges),
        flow_calibration='LOW_CURRENT_ONLY',AON_role='LOW_DOMAIN_EVIDENCE_ONLY',HIGH_role='EVALUATION_ONLY',
        meaningful_output_count=len(links)))


def topology(r):
    out=r.root/'topology'
    if out.exists():raise FileExistsError('Fresh topology root required')
    out.mkdir()
    physical={name:getattr(r.graph,name) for name in ('node_idx_to_uid','arc_idx_to_directed_arc_id','tail','head','free_time','capacity','alpha','beta')}
    links=pd.read_parquet(r.root/'consolidation/runtime_links.parquet');charges=pd.read_parquet(r.root/'consolidation/charges.parquet')
    original=arrays(r.root/'derived_inputs/aggregated_od.npz');closed=read(r.plan/'represented_i95_ids.json')
    a,o,f,report=assemble_common(physical,links,r.frame('retained_nodes'),r.frame('physical_ownership'),
        r.frame('bacn_membership'),charges,original,closed,
        expected_backbone_count=r.metadata['frozen_backbone_coordinate_count'],capacity_multiplier=float(r.c['physical']['capacity_multiplier']))
    pc=r.c['partition'];suites=r.c['suite_ids'];resolutions=r.c['resolutions']
    p,regions,nodepart,ps=partition_arrays(a,physical,r.frame('physical_ownership'),suites=suites,resolutions=resolutions,partition_config=pc,guard=r.guard)
    f.update(regions=regions,rebuilt_node_partitions=nodepart,interfaces=interface_registry(a,p),frozen_macro_sequences=r.frame('macro_sequences'))
    save(out/'common_graph.npz',**a);save(out/'common_od.npz',**o);save(out/'partitions.npz',**p)
    for name,frame in f.items():r.guard();parquet(out/(name+'.parquet'),frame)
    gid,oid=runtime_identities(a,o)
    report.update(status='PASS',schema='pacnhn.common_topology.v1',partitions=ps,graph_identity=gid,
        OD_identity=oid,graph_array_identity=array_hash(a),OD_array_identity=array_hash(o),partitions_identity=array_hash(p),
        physical_graph_identity=r.graph.graph_hash,original_od_identity=r.input_config['input_versions']['od_identity'],
        equilibrium_executed=False,I95_scope='Frozen represented backbone-coordinate mask; residual physical graph remains available',
        witness_feasibility='PENDING_FRESH_REFERENCE_AND_SUPPORT_AUDITS')
    write(out/'RESULT.json',report)
    r.finish(dict(status='PASS',meaningful_output_count=len(a['tail']),common_node_count=len(a['node_ids']),common_edge_count=len(a['edge_ids'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step',choices=['physical','consolidate','topology'])
    parser.add_argument('--config',required=True);args=parser.parse_args()
    r=Run(args.config,args.step)
    try:globals()[args.step](r)
    except Exception as exc:
        write(r.root/(args.step+'_STATUS.json'),dict(status='FAIL',step_id=args.step,run_id=r.c['run_id'],
            error=repr(exc),wall_seconds=time.monotonic()-r.started,peak_rss_bytes=r.peak,partial_outputs_preserved=True))
        raise
if __name__=='__main__':main()
