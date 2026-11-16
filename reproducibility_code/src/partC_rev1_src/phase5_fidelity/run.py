"""Explicit F0-F6 fidelity dispatcher. No implicit task or scientific execution."""
import argparse
import gc
import json
from pathlib import Path
import time
import traceback
import sys
import os
import numpy as np
import pandas as pd
from .context import Context
from .audit import source_invariance,count_addendum,fit_population_addendum,held_state_fit_audit
from .graph import build_hybrid,physical_graph
from .lift import PhysicalLifter,project_witness
from .queries import query_batches,paths_from_rows,array_hash
from partC_rev1_src.phase5.campaign_data import load_accepted,PRIMARY,STRUCTURAL
from partC_rev1_src.phase5.pilot import _suite_frames
from partC_rev1_src.phase5.scc_reachability import load_reachable_od_supply_cache
from partC_rev1_src.phase5.bacn_charges import StaticPathProjector
from partC_rev1_src.phase4.tracing import load_suite_trace_maps
from partC_rev1_src.phase5.campaign_io import sha,frame_write,write_json,npz_write,git,verify_receipt
from partC_rev1_src.phase5.campaign_io import append_event
from .sampling import freeze_sample

VARIANTS=['COMPILED_B_R_FINE','COMPILED_B_R_COARSE','EXACT_DYNAMIC_SHADOW_R_FINE','EXACT_DYNAMIC_SHADOW_R_COARSE']


def publish_identity_summary(out,payload,state):
    """Keep F1's receipt-bound LOW identity immutable when HIGH is added."""
    path=out/'validation'/f'all_B_restored_identity_{state}.json'
    write_json(path,payload)
    states=[json.loads(p.read_text()) for p in sorted((out/'validation').glob('all_B_restored_identity_*.json'))
            if p.name in ['all_B_restored_identity_LOW.json','all_B_restored_identity_HIGH.json']]
    summary=out/'validation'/('all_B_restored_identity.json' if state=='LOW' else 'all_B_restored_identity_both_states.json')
    write_json(summary,dict(status='PASS',states=states,production_adoption=False))


def samples(ctx):
    return pd.read_csv(ctx.out/'sampling/frozen_od_sample.csv',dtype={'origin_uid':str,'destination_uid':str,'stable_od_key':str,'source_sample_od_id':str},keep_default_na=False)


class Run:
    def __init__(self,ctx):
        self.ctx=ctx;self.b=None;self.t_cache={};self.frames_cache={}

    def base(self):
        if self.b is None:
            self.ctx.mark('Loading accepted physical CURRENT states and existing reachability/OD identities')
            self.b=load_accepted(self.ctx.source_config,'s03')
            self.t_cache['s03']=self.b.trace_maps[0]
        return self.b

    def suite(self,suite):
        b=self.base()
        if suite not in self.t_cache:
            p4=self.ctx.repo/'partC_doc_icnn/phase_4_work/phase4_dual_accuracy_benchmark_execution_config.json'
            self.ctx.mark('Loading immutable trace map '+suite)
            self.t_cache[suite]=load_suite_trace_maps(p4,b.graph,suite_ids=[suite],verify_directory_digests=False)[0]
        t=self.t_cache[suite]
        f=_suite_frames(t.root)
        f['terminal_fine_members']=pd.read_parquet(t.root/'terminal_fine_members.parquet')
        links=pd.read_parquet(self.ctx.source/f'suites/{suite}/runtime_links.parquet')
        seq=pd.read_parquet(self.ctx.source/f'suites/{suite}/macro_sequences.parquet')
        return b,t,f,links,seq

    def charges(self,suite,variant):
        ch=pd.read_parquet(self.ctx.source/'registries/charges.parquet')
        return ch.loc[ch.suite_id.eq(suite)&ch.model_variant.eq(PRIMARY if variant=='PRIMARY' else STRUCTURAL)]

    def shadow(self,suite,state):
        df=pd.read_csv(self.ctx.out/'reports/all_fitted_LOW_HIGH_point_queries.csv')
        return df.loc[df.suite_id.eq(suite)&df.state.eq(state)].set_index('link_id').exact_cost_s.to_dict()

    def F0(self):
        import xml.etree.ElementTree as ET
        ctx=self.ctx
        root=ET.parse(ctx.out/'validation/tests.xml').getroot()
        suites=list(root.iter('testsuite'))
        if sum(int(x.attrib.get('failures',0))+int(x.attrib.get('errors',0)) for x in suites):raise ValueError('Regression failure before F0')
        def identity():return [source_invariance(ctx)],dict(status_detail='ALL_BOUND_SOURCE_FILES_HASHED')
        ctx.stage('F0_source',identity,ctx.dependency(modules=['audit.py']))
        ctx.stage('F0_counts',lambda:(count_addendum(ctx),{}),ctx.dependency(modules=['audit.py']))
        ctx.stage('F0_population',lambda:(fit_population_addendum(ctx),{}),ctx.dependency(modules=['audit.py']))
        p=ctx.out/'reports/F0_summary.json'
        write_json(p,dict(status='PASS',tests=sum(int(x.attrib['tests']) for x in suites),source_compiler='COMPLETE_WITH_MODEL_LIMITATIONS',
            source_compiler_rerun=False,full_population=True,next_internal_step='F1',next_external_step_authorized=False))
        return [p],{}

    def F1(self):
        ctx=self.ctx;b=self.base()
        def freeze():
            old=pd.read_parquet(b.sources['frozen_path_sample'])
            supply=load_reachable_od_supply_cache(Path(ctx.source_config['source_pilot_output_root'])/'cache/reachable_od_supply_v2.npz',
                expected_graph_identity=b.graph.graph_hash,expected_od_identity=b.od.od_hash,
                expected_node_count=len(b.graph.node_idx_to_uid),expected_row_count=len(b.od.demand))
            sample=freeze_sample(b,old,supply,ctx.config['sample']['seed'],ctx.mark)
            p=ctx.out/'sampling/frozen_od_sample.csv';frame_write(p,sample)
            design=dict(sample_sha256=sha(p),source_graph_hash=b.graph.graph_hash,source_od_hash=b.od.od_hash,
                stable_od_key='accepted origin UID + | + accepted destination UID',
                sampling='Eight equal-count rank strata by demand and lexical stable OD key; ascending full SHA256(seed+key), first16 SELECT next16 AUDIT',
                variants=VARIANTS,charges=['PRIMARY','STRUCTURAL'],all_five_suites=True,
                restoration_sizes=ctx.config['refinement']['sizes'],ranking_source='LOW_DISCOVERY_ONLY',selection_source='LOW_NEW_SELECT128',
                screening=ctx.config['screen'],high_tunes_nothing=True,frozen_before_any_query=True,
                source_set_counts=sample.source_set.value_counts().to_dict(),cross_scc_reachable_rows=0)
            d=ctx.out/'sampling/design_manifest.json';write_json(d,design)
            states={};costcheck=[]
            for state,physical in [('LOW',b.low_physical),('HIGH',b.high_physical)]:
                exact=b.graph.free_time*60.*(1.+b.graph.alpha*(physical.current_flow/physical.used_capacity)**b.graph.beta)
                if not np.allclose(exact,physical.current_cost_s,rtol=1e-12,atol=1e-7):raise ValueError('One-time seconds conversion/BPR/current cost identity failed')
                states[state+'_cost_s']=physical.current_cost_s;states[state+'_current_flow']=physical.current_flow
                costcheck.append(dict(state=state,flow_role='CURRENT',seconds_conversion_factor=60.,conversion_count=1,
                    capacity_factor_already_applied=.3,max_abs_cost_difference=float(np.max(abs(exact-physical.current_cost_s)))))
            npz_write(ctx.out/'sampling/frozen_current_states.npz',states)
            write_json(ctx.out/'validation/units_and_CURRENT_identity.json',dict(status='PASS',states=costcheck))
            return [p,d,ctx.out/'sampling/frozen_current_states.npz',ctx.out/'validation/units_and_CURRENT_identity.json'],{}
        ctx.stage('F1_sample',freeze,ctx.dependency(modules=['sampling.py','run.py']))
        ctx.stage('F1_fit_queries',lambda:(held_state_fit_audit(ctx),{}),ctx.dependency(modules=['audit.py']))
        sample=samples(ctx)
        pg=physical_graph(b,'LOW')
        prows,preceipts,pkey=query_batches(ctx,pg,sample,b,max_batches=1)
        reference=prows.set_index('stable_od_key').D_phys_s.to_dict()
        b,t,f,links,seq=self.suite('s03');lifter=PhysicalLifter(b,t,f,links,seq,'LOW')
        g=build_hybrid(b,t,f,links,seq,self.charges('s03','PRIMARY'),sample,'LOW',VARIANTS[0],'PRIMARY',self.shadow('s03','LOW'))
        self.project_old(g,lifter,'LOW')
        rows,receipts,key=query_batches(ctx,g,sample,b,lifter,reference,max_batches=1)
        identity=self.identity(sample,prows,b,t,f,links,seq,'LOW')
        first=receipts[0]
        seconds_per_origin=first['elapsed_seconds']/first['unique_origins']
        expected_queries=98*sample.origin_idx.nunique()
        forecast=1.5*seconds_per_origin*expected_queries+1800
        elapsed=time.time()-ctx.config['task_started_unix']
        available=ctx.config['resources']['base_seconds']-ctx.config['resources']['sealing_reserve_seconds']-elapsed
        if forecast>available:raise RuntimeError(f'Measured 32-origin forecast {forecast}s exceeds {available}s; stage segmentation/throughput diagnosis required')
        p=ctx.out/'validation/32_origin_preflight.json'
        write_json(p,dict(status='PASS',unique_origins=first['unique_origins'],OD_rows=first['OD_rows'],actual_graph_nodes=len(g.nodes),actual_graph_edges=len(g.edges),
            measured_seconds_per_origin=seconds_per_origin,forecast_seconds_with_50pct_margin_and_1800s_closeout=forecast,
            available_science_seconds=available,expected_main_graph_configurations=98,workers=1,batch_origin_cap=32,
            all_B_identity=identity['status'],no_assignment=True))
        return [p,ctx.out/'validation/all_B_restored_identity.json'],dict(preflight='ADMITTED')

    def identity(self,sample,physical_rows,b,t,f,links,seq,state):
        ctx=self.ctx
        identity_sample=sample.set_index('stable_od_key').loc[physical_rows.stable_od_key].reset_index().iloc[:32].copy()
        graph=build_hybrid(b,t,f,links,seq,self.charges('s03','PRIMARY'),sample,state,'RESTORED_B_R_FINE','PRIMARY',restored=t.bacn_ids)
        lifter=PhysicalLifter(b,t,f,links,seq,state)
        reference=physical_rows.set_index('stable_od_key').D_phys_s.to_dict()
        rows,_,key=query_batches(ctx,graph,identity_sample,b,lifter,reference)
        error=pd.to_numeric(rows.D_hybrid_s)-pd.to_numeric(rows.D_phys_s)
        lift_error=pd.to_numeric(rows.D_lift_s)-pd.to_numeric(rows.D_phys_s)
        tolerance=1e-7+1e-10*np.maximum(1.,abs(pd.to_numeric(rows.D_phys_s)))
        if len(rows)!=32 or rows.new_unreachable.any() or not rows.lift_status.eq('VERIFIED').all() or (abs(error)>tolerance).any() or (abs(lift_error)>tolerance).any():
            raise ValueError('All-B-restored identity differs from physical reference')
        for row in rows.itertuples(index=False):
            physical_path=paths_from_rows(ctx.out,physical_rows.loc[physical_rows.stable_od_key.eq(row.stable_od_key)])[row.stable_od_key]
            projected=project_witness(graph,lifter,physical_path,row.origin_idx,row.destination_idx)
            if abs(graph.cost[projected].sum()-row.D_phys_s)>1e-7+1e-10*max(1.,row.D_phys_s):raise ValueError('All-B physical witness projection cost mismatch')
        payload=dict(status='PASS',state=state,OD_count=32,all_original_arcs_owned_once=True,all_fixed_charges_zero=True,
            graph_key=key,physical_arc_count=len(b.graph.tail),max_distance_error_s=float(abs(error).max()),
            max_lift_error_s=float(abs(lift_error).max()),production_adoption=False)
        publish_identity_summary(ctx.out,payload,state)
        return payload

    def project_old(self,graph,lifter,state):
        ctx=self.ctx;b=self.base()
        dest=ctx.out/'projections'/(graph.fingerprint+'.csv')
        if dest.exists():return dest
        sample=samples(ctx);old=sample.loc[sample.source_set.eq('DISCOVERY_OLD200')].set_index('source_sample_od_id')
        tables=[('FROZEN_FREE_FLOW',pd.read_parquet(b.sources['frozen_path_sample']),'arc_indices_json')]
        if state=='HIGH':tables.append(('HIGH_SELECTED',pd.read_parquet(Path(b.sources['high_benchmark']['root'])/'routes/sampled_route_switches.parquet'),'selected_arc_indices_json'))
        rows=[]
        for role,table,column in tables:
            for i,r in enumerate(table.itertuples(index=False)):
                arcs=np.asarray(json.loads(getattr(r,column)),np.int64)
                p=project_witness(graph,lifter,arcs,int(r.origin_idx),int(r.destination_idx))
                if len(p) and np.any(graph.head[p[:-1]]!=graph.tail[p[1:]]):raise ValueError('Projected original walk discontinuity')
                rows.append(dict(stable_od_key=old.loc[str(r.sample_od_id)].stable_od_key,source_sample_od_id=str(r.sample_od_id),
                    graph_fingerprint=graph.fingerprint,state=state,source_role=role,status='PASS',physical_arc_count=len(arcs),
                    projected_edge_count=len(p),projected_cost_s=float(graph.cost[p].sum()),projected_edge_hash=array_hash(p)))
                if i%64==0:ctx.mark(f'Original witness projection {graph.metadata["suite"]} {graph.metadata["variant"]} {state} {role} {i+1}/200')
        frame_write(dest,pd.DataFrame(rows));return dest

    def physical(self,state):
        ctx=self.ctx;b=self.base();sample=samples(ctx)
        graph=physical_graph(b,state)
        rows,receipts,key=query_batches(ctx,graph,sample,b)
        frame_write(ctx.out/f'results/PHYSICAL_{state}.csv',rows)
        return rows

    def hybrid(self,suite,state,variant,charge,physical_rows,restored=(),suite_data=None,lifter=None):
        ctx=self.ctx;sample=samples(ctx)
        b,t,f,links,seq=self.suite(suite) if suite_data is None else suite_data
        if lifter is None:lifter=PhysicalLifter(b,t,f,links,seq,state)
        graph=build_hybrid(b,t,f,links,seq,self.charges(suite,charge),sample,state,variant,charge,
            self.shadow(suite,state),restored)
        self.project_old(graph,lifter,state)
        rows,receipts,key=query_batches(ctx,graph,sample,b,lifter,physical_rows.set_index('stable_od_key').D_phys_s.to_dict())
        result=ctx.out/'results/configurations'/f'{suite}_{state}_{variant}_{charge}_M{len(restored)}.csv'
        if result.exists():
            previous=pd.read_csv(result)
            if not previous.graph_key.eq(key).all():raise ValueError('Changed graph requires versioned dependent-output invalidation')
        frame_write(result,rows)
        return rows,result

    def core(self,state):
        ctx=self.ctx;physical=self.physical(state);files=[ctx.out/f'results/PHYSICAL_{state}.csv']
        for suite in ctx.config['suites']:
            data=self.suite(suite);b,t,f,links,seq=data
            lifter=PhysicalLifter(b,t,f,links,seq,state)
            for variant in VARIANTS:
                for charge in ['PRIMARY','STRUCTURAL']:
                    rows,result=self.hybrid(suite,state,variant,charge,physical,suite_data=data,lifter=lifter);files.append(result)
            gc.collect()
        return files

    def F2(self):return self.core('LOW'),dict(all_five_suites=True)

    def F3(self):
        from .report import rank_B,freeze_selection,candidate_summary,interface_impact
        ctx=self.ctx;physical=pd.read_csv(ctx.out/'results/PHYSICAL_LOW.csv',float_precision='round_trip');data=self.suite('s03')
        b,t,f,links,seq=data;lifter=PhysicalLifter(b,t,f,links,seq,'LOW')
        results=[]
        for charge in ['PRIMARY','STRUCTURAL']:
            rows,p=self.hybrid('s03','LOW','MEMBER_B_R_FINE',charge,physical,suite_data=data,lifter=lifter);results.append(p)
        ranking=rank_B(ctx,b,t,f,physical,self.charges('s03','PRIMARY'))
        manifest=ctx.out/'refinement/candidate_ladder_frozen.json'
        ladder={str(m):ranking.head(m).bacn_id.tolist() for m in ctx.config['refinement']['sizes']}
        write_json(manifest,dict(ranking_source='LOW_DISCOVERY_ONLY',ranking_sha256=sha(ctx.out/'refinement/low_frozen_B_ranking.csv'),
            sample_sha256=sha(ctx.out/'sampling/frozen_od_sample.csv'),candidate_sets=ladder,
            M0_alias='MEMBER_B_R_FINE',selection_source='LOW_NEW_SELECT128',high_or_audit_changes_selection=False))
        for m in [16,64,256]:
            for charge in ['PRIMARY','STRUCTURAL']:
                _,p=self.hybrid('s03','LOW','RESTORED_B_R_FINE',charge,physical,ladder[str(m)],data,lifter);results.append(p)
        selection=freeze_selection(ctx)
        impact=interface_impact(ctx,b,t,f,ladder)
        return results+[manifest,ctx.out/'refinement/low_frozen_B_ranking.csv',ctx.out/'refinement/selection.json',impact],dict(selection=selection)

    def F4(self):
        ctx=self.ctx
        if not (ctx.out/'refinement/selection.json').exists():raise ValueError('Low-only selection must be frozen before HIGH queries')
        before=sha(ctx.out/'refinement/selection.json');files=self.core('HIGH')
        physical=pd.read_csv(ctx.out/'results/PHYSICAL_HIGH.csv',float_precision='round_trip');data=self.suite('s03');b,t,f,links,seq=data
        lifter=PhysicalLifter(b,t,f,links,seq,'HIGH')
        for charge in ['PRIMARY','STRUCTURAL']:
            _,p=self.hybrid('s03','HIGH','MEMBER_B_R_FINE',charge,physical,suite_data=data,lifter=lifter);files.append(p)
        ladder=json.loads((ctx.out/'refinement/candidate_ladder_frozen.json').read_text())['candidate_sets']
        for m in [16,64,256]:
            for charge in ['PRIMARY','STRUCTURAL']:
                _,p=self.hybrid('s03','HIGH','RESTORED_B_R_FINE',charge,physical,ladder[str(m)],data,lifter);files.append(p)
        self.identity(samples(ctx),physical,b,t,f,links,seq,'HIGH')
        if sha(ctx.out/'refinement/selection.json')!=before:raise ValueError('HIGH changed frozen selection')
        write_json(ctx.out/'validation/low_selection_invariance_to_HIGH.json',dict(status='PASS',selection_before_sha256=before,selection_after_sha256=before,
            HIGH_role='REUSED_SENSITIVITY_NOT_PRISTINE_TEST',audit_and_HIGH_used_for_selection=False))
        return files+[ctx.out/'validation/low_selection_invariance_to_HIGH.json'],{}

    def F5(self):
        from .report import final_report
        paths=final_report(self.ctx)
        paths.append(source_invariance(self.ctx,final=True))
        return paths,{}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--stage',required=True,choices=['F0','F1','F2','F3','F4','F5'])
    args=parser.parse_args();ctx=Context(args.config)
    append_event(ctx.out/'validation/command_log.jsonl',dict(command=[sys.executable,'-m','partC_rev1_src.phase5_fidelity.run',*sys.argv[1:]],
        phase_id='phase_05',step_id=args.stage,run_id=ctx.config['run_id'],pid=os.getpid(),config_sha256=sha(ctx.config_path)))
    try:
        if git(ctx.repo,'branch','--show-current')!=ctx.config['branch']:raise ValueError('Wrong branch')
        if args.stage!='F0':
            checkpoints=[json.loads((ctx.out/'validation/pre_run_checkpoint.json').read_text())]
            checkpoints += [json.loads(p.read_text()) for p in sorted((ctx.out/'engineering_epochs').glob('epoch_*/checkpoint.json'))]
            if not any(checkpoint['code']==ctx.code for checkpoint in checkpoints):raise ValueError('Code changed after checkpoint; evidenced repair checkpoint required')
        run=Run(ctx)
        ctx.stage(args.stage,getattr(run,args.stage))
    except BaseException as exc:
        folder=ctx.out/'failures';folder.mkdir(exist_ok=True)
        name=time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+args.stage
        (folder/(name+'.txt')).write_text(traceback.format_exc(),encoding='utf-8')
        write_json(folder/(name+'.json'),dict(stage=args.stage,exception=repr(exc),activity=ctx.activity,
            status='INTERNAL_FAILURE_REQUIRES_DIAGNOSIS',time_unix=time.time()))
        raise
    finally:ctx.close()


if __name__=='__main__':main()
