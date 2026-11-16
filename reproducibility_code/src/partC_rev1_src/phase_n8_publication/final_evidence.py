"""Outcome inventories and final immutable-source/publication admission gates."""
from pathlib import Path
import json
import shutil
import time
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .publication_pipeline import ASSETS


def publication_gate(ctx):
    root=ctx.root;pub=root/'publication_bundle'
    automatic=read_json(pub/'AUTOMATED_VERIFICATION.json');qa=Path(automatic['qa_root'])
    isolation=read_json(qa/'ISOLATION_RESULT.json')
    if isolation['status']!='PASS' or {a['asset'] for a in isolation['assets']}!=set(ASSETS):raise ValueError('Seventeen isolation passes required')
    for asset in isolation['assets']:
        if asset['status']!='PASS' or len(asset['runs'])!=2:raise ValueError('Incomplete independent reproduction')
        for run in asset['runs']:
            if not run['runtime_access'].get('module_origins_validated'):raise ValueError('Isolated module origins not checked')
        for name,h in asset['verified_files'].items():
            if sha(pub/'numerical_data'/asset['asset']/name)!=h:raise ValueError('Publication asset changed since isolation: '+name)
    manifest=pd.read_csv(pub/'figure_manifest.csv')
    if set(manifest.asset)!=set(ASSETS):raise ValueError('Final figure/table file manifest incomplete')
    for row in manifest.itertuples():
        if sha(pub/row.relative_path)!=row.sha256:raise ValueError('Final file manifest mismatch: '+row.relative_path)
    visual=read_json(root/'validation/VISUAL_QA.json')
    pdf=read_json(qa/'PDF_BUILD.json')
    if visual['status']!='PASS' or visual['main_pdf_sha256']!=sha(pub/'main_numerical.pdf'):raise ValueError('Actual current PDF visual inspection required')
    if pdf['pdf_sha256']!=sha(pub/'main_numerical.pdf'):raise ValueError('Current PDF differs from compiled/rendered PDF')
    if visual['figure_manifest_sha256']!=sha(pub/'figure_manifest.csv'):raise ValueError('Assets changed after visual inspection')
    for name,h in pdf['tex_source_hashes'].items():
        if sha(pub/name)!=h:raise ValueError('TeX source changed after compilation: '+name)
    if set(visual['assets_inspected'])!=set(ASSETS) or set(visual['pages_inspected'])!=set(range(1,pdf['pages']+1)):
        raise ValueError('Visual inspection must cover every asset and numerical PDF page')
    review=automatic.get('v5_review_copy',{})
    if review.get('status')=='COMPILED_REVIEW_COPY':
        if visual.get('v5_pdf_sha256')!=review['pdf_sha256'] or review['pdf_sha256']!=sha(pub/'NUMERICAL_REPLACEMENT_REVIEW_COPY.pdf'):
            raise ValueError('Current v5 review-copy PDF requires visual inspection')
        if review['tex_sha256']!=sha(pub/'NUMERICAL_REPLACEMENT_REVIEW_COPY.tex'):
            raise ValueError('v5 review-copy source changed after compilation')
        if set(visual.get('v5_pages_inspected',[]))!=set(range(1,review['pages']+1)):
            raise ValueError('Inspect every compiled v5 review-copy page')
    from .context import code_identity
    tests=read_json(root/'validation/TESTS.json')
    if tests['status']!='PASS' or tests['code_identity']!=code_identity(ctx.repo)[0]:raise ValueError('Current code requires matching regression PASS')
    if automatic['arithmetic']['status']!='PASS' or read_json(root/'metrics/RESULT.json')['status']!='PASS':raise ValueError('Numerical arithmetic gate failed')
    return dict(status='PASS',assets=17,main_pdf_pages=pdf['pages'],manifest_sha256=sha(pub/'figure_manifest.csv'),
        isolation_receipt_sha256=sha(qa/'ISOLATION_RESULT.json'),visual_receipt_sha256=sha(root/'validation/VISUAL_QA.json'))


def source_seal(ctx):
    records={};excluded=[]
    for receipt in sorted((ctx.root/'receipts').glob('*_SOURCE_BINDINGS.json')):
        for row in read_json(receipt):
            p=Path(row['native_path']).resolve()
            if p in {ctx.root/'registries/M1_CURRENT.json',ctx.root/'logs/resource_usage.jsonl'}:
                excluded.append(dict(native_path=str(p),reason='NAMED_MUTABLE_POINTER_OR_LIVE_LOG; FROZEN_REGISTRIES_AND_RESOURCE_SNAPSHOT_VERIFIED_SEPARATELY'))
                continue
            key=str(p)
            if key in records and records[key]['sha256']!=row['sha256']:raise ValueError('Conflicting immutable source versions: '+key)
            records[key]=row
    checked=[]
    for i,(key,row) in enumerate(sorted(records.items())):
        if i%64==0:ctx.guard()
        p=Path(key);st=p.stat();same=st.st_size==row['size_bytes'] and st.st_mtime_ns==row['mtime_ns']
        # Previously authenticated large files may reuse their digest only when
        # BOTH size and nanosecond modification time remain unchanged. Never
        # rehash the national GIS database for a packaging convenience check.
        metadata_reuse=same and st.st_size>256_000_000
        observed=row['sha256'] if metadata_reuse else sha(p)
        if observed!=row['sha256']:raise ValueError('Immutable source changed: '+key)
        checked.append(dict(native_path=key,sha256=observed,size_bytes=st.st_size,mtime_ns=st.st_mtime_ns,
            verification='PRIOR_DIGEST_WITH_UNCHANGED_SIZE_MTIME' if metadata_reuse else 'REHASHED_AT_CLOSEOUT',status='PASS'))
    pd.DataFrame(checked).to_csv(ctx.root/'IMMUTABLE_SOURCE_SEAL.csv',index=False)
    result=dict(status='PASS',sources=len(checked),rehash_count=sum(r['verification']=='REHASHED_AT_CLOSEOUT' for r in checked),
        large_unchanged_metadata_count=sum(r['verification']!='REHASHED_AT_CLOSEOUT' for r in checked),
        named_mutable_bindings_excluded=excluded,current_output_proof='All immutable current-run input bindings rechecked against scientific/extraction hashes; figure_manifest.csv; NATIVE_ARTIFACT_INDEX.csv',
        sealed_epoch=time.time(),scientific_source_mutations=0)
    write_json(ctx.root/'IMMUTABLE_SOURCE_SEAL.json',result);return result


def inventories(ctx):
    root=ctx.root;pub=root/'publication_bundle'
    opened=pd.read_csv(root/'metrics/open_summary_metrics.csv',float_precision='round_trip')
    closed=pd.read_csv(root/'metrics/closed_summary_metrics.csv',float_precision='round_trip')
    combined=pd.concat([opened.assign(network_state='OPEN'),closed.assign(network_state='CLOSED')],ignore_index=True)
    b=combined.loc[combined.population.eq('BACKBONE')].copy()
    intended={('OPEN','s03',m) for m in ('REFERENCE_X45','PHYSICAL_CFW','PHYSICAL_FW','PHYSICAL_MSA','M1','M2','M3')}
    intended.update(('OPEN',s,'M3') for s in ctx.c['active_suites'])
    intended.update(('CLOSED','s03',m) for m in ('M1','M2','M3','REFERENCE_CLOSED'))
    observed={(r.network_state,r.suite_id,r.method_id) for r in b.itertuples()}
    columns=[c for c in ('method_id','method_label','suite_id','network_state','status','state_id','reference_id','graph_identity','mapping_identity','registry_identity','reference_RG_SPTT','RG_SPTT','online_seconds','timing_scope','represented_demand','original_demand') if c in b]
    b[columns].to_csv(root/'METHOD_REGISTRY.csv',index=False)
    outcomes=[]
    def outcome(experiment,method,status,artifact,**kw):
        outcomes.append(dict(experiment_id=experiment,method_or_case=method,status=status,artifact=artifact,**kw))
    outcome('E0','All five native forward references and baseline-only keys','PASS','reference/REFERENCE_FREEZE.json',experiment_complete=True)
    for r in b.itertuples():
        exp='E6' if r.network_state=='CLOSED' else ('E1' if r.suite_id=='s03' else 'E5')
        outcome(exp,r.suite_id+'/'+r.method_id,r.status,'metrics/'+('closed' if r.network_state=='CLOSED' else 'open')+'_summary_metrics.csv',
            experiment_complete=True,measured_gap=r.RG_SPTT,online_seconds=r.online_seconds,
            reused=r.network_state=='OPEN' and r.method_id not in ('M1','M2'),state_id=r.state_id,network_state=r.network_state)
    for state,sid,method in sorted(intended-observed):
        outcome('E6' if state=='CLOSED' else ('E1' if sid=='s03' else 'E5'),sid+'/'+method,
            'NO_ADMITTED_RESULT','jobs/;receipts/;run_matrix.csv',experiment_complete=False,network_state=state,
            limitation='Intended case lacks an admitted metric row; failed attempts and unexecuted planned jobs remain separately recorded.')
    outcome('E2','Complete backbone and fixed200 visual comparisons','PASS','publication_bundle/numerical_data/figure3',experiment_complete=True)
    costs=pd.read_csv(root/'regional_metrics/region_summary.csv') if (root/'regional_metrics/region_summary.csv').exists() else None
    reg=read_json(root/'regional_metrics/RESULT.json')
    for kind in ('FFTT','POWER4','BEST_NONCONSTANT_FALLBACK','SKIP_ICNN_WITHOUT_FALLBACK'):
        outcome('E3',kind,reg['status'],'regional_metrics/RESULT.json',experiment_complete=reg['status']=='EXECUTED')
    fits=read_json(root/'learning/COMBINED_FIT_INVENTORY.json')
    for fit in fits:
        outcome('E4',fit['region_id']+'/fraction='+str(fit['fraction']),fit['status'],'learning/COMBINED_FIT_INVENTORY.json',
            experiment_complete=fit['status']=='COMPLETE',reused=bool(fit.get('reused',fit.get('fit_reused',False))))
    learning=read_json(root/'receipts/C02_LEARNING_COMPLETION_RESULT.json')
    outcome('E4','Post-freeze label evaluation',learning['evaluation']['status'],'learning/EVALUATION_RESULT.json',
        experiment_complete=learning['evaluation']['status']=='EXECUTED')
    outcome('E4','Direct native cost-WMAPE export',learning['costs']['status'],'learning/cost_wmape/RESULT.json',
        experiment_complete=learning['costs']['status']=='PASS')
    for sid in ctx.c['active_suites']:
        selected=b.loc[b.suite_id.eq(sid)&b.method_id.eq('M3')&b.network_state.eq('OPEN')]
        if len(selected)>1:raise ValueError('Five-resolution matrix duplicates '+sid)
        if sid=='s03':outcome('E5','s03/M3',selected.iloc[0].status if len(selected) else 'NO_ADMITTED_RESULT',
            'metrics/open_summary_metrics.csv',experiment_complete=bool(len(selected)),reused=True,network_state='OPEN')
    outcome('E7','Measured boundaries, offline coverage and amortization','NOT_COMPARABLE_FOR_COLD_START_AMORTIZATION',
        'publication_bundle/numerical_data/table6',experiment_complete=True,limitation='Offline attribution and initialization boundaries incomplete; no crossing claimed')
    previous=root/'EXPERIMENT_INVENTORY.csv'
    if previous.exists() and not (root/'EXPERIMENT_INVENTORY_INITIAL_PLAN.csv').exists():shutil.copy2(previous,root/'EXPERIMENT_INVENTORY_INITIAL_PLAN.csv')
    pd.DataFrame(outcomes).to_csv(previous,index=False)
    epochs=[]
    for p in sorted((root/'runs').glob('*/EPOCHS.json')):
        epochs.extend(dict(r,job_id=p.parent.name,source_sha256=sha(p)) for r in read_json(p))
    pd.DataFrame(epochs).to_csv(root/'FALLBACK_EPOCHS.csv',index=False)
    mapping=[];keys=[]
    for sid in ctx.c['active_suites']:
        path=root/'reference'/sid/'REFERENCE_AUDIT.json';d=read_json(path)
        mapping.append(dict({k:v for k,v in d.items() if not isinstance(v,(dict,list))},audit_path=str(path),audit_sha256=sha(path)))
        k=pd.read_csv(root/'reference'/sid/'key_links_200.csv',dtype={'abstract_link_id':str});k['suite_id']=sid
        k['selection_freeze_sha256']=sha(root/'reference'/sid/'KEY_SELECTION_FREEZE.json');keys.append(k)
    pd.DataFrame(mapping).to_csv(root/'MAPPING_REFERENCE_AUDIT.csv',index=False)
    pd.concat(keys,ignore_index=True).to_csv(root/'KEY_LINK_SELECTION.csv',index=False)
    manifest=pd.read_csv(pub/'figure_manifest.csv');manifest['native_path']=manifest.relative_path.map(lambda p:str(pub/p))
    manifest.to_csv(root/'FIGURE_TABLE_INVENTORY.csv',index=False)
    crosswalk=pd.read_csv(pub/'reviewer_point_crosswalk.csv')
    refs={'Correct numerical estimand':('5.2;5.4;5.5','Figure3;Figure6;Table2;Table4;Table5','Projected approximate CURRENT is not physical or observed traffic truth'),
        'Practical safety fallback':('5.2','FigureS3;Table2','Final analytic rescue retains zero neural regions; no national neural-performance claim'),
        'Actual I-95 response':('5.5','Figure7;FigureS4;Table5','608 hidden interior arcs omitted; closed control approximate and demand reachability differs'),
        'Learning curves and costs':('5.3','Figure4;Figure5;FigureS1;Table3','Selected sixteen regions; severe/missing labels and now-seen cohorts preserved'),
        'Timing and empirical limits':('5.1;5.6','Figure8;Table1;Table6;TableS1','Unequal initialization clocks and incomplete attributable offline cost; no amortization or equal-accuracy speedup')}
    for col,pos in [('section',0),('figure_table',1),('residual_limitation',2)]:crosswalk[col]=crosswalk.reviewer_point.map(lambda x:refs[x][pos])
    extra=[
        ('Data lineage','5.1','Table1;TableS1','Factor counties, matched centroid records, graph nodes, original OD rows and aggregated rows are separately reported.','Factor release year is not established; upstream exclusions remain recorded.'),
        ('Passenger/background and long-haul realism','5.1;5.6','Table1','Capacity preprocessing and the once-applied 0.30 multiplier are preserved.','Static annual assignment does not model dynamic passenger equilibrium, parking, delivery windows or hours of service; AADT is a capacity input, not independent validation.'),
        ('Stronger algorithm baselines','5.2;5.6','Table2;TableS1','Same-codebase CFW/FW/MSA controls have fresh gaps, achieved outcomes and explicit clock boundaries.','No measured bush-based or origin-based algorithm comparison; time limits and initialization boundaries differ.'),
        ('Convexity, Jacobian and convergence','5.2','Figure3;FigureS3','One scalar potential per epoch; tested gradients/HVPs and fresh same-state certificates.','A small surrogate gap does not certify physical or empirical accuracy; abandoned neural models are not globally nonnegative.'),
        ('Regional versus national error','5.2;5.3','Table2;Table3','Native cost WMAPE and national backbone flow WMAPE use distinct populations and denominators.','No causal error-propagation theorem or unconditional strong-monotonicity bound is asserted.'),
        ('Partition sensitivity','5.4','Figure6;FigureS2;Table4','Five actual constant-region suites retain their graph-specific denominators and full stability population.','No trained models at the other resolutions; map thinning is separate from statistical coverage.'),
        ('Boundary feasibility and conservation','5.2;5.5','Figure7;Table5','Native role/order/support and served-demand conservation are checked; unavailable coordinates carry zero flow.','Suppressed backbone interiors and regional abstractions do not preserve every physical turn; no predicted physical-flow recovery used.'),
        ('I95 event and field validation','5.5;5.6','Figure7;FigureS4;Table5','Actual represented-subset counterfactual and matched physical closure comparison.','No observed event validation or fabricated external count/year data.'),
        ('Public reproducibility','5.1;Numerical supplement','All seventeen folders;claim_to_data.csv','Each folder regenerated alone twice from local CSVs; exact sources and output hashes are recorded.','Prepared local release is not a published repository or DOI.'),
        ('Nonnumerical manuscript revisions','Later author edits','LATER_MANUSCRIPT_EDITS.md','A separate Figure1/terminology/literature/Remark2/method-theory list is supplied.','These nonnumerical revisions and proof changes were outside the authorized scope.'),
        ('Emissions and Key Bridge scope','5.6;scope annex','NUMERICAL_SCOPE_CHANGES.md','These studies are excluded from the v5-aligned numerical sequence.','No emissions factor, welfare, event-diversion or field-validation claim is supplied.')]
    crosswalk=pd.concat([crosswalk,pd.DataFrame([dict(reviewer_point=p,section=s,figure_table=f,response=a,residual_limitation=l,claims='See specified asset') for p,s,f,a,l in extra])],ignore_index=True)
    crosswalk.to_csv(root/'REVIEWER_RESPONSE_CROSSWALK.csv',index=False)
    (root/'LATER_MANUSCRIPT_EDITS.md').write_text('''# Later author edits outside this numerical amendment

- **Figure 1:** reconcile the conceptual network illustration and caption with the retained active support and actual abstraction roles. The optional v5 review copy uses only the exact supplied historical rasters and does not claim this revision complete.
- **Terminology:** harmonize ACN/ACNHN, backbone, regional terminal pairs, native cost WMAPE, flow WMAPE and approximate reference throughout the abstract, introduction, model and conclusion.
- **Literature:** revise the literature review and algorithm-comparison positioning. No new literature search or unimplemented solver benchmark was part of this run.
- **Remark 2:** review its mathematical assumptions, particularly any strong-monotonicity or error-transfer statement. This amendment supplies no new proof and does not apply such a bound unconditionally.
- **Method/theory sections:** reconcile whole-region fallback, immutable objective epochs, restart semantics, final analytic rescue and the scope of convergence certificates with the formal presentation. Any mathematical change needs separate authorization.
- **Remaining text and submission metadata:** reconcile abstract, highlights, introduction, conclusion and statements with the measured numerical outcomes; replace any historical repository placeholder only after a real publication exists.

The numerical review crosswalk records supported results and residual limitations. It is not a final rebuttal letter. No next round, emissions study or Key Bridge study is authorized by this handoff.
''',encoding='utf-8')
    (root/'NUMERICAL_SCOPE_CHANGES.md').write_text('''# Numerical scope changes from supplied v5

Section 5 preserves the setting, backbone validation, regional learning, resolution, I-95 and computational-cost sequence. Figure purposes, rather than obsolete filename references or numerical values, determine the new assets.

- Assignment KPIs now compare native abstract flows with the forward projection of the fixed physical CURRENT reference. No predicted physical-flow reconstruction is used. Full backbone coordinates include reference zeros; the 200 keys were frozen from the reference alone.
- The actual reference self-row is zero by computation. Independent time-limited CFW/FW/MSA controls remain separate measured rows.
- The fixed VALID nonconstant portfolio uses whole-region permanent fallback and bounded analytic rescue. Its final 1,322 current and 244 legacy power-4 regions contain no neural regions. Open and closed use the same final registry. A fresh final-registry open solve avoids counting a rescued warm flow as an independent online query.
- Five existing constant-region resolutions retain their exact graph-specific denominators; no new trained resolution or active-support expansion was introduced.
- Regional cost WMAPE comes from native cost predictions/truths. MARE, optimizer losses, selected updates and executed updates are distinct. Forty old fits are unchanged; only the missing frozen eight-by-five slots were eligible for new fits under a shared three-hour cap.
- The actual represented I-95 scenario disables 869 abstract coordinates and exactly 1,105 represented physical source arcs. It omits 608 suppressed interior arcs and retains 55 healthy alternatives inside disabled macros. Redistribution and matched closed-reference accuracy are separate.
- Figure8 and supplements expose timing boundaries, fixed-data optimization, ownership stability, fallback and the remaining closure methods. Incompatible cold-start boundaries and incomplete offline attribution preclude an amortization crossing.
- All seventeen publication folders regenerate independently from local CSVs. The numerical-only PDF is current; any v5 derivative retains a visible warning that its nonnumerical sections require revision.

The original v5 archive and all authoritative earlier data, graphs, models and result files are preserved. This document is a numerical review handoff, not whole-paper acceptance or authorization for another round.
''',encoding='utf-8')
    primary_complete=intended<=observed
    result=dict(status='PASS',primary_matrix_complete=primary_complete,assignment_cases=len(b),fit_slots=len(fits),
        experiments=sorted({r['experiment_id'] for r in outcomes}),required_outcome_rows=len(outcomes),
        experiment_complete={e:all(r.get('experiment_complete',False) for r in outcomes if r['experiment_id']==e) for e in sorted({r['experiment_id'] for r in outcomes})})
    write_json(root/'INVENTORY_RESULT.json',result);return result
