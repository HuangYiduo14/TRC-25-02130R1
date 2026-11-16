"""Extract measured publication observations; drawing code remains independent."""
from pathlib import Path
import csv
import json
import shutil
import time
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .path_metrics import cases,missing_state

LABELS={'N8R4_PRIMARY_BOUND_SAFE_V1':'s03 primary safe','N8R4_BEST_ANY_BOUND_SAFE_V1':'s03 best-any safe',
    'N8R4_NONCONSTANT_SOURCE_BOUND_SAFE_V1':'s03 nonconstant safe','S03_ANALYTIC_VALID':'s03 analytic',
    'E2_PRIMARY_FROZEN':'Original primary','E2_BEST_ANY_FROZEN':'Original best-any',
    'N8_PROPOSED_NONCONSTANT_VALID_V1':'Original nonconstant','N8_PROPOSED_ANALYTIC_SAFE_VALID_V1':'Analytic',
    'PHYSICAL_CFW':'Physical safeguarded CFW','PHYSICAL_FW':'Physical FW','PHYSICAL_MSA':'Physical MSA',
    'ORIGINAL_PRIMARY_WARM':'Original primary warm','ORIGINAL_BEST_ANY_WARM':'Original best-any warm',
    'ORIGINAL_NONCONSTANT_WARM':'Original nonconstant warm'}
for _s in ('s02','s03','s04','s05','s06'):LABELS['FFTT_'+_s]=_s+' FFTT'


def short(value):return LABELS.get(str(value),str(value))


def paired_regional(frame,registry_count):
    if frame.duplicated(['region_id','cohort','registry']).any():raise ValueError('Duplicated regional registry/cohort metric')
    matched=frame.groupby(['region_id','cohort']).registry.transform('nunique')
    return frame.loc[matched.eq(registry_count)].copy()


def available_diagnostics(release,fig,folder,names):
    rows=[]
    for name in names:
        p=folder/name;exists=p.is_file()
        rows.append(dict(expected_file=name,status='PRESENT' if exists else 'NOT_PRODUCED',
            size_bytes=p.stat().st_size if exists else None,sha256=sha(p) if exists else None))
        if exists:
            try:release.copy(fig,p)
            except pd.errors.EmptyDataError:
                release.folder(fig);release.source(fig,p)
                rows[-1]['status']='PRESENT_WITHOUT_CSV_SCHEMA_NO_MEASURED_OBSERVATIONS'
    source=folder/'RESULT.json'
    if not source.exists():source=release.ctx.root/'jobs/E03_BOUNDARY_PATH_DIAGNOSTICS/EXIT.json'
    release.csv(fig,'diagnostic_file_coverage.csv',pd.DataFrame(rows),
        'Actual diagnostic artifacts; absent outputs are not interpreted as successful empty results',[source])
    return source


class Release:
    def __init__(self,ctx):
        self.ctx=ctx;self.root=ctx.root/'numerical_release';self.root.mkdir(exist_ok=True)
        self.sources={};self.specs={};self.notes={};self.files={}

    def folder(self,name):
        p=self.root/name;p.mkdir(exist_ok=True)
        self.sources.setdefault(name,{});self.specs.setdefault(name,[]);self.files.setdefault(name,[])
        return p

    def source(self,name,path):
        p=Path(path);h=self.ctx.bind(p,'PUBLICATION_SOURCE_OBSERVATIONS')
        self.sources[name][str(p)]=h

    def read(self,path):
        return pd.read_csv(path,float_precision='round_trip',dtype={'physical_arc_id':str,'physical_node_uid':str,
            'region_id':str,'input_hash':str,'stable_od_key':str})

    def csv(self,folder,name,data,description,source_paths=()):
        dst=self.folder(folder)
        for p in source_paths:self.source(folder,p)
        # Keep each ordinary CSV member comfortably below the 25 MB packaging cap.
        sample=data.head(1000).to_csv(index=False,float_format='%.17g').encode('utf-8')
        approximate_per_row=max(1.,len(sample)/max(1,len(data.head(1000))))
        files=[];step=max(1,min(40000,int(18_000_000/approximate_per_row)))
        if len(data)<=step:
            pieces=[data]
        else:pieces=[data.iloc[i:i+step] for i in range(0,len(data),step)]
        for i,piece in enumerate(pieces):
            filename=name if len(pieces)==1 else Path(name).stem+f'_{i:03d}.csv'
            path=dst/filename;piece.to_csv(path,index=False,float_format='%.17g',encoding='utf-8')
            if path.stat().st_size>=25_000_000:raise ValueError('Publication CSV member too large: '+str(path))
            files.append(filename);self.files[folder].append(dict(file=filename,description=description,rows=len(piece),
                columns=list(piece.columns),sha256=sha(path),bytes=path.stat().st_size))
        return '|'.join(files)

    def copy(self,folder,path,name=None,description='Unaggregated analysis-ready observations'):
        return self.csv(folder,name or Path(path).name,self.read(path),description,[path])

    def panel(self,folder,panel,kind,data,title,x='',y='',group='',**kwargs):
        supplied={row['file']:row for row in self.files[folder]}
        members=data.split('|')
        if kind!='status' and all(name in supplied for name in members) and sum(supplied[name]['rows'] for name in members)==0:
            self.status(folder,panel,title+' (not measured)',
                'The supplied observation tables contain no completed rows for this panel. No value, successful outcome, or zero error is inferred. '
                'The empty schema-bearing source tables and the associated coverage/disposition records remain in this figure folder. Sources: '+data,[])
            return
        for axis,col in [('x',x),('y',y)]:
            if kwargs.get(axis+'scale')=='symlog':
                kwargs.setdefault(axis+'linthresh',.01 if any(word in col.lower() for word in ('relative','mare','score','loss')) else 1.)
        self.specs[folder].append(dict(panel='panel_'+panel,kind=kind,data=data,title=title,x=x,y=y,group=group,**kwargs))

    def status(self,folder,panel,title,text,source_paths):
        name=self.csv(folder,'status_'+panel+'.csv',pd.DataFrame([dict(status='NOT_MEASURED',text=text)]),
            'Explicit evidence-backed disposition; no fabricated values',source_paths)
        self.panel(folder,panel,'status',name,title)

    def finish_figures(self):
        code=Path(__file__).with_name('plot_standalone.py')
        for name,specs in self.specs.items():
            if not name.startswith('fig_'):continue
            folder=self.folder(name);shutil.copy2(code,folder/'plot.py')
            pd.DataFrame(specs).fillna('').to_csv(folder/'plots.csv',index=False)
            self.document(name,'python plot.py')

    def document(self,name,command):
        folder=self.folder(name)
        intro=self.notes.get(name,'')
        files=self.files[name]
        readme=f'# {name}\n\nRun `{command}` from any working directory. Inputs resolve beside the script.\n\n'+intro+'\n\n'
        readme+='All numeric inputs and geometry are local UTF-8 CSVs. Stable IDs are strings. Native floating-point values use 17 significant digits. Missing/nonfinite results remain missing with status; they are never zero-imputed. Drawing code reports its excluded nonfinite values.\n\n'
        readme+='These files are plot-source data and analysis-ready observations, not the complete raw national database. No external repository, model, network service or raster map is needed for drawing.\n'
        (folder/'README.md').write_text(readme,encoding='utf-8')
        dictionary=['# Drawing data dictionary','',intro,'']
        for f in files:
            dictionary.extend([f'## {f["file"]}',f'{f["description"]} Rows: {f["rows"]}.','',
                '| Field | Meaning / convention |','|---|---|'])
            for col in f['columns']:dictionary.append('| '+col+' | '+field_meaning(col)+' |')
            dictionary.append('')
        (folder/'DATA_DICTIONARY.md').write_text('\n'.join(dictionary),encoding='utf-8')
        write_json(folder/'PROVENANCE.json',dict(source_bindings=self.sources[name],drawing_files=files,
            extraction_code_sha256=sha(Path(__file__)),run_id=self.ctx.c['run_id'],source_models_unchanged=True,
            data_kind='PLOT_SOURCE_DATA_AND_ANALYSIS_READY_OBSERVATIONS',external_public_deposit=False))


def field_meaning(col):
    if col=='unique_n':return 'Available distinct TRAIN inputs in the declared nested subset; replacement sampling may expose fewer distinct inputs.'
    if col in ('unique_rows_consumed','unique_rows_exposed'):return 'Distinct TRAIN inputs actually drawn by the recorded optimizer updates; excludes available but unexposed inputs.'
    if col=='selected_update':return 'Optimizer update of the fixed-VALID-selected checkpoint; zero selects initialization even when later updates were executed.'
    if col=='n_evaluation_rows':return 'Admitted certified query count for this curve point; shared across model families and fractions within a fresh region/cohort.'
    if col in ('failure_from_physical_nodes','failure_to_physical_nodes'):return 'Physical graph node indices for the failed internal bridge, not stable physical-node UIDs; use the authenticated graph index-to-UID mapping.'
    if col=='failure_layer':return 'Position of the failed operation in the ordered stitch diagnostic, not a physical network layer identifier.'
    if col=='selected_global_lift_interior_centroid_occurrences':return 'Interior-centroid count only for a verified lifted path; a zero with NO_VERIFIED_GLOBAL_PATH is unavailable evidence, not a centroid pass.'
    if col in ('x','y','mean_x','mean_y'):return 'Actual projected coordinate, EPSG:5070, metres; see geometry role.'
    if col in ('longitude','latitude'):return 'Source NAD83 coordinate, degrees (EPSG:4269).'
    if col in ('x45_flow','x200_flow','predicted_physical_flow','absolute_flow_error'):return 'Annual kilotons in the source assignment; same directed physical arc.'
    if col=='signed_bounded_difference':return '(prediction - x45) / max(abs(prediction), abs(x45)); both zero gives zero; dimensionless.'
    if 'sha256' in col or 'hash' in col:return 'SHA-256 identity; exact source/state binding, not a numerical score.'
    if col.endswith('_id') or col in ('region_id','physical_arc_id','physical_node_uid','stable_od_key'):return 'Stable source identifier, preserved as a string.'
    if col=='rho':return 'Fixed neural residual attenuation coefficient; no per-query clipping or switching.'
    if col in ('WREL','flow_WMAPE'):return 'Ratio of absolute-error sum to declared reference sum; fraction, not percent. See population and weights.'
    if col=='WMEANREL_1S' or 'MARE' in col:return 'Mean absolute relative error using max(abs(reference), 1 second); fraction, not percent; cohort/weights named.'
    if col=='RG_SPTT':return 'Fresh signed (TSTT - SPTT) / SPTT at exactly one state; dimensionless.'
    if 'seconds' in col or col.endswith('_s'):return 'Seconds, with measurement scope identified by neighboring role/measure/cohort fields.'
    if 'fraction' in col:return 'Dimensionless fraction, not percent.'
    if 'status' in col or 'role' in col or col in ('scenario','track','cohort','population'):return 'Explicit scientific scope or disposition. Distinct labels are not interchangeable.'
    if col in ('ARI','NMI_arithmetic','boundary_Jaccard'):return 'Partition statistic on the explicitly named common stable-ID population.'
    return col.replace('_',' ')+'; retained source observation or explicitly documented display/control field.'


def scenario_evidence(ctx):
    rows=[];curves=[];all_cases=cases(ctx)
    for method,job in [('CFW','C02_PHYSICAL_CFW'),('FW','C03_PHYSICAL_FW'),('MSA','C04_PHYSICAL_MSA')]:
        all_cases.append(dict(scenario='PHYSICAL_'+method,suite='PHYSICAL',root=str(ctx.root/'runs'/job)))
    for scenario,job in [('ORIGINAL_PRIMARY_WARM','B05_ORIGINAL_PRIMARY_WARM'),('ORIGINAL_BEST_ANY_WARM','B06_ORIGINAL_BEST_ANY_WARM'),
                         ('ORIGINAL_NONCONSTANT_WARM','B07_ORIGINAL_NONCONSTANT_WARM')]:
        all_cases.append(dict(scenario=scenario,suite='s03',root=str(ctx.root/'runs'/job)))
    for case in all_cases:
        root=Path(case['root']);p=root/'RESULT.json';scenario=case['scenario'];physical=case['suite']=='PHYSICAL'
        if not p.exists():
            missing=missing_state(ctx,root)
            if missing.get('exit_path'):ctx.bind(missing['exit_path'],'PUBLICATION_FAILED_ATTEMPT_TIMER',missing['exit_sha256'])
            rows.append(dict(scenario=scenario,label=short(scenario),suite=case['suite'],source_root=str(root),
                measured_seconds=missing.get('complete_child_seconds'),timing_scope='R4_FAILED_ATTEMPT_SUPERVISED_JOB_WALL'
                    if missing.get('complete_child_seconds') is not None else 'NO_MEASURED_ATTEMPT_WALL',
                certificate_fresh=False,**missing));continue
        ctx.bind(p,'PUBLICATION_SCENARIO_RESULT');r=read_json(p);certificate=r.get('final_certificate') or {}
        job=root.name;exitpath=ctx.root/'jobs'/job/'EXIT.json'
        # Inherited native complete-job records retain their own timing scope.
        if exitpath.exists():
            complete=read_json(exitpath)['elapsed_seconds'];scope='R4_COMPLETE_SUPERVISED_JOB_WALL';ctx.bind(exitpath,'PUBLICATION_COMPLETE_JOB_TIMER')
        else:
            complete=r.get('elapsed_seconds');scope='INHERITED_RECORDED_ASSIGNMENT_WALL_NOT_COMPLETE_PROCESS'
        offset=r.get('source_load_seconds',r.get('source_load_and_admission_seconds',0.))
        status=r.get('stop_reason',r.get('status','UNKNOWN'))
        row=dict(scenario=scenario,label=short(scenario),suite=case['suite'],status=status,RG_SPTT=certificate.get('RG_SPTT'),
            native_stop_reason=r.get('native_stop_reason'),
            certificate_fresh=r.get('final_certificate_fresh',False),accepted_updates=r.get('accepted_iterations'),
            measured_seconds=complete,timing_scope=scope,solver_seconds=r.get('solver_seconds',r.get('solver_elapsed_seconds',r.get('elapsed_seconds'))),
            source_load_seconds=offset,source_root=str(root),source_result_sha256=sha(p),
            initialization_seconds=r.get('initialization_seconds'),final_refresh_seconds=r.get('final_refresh_seconds'),
            final_refresh_scope=('Native final-cost recomputation and optional final AON; reuses an already certified same-state AON when available. Fresh-AON work is already inside solver elapsed.' if physical else None),
            component_nesting=('Initialization and final refresh are inside solver elapsed; source loading is outside solver elapsed; all are inside supervised-job wall'
                if physical and exitpath.exists() else 'Physical initialization/final-refresh breakdown not supplied for this scenario; see its recorded timing scope'),
            represented_demand=r.get('represented_demand'),initialization=r.get('initialization','SEE_PRE_RUN_FREEZE'),
            historical_analytic_startup_seconds=r.get('historical_analytic_startup_seconds'),
            cold_does_not_imply_empty_compilation_or_OS_cache=True)
        rows.append(row)
        hp=root/('fresh_certificates.jsonl' if physical else 'HISTORY.json')
        history=[]
        if hp.exists():
            ctx.bind(hp,'PUBLICATION_FRESH_GAP_HISTORY')
            history=[json.loads(line) for line in hp.read_text().splitlines()] if physical else read_json(hp)
        for item in history:
            if not item.get('fresh',False):raise ValueError('Uncertified history point')
            # N8 HISTORY uses the run/job epoch already; only physical solver
            # perf_counter histories require source-load offset.
            curve_offset=offset if physical else 0.
            original={k:(json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v) for k,v in item.items()}
            normalized=dict(scenario=scenario,label=short(scenario),kind='Physical' if physical else 'ACNHN',
                elapsed_seconds=float(item['elapsed_seconds'])+curve_offset,RG_SPTT=item['RG_SPTT'],state_hash=item['state_hash'],
                state_iteration=item.get('iteration',item.get('state_iteration')),source_history_sha256=sha(hp),
                timing_scope='SOURCE_LOAD_PLUS_SOLVER_FRESH_CERTIFICATE' if physical else 'N8_HISTORY_JOB_EPOCH_ALREADY_INCLUDES_LOADING',status='FRESH_INTERMEDIATE',
                display_series=short(scenario)+' ['+status+']',source_elapsed_seconds=item['elapsed_seconds'],
                complete_supervised_job_elapsed_seconds=complete if exitpath.exists() else None,
                terminal_source_load_seconds=offset,history_or_terminal='HISTORY_FRESH_CERTIFICATE')
            curves.append(dict(original,**normalized))
        if certificate and r.get('final_certificate_fresh'):
            terminal=dict(certificate)
            terminal.update(dict(scenario=scenario,label=short(scenario),kind='Physical' if physical else 'ACNHN',
                elapsed_seconds=complete,RG_SPTT=certificate['RG_SPTT'],state_hash=certificate.get('state_hash',r.get('state_hash')),
                state_iteration=r.get('accepted_iterations'),source_history_sha256=sha(p),timing_scope=scope,status=status,
                display_series=short(scenario)+' ['+status+']',history_or_terminal='TERMINAL_CERTIFICATE_AT_COMPLETE_RECORDED_END',
                complete_supervised_job_elapsed_seconds=complete if exitpath.exists() else None))
            curves.append(terminal)
    # Original cold admission failures remain explicit even though no valid solve exists.
    for original,job in [('Original primary','P03_S03_PRIMARY'),('Original best-any','P08_S03_BEST_ANY'),('Original nonconstant','P09_S03_NONCONSTANT_VALID')]:
        candidates=list((ctx.prior/'runs').glob(job+'*'))
        if not candidates:
            candidates=[p for p in (ctx.prior/'runs').iterdir() if p.name.startswith(job[:3]+'_')]
        root=candidates[0] if len(candidates)==1 else None;p=root/'RESULT.json' if root else None
        r=read_json(p) if p is not None and p.exists() else {}
        rows.append(dict(scenario=job,label=original+' cold',suite='s03',status='SAFETY_REJECTED',
            measured_seconds=r.get('elapsed_seconds'),timing_scope='ORIGINAL_REJECTED_ADMISSION_NOT_SOLVE_TIME',
            RG_SPTT=None,certificate_fresh=False,accepted_updates=0,source_root=str(root) if root else None,
            source_result_sha256=sha(p) if p is not None and p.exists() else None,
            initialization=r.get('initialization'),
            original_failure_location='First objective evaluation after feasible AON using scenario costs evaluated at zero flow; no accepted updates',
            original_failure_location_source='RESULT initialization + STAGES feasible AON + solver control flow; not fixed-x45 admission'))
    return pd.DataFrame(rows),pd.DataFrame(curves)


def figures_3_6_7(release,scenarios,curves):
    ctx=release.ctx;root=ctx.root;poproot=root/'physical_population'
    fig='fig_3';release.folder(fig)
    selection=release.read(root/'geography/reference_selected_interstate_100.csv')
    geo=release.read(root/'geography/comparison_100_road_vertices.csv')
    geo['segment_id']=geo.physical_arc_id+'|'+geo.part_index.astype(str)
    a=release.csv(fig,'selected_road_vertices.csv',geo,'True source road vertices, reference-only geographic selection',[root/'geography/comparison_100_road_vertices.csv'])
    release.copy(fig,root/'geography/reference_selected_interstate_100.csv')
    release.copy(fig,poproot/'COMMON_POPULATION_COVERAGE.csv')
    release.csv(fig,'physical_comparison_dispositions.csv',pd.DataFrame(read_json(poproot/'COMPARISON_STATES.json')),
        'Complete planned physical-comparison case ledger; measured, rejected, attempted and unvisited cases remain distinct',
        [poproot/'COMPARISON_STATES.json'])
    release.copy(fig,root/'accounting/historical_physical_benchmark_timing.csv',
        description='Accepted historical iteration-wall, broader campaign and separate zero-update refresh scopes; unknown standalone/hardware fields preserved')
    physical_timers=scenarios.loc[scenarios.suite.eq('PHYSICAL')].copy()
    release.csv(fig,'contemporary_physical_timer_components.csv',physical_timers,
        'Current supervised-job, solver, initialization and final-refresh times; nested components are not added twice',[root/'tables/SCENARIO_RESULTS.csv'])
    for job in ('C02_PHYSICAL_CFW','C03_PHYSICAL_FW','C04_PHYSICAL_MSA'):
        p=root/'runs'/job/'iteration_history.csv'
        if p.exists() and p.stat().st_size>2:
            iteration_rows=release.read(p)
            iteration_rows['native_cost_unit']='minutes'
            iteration_rows['native_objective_TSTT_SPTT_gap_unit']='annual_kiloton_minutes'
            iteration_rows['wall_timer_unit']='seconds'
            release.csv(fig,job+'_iteration_history.csv',iteration_rows,
                'Original physical iteration values: objectives, TSTT, SPTT and absolute gap use annual-kiloton minutes; wall timers use seconds and are nested in solver/job elapsed. Fresh exported certificate histories separately use seconds.',[p])
        else:
            source=p if p.exists() else poproot/'COMPARISON_STATES.json'
            release.csv(fig,job+'_iteration_history_disposition.csv',pd.DataFrame([dict(job_id=job,
                status='ZERO_RECORDED_UPDATES' if p.exists() else 'ITERATION_HISTORY_NOT_PRODUCED')]),
                'Explicit absence of recorded update timing rows; not an inferred successful zero-cost assignment',[source])
    release.panel(fig,'a','map',a,'Reference-selected Interstate sample','x','y','segment_id',xlabel='Albers easting (m)',ylabel='Albers northing (m)',
        vertex_markers='true',note='Actual directed road geometry; dots mark one supplied road vertex per sampled segment for visibility. Selection uses reference flow and geographic dispersion; no prediction errors.')
    pop=pd.read_parquet(poproot/'population.parquet');common=pop.loc[pop.common_all_five_suites].copy()
    b=release.csv(fig,'reference_physical_arcs.csv',common[['physical_arc_id','x45_flow','x200_flow','x45_cost_seconds','x200_cost_seconds']],
        'Every common recoverable physical arc; no interior zero imputation',[poproot/'population.parquet'])
    release.panel(fig,'b','ecdf',b,'Reference flow on the common physical population','x45_flow',ylabel='Empirical cumulative fraction',
        xlabel='Annual kilotons; symlog includes exact zero',xscale='symlog',note='Each directed physical arc has equal distribution weight. Flow-weighted summary metrics are supplied separately.')
    frames=[];source=[]
    for p in sorted(poproot.glob('*_COMMON_ALL_FIVE_SUITES_ARC_ROWS.csv')):
        v=release.read(p);statuses=scenarios.set_index('scenario').status.to_dict()
        v['label']=v.scenario.map(lambda s:short(s)+' ['+statuses[s]+']')
        frames.append(v[['physical_arc_id','scenario','label','population','x45_flow','x200_flow','predicted_physical_flow',
            'signed_bounded_difference','absolute_flow_error','physical_cost_at_prediction_seconds']]);source.append(p)
    if frames:
        c=release.csv(fig,'physical_arc_comparison_rows.csv',pd.concat(frames,ignore_index=True),
            'Full per-scenario common-population observations; annual kilotons and seconds',source)
        release.panel(fig,'c','ecdf',c,'Signed physical-flow differences','signed_bounded_difference',group='label',
            xlabel='(prediction - x45) / max(abs(prediction), abs(x45))',ylabel='Empirical cumulative fraction',
            note='Exact common physical arc population for every series; zero/zero is defined as zero.')
    else:release.status(fig,'c','Physical-flow comparison not measured',
        'No admitted common-population comparison rows were produced. The source assignments and frozen recovery population remain separate evidence; missing predictions are not physical zeros.',
        [poproot/'COMPARISON_STATES.json'])
    for panel,kind in [('d','Physical'),('e','ACNHN')]:
        data=curves.loc[curves.kind.eq(kind)].copy()
        data['positive_gap_for_log']=data.RG_SPTT.where(data.RG_SPTT>0)
        f=release.csv(fig,kind.lower()+'_fresh_gap_history.csv',data,'Full fresh same-state histories; no stale post-update gap',[])
        release.panel(fig,panel,'line',f,kind+' fresh-gap histories','elapsed_seconds','positive_gap_for_log','display_series',
            xlabel='Recorded wall seconds (scope in CSV)',ylabel='Fresh RG-SPTT',yscale='log',mark_terminal='true',
            note='Terminal x: time limit; circle: other terminal state. Nonpositive gaps remain in CSV and are omitted from the log axis. Rejected models have no gap curve.')
    common['reference_signed_difference']=(common.x200_flow-common.x45_flow)/np.maximum(abs(common.x200_flow),abs(common.x45_flow))
    common['reference_signed_difference']=common.reference_signed_difference.fillna(0)
    f=release.csv(fig,'reference_sensitivity_rows.csv',common[['physical_arc_id','x45_flow','x200_flow','reference_signed_difference']],
        'x200 sensitivity relative to accepted primary x45 on the same physical population',[poproot/'population.parquet'])
    release.panel(fig,'f','ecdf',f,'Reference-state sensitivity','reference_signed_difference',xlabel='(x200 - x45) / max(abs(x200), abs(x45))',ylabel='Empirical cumulative fraction')
    release.notes[fig]='Physical errors use the pre-frozen COMMON_ALL_FIVE_SUITES population. Equal-arc ECDFs show distributions; WMAPE uses sum absolute flow error / sum reference flow. TSTT on this subset uses physical BPR only, without fixed coarse charges. The subset does not certify complete physical lifting. x45 is primary; x200 is sensitivity. Timed controls run serially with one worker, but inherited assignment-only and new complete supervised-job scopes remain distinguished. The latter include prelaunch snapshotting. Historical four-worker cumulative iteration-wall timers include recorded checkpoint/refresh work; the later x45 zero-update refresh is separate. Initialization and final-refresh components are nested inside current solver elapsed, not additional time.'
    # Five-suite resolution sensitivity with constant-regional FFTT policies.
    fig='fig_6';release.folder(fig);st=root/'stability'
    release.csv(fig,'stability_coverage.csv',pd.DataFrame(read_json(st/'STABILITY_STATES.json')),
        'Complete planned suite/statistic/map ledger; missing units are not successful empty results',[st/'STABILITY_STATES.json',st/'RESULT.json'])
    count=release.read(st/'resolution_counts.csv')
    d=count.melt(id_vars=['suite','resolution'],value_vars=['final_region_count','active_regions','physical_portal_nodes','active_regional_coordinates'],var_name='quantity',value_name='count')
    f=release.csv(fig,'resolution_count_observations.csv',d,'Accepted final counts; raw pre-correction counts separately supplied',[st/'resolution_counts.csv'])
    release.copy(fig,st/'resolution_counts.csv');release.copy(fig,st/'active_dimension_observations.csv')
    release.panel(fig,'a','line',f,'Resolution and active representation','resolution','count','quantity',xscale='log',yscale='log',xlabel='Accepted resolution parameter',ylabel='Count')
    timing=scenarios.loc[scenarios.scenario.str.startswith('FFTT_')].copy();timing['label_status']=timing.label+' ['+timing.status+']'
    f=release.csv(fig,'resolution_runtime_status.csv',timing,'Exact inherited current FFTT assignments and their recorded timing boundaries')
    release.panel(fig,'b','bar',f,'Recorded FFTT assignment runtime','label_status','measured_seconds',xlabel='Seconds; inherited assignment scope',ylabel='')
    metric=release.read(poproot/'PHYSICAL_FLOW_METRICS.csv');metric=metric.loc[metric.scenario.str.startswith('FFTT_')&metric.population.eq('COMMON_ALL_FIVE_SUITES')]
    f=release.csv(fig,'resolution_physical_metrics.csv',metric,'Full common-population errors, not errors on selected map arcs',[poproot/'PHYSICAL_FLOW_METRICS.csv'])
    release.copy(fig,poproot/'COMMON_POPULATION_COVERAGE.csv')
    release.panel(fig,'c','bar',f,'Common physical-population flow error','suite','flow_WMAPE',xlabel='WMAPE (fraction)',ylabel='',
        note='Same directed arc population at every resolution; flow and flow-cost coverage are supplied in COMMON_POPULATION_COVERAGE.csv.')
    f=release.copy(fig,st/'active_dimension_observations.csv')
    release.panel(fig,'d','ecdf',f,'Active regional dimensions','m_active',group='suite',xscale='log',xlabel='Active coarse-pair coordinates per region',ylabel='Empirical cumulative fraction')
    ari=release.read(st/'partition_stability.csv');ari=ari.loc[ari.population.eq('COMMON_FINAL_REGION_OWNED_PHYSICAL_ARCS')]
    f=release.csv(fig,'final_ownership_stability.csv',ari,'ARI and arithmetic NMI on fixed common region-owned directed physical arcs',[st/'partition_stability.csv'])
    release.panel(fig,'e','heatmap',f,'Final ownership agreement','suite_a','suite_b',value='ARI',value_label='Adjusted Rand index')
    release.notes[fig]='All five suites are s02-s06; s01 is excluded. This is constant-regional FFTT resolution sensitivity. Backbone membership changes with resolution, and the evidence does not establish learned accuracy across resolutions or an optimal universal partition. Stability labels are categorical stable ownership IDs on the common physical-arc population.'
    # Faithful closure was impossible; preserve the actual intended geometry and source open reference.
    fig='fig_7';release.folder(fig);groot=root/'geography';closure=read_json(groot/'I95_CLOSURE_STATUS.json')
    geo=release.read(groot/'I95_intended_closed_physical_road_vertices.csv');geo['segment_id']=geo.physical_arc_id+'|'+geo.part_index.astype(str)
    a=release.csv(fig,'intended_closure_vertices.csv',geo,'True intended Interstate-95 physical road geometry',[groot/'I95_intended_closed_physical_road_vertices.csv'])
    release.panel(fig,'a','map',a,'Audited intended I-95 closure','x','y','segment_id',xlabel='Albers easting (m)',ylabel='Albers northing (m)',
        note='Hypothetical corridor defined by source class, route and states. AB is digitization direction; north/south labels are not inferred.')
    geo=geo.merge(pop[['physical_arc_id','x45_flow']],on='physical_arc_id',validate='many_to_one')
    f=release.csv(fig,'open_physical_reference_vertices.csv',geo,'Accepted open physical x45 reference only; not a completed hybrid closure comparison',[groot/'I95_intended_closed_physical_road_vertices.csv',poproot/'population.parquet'])
    release.panel(fig,'b','map',f,'Open physical x45 reference on the intended corridor','x','y','segment_id',value='x45_flow',value_label='Annual kilotons',
        xlabel='Albers easting (m)',ylabel='Albers northing (m)',note='This is the unchanged physical reference. A paired faithful closed state was not obtained.')
    counts=closure['counts'];reason=closure['reason']
    release.status(fig,'c','Closure experiment: unsupported mapping',
        f"{closure['status']}. {counts['suppressed_B_closed_arcs']} of {counts['intended_closed_physical_arcs']} intended road arcs are hidden inside {counts['suppressed_B_nodes']} suppressed B nodes. "+reason,[groot/'I95_CLOSURE_STATUS.json'])
    release.status(fig,'d','Signed closure change: not measured',
        'No faithful paired closure state is available. Closed flow, signed change, diversion and additional unreachability were not measured. Baseline capacities, support and regional models are preserved. No arbitrary logical-edge cut is presented as physical closure.',[groot/'I95_CLOSURE_STATUS.json'])
    for name in ('I95_s03_mapping_audit.csv','I95_affected_macro_essential_members.csv','I95_affected_primitive_bundles.csv','I95_intended_closed_physical_ids.csv'):release.copy(fig,groot/name)
    release.notes[fig]='The intended physical closure is audited, but UNSUPPORTED_MAPPING prevents a faithful overlay in the fixed contracted representation. Panels c and d are failed-experiment dispositions, not scientific closed-flow maps. No observed-event, welfare, emissions or ton-mile inference is made.'


def figures_4_5_S1(release):
    root=release.ctx.root;geo=root/'regional_geography';learning=root/'learning';fig='fig_4';release.folder(fig)
    summary=release.read(geo/'region_geography_summary.csv');points=release.read(geo/'true_portal_and_centroid_points.csv')
    release.csv(fig,'geography_coverage.csv',pd.DataFrame(read_json(geo/'GEOGRAPHY_STATES.json')),
        'Every preselected region and its completed or unavailable geography disposition',[geo/'GEOGRAPHY_STATES.json',geo/'RESULT.json'])
    release.copy(fig,geo/'region_geography_summary.csv')
    loc=summary.rename(columns={'mean_x':'x','mean_y':'y'}).copy();loc['display_role']=loc.figure_role.str.replace('|',' / ',regex=False)
    overlay=release.csv(fig,'region_locations.csv',loc,'Illustrative regional location markers: mean of actual road vertices, not administrative centroids',[geo/'region_geography_summary.csv'])
    background=release.read(root/'geography/comparison_100_road_vertices.csv');background['segment_id']=background.physical_arc_id+'|'+background.part_index.astype(str)
    name=release.csv(fig,'reference_road_context.csv',background,'Reference-selected true road geometry provides location context',[root/'geography/comparison_100_road_vertices.csv'])
    if summary.empty:
        release.status(fig,'a','Representative subnetwork locations: unavailable',
            'No selected region completed geometry extraction. The intended docket, source road context and coverage ledger are supplied; no regional location is inferred.',[geo/'RESULT.json'])
    else:
        release.panel(fig,'a','map',name,'Representative subnetwork locations','x','y','segment_id',overlay=overlay,point_group='display_role',xlabel='Albers easting (m)',ylabel='Albers northing (m)',vertex_markers='true',
            note='Grey dots mark supplied vertices of sampled roads. Colored region markers are means of actual road vertices, not administrative centroids.')
    for panel,role,title in [('b','large','Large region'),('c','good_VALID','Low-VALID-error region'),
        ('d','difficult_TRAIN_VALID','Difficult TRAIN/VALID region'),('e','LARGE_ZERO_E2_UPDATES_DIAGNOSTIC','Large zero-E2-update diagnostic'),
        ('f','NAMED_M28_DIAGNOSTIC','Named m28 diagnostic')]:
        chosen=summary.loc[summary.figure_role.str.split('|').map(lambda values:role in values).astype(bool)]
        if len(chosen)>1:raise ValueError('Ambiguous preselected geographic role: '+role)
        if chosen.empty:
            release.status(fig,panel,title+': geometry unavailable',
                'The preselected role '+role+' has no completed verified road-geometry export. No substitute region or invented coordinates are shown.',
                [geo/'RESULT.json',geo/'GEOGRAPHY_STATES.json'])
            continue
        row=chosen.iloc[0];rid=row.region_id
        v=release.read(geo/(rid+'_road_vertices.csv'));v['segment_id']=v.physical_arc_id+'|'+v.part_index.astype(str)
        roads=release.csv(fig,panel+'_road_vertices.csv',v,'True owned regional physical road vertices',[geo/(rid+'_road_vertices.csv')])
        pts=points.loc[points.region_id.eq(rid)].copy()
        point_file=release.csv(fig,panel+'_true_terminal_points.csv',pts,'Actual fine-member portal and centroid locations with native terminal types',[geo/'true_portal_and_centroid_points.csv'])
        release.panel(fig,panel,'map',roads,title,'x','y','segment_id',overlay=point_file,point_group='terminal_type',
            xlabel='Albers easting (m)',ylabel='Albers northing (m)',note='Native region '+rid+'. Roads are physical polylines; markers are actual terminal fine members.')
    release.copy(fig,geo/'selected_region_metadata.csv')
    release.notes[fig]='Selection uses frozen active dimension and TRAIN/VALID criteria before new evaluation outcomes. Region maps show actual road vertices and fine-member terminal locations in EPSG:5070. Multiple terminal roles can share a physical node. Location markers are illustrative means of road vertices, not geographic centroids or administrative boundaries. The named m28 and large zero-update cases are diagnostic additions, not replacements for failed selected regions.'
    fig='fig_5';release.folder(fig)
    summary_path=learning/'learning_curve_summary.csv'
    if summary_path.exists() and summary_path.stat().st_size>2:
        summary=release.read(summary_path);release.copy(fig,summary_path)
    else:
        summary=pd.DataFrame(columns=['stratum','cohort','status'])
        release.source(fig,learning/'EVALUATION_RESULT.json')
    release.copy(fig,learning/'selected_regions_16.csv')
    release.copy(fig,learning/'runtime_pilot.csv');release.copy(fig,learning/'new_label_coverage.csv')
    release.copy(fig,learning/'fresh_evaluation_planned_coverage.csv')
    source=[];all_rows=[]
    for p in sorted((learning/'regions').glob('*/post_freeze_evaluation/row_level_errors.csv')):
        all_rows.append(release.read(p));source.append(p)
    if all_rows:release.csv(fig,'evaluation_query_observations.csv',pd.concat(all_rows,ignore_index=True),'All evaluated query/region/seed/fraction/family observations, reused and postfreeze',source)
    for panel,stratum,title in [('a','large','Large regions'),('b','small','Small regions'),('c','good_VALID','Low-VALID-error regions'),('d','difficult_TRAIN_VALID','Difficult TRAIN/VALID regions')]:
        rows=summary.loc[summary.stratum.eq(stratum)&summary.cohort.isin(['NEW_OPERATING','NEW_STRESS'])&summary.status.eq('MEASURED')].copy()
        if rows.empty:
            release.status(fig,panel,title+': no eligible fresh curve',
                'No eligible postfreeze A/B evaluation curve was completed for this stratum. The supplied coverage and fit records give the failed or budget-limited cases; reused heldout results remain separately labeled.',[learning/'EVALUATION_RESULT.json']);continue
        rows['curve_id']=rows.region_id+'|seed'+rows.seed.astype(str)+'|'+rows.family+'|'+rows.cohort+'|'+rows.fit_status
        rows['model_and_cohort']=rows.family+' / '+rows.cohort+' / '+rows.fit_status
        file=release.csv(fig,'fresh_'+panel+'_curve_observations.csv',rows,'Every measured region/seed/fraction point; available distinct TRAIN inputs n, not the number actually exposed by replacement sampling; no monotonic smoothing',[learning/'learning_curve_summary.csv'])
        counts=rows.groupby(['region_id','cohort']).n_evaluation_rows.agg(['min','max'])
        if not counts['min'].eq(counts['max']).all():raise ValueError('Fresh curve evaluation denominators changed across fits')
        denominator_note=' Certified queries per region/cohort curve: '+str(int(counts['min'].min()))+' to '+str(int(counts['max'].max()))+'.'
        named=rows.loc[rows.region_id.eq('partC_p2_region_0ce745b1cfa26af63aab')].groupby('cohort').n_evaluation_rows.first()
        if len(named):denominator_note+=' Named m28: '+', '.join(str(cohort)+' n='+str(int(n)) for cohort,n in named.items())+'.'
        initial=rows.loc[rows.family.eq('SKIP_ICNN32')].groupby('region_id').selected_update.agg(['min','max'])
        initialized=initial.index[initial['min'].eq(0)&initial['max'].eq(0)]
        if len(initialized):denominator_note+=' Shared initialization selected for all fractions in region(s): '+', '.join(r.replace('partC_p2_region_','')[:6] for r in initialized)+'.'
        release.panel(fig,panel,'budget',file,title,'unique_n','MARE_equal_pairs','curve_id',color_group='model_and_cohort',
            xlabel='Available distinct TRAIN inputs n',ylabel='Mean absolute relative error (fraction)',yscale='symlog',
            note='Each thin line is one region/seed/model/cohort. Lines only connect the measured nested budgets; spread is empirical, not a confidence interval.'+denominator_note)
    release.notes[fig]='The five nested TRAIN fractions share a calibration subset, normalization and complete initialization within each region/seed. Analytic power4 is refit on each fraction. NN fits use the same configured optimizer-update budget and fixed VALID checkpoint policy; actual updates, selected checkpoints, exposure counts and time are supplied. Current national portfolios never absorb these experimental weights. Fresh operating/stress A/B labels are generated only after all experiment models freeze. Reused heldouts are separately labeled in the full observation CSVs. No failed label or region is zero-imputed. Symlog retains exact zero and does not add epsilon.'
    fit_results=read_json(learning/'FIT_RESULTS.json')
    flat=[]
    for row in fit_results:flat.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))})
    release.csv(fig,'fit_status_and_compute.csv',pd.DataFrame(flat),'All planned fit dispositions, including failures/censoring',[learning/'FIT_RESULTS.json'])
    for p in sorted((learning/'regions').glob('*/nested_subsets.csv')):release.copy(fig,p,name=p.parent.name+'_nested_subsets.csv')
    fig='fig_S1';release.folder(fig);loss=[];valid=[];exposures=[];sources=[]
    release.csv(fig,'all_fit_status_and_compute.csv',pd.DataFrame(flat),'All fit dispositions; zero-update/time-limited cases cannot supply optimizer observations',[learning/'FIT_RESULTS.json'])
    for row in fit_results:
        saved_path=row.get('model_path') or row.get('retained_checkpoint_path')
        if row.get('fraction')!=1. or not saved_path:continue
        path=Path(saved_path);key=row['region_id']+'|seed'+str(row['seed'])+'|'+row['status']
        unique_n=row.get('unique_n')
        if unique_n is None:
            prepared_path=path.parent.parent/'PREPARED.json'
            release.source(fig,prepared_path)
            unique_n=next(item['unique_n'] for item in read_json(prepared_path)['subsets'] if item['fraction']==1.)
        for name,bucket in [('training_batch_losses.csv',loss),('optimization_curve.csv',valid),('exposure_counts.csv',exposures)]:
            p=path/name
            if not p.exists():continue
            if p.stat().st_size<=2:
                release.source(fig,p)
                continue
            v=release.read(p);v['curve_id']=key;v['stratum']=row['stratum'];v['region_id']=row['region_id'];v['seed']=row['seed'];v['unique_n']=unique_n
            v['fit_status']=row['status'];v['stratum_and_fit_status']=row['stratum']+' / '+row['status']
            bucket.append(v);sources.append(p)
    if loss:
        f=release.csv(fig,'fixed_data_batch_losses.csv',pd.concat(loss,ignore_index=True),'Every optimizer batch at the fixed 100% TRAIN fraction; plotted means in 100-update bins',sources)
        release.panel(fig,'a','optimization',f,'Fixed-data training optimization','updates','robust_training_batch_loss','curve_id',color_group='stratum_and_fit_status',bin_width=100,
            xlabel='Optimizer updates',ylabel='Original robust batch loss',yscale='symlog',note='Only 100% TRAIN fits. Binned means improve readability; all raw batch losses are supplied.')
        release.panel(fig,'c','line',f,'Fixed-data cumulative exposure','updates','cumulative_row_exposures','curve_id',color_group='stratum_and_fit_status',xlabel='Optimizer updates',ylabel='Rows consumed, with replacement')
    else:
        for panel in ('a','c'):release.status(fig,panel,'Fixed-data optimization unavailable','No 100% TRAIN fit produced eligible optimizer observations within the fixed stage budget.',[learning/'FIT_RESULTS.json'])
    if valid:
        f=release.csv(fig,'fixed_data_VALID_checkpoints.csv',pd.concat(valid,ignore_index=True),'All VALID checkpoints of the fixed 100% TRAIN fits',sources)
        release.panel(fig,'b','line',f,'Fixed-data VALID checkpoints','updates','validation_MARE_equal_pairs','curve_id',color_group='stratum_and_fit_status',xlabel='Optimizer updates',ylabel='VALID MARE (fraction)',yscale='symlog')
    else:release.status(fig,'b','Fixed-data VALID curve unavailable','No eligible 100% TRAIN VALID checkpoints were completed.',[learning/'FIT_RESULTS.json'])
    if exposures:release.csv(fig,'fixed_data_input_exposure_counts.csv',pd.concat(exposures,ignore_index=True),'Per-unique-input replacement exposure counts at 100% TRAIN',sources)
    release.notes[fig]='This figure fixes the training-data fraction at 100% and changes optimizer updates. It is distinct from the nested sample-size experiment. Checkpoint VALID metrics and raw batch losses measure different quantities; both retain their own units. Failed fits contribute their persisted checkpoint prefix with an explicit FAILED label; unrecorded later updates are not reconstructed. Curves show actual regions/seeds and do not imply convergence or heldout monotonic improvement.'


def figure_S3(release):
    root=release.ctx.root;fig='fig_S3';release.folder(fig);rm=root/'regional_metrics'
    frame=release.read(rm/'region_registry_metrics.csv');frame['label']=frame.registry.map(short)
    release.csv(fig,'available_regional_registry_cohort_metrics.csv',frame,'All available per-region predictions, including incomplete registry groups',[rm/'region_registry_metrics.csv'])
    registry_count=int(read_json(rm/'RESULT.json')['registries'])
    frame=paired_regional(frame,registry_count)
    full=release.csv(fig,'regional_registry_cohort_metrics.csv',frame,'Paired region/cohort groups completed by every registry; per-region query means and coordinate denominators',[rm/'region_registry_metrics.csv',rm/'RESULT.json'])
    release.copy(fig,rm/'coverage.csv')
    release.copy(fig,rm/'model_prediction_coverage.csv')
    for p in sorted(rm.glob('current_E2_row_observations_*.csv')):release.copy(fig,p,description='Unaggregated query-level means/maxima; not a distribution of individual pair-coordinate errors')
    for panel,cohort,title in [('a','REUSED_NOW_SEEN_E2_AUDIT_OPERATING','Reused E2 operating queries'),('b','REUSED_NOW_SEEN_E2_AUDIT_STRESS','Reused E2 stress queries'),
                              ('e','REUSED_NOW_SEEN_LOW_TEST','Reused protected LOW_TEST'),('f','REUSED_NOW_SEEN_HIGH_PROJECTED','Reused protected HIGH_PROJECTED')]:
        v=frame.loc[frame.cohort.eq(cohort)]
        if v.empty:
            release.status(fig,panel,title+': no complete paired population',
                'No region/cohort group completed every frozen registry. All available predictions and missing coverage are supplied separately.',
                [rm/'RESULT.json',rm/'model_prediction_coverage.csv']);continue
        f=release.csv(fig,'region_error_'+panel+'.csv',v,'Equal-region distribution of mean query MARE for the named cohort',[rm/'region_registry_metrics.csv'])
        release.panel(fig,panel,'ecdf',f,title,'MARE_equal_pairs',group='label',xscale='symlog',xlabel='Region mean of query MARE (fraction)',ylabel='Equal-region empirical cumulative fraction',
            note='Only region/cohort groups completed by all registries. Source and completed-prediction coverage are separate; no zero imputation.')
    safety=release.read(root/'regional_safety_transform.csv');safety['label']=safety.registry_id.map(short)
    release.csv(fig,'fixed_safety_transform_observations.csv',safety,'Every active source region and separately named bound-safe registry',[root/'regional_safety_transform.csv'])
    neural=safety.source_kind.str.contains('ICNN')&safety.source_head_power.eq(1)&safety.rho.notna()
    denom=safety.groupby('registry_id').size().rename('all_active_regions').reset_index()
    eligible=safety.loc[neural].groupby('registry_id').size();denom['linear_head_neural_rho_regions']=denom.registry_id.map(eligible).fillna(0)
    denom['other_or_missing_rho_regions']=denom.all_active_regions-denom.linear_head_neural_rho_regions
    release.csv(fig,'rho_plot_denominators.csv',denom,'Analytic entries and unsupported-head fallbacks are outside the linear-head neural rho ECDF; full registry rows retained',[root/'regional_safety_transform.csv'])
    f=release.csv(fig,'linear_head_neural_rho_observations.csv',safety.loc[neural],'Only eligible linear-head neural transform coefficients; explicit full/plot denominators supplied',[root/'regional_safety_transform.csv'])
    release.panel(fig,'c','ecdf',f,'Fixed attenuation of linear-head neural residuals','rho',group='label',xlabel='Fixed rho',ylabel='Equal-eligible-region empirical cumulative fraction',note='Full registry and plot denominators are supplied; analytic entries and unsupported-head fallbacks are recorded separately.')
    path=root/'validation/original_failure_queries/N8R4_PRIMARY_BOUND_SAFE_V1.csv';query=release.read(path)
    release.copy(fig,path,name='exact_original_failure_complete_regional_query.csv')
    # Column names are the explicit native safety export, checked at execution.
    columns=[col for col in query if col in ('original_cost_seconds','safe_cost_seconds','c0_seconds')]
    if len(columns)<2:raise ValueError('Native negative-query cost fields changed')
    idx='native_index'
    if idx not in query:query[idx]=np.arange(len(query))
    data=query.melt(id_vars=[idx],value_vars=columns,var_name='cost_role',value_name='cost_seconds')
    f=release.csv(fig,'failure_query_cost_coordinates.csv',data,'Every coordinate of the complete original negative-query region; own coordinate zero does not remove cross effects',[path])
    release.panel(fig,'d','line',f,'Complete saved negative-query region',idx,'cost_seconds','cost_role',xlabel='Native coordinate index',ylabel='Cost (seconds; signed symlog)',yscale='symlog')
    release.csv(fig,'m28_current_disposition.csv',pd.DataFrame([read_json(rm/'M28_DISPOSITION.json')]),
        'Actual named m28 current diagnostic disposition; preserved R1 training evidence remains separate',[rm/'M28_DISPOSITION.json'])
    if (rm/'m28_current_TRAIN_e2_rows.csv').exists():release.copy(fig,rm/'m28_current_TRAIN_e2_rows.csv')
    stress=frame.loc[frame.cohort.eq('REUSED_NOW_SEEN_E2_AUDIT_STRESS')].copy()
    stress['below_fftt_observation_fraction']=stress.below_fftt_coordinates/stress.coordinate_observations
    f=release.csv(fig,'stress_tail_and_floor_observations.csv',stress,'Every region maximum query-coordinate relative error and below-freeflow frequency; same certified stress observations',[rm/'region_registry_metrics.csv'])
    if len(stress):
        release.panel(fig,'g','ecdf',f,'Stress-query regional error tails','maximum_relative_error',group='label',xscale='symlog',
            xlabel='Maximum relative coordinate error within the region/cohort',ylabel='Equal-region empirical cumulative fraction')
        release.panel(fig,'h','ecdf',f,'Stress-query costs below free flow','below_fftt_observation_fraction',group='label',
            xlabel='Below-freeflow coordinate observations / all coordinate observations',ylabel='Equal-region empirical cumulative fraction')
    else:
        for panel in ('g','h'):release.status(fig,panel,'No complete paired stress population',
            'Stress predictions did not complete for every registry on any common region/cohort group. Available predictions and missing coverage are retained.',
            [rm/'RESULT.json',rm/'model_prediction_coverage.csv'])
    release.copy(fig,root/'learning/large_zero_E2_docket.csv');release.copy(fig,root/'learning/m28_docket.csv')
    r1=Path(release.ctx.c['input_versions']['current_R1_native_root'])/'tables'
    for name in ('m28_training_probe.csv','E2_label_role_quality_counts.csv','priority_case_training_counters.csv'):
        release.copy(fig,r1/name,name='preserved_R1_'+name,description='Preserved prior seen-data diagnostic and quality/update counts; not new experimental outcomes')
    release.notes[fig]='Current frozen source portfolios and their separately named fixed bound-safe transforms use identical certified observations within each cohort. Error/floor distributions use only region/cohort groups completed by every registry, matching Table 3; all available predictions remain separately supplied. Distributions are of per-region averages of query-level MARE, not individual coordinate errors. E2 source coverage, completed prediction coverage and old-test coverage are distinct. The named m28 TRAIN_e2 diagnostic is seen training evidence. Safety rho is frozen from a weight-only lower bound and defines a different potential; no online clipping, scale cap, per-query fallback or silent model replacement is used. Nonnegative costs do not imply that every cost exceeds free flow.'


def figure_S2(release):
    ctx=release.ctx;root=ctx.root;st=root/'stability';bd=root/'boundary_diagnostics';fig='fig_S2';release.folder(fig)
    national=[];chicago=[];coverage=[];source=[]
    columns=['physical_arc_id','suite','suite_label','is_final_backbone','x','y','longitude','latitude','geometry_role','plot_crs']
    for sid in ctx.c['active_suites']:
        p=st/(sid+'_map_observations.parquet')
        if not p.exists():
            coverage.extend([dict(suite=sid,panel=panel,status='MAP_NOT_PRODUCED',source_arcs=None,displayed_arcs=None)
                for panel in ('national','Chicago_window')])
            continue
        v=pd.read_parquet(p);v['suite']=sid;source.append(p)
        if not np.isfinite(v[['x','y','longitude','latitude']].to_numpy(float)).all():
            raise ValueError('Ownership map has nonfinite source coordinates: '+sid)
        v=v.sort_values('physical_arc_id',key=lambda s:s.astype(np.int64))
        full=v.loc[v.longitude.between(-125,-66)&v.latitude.between(24,50)]
        chi=v.loc[v.longitude.between(-89,-86)&v.latitude.between(40.5,43)]
        step=max(1,int(np.ceil(len(full)/40000)));shown=full.iloc[::step]
        national.append(shown[columns]);chicago.append(chi[columns])
        coverage.extend([dict(suite=sid,panel='national',status='MEASURED',source_arcs=len(v),nonfinite_coordinate_rows=0,
            outside_declared_window=len(v)-len(full),eligible_arcs=len(full),displayed_arcs=len(shown),stable_ID_sort_stride=step,
            longitude_min=-125,longitude_max=-66,latitude_min=24,latitude_max=50),
            dict(suite=sid,panel='Chicago_window',status='MEASURED',source_arcs=len(v),nonfinite_coordinate_rows=0,
            outside_declared_window=len(v)-len(chi),eligible_arcs=len(chi),displayed_arcs=len(chi),stable_ID_sort_stride=1,
            longitude_min=-89,longitude_max=-86,latitude_min=40.5,latitude_max=43)])
    for panel,name,frames,title in [('a','national',national,'Final physical-arc ownership: CONUS window'),('b','Chicago',chicago,'Final physical-arc ownership: Chicago window')]:
        data=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=columns)
        f=release.csv(fig,name+'_ownership_markers.csv',data,'Actual physical-arc endpoint midpoint markers; missing suites remain unavailable, not fabricated boundaries',source+[st/'RESULT.json'])
        release.panel(fig,panel,'ownership',f,title,'x','y','suite_label',facet='suite',
            note='Grey: backbone. Colors: categorical regional ownership within each suite; colors do not identify matched regions. Midpoints, not boundaries.')
    release.csv(fig,'map_display_selection.csv',pd.DataFrame(coverage),'Deterministic graphical reduction; all accuracy summaries use their full populations',source)
    release.csv(fig,'stability_coverage.csv',pd.DataFrame(read_json(st/'STABILITY_STATES.json')),
        'Complete suite/statistic/map ledger; five-suite common-population metrics require all five input suites',[st/'STABILITY_STATES.json',st/'RESULT.json'])
    for name in ('common_final_regional_arc_labels.csv','common_raw_precorrection_node_labels.csv','stability_contingency_counts.csv',
                 'partition_stability.csv','physical_boundary_node_membership.csv','boundary_Jaccard.csv'):release.copy(fig,st/name)
    ari=release.read(st/'partition_stability.csv');ari=ari.loc[ari.population.eq('COMMON_FINAL_REGION_OWNED_PHYSICAL_ARCS')]
    f=release.csv(fig,'final_ownership_NMI.csv',ari,'Arithmetic normalized mutual information, fixed common owned-arc IDs',[st/'partition_stability.csv'])
    release.panel(fig,'c','heatmap',f,'Final ownership normalized mutual information','suite_a','suite_b',value='NMI_arithmetic',value_label='Arithmetic NMI')
    f=release.copy(fig,st/'boundary_Jaccard.csv')
    release.panel(fig,'d','heatmap',f,'Physical portal-set agreement','suite_a','suite_b',value='boundary_Jaccard',value_label='Jaccard index')
    disposition=available_diagnostics(release,fig,bd,('selected_path_diagnostics.csv','selected_native_path_edges.csv','positive_coarse_pair_local_witnesses.csv',
        'selected_physical_reference_path_arcs.csv','verified_selected_global_lift_arcs.csv','selected_regional_witness_arcs.csv',
        'accepted_graph_centroid_flags.csv','interior_centroid_violations.csv'))
    p=bd/'selected_path_diagnostics.csv'
    rows=release.read(p) if p.exists() and p.stat().st_size>2 else pd.DataFrame()
    if not rows.empty:
        rows['joint_audit']=rows.lift_status+' / '+rows.selected_global_lift_centroid_audit
        counts=rows.groupby('joint_audit',sort=False).size().rename('distinct_ODs').reset_index()
        counts['display_disposition']=counts.joint_audit.map({
            'SELECTED_LOCAL_WITNESS_STITCH_FAILURE_NOT_EXHAUSTIVE_UNLIFTABILITY / NO_VERIFIED_GLOBAL_PATH':'Selected witness stitch failed\nGlobal path unverified',
            'VERIFIED / PASS':'Verified selected path\nNo interior centroid'})
        if counts.display_disposition.isna().any():raise ValueError('Unexpected selected-path disposition requires an explicit display label')
        f=release.csv(fig,'selected_lift_dispositions.csv',counts,'Completed unique-OD diagnostic outcomes; not probability-sample success estimates',[p,disposition])
        endpoints=rows.stable_od_key.str.split('|',regex=False,expand=True)
        verified_self=int((endpoints[0].eq(endpoints[1])&rows.selected_global_lift_continuity_pass.eq(True)).sum())
        release.panel(fig,'e','bar',f,'Completed global-path diagnostic outcomes','display_disposition','distinct_ODs',xlabel='Distinct diagnostic ODs',ylabel='',
            note='Verified counts include '+str(verified_self)+' empty self-OD path(s). Unverified paths have no centroid-pass claim. A selected witness stitch failure is not exhaustive unliftability.')
    else:release.status(fig,'e','Global-path diagnostics not completed',
        'The bounded diagnostic did not produce completed selected global-path outcomes. This is missing evidence, not a zero failure or success rate. See diagnostic_file_coverage.csv and the terminal disposition.',[disposition])
    p=bd/'positive_coarse_pair_local_witnesses.csv'
    witness=release.read(p) if p.exists() and p.stat().st_size>2 else pd.DataFrame()
    if not witness.empty and witness.national_coarse_pair_flow.gt(0).any():
        witness['transition']=witness.source_terminal_type+' to '+witness.target_terminal_type
        transitions=witness.loc[witness.national_coarse_pair_flow>0].groupby('transition',sort=False).agg(
            positive_pairs=('pair_id','size'),coarse_pair_flow_sum=('national_coarse_pair_flow','sum')).reset_index()
        f=release.csv(fig,'positive_pair_terminal_transitions.csv',transitions,'Completed positive-flow pairs with validated frozen-physical-cost local witnesses',[p,disposition])
        release.panel(fig,'f','bar',f,'Completed positive-flow local transitions','transition','positive_pairs',xlabel='Positive coarse pairs',ylabel='')
    else:release.status(fig,'f','Positive-flow local witness data unavailable',
        ('No completed local-witness rows were produced by the bounded diagnostic. ' if witness.empty else
         'The completed local-witness rows contain no positive coarse-pair flows. ')+
        'Missing regions are not imputed as valid or invalid witnesses.',[disposition])
    release.notes[fig]='Ownership uses common stable physical IDs. Raw pre-correction and final corrected partition statistics remain distinct. Map points are physical arc midpoints, not fabricated roads or boundaries. Global lifting is a selected unique-OD diagnostic using independently shortest regional witnesses at frozen x45 physical costs plus exact backbone/member stitching; it is not joint regional equilibrium or an all-OD claim. Full physical centroid flags audit interior occurrences separately from continuity. Source physical assignment permits centroid transit; the reference is preserved. Repeated regional visits remain ordered occurrences.'


def figure_S4(release):
    root=release.ctx.root;pm=root/'path_metrics';bd=root/'boundary_diagnostics';fig='fig_S4';release.folder(fig)
    release.csv(fig,'path_evaluation_dispositions.csv',pd.DataFrame(read_json(pm/'METRIC_STATES.json')),
        'Complete planned scenario/track ledger, including rejected states and unvisited or interrupted work',[pm/'METRIC_STATES.json'])
    metric=release.read(pm/'PATH_METRICS.csv');metric['label']=metric.scenario.map(short)
    release.copy(fig,pm/'PATH_METRICS.csv');release.copy(fig,pm/'PHYSICAL_SAMPLE_ROWS.csv')
    all_rows=[];source=[]
    for p in sorted(pm.glob('*_PATH_ROWS.csv')):
        v=release.read(p);v['label']=v.scenario.map(short);all_rows.append(v);source.append(p)
    observations=pd.concat(all_rows,ignore_index=True) if all_rows else pd.DataFrame(columns=[
        'scenario','label','source_set','sample_role','track','relative_error_1s','absolute_error_seconds'])
    release.csv(fig,'all_paired_path_observations.csv',observations,'Every historical and PPS paired observation, both frozen and solved tracks, with original weights/state/status',source)
    pps=metric.loc[metric.cohort.eq('FRESH_PPS_DRAW')].copy();pps['series']=pps.label+' / '+pps.track.str.replace('_X45_REFERENCE','',regex=False)
    f=release.csv(fig,'PPS_ratio_estimates.csv',pps,'Demand-PPS draw-unit estimates and paired-draw bootstrap; no second demand weighting',[pm/'PATH_METRICS.csv'])
    release.panel(fig,'a','errorbar',f,'New demand-PPS path estimates','series','WREL',low='WREL_bootstrap_draw_95_low',high='WREL_bootstrap_draw_95_high',
        xlabel='WREL (fraction), with paired-draw 95% bootstrap interval',ylabel='',note='Uncertainty is conditional on finite paired queries and this frozen design; it is not an all-OD bound.')
    solved=observations.loc[observations.source_set.eq('FRESH_PPS_DRAW')&observations.track.eq('SOLVED_X45_REFERENCE')].copy()
    f=release.csv(fig,'PPS_solved_draw_observations.csv',solved,'All draw units including duplicates; solved native times versus unchanged x45 physical reference',source)
    release.panel(fig,'b','ecdf',f,'PPS solved-state relative path errors','relative_error_1s',group='label',xscale='symlog',
        xlabel='Absolute error / max(abs(x45 path time), 1 second)',ylabel='Empirical fraction of draw units',
        note='Primary-safe and best-any-safe are certified TIME_LIMIT snapshots. Other supplied solved states reached the declared optimization target; this does not certify physical accuracy.')
    release.panel(fig,'c','tail',f,'PPS error-numerator concentration','absolute_error_seconds',group='label',
        xlabel='Fraction of draw units, ordered by absolute error',ylabel='Cumulative share of WREL numerator',
        note='PPS already samples demand. Draws retain multiplicity and receive equal weights. Primary-safe and best-any-safe are certified TIME_LIMIT snapshots.')
    old=metric.loc[~metric.cohort.eq('FRESH_PPS_DRAW')&metric.track.eq('SOLVED_X45_REFERENCE')&metric.weight_definition.eq('RAW_ORIGINAL_OD_DEMAND')].copy()
    old['series']=old.label+' / '+old.cohort
    f=release.csv(fig,'historical_cohort_ratio_estimates.csv',old,'All historical cohorts, raw-demand weights; source expansion-weight alternatives supplied in full metrics',[pm/'PATH_METRICS.csv'])
    release.panel(fig,'d','bar',f,'Historical now-seen cohort path error','series','WREL',xlabel='WREL (fraction), raw original demand weights',ylabel='',bar_label_fontsize='9')
    disposition=available_diagnostics(release,fig,bd,('selected_path_diagnostics.csv','selected_native_path_edges.csv',
        'selected_physical_reference_path_arcs.csv','selected_regional_witness_arcs.csv','interior_centroid_violations.csv'))
    p=bd/'selected_path_diagnostics.csv'
    diagnostics=release.read(p) if p.exists() and p.stat().st_size>2 else pd.DataFrame()
    if not diagnostics.empty:
        named=diagnostics.loc[diagnostics.selection_roles.str.split('|').map(
            lambda roles:any(role!='FIRST32_PREFROZEN_PPS_DRAWS' for role in roles))].copy()
        columns=['D_phys_s','D_native_s','D_native_FIXED_X45_s','D_frozen_regional_cost_native_solved_backbone_s','D_frozen_regional_cost_native_fixed_backbone_s']
        v=named.melt(id_vars=['stable_od_key','selection_roles'],value_vars=columns,var_name='distance_role',value_name='seconds')
        role_labels={'D_phys_s':'physical x45', 'D_native_s':'native solved state',
            'D_native_FIXED_X45_s':'native at fixed x45 state',
            'D_frozen_regional_cost_native_solved_backbone_s':'frozen regional cost / solved backbone',
            'D_frozen_regional_cost_native_fixed_backbone_s':'frozen regional cost / fixed backbone'}
        v['cost_role_label']=v.distance_role.map(role_labels)
        role_keys={'D_phys_s':'P45','D_native_s':'NS','D_native_FIXED_X45_s':'N45',
            'D_frozen_regional_cost_native_solved_backbone_s':'FR-S','D_frozen_regional_cost_native_fixed_backbone_s':'FR-45'}
        v['series']=v.stable_od_key+' / '+v.distance_role.map(role_keys)
        f=release.csv(fig,'named_tail_distance_comparisons.csv',v,'Completed fixed/solved backbone comparisons for the selected diagnostics',[p,disposition])
        release.panel(fig,'e','bar',f,'Completed named and extreme path diagnostics','series','seconds',xscale='symlog',xlabel='Path cost (seconds; symlog)',ylabel='',bar_label_fontsize='9',note_fontsize='9',note_wrap_width='72',note_after_axis='true',
            note='P45: physical x45; NS: native solved; N45: native at fixed x45; FR-S / FR-45: frozen regional physical costs with solved / fixed backbone. Frozen regional paths are diagnostics, not joint equilibrium labels. Uncompleted cases remain missing.')
    else:release.status(fig,'e','Named path diagnostics not completed',
        'The selected boundary/lift diagnostic did not produce completed named path-distance comparisons. Independent paired path estimates in the preceding panels retain their own coverage.',[disposition])
    p=bd/'selected_native_path_edges.csv'
    if p.exists() and p.stat().st_size>2:
        native=release.read(p)
        costs=native.groupby('stable_od_key')[['fixed_B_charge_seconds','packed_dynamic_including_fftt_seconds','regional_surrogate_seconds']].sum().reset_index()
        release.csv(fig,'native_selected_path_cost_components.csv',costs,'Exact cost components summed over the recorded ordered selected native paths',[p])
    release.notes[fig]='WREL is sum(w abs(error))/sum(w reference time). WMEANREL_1S is sum(w abs(error)/max(abs(reference),1 second))/sum(w). Historical raw demand and historical expansion weights are distinct records. New represented-demand PPS uses equal draw weights, includes duplicates and has a paired-draw bootstrap. No headline trimming: top1/top5 exclusion metrics are sensitivity fields only. Original rejected neural portfolios have no admissible national path result. Solved comparisons retain assignment censoring. Tiny physical capacities and minute-to-second conversion appear in the selected physical-arc audit; no reference retuning occurs.'


def figure_8(release,scenarios):
    root=release.ctx.root;acc=root/'accounting';fig='fig_8';release.folder(fig)
    stages=release.read(acc/'offline_stage_subtotals.csv');stages['hours']=stages.seconds/3600.
    stages['label']=stages.stage.str.replace('_',' ',regex=False)
    release.copy(fig,acc/'offline_disjoint_components.csv');release.copy(fig,acc/'offline_missing_scope.csv')
    release.copy(fig,acc/'R4_complete_job_clock_ledger.csv',description='Pre-release completed-job snapshot assembled during E04; excludes E04 itself and later publication/finalization jobs, not the full R4 campaign total')
    release.copy(fig,acc/'historical_physical_authoritative_update_history.csv')
    worker=stages.measure.str.contains('WORKER')
    for panel,mask,title in [('a',~worker,'Recorded offline stage wall subtotals'),('b',worker,'Recorded disjoint worker-wall subtotals')]:
        f=release.csv(fig,'offline_'+panel+'_subtotals.csv',stages.loc[mask],'Measured partial subtotals; different timing scopes are not summed into a complete campaign total',[acc/'offline_stage_subtotals.csv'])
        release.panel(fig,panel,'bar',f,title,'label','hours',xlabel='Hours in explicitly named measurement scope',ylabel='',
            note='Missing preparation/generation scope remains unknown. Worker wall is not CPU time and is not added to parent elapsed time.')
    converged=scenarios.loc[scenarios.certificate_fresh.fillna(False)&pd.to_numeric(scenarios.RG_SPTT,errors='coerce').le(.001)&
        scenarios.status.str.startswith('CONVERGED')&~scenarios.scenario.str.endswith('_WARM')].copy()
    qrows=[]
    for row in converged.itertuples(index=False):
        if not np.isfinite(row.measured_seconds):continue
        for q in (1,2,5,10,20,50,100):
            qrows.append(dict(scenario=row.scenario,label=row.label+' / '+row.timing_scope,Q=q,
                observed_single_query_seconds=row.measured_seconds,online_only_seconds=q*row.measured_seconds,
                complete_offline_intercept=None,complete_cost=None,full_break_even='NOT_ESTABLISHED',
                model='Illustrative identical-query serial repetition at observed qualified per-query time; no cache or learning curve forecast',
                timing_scope=row.timing_scope,terminal_status=row.status,RG_SPTT=row.RG_SPTT))
    if qrows:
        f=release.csv(fig,'qualified_repeat_query_online_only.csv',pd.DataFrame(qrows),'Explicit zero-intercept ONLINE-ONLY arithmetic scenarios; not complete amortized costs')
        release.panel(fig,'c','line',f,'Qualified online-only repetition scenarios','Q','online_only_seconds','label',
            xlabel='Illustrative number of identical serial queries Q',ylabel='Q times observed per-query seconds',xscale='log',yscale='log',
            note='Offline intercepts are unknown. Mixed recorded timing scopes are labeled; no matched speedup or full break-even follows from these slopes.')
    else:
        release.status(fig,'c','No qualified completed-query timing',
            'No observed scenario has both a fresh target-gap certificate and a converged terminal status with a finite recorded runtime. No completed-query slope is estimated.',
            [root/'tables/SCENARIO_RESULTS.csv'])
    release.status(fig,'d','Complete offline amortization: not established',
        'Complete physical and hybrid offline elapsed costs are not established from the retained historical receipts. Worker-wall counters, nested solver timers and parent elapsed intervals are not interchangeable. Therefore a complete-cost intercept and a confident full break-even query count are not reported. Time-limited and safety-rejected assignments are excluded from completed-query slopes.',[acc/'RESULT.json'])
    release.notes[fig]='Offline components are disjoint within their named scope; inherited training counters are subtracted once and resumed label jobs are deduplicated or excluded as ambiguous. Summed worker wall is neither elapsed critical path nor CPU/GPU time. Historical x45/x200 update timers and the later x45 refresh have different scopes. R4_complete_job_clock_ledger.csv is the pre-release completed-job snapshot assembled during E04: it excludes E04 itself and later publication/finalization jobs, and is not the complete campaign total. The final root time_budget_ledger.csv and handoff receipt retain the later campaign clock. The repeat-Q panel is an illustrative online-only arithmetic scenario based on converged assignments and individually labeled timing scopes. Complete intercepts and full break-even remain unknown.'
