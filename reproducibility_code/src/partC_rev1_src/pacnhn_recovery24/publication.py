"""Explicit recovery publication and byte-preserved v5 numerical amendment.

No work runs on import. Scientific inputs are authenticated CSV exports only;
the manuscript bundle is constructed only in the separately opened manuscript
window. Existing v5 sources, supplements and shared figures remain untouched.
"""
from pathlib import Path
import csv
import html
import json
import math
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd
from ..pacnhn_v1.control import read_json, sha
from ..pacnhn_v1 import publication as portable
from ..pacnhn_v1.publication_templates.csv_support import (CSV_PART_COLUMNS, read_logical_csv,
    csv_logical_names, csv_expected_files)
from .ablation import _new
from .publication_templates.build_table import escape as table_escape

TEMPLATES=Path(__file__).with_name('publication_templates')
ASSETS=[f'figure{i}' for i in range(3,8)]+[f'table{i}' for i in range(1,5)]
HEADINGS=['Study setting and reproducibility','Full-network benchmark and backbone-flow validation',
          'Regional surrogate training','Partition sensitivity and boundary consistency','Counterfactual I-95 closure']
LABELS=['sec:setting','sec:validation','sec:regional_training','sec:partition_results','sec:i95']
GEO=['tail_longitude','tail_latitude','head_longitude','head_latitude']
SCHEMAS={
 'networks':['suite_id','resolution'],
 'methods':['case_id','network','suite_id','scenario','family','effective_method','solver','status','RG_SPTT',
    'elapsed_seconds','wmape_backbone_percent','wmape_key_percent','backbone_cost_error_percent','assignment_role','execution_status'],
 'backbone_links':['case_id','link_id','native_index','reference_flow','predicted_flow','signed_bounded_error','is_key']+GEO,
 'convergence':['case_id','iteration','elapsed_seconds','RG_SPTT'],
 'learning':['suite_id','region_index','family','version','role','WMAPE_percent','absolute_error_sum','true_cost_sum',
    'coordinates','admitted_inputs','requested_inputs','dataset_identity','model_sha256','target_met','fallback_component','fallback_reason','status'],
 'ablation_summary':['stratum','model_family','training_fraction','median_WMAPE','q25_WMAPE','q75_WMAPE','region_seed_rows'],
 'ablation_observations':['task_id','region_key','stratum','family','fraction','fit_status','TEST_status','TEST_WMAPE_percent','requested_TEST_inputs','admitted_TEST_inputs'],
 'region_geometry':['suite_id','region_index','stratum','geometry_role','native_id']+GEO,
 'region_locations':['suite_id','region_index','stratum','geometry_role','label']+GEO,
 'resolution_counts':['suite_id','resolution','regions','portal_nodes','active_regions','active_pairs','mean_physical_arcs'],
 'partition_stability':['suite_a','suite_b','ARI','NMI_arithmetic','common_population_n','population'],
 'boundary_jaccard':['suite_a','suite_b','boundary_Jaccard','population'],
 'primary_map_flows':['link_id','edge_kind','flow','scenario','is_closed']+GEO,
 'closure_geometry':['link_id']+GEO,
 'served_membership':['population','aggregate_OD_rows','original_OD_row_multiplicity','demand'],
 'timing':['case_id','seconds','status','timing_boundary','time_kind'],
 'settings':['setting','value'],
}
TITLES={
 'figure3':'Full-network benchmark and backbone-flow validation',
 'figure4':'Representative regional subnetworks',
 'figure5':'Nested TRAIN-size experiment on fixed blind cohorts',
 'figure6':'Partition sensitivity on the common backbone',
 'figure7':'Represented I-95 closure',
 'table1':'Study setting and computing environment',
 'table2':'P-ACNHN and hybrid assignment results',
 'table3':'Regional fitting, fallback and new heldout accuracy',
 'table4':'Constant-FFTT partition sensitivity',
}
CAPTIONS={
 'figure3':'Native backbone coordinates and the fixed 200-link subset are retained. Maps use real endpoint longitude/latitude; the continental-US crop affects display only. Signed bounded errors include zero/zero as zero.',
 'figure4':'Physical regional links are endpoint chords in the fixed P graph. Circles mark portals; stars mark exact physical centroid UIDs. Regional location markers are means of actual graph-node coordinates, not administrative centroids.',
 'figure5':'Available-case median native TEST WMAPE and interquartile ranges over up to four frozen regions per stratum. Unconnected markers and per-point n preserve missing fractions; observed regions can differ across family/fraction, so these are not paired learning curves. The design specified five nested TRAIN fractions, independent subset-only calibration, and fixed per-family fitting caps; actual fit and evaluation coverage is reported explicitly. VALID and new TEST cohorts stay fixed; missing fits or labels are explicit.',
 'figure6':'The same backbone, OD mappings and residual physical arc population are retained across five frozen resolutions. FFTT provides the constant-policy comparison. Full learned/hybrid outcomes and fixed-topology stability observations are included in CSV.',
 'figure7':'Closure applies to represented B coordinates. B and regional shortcut flows have separate logarithmic scales. Changes can combine rerouting and lost served OD demand; they are not a full physical I-95 closure or welfare estimate.',
 'table1':'Counts and environment values are taken from the bound current export. Native freight flow is annual thousand tons; costs use seconds. Historical values are identified by source and version.',
 'table2':'OPEN primary s03 and P rows. B/200 errors are ratio-of-sums flow WMAPE with every coordinate retained. B cost error is the absolute aggregate backbone travel-cost difference divided by reference B travel cost, not per-link cost WMAPE. Reference-self and controls remain separate. Recovery family fit/audit worker-seconds include failed and unqualified candidates, exclude shared labels and historical stage wall, and are repeated across that family\'s reporting rows. Power4 timing is inseparable from shared baseline work and remains N/A; P/FFTT require no incremental regional fit. Absent bound timing is N/A.',
 'table3':'Regional summaries use finite equal-region errors; pooled TEST error uses native absolute-error/truth sums. Targets, numerical admission, fallback and requested/admitted holdout coverage are separate. Missing measurements remain N/A.',
 'table4':'OPEN FFTT uses an unchanged analytic cost policy across all five suites. Same-native-B errors and aggregate B cost errors use the same P reference. The full 48-row method export remains available separately.',
}


def code_bindings():
    paths=[Path(__file__),Path(__file__).with_name('context.py'),Path(__file__).with_name('latex_compat.py'),Path(portable.__file__),
           Path(__file__).with_name('publication_geometry.py'),Path(__file__).with_name('ablation.py'),
           Path(portable.__file__).with_name('control.py'),
           portable.TEMPLATES/'audit_runner.py',portable.TEMPLATES/'csv_support.py']+sorted(TEMPLATES.glob('*.py'))
    return {str(p.resolve()):sha(p) for p in paths}


def read_contract(ctx,path,action):
    ctx.require_action(action);ctx.guard()
    path=Path(path).resolve(strict=True);digest=ctx.bind(path,'RECOVERY_PUBLICATION_CONFIG')
    cfg=read_json(path)
    if (cfg.get('schema')!='RECOVERY24_V5_PUBLICATION_V1' or cfg.get('phase_id')!=ctx.c['phase_id']
        or cfg.get('run_id')!=ctx.c['run_id'] or cfg.get('campaign_config_sha256')!=ctx.config_sha256
        or action not in cfg.get('tasks',[]) or set(cfg.get('inputs',{}))!=set(SCHEMAS)):
        raise ValueError('Explicit matching publication authority and complete CSV schema map required')
    if not isinstance(cfg.get('deadline_epoch'),(float,int)) or not math.isfinite(cfg['deadline_epoch']):
        raise ValueError('Finite explicit publication deadline required')
    root=ctx.output_path(cfg['output_relative'])
    if not root.is_relative_to(Path(ctx.root)/'publication'):raise ValueError('New publication namespace required')
    return cfg,digest,root,min(float(cfg['deadline_epoch']),ctx.deadline(action))


def load_inputs(ctx,cfg,deadline):
    data={};records={}
    # Read small method rows first; the link export may contain millions of rows.
    names=['methods']+[x for x in SCHEMAS if x!='methods']
    for name in names:
        ctx.guard(deadline);b=cfg['inputs'][name];p=Path(b['path']).resolve(strict=True)
        if p.suffix.lower()!='.csv':raise ValueError('Scientific publication inputs must be explicit CSVs')
        digest=ctx.bind(p,'PUBLICATION_'+name,b['sha256'])
        if name=='backbone_links':
            primary=data['methods'].loc[data['methods'].scenario.eq('OPEN') &
                (data['methods'].network.eq('P')|data['methods'].suite_id.eq('s03')),'case_id']
            pieces=[]
            for part in pd.read_csv(p,dtype={'link_id':str,'case_id':str},chunksize=50000):
                ctx.guard(deadline);pieces.append(part.loc[part.case_id.isin(primary)])
            frame=pd.concat(pieces,ignore_index=True) if pieces else pd.DataFrame(columns=SCHEMAS[name])
        else:
            frame=pd.read_csv(p,dtype={x:str for x in ['case_id','suite_id','link_id','native_id','model_sha256']})
        if set(SCHEMAS[name])-set(frame):raise ValueError('Missing columns in '+name+': '+str(set(SCHEMAS[name])-set(frame)))
        private=[x for x in frame if x.lower().endswith(('_path','_file','_directory'))]
        data[name]=frame.drop(columns=private)
        records[name]=dict(path=str(p),sha256=digest,bytes=p.stat().st_size,retained_rows=len(frame),
            selection='OPEN P and s03 only' if name=='backbone_links' else 'All supplied rows')
    validate_data(data)
    data['learning_observations']=data['learning']
    data['learning']=normalize_learning(data['learning'])
    return data,records


def validate_data(d):
    m=d['methods']
    validate_training_timing(m)
    if len(m)!=48 or m.case_id.duplicated().any():raise ValueError('All 48 explicit reporting rows required')
    if set(m.scenario)!={'OPEN','REPRESENTED_I95'}:raise ValueError('Exact OPEN/represented-I95 scenario vocabulary required')
    if not set(m.network)<= {'P','AC'}:raise ValueError('Unknown network scope')
    if set(d['resolution_counts'].suite_id)!= {'s02','s03','s04','s05','s06'}:raise ValueError('Five frozen suites required')
    if len(d['ablation_observations'])!=160 or d['ablation_observations'].task_id.duplicated().any():
        raise ValueError('All 160 ablation dispositions required')
    for name in ('backbone_links','region_geometry','region_locations','primary_map_flows'):
        values=d[name][GEO].to_numpy(float)
        if len(values) and (not np.isfinite(values).all() or (np.abs(values[:,[0,2]])>180).any() or (np.abs(values[:,[1,3]])>90).any()):
            raise ValueError('Invalid actual longitude/latitude: '+name)
    for case,frame in d['backbone_links'].groupby('case_id'):
        if len(frame)!=87582 or frame.link_id.duplicated().any() or int(portable.truth(frame.is_key).sum())!=200:
            raise ValueError('Incomplete native B/200 population: '+case)
    summary=d['ablation_summary']
    if len(summary)!=40 or summary.duplicated(['stratum','model_family','training_fraction']).any():
        raise ValueError('Forty explicit stratum/family/fraction dispositions required')


def validate_training_timing(methods):
    if 'recorded_training_seconds' not in methods:return
    values=pd.to_numeric(methods.recorded_training_seconds,errors='raise')
    measured=values.notna()
    if (not np.isfinite(values[measured]).all() or (values[measured]<0).any()
        or measured.any() and ('recorded_training_scope' not in methods
        or not methods.loc[measured,'recorded_training_scope'].eq('RECOVERY_FAMILY_FIT_AND_AUDIT_WORKER_SECONDS').all())):
        raise ValueError('Training timing requires explicit recovery family fit/audit worker-second scope')


def display(value):
    if pd.isna(value):return 'N/A'
    if isinstance(value,(float,np.floating)):return f'{value:.4g}'
    return str(value)


def normalize_learning(frame):
    keys=['suite_id','region_index','family','version']
    if frame.duplicated(keys+['role']).any() or not set(frame.role)<={'TRAIN','VALID','TEST'}:
        raise ValueError('Duplicate or foreign regional role')
    rows=[]
    for identity,group in frame.groupby(keys,dropna=False,sort=True):
        row=dict(zip(keys,identity))
        for field in ('model_sha256','target_met','fallback_component','fallback_reason',
                      'neural_amplitude','neural_amplitude_is_zero','neural_activity_scope'):
            if field not in group:continue
            values=group[field].dropna().unique()
            if len(values)>1:raise ValueError('Model/selection identity changes across reported roles')
            row['selected_family' if field=='fallback_component' else field]=values[0] if len(values) else None
        if set(group.role)!={'TRAIN','VALID','TEST'}:raise ValueError('Every model requires explicit dispositions for all three roles')
        for observation in group.to_dict('records'):
            role=observation['role'];row[role+'_WMAPE_percent']=observation['WMAPE_percent']
            row[role+'_status']=observation['status'];row[role+'_dataset_identity']=observation['dataset_identity']
            if role=='TEST':
                row.update(TEST_absolute_error_sum=observation['absolute_error_sum'],TEST_true_cost_sum=observation['true_cost_sum'],
                    TEST_coordinates=observation['coordinates'],admitted_TEST_inputs=observation['admitted_inputs'],
                    requested_TEST_inputs=observation['requested_inputs'])
        rows.append(row)
    return pd.DataFrame(rows)


def learning_summary(frame):
    rows=[]
    for (suite,family,version),group in frame.groupby(['suite_id','family','version'],dropna=False,sort=True):
        values=pd.to_numeric(group.TEST_WMAPE_percent,errors='raise').to_numpy(float)
        good=np.isfinite(values);values=values[good]
        err=pd.to_numeric(group.TEST_absolute_error_sum,errors='raise').to_numpy(float)
        truth=pd.to_numeric(group.TEST_true_cost_sum,errors='raise').to_numpy(float)
        attached=np.isfinite(err)&np.isfinite(truth)
        if (np.isinf(pd.to_numeric(group.TEST_WMAPE_percent,errors='raise')).any() or np.isinf(err).any() or np.isinf(truth).any()
            or np.any(err[attached]<0) or np.any(truth[attached]<0) or np.any(np.isfinite(err)!=np.isfinite(truth))):
            raise ValueError('Invalid or half-missing native TEST aggregate sums')
        try:
            total=math.fsum(truth[attached]);error=math.fsum(err[attached])
        except OverflowError as exc:raise ValueError('Pooled native sum overflow') from exc
        pooled=100*(error/total) if total>0 else np.nan
        if not math.isfinite(total) or not math.isfinite(error) or total>0 and not math.isfinite(pooled):
            raise ValueError('Pooled native sum/ratio is nonfinite')
        amplitude=pd.to_numeric(group.get('neural_amplitude',pd.Series(np.nan,index=group.index)),errors='raise')
        if np.isinf(amplitude).any() or (amplitude.dropna()<0).any():raise ValueError('Invalid recorded ICNN neural amplitude')
        if 'neural_amplitude_is_zero' in group:
            flag=group.neural_amplitude_is_zero;known=flag.notna()&amplitude.notna()
            if not (portable.truth(flag[known])==amplitude[known].eq(0)).all():raise ValueError('Contradictory Gamma-zero metadata')
        rows.append(dict(suite_id=suite,family=family,version=version,regions=len(group),measured_regions=len(values),
            TRAIN_target_pass=int(portable.truth(group.target_met).sum()),
            mean_TRAIN_percent=float(pd.to_numeric(group.TRAIN_WMAPE_percent,errors='raise').mean()),
            mean_VALID_percent=float(pd.to_numeric(group.VALID_WMAPE_percent,errors='raise').mean()),
            mean_TEST_percent=float(values.mean()) if len(values) else np.nan,
            median_TEST_percent=float(np.median(values)) if len(values) else np.nan,
            p90_TEST_percent=float(np.quantile(values,.9)) if len(values) else np.nan,pooled_TEST_percent=pooled,
            pooled_regions=int(attached.sum()),admitted_TEST_inputs=int(group.admitted_TEST_inputs.sum()),
            requested_TEST_inputs=int(group.requested_TEST_inputs.sum()),
            fallback_regions=int((group.selected_family.isin(['POWER4','FFTT'])&(group.selected_family!=family)).sum()) if version=='portfolio' else 0,
            unavailable_component_regions=int(group.selected_family.isna().sum()) if version=='portfolio' else 0,
            ICNN_zero_Gamma=int(amplitude.eq(0).sum()) if family=='ICNN' else 0,
            ICNN_positive_Gamma=int(amplitude.gt(0).sum()) if family=='ICNN' else 0,
            ICNN_unknown_Gamma=int(amplitude.isna().sum()) if family=='ICNN' else 0))
    return pd.DataFrame(rows)


def asset_folder(root,name,frames,records,*,table=None,columns=None,panels=None):
    folder=root/'numerical_data'/name;folder.mkdir(parents=True,exist_ok=False);parts=[];dictionary=[]
    def emit(filename,frame):
        portable.public_csv(folder/filename,frame,part_registry=parts)
        dictionary.extend(portable.dictionary(frame,filename))
    emit('asset.csv',pd.DataFrame([dict(asset_id=name,title=TITLES[name],caption=CAPTIONS[name])]))
    for filename,frame in frames.items():emit(filename,frame)
    if table is not None:
        emit('table.csv',table);emit('columns.csv',pd.DataFrame(columns,columns=['column','label','width']));script='build_table.py'
    else:
        emit('panels.csv',pd.DataFrame(panels,columns=portable.PANEL_COLUMNS).fillna(''));script='plot.py'
    emit('data_dictionary.csv',pd.DataFrame(dictionary))
    pd.DataFrame(parts,columns=CSV_PART_COLUMNS).to_csv(folder/'csv_parts.csv',index=False)
    text=(TEMPLATES/script).read_text(encoding='utf-8');marker='# EMBED_LOCAL_LOGICAL_CSV_SUPPORT_HERE'
    if text.count(marker)!=1:raise ValueError('CSV-only renderer insertion marker differs')
    (folder/script).write_text(text.replace(marker,(TEMPLATES/'csv_support.py').read_text(encoding='utf-8')),encoding='utf-8')
    _new(folder/'provenance.json',dict(asset_id=name,source_datasets={k:{x:v[x] for x in ('sha256','bytes','retained_rows','selection')} for k,v in records.items()},
        script_sha256=sha(folder/script),input_rule='Only local CSV scientific inputs; no private path dependency',visual_qa='PENDING'))
    (folder/'README.md').write_text('# '+TITLES[name]+'\n\n'+CAPTIONS[name]+'\n\nRun `python '+script+'`. '
        'Only local CSV inputs are read; large CSVs are losslessly sharded with ordered SHA-bound parts. '
        'Keep every part and csv_parts.csv. Python, NumPy, pandas and Matplotlib are sufficient. '
        'Unavailable observations retain explicit statuses; a render receipt does not establish visual QA.\n',encoding='utf-8')
    return folder


def build_tables(root,d,records):
    method_columns=[('effective_method','Method',1.2),('solver','Solver',.6),('status','Status',1.2),
        ('elapsed_seconds','Query seconds',.7),('recorded_training_seconds','Recovery fit/audit worker-s',.8),('RG_SPTT','Gap/VI',.7),('wmape_backbone_percent','B WMAPE (%)',.85),
        ('wmape_key_percent','200 WMAPE (%)',.85),('backbone_cost_error_percent','B cost error (%)',.85)]
    main=d['methods'].loc[d['methods'].scenario.eq('OPEN')&(d['methods'].network.eq('P')|d['methods'].suite_id.eq('s03'))].copy()
    tables={
      'table1':(d['settings'],[('setting','Setting',1),('value','Value',2)],{'observations_networks.csv':d['networks']}),
      'table2':(main,method_columns,{'observations_all_48_methods.csv':d['methods'],'observations_timing.csv':d['timing']}),
    }
    all_summary=learning_summary(d['learning']);summary=all_summary.loc[all_summary.suite_id.eq('s03')].copy()
    summary['method']=summary['version']+' / '+summary['family']
    summary['coverage']=summary.measured_regions.astype(str)+'/'+summary.regions.astype(str)
    summary['holdout']=summary.admitted_TEST_inputs.astype(str)+'/'+summary.requested_TEST_inputs.astype(str)
    tables['table3']=(summary,[('method','Model / version',1.6),('coverage','Finite regions',.8),('TRAIN_target_pass','TRAIN <1%',.75),
        ('median_TEST_percent','TEST median (%)',.85),('mean_TEST_percent','Mean (%)',.8),('p90_TEST_percent','p90 (%)',.8),
        ('pooled_TEST_percent','Pooled (%)',.8),('holdout','Admitted / requested',1.2),('fallback_regions','Fallback regions',.8)],
        {'observations_learning.csv':d['learning_observations'],'observations_all_suites_summary.csv':all_summary})
    fftt=d['methods'].loc[d['methods'].network.eq('AC')&d['methods'].family.eq('FFTT')&d['methods'].scenario.eq('OPEN')].drop(columns=['resolution'],errors='ignore')
    resolution=d['resolution_counts'].merge(fftt,on='suite_id',how='left',validate='one_to_one')
    tables['table4']=(resolution,[('resolution','Resolution',.75),('regions','Regions',.7),('portal_nodes','Portals',.7),
        ('active_pairs','Pairs',.8),('elapsed_seconds','Seconds',.8),('wmape_backbone_percent','B WMAPE (%)',1),
        ('wmape_key_percent','200 WMAPE (%)',1),('backbone_cost_error_percent','B cost error (%)',1)],
        {'observations_all_methods.csv':d['methods'],'observations_counts.csv':d['resolution_counts'],
         'observations_stability.csv':d['partition_stability'],'observations_boundary.csv':d['boundary_jaccard']})
    for name,(frame,columns,extra) in tables.items():
        table=frame.reindex(columns=[x[0] for x in columns]).map(display)
        asset_folder(root,name,extra,records,table=table,columns=columns)
    return tables


def p(letter,title,kind,source,**kw):return portable.panel(letter,title,kind,source,**kw)


def build_figures(root,d,records):
    methods=d['methods'];links=d['backbone_links'].copy()
    links=links.merge(methods[['case_id','effective_method','network','assignment_role']],on='case_id',validate='many_to_one')
    refs=links.loc[links.assignment_role.eq('REFERENCE_SELF')]
    if refs.case_id.nunique()!=1:raise ValueError('Exactly one OPEN reference-self link population is required')
    selected=refs.loc[portable.truth(refs.is_key)].copy()
    errors=links.loc[links.network.eq('AC')&portable.truth(links.is_key)].copy()
    errors['signed_percent']=100*errors.signed_bounded_error
    errors['key_order']=errors.groupby('case_id').native_index.rank(method='first')
    physical=methods.loc[methods.network.eq('P')&methods.scenario.eq('OPEN')&~methods.assignment_role.eq('REFERENCE_SELF')]
    curves=d['convergence'].loc[d['convergence'].case_id.isin(physical.case_id)].merge(physical[['case_id','solver']],on='case_id',validate='many_to_one')
    asset_folder(root,'figure3',{'keys.csv':selected,'reference.csv':refs,'errors.csv':errors,'curves.csv':curves},records,panels=[
        p('a','Frozen 200 spatially distributed backbone links','map','keys.csv',point_markers='true',note='Native endpoint chords; OPEN-reference selection is fixed across all cases.'),
        p('b','Reference backbone-flow distribution','ecdf','reference.csv',x='reference_flow',xlabel='Annual thousand tons',ylabel='Empirical cumulative fraction',xscale='symlog',xlinthresh=1,note='All native B links including exact zero flows.'),
        p('c','Signed bounded flow errors on the fixed 200 links','line','errors.csv',x='key_order',y='signed_percent',group='effective_method',xlabel='Frozen native key order',ylabel='Signed bounded error (%)',note='(prediction-reference)/max(prediction,reference); zero/zero is zero.'),
        p('d','Independent P solver convergence','line','curves.csv',x='elapsed_seconds',y='RG_SPTT',group='solver',xlabel='Recorded solver elapsed seconds',ylabel='Fresh relative gap',yscale='log',threshold=.001,note='Historical complete-query duties and saved iteration timing remain distinguished in the method table.')])
    asset_folder(root,'figure4',{'locations.csv':d['region_locations'],'geometry.csv':d['region_geometry']},records,panels=[
        p('a','Representative region locations','location_map','locations.csv',note='Selected before new TEST outcomes; location is the mean of actual graph coordinates.'),
        *[p(letter,title,'region_map','geometry.csv',filter_column='stratum',filter_value=stratum,
            note='Native P regional endpoint chords; exact physical centroid UID and portal markers.')
            for letter,stratum,title in [('b','large','Large region'),('c','good','Low-VALID-error region'),('d','bad','Difficult-VALID region')]]])
    curves=d['ablation_summary'].copy()
    for col in ['median_WMAPE','q25_WMAPE','q75_WMAPE']:curves[col]=100*pd.to_numeric(curves[col],errors='raise')
    asset_folder(root,'figure5',{'curves.csv':curves,'observations_all_160_fits.csv':d['ablation_observations']},records,panels=[
        p(letter,title,'iqr','curves.csv',x='training_fraction',y='median_WMAPE',group='model_family',filter_column='stratum',filter_value=stratum,
          xlabel='Available unique TRAIN fraction (%)',ylabel='Native TEST WMAPE (%)',yscale='symlog',ylinthresh=.1,
          note='Four regions requested per stratum. Unconnected medians/IQR; per-point n counts available regions. Missing fractions stay absent. Family/fraction populations may differ; this is not a paired learning curve.')
        for letter,stratum,title in [('a','large','Large regions'),('b','small','Small regions'),('c','good','Low-VALID-error regions'),('d','bad','Difficult regions')]])
    counts=d['resolution_counts'];ft=methods.loc[methods.network.eq('AC')&methods.family.eq('FFTT')&methods.scenario.eq('OPEN')].drop(columns=['resolution'],errors='ignore')
    joined=counts.merge(ft,on='suite_id',validate='one_to_one')
    countplot=counts.melt(['suite_id','resolution'],['regions','portal_nodes'],var_name='measure',value_name='count')
    errorplot=joined.melt(['suite_id','resolution'],['wmape_backbone_percent','wmape_key_percent','backbone_cost_error_percent'],var_name='measure',value_name='error_percent')
    sizeplot=counts.melt(['suite_id','resolution'],['active_pairs','mean_physical_arcs'],var_name='measure',value_name='value')
    asset_folder(root,'figure6',{'counts.csv':countplot,'runtime.csv':joined,'errors.csv':errorplot,'sizes.csv':sizeplot,
        'observations_stability.csv':d['partition_stability'],'observations_boundary.csv':d['boundary_jaccard']},records,panels=[
        p('a','Regions and interface nodes','points','counts.csv',x='resolution',y='count',group='measure',xlabel='Frozen resolution',ylabel='Count',yscale='log'),
        p('b','Constant-policy FFTT assignment time','points','runtime.csv',x='resolution',y='elapsed_seconds',xlabel='Frozen resolution',ylabel='Complete-query seconds'),
        p('c','Constant-policy native B accuracy','points','errors.csv',x='resolution',y='error_percent',group='measure',xlabel='Frozen resolution',ylabel='Error (%)',yscale='symlog',ylinthresh=.1),
        p('d','Active pairs and mean regional size','points','sizes.csv',x='resolution',y='value',group='measure',xlabel='Frozen resolution',ylabel='Count / mean physical arcs',yscale='log')])
    maps=d['primary_map_flows'];opened=maps.loc[maps.scenario.eq('open')];closed=maps.loc[maps.scenario.eq('i95')]
    delta=opened.merge(closed[['link_id','flow']],on='link_id',suffixes=('_open','_closed'),validate='one_to_one')
    delta['delta']=delta.flow_closed-delta.flow_open
    closure=d['closure_geometry']
    asset_folder(root,'figure7',{'closure.csv':closure,'open.csv':opened,'closed.csv':closed,'delta.csv':delta,
        'observations_served_OD_membership.csv':d['served_membership']},records,panels=[
        p('a','Represented I-95 closure mask','map','closure.csv',note='Only represented B coordinates are closed; hidden physical interiors are outside this intervention.'),
        p('b','OPEN ICNN-hybrid flows','flow_map','open.csv',y='flow',colorbar='annual thousand tons'),
        p('c','Represented-I95 ICNN-hybrid flows','flow_map','closed.csv',y='flow',colorbar='annual thousand tons'),
        p('d','Closed-minus-open redistribution','flow_map','delta.csv',y='delta',colorbar='absolute change, annual thousand tons',
          note='Solid lines denote gains; dashed lines denote losses. Link traversal changes can include newly unserved OD demand.')])


def extract(ctx,config_path):
    cfg,digest,root,deadline=read_contract(ctx,config_path,'publication');before=code_bindings()
    if root.exists():raise FileExistsError('Preserve prior publication; use a new output version')
    d,records=load_inputs(ctx,cfg,deadline);root.mkdir(parents=True)
    build_tables(root,d,records);ctx.guard(deadline);build_figures(root,d,records)
    provenance=dict(schema='RECOVERY24_NUMERICAL_PROVENANCE_V1',config_sha256=digest,inputs=records,code_sha256=before,
        preserved_nonnumerical_source=True,publication_test_selection=False)
    _new(root/'PROVENANCE.json',provenance)
    index=pd.DataFrame([dict(asset_id=name,title=TITLES[name],folder='numerical_data/'+name,
        status='EXTRACTED_RENDER_PENDING') for name in ASSETS])
    index.to_csv(root/'INDEX.csv',index=False)
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Numerical assets</title>'
        '<h1>Numerical assets</h1><p>CSV inputs and standalone scripts; render and visual QA receipts are separate.</p><ul>'+
        ''.join('<li><a href="'+html.escape(row.folder)+'/README.md">'+html.escape(row.asset_id+': '+row.title)+'</a></li>'
                for row in index.itertuples())+'</ul>',encoding='utf-8')
    if before!=code_bindings():raise ValueError('Publication source changed during extraction')
    for name,binding in cfg['inputs'].items():ctx.bind(binding['path'],'FINAL_PUBLICATION_'+name,binding['sha256'])
    files=[dict(relative_path=p.relative_to(root).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in root.rglob('*') if p.is_file()]
    receipt=dict(status='EXTRACTED_RENDER_PENDING',config_sha256=digest,code_sha256=before,files=files,
        assets=ASSETS,declared_figure_panels=20,main_tables=4,actual_visual_qa=False)
    _new(root/'EXTRACTION_RESULT.json',receipt);ctx.flush_bindings();return receipt


def verify_extraction(ctx,cfg,digest,root):
    result=read_json(root/'EXTRACTION_RESULT.json')
    if result['config_sha256']!=digest or result['code_sha256']!=code_bindings():raise ValueError('Publication source/config changed')
    for r in result['files']:
        if sha(root/r['relative_path'])!=r['sha256']:raise ValueError('Extracted CSV/script/provenance changed')
    for name,b in cfg['inputs'].items():ctx.bind(b['path'],'RECHECK_PUBLICATION_'+name,b['sha256'])
    return result


def render(ctx,config_path):
    cfg,digest,root,deadline=read_contract(ctx,config_path,'publication');verify_extraction(ctx,cfg,digest,root)
    evidence=root/'validation/asset_reproduction';evidence.mkdir(parents=True,exist_ok=False)
    config=dict(python_path=cfg['python_path'],deadline_epoch=deadline,resource_budget=cfg['resource_budget'])
    records=[]
    # Existing audited runner accepts standalone scripts; copied renderers carry
    # their embedded logical-CSV reader and use no repository scientific data.
    for name in ASSETS:
        folder=root/'numerical_data'/name;replicas=[]
        for label in ('A','B'):
            copy=evidence/label/name;shutil.copytree(folder,copy);replicas.append((label,copy))
        native=portable.render_one(folder,config,ctx,evidence/'native'/name,'native')
        checks=[]
        for label,copy in replicas:
            replica=portable.render_one(copy,config,ctx,evidence/('audit_'+label)/name,label)
            if replica['inputs']!=native['inputs'] or replica['csv_read_sha256']!=native['csv_read_sha256'] or replica['status']!=native['status']:
                raise ValueError('Isolated scientific input/disposition differs')
            if set(replica['outputs'])!=set(native['outputs']):raise ValueError('Isolated output list differs')
            for filename,value in native['outputs'].items():
                if filename.endswith(('.png','.tex')) and replica['outputs'][filename]!=value:raise ValueError('Isolated output bytes differ')
            checks.append(replica)
        records.append(dict(asset=name,native=native,independent=checks));print('ASSET_ISOLATION_PASS '+name,flush=True)
    verify_extraction(ctx,cfg,digest,root)
    receipt=dict(status='REPRODUCED_VISUAL_QA_PENDING',config_sha256=digest,code_sha256=code_bindings(),assets=records)
    _new(root/'REPRODUCTION_RESULT.json',receipt);return receipt


def safe_member(root,relative):
    p=Path(relative)
    if p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Unsafe baseline member path')
    destination=(Path(root)/p).resolve()
    if not destination.is_relative_to(Path(root).resolve()):raise ValueError('Baseline path escape')
    return destination


def verify_reproduction(root,digest):
    result=read_json(Path(root)/'REPRODUCTION_RESULT.json')
    if (result.get('status')!='REPRODUCED_VISUAL_QA_PENDING' or result.get('config_sha256')!=digest
        or result.get('code_sha256')!=code_bindings() or [r['asset'] for r in result.get('assets',[])]!=ASSETS):
        raise ValueError('Incomplete or unbound reproduction receipt')
    for row in result['assets']:
        folder=Path(root)/'numerical_data'/row['asset']
        for name,digest in row['native']['outputs'].items():
            if Path(name).name!=name or sha(folder/name)!=digest:raise ValueError('Rendered scientific asset changed')
    return result


def verify_bundle_assets(bundle,records):
    for row in records:
        if sha(safe_member(bundle,row['relative_path']))!=row['sha256']:
            raise ValueError('Copied manuscript asset changed')


def copy_baseline(cfg,bundle,ctx):
    records=[]
    for key in ('baseline','dependencies'):
        section=cfg[key];manifest=Path(section['manifest']['path'])
        ctx.bind(manifest,'V5_'+key.upper(),section['manifest']['sha256'])
        data=read_json(manifest);snapshot=Path(section['snapshot_root']).resolve(strict=True)
        for r in data['members']:
            source=safe_member(snapshot,r['relative_path']);dest=safe_member(bundle,r['relative_path'])
            if source.is_symlink() or not source.is_file() or sha(source)!=r['sha256']:raise ValueError('Baseline member changed')
            if dest.exists():raise ValueError('Baseline/dependency destination collision')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
            if sha(dest)!=r['sha256']:raise ValueError('Baseline copy mismatch')
            records.append(dict(relative_path=r['relative_path'],sha256=r['sha256'],source_kind=key))
    return records


def verify_preservation(bundle,records):
    changed=[]
    for row in records:
        path=safe_member(bundle,row['relative_path'])
        if sha(path)!=row['sha256']:changed.append(row['relative_path'])
    if changed!=['numerical example.tex']:raise ValueError('Only numerical example.tex may change: '+str(changed))
    return dict(status='PASS_NUMERICAL_ONLY_SOURCE_PRESERVATION',existing_members=len(records),changed=changed,
                all_other_existing_members_byte_identical=True)


def v5_table(frame,columns,title,label,note=None):
    escape=table_escape;weight=sum(x[2] for x in columns)
    spec=''.join(r'>{\raggedright\arraybackslash}p{\dimexpr '+f'{w/weight:.4f}'+r'\linewidth-2\tabcolsep\relax}' for _,_,w in columns)
    lines=[r'\begin{table}[htbp]',r'\centering\footnotesize\setlength{\tabcolsep}{2.5pt}',
        r'\caption{'+escape(title)+r'}\label{'+label+r'}',r'\begin{tabular}{'+spec+r'}\toprule',
        ' & '.join(escape(x[1]) for x in columns)+r'\\\midrule']
    for row in frame.reindex(columns=[x[0] for x in columns]).itertuples(index=False,name=None):
        lines.append(' & '.join(escape(display(x)) for x in row)+r'\\')
    lines.append(r'\bottomrule\end{tabular}')
    if note:
        lines.append(r'\par\smallskip{\raggedright\footnotesize '+escape(note)+r'\par}')
    lines.append(r'\end{table}');return '\n'.join(lines)


def scientific_paragraphs(d):
    m=d['methods'];converged=int(m.status.eq('CONVERGED').sum());summary=learning_summary(d['learning'])
    regional=[]
    for r in summary.loc[summary.suite_id.eq('s03')].itertuples():
        regional.append(f'{r.version} {r.family}: {r.TRAIN_target_pass}/{r.regions} TRAIN targets, TEST median {display(r.median_TEST_percent)}%, '
            f'p90 {display(r.p90_TEST_percent)}%, pooled {display(r.pooled_TEST_percent)}% over {r.measured_regions} regions')
        if r.family=='ICNN':
            regional.append(f'For these ICNN-designated records, Gamma is zero in {r.ICNN_zero_Gamma}, positive in {r.ICNN_positive_Gamma}, '
                f'and unknown in {r.ICNN_unknown_Gamma} (unknown includes non-ICNN fallback or unavailable metadata); '
                'positive amplitude is structural metadata, not measured predictive benefit')
    pairs=d['ablation_observations'];fit=int(pairs.fit_status.eq('AVAILABLE').sum());measured=int(pairs.TEST_status.eq('MEASURED').sum())
    accuracy=[]
    for r in m.loc[m.scenario.eq('OPEN')&m.network.eq('AC')&m.suite_id.eq('s03')].itertuples():
        accuracy.append(f'{r.effective_method}: B {display(r.wmape_backbone_percent)}%, fixed-200 {display(r.wmape_key_percent)}% ({r.status})')
    served=d['served_membership'];loss=served.loc[served.population.eq('LOST_CLOSED_VS_OPEN')]
    lost='The source membership export does not supply a lost-OD row.' if loss.empty else f"The represented closure loses {int(loss.aggregate_OD_rows.sum()):,} aggregate OD rows ({int(loss.original_OD_row_multiplicity.sum()):,} original rows) and {display(loss.demand.sum())} annual thousand tons of served demand, without renormalization."
    return [
      'This section reports the current P-ACNHN recovery experiment. Other sections and the inherited supplement are preserved verbatim from v5 and retain historical terminology and claims; the current numerical evidence is delimited here. The backbone, physical regional graph, OD mappings and represented closure are frozen. Models and whole-region fallback choices are fixed across OPEN and I95. Native flow is annual thousand tons.',
      f'The reporting matrix contains {len(m)} rows, including two reference-self rows and 46 computational cases; {converged} rows report CONVERGED. '
      +'OPEN s03 native accuracy is '+('; '.join(accuracy) or 'unavailable')+'. Flow WMAPE uses the ratio of summed absolute differences to summed reference flows, retaining all zeros. The B cost statistic compares aggregate backbone travel costs. Reference preparation includes witness duties absent from independent controls; reused historical and new query timings are distinguished. Queue validation is included in stage wall time, outside per-query clocks.',
      'Regional costs are fit from complete flow vectors. ICNN learns a constrained convex potential; Power4 and XGBoost fit cost targets directly, with an analytic integral only for Power4. Learned methods require the strict current TRAIN target and numerical admission; otherwise the smaller fixed-VALID-error Power4/FFTT is used, with FFTT for ties or unavailable ranking. The raw Power4 arm uses valid Power4 regardless of the target and falls back only when missing or invalid. '
      +' '.join(regional)+'. Pooled native-cost error and equal-region medians have different denominators; neither establishes national backbone accuracy. '
      +f'The separate nested-fraction experiment has {fit}/160 available fits and {measured}/160 measured blind outcomes; unavailable cases are retained. The design specifies subset-only calibration at each fraction, fixed VALID and the same newly frozen 48-request TEST cohort. New TEST targets were opened only after production selections and all 160 fit dispositions, including unattempted fits, were frozen. No TEST outcome selects a model. Coordinate, region and admitted/requested coverage are reported in the data release.',
      'Five fixed residual partitions share identical native B and OD mappings. Constant FFTT isolates partition sensitivity, while complete hybrid outcomes remain in the accompanying CSVs. Adjusted Rand index and arithmetic normalized mutual information compare current common residual directed-arc ownership; boundary Jaccard compares current common interface-node identities. These are descriptive stability statistics, not proof of an optimal resolution. Directed projection/lifting holds on represented frozen support, not all physical route-choice alternatives.',
      lost+' Closed accuracy uses the separately frozen closed P reference and the OPEN-frozen 200 keys. Within each scenario the hybrid and P served populations agree; cross-scenario losses can contribute to link-flow redistribution. The closure is limited to represented B coordinates. No full physical-corridor, observed-event, welfare, emissions or amortized-speedup claim follows.'
    ]


def assemble_v5(ctx,config_path):
    cfg,digest,root,deadline=read_contract(ctx,config_path,'manuscript');verify_extraction(ctx,cfg,digest,root)
    verify_reproduction(root,digest)
    d,records=load_inputs(ctx,cfg,deadline);bundle=root/'v5_numerical_revision';bundle.mkdir(exist_ok=False)
    baseline=copy_baseline(cfg,bundle,ctx);revision=bundle/'numerical_revision';revision.mkdir()
    shutil.copytree(root/'numerical_data',revision/'numerical_data')
    asset_records=[dict(relative_path=p.relative_to(bundle).as_posix(),sha256=sha(p))
                   for p in revision.rglob('*') if p.is_file()]
    tables={}
    for i in range(1,5):
        folder=root/'numerical_data'/f'table{i}'
        tables[i]=(read_logical_csv(folder,'table.csv',{},dtype=str),read_logical_csv(folder,'columns.csv',{},dtype=str))
    paragraphs=scientific_paragraphs(d)
    lines=[r'\section{Numerical experiments}\label{sec:numerical}']
    figures_by_section={1:[3],2:[4,5],3:[6],4:[7]};table_labels=['tab:setting','tab:assignment','tab:regional','tab:partition']
    figure_labels={3:'fig:network_validation',4:'fig:regions',5:'fig:sample_size',6:'fig:partition_sensitivity',7:'fig:i95'}
    for i,(heading,label,paragraph) in enumerate(zip(HEADINGS,LABELS,paragraphs)):
        lines.extend([r'\subsection{'+heading+r'}\label{'+label+r'}',portable.tex_escape(paragraph)])
        if i<4:
            frame,columns=tables[i+1];cols=[(r.column,r.label,float(r.width)) for r in columns.itertuples()]
            lines.append(v5_table(frame,cols,TITLES[f'table{i+1}'],table_labels[i],note=CAPTIONS[f'table{i+1}']))
        for number in figures_by_section.get(i,[]):
            name=f'figure{number}';folder=root/'numerical_data'/name
            dispositions=pd.read_csv(folder/'panel_status.csv',dtype=str)
            lines.extend([r'\begin{figure}[htbp]',r'\centering'])
            for j,row in enumerate(dispositions.itertuples()):
                if row.status=='RENDERED':
                    lines.extend([r'\begin{subfigure}[t]{0.48\linewidth}',r'\centering\includegraphics[width=\linewidth]{numerical_revision/numerical_data/'+name+'/panel_'+row.panel_id+r'.pdf}',
                        r'\caption{'+portable.tex_escape(row.title)+r'}',r'\end{subfigure}'+(r'\hfill' if j%2==0 else r'\par\medskip')])
                else:
                    # A failed experiment is text, never a fabricated chart box.
                    lines.append(r'\par\small '+portable.tex_escape(f'Panel {row.panel_id}: {row.status}. {row.note}')+r'\par')
            lines.extend([r'\caption{'+portable.tex_escape(CAPTIONS[name])+r'}\label{'+figure_labels[number]+r'}',r'\end{figure}',r'\FloatBarrier'])
    target=bundle/'numerical example.tex';target.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    from .latex_compat import write_wrapper
    compatibility=write_wrapper(bundle,output_root=ctx.root,
        expected_main_sha256=next(r['sha256'] for r in baseline if r['relative_path']=='main.tex'))
    preservation=verify_preservation(bundle,baseline)
    _new(bundle/'NUMERICAL_REVISION_PRESERVATION.json',dict(preservation,baseline_members=baseline,inputs=records,
        code_sha256=code_bindings(),numerical_tex_sha256=sha(target),asset_records=asset_records,compatibility_wrapper=compatibility))
    (bundle/'NUMERICAL_REVISION_README.md').write_text('# Numerical-only v5 amendment\n\nOnly numerical example.tex changes among existing members. '
        'All other sections, original figures, supplement and responses retain their original bytes and historical claims. '
        'New numerical assets use numerical_revision/. The main PDF is a newly compiled derivative. '
        'Full source/data/provenance and explicit failed/unavailable experiment outcomes remain included. '
        'Compile main_recovery_compat.tex as the entry point; it inputs byte-identical main.tex and maps only '
        'Unicode U+2011/U+2013/U+2014 to their standard TeX punctuation, restoring missing dash glyphs. '
        'The wrapper adds no package and changes no mathematical command or existing source member. '
        'A successful build is not external scientific acceptance.\n',encoding='utf-8')
    verify_extraction(ctx,cfg,digest,root)
    verify_reproduction(root,digest);verify_bundle_assets(bundle,asset_records)
    _new(root/'MANUSCRIPT_RESULT.json',dict(status='ASSEMBLED_COMPILE_AND_VISUAL_QA_PENDING',config_sha256=digest,
        bundle=str(bundle),preservation=preservation,code_sha256=code_bindings(),numerical_tex_sha256=sha(target),
        compatibility_wrapper=compatibility))
    return bundle


def compile_v5(ctx,config_path):
    cfg,digest,root,deadline=read_contract(ctx,config_path,'manuscript');verify_extraction(ctx,cfg,digest,root)
    assembled=read_json(root/'MANUSCRIPT_RESULT.json');bundle=Path(assembled['bundle'])
    if assembled['config_sha256']!=digest or assembled['code_sha256']!=code_bindings():raise ValueError('Manuscript source changed')
    from .latex_compat import verify_wrapper
    compatibility=assembled['compatibility_wrapper']
    entry=verify_wrapper(bundle,compatibility,output_root=ctx.root)
    evidence=root/'validation/v5_pdf';evidence.mkdir(parents=True,exist_ok=False)
    preservation=read_json(bundle/'NUMERICAL_REVISION_PRESERVATION.json')
    verify_preservation(bundle,preservation['baseline_members'])
    verify_reproduction(root,digest);verify_bundle_assets(bundle,preservation['asset_records'])
    if sha(bundle/'numerical example.tex')!=assembled['numerical_tex_sha256']:raise ValueError('Numerical manuscript changed before compile')
    compiler=Path(cfg['compiler']['path']);ctx.bind(compiler,'APPROVED_PORTABLE_TECTONIC',cfg['compiler']['sha256'])
    poppler=Path(cfg['pdftoppm']['path']);ctx.bind(poppler,'APPROVED_POPPLER',cfg['pdftoppm']['sha256'])
    # Separate cache/output, never modify the original approved toolchain cache.
    cache=evidence/'cache';shutil.copytree(cfg['compiler_cache_source'],cache)
    temporary=evidence/'tmp';temporary.mkdir()
    env=dict(os.environ,TECTONIC_CACHE_DIR=str(cache),XDG_CACHE_HOME=str(cache),
             TEMP=str(temporary),TMP=str(temporary),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[str(compiler),'-X','compile','--only-cached','--keep-logs','--keep-intermediates','--outdir',str(evidence),str(entry)]
    build=portable.run_process(command,bundle,env,ctx,deadline,evidence/'tectonic',cfg['resource_budget']['compile_seconds'])
    pdf=evidence/(entry.stem+'.pdf')
    if not pdf.exists() or pdf.stat().st_size==0:raise RuntimeError('No compiled v5 PDF')
    from pypdf import PdfReader
    count=len(PdfReader(pdf).pages);pages=evidence/'pages';pages.mkdir()
    render=portable.run_process([str(poppler),'-r','120','-png',str(pdf),str(pages/'page')],bundle,env,ctx,deadline,
        evidence/'pdftoppm',cfg['resource_budget']['pdf_render_seconds'])
    images=sorted(pages.glob('page-*.png'))
    if len(images)!=count:raise ValueError('Incomplete actual PDF page render')
    verify_extraction(ctx,cfg,digest,root);verify_preservation(bundle,preservation['baseline_members'])
    verify_reproduction(root,digest);verify_bundle_assets(bundle,preservation['asset_records'])
    if sha(bundle/'numerical example.tex')!=assembled['numerical_tex_sha256']:raise ValueError('Numerical manuscript changed during compile')
    verify_wrapper(bundle,compatibility,output_root=ctx.root)
    warnings=[]
    for log in evidence.glob('*.log'):
        warnings.extend(line for line in log.read_text(errors='replace').splitlines() if any(s in line for s in ('Overfull','Missing character','undefined','LaTeX Warning')))
    shutil.copyfile(pdf,bundle/'NUMERICAL_REVISION_main.pdf')
    result=dict(status='COMPILED_EVERY_PAGE_VISUAL_QA_PENDING',pdf_path=str(pdf),pdf_sha256=sha(pdf),pages=count,
        page_images=[dict(path=str(p),sha256=sha(p)) for p in images],build=build,render=render,warnings=warnings,
        source_preservation=preservation['status'],source_sha256=code_bindings(),external_acceptance=False,
        compatibility_wrapper=compatibility,compiled_entry=str(entry))
    _new(root/'PDF_BUILD_RESULT.json',result);return result
