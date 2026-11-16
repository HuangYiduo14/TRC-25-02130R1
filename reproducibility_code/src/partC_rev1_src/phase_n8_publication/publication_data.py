"""Authenticated extraction into the seventeen independent publication folders."""
from pathlib import Path
import json
import shutil
import html
import numpy as np
import pandas as pd
from pyproj import Transformer
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .metrics import LABELS,final_new_case,job_timing

FAMILIES={'FREEFLOW_LINEAR':'FFTT','NEW_POWER4_RELATIVE_NNLS':'Power-4','OLD_POWER4_NNLS':'Legacy power-4',
    'POWER4_SKIP_ICNN32':'Power-4 + ICNN','OLD_ICNN_FROZEN':'Legacy ICNN',
    'FFTT':'FFTT','POWER4':'Power-4','ICNN':'ICNN','SKIP_ICNN32':'Power-4 + ICNN',
    'SKIP_ICNN_WITHOUT_FALLBACK':'Power-4 + ICNN without fallback',
    'BEST_NONCONSTANT_FALLBACK':'Best nonconstant with safety fallback','EFFECTIVE_M1':'Best nonconstant with safety fallback'}

STEMS={'figure4':{'geometry':'road_vertices'},'figure5':{'observations':'region_fraction_observations'},
    'figure6':{'active_dimensions':'dimensions'},'figure7':{'backbone_flows':'flow_open_closed','mask_scope':'source_coverage','unavailable_abstract_links':'closure_links'},
    'figure8':{'components':'cost_components','query_costs':'cumulative_costs','offline_scopes':'comparison_scope'},
    'figureS1':{'batch_losses':'optimizer_observations'},'figureS2':{'ownership':'membership'},
    'figureS3':{'fallback':'regional_fallback','safety_queries':'safety_diagnostics'},'figureS4':{'backbone_flows':'flow_open_closed'}}


def public_role(s):
    return {'NEW_OP':'Fresh operating','NEW_STRESS':'Fresh stress','NEW_OPERATING':'Fresh operating',
        'NEW_STRESS_FROZEN':'Fresh stress'}.get(s,str(s).replace('REUSED_NOW_SEEN_','Reused ').replace('E2_AUDIT_OPERATING','operating audit').replace('E2_AUDIT_STRESS','stress audit').replace('E2_','audit ').replace('_',' '))


def write_index(data,index):
    index.to_csv(data/'INDEX.csv',index=False)
    links='\n'.join('<tr><td>'+html.escape(r.asset)+'</td><td>'+html.escape(r.purpose)+'</td><td>'+html.escape(r.kind)+'</td><td><a href="'+html.escape(r.relative_path)+'">'+html.escape(r.relative_path)+'</a></td></tr>' for r in index.itertuples())
    (data/'INDEX.html').write_text('<!doctype html><meta charset="utf-8"><title>Numerical publication assets</title><style>body{font:15px sans-serif;margin:3em}td,th{padding:.4em;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Numerical publication assets</h1><p>Each folder reproduces independently from its own CSVs.</p><table><tr><th>Asset</th><th>Purpose</th><th>Type</th><th>File</th></tr>'+links+'</table>',encoding='utf-8')


def refresh_generated_index(data,assets):
    """Index every final local input and output, including multi-page tables."""
    old=pd.read_csv(data/'INDEX.csv');purposes=old.groupby('asset').purpose.first().to_dict();rows=[]
    known=old.set_index('relative_path').kind.to_dict()
    for asset in assets:
        for path in sorted((data/asset).iterdir()):
            if not path.is_file():continue
            relative=asset+'/'+path.name
            rows.append(dict(asset=asset,purpose=purposes[asset],kind=known.get(relative,path.suffix.lstrip('.').upper()),relative_path=relative))
    write_index(data,pd.DataFrame(rows))


class Release:
    def __init__(self,ctx):
        self.ctx=ctx;self.root=ctx.root/'publication_bundle';self.data=self.root/'numerical_data'
        self.private=self.root/'private_provenance';self.private.mkdir(parents=True,exist_ok=True)
        self.data.mkdir(parents=True,exist_ok=True)
        self.files={};self.bindings={};self.notes={};self.purposes={}

    def read(self,path):
        path=Path(path);h=self.ctx.bind(path,'PUBLICATION_EXACT_INPUT');self.bindings[str(path)]=h
        return pd.read_csv(path,float_precision='round_trip',dtype={'region_id':str,'physical_arc_id':str,'abstract_link_id':str,'input_hash':str})

    def record(self,path):
        path=Path(path);h=self.ctx.bind(path,'PUBLICATION_EXACT_RECORD');self.bindings[str(path)]=h;return read_json(path)

    def csv(self,folder,stem,frame,description):
        stem=STEMS.get(folder,{}).get(stem,stem)
        dst=self.data/folder;dst.mkdir(exist_ok=True);self.files.setdefault(folder,[])
        # Source paths belong only in the private binding index.
        for c in frame.select_dtypes(include=['object','string']).columns:
            if frame[c].astype(str).str.contains(r'[A-Za-z]:[\\/]',regex=True).any():
                raise ValueError('Machine path in public CSV '+folder+'/'+stem+':'+c)
        sample=frame.head(1000).to_csv(index=False,float_format='%.17g').encode()
        step=max(1,min(80000,int(16_000_000/max(1,len(sample)/max(1,min(len(frame),1000))))))
        starts=list(range(0,len(frame),step)) or [0]
        for i,start in enumerate(starts):
            name=stem+'.csv' if len(starts)==1 else stem+f'_{i:03d}.csv';p=dst/name;chunk=frame.iloc[start:start+step]
            if p.exists():raise FileExistsError('Publication extraction cannot silently replace '+str(p))
            chunk.to_csv(p,index=False,float_format='%.17g')
            if p.stat().st_size>25_000_000:raise ValueError('CSV member exceeds limit')
            self.files[folder].append(dict(file=name,rows=len(chunk),columns=list(chunk.columns),description=description,sha256=sha(p)))

    def finish(self):
        source=Path(__file__).with_name('publication_plot.py');inventory=[]
        for folder,files in self.files.items():
            dst=self.data/folder;table=folder.startswith('table');script='build_table.py' if table else 'plot.py'
            shutil.copy2(source,dst/script)
            intro=self.notes.get(folder,'');purpose=self.purposes[folder]
            text=f'# {folder}: {purpose}\n\nRun `python {script}` from any working directory. All drawing inputs are CSVs beside the script.\n\n{intro}\n\n'
            text+='Stable IDs are strings. Flows are annual kilotons; marginal costs are seconds. Percentages and fractions are explicitly named. Missing values are unavailable, never zero-imputed. Geometry uses EPSG:5070 metres. Abstract segments are native endpoint chords, and regional segments represent terminal pairs rather than observed routes.\n\n## Local inputs\n\n'
            text+='\n'.join('- '+r['file']+': '+r['description'] for r in files)+'\n'
            (dst/'README.md').write_text(text,encoding='utf-8')
            dictionary=['# Data dictionary','',intro,'']
            from .publication_dictionary import field_meaning
            for row in files:
                dictionary.extend(['## '+row['file'],row['description'],'','|Field|Meaning|','|---|---|'])
                for c in row['columns']:dictionary.append('|'+c+'|'+field_meaning(c)+'|')
                dictionary.append('')
            (dst/'DATA_DICTIONARY.md').write_text('\n'.join(dictionary),encoding='utf-8')
            write_json(dst/'PROVENANCE.json',dict(drawing_inputs=files,script_sha256=sha(dst/script),
                source_index='Bindings and original paths are in the companion private provenance directory; drawing does not read it.',
                no_model_inference=True,all_observations_local=True))
            for row in files:inventory.append(dict(asset=folder,purpose=purpose,kind='CSV',relative_path=folder+'/'+row['file']))
            for name,kind in [(script,'Code'),('README.md','Readme'),('DATA_DICTIONARY.md','Dictionary'),('PROVENANCE.json','Provenance'),
                    ('table.pdf' if table else folder+'.pdf','PDF'),('table.png' if table else folder+'.png','PNG')]+([('table.tex','TeX')] if table else []):
                inventory.append(dict(asset=folder,purpose=purpose,kind=kind,relative_path=folder+'/'+name))
        write_index(self.data,pd.DataFrame(inventory))
        pd.DataFrame([dict(native_path=k,sha256=v) for k,v in self.bindings.items()]).to_csv(self.private/'SOURCE_BINDINGS.csv',index=False)
        write_json(self.private/'EXTRACTION_MANIFEST.json',dict(files=self.files,source_code_sha256=sha(Path(__file__))))

    def spec(self,asset,title,note='',widths='1'):
        self.purposes[asset]=title;self.notes[asset]=note
        self.csv(asset,'specification',pd.DataFrame([dict(asset_id=asset,title=title,column_widths=widths,caption_note=note)]),'Asset title and explicit rendering configuration')


def geometry(frame,layer='Backbone'):
    rows=[]
    for j,prefix in enumerate(('tail','head')):
        f=frame[['abstract_link_id']].copy();f['feature_id']=f.abstract_link_id;f['layer']=layer;f['part_id']=0;f['vertex_order']=j
        f['x']=frame[prefix+'_x'];f['y']=frame[prefix+'_y'];f['crs']='EPSG:5070';rows.append(f)
    return pd.concat(rows,ignore_index=True).sort_values(['feature_id','vertex_order'])


def primary(release):
    c=release.ctx;root=c.root;met=root/'metrics';ref=release.read(root/'reference/s03/backbone_reference.csv')
    geo=geometry(release.read(met/'s03/backbone_geometry.csv'))
    key=release.read(root/'reference/s03/key_links_200.csv');key['reference_flow']=key.lambda_reference;key['selection_rule_id']='BASELINE_POSITIVE_CONUS_GRID250KM_ROUND_ROBIN'
    ref=ref.rename(columns={'role':'link_role','units':'flow_unit'})
    per=[]
    for method in ('M1','M2','M3'):
        f=release.read(met/'s03'/(method+'_open_per_link.csv'));f['method_label']=LABELS[method];per.append(f)
    per=pd.concat(per,ignore_index=True)
    h=release.read(met/'all_fixed_epoch_histories.csv');h=h.loc[h.suite_id.eq('s03')&h.network_state.eq('OPEN')&h.method_id.isin(['M1','M2','M3'])].copy()
    h['method_label']=h.method_id.map(LABELS)
    h['curve_label']=h.method_label+' / epoch '+h.registry_epoch.astype(int).astype(str)+' / '+h.assignment_role.map({'FINAL_COMPARISON':'comparison','RECOVERY_CAMPAIGN':'recovery'})
    release.spec('figure3','Backbone validation in native abstract coordinates',
        'The complete 87,582-link backbone includes reference zeros; every abstract link contributes once. Fixed 200 IDs were chosen before predictions. Bounded error is (estimate-reference)/max(abs(estimate),abs(reference)), with zero when both vanish. The benchmark is a projected physical CFW CURRENT state, not ground truth.')
    for stem,f,desc in [('reference',ref,'Complete baseline reference and candidate locations'),('per_link',per,'Every method/backbone-coordinate observation; no physical-flow reconstruction'),
            ('key_links',key,'Frozen dispersed 200-link selection including grid cells and ranks'),('geometry',geo,'Native backbone endpoint chords'),('convergence',h,'All fixed-registry objective epochs, including abandoned epochs')]:release.csv('figure3',stem,f,desc)
    selection=release.record(root/'reference/s03/KEY_SELECTION_FREEZE.json')
    release.csv('figure3','selection_metadata',pd.DataFrame([dict(field=k,value=json.dumps(v)) for k,v in selection.items()]),'Frozen baseline-only selection policy, candidate counts and exact selected-file hash')
    summary=release.read(met/'open_summary_metrics.csv');counts=release.read(c.r4/'stability/resolution_counts.csv')
    counts=counts[['suite','resolution','active_regions','inactive_final_regions','final_region_count','active_regional_coordinates','P5_actual_backbone_links','N8_runtime_nodes','N8_runtime_links']].rename(columns={'suite':'suite_id','P5_actual_backbone_links':'backbone_count','N8_runtime_nodes':'runtime_nodes','N8_runtime_links':'runtime_links'})
    b=summary.loc[summary.method_id.eq('M3')&summary.population.eq('BACKBONE')].copy();k=summary.loc[summary.method_id.eq('M3')&summary.population.eq('KEY_LINKS')]
    sm=counts.merge(b[['suite_id','wmape_percent','backbone_cost_error_percent','online_seconds','RG_SPTT','status','reference_flow_sum','absolute_error_sum',
        'link_count','reference_id','state_id','graph_identity','mapping_identity','timing_scope','represented_demand','original_demand']],on='suite_id',validate='one_to_one').rename(columns={
        'wmape_percent':'backbone_wmape','backbone_cost_error_percent':'backbone_cost_error','reference_flow_sum':'backbone_reference_flow_sum',
        'absolute_error_sum':'backbone_absolute_error_sum','link_count':'backbone_link_count'})
    sm=sm.merge(k[['suite_id','wmape_percent','reference_flow_sum','absolute_error_sum','link_count']].rename(columns={
        'wmape_percent':'key_wmape','reference_flow_sum':'key_reference_flow_sum','absolute_error_sum':'key_absolute_error_sum','link_count':'key_link_count'}),on='suite_id',validate='one_to_one')
    dims=release.read(c.r4/'stability/active_dimension_observations.csv').rename(columns={'suite':'suite_id'})
    release.spec('figure6','Resolution sensitivity of constant regional models','Each resolution has its own graph, full backbone, forward reference and frozen 200-link population. All five assignments use constant FFTT regional costs. Exact resolution values, including 0.3107232505953859, are retained.')
    release.csv('figure6','suite_metrics',sm,'Exact five-suite metrics and graph counts');release.csv('figure6','active_dimensions',dims,'All active-region dimensions underlying the cumulative distributions')
    for folder,methods in [('figure7',['M1']),('figureS4',['M2','M3'])]:
        release.spec(folder,'I-95 represented abstract-backbone response','The intended route contains 1,713 physical arcs. Only 1,105 represented source arcs, corresponding to 869 unavailable abstract backbone coordinates, are closed. The 608 suppressed interior arcs are omitted. Healthy macro alternatives remain available physically. No demand renormalization. Open-to-closed change is redistribution, not accuracy.')
        bs=[];rs=[];rgs=[]
        for method in methods:
            f=release.read(met/'s03'/(method+'_backbone_open_closed.csv'));f['method_label']=LABELS[method];bs.append(f)
            r=release.read(met/'s03'/(method+'_regional_open_closed.csv')).rename(columns={'pair_id':'abstract_link_id','open_flow':'flow_open','closed_flow':'flow_closed','signed_change':'flow_delta'})
            r['method_id']=method;r['method_label']=LABELS[method];r['suite_id']='s03';rs.append(r)
            trans=Transformer.from_crs(4269,5070,always_xy=True)
            for prefix in ('tail','head'):r[prefix+'_x'],r[prefix+'_y']=trans.transform(r[prefix+'_longitude'].to_numpy(),r[prefix+'_latitude'].to_numpy())
            rgs.append(geometry(r,'Regional terminal-pair chord'))
        for stem,f,desc in [('backbone_flows',pd.concat(bs),'All backbone open/closed values, including unavailable zero coordinates'),('regional_flows',pd.concat(rs),'All active regional traversal coordinates; no physical recovery'),('geometry',geo,'Native abstract backbone endpoint geometry'),('regional_geometry',pd.concat(rgs).drop_duplicates(),'Terminal-pair chords; separate layer and colour scale')]:release.csv(folder,stem,f,desc)
        freeze=release.record(root/'closure/MASK_FREEZE.json')
        release.csv(folder,'mask_scope',pd.DataFrame([dict(intended_arcs=1713,matched_arcs=1105,omitted_interior_arcs=608,disabled_abstract_links=869,mask_hash=freeze['mask_sha256'])]),'Exact represented scenario counts and mask identity')
        for name in ('matched_physical_arcs','omitted_suppressed_interior_arcs','unavailable_abstract_links','intended_I95_route_metadata'):
            f=release.read(root/'closure'/(name+'.csv'));release.csv(folder,name,f,'Complete mask/source scope audit')
        if folder=='figureS4':release.csv(folder,'method_metrics',release.read(met/'closed_summary_metrics.csv'),'All closed-reference method metrics and termination statuses')
    return summary,sm


def regions(release):
    c=release.ctx;old=c.r4/'regional_geography'
    selection=[('partC_p2_region_0cd73ab14d25f174fdd3','Large region'),('partC_p2_region_f0b86317c9795e062d4a','Low VALID error'),('partC_p2_region_0ce745b1cfa26af63aab','Difficult region')]
    roads=[];locations=[]
    for rid,label in selection:
        f=release.read(old/(rid+'_road_vertices.csv'))
        f=f[['physical_arc_id','part_index','vertex_order','x','y','region_id']].rename(columns={'physical_arc_id':'feature_id','part_index':'part_id'})
        f['abstract_link_id']=f.feature_id;f['layer']='Owned physical road';f['crs']='EPSG:5070';roads.append(f)
        locations.append(dict(region_id=rid,role_label=label,x=float(f.x.mean()),y=float(f.y.mean()),location_rule='MEAN_OF_EXPORTED_ROAD_VERTICES'))
    points=release.read(old/'true_portal_and_centroid_points.csv');points=points.loc[points.region_id.isin([r[0] for r in selection])]
    bg=release.read(c.r4/'geography/comparison_100_road_vertices.csv').rename(columns={'physical_arc_id':'feature_id','part_index':'part_id'})
    bg=bg[['feature_id','part_id','vertex_order','x','y']];bg['abstract_link_id']=bg.feature_id;bg['layer']='National road context';bg['crs']='EPSG:5070'
    release.spec('figure4','Representative regional physical geometry','Owned roads are actual stored geometries; portal and centroid symbols are actual fine-member positions. Multiple terminal roles may share a physical point. National marker locations are illustrative means of road vertices.')
    for stem,f,desc in [('geometry',pd.concat(roads),'Every displayed regional road vertex, multipart order retained'),('terminals',points,'Actual fine-member portal/centroid locations and allowed roles'),('locations',pd.DataFrame(locations),'Representative-region markers and selection labels'),('national_geometry',bg,'Actual stored national road context')]:release.csv('figure4',stem,f,desc)
    region_summary=release.read(old/'region_geography_summary.csv');region_summary=region_summary.loc[region_summary.region_id.isin([r[0] for r in selection])].copy()
    region_summary['figure_role']=region_summary.region_id.map(dict(selection))
    release.csv('figure4','region_summary',region_summary,'Representative native regional dimensions and geometry/terminal coverage')
    cost=release.read(c.root/'learning/cost_wmape/region_fraction_cost_wmape.csv').rename(columns={'family':'model_family','cohort':'role','unique_n':'n_train','selected_update':'selected_updates','optimizer_updates':'executed_updates'})
    cost['model_family']=cost.model_family.map(lambda x:FAMILIES.get(x,x));cost['role']=cost.role.map(public_role)
    cost['label_quality_status']='CERTIFIED_A_OR_B_ONLY; EXCLUSIONS_IN_COVERAGE'
    cost['label_collection_status']=np.where(cost.role.isin(['Fresh operating','Fresh stress']),
        np.where(cost.fit_reused,'POST_FIT_LABELS_REUSED_UNCHANGED','NEW_POST_FIT_LABELS'),'REUSED_SOURCE_AUDIT_NOW_SEEN')
    fits=release.record(c.root/'learning/COMBINED_FIT_INVENTORY.json')
    allowed={'region_id','stratum','seed','fraction','status','unique_n','optimizer_updates','selected_update','optimizer_seconds','validation_seconds',
        'elapsed_seconds','unique_rows_consumed','selected_VALID_score','reused','error_type','subset_input_hash','calibration_input_hash'}
    fit_frame=pd.DataFrame([{k:v for k,v in r.items() if k in allowed} for r in fits])
    coverage=release.read(c.root/'learning/combined_label_coverage.csv')
    if 'error' in coverage and 'error_type' not in coverage:
        coverage['error_type']=coverage.error.map(lambda x:str(x).split('(')[0] if pd.notna(x) else '')
    coverage=coverage.drop(columns=[x for x in coverage if 'path' in x or x in ('error','traceback')],errors='ignore')
    if 'cohort' in coverage:coverage['cohort']=coverage.cohort.map(public_role)
    release.spec('figure5','Regional data requirements','Five nested TRAIN fractions, one seed, 4,000 executed updates per completed neural fit, unchanged VALID checkpoint selection. Bands are region-level median/IQR, not independent-seed uncertainty. Fresh operating/stress labels were collected after each model freeze: eight regional datasets are reused unchanged and eight are newly attempted, as labeled by label_collection_status. All operating, stress and now-seen audit cohorts are retained. Missing/severe labels are excluded only by the fixed A/B policy and remain in coverage.')
    for stem,f,desc in [('observations',cost,'All measured region/fraction/model/cohort ratios of summed native cost errors'),('label_coverage',coverage,'All sixteen regions and planned fresh-label slots, including failures and exclusions'),('fit_coverage',fit_frame,'All reused and added fit statuses with executed and selected counters')]:release.csv('figure5',stem,f,desc)
    selected=release.read(c.root/'learning/selected_regions_16.csv')[['region_id','m_active','trained_n','frozen_VALID_MARE',
        'frozen_TRAIN_VALID_max_MARE','selection_tie_sha256','stratum','rank_within_stratum']].rename(columns={'trained_n':'source_TRAIN_count'})
    release.csv('figure5','selected_regions',selected,'All sixteen preselected regions and frozen TRAIN/VALID selection criteria; independent of later evaluation errors')
    # Query-level numerators/denominators establish all plotted ratios independently.
    for p in sorted((c.root/'learning/cost_wmape').glob('*_query_rows.csv')):
        f=release.read(p);f['cohort']=f.cohort.map(public_role);release.csv('figure5','queries_'+p.stem,f,'Native query-level absolute-error and true-cost sums behind region aggregates')
    return cost,fit_frame,coverage


def supplements(release):
    c=release.ctx;old=c.r4/'numerical_release/fig_S1';new=c.root/'learning'
    losses=[release.read(old/'fixed_data_batch_losses.csv')];valid=[release.read(old/'fixed_data_VALID_checkpoints.csv')];exposure=[release.read(old/'fixed_data_input_exposure_counts.csv')]
    counters=[]
    for episode,reused in [(c.r4,True),(c.root,False)]:
        for row in release.record(episode/'learning/FIT_RESULTS.json'):
            if row.get('fraction')!=1:continue
            checkpoint=row.get('last_saved_checkpoint',{})
            counters.append(dict(region_id=row['region_id'],fraction=row['fraction'],executed_updates=row.get('optimizer_updates',checkpoint.get('last_saved_update')),
                selected_updates=row.get('selected_update',checkpoint.get('selected_update')),status=row['status'],reused=reused))
            if reused:continue
            location=row.get('model_path') or row.get('retained_checkpoint_path') or row.get('evaluation_artifact_path')
            if not location:continue
            dest=Path(location)
            for name,target in [('training_batch_losses.csv',losses),('optimization_curve.csv',valid),('exposure_counts.csv',exposure)]:
                if not (dest/name).exists():continue
                if (dest/name).stat().st_size<=2:
                    release.ctx.bind(dest/name,'EMPTY_OPTIMIZER_LOG_NO_OBSERVATIONS')
                    counters[-1][name.removesuffix('.csv')+'_status']='EMPTY_NO_OBSERVATIONS';continue
                f=release.read(dest/name)
                for k in ('region_id','stratum','seed','fraction'):f[k]=row[k]
                target.append(f)
    lf=pd.concat(losses,ignore_index=True).rename(columns={'updates':'optimizer_updates','robust_training_batch_loss':'loss'})
    vf=pd.concat(valid,ignore_index=True).rename(columns={'updates':'update','validation_MARE_equal_pairs':'VALID_MARE'})
    release.spec('figureS1','Fixed-data optimizer progress','Only the 100%-TRAIN fits are shown. Training loss and intermediate VALID MARE are the recorded optimizer quantities, not cost WMAPE. A selected initialization remains update 0 even after all 4,000 updates were executed.')
    for stem,f,desc in [('batch_losses',lf,'All recorded optimizer batch losses'),('valid_checkpoints',vf,'All fixed-VALID selection checkpoints'),('input_exposure',pd.concat(exposure),'Per-input replacement-sampling exposure counts'),('fit_counters',pd.DataFrame(counters),'Executed/selected counters and fit statuses')]:release.csv('figureS1',stem,f,desc)
    release.spec('figureS2','Resolution ownership and stability','Maps show stored endpoint-midpoint ownership markers selected by stable-ID stride; they are not road polylines or region boundaries. Colours are suite-local categorical labels. Stability uses all 324,602 common final region-owned physical arcs, not map samples.')
    maps=pd.concat([release.read(p) for p in sorted((c.r4/'numerical_release/fig_S2').glob('national_ownership_markers_*.csv'))],ignore_index=True)
    maps=maps.rename(columns={'suite':'suite_id','suite_label':'region_id'})
    release.csv('figureS2','ownership',maps,'All displayed sampled ownership midpoint observations')
    resolution=release.read(c.r4/'stability/resolution_counts.csv')[['suite','resolution']].rename(columns={'suite':'suite_id'})
    release.csv('figureS2','suite_resolutions',resolution,'Exact resolution for each stored ownership suite')
    pointgeo=maps[['physical_arc_id','suite_id','x','y']].copy();pointgeo['feature_id']=pointgeo.suite_id+'|'+pointgeo.physical_arc_id
    pointgeo['abstract_link_id']=pointgeo.physical_arc_id;pointgeo['layer']='Physical ownership midpoint';pointgeo['part_id']=0;pointgeo['vertex_order']=0;pointgeo['crs']='EPSG:5070'
    release.csv('figureS2','geometry',pointgeo,'Actual displayed midpoint geometry; these are points, not inferred boundaries')
    stability=release.read(c.r4/'stability/partition_stability.csv');stability=stability.loc[stability.population.eq('COMMON_FINAL_REGION_OWNED_PHYSICAL_ARCS')]
    release.csv('figureS2','stability',stability,'ARI and NMI on the full common final ownership population')
    for stem,path,desc in [('contingency',c.r4/'stability/stability_contingency_counts.csv','Exact pairwise label contingency counts underlying stability'),('selection',c.r4/'numerical_release/fig_S2/map_display_selection.csv','Stable-ID display thinning policy and complete coverage counts')]:release.csv('figureS2',stem,release.read(path),desc)
    current=release.record(c.root/'registries/M1_CURRENT.json');effective=release.record(c.root/'registries'/(current['name']+'.json'))
    admission=release.read(c.root/'registries/source_TRAIN_VALID_admission.csv').set_index('region_id');rows=[]
    events_by_region={};witnesses={};query_rows=[]
    for p in sorted((c.root/'registries').glob('M1_EPOCH_*.json')):
        for event in release.record(p)['events']:events_by_region.setdefault(event['region_id'],[]).append(event)
    p=c.root/'registries/PREFLIGHT_FAILURES.json';preflight=release.record(p);preflight_hash=sha(p)
    for event in preflight:
        rid=event['region_id'];witnesses.setdefault(rid,set()).add(preflight_hash)
        query_rows.append(dict(region_id=rid,minimum_cost_seconds=event.get('minimum_cost_seconds'),query_role=public_role(event.get('witness','Saved admission')),
            witness_hash=preflight_hash,witness_binding='PREFLIGHT_FAILURE_RECORD',negative_coordinates=len(event.get('negative_coordinates',[]))))
    for discarded in current['discarded_epochs']:
        p=Path(discarded['path'])/'SAFETY_STOP.json'
        if p.exists():
            stop=release.record(p);witness=stop.get('offending_query_sha256',sha(p))
            for event in discarded['failures']:
                rid=event['region_id'];witnesses.setdefault(rid,set()).add(witness)
                query_rows.append(dict(region_id=rid,minimum_cost_seconds=event.get('minimum_cost_seconds'),query_role='Aborted assignment trial',
                    witness_hash=witness,witness_binding='OFFENDING_QUERY_ARRAY',negative_coordinates=len(event.get('negative_coordinates',[]))))
    supplemental=c.root/'registries/all_recorded_witnesses'
    replay=release.read(supplemental/'SCREEN_SUMMARY.csv')
    replay['portfolio']=replay.portfolio.map({'ORIGINAL_NONCONSTANT':'Frozen nonconstant','FINAL_EFFECTIVE':'Effective safety portfolio'})
    for event in release.record(supplemental/'FAILURES.json'):
        query_rows.append(dict(region_id=event['region_id'],minimum_cost_seconds=event.get('minimum_cost_seconds'),
            query_role='Supplemental recorded-query replay: '+('frozen nonconstant' if event['portfolio']=='ORIGINAL_NONCONSTANT' else 'effective safety portfolio'),
            witness_hash=event['input_state_hash'],witness_binding='AUTHENTICATED_OFFENDING_QUERY_NATIVE_STATE',
            negative_coordinates=len(event.get('negative_coordinates',[]))))
    rescue_hash=sha(c.root/'registries/FALLBACK_CHAINS.json')
    for e in effective['rows']:
        a=admission.loc[e['region_id']];events=events_by_region.get(e['region_id'],[])
        rows.append(dict(region_id=e['region_id'],source_kind=a.source_kind,source_hash=a.source_sha256,effective_kind=e['model_kind'],
            effective_hash=e['portable_sha256'],effective_family=FAMILIES.get(e['model_kind'],e['model_kind']),epoch=effective['epoch'],
            reason=' | '.join(str(x.get('reason','Screened unsafe query')) for x in events) or 'SOURCE_RETAINED',
            witness_hash=' | '.join(sorted(witnesses.get(e['region_id'],set()))) or (rescue_hash if events else ''),native_order_hash=e['pair_order_hash'],
            was_replaced=a.source_sha256!=e['portable_sha256'],
            baseline_exposure=a.baseline_regional_traversal_flow,source_minimum_cost_seconds=a.source_minimum_observed_cost_seconds))
    fallback=pd.DataFrame(rows);query=fallback[['region_id','source_minimum_cost_seconds']].rename(columns={'source_minimum_cost_seconds':'minimum_cost_seconds'})
    query['query_role']='Saved TRAIN/VALID/guard minimum'
    query=pd.concat([query,pd.DataFrame(query_rows)],ignore_index=True)
    release.spec('figureS3','Whole-region safety fallback','Saved minima and every failed trial trigger only whole-region replacement. Final composition and baseline traversal exposure are reported separately. Nonnegative predictions below FFTT remain fidelity diagnostics, not fallback triggers. All objective epochs and witness bindings are in private provenance.')
    release.csv('figureS3','fallback',fallback,'Complete 1,566-region source/effective model ledger');release.csv('figureS3','safety_queries',query,'Saved minima and all discovered negative-query minima')
    release.csv('figureS3','recorded_query_replay',replay,'Late supplemental audit of all twelve recorded witnesses, deduplicated to eight native vectors; initial warm-witness omission disclosed in the handoff')
    floor=release.read(c.root/'regional_metrics/region_registry_metrics.csv')
    floor=floor[['region_id','registry','cohort','coordinate_observations','negative_coordinates','below_fftt_coordinates','minimum_cost_seconds']]
    floor['registry']=floor.registry.map(lambda x:FAMILIES.get(x,x));floor['cohort']=floor.cohort.map(public_role)
    release.csv('figureS3','floor_diagnostics',floor,'All native cost comparisons below FFTT and below zero, separately; positive below-floor costs never trigger replacement')
    return fallback
