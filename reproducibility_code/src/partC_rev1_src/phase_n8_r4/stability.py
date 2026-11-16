"""Final ownership stability on common physical arcs, with raw partition distinguished."""
import math
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .context import sources
from .diagnostic_handoffs import bounded,progress

SUITE_DIRS={'s02':'phase2_suite_02_r0p310723250595','s03':'phase2_suite_03_r1',
            's04':'phase2_suite_04_r1p6','s05':'phase2_suite_05_r4','s06':'phase2_suite_06_r12'}
COUNT_COLUMNS=['suite','resolution','raw_pre_correction_region_count','raw_partition_nodes','final_region_count',
    'active_regions','inactive_final_regions','regional_owned_physical_arcs','final_backbone_physical_arcs',
    'physical_portal_nodes','active_regional_coordinates','pre_superport_runtime_nodes','packed_links',
    'runtime_regional_links','raw_partition_is_final','backbone_is_constant_across_resolution',
    *['P3_'+key for key in ('bacn_count','racn_count','summarized_fixed_rule_racn_count','singleton_ineligible_racn_control_count',
        'terminal_count','actual_online_node_count','actual_static_link_count','exact_regional_pair_count',
        'full_logical_link_count','regional_support_region_count')],
    *['P3_'+kind+'_terminal_copies' for kind in ('BACN','RACN','ORIGIN_CENTROID','DESTINATION_CENTROID')],
    *['P5_'+key for key in ('retained_B_nodes','retained_nodes','actual_backbone_links','terminal_connectors','actual_total_links')],
    'P5_count_source','P5_count_source_sha256','N8_runtime_nodes','N8_OD_wrapper_nodes','N8_OD_attachment_links',
    'N8_runtime_links','P3_count_source','P3_count_source_sha256','N8_count_source','N8_count_source_sha256']
STABILITY_COLUMNS=['ARI','NMI_arithmetic','common_population_n','regions_a','regions_b','MI_nats',
    'entropy_a_nats','entropy_b_nats','population','suite_a','suite_b']
BOUNDARY_COLUMNS=['suite_a','suite_b','physical_nodes_union','physical_nodes_intersection','boundary_Jaccard','population']


def information(a,b):
    av,ai=np.unique(np.asarray(a,str),return_inverse=True);bv,bi=np.unique(np.asarray(b,str),return_inverse=True)
    n=len(ai)
    if n==0:raise ValueError('No common stability observations')
    keys,count=np.unique(ai.astype(np.int64)*len(bv)+bi,return_counts=True)
    aa=keys//len(bv);bb=keys%len(bv);ca=np.bincount(ai);cb=np.bincount(bi)
    choose=lambda v:float(np.sum(np.asarray(v,dtype=np.float64)*(np.asarray(v,dtype=np.float64)-1.)/2.))
    joint=choose(count);ra=choose(ca);rb=choose(cb);total=n*(n-1)/2.
    expected=ra*rb/total if total else 0.;maximum=(ra+rb)/2
    ari=(joint-expected)/(maximum-expected) if maximum!=expected else 1.
    mi=float(np.sum(count/n*np.log(count*n/(ca[aa].astype(float)*cb[bb]))))
    ha=float(-np.sum(ca/n*np.log(ca/n)));hb=float(-np.sum(cb/n*np.log(cb/n)))
    nmi=2*mi/(ha+hb) if ha+hb>0 else 1.
    table=pd.DataFrame(dict(label_a=av[aa],label_b=bv[bb],physical_observations=count))
    return dict(ARI=ari,NMI_arithmetic=nmi,common_population_n=n,regions_a=len(av),regions_b=len(bv),
                MI_nats=mi,entropy_a_nats=ha,entropy_b_nats=hb),table


def _run(ctx,args):
    src=sources(ctx);dst=ctx.root/'stability';completed=[]
    p2=src.base/'phase_02/split_mapping_campaign/phase2_split_mapping_20260817_v1_run_001/suites'
    final_labels={};raw_labels={};boundary={};counts=[];dimensions=[];arc_frames={}
    for sid in ctx.c['active_suites']:
        ctx.guard();progress(dst,'STABILITY_STATES.json',completed,active=dict(unit='inputs_'+sid))
        root=p2/SUITE_DIRS[sid]
        src.bind(root/'artifact_manifest.csv',sid+'_ACCEPTED_PHASE2_ARTIFACT_MANIFEST')
        manifest=pd.read_csv(root/'artifact_manifest.csv')
        # P2 manifests use relative_path/sha256, bound independently before reads.
        for row in manifest.itertuples(index=False):
            key=getattr(row,'relative_path',getattr(row,'path',None))
            if key is not None:src.expected[str((root/key).resolve())]=row.sha256
        raw=src.frame(root/'node_partition_membership.parquet',sid+'_RAW_PRECORRECTION_PARTITION')
        incident=src.frame(root/'node_incident_registry.parquet',sid+'_FINAL_PHYSICAL_BOUNDARIES')
        arcs=src.frame(root/'directed_arc_membership.parquet',sid+'_FINAL_PHYSICAL_ARC_OWNERSHIP')
        raw_labels[sid]=raw.set_index(raw.partC_rev1_node_uid.astype(np.int64).astype(str)).region_uuid.astype(str)
        mask=~arcs.is_final_backbone.astype(bool)
        labels=arcs.loc[mask].set_index(arcs.loc[mask].directed_arc_id.astype(np.int64).astype(str)).suite_label.astype(str)
        if labels.index.duplicated().any():raise ValueError('Duplicate final owned physical arc')
        final_labels[sid]=labels
        boundary[sid]=set(incident.loc[incident.is_physical_portal.astype(bool),'partC_rev1_node_uid'].astype(np.int64).astype(str))
        s=src.suite(sid);dims=s['pairs'].groupby('region_id',sort=False).size()
        dimensions.append(pd.DataFrame(dict(suite=sid,region_id=dims.index,m_active=dims.values)))
        values=raw.resolution.unique()
        if len(values)!=1:raise ValueError('Ambiguous source resolution')
        counts.append(dict(suite=sid,resolution=float(values[0]),raw_pre_correction_region_count=raw.region_uuid.nunique(),
            raw_partition_nodes=len(raw),final_region_count=labels.nunique(),active_regions=len(dims),
            inactive_final_regions=labels.nunique()-len(dims),regional_owned_physical_arcs=int(mask.sum()),
            final_backbone_physical_arcs=int((~mask).sum()),physical_portal_nodes=len(boundary[sid]),
            active_regional_coordinates=int(dims.sum()),pre_superport_runtime_nodes=len(s['nodes']),
            packed_links=s['packed_count'],runtime_regional_links=len(s['pairs']),
            raw_partition_is_final=False,backbone_is_constant_across_resolution=False))
        p3=src.p3/'suites'/sid/'parquet'
        metadata_path=p3/'suite_metadata.json';src.bind(metadata_path,sid+'_P3_GRAPH_STAGE_COUNTS')
        metadata=read_json(metadata_path)
        for key in ('bacn_count','racn_count','summarized_fixed_rule_racn_count','singleton_ineligible_racn_control_count',
                    'terminal_count','actual_online_node_count','actual_static_link_count','exact_regional_pair_count',
                    'full_logical_link_count','regional_support_region_count'):
            counts[-1]['P3_'+key]=int(metadata[key])
        terms=s['terms']
        for kind in ('BACN','RACN','ORIGIN_CENTROID','DESTINATION_CENTROID'):
            counts[-1]['P3_'+kind+'_terminal_copies']=int(terms.terminal_type.eq(kind).sum())
        p5counts=src.p5/'suites'/sid/'counts.json';src.bind(p5counts,sid+'_P5_COMPILED_GRAPH_STAGE_COUNTS')
        compiled=read_json(p5counts)
        for key in ('retained_B_nodes','retained_nodes','actual_backbone_links','terminal_connectors','actual_total_links'):
            counts[-1]['P5_'+key]=int(compiled[key])
        if compiled['retained_nodes']!=len(s['nodes']) or compiled['actual_total_links']!=s['packed_count']:
            raise ValueError('Compiled metadata and reopened graph counts differ')
        counts[-1].update(P5_count_source=str(p5counts),P5_count_source_sha256=sha(p5counts))
        coverage_path=ctx.prior/'graphs'/sid/'IDENTITY_AND_COVERAGE_SELF_OD_V2.json'
        if not coverage_path.exists():coverage_path=coverage_path.with_name('IDENTITY_AND_COVERAGE.json')
        ctx.bind(coverage_path,sid+'_FROZEN_N8_GRAPH_STAGE_COUNTS');coverage=read_json(coverage_path)
        if coverage['endpoint_mapping_version']!='EXACT_NATIVE_ROLES_WITH_EMPTY_SELF_OD_V2':
            raise ValueError('Historical graph-stage metadata is not the accepted self-OD graph')
        if coverage['packed_links']!=s['packed_count'] or coverage['regional_links']!=len(s['pairs']):
            raise ValueError('Graph-stage counts differ from reopened runtime')
        counts[-1].update(N8_runtime_nodes=int(coverage['nodes']),
            N8_OD_wrapper_nodes=int(coverage['nodes'])-len(s['nodes']),
            N8_OD_attachment_links=int(coverage['endpoint_attachment_links']),
            N8_runtime_links=int(coverage['packed_links']+coverage['regional_links']+coverage['endpoint_attachment_links']),
            P3_count_source=str(metadata_path),P3_count_source_sha256=sha(metadata_path),
            N8_count_source=str(coverage_path),N8_count_source_sha256=sha(coverage_path))
        arc_frames[sid]=arcs[['directed_arc_id','tail_node_uid','head_node_uid','suite_label','is_final_backbone']].copy()
        pd.DataFrame(counts,columns=COUNT_COLUMNS).to_csv(dst/'resolution_counts.csv',index=False,float_format='%.17g')
        pd.concat(dimensions,ignore_index=True).to_csv(dst/'active_dimension_observations.csv',index=False)
        completed.append(dict(unit='inputs_'+sid,status='MEASURED'));progress(dst,'STABILITY_STATES.json',completed)
    ctx.guard();progress(dst,'STABILITY_STATES.json',completed,active=dict(unit='five_suite_common_population'))
    common=pd.concat(final_labels,axis=1,join='inner').sort_index();common.index.name='physical_arc_id'
    raw_common=pd.concat(raw_labels,axis=1,join='inner').sort_index();raw_common.index.name='physical_node_uid'
    if common.empty or raw_common.empty:raise ValueError('Empty common ownership population')
    common.to_csv(dst/'common_final_regional_arc_labels.csv',float_format='%.17g')
    raw_common.to_csv(dst/'common_raw_precorrection_node_labels.csv',float_format='%.17g')
    completed.append(dict(unit='five_suite_common_population',status='MEASURED'))
    progress(dst,'STABILITY_STATES.json',completed)
    metrics=[];tables=[]
    for population,data in [('COMMON_FINAL_REGION_OWNED_PHYSICAL_ARCS',common),('RAW_PRECORRECTION_COMMON_PHYSICAL_NODES',raw_common)]:
        for i,a in enumerate(ctx.c['active_suites']):
            for b in ctx.c['active_suites'][i:]:
                ctx.guard();unit=population+'_'+a+'_'+b
                progress(dst,'STABILITY_STATES.json',completed,active=dict(unit=unit))
                result,table=information(data[a].to_numpy(),data[b].to_numpy())
                result.update(population=population,suite_a=a,suite_b=b);metrics.append(result)
                table['population']=population;table['suite_a']=a;table['suite_b']=b;tables.append(table)
                pd.DataFrame(metrics,columns=STABILITY_COLUMNS).to_csv(dst/'partition_stability.csv',index=False,float_format='%.17g')
                pd.concat(tables,ignore_index=True).to_csv(dst/'stability_contingency_counts.csv',index=False)
                completed.append(dict(unit=unit,status='MEASURED'));progress(dst,'STABILITY_STATES.json',completed)
    ctx.guard();progress(dst,'STABILITY_STATES.json',completed,active=dict(unit='physical_portal_sets'))
    b_rows=[]
    for i,a in enumerate(ctx.c['active_suites']):
        for b in ctx.c['active_suites'][i:]:
            union=boundary[a]|boundary[b];intersection=boundary[a]&boundary[b]
            b_rows.append(dict(suite_a=a,suite_b=b,physical_nodes_union=len(union),physical_nodes_intersection=len(intersection),
                boundary_Jaccard=len(intersection)/len(union) if union else 1.,population='ACCEPTED_PHYSICAL_NODE_IDS'))
    all_boundary=sorted(set.union(*boundary.values()),key=int)
    boundaries=pd.DataFrame(dict(physical_node_uid=all_boundary))
    for sid in ctx.c['active_suites']:boundaries[sid+'_is_physical_portal']=boundaries.physical_node_uid.isin(boundary[sid])
    boundaries.to_csv(dst/'physical_boundary_node_membership.csv',index=False)
    pd.DataFrame(metrics).to_csv(dst/'partition_stability.csv',index=False,float_format='%.17g')
    pd.concat(tables,ignore_index=True).to_csv(dst/'stability_contingency_counts.csv',index=False)
    pd.DataFrame(b_rows).to_csv(dst/'boundary_Jaccard.csv',index=False,float_format='%.17g')
    pd.DataFrame(counts).to_csv(dst/'resolution_counts.csv',index=False,float_format='%.17g')
    pd.concat(dimensions,ignore_index=True).to_csv(dst/'active_dimension_observations.csv',index=False)
    completed.append(dict(unit='physical_portal_sets',status='MEASURED'));progress(dst,'STABILITY_STATES.json',completed)
    ctx.bind(ctx.root/'geography/physical_node_coordinates.csv','FROZEN_TRUE_PHYSICAL_NODE_COORDINATES')
    coords=pd.read_csv(ctx.root/'geography/physical_node_coordinates.csv',dtype={'physical_node_uid':str})
    xy=coords.set_index('physical_node_uid')
    for sid,arcs in arc_frames.items():
        ctx.guard();progress(dst,'STABILITY_STATES.json',completed,active=dict(unit='map_'+sid))
        # The maps explicitly show physical arc midpoint ownership, not fabricated boundaries.
        arcs['physical_arc_id']=arcs.directed_arc_id.astype(np.int64).astype(str)
        tail=arcs.tail_node_uid.astype(np.int64).astype(str);head=arcs.head_node_uid.astype(np.int64).astype(str)
        arcs['x']=(tail.map(xy.x)+head.map(xy.x))/2;arcs['y']=(tail.map(xy.y)+head.map(xy.y))/2
        arcs['longitude']=(tail.map(xy.partC_rev1_lon)+head.map(xy.partC_rev1_lon))/2
        arcs['latitude']=(tail.map(xy.partC_rev1_lat)+head.map(xy.partC_rev1_lat))/2
        arcs['plot_crs']='EPSG:5070';arcs['geometry_role']='PHYSICAL_ARC_ENDPOINT_MIDPOINT_OWNERSHIP_MARKER'
        arcs.to_parquet(dst/(sid+'_map_observations.parquet'),index=False)
        completed.append(dict(unit='map_'+sid,status='MEASURED'));progress(dst,'STABILITY_STATES.json',completed)
    result=dict(status='EXECUTED',suites=ctx.c['active_suites'],common_final_owned_arcs=len(common),raw_common_nodes=len(raw_common),
        final_ARInormalization='Same physical arcs owned by regions in every suite; numeric region labels treated categorically',
        raw_partition_warning='Raw partitions precede small-region correction and include later backbone absorptions',
        interpretation='Constant-regional FFTT resolution sensitivity; backbone changes; not learned all-resolution accuracy',
        common_labels_sha256=sha(dst/'common_final_regional_arc_labels.csv'))
    write_json(dst/'RESULT.json',result);return result


def run(ctx,args):
    if (ctx.root/'stability').exists():raise FileExistsError('Preserve existing stability results before explicit recovery')
    suites=ctx.c['active_suites']
    plans=[dict(unit='inputs_'+sid,scope='COMPLETE_ACCEPTED_SUITE_INPUTS_AND_COUNTS',suite=sid) for sid in suites]
    plans.append(dict(unit='five_suite_common_population',scope='INTERSECTION_REQUIRES_ALL_FIVE_SUITES'))
    for population in ('COMMON_FINAL_REGION_OWNED_PHYSICAL_ARCS','RAW_PRECORRECTION_COMMON_PHYSICAL_NODES'):
        plans.extend(dict(unit=population+'_'+a+'_'+b,scope=population,suite_a=a,suite_b=b)
            for i,a in enumerate(suites) for b in suites[i:])
    plans.append(dict(unit='physical_portal_sets',scope='ACCEPTED_PHYSICAL_NODE_IDS_ALL_FIVE_SUITES'))
    plans.extend(dict(unit='map_'+sid,scope='TRUE_ENDPOINT_MIDPOINT_GEOMETRY',suite=sid) for sid in suites)
    schemas={'resolution_counts.csv':COUNT_COLUMNS,'active_dimension_observations.csv':['suite','region_id','m_active'],
        'partition_stability.csv':STABILITY_COLUMNS,'boundary_Jaccard.csv':BOUNDARY_COLUMNS,
        'stability_contingency_counts.csv':['label_a','label_b','physical_observations','population','suite_a','suite_b'],
        'common_final_regional_arc_labels.csv':['physical_arc_id']+suites,
        'common_raw_precorrection_node_labels.csv':['physical_node_uid']+suites,
        'physical_boundary_node_membership.csv':['physical_node_uid']+[sid+'_is_physical_portal' for sid in suites]}
    return bounded(ctx,args,'stability',plans,['unit'],'STABILITY_STATES.json',schemas,_run,reserve=60)
