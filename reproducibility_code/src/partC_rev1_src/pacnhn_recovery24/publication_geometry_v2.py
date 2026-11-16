"""Additive v2 fixed-topology geometry export; complete frozen BACN coordinates.

ARI/NMI uses the sparse contingency formula already implemented in
phase_n8_r4.stability.information. The population here is the CURRENT common
residual physical arcs; historical scores are never relabeled as current.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ..pacnhn_v1.control import read_json, sha, array_hash
from .ablation import _new

GEO=['tail_longitude','tail_latitude','head_longitude','head_latitude']


def information(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.ndim!=1 or b.shape!=a.shape or not len(a):raise ValueError('Identical nonempty stability population required')
    av,ai=np.unique(a,return_inverse=True);bv,bi=np.unique(b,return_inverse=True)
    keys,count=np.unique(ai.astype(np.int64)*len(bv)+bi,return_counts=True)
    aa=keys//len(bv);bb=keys%len(bv);ca=np.bincount(ai);cb=np.bincount(bi);n=len(a)
    choose=lambda v:float(np.sum(np.asarray(v,float)*(np.asarray(v,float)-1)/2))
    joint=choose(count);ra=choose(ca);rb=choose(cb);total=n*(n-1)/2
    expected=ra*rb/total if total else 0.;maximum=(ra+rb)/2
    mi=float(np.sum(count/n*np.log(count*n/(ca[aa].astype(float)*cb[bb]))))
    ha=float(-np.sum(ca/n*np.log(ca/n)));hb=float(-np.sum(cb/n*np.log(cb/n)))
    return dict(ARI=(joint-expected)/(maximum-expected) if maximum!=expected else 1.,
        NMI_arithmetic=2*mi/(ha+hb) if ha+hb else 1.,common_population_n=n,
        regions_a=len(av),regions_b=len(bv)),pd.DataFrame(dict(label_a=av[aa],label_b=bv[bb],observations=count))


def common_coordinates(node_ids,physical,backbone,retained_nodes):
    """Exact physical/logical-role UIDs plus every frozen retained BACN node.

    Some retained BACN nodes are not endpoints of a B link. Endpoint-only
    geography is therefore incomplete, even though the node remains in the
    common topology. No interpolation, node dropping or new centroid is used.
    """
    def coordinates(frame):
        xy=frame[['longitude','latitude']].to_numpy(float)
        if not np.isfinite(xy).all() or (np.abs(xy[:,0])>180).any() or (np.abs(xy[:,1])>90).any():
            raise ValueError('Missing or invalid native longitude/latitude')
        return xy
    p=physical[['physical_node_uid','partC_rev1_lon','partC_rev1_lat']].copy()
    if p.physical_node_uid.isna().any():raise ValueError('Missing physical coordinate UID')
    p['node_id']=p.physical_node_uid.astype(str)
    p=p.rename(columns={'partC_rev1_lon':'longitude','partC_rev1_lat':'latitude'})[['node_id','longitude','latitude']]
    if p.node_id.duplicated().any():raise ValueError('Duplicate physical coordinate UID')
    coordinates(p)
    b=pd.concat([backbone[[end+'_node_id',end+'_longitude',end+'_latitude']].rename(columns={
        end+'_node_id':'node_id',end+'_longitude':'longitude',end+'_latitude':'latitude'}) for end in ('tail','head')])
    if b.node_id.isna().any():raise ValueError('Missing frozen B endpoint UID')
    b.node_id=b.node_id.astype(str);coordinates(b)
    if (b.groupby('node_id')[['longitude','latitude']].nunique()>1).any().any():raise ValueError('Conflicting frozen B coordinates')
    b=b.drop_duplicates('node_id')
    r=retained_nodes[['node_id','node_type','longitude','latitude']].copy()
    if r.node_id.isna().any() or r.node_type.isna().any():raise ValueError('Missing retained node identity/type')
    r.node_id=r.node_id.astype(str)
    if r.node_id.duplicated().any():raise ValueError('Duplicate retained node UID')
    r=r.loc[r.node_type.eq('BACN'),['node_id','longitude','latitude']]
    if not len(r):raise ValueError('No frozen retained BACN coordinates')
    coordinates(r)
    if not set(b.node_id)<=set(r.node_id):raise ValueError('Frozen B endpoint absent from retained BACN nodes')
    overlap=r.set_index('node_id').reindex(b.node_id)
    if not np.array_equal(coordinates(b),coordinates(overlap)):
        raise ValueError('Frozen B endpoint and retained BACN coordinates differ')
    if set(r.node_id)&set(p.node_id):raise ValueError('Physical and B coordinate namespaces collide')
    lookup=pd.concat([p,r]).set_index('node_id')
    if np.asarray(node_ids).ndim!=1 or not len(node_ids):raise ValueError('Nonempty common node vector required')
    ids=pd.Series(np.asarray(node_ids,str))
    if ids.duplicated().any():raise ValueError('Duplicate common node identity')
    keys=ids.str.replace(r'^n8_od_(?:origin|destination):','',regex=True)
    return coordinates(lookup.reindex(keys))


def segments(indices,tail,head,xy):
    ix=np.asarray(indices,np.int64)
    tail=np.asarray(tail);head=np.asarray(head);xy=np.asarray(xy,float)
    if (tail.ndim!=1 or head.shape!=tail.shape or xy.ndim!=2 or xy.shape[1]!=2 or ix.ndim!=1
            or np.any(ix<0) or np.any(ix>=len(tail)) or not np.isfinite(xy).all()
            or np.any(tail[ix]<0) or np.any(head[ix]<0) or np.any(tail[ix]>=len(xy)) or np.any(head[ix]>=len(xy))):
        raise ValueError('Invalid native geometry indices or coordinates')
    return pd.DataFrame(np.column_stack([xy[tail[ix]],xy[head[ix]]]),columns=GEO)


def stability_frames(suites,owners,mask,interfaces):
    owners=np.asarray(owners);mask=np.asarray(mask,bool)
    if owners.shape!=(len(suites),len(mask)) or not mask.any() or (owners[:,mask]<0).any():
        raise ValueError('All suites must own the identical residual arc population')
    scores=[];contingencies=[];boundary=[]
    sets={s:set(interfaces.loc[interfaces.suite_id.eq(s),'node_id'].astype(str)) for s in suites}
    for i,left in enumerate(suites):
        for j in range(i,len(suites)):
            right=suites[j];score,cross=information(owners[i,mask],owners[j,mask])
            score.update(suite_a=left,suite_b=right,population='CURRENT_COMMON_RESIDUAL_DIRECTED_PHYSICAL_ARCS')
            scores.append(score);cross['suite_a']=left;cross['suite_b']=right;contingencies.append(cross)
            union=sets[left]|sets[right];inter=sets[left]&sets[right]
            boundary.append(dict(suite_a=left,suite_b=right,interface_union=len(union),interface_intersection=len(inter),
                boundary_Jaccard=len(inter)/len(union) if union else 1.,population='CURRENT_COMMON_INTERFACE_NODE_IDS'))
    return pd.DataFrame(scores),pd.concat(contingencies,ignore_index=True),pd.DataFrame(boundary)


def export(ctx, config_path, *, deadline):
    """Explicit metadata/array reads after assignment, writing a NEW export only.

    Configuration binds common_graph, partitions, backbone, interfaces, regions,
    od_endpoints, physical_coordinates, retained_nodes, support_freeze, ablation_plan and
    reporting_results. All fields are {path,sha256}; no discovery/default files.
    """
    ctx.require_action('publication');ctx.guard(deadline)
    source_sha=sha(__file__)
    config_path=Path(config_path);config_sha=ctx.bind(config_path,'GEOMETRY_EXPORT_CONFIG')
    cfg=read_json(config_path)
    if cfg.get('phase_id')!=ctx.c['phase_id'] or cfg.get('run_id')!=ctx.c['run_id']:
        raise ValueError('Foreign geometry export authority')
    inputs=cfg['inputs'];bindings={}
    def path(name):
        b=inputs[name];p=Path(b['path']);ctx.bind(p,'FIXED_GEOMETRY_'+name,b['sha256']);bindings[name]=b;return p
    with np.load(path('common_graph'),allow_pickle=False) as z:
        a={k:z[k].copy() for k in ('tail','head','node_ids','edge_ids','edge_role','backbone_count')}
    with np.load(path('partitions'),allow_pickle=False) as z:
        suites=z['suite_ids'].astype(str).tolist();owners=z['edge_region'].copy()
    if suites!=ctx.c['suite_ids']:raise ValueError('Geometry suites differ from current campaign')
    b=pd.read_parquet(path('backbone'));interfaces=pd.read_parquet(path('interfaces'))
    regions=pd.read_parquet(path('regions'));od=pd.read_parquet(path('od_endpoints'))
    physical=pd.read_csv(path('physical_coordinates'),dtype={'physical_node_uid':str})
    retained=pd.read_parquet(path('retained_nodes'),columns=['node_id','node_type','longitude','latitude'])
    support=read_json(path('support_freeze'));plan=read_json(path('ablation_plan'))
    if support.get('status')!='PASS' or len(plan.get('regions',[]))!=16:raise ValueError('Frozen support/region selection required')
    xy=common_coordinates(a['node_ids'],physical,b,retained);ctx.guard(deadline)
    scores,cross,boundary=stability_frames(suites,owners,a['edge_role']==1,interfaces)
    chosen=[next(r for r in plan['regions'] if r['stratum']==s) for s in ('large','good','bad')]
    mapped=[];locations=[]
    physical_xy=physical.set_index('physical_node_uid')[['partC_rev1_lon','partC_rev1_lat']]
    for r in chosen:
        ctx.guard(deadline);sid='s03';rid=r['region_index'];ix=np.flatnonzero(owners[suites.index(sid)]==rid)
        if not len(ix):raise ValueError('Selected regional geometry has no physical arcs')
        part=segments(ix,a['tail'],a['head'],xy);part['geometry_role']='PHYSICAL_LINK';part['native_id']=a['edge_ids'][ix]
        nodes=np.unique(np.r_[a['tail'][ix],a['head'][ix]])
        ports=interfaces.loc[interfaces.suite_id.eq(sid)&interfaces.region_index.eq(rid)]
        for row in ports.itertuples():
            point=xy[int(row.node_index)]
            part.loc[len(part)]=[*point,*point,'PORTAL',str(row.node_id)]
        for row in od.loc[od.road_node_idx.isin(nodes)].itertuples():
            point=physical_xy.loc[str(row.physical_node_uid)].to_numpy(float)
            part.loc[len(part)]=[*point,*point,'CENTROID',str(row.physical_node_uid)]
        part['stratum']=r['stratum'];part['region_index']=rid;part['suite_id']=sid;part['label']='';mapped.append(part)
        point=xy[nodes].mean(axis=0)
        locations.append(dict(zip(GEO,[*point,*point]),geometry_role='REGION_LOCATION',stratum=r['stratum'],
            region_index=rid,suite_id=sid,label=f"{r['stratum']} r{rid}"))
    counts=[]
    for sid in suites:
        ss=next(x for x in support['suites'] if x['suite_id']==sid)
        rr=regions.loc[regions.suite_id.eq(sid)]
        counts.append(dict(suite_id=sid,resolution=ctx.c['resolutions'][sid],regions=len(rr),
            portal_nodes=interfaces.loc[interfaces.suite_id.eq(sid),'node_id'].nunique(),
            active_regions=len(ss['region_records']),active_pairs=sum(x['pairs'] for x in ss['region_records']),
            mean_physical_arcs=float(rr.physical_arc_count.mean())))
    # Regional shortcuts use the same frozen common-node coordinates, never a
    # regional centroid approximation. The input result must contain exact native
    # state artifacts and a fresh matching flow certificate.
    results=read_json(path('reporting_results'));pair_maps=[];map_dispositions=[]
    ss=next(x for x in support['suites'] if x['suite_id']=='s03')
    graph_path=Path(ctx.source_root)/'acnhn/s03/graph.npz'
    ctx.bind(graph_path,'CURRENT_S03_AC_GRAPH',ss['graph_sha256'])
    bindings['current_s03_graph']={'path':str(graph_path),'sha256':ss['graph_sha256']}
    with np.load(graph_path,allow_pickle=False) as z:
        ag={k:z[k].copy() for k in ('tail','head','edge_ids','backbone_count','pair_keys','closed_mask')}
    nb=int(ag['backbone_count']);nreg=len(ag['pair_keys']);ix=np.arange(nb+nreg)
    for scenario in ('open','i95'):
        match=[r for r in results if r.get('case',{}).get('case_id')==f'AC_s03_ICNN_{scenario}']
        if len(match)!=1:raise ValueError('Missing/duplicate primary closure case')
        result=match[0]
        if result.get('final_certificate_fresh') is not True:
            map_dispositions.append(dict(scenario=scenario,status=result.get('status'),geometry_flow_available=False));continue
        artifact=result['native_artifacts']['final_flow.npz']
        ctx.bind(artifact['path'],'PRIMARY_REGIONAL_FLOW_'+scenario,artifact['sha256'])
        bindings['primary_flow_'+scenario]=artifact
        with np.load(artifact['path'],allow_pickle=False) as z:flow=z['flow'].copy()
        if (len(flow)!=len(ag['tail']) or array_hash({'flow':flow})!=result['certificate']['state_hash']
                or not np.isfinite(flow).all() or np.any(flow<0)):
            raise ValueError('Primary map flow is not the freshly certified state')
        part=segments(ix,ag['tail'],ag['head'],xy);part['link_id']=ag['edge_ids'][ix]
        part['edge_kind']=np.where(ix<nb,'BACKBONE','REGIONAL');part['flow']=flow[ix]
        part['scenario']=scenario;part['is_closed']=ag['closed_mask'][ix];pair_maps.append(part)
        map_dispositions.append(dict(scenario=scenario,status=result['status'],geometry_flow_available=True,
            flow_sha256=artifact['sha256'],state_hash=result['certificate']['state_hash']))
    closed_indices=np.flatnonzero(ag['closed_mask'])
    closure=segments(closed_indices,ag['tail'],ag['head'],xy);closure['link_id']=ag['edge_ids'][closed_indices]
    outputs={'region_geometry.csv':pd.concat(mapped,ignore_index=True),'region_locations.csv':pd.DataFrame(locations),
        'closure_geometry.csv':closure,
        'partition_stability.csv':scores,'stability_contingency.csv':cross,'boundary_jaccard.csv':boundary,
        'resolution_geometry_counts.csv':pd.DataFrame(counts),
        'primary_i95_map_flows.csv':pd.concat(pair_maps,ignore_index=True) if pair_maps else pd.DataFrame(columns=GEO+['link_id','edge_kind','flow','scenario','is_closed'])}
    root=ctx.output_path(cfg['output_relative']);root.mkdir(parents=True,exist_ok=False)
    for name,frame in outputs.items():ctx.guard(deadline);frame.to_csv(root/name,index=False,float_format='%.17g')
    for name,binding in bindings.items():ctx.bind(binding['path'],'FINAL_FIXED_GEOMETRY_'+name,binding['sha256'])
    ctx.bind(graph_path,'FINAL_CURRENT_S03_AC_GRAPH',ss['graph_sha256'])
    if sha(__file__)!=source_sha:raise ValueError('Geometry export source changed')
    receipt=dict(status='EXPORTED_WITH_EXPLICIT_FLOW_COVERAGE',config_sha256=config_sha,source_sha256=source_sha,
        input_bindings=bindings,source_topology_changed=False,partition_tuning=False,
        geometry='Native endpoint chords in longitude/latitude; exact physical centroid UID markers; no invented shape.',
        map_dispositions=map_dispositions,outputs={name:dict(path=str(root/name),sha256=sha(root/name),rows=len(frame)) for name,frame in outputs.items()})
    _new(root/'RESULT.json',receipt);ctx.flush_bindings();return receipt
