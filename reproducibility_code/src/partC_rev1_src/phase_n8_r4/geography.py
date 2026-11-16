"""Actual NAD83 road vertices, reference-only selection and closure feasibility audit."""
from pathlib import Path
import sqlite3
import time
import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .context import sources

ARC_TABLE='partC_rev1_directed_arcs_enriched'
NODE_TABLE='partC_rev1_clean_physical_nodes'
FIELDS=['directed_arc_id','physical_link_id','from_node_uid','to_node_uid','direction','class_code',
        'route_type','route_number','sign_route','partC_rev1_faf_stfips',
        'partC_rev1_prompt3_is_final_backbone','partC_rev1_prompt3_backbone_stage','partC_rev1_prompt3_arc_partition_type']


def database(ctx):
    path=ctx.repo.parent/'doe_solver_v3/partC_rev1_cache/partC_rev1_network.sqlite'
    if not path.exists():raise FileNotFoundError(path)
    connection=sqlite3.connect(path.as_uri()+'?mode=ro&immutable=1',uri=True)
    connection.execute('PRAGMA query_only=ON')
    return path,connection


def vertices(connection,ids):
    transform=Transformer.from_crs('EPSG:4269','EPSG:5070',always_xy=True)
    records=[];endpoints=[]
    for begin in range(0,len(ids),500):
        batch=[int(x) for x in ids[begin:begin+500]]
        query='SELECT directed_arc_id,partC_rev1_geom_wkb FROM '+ARC_TABLE+' WHERE directed_arc_id IN ('+','.join('?' for _ in batch)+')'
        for arc,hex_wkb in connection.execute(query,batch):
            geom=shapely.from_wkb(bytes.fromhex(hex_wkb));parts=list(geom.geoms) if hasattr(geom,'geoms') else [geom]
            for part,line in enumerate(parts):
                coordinates=np.asarray(line.coords);x,y=transform.transform(coordinates[:,0],coordinates[:,1])
                for order,(lon,lat,xx,yy) in enumerate(zip(coordinates[:,0],coordinates[:,1],x,y)):
                    records.append((str(arc),part,order,float(lon),float(lat),float(xx),float(yy)))
            start=np.asarray(parts[0].coords)[0];end=np.asarray(parts[-1].coords)[-1]
            endpoints.append((str(arc),float(start[0]),float(start[1]),float(end[0]),float(end[1]),len(parts)))
    frame=pd.DataFrame(records,columns=['physical_arc_id','part_index','vertex_order','longitude','latitude','x','y'])
    frame['source_crs']='EPSG:4269';frame['plot_crs']='EPSG:5070';frame['xy_units']='metres'
    ends=pd.DataFrame(endpoints,columns=['physical_arc_id','start_lon','start_lat','end_lon','end_lat','part_count'])
    return frame,ends


def run(ctx,args):
    src=sources(ctx);dst=ctx.root/'geography';dst.mkdir(exist_ok=False)
    frozen=read_json(ctx.root/'physical_population/POPULATION_FREEZE.json')
    ctx.bind(ctx.root/'physical_population/population.parquet','PREOUTCOME_COMPARISON_POPULATION',frozen['native_sha256'])
    pop=pd.read_parquet(ctx.root/'physical_population/population.parquet')
    path,con=database(ctx);before=path.stat()
    config=ctx.repo/'partC_rev1_configs/partC_rev1_real_preprocess_config.json';ctx.bind(config,'SOURCE_GEOGRAPHIC_CRS')
    try:
        metadata=pd.read_sql_query('SELECT '+','.join(FIELDS)+' FROM '+ARC_TABLE,con)
        metadata['physical_arc_id']=metadata.directed_arc_id.astype(np.int64).astype(str)
        if metadata.physical_arc_id.duplicated().any():raise ValueError('Duplicate source directed arc IDs')
        population=pop.merge(metadata,on='physical_arc_id',how='left',validate='one_to_one')
        if population.directed_arc_id.isna().any():raise ValueError('Physical graph metadata is missing')
        nodes=pd.read_sql_query('SELECT partC_rev1_node_uid,partC_rev1_lon,partC_rev1_lat FROM '+NODE_TABLE,con)
        nodes['physical_node_uid']=nodes.partC_rev1_node_uid.astype(np.int64).astype(str)
        node_index=nodes.set_index('physical_node_uid')
        population['longitude']=population.tail_physical_uid.map(node_index.partC_rev1_lon)
        population['latitude']=population.tail_physical_uid.map(node_index.partC_rev1_lat)
        conus=population.longitude.between(-125,-66)&population.latitude.between(24,50)
        interstate=population.class_code.eq(11)&population.route_type.eq('I')
        candidate=population.common_all_five_suites & interstate & conus & population.x45_flow.gt(0)
        population['comparison_map_candidate']=candidate
        population['map_candidate_exclusion']=np.select([~population.common_all_five_suites,~interstate,~conus,~population.x45_flow.gt(0)],
            ['NOT_COMMON_RECOVERABLE','NOT_CLASS11_INTERSTATE','OUTSIDE_DECLARED_CONUS_MAP','NONPOSITIVE_REFERENCE_FLOW'],default='ELIGIBLE')
        # Each 2-degree source-coordinate cell contributes its top-reference arc
        # before any cell contributes another. No predicted values enter selection.
        c=population.loc[candidate].sort_values(['x45_flow','physical_arc_id'],ascending=[False,True],kind='stable').copy()
        c['grid_lon']=np.floor((c.longitude+180)/2).astype(int);c['grid_lat']=np.floor((c.latitude+90)/2).astype(int)
        c['within_cell_rank']=c.groupby(['grid_lon','grid_lat'],sort=False).cumcount()
        selected=c.sort_values(['within_cell_rank','x45_flow','physical_arc_id'],ascending=[True,False,True],kind='stable').head(100)
        selected.to_csv(dst/'reference_selected_interstate_100.csv',index=False,float_format='%.17g')
        candidate_fields=['physical_arc_id','class_code','route_type','route_number','direction','partC_rev1_faf_stfips',
            'longitude','latitude','x45_flow','common_all_five_suites','comparison_map_candidate','map_candidate_exclusion']
        population[candidate_fields].to_csv(dst/'physical_map_candidate_population.csv',index=False,float_format='%.17g')
        geometry,ends=vertices(con,selected.physical_arc_id.tolist())
        geometry.to_csv(dst/'comparison_100_road_vertices.csv',index=False,float_format='%.17g')
        # Intended full Virginia-to-Florida I-95 mainline, both source travel orientations.
        closure=population.loc[population.class_code.eq(11)&population.route_type.eq('I')&population.route_number.eq('95')&
            population.sign_route.eq('I 95')&population.partC_rev1_faf_stfips.isin(['51','37','45','13','12'])].copy()
        if closure.empty:raise ValueError('Intended physical closure has no source-matched roads')
        geometry,ends=vertices(con,closure.physical_arc_id.tolist());closure=closure.merge(ends,on='physical_arc_id',validate='one_to_one')
        for prefix,node_field in [('start','from_node_uid'),('end','to_node_uid')]:
            uid=closure[node_field].astype(np.int64).astype(str)
            lon=uid.map(node_index.partC_rev1_lon).to_numpy();lat=uid.map(node_index.partC_rev1_lat).to_numpy()
            error=np.maximum(np.abs(lon-closure[prefix+'_lon']),np.abs(lat-closure[prefix+'_lat']))
            closure[prefix+'_endpoint_max_degree_error']=error
            if not np.isfinite(error).all() or np.max(error)>1e-10:raise ValueError('Road geometry orientation disagrees with physical nodes')
        closure['geometry_latitude_direction']=np.sign(closure.end_lat-closure.start_lat).astype(int)
        closure['direction_interpretation']='AB is source digitization; latitude sign is geographic only, not authoritative NB/SB'
        geometry.to_csv(dst/'I95_intended_closed_physical_road_vertices.csv',index=False,float_format='%.17g')
        closure.to_csv(dst/'I95_intended_closed_physical_ids.csv',index=False,float_format='%.17g')
        ledger=pd.read_csv(ctx.root/'physical_population/s03_recovery.csv',dtype={'physical_arc_id':str})
        affected=ledger.loc[ledger.physical_arc_id.isin(closure.physical_arc_id)].copy()
        affected.to_csv(dst/'I95_s03_mapping_audit.csv',index=False)
        inter=ledger.loc[ledger.ownership_role.eq('INTER_B')]
        total=inter.groupby('original_owner_id').size().rename('all_members')
        closed=affected.loc[affected.ownership_role.eq('INTER_B')].groupby('original_owner_id').size().rename('closed_members')
        bundle=pd.concat([total,closed],axis=1).fillna({'closed_members':0});bundle=bundle.loc[bundle.closed_members>0].reset_index()
        bundle['wholly_closed']=bundle.closed_members.eq(bundle.all_members)
        bundle.to_csv(dst/'I95_affected_primitive_bundles.csv',index=False)
        links=src.frame(src.p5/'suites/s03/runtime_links.parquet','S03_CLOSURE_NATIVE_LINKS')
        macro_members=inter.loc[inter.source_logical_role.eq('MACRO'),['source_logical_id','original_owner_id']].drop_duplicates()
        macro=macro_members.merge(bundle,on='original_owner_id',how='inner').groupby('source_logical_id').agg(
            affected_primitive_count=('original_owner_id','size'),wholly_closed_primitives=('wholly_closed','sum')).reset_index()
        macro['can_disable_due_to_closed_essential_constituent']=macro.wholly_closed_primitives.gt(0)
        macro.to_csv(dst/'I95_affected_macro_essential_members.csv',index=False)
        suppressed=affected.ownership_role.eq('SUPPRESSED_B');regional=affected.ownership_role.eq('REGION')
        count=dict(intended_closed_physical_arcs=len(closure),suppressed_B_closed_arcs=int(suppressed.sum()),
            suppressed_B_nodes=int(affected.loc[suppressed,'original_owner_id'].nunique()),
            region_owned_closed_arcs=int(regional.sum()),interB_closed_arcs=int(affected.ownership_role.eq('INTER_B').sum()),
            affected_macros=len(macro),macro_with_closed_essential_constituent=int(macro.can_disable_due_to_closed_essential_constituent.sum()),
            geometry_vertices=len(geometry),source_direction_values=sorted(closure.direction.unique().tolist()))
        if not suppressed.any() and not regional.any():
            raise RuntimeError('Closure representability differs from pre-audit; inspect before deciding execution')
        result=dict(status='UNSUPPORTED_MAPPING',counts=count,
            reason='Intended backbone road closures include suppressed B interior movements with no faithful remaining boundary-state mask. Deleting mapped inter-B links leaves those movements implicitly available.',
            macro_partial_closure_note='Affected macro may be disabled only with a wholly closed essential constituent; per-macro evidence exported.',
            arbitrary_logical_cut_executed=False,baseline_capacity_changes=0,regional_predictor_changes=0,
            B_restoration=False,closure_assignment_executed=False,closed_flow_and_diversion_results='NOT_MEASURED',
            baseline_rule='PRIMARY_BOUND_SAFE when eligible and admitted; otherwise predeclared ANALYTIC_VALID; no result-based choice',
            applicable_authority='03_EXPERIMENTS_AND_METRICS.md section5.5',physical_scenario_is_hypothetical=True)
        write_json(dst/'I95_CLOSURE_STATUS.json',result)
        transformer=Transformer.from_crs('EPSG:4269','EPSG:5070',always_xy=True)
        nodes['x'],nodes['y']=transformer.transform(nodes.partC_rev1_lon.to_numpy(),nodes.partC_rev1_lat.to_numpy())
        nodes['source_crs']='EPSG:4269';nodes['plot_crs']='EPSG:5070';nodes['xy_units']='metres'
        nodes.to_csv(dst/'physical_node_coordinates.csv',index=False,float_format='%.17g')
        after=path.stat()
        if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Read-only source SQLite identity changed')
        write_json(dst/'GEOGRAPHY_PROVENANCE.json',dict(source_path=str(path),source_size_bytes=before.st_size,
            source_mtime_ns=before.st_mtime_ns,source_open_mode='ro immutable query_only',source_crs='EPSG:4269',plot_crs='EPSG:5070',
            relevant_query_outputs_sha256={p.name:sha(p) for p in dst.glob('*.csv')},
            whole_database_sha256='NOT_REHASHED_21GB; extracted rows hashed and checked against accepted physical IDs',
            sample_policy='CONUS class11 Interstate positive-x45 flow, common recoverable; 2degree grid round-robin then descending reference flow, stable ID tie',
            candidate_arcs=int(candidate.sum()),selected_arcs=len(selected),prediction_errors_used=False))
        return dict(status='EXECUTED',selected_map_arcs=len(selected),I95=result)
    finally:con.close()
