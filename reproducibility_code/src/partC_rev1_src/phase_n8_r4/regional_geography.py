"""Source road polylines and true fine-member locations for frozen representative regions."""
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,write_json,sha
from .geography import database,vertices
from .learning_data import RegionalSources
from .diagnostic_handoffs import bounded,progress

SUMMARY_COLUMNS=['region_id','figure_role','active_dimension','regional_physical_arcs','requested_physical_arcs',
    'geometry_ID_coverage_exact','regional_physical_nodes','true_road_vertices','terminal_role_records',
    'unique_terminal_physical_points','centroid_role_records','mean_x','mean_y','location_marker_definition',
    'selection_is_preoutcome']
POINT_COLUMNS=['region_id','terminal_id','terminal_index','terminal_type','physical_node_uid',
    'x','y','longitude','latitude','plot_crs','source_crs','xy_units']


def selected_regions(ctx):
    select_path=ctx.root/'learning/selected_regions_16.csv';ctx.bind(select_path,'PREOUTCOME_REGION_SELECTION_DOCKET')
    selection=pd.read_csv(select_path)
    chosen=selection.loc[selection.rank_within_stratum.eq(1)].copy()
    chosen['figure_role']=chosen.stratum
    zero_path=ctx.root/'learning/large_zero_E2_docket.csv';m28_path=ctx.root/'learning/m28_docket.csv'
    ctx.bind(zero_path,'PREOUTCOME_LARGE_ZERO_UPDATE_DOCKET');ctx.bind(m28_path,'PREOUTCOME_NAMED_M28_DOCKET')
    zero=pd.read_csv(zero_path).head(1).copy();zero['figure_role']='LARGE_ZERO_E2_UPDATES_DIAGNOSTIC'
    m28=pd.read_csv(m28_path)
    m28=m28.loc[m28.region_id.eq('partC_p2_region_0ce745b1cfa26af63aab')].copy();m28['figure_role']='NAMED_M28_DIAGNOSTIC'
    selected=pd.concat([chosen,zero,m28],ignore_index=True)
    roles=selected.groupby('region_id',sort=False).figure_role.agg('|'.join)
    selected=selected.drop_duplicates('region_id').copy();selected['figure_role']=selected.region_id.map(roles)
    return selected


def _run(ctx,args,selected):
    dst=ctx.root/'regional_geography';select_path=ctx.root/'learning/selected_regions_16.csv'
    selected.to_csv(dst/'selected_region_metadata.csv',index=False,float_format='%.17g')
    source=RegionalSources(ctx);con=None
    try:
        path,con=database(ctx);before=path.stat()
        coordinate_path=ctx.root/'geography/physical_node_coordinates.csv'
        ctx.bind(coordinate_path,'FROZEN_TRUE_PHYSICAL_NODE_COORDINATES')
        coordinates=pd.read_csv(coordinate_path,dtype={'physical_node_uid':str})
        xy=coordinates.set_index('physical_node_uid')
        terminals=source.source.frame6('registry/coarse_terminals.parquet')
        fine=source.source.frame6('registry/terminal_fine_members.parquet')
        terminals=terminals.loc[terminals.region_id.isin(selected.region_id)]
        fine=fine.loc[fine.region_id.isin(selected.region_id)]
        points=fine.merge(terminals,on=['region_id','terminal_id','terminal_index'],validate='many_to_one')
        points['physical_node_uid']=points.partC_rev1_node_uid.astype(np.int64).astype(str)
        for name,column in [('x','x'),('y','y'),('longitude','partC_rev1_lon'),('latitude','partC_rev1_lat')]:
            points[name]=points.physical_node_uid.map(xy[column])
        if not np.isfinite(points[['x','y','longitude','latitude']].to_numpy()).all():raise ValueError('Actual terminal coordinates missing')
        points['plot_crs']='EPSG:5070';points['source_crs']='EPSG:4269';points['xy_units']='metres'
        points.to_csv(dst/'true_portal_and_centroid_points.csv',index=False,float_format='%.17g')
        summaries=[];completed=[]
        for r in selected.itertuples(index=False):
            ctx.guard();progress(dst,'GEOGRAPHY_STATES.json',completed,active=dict(region_id=r.region_id))
            desc,a,pairs,scales=source.source.local(r.region_id)
            roads,ends=vertices(con,a['physical_arc_id'].astype(str).tolist())
            requested=set(a['physical_arc_id'].astype(str))
            returned=set(ends.physical_arc_id.astype(str))
            if requested!=returned or roads.empty or set(roads.physical_arc_id.astype(str))!=requested:
                raise ValueError('Regional road geometry coverage differs from requested physical IDs: '+r.region_id)
            if not np.isfinite(roads[['x','y','longitude','latitude']].to_numpy(float)).all():
                raise ValueError('Nonfinite regional road vertices: '+r.region_id)
            roads['region_id']=r.region_id;roads['figure_role']=r.figure_role
            roads.to_csv(dst/(r.region_id+'_road_vertices.csv'),index=False,float_format='%.17g')
            pt=points.loc[points.region_id.eq(r.region_id)]
            summaries.append(dict(region_id=r.region_id,figure_role=r.figure_role,active_dimension=len(pairs),
                regional_physical_arcs=len(returned),requested_physical_arcs=len(requested),geometry_ID_coverage_exact=True,
                regional_physical_nodes=len(a['global_node_index']),
                true_road_vertices=len(roads),terminal_role_records=len(pt),unique_terminal_physical_points=pt.physical_node_uid.nunique(),
                centroid_role_records=int(pt.terminal_type.isin(['ORIGIN_CENTROID','DESTINATION_CENTROID']).sum()),
                mean_x=float(roads.x.mean()),mean_y=float(roads.y.mean()),
                location_marker_definition='Mean of actual exported road vertices, an illustrative subnetwork marker not an administrative centroid',
                selection_is_preoutcome=True))
            pd.DataFrame(summaries,columns=SUMMARY_COLUMNS).to_csv(dst/'region_geography_summary.csv',index=False,float_format='%.17g')
            completed.append(dict(region_id=r.region_id,status='MEASURED'))
            progress(dst,'GEOGRAPHY_STATES.json',completed)
        source.flush_bindings()
        after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('Source geography SQLite changed')
        pd.DataFrame(summaries).to_csv(dst/'region_geography_summary.csv',index=False,float_format='%.17g')
        result=dict(status='EXECUTED',distinct_regions=len(summaries),selected_docket_sha256=sha(select_path),
            geography='True physical road vertices and fine-member locations; no logical chords or invented boundaries',
            source_weights_mutated=False,selection_by_final_test_results=False)
        write_json(dst/'RESULT.json',result);return result
    finally:
        try:source.close()
        finally:
            if con is not None:con.close()


def run(ctx,args):
    if (ctx.root/'regional_geography').exists():raise FileExistsError('Preserve existing regional geography before explicit recovery')
    selected=selected_regions(ctx)
    plans=selected[['region_id','figure_role']].to_dict('records')
    schemas={'selected_region_metadata.csv':list(selected.columns),'region_geography_summary.csv':SUMMARY_COLUMNS,
        'true_portal_and_centroid_points.csv':POINT_COLUMNS}
    return bounded(ctx,args,'regional_geography',plans,['region_id'],'GEOGRAPHY_STATES.json',schemas,
        lambda context,arguments:_run(context,arguments,selected),reserve=60)
