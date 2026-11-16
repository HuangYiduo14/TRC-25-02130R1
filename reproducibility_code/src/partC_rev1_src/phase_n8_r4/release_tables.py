"""Row-traced settings, scenario tables and data-derived manuscript facts."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
from partC_rev1_src.phase_n8.common import read_json,sha,write_json
from .release_data import short,paired_regional
from .safety import REGISTRIES,ANALYTIC
from .context import sources


def number(value,digits=5):
    if value is None or (isinstance(value,(float,np.floating)) and not np.isfinite(value)):return 'not measured'
    if isinstance(value,(int,np.integer)):return f'{value:,}'
    if isinstance(value,(float,np.floating)):return f'{value:,.{digits}g}'
    return str(value)


def primary_graph_counts(ctx):
    """Reopen accepted small metadata independently of the bounded stability audit."""
    src=sources(ctx);p3=src.p3/'suites/s03/parquet'
    metadata_path=p3/'suite_metadata.json';packed_path=src.p5/'suites/s03/counts.json'
    terminal_path=p3/'consolidated_terminals.parquet'
    coverage_path=ctx.prior/'graphs/s03/IDENTITY_AND_COVERAGE_SELF_OD_V2.json'
    for p in (metadata_path,packed_path,terminal_path):src.bind(p,'ACCEPTED_PRIMARY_GRAPH_STAGE_METADATA')
    ctx.bind(coverage_path,'ACCEPTED_PRIMARY_N8_GRAPH_COVERAGE')
    metadata=read_json(metadata_path);packed=read_json(packed_path);coverage=read_json(coverage_path)
    terms=pd.read_parquet(terminal_path,columns=['terminal_type'])
    if coverage['endpoint_mapping_version']!='EXACT_NATIVE_ROLES_WITH_EMPTY_SELF_OD_V2':
        raise ValueError('Primary graph coverage is not the accepted self-OD version')
    if packed['actual_total_links']!=coverage['packed_links'] or len(terms)!=metadata['terminal_count']:
        raise ValueError('Primary accepted graph-stage counts disagree')
    row=dict(suite='s03',active_regional_coordinates=coverage['regional_links'],N8_runtime_nodes=coverage['nodes'],
        N8_runtime_links=coverage['packed_links']+coverage['regional_links']+coverage['endpoint_attachment_links'],
        N8_OD_wrapper_nodes=coverage['nodes']-packed['retained_nodes'],N8_OD_attachment_links=coverage['endpoint_attachment_links'])
    for key in ('bacn_count','racn_count','singleton_ineligible_racn_control_count','terminal_count',
                'actual_online_node_count','actual_static_link_count','exact_regional_pair_count'):
        row['P3_'+key]=metadata[key]
    for kind in ('ORIGIN_CENTROID','DESTINATION_CENTROID'):
        row['P3_'+kind+'_terminal_copies']=int(terms.terminal_type.eq(kind).sum())
    row['source_bindings']=json.dumps({str(p):sha(p) for p in (metadata_path,packed_path,terminal_path,coverage_path)},sort_keys=True)
    path=ctx.root/'tables/PRIMARY_GRAPH_STAGE_COUNTS.csv'
    pd.DataFrame([row]).to_csv(path,index=False,float_format='%.17g')
    return path,pd.DataFrame([row])


def build(release,scenarios):
    ctx=release.ctx;root=ctx.root;repo=ctx.repo;facts=[]
    def fact(key,value,unit,source,locator,denominator='',cohort=''):
        ctx.bind(source,'PUBLICATION_FACT_SOURCE')
        if isinstance(value,np.generic):value=value.item()
        facts.append(dict(fact_id=key,value=value,unit=unit,source_path=str(source),source_sha256=sha(source),
            source_locator=locator,denominator=denominator,cohort=cohort))
    config=repo/'partC_rev1_configs/partC_rev1_real_preprocess_config.json';c=read_json(config)
    pre=repo.parent/'doe_solver_v3/partC_rev1_outputs/partC_rev1_preprocess_summary.csv';pr=pd.read_csv(pre).iloc[0]
    p4=repo/'partC_doc_icnn/phase_4_work/phase4_dual_accuracy_benchmark_execution_config.json';pc=read_json(p4)
    graph=Path(pc['input_versions']['graph_metadata_json']);gm=read_json(graph)
    agg=Path(pc['input_versions']['accepted_aggregated_od_root'])/'derived_inputs/aggregated_od_metadata.json';am=read_json(agg)
    supply=repo.parent/(repo.name+'_outputs')/'phase_05/pilot_v4_0905/run_001/cache/reachable_od_supply_v2.metadata.json';sm=read_json(supply)
    env=root/'receipts/R4_ENVIRONMENT.json';en=read_json(env)
    cap=repo.parent/'doe_solver_v3/partC_rev1_outputs/partC_rev1_capacity_enrichment_summary.csv';ca=pd.read_csv(cap).iloc[0]
    fact('FAFFile',Path(c['inputs']['od_path']).name,'source filename',config,'inputs.od_path')
    fact('BaseYear',c['od']['baseline_year'],'year',config,'od.baseline_year')
    fact('DemandColumn',c['od']['demand_col'],'field',config,'od.demand_col')
    fact('DemandUnits',c['od']['demand_units'],'units',config,'od.demand_units')
    fact('FactorYear','NOT_ESTABLISHED','status',root/'audits/DATA_PROVENANCE_AND_OBSERVATIONS.md','Consumed freight inputs')
    for key,column,unit in [('RawPhysicalLinks','clean link rows','links'),('CleanNodes','clean node rows','nodes'),
        ('PreprocessArcs','directed arc rows','directed arcs'),('Centroids','centroid count','centroids'),
        ('CountyODRows','county OD rows','county OD rows'),('PreprocessOriginalRows','centroid OD rows','centroid OD rows'),
        ('CountyDemand','total county demand annual ktons','annual kilotons'),('CentroidDemand','total centroid demand annual ktons','annual kilotons')]:
        value=pr[column];value=int(value) if unit not in ('annual kilotons',) else float(value)
        fact(key,value,unit,pre,'row0 / '+column)
    for key,column in [('PhysicalNodes','node_count'),('PhysicalArcs','arc_count'),('ExcludedArcs','excluded_arc_rows')]:fact(key,gm[column],'count',graph,column)
    for key,column,unit in [('OriginalODRows','original_row_count','original rows'),('AggregatedODRows','aggregated_row_count','aggregated rows'),
        ('Origins','origin_count','physical origins'),('OriginalDemand','aggregated_total_demand','annual kilotons')]:fact(key,am[column],unit,agg,column)
    for key,column,unit in [('RepresentedODRows','reachable_row_count','aggregated rows'),('UnreachableODRows','unreachable_row_count','aggregated rows'),
        ('RepresentedDemand','reachable_demand','annual kilotons'),('UnreachableDemand','unreachable_demand','annual kilotons')]:fact(key,sm[column],unit,supply,column)
    capacity_freeze=root/'runs/C02_PHYSICAL_CFW/PRE_RUN_FREEZE.json'
    fact('CapacityMultiplier',read_json(capacity_freeze)['capacity_multiplier'],'prepared-capacity multiplier once',capacity_freeze,'capacity_multiplier')
    for key,col,unit in [('Payload','payload_tons_per_truck','tons/truck'),('PCE','truck_pce','passenger-car equivalents/truck'),
        ('DirectionSplit','direction_split','fraction'),('Kfactor','k_factor','fraction'),('Dfactor','d_factor','fraction'),
        ('PHF','fallback_phf','fraction'),('CapacityFloor','min_remaining_capacity_fraction','fraction'),('TruckAADTDefault','default_truck_ratio','fraction')]:
        fact(key,c['capacity_enrichment'][col],unit,config,'capacity_enrichment.'+col)
    for key,col in [('AADTMatched','matched_buffer_geometry_total'),('AADTNoGood','unassigned_no_good_match'),
        ('AADTDefaults','links_using_default_aadt'),('AADTCloseCompetitors','matched_close_competitor')]:
        fact(key,int(ca[col]),'physical links',cap,'row0 / '+col,str(int(ca.faf_cleaned_links)))
    for key,col,unit in [('LogicalCPUs','cpu_logical_count','cores'),('PhysicalCPUs','cpu_physical_count','cores'),
        ('SystemRAMBytes','memory_total','bytes'),('NumericWorkers','numerical_workers','workers'),('NestedThreads','nested_threads','threads'),
        ('GPUUsed','gpu_used','boolean')]:fact(key,en[col],unit,env,col)
    for col in ('python','numpy','scipy','pandas','torch','platform','processor'):fact('Environment_'+col,en[col],'version or identity',env,col)
    stagepath=root/'stability/resolution_counts.csv';stages=release.read(stagepath)
    observed_primary=stages.loc[stages.suite.eq('s03')]
    if len(observed_primary)>1:raise ValueError('Ambiguous primary graph-stage counts')
    primarypath,stage=primary_graph_counts(ctx)
    if len(observed_primary):
        for column in stage.columns:
            if column in observed_primary and column not in ('suite','source_bindings'):
                if stage.iloc[0][column]!=observed_primary.iloc[0][column]:raise ValueError('Independent primary metadata count mismatch: '+column)
    for key,col,unit in [('PrimaryBnodes','P3_bacn_count','P3 B nodes'),('PrimaryRnodes','P3_racn_count','P3 R nodes including singleton controls'),
        ('PrimaryRcontrols','P3_singleton_ineligible_racn_control_count','P3 singleton controls'),
        ('PrimaryTerminals','P3_terminal_count','region-specific P3 terminal copies'),
        ('PrimaryOriginCopies','P3_ORIGIN_CENTROID_terminal_copies','region-specific origin copies'),
        ('PrimaryDestinationCopies','P3_DESTINATION_CENTROID_terminal_copies','region-specific destination copies'),
        ('PrimaryPthreeNodes','P3_actual_online_node_count','P3 graph nodes'),('PrimaryStaticLinks','P3_actual_static_link_count','P3 static links'),
        ('PrimaryConceptualPairs','P3_exact_regional_pair_count','P3 conceptual regional pairs'),
        ('PrimaryActiveCoordinates','active_regional_coordinates','accepted active regional coordinates'),
        ('PrimaryRuntimeNodes','N8_runtime_nodes','N8 runtime nodes'),('PrimaryRuntimeLinks','N8_runtime_links','N8 runtime links'),
        ('PrimaryWrapperNodes','N8_OD_wrapper_nodes','N8 directional OD wrapper nodes'),
        ('PrimaryAttachmentLinks','N8_OD_attachment_links','N8 directional OD attachment links')]:
        fact(key,int(stage.iloc[0][col]),unit,primarypath,'suite=s03 / '+col,'graph stage explicitly named; independently reopened accepted metadata')
    poppath=root/'physical_population/COMMON_POPULATION_COVERAGE.csv';pop=release.read(poppath)
    for r in pop.itertuples(index=False):
        prefix='Common' if r.population=='COMMON_ALL_FIVE_SUITES' else 'SthreeCommon'
        for suffix,col in [('Arcs','physical_arc_count'),('FlowShare','x45_flow_share'),('FlowCostShare','x45_flow_cost_share')]:
            fact(prefix+suffix,getattr(r,col),'count' if suffix=='Arcs' else 'fraction',poppath,r.population+' / '+col,
                'all accepted physical arcs / corresponding x45 sums',r.population)
    freeze=root/'sampling/DESIGN_FREEZE.json';d=read_json(freeze)
    draws=release.read(root/'sampling/PPS_draws.csv')
    fact('PPSDraws',len(draws),'draw units',root/'sampling/PPS_draws.csv','all rows','represented-demand PPS with replacement')
    fact('PPSUniqueODs',draws.accepted_od_row.nunique(),'distinct aggregated rows',root/'sampling/PPS_draws.csv','nunique accepted_od_row')
    fact('PPSOrigins',draws.origin_idx.nunique(),'physical origins',root/'sampling/PPS_draws.csv','nunique origin_idx')
    replay=root/'validation/ORIGINAL_FAILURE_REPLAY.json';rr=read_json(replay)[0]
    for key,col,unit in [('NegativeQueryDimension','dimensions','active coordinates'),('NegativeCoordinate','native_coordinate','zero-based native index'),
        ('NegativeOriginalCost','original_cost_seconds','seconds'),('NegativeSafeCost','safe_cost_seconds','seconds'),
        ('NegativeOwnFlow','own_coordinate_flow','annual kilotons'),('NegativeRho','rho','fraction'),
        ('NegativeRegionScaleRatio','source_region_max_native_scale_ratio','dimensionless'),('NegativeGlobalScaleRatio','original_portfolio_max_native_scale_ratio','dimensionless')]:
        fact(key,rr[col],unit,replay,'row0 / '+col,rr['region_id'])
    for row in scenarios.itertuples(index=False):
        key=row.scenario.replace('_','')
        for col in ('status','measured_seconds','RG_SPTT','accepted_updates'):
            fact('Scenario'+key+col.replace('_',''),getattr(row,col,None),'seconds' if col=='measured_seconds' else 'fraction' if col=='RG_SPTT' else 'status/count',
                root/'tables/SCENARIO_RESULTS.csv',row.scenario+' / '+col,row.timing_scope)
    warm=[]
    for job in ('B05_ORIGINAL_PRIMARY_WARM','B06_ORIGINAL_BEST_ANY_WARM','B07_ORIGINAL_NONCONSTANT_WARM'):
        rp=root/'runs'/job/'RESULT.json';sp=rp.with_name('SAFETY_STOP.json')
        ctx.bind(rp,'COMPLETED_UNCHANGED_WARM_DIAGNOSTIC');ctx.bind(sp,'RECORDED_WARM_NEGATIVE_COST')
        result=read_json(rp);stop=read_json(sp)
        if result['stop_reason']!='STOP_NEGATIVE_COST' or result['accepted_iterations']!=0 or result['final_certificate_fresh']:
            raise ValueError('Warm narrative no longer matches the actual terminal evidence')
        warm.append((rp,sp,result,stop))
    if len({(r[3]['minimum_cost_seconds'],r[3]['native_coordinate'],r[3]['input_state_hash']) for r in warm})!=1:
        raise ValueError('Warm diagnostics no longer share the same recorded negative query')
    fact('WarmRejectedPortfolios',len(warm),'unchanged portfolios',root/'tables/SCENARIO_RESULTS.csv',
        'ORIGINAL_PRIMARY_WARM / ORIGINAL_BEST_ANY_WARM / ORIGINAL_NONCONSTANT_WARM; STOP_NEGATIVE_COST before updates')
    fact('WarmNegativeCost',warm[0][3]['minimum_cost_seconds'],'seconds',warm[0][1],'minimum_cost_seconds',
        'same recorded minimum and full input state in all three unchanged warm diagnostics')
    fact('WarmNegativeCoordinate',warm[0][3]['native_coordinate'],'zero-based native index',warm[0][1],'native_coordinate')
    fact('WarmHistoricalSeedSeconds',warm[0][2]['historical_analytic_startup_seconds'],'recorded assignment seconds',
        warm[0][0],'historical_analytic_startup_seconds','inherited analytic assignment record; separate from failed warm diagnostic supervised-job times')
    regpath=root/'regional_metrics/RESULT.json';reg=read_json(regpath)
    for key,col in [('ActiveRegions','active_regions'),('E2SourceRegions','evaluated_regions'),('E2MissingRegions','missing_regions'),
                    ('E2MissingCoordinates','missing_active_coordinates')]:fact(key,reg[col],'count',regpath,col,'E2 source-reopened denominator; completed registry predictions separate')
    closurepath=root/'geography/I95_CLOSURE_STATUS.json';closure=read_json(closurepath)
    for key,col in [('ClosureArcs','intended_closed_physical_arcs'),('ClosureHiddenArcs','suppressed_B_closed_arcs'),
        ('ClosureHiddenBnodes','suppressed_B_nodes'),('ClosureMacros','affected_macros')]:fact(key,closure['counts'][col],'count',closurepath,'counts.'+col)
    fact('ClosureStatus',closure['status'],'status',closurepath,'status')
    policy=root/'resolved_config.json'
    fact('NominalGap',ctx.c['nominal_RG_SPTT'],'RG-SPTT fraction',policy,'nominal_RG_SPTT')
    for method in ('CFW','FW','MSA'):
        fact('Cap'+method,ctx.c['experiment_defaults']['physical_controls_caps_seconds'][method],'maximum job seconds',policy,'experiment_defaults.physical_controls_caps_seconds.'+method)
    historical_path=root/'accounting/historical_physical_benchmark_timing.csv';historical=release.read(historical_path)
    for prefix,benchmark in [('HistoricalXFortyFive','physical_mu030_low_rg1e3_CURRENT_x45'),
                             ('HistoricalXTwoHundred','physical_mu030_high_iter0200_CURRENT')]:
        row=historical.loc[historical.benchmark_id.eq(benchmark)]
        if len(row)!=1:raise ValueError('Ambiguous accepted physical timing row')
        for suffix,col,unit in [('IterationWallSeconds','update_wall_seconds','cumulative recorded iteration-wall seconds including nested checkpoint/refresh work'),
                               ('LaterRefreshSeconds','later_refresh_only_seconds','separate zero-update refresh seconds'),
                               ('BroaderCampaignSeconds','broader_campaign_elapsed_seconds','broader campaign elapsed seconds'),
                               ('Workers','worker_count','historical numerical workers')]:
            fact(prefix+suffix,row.iloc[0][col],unit,historical_path,benchmark+' / '+col,
                'Unknown complete standalone assignment and hardware fields are preserved; no matched speedup follows')
    fact('SafeRunCap',ctx.c['experiment_defaults']['safe_variant_cap_seconds'],'maximum job seconds',policy,'experiment_defaults.safe_variant_cap_seconds')
    pathmetrics=root/'path_metrics/PATH_METRICS.csv';pm=release.read(pathmetrics)
    primary=pm.loc[pm.scenario.eq('N8R4_PRIMARY_BOUND_SAFE_V1')&pm.track.eq('SOLVED_X45_REFERENCE')&pm.cohort.eq('FRESH_PPS_DRAW')]
    if len(primary)>1:raise ValueError('Primary PPS manuscript estimate is ambiguous')
    fact('PrimaryPPSStatus','MEASURED' if len(primary) else 'NOT_MEASURED','status',pathmetrics,
        'primary safe / SOLVED_X45_REFERENCE / FRESH_PPS_DRAW; absence is missing evidence')
    for key,col,scale,unit in [('PrimaryPPSWrel','WREL',100.,'percent'),('PrimaryPPSWmeanrel','WMEANREL_1S',100.,'percent'),
        ('PrimaryPPSLow','WREL_bootstrap_draw_95_low',100.,'percent'),('PrimaryPPSHigh','WREL_bootstrap_draw_95_high',100.,'percent'),
        ('PrimaryPPSTopOne','top1_WREL_numerator_share',100.,'percent'),('PrimaryPPSPairs','valid_paired_rows',1.,'draw units')]:
        fact(key,float(primary.iloc[0][col])*scale if len(primary) else None,unit,pathmetrics,'primary safe / SOLVED_X45_REFERENCE / FRESH_PPS_DRAW / '+col,
            'equal PPS draw units; no second demand weighting','fresh probability sample')
    studypath=root/'learning/STUDY_FREEZE.json';study=read_json(studypath)
    fitpath=root/'learning/FIT_RESULTS.json';fits=read_json(fitpath)
    fact('LearningRegions',len(study['selected_region_ids']),'selected regions',studypath,'len selected_region_ids','pilot-frozen deterministic strata')
    fact('LearningSeeds',len(study['seeds']),'seeds per region',studypath,'len seeds')
    fact('LearningUpdates',study['updates_per_fit'],'configured optimizer updates per fit',studypath,'updates_per_fit')
    fact('LearningFractions',','.join(str(f) for f in study['fractions']),'nested TRAIN fractions',studypath,'fractions')
    fact('LearningCompleteFits',sum(r['status']=='COMPLETE' for r in fits),'fits',fitpath,'status=COMPLETE')
    fact('LearningPlannedFits',len(fits),'fits',fitpath,'all planned result records')
    fact('LearningInitializationSelectedFits',sum(r.get('selected_update')==0 for r in fits),'fits',fitpath,
        'selected_update=0; actual optimizer_updates reported separately')
    fact('LearningInitializationSelectedRegions',len({r['region_id'] for r in fits if r.get('selected_update')==0}),
        'distinct regions',fitpath,'distinct region_id where selected_update=0')
    fact('LearningNonfinalSelectedFits',sum(r.get('selected_update')!=r.get('optimizer_updates') for r in fits),'fits',fitpath,
        'selected_update differs from actual optimizer_updates')
    fact('LearningArchitecture',study['source_architecture'],'architecture',studypath,'source_architecture')
    for key,col,unit in [('LearningOptimizer','optimizer_name','optimizer'),('LearningRate','learning_rate','learning rate'),
        ('LearningDtype','dtype','numeric precision'),('LearningClip','gradient_norm_clip','gradient norm'),
        ('LearningBatchLarge','batch_large_dimension','draws per update'),('LearningBatchSmall','batch_small_dimension','draws per update'),
        ('LearningBatchThreshold','batch_dimension_threshold','active coordinates'),('LearningValidPeriod','validation_checks_every_updates','updates'),
        ('LearningWeightDecay','adam_weight_decay','coefficient'),('LearningHuberDelta','huber_delta','scaled residual'),
        ('LearningTotalWeight','total_cost_loss_weight','loss coefficient'),('LearningResidualWeight','congestion_loss_weight','loss coefficient'),
        ('LearningValueWeight','value_interval_loss_weight','loss coefficient'),('LearningFloorWeight','below_c0_penalty_weight','loss coefficient'),
        ('LearningOperatingMass','sampling_operating_probability_when_both','sampling probability'),
        ('LearningStressMass','sampling_stress_probability_when_both','sampling probability')]:
        fact(key,study[col],unit,studypath,col,'frozen study policy; not a fitted empirical effect')
    freshpath=root/'learning/new_label_coverage.csv';fresh=release.read(freshpath)
    fact('FreshLabelPlanned',len(fresh),'inputs',freshpath,'all coverage rows')
    fact('FreshLabelAB',int(fresh.quality.isin(['A','B']).sum()),'certified A/B inputs',freshpath,'quality in A/B')
    for quality,suffix in [('C','C'),('QUARANTINED','Quarantined')]:
        fact('FreshLabel'+suffix,int(fresh.quality.eq(quality).sum()),'excluded inputs',freshpath,'quality='+quality)
    m28fresh=fresh.loc[fresh.region_id.eq('partC_p2_region_0ce745b1cfa26af63aab')]
    for cohort,suffix in [('NEW_OPERATING','Operating'),('NEW_STRESS','Stress')]:
        cohort_rows=m28fresh.loc[m28fresh.cohort.eq(cohort)]
        fact('FreshMtwentyEight'+suffix+'Planned',len(cohort_rows),'inputs',freshpath,'m28 / '+cohort+' / all rows')
        fact('FreshMtwentyEight'+suffix+'AB',int(cohort_rows.quality.isin(['A','B']).sum()),'certified A/B inputs',freshpath,
            'm28 / '+cohort+' / quality in A/B','Per-curve query denominator shared across model families and fractions')
    bdp=root/'boundary_diagnostics/RESULT.json';bd=read_json(bdp)
    for key,col in [('LiftSelectedODs','selected_distinct_ODs'),('LiftContinuityVerified','verified_selected_global_lifts'),
                    ('LiftCompletedODs','completed_selected_ODs'),('PositiveLocalWitnesses','local_positive_pair_witness_rows')]:
        fact(key,bd[col],'count',bdp,col,'planned/actually completed distinct diagnostic ODs / positive active pairs as named')
    fact('BoundaryStatus',bd['status'],'terminal diagnostic status',bdp,'status')
    selected_path=root/'boundary_diagnostics/selected_path_diagnostics.csv'
    if selected_path.is_file():
        selected=release.read(selected_path)
        endpoints=selected.stable_od_key.str.split('|',regex=False,expand=True)
        self_verified=int((endpoints[0].eq(endpoints[1])&selected.selected_global_lift_continuity_pass.eq(True)).sum()) if len(selected) else 0
        fact('LiftVerifiedSelfODs',self_verified,'empty self-OD paths',selected_path,
            'same stable origin/destination and selected_global_lift_continuity_pass=True','Subset of verified selected paths, not nontrivial routes')
    else:fact('LiftVerifiedSelfODs',None,'empty self-OD paths',bdp,'Selected-path observations unavailable')
    old_reports=repo.parent/(repo.name+'_outputs')/'phase_07/p67_postaudit_e2_0908/run_001/reports'
    old_summary=old_reports/'standalone_ICNN_summary.csv'
    ctx.bind(old_summary,'PRESERVED_NATIONAL_TRAIN_E2_HEADLINE','7d5c0d978815933104fa98e8a2e0aeb81add08b9be35efa151c31fa9112c6722')
    old=release.read(old_summary)
    old=old.loc[old.split.eq('TRAIN_e2')&old.model_kind.eq('POWER4_SKIP_ICNN32')&old.aggregation.eq('EQUAL_REGION')]
    if len(old)!=1:raise ValueError('Ambiguous protected TRAIN_e2 headline')
    headline=float(old.MARE_equal_pairs.iloc[0]);nregions=int(old.regions.iloc[0])
    probe=Path(ctx.c['input_versions']['current_R1_native_root'])/'tables/m28_training_probe.csv'
    ctx.bind(probe,'PRESERVED_M28_SEEN_TRAIN_DIAGNOSTIC','24c3316f535dc41d8783468bed923657522d607359e2b25943d598ded79a8eaa')
    m28=release.read(probe);mean=float(m28.MARE_equal_pairs.mean())
    fact('HistoricalNNTraineHeadline',headline*100,'percent',old_summary,'TRAIN_e2 / POWER4_SKIP_ICNN32 / EQUAL_REGION / MARE_equal_pairs',str(nregions)+' equally weighted regions','historical standalone NN seen TRAIN_e2')
    fact('HistoricalMtwentyEightError',mean*100,'percent',probe,'mean MARE_equal_pairs',str(len(m28))+' queries times 28 coordinates','named region seen TRAIN_e2')
    fact('HistoricalMtwentyEightShare',100*mean/(headline*nregions),'percent of regional-error sum',old_summary,'headline regional sum and authenticated m28 query mean',str(nregions)+' region MAREs','outlier contribution, not demand weighting')
    fact('HistoricalLeaveMtwentyEightOut',100*(headline*nregions-mean)/(nregions-1),'percent',old_summary,'leave-one-region-out arithmetic with authenticated m28 query mean',str(nregions-1)+' regions','sensitivity only; headline unchanged')
    current_metric_source=root/'regional_metrics/region_registry_metrics.csv'
    current_regional=paired_regional(release.read(current_metric_source),int(reg['registries']))
    current_grouped=current_regional.groupby(['registry','cohort'],sort=False).agg(
        region_mean_MARE=('MARE_equal_pairs','mean'),negative_coordinates=('negative_coordinates','sum')).reset_index()
    for registry,label in [('E2_PRIMARY_FROZEN','Primary'),('N8R4_PRIMARY_BOUND_SAFE_V1','PrimarySafe')]:
        for cohort,suffix in [('REUSED_NOW_SEEN_E2_AUDIT_OPERATING','Operating'),('REUSED_NOW_SEEN_E2_AUDIT_STRESS','Stress')]:
            observed=current_grouped.loc[current_grouped.registry.eq(registry)&current_grouped.cohort.eq(cohort)]
            if len(observed)>1:raise ValueError('Ambiguous primary regional manuscript metric')
            fact('Regional'+label+suffix+'MARE',float(observed.region_mean_MARE.iloc[0])*100 if len(observed) else None,
                'percent',current_metric_source,registry+' / '+cohort+' / equal-region query-mean MARE',
                'Only region/cohort groups completed by every frozen registry; query MARE then equal-region mean')
            fact('Regional'+label+suffix+'Negative',int(observed.negative_coordinates.iloc[0]) if len(observed) else None,
                'coordinate observations',current_metric_source,registry+' / '+cohort+' / sum negative_coordinates')
    m28path=root/'regional_metrics/m28_current_TRAIN_e2_rows.csv'
    if m28path.is_file():
        m28now=release.read(m28path)
        for registry,label in [('E2_PRIMARY_FROZEN','Primary'),('N8R4_PRIMARY_BOUND_SAFE_V1','PrimarySafe')]:
            observed=m28now.loc[m28now.registry.eq(registry)]
            fact('CurrentMtwentyEight'+label+'MARE',float(observed.MARE_equal_pairs.mean())*100 if len(observed) else None,
                'percent',m28path,registry+' / mean query MARE','Named m28 seen TRAIN_e2 diagnostic, not fresh test')
            fact('CurrentMtwentyEight'+label+'Queries',len(observed),'queries',m28path,registry+' / row count')
    else:
        for label in ('Primary','PrimarySafe'):
            for suffix,unit in [('MARE','percent'),('Queries','queries')]:
                fact('CurrentMtwentyEight'+label+suffix,None,unit,root/'regional_metrics/M28_DISPOSITION.json','Current named diagnostic unavailable')
    factpath=release.root/'numerical_facts.csv';pd.DataFrame(facts).to_csv(factpath,index=False,float_format='%.17g')
    lookup={r['fact_id']:r for r in facts}
    # Table 1 uses compact selected rows; complete provenance settings are supplied.
    folder='table_1';release.folder(folder);release.copy(folder,factpath)
    display=[]
    for label,keys in [('Freight release / base year',['FAFFile','BaseYear']),('County-factor release year',['FactorYear']),
        ('Accepted physical graph',['PhysicalNodes','PhysicalArcs']),('Original / aggregated OD rows',['OriginalODRows','AggregatedODRows']),
        ('s03 P3 B / R nodes / singleton controls',['PrimaryBnodes','PrimaryRnodes','PrimaryRcontrols']),
        ('s03 region-specific terminals / origin / destination copies',['PrimaryTerminals','PrimaryOriginCopies','PrimaryDestinationCopies']),
        ('s03 P3 nodes / static links / conceptual pairs',['PrimaryPthreeNodes','PrimaryStaticLinks','PrimaryConceptualPairs']),
        ('s03 active regional coordinates',['PrimaryActiveCoordinates']),
        ('s03 N8 runtime nodes / links',['PrimaryRuntimeNodes','PrimaryRuntimeLinks']),
        ('s03 N8 OD wrapper nodes / attachment links',['PrimaryWrapperNodes','PrimaryAttachmentLinks']),
        ('Physical origins / extracted centroids',['Origins','Centroids']),('Represented / inherited unreachable ODs',['RepresentedODRows','UnreachableODRows']),
        ('Original / represented annual kilotons',['OriginalDemand','RepresentedDemand']),('Inherited unreachable annual kilotons',['UnreachableDemand']),
        ('Prepared-capacity multiplier',['CapacityMultiplier']),('Payload / truck PCE',['Payload','PCE']),
        ('Direction / K / D / PHF assumptions',['DirectionSplit','Kfactor','Dfactor','PHF']),('Capacity floor / missing-truck AADT fraction',['CapacityFloor','TruckAADTDefault']),
        ('Geometric AADT matches / defaults',['AADTMatched','AADTDefaults']),('No-good matches / close competitors',['AADTNoGood','AADTCloseCompetitors']),
        ('Logical / physical CPUs',['LogicalCPUs','PhysicalCPUs']),('Numerical workers / nested threads',['NumericWorkers','NestedThreads']),('GPU used',['GPUUsed'])]:
        display.append(dict(Setting=label,Value=' / '.join(str(int(float(lookup[k]['value']))) if lookup[k]['unit']=='year' else number(lookup[k]['value']) for k in keys),
            Scope='; '.join(dict.fromkeys(lookup[k]['unit'] for k in keys))))
    release.csv(folder,'table.csv',pd.DataFrame(display),'Display values generated only from numerical_facts.csv',[factpath])
    release.copy(folder,stagepath,description='All-suite companion graph-stage counts: P3 conceptual representation, P5 packing and accepted N8 active runtime remain separate')
    release.copy(folder,primarypath,description='Primary accepted graph-stage metadata reopened independently; not inferred from a partial stability population')
    release.notes[folder]='Native source metadata distinguish preprocessing, accepted physical assignment and reduced-network stages. Annual tonnage is not AADT. County-factor release year remains unestablished. The current multiplier acts once on already prepared capacities. Background traffic enters capacity preparation; calibration AADT is not independent validation. Complete row-level fact/source bindings are in numerical_facts.csv.'
    # Table 2 keeps every successful, time-limited and rejected scenario.
    folder='table_2';release.folder(folder);release.csv(folder,'scenario_results.csv',scenarios,'Every current scenario and original cold failure with status and timing scope',[root/'tables/SCENARIO_RESULTS.csv'])
    metrics=release.read(root/'physical_population/PHYSICAL_FLOW_METRICS.csv')
    common=metrics.loc[metrics.population.eq('COMMON_ALL_FIVE_SUITES')].set_index('scenario')
    display=[]
    for row in scenarios.itertuples(index=False):
        error=common.loc[row.scenario,'flow_WMAPE'] if row.scenario in common.index else None
        scope=('failed attempt job' if row.timing_scope=='R4_FAILED_ATTEMPT_SUPERVISED_JOB_WALL' else
            'no measured attempt' if row.timing_scope=='NO_MEASURED_ATTEMPT_WALL' else
            'conditional warm job' if row.scenario.endswith('_WARM') else 'complete job' if row.timing_scope=='R4_COMPLETE_SUPERVISED_JOB_WALL' else
            'rejected startup' if row.timing_scope=='ORIGINAL_REJECTED_ADMISSION_NOT_SOLVE_TIME' else 'recorded assignment')
        display.append(dict(Scenario=row.label,Status=row.status.replace('CONVERGED_REPRESENTED_DEMAND_ONLY','converged represented').replace('CONVERGED','converged').replace('_',' '),
            **{'Time scope':scope},
            Seconds=number(row.measured_seconds),RG=number(row.RG_SPTT),Updates=number(row.accepted_updates),
            **{'Flow WMAPE (%)':number(error*100 if error is not None else None)}))
    release.csv(folder,'table.csv',pd.DataFrame(display),'Display percentages are 100 times the supplied fraction; missing values are explicit',
        [root/'tables/SCENARIO_RESULTS.csv',root/'physical_population/PHYSICAL_FLOW_METRICS.csv'])
    release.copy(folder,root/'physical_population/PHYSICAL_FLOW_METRICS.csv');release.copy(folder,poppath)
    release.copy(folder,historical_path,description='Accepted historical physical timer scopes, including unknown complete standalone clock and hardware identity')
    release.notes[folder]='RG is a fresh same-state signed gap ratio. A valid final state may remain time-limited. Historical unchanged results and new complete supervised-job timers have different scopes in scenario_results.csv. Job timing starts before code snapshotting and child launch and includes those costs. Original rejected admission time is not an assignment runtime. Physical WMAPE uses the identical common recoverable directed-arc population. Unchanged warm diagnostics and fixed-safe variants are separate mathematical/scenario identities. Historical physical cumulative iteration-wall timers include nested checkpoint/refresh work; the later x45 zero-update refresh is separate. No historical matched-hardware speedup is claimed.'
    # Table 3: equal-region means, with source and coordinate denominators explicit.
    folder='table_3';release.folder(folder);p=root/'regional_metrics/region_registry_metrics.csv';v=release.read(p)
    release.csv(folder,'learning_study_settings.csv',pd.DataFrame([dict(setting=k,value=json.dumps(val),
        source_path=str(studypath),source_sha256=sha(studypath),source_locator=k) for k,val in study.items()]),
        'Complete pilot-frozen study policy, including optimizer, sampling and actual selected docket',[studypath])
    release.copy(folder,p,name='available_region_registry_metrics.csv',description='All completed registry-specific predictions, including any incompletely matched cohort/region')
    registry_count=int(reg['registries'])
    v=paired_regional(v,registry_count)
    release.csv(folder,'matched_region_registry_metrics.csv',v,'Only region/cohort populations completed by every frozen registry enter paired portfolio summaries',[p,regpath])
    grouped=v.groupby(['registry','cohort'],sort=False).agg(region_mean_MARE=('MARE_equal_pairs','mean'),regions=('region_id','nunique'),
        source_queries=('rows','sum'),coordinate_observations=('coordinate_observations','sum'),negative_coordinates=('negative_coordinates','sum'),
        below_fftt_coordinates=('below_fftt_coordinates','sum')).reset_index()
    release.csv(folder,'registry_cohort_metrics.csv',grouped,'Equal-region mean of query MARE; sums show full denominators',[p])
    byfamily=v.groupby(['registry','cohort','model_kind'],sort=False).agg(region_mean_MARE=('MARE_equal_pairs','mean'),
        regions=('region_id','nunique'),source_queries=('rows','sum'),coordinate_observations=('coordinate_observations','sum'),
        negative_coordinates=('negative_coordinates','sum')).reset_index()
    release.csv(folder,'registry_family_cohort_metrics.csv',byfamily,'Source model-family composition and within-family errors/denominators',[p])
    display=[]
    registry_ids=[item for safe,original in REGISTRIES.items() for item in (original,safe)]+[ANALYTIC,'FFTT']
    if len(registry_ids)!=registry_count:raise ValueError('Regional table registry population changed')
    for registry in registry_ids:
        q=grouped.loc[grouped.registry.eq(registry)].set_index('cohort')
        def metric(cohort):return number(float(q.loc['REUSED_NOW_SEEN_'+cohort,'region_mean_MARE'])*100) if 'REUSED_NOW_SEEN_'+cohort in q.index else 'not measured'
        e=q.loc[q.index.str.contains('E2_AUDIT')]
        display.append({'Registry':short(registry),'E2 op. MARE (%)':metric('E2_AUDIT_OPERATING'),'E2 stress MARE (%)':metric('E2_AUDIT_STRESS'),
            'Old LOW MARE (%)':metric('LOW_TEST'),
            'E2 negative obs.':number(int(e.negative_coordinates.sum())) if len(e) else 'not measured',
            'E2 coordinate obs.':number(int(e.coordinate_observations.sum())) if len(e) else 'not measured'})
    release.csv(folder,'table.csv',pd.DataFrame(display),'Per-registry display summary, no silent model exclusions',[p])
    release.copy(folder,root/'regional_metrics/coverage.csv');release.copy(folder,root/'regional_metrics/model_prediction_coverage.csv')
    release.copy(folder,root/'tables/safety_registry_summary.csv')
    release.notes[folder]='MARE uses max(abs(reference), 1 second). Displayed MARE is the equal-region mean of each region\'s query mean; it is not a pooled coordinate or LOW-weighted mean. Corresponding LOW-weighted and other metrics remain in per-region observations. E2 coverage stays distinct from protected old-test coverage. All old-test and E2 rows are reused now-seen diagnostics; the selected learning study has separately labeled postfreeze tests. Negative and below-freeflow coordinate counts use the supplied observation denominators.'
    # Table 4: resolution sensitivity on common physical IDs.
    folder='table_4';release.folder(folder);p=root/'stability/resolution_counts.csv';v=release.read(p)
    aripath=root/'stability/partition_stability.csv';ari=release.read(aripath);ari=ari.loc[ari.population.eq('COMMON_FINAL_REGION_OWNED_PHYSICAL_ARCS')]
    display=[]
    if v.suite.duplicated().any():raise ValueError('Duplicated resolution count row')
    for sid in ctx.c['active_suites']:
        observed=v.loc[v.suite.eq(sid)];row=observed.iloc[0] if len(observed) else {}
        pair=ari.loc[(ari.suite_a.eq(sid)&ari.suite_b.eq('s03'))|(ari.suite_b.eq(sid)&ari.suite_a.eq('s03'))]
        if len(pair)>1:raise ValueError('Ambiguous s03 stability comparator')
        error=common.loc['FFTT_'+sid,'flow_WMAPE'] if 'FFTT_'+sid in common.index else None
        display.append({'Suite':sid,'Resolution':number(row.get('resolution')),'Final regions':number(row.get('final_region_count')),
            'Active coords.':number(row.get('active_regional_coordinates')),'Flow WMAPE (%)':number(error*100 if error is not None else None),
            'ARI vs s03':number(pair.ARI.iloc[0] if len(pair) else None)})
    release.csv(folder,'table.csv',pd.DataFrame(display),'Five accepted FFTT suite outcomes with common-population error and final ownership ARI',[p,aripath,root/'physical_population/PHYSICAL_FLOW_METRICS.csv'])
    for path in [p,aripath,root/'stability/boundary_Jaccard.csv',poppath]:release.copy(folder,path)
    release.csv(folder,'stability_coverage.csv',pd.DataFrame(read_json(root/'stability/STABILITY_STATES.json')),
        'Every planned suite/statistic/map disposition; missing ARI is not zero',[root/'stability/STABILITY_STATES.json',root/'stability/RESULT.json'])
    release.notes[folder]='All suites use constant-regional FFTT policies. Final counts follow small-region correction. Backbone changes across resolution; ARI uses only the intersection of final region-owned physical arcs. This table does not compare trained surrogates across resolutions.'
    folder='table_5';release.folder(folder);p=root/'accounting/offline_stage_subtotals.csv';v=release.read(p)
    display=[dict(Stage=r.stage.replace('_',' '),Hours=number(r.seconds/3600.),Scope=r.measure.replace('_',' ')) for r in v.itertuples(index=False)]
    display.extend([dict(Stage='Complete offline elapsed cost',Hours='not established',Scope='Missing historical scope; not the sum of worker and elapsed timers'),
        dict(Stage='Complete CPU / GPU hours',Hours='not established',Scope='No fabricated nominal-core multiplier'),
        dict(Stage='Full break-even query count',Hours='not established',Scope='Unknown comparable offline intercepts')])
    release.csv(folder,'table.csv',pd.DataFrame(display),'Measured disjoint subtotals by scope, with unknown complete costs explicit',[p,root/'accounting/RESULT.json'])
    for path in [p,root/'accounting/offline_missing_scope.csv',root/'accounting/offline_disjoint_components.csv',historical_path]:
        release.copy(folder,path)
    release.notes[folder]='Hours are seconds divided by 3600 within each named measurement scope. Worker wall is not CPU time or critical-path elapsed time. Historical resumed records are deduplicated, ambiguous counters excluded and inherited training subtracted exactly once. Complete offline intercepts and full break-even remain unknown.'
    code=Path(__file__).with_name('table_standalone.py')
    titles=[('settings','Setting and provenance'),('scenarios','All scenario outcomes and declared timing scopes'),
        ('regional','Current regional portfolios; equal-region MARE summaries'),('resolution','Accepted FFTT resolution sensitivity'),
        ('offline','Partial offline cost components by measurement scope')]
    for i in range(1,6):
        name='table_'+str(i);shutil.copy2(code,release.folder(name)/'build_table.py')
        label,title=titles[i-1];release.csv(name,'table_metadata.csv',pd.DataFrame([dict(caption=title,label='tab:'+label)]),'Table caption and reference label')
        release.document(name,'python build_table.py')
    # Every displayed table row is a reviewable CSV claim.
    claims=[]
    for f in facts:
        claims.append(dict(claim_id=f['fact_id'],claim_scope='MANUSCRIPT_FACT',csv='numerical_release/numerical_facts.csv',
            row_selector='fact_id='+f['fact_id'],value=f['value'],unit=f['unit'],denominator=f['denominator'],cohort=f['cohort'],
            native_source=f['source_path'],native_source_sha256=f['source_sha256'],source_locator=f['source_locator']))
    for i in range(1,6):
        p=release.root/('table_'+str(i))/'table.csv';data=release.read(p)
        for j,row in data.iterrows():
            claims.append(dict(claim_id=f'Table{i}Row{j+1}',claim_scope='TABLE_DISPLAY_ROW',csv='numerical_release/'+p.relative_to(release.root).as_posix(),
                row_selector='data_row='+str(j+1),value=json.dumps(row.to_dict()),unit='see table dictionary and source CSV',
                source_locator='Generated table row; exact numeric source in accompanying table data'))
    pd.DataFrame(claims).to_csv(root/'claim_to_result.csv',index=False,float_format='%.17g')
    return pd.DataFrame(facts)
