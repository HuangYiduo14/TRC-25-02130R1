"""Measured table rows and timing with explicit, non-additive clock boundaries."""
from pathlib import Path
import math
import numpy as np
import pandas as pd
from .metrics import LABELS,final_new_case,job_timing
from .publication_data import public_role,FAMILIES


def fmt(value,digits=3):
    if value is None or pd.isna(value):return 'Unavailable'
    return format(float(value),'.'+str(digits)+'f')


def table(release,name,title,data,widths,note):
    release.spec(name,title,note,widths)
    release.csv(name,'data',data,'Exact display rows; units and missing-value dispositions in headers and notes')


def timing(release,summary):
    c=release.ctx;parts=[]
    for p in sorted((c.root/'jobs').glob('*/EXIT.json')):
        r=release.record(p)
        if 'elapsed_seconds' not in r:continue
        parts.append(dict(component_id=p.parent.name,component_label=r['task'].replace('_',' ').title(),seconds=r['elapsed_seconds'],
            measurement_scope='COMPLETE_SUPERVISED_PROCESS_WALL',display_in_composition=False,method_id=r.get('explicit_task',{}).get('scenario',''),
            nesting='Disjoint serial job interval; source loading, solver, export nested inside',exit_code=r.get('exit_code')))
    m1_path=final_new_case(c,'M1_OPEN');m1=release.record(m1_path/'RESULT.json')
    m1_complete=release.record(c.root/'jobs'/m1_path.name/'EXIT.json')['elapsed_seconds']
    discarded=float(m1['discarded_epoch_seconds'])
    if discarded:raise ValueError('Final fixed-portfolio timing must be independently initialized; execute the declared final-registry open rerun')
    final_online=m1_complete
    # These discarded epochs are a directly measured attributable setup cost. Prior
    # training and shared preflight are explicitly not treated as fully measured C.
    discovery=[]
    for p in (c.root/'runs').glob('*/RESULT.json'):
        r=release.record(p)
        if r.get('scenario')=='M1_OPEN':discovery.append(float(r.get('discarded_epoch_seconds',0)))
    cost_partial=sum(discovery)
    for component,label,seconds in [('M1_DISCARDED','Safety discovery epochs',cost_partial),('M1_FINAL','Final fixed-portfolio assignment',final_online)]:
        parts.append(dict(component_id=component,component_label=label,seconds=seconds,
            measurement_scope='DISCARDED_EPOCH_SUM_OR_FINAL_SEPARATE_JOB',display_in_composition=True,method_id='M1',
            nesting='Already inside parent receipts; do not add again to parent',exit_code=0))
    components=pd.DataFrame(parts)
    offline=release.read(c.r4/'accounting/offline_stage_subtotals.csv')
    offline['stage_public']=offline.stage.str.replace('E2','regional audit',regex=False).str.replace('P67','extended regional',regex=False).str.replace('P6','regional',regex=False).str.replace('P7_ORIGINAL','initial model',regex=False)
    offline=offline.rename(columns={'stage':'source_stage'});offline['source_stage']=offline.stage_public
    physical=summary.loc[summary.suite_id.eq('s03')&summary.method_id.eq('PHYSICAL_CFW')&summary.population.eq('BACKBONE')].iloc[0]
    tp=float(physical.online_seconds);queries=[];crossings=[]
    for method in ('M1','M2','M3'):
        row=summary.loc[summary.suite_id.eq('s03')&summary.method_id.eq(method)&summary.population.eq('BACKBONE')].iloc[0]
        th=final_online if method=='M1' else float(row.online_seconds);intercept=cost_partial if method=='M1' else None
        qstar=None
        crossings.append(dict(method_id=method,method_label=LABELS[method],physical_seconds=tp,online_seconds=th,
            recorded_attributable_setup_seconds=intercept,break_even_queries=qstar,
            status='NOT_COMPARABLE_COLD_START_AMORTIZATION',
            initialization_scope='SHARED_PREFLIGHT_FEASIBLE_AON_OUTSIDE_JOB' if method in ('M1','M2') else 'FRESH_FEASIBLE_INITIALIZATION_INSIDE_JOB',
            full_offline_cost_measured=False,quality_comparability='DIFFERENT_ACHIEVED_GAPS; PHYSICAL_CONTROL_TIME_LIMITED'))
        queries.append(dict(method_id=method,method_label=LABELS[method],queries=None,cumulative_seconds=None,
            recorded_setup_seconds=intercept,observed_job_seconds=th,status='NOT_COMPARABLE_INITIALIZATION_AND_OFFLINE_COST_SCOPE'))
    release.spec('figure8','Recorded computation components and timing completeness',
        'Same recorded host and one worker; different initialization boundaries. Final-portfolio open time is a separately executed assignment from the common preflight feasible state. Discovery epochs are shown separately. Prior training and shared infrastructure costs are incompletely attributable. Cold-start amortization and a defensible repeated-query crossing are unavailable; no cumulative-cost curve is fabricated.')
    for stem,f,desc in [('components',components,'Disjoint supervisor jobs and explicitly nested partition of the mixed-method open job'),
            ('query_costs',pd.DataFrame(queries),'Explicit unavailable cumulative-query comparison with observed component times'),('offline_scopes',offline,'Historical elapsed and worker-wall subtotals kept separate'),
            ('crossings',pd.DataFrame(crossings),'Not-comparable amortization dispositions and exact observed boundaries')]:release.csv('figure8',stem,f,desc)
    # Large original component inventory stays outside the plotter, but the full
    # underlying rows are included for independent subtotal reconstruction.
    source=c.r4/'accounting/offline_disjoint_components.csv';allrows=release.read(source)
    pathcols=[col for col in allrows if 'path' in col or col.endswith('_root')]
    allrows=allrows.drop(columns=pathcols,errors='ignore')
    for col in allrows.select_dtypes(include=['object','string']).columns:
        allrows[col]=allrows[col].astype(str).str.replace('E2','regional audit',regex=False).str.replace('N8','assignment',regex=False).str.replace('ORIGINAL','initial',regex=False)
    release.csv('figure8','offline_component_observations',allrows,'Disjoint historical component observations behind subtotals; scopes must not be summed together')
    return components,pd.DataFrame(crossings),offline


def build(release,summary,suites,costs,fits,coverage,fallback):
    c=release.ctx;components,crossings,offline=timing(release,summary)
    original=c.repo.parent/'doe_solver_v3'
    origin=release.read(original/'faf_assignment_result/All_Experimental_Disaggregation_Factors/truck_origin_factors.csv')
    destination=release.read(original/'faf_assignment_result/All_Experimental_Disaggregation_Factors/truck_destination_factors.csv')
    centroids=release.read(original/'partC_rev1_outputs/partC_rev1_county_centroids_report.csv')
    county=lambda s:s.dropna().astype(str).str.replace(r'\.0$','',regex=True).str.zfill(5)
    a=set(county(origin.dms_orig_cnty));b=set(county(destination.dms_dest_cnty));centroid_county=county(centroids.county_fips)
    values=dict(origin_factor_rows=len(origin),destination_factor_rows=len(destination),factor_counties=len(a),
        origin_FAF_regions=origin.dms_orig.nunique(),destination_FAF_regions=destination.dms_dest.nunique(),
        extracted_centroid_records=len(centroids),extracted_centroid_counties=centroid_county.nunique(),
        factor_counties_with_centroids=len(a&set(centroid_county)),factor_counties_without_centroids=len(a-set(centroid_county)),
        centroid_counties_outside_factor_universe=len(set(centroid_county)-a),matched_centroid_records=int(centroid_county.isin(a).sum()))
    expected=dict(origin_factor_rows=15720,destination_factor_rows=15720,factor_counties=3144,origin_FAF_regions=132,destination_FAF_regions=132,
        extracted_centroid_records=3497,extracted_centroid_counties=3191,factor_counties_with_centroids=3133,factor_counties_without_centroids=11,
        centroid_counties_outside_factor_universe=58,matched_centroid_records=3225)
    if a!=b or values!=expected:raise ValueError('Study-setting lineage differs from reviewed local source facts')
    prep=release.read(original/'partC_rev1_outputs/partC_rev1_real_preprocess_summary.csv')
    for key,value in [('unmatched_od_county_rows',185778),('raw_link_rows',487394),('clean_node_rows',348495),('directed_arc_rows',665303)]:
        if int(prep.iloc[0][key])!=value:raise ValueError('Native preprocessing lineage differs: '+key)
        values[key]=value
    config=release.record(c.repo/'partC_rev1_configs/partC_rev1_real_preprocess_config.json')
    if (Path(config['inputs']['od_path']).name,config['od']['baseline_year'],config['od']['demand_col'],config['od']['demand_units'])!=('FAF5.6.1.csv',2017,'tons_2017','annual_ktons'):
        raise ValueError('Freight release/year/unit source differs')
    settings=[
        ('Freight input','FAF5.6.1; baseline 2017; tons_2017'),('Demand unit','Annual kilotons; synthetic annual-cost assignment regime'),
        ('County-to-centroid lineage','3,144 factor counties; 3,133 matched; 3,225 solver origins'),('Graph centroid flags','3,497 nodes across 3,191 county-code values'),
        ('BTS experimental county factors','15,720 origin and 15,720 destination rows; 132 FAF region codes; 3,144 county codes; local factor release year unestablished'),
        ('Unmatched county scope','11 factor counties without extracted centroids; 185,778 upstream county OD rows unmatched'),
        ('Raw / prepared physical network','487,394 physical links; 348,495 clean nodes; 665,303 prepared directed arcs'),
        ('Accepted assignment graph','346,373 nodes; 661,644 directed arcs; 3,659 prepared arcs excluded'),
        ('OD populations','41,637,922 original centroid OD rows; 10,193,770 aggregated rows'),
        ('Open represented OD','10,190,380 aggregated rows; 3,390 inherited unreachable rows'),
        ('Demand totals','12,667,037.704539 total; 12,654,353.578362 represented annual kilotons'),
        ('Current capacity','0.30 times authenticated prepared capacity, applied once'),
        ('Capacity preparation','18 tons/truck; PCE 2; directional split 0.5; K 0.10; D 0.55; PHF 0.90; 365 days'),
        ('Background assumptions','Subtract nontruck/truck background; missing truck AADT=12%; remaining vehicle-capacity floor 5%'),
        ('Main abstract network','79,174 runtime nodes; 230,383 links; 87,582 backbone links; 80,316 active regional pairs'),
        ('Active regional scope','1,566 active / 281 inactive regions; no inactive promotion or support expansion'),
        ('Regional potentials','Frozen VALID-selected nonconstant; full current relative-NNLS power-4; constant FFTT'),
        ('Stopping rule','Fresh signed relative gap <=0.001 or stated wall-time cap; TIME_LIMIT retained'),
        ('Learning protocol','16 preselected regions; fractions 10/25/50/70/100%; seed 0; 4,000 updates; fixed VALID selection'),
        ('Hardware / concurrency','16 physical / 24 logical CPU cores; 34,053,951,488 RAM bytes; one numerical worker and one nested thread'),
        ('Software','Windows; Python 3.12.3; NumPy 1.26.4; SciPy 1.13.1; PyTorch 2.4.1; pandas 2.2.3'),
        ('Unestablished metadata','County-factor release year and matched independent traffic observations unavailable')]
    table(release,'table1','Study setting and reproducibility',pd.DataFrame(settings,columns=['Setting','Verified value']),'1|3',
        'County codes, extracted centroid nodes and solver origins are distinct populations. Historical capacity preparation factors are not applied again. Freight/year and factor-source release identity are distinguished.')
    release.csv('table1','source_lineage',pd.DataFrame([dict(population=k,count=v) for k,v in values.items()]),
        'Recomputed counts from authenticated county factors and centroid rows; no equation treats counties as solver origins')
    for p in [c.repo/'partC_rev1_configs/partC_rev1_real_preprocess_config.json',c.r4/'boundary_diagnostics/CENTROID_SOURCE.json',c.prior/'receipts/ENVIRONMENT.json']:
        if p.exists():release.record(p)
    main=summary.loc[summary.suite_id.eq('s03')];rows=[]
    portfolio_note=' Final safety-fallback portfolio: '+', '.join(format(n,',')+' '+family+' regions' for family,n in fallback.effective_family.value_counts().items())+'; '+str(int(fallback.effective_kind.str.contains('ICNN').sum()))+' neural regions.'
    for method in ('REFERENCE_X45','PHYSICAL_CFW','PHYSICAL_FW','PHYSICAL_MSA','M1','M2','M3'):
        b=main.loc[main.method_id.eq(method)&main.population.eq('BACKBONE')].iloc[0];k=main.loc[main.method_id.eq(method)&main.population.eq('KEY_LINKS')].iloc[0]
        rows.append({'Method':LABELS[method],'B WMAPE (%)':fmt(b.wmape_percent),'200-key WMAPE (%)':fmt(k.wmape_percent),
            'B cost error (%)':fmt(b.backbone_cost_error_percent),'Time (s)':fmt(b.online_seconds,1),'Fresh gap':format(b.RG_SPTT,'.5g'),
            'State':str(b.status).replace('_',' ').title()})
    table(release,'table2','Main-suite assignment comparison',pd.DataFrame(rows),'3|1.1|1.1|1.1|1|1|1.8',
        'All errors compare native abstract coordinates with the forward physical reference. Reference self-error is calculated and zero. Independent physical controls are distinct time-limited states. B denotes all 87,582 backbone coordinates; 200-key set is fixed before predictions. Time is complete supervised process wall; historical reference complete wall time is unavailable. New abstract runs reuse a feasible initialization prepared in the separately timed shared preflight; physical controls include fresh initialization. These are not matched cold-start timings. Represented open demand is 12,654,353.578362 annual kilotons for every method.'+portfolio_note)
    release.csv('table2','full_metrics',main,'Complete numeric numerators, denominators, state hashes, statuses and timing scopes')
    regional=release.read(c.root/'regional_metrics/region_registry_metrics.csv')
    regional['model_label']=regional.registry.map(lambda x:FAMILIES.get(x,LABELS.get(x,x)))
    regional['role']=regional.cohort.map(public_role);rows=[]
    for (model,role),g in regional.groupby(['model_label','role'],sort=False):
        measured=g.loc[g.cost_wmape_percent.notna()]
        rows.append({'Regional potential':model,'Query role':role,'Regions':len(measured),'Mean WMAPE (%)':fmt(measured.cost_wmape_percent.mean()),
            'Median (%)':fmt(measured.cost_wmape_percent.median()),'P90 (%)':fmt(measured.cost_wmape_percent.quantile(.9)),'Admitted queries':int(measured.rows.sum())})
    if not rows:rows=[{'Regional potential':'Unavailable','Query role':'No admitted completed evaluation','Regions':0,
        'Mean WMAPE (%)':'Unavailable','Median (%)':'Unavailable','P90 (%)':'Unavailable','Admitted queries':0}]
    table(release,'table3','Regional native marginal-cost errors',pd.DataFrame(rows),'2|2|.8|1.2|1|1|1',
        'Each regional WMAPE is a ratio of summed absolute native cost errors to summed true costs, then summarized equally across regions. MARE is separately retained. These reopened audit cohorts are now seen, not fresh blind tests. Missing m=28 and m=39 regions apply to the operating/stress audits and remain in coverage; low-flow and high projected test cohorts have separate denominators. The standalone ICNN registry contains 1,566 frozen models; each row states its measured regional coverage.')
    release.csv('table3','region_observations',regional,'Every regional cost numerator/denominator underlying displayed mean/median/p90')
    release.csv('table3','coverage',release.read(c.root/'regional_metrics/coverage.csv'),'Exhaustive region/cohort source and A/B coverage')
    rows=[]
    for r in suites.sort_values('resolution').itertuples():rows.append({'Resolution':format(r.resolution,'.17g'),'Active regions':r.active_regions,'Active pairs':r.active_regional_coordinates,
        'Backbone links':r.backbone_count,'Time (s)':fmt(r.online_seconds,1),'B WMAPE (%)':fmt(r.backbone_wmape),'Key WMAPE (%)':fmt(r.key_wmape),'B cost error (%)':fmt(r.backbone_cost_error),'Gap':format(r.RG_SPTT,'.5g')})
    table(release,'table4','Five-resolution constant-region comparison',pd.DataFrame(rows),'1.6|1|1|1|1|1|1|1|1',
        'Each suite uses its own full backbone and frozen dispersed 200 links. All rows use constant FFTT regional costs. Exact resolution is retained in the local source CSV.')
    release.csv('table4','suite_metrics',suites,'Complete suite-specific graph counts, denominator totals, statuses and metrics')
    closed=release.read(c.root/'metrics/closed_summary_metrics.csv');response=release.read(c.root/'metrics/closure_redistribution.csv')
    registry_by_method=closed.loc[closed.population.eq('BACKBONE')].set_index('method_id').registry_identity
    response['registry_identity']=response.registry_identity.fillna(response.method_id.map(registry_by_method))
    rows=[]
    for method in ('M1','M2','M3'):
        b=closed.loc[closed.method_id.eq(method)&closed.population.eq('BACKBONE')].iloc[0]
        k=closed.loc[closed.method_id.eq(method)&closed.population.eq('KEY_LINKS')].iloc[0]
        d=response.loc[response.method_id.eq(method)&response.population.eq('ALL_BACKBONE')].iloc[0]
        dd=response.loc[response.method_id.eq(method)&response.population.eq('BACKBONE_EXCLUDING_CLOSED')].iloc[0]
        rows.append({'Method':LABELS[method],'B redistribution (%)':fmt(d.redistribution_percent),'Outside mask (%)':fmt(dd.redistribution_percent),
            'Closed B WMAPE (%)':fmt(b.wmape_percent),'Closed key WMAPE (%)':fmt(k.wmape_percent),'Gap':format(b.RG_SPTT,'.5g'),'State':b.status.replace('_',' ').title()})
    reach=release.record(c.root/'closure/CLOSED_REACHABILITY.json')
    paired=release.record(c.root/'metrics/CLOSED_OD_COMPARABILITY.json')
    paired_rows={r['population']:r for r in paired['rows']}
    aa=paired_rows['ABSTRACT_SERVED_TOTAL'];ppop=paired_rows['PHYSICAL_SERVED_TOTAL']
    control=closed.loc[closed.method_id.eq('REFERENCE_CLOSED')&closed.population.eq('BACKBONE')].iloc[0]
    note=('1,105 matched represented physical arcs disable 869 abstract backbone coordinates; 608 intended suppressed-interior arcs omitted. '
        'Open-to-closed redistribution uses open-flow denominators and includes lost demand. Closed-reference WMAPE uses the independently assigned matched-mask physical closure. '
        'Matched physical control: '+control.status.replace('_',' ')+'; fresh gap '+format(control.RG_SPTT,'.6g')+'; complete time '+fmt(control.online_seconds,1)+' s. '
        'Additional abstract unreachable demand: '+fmt(reach['additional_unreachable_demand'],6)+' annual kilotons across '+str(reach['additional_unreachable_od_count'])+' aggregated rows. '
        'Abstract served: '+str(aa['aggregated_rows'])+' aggregated rows / '+fmt(aa['demand_annual_ktons'],6)+' annual kilotons; physical served: '+str(ppop['aggregated_rows'])+' rows / '+fmt(ppop['demand_annual_ktons'],6)+'. '+
        ('Served OD membership matches. ' if paired['same_served_OD_membership'] else 'Served OD membership differs: this full-backbone discrepancy includes differing served demand and is not common-served-OD accuracy. ')+
        'No renormalization. Same final fallback registry for both abstract states.')
    table(release,'table5','Represented I-95 response and closed-reference comparison',pd.DataFrame(rows),'3|1.1|1.1|1.1|1.1|1|1.7',note+portfolio_note)
    release.csv('table5','accuracy_metrics',closed,'All matched closed-reference numeric metrics and approximate-reference status')
    release.csv('table5','redistribution',response,'All open-to-closed changes; distinct from reference accuracy')
    scalar={k:v for k,v in reach.items() if isinstance(v,(int,float,bool,str))};release.csv('table5','abstract_coverage',pd.DataFrame([scalar]),'Original/served and additional unreachable OD counts and demand')
    pp=final_new_case(c,'P_CLOSED');release.csv('table5','physical_coverage',release.read(pp/'OD_COVERAGE.csv'),'Matched physical original and closed-reachable OD ledger')
    release.csv('table5','paired_OD_membership',pd.DataFrame(paired['rows']),'Exact overlap, unique served sets and totals on the unchanged original aggregated OD order')
    rows=[]
    for r in crossings.itertuples():rows.append({'Method':r.method_label,'Recorded setup (s)':fmt(r.recorded_attributable_setup_seconds,1),'Online (s)':fmt(r.online_seconds,1),
        'Crossing Q':'Not comparable','Scope':'Different initialization boundaries; full attributable offline cost unknown'})
    table(release,'table6','Recorded offline and online cost boundaries',pd.DataFrame(rows),'3|1.2|1.2|1.2|3',
        'Physical CFW comparator: 4,990.219659 s complete one-worker supervised process, TIME_LIMIT at gap 0.0016196. New power-4 and final-mixture jobs reuse separately timed preflight initialization; physical and historical FFTT jobs initialize inside their timers. Cold-start amortization is not comparable. Historical worker wall, parent elapsed and iteration-only clocks are never added as one intercept. Unmeasured setup costs remain unavailable, not zero.'+portfolio_note)
    for stem,f,desc in [('online_and_crossings',crossings,'Unrounded observed times, recorded discovery cost and not-comparable amortization dispositions'),('component_observations',components,'Measured job/component nesting'),('historical_subtotals',offline,'Historical timing scopes retained separately')]:release.csv('table6',stem,f,desc)
    historical=release.read(c.r4/'accounting/historical_physical_benchmark_timing.csv').drop(columns=['source_path'],errors='ignore')
    historical['benchmark_id']=historical.benchmark_id.map(lambda x:'Physical CFW reference (45)' if 'x45' in x else 'Physical CFW sensitivity (200)' if '200' in x else 'Historical scalar checkpoint (33)')
    release.csv('tableS1','historical_physical_timing',historical,'Historical iteration, broader campaign and later-refresh clocks, with availability flags')
    forecast=release.read(c.prior/'tables/runtime_forecast.csv')
    names={'S03_ANALYTIC_VALID':'Earlier analytic VALID mixture','S03_PRIMARY':'Selected proposal without safety fallback',
        'S03_BEST_ANY':'Best VALID any-family without safety fallback','S03_NONCONSTANT_VALID':'Best nonconstant without safety fallback'}
    forecast['method_label']=forecast.scenario.map(lambda s:'ACNHN–FFTT ('+s.split('_')[1]+')' if s.startswith('FFTT_') else names[s])
    forecast=forecast[['method_label','stop_reason','accepted_iterations','fresh','achieved_RG','observed_end_to_end_seconds',
        'cold_input_load_seconds','feasible_initialization_seconds','full_AON_p50_seconds','full_AON_p90_seconds',
        'warm_iteration_p50_seconds','warm_iteration_p90_seconds','fresh_assignment_base_seconds','fresh_assignment_high_seconds',
        'metrics_base_seconds','metrics_high_seconds','censored','gate_status','estimate_type','forecast_method','registry','graph','support']]
    release.csv('tableS1','historical_pilot_forecast',forecast,
        'Previously measured pilot-based forecasts; historical method and initialization scope, not substituted for current actual times')
    annex=summary.copy();annex['method_id']=annex.method_id.map(LABELS)
    release.csv('tableS1','assignment_metadata',annex,'All suites, populations, graph/reference/mapping/state identities, statuses and timing boundaries')
    release.csv('tableS1','fallback_registry',fallback,'Complete immutable source/effective regional hashes and support/order identities')
    rows=[{'Item':'Reference definition','Evidence':'Projected physical CFW CURRENT state, x45; fresh RG=0.0006949331890154105'},
        {'Item':'Physical CFW sensitivity','Evidence':'x200 RG=0.00044683233333997485; not exact truth'},
        {'Item':'Reference clock','Evidence':'3,353.947175 s cumulative iteration wall; separate later 138.043553 s zero-update refresh; complete cold time unknown'},
        {'Item':'Sensitivity clock','Evidence':'14,614.275031 s cumulative iteration wall; 17,724.460237 s broader campaign; not complete standalone assignment'},
        {'Item':'Backbone role population','Evidence':'s03: 38,944 primitive; 11,635 macro; 37,003 physical-member coordinates'},
        {'Item':'Series projection','Evidence':'Minimum of parallel-bundle sums; baseline series CURRENT consistency audited; comparator residuals retained'},
        {'Item':'Regional reference','Evidence':'Saved CURRENT regional path-mixture traces, exact source/target key joins; not marginal physical flows'},
        {'Item':'Demand loss policy','Evidence':'Original rows and demand retained; unreachable demand reported; no hidden renormalization'},
        {'Item':'Safety registry','Evidence':'Whole-region potential immutable within each objective epoch; shared final registry for open and closed'},
        {'Item':'Model fidelity','Evidence':'Finite nonnegative below-FFTT predictions are diagnostics; negative/nonfinite costs trigger fallback'},
        {'Item':'Local labels','Evidence':'Certified A/B only; failed, severe and missing slots remain in exhaustive coverage'},
        {'Item':'Independent validation','Evidence':'No matched observed-traffic validation; no causal/economic policy validation asserted'}]
    table(release,'tableS1','Expanded identity, coverage and interpretation notes',pd.DataFrame(rows),'1|4',
        'Source paths are stored privately. Public supporting CSVs retain hashes, coordinate/role counts, termination and timing scopes. Pilot estimates are separate from actual complete-job measurements.')
    return dict(summary=summary,suites=suites,costs=costs,fits=fits,coverage=coverage,fallback=fallback,closed=closed,
        response=response,crossings=crossings,reachability=reach,paired_coverage=paired)
