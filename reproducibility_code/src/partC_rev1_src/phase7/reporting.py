"""Population reports over explicitly named, trained and evaluated cohorts."""
from __future__ import annotations
import argparse,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
from .common import root,region_out,read_json,write_json,load_config,sha,elapsed
from .campaign import summarize

ERROR_COLUMNS=['MAE_seconds','RMSE_seconds','MARE_equal_pairs','MARE_LOW_pair_weighted','p50_relative','p90_relative','p95_relative','max_relative','bias_seconds',
    'mean_interval_violation_relative','mean_interval_violation_native','residual_cost_RMSE_scaled','residual_value_interval_RMSE_scaled','congestion_relative_MAE','floor_violation_count','negative_cost_count','composite_loss']

def frame_write(frame,path):
    frame.to_parquet(path.with_suffix('.parquet'),index=False)
    frame.to_csv(path.with_suffix('.csv'),index=False)

def aggregate(metrics,registry):
    rows=[];population=registry.loc[registry.m_active.gt(0)];exposure=float(population.low_exposure.sum())
    for keys,g in metrics.groupby(['model','split','family','grade'],dropna=False):
        assert g.region_id.is_unique
        weights=g.LOW_exposure.to_numpy(float);total=weights.sum()
        for method,w in [('REGION_UNWEIGHTED',np.ones(len(g))/len(g)),('LOW_TRAVERSAL_EXPOSURE_WEIGHTED',weights/total)]:
            r=dict(zip(['model','split','family','grade'],keys));r.update(aggregation=method,cohort='AMONG_TRAINED_AND_EVALUATED_REGIONS_WITH_THIS_SPLIT_FAMILY_GRADE',
                covered_regions=len(g),total_active_regions=1566,covered_region_fraction=len(g)/1566,covered_active_pairs=int(g.m_active.sum()),total_active_pairs=80316,
                covered_pair_fraction=float(g.m_active.sum()/80316),covered_LOW_exposure=float(total),total_LOW_exposure=exposure,covered_LOW_exposure_fraction=float(total/exposure),
                scenario_rows=int(g.n_rows.sum()))
            for k in ERROR_COLUMNS:r[k]=float(g[k].to_numpy(float)@w)
            r['regions_passing_screen']=int(g.accuracy_screen.sum());r['any_negative_cost_regions']=int(g.negative_cost_count.gt(0).sum())
            r['negative_cost_count_sum']=int(g.negative_cost_count.sum());r['floor_violation_count_sum']=int(g.floor_violation_count.sum())
            r['max_relative_global']=float(g.max_relative.max());r['max_seconds_global']=float(g.max_seconds.max())
            r['note']='p95/max fields are weighted summaries of regional tail statistics; they are not pooled-coordinate quantiles.'
            rows.append(r)
    return pd.DataFrame(rows)

def recommend(registry,metrics,curves,c,diagnostics=None):
    output=[];expansion=[];main=metrics.loc[metrics.family.eq('ALL')&metrics.grade.eq('ALL')]
    for r in registry.to_dict('records'):
        if r['m_active']==0:continue
        rid=r['region_id'];mm=main.loc[main.region_id.eq(rid)];cc=curves.loc[curves.region_id.eq(rid)&curves.scheduled.eq(True)].sort_values('steps') if len(curves) else pd.DataFrame()
        own=mm.loc[mm.model.eq('ICNN')].set_index('split')
        flags=[];evidence=[];train_improve=None;valid_improve=None
        if len(cc)>=10:
            k=max(2,math.ceil(len(cc)*.2));recent=cc.iloc[-k:];prior=cc.iloc[-2*k:-k]
            train_improve=float((prior.train_composite.mean()-recent.train_composite.mean())/max(abs(prior.train_composite.mean()),1e-15))
            valid_improve=float((prior.valid_composite.mean()-recent.valid_composite.mean())/max(abs(prior.valid_composite.mean()),1e-15))
        if own.empty:
            flags=['UNTRAINED_AT_BUDGET' if r['status'].startswith('PENDING') else 'REGION_FAILURE_REVIEW']
            evidence=['No validated frozen ICNN is included in the error cohort; no zero-error substitution.']
        else:
            tr=own.loc['TRAIN'];va=own.loc['VALID'];partial='PARTIAL' in r['status']
            gaps=float(va.MARE_equal_pairs-tr.MARE_equal_pairs)
            if bool(va.accuracy_screen) and not partial:
                flags.append('DONE_FOR_CURRENT_SCOPE');evidence.append('Frozen checkpoint passes the independent VALID research screen; test sensitivity remains separately reported.')
            compute_limited=partial or r['status']=='TRAINED_EPOCH_LIMIT'
            if compute_limited and train_improve is not None and train_improve>.001 and valid_improve>.001 and gaps<.05:
                flags.append('MORE_EPOCHS_CANDIDATE');evidence.append('At cap/cutoff, recent TRAIN and VALID composite losses both continue to improve without a large total-cost generalization gap.')
            if tr.MARE_equal_pairs<=.05 and va.MARE_equal_pairs>.05:
                flags.append('MORE_DATA_CANDIDATE');evidence.append('Low TRAIN and elevated independent VALID errors suggest a coverage or generalization issue; finite split counts, regularization and model mismatch remain alternatives.')
                expansion.append(dict(region_id=rid,m_active=r['m_active'],coordinates='Existing ordered active coordinates only; exact candidates require external review',
                    family='Inspect family-conditioned VALID/LOW_TEST residual and observed-domain tables',current_TRAIN_AB=int(r.get('n_AB_TRAIN',0)),current_VALID_AB=int(r.get('n_AB_VALID',0)),
                    source_support_hash=r['support_hash'],coverage_evidence='TRAIN/VALID gap; no causal proof that more labels suffice',intended_query_range='To be selected only after external review; no new sample generated',executed=False))
            base=mm.loc[mm.model.eq('SEPARABLE_POWER4_NNLS')&mm.split.eq('VALID')]
            baseline_better=bool(len(base) and base.iloc[0].MARE_equal_pairs+.001 < va.MARE_equal_pairs)
            plateau=train_improve is not None and train_improve<.01 and valid_improve<.01
            if baseline_better or (tr.MARE_equal_pairs>.05 and va.MARE_equal_pairs>.05 and plateau):
                flags.append('CAPACITY_OR_OPTIMIZER_REVIEW');evidence.append('The TRAIN-fit separable baseline beats ICNN on VALID, or both TRAIN and VALID errors remain high on a plateau. This does not isolate width/depth from optimization, initialization or smooth-model mismatch.')
            grades=metrics.loc[metrics.region_id.eq(rid)&metrics.model.eq('ICNN')&metrics.split.eq('VALID')&metrics.family.eq('ALL')].set_index('grade')
            if 'A' in grades.index and 'B' in grades.index and grades.loc['B','MARE_equal_pairs']>1.5*max(grades.loc['A','MARE_equal_pairs'],1e-12):
                flags.append('LABEL_PRECISION_REVIEW');evidence.append('VALID grade-B marginal errors exceed grade-A errors; grade composition and input difficulty confound attribution. Finite-gap labels need not have exact marginal coordinates.')
            if partial and not flags:
                flags.append('PARTIAL_TRAINING_REVIEW');evidence.append('Trained partial checkpoint retained; available curves do not yet identify whether epochs, data or capacity is limiting.')
            if not flags:
                flags.append('CAPACITY_OR_OPTIMIZER_REVIEW');evidence.append('Current fit is inadequate without decisive support for more epochs or data; review curves and baseline residuals before changing the model.')
            dd=diagnostics.loc[diagnostics.region_id.eq(rid)] if diagnostics is not None and len(diagnostics) else pd.DataFrame()
            if len(dd)==3:
                dd=dd.sort_values('fraction');vals=dd.VALID_MARE_equal_pairs.to_numpy()
                evidence.append('Preselected nested TRAIN diagnostics report VALID MARE for 25%, 50% and full data: '+', '.join(f'{v:.6g}' for v in vals)+'. Different convergence, optimizer updates and TRAIN-derived scales limit causal interpretation.')
            else:evidence.append('No nested data-size probe was run for this region; the generalization gap alone does not establish a need for more labels.')
        output.append(dict(region_id=rid,m_active=r['m_active'],status=r['status'],recommendations=';'.join(flags),evidence=' '.join(evidence),
            recent20pct_TRAIN_composite_improvement=train_improve,recent20pct_VALID_composite_improvement=valid_improve,scheduled_checks=len(cc),
            pack_path=str(region_out(c,rid)/'pack'),checkpoint_path=str(region_out(c,rid)/'last.pt'),
            continuation_authorized=False,next_command='Requires new explicit authority; original campaign deadline must not be extended. See resume_queue.json.'))
    return pd.DataFrame(output),pd.DataFrame(expansion,columns=['region_id','m_active','coordinates','family','current_TRAIN_AB','current_VALID_AB','source_support_hash','coverage_evidence','intended_query_range','executed'])


def diagnostic_summary(c,metrics):
    file=root(c)/'diagnostics/nested_train_subsets/DIAGNOSTIC_RESULT.json';records=[]
    if not file.exists():return pd.DataFrame()
    result=read_json(file)
    for r in result['results']:
        if 'valid' not in r:continue
        records.append(dict(region_id=r['region_id'],m_active=r['m_active'],fraction=r['fraction'],TRAIN_rows=r['TRAIN_rows'],VALID_rows=r['VALID_rows'],
            steps=r['steps'],epochs=r['epochs'],best_steps=r['best_steps'],best_epoch=r['best_epoch'],status=r['status'],wall_seconds=r['wall_seconds'],
            VALID_MARE_equal_pairs=r['valid']['MARE_equal_pairs'],VALID_MARE_LOW_pair_weighted=r['valid']['MARE_LOW_pair_weighted'],
            VALID_interval_violation=r['valid']['mean_interval_violation_relative'],
            comparison='VALID only; same regional seed and architecture; convergence, updates and numerical TRAIN scales differ'))
    for rid in result['regions']:
        rd=region_out(c,rid);st=read_json(rd/'status.json');p=read_json(rd/'pack/manifest.json')
        main=metrics.loc[metrics.region_id.eq(rid)&metrics.model.eq('ICNN')&metrics.split.eq('VALID')&metrics.family.eq('ALL')&metrics.grade.eq('ALL')]
        if not len(main):continue
        r=main.iloc[0]
        records.append(dict(region_id=rid,m_active=p['m_active'],fraction=1.,TRAIN_rows=p['counts_AB']['TRAIN'],VALID_rows=p['counts_AB']['VALID'],
            steps=st['steps'],epochs=st['epoch'],best_steps=st['best_steps'],best_epoch=st['best_epoch'],status=st['status'],wall_seconds=None,
            VALID_MARE_equal_pairs=r.MARE_equal_pairs,VALID_MARE_LOW_pair_weighted=r.MARE_LOW_pair_weighted,VALID_interval_violation=r.mean_interval_violation_relative,
            comparison='Frozen main model; no retraining or selection from diagnostics'))
    return pd.DataFrame(records)

def collect(c,*,preview=False):
    out=root(c);reports=out/('preview_reports' if preview else 'reports');reports.mkdir(exist_ok=True)
    source_registry=pd.read_csv(out/'all_region_registry.csv')
    if preview:
        registry=pd.read_csv(out/'region_status.csv');cov=read_json(out/'coverage_summary.json')
    else:registry,cov=summarize(c,source_registry)
    mframes=[];curves=[];splits=[];mathrows=[];benches=[];modelrows=[]
    for rid in registry.region_id:
        rd=region_out(c,rid);pm=rd/'pack/manifest.json'
        if pm.exists():
            meta=read_json(pm)
            for split,grades in meta['split_quality_counts'].items():
                for grade,n in grades.items():splits.append(dict(region_id=rid,split=split,quality=grade,count=n,eligible_AB=grade in ['A','B']))
        lp=rd/'learning_curves.jsonl'
        if lp.exists():
            for line in lp.read_text().splitlines():
                r=json.loads(line);tr=r.pop('train');va=r.pop('valid')
                curves.append(dict(region_id=rid,**r,train_composite=tr['composite_loss'],valid_composite=va['composite_loss'],
                    train_MARE=tr['MARE_equal_pairs'],valid_MARE=va['MARE_equal_pairs'],train_LOW_MARE=tr['MARE_LOW_pair_weighted'],valid_LOW_MARE=va['MARE_LOW_pair_weighted'],
                    train_cost_loss=tr['cost_loss'],valid_cost_loss=va['cost_loss'],train_value_loss=tr['value_interval_loss'],valid_value_loss=va['value_interval_loss'],train_floor_loss=tr['floor_loss'],valid_floor_loss=va['floor_loss']))
        ep=rd/'evaluation/manifest.json'
        if ep.exists():
            e=read_json(ep)
            if e['status']!='PASS' or e['steps']<1:continue
            mframes.append(pd.read_parquet(rd/'evaluation/metrics.parquet'))
            mr=read_json(rd/'evaluation/mathematical_validation.json');mr.pop('normalized_finite_differences',None);mathrows.append(dict(region_id=rid,**mr))
            br=read_json(rd/'evaluation/benchmarks.json')
            for b in br['batches']:benches.append(dict(region_id=rid,m_active=read_json(pm)['m_active'],**b,device=br['device'],dtype=br['dtype'],threads=br['threads'],batch1_HVP_seconds=br['batch1_HVP_seconds']))
            st=read_json(rd/'status.json')
            with np.load(rd/'pack/scales.npz',allow_pickle=False) as z:
                numerical_scale=dict(Vscale=float(z['Vscale']),TRAIN_gain=float(z['gain']),effective_residual_scale=float(z['Vscale']*z['gain']),
                    input_scale_min=float(z['scale'].min()),input_scale_max=float(z['scale'].max()),TRAIN_value_scale=float(z['value_scale']))
            modelrows.append(dict(region_id=rid,m_active=st['m_active'],steps=st['steps'],best_steps=st['best_steps'],best_epoch=st['best_epoch'],**numerical_scale,
                portable_path=str(rd/'evaluation/portable.npz'),portable_sha256=e['files']['portable.npz']['sha256'],pack_manifest_sha256=sha(pm),
                native_runtime_valid=True,national_readiness=False))
    if not mframes:raise ValueError('No validated trained model cohort exists')
    metrics=pd.concat(mframes,ignore_index=True);curve=pd.DataFrame(curves);split_counts=pd.DataFrame(splits);models=pd.DataFrame(modelrows)
    keys=['region_id','steps','scheduled']
    curve['checkpoint_replay_duplicate']=curve.duplicated(keys,keep='last')
    duplicate_keys=curve.loc[curve.duplicated(keys,keep=False)]
    for key,g in duplicate_keys.groupby(keys):
        cols=[n for n in curve.columns if n.startswith(('train_','valid_')) and not n.endswith('_seconds')]
        assert all(g[n].nunique(dropna=False)==1 for n in cols),'Replayed checkpoint changed logical TRAIN/VALID metrics: '+str(key)
    logical_curve=curve.loc[~curve.checkpoint_replay_duplicate]
    frozen_valid=metrics.loc[metrics.model.eq('ICNN')&metrics.split.eq('VALID')&metrics.family.eq('ALL')&metrics.grade.eq('ALL')].set_index('region_id').accuracy_screen
    registry['last_validation_accuracy_adequacy']=registry.get('accuracy_adequacy')
    registry['frozen_VALID_accuracy_screen']=registry.region_id.map(frozen_valid).astype('boolean')
    models['frozen_VALID_accuracy_screen']=models.region_id.map(frozen_valid).astype('boolean')
    registry['learning_accuracy_source']='frozen_VALID_accuracy_screen refers to the selected checkpoint; last_validation_accuracy_adequacy refers to the final optimizer state'
    diagnostics=diagnostic_summary(c,metrics)
    ag=aggregate(metrics,registry);docket,expansion=recommend(registry,metrics,logical_curve,c,diagnostics)
    main=metrics.loc[metrics.family.eq('ALL')&metrics.grade.eq('ALL')]
    icnn=main.loc[main.model.eq('ICNN')];baseline=main.loc[~main.model.eq('ICNN')]
    # Baselines are constant across epochs; attach their TRAIN/VALID errors to every
    # curve row using the exact same frozen TRAIN fit, never heldout-based scales.
    for model_name,prefix in [('FREEFLOW_LINEAR','freeflow'),('SEPARABLE_POWER4_NNLS','power4')]:
        for split in ['TRAIN','VALID']:
            b=baseline.loc[baseline.model.eq(model_name)&baseline.split.eq(split),['region_id','composite_loss','MARE_equal_pairs']]
            curve=curve.merge(b.rename(columns={'composite_loss':prefix+'_'+split+'_composite','MARE_equal_pairs':prefix+'_'+split+'_MARE'}),on='region_id',how='left',validate='many_to_one')
    paired=baseline.merge(icnn[['region_id','split',*ERROR_COLUMNS]],on=['region_id','split'],suffixes=('_baseline','_ICNN'),validate='many_to_one')
    for k in ERROR_COLUMNS:paired[k+'_ICNN_minus_baseline']=paired[k+'_ICNN']-paired[k+'_baseline']
    for name,frame in [('all_region_registry',registry),('source_split_quality_counts',split_counts),('training_curves',curve),
        ('metrics_by_region_split_family',metrics),('metrics_by_region_split',main),('aggregate_metrics',ag),('baseline_comparison',paired),
        ('mathematical_export_validation',pd.DataFrame(mathrows)),('inference_benchmarks',pd.DataFrame(benches)),('model_manifest',models),('next_action_docket',docket),('phase6_expansion_candidates_NOT_EXECUTED',expansion)]:
        frame_write(frame,reports/name)
    exceptions=registry.loc[registry.m_active.gt(0)&(~registry.region_id.isin(models.region_id)|registry.status.str.contains('PARTIAL'))]
    frame_write(exceptions,reports/'untrained_or_failed_regions')
    pending=exceptions[['region_id','status','m_active','steps']].to_dict('records')
    for r in pending:r.update(pack_path=str(region_out(c,r['region_id'])/'pack'),checkpoint_path=str(region_out(c,r['region_id'])/'last.pt'),authorization_required=True)
    write_json(reports/'resume_queue.json',dict(original_campaign_must_not_resume_after_deadline=True,original_deadline=c['hard_deadline_unix'],continuation_authorized=False,regions=pending,
        future_command_template='After a new reviewed authority only: use its explicit phase, step, run ID and configuration, importing the listed same-model checkpoint states. No command in this handoff extends this clock.'))
    verified=registry.loc[registry.region_id.isin(models.region_id)]
    coverage=dict(total_regions=1847,total_active_regions=1566,total_inactive_skips=281,total_active_pairs=80316,total_LOW_exposure=float(registry.low_exposure.sum()),
        trained_evaluated_regions=len(models),trained_evaluated_active_pairs=int(verified.m_active.sum()),trained_evaluated_LOW_exposure=float(verified.low_exposure.sum()),
        trained_region_fraction=len(models)/1566,trained_pair_fraction=float(verified.m_active.sum()/80316),trained_exposure_fraction=float(verified.low_exposure.sum()/registry.low_exposure.sum()),
        pending_or_partial_or_failed_regions=len(exceptions),status_counts=registry.status.value_counts().to_dict(),missing_models_excluded_from_error_aggregates=True)
    write_json(reports/'coverage_summary.json',coverage)
    frame_write(split_counts.groupby(['split','quality'],as_index=False)['count'].sum(),reports/'source_split_quality_totals')
    resources=[]
    for name in ['resource_timeline.jsonl','guardian_resource_timeline.jsonl']:
        if (out/name).exists():resources.extend(json.loads(s) for s in (out/name).read_text().splitlines())
    frame_write(pd.DataFrame(resources).sort_values('unix'),reports/'resource_usage')
    write_json(reports/'REPORT_COLLECTION.json',dict(status='PREVIEW' if preview else 'PASS',unix=time.time(),elapsed_seconds=elapsed(c),metrics_rows=len(metrics),curve_rows=len(curve),coverage=coverage))
    write_json(reports/'CHECKPOINT_REPLAY_CURVE_AUDIT.json',dict(status='PASS',raw_curve_rows=len(curve),duplicate_replayed_checks=int(curve.checkpoint_replay_duplicate.sum()),
        all_repeated_logical_metrics_identical=True,raw_evidence_preserved=True,analysis_policy='Every row remains in training_curves; the final copy per region/step/scheduled key is used for trends and plots.'))
    if len(diagnostics):frame_write(diagnostics,reports/'nested_data_size_diagnostics')
    curve_summary=docket[['region_id','m_active','scheduled_checks','recent20pct_TRAIN_composite_improvement','recent20pct_VALID_composite_improvement']].merge(
        registry[['region_id','epoch','steps','rows_seen','unique_rows_seen','train_seconds','validation_seconds','load_seconds','best_epoch','best_steps']],on='region_id',validate='one_to_one')
    frame_write(curve_summary,reports/'learning_curve_summary')
    if not preview:
        frame_write(registry,out/'all_region_registry')
    return reports,registry,metrics,curve,ag,coverage

def figures(reports,registry,metrics,curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fd=reports/'figures';fd.mkdir(exist_ok=True)
    main=metrics.loc[metrics.model.eq('ICNN')&metrics.family.eq('ALL')&metrics.grade.eq('ALL')]
    valid=main.loc[main.split.eq('VALID')].sort_values('m_active')
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    for split,color in [('TRAIN','#246b82'),('VALID','#d18135'),('LOW_TEST','#895a91'),('HIGH_PROJECTED','#638743')]:
        g=main.loc[main.split.eq(split)];axs[0].scatter(g.m_active,g.MARE_equal_pairs*100,s=13,alpha=.45,label=split,color=color)
    axs[0].set(xscale='log',yscale='symlog',xlabel='Active input dimension',ylabel='Equal-pair MARE (%)');axs[0].set_ylim(bottom=0);axs[0].axhline(5,color='gray',ls='--',lw=1);axs[0].legend(fontsize=8)
    statuses=registry.loc[registry.m_active.gt(0)].status.value_counts().sort_values();bars=axs[1].barh([s.replace('TRAINED_','').replace('_',' ') for s in statuses.index],statuses.values,color='#246b82');axs[1].bar_label(bars,padding=3);axs[1].set_xlim(right=statuses.max()*1.12);axs[1].set_xlabel('Active regions');axs[1].set_title('Execution status; accuracy is a separate axis')
    fig.suptitle('s03 ICNN errors among trained/evaluated regions and full cohort coverage');fig.tight_layout();fig.savefig(fd/'errors_and_coverage.png',dpi=170);plt.close(fig)
    groups=[valid.loc[valid.m_active.eq(1)],valid.loc[valid.m_active.between(2,16)],valid.loc[valid.m_active.between(17,64)],valid.loc[valid.m_active.between(65,256)],valid.loc[valid.m_active.ge(257)]]
    selected=list(dict.fromkeys([g.iloc[len(g)//2].region_id for g in groups if len(g)]+[valid.iloc[-1].region_id]))
    fig,axs=plt.subplots(2,3,figsize=(12,7.2))
    for ax,rid in zip(axs.flat,selected):
        g=curves.loc[curves.region_id.eq(rid)&~curves.checkpoint_replay_duplicate].sort_values('steps');m=int(valid.loc[valid.region_id.eq(rid),'m_active'].iloc[0])
        ax.plot(g.fractional_epoch,g.train_composite,label='TRAIN',color='#246b82');ax.plot(g.fractional_epoch,g.valid_composite,label='VALID',color='#d18135');ax.set(yscale='log',xlabel='Epochs',ylabel='Composite loss',title=f'm={m} | {rid[-8:]}');ax.legend(fontsize=8)
    for ax in list(axs.flat)[len(selected):]:ax.set_visible(False)
    fig.suptitle('Learning curves: dimension representatives chosen without error ranking');fig.tight_layout();fig.savefig(fd/'learning_curves.png',dpi=170);plt.close(fig)
    paired=pd.read_parquet(reports/'baseline_comparison.parquet');g=paired.loc[paired.split.eq('LOW_TEST')]
    fig,ax=plt.subplots(figsize=(7,5))
    for name,color in [('FREEFLOW_LINEAR','#638743'),('SEPARABLE_POWER4_NNLS','#895a91')]:
        b=g.loc[g.model.eq(name)];ax.scatter(100*b.MARE_equal_pairs_baseline,100*b.MARE_equal_pairs_ICNN,s=14,alpha=.45,label=name,color=color)
    ax.plot([0,100],[0,100],color='gray',ls='--');ax.set(xscale='symlog',yscale='symlog',xlabel='Baseline LOW_TEST MARE (%)',ylabel='ICNN LOW_TEST MARE (%)',title='Paired heldout comparisons on identical canonical inputs');ax.set_xlim(left=0);ax.set_ylim(bottom=0);ax.legend();fig.tight_layout();fig.savefig(fd/'baseline_comparison.png',dpi=170);plt.close(fig)
    write_json(fd/'FIGURE_SOURCES.json',dict(error='metrics_by_region_split.parquet',coverage='all_region_registry.parquet',curves='training_curves.parquet',baseline='baseline_comparison.parquet',representative_regions=selected))
    dp=reports/'nested_data_size_diagnostics.parquet'
    if dp.exists():
        data=pd.read_parquet(dp);fig,axs=plt.subplots(2,4,figsize=(13,6.6))
        for ax,(rid,g) in zip(axs.flat,data.groupby('region_id',sort=False)):
            g=g.sort_values('fraction');ax.plot(g.fraction*100,g.VALID_MARE_equal_pairs*100,marker='o',color='#246b82')
            for r in g.itertuples():ax.annotate(f'{r.epochs} ep', (r.fraction*100,r.VALID_MARE_equal_pairs*100),xytext=(3,5),textcoords='offset points',fontsize=7)
            ax.set(title=f'm={int(g.m_active.iloc[0])} | {rid[-8:]}',xlabel='TRAIN rows (% of full)',ylabel='VALID MARE (%)',xticks=[25,50,100])
            ax.set_ylim(0,max(float(g.VALID_MARE_equal_pairs.max()*100)*1.2,1e-12));ax.ticklabel_format(axis='y',useOffset=False,style='sci',scilimits=(-3,4))
            ax.set_title(f'm={int(g.m_active.iloc[0])} | {rid[-8:]}',pad=22)
        for ax in list(axs.flat)[data.region_id.nunique():]:ax.set_visible(False)
        fig.suptitle('Nested data-size probes: same seed; convergence and TRAIN scales differ')
        fig.tight_layout();fig.savefig(fd/'nested_data_size_diagnostics.png',dpi=170);plt.close(fig)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--preview',action='store_true');a=p.parse_args();c=load_config(a.config)
    if not a.preview:assert (root(c)/'CAMPAIGN_EXECUTION_DONE.json').exists()
    reports,registry,metrics,curves,ag,cov=collect(c,preview=a.preview);figures(reports,registry,metrics,curves);print(json.dumps(cov))
if __name__=='__main__':main()
