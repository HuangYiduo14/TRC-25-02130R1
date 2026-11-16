"""Approved continuous campaign scheduling; every action has an append-only intent.

This controller changes allocation and time policy only. Numerical primitives,
labels, portfolios and post-selection evaluation retain their own hard checks.
"""
from __future__ import annotations
import argparse,copy,json,math,os,time,traceback
from pathlib import Path
from collections import Counter
from .context import RecoveryContext
from .labels import new_json,support_records,key,_parallel
from ..pacnhn_v1.control import read_json,sha,write_json


def binding(path):
    return dict(path=str(Path(path).resolve(strict=True)),sha256=sha(path))


def revised_config(ctx,amendment,*,reason):
    """New configuration version; previous versions and source files are immutable."""
    c=copy.deepcopy(ctx.c)
    for field,value in amendment.items():c[field]=value
    version=int(c['config_revision'])+1
    while (ctx.root/f'campaign_v{version}.json').exists():version+=1
    c.update(config_revision=version,supersedes_config_sha256=ctx.config_sha256,
             revision_reason=reason,revision_created_epoch=time.time())
    path=ctx.root/f'campaign_v{version}.json';new_json(path,c)
    new_json(ctx.root/'config'/f'REVISION_{version:03d}.json',dict(parent=binding(ctx.path),new=binding(path),reason=reason,epoch=time.time()))
    write_json(ctx.root/'ACTIVE_CONFIG.json',binding(path))
    return RecoveryContext(path,step_id=ctx.step_id)


def stage_intent(ctx,action,tasks,deadline,outputs):
    ctx.require_action(action);ctx.guard(deadline=deadline)
    if not tasks:raise ValueError('A real campaign stage needs nonempty explicit tasks')
    folder=ctx.root/'controller'/f'{action}_{time.time_ns()}';folder.mkdir(parents=True,exist_ok=False)
    sources={}
    for name in ('campaign.py','context.py','disk_accounting.py','training_recovery.py','labels.py','regional_solver.py','quota_allocation.py'):
        path=Path(__file__).with_name(name);sources[str(path)]=ctx.bind(path,'CONTROLLER_BOUND_SOURCE')
    new_json(folder/'INTENT.json',dict(phase_id=ctx.c['phase_id'],step_id=action,run_id=ctx.c['run_id'],
        config=binding(ctx.path),tasks=tasks,expected_outputs=outputs,deadline_epoch=deadline,started_epoch=time.time(),
        resource_budget=ctx.c['resources'],sources=sources,acceptance_tests=ctx.c['acceptance_tests'],
        external_review_required=False,authority='APPROVED_CONTINUOUS_RECOVERY_GOAL'))
    return folder,sources


def verify_sources(ctx,sources):
    for path,digest in sources.items():ctx.bind(path,'UNCHANGED_CONTROLLER_BOUND_SOURCE',digest)


def current_state(ctx,record,*,allow_old=True):
    rel=Path('training/regions')/record['suite_id']/f"{record['region_index']:06d}"/'STATE.json'
    path=ctx.root/rel
    if not path.exists() and allow_old:path=ctx.source_root/rel
    if not path.exists():return None,None
    ctx.bind(path,'CURRENT_RECOVERY_STATE' if path.is_relative_to(ctx.root) else 'INHERITED_REGION_STATE')
    return read_json(path),path


def family_retry_allowed(state,family,quota):
    """No known zero-tree retry or exhausted spending rule becomes a fit task."""
    from .training_recovery import target_met
    from .quota_allocation import headroom
    if target_met(state.get('selected',{}).get(family),state.get('dataset_identity'),state.get('valid_dataset_identity')):
        return False,'CURRENT_TRAIN_TARGET_MET'
    policy=state.get('recovery_family_policy',{}).get(family,{})
    if policy.get('invariant_failure'):return False,'INVARIANT_FAILURE'
    if policy.get('failures',0)>=2:return False,'REPEATED_FAILURE_AFTER_RETRY'
    if policy.get('elapsed_seconds',0)>=1200:return False,'CUMULATIVE_WORKER_TIME_EXHAUSTED'
    if policy.get('unproductive_tranches',0)>=2:return False,'TWO_UNPRODUCTIVE_TRANCHES'
    if family=='XGBOOST' and state.get('xgb_latest'):
        try:
            increment=(8,16,32)[min(policy.get('cohort_tranches',policy.get('evaluated_tranches',0)),2)]
            room=headroom(state['xgb_latest'],quota,state.get('dataset_identity'),increment=increment)
            if room['status'] in {'NO_NEW_TREE','ZERO_ROUND_ANCHOR'}:return False,'NO_NEW_TREE_WITHIN_CURRENT_QUOTA_AND_COMPLEXITY'
        except (KeyError,TypeError,ValueError):
            return True,'STATE_NEEDS_BOUNDED_AUTHENTICATION'
    return True,'APPROVED_RETRY_AVAILABLE'


def next_primary_tasks(ctx,records):
    tasks=[];ledger=[]
    quotas=ctx.c['recovery']['xgb_region_quotas']
    for record in records:
        state,path=current_state(ctx,record,allow_old=False)
        if state is None:
            tasks.append(key(record));ledger.append(dict(region_key=key(record),family='ALL',reason='NO_RECOVERY_VISIT'));continue
        outer_failures=[e for e in state.get('recovery_failures',[]) if 'visit' in e and 'type' in e]
        invariant=any(e['type'] in {'ValueError','PermissionError'} for e in outer_failures)
        if invariant or len(outer_failures)>=2:
            ledger.append(dict(region_key=key(record),family='ALL',retry_allowed=False,
                reason='REGION_INVARIANT_FAILURE' if invariant else 'REGION_REPEATED_FAILURE_AFTER_RETRY'));continue
        eligible=False
        for family in ('ICNN','XGBOOST'):
            yes,reason=family_retry_allowed(state,family,quotas[key(record)]);eligible|=yes
            ledger.append(dict(region_key=key(record),family=family,retry_allowed=yes,reason=reason))
        if eligible:tasks.append(key(record))
    return tasks,ledger


def reallocate_fourfold(ctx,records,*,pass_root):
    from .quota_allocation import observation_from_state,allocate_quotas
    prior=ctx.c['recovery']['xgb_quota_manifest'];ctx.bind(prior['path'],'ADMITTED_TWOFOLD_QUOTAS',prior['sha256'])
    allocation=read_json(prior['path'])
    if allocation.get('requested_multiplier')!=2:raise ValueError('Fourfold escalation requires a prior complete twofold allocation')
    pass_root=Path(pass_root).resolve(strict=True)
    if not pass_root.is_relative_to(ctx.root/'training/passes'):raise ValueError('Foreign twofold trial pass')
    intent_path=pass_root/'INTENT.json';ctx.bind(intent_path,'TWOFOLD_TRIAL_INTENT');intent=read_json(intent_path)
    if intent.get('config_path')!=str(ctx.path) or intent.get('config_sha256')!=ctx.config_sha256:
        raise ValueError('Latest trial pass does not use the exact current twofold configuration')
    observations=[]
    for record in records:
        state,path=current_state(ctx,record)
        if state is None:continue
        trial=None
        if path.is_relative_to(ctx.root):
            latest=int(state.get('recovery_visits',0));v=pass_root/(key(record).replace(':','_')+'.json')
            if v.exists():
                result=read_json(v)
                if result.get('visit')!=latest or result.get('region_key')!=key(record):continue
                out=[r for r in result.get('outcomes',[]) if r.get('family')=='XGBOOST']
                rounds=sum(int(r.get('rounds',0)) for r in out)
                if rounds:status='FITTED'
                elif any(r.get('status')=='QUOTA_OR_COMPLEXITY_BOUND_PRESERVED_CURRENT_MODEL' for r in out):status='NO_HEADROOM'
                else:status=None
                if status:trial=dict(allocation_sha256=prior['sha256'],status=status,new_boosting_rounds=rounds,result_binding=binding(v))
        observations.append(observation_from_state(key(record),state,source_binding=binding(path),twofold_trial=trial))
    frozen_keys=[key(r) for r in support_records(ctx)]
    manifest,mapping=allocate_quotas(allocation,observations,prior_binding=prior,expected_keys=frozen_keys,multiplier=4)
    folder=ctx.root/'config'/f'fourfold_{time.time_ns()}';new_json(folder/'OBSERVATIONS.json',observations)
    new_json(folder/'QUOTAS.json',manifest)
    recovery=dict(ctx.c['recovery'],xgb_quota_manifest=binding(folder/'QUOTAS.json'),xgb_region_quotas=mapping)
    return revised_config(ctx,dict(recovery=recovery),reason='Automatic approved at-most4x XGB allocation after bound2x trials; unchanged complete-suite cap')


def run_primary(ctx):
    from .training_recovery import run
    records=[r for r in support_records(ctx) if r['suite_id']=='s03']
    if len(records)!=1499:raise ValueError('All1499 primary regions required')
    # Reserve30minutes for four first-fit suite allocations and30minutes for the
    # independent160-fit figure experiment before the declared s03/ablation cap.
    # Released time moves downstream; no additional campaign clock is created.
    end=min(ctx.deadline('labels')-3600,ctx.deadline('training')-3600)
    folder,sources=stage_intent(ctx,'training',[key(r) for r in records],end,
        ['all1499 current cohorts audited','preserved passing models','bounded recovery candidates','final per-family stop reasons'])
    # Start from a fair near-target order while still visiting every region once.
    ranked=[]
    for r in records:
        state,_=current_state(ctx,r)
        values=[(state or {}).get('selected',{}).get(f,{}).get('audit',{}).get('TRAIN',{}).get('WMAPE_percent') for f in ('ICNN','XGBOOST')]
        errors=[float(v) if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and v>=0 else math.inf for v in values]
        unresolved=any(x>=1 for x in errors)
        ranked.append((not unresolved,max(errors),key(r)))
    tasks=[r[-1] for r in sorted(ranked)];passes=[];escalated=False
    while tasks and time.time()+30<end:
        verify_sources(ctx,sources)
        result=run(ctx,deadline=end-10,region_keys=tasks,max_workers=4)
        passes.append(binding(Path(result['output_directory'])/'RESULT.json'))
        if result['completed_visits']<len(tasks):break
        ctx=RecoveryContext(ctx.path,step_id='training')
        if not escalated:
            ctx=reallocate_fourfold(ctx,records,pass_root=result['output_directory']);escalated=True
        # STATE files are mutable only in this new run. A fresh context avoids
        # treating the next completed state version as an immutable-input change.
        ctx=RecoveryContext(ctx.path,step_id='training')
        tasks,ledger=next_primary_tasks(ctx,records)
        new_json(folder/f'PASS_{len(passes):03d}_DECISIONS.json',dict(next_keys=tasks,ledger=ledger,config=binding(ctx.path)))
        if len(passes)>=100:raise RuntimeError('Unexpected nonterminating recovery scheduling')
    ctx=RecoveryContext(ctx.path,step_id='training');tasks,ledger=next_primary_tasks(ctx,records)
    result=dict(status='POLICY_EXHAUSTED_OR_TARGETS_MET' if not tasks else 'PRIMARY_TIME_LIMIT',epoch=time.time(),
        remaining_retry_eligible=tasks,ledger=ledger,passes=passes,config=binding(ctx.path),TEST_used=False)
    new_json(folder/'RESULT.json',result);new_json(ctx.root/'training/PRIMARY_RESULT.json',result)
    verify_sources(ctx,sources);ctx.flush_bindings();return result


def _first_fit_worker(config,record,deadline):
    from .labels import prepare_initial_training
    from .training_recovery import run_region
    from threadpoolctl import threadpool_limits
    import torch
    torch.set_num_threads(1)
    ctx=RecoveryContext(config,step_id='training')
    # Preserve room for fitting within the fixed suite allowance.
    label_end=min(deadline-20,time.time()+240)
    with threadpool_limits(limits=1):
        labels=prepare_initial_training(ctx,record,deadline=label_end)
        if labels['status']!='ADMITTED_NONEMPTY_TRAIN_VALID':return dict(region_key=key(record),status='LABEL_UNRESOLVED',labels=labels)
        if time.time()+3>=deadline:return dict(region_key=key(record),status='UNTRAINED_TIME_LIMIT',labels=labels)
        fit=run_region(ctx,record,deadline=deadline,labels_root=labels['path'],labels_sha256=labels['sha256'])
        return dict(region_key=key(record),status=fit['status'],labels=labels,fit=fit)


def run_other_suites(ctx):
    ctx.require_action('training')
    frozen=read_json(ctx.root/'holdout/INPUT_FREEZE.json')
    by_suite={sid:[r['record'] for r in frozen['rows'] if r['record']['suite_id']==sid] for sid in ('s04','s02','s05','s06')}
    tasks=[key(r) for rows in by_suite.values() for r in rows]
    folder,sources=stage_intent(ctx,'training',tasks,ctx.deadline('training'),['bounded first fits','explicit labels/fitting dispositions','additional-suite overlay registry'])
    results=[];overlays=dict(ctx.c['recovery'].get('training_overlays',{}))
    for sid in ('s04','s02','s05','s06'):
        end=min(ctx.deadline('training'),time.time()+450)
        ordered=sorted(by_suite[sid],key=lambda r:(r['pairs'],r['region_index']))
        if time.time()>=end:rows=[dict(region_key=key(r),status='UNATTEMPTED_TIME_LIMIT') for r in ordered]
        else:rows=_parallel(ctx,_first_fit_worker,[(str(ctx.path),r) for r in ordered],end,max_workers=4)
        for row in rows:
            labels=row.get('labels')
            if labels and labels['status']=='ADMITTED_NONEMPTY_TRAIN_VALID':overlays[row['region_key']]={k:labels[k] for k in ('path','sha256')}
        results.append(dict(suite_id=sid,deadline_epoch=end,results=rows))
        new_json(folder/(sid+'.json'),results[-1])
    recovery=dict(ctx.c['recovery'],training_overlays=overlays)
    ctx=revised_config(ctx,dict(recovery=recovery),reason='Bounded additional-suite first-fit TRAIN/VALID index bindings; same frozen holdout inputs')
    result=dict(status='ALL_FOUR_SUITE_ALLOCATIONS_ACCOUNTED',suites=results,config=binding(ctx.path),TEST_used=False)
    new_json(folder/'RESULT.json',result);new_json(ctx.root/'training/OTHER_SUITES_RESULT.json',result)
    verify_sources(ctx,sources);ctx.flush_bindings();return result


def collect_records(ctx):
    records=[];dispositions=[]
    for region in support_records(ctx):
        state,path=current_state(ctx,region,allow_old=False)
        if state is None:
            dispositions.append(dict(region_key=key(region),status='NO_RECOVERY_MODEL_STATE'));continue
        rows=state.get('portable_records',[])
        for row in rows:
            if (row.get('current_train_dataset_identity')!=state.get('dataset_identity')
                or row.get('current_valid_dataset_identity')!=state.get('valid_dataset_identity')):
                raise ValueError('Portable record is stale on current frozen cohort: '+key(region))
            records.append(copy.deepcopy(row))
        dispositions.append(dict(region_key=key(region),status=state['status'],state=binding(path),families=[r['family'] for r in rows],
            failures=state.get('recovery_failures',[]),policy=state.get('recovery_family_policy',{})))
    if not records:raise ValueError('No available normalized records')
    return records,dispositions


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--action',required=True,choices=['primary','other','ablation'])
    args=parser.parse_args();ctx=RecoveryContext(args.config,step_id='training')
    if args.action=='primary':result=run_primary(ctx)
    elif args.action=='other':result=run_other_suites(ctx)
    else:
        from .ablation import prepare,run
        if not (ctx.root/'ablation/PLAN.json').exists():prepare(ctx)
        result=run(ctx,deadline=min(ctx.deadline('training'),ctx.c['ablation']['deadline_epoch']),pilot_only=False)
    print(json.dumps(dict(action=args.action,status=result.get('status'),config=str(ctx.path))))


if __name__=='__main__':main()
