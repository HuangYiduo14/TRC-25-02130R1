"""Synthetic metadata/API stubs only; no real models, arrays, tools or renders."""
from pathlib import Path
import copy
import json
import time
from types import SimpleNamespace
import pytest
from partC_rev1_src.pacnhn_recovery24 import publication_driver as d


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return d.binding(path)


def fixture(tmp_path, monkeypatch):
    root=tmp_path/'run';root.mkdir(); now=time.time()
    c=dict(phase_id='PACNHN_RECOVERY24',run_id='fixture',original_start_epoch=now-100,
        hard_deadline_epoch=now+10000,manuscript_outer_deadline_epoch=now+20800,offline_timing={})
    config=save(root/'campaign.json',c)
    class Context:
        def require_action(self,action):self.action=action
        def deadline(self,action=None):return self.c['manuscript_outer_deadline_epoch'] if (action or self.action)=='manuscript' else self.c['hard_deadline_epoch']
        def guard(self,deadline=None):
            if time.time()>=min(self.deadline(),deadline or self.deadline()):raise TimeoutError('fixture deadline')
        def output_path(self,relative):
            path=(self.root/relative).resolve()
            if not path.is_relative_to(self.root):raise ValueError('outside')
            return path
    ctx=Context();ctx.root=root;ctx.c=c;ctx.path=Path(config['path']);ctx.config_sha256=config['sha256'];ctx.action='publication'
    source=root/'source.py';source.write_text('synthetic source')
    monkeypatch.setattr(d,'source_paths',lambda:[source])
    def put(name,value):return save(root/d.TERMINALS[name],value)
    terminals={};plan=save(root/'ablation/PLAN.json',dict(regions=[{'region_index':i} for i in range(16)]))
    terminals['fit_freeze']=put('fit_freeze',dict(status='ALL_160_DISPOSITIONS_FROZEN',total=160,TEST_used=False,
        plan_sha256=plan['sha256'],outcomes=[dict(task={'task_id':str(i)},status='AVAILABLE' if i else 'FAILED') for i in range(160)]))
    portfolio=save(root/'portfolios/PORTFOLIOS.json',dict(status='fixture'))
    terminals['selection']=put('selection',dict(status='FROZEN',phase_id=c['phase_id'],run_id=c['run_id'],
        config_sha256=config['sha256'],config_path=config['path'],test_targets_opened=False,
        ablation_fit_freeze=terminals['fit_freeze'],portfolio=portfolio))
    terminals['regional']=put('regional',dict(status='TIME_LIMIT_COMPLETE_DISPOSITION_ACCOUNTING',regions=10442,row_count=10442*36,
        selection_freeze=terminals['selection'],TEST_used_for_selection=False,outputs={}))
    terminals['ablation']=put('ablation',dict(status='COMPLETE_WITH_EXPLICIT_LABEL_COVERAGE',fit_cases=160,
        global_selection_freeze=terminals['selection'],TEST_used_for_selection=False,output_sha256={}))
    matrix=[dict(case_id='case'+str(i)) for i in range(46)]
    rows=[dict(case=r,status='TIME_LIMIT') for r in matrix]
    reports=rows+[dict(case={'case_id':'P_REFERENCE_'+s},status='CONVERGED') for s in ('open','i95')]
    terminals['matrix']=put('matrix',matrix);terminals['computational']=put('computational',rows);terminals['reporting']=put('reporting',reports)
    terminals['assignment']=put('assignment',dict(status='COMPLETE_WITH_RECORDED_OUTCOMES',phase_id=c['phase_id'],run_id=c['run_id'],
        rows=46,reporting_rows=48,portfolio_sha256=portfolio['sha256']))
    terminals['metrics_result']=put('metrics_result',dict(status='PASS_WITH_EXPLICIT_CASE_DISPOSITIONS',phase_id=c['phase_id'],run_id=c['run_id'],reporting_rows=48,computational_cases=46))
    terminals['metrics_provenance']=put('metrics_provenance',dict(bindings=[terminals['reporting'],terminals['assignment']],portfolio_sha256=portfolio['sha256']))
    terminals['metrics']=put('metrics',dict(status='VERIFIED_REPORT_DATA',reporting_rows=48,files=[terminals['metrics_result'],terminals['metrics_provenance']]))
    a=dict(schema=d.SCHEMA,phase_id=c['phase_id'],run_id=c['run_id'],action='assets',campaign_config=config,
        tasks=['timing','geometry','configuration','extract','render'],expected_outputs=['assets'],acceptance_tests=['exact receipts'],
        root_honors_publication_lock=True,budget_seconds=300,deadline_epoch=now+1000,
        source_bindings={str(source):d.binding(source)['sha256']},terminal_bindings=terminals,output_relative='publication/driver/assets_v1')
    settings={k:None for k in d.SETTINGS_FIELDS};settings.update(schema='RECOVERY24_PUBLICATION_SETTINGS_V1',phase_id=c['phase_id'],run_id=c['run_id'])
    external=tmp_path/'external';external.mkdir();cache=external/'cache';cache.mkdir()
    for name in ('environment','compiler','pdftoppm'):settings[name]=save(external/(name+'.json'),{'fixture':name})
    settings['compiler_cache_source']=str(cache)
    settings['toolchain_authority']=save(external/'authority.json',{k:settings[k] for k in ('compiler','pdftoppm','compiler_cache_source')})
    for name,files in [('baseline',['main.tex','numerical example.tex']),('dependencies',['extra.tex'])]:
        snapshot=external/name;snapshot.mkdir();members=[]
        for filename in files:
            p=snapshot/filename;p.write_text('synthetic content');members.append(dict(relative_path=filename,sha256=d.binding(p)['sha256']))
        settings[name]=dict(snapshot_root=str(snapshot),manifest=save(external/(name+'_manifest.json'),{'members':members}))
    settings.update(resource_budget=dict(per_asset_seconds=10,compile_seconds=10,pdf_render_seconds=10),deadline_epoch=now+2000,
        output_relative='publication/config_v1',publication_output_relative='publication/final_v1')
    a['publication_settings']=save(root/'settings.json',settings)
    a['geometry_config']=save(root/'geometry_config.json',dict(phase_id=c['phase_id'],run_id=c['run_id'],
        output_relative='publication/geometry_v1',inputs={'reporting_results':terminals['reporting'],'ablation_plan':plan}))
    adm=save(root/'admission.json',a)
    return SimpleNamespace(root=root,ctx=ctx,a=a,admission=adm,terminals=terminals,put=put,source=source)


def execute(f):return d.run(f.ctx,f.admission['path'],admission_sha256=f.admission['sha256'],action=f.a['action'])


def test_exact_terminal_partial_science_is_valid_complete_accounting(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);r=d.terminal_inputs(f.ctx,f.terminals)
    assert r['fit_freeze']['outcomes'][0]['status']=='FAILED'
    assert r['computational'][0]['status']=='TIME_LIMIT'


@pytest.mark.parametrize('name,field,value',[
    ('regional','regions',10441),('ablation','fit_cases',159),('selection','config_sha256','a'*64),
    ('selection','test_targets_opened',True),('regional','TEST_used_for_selection',True),
    ('metrics_result','run_id','foreign'),('assignment','portfolio_sha256','a'*64)])
def test_rebound_wrong_native_semantics_rejected(tmp_path,monkeypatch,name,field,value):
    f=fixture(tmp_path,monkeypatch);row=d.read(f.terminals[name]['path']);row[field]=value;f.terminals[name]=f.put(name,row)
    with pytest.raises(ValueError):d.terminal_inputs(f.ctx,f.terminals)


def test_changed_bytes_and_canonical_alias_rejected(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);p=Path(f.terminals['matrix']['path']);old=p.read_bytes();p.write_bytes(old+b' ')
    with pytest.raises(ValueError,match='SHA256'):d.terminal_inputs(f.ctx,f.terminals)
    p.write_bytes(old);alternate=f.root/'other.json';alternate.write_bytes(old);f.terminals['matrix']=d.binding(alternate)
    with pytest.raises(ValueError,match='Noncanonical'):d.terminal_inputs(f.ctx,f.terminals)


def test_wrong_matrix_reference_or_duplicate_case_fails(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);rows=d.read(f.terminals['reporting']['path']);rows[-1]=rows[0]
    f.terminals['reporting']=f.put('reporting',rows)
    with pytest.raises(ValueError,match='matrix'):d.terminal_inputs(f.ctx,f.terminals)


def test_provenance_must_bind_exact_reporting_bytes(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);r=d.read(f.terminals['metrics_provenance']['path']);r['bindings'][0]['sha256']='a'*64
    f.terminals['metrics_provenance']=f.put('metrics_provenance',r)
    m=d.read(f.terminals['metrics']['path']);m['files'][1]=f.terminals['metrics_provenance'];f.terminals['metrics']=f.put('metrics',m)
    with pytest.raises(ValueError,match='provenance'):d.terminal_inputs(f.ctx,f.terminals)


@pytest.mark.parametrize('change',['plan','tool','baseline','cache'])
def test_static_prerequisite_failure_precedes_all_export_calls(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);calls=fake_stages(f,monkeypatch)
    if change=='plan':
        g=d.read(f.a['geometry_config']['path']);g['inputs']['ablation_plan']=save(f.root/'other_plan.json',{'regions':[{}]*16})
        f.a['geometry_config']=save(Path(f.a['geometry_config']['path']),g)
    elif change=='tool':Path(d.read(f.a['publication_settings']['path'])['compiler']['path']).unlink()
    elif change=='baseline':
        s=d.read(f.a['publication_settings']['path']);(Path(s['baseline']['snapshot_root'])/'main.tex').write_text('changed')
    else:Path(d.read(f.a['publication_settings']['path'])['compiler_cache_source']).rmdir()
    f.admission=save(f.root/'admission.json',f.a);r=execute(f)
    assert r['status']=='FAILED_PRESERVED_OUTPUTS' and not calls
    assert not (f.root/'report_timing').exists()


def clock_fixture(f):
    now=time.time();assets={'ended_epoch':now-5};start=now-1
    record=dict(schema='RECOVERY24_MANUSCRIPT_CLOCK_V1',phase_id=f.ctx.c['phase_id'],run_id=f.ctx.c['run_id'],
        campaign_config_sha256=f.ctx.config_sha256,start_epoch=start,deadline_epoch=min(start+10800,f.ctx.c['manuscript_outer_deadline_epoch']))
    return assets,record


def test_canonical_original_three_hour_clock(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);assets,clock=clock_fixture(f);b=save(f.root/'MANUSCRIPT_CLOCK.json',clock)
    assert d.validate_clock(f.ctx,b,assets)==clock['deadline_epoch']


@pytest.mark.parametrize('change',['extension','early','foreign','expired','nan','alias'])
def test_clock_cannot_reset_extend_or_precede_assets(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);assets,clock=clock_fixture(f)
    if change=='extension':clock['deadline_epoch']+=1
    if change=='early':clock['start_epoch']=assets['ended_epoch']-1
    if change=='foreign':clock['campaign_config_sha256']='a'*64
    if change=='nan':clock['start_epoch']=float('nan')
    if change=='expired':clock['start_epoch']=time.time()-20000;clock['deadline_epoch']=clock['start_epoch']+10800
    b=save(f.root/('alias.json' if change=='alias' else 'MANUSCRIPT_CLOCK.json'),clock)
    with pytest.raises(ValueError):d.validate_clock(f.ctx,b,assets)


def test_nested_context_guards_respect_driver_budget(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);b=d.BoundedContext(f.ctx,time.time()-1)
    assert b.deadline()<f.ctx.deadline()
    with pytest.raises(TimeoutError):b.guard(f.ctx.deadline())
    with pytest.raises(TimeoutError):b.require_action('publication')


@pytest.mark.parametrize('lock',['NATIONAL_TIMING_ACTIVE.json','ADVANCE_CONTROLLER_LOCK.json','PUBLICATION_DRIVER_LOCK.json'])
def test_existing_science_or_publication_lock_blocks_before_api(tmp_path,monkeypatch,lock):
    f=fixture(tmp_path,monkeypatch);save(f.root/lock,{'owner':'other'});calls=[]
    monkeypatch.setattr(d,'call',lambda *a,**k:calls.append(a))
    with pytest.raises((PermissionError,FileExistsError)):execute(f)
    assert not calls and d.read(f.root/lock)=={'owner':'other'}


def test_assets_admission_cannot_borrow_manuscript_time(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);f.a['deadline_epoch']=f.ctx.c['hard_deadline_epoch']+1;f.admission=save(f.root/'admission.json',f.a)
    with pytest.raises(ValueError,match='extension'):execute(f)


def fake_stages(f,monkeypatch,fail=None):
    calls=[]
    def stub(stage,ctx,**kw):
        calls.append(stage);assert isinstance(ctx,d.BoundedContext)
        if stage==fail:raise RuntimeError('synthetic failure')
        if stage=='timing':save(f.root/'report_timing/v1/MANIFEST.json',{'status':'VERIFIED_TIMING_EXPORT'})
        elif stage=='geometry':save(f.root/'publication/geometry_v1/RESULT.json',{'status':'EXPORTED_WITH_EXPLICIT_FLOW_COVERAGE'})
        elif stage=='configuration':
            cfg=save(f.root/'publication/config_v1/PUBLICATION_CONFIG.json',{'output_relative':'publication/final_v1'})
            return {'publication_config':cfg}
        elif stage=='extract':save(f.root/'publication/final_v1/EXTRACTION_RESULT.json',{'status':'EXTRACTED_RENDER_PENDING'})
        elif stage=='render':save(f.root/'publication/final_v1/REPRODUCTION_RESULT.json',{'status':'REPRODUCED_VISUAL_QA_PENDING'})
        elif stage=='assemble':save(f.root/'publication/final_v1/MANUSCRIPT_RESULT.json',{'status':'ASSEMBLED_COMPILE_AND_VISUAL_QA_PENDING'})
        elif stage=='compile':
            pdf=f.root/'publication/final_v1/toy.pdf';pdf.write_bytes(b'synthetic bytes, not a PDF')
            page=pdf.with_name('page-1.png');page.write_bytes(b'synthetic bytes, not an image')
            save(f.root/'publication/final_v1/PDF_BUILD_RESULT.json',{'status':'COMPILED_EVERY_PAGE_VISUAL_QA_PENDING',
                'pdf_path':str(pdf),'pdf_sha256':d.binding(pdf)['sha256'],'pages':1,'page_images':[d.binding(page)],
                'source_preservation':'PASS_NUMERICAL_ONLY_SOURCE_PRESERVATION'})
        return {'status':'synthetic return'}
    monkeypatch.setattr(d,'call',stub);monkeypatch.setattr(d,'request_from_outputs',lambda *a:{'synthetic':True})
    return calls


def test_assets_exact_sequence_and_explicit_qa_pending(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);calls=fake_stages(f,monkeypatch);r=execute(f)
    assert r['status']=='ASSETS_REPRODUCED_VISUAL_QA_PENDING' and calls==f.a['tasks']
    assert not r['actual_visual_qa'] and not r['clock_created'] and r['error'] is None
    assert not (f.root/'MANUSCRIPT_CLOCK.json').exists()
    assert not (f.root/'PUBLICATION_DRIVER_LOCK.json').exists()


def test_failure_preserves_earlier_outputs_and_never_launches_later_step(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);calls=fake_stages(f,monkeypatch,fail='geometry');r=execute(f)
    assert r['status']=='FAILED_PRESERVED_OUTPUTS' and calls==['timing','geometry']
    assert Path(r['outputs']['timing']['path']).exists()
    assert r['error']['message']=='synthetic failure'
    with pytest.raises(FileExistsError):execute(f)


def test_source_change_mid_stage_preserved_as_failure(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);fake_stages(f,monkeypatch);original=d.call
    def changing(*args,**kw):
        result=original(*args,**kw);f.source.write_text('changed');return result
    monkeypatch.setattr(d,'call',changing);r=execute(f)
    assert r['status']=='FAILED_PRESERVED_OUTPUTS' and 'SHA256' in r['error']['message']


def test_manuscript_is_separately_admitted_after_assets_with_qa_pending(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);calls=fake_stages(f,monkeypatch);assets=execute(f)
    assets_binding=d.binding(f.root/'publication/driver/assets_v1/RESULT.json')
    now=time.time();clock=dict(schema='RECOVERY24_MANUSCRIPT_CLOCK_V1',phase_id=f.ctx.c['phase_id'],run_id=f.ctx.c['run_id'],
        campaign_config_sha256=f.ctx.config_sha256,start_epoch=now,deadline_epoch=min(now+10800,f.ctx.c['manuscript_outer_deadline_epoch']))
    f.a.update(action='manuscript',tasks=['assemble','compile'],output_relative='publication/driver/manuscript_v1',
        assets_result=assets_binding,publication_config=assets['outputs']['publication_config'],manuscript_clock=save(f.root/'MANUSCRIPT_CLOCK.json',clock))
    for key in ('geometry_config','publication_settings'):f.a.pop(key)
    f.admission=save(f.root/'manuscript_admission.json',f.a)
    r=execute(f)
    assert r['status']=='MANUSCRIPT_COMPILED_VISUAL_QA_PENDING' and calls[-2:]==['assemble','compile']
    assert r['actual_visual_qa'] is False


@pytest.mark.parametrize('change',['zero','missing','duplicate','preservation'])
def test_manuscript_cannot_accept_empty_incomplete_or_nonpreserved_build(tmp_path,monkeypatch,change):
    f=fixture(tmp_path,monkeypatch);fake_stages(f,monkeypatch);assets=execute(f)
    now=time.time();clock=dict(schema='RECOVERY24_MANUSCRIPT_CLOCK_V1',phase_id=f.ctx.c['phase_id'],run_id=f.ctx.c['run_id'],
        campaign_config_sha256=f.ctx.config_sha256,start_epoch=now,deadline_epoch=min(now+10800,f.ctx.c['manuscript_outer_deadline_epoch']))
    f.a.update(action='manuscript',tasks=['assemble','compile'],output_relative='publication/driver/manuscript_v1',
        assets_result=d.binding(f.root/'publication/driver/assets_v1/RESULT.json'),publication_config=assets['outputs']['publication_config'],
        manuscript_clock=save(f.root/'MANUSCRIPT_CLOCK.json',clock))
    f.admission=save(f.root/'manuscript_admission.json',f.a);original=d.call
    def changed(stage,*args,**kw):
        result=original(stage,*args,**kw)
        if stage=='compile':
            p=f.root/'publication/final_v1/PDF_BUILD_RESULT.json';r=d.read(p)
            if change=='zero':r['pages']=0;r['page_images']=[]
            elif change=='missing':r['page_images']=[]
            elif change=='duplicate':r['pages']=2;r['page_images']*=2
            else:r['source_preservation']='FAILED'
            save(p,r)
        return result
    monkeypatch.setattr(d,'call',changed);result=execute(f)
    assert result['status']=='FAILED_PRESERVED_OUTPUTS'
    assert 'nonempty rendered PDF' in result['error']['message']


def test_request_uses_only_actual_producer_output_fingerprints(tmp_path,monkeypatch):
    f=fixture(tmp_path,monkeypatch);producers={};files={};settings={k:None for k in d.SETTINGS_FIELDS}
    settings.update(schema='RECOVERY24_PUBLICATION_SETTINGS_V1',phase_id=f.ctx.c['phase_id'],run_id=f.ctx.c['run_id'])
    statuses={'metrics':'VERIFIED_REPORT_DATA','timing':'VERIFIED_TIMING_EXPORT','regional':'COMPLETE_ACCOUNTING_WITH_SCOPE_LIMITATIONS',
        'ablation':'COMPLETE_WITH_EXPLICIT_LABEL_COVERAGE','geometry':'EXPORTED_WITH_EXPLICIT_FLOW_COVERAGE'}
    for name,(producer,filename) in d.DATASETS.items():
        p=f.root/'inputs'/producer/filename;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('x\n1\n');files[name]=d.binding(p)
    for producer,status in statuses.items():
        rows=[files[n] for n,(p,_) in d.DATASETS.items() if p==producer]
        record={'status':status}
        if producer in ('metrics','timing'):record['files']=rows
        elif producer in ('geometry','regional'):record['outputs']={Path(x['path']).name:x for x in rows}
        else:record['output_sha256']={Path(x['path']).name:x['sha256'] for x in rows}
        producers[producer]=save(f.root/'inputs'/producer/'RESULT.json',record)
    request=d.request_from_outputs(f.ctx,settings,producers)
    assert request['inputs']==files and request['campaign_config_sha256']==f.ctx.config_sha256
    row=d.read(producers['metrics']['path']);row['files']=[];producers['metrics']=save(Path(producers['metrics']['path']),row)
    with pytest.raises(ValueError,match='Missing'):d.request_from_outputs(f.ctx,settings,producers)
