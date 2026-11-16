"""Synthetic metadata only: no real outputs, models, labels, tools or rendering."""
from pathlib import Path
import csv
import json
import time
import pytest
from partC_rev1_src.pacnhn_recovery24 import publication_configuration as p


def config():
    return dict(phase_id='PACNHN_RECOVERY24',run_id='synthetic',suite_ids=['s02','s03','s04','s05','s06'],
        resources={'regional_workers':4},assignment={'RG_SPTT':.001,'cap_seconds':{'ICNN':1500}},
        recovery={'target_train_percent':1.},holdout={'s03_inputs_per_region':48,'other_regions_per_suite':32,'other_inputs_per_region':8},
        ablation={'fractions':[.1,.25,.5,.7,1.],'fit_policy':{'icnn_updates':1000}},manuscript_outer_deadline_epoch=time.time()+10000)


def networks():
    return [dict(suite_id=s,resolution='' if s=='P' else 1,nodes=10,edges=20,backbone_links=3,regional_links=17)
        for s in ['P','s02','s03','s04','s05','s06']]


def test_settings_separate_requested_configured_measured_and_unknown():
    rows=p.settings_rows(config(),{'python_version':'synthetic','packages':{'torch':'fixture'}},networks())
    byname={r['setting']:r for r in rows}
    assert byname['cpu']['value']=='N/A (not recorded)'
    assert byname['Configured regional workers']['evidence_scope'].startswith('Configured')
    assert byname['New holdout requested per s03 region']['value']=='48'
    assert 'admitted' in byname['New holdout requested per s03 region']['evidence_scope']
    assert byname['s03 frozen network']['evidence_scope']=='Measured frozen network counts'
    assert not any('GPU hours' in r['setting'] for r in rows)


@pytest.mark.parametrize('change',['duplicate','missing','negative','nan','noninteger','empty'])
def test_frozen_network_counts_fail_closed(change):
    rows=networks()
    if change=='duplicate':rows[-1]=dict(rows[0])
    elif change=='missing':rows.pop()
    elif change=='empty':rows[0]={'suite_id':'P'}
    else:rows[0]['nodes']={'negative':-1,'nan':float('nan'),'noninteger':2.5}[change]
    with pytest.raises(ValueError):p.settings_rows(config(),{},rows)


def test_producer_shapes_normalized_without_payload_reads(tmp_path):
    digest='a'*64;expected=tmp_path/'a.csv'
    assert (str(expected),digest) in p.recorded_bindings({'outputs':{'a':{'path':str(expected),'sha256':digest}}},tmp_path)
    assert (str(expected),digest) in p.recorded_bindings({'output_sha256':{'a.csv':digest}},tmp_path)
    with pytest.raises(ValueError):p.recorded_bindings({'output_sha256':{'../a.csv':digest}},tmp_path)
    with pytest.raises(ValueError):p.recorded_bindings({'files':[{'path':str(expected),'sha256':'future'}]},tmp_path)


def fixture(tmp_path):
    root=tmp_path/'run';root.mkdir();admin=tmp_path/'admin';admin.mkdir();outside=tmp_path/'external';outside.mkdir()
    def save(path,obj):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return {'path':str(path),'sha256':p.sha(path)}
    class Context:
        config_sha256='b'*64
        def require_action(self,action):assert action=='publication'
        def guard(self,*args):pass
        def output_path(self,rel):
            out=(self.root/rel).resolve()
            if not out.is_relative_to(self.root):raise ValueError('outside')
            return out
        def bind(self,path,role,expected=None):
            digest=p.sha(path)
            if expected is not None and digest!=expected:raise ValueError('hash mismatch')
            return digest
        def flush_bindings(self):pass
    ctx=Context();ctx.root=root;ctx.c=config();ctx.c['admin_root']=str(admin)
    env=save(admin/'ENVIRONMENT.json',dict(python_executable='python_fixture',python_version='synthetic',packages={}))
    inputs={};byproducer={name:[] for name in p.PRODUCER_STATUSES}
    for name in p.INPUT_NAMES:
        file=(root/'report_timing/v1/methods_with_training_timing.csv') if name=='methods' else root/'data'/p.INPUT_PRODUCERS[name]/(name+'.csv')
        file.parent.mkdir(parents=True,exist_ok=True)
        if name=='networks':
            with file.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(networks()[0]));writer.writeheader();writer.writerows(networks())
        else:file.write_text('fixture\n1\n')
        inputs[name]={'path':str(file),'sha256':p.sha(file)};byproducer[p.INPUT_PRODUCERS[name]].append(inputs[name])
    producers={}
    for name,files in byproducer.items():
        folder=root/'data'/name
        if name in {'metrics','timing'}:body={'files':files}
        elif name in {'regional','geometry'}:body={'outputs':{Path(r['path']).name:r for r in files}}
        else:body={'output_sha256':{Path(r['path']).name:r['sha256'] for r in files}}
        producers[name]=save(folder/'RESULT.json',{'status':sorted(p.PRODUCER_STATUSES[name])[0],**body})
    compiler=save(outside/'tectonic.exe',{'fake':'not executable'});poppler=save(outside/'pdftoppm.exe',{'fake':'not executable'})
    cache=outside/'cache';cache.mkdir()
    authority=save(outside/'approved_toolchain.json',dict(compiler=compiler,pdftoppm=poppler,compiler_cache_source=str(cache)))
    preserved={}
    for name in ('baseline','dependencies'):
        snapshot=admin/name;snapshot.mkdir()
        preserved[name]={'snapshot_root':str(snapshot),'manifest':save(admin/(name+'.json'),{'files':[]})}
    request=dict(schema=p.SCHEMA,phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],campaign_config_sha256=ctx.config_sha256,
        output_relative='publication/config_v1',publication_output_relative='publication/final_v1',inputs=inputs,
        producers=producers,environment=env,toolchain_authority=authority,compiler=compiler,pdftoppm=poppler,
        compiler_cache_source=str(cache),resource_budget={'per_asset_seconds':10,'compile_seconds':20,'pdf_render_seconds':20},
        deadline_epoch=ctx.c['manuscript_outer_deadline_epoch'],**preserved)
    request_path=root/'REQUEST.json';save(request_path,request)
    return ctx,request,request_path,save


@pytest.mark.parametrize('mutation',['none','missing_input','wrong_campaign','wrong_status','missing_producer_binding','changed_csv','unapproved_binary','cache','expired','budget','environment','raw_methods_without_disposition'])
def test_prepare_authenticated_metadata_only(tmp_path,mutation):
    ctx,request,path,save=fixture(tmp_path)
    if mutation=='missing_input':request['inputs'].pop('learning')
    if mutation=='wrong_campaign':request['campaign_config_sha256']='c'*64
    if mutation in {'wrong_status','missing_producer_binding'}:
        producer=Path(request['producers']['metrics']['path']);record=json.loads(producer.read_text())
        if mutation=='wrong_status':record['status']='RUNNING'
        else:record['files']=[]
        request['producers']['metrics']=save(producer,record)
    if mutation=='changed_csv':Path(request['inputs']['learning']['path']).write_text('changed')
    if mutation in {'unapproved_binary','cache'}:
        authority=Path(request['toolchain_authority']['path']);record=json.loads(authority.read_text())
        record.pop('compiler' if mutation=='unapproved_binary' else 'compiler_cache_source')
        request['toolchain_authority']=save(authority,record)
    if mutation=='expired':request['deadline_epoch']=1.
    if mutation=='budget':request['resource_budget']['compile_seconds']=float('inf')
    if mutation=='environment':
        request['environment']=save(ctx.root/'OTHER_ENV.json',{'python_executable':'fixture'})
    if mutation=='raw_methods_without_disposition':
        methods=ctx.root/'report_data/table2.csv';methods.parent.mkdir();methods.write_text('fixture\n1\n')
        request['inputs']['methods']={'path':str(methods),'sha256':p.sha(methods)}
    save(path,request);before={str(f):p.sha(f) for f in tmp_path.rglob('*') if f.is_file()}
    if mutation!='none':
        with pytest.raises((ValueError,TimeoutError)):p.prepare(ctx,path,request_sha256=p.sha(path),deadline=1e12)
        assert not (ctx.root/'publication/config_v1/PUBLICATION_CONFIG.json').exists()
    else:
        result=p.prepare(ctx,path,request_sha256=p.sha(path),deadline=1e12)
        assert result['status']=='PREPARED_FROM_TERMINAL_BOUND_INPUTS' and not result['publication_executed']
        actual=json.loads(Path(result['publication_config']['path']).read_text())
        assert set(actual['inputs'])==p.INPUT_NAMES|{'settings'}
        assert actual['campaign_config_sha256']==ctx.config_sha256 and actual['compiler']==dict(request['compiler'],bytes=Path(request['compiler']['path']).stat().st_size)
        assert actual['inputs']['methods']['path']==str(ctx.root/'report_timing/v1/methods_with_training_timing.csv')
        assert not (ctx.root/'publication/final_v1').exists()
        with pytest.raises(FileExistsError):p.prepare(ctx,path,request_sha256=p.sha(path),deadline=1e12)
    assert before=={name:p.sha(name) for name in before}


def test_wrong_request_hash_and_path_escape_are_rejected(tmp_path):
    ctx,request,path,save=fixture(tmp_path)
    with pytest.raises(ValueError):p.prepare(ctx,path,request_sha256='f'*64,deadline=1e12)
    request['output_relative']='../outside';save(path,request)
    with pytest.raises(ValueError):p.prepare(ctx,path,request_sha256=p.sha(path),deadline=1e12)


def test_raw_methods_fallback_requires_missing_augmented_output_and_explicit_disposition(tmp_path):
    ctx,request,path,save=fixture(tmp_path)
    old=Path(request['inputs']['methods']['path']);old.unlink()  # synthetic fixture only
    methods=ctx.root/'report_data/table2.csv';methods.parent.mkdir();methods.write_text('fixture\n1\n')
    binding={'path':str(methods),'sha256':p.sha(methods)};request['inputs']['methods']=binding
    producer=Path(request['producers']['metrics']['path']);record=json.loads(producer.read_text());record['files'].append(binding)
    request['producers']['metrics']=save(producer,record);request['timing_unavailable_disposition']='EXPLICIT_TIMING_UNAVAILABLE'
    save(path,request)
    assert not p.prepare(ctx,path,request_sha256=p.sha(path),deadline=1e12)['publication_executed']


def test_producer_upstream_input_is_not_a_generated_output(tmp_path):
    path=tmp_path/'upstream.csv';digest='a'*64
    record={'inputs':[{'path':str(path),'sha256':digest}],'outputs':{}}
    assert not p.producer_outputs('regional',record,tmp_path)
    record={'input_bindings':{'input':{'path':str(path),'sha256':digest}},'outputs':{}}
    assert not p.producer_outputs('geometry',record,tmp_path)


@pytest.mark.parametrize('case',['valid','changed_packages','wrong_original','changed_original_hash'])
def test_hardware_supplement_binds_and_preserves_original_environment(tmp_path,case):
    ctx,request,path,save=fixture(tmp_path)
    baseline=request['environment'];record=json.loads(Path(baseline['path']).read_text())
    record.update(cpu='Recorded CPU',gpu=['Recorded GPU'],total_ram_bytes=123456789,
        hardware_observation={'cpu':{'NumberOfCores':16,'NumberOfLogicalProcessors':24}},baseline_environment_binding=baseline)
    if case=='changed_packages':record['packages']={'torch':'invented'}
    if case=='wrong_original':record['baseline_environment_binding']=save(ctx.root/'FOREIGN_ENV.json',{'python_executable':'fixture'})
    if case=='changed_original_hash':Path(baseline['path']).write_text('{}')
    request['environment']=save(Path(ctx.c['admin_root'])/'ENVIRONMENT_V2_HARDWARE.json',record);save(path,request)
    if case!='valid':
        with pytest.raises(ValueError):p.prepare(ctx,path,request_sha256=p.sha(path),deadline=1e12)
    else:
        result=p.prepare(ctx,path,request_sha256=p.sha(path),deadline=1e12)
        with (Path(result['publication_config']['path']).parent/'settings.csv').open(newline='') as stream:rows=list(csv.DictReader(stream))
        byname={r['setting']:r for r in rows}
        assert byname['CPU physical cores']['value']=='16' and byname['CPU logical processors']['value']=='24'
        assert 'not measured worker utilization' in byname['CPU physical cores']['evidence_scope']
        assert byname['gpu']['value']=='["Recorded GPU"]'
