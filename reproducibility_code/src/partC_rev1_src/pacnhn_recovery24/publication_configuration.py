"""Build a separate publication contract from authenticated terminal metadata.

No fitting, label/model arrays, rendering, clock creation or campaign-config edits.
The caller supplies an explicit SHA-bound request after the producers finish.
"""
from pathlib import Path
import csv
import json
import math
import re
import time
from ..pacnhn_v1.control import read_json, sha
from .ablation import _new

from . import assignment_reconciliation as repair

SCHEMA='RECOVERY24_PUBLICATION_CONFIGURATION_REQUEST_V1'
INPUT_NAMES={'networks','methods','backbone_links','convergence','learning','ablation_summary',
    'ablation_observations','region_geometry','region_locations','resolution_counts',
    'partition_stability','boundary_jaccard','primary_map_flows','closure_geometry',
    'served_membership','timing'}
PRODUCER_STATUSES={
    'metrics':{'VERIFIED_REPORT_DATA'},
    'timing':{'VERIFIED_TIMING_EXPORT'},
    'geometry':{'EXPORTED_WITH_EXPLICIT_FLOW_COVERAGE'},
    'regional':{'COMPLETE_ACCOUNTING_WITH_SCOPE_LIMITATIONS','TIME_LIMIT_COMPLETE_DISPOSITION_ACCOUNTING'},
    'ablation':{'COMPLETE_WITH_EXPLICIT_LABEL_COVERAGE'},
}
INPUT_PRODUCERS={
    'networks':'metrics','methods':'timing','backbone_links':'metrics','convergence':'metrics',
    'learning':'regional','ablation_summary':'ablation','ablation_observations':'ablation',
    'region_geometry':'geometry','region_locations':'geometry','resolution_counts':'geometry',
    'partition_stability':'geometry','boundary_jaccard':'geometry','primary_map_flows':'geometry',
    'closure_geometry':'geometry','served_membership':'metrics','timing':'metrics',
}
COUNT_COLUMNS=['nodes','edges','backbone_links','regional_links','regions','active_regions','active_pairs']


def _digest(value):
    if not isinstance(value,str) or re.fullmatch('[0-9a-f]{64}',value) is None:
        raise ValueError('An explicit lowercase SHA256 is required')
    return value


def _binding(ctx,record,role,bindings,deadline):
    ctx.guard(deadline)
    path=Path(record['path']).resolve(strict=True);expected=_digest(record['sha256'])
    digest=ctx.bind(path,role,expected)
    if digest!=expected:raise ValueError('Binding implementation did not authenticate '+role)
    item=dict(path=str(path),sha256=digest,bytes=path.stat().st_size)
    prior=bindings.get(str(path))
    if prior is not None and prior!=item:raise ValueError('Conflicting input binding')
    bindings[str(path)]=item
    return path,item


def recorded_bindings(record,folder):
    """Normalize actual producer manifest shapes, without opening their payloads."""
    result=set()
    def walk(value):
        if isinstance(value,dict):
            if 'path' in value and 'sha256' in value:
                result.add((str(Path(value['path']).resolve()),_digest(value['sha256'])))
            for name,digest in value.get('output_sha256',{}).items():
                p=Path(name)
                if p.is_absolute() or len(p.parts)!=1 or p.name!=name:raise ValueError('Unsafe output filename')
                result.add((str((Path(folder)/p).resolve()),_digest(digest)))
            for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    walk(record)
    return result


def producer_outputs(name,record,folder):
    if name in {'metrics','timing'}:body={'files':record['files']}
    elif name in {'regional','geometry'}:body={'outputs':record['outputs']}
    elif name=='ablation':body={'output_sha256':record['output_sha256']}
    else:raise ValueError('Unknown producer output schema')
    return recorded_bindings(body,folder)


def _value(value):
    if value is None or value=='':return 'N/A (not recorded)'
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite setting')
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False) if isinstance(value,(dict,list)) else str(value)


def environment_record(ctx,path,bindings,deadline):
    """Admit original environment or its explicitly bound hardware supplement."""
    canonical=Path(ctx.c['admin_root']).resolve()/'ENVIRONMENT.json'
    record=read_json(path)
    if path==canonical:return record
    if path!=canonical.with_name('ENVIRONMENT_V2_HARDWARE.json'):
        raise ValueError('Canonical campaign environment or its V2 hardware supplement required')
    baseline=record.get('baseline_environment_binding')
    if not isinstance(baseline,dict):raise ValueError('Hardware supplement must bind original environment')
    original,_=_binding(ctx,baseline,'ORIGINAL_ENVIRONMENT_UNCHANGED',bindings,deadline)
    if original!=canonical:raise ValueError('Hardware supplement has foreign original environment')
    prior=read_json(original)
    for name,value in prior.items():
        if json.dumps(record.get(name),sort_keys=True,allow_nan=False)!=json.dumps(value,sort_keys=True,allow_nan=False):
            raise ValueError('Hardware supplement changed original environment field: '+name)
    return record


def settings_rows(config,environment,networks):
    """Configured limits and measured metadata are labelled as different evidence."""
    suites=list(config['suite_ids']);expected={'P',*suites}
    if len(networks)!=len(expected) or {r['suite_id'] for r in networks}!=expected:
        raise ValueError('Exactly P plus the configured five frozen network records required')
    result=[]
    def add(setting,value,scope,source):
        result.append(dict(setting=setting,value=_value(value),evidence_scope=scope,source_field=source))
    add('Flow and cost units','Annual thousand tons; seconds','Declared native units','Approved method convention')
    add('Python version',environment.get('python_version'),'Recorded environment metadata','ENVIRONMENT.python_version')
    add('Operating system',environment.get('platform'),'Recorded environment metadata','ENVIRONMENT.platform')
    for name in ('numpy','scipy','pandas','torch','xgboost','numba','matplotlib'):
        add(name+' version',environment.get('packages',{}).get(name),'Recorded environment metadata','ENVIRONMENT.packages.'+name)
    for name in ('cpu','gpu','total_ram_bytes'):
        add(name.replace('_',' '),environment.get(name),'Recorded environment metadata; availability is not utilization','ENVIRONMENT.'+name)
    for name in ('NumberOfCores','NumberOfLogicalProcessors'):
        add('CPU '+('physical cores' if name=='NumberOfCores' else 'logical processors'),
            environment.get('hardware_observation',{}).get('cpu',{}).get(name),
            'Recorded hardware availability; not measured worker utilization','ENVIRONMENT.hardware_observation.cpu.'+name)
    for name in ('regional_workers','torch_threads','xgboost_threads','tree_rss_bytes','minimum_available_bytes'):
        add('Configured '+name.replace('_',' '),config.get('resources',{}).get(name),'Configured limit; not measured usage','campaign.resources.'+name)
    for name in ('RG_SPTT','cap_seconds'):
        add('Assignment '+name,config.get('assignment',{}).get(name),'Configured stopping/budget rule','campaign.assignment.'+name)
    add('Strict TRAIN target (%)',config.get('recovery',{}).get('target_train_percent'),'Configured strict less-than target; attainment reported separately','campaign.recovery.target_train_percent')
    add('New holdout requested per s03 region',config.get('holdout',{}).get('s03_inputs_per_region'),'Requested cohort size; admitted coverage reported separately','campaign.holdout.s03_inputs_per_region')
    add('New holdout requested other-suite regions/inputs',
        {k:config.get('holdout',{}).get(k) for k in ('other_regions_per_suite','other_inputs_per_region')},
        'Requested cohort size; admitted coverage reported separately','campaign.holdout')
    add('Nested ablation fractions',config.get('ablation',{}).get('fractions'),'Configured experiment; completion reported separately','campaign.ablation.fractions')
    add('Ablation fitting allowance',config.get('ablation',{}).get('fit_policy'),'Equal configured allowance per fraction; no completed-fit claim','campaign.ablation.fit_policy')
    for row in networks:
        sid=row['suite_id'];values=[]
        for name in COUNT_COLUMNS:
            raw=row.get(name)
            if raw in (None,''):continue
            try:number=float(raw)
            except (ValueError,TypeError) as exc:raise ValueError('Invalid frozen network count') from exc
            if not math.isfinite(number) or number<0 or not number.is_integer():raise ValueError('Invalid frozen network count')
            values.append(name.replace('_',' ')+'='+str(int(number)))
        if not values:raise ValueError('Frozen network count record is empty')
        add(sid+' frozen network', '; '.join(values),'Measured frozen network counts','networks.csv:'+sid)
    return result


def prepare(ctx,request_path,*,request_sha256,deadline):
    """Write new settings/config files only, from exact terminal producer bindings."""
    ctx.require_action('publication');ctx.guard(deadline);started=time.time();own_sha=sha(__file__);bindings={}
    request_file,request_binding=_binding(ctx,dict(path=str(request_path),sha256=request_sha256),'PUBLICATION_CONFIGURATION_REQUEST',bindings,deadline)
    request=read_json(request_file)
    if (request.get('schema')!=SCHEMA or request.get('phase_id')!=ctx.c['phase_id']
        or request.get('run_id')!=ctx.c['run_id'] or request.get('campaign_config_sha256')!=ctx.config_sha256
        or set(request.get('inputs',{}))!=INPUT_NAMES):
        raise ValueError('Exact campaign and complete scientific input request required')
    assignment_revision=request.get('assignment_revision');layout=repair.resolve(ctx,assignment_revision)
    ctx.bind(repair.__file__,'ASSIGNMENT_REVISION_RESOLVER_SOURCE',sha(repair.__file__))
    if assignment_revision is not None:
        _binding(ctx,assignment_revision,'EXPLICIT_ASSIGNMENT_REVISION',bindings,deadline)
    destination=ctx.output_path(request['output_relative'])
    if not destination.is_relative_to(Path(ctx.root)/'publication') or destination==Path(ctx.root)/'publication':
        raise ValueError('New dedicated publication configuration directory required')
    if destination.exists():raise FileExistsError('Configuration preparation is append-only')
    env_path,_=_binding(ctx,request['environment'],'RECORDED_ENVIRONMENT',bindings,deadline)
    environment=environment_record(ctx,env_path,bindings,deadline)
    producers={}
    for name,binding in request['producers'].items():
        if name not in PRODUCER_STATUSES:raise ValueError('Unknown terminal producer')
        path,_=_binding(ctx,binding,'TERMINAL_'+name,bindings,deadline)
        if not path.is_relative_to(Path(ctx.root)):raise ValueError('Current terminal producer must belong to this recovery')
        record=read_json(path)
        if record.get('status') not in PRODUCER_STATUSES[name]:raise ValueError('Producer is not terminal: '+name)
        if name in {'metrics','timing'} and record.get('assignment_revision')!=assignment_revision:
            raise ValueError('Producer assignment revision differs: '+name)
        producers[name]=producer_outputs(name,record,path.parent)
    inputs={}
    for name,binding in request['inputs'].items():
        path=Path(binding['path']).resolve(strict=True);wanted=(str(path),_digest(binding['sha256']))
        producer=INPUT_PRODUCERS[name]
        if name=='methods':
            preferred=layout['timing_root']/'methods_with_training_timing.csv'
            if path!=preferred.resolve():
                if (preferred.exists() or path!=(layout['metrics_root']/'table2.csv').resolve()
                    or request.get('timing_unavailable_disposition')!='EXPLICIT_TIMING_UNAVAILABLE'):
                    raise ValueError('Use exact augmented timing methods or explicit unavailable fallback')
                producer='metrics'
        if wanted not in producers.get(producer,set()):raise ValueError('CSV lacks matching terminal producer binding: '+name)
        if not path.is_relative_to(Path(ctx.root)) or path.suffix.lower()!='.csv':raise ValueError('Current scientific CSV required')
        _,inputs[name]=_binding(ctx,binding,'SCIENTIFIC_'+name,bindings,deadline)
    # The request authenticates authority metadata; binaries must ALSO occur in it,
    # rather than being accepted just because their current filesystem hash matches.
    authority_path,_=_binding(ctx,request['toolchain_authority'],'EXISTING_TOOLCHAIN_AUTHORITY',bindings,deadline)
    authority=read_json(authority_path);authority_files=recorded_bindings(authority,authority_path.parent)
    toolchain={}
    for name in ('compiler','pdftoppm'):
        p,item=_binding(ctx,request[name],'APPROVED_'+name.upper(),bindings,deadline)
        if (str(p),item['sha256']) not in authority_files:raise ValueError('Binary is absent from approved toolchain metadata')
        toolchain[name]=item
    cache=Path(request['compiler_cache_source']).resolve(strict=True)
    if not cache.is_dir() or cache.is_relative_to(Path(ctx.root)):raise ValueError('Existing external compiler cache required')
    if str(cache) not in _directory_values(authority):raise ValueError('Cache directory not recorded in toolchain authority')
    preserved={}
    for name in ('baseline','dependencies'):
        entry=request[name];path,item=_binding(ctx,entry['manifest'],'PRESERVED_'+name.upper(),bindings,deadline)
        snapshot=Path(entry['snapshot_root']).resolve(strict=True)
        if not snapshot.is_dir():raise ValueError('Existing immutable manuscript snapshot required')
        preserved[name]=dict(manifest=item,snapshot_root=str(snapshot))
    with Path(inputs['networks']['path']).open(newline='',encoding='utf-8') as stream:networks=list(csv.DictReader(stream))
    settings=settings_rows(ctx.c,environment,networks)
    budget=request['resource_budget']
    if set(budget)!={'per_asset_seconds','compile_seconds','pdf_render_seconds'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in budget.values()):
        raise ValueError('Explicit finite positive publication budgets required')
    publication_deadline=request['deadline_epoch']
    if type(publication_deadline) not in (int,float) or not math.isfinite(publication_deadline) or publication_deadline>ctx.c['manuscript_outer_deadline_epoch']:
        raise ValueError('Publication deadline exceeds the recorded outer authority')
    if publication_deadline<=time.time():raise TimeoutError('Publication request deadline expired')
    final_root=ctx.output_path(request['publication_output_relative'])
    if not final_root.is_relative_to(Path(ctx.root)/'publication') or final_root==destination or final_root.exists():
        raise ValueError('Unused separate publication output directory required')
    ctx.guard(deadline);destination.mkdir(parents=True,exist_ok=False)
    settings_path=destination/'settings.csv'
    with settings_path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=['setting','value','evidence_scope','source_field']);writer.writeheader();writer.writerows(settings)
    inputs['settings']=dict(path=str(settings_path),sha256=sha(settings_path),bytes=settings_path.stat().st_size)
    config=dict(schema='RECOVERY24_V5_PUBLICATION_V1',phase_id=ctx.c['phase_id'],run_id=ctx.c['run_id'],
        campaign_config_sha256=ctx.config_sha256,assignment_revision=assignment_revision,tasks=['publication','manuscript'],deadline_epoch=publication_deadline,
        output_relative=request['publication_output_relative'],inputs=inputs,python_path=environment['python_executable'],
        resource_budget=budget,compiler_cache_source=str(cache),**toolchain,**preserved)
    for item in bindings.values():ctx.guard(deadline);ctx.bind(item['path'],'FINAL_CONFIGURATION_INPUT_RECHECK',item['sha256'])
    if sha(__file__)!=own_sha:raise ValueError('Configuration adapter changed')
    ctx.require_action('publication')
    _new(destination/'PUBLICATION_CONFIG.json',config)
    _new(destination/'PROVENANCE.json',dict(request=request_binding,source_sha256=own_sha,config_sha256=ctx.config_sha256,
        inputs=list(bindings.values()),settings_semantics='Recorded environment and frozen counts; configured limits separately labelled; unknown fields remain N/A.',
        toolchain_cache_scope='Recorded existing cache directory; compilation copies it and runs cache-only. Cache contents are not measured outcomes.',
        original_payload_writes=False,scientific_computation=False,rendering=False,clock_created=False))
    result=dict(status='PREPARED_FROM_TERMINAL_BOUND_INPUTS',elapsed_seconds=time.time()-started,settings_rows=len(settings),scientific_datasets=17,
        publication_config=dict(path=str(destination/'PUBLICATION_CONFIG.json'),sha256=sha(destination/'PUBLICATION_CONFIG.json')),
        publication_executed=False,visual_QA='PENDING')
    _new(destination/'RESULT.json',result);ctx.flush_bindings();return result


def _directory_values(record):
    result=set()
    if isinstance(record,dict):
        for key,value in record.items():
            if key in {'compiler_cache_source','tectonic_cache','cache_dir','cache_directory'} and isinstance(value,str):
                result.add(str(Path(value).resolve()))
            result.update(_directory_values(value))
    elif isinstance(record,list):
        for value in record:result.update(_directory_values(value))
    return result
