"""Explicit publication integration; no work or scientific imports on import.

Assets and manuscript are separately admitted actions. This driver never creates
a clock, launches an assignment, fits a model, opens TEST targets, or certifies
visual quality. Existing producer APIs retain their own guards and provenance.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import hashlib
import time
import traceback

SCHEMA = 'RECOVERY24_PUBLICATION_DRIVER_ADMISSION_V1'
TERMINALS = {
    'metrics': 'report_data/MANIFEST.json',
    'metrics_result': 'report_data/RESULT.json',
    'metrics_provenance': 'report_data/PROVENANCE.json',
    'regional': 'evaluation/regional_v1/RESULT.json',
    'ablation': 'ablation/evaluation/RESULT.json',
    'assignment': 'ASSIGNMENT_STAGE_RESULT.json',
    'matrix': 'ASSIGNMENT_MATRIX.json',
    'computational': 'ASSIGNMENT_RESULTS.json',
    'reporting': 'REPORTING_RESULTS.json',
    'selection': 'training/MODEL_SELECTION_FREEZE.json',
    'fit_freeze': 'ablation/FIT_FREEZE.json',
}
DATASETS = {
    'networks': ('metrics', 'networks.csv'),
    'methods': ('timing', 'methods_with_training_timing.csv'),
    'backbone_links': ('metrics', 'backbone_links.csv'),
    'convergence': ('metrics', 'convergence.csv'),
    'learning': ('regional', 'learning.csv'),
    'ablation_summary': ('ablation', 'FIGURE5_SUMMARIES.csv'),
    'ablation_observations': ('ablation', 'REGION_FRACTION_OBSERVATIONS.csv'),
    'region_geometry': ('geometry', 'region_geometry.csv'),
    'region_locations': ('geometry', 'region_locations.csv'),
    'resolution_counts': ('geometry', 'resolution_geometry_counts.csv'),
    'partition_stability': ('geometry', 'partition_stability.csv'),
    'boundary_jaccard': ('geometry', 'boundary_jaccard.csv'),
    'primary_map_flows': ('geometry', 'primary_i95_map_flows.csv'),
    'closure_geometry': ('geometry', 'closure_geometry.csv'),
    'served_membership': ('metrics', 'served_membership.csv'),
    'timing': ('metrics', 'timing.csv'),
}
SETTINGS_FIELDS = {'schema', 'phase_id', 'run_id', 'environment', 'toolchain_authority',
    'baseline', 'dependencies', 'compiler', 'pdftoppm', 'compiler_cache_source',
    'resource_budget', 'deadline_epoch', 'output_relative', 'publication_output_relative'}


def read(path):
    return json.loads(Path(path).read_bytes())


def binding(path):
    p = Path(path).resolve(strict=True); before = p.stat(); h = hashlib.sha256()
    if not p.is_file(): raise ValueError('A file binding is required')
    with p.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''): h.update(chunk)
    after = p.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('File changed while hashing')
    return dict(path=str(p), sha256=h.hexdigest())


def authenticate(record, *, canonical=None):
    actual = binding(record['path'])
    if actual['sha256'] != record['sha256']: raise ValueError('SHA256 mismatch: ' + actual['path'])
    if canonical is not None and Path(actual['path']) != Path(canonical).resolve():
        raise ValueError('Noncanonical evidence path')
    return Path(actual['path'])


def new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def source_paths():
    here = Path(__file__).resolve().parent; old = here.parent / 'pacnhn_v1'
    paths = [here / (n + '.py') for n in ('publication_driver', 'context', 'disk_accounting',
        'offline_timing', 'publication_geometry', 'publication_configuration', 'publication', 'ablation', 'latex_compat', 'assignment_reconciliation')]
    paths += [old / (n + '.py') for n in ('control', 'continuation', 'publication')]
    paths += list((here / 'publication_templates').glob('*.py'))
    paths += list((old / 'publication_templates').glob('*.py'))
    return sorted(set(paths))


def check_sources(admission):
    expected = {str(p.resolve()) for p in source_paths()}
    if not expected.issubset(admission['source_bindings']): raise ValueError('Publication source closure incomplete')
    for path, digest in admission['source_bindings'].items():
        authenticate(dict(path=path, sha256=digest))


def terminal_inputs(ctx, supplied, assignment_revision=None):
    """Match actual canonical producer fingerprints and their cross-bindings."""
    root = Path(ctx.root)
    if set(supplied) != set(TERMINALS): raise ValueError('Complete explicit terminal binding map required')
    resolver=importlib.import_module('.assignment_reconciliation',__package__)
    layout=resolver.resolve(ctx,assignment_revision); canonical={name:root/rel for name,rel in TERMINALS.items()}
    for name,key in [('assignment','terminal'),('matrix','matrix'),('computational','computational'),('reporting','reporting')]:
        canonical[name]=layout['paths'][key]
    for name,file in [('metrics','MANIFEST.json'),('metrics_result','RESULT.json'),('metrics_provenance','PROVENANCE.json')]:
        canonical[name]=layout['metrics_root']/file
    records = {name: read(authenticate(supplied[name], canonical=canonical[name])) for name in TERMINALS}
    selection = records['selection']; selected = supplied['selection']
    if (selection.get('status') != 'FROZEN' or selection.get('phase_id') != ctx.c['phase_id']
            or selection.get('run_id') != ctx.c['run_id'] or selection.get('config_sha256') != ctx.config_sha256
            or Path(selection['config_path']).resolve() != Path(ctx.path).resolve()
            or selection.get('test_targets_opened') is not False):
        raise ValueError('Frozen selection/campaign identity mismatch')
    if authenticate(selection['ablation_fit_freeze']) != Path(supplied['fit_freeze']['path']).resolve():
        raise ValueError('Ablation freeze is not the selected freeze')
    # Authenticating both references separately also proves equal bytes.
    f = records['fit_freeze']; outcomes = f.get('outcomes', [])
    if (f.get('status') != 'ALL_160_DISPOSITIONS_FROZEN' or f.get('total') != 160
            or len(outcomes) != 160 or len({r['task']['task_id'] for r in outcomes}) != 160
            or any(r.get('status') in (None, 'UNATTEMPTED') for r in outcomes)
            or f.get('TEST_used') is not False): raise ValueError('Ablation terminal accounting incomplete')
    for name, field in [('regional', 'selection_freeze'), ('ablation', 'global_selection_freeze')]:
        item = records[name]
        if authenticate(item[field]) != Path(selected['path']).resolve(): raise ValueError('Evaluation selection mismatch')
        if item.get('TEST_used_for_selection') is not False: raise ValueError('TEST selection is prohibited')
    if (records['regional'].get('status') not in {'COMPLETE_ACCOUNTING_WITH_SCOPE_LIMITATIONS', 'TIME_LIMIT_COMPLETE_DISPOSITION_ACCOUNTING'}
            or records['regional'].get('regions') != 10442 or records['regional'].get('row_count') != 10442*36):
        raise ValueError('Regional result is not complete disposition accounting')
    if records['ablation'].get('status') != 'COMPLETE_WITH_EXPLICIT_LABEL_COVERAGE' or records['ablation'].get('fit_cases') != 160:
        raise ValueError('Ablation evaluation accounting incomplete')
    assignment = records['assignment']; matrix = records['matrix']; rows = records['computational']; reports = records['reporting']
    ids = [r['case_id'] for r in matrix]; actual = [r['case']['case_id'] for r in rows]; all_ids = [r['case']['case_id'] for r in reports]
    if (assignment.get('status') != 'COMPLETE_WITH_RECORDED_OUTCOMES' or assignment.get('phase_id') != ctx.c['phase_id']
            or assignment.get('run_id') != ctx.c['run_id'] or assignment.get('rows') != 46 or assignment.get('reporting_rows') != 48
            or len(ids) != 46 or len(set(ids)) != 46 or len(actual) != 46 or set(actual) != set(ids)
            or len(all_ids) != 48 or len(set(all_ids)) != 48 or set(all_ids)-set(ids) != {'P_REFERENCE_open', 'P_REFERENCE_i95'}
            or any(not r.get('status') for r in rows+reports)):
        raise ValueError('Assignment matrix/disposition mismatch')
    if assignment.get('portfolio_sha256') != selection['portfolio']['sha256']:
        raise ValueError('Assignments used another portfolio')
    metrics = records['metrics']; result = records['metrics_result']; provenance = records['metrics_provenance']
    if any(item.get('assignment_revision')!=assignment_revision for item in (metrics,result,provenance)):
        raise ValueError('Exact metrics revision lineage required')
    if (metrics.get('status') != 'VERIFIED_REPORT_DATA' or metrics.get('reporting_rows') != 48
            or result.get('status') != 'PASS_WITH_EXPLICIT_CASE_DISPOSITIONS' or result.get('reporting_rows') != 48
            or result.get('computational_cases') != 46 or result.get('phase_id') != ctx.c['phase_id']
            or result.get('run_id') != ctx.c['run_id']): raise ValueError('Metrics producer terminal identity mismatch')
    for name in ('metrics_result', 'metrics_provenance'):
        if not any(Path(r['path']).resolve() == Path(supplied[name]['path']).resolve() and r['sha256'] == supplied[name]['sha256'] for r in metrics['files']):
            raise ValueError('Metrics output is not bound by its terminal manifest')
    for name in ('reporting', 'assignment'):
        if not any(Path(r['path']).resolve() == Path(supplied[name]['path']).resolve() and r['sha256'] == supplied[name]['sha256'] for r in provenance['bindings']):
            raise ValueError('Metrics provenance differs from exact assignment evidence')
    if provenance.get('portfolio_sha256') != selection['portfolio']['sha256']:
        raise ValueError('Metrics portfolio differs from frozen selection')
    return records


def validate_clock(ctx, record, assets, *, now=None):
    """Authenticate the already opened root clock, never create/reset one."""
    path = authenticate(record, canonical=Path(ctx.root)/'MANUSCRIPT_CLOCK.json'); clock = read(path)
    start, end = clock.get('start_epoch'), clock.get('deadline_epoch'); now = time.time() if now is None else now
    if (clock.get('schema') != 'RECOVERY24_MANUSCRIPT_CLOCK_V1' or clock.get('phase_id') != ctx.c['phase_id']
            or clock.get('run_id') != ctx.c['run_id'] or clock.get('campaign_config_sha256') != ctx.config_sha256
            or not all(finite(x) for x in (start, end, now)) or start < assets['ended_epoch']
            or start < ctx.c['original_start_epoch'] or end != min(start+10800., ctx.c['manuscript_outer_deadline_epoch'])
            or not start <= now < end): raise ValueError('Invalid, premature or expired manuscript clock')
    return end


def validate_admission(ctx, path, digest, action):
    admission_path = authenticate(dict(path=str(path), sha256=digest)); a = read(admission_path)
    tasks = ['timing', 'geometry', 'configuration', 'extract', 'render'] if action == 'assets' else ['assemble', 'compile']
    if (action not in {'assets', 'manuscript'} or a.get('schema') != SCHEMA or a.get('action') != action
            or a.get('phase_id') != ctx.c['phase_id'] or a.get('run_id') != ctx.c['run_id']
            or a.get('tasks') != tasks or not a.get('expected_outputs') or not a.get('acceptance_tests')
            or a.get('root_honors_publication_lock') is not True): raise ValueError('Explicit complete publication admission required')
    if authenticate(a['campaign_config']) != Path(ctx.path).resolve() or a['campaign_config']['sha256'] != ctx.config_sha256:
        raise ValueError('Driver campaign differs from Context')
    budget = a.get('budget_seconds'); end = a.get('deadline_epoch')
    if not finite(budget) or budget <= 0 or not finite(end): raise ValueError('Finite positive driver budget required')
    check_sources(a); terminal_inputs(ctx, a['terminal_bindings'],a.get('assignment_revision'))
    if action == 'assets':
        ctx.require_action('publication')
        if end > ctx.deadline('publication'): raise ValueError('Assets cannot use the manuscript extension')
    else:
        assets_path = authenticate(a['assets_result']); assets = read(assets_path)
        if (not assets_path.is_relative_to(Path(ctx.root)/'publication/driver') or assets_path.name != 'RESULT.json'
                or assets.get('status') != 'ASSETS_REPRODUCED_VISUAL_QA_PENDING' or assets.get('error') is not None
                or assets.get('campaign_config') != a['campaign_config'] or assets.get('terminal_bindings') != a['terminal_bindings']
                or assets.get('source_bindings') != a['source_bindings'] or assets.get('assignment_revision')!=a.get('assignment_revision')): raise ValueError('Exact successful assets driver result required')
        for item in assets['outputs'].values(): authenticate(item)
        if a['publication_config'] != assets['outputs']['publication_config']: raise ValueError('Manuscript changed the assets configuration')
        limit = validate_clock(ctx, a['manuscript_clock'], assets)
        ctx.require_action('manuscript')
        if end > min(limit, ctx.deadline('manuscript')): raise ValueError('Manuscript admission exceeds its existing clock')
    if time.time() >= end: raise TimeoutError('Driver admission expired')
    return a, dict(path=str(admission_path), sha256=digest)


def request_from_outputs(ctx, settings, producer_bindings, assignment_revision=None):
    """Derive input hashes only from the completed producers' output sections."""
    module = importlib.import_module('.publication_configuration', __package__)
    if (set(settings) != SETTINGS_FIELDS or settings['schema'] != 'RECOVERY24_PUBLICATION_SETTINGS_V1'
            or settings['phase_id'] != ctx.c['phase_id'] or settings['run_id'] != ctx.c['run_id']):
        raise ValueError('Explicit publication settings schema required')
    request = dict(settings); request['schema'] = module.SCHEMA
    request.update(campaign_config_sha256=ctx.config_sha256, producers=producer_bindings, inputs={},assignment_revision=assignment_revision)
    for name, (producer, filename) in DATASETS.items():
        p = authenticate(producer_bindings[producer]); record = read(p)
        if record.get('status') not in module.PRODUCER_STATUSES[producer]: raise ValueError('Nonterminal producer')
        expected = p.parent/filename
        matches = [(path, sha) for path, sha in module.producer_outputs(producer, record, p.parent) if Path(path) == expected]
        if len(matches) != 1: raise ValueError('Missing/ambiguous producer CSV: '+name)
        request['inputs'][name] = dict(path=matches[0][0], sha256=matches[0][1])
    return request


def static_prerequisites(ctx, settings, geometry, terminal_bindings):
    """Fail before append-only exports on missing tools/snapshots or a wrong plan."""
    module = importlib.import_module('.publication_configuration', __package__)
    if (set(settings) != SETTINGS_FIELDS or settings['schema'] != 'RECOVERY24_PUBLICATION_SETTINGS_V1'
            or settings['phase_id'] != ctx.c['phase_id'] or settings['run_id'] != ctx.c['run_id']):
        raise ValueError('Explicit publication settings schema required')
    if (not finite(settings['deadline_epoch']) or settings['deadline_epoch'] <= time.time()
            or settings['deadline_epoch'] > ctx.c['manuscript_outer_deadline_epoch']):
        raise ValueError('Settings deadline is outside original authority')
    budget=settings['resource_budget']
    if (set(budget) != {'per_asset_seconds','compile_seconds','pdf_render_seconds'}
            or any(not finite(x) or x<=0 for x in budget.values())): raise ValueError('Finite positive publication budgets required')
    authenticate(settings['environment'])
    authority_path=authenticate(settings['toolchain_authority']);authority=read(authority_path)
    files=module.recorded_bindings(authority,authority_path.parent)
    for name in ('compiler','pdftoppm'):
        path=authenticate(settings[name])
        if (str(path),settings[name]['sha256']) not in files:raise ValueError('Tool absent from exact authority')
    cache=Path(settings['compiler_cache_source']).resolve(strict=True)
    if (not cache.is_dir() or cache.is_relative_to(Path(ctx.root))
            or str(cache) not in module._directory_values(authority)):raise ValueError('Unbound external compiler cache')
    names=set()
    for name in ('baseline','dependencies'):
        section=settings[name];manifest=read(authenticate(section['manifest']));snapshot=Path(section['snapshot_root']).resolve(strict=True)
        if not snapshot.is_dir() or not manifest.get('members'):raise ValueError('Missing immutable manuscript snapshot')
        for row in manifest['members']:
            ctx.guard();relative=Path(row['relative_path']);path=(snapshot/relative).resolve(strict=True)
            if (relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(snapshot)
                    or relative.as_posix() in names):raise ValueError('Unsafe/colliding manuscript member')
            names.add(relative.as_posix());authenticate(dict(path=str(path),sha256=row['sha256']))
    if not {'main.tex','numerical example.tex'}.issubset(names):raise ValueError('Required v5 source members missing')
    plan=geometry['inputs']['ablation_plan'];path=authenticate(plan,canonical=Path(ctx.root)/'ablation/PLAN.json')
    frozen=read(authenticate(terminal_bindings['fit_freeze']))
    if plan['sha256']!=frozen.get('plan_sha256'):raise ValueError('Geometry plan differs from final ablation fit freeze')
    if len(read(path).get('regions',[]))!=16:raise ValueError('Exact 16-region ablation design required')
    for key in ('output_relative','publication_output_relative'):
        dest=ctx.output_path(settings[key])
        if not dest.is_relative_to(Path(ctx.root)/'publication') or dest.exists():raise ValueError('Unused publication destination required')


def call(stage, ctx, **kwargs):
    """Delayed imports: CLI admission checks precede scientific CSV/array reads."""
    if stage == 'timing': return importlib.import_module('.offline_timing', __package__).export(ctx, **kwargs)
    if stage == 'geometry': return importlib.import_module('.publication_geometry', __package__).export(ctx, **kwargs)
    if stage == 'configuration': return importlib.import_module('.publication_configuration', __package__).prepare(ctx, **kwargs)
    functions = {'extract': 'extract', 'render': 'render', 'assemble': 'assemble_v5', 'compile': 'compile_v5'}
    return getattr(importlib.import_module('.publication', __package__), functions[stage])(ctx, **kwargs)


class BoundedContext:
    """Propagate the admitted driver limit through existing nested API guards."""
    def __init__(self, context, end): self._context, self._end = context, end
    def __getattr__(self, name): return getattr(self._context, name)
    def deadline(self, action=None): return min(self._end, self._context.deadline(action))
    def guard(self, deadline=None):
        return self._context.guard(self._end if deadline is None else min(self._end, deadline))
    def require_action(self, action):
        result = self._context.require_action(action); self.guard(); return result


def run(ctx, admission_path, *, admission_sha256, action):
    a, admitted = validate_admission(ctx, admission_path, admission_sha256, action)
    root = Path(ctx.root); folder = ctx.output_path(a['output_relative'])
    if not folder.is_relative_to(root/'publication/driver') or folder == root/'publication/driver':
        raise ValueError('Dedicated driver output directory required')
    if folder.exists(): raise FileExistsError('Preserve every prior driver attempt; use a new directory')
    if (root/'NATIONAL_TIMING_ACTIVE.json').exists() or (root/'ADVANCE_CONTROLLER_LOCK.json').exists():
        raise PermissionError('Scientific stage/controller still active; publication must not overlap')
    lock = root/'PUBLICATION_DRIVER_LOCK.json'
    token = dict(pid=os.getpid(), action=action, admission=admitted, created_epoch=time.time())
    # An existing lock is evidence for root diagnosis, never automatically removed.
    new(lock, token)
    started = time.time(); end = min(a['deadline_epoch'], started+a['budget_seconds']); outputs = {}; steps = []; error = None
    bounded = BoundedContext(ctx, end)
    try:
        folder.mkdir(parents=True, exist_ok=False)
        new(folder/'INTENT.json', dict(admission=admitted, action=action, started_epoch=started, deadline_epoch=end,
            campaign_config=a['campaign_config'], tasks=a['tasks'], source_bindings=a['source_bindings'], scientific_execution=False))
        def guard():
            if (root/'NATIONAL_TIMING_ACTIVE.json').exists() or (root/'ADVANCE_CONTROLLER_LOCK.json').exists():
                raise PermissionError('Scientific activity overlaps publication')
            ctx.guard(end); check_sources(a)
            authenticate(admitted); authenticate(a['campaign_config'])
            for key in ('publication_settings', 'geometry_config', 'manuscript_clock', 'assets_result', 'assignment_revision'):
                if key in a and a[key] is not None: authenticate(a[key])
            for item in a['terminal_bindings'].values(): authenticate(item)
            for item in outputs.values(): authenticate(item)
        def step(name, **kwargs):
            guard(); tick=time.time(); value=call(name, bounded, **kwargs); guard()
            steps.append(dict(stage=name, started_epoch=tick, ended_epoch=time.time()))
            new(folder/(name+'_RETURN.json'), dict(stage=name, completed_epoch=time.time(),
                status=value.get('status') if isinstance(value, dict) else None))
            return value
        if action == 'assets':
            settings_path=authenticate(a['publication_settings']); settings=read(settings_path)
            geo_path=authenticate(a['geometry_config']); geo=read(geo_path)
            if geo.get('phase_id') != ctx.c['phase_id'] or geo.get('run_id') != ctx.c['run_id']:
                raise ValueError('Geometry configuration campaign mismatch')
            if geo['inputs']['reporting_results'] != a['terminal_bindings']['reporting']:
                raise ValueError('Geometry does not use the exact completed assignment reporting ledger')
            static_prerequisites(bounded,settings,geo,a['terminal_bindings'])
            layout=importlib.import_module('.assignment_reconciliation',__package__).resolve(bounded,a.get('assignment_revision'))
            timing_root=layout['timing_root']
            geometry_root=ctx.output_path(geo['output_relative'])
            if not geometry_root.is_relative_to(root/'publication') or geometry_root.exists() or timing_root.exists():
                raise ValueError('Timing/geometry output must be new; preserve any prior partial attempt')
            step('timing', deadline=end,assignment_revision=a.get('assignment_revision')); outputs['timing']=binding(timing_root/'MANIFEST.json')
            step('geometry', config_path=geo_path, deadline=end); outputs['geometry']=binding(geometry_root/'RESULT.json')
            producers={name:a['terminal_bindings'][name] for name in ('metrics','regional','ablation')}
            producers.update(timing=outputs['timing'],geometry=outputs['geometry'])
            request=request_from_outputs(ctx,settings,producers,a.get('assignment_revision')); request_path=folder/'PUBLICATION_REQUEST.json';new(request_path,request)
            outputs['request']=binding(request_path)
            result=step('configuration', request_path=request_path,request_sha256=outputs['request']['sha256'],deadline=end)
            outputs['publication_config']=result['publication_config'];config_path=authenticate(outputs['publication_config'])
            cfg=read(config_path);pub=ctx.output_path(cfg['output_relative'])
            step('extract',config_path=config_path);outputs['extraction']=binding(pub/'EXTRACTION_RESULT.json')
            step('render',config_path=config_path);outputs['reproduction']=binding(pub/'REPRODUCTION_RESULT.json')
            if read(outputs['extraction']['path']).get('status')!='EXTRACTED_RENDER_PENDING' or read(outputs['reproduction']['path']).get('status')!='REPRODUCED_VISUAL_QA_PENDING':
                raise ValueError('Publication producer did not complete the expected asset stages')
            authenticate(a['publication_settings']);authenticate(a['geometry_config'])
        else:
            config_path=authenticate(a['publication_config']);outputs['publication_config']=a['publication_config']
            cfg=read(config_path);pub=ctx.output_path(cfg['output_relative'])
            step('assemble',config_path=config_path);outputs['manuscript']=binding(pub/'MANUSCRIPT_RESULT.json')
            step('compile',config_path=config_path);outputs['pdf_build']=binding(pub/'PDF_BUILD_RESULT.json')
            build=read(outputs['pdf_build']['path'])
            if build.get('status')!='COMPILED_EVERY_PAGE_VISUAL_QA_PENDING':raise ValueError('Unexpected PDF build disposition')
            pages=build.get('pages');images=build.get('page_images',[])
            if (type(pages) is not int or pages<=0 or len(images)!=pages
                    or len({str(Path(p['path']).resolve()) for p in images})!=pages
                    or build.get('source_preservation')!='PASS_NUMERICAL_ONLY_SOURCE_PRESERVATION'):
                raise ValueError('Complete nonempty rendered PDF and numerical-only preservation required')
            authenticate(dict(path=build['pdf_path'],sha256=build['pdf_sha256']))
            for page in images:authenticate(page)
            authenticate(a['manuscript_clock']);authenticate(a['assets_result'])
        guard()
    except Exception as exc:
        error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        if not folder.exists(): folder.mkdir(parents=True,exist_ok=False)
    result=dict(schema='RECOVERY24_PUBLICATION_DRIVER_RESULT_V1', status='FAILED_PRESERVED_OUTPUTS' if error else
        ('ASSETS_REPRODUCED_VISUAL_QA_PENDING' if action=='assets' else 'MANUSCRIPT_COMPILED_VISUAL_QA_PENDING'),
        action=action,admission=admitted,campaign_config=a['campaign_config'],terminal_bindings=a['terminal_bindings'],
        source_bindings=a['source_bindings'],started_epoch=started,ended_epoch=time.time(),deadline_epoch=end,
        steps=steps,outputs=outputs,error=error,assignment_revision=a.get('assignment_revision'),actual_visual_qa=False,external_acceptance=False,clock_created=False)
    new(folder/'RESULT.json',result)
    if read(lock)!=token:raise RuntimeError('Publication lock changed; preserve foreign evidence')
    lock.rename(folder/'PUBLICATION_DRIVER_LOCK_CLOSED.json')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--admission',required=True)
    parser.add_argument('--admission-sha256',required=True);parser.add_argument('--action',choices=['assets','manuscript'],required=True)
    args=parser.parse_args(argv)
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    from .context import RecoveryContext
    ctx=RecoveryContext(args.config,step_id='publication' if args.action=='assets' else 'manuscript')
    result=run(ctx,args.admission,admission_sha256=args.admission_sha256,action=args.action)
    print(json.dumps(dict(status=result['status'],error=result['error'])))
    if result['error']:raise SystemExit(1)


if __name__=='__main__':main()
