"""Additive publication continuation: reuse sealed timing, restart geometry only.

No old driver/exporter/revision source is changed. Installation and real execution
require root's explicit admission. No producer runs on import. The existing
manuscript action retains its separate original clock through the old driver.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import time
import traceback

from . import publication_driver as legacy

SCHEMA = 'RECOVERY24_PUBLICATION_CONTINUATION_ADMISSION_V2'
TASKS = ['reuse_verified_timing', 'geometry', 'configuration', 'extract', 'render']
TIMING_NAMES = {'family_training_timing.csv', 'methods_with_training_timing.csv',
    'PROVENANCE.json', 'RESULT.json', 'timing_observations.csv', 'TRAINING_TASK_RECONCILIATION.json'}
OUTPUTS = {'driver': 'publication/driver/assets_v2', 'geometry': 'publication/geometry_v2',
    'configuration': 'publication/config_v2', 'publication': 'publication/final_v2'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def source_paths():
    here = Path(legacy.__file__).resolve().parent
    return sorted(set(legacy.source_paths() + [Path(__file__).resolve(), here/'publication_geometry_v2.py']))


def check_sources(a):
    require({str(p.resolve()) for p in source_paths()}.issubset(a['source_bindings']),
        'Continuation source closure incomplete')
    legacy.check_sources(a)


def reuse_timing(ctx, a):
    """Authenticate completed bytes, not the 84k original timing-input receipts."""
    root = Path(ctx.root)
    prior_path = legacy.authenticate(a['prior_assets_result'], canonical=root/'publication/driver/assets_v1/RESULT.json')
    prior = legacy.read(prior_path)
    require(prior.get('schema') == 'RECOVERY24_PUBLICATION_DRIVER_RESULT_V1'
        and prior.get('status') == 'FAILED_PRESERVED_OUTPUTS' and prior.get('action') == 'assets'
        and prior.get('error', {}).get('type') == 'ValueError'
        and prior['error'].get('message') == 'Missing or invalid native longitude/latitude'
        and [s.get('stage') for s in prior.get('steps', [])] == ['timing']
        and set(prior.get('outputs', {})) == {'timing'}, 'Exact closed geometry-failure predecessor required')
    for key in ('campaign_config', 'terminal_bindings', 'assignment_revision'):
        require(prior.get(key) == a.get(key), 'Continuation changed predecessor '+key)
    require(all(a['source_bindings'].get(p) == sha for p, sha in prior['source_bindings'].items()),
        'A previously pinned source changed')
    prior_admission = legacy.read(legacy.authenticate(prior['admission']))
    require(prior_admission['source_bindings'] == prior['source_bindings']
        and prior_admission['terminal_bindings'] == prior['terminal_bindings']
        and prior_admission['campaign_config'] == prior['campaign_config']
        and prior_admission.get('assignment_revision') == prior.get('assignment_revision'),
        'Prior admission/result authority differs')
    closed = legacy.read(legacy.authenticate(a['prior_closed_lock'],
        canonical=prior_path.parent/'PUBLICATION_DRIVER_LOCK_CLOSED.json'))
    require(closed.get('admission') == prior['admission'] and closed.get('action') == 'assets'
        and type(closed.get('pid')) is int and closed['pid'] > 0, 'Prior driver lock closure differs')
    require(a['timing_reuse']['manifest'] == prior['outputs']['timing'], 'Timing was not completed by predecessor')
    resolver = importlib.import_module('.assignment_reconciliation', __package__)
    layout = resolver.resolve(ctx, a['assignment_revision'])
    timing_root = Path(layout['timing_root']).resolve()
    manifest_path = legacy.authenticate(a['timing_reuse']['manifest'], canonical=timing_root/'MANIFEST.json')
    manifest = legacy.read(manifest_path)
    require(manifest.get('schema') == 'RECOVERY24_OFFLINE_TIMING_V1'
        and manifest.get('status') == 'VERIFIED_TIMING_EXPORT'
        and manifest.get('assignment_revision') == a['assignment_revision'], 'Unverified or wrong timing revision')
    files = manifest.get('files', [])
    require(len(files) == len(TIMING_NAMES) and {Path(x['path']).name for x in files} == TIMING_NAMES,
        'Incomplete or duplicate timing output inventory')
    by_name = {}
    for item in files:
        ctx.guard()
        path = legacy.authenticate(item, canonical=timing_root/Path(item['path']).name)
        require(type(item.get('bytes')) is int and path.stat().st_size == item['bytes'], 'Timing size differs')
        by_name[path.name] = item
    for name, filename in (('result', 'RESULT.json'), ('provenance', 'PROVENANCE.json')):
        require(a['timing_reuse'][name] == by_name[filename], 'Explicit timing '+name+' binding differs')
    result = legacy.read(by_name['RESULT.json']['path'])
    require(result.get('schema') == 'RECOVERY24_OFFLINE_TIMING_V1'
        and result.get('status') == 'EXPORTED_WITH_EXPLICIT_TIMING_SCOPE_AND_MISSINGNESS'
        and result.get('method_rows') == 48 and result.get('observation_rows', 0) > 0,
        'Empty/nonterminal timing output')
    # This reads one already sealed provenance JSON, never opens its input list.
    proof = legacy.read(by_name['PROVENANCE.json']['path'])
    timing_source = str(Path(legacy.__file__).resolve().with_name('offline_timing.py'))
    require(proof.get('schema') == 'RECOVERY24_OFFLINE_TIMING_V1'
        and proof.get('source_sha256') == a['source_bindings'][timing_source]
        and proof.get('config_sha256') == ctx.config_sha256
        and proof.get('assignment_revision') == a['assignment_revision']
        and proof.get('original_payload_writes') is False
        and proof.get('model_or_label_array_reads') is False,
        'Timing provenance authority differs')
    del proof
    ctx.guard()
    return dict(manifest=a['timing_reuse']['manifest'], result=a['timing_reuse']['result'],
        provenance=a['timing_reuse']['provenance'], files=files, predecessor=a['prior_assets_result'],
        disposition='REUSED_EXACT_VERIFIED_OUTPUTS_NO_TIMING_OR_METRICS_RERUN',
        source_input_receipts_replayed=False)


def validate_admission(ctx, path, digest):
    admitted_path = legacy.authenticate(dict(path=str(path), sha256=digest))
    a = legacy.read(admitted_path)
    require(a.get('schema') == SCHEMA and a.get('action') == 'assets'
        and a.get('phase_id') == ctx.c['phase_id'] and a.get('run_id') == ctx.c['run_id']
        and a.get('tasks') == TASKS and a.get('expected_outputs') and a.get('acceptance_tests')
        and a.get('root_honors_publication_lock') is True, 'Explicit continuation admission required')
    require(legacy.authenticate(a['campaign_config']) == Path(ctx.path).resolve()
        and a['campaign_config']['sha256'] == ctx.config_sha256, 'Wrong current campaign')
    ctx.require_action('publication')
    require(legacy.finite(a.get('budget_seconds')) and 0 < a['budget_seconds'] <= 3600
        and legacy.finite(a.get('deadline_epoch')) and time.time() < a['deadline_epoch'] <= ctx.deadline('publication'),
        'Continuation exceeds original publication deadline/budget')
    require(a.get('output_relative') == OUTPUTS['driver'], 'New fixed continuation namespace required')
    check_sources(a)
    legacy.terminal_inputs(ctx, a['terminal_bindings'], a['assignment_revision'])
    return a, dict(path=str(admitted_path), sha256=digest)


def call(stage, ctx, **kwargs):
    if stage == 'geometry':
        return importlib.import_module('.publication_geometry_v2', __package__).export(ctx, **kwargs)
    require(stage in {'configuration', 'extract', 'render'}, 'Continuation cannot dispatch timing, metrics, or science')
    return legacy.call(stage, ctx, **kwargs)


def run(ctx, admission_path, *, admission_sha256, action='assets'):
    if action == 'manuscript':
        # Unchanged manuscript validation authenticates the successful assets V2
        # record, identical full source map, and existing original three-hour clock.
        a = legacy.read(legacy.authenticate(dict(path=str(admission_path), sha256=admission_sha256)))
        check_sources(a)
        return legacy.run(ctx, admission_path, admission_sha256=admission_sha256, action=action)
    require(action == 'assets', 'Unknown continuation action')
    a, admitted = validate_admission(ctx, admission_path, admission_sha256)
    root = Path(ctx.root); folder = ctx.output_path(a['output_relative'])
    require(not folder.exists(), 'Preserve prior continuation attempt')
    require(not any((root/name).exists() for name in ('NATIONAL_TIMING_ACTIVE.json', 'ADVANCE_CONTROLLER_LOCK.json')),
        'Science/controller still active')
    lock = root/'PUBLICATION_DRIVER_LOCK.json'
    token = dict(pid=os.getpid(), action='assets', admission=admitted, created_epoch=time.time())
    legacy.new(lock, token)
    started = time.time(); end = min(a['deadline_epoch'], started+a['budget_seconds'])
    bounded = legacy.BoundedContext(ctx, end); outputs = {}; steps = []; error = None; reuse = None
    try:
        folder.mkdir(parents=True, exist_ok=False)
        legacy.new(folder/'INTENT.json', dict(admission=admitted, action='assets', started_epoch=started,
            deadline_epoch=end, campaign_config=a['campaign_config'], tasks=TASKS,
            source_bindings=a['source_bindings'], scientific_execution=False, timing_rerun=False, metrics_rerun=False))
        def guard():
            require(not any((root/name).exists() for name in ('NATIONAL_TIMING_ACTIVE.json', 'ADVANCE_CONTROLLER_LOCK.json')),
                'Scientific activity overlaps publication')
            bounded.guard(); check_sources(a); legacy.authenticate(admitted); legacy.authenticate(a['campaign_config'])
            for key in ('prior_assets_result', 'prior_closed_lock', 'publication_settings', 'geometry_config', 'assignment_revision'):
                legacy.authenticate(a[key])
            for item in a['terminal_bindings'].values(): legacy.authenticate(item)
            for item in a['timing_reuse'].values(): legacy.authenticate(item)
            for item in outputs.values(): legacy.authenticate(item)
        def step(name, **kwargs):
            guard(); tick = time.time(); value = call(name, bounded, **kwargs); guard()
            steps.append(dict(stage=name, started_epoch=tick, ended_epoch=time.time()))
            legacy.new(folder/(name+'_RETURN.json'), dict(stage=name, completed_epoch=time.time(),
                status=value.get('status') if isinstance(value, dict) else None))
            return value
        guard()
        settings = legacy.read(legacy.authenticate(a['publication_settings']))
        geo_path = legacy.authenticate(a['geometry_config']); geo = legacy.read(geo_path)
        require(settings['output_relative'] == OUTPUTS['configuration']
            and settings['publication_output_relative'] == OUTPUTS['publication']
            and geo['output_relative'] == OUTPUTS['geometry'], 'Only new fixed geometry/config/publication namespaces admitted')
        require(geo.get('phase_id') == ctx.c['phase_id'] and geo.get('run_id') == ctx.c['run_id']
            and geo['inputs']['reporting_results'] == a['terminal_bindings']['reporting'], 'Geometry authority mismatch')
        legacy.static_prerequisites(bounded, settings, geo, a['terminal_bindings'])
        geometry_root = ctx.output_path(geo['output_relative'])
        require(not geometry_root.exists(), 'Preserve failed geometry output')
        tick = time.time(); reuse = reuse_timing(bounded, a)
        legacy.new(folder/'TIMING_REUSE.json', reuse)
        steps.append(dict(stage='reuse_verified_timing', started_epoch=tick, ended_epoch=time.time(), rerun=False))
        outputs['timing'] = reuse['manifest']
        step('geometry', config_path=geo_path, deadline=end)
        outputs['geometry'] = legacy.binding(geometry_root/'RESULT.json')
        producers = {name:a['terminal_bindings'][name] for name in ('metrics', 'regional', 'ablation')}
        producers.update(timing=outputs['timing'], geometry=outputs['geometry'])
        request = legacy.request_from_outputs(bounded, settings, producers, a['assignment_revision'])
        request_path = folder/'PUBLICATION_REQUEST.json'; legacy.new(request_path, request)
        outputs['request'] = legacy.binding(request_path)
        config_result = step('configuration', request_path=request_path, request_sha256=outputs['request']['sha256'], deadline=end)
        outputs['publication_config'] = config_result['publication_config']
        config_path = legacy.authenticate(outputs['publication_config']); cfg = legacy.read(config_path)
        pub = ctx.output_path(cfg['output_relative'])
        require(cfg['output_relative'] == OUTPUTS['publication'], 'Configuration changed publication destination')
        step('extract', config_path=config_path); outputs['extraction'] = legacy.binding(pub/'EXTRACTION_RESULT.json')
        step('render', config_path=config_path); outputs['reproduction'] = legacy.binding(pub/'REPRODUCTION_RESULT.json')
        require(legacy.read(outputs['extraction']['path']).get('status') == 'EXTRACTED_RENDER_PENDING'
            and legacy.read(outputs['reproduction']['path']).get('status') == 'REPRODUCED_VISUAL_QA_PENDING',
            'Asset producers did not reach terminal statuses')
        for item in reuse['files']: bounded.guard(); legacy.authenticate(item)
        guard()
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if not folder.exists(): folder.mkdir(parents=True, exist_ok=False)
    result = dict(schema='RECOVERY24_PUBLICATION_DRIVER_RESULT_V2', status='FAILED_PRESERVED_OUTPUTS' if error else
        'ASSETS_REPRODUCED_VISUAL_QA_PENDING', action='assets', admission=admitted,
        campaign_config=a['campaign_config'], terminal_bindings=a['terminal_bindings'], source_bindings=a['source_bindings'],
        started_epoch=started, ended_epoch=time.time(), deadline_epoch=end, steps=steps, outputs=outputs, error=error,
        assignment_revision=a['assignment_revision'], prior_assets_result=a['prior_assets_result'], timing_reuse=reuse,
        timing_rerun=False, metrics_rerun=False, actual_visual_qa=False, external_acceptance=False, clock_created=False)
    legacy.new(folder/'RESULT.json', result)
    require(legacy.read(lock) == token, 'Foreign publication lock; preserve it')
    lock.rename(folder/'PUBLICATION_DRIVER_LOCK_CLOSED.json')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--admission', required=True)
    parser.add_argument('--admission-sha256', required=True)
    parser.add_argument('--action', choices=['assets', 'manuscript'], required=True)
    args = parser.parse_args(argv)
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[name] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    from .context import RecoveryContext
    ctx = RecoveryContext(args.config, step_id='publication' if args.action == 'assets' else 'manuscript')
    result = run(ctx, args.admission, admission_sha256=args.admission_sha256, action=args.action)
    print(json.dumps(dict(status=result['status'], error=result['error'])))
    if result['error']: raise SystemExit(1)


if __name__ == '__main__':
    main()
