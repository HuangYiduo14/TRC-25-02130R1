"""Explicit, supervised execution of the declared reproduction stage list."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)

def resolved(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

def footprint(paths):
    total = 0
    for root in paths:
        if not root.exists():
            continue
        for current, _, files in os.walk(root):
            for filename in files:
                try:
                    total += (Path(current) / filename).stat().st_size
                except FileNotFoundError:
                    pass
    return total

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-id', required=True)
    task = parser.add_mutually_exclusive_group(required=True)
    task.add_argument('--stage')
    task.add_argument('--all', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Explicitly emit the command plan without computation')
    args = parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_id):
        parser.error('run-id must be a nonempty path-safe identifier')
    config_path = resolved(args.config)
    cfg = json.loads(config_path.read_text(encoding='utf-8'))
    for key in ('phase_id', 'input_versions', 'tasks', 'resource_budget'):
        if not cfg.get(key):
            raise ValueError(f'Nonempty {key} is required')
    stages = cfg['tasks']
    stages = [dict(s, args=[x.replace('{run_id}', args.run_id) for x in s['args']],
                   expected_outputs=[x.replace('{run_id}', args.run_id) for x in s['expected_outputs']]) for s in stages]
    names = [s['step_id'] for s in stages]
    if len(names) != len(set(names)) or not stages:
        raise ValueError('Unique nonempty steps are required')
    if args.stage and args.stage not in names:
        parser.error(f'Unknown stage. Choose one of: {", ".join(names)}')
    selected = stages if args.all else [s for s in stages if s['step_id'] == args.stage]
    plans = []
    bindings = {str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path): digest(config_path)}
    # Bind imported numerical modules, adapter helpers and frozen mapping decisions,
    # not just the stage entrypoints. Runtime outputs are deliberately excluded.
    for directory, pattern in [('src', '*.py'), ('scripts', '*.py'), ('configs', '*.json'), ('metadata', '*')]:
        for bound_path in sorted((ROOT / directory).rglob(pattern)):
            if bound_path.is_file() and '__pycache__' not in bound_path.parts:
                bindings[bound_path.relative_to(ROOT).as_posix()] = digest(bound_path)
    for stage in stages:
        script_path = resolved(stage['script'])
        bindings[stage['script']] = digest(script_path)
        if '--config' in stage['args']:
            child_config = stage['args'][stage['args'].index('--config')+1]
            bindings[child_config] = digest(resolved(child_config))
    for stage in selected:
        if not stage.get('expected_outputs') or not stage.get('acceptance_tests'):
            raise ValueError(f'Incomplete contract for {stage["step_id"]}')
        command = [sys.executable, '-X', 'utf8', '-B', str(resolved(stage['script']))] + stage['args']
        plans.append({'step_id': stage['step_id'], 'command': command,
                      'expected_outputs': stage['expected_outputs'], 'acceptance_tests': stage['acceptance_tests']})
    if args.dry_run:
        print(json.dumps({'status': 'EXPLICIT_DRY_RUN', 'phase_id': cfg['phase_id'], 'run_id': args.run_id,
                          'config_sha256': digest(config_path), 'tasks': plans}, indent=2))
        return 0
    import psutil
    receipts = ROOT / 'runs' / args.run_id
    receipts.mkdir(parents=True, exist_ok=True)
    state_path = receipts / 'CAMPAIGN.json'
    if state_path.exists():
        campaign = json.loads(state_path.read_text(encoding='utf-8'))
        if campaign['config_sha256'] != digest(config_path) or campaign.get('execution_bindings') != bindings:
            raise ValueError('Code/config/metadata changed for this run-id; preserve this run and choose a new workspace/run-id')
    else:
        campaign = {'phase_id': cfg['phase_id'], 'run_id': args.run_id, 'start_epoch': time.time(),
                    'config_sha256': digest(config_path), 'input_versions': cfg['input_versions'],
                    'execution_bindings': bindings,
                    'resource_budget': cfg['resource_budget'], 'completed_steps': []}
        write_json(state_path, campaign)
    budget = cfg['resource_budget']
    deadline = campaign['start_epoch'] + budget['wall_seconds_max']
    monitored = [ROOT / p for p in cfg.get('monitored_output_roots', ['data/processed', 'runs'])]
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(ROOT / 'src' / 'partC_rev1_src'), str(ROOT / 'scripts')])
    env['REPRODUCTION_RUN_ID'] = args.run_id
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    for stage, plan in zip(selected, plans):
        stage_id = stage['step_id']
        out = receipts / stage_id
        if (out / 'RESULT.json').exists():
            raise FileExistsError(f'Stage receipt exists: {out}. Select remaining stages explicitly; preserve prior results.')
        for dependency in stage.get('depends_on', []):
            previous = receipts / dependency / 'RESULT.json'
            if not previous.exists() or json.loads(previous.read_text(encoding='utf-8'))['status'] != 'PASS':
                raise RuntimeError(f'{stage_id} requires successful stage {dependency} under this run-id')
        for required in stage.get('required_inputs', []):
            if not resolved(required).exists():
                raise FileNotFoundError(f'Missing declared input: {required}')
        if time.time() >= deadline:
            raise TimeoutError('Whole-campaign wall budget has expired')
        out.mkdir(parents=True, exist_ok=False)
        started = time.time()
        record = {'phase_id': cfg['phase_id'], 'step_id': stage_id, 'run_id': args.run_id,
                  'config': str(config_path), 'config_sha256': digest(config_path),
                  'input_versions': cfg['input_versions'], 'tasks': [plan['command']],
                  'expected_outputs': stage['expected_outputs'], 'acceptance_tests': stage['acceptance_tests'],
                  'resource_budget': budget, 'started_epoch': started, 'status': 'RUNNING'}
        write_json(out / 'EXECUTION.json', record)
        reason = None
        peak_rss = 0
        min_free = psutil.virtual_memory().available
        peak_disk = footprint(monitored)
        next_disk = started
        with (out / 'stdout.log').open('w', encoding='utf-8') as stdout, (out / 'stderr.log').open('w', encoding='utf-8') as stderr:
            proc = subprocess.Popen(plan['command'], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
            parent = psutil.Process(proc.pid)
            while proc.poll() is None:
                now = time.time()
                try:
                    tree = [parent] + parent.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in tree if p.is_running())
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    rss = 0
                peak_rss = max(peak_rss, rss)
                free = psutil.virtual_memory().available
                min_free = min(min_free, free)
                if now >= next_disk:
                    peak_disk = max(peak_disk, footprint(monitored))
                    next_disk = now + 15
                if now >= deadline or now-started >= stage.get('wall_seconds_max', budget['wall_seconds_max']):
                    reason = 'TIME_LIMIT'
                elif rss > budget['max_process_tree_rss_bytes'] or free < budget['min_available_ram_bytes']:
                    reason = 'RAM_LIMIT'
                elif peak_disk > budget['max_generated_output_bytes']:
                    reason = 'DISK_LIMIT'
                if reason:
                    try:
                        for child in reversed(parent.children(recursive=True)):
                            child.terminate()
                        parent.terminate()
                    except psutil.NoSuchProcess:
                        pass
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    break
                time.sleep(0.5)
        outputs = []
        for pattern in stage['expected_outputs']:
            matches = sorted(ROOT.glob(pattern))
            if not matches:
                reason = reason or f'MISSING_EXPECTED_OUTPUT:{pattern}'
            for file in matches:
                if file.is_file():
                    if file.stat().st_size == 0:
                        reason = reason or f'EMPTY_OUTPUT:{file.relative_to(ROOT)}'
                    outputs.append({'path': str(file.relative_to(ROOT)), 'bytes': file.stat().st_size,
                                    'sha256': digest(file)})
        record.update(status='PASS' if proc.returncode == 0 and not reason and outputs else 'FAILED',
                      exit_code=proc.returncode, failure=reason, elapsed_seconds=time.time()-started,
                      ended_utc=datetime.now(timezone.utc).isoformat(), artifacts=outputs,
                      resource_use={'peak_sampled_tree_rss_bytes': peak_rss, 'minimum_available_ram_bytes': min_free,
                                    'peak_generated_output_bytes': peak_disk},
                      next_step_authorized=False)
        write_json(out / 'RESULT.json', record)
        print(json.dumps({'stage': stage_id, 'status': record['status'], 'receipt': str(out / 'RESULT.json')}), flush=True)
        if record['status'] != 'PASS':
            return 1
        campaign['completed_steps'].append(stage_id)
        write_json(state_path, campaign)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
