"""Tests of execution safety and missing-stage detection, without FAF data."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import pytest

SOURCE = Path(__file__).resolve().parents[1]

def fixture(tmp_path):
    (tmp_path / 'scripts').mkdir()
    shutil.copyfile(SOURCE / 'scripts' / 'run_pipeline.py', tmp_path / 'scripts' / 'run_pipeline.py')
    (tmp_path / 'scripts' / 'worker.py').write_text(
        'from pathlib import Path\nimport sys\np=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True);p.write_text("meaningful fixture output")\n', encoding='utf-8')
    cfg = {'phase_id': 'REPRODUCTION_TEST', 'input_versions': {'fixture': 'directed-graph-fixture-v1'},
           'resource_budget': {'wall_seconds_max': 30, 'max_process_tree_rss_bytes': 1073741824,
                               'min_available_ram_bytes': 1, 'max_generated_output_bytes': 1000000},
           'monitored_output_roots': ['data', 'runs'],
           'tasks': [
               {'step_id': 'one', 'script': 'scripts/worker.py', 'args': ['data/one.txt'],
                'expected_outputs': ['data/one.txt'], 'acceptance_tests': ['nonempty output']},
               {'step_id': 'two', 'script': 'scripts/worker.py', 'args': ['data/two.txt'], 'depends_on': ['one'],
                'expected_outputs': ['data/two.txt'], 'acceptance_tests': ['prerequisite receipt and nonempty output']}]
           }
    (tmp_path / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    return cfg

def invoke(root, *args):
    return subprocess.run([sys.executable, '-B', str(root / 'scripts' / 'run_pipeline.py'),
                           '--config', 'config.json', '--run-id', 'fixture_v1', *args],
                          cwd=root, capture_output=True, text=True, timeout=40)

def test_end_to_end_and_receipts(tmp_path):
    fixture(tmp_path)
    r = invoke(tmp_path, '--all')
    assert r.returncode == 0, r.stderr
    for step in ['one', 'two']:
        receipt = json.loads((tmp_path / 'runs' / 'fixture_v1' / step / 'RESULT.json').read_text())
        assert receipt['status'] == 'PASS'
        assert receipt['artifacts'][0]['sha256']
        assert receipt['resource_use']['peak_sampled_tree_rss_bytes'] > 0
        assert receipt['tasks']
    assert invoke(tmp_path, '--all').returncode != 0

def test_dependency_cannot_be_skipped(tmp_path):
    fixture(tmp_path)
    r = invoke(tmp_path, '--stage', 'two')
    assert r.returncode != 0
    assert 'requires successful stage one' in r.stderr
    assert not (tmp_path / 'data' / 'two.txt').exists()

def test_nonempty_output_is_required(tmp_path):
    cfg = fixture(tmp_path)
    cfg['tasks'][0]['expected_outputs'] = ['data/absent.txt']
    (tmp_path / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    r = invoke(tmp_path, '--stage', 'one')
    assert r.returncode != 0
    receipt = json.loads((tmp_path / 'runs' / 'fixture_v1' / 'one' / 'RESULT.json').read_text())
    assert receipt['status'] == 'FAILED'
    assert 'MISSING_EXPECTED_OUTPUT' in receipt['failure']

def test_dry_run_is_explicit_and_has_no_outputs(tmp_path):
    fixture(tmp_path)
    r = invoke(tmp_path, '--all', '--dry-run')
    assert r.returncode == 0
    assert json.loads(r.stdout)['status'] == 'EXPLICIT_DRY_RUN'
    assert not (tmp_path / 'runs').exists()
    assert not (tmp_path / 'data').exists()

def test_empty_task_list_fails(tmp_path):
    cfg = fixture(tmp_path)
    cfg['tasks'] = []
    (tmp_path / 'config.json').write_text(json.dumps(cfg), encoding='utf-8')
    assert invoke(tmp_path, '--all').returncode != 0

@pytest.mark.parametrize('relative', ['src/numerical.py', 'scripts/imported_helper.py', 'metadata/frozen_plan.json'])
def test_imported_source_and_metadata_changes_invalidate_run(tmp_path, relative):
    fixture(tmp_path)
    bound = tmp_path / relative
    bound.parent.mkdir(parents=True, exist_ok=True)
    bound.write_text('{}' if bound.suffix == '.json' else 'VALUE = 1\n', encoding='utf-8')
    assert invoke(tmp_path, '--stage', 'one').returncode == 0
    bound.write_text('{"changed":true}' if bound.suffix == '.json' else 'VALUE = 2\n', encoding='utf-8')
    result = invoke(tmp_path, '--stage', 'two')
    assert result.returncode != 0
    assert 'Code/config/metadata changed' in result.stderr
    assert not (tmp_path / 'data' / 'two.txt').exists()
