"""A preflight step name must never authorize a physical pilot or new key freeze."""
from types import SimpleNamespace
import sys
import pytest


def rejecting_context(tmp_path):
    seen = []
    def require(action):
        seen.append(action)
        if action not in {'preflight', 'inference', 'assignments', 'analysis', 'publication', 'handoff'}:
            raise ValueError('Continuation action not authorized: ' + action)
    def guard(*args, **kwargs):
        seen.append('guard')
    return SimpleNamespace(c={'continuation': True}, root=tmp_path, require_action=require, guard=guard), seen


def test_reference_pilot_alias_rejected_before_guard_or_dispatch(tmp_path, monkeypatch):
    from partC_rev1_src.pacnhn_v1 import cli
    ctx, seen = rejecting_context(tmp_path)
    monkeypatch.setattr(cli, 'Context', lambda *a, **k: ctx)
    monkeypatch.setattr(sys, 'argv', ['cli', '--config', 'unused', '--action', 'pilot', '--scenario', 'open'])
    monkeypatch.setattr(cli, 'reference', lambda *a, **k: pytest.fail('Unauthorized numerical pilot dispatched'))
    with pytest.raises(ValueError, match='not authorized'):
        cli.main()
    assert seen == ['pilot']
    assert not (tmp_path / 'processes').exists()


def test_actual_preflight_can_run_through_same_step_alias(tmp_path, monkeypatch):
    from partC_rev1_src.pacnhn_v1 import cli
    ctx, seen = rejecting_context(tmp_path)
    monkeypatch.setattr(cli, 'Context', lambda *a, **k: ctx)
    monkeypatch.setattr(sys, 'argv', ['cli', '--config', 'unused', '--action', 'preflight'])
    monkeypatch.setattr(cli, 'preflight', lambda *a: seen.append('preflight_dispatched'))
    cli.main()
    assert seen == ['preflight', 'guard', 'preflight_dispatched']


@pytest.mark.parametrize('component', ['reference', 'support', 'training', 'pilot'])
def test_component_cannot_bypass_scope_with_permitted_context(tmp_path, component):
    from partC_rev1_src.pacnhn_v1.gates import require_component
    ctx, seen = rejecting_context(tmp_path)
    with pytest.raises(ValueError, match='not authorized'):
        require_component(ctx, component)
    assert seen == [component]


def test_direct_key_freeze_rejected_before_reference_access(tmp_path):
    from partC_rev1_src.pacnhn_v1.analysis_export import freeze_reference_keys
    ctx, seen = rejecting_context(tmp_path)
    with pytest.raises(ValueError, match='not authorized'):
        freeze_reference_keys(ctx)
    assert seen == ['freeze-keys']
