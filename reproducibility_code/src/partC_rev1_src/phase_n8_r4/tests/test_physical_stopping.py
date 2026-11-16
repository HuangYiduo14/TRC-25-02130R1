"""Compare unchanged numerical updates; test newly required fresh stopping."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from partC_rev1_src.phase4.solver import Phase4SolverConfig as OldConfig,solve_assignment as old_solve
from partC_rev1_src.phase_n8_r4.physical_solver import Phase4SolverConfig,solve_assignment

spec=importlib.util.spec_from_file_location('phase4_existing_fixtures',Path(__file__).resolve().parents[3]/'tests/test_phase4_preflight.py')
fixtures=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixtures)


@pytest.mark.parametrize('method',['MSA','FW','CFW'])
def test_exact_existing_updates(method):
    graph=fixtures._parallel_graph();od=fixtures._parallel_od()
    old=old_solve(graph,od,trace_maps=[],config=OldConfig(method=method,capacity_multiplier=.3,max_iterations=4))
    new=solve_assignment(graph,od,trace_maps=[],config=Phase4SolverConfig(method=method,capacity_multiplier=.3,max_iterations=4))
    np.testing.assert_array_equal(new.flow,old.flow)
    np.testing.assert_array_equal(new.cost,old.cost)
    assert new.final_metrics==old.final_metrics


def test_fresh_gap_stops_without_checkpoint_schedule():
    graph=fixtures._parallel_graph();od=fixtures._parallel_od()
    result=solve_assignment(graph,od,trace_maps=[],config=Phase4SolverConfig(method='FW',capacity_multiplier=.3,max_iterations=100,gap_target=.001))
    assert result.stop_reason=='refreshed_gap_target_met'
    assert result.final_metrics.relative_gap_sptt<=.001
    assert len(result.histories)<100
    assert result.latest_aon.sptt==result.final_metrics.sptt


def test_time_reserve_keeps_fresh_state():
    graph=fixtures._parallel_graph();od=fixtures._parallel_od()
    result=solve_assignment(graph,od,trace_maps=[],config=Phase4SolverConfig(method='FW',capacity_multiplier=.3,max_iterations=100,max_runtime_seconds=1e-8))
    assert result.stop_reason=='runtime_limit_with_fresh_current_state'
    assert result.latest_aon.sptt==result.final_metrics.sptt
    assert np.isfinite(result.flow).all() and (result.flow>=0).all()


def test_certified_state_survives_before_next_update(tmp_path):
    import json
    from partC_rev1_src.phase_n8.common import read_npz,array_hash,sha
    graph=fixtures._parallel_graph();od=fixtures._parallel_od()
    result=solve_assignment(graph,od,trace_maps=[],config=Phase4SolverConfig(method='CFW',capacity_multiplier=.3,
        max_iterations=3,iteration_record_path=tmp_path/'iterations.jsonl'))
    records=[json.loads(line) for line in (tmp_path/'fresh_certificates.jsonl').read_text().splitlines()]
    assert len(records)==3
    for record in records:
        path=Path(record['certified_state_path']);a=read_npz(path)
        assert sha(path)==record['certified_state_sha256']
        assert array_hash({'flow':a['flow']})==record['state_hash']
        assert float(a['flow']@a['cost_seconds'])==pytest.approx(record['TSTT'])
        assert float(a['aon_flow']@a['cost_seconds'])==pytest.approx(record['SPTT'])


def test_interior_cfw_mixture_satisfies_diagonal_conjugacy():
    from partC_rev1_src.partC_rev1_assignment import partC_rev1_cfw_alpha
    previous=np.array([1.,-1.,0.]);fw=np.array([-.1,1.,-.9]);h=np.array([2.,3.,4.]);tau=.2
    alpha,_=partC_rev1_cfw_alpha(np.zeros(3),np.zeros(3),fw,tau,previous,h,denominator_tol=1e-12,delta_cfw=1e-4)
    assert 0<alpha<1-1e-4
    dbar=(1-tau)*previous;direction=alpha*dbar+(1-alpha)*fw
    assert float(dbar@(h*direction))==pytest.approx(0.,abs=1e-12)
