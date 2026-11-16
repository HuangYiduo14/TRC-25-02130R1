import time
import numpy as np
import pytest

from partC_rev1_src.pacnhn_v1.control import save_npz, sha
from partC_rev1_src.pacnhn_v1.regional import (
    generate_labels, load_admitted_dataset, load_region_graph, sample_requests, solve_label,
)


def toy(tmp_path, *, cycle=False):
    path = tmp_path/"graph.npz"
    if cycle:
        save_npz(path, tail=np.array([0, 1, 2], np.int64), head=np.array([1, 2, 0], np.int64),
            node_ids=np.array(["P0", "P1", "P2"]), edge_ids=np.array(["e01", "e12", "e20"]),
            base=np.array([1., 2., 4.]), k4=np.zeros(3), pair_source=np.array([2, 0], np.int64),
            pair_target=np.array([0, 2], np.int64), pair_keys=np.array([6, 2], np.uint64),
            baseline_open=np.array([1., 1.]), baseline_i95=np.array([2., 2.]))
    else:
        save_npz(path, tail=np.array([0, 0], np.int64), head=np.array([1, 1], np.int64),
            node_ids=np.array(["P0", "P1"]), edge_ids=np.array(["a", "b"]),
            base=np.array([1., 2.]), k4=np.array([1., 0.]), pair_source=np.array([0], np.int64),
            pair_target=np.array([1], np.int64), pair_keys=np.array([1], np.uint64),
            baseline_open=np.array([2.]), baseline_i95=np.array([3.]))
    return load_region_graph(path, expected_sha256=sha(path))


def test_regional_parallel_exact_equilibrium_and_zero(tmp_path):
    graph = toy(tmp_path)
    result = solve_label(graph, [2.], max_seconds=30., max_iterations=20)
    assert result["record"]["admitted"] and result["record"]["quality"] == "A"
    np.testing.assert_allclose(result["flow"], [1., 1.], atol=1e-9)
    np.testing.assert_allclose(result["tau"], [2.], atol=1e-9)
    assert abs(result["record"]["potential_upper"]-3.2) < 1e-9
    assert not result["record"]["marginal_cost_error_bound_available"]
    zero = solve_label(graph, [0.], max_seconds=30., max_iterations=20)
    np.testing.assert_array_equal(zero["flow"], [0., 0.])
    np.testing.assert_array_equal(zero["tau"], [1.])
    assert zero["record"]["zero_input"] and zero["record"]["quality"] == "A"


def test_regional_pair_order_and_zero_query_through_vertices(tmp_path):
    graph = toy(tmp_path, cycle=True)
    result = solve_label(graph, [0., 1.], max_seconds=30., max_iterations=10)
    np.testing.assert_array_equal(result["tau"], [4., 3.])
    np.testing.assert_array_equal(result["flow"], [1., 1., 0.])
    assert not graph.network.blocked.any()
    with pytest.raises(ValueError, match="hash"):
        load_region_graph(graph.source_path, expected_sha256="0"*64)


def test_regional_sampling_prefix_unique_and_disjoint_roles(tmp_path):
    graph = toy(tmp_path)
    a = sample_requests(graph, role="TRAIN", count=5, seed=42)
    b = sample_requests(graph, role="TRAIN", count=8, seed=42)
    assert [r["input_id"] for r in a] == [r["input_id"] for r in b[:5]]
    valid = sample_requests(graph, role="VALID", count=5, seed=42, existing_input_ids=[r["input_id"] for r in b])
    assert not ({r["input_id"] for r in b} & {r["input_id"] for r in valid})
    assert len({r["input_id"] for r in b}) == len(b)
    assert a[0]["family"] == "ZERO"


def config(graph):
    return dict(phase_id="PACNHN_V1", run_id="synthetic", region_id="toy", graph_path=graph.source_path,
        graph_sha256=graph.graph_sha256, deadline_epoch=time.time()+60., per_label_seconds=30.,
        max_iterations=20, relative_gap_tolerance=1e-6)


def test_regional_immutable_roles_exact_resume_and_blind_test(tmp_path):
    graph = toy(tmp_path); root = tmp_path/"labels"
    train = sample_requests(graph, role="TRAIN", count=2, seed=42)
    tests = sample_requests(graph, role="TEST", count=2, seed=42,
        existing_input_ids=[r["input_id"] for r in train])
    result = generate_labels(config(graph), root, requests=train+tests)
    assert result["admitted_current_batch"] == 4
    dataset = load_admitted_dataset(root, role="TRAIN")
    assert dataset["unique_rows"] == 2 and dataset["tau"].shape == (2, 1)
    assert dataset["train_cost_mean"] == dataset["tau"].mean()
    again = generate_labels(config(graph), root, requests=train)
    assert again["reused"] == 2 and again["attempted"] == 0
    assert len(list((root/"request_versions").glob("v*.json"))) == 2
    with pytest.raises(PermissionError):
        load_admitted_dataset(root, role="TEST")
    tampered = dict(train[0], role="VALID")
    with pytest.raises(ValueError):
        generate_labels(config(graph), root, requests=[tampered])


def test_regional_failure_ledger_preserves_healthy_labels(tmp_path, monkeypatch):
    import partC_rev1_src.pacnhn_v1.regional as module
    graph = toy(tmp_path); root = tmp_path/"labels"
    requests = sample_requests(graph, role="TRAIN", count=2, seed=1)
    original = module.solve_label
    def fail_zero(region, u, **kwargs):
        if np.all(u == 0):
            raise RuntimeError("deliberate synthetic failure")
        return original(region, u, **kwargs)
    monkeypatch.setattr(module, "solve_label", fail_zero)
    result = generate_labels(config(graph), root, requests=requests)
    assert result["attempted"] == 2 and result["admitted_current_batch"] == 1
    assert len(list(root.rglob("FAILED.json"))) == 1
    assert load_admitted_dataset(root, role="TRAIN")["unique_rows"] == 1
    assert len((root/"ATTEMPTS.jsonl").read_text().splitlines()) == 2


def test_regional_expired_label_cannot_fabricate_certificate(tmp_path):
    graph = toy(tmp_path)
    with pytest.raises(TimeoutError):
        solve_label(graph, [2.], max_seconds=30., max_iterations=20, deadline_epoch=time.time()-1.)


def test_regional_interrupted_next_update_returns_last_fresh_state(tmp_path, monkeypatch):
    import partC_rev1_src.pacnhn_v1.regional as module
    graph = toy(tmp_path)
    def interrupted(*args, **kwargs):
        raise TimeoutError("synthetic next-step deadline")
    monkeypatch.setattr(module, "line_search", interrupted)
    result = solve_label(graph, [2.], max_seconds=30., max_iterations=20)
    assert result["record"]["status"] == "TIME_LIMIT"
    np.testing.assert_array_equal(result["flow"], [2., 0.])
    np.testing.assert_array_equal(result["edge_cost"], [17., 2.])
    np.testing.assert_array_equal(result["tau"], [2.])
    assert result["record"]["certificate"]["raw_signed_gap"] == 30.
    assert not result["record"]["admitted"]


def test_regional_invariant_failure_stops_dependent_requests(tmp_path, monkeypatch):
    import partC_rev1_src.pacnhn_v1.regional as module
    graph = toy(tmp_path)
    requests = sample_requests(graph, role="TRAIN", count=3, seed=1)
    def invariant(*args, **kwargs):
        raise module.RegionalInvariantError("synthetic support failure")
    monkeypatch.setattr(module, "solve_label", invariant)
    result = generate_labels(config(graph), tmp_path/"labels", requests=requests)
    assert result["status"] == "BLOCKED_REGIONAL_INVARIANT"
    assert result["attempted"] == 1 and result["unattempted"] == 2
