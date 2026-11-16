"""Meaningful metric/identity regressions; no campaign assignments are launched."""
import json

import numpy as np
import pandas as pd
import pytest

from partC_rev1_src.pacnhn_v1.control import array_hash, sha
from partC_rev1_src.pacnhn_v1.metrics import (
    admit_assignment, backbone_frame, compare_native, compare_vectors,
    extract_backbone, freeze_keys, load_frozen_keys, select_keys,
)


def backbone(n=4):
    return pd.DataFrame(dict(link_id=[f"e{i:04d}" for i in range(n)],
        tail_longitude=np.linspace(-124., -67., n), head_longitude=np.linspace(-123.9, -66.9, n),
        tail_latitude=np.resize([30., 35., 40., 45.], n), head_latitude=np.resize([30.1, 35.1, 40.1, 45.1], n)))


def test_paper_metric_ratio_of_sums_retains_reference_zeros():
    rows, per = compare_vectors([100., 0., 0., 1.], [90., 7., 0., 2.],
        np.array([True, True, False, False]), np.ones(4), np.zeros(4))
    assert rows[0]["wmape_percent"] == pytest.approx(100*18/101)
    assert rows[1]["wmape_percent"] == 17.
    assert rows[0]["link_count"] == 4 and rows[0]["reference_zero_link_count"] == 2
    assert rows[0]["reference_zero_positive_prediction_count"] == 1
    np.testing.assert_allclose(per.signed_bounded_error, [-.1, 1., 0., .5])
    assert np.isnan(per.signed_relative_error.iloc[1]) and np.isnan(per.signed_relative_error.iloc[2])
    assert per.backbone_wmape_contribution_percent.sum() == pytest.approx(rows[0]["wmape_percent"])
    assert rows[0]["backbone_cost_error_percent"] == pytest.approx(100*abs(99-101)/101)


def test_zero_reference_total_is_undefined_not_zero_or_epsilon():
    rows, per = compare_vectors([0., 0.], [1., 0.], np.array([True, True]))
    assert all(r["wmape_percent"] is None for r in rows)
    assert all(r["wmape_status"] == "UNDEFINED_ZERO_REFERENCE_TOTAL" for r in rows)
    assert rows[0]["absolute_error_sum"] == 1.
    assert per.backbone_wmape_contribution_percent.isna().all()


def test_native_prefix_reordering_is_rejected():
    b = backbone(3)
    np.testing.assert_array_equal(extract_backbone([1., 2., 3., 4.], ["e0000", "e0001", "e0002", "regional"], b, expected_count=3), [1., 2., 3.])
    with pytest.raises(ValueError, match="prefix identity"):
        extract_backbone([1., 2., 3., 4.], ["e0001", "e0000", "e0002", "regional"], b, expected_count=3)


@pytest.mark.parametrize("bad", [np.array([1, 0]), np.array([[True, False]])])
def test_key_mask_must_be_boolean_native_vector(bad):
    with pytest.raises(ValueError, match="boolean"):
        compare_vectors([1., 2.], [1., 2.], bad)


def test_key_selection_uses_open_reference_only_and_is_order_invariant():
    n = 1200; b = backbone(n); ref = np.arange(n, 0, -1, dtype=float)
    keys, full, meta = select_keys(b, ref, reference_id="open-P", reference_scenario="OPEN", expected_count=n)
    shuffled = b.iloc[::-1].copy(); shuffled["prediction_error"] = np.arange(n)*999.
    other, _, _ = select_keys(shuffled, ref[::-1], reference_id="open-P", reference_scenario="OPEN", expected_count=n)
    assert keys.link_id.tolist() == other.link_id.tolist()
    assert len(keys) == 200 and full.is_key_link.sum() == 200
    assert meta["candidate_fraction"] == .2 and meta["candidate_count"] == 240
    assert meta["prediction_errors_used"] is False and meta["full_backbone_population_filtered"] is False
    assert meta["occupied_selected_cells"] > 1
    with pytest.raises(ValueError, match="OPEN P"):
        select_keys(b, ref, reference_id="closed-P", reference_scenario="REPRESENTED_I95", expected_count=n)


def test_key_pool_exclusions_do_not_filter_full_backbone():
    b = backbone(12); b.loc[0, ["tail_longitude", "head_longitude"]] = [-150., -149.]
    ref = np.arange(12, 0, -1, dtype=float); ref[1] = 0
    keys, full, meta = select_keys(b, ref, reference_id="open", reference_scenario="OPEN", target=3, expected_count=12)
    assert len(full) == 12 and len(keys) == 3
    assert meta["geographic_or_zero_exclusions"] == 2
    assert not set(keys.link_id) & {"e0000", "e0001"}


def test_frozen_key_file_and_source_identity_are_immutable(tmp_path):
    b = backbone(12); ref = np.arange(12, 0, -1, dtype=float)
    source = tmp_path/"open_reference_source"; source.write_bytes(b"test-only reference source")
    binding = [{"path": str(source), "sha256": sha(source)}]
    dest = tmp_path/"freeze"
    freeze_keys(dest, b, ref, reference_id="open", reference_scenario="OPEN", source_bindings=binding, target=3, expected_count=12)
    keys, _ = load_frozen_keys(dest, b, ref, reference_id="open", expected_count=12, expected_key_count=3)
    assert len(keys) == 3
    with pytest.raises(FileExistsError):
        freeze_keys(dest, b, ref, reference_id="open", reference_scenario="OPEN", source_bindings=binding, target=3, expected_count=12)
    with pytest.raises(ValueError, match="identity"):
        load_frozen_keys(dest, b, ref+1, reference_id="open", expected_count=12, expected_key_count=3)
    with (dest/"key_links_200.csv").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="file changed"):
        load_frozen_keys(dest, b, ref, reference_id="open", expected_count=12, expected_key_count=3)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="input source changed"):
        freeze_keys(tmp_path/"different", b, ref, reference_id="open", reference_scenario="OPEN", source_bindings=binding, target=3, expected_count=12)


def compare_case(reference, predicted, **overrides):
    b = backbone(4); keys = pd.DataFrame({"link_id": ["e0000", "e0002"]})
    kwargs = dict(scenario="OPEN", reference_case_id="open-reference-job", case_id="independent-cfw-job",
        method_id="P_CFW", assignment_role="INDEPENDENT_CONTROL", reference_serve=np.array([True, False, True]),
        predicted_serve=np.array([True, False, True]), expected_count=4, expected_key_count=2)
    kwargs.update(overrides)
    return compare_native(b, reference, predicted, keys, **kwargs)


def test_independent_cfw_is_never_forced_to_reference_self_zero():
    rows, _ = compare_case([10., 0., 5., 2.], [9., 1., 5., 2.])
    assert rows[0]["wmape_percent"] == pytest.approx(100*2/17)
    assert rows[0]["assignment_role"] == "INDEPENDENT_CONTROL"
    zero, _ = compare_case([10., 0., 5., 2.], [10., 0., 5., 2.])
    assert zero[0]["wmape_percent"] == 0  # An actually identical independent result is valid.
    self_rows, _ = compare_case([10., 0., 5., 2.], [10., 0., 5., 2.],
        case_id="open-reference-job", assignment_role="REFERENCE_SELF")
    assert self_rows[0]["assignment_role"] == "REFERENCE_SELF"
    with pytest.raises(ValueError, match="not identical"):
        compare_case([10., 0., 5., 2.], [9., 1., 5., 2.], case_id="open-reference-job", assignment_role="REFERENCE_SELF")


def test_closed_scenario_retains_zero_flow_open_frozen_keys():
    rows, per = compare_case([0., 5., 0., 5.], [0., 4., 0., 5.], scenario="REPRESENTED_I95",
        reference_case_id="closed-reference-job", closed_mask=np.array([True, False, True, False]))
    assert rows[0]["wmape_percent"] == 10.
    assert rows[1]["link_count"] == 2 and rows[1]["wmape_percent"] is None
    assert per.loc[per.is_key_link, "is_closed"].all()
    with pytest.raises(ValueError, match="closed-backbone"):
        compare_case([0., 5., 0., 5.], [1., 4., 0., 5.], scenario="REPRESENTED_I95",
            reference_case_id="closed-reference-job", closed_mask=np.array([True, False, True, False]))


def test_equal_served_counts_do_not_substitute_for_exact_membership():
    with pytest.raises(ValueError, match="served OD membership"):
        compare_case([1., 2., 3., 4.], [1., 2., 3., 4.], predicted_serve=np.array([False, True, True]))


def write_assignment(root, *, potential=True, status="CONVERGED"):
    root.mkdir(); flow = np.array([10., 0., 4.]); np.savez(root/"final_flow.npz", flow=flow)
    np.savez(root/"coverage.npz", serve=np.array([True, False, True]))
    cert = dict(state_hash=array_hash({"flow": flow}), TSTT=14., SPTT=13.99,
        raw_signed_gap=14.-13.99, RG_SPTT=(14.-13.99)/13.99,
        arithmetic_allowance=1.4e-9, value=10. if potential else None, fresh=True,
        interpretation="CONVEX_POTENTIAL_FW_GAP" if potential else "POINTWISE_VI_RESIDUAL_NO_POTENTIAL")
    result = dict(status=status, method="CFW" if potential else "MSA", certificate=cert,
        final_certificate_fresh=True, reported_state="LAST_FRESH_CERTIFIED_STATE", reported_state_iteration=1,
        graph_identity="graph", OD_identity="served-od", objective_has_potential=potential, nominal_tolerance=.001)
    (root/"RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    return result


@pytest.mark.parametrize("status", ["CONVERGED", "TIME_LIMIT", "STAGNATION"])
def test_admission_preserves_actual_certified_disposition(tmp_path, status):
    root = tmp_path/"case"; write_assignment(root, status=status)
    flow, result, serve, bindings = admit_assignment(root, expected_graph_identity="graph",
        expected_edge_ids=["a", "b", "c"], expected_OD_identity="served-od")
    assert result["status"] == status and len(bindings) == 3
    np.testing.assert_array_equal(flow, [10., 0., 4.]); np.testing.assert_array_equal(serve, [True, False, True])


def test_admission_rejects_flow_certificate_mismatch(tmp_path):
    root = tmp_path/"case"; write_assignment(root)
    np.savez(root/"final_flow.npz", flow=np.array([9., 1., 4.]))
    with pytest.raises(ValueError, match="state identity"):
        admit_assignment(root, expected_graph_identity="graph", expected_edge_ids=["a", "b", "c"])


def test_cost_only_admission_does_not_invent_potential(tmp_path):
    root = tmp_path/"case"; result = write_assignment(root, potential=False)
    _, admitted, _, _ = admit_assignment(root, expected_graph_identity="graph", expected_edge_ids=["a", "b", "c"])
    assert admitted["certificate"]["value"] is None
    result["certificate"]["value"] = 10.; (root/"RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="invent a potential"):
        admit_assignment(root, expected_graph_identity="graph", expected_edge_ids=["a", "b", "c"])


def test_pilot_and_uncertified_states_are_not_final_comparisons(tmp_path):
    root = tmp_path/"case"; result = write_assignment(root, status="PILOT_UPDATE_LIMIT")
    with pytest.raises(ValueError, match="status"):
        admit_assignment(root, expected_graph_identity="graph", expected_edge_ids=["a", "b", "c"])
    result["status"] = "TIME_LIMIT"; result["final_certificate_fresh"] = False
    (root/"RESULT.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="fresh"):
        admit_assignment(root, expected_graph_identity="graph", expected_edge_ids=["a", "b", "c"])


def test_declared_native_order_and_nonfinite_coordinates_fail():
    b = backbone(4); b["native_index"] = [1, 0, 2, 3]
    with pytest.raises(ValueError, match="native order"):
        backbone_frame(b, expected_count=4)
    b = backbone(4); b.loc[0, "tail_latitude"] = np.nan
    with pytest.raises(ValueError, match="geography"):
        backbone_frame(b, expected_count=4)
