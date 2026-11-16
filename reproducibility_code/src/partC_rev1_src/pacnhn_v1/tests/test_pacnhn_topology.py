"""Small fixtures for fixed-B costs, exact endpoints and partition-only changes."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from partC_rev1_src.pacnhn_v1.topology import (
    assemble_common, edge_owners_from_nodes, partition_arrays, interface_registry, runtime_identities,
)


def fixture():
    physical = dict(node_idx_to_uid=np.array(["0", "1", "2", "3", "4"]),
        arc_idx_to_directed_arc_id=np.array(["10", "11", "12", "13", "14", "15", "16"]),
        tail=np.array([0, 2, 1, 3, 3, 4, 3]), head=np.array([1, 2, 3, 4, 4, 3, 3]),
        free_time=np.array([1., 1., 1., 2., 3., 2., 4.]),
        capacity=np.full(7, 100.), alpha=np.full(7, .15), beta=np.full(7, 4.))
    links = pd.DataFrame([dict(link_id="fixed_B", tail_node_id="B0", head_node_id="B1",
        role="EXACT_PRIMITIVE", runtime_class="EXACT_PRIMITIVE_BPR", dynamic_model_id="fixed_model",
        tau_s=10., k4=.25, fixed_internal_H_s=0., tail_half_charge_s=2., head_half_charge_s=3.,
        structural_fixed_s=9., domain_Q=np.inf, beta=4.)])
    nodes = pd.DataFrame(dict(node_id=["B0", "B1"], node_type=["BACN", "BACN"],
                              longitude=[-90., -89.], latitude=[35., 36.]))
    ownership = pd.DataFrame(dict(physical_arc_idx=np.arange(7), physical_arc_id=physical["arc_idx_to_directed_arc_id"],
        ownership_role=["INTER_B", "SUPPRESSED_B"]+["REGION"]*5,
        original_owner_id=["bundle", "B2", "old_region", "old_region", "old_region", "old_region", "old_region"],
        final_link_id=["fixed_B"]+[""]*6, sequence_position=[0]+[-1]*6, owner_count=np.ones(7, int)))
    membership = pd.DataFrame(dict(partC_rev1_node_uid=["0", "1", "2"], bacn_id=["B0", "B1", "B2"]))
    charges = pd.DataFrame(dict(suite_id=["s03"]*3, bacn_id=["B0", "B1", "B2"],
        model_variant=["LOW_MEAN_STATIC_INCIDENCE_V1"]*3, chi_s=[4., 6., 8.]))
    od = dict(origin_group_node_idx=np.array([0, 3]), origin_offsets=np.array([0, 1, 2]),
              destination_node_idx=np.array([4, 1]), demand=np.array([7., 11.]),
              original_row_multiplicity=np.array([2, 3]))
    return physical, links, nodes, ownership, membership, charges, od


def build_fixture(values=None):
    return assemble_common(*(values or fixture()), ["fixed_B"],
                           expected_backbone_count=1, capacity_multiplier=.3)


class CommonTopologyTests(unittest.TestCase):
    def test_runtime_identities_use_inherited_constructor_convention(self):
        from partC_rev1_src.phase_n8.oracle import Network, Demand
        a, o, _, _ = build_fixture()
        graph = Network(a["tail"], a["head"], a["blocked"], a["edge_ids"], a["node_ids"])
        demand = Demand(o["origins"], o["offsets"], o["destinations"], o["demand"], o["serve"])
        self.assertEqual(runtime_identities(a, o), (graph.identity, demand.identity))

    def test_frozen_B_and_parallel_residual_costs(self):
        values = fixture()
        a, od, tables, report = build_fixture(values)
        self.assertEqual(a["edge_ids"][0], "fixed_B")
        self.assertEqual(a["base"][0], 15.)
        self.assertEqual(a["k4"][0], .25)
        self.assertEqual(a["structural_fixed_s"][0], 9.)
        self.assertEqual(a["capacity"][1], 30.)
        self.assertEqual(a["base"][1], 63.)  # one B1 half-charge, once
        self.assertAlmostEqual(a["k4"][1], 60*.15/30**4)
        self.assertEqual(a["tail"][2], a["tail"][3])
        self.assertEqual(a["head"][2], a["head"][3])
        self.assertNotEqual(a["edge_ids"][2], a["edge_ids"][3])
        self.assertNotEqual(a["base"][2], a["base"][3])
        self.assertEqual(report["regional_self_loop_count"], 1)
        self.assertEqual(int(a["closed_mask"].sum()), 1)
        self.assertFalse(a["closed_mask"][a["edge_role"] == 1].any())
        self.assertEqual(len(tables["physical_cost_ownership"]), 7)
        self.assertEqual(report["original_OD_rows"], 5)
        np.testing.assert_array_equal(od["demand"], values[-1]["demand"])
        self.assertEqual(a["physical_to_node"][2], -1)

    def test_physical_centroid_may_transit_but_roles_cannot(self):
        a, od, frames, _ = build_fixture()
        physical_centroid = int(a["physical_to_node"][3])
        self.assertFalse(a["blocked"][physical_centroid])
        self.assertEqual(a["node_ids"][physical_centroid], "3")
        self.assertTrue(np.any(a["head"][a["edge_role"] == 1] == physical_centroid))
        self.assertTrue(np.any(a["tail"][a["edge_role"] == 1] == physical_centroid))
        mapping = frames["OD_endpoint_mapping"]
        for row in mapping.itertuples():
            self.assertTrue(a["blocked"][row.origin_node_idx])
            self.assertTrue(a["blocked"][row.destination_node_idx])
            self.assertFalse(np.any(a["head"] == row.origin_node_idx))
            self.assertFalse(np.any(a["tail"] == row.destination_node_idx))
        self.assertEqual(mapping.loc[mapping.physical_node_uid.eq("0"), "road_node_id"].iloc[0], "B0")
        self.assertEqual(mapping.loc[mapping.physical_node_uid.eq("3"), "road_node_id"].iloc[0], "3")
        self.assertTrue((a["base"][a["edge_role"] == 2] == 0).all())
        self.assertTrue(od["serve"].all())  # initial eligibility, not a false reachability claim

    def test_removed_B_needed_by_residual_is_hard_failure(self):
        values = list(fixture())
        values[0]["tail"][2] = 2
        with self.assertRaisesRegex(ValueError, "removed B interior"):
            build_fixture(values)

    def test_removed_B_OD_endpoint_is_hard_failure(self):
        values = list(fixture())
        values[-1]["origin_group_node_idx"][0] = 2
        with self.assertRaisesRegex(ValueError, "Protected physical OD"):
            build_fixture(values)

    def test_no_missing_owner_or_unfrozen_charge(self):
        values = list(fixture())
        values[3].loc[0, "owner_count"] = 2
        with self.assertRaisesRegex(ValueError, "ownership"):
            build_fixture(values)
        values = list(fixture())
        values[1].loc[0, "head_half_charge_s"] = 6.
        with self.assertRaisesRegex(ValueError, "incidence charges"):
            build_fixture(values)

    def test_partition_assignment_never_promotes_cross_region_arc(self):
        t, h = np.array([0, 1, 2, 2]), np.array([1, 2, 1, 2])
        owner = edge_owners_from_nodes(t, h, np.arange(4), np.array(["", "A", "B"]), np.array([True, False, False]))
        np.testing.assert_array_equal(owner, ["A", "A", "B", "B"])

    def test_residual_B_to_B_arc_and_quotient_loop_are_retained(self):
        values = list(fixture())
        values[0]["tail"][2], values[0]["head"][2] = 0, 1
        values[0]["tail"][6], values[0]["head"][6] = 0, 0
        a, _, _, report = build_fixture(values)
        self.assertEqual(report["residual_B_to_B_arc_count"], 2)
        self.assertEqual(a["base"][1], 60.+2.+3.)
        self.assertTrue(a["self_loop"][5])
        self.assertEqual(a["edge_ids"][5], "16")
        labels = np.array(["", "", "", "R", "R"])
        both = np.array([True, True, True, False, False])
        ridx = np.arange(2, 7)
        with self.assertRaisesRegex(ValueError, "Explicit residual B-to-B"):
            edge_owners_from_nodes(values[0]["tail"], values[0]["head"], ridx, labels, both)
        owner = edge_owners_from_nodes(values[0]["tail"], values[0]["head"], ridx, labels, both,
                                      bridge_labels=np.array(["bridge12", "", "", "", "bridge16"]))
        np.testing.assert_array_equal(owner, ["bridge12", "R", "R", "R", "bridge16"])

    def test_all_five_partitions_share_graph_and_exact_interfaces(self):
        values = fixture()
        a, _, _, _ = build_fixture(values)
        before = {key: value.copy() for key, value in a.items()}
        called = []
        def fake_leiden(nodes, arcs, config, *, resolution, random_seed):
            called.append((resolution, random_seed))
            self.assertEqual(set(arcs.directed_arc_id), {"12", "13", "14", "15", "16"})
            return dict(node_partition=pd.DataFrame(dict(partC_rev1_node_uid=["3", "4"], region_id=["r0", "r1"])))
        suites = ["s02", "s03", "s04", "s05", "s06"]
        res = dict(zip(suites, [.3107232505953859, 1., 1.6, 4., 12.]))
        pc = dict(random_seed=42, objective_function="modularity", beta=.01, n_iterations=-1,
                  backbone_bridge_policy="SINGLETON_DIRECTED_PHYSICAL_ARC_REGION")
        with patch("partC_rev1_src.partC_rev1_partition_igraph.partC_rev1_run_leiden_partition", side_effect=fake_leiden):
            p, regions, nodes, summary = partition_arrays(a, values[0], values[3], suites=suites, resolutions=res, partition_config=pc)
        self.assertEqual(len(called), 4)
        self.assertEqual(p["s03_region_ids"].tolist(), ["old_region"])
        self.assertTrue((p["edge_region"][:, a["edge_role"] != 1] == -1).all())
        self.assertTrue((p["edge_region"][:, a["edge_role"] == 1] >= 0).all())
        for key in a:
            np.testing.assert_array_equal(a[key], before[key])
        registry = interface_registry(a, p)
        shared = registry.loc[(registry.suite_id == "s02") & (registry.node_id == "4")]
        self.assertEqual(len(shared), 2)
        self.assertTrue(shared.fine_state_is_singleton.all())
        self.assertEqual(shared.node_index.nunique(), 1)
        self.assertTrue((regions.physical_arc_count.groupby(regions.suite_id).sum() == 5).all())


if __name__ == "__main__":
    unittest.main()
