"""Mutation tests for the independent source-to-P audit, on small fixtures."""
import copy
import unittest

import numpy as np
import pandas as pd

from partC_rev1_src.phase5.campaign_runtime import pack_runtime
from partC_rev1_src.pacnhn_v1.tests.test_pacnhn_topology import fixture, build_fixture
from partC_rev1_src.pacnhn_v1.topology_audit import (
    Checks, audit_backbone, audit_residual, audit_mapping, audit_charge_partition,
    audit_compiled_ac,audit_region_copy,native_array_hash,array_hash,
)


class TopologyAuditTests(unittest.TestCase):
    def backbone_inputs(self):
        values = list(fixture())
        values[1]["low_current_flow"] = 12.
        values[1]["member_arc_idx"] = -1
        a, od, tables, report = build_fixture(values)
        packed = pack_runtime(values[1], values[2]).arrays
        mask = dict(abstract_link_ids=np.array(["fixed_B"]), unavailable=np.array([True]))
        return values, a, packed, mask

    def test_full_backbone_three_probes_and_mutated_coefficient_rejected(self):
        values, a, packed, mask = self.backbone_inputs()
        checks = Checks()
        probes, indices = audit_backbone(a, values[1], values[2], packed, mask, checks)
        self.assertEqual(len(probes), 9)
        self.assertEqual(indices.tolist(), [0])
        self.assertTrue(probes.status.eq("PASS").all())
        a["k4"][0] *= 1.01
        with self.assertRaisesRegex(ValueError, "B frozen field k4"):
            audit_backbone(a, values[1], values[2], packed, mask, Checks())

    def test_residual_rejects_double_capacity_factor_and_double_charge(self):
        values = fixture()
        a, _, _, _ = build_fixture(values)
        charge = np.zeros(len(a["node_ids"])); charge[:2] = [4., 6.]
        probes, _ = audit_residual(a, values[0], values[3], charge, capacity_multiplier=.3, checks=Checks())
        self.assertEqual(len(probes), 3)
        bad = copy.deepcopy(a); bad["capacity"][1] *= .3
        with self.assertRaisesRegex(ValueError, "capacity factor applied once"):
            audit_residual(bad, values[0], values[3], charge, capacity_multiplier=.3, checks=Checks())
        bad = copy.deepcopy(a); bad["tail_half_charge_s"][1] *= 2
        with self.assertRaisesRegex(ValueError, "tail half charge once"):
            audit_residual(bad, values[0], values[3], charge, capacity_multiplier=.3, checks=Checks())

    def test_mapping_rejects_centroid_block_and_role_remap(self):
        values = fixture()
        a, od, _, _ = build_fixture(values)
        audit_mapping(a, od, values[0], values[4], Checks())
        bad = copy.deepcopy(a); bad["blocked"][bad["physical_to_node"][3]] = True
        with self.assertRaisesRegex(ValueError, "Only logical roles"):
            audit_mapping(bad, od, values[0], values[4], Checks())
        bad_od = copy.deepcopy(od); bad_od["destinations"][0] = od["destinations"][1]
        with self.assertRaisesRegex(ValueError, "Every original destination"):
            audit_mapping(a, bad_od, values[0], values[4], Checks())

    def test_cost_partition_rejects_lost_edge_or_extra_backbone_owner(self):
        a, _, frames, _ = build_fixture()
        p = dict(suite_ids=np.array(["s02", "s03", "s04", "s05", "s06"]),
                 edge_region=np.tile(np.where(a["edge_role"] == 1, 0, -1), (5, 1)))
        p.update({sid+"_region_ids": np.array(["r0"]) for sid in p["suite_ids"]})
        records = audit_charge_partition(a, p, frames["cost_ownership"], Checks())
        self.assertEqual(len(records), 5)
        bad = copy.deepcopy(p); bad["edge_region"][0, 2] = -1
        with self.assertRaisesRegex(ValueError, "every residual has one owner"):
            audit_charge_partition(a, bad, frames["cost_ownership"], Checks())
        bad = copy.deepcopy(p); bad["edge_region"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "B/role edges not regional"):
            audit_charge_partition(a, bad, frames["cost_ownership"], Checks())


def generated_fixture():
    p=dict(tail=np.array([1,2,3,0,4],np.int64),head=np.array([2,3,4,1,5],np.int64),
        node_ids=np.array(['O','A','B','C','D','Z']),blocked=np.array([True,False,False,False,False,True]),
        edge_ids=np.array(['b','r0','r1','port_o','port_d']),base=np.array([10.,2.,3.,0.,0.]),
        k4=np.array([.02,.1,.2,0.,0.]),closed_mask=np.array([True,False,False,False,False]),
        edge_role=np.array([0,1,1,2,2]),capacity=np.array([0.,4.,6.,0.,0.]))
    ac=dict(tail=np.array([1,2,0,4],np.int64),head=np.array([2,4,1,5],np.int64),
        node_ids=p['node_ids'].copy(),blocked=p['blocked'].copy(),
        edge_ids=np.array(['b','pac_pair:s03:16','port_o','port_d']),
        base=np.array([10.,5.,0.,0.]),k4=np.array([.02,0.,0.,0.]),closed_mask=np.array([True,False,False,False]),
        backbone_count=np.array(1),port_count=np.array(2),pair_keys=np.array([16],np.uint64),
        pair_region=np.array([0],np.int32),pair_source=np.array([2],np.int64),pair_target=np.array([4],np.int64),
        c0=np.array([5.]),baseline_open=np.array([2.]),baseline_i95=np.array([0.]))
    report=dict(graph_identity=native_array_hash({k:ac[k] for k in ('tail','head','blocked','edge_ids','node_ids')}),
        OD_identity='common_od',backbone_count=1,active_pairs=1,active_regions=1,support_frozen_before_labels=True,
        backbone_identity=array_hash(dict(tail=ac['tail'][:1],head=ac['head'][:1],ids=ac['edge_ids'][:1],
            base=ac['base'][:1],k4=ac['k4'][:1],closed=ac['closed_mask'][:1])))
    pairs=pd.DataFrame(dict(pair_key=ac['pair_keys'],region_index=ac['pair_region'],native_index=[0],
        source_node=ac['pair_source'],target_node=ac['pair_target'],c0=ac['c0'],
        baseline_open=ac['baseline_open'],baseline_i95=ac['baseline_i95']))
    local=dict(tail=np.array([0,1],np.int64),head=np.array([1,2],np.int64),node_ids=p['node_ids'][[2,3,4]],
        edge_ids=p['edge_ids'][[1,2]],base=p['base'][[1,2]],k4=p['k4'][[1,2]],
        pair_source=np.array([0],np.int64),pair_target=np.array([2],np.int64),pair_keys=ac['pair_keys'].copy(),
        c0=ac['c0'].copy(),baseline_open=ac['baseline_open'].copy(),baseline_i95=ac['baseline_i95'].copy(),
        P_edge_indices=np.array([1,2],np.int64),P_node_indices=np.array([2,3,4],np.int64),physical_capacity_reference=np.array(5.))
    record=dict(region_index=0,input_identity=array_hash(local),
        graph_identity=native_array_hash(dict(tail=local['tail'],head=local['head'],blocked=np.zeros(3,bool),
            edge_ids=local['edge_ids'],node_ids=local['node_ids'])),
        physical_capacity_reference=5.,all_pairs_lift=True,pairs=1,edges=2,nodes=3,local_FFTT_seconds=.1)
    return p,ac,report,pairs,local,record


class GeneratedACTopologyTests(unittest.TestCase):
    def test_complete_compiled_coordinates_and_region_copy(self):
        p,ac,report,pairs,local,record=generated_fixture()
        regions,starts,counts=audit_compiled_ac(p,ac,report,pairs,'s03','common_od',Checks(),expected_backbone_count=1)
        self.assertEqual(regions.tolist(),[0]);self.assertEqual(starts.tolist(),[0]);self.assertEqual(counts.tolist(),[1])
        row=audit_region_copy(p,local,record,ac,np.array([1,2]),0,1,'s03',Checks())
        self.assertEqual(row['physical_edges'],2)
        self.assertEqual(row['AC_interface_added_charge'],0)
        self.assertFalse(row['shortest_path_replayed'])

    def test_compiled_mutated_B_double_interface_charge_mask_or_OD_rejected(self):
        for key,index,match in [('base',0,'exact B base'),('base',1,'no added interface charge'),
                ('closed_mask',1,'no regional closure'),('head',0,'exact B head')]:
            p,ac,report,pairs,_,_=generated_fixture()
            if key=='closed_mask':ac[key][index]=True
            else:ac[key][index]+=1
            with self.assertRaisesRegex(ValueError,match):
                audit_compiled_ac(p,ac,report,pairs,'s03','common_od',Checks(),expected_backbone_count=1)
        p,ac,report,pairs,_,_=generated_fixture()
        with self.assertRaisesRegex(ValueError,'same original global OD'):
            audit_compiled_ac(p,ac,report,pairs,'s03','other_od',Checks(),expected_backbone_count=1)

    def test_local_lost_owner_rewired_endpoint_or_changed_c0_rejected(self):
        for key,index,match in [('P_edge_indices',1,'full owned P edges'),('head',0,'local head'),
                ('base',0,'cost ownership once base'),('c0',0,'full c0')]:
            p,ac,_,_,local,record=generated_fixture();local[key][index]+=1
            with self.assertRaisesRegex(ValueError,match):
                audit_region_copy(p,local,record,ac,np.array([1,2]),0,1,'s03',Checks())
        p,ac,_,_,local,record=generated_fixture();record['all_pairs_lift']=False
        with self.assertRaisesRegex(ValueError,'complete shortest-path lift receipt'):
            audit_region_copy(p,local,record,ac,np.array([1,2]),0,1,'s03',Checks())


if __name__ == "__main__":
    unittest.main()
