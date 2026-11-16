"""Independent full-population cost, mapping and ownership audit of common P.

No builder, assignment, training, partitioner or model-inference job is run.
The audit writes only a new, explicitly named directory below
``OUT/audits/topology_cost_identity``. Existing graph and source files are read
only. Scalar floating-point evaluation is compared within 32 machine epsilons;
all stored coefficients, capacities, endpoint IDs and OD values must be exact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import psutil

from partC_rev1_src.phase5.campaign_runtime import PackedRuntime
from partC_rev1_src.phase_n8.common import array_hash as native_array_hash
from .control import array_hash, read_json, sha, write_json


class Checks:
    def __init__(self):
        self.rows = []

    def require(self, name, condition, *, elements=1, detail=""):
        passed = bool(condition)
        self.rows.append(dict(check=name, comparison="PREDICATE", elements=int(elements),
                              status="PASS" if passed else "FAIL", detail=detail))
        if not passed:
            raise ValueError("Independent topology audit: " + name)

    def exact(self, name, actual, expected):
        a, b = np.asarray(actual), np.asarray(expected)
        passed = a.shape == b.shape and np.array_equal(a, b)
        self.rows.append(dict(check=name, comparison="EXACT_ARRAY", elements=int(b.size),
                              status="PASS" if passed else "FAIL", detail=f"actual={a.shape}; expected={b.shape}"))
        if not passed:
            raise ValueError("Independent topology audit exact mismatch: " + name)

    def close(self, name, actual, expected):
        a, b = np.asarray(actual, float), np.asarray(expected, float)
        valid = a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all()
        difference = np.abs(a-b) if valid else np.array([np.inf])
        scale = np.maximum(1., np.maximum(np.abs(a), np.abs(b))) if valid else np.array([1.])
        normalized = difference/scale
        passed = valid and np.all(normalized <= 32*np.finfo(np.float64).eps)
        row = dict(check=name, comparison="FLOAT64_32_EPS_SCALED", elements=int(b.size),
                   status="PASS" if passed else "FAIL",
                   maximum_absolute_error=float(difference.max()) if valid else None,
                   maximum_scaled_error=float(normalized.max()) if valid else None,
                   maximum_scaled_eps=float(normalized.max()/np.finfo(np.float64).eps) if valid else None)
        self.rows.append(row)
        if not passed:
            raise ValueError("Independent topology audit numerical mismatch: " + name)
        return row


def audit_backbone(a, links, nodes, packed_arrays, mask_arrays, checks):
    """Every B coordinate, in the exact original filtered runtime row order."""
    original_indices = np.flatnonzero(links.role.isin(["EXACT_PRIMITIVE", "MACRO", "PHYSICAL_MEMBER"]))
    b = links.iloc[original_indices]
    n = len(b)
    checks.exact("B prefix indices", np.flatnonzero(a["edge_role"] == 0), np.arange(n))
    checks.exact("B exact stable IDs", a["edge_ids"][:n], b.link_id.to_numpy(str))
    checks.exact("B tail stable IDs", a["node_ids"][a["tail"][:n]], b.tail_node_id.to_numpy(str))
    checks.exact("B head stable IDs", a["node_ids"][a["head"][:n]], b.head_node_id.to_numpy(str))
    old_nodes = nodes.node_id.to_numpy(str)
    checks.exact("Source packed tails", old_nodes[packed_arrays["tail"][original_indices]], b.tail_node_id.to_numpy(str))
    checks.exact("Source packed heads", old_nodes[packed_arrays["head"][original_indices]], b.head_node_id.to_numpy(str))
    for key in ("tau_s", "k4", "fixed_internal_H_s", "tail_half_charge_s", "head_half_charge_s", "structural_fixed_s", "domain_Q"):
        checks.exact("B source table/packed " + key, b[key].to_numpy(float), packed_arrays[key][original_indices])
        checks.exact("B frozen field " + key, a[key][:n], b[key].to_numpy(float))
    checks.exact("B fitted type flags", a["is_fitted"][:n], b.runtime_class.eq("FIT_POWER4").to_numpy())
    checks.exact("B source exponent", b.beta.to_numpy(float), np.full(n, 4.))
    ids = pd.Index(mask_arrays["abstract_link_ids"].astype(str))
    at = ids.get_indexer(b.link_id.astype(str))
    checks.require("All frozen B IDs in source availability registry", np.all(at >= 0), elements=n)
    checks.exact("B represented I95 mask", a["closed_mask"][:n], mask_arrays["unavailable"][at])
    checks.require("No closure outside B", not a["closed_mask"][n:].any(), elements=len(a["tail"])-n)
    packed = PackedRuntime(packed_arrays)
    low = links.low_current_flow.to_numpy(float)
    probes = []
    for label, flow in (("ZERO", np.zeros(len(links))), ("NATIVE_LOW_CURRENT", low), ("TWICE_NATIVE_LOW_CURRENT", 2*low)):
        expected, flags = packed.evaluate(flow, operation="combined")
        q = flow[original_indices]
        # This matches the new solver's evaluation arithmetic, independently of
        # PackedRuntime's q*q*q*q expression. Coefficients are checked exactly.
        dynamic = a["k4"][:n]*q**4
        cost = a["base"][:n] + dynamic
        potential = q*(a["base"][:n]+dynamic/5)
        derivative = 4*a["k4"][:n]*q**3
        for operation, actual, source in zip(("COST", "POTENTIAL", "DERIVATIVE"),
                (cost, potential, derivative), expected):
            row = checks.close("B " + label + " " + operation, actual, source[original_indices])
            probes.append(dict(probe=label, operation=operation, coordinates=n,
                zero_flow_coordinates=int((q == 0).sum()), maximum_flow=float(q.max()),
                **{key: row[key] for key in ("status", "maximum_absolute_error", "maximum_scaled_error", "maximum_scaled_eps")}))
        actual_flags = a["is_fitted"][:n] & (q > a["domain_Q"][:n])
        checks.exact("B fitted domain flags " + label, actual_flags, flags[original_indices])
    return pd.DataFrame(probes), original_indices


def audit_residual(a, physical, ownership, charge_by_node, *, capacity_multiplier, checks):
    """Every exact residual BPR term and both directional incidence charges."""
    idx = np.flatnonzero(a["edge_role"] == 1)
    source = np.flatnonzero(ownership.ownership_role.eq("REGION").to_numpy())
    checks.exact("Residual exact source population/order", a["physical_arc_idx"][idx], source)
    checks.exact("Residual exact directed IDs", a["edge_ids"][idx], physical["arc_idx_to_directed_arc_id"].astype(str)[source])
    used = physical["capacity"][source]*capacity_multiplier
    t0 = physical["free_time"][source]*60.
    alpha = physical["alpha"][source]
    checks.exact("Residual capacity factor applied once", a["capacity"][idx], used)
    checks.exact("Residual minutes converted once", a["tau_s"][idx], t0)
    checks.exact("Residual physical alpha", a["physical_alpha"][idx], alpha)
    checks.exact("Residual exponent exactly four", physical["beta"][source], np.full(len(source), 4.))
    checks.exact("Residual BPR coefficient", a["k4"][idx], t0*alpha/used**4)
    checks.exact("Residual no macro-internal charge", a["fixed_internal_H_s"][idx], np.zeros(len(idx)))
    expected_tail = charge_by_node[a["tail"][idx]]/2
    expected_head = charge_by_node[a["head"][idx]]/2
    checks.exact("Residual tail half charge once", a["tail_half_charge_s"][idx], expected_tail)
    checks.exact("Residual head half charge once", a["head_half_charge_s"][idx], expected_head)
    checks.exact("Residual base charge decomposition", a["base"][idx], t0+(expected_tail+expected_head))
    probes = []
    for ratio in (0., 1., 2.):
        q = ratio*used
        expected = t0*(1+alpha*(q/used)**4)+expected_tail+expected_head
        actual = a["base"][idx]+a["k4"][idx]*q**4
        row = checks.close("Residual physical BPR at used-capacity ratio " + str(ratio), actual, expected)
        probes.append(dict(probe="FLOW_OVER_USED_CAPACITY_"+str(ratio), coordinates=len(idx),
                           **{key: row[key] for key in ("status", "maximum_absolute_error", "maximum_scaled_error", "maximum_scaled_eps")}))
    return pd.DataFrame(probes), source


def audit_mapping(a, od, physical, membership, checks):
    uid = physical["node_idx_to_uid"].astype(str)
    checks.exact("Physical UID universe", a["physical_node_uids"], uid)
    checks.require("B membership is single valued", membership.partC_rev1_node_uid.is_unique, elements=len(membership))
    ui, ni = pd.Index(uid), pd.Index(a["node_ids"])
    member_at = ui.get_indexer(membership.partC_rev1_node_uid.astype(str))
    checks.require("B membership source UID coverage", np.all(member_at >= 0), elements=len(member_at))
    expected = ni.get_indexer(uid)
    expected[member_at] = ni.get_indexer(membership.bacn_id.astype(str))
    checks.exact("Independent physical UID to frozen quotient map", a["physical_to_node"], expected)
    ridx = np.flatnonzero(a["edge_role"] == 1)
    physical_ridx = a["physical_arc_idx"][ridx]
    checks.exact("Residual rewired tails", a["tail"][ridx], expected[physical["tail"][physical_ridx]])
    checks.exact("Residual rewired heads", a["head"][ridx], expected[physical["head"][physical_ridx]])
    checks.require("Residual no removed endpoint", (a["tail"][ridx] >= 0).all() and (a["head"][ridx] >= 0).all(), elements=2*len(ridx))
    origins = od["origin_group_node_idx"]
    destinations = od["destination_node_idx"]
    endpoints = np.unique(np.r_[origins, destinations])
    checks.require("OD no removed quotient endpoint", (expected[endpoints] >= 0).all(), elements=len(endpoints))
    onames, dnames = np.char.add("n8_od_origin:", uid), np.char.add("n8_od_destination:", uid)
    checks.exact("Every original origin role mapping", a["node_ids"][od["origins"]], onames[origins])
    checks.exact("Every original destination role mapping", a["node_ids"][od["destinations"]], dnames[destinations])
    oidx, didx = ni.get_indexer(onames[endpoints]), ni.get_indexer(dnames[endpoints])
    checks.require("All logical roles exist", (oidx >= 0).all() and (didx >= 0).all(), elements=2*len(endpoints))
    ports = np.flatnonzero(a["edge_role"] == 2)
    checks.exact("Exact logical attachment population", a["edge_ids"][ports], np.r_[
        np.char.add("pacnhn_attachment:origin:", uid[endpoints]),
        np.char.add("pacnhn_attachment:destination:", uid[endpoints])])
    checks.exact("Logical attachment tails", a["tail"][ports], np.r_[oidx, expected[endpoints]])
    checks.exact("Logical attachment heads", a["head"][ports], np.r_[expected[endpoints], didx])
    blocked = np.zeros(len(a["node_ids"]), bool)
    blocked[np.r_[oidx, didx]] = True
    checks.exact("Only logical roles are nontransit", a["blocked"], blocked)
    indegree = np.bincount(a["head"], minlength=len(blocked))
    outdegree = np.bincount(a["tail"], minlength=len(blocked))
    checks.require("Origin roles cannot be entered", (indegree[oidx] == 0).all(), elements=len(oidx))
    checks.require("Destination roles cannot be exited", (outdegree[didx] == 0).all(), elements=len(didx))
    for field in ("base", "tau_s", "k4", "fixed_internal_H_s", "tail_half_charge_s", "head_half_charge_s"):
        checks.exact("Zero role cost " + field, a[field][ports], np.zeros(len(ports)))
    checks.exact("Every original OD row initially retained", od["serve"], np.ones(len(od["demand"]), bool))
    return expected


def audit_charge_partition(a, p, cost_ledger, checks):
    """Disjoint cost ownership implies exact P-to-local decomposition.

    For each suite and arbitrary nonnegative edge flows, summing regional
    physical-edge potentials plus frozen B potentials equals the P potential.
    Regional half charges are part of those physical-edge potentials. They must
    not be added to the AC regional-edge endpoints a second time.
    """
    checks.exact("Cost ledger edge order", cost_ledger.edge_idx, np.arange(len(a["tail"])))
    checks.exact("Cost ledger IDs", cost_ledger.edge_id.astype(str), a["edge_ids"])
    for field in ("edge_role", "physical_arc_idx", "tau_s", "k4", "fixed_internal_H_s",
                  "tail_half_charge_s", "head_half_charge_s", "structural_fixed_s"):
        checks.exact("Cost ledger " + field, cost_ledger[field].to_numpy(), a[field])
    residual = a["edge_role"] == 1
    checks.require("All five explicit partitions", p["edge_region"].shape == (5, len(residual)), elements=5)
    records = []
    # A deterministic nonuniform flow vector checks the sum algebra without
    # running a solver or claiming that these independent edge flows are feasible.
    flow = (1+(np.arange(len(residual)) % 11))/11.
    fixed = a["tau_s"]+a["fixed_internal_H_s"]+a["tail_half_charge_s"]+a["head_half_charge_s"]
    edge_value = flow*(fixed+a["k4"]*flow**4/5)
    for si, sid in enumerate(p["suite_ids"]):
        labels = p["edge_region"][si]
        checks.require(str(sid)+" every residual has one owner", (labels[residual] >= 0).all(), elements=int(residual.sum()))
        checks.exact(str(sid)+" B/role edges not regional", labels[~residual], np.full(int((~residual).sum()), -1, np.int32))
        checks.require(str(sid)+" regional owner bounds", labels[residual].max() < len(p[str(sid)+"_region_ids"]), elements=int(residual.sum()))
        local = np.bincount(labels[residual], weights=edge_value[residual], minlength=len(p[str(sid)+"_region_ids"]))
        checks.close(str(sid)+" disjoint potential decomposition", np.array([local.sum()+edge_value[~residual].sum()]), np.array([edge_value.sum()]))
        records.append(dict(suite_id=str(sid), residual_edge_occurrences=int(residual.sum()),
            unique_residual_edge_owners=int(residual.sum()), regions=len(local),
            B_cost_owner="UNCHANGED_B_COORDINATE", regional_cost_owner="ONE_DETAILED_REGION_VALUE_FUNCTION",
            regional_half_charges="INCLUDED_IN_LOCAL_PHYSICAL_EDGE_BASE", extra_AC_interface_charge=0,
            proof="PER_EDGE_PARTITION_OF_SUM; VALID_FOR_ARBITRARY_NONNEGATIVE_EDGE_FLOWS"))
    return pd.DataFrame(records)


def audit_compiled_ac(common,ac,report,pairs,sid,od_identity,checks, *,expected_backbone_count=87582):
    """All actual AC coordinates, including zero-flow pairs and logical ports."""
    label=sid+' AC ';b=int(ac['backbone_count']);keys=ac['pair_keys'];m=len(keys);n=len(common['node_ids'])
    ports=np.flatnonzero(common['edge_role']==2)
    checks.require(label+'nonempty native B and active pairs',b==expected_backbone_count and m>0,elements=b+m)
    checks.exact(label+'P B prefix',np.flatnonzero(common['edge_role']==0),np.arange(b))
    checks.require(label+'strict uint64 pair order',keys.dtype==np.dtype('uint64') and np.all(keys[1:]>keys[:-1]),elements=m)
    checks.require(label+'edge and port count',len(ac['tail'])==b+m+len(ports) and int(ac['port_count'])==len(ports))
    for key in ('node_ids','blocked'):checks.exact(label+key,ac[key],common[key])
    for key in ('tail','head','edge_ids','base','k4','closed_mask'):
        checks.exact(label+'exact B '+key,ac[key][:b],common[key][:b])
        checks.exact(label+'exact port '+key,ac[key][b+m:],common[key][ports])
    checks.exact(label+'regional edge IDs',ac['edge_ids'][b:b+m],np.char.add('pac_pair:'+sid+':',keys.astype(str)))
    regions=(keys//np.uint64(n*n)).astype(np.int32)
    sources=((keys//np.uint64(n))%np.uint64(n)).astype(np.int64)
    targets=(keys%np.uint64(n)).astype(np.int64)
    for key,expected in (('pair_region',regions),('pair_source',sources),('pair_target',targets)):
        checks.exact(label+key,ac[key],expected)
    checks.require(label+'no self pair',np.all(sources!=targets),elements=m)
    checks.exact(label+'regional tails',ac['tail'][b:b+m],sources)
    checks.exact(label+'regional heads',ac['head'][b:b+m],targets)
    checks.require(label+'finite nonnegative c0',ac['c0'].shape==(m,) and np.isfinite(ac['c0']).all() and (ac['c0']>=0).all(),elements=m)
    checks.exact(label+'regional FFTT has no added interface charge',ac['base'][b:b+m],ac['c0'])
    checks.exact(label+'regional FFTT dynamic coefficient zero',ac['k4'][b:b+m],np.zeros(m))
    checks.exact(label+'no regional closure',ac['closed_mask'][b:b+m],np.zeros(m,bool))
    for scenario in ('open','i95'):
        values=ac['baseline_'+scenario]
        checks.require(label+'complete nonnegative baseline '+scenario,values.shape==(m,) and np.isfinite(values).all() and (values>=0).all(),elements=m)
    unique,starts,counts=np.unique(regions,return_index=True,return_counts=True)
    native=np.arange(m)-np.repeat(starts,counts)
    for key,expected in (('pair_key',keys),('region_index',regions),('native_index',native),
        ('source_node',sources),('target_node',targets),('c0',ac['c0']),
        ('baseline_open',ac['baseline_open']),('baseline_i95',ac['baseline_i95'])):
        checks.exact(label+'pair ledger '+key,pairs[key].to_numpy(),expected)
    identity=native_array_hash({k:ac[k] for k in ('tail','head','blocked','edge_ids','node_ids')})
    checks.require(label+'runtime graph identity',identity==report['graph_identity'])
    checks.require(label+'same original global OD',report['OD_identity']==od_identity)
    checks.require(label+'report full counts',report['backbone_count']==b and report['active_pairs']==m and report['active_regions']==len(unique))
    bid=array_hash(dict(tail=ac['tail'][:b],head=ac['head'][:b],ids=ac['edge_ids'][:b],base=ac['base'][:b],k4=ac['k4'][:b],closed=ac['closed_mask'][:b]))
    checks.require(label+'reported B coordinate identity',bid==report['backbone_identity'])
    checks.require(label+'support frozen before labels',report['support_frozen_before_labels'] is True)
    return unique,starts,counts


def audit_region_copy(common,local,record,ac,edges,begin,end,sid,checks):
    """Exact detailed-region slices certify ownership and absence of double charges."""
    prefix=sid+' region '+str(record['region_index'])+' ';nodes=np.unique(np.r_[common['tail'][edges],common['head'][edges]])
    checks.require(prefix+'nonempty edges and support',len(edges)>0 and end>begin)
    checks.exact(prefix+'full owned P edges',local['P_edge_indices'],edges)
    checks.exact(prefix+'full incident P nodes',local['P_node_indices'],nodes)
    checks.exact(prefix+'exact edge IDs',local['edge_ids'],common['edge_ids'][edges])
    checks.exact(prefix+'exact node IDs',local['node_ids'],common['node_ids'][nodes])
    for key in ('tail','head'):
        checks.exact(prefix+'local '+key,local[key],np.searchsorted(nodes,common[key][edges]))
    for key in ('base','k4'):
        checks.exact(prefix+'cost ownership once '+key,local[key],common[key][edges])
    for key in ('pair_source','pair_target'):
        endpoints=ac[key][begin:end];at=np.searchsorted(nodes,endpoints)
        checks.require(prefix+key+' retained bounds',np.all(at<len(nodes)),elements=len(at))
        checks.exact(prefix+key+' local/global endpoint identity',nodes[at],endpoints)
        checks.exact(prefix+key+' local coordinate',local[key],at)
    for key in ('pair_keys','c0','baseline_open','baseline_i95'):
        checks.exact(prefix+'full '+key,local[key],ac[key][begin:end])
    checks.require(prefix+'finite lifted c0',np.isfinite(local['c0']).all() and (local['c0']>=0).all(),elements=end-begin)
    checks.require(prefix+'exact local payload identity',array_hash(local)==record['input_identity'])
    identity=native_array_hash(dict(tail=local['tail'],head=local['head'],blocked=np.zeros(len(nodes),bool),
        edge_ids=local['edge_ids'],node_ids=local['node_ids']))
    checks.require(prefix+'local graph/physical-transit identity',identity==record['graph_identity'])
    expected_capacity=float(np.median(common['capacity'][edges]))
    checks.exact(prefix+'capacity reference from once-scaled P',local['physical_capacity_reference'],np.array(expected_capacity))
    checks.require(prefix+'capacity reference receipt',record['physical_capacity_reference']==expected_capacity)
    checks.require(prefix+'complete shortest-path lift receipt',record['all_pairs_lift'] is True and
        record['pairs']==end-begin and record['edges']==len(edges) and record['nodes']==len(nodes) and
        np.isfinite(record['local_FFTT_seconds']) and record['local_FFTT_seconds']>0,elements=end-begin)
    return dict(suite_id=sid,region_index=int(record['region_index']),physical_edges=len(edges),nodes=len(nodes),
        pairs=end-begin,zero_c0_pairs=int((local['c0']==0).sum()),
        physical_cost_ownership='EXACT_P_SLICE_INCLUDING_ONE_REGIONAL_INCIDENCE_CHARGE',
        AC_interface_added_charge=0,c0_status='EXACT_COPY_OF_FROZEN_LOCAL_SHORTEST_PATH_COST_VECTOR',
        lift_evidence='HASH_BOUND_REGION_RECEIPT_ALL_PAIRS_LIFT_AND_FINITE_C0',shortest_path_replayed=False)


class Audit:
    def __init__(self, config, audit_id):
        self.config_path = Path(config).resolve(strict=True)
        self.c = read_json(self.config_path)
        if self.c["phase_id"] != "PACNHN_V1" or not self.c["tasks"] or "topology" not in self.c["tasks"]:
            raise ValueError("Explicit approved topology campaign configuration required")
        if self.c["suite_ids"] != ["s02", "s03", "s04", "s05", "s06"]:
            raise ValueError("Five-suite audit matrix differs")
        if not audit_id or Path(audit_id).name != audit_id:
            raise ValueError("An explicit one-component audit ID is required")
        self.root = Path(self.c["output_root"]).resolve(strict=True)
        self.dst = self.root/"audits/topology_cost_identity"/audit_id
        self.dst.mkdir(parents=True, exist_ok=False)
        self.started = time.time()
        self.bindings, self.expected, self.checks = {}, {}, Checks()
        self.peak_rss, self.minimum_available = 0, 2**63-1
        if sha(self.c["contract_path"]) != self.c["contract_sha256"]:
            raise ValueError("Approved contract hash changed")

    def guard(self):
        if time.time() >= min(self.started+600, self.c["maximum_cumulative_stage_deadlines"][0]):
            raise TimeoutError("Bounded independent audit clock reached")
        rss = psutil.Process().memory_info().rss
        available = psutil.virtual_memory().available
        self.peak_rss, self.minimum_available = max(self.peak_rss, rss), min(self.minimum_available, available)
        if rss > self.c["resources"]["tree_rss_bytes"] or available < self.c["resources"]["minimum_available_bytes"]:
            raise MemoryError("Independent audit resource guard")

    def bind(self, path, role, *, expected=None, inherited=False):
        p = Path(path).resolve(strict=True)
        if inherited:
            if str(p) not in self.expected:
                raise ValueError("Audited source absent from accepted binding index: " + str(p))
            if expected is not None and expected != self.expected[str(p)]:
                raise ValueError("Accepted source/index hash conflict")
            expected = self.expected[str(p)]
        digest = sha(p)
        if expected is not None and digest != expected:
            raise ValueError("Independent audit source hash mismatch: " + str(p))
        stat = p.stat()
        self.bindings[str(p)] = dict(path=str(p), role=role, sha256=digest, expected_sha256=expected,
                                    bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        return p

    def arrays(self, path, role, **kwargs):
        p = self.bind(path, role, **kwargs)
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    def frame(self, path, role, **kwargs):
        return pd.read_parquet(self.bind(path, role, **kwargs))

    def initialize_sources(self):
        self.bind(self.config_path, "EXPLICIT_AUDIT_CONFIGURATION")
        self.bind(__file__, "INDEPENDENT_AUDIT_IMPLEMENTATION")
        topology_bindings = self.bind(self.root/"manifests/inputs_topology.json", "ORIGINAL_TOPOLOGY_INPUT_BINDINGS")
        for row in read_json(topology_bindings):
            key = str(Path(row["path"]).resolve())
            self.expected[key] = row["sha256"]
        seal = Path(self.c["input_versions"]["prior_publication"])/"IMMUTABLE_SOURCE_SEAL.csv"
        self.bind(seal, "ACCEPTED_INHERITED_SOURCE_SEAL", expected=self.c["topology"]["source_seal_sha256"])
        rows = pd.read_csv(seal, usecols=["native_path", "sha256", "status"])
        self.checks.require("Accepted source seal PASS", rows.status.eq("PASS").all(), elements=len(rows))
        for row in rows.itertuples():
            key = str(Path(row.native_path).resolve())
            if key in self.expected and self.expected[key] != row.sha256:
                raise ValueError("Topology/accepted source hash conflict: " + key)
            self.expected[key] = row.sha256

    def run(self):
        self.guard()
        self.initialize_sources()
        inp, checks = self.c["input_versions"], self.checks
        report = read_json(self.bind(self.root/"topology/RESULT.json", "COMPLETED_COMMON_P_CERTIFICATE"))
        checks.require("Completed common P certificate", report["status"] == "PASS")
        a = self.arrays(self.root/"topology/common_graph.npz", "COMPLETE_COMMON_P_ARRAYS")
        od = self.arrays(self.root/"topology/common_od.npz", "COMPLETE_COMMON_OD_ARRAYS")
        p = self.arrays(self.root/"topology/partitions.npz", "COMPLETE_FIVE_PARTITION_ARRAYS")
        checks.require("Common graph payload hash", array_hash(a) == report["graph_array_identity"])
        checks.require("Common OD payload hash", array_hash(od) == report["OD_array_identity"])
        checks.require("Partition payload hash", array_hash(p) == report["partitions_identity"])
        checks.require("Runtime graph hash", native_array_hash({k:a[k] for k in ("tail", "head", "blocked", "edge_ids", "node_ids")}) == report["graph_identity"])
        checks.require("Runtime OD hash", native_array_hash({k:od[k] for k in ("origins", "offsets", "destinations", "demand", "serve")}) == report["OD_identity"])
        p5 = Path(inp["P5_frozen_runtime"])
        p3 = Path(inp["P3_suite_root"])/"suites/s03/parquet"
        links = self.frame(p5/"runtime_links.parquet", "EXACT_ACCEPTED_S03_LINK_FIELDS", inherited=True)
        nodes = self.frame(p5/"retained_nodes.parquet", "EXACT_ACCEPTED_S03_NODES", inherited=True)
        packed = self.arrays(p5/"runtime_arrays.npz", "EXACT_ACCEPTED_PACKED_RUNTIME", inherited=True)
        pub = Path(inp["prior_publication"])
        maskmeta = read_json(self.bind(pub/"closure/MASK_FREEZE.json", "ACCEPTED_I95_SCOPE", inherited=True))
        mask = self.arrays(pub/"closure/availability.npz", "ACCEPTED_I95_TYPED_MASK", inherited=True, expected=maskmeta["mask_sha256"])
        bprobes, old_bi = audit_backbone(a, links, nodes, packed, mask, checks)
        bprobes.to_csv(self.dst/"backbone_cost_probes.csv", index=False, float_format="%.17g")
        checks.require("All 87582 backbone coordinates audited", len(old_bi) == self.c["topology"]["expected_backbone_count"], elements=len(old_bi))
        geometry = self.frame(self.root/"topology/backbone.parquet", "FROZEN_B_METADATA_AND_REPRESENTATIVES")
        bn = nodes.set_index("node_id")
        for key in ("link_id", "tail_node_id", "head_node_id", "role", "runtime_class", "dynamic_model_id"):
            checks.exact("B frozen metadata " + key, geometry[key].to_numpy(str), links.iloc[old_bi][key].to_numpy(str))
        for endpoint in ("tail", "head"):
            for coordinate in ("longitude", "latitude"):
                checks.exact("B frozen " + endpoint + " " + coordinate, geometry[endpoint+"_"+coordinate],
                             links.iloc[old_bi][endpoint+"_node_id"].map(bn[coordinate]))
        self.guard()
        p4 = read_json(self.bind(inp["P4_physical_config"], "ACCEPTED_P4_GRAPH_AND_OD_IDENTITIES", inherited=True))
        pi = p4["input_versions"]
        gm = read_json(self.bind(pi["graph_metadata_json"], "ACCEPTED_PHYSICAL_GRAPH_METADATA", inherited=True))
        physical = self.arrays(pi["graph_npz"], "ACCEPTED_FULL_PHYSICAL_GRAPH", inherited=True)
        checks.require("Accepted physical graph semantic identity", gm["graph_hash"] == gm["assignment_graph_hash"] == pi["graph_identity"])
        caphash = hashlib.sha256(np.ascontiguousarray(physical["capacity"], dtype=np.float64).tobytes()).hexdigest()
        checks.require("Clean prepared capacity payload SHA256", caphash == pi["clean_capacity_payload_sha256"])
        own = self.frame(p5/"physical_link_ownership.parquet", "ACCEPTED_PHYSICAL_COST_OWNERSHIP", inherited=True).sort_values("physical_arc_idx", kind="stable").reset_index(drop=True)
        checks.exact("Accepted ownership full physical index", own.physical_arc_idx, np.arange(len(physical["tail"])))
        checks.exact("Accepted ownership full physical IDs", own.physical_arc_id.astype(str), physical["arc_idx_to_directed_arc_id"].astype(str))
        checks.exact("Accepted physical owner count once", own.owner_count, np.ones(len(own), np.int64))
        membership = self.frame(p3/"bacn_membership.parquet", "FROZEN_PHYSICAL_B_QUOTIENT", inherited=True)
        charges = self.frame(p5.parents[1]/"registries/charges.parquet", "ACCEPTED_CURRENT_CHARGE_REGISTRY", inherited=True)
        charges = charges.loc[charges.suite_id.eq("s03") & charges.model_variant.eq("LOW_MEAN_STATIC_INCIDENCE_V1")]
        checks.require("Primary charge registry unique", charges.bacn_id.is_unique, elements=len(charges))
        chi = charges.set_index("bacn_id").chi_s
        per_node = np.zeros(len(a["node_ids"]))
        retained = np.arange(int(a["backbone_node_count"]))
        per_node[retained] = pd.Series(a["node_ids"][retained]).map(chi).to_numpy(float)
        checks.require("Every retained B charge source valid", np.isfinite(per_node).all() and (per_node >= 0).all(), elements=len(retained))
        for side in ("tail", "head"):
            checks.exact("Frozen B " + side + " charge registry half", a[side+"_half_charge_s"][:len(old_bi)], per_node[a[side][:len(old_bi)]]/2)
        seq = self.frame(p5/"macro_sequences.parquet", "ACCEPTED_MACRO_SERIES_CHARGE_OWNERSHIP", inherited=True)
        inside = seq.loc[seq.internal_node_after.astype(str).ne("")].copy()
        checks.require("No repeated internal node within one macro", not inside.duplicated(["corridor_id", "internal_node_after"]).any(), elements=len(inside))
        inside["charge"] = inside.internal_node_after.map(chi)
        checks.require("Every macro internal charge bound", np.isfinite(inside.charge).all(), elements=len(inside))
        summed = inside.groupby("corridor_id").charge.sum()
        macro = links.iloc[old_bi].role.eq("MACRO").to_numpy()
        expected_h = links.iloc[old_bi].link_id.map(summed).fillna(0.).to_numpy()
        checks.close("Macro internal charge counted once per traversal", a["fixed_internal_H_s"][:len(old_bi)], expected_h)
        checks.exact("Nonmacro internal charge zero", a["fixed_internal_H_s"][:len(old_bi)][~macro], np.zeros(int((~macro).sum())))
        rprobes, residual = audit_residual(a, physical, own, per_node, capacity_multiplier=self.c["topology"]["capacity_multiplier"], checks=checks)
        rprobes.to_csv(self.dst/"residual_cost_probes.csv", index=False, float_format="%.17g")
        omroot = Path(pi["accepted_aggregated_od_root"])/"derived_inputs"
        om = read_json(self.bind(omroot/"aggregated_od_metadata.json", "ACCEPTED_ORIGINAL_OD_METADATA", inherited=True))
        original = self.arrays(omroot/"aggregated_od.npz", "ACCEPTED_EXACT_ORIGINAL_OD_AGGREGATE", inherited=True, expected=om["aggregate_npz_sha256"])
        self.bind(omroot/"original_to_aggregated_od_index.npz", "ACCEPTED_ORIGINAL_ROW_REVERSIBLE_MAP", inherited=True, expected=om["mapping_npz_sha256"])
        checks.require("Original OD semantic identity", om["od_hash"] == pi["od_identity"])
        for key in ("origin_group_node_idx", "destination_node_idx", "demand", "original_row_multiplicity"):
            checks.exact("Original OD exact " + key, od[key], original[key])
        checks.exact("Original OD exact grouping offsets", od["offsets"], original["origin_offsets"])
        quotient = audit_mapping(a, od, physical, membership, checks)
        produced_own = self.frame(self.root/"topology/physical_cost_ownership.parquet", "EXPORTED_FULL_PHYSICAL_OWNERSHIP_MAPPING")
        for column in own.columns:
            checks.exact("Exported physical ownership " + column, produced_own[column], own[column])
        expected_owner = np.full(len(own), -1, np.int64)
        bowner = own.ownership_role.eq("INTER_B").to_numpy()
        expected_owner[bowner] = pd.Index(a["edge_ids"][:len(old_bi)]).get_indexer(own.loc[bowner, "final_link_id"])
        expected_owner[residual] = np.flatnonzero(a["edge_role"] == 1)
        checks.require("Every represented source B arc has its frozen B coordinate", (expected_owner[bowner] >= 0).all(), elements=int(bowner.sum()))
        checks.exact("Exported physical cost owner common edge index", produced_own.common_edge_idx, expected_owner)
        produced_sequence = self.frame(self.root/"topology/frozen_macro_sequences.parquet", "EXPORTED_FROZEN_MACRO_SEQUENCE")
        for column in seq.columns:
            checks.exact("Exported macro sequence " + column, produced_sequence[column], seq[column])
        node_map = self.frame(self.root/"topology/physical_node_mapping.parquet", "EXPORTED_PHYSICAL_UID_QUOTIENT_MAPPING")
        checks.exact("Exported node mapping physical indices", node_map.physical_node_idx, np.arange(len(quotient)))
        checks.exact("Exported node mapping physical UIDs", node_map.physical_node_uid.astype(str), physical["node_idx_to_uid"].astype(str))
        checks.exact("Exported node mapping quotient indices", node_map.common_node_idx, quotient)
        checks.exact("Exported node mapping disposition", node_map.disposition, np.where(quotient < 0, "CONSUMED_MACRO_INTERIOR", "RETAINED"))
        od_map = self.frame(self.root/"topology/OD_endpoint_mapping.parquet", "EXPORTED_EXACT_GLOBAL_OD_MAPPING")
        endpoints = np.unique(np.r_[original["origin_group_node_idx"], original["destination_node_idx"]])
        endpoint_uids = physical["node_idx_to_uid"].astype(str)[endpoints]
        checks.exact("Exported OD original endpoint indices", od_map.physical_node_idx, endpoints)
        checks.exact("Exported OD original endpoint UIDs", od_map.physical_node_uid.astype(str), endpoint_uids)
        checks.exact("Exported OD underlying road indices", od_map.road_node_idx, quotient[endpoints])
        checks.exact("Exported OD underlying road IDs", od_map.road_node_id.astype(str), a["node_ids"][quotient[endpoints]])
        for role in ("origin", "destination"):
            expected_names = np.char.add("n8_od_"+role+":", endpoint_uids)
            checks.exact("Exported OD " + role + " names", od_map[role+"_node_id"].astype(str), expected_names)
            checks.exact("Exported OD " + role + " indices", od_map[role+"_node_idx"], pd.Index(a["node_ids"]).get_indexer(expected_names))
        self.guard()
        costledger = self.frame(self.root/"topology/cost_ownership.parquet", "COMPLETE_P_COST_OWNERSHIP_LEDGER")
        partitions = audit_charge_partition(a, p, costledger, checks)
        partitions.to_csv(self.dst/"charge_partition_algebra.csv", index=False)
        s03_index = list(p["suite_ids"]).index("s03")
        rslots = np.flatnonzero(a["edge_role"] == 1)
        checks.exact("s03 inherited residual region ownership", p["s03_region_ids"][p["edge_region"][s03_index, rslots]], own.loc[own.ownership_role.eq("REGION"), "original_owner_id"].to_numpy(str))
        physical_uids = physical["node_idx_to_uid"].astype(str)
        in_b = np.isin(physical_uids, membership.partC_rev1_node_uid.to_numpy(str))
        bridge = in_b[physical["tail"][residual]] & in_b[physical["head"][residual]]
        partition_nodes = self.frame(self.root/"topology/rebuilt_node_partitions.parquet", "REBUILT_FIXED_UNIVERSE_NODE_PARTITIONS")
        for si, sid in enumerate(p["suite_ids"]):
            if sid == "s03": continue
            nr = partition_nodes.loc[partition_nodes.suite_id.eq(sid)]
            checks.require(str(sid)+" unique physical node partition", nr.physical_node_uid.is_unique, elements=len(nr))
            assignment = pd.Series(physical_uids).map(nr.set_index("physical_node_uid").primary_region_id).fillna("").to_numpy()
            checks.require(str(sid)+" full non-B physical partition", (assignment[~in_b] != "").all() and (assignment[in_b] == "").all(), elements=len(physical_uids))
            tail, head = physical["tail"][residual], physical["head"][residual]
            labels = np.where(in_b[tail], assignment[head], assignment[tail])
            labels[bridge] = np.char.add("pacnhn_"+str(sid)+"_Bbridge_arc_", physical["arc_idx_to_directed_arc_id"].astype(str)[residual][bridge])
            checks.exact(str(sid)+" complete directed residual ownership", p[str(sid)+"_region_ids"][p["edge_region"][si, rslots]], labels)
        # The AC build has not run yet during the reference stage. Bind and name
        # the reviewed implementation; never call an absent future file passed.
        repo = Path(__file__).resolve().parents[2]
        self.bind(repo/"partC_rev1_src/pacnhn_v1/support.py", "REVIEWED_AC_LOCAL_COST_COPY_IMPLEMENTATION")
        ac_status = {sid: "PRESENT_REQUIRES_SEPARATE_GENERATED_ARTIFACT_AUDIT" if (self.root/"acnhn"/sid/"graph.npz").exists()
                     else "NOT_YET_GENERATED" for sid in self.c["suite_ids"]}
        return dict(status="PASS", phase_id=self.c["phase_id"], step_id="INDEPENDENT_TOPOLOGY_COST_IDENTITY_AUDIT", run_id=self.c["run_id"],
            common_graph_identity=report["graph_identity"], common_graph_array_identity=report["graph_array_identity"],
            common_OD_identity=report["OD_identity"], common_OD_array_identity=report["OD_array_identity"],
            partitions_identity=report["partitions_identity"], all_backbone_coordinates=len(old_bi), all_residual_physical_arcs=len(residual),
            original_OD_aggregate_rows=len(od["demand"]), original_OD_rows=int(od["original_row_multiplicity"].sum()),
            original_total_demand=float(od["demand"].sum()), physical_graph_identity=pi["graph_identity"], original_OD_identity=pi["od_identity"],
            clean_prepared_capacity_payload_sha256=caphash, expected_clean_prepared_capacity_payload_sha256=pi["clean_capacity_payload_sha256"],
            used_residual_capacity_payload_sha256=hashlib.sha256(np.ascontiguousarray(a["capacity"][rslots], dtype=np.float64).tobytes()).hexdigest(),
            used_capacity_factor=self.c["topology"]["capacity_multiplier"], factor_application_count=1,
            physical_time_unit_source="MINUTES", output_time_unit="SECONDS", physical_time_conversion_factor=60,
            full_P_physical_ownership_rows=len(own), residual_B_to_B_arcs=int(bridge.sum()),
            all_residual_self_loops=int((a["tail"][rslots] == a["head"][rslots]).sum()),
            charge_algebra="ONE_FROZEN_B_EDGE_TERM_OR_ONE_OWNED_RESIDUAL_EDGE_TERM; REGIONAL_HALF_CHARGES_INSIDE_LOCAL_VALUE",
            AC_generated_artifact_status=ac_status,
            AC_implementation_review="support.py local base/k4 are slices of P; B base/k4/mask copied once; no additional endpoint charges",
            AC_generated_cost_identity_certified=False, assignment_executed=False, training_executed=False,
            physical_vertex_transit_allowed=True, logical_OD_role_transit_allowed=False,
            completeness="FULL_NATIVE_ARRAY_POPULATIONS; NO_COORDINATE_SAMPLING")

    def execute(self):
        try:
            report = self.run()
        except Exception as exc:
            import traceback
            report = dict(status="FAIL", exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            # Only reads have touched external inputs. Detect concurrent source
            # replacement without rehashing unchanged large arrays a second time.
            changed = [p for p, r in self.bindings.items() if
                       Path(p).stat().st_size != r["bytes"] or Path(p).stat().st_mtime_ns != r["mtime_ns"]]
            if changed:
                report.update(status="FAIL", concurrently_changed_sources=changed)
            report.update(audit_directory=str(self.dst), complete_wall_seconds=time.time()-self.started,
                          checks=len(self.checks.rows), source_binding_count=len(self.bindings),
                          peak_audit_process_rss_bytes=self.peak_rss, minimum_available_bytes=self.minimum_available)
            write_json(self.dst/"REPORT.json", report)
            write_json(self.dst/"SOURCE_BINDINGS.json", list(self.bindings.values()))
            pd.DataFrame(self.checks.rows).to_csv(self.dst/"checks.csv", index=False, float_format="%.17g")
        if report["status"] != "PASS": raise ValueError("Audit concurrent-source freeze failed")
        return report


class GeneratedACAudit(Audit):
    """Full generated-artifact gate; no duplicate shortest-path replay."""
    def __init__(self,config,audit_id,p_audit_report,p_audit_sha256):
        super().__init__(config,audit_id)
        if not p_audit_report or not p_audit_sha256:
            raise ValueError('Exact prior full-P audit path and SHA256 are required')
        self.p_audit_report=Path(p_audit_report).resolve(strict=True)
        self.p_audit_sha256=p_audit_sha256
        write_json(self.dst/'INTENT.json',dict(phase_id=self.c['phase_id'],run_id=self.c['run_id'],
            step_id='GENERATED_AC_FULL_ARRAY_AUDIT',config_path=str(self.config_path),config_sha256=sha(self.config_path),
            input_versions=self.c['input_versions'],prior_P_audit=str(self.p_audit_report),prior_P_audit_sha256=p_audit_sha256,
            task_list=['VERIFY_P_PAYLOAD_AND_PRIOR_COST_GATE','CHECK_FIVE_COMPILED_AC_GRAPHS','CHECK_ALL_LOCAL_ARRAYS_AND_OWNERS',
                'VERIFY_FROZEN_TRACE_SUPPORT_AND_C0_RECEIPTS','VERIFY_EXACT_OD_MEMBERSHIP_AND_PROJECTION','BIND_COMPLETE_WITNESS_EVIDENCE'],
            expected_outputs=['REPORT.json','SOURCE_BINDINGS.json','checks.csv','generated_ac_suites.csv','generated_regions.csv'],
            acceptance_tests=['all_native_B_coordinates','all_local_cost_ownership_once','all_frozen_active_pairs',
                'common_OD_and_roles','exact_mask','all_served_OD_projection_conservation','hash_bound_lift_receipts'],
            resources=self.c['resources'],deadline_epoch=min(self.started+600,self.c['maximum_cumulative_stage_deadlines'][0]),
            shortest_path_replay=False,assignment_execution=False,training_execution=False))

    def run(self):
        from .oracle import Demand,conservation
        from types import SimpleNamespace
        self.guard();checks=self.checks
        self.bind(self.config_path,'EXPLICIT_GENERATED_AC_AUDIT_CONFIG')
        self.bind(__file__,'INDEPENDENT_GENERATED_AC_ARRAY_AUDIT_SOURCE')
        prior=read_json(self.bind(self.p_audit_report,'ACCEPTED_FULL_POPULATION_P_COST_AUDIT',expected=self.p_audit_sha256))
        checks.require('Prior full-P native cost audit PASS',prior.get('status')=='PASS' and
            prior.get('completeness')=='FULL_NATIVE_ARRAY_POPULATIONS; NO_COORDINATE_SAMPLING')
        topo=read_json(self.bind(self.root/'topology/RESULT.json','CURRENT_P_TOPOLOGY_HEADER'))
        checks.require('Current P topology PASS',topo['status']=='PASS')
        a=self.arrays(self.root/'topology/common_graph.npz','FROZEN_FULL_P_GRAPH')
        od=self.arrays(self.root/'topology/common_od.npz','FROZEN_FULL_GLOBAL_OD')
        partitions=self.arrays(self.root/'topology/partitions.npz','FROZEN_FULL_FIVE_PARTITIONS')
        for name,actual,tkey,pkey in [('P graph',array_hash(a),'graph_array_identity','common_graph_array_identity'),
            ('global OD',array_hash(od),'OD_array_identity','common_OD_array_identity'),
            ('five partitions',array_hash(partitions),'partitions_identity','partitions_identity')]:
            checks.require(name+' same payload as independent P audit',actual==topo[tkey]==prior[pkey])
        graph_identity=native_array_hash({k:a[k] for k in ('tail','head','blocked','edge_ids','node_ids')})
        od_identity=native_array_hash({k:od[k] for k in ('origins','offsets','destinations','demand','serve')})
        checks.require('Current P graph native identity',graph_identity==topo['graph_identity']==prior['common_graph_identity'])
        checks.require('Current global OD native identity',od_identity==topo['OD_identity']==prior['common_OD_identity'])
        checks.exact('Five partition suite order',partitions['suite_ids'],np.asarray(self.c['suite_ids']))
        checks.require('Fixed full native B count',int((a['edge_role']==0).sum())==87582==prior['all_backbone_coordinates'])
        freeze_path=self.root/'SUPPORT_FREEZE.json';witness_path=self.root/'COMPLETE_WITNESS_AUDIT.json'
        freeze=read_json(self.bind(freeze_path,'FROZEN_COMPLETE_GENERATED_SUPPORT'))
        freeze_sha=self.bindings[str(freeze_path.resolve())]['sha256']
        witness=read_json(self.bind(witness_path,'COMPLETED_ALL_OD_WITNESS_GATE'))
        checks.require('Complete support/witness hash chain',freeze['status']=='PASS' and witness['status']=='PASS'
            and witness['support_freeze_sha256']==freeze_sha and witness['graph_identity']==graph_identity)
        frozen={r['suite_id']:r for r in freeze['suites']}
        checks.require('Exactly five support suites',len(frozen)==len(freeze['suites'])==5 and set(frozen)==set(self.c['suite_ids']))
        witness_checks={r['scenario']:r for r in witness['checks']}
        checks.require('Exactly open and represented-I95 witness checks',len(witness['checks'])==2 and set(witness_checks)=={'open','i95'})
        # Bind the exact manifests consumed by the completed witness job. The
        # forest arrays themselves were checked by that gate, not replayed here.
        index_path=self.root/'manifests/inputs_support.json'
        evidence=read_json(self.bind(index_path,'LATEST_COMPLETED_SUPPORT_WITNESS_INPUT_BINDINGS'))
        expected={str(Path(r['path']).resolve()):r['sha256'] for r in evidence}
        checks.require('Support source-index freeze identity',expected.get(str(freeze_path.resolve()))==freeze_sha)
        refs={};reference_rows=[]
        for si,scenario in enumerate(('open','i95')):
            self.guard();root=self.root/'references'/scenario
            result=read_json(self.bind(root/'RESULT.json','FROZEN_P_REFERENCE_RESULT'))
            checks.require(scenario+' exact support reference result',result==freeze['source_references'][si])
            checks.require(scenario+' reference graph/fresh certificate',result['graph_identity']==graph_identity and
                result['final_certificate_fresh'] is True and result['certificate']['fresh'] is True and result['trace_count']==5)
            flow=self.arrays(root/'final_flow.npz','CERTIFIED_REFERENCE_FLOW')['flow']
            checks.require(scenario+' reference flow state identity',array_hash({'flow':flow})==result['certificate']['state_hash'])
            coverage=self.arrays(root/'coverage.npz','CERTIFIED_REFERENCE_OD_MEMBERSHIP',expected=expected.get(str((root/'coverage.npz').resolve())))
            serve=coverage['serve']
            checks.require(scenario+' full boolean served OD vector',serve.dtype==bool and serve.shape==od['serve'].shape and serve.any(),elements=len(serve))
            served=Demand(od['origins'],od['offsets'],od['destinations'],od['demand'],serve)
            checks.require(scenario+' served OD identity',served.identity==result['OD_identity'])
            traces=self.arrays(root/'final_traces.npz','CERTIFIED_STATE_REGIONAL_TRACES')
            ever=self.arrays(root/'ever_active_pairs.npz','FULL_ACCEPTED_TRACE_SUPPORT_UNION')
            checks.require(scenario+' complete trace suite fields',set(traces)=={f'{kind}_{s}' for s in range(5) for kind in ('keys','values')})
            checks.require(scenario+' complete accepted-support suite fields',set(ever)=={f'keys_{s}' for s in range(5)})
            manifest_path=root/'initial_witnesses/MANIFEST.json'
            checks.require(scenario+' witness manifest is indexed',str(manifest_path.resolve()) in expected)
            manifest=read_json(self.bind(manifest_path,'BOUND_COMPLETE_OD_FOREST_MANIFEST',expected=expected[str(manifest_path.resolve())]))
            enabled=~a['closed_mask'] if scenario=='i95' else np.ones(len(a['tail']),bool)
            checks.require(scenario+' witness graph OD mask cost identity',manifest['graph_identity']==graph_identity and
                manifest['OD_identity']==od_identity and manifest['enabled_identity']==array_hash({'enabled':enabled}) and
                manifest['cost_identity']==array_hash({'cost':a['base']}))
            wc=witness_checks[scenario]
            checks.require(scenario+' full OD witness acceptance',wc['checked_OD_rows']==int(serve.sum()) and
                manifest['traced_OD_rows']==wc['checked_OD_rows'] and
                all(wc[k] is True for k in ('all_five_resolutions','all_directed_P_arcs_and_masks_valid','all_segments_in_frozen_support')) and
                len(wc['projected_segments'])==5,elements=int(serve.sum()))
            checks.exact(scenario+' initial forest segment-count receipt',wc['projected_segments'],manifest['regional_segments'])
            refs[scenario]=dict(flow=flow,serve=serve,served=served,balance=served.balance(len(a['node_ids'])),traces=traces,ever=ever)
            reference_rows.append(dict(scenario=scenario,served_rows=int(serve.sum()),served_demand=served.total,
                reference_state_hash=result['certificate']['state_hash'],complete_witness_rows=wc['checked_OD_rows']))
        b=87582;ports=np.flatnonzero(a['edge_role']==2);suite_rows=[];region_rows=[]
        for si,sid in enumerate(self.c['suite_ids']):
            self.guard();root=self.root/'acnhn'/sid
            report=read_json(self.bind(root/'REPORT.json','GENERATED_AC_GRAPH_REPORT'))
            checks.require(sid+' report equals support freeze',report==frozen[sid])
            checks.require(sid+' exact graph consumed by complete witness gate',
                expected.get(str((root/'graph.npz').resolve()))==report['graph_sha256'])
            ac=self.arrays(root/'graph.npz','FULL_GENERATED_AC_GRAPH',expected=report['graph_sha256'])
            pairs=self.frame(root/'pairs.parquet','FULL_GENERATED_AC_PAIR_LEDGER')
            regions,starts,counts=audit_compiled_ac(a,ac,report,pairs,sid,od_identity,checks)
            expected_keys=np.union1d(refs['open']['ever'][f'keys_{si}'],refs['i95']['ever'][f'keys_{si}'])
            checks.exact(sid+' exact accepted open/I95 support union',ac['pair_keys'],expected_keys)
            records={int(r['region_index']):r for r in report['region_records']}
            checks.require(sid+' exact unique active region receipt population',len(records)==len(report['region_records']) and set(records)==set(regions))
            owner=partitions['edge_region'][si];order=np.flatnonzero(owner>=0)
            order=order[np.argsort(owner[order],kind='stable')];labels=owner[order]
            occurrences=np.zeros(len(a['tail']),np.uint8)
            for region,begin,count in zip(regions,starts,counts):
                self.guard();region=int(region);begin=int(begin);end=begin+int(count)
                lo,hi=np.searchsorted(labels,[region,region+1]);edges=order[lo:hi]
                record=records[region];directory=self.root/'regions'/sid/f'{region:06d}'
                sidecar=read_json(self.bind(directory/'REGION.json','EXACT_LOCAL_LIFT_AND_COST_RECEIPT'))
                checks.require(sid+' region '+str(region)+' frozen receipt identity',sidecar==record and
                    Path(record['path']).resolve()==(directory/'graph.npz').resolve())
                local=self.arrays(directory/'graph.npz','FULL_LOCAL_PHYSICAL_GRAPH_AND_PAIR_COSTS',expected=record['sha256'])
                region_rows.append(audit_region_copy(a,local,record,ac,edges,begin,end,sid,checks))
                occurrences[edges]+=1
                del local
            selected=np.isin(owner,regions)
            checks.exact(sid+' each active local physical edge appears once',occurrences,selected.astype(np.uint8))
            checks.require(sid+' local edge copies never contain B or ports',not occurrences[a['edge_role']!=1].any())
            residuals={};reach={r['scenario']:r for r in report['coverage']}
            checks.require(sid+' complete reachability report',len(report['coverage'])==2 and set(reach)=={'open','i95'})
            for scenario in ('open','i95'):
                self.guard();ref=refs[scenario];tk=ref['traces'][f'keys_{si}'];tv=ref['traces'][f'values_{si}']
                checks.require(sid+' '+scenario+' strict final trace coordinates',tk.dtype==np.dtype('uint64') and tk.shape==tv.shape and
                    np.all(tk[1:]>tk[:-1]) and np.isfinite(tv).all() and (tv>=0).all(),elements=len(tk))
                at=np.searchsorted(ac['pair_keys'],tk)
                checks.require(sid+' '+scenario+' trace support membership',np.all(at<len(ac['pair_keys'])),elements=len(tk))
                checks.exact(sid+' '+scenario+' exact trace keys',ac['pair_keys'][at],tk)
                baseline=np.zeros(len(ac['pair_keys']));baseline[at]=tv
                checks.exact(sid+' '+scenario+' full baseline including zero coordinates',ac['baseline_'+scenario],baseline)
                cov=self.arrays(root/f'coverage_{scenario}.npz','FULL_AC_OD_REACHABILITY_RECEIPT')
                checks.exact(sid+' '+scenario+' exact served OD membership',cov['serve'],ref['serve'])
                checks.require(sid+' '+scenario+' coverage shapes and nonnegative distances',cov['serve'].dtype==bool and
                    cov['shortest_cost'].shape==od['serve'].shape and not np.isnan(cov['shortest_cost']).any() and
                    np.all(cov['shortest_cost'][np.isfinite(cov['shortest_cost'])]>=0),elements=len(od['serve']))
                checks.exact(sid+' '+scenario+' finite path membership',od['serve']&np.isfinite(cov['shortest_cost']),ref['serve'])
                checks.require(sid+' '+scenario+' coverage receipt count/membership',reach[scenario]['exact_OD_membership'] is True and
                    reach[scenario]['served_rows']==int(ref['serve'].sum()))
                checks.close(sid+' '+scenario+' served demand receipt',[reach[scenario]['served_demand']],[ref['served'].total])
                projected=np.r_[ref['flow'][:b],baseline,ref['flow'][ports]]
                if scenario=='i95':checks.exact(sid+' masked projected flow',projected[ac['closed_mask']],np.zeros(int(ac['closed_mask'].sum())))
                residuals[scenario]=conservation(SimpleNamespace(tail=ac['tail'],head=ac['head'],node_ids=ac['node_ids']),
                    projected,ref['balance'],ref['served'].total)
                del cov,projected
            suite_rows.append(dict(suite_id=sid,backbone_coordinates=b,AC_edges=len(ac['tail']),active_pairs=len(ac['pair_keys']),
                active_regions=len(regions),local_physical_edge_occurrences=int(occurrences.sum()),
                inactive_owned_physical_edges=int(((owner>=0)&~selected).sum()),
                graph_sha256=report['graph_sha256'],graph_identity=report['graph_identity'],OD_identity=report['OD_identity'],
                open_conservation_max=residuals['open']['max_absolute_node_residual'],
                closed_conservation_max=residuals['i95']['max_absolute_node_residual'],status='PASS'))
            pd.DataFrame(suite_rows).to_csv(self.dst/'generated_ac_suites.csv',index=False,float_format='%.17g')
            del ac,pairs
        pd.DataFrame(region_rows).to_csv(self.dst/'generated_regions.csv',index=False,float_format='%.17g')
        return dict(status='PASS',phase_id=self.c['phase_id'],step_id='GENERATED_AC_FULL_ARRAY_AUDIT',run_id=self.c['run_id'],
            support_freeze_sha256=freeze_sha,complete_witness_sha256=self.bindings[str(witness_path.resolve())]['sha256'],
            prior_full_P_cost_audit_sha256=self.p_audit_sha256,common_graph_identity=graph_identity,common_OD_identity=od_identity,
            generated_suites=len(suite_rows),generated_regions=len(region_rows),suites=suite_rows,reference_OD_populations=reference_rows,
            generated_graph_hashes={r['suite_id']:r['graph_sha256'] for r in suite_rows},all_backbone_coordinates_per_suite=b,
            total_active_pairs=sum(r['active_pairs'] for r in suite_rows),all_generated_AC_cost_identity_certified=True,
            AC_generated_cost_identity_certified=True,
            completeness='FULL_GENERATED_AC_AND_LOCAL_ARRAY_POPULATIONS; NO_COORDINATE_SAMPLING',
            proof_scope='COMPLETE_STORED_ARRAY_IDENTITY_OWNERSHIP_SUPPORT_AND_FEASIBLE_PROJECTION; NO_COORDINATE_SAMPLING',
            local_c0_scope='Every c0 equals the hash-bound local shortest-path result and AC regional base; no duplicate shortest-path replay.',
            lift_scope='Exact local P edge/node slices and all pair endpoints; finite c0 and all_pairs_lift from authenticated local solve receipts. Complete OD forest replay accepted through its bound witness gate.',
            shortest_path_replayed=False,assignment_executed=False,training_executed=False,
            cost_ownership='Unchanged B base/k4; regional owned P base/k4 include original half charges once; AC adds no new interface charge.',
            inactive_region_scope='Residual P owners without any frozen active pair have no generated local training graph; counts remain explicit.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit-id", required=True)
    parser.add_argument('--scope',choices=('common-p','generated-ac'),default='common-p')
    parser.add_argument('--p-audit-report')
    parser.add_argument('--p-audit-sha256')
    args = parser.parse_args()
    audit=GeneratedACAudit(args.config,args.audit_id,args.p_audit_report,args.p_audit_sha256) if args.scope=='generated-ac' else Audit(args.config,args.audit_id)
    result = audit.execute()
    print(json.dumps({k:result[k] for k in ("status", "audit_directory", "checks", "complete_wall_seconds")}), flush=True)
