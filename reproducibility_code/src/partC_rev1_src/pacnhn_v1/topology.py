"""One fixed s03 quotient backbone plus exact residual physical multiarcs.

This module constructs data only.  It never performs a national assignment.
The same graph and OD arrays are shared by every resolution.  Partitions change
only the owner of a residual physical edge; they cannot change its endpoints,
cost, availability or identity.  A physical network vertex may be traversed;
only its separate logical origin/destination role nodes are nontransit.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .control import Context, array_hash, read_json, save_npz, write_json

B_ROLES = ("EXACT_PRIMITIVE", "MACRO", "PHYSICAL_MEMBER")
COST_FIELDS = ("tau_s", "k4", "fixed_internal_H_s", "tail_half_charge_s",
               "head_half_charge_s", "structural_fixed_s", "domain_Q")
SUITES = ("s02", "s03", "s04", "s05", "s06")
PRIMARY = "LOW_MEAN_STATIC_INCIDENCE_V1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _strings(value):
    return np.asarray(value, dtype=str)


def _index(index, keys, name):
    result = pd.Index(index).get_indexer(keys).astype(np.int64)
    _require(np.all(result >= 0), "Unmapped " + name)
    return result


def runtime_identities(graph_arrays, od_arrays):
    """Use the inherited Network/Demand header convention for their identities."""
    from partC_rev1_src.phase_n8.common import array_hash as native_array_hash
    return (native_array_hash({k:graph_arrays[k] for k in ("tail", "head", "blocked", "edge_ids", "node_ids")}),
            native_array_hash({k:od_arrays[k] for k in ("origins", "offsets", "destinations", "demand", "serve")}))


def assemble_common(physical, links, nodes, ownership, bacn_membership,
                    charges, original_od, closed_backbone_ids, *,
                    expected_backbone_count, capacity_multiplier):
    """Pure constructor, usable with small fixtures before any source build.

    ``physical`` uses the accepted graph-snapshot array names.  Capacity is the
    accepted prepared capacity; the explicitly supplied benchmark multiplier is
    applied exactly once here.  Output field ``capacity`` is the resulting used
    capacity for residual physical edges and zero (not applicable) elsewhere.
    """
    uid = _strings(physical["node_idx_to_uid"])
    arcid = _strings(physical["arc_idx_to_directed_arc_id"])
    pt = np.asarray(physical["tail"], np.int64)
    ph = np.asarray(physical["head"], np.int64)
    _require(len(pd.Index(uid).unique()) == len(uid), "Duplicate physical node UID")
    _require(len(pd.Index(arcid).unique()) == len(arcid), "Duplicate physical arc ID")
    _require(pt.shape == ph.shape == arcid.shape, "Physical incidence shape differs")
    _require(len(pt) > 0 and min(pt.min(), ph.min()) >= 0 and
             max(pt.max(), ph.max()) < len(uid), "Invalid physical incidence")
    _require(np.isfinite(capacity_multiplier) and capacity_multiplier > 0,
             "Explicit positive used-capacity multiplier required")
    for key in ("free_time", "capacity", "alpha", "beta"):
        a = np.asarray(physical[key], float)
        _require(a.shape == pt.shape and np.isfinite(a).all(), "Invalid physical " + key)
    _require((physical["free_time"] > 0).all() and (physical["capacity"] > 0).all()
             and (physical["alpha"] >= 0).all() and (physical["beta"] == 4).all(),
             "Physical beta=4 and valid BPR parameters required")

    own = ownership.sort_values("physical_arc_idx", kind="stable").reset_index(drop=True).copy()
    _require(np.array_equal(own.physical_arc_idx, np.arange(len(pt))) and
             np.array_equal(_strings(own.physical_arc_id), arcid),
             "Ownership must cover exactly the accepted physical arc order")
    _require(own.owner_count.eq(1).all() and
             set(own.ownership_role) <= {"REGION", "INTER_B", "SUPPRESSED_B"},
             "Unknown or duplicate physical cost ownership")
    backbone = links.loc[links.role.isin(B_ROLES)].copy().reset_index(drop=True)
    _require(len(backbone) == expected_backbone_count and len(backbone) > 0,
             "Frozen backbone coordinate count differs")
    _require(backbone.link_id.is_unique and nodes.node_id.is_unique,
             "Duplicate frozen node/link identity")
    _require((backbone.beta == 4).all(), "Frozen backbone exponent differs")
    for key in COST_FIELDS[:-1]:
        _require(np.isfinite(backbone[key]).all() and backbone[key].ge(0).all(),
                 "Invalid frozen cost field " + key)
    _require(set(own.loc[own.ownership_role.eq("INTER_B"), "final_link_id"]) ==
             set(backbone.link_id), "Backbone physical-member provenance incomplete")

    bnodes = nodes.loc[nodes.node_type.eq("BACN")].copy().reset_index(drop=True)
    _require(len(bnodes) > 0, "No retained backbone nodes")
    bi = pd.Index(_strings(bnodes.node_id))
    bt = _index(bi, _strings(backbone.tail_node_id), "frozen B tail")
    bh = _index(bi, _strings(backbone.head_node_id), "frozen B head")
    membership = bacn_membership[["partC_rev1_node_uid", "bacn_id"]].copy()
    _require(membership.partC_rev1_node_uid.is_unique, "Multiple B quotients for physical UID")
    pidx = _index(uid, _strings(membership.partC_rev1_node_uid), "B physical member")
    quotient = np.full(len(uid), -1, np.int64)
    quotient[pidx] = bi.get_indexer(_strings(membership.bacn_id))
    in_b = np.zeros(len(uid), bool)
    in_b[pidx] = True
    non_b = np.flatnonzero(~in_b)
    quotient[non_b] = len(bi) + np.arange(len(non_b))
    node_ids = np.r_[_strings(bnodes.node_id), uid[non_b]]
    _require(len(pd.Index(node_ids).unique()) == len(node_ids), "B and physical node ID collision")
    residual = np.flatnonzero(own.ownership_role.eq("REGION").to_numpy())
    _require(len(residual) > 0, "No residual physical population")
    _require(np.all(quotient[pt[residual]] >= 0) and np.all(quotient[ph[residual]] >= 0),
             "A removed B interior node is required by a residual physical edge")
    # The accepted s03 ledger contains residual arcs whose two physical
    # endpoints are B members. They remain exact residual edges, including
    # quotient self-loops; physical ownership outranks an endpoint inference.
    residual_b_bridge = in_b[pt[residual]] & in_b[ph[residual]]
    charge = charges.loc[charges.model_variant.eq(PRIMARY)].copy()
    if "suite_id" in charge:
        charge = charge.loc[charge.suite_id.eq("s03")]
    _require(charge.bacn_id.is_unique and set(membership.bacn_id) <= set(charge.bacn_id),
             "Frozen B charge registry incomplete or duplicated")
    chi = charge.set_index("bacn_id").chi_s.reindex(bi).to_numpy(float)
    _require(np.isfinite(chi).all() and (chi >= 0).all(), "Invalid frozen B incidence charge")
    _require(np.array_equal(backbone.tail_half_charge_s.to_numpy(float), chi[bt]/2) and
             np.array_equal(backbone.head_half_charge_s.to_numpy(float), chi[bh]/2),
             "Frozen backbone incidence charges disagree with charge registry")
    rt, rh = quotient[pt[residual]], quotient[ph[residual]]
    node_chi = np.r_[chi, np.zeros(len(non_b))]
    used_capacity = np.asarray(physical["capacity"], float)[residual] * capacity_multiplier
    tau = np.asarray(physical["free_time"], float)[residual] * 60.
    rk4 = tau * np.asarray(physical["alpha"], float)[residual] / used_capacity**4

    origins_physical = np.asarray(original_od["origin_group_node_idx"], np.int64)
    destinations_physical = np.asarray(original_od["destination_node_idx"], np.int64)
    endpoints = np.unique(np.r_[origins_physical, destinations_physical])
    _require(len(endpoints) > 0 and endpoints.min() >= 0 and endpoints.max() < len(uid),
             "Invalid physical OD endpoints")
    _require((quotient[endpoints] >= 0).all(), "Protected physical OD endpoint removed by frozen B compiler")
    offsets = np.asarray(original_od["origin_offsets"], np.int64)
    demand = np.asarray(original_od["demand"], float)
    multiplicity = np.asarray(original_od["original_row_multiplicity"], np.int64)
    _require(offsets.shape == (len(origins_physical)+1,) and offsets[0] == 0 and
             offsets[-1] == len(demand) and (np.diff(offsets) >= 0).all() and
             destinations_physical.shape == demand.shape == multiplicity.shape,
             "OD ordering/shape differs")
    _require(np.isfinite(demand).all() and (demand > 0).all() and (multiplicity > 0).all(),
             "OD values/multiplicities invalid")
    nroad, nport = len(node_ids), len(endpoints)
    origin_port = nroad + np.arange(nport, dtype=np.int64)
    destination_port = origin_port + nport
    origin_map = np.full(len(uid), -1, np.int64)
    destination_map = origin_map.copy()
    origin_map[endpoints], destination_map[endpoints] = origin_port, destination_port
    oids = np.char.add("n8_od_origin:", uid[endpoints])
    dids = np.char.add("n8_od_destination:", uid[endpoints])
    node_ids = np.r_[node_ids, oids, dids]
    port_edges = np.r_[np.char.add("pacnhn_attachment:origin:", uid[endpoints]),
                       np.char.add("pacnhn_attachment:destination:", uid[endpoints])]
    tail = np.r_[bt, rt, origin_port, quotient[endpoints]]
    head = np.r_[bh, rh, quotient[endpoints], destination_port]
    # Residual physical arc IDs themselves are kept.  Namespaces of frozen B
    # IDs and role attachments are disjoint and are checked rather than assumed.
    edge_ids = np.r_[_strings(backbone.link_id), arcid[residual], port_edges]
    _require(len(pd.Index(edge_ids).unique()) == len(edge_ids), "Common edge IDs collide")
    nb, nr, ne = len(backbone), len(residual), len(edge_ids)
    zero_ports = np.zeros(2*nport)
    a = dict(tail=tail, head=head, edge_ids=edge_ids, node_ids=node_ids,
             blocked=np.r_[np.zeros(nroad, bool), np.ones(2*nport, bool)],
             edge_role=np.r_[np.zeros(nb, np.int8), np.ones(nr, np.int8), np.full(2*nport, 2, np.int8)],
             physical_arc_idx=np.r_[np.full(nb, -1, np.int64), residual, np.full(2*nport, -1, np.int64)],
             capacity=np.r_[np.zeros(nb), used_capacity, zero_ports],
             physical_alpha=np.r_[np.zeros(nb), physical["alpha"][residual], zero_ports],
             is_fitted=np.r_[backbone.runtime_class.eq("FIT_POWER4").to_numpy(), np.zeros(nr+2*nport, bool)],
             backbone_count=np.array(nb, np.int64), regional_count=np.array(nr, np.int64),
             backbone_node_count=np.array(len(bi), np.int64), physical_to_node=quotient,
             physical_node_uids=uid, road_node_count=np.array(nroad, np.int64))
    regional_fields = dict(tau_s=tau, k4=rk4, fixed_internal_H_s=np.zeros(nr),
                           tail_half_charge_s=node_chi[rt]/2, head_half_charge_s=node_chi[rh]/2,
                           structural_fixed_s=np.zeros(nr), domain_Q=np.full(nr, np.inf))
    for key in COST_FIELDS:
        a[key] = np.r_[backbone[key].to_numpy(float), regional_fields[key],
                       np.full(2*nport, np.inf) if key == "domain_Q" else zero_ports]
    a["base"] = a["tau_s"] + (a["fixed_internal_H_s"] + a["tail_half_charge_s"] + a["head_half_charge_s"])
    _require(np.isfinite(a["base"]).all() and np.isfinite(a["k4"]).all(), "Invalid assembled cost")
    closed_ids = _strings(closed_backbone_ids)
    _require(len(np.unique(closed_ids)) == len(closed_ids), "Duplicate closed B coordinate")
    closed_idx = _index(backbone.link_id, closed_ids, "represented I95 B coordinate")
    a["closed_mask"] = np.zeros(ne, bool)
    a["closed_mask"][closed_idx] = True
    a["self_loop"] = tail == head
    _require(not np.any(a["closed_mask"] & (a["edge_role"] != 0)), "Closure changed residual graph")
    od = dict(origins=origin_map[origins_physical], offsets=offsets,
              destinations=destination_map[destinations_physical], demand=demand,
              serve=np.ones(len(demand), bool), original_row_multiplicity=multiplicity,
              origin_group_node_idx=origins_physical, destination_node_idx=destinations_physical)
    mapping = pd.DataFrame(dict(physical_node_idx=np.arange(len(uid)), physical_node_uid=uid,
                               common_node_idx=quotient, is_frozen_backbone_member=in_b,
                               disposition=np.where(quotient < 0, "CONSUMED_MACRO_INTERIOR", "RETAINED")))
    # This full ledger assigns an existing fitted field or one physical BPR term
    # to exactly one graph edge. AC local value functions include regional half
    # charges; no AC boundary connector may add them a second time.
    costs = pd.DataFrame({"edge_idx": np.arange(ne), "edge_id": edge_ids, "edge_role": a["edge_role"],
                          "physical_arc_idx": a["physical_arc_idx"],
                          **{key: a[key] for key in COST_FIELDS[:-1]}})
    costs["charge_owner"] = np.select([a["edge_role"] == 0, a["edge_role"] == 1],
        ["FROZEN_B_EDGE", "REGIONAL_PHYSICAL_EDGE_INCLUDED_IN_LOCAL_VALUE"], default="ZERO_LOGICAL_ROLE")
    own["common_edge_idx"] = -1
    inter = own.ownership_role.eq("INTER_B")
    own.loc[inter, "common_edge_idx"] = _index(backbone.link_id, own.loc[inter, "final_link_id"], "physical B owner")
    own.loc[own.ownership_role.eq("REGION"), "common_edge_idx"] = nb + np.arange(nr)
    endpoint_rows = pd.DataFrame(dict(physical_node_idx=endpoints, physical_node_uid=uid[endpoints],
        road_node_idx=quotient[endpoints], road_node_id=node_ids[quotient[endpoints]],
        origin_node_idx=origin_port, destination_node_idx=destination_port,
        origin_node_id=oids, destination_node_id=dids,
        map_rule=np.where(in_b[endpoints], "FROZEN_S03_B_QUOTIENT", "UNCHANGED_PHYSICAL_UID")))
    bgeo = backbone[["link_id", "tail_node_id", "head_node_id", "role", "runtime_class", "dynamic_model_id"]].copy()
    bn = bnodes.set_index("node_id")
    for end in ("tail", "head"):
        for coordinate in ("longitude", "latitude"):
            bgeo[end+"_"+coordinate] = bgeo[end+"_node_id"].map(bn[coordinate]).to_numpy()
    _require(np.isfinite(bgeo.filter(regex="longitude|latitude").to_numpy()).all(), "Missing frozen B coordinates")
    report = dict(backbone_count=nb, residual_physical_arc_count=nr, common_edge_count=ne,
        common_node_count=len(node_ids), retained_B_node_count=len(bi), physical_road_node_count=len(non_b),
        logical_role_nodes=2*nport, original_OD_aggregate_rows=len(demand),
        original_OD_rows=int(multiplicity.sum()), original_total_demand=float(demand.sum()),
        regional_self_loop_count=int(np.sum(a["self_loop"] & (a["edge_role"] == 1))),
        residual_B_to_B_arc_count=int(residual_b_bridge.sum()),
        residual_B_to_B_inherited_region_count=int(own.loc[residual[residual_b_bridge], "original_owner_id"].nunique()),
        backbone_self_loop_count=int(np.sum(a["self_loop"] & (a["edge_role"] == 0))),
        represented_closed_B_count=int(a["closed_mask"].sum()),
        suppressed_physical_B_arc_count=int(own.ownership_role.eq("SUPPRESSED_B").sum()),
        capacity_multiplier=capacity_multiplier, capacity_multiplier_applications=1,
        physical_cost_minutes_to_seconds_applications=1, residual_parallel_arcs_preserved=True,
        physical_vertex_transit_allowed=True, logical_role_transit_allowed=False,
        self_loops="RETAINED_WITH_EXACT_COST_AND_OWNERSHIP; NEVER_REQUIRED_BY_NONNEGATIVE_SHORTEST_PATH",
        demand_renormalized=False, reachability="NOT_YET_MEASURED; ALL ORIGINAL OD ROWS RETAINED",
        regional_half_charge_policy="INCLUDED_ON_PHYSICAL_EDGE; DO_NOT_ADD_AGAIN_AT_AC_INTERFACE")
    return a, od, dict(physical_node_mapping=mapping, physical_cost_ownership=own,
                      cost_ownership=costs, OD_endpoint_mapping=endpoint_rows, backbone=bgeo), report


def edge_owners_from_nodes(physical_tail, physical_head, residual, node_region, in_b, *, bridge_labels=None):
    """Exact directed-arc ownership; boundary arcs belong to their tail region.

    A B-to-R arc belongs to its R head. No branch is allowed to promote an edge
    into B, merge parallel edges, or change a global vertex identity.
    """
    t, h = np.asarray(physical_tail)[residual], np.asarray(physical_head)[residual]
    label = np.asarray(node_region)
    owner = np.where(in_b[t], label[h], label[t])
    both = in_b[t] & in_b[h]
    if both.any():
        _require(bridge_labels is not None and len(bridge_labels) == len(owner),
                 "Explicit residual B-to-B region ownership required")
        owner = owner.astype(object)
        owner[both] = np.asarray(bridge_labels)[both]
    _require(np.all(owner != ""), "Residual regional ownership is incomplete")
    return owner


def partition_arrays(a, physical, ownership, *, suites, resolutions, partition_config, guard=None):
    """Rebuild four Leiden partitions, retain exact s03 residual ownership.

    Reuses the existing weighted-igraph implementation. Small-region correction
    is deliberately absent: the residual universe is fixed once by s03.
    """
    from partC_rev1_src.partC_rev1_partition_igraph import partC_rev1_run_leiden_partition
    _require(tuple(suites) == SUITES and set(resolutions) == set(SUITES), "Five-suite matrix differs")
    for key in ("random_seed", "objective_function", "beta", "n_iterations"):
        _require(key in partition_config, "Explicit partition parameter missing: " + key)
    _require(partition_config.get("backbone_bridge_policy") == "SINGLETON_DIRECTED_PHYSICAL_ARC_REGION",
             "Explicit residual B-to-B region policy required")
    own = ownership.sort_values("physical_arc_idx", kind="stable").reset_index(drop=True)
    ridx = a["physical_arc_idx"][a["edge_role"] == 1]
    uid = _strings(physical["node_idx_to_uid"])
    in_b = np.ones(len(uid), bool)
    # Every original UID mapped to a non-B retained node is a residual vertex.
    q = a["physical_to_node"]
    in_b[(q >= int(a["backbone_node_count"])) & (q >= 0)] = False
    nodes = pd.DataFrame(dict(partC_rev1_node_uid=uid, is_backbone_node=in_b))
    t, h = physical["tail"], physical["head"]
    arcs = pd.DataFrame(dict(directed_arc_id=_strings(physical["arc_idx_to_directed_arc_id"])[ridx],
        tail_node_uid=uid[t[ridx]], head_node_uid=uid[h[ridx]],
        free_time_min=physical["free_time"][ridx], capacity=physical["capacity"][ridx]))
    rows, node_rows, summaries = [], [], []
    all_owners = np.full((len(suites), len(a["tail"])), -1, np.int32)
    out = dict(suite_ids=_strings(suites), edge_region=all_owners)
    for si, sid in enumerate(suites):
        if guard: guard()
        started = time.perf_counter()
        if sid == "s03":
            labels = _strings(own.loc[own.ownership_role.eq("REGION"), "original_owner_id"])
            source = "EXACT_FROZEN_S03_PHYSICAL_OWNERSHIP"
        else:
            cfg = {"backbone_partition": {"partition": dict(partition_config)}}
            result = partC_rev1_run_leiden_partition(nodes, arcs, cfg,
                resolution=float(resolutions[sid]), random_seed=int(partition_config["random_seed"]))
            partitions = result["node_partition"]
            region_map = {}
            for raw, members in partitions.groupby("region_id", sort=True):
                digest = hashlib.sha256(("\n".join(sorted(members.partC_rev1_node_uid.astype(str)))+"\n").encode()).hexdigest()[:24]
                region_map[str(raw)] = "pacnhn_"+sid+"_region_"+digest
            labels_by_node = np.full(len(uid), "", dtype=object)
            at = _index(uid, _strings(partitions.partC_rev1_node_uid), "partition physical node")
            labels_by_node[at] = partitions.region_id.astype(str).map(region_map).to_numpy()
            _require(np.all(labels_by_node[~in_b] != ""), "Non-B physical partition coverage incomplete")
            bridges = np.char.add("pacnhn_"+sid+"_Bbridge_arc_", arcs.directed_arc_id.to_numpy(str))
            labels = edge_owners_from_nodes(t, h, ridx, labels_by_node, in_b, bridge_labels=bridges)
            node_rows.append(pd.DataFrame(dict(suite_id=sid, physical_node_uid=uid[~in_b],
                                               primary_region_id=labels_by_node[~in_b])))
            source = "LEIDEN_FIXED_RESIDUAL_S03_UNIVERSE_NO_BACKBONE_PROMOTION"
        regions, local = np.unique(labels, return_inverse=True)
        _require(len(regions) > 0 and np.all(regions != ""), "Empty regional partition")
        all_owners[si, a["edge_role"] == 1] = local.astype(np.int32)
        out[sid+"_region_ids"] = _strings(regions)
        counts = np.bincount(local, minlength=len(regions))
        rows.append(pd.DataFrame(dict(suite_id=sid, region_index=np.arange(len(regions)), region_id=regions,
            physical_arc_count=counts, resolution=float(resolutions[sid]), source=source)))
        summaries.append(dict(suite_id=sid, resolution=float(resolutions[sid]), region_count=len(regions),
            residual_arcs=len(labels), promoted_to_B=0, elapsed_seconds=time.perf_counter()-started, source=source,
            residual_B_to_B_policy="INHERITED_S03_OWNER" if sid == "s03" else partition_config["backbone_bridge_policy"],
            residual_B_to_B_count=int(np.sum(in_b[t[ridx]] & in_b[h[ridx]]))))
        if guard: guard()
    _require((all_owners[:, a["edge_role"] == 1] >= 0).all() and
             (all_owners[:, a["edge_role"] != 1] == -1).all(), "Partition edge-role invariant failed")
    # uint64 path trace key=(region*nnode+source)*nnode+target must be lossless.
    largest = max(len(out[s+"_region_ids"]) for s in suites)
    _require(largest*len(a["node_ids"])**2 <= np.iinfo(np.uint64).max, "Trace key uint64 overflow")
    return out, pd.concat(rows, ignore_index=True), (pd.concat(node_rows, ignore_index=True) if node_rows else pd.DataFrame()), summaries


def interface_registry(a, partitions):
    """Exact global nodes shared by regional boundaries, B and logical OD roles."""
    rows = []
    ne, nn = len(a["tail"]), len(a["node_ids"])
    od_roads = np.unique(np.r_[a["head"][(a["edge_role"] == 2) & a["blocked"][a["tail"]]],
                                a["tail"][(a["edge_role"] == 2) & a["blocked"][a["head"]]]])
    for si, sid in enumerate(partitions["suite_ids"]):
        owners = partitions["edge_region"][si]
        edge = np.flatnonzero(owners >= 0)
        incident = pd.DataFrame(dict(node_index=np.r_[a["tail"][edge], a["head"][edge]],
            region_index=np.r_[owners[edge], owners[edge]],
            can_start=np.r_[np.ones(len(edge), bool), np.zeros(len(edge), bool)],
            can_end=np.r_[np.zeros(len(edge), bool), np.ones(len(edge), bool)]))
        incident = incident.groupby(["node_index", "region_index"], sort=True, as_index=False).agg(
            can_start=("can_start", "any"), can_end=("can_end", "any"))
        degree = np.bincount(incident.node_index.to_numpy(np.int64), minlength=nn)
        boundary = degree > 1
        boundary[:int(a["backbone_node_count"])] = True
        boundary[od_roads] = True
        incident = incident.loc[boundary[incident.node_index.to_numpy(np.int64)]].copy()
        incident["suite_id"] = sid
        incident["node_id"] = a["node_ids"][incident.node_index]
        incident["region_id"] = partitions[str(sid)+"_region_ids"][incident.region_index]
        incident["fine_state_is_singleton"] = incident.node_index.ge(int(a["backbone_node_count"]))
        incident["interface_policy"] = np.where(incident.fine_state_is_singleton,
            "EXACT_SHARED_PHYSICAL_UID", "USER_FROZEN_S03_B_QUOTIENT")
        rows.append(incident)
    return pd.concat(rows, ignore_index=True)


class _Sources:
    """Accepted native index union, with conflicts and absent entries rejected."""
    def __init__(self, ctx):
        self.ctx, self.expected = ctx, {}
        inp = ctx.c["input_versions"]
        publication = Path(inp["prior_publication"])
        seal = publication/"IMMUTABLE_SOURCE_SEAL.csv"
        expected_seal = ctx.c["topology"]["source_seal_sha256"]
        ctx.bind(seal, "PRIOR_ACCEPTED_NATIVE_SOURCE_SEAL", expected_seal)
        f = pd.read_csv(seal, usecols=["native_path", "sha256", "status"])
        _require(f.status.eq("PASS").all(), "Prior native source seal contains failures")
        self.add(f.native_path, f.sha256)
        p5 = Path(inp["P5_frozen_runtime"]).parents[1]
        manifest = p5/"manifests/MANIFEST.csv"
        self.bind(manifest, "ACCEPTED_P5_OUTPUT_MANIFEST")
        f = pd.read_csv(manifest, usecols=["absolute_path", "sha256"])
        self.add(f.absolute_path, f.sha256)
        p3 = Path(inp["P3_suite_root"])/"suites/s03/parquet"
        manifest = p3/"artifact_manifest.csv"
        self.bind(manifest, "ACCEPTED_S03_P3_MANIFEST")
        f = pd.read_csv(manifest, usecols=["relative_path", "sha256"])
        self.add([p3/str(p) for p in f.relative_path], f.sha256)

    def add(self, paths, hashes):
        for path, digest in zip(paths, hashes):
            key = str(Path(path).resolve())
            _require(key not in self.expected or self.expected[key] == digest,
                     "Conflicting accepted native identities: " + key)
            self.expected[key] = str(digest)

    def bind(self, path, role, expected=None):
        p = Path(path).resolve(strict=True)
        key = str(p)
        _require(key in self.expected, "Source absent from accepted native index: " + key)
        if expected is not None:
            _require(self.expected[key] == expected, "Metadata/index digest disagreement: " + key)
        return self.ctx.bind(p, role, self.expected[key])

    def frame(self, path, role, **kwargs):
        self.bind(path, role)
        return pd.read_parquet(path, **kwargs)

    def arrays(self, path, role, expected=None):
        self.bind(path, role, expected)
        with np.load(path, allow_pickle=False) as z:
            return {key: z[key] for key in z.files}

    def json(self, path, role):
        self.bind(path, role)
        return read_json(path)


def _write_frame(path, frame):
    path = Path(path)
    tmp = path.with_name(path.name+".tmp."+str(os.getpid()))
    frame.to_parquet(tmp, index=False, compression="zstd")
    os.replace(tmp, path)


def build(ctx):
    """Explicit source-bound builder; called only after campaign preflight."""
    _require(ctx.step_id == "topology", "Topology step context required")
    cfg, inp = ctx.c["topology"], ctx.c["input_versions"]
    deadline = ctx.c["maximum_cumulative_stage_deadlines"][0]
    guard = lambda: ctx.guard(deadline)
    guard()
    dst = ctx.root/"topology"
    _require(not (dst/"RESULT.json").exists(), "Completed topology is immutable")
    dst.mkdir(parents=True, exist_ok=True)
    src = _Sources(ctx)
    ctx.bind(ctx.path, "EXPLICIT_TOPOLOGY_CONFIGURATION")
    ctx.bind(Path(__file__), "CURRENT_TOPOLOGY_IMPLEMENTATION")
    ctx.bind(ctx.repo/"partC_rev1_src/partC_rev1_partition_igraph.py", "REUSED_LEIDEN_IMPLEMENTATION")
    p4 = src.json(inp["P4_physical_config"], "ACCEPTED_PHYSICAL_INPUT_CONFIG")
    pi = p4["input_versions"]
    meta = src.json(pi["graph_metadata_json"], "ACCEPTED_PHYSICAL_GRAPH_METADATA")
    _require(meta["graph_hash"] == pi["graph_identity"] == meta["assignment_graph_hash"], "Physical graph identity mismatch")
    physical = src.arrays(pi["graph_npz"], "ACCEPTED_PHYSICAL_GRAPH")
    _require(len(physical["tail"]) == pi["graph_arc_count"] and len(physical["node_idx_to_uid"]) == pi["graph_node_count"], "Physical graph dimensions differ")
    ch = hashlib.sha256(np.ascontiguousarray(physical["capacity"], dtype=np.float64).tobytes()).hexdigest()
    _require(ch == pi["clean_capacity_payload_sha256"], "Prepared capacity payload differs")
    odroot = Path(pi["accepted_aggregated_od_root"])/"derived_inputs"
    om = src.json(odroot/"aggregated_od_metadata.json", "ORIGINAL_OD_METADATA")
    original = src.arrays(odroot/"aggregated_od.npz", "ORIGINAL_OD_EXACT_AGGREGATE", om["aggregate_npz_sha256"])
    src.bind(odroot/"original_to_aggregated_od_index.npz", "EXACT_ORIGINAL_ROW_MAPPING", om["mapping_npz_sha256"])
    _require(om["od_hash"] == pi["od_identity"] and len(original["demand"]) == om["aggregated_row_count"], "Original OD identity differs")
    _require(int(original["original_row_multiplicity"].sum()) == pi["od_row_count"] and
             abs(float(original["demand"].sum())-pi["total_demand"]) <= 1e-10*pi["total_demand"], "Original OD values differ")
    guard()
    p5 = Path(inp["P5_frozen_runtime"])
    p3 = Path(inp["P3_suite_root"])/"suites/s03/parquet"
    links = src.frame(p5/"runtime_links.parquet", "FROZEN_S03_RUNTIME_LINKS")
    nodes = src.frame(p5/"retained_nodes.parquet", "FROZEN_S03_RUNTIME_NODES")
    packed = src.arrays(p5/"runtime_arrays.npz", "FROZEN_S03_RUNTIME_ARRAYS")
    from partC_rev1_src.phase5.campaign_runtime import pack_runtime
    replay = pack_runtime(links, nodes).arrays
    _require(set(packed) == set(replay) and all(np.array_equal(packed[k], replay[k], equal_nan=True) for k in packed), "Frozen runtime table/array disagreement")
    ownership = src.frame(p5/"physical_link_ownership.parquet", "FROZEN_S03_PHYSICAL_OWNERSHIP")
    bacn = src.frame(p3/"bacn_membership.parquet", "FROZEN_S03_B_QUOTIENT")
    charges = src.frame(p5.parents[1]/"registries/charges.parquet", "FROZEN_STATIC_INCIDENCE_CHARGES",
                       filters=[("suite_id", "=", "s03")])
    sequence = src.frame(p5/"macro_sequences.parquet", "FROZEN_MACRO_PROVENANCE")
    counts = src.json(p5/"counts.json", "FROZEN_B_INCIDENCE_CERTIFICATE")
    inc = counts["incidence"]
    _require(inc["status"] == "PASS" and inc["exactly_one_owner"] and all(inc[k] == 0 for k in
             ("dangling_surviving_incidence", "removed_od_nodes", "replacement_endpoint_conflicts")), "Frozen B incidence certificate failed")
    pub = Path(inp["prior_publication"])
    maskmeta = src.json(pub/"closure/MASK_FREEZE.json", "FROZEN_REPRESENTED_I95_SCOPE")
    _require(maskmeta["status"] == "EXACT_REPRESENTED_MASK_EQUIVALENCE_PASS", "Represented I95 certificate failed")
    ma = src.arrays(pub/"closure/availability.npz", "FROZEN_REPRESENTED_I95_MASK", maskmeta["mask_sha256"])
    closed = ma["abstract_link_ids"][ma["unavailable"]]
    _require(len(closed) == maskmeta["unavailable_logical_links"], "Frozen closure coordinate count differs")
    guard()
    a, od, frames, report = assemble_common(physical, links, nodes, ownership, bacn, charges, original, closed,
        expected_backbone_count=cfg["expected_backbone_count"], capacity_multiplier=cfg["capacity_multiplier"])
    guard()
    pc = dict(random_seed=cfg["leiden_seed"], objective_function=cfg["leiden_objective"],
              beta=cfg["leiden_beta"], n_iterations=cfg["leiden_n_iterations"],
              backbone_bridge_policy=cfg["backbone_bridge_policy"])
    partitions, regions, nodepart, ps = partition_arrays(a, physical, ownership,
        suites=ctx.c["suite_ids"], resolutions=ctx.c["resolutions"], partition_config=pc, guard=guard)
    frames["regions"] = regions
    frames["rebuilt_node_partitions"] = nodepart
    frames["interfaces"] = interface_registry(a, partitions)
    frames["frozen_macro_sequences"] = sequence
    frames["residual_B_to_B"] = frames["physical_cost_ownership"].loc[
        frames["physical_cost_ownership"].ownership_role.eq("REGION") &
        (a["physical_to_node"][physical["tail"]] >= 0) &
        (a["physical_to_node"][physical["tail"]] < int(a["backbone_node_count"])) &
        (a["physical_to_node"][physical["head"]] >= 0) &
        (a["physical_to_node"][physical["head"]] < int(a["backbone_node_count"]))].copy()
    save_npz(dst/"common_graph.npz", **a)
    save_npz(dst/"common_od.npz", **od)
    save_npz(dst/"partitions.npz", **partitions)
    for name, frame in frames.items():
        guard()
        _write_frame(dst/(name+".parquet"), frame)
    (dst/"CHANGELOG.md").write_text(
        "# Common P topology construction\n\n"
        "The new graph freezes all s03 backbone coordinates and cost fields. "
        "Every residual physical directed arc retains its ID, BPR cost and used capacity. "
        "B endpoint incidence half-charges occur once on the residual edge. "
        "Global O/D role nodes are separate and nontransit; their underlying network vertices permit transit.\n\n"
        f"Source inspection found {report['residual_B_to_B_arc_count']} residual B-to-B arcs across "
        f"{report['residual_B_to_B_inherited_region_count']} inherited regions. "
        f"There are {report['regional_self_loop_count']} retained residual quotient self-loops. "
        "The earlier endpoint-only assumption was rejected before graph acceptance. "
        "s03 keeps exact inherited ownership. Each rebuilt resolution uses an explicit singleton "
        "directed-arc region for a residual B-to-B edge. No residual edge is promoted into B.\n\n"
        "All original OD rows remain represented; scenario reachability and witness feasibility "
        "are pending reference/support audits. No assignment was executed by this builder.\n",
        encoding="utf-8")
    graph_identity, od_identity = runtime_identities(a, od)
    report.update(status="PASS", schema="pacnhn.common_topology.v1", partitions=ps,
        graph_identity=graph_identity,
        graph_array_identity=array_hash(a),
        OD_identity=od_identity,
        OD_array_identity=array_hash(od), partitions_identity=array_hash(partitions),
        I95_scope=maskmeta, physical_graph_identity=pi["graph_identity"], original_od_identity=pi["od_identity"],
        trace_key="(region_index * node_count + source_node_index) * node_count + target_node_index",
        equilibrium_executed=False, witness_feasibility="PENDING_REFERENCE_AND_SUPPORT_AUDITS",
        construction_complete_elapsed_seconds=time.time()-ctx.started)
    # Final-source metadata hashes are flushed before the completion marker.
    ctx.flush_bindings()
    write_json(dst/"RESULT.json", report)
    return report


def load_common(root):
    """Return ``Network, Demand, graph_arrays, partition_arrays`` in frozen order."""
    from partC_rev1_src.phase_n8.oracle import Network, Demand
    root = Path(root)
    if root.name != "topology": root = root/"topology"
    _require(read_json(root/"RESULT.json")["status"] == "PASS", "Common topology is not complete")
    def read(name):
        with np.load(root/name, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    a, o, p = read("common_graph.npz"), read("common_od.npz"), read("partitions.npz")
    report = read_json(root/"RESULT.json")
    _require(array_hash(a) == report["graph_array_identity"] and
             array_hash(o) == report["OD_array_identity"] and
             array_hash(p) == report["partitions_identity"], "Frozen common topology payload changed")
    graph = Network(a["tail"], a["head"], a["blocked"], a["edge_ids"], a["node_ids"])
    od = Demand(o["origins"], o["offsets"], o["destinations"], o["demand"], o["serve"])
    _require(graph.identity == report["graph_identity"] and od.identity == report["OD_identity"],
             "Runtime Network/Demand identity differs from topology certificate")
    return graph, od, a, p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--step-id", required=True, choices=["topology"])
    args = parser.parse_args()
    ctx = Context(args.config, step_id=args.step_id)
    try:
        result = build(ctx)
    except Exception as exc:
        import traceback
        write_json(ctx.root/"topology"/(f"FAILED_ATTEMPT_{time.time_ns()}_{os.getpid()}.json"),
                   dict(status="FAIL", exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        raise
    finally:
        ctx.flush_bindings()
    print({k: result[k] for k in ("status", "backbone_count", "residual_physical_arc_count", "common_edge_count")}, flush=True)
