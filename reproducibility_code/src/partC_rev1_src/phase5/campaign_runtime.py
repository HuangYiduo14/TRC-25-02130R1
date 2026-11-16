"""Complete Phase 5 explicit kernels. No oracle is imported by this module.

The packed implementation uses the authenticated beta=4 physical population.
An unexpected exponent is rejected at packing, never silently reinterpreted.
Physical member variables remain distinct even when endpoints coincide.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

CLASSES = ("EXACT_POWER_SUM", "FIT_POWER4", "EXACT_PHYSICAL_MEMBER_BPR",
           "EXACT_PRIMITIVE_BPR", "STATIC_CONNECTOR")


def stable_id(prefix, *parts):
    return prefix + hashlib.sha256(json.dumps(parts, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def native_adjacency(tail, head, node_count):
    tail = np.asarray(tail, dtype=np.int64)
    head = np.asarray(head, dtype=np.int64)
    if tail.shape != head.shape or tail.ndim != 1 or not len(tail):
        raise ValueError("Nonempty aligned native edges required")
    if min(tail.min(), head.min()) < 0 or max(tail.max(), head.max()) >= node_count:
        raise ValueError("Dangling native edge")
    edge_idx = np.argsort(tail, kind="stable").astype(np.int64)
    offsets = np.r_[0, np.cumsum(np.bincount(tail, minlength=node_count))]
    return offsets.astype(np.int64), edge_idx


def recover_edge_path(predecessor_edge, tail, head, origin, destination):
    """Recover by edge index; endpoint pairs are never used as link identity."""
    result = []
    node = int(destination)
    seen = set()
    while node != origin:
        if node in seen or node < 0 or node >= len(predecessor_edge):
            raise ValueError("Cyclic/invalid predecessor path")
        seen.add(node)
        edge = int(predecessor_edge[node])
        if edge < 0 or edge >= len(tail) or int(head[edge]) != node:
            raise ValueError("Missing/invalid predecessor edge")
        result.append(edge)
        node = int(tail[edge])
    return np.asarray(result[::-1], dtype=np.int64)


@dataclass(frozen=True)
class PackedRuntime:
    arrays: dict

    def _flow(self, flow):
        q = np.asarray(flow, dtype=np.float64)
        if q.shape != self.arrays["tau_s"].shape or not np.isfinite(q).all() or (q < 0).any():
            raise ValueError("One finite nonnegative flow per native edge is required")
        return q

    def evaluate(self, flow, operation="combined", *, structural=False):
        q = self._flow(flow)
        a = self.arrays
        h = a["structural_fixed_s"] if structural else (
            a["fixed_internal_H_s"] + a["tail_half_charge_s"] + a["head_half_charge_s"])
        base = a["tau_s"] + h
        with np.errstate(over="raise", invalid="raise"):
            q2 = q*q
            dynamic = a["k4"]*q2*q2
            if operation == "cost":
                result = base + dynamic
            elif operation == "potential":
                result = q*(base + dynamic/5.0)
            elif operation == "derivative":
                result = 4.0*a["k4"]*q2*q
            elif operation == "combined":
                result = (base+dynamic, q*(base+dynamic/5.0), 4.0*a["k4"]*q2*q)
            else:
                raise ValueError("Unknown explicit kernel operation")
        values = result if isinstance(result, tuple) else (result,)
        if any(not np.isfinite(v).all() for v in values):
            raise ArithmeticError("Nonfinite runtime result")
        flag = a["is_fitted"] & (q > a["domain_Q"])
        return result, flag


def pack_runtime(links, nodes):
    if links.empty or links.link_id.duplicated().any() or nodes.node_id.duplicated().any():
        raise ValueError("Empty or duplicate native registry")
    if not set(links.runtime_class).issubset(CLASSES):
        raise ValueError("Unknown runtime class; no proxies allowed")
    if not np.all(links.beta.to_numpy(dtype=float) == 4.0):
        raise ValueError("Unexpected physical exponent: reconcile source before packing")
    local = pd.Index(nodes.node_id.astype(str))
    tail = local.get_indexer(links.tail_node_id.astype(str)).astype(np.int64)
    head = local.get_indexer(links.head_node_id.astype(str)).astype(np.int64)
    offsets, edges = native_adjacency(tail, head, len(local))
    a = {n: links[n].to_numpy(dtype=np.float64) for n in (
        "tau_s", "k4", "fixed_internal_H_s", "tail_half_charge_s", "head_half_charge_s",
        "structural_fixed_s", "domain_Q")}
    for name in ("tau_s", "k4", "fixed_internal_H_s", "tail_half_charge_s",
                 "head_half_charge_s", "structural_fixed_s"):
        if not np.isfinite(a[name]).all() or (a[name] < 0).any():
            raise ValueError("Invalid nonnegative runtime parameter: "+name)
    a.update(tail=tail, head=head, adjacency_offsets=offsets, adjacency_edge_idx=edges,
        is_fitted=links.runtime_class.eq("FIT_POWER4").to_numpy(dtype=bool),
        class_code=pd.Categorical(links.runtime_class, categories=CLASSES).codes.astype(np.int8),
        member_arc_idx=links.member_arc_idx.to_numpy(dtype=np.int64))
    if np.any(a["is_fitted"] & (~np.isfinite(a["domain_Q"]) | (a["domain_Q"] <= 0))):
        raise ValueError("Fitted link has no valid low-only domain")
    return PackedRuntime(a)


def independent_validation_grid(training, bundles, domain, low_flow, activation_function):
    """Coefficient-frozen low-only probes, deterministic float64 deduplication."""
    q = np.asarray(training, dtype=np.float64)
    if q.ndim != 1 or len(q) < 2 or np.any(np.diff(q) <= 0) or domain <= 0:
        raise ValueError("Invalid training domain")
    offset = (np.arange(33, dtype=np.float64) + 0.5)/33.0
    pieces = [0.5*(q[:-1]+q[1:]), domain*offset, domain*offset**2,
              np.array([0.0, float(low_flow), float(domain)])]
    for spec in bundles:
        activation = activation_function(spec)
        activation = activation[(activation >= 0) & (activation <= domain)]
        pieces.extend([activation, np.nextafter(activation, 0.0),
                       np.nextafter(activation, np.inf)])
    result = np.unique(np.concatenate(pieces))
    return result[(result >= 0) & (result <= domain)]


def choose_split(directions):
    """One physical cut using summed reciprocal burden; no flow-based selection."""
    if not directions:
        raise ValueError("No corridor directions")
    common = set(directions[0]["internal_nodes"])
    for d in directions[1:]:
        common &= set(d["internal_nodes"])
    scores = []
    for node in common:
        burden = 0
        distance = 0.0
        for d in directions:
            j = d["internal_nodes"].index(node)+1
            counts = d["member_counts"]
            if len(counts) != len(d["internal_nodes"])+1:
                raise ValueError("Invalid split sequence")
            burden += int(counts[j-1])+int(counts[j])
            distance += abs(j-len(counts)/2.0)
        scores.append((-burden, distance, str(node)))
    return min(scores)[2] if scores else None


def atomic_action(parent_pass, children_pass=None, *, saving=0):
    if parent_pass and all(parent_pass):
        return "ACCEPT_PARENT"
    if saving > 0 and children_pass and all(children_pass):
        return "ACCEPT_SPLIT"
    return "RETAIN"
