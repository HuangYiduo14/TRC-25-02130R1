"""Availability overlays retain every native coordinate and finite cost."""
from __future__ import annotations
import numpy as np
from partC_rev1_src.phase_n8.common import array_hash, digest
from partC_rev1_src.phase_n8.oracle import Network, Demand, aon, conservation


def apply_overlay(graph, unavailable):
    mask = np.asarray(unavailable, bool)
    if mask.shape != graph.tail.shape:
        raise ValueError('Availability mask is not in native edge order')
    out = Network(graph.tail, graph.head, graph.blocked, graph.edge_ids, graph.node_ids)
    out.base_identity = graph.identity
    out.unavailable = mask.copy()
    out.order = out.order[~mask[out.order]].copy()
    counts = np.bincount(out.tail[out.order], minlength=len(out.node_ids))
    out.indptr = np.r_[0, np.cumsum(counts)].astype(np.int64)
    out.identity = digest(dict(base_graph=graph.identity, availability=array_hash({'unavailable': mask}),
                              semantics='ORIGINAL_COORDINATES_FILTERED_ADJACENCY_V1'))
    if len(out.order) != int((~mask).sum()):
        raise AssertionError('Overlay adjacency lost an available edge')
    return out


def closed_initialization(graph, baseline_od, cost, guard=None):
    loaded = aon(graph, baseline_od, cost, guard=guard, return_distances=True)
    serve = baseline_od.serve & np.isfinite(loaded['distances'])
    od = Demand(baseline_od.origins, baseline_od.offsets, baseline_od.destinations,
                baseline_od.demand, serve)
    if od.total <= 0:
        raise ValueError('No served demand under closure')
    balance = od.balance(len(graph.node_ids))
    flow = loaded['flow']
    if np.any(flow[graph.unavailable] != 0):
        raise ArithmeticError('Unavailable edge carries initialization flow')
    residual = conservation(graph, flow, balance, od.total)
    if not np.isclose(loaded['reachable_demand'], od.total, rtol=1e-10, atol=1e-8):
        raise ArithmeticError('Closure reachability/load disagreement')
    ledger = dict(original_aggregated_od_count=len(serve), baseline_served_od_count=int(baseline_od.serve.sum()),
                  closed_served_od_count=int(serve.sum()), additional_unreachable_od_count=int((baseline_od.serve & ~serve).sum()),
                  original_demand=float(od.demand.sum()), baseline_served_demand=baseline_od.total,
                  closed_served_demand=od.total, additional_unreachable_demand=float(od.demand[baseline_od.serve & ~serve].sum()),
                  demand_renormalized=False, initialization='FRESH_MASKED_FFTT_AON',
                  aon_seconds=loaded['elapsed_seconds'], conservation=residual)
    return od, balance, flow, ledger
