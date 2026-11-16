"""Settlement-safe exact multigraph loading, with no predecessor-path loops.

Each native batch is bounded in origins; the controller polls between batches.
Heap operations are reused unchanged from the audited Phase 4 implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np
from numba import njit
from partC_rev1_src.phase4.kernels import _heap_push_or_decrease, _heap_pop
from partC_rev1_src.phase5.campaign_runtime import native_adjacency
from .common import array_hash


@njit(cache=True, nogil=True)
def load_range(n, tail, head, indptr, order, ranks, blocked, cost,
               origins, offsets, destinations, demand, serve, begin, end):
    flow = np.zeros(len(tail))
    distances = np.full(int(offsets[end]-offsets[begin]), np.inf)
    dist = np.empty(n)
    pred = np.empty(n, np.int64)
    seen = np.zeros(n, np.int32)
    settled = np.zeros(n, np.int32)
    settlement = np.empty(n, np.int64)
    heapnodes = np.empty(n, np.int64)
    heapkeys = np.empty(n)
    heappos = np.full(n, -1, np.int64)
    subtree = np.zeros(n)
    is_target = np.zeros(n, np.int32)
    sptt = 0.0
    reachable = 0.0
    nsettled = 0
    for group in range(begin, end):
        epoch = group-begin+1
        source = origins[group]
        if source < 0 or source >= n:
            raise ValueError("Invalid mapped origin")
        count = 0
        for row in range(offsets[group], offsets[group+1]):
            if not serve[row]:
                continue
            dest = destinations[row]
            if dest < 0 or dest >= n:
                raise ValueError("Invalid mapped destination")
            if is_target[dest] != epoch:
                is_target[dest] = epoch
                count += 1
        if count == 0:
            continue
        heap_size = 0
        settled_count = 0
        dist[source] = 0.
        seen[source] = epoch
        pred[source] = -1
        heap_size = _heap_push_or_decrease(heapnodes, heapkeys, heappos, heap_size, source, 0.)
        while heap_size > 0 and count > 0:
            node, distance, heap_size = _heap_pop(heapnodes, heapkeys, heappos, heap_size)
            if settled[node] == epoch:
                continue
            settled[node] = epoch
            settlement[settled_count] = node
            settled_count += 1
            if is_target[node] == epoch:
                count -= 1
            if blocked[node] and node != source:
                continue
            for at in range(indptr[node], indptr[node+1]):
                edge = order[at]
                child = head[edge]
                # A settled node, including the source, is never rewritten.
                if settled[child] == epoch:
                    continue
                candidate = distance+cost[edge]
                if seen[child] != epoch or candidate < dist[child]:
                    seen[child] = epoch
                    dist[child] = candidate
                    pred[child] = edge
                    heap_size = _heap_push_or_decrease(heapnodes, heapkeys, heappos, heap_size, child, candidate)
                elif candidate == dist[child] and ranks[edge] < ranks[pred[child]]:
                    pred[child] = edge
        for k in range(heap_size):
            heappos[heapnodes[k]] = -1
        for k in range(settled_count):
            subtree[settlement[k]] = 0.
        for row in range(offsets[group], offsets[group+1]):
            if not serve[row]:
                continue
            dest = destinations[row]
            if settled[dest] == epoch:
                distances[row-offsets[begin]] = dist[dest]
                amount = demand[row]
                subtree[dest] += amount
                reachable += amount
                sptt += amount*dist[dest]
        for k in range(settled_count-1, -1, -1):
            node = settlement[k]
            if node != source and subtree[node] > 0:
                edge = pred[node]
                if edge < 0 or head[edge] != node:
                    raise ValueError("Invalid predecessor edge")
                flow[edge] += subtree[node]
                subtree[tail[edge]] += subtree[node]
        nsettled += settled_count
    return flow, distances, sptt, reachable, nsettled


@dataclass
class Network:
    tail: np.ndarray
    head: np.ndarray
    blocked: np.ndarray
    edge_ids: np.ndarray
    node_ids: np.ndarray

    def __post_init__(self):
        self.tail = np.asarray(self.tail, np.int64)
        self.head = np.asarray(self.head, np.int64)
        self.blocked = np.asarray(self.blocked, bool)
        n = len(self.node_ids)
        if self.blocked.shape != (n,) or len(self.edge_ids) != len(self.tail):
            raise ValueError("Graph shape mismatch")
        if len(set(self.edge_ids)) != len(self.edge_ids) or len(set(self.node_ids)) != n:
            raise ValueError("Duplicate graph identities")
        self.indptr, self.order = native_adjacency(self.tail, self.head, n)
        sorted_ids = np.argsort(self.edge_ids, kind="stable")
        self.ranks = np.empty(len(self.tail), np.int64)
        self.ranks[sorted_ids] = np.arange(len(self.tail))
        self.identity = array_hash(dict(tail=self.tail, head=self.head, blocked=self.blocked,
                                        edge_ids=self.edge_ids, node_ids=self.node_ids))


@dataclass
class Demand:
    origins: np.ndarray
    offsets: np.ndarray
    destinations: np.ndarray
    demand: np.ndarray
    serve: np.ndarray

    def __post_init__(self):
        self.origins = np.asarray(self.origins, np.int64)
        self.offsets = np.asarray(self.offsets, np.int64)
        self.destinations = np.asarray(self.destinations, np.int64)
        self.demand = np.asarray(self.demand, np.float64)
        self.serve = np.asarray(self.serve, bool)
        if self.offsets.shape != (len(self.origins)+1,) or self.offsets[0] != 0 or self.offsets[-1] != len(self.demand):
            raise ValueError("Invalid OD group offsets")
        if np.any(np.diff(self.offsets) < 0) or self.destinations.shape != self.demand.shape or self.serve.shape != self.demand.shape:
            raise ValueError("Unaligned OD arrays")
        if not np.isfinite(self.demand).all() or (self.demand < 0).any():
            raise ValueError("Invalid original demand")
        self.total = float(self.demand[self.serve].sum())
        self.identity = array_hash(dict(origins=self.origins, offsets=self.offsets,
                                        destinations=self.destinations, demand=self.demand, serve=self.serve))

    def balance(self, n):
        routed = np.where(self.serve, self.demand, 0.)
        # reduceat is unsafe for empty origin groups. Cumulative sums preserve them.
        prefix = np.r_[0., np.cumsum(routed)]
        outgoing = prefix[self.offsets[1:]]-prefix[self.offsets[:-1]]
        return np.bincount(self.origins, weights=outgoing, minlength=n)-np.bincount(
            self.destinations[self.serve], weights=self.demand[self.serve], minlength=n)


def aon(graph, od, cost, *, guard=None, batch_origins=16, return_distances=False):
    c = np.asarray(cost, np.float64)
    if c.shape != graph.tail.shape or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("Dijkstra requires actual finite nonnegative costs")
    started = time.perf_counter()
    flow = np.zeros(len(c))
    distance = np.full(len(od.demand), np.inf) if return_distances else None
    sptts = []
    reached = []
    settled = 0
    for first in range(0, len(od.origins), batch_origins):
        if guard:
            guard()
        end = min(first+batch_origins, len(od.origins))
        result = load_range(len(graph.node_ids), graph.tail, graph.head, graph.indptr,
                            graph.order, graph.ranks, graph.blocked, c, od.origins, od.offsets,
                            od.destinations, od.demand, od.serve, first, end)
        flow += result[0]
        if return_distances:
            distance[od.offsets[first]:od.offsets[end]] = result[1]
        sptts.append(result[2])
        reached.append(result[3])
        settled += result[4]
    import math
    sptt = math.fsum(sptts)
    loaded_sptt = float(flow@c)
    allowance = 1e-10*max(1., abs(sptt), abs(loaded_sptt))
    if abs(sptt-loaded_sptt) > allowance:
        raise ArithmeticError("INVALID_ORACLE: y dot cost differs from OD-distance SPTT")
    return dict(flow=flow, distances=distance, sptt=sptt, loaded_sptt=loaded_sptt,
                sptt_difference=loaded_sptt-sptt, reachable_demand=math.fsum(reached),
                settled_nodes=settled, elapsed_seconds=time.perf_counter()-started)


def conservation(graph, flow, balance, total):
    if flow.shape != graph.tail.shape or not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError("Invalid feasible flow; clipping prohibited")
    out = np.bincount(graph.tail, weights=flow, minlength=len(graph.node_ids))
    inc = np.bincount(graph.head, weights=flow, minlength=len(graph.node_ids))
    residual = out-inc-balance
    maximum = float(np.max(np.abs(residual), initial=0.))
    if maximum > 1e-10*max(1., total):
        raise ArithmeticError("Flow conservation failed")
    return dict(max_absolute_node_residual=maximum,
                max_incident_scaled_residual=float(np.max(np.abs(residual)/np.maximum(1., out+inc+np.abs(balance)), initial=0.)))
