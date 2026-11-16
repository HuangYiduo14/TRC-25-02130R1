"""Settlement-safe P-ACNHN loader with exact regional segment traces.

Adapted from phase_n8/oracle.py, retaining audited heap/tie/settlement rules.
Only the new file is modified. Parallel arcs remain distinct coordinates.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import time
import numpy as np
from numba import njit
from partC_rev1_src.phase4.kernels import _heap_push_or_decrease, _heap_pop, _hash_add, UINT64_SENTINEL
from partC_rev1_src.phase_n8.oracle import Network, Demand, conservation
from partC_rev1_src.phase4.models import SparseFlowVector, linear_combine_sparse
from .control import save_npz, write_json, array_hash, sha

@njit(cache=True, nogil=True)
def trace_range(n, tail, head, indptr, order, ranks, blocked, cost,
               origins, offsets, destinations, demand, serve, begin, end, enabled, edge_regions, hash_capacity, keep_forest):
    flow = np.zeros(len(tail))
    suites = len(edge_regions)
    pair_keys = np.full((suites, hash_capacity), UINT64_SENTINEL, np.uint64)
    pair_values = np.zeros((suites, hash_capacity))
    forest = np.full((end-begin if keep_forest else 0, n), -2, np.int32)
    active_region = np.empty(suites, np.int64)
    exit_node = np.empty(suites, np.int64)
    traced_rows = 0
    trace_segments = np.zeros(suites, np.int64)
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
                if not enabled[edge]:
                    continue
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
        # Walk exact predecessor paths after loading. Regional traces are
        # keyed by the actual P vertices; no coarse portal selector is used.
        if keep_forest:
            forest[group-begin, source] = -1
            for k in range(settled_count):
                node = settlement[k]
                if node != source and subtree[node] > 0:
                    forest[group-begin, node] = pred[node]
        for row in range(offsets[group], offsets[group+1]):
            if not serve[row] or settled[destinations[row]] != epoch:
                continue
            traced_rows += 1
            if suites == 0:
                continue
            node = destinations[row]
            amount = demand[row]
            active_region[:] = -1
            exit_node[:] = -1
            hops = 0
            while node != source:
                edge = pred[node]
                if edge < 0 or head[edge] != node or not enabled[edge]:
                    raise ValueError("Invalid traced predecessor")
                for suite in range(suites):
                    region = edge_regions[suite, edge]
                    if region != active_region[suite]:
                        if active_region[suite] >= 0:
                            key = (np.uint64(active_region[suite])*np.uint64(n)+np.uint64(node))*np.uint64(n)+np.uint64(exit_node[suite])
                            if not _hash_add(pair_keys[suite], pair_values[suite], key, amount):
                                raise RuntimeError("Trace hash capacity exhausted")
                            trace_segments[suite] += 1
                        active_region[suite] = region
                        exit_node[suite] = node
                node = tail[edge]
                hops += 1
                if hops > n:
                    raise ValueError("Predecessor cycle")
            for suite in range(suites):
                if active_region[suite] >= 0:
                    key = (np.uint64(active_region[suite])*np.uint64(n)+np.uint64(source))*np.uint64(n)+np.uint64(exit_node[suite])
                    if not _hash_add(pair_keys[suite], pair_values[suite], key, amount):
                        raise RuntimeError("Trace hash capacity exhausted")
                    trace_segments[suite] += 1
        nsettled += settled_count
    return flow, distances, sptt, reachable, nsettled, pair_keys, pair_values, forest, traced_rows, trace_segments


def decode_keys(keys, node_count):
    keys=np.asarray(keys,np.uint64); n=np.uint64(node_count)
    return (keys//(n*n)).astype(np.int32), ((keys//n)%n).astype(np.int64), (keys%n).astype(np.int64)


def aon(graph, od, cost, *, edge_regions=None, enabled=None, guard=None,
        workers=1, batch_origins=16, hash_capacity=262144, witness_root=None):
    """Exact loading and optional path-segment tracing, with deterministic reduction.

    Witness forests keep predecessors for every demand-loaded path; original OD
    rows identify the destinations. A forest therefore encodes an explicit path
    for every reachable OD without duplicating common path prefixes.
    """
    c=np.ascontiguousarray(cost,dtype=np.float64)
    if c.shape!=graph.tail.shape or not np.isfinite(c).all() or (c<0).any():
        raise ValueError('Dijkstra needs finite nonnegative actual costs')
    if workers<1 or batch_origins<1 or hash_capacity<2 or hash_capacity&(hash_capacity-1):
        raise ValueError('Invalid bounded loader configuration')
    enabled=np.ones(len(c),bool) if enabled is None else np.asarray(enabled,bool)
    regions=np.empty((0,len(c)),np.int32) if edge_regions is None else np.ascontiguousarray(edge_regions,dtype=np.int32)
    if enabled.shape!=c.shape or regions.ndim!=2 or regions.shape[1]!=len(c):raise ValueError('Misaligned trace/mask')
    n=len(graph.node_ids)
    max_region=int(regions.max(initial=-1))
    if (max_region+1)*n*n>=int(UINT64_SENTINEL):raise OverflowError('Regional key range exceeds uint64')
    if witness_root is not None:
        witness_root=Path(witness_root); witness_root.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();flow=np.zeros(len(c));distances=np.full(len(od.demand),np.inf)
    all_keys=[[] for _ in regions];all_values=[[] for _ in regions]
    sptts=[];reached=[];settled=0;traced=0;segments=np.zeros(len(regions),np.int64);witnesses=[]
    bounds=[(i,min(i+batch_origins,len(od.origins))) for i in range(0,len(od.origins),batch_origins)]

    def run(bound):
        begin,end=bound
        return trace_range(n,graph.tail,graph.head,graph.indptr,graph.order,graph.ranks,
            graph.blocked,c,od.origins,od.offsets,od.destinations,od.demand,od.serve,
            begin,end,enabled,regions,hash_capacity,witness_root is not None)

    # Bound outstanding results to the worker count, rather than submitting the
    # entire national task and retaining many full forest/flow arrays in futures.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for at in range(0,len(bounds),workers):
            if guard:guard()
            chunk=bounds[at:at+workers]
            futures=[pool.submit(run,b) for b in chunk]
            for (begin,end),future in zip(chunk,futures):
                result=future.result()
                flow+=result[0];distances[od.offsets[begin]:od.offsets[end]]=result[1]
                sptts.append(result[2]);reached.append(result[3]);settled+=result[4]
                for suite in range(len(regions)):
                    keep=result[5][suite]!=UINT64_SENTINEL
                    all_keys[suite].append(result[5][suite,keep]);all_values[suite].append(result[6][suite,keep])
                traced+=result[8];segments+=result[9]
                if witness_root is not None:
                    p=witness_root/f'forest_{begin:05d}_{end:05d}.npz'
                    save_npz(p,predecessor=result[7],origin_group_indices=np.arange(begin,end,dtype=np.int32))
                    witnesses.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,begin=begin,end=end))
                del result
    sptt=math.fsum(sptts);loaded=float(flow@c);reachable=math.fsum(reached)
    allowance=1e-10*max(1.,abs(sptt),abs(loaded))
    if abs(sptt-loaded)>allowance:raise ArithmeticError('Loaded cost and shortest-path distance disagree')
    if np.any(flow[~enabled]!=0):raise ArithmeticError('Masked edge carries flow')
    if traced!=int(np.count_nonzero(od.serve&np.isfinite(distances))):raise ArithmeticError('Incomplete OD trace coverage')
    traces=[SparseFlowVector.from_unsorted(np.concatenate(k),np.concatenate(v)) if k else SparseFlowVector.empty()
            for k,v in zip(all_keys,all_values)]
    if witness_root is not None:
        write_json(witness_root/'MANIFEST.json',dict(graph_identity=graph.identity,OD_identity=od.identity,
            enabled_identity=array_hash({'enabled':enabled}),cost_identity=array_hash({'cost':c}),
            traced_OD_rows=traced,regional_segments=segments,forests=witnesses))
    return dict(flow=flow,distances=distances,sptt=sptt,loaded_sptt=loaded,reachable_demand=reachable,
        settled_nodes=settled,traces=traces,traced_OD_rows=traced,regional_segments=segments,
        elapsed_seconds=time.perf_counter()-start,workers=workers)


def combine_traces(weight, left, right):
    if not 0<=weight<=1 or len(left)!=len(right):raise ValueError('Invalid feasible trace combination')
    return [linear_combine_sparse([(1-weight,a),(weight,b)],zero_tol=0.) for a,b in zip(left,right)]


def save_traces(path, traces):
    arrays={}
    for i,trace in enumerate(traces):
        arrays[f'keys_{i}']=trace.keys;arrays[f'values_{i}']=trace.values
    save_npz(path,**arrays)


def load_traces(path):
    with np.load(path,allow_pickle=False) as z:
        return [SparseFlowVector(z[f'keys_{i}'],z[f'values_{i}']) for i in range(len(z.files)//2)]

