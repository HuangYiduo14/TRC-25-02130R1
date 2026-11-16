"""Array-only directed multigraph search and exact fine-state lift kernels."""
import numpy as np
from numba import njit
from partC_rev1_src.partC_rev1_shortest_paths import (
    _partC_rev1_heap_push, _partC_rev1_heap_pop,
)


@njit(cache=True, nogil=True)
def shortest_tree(n, indptr, order, head, cost, source):
    """Strict improvements, stable edge order, no equal-distance rewrites."""
    distance = np.full(n, np.inf)
    pred = np.full(n, -1, np.int64)
    settled = np.zeros(n, np.uint8)
    hn = np.empty(2 * len(head) + n + 2, np.int64)
    hd = np.empty(len(hn), np.float64)
    distance[source] = 0.
    size = _partC_rev1_heap_push(hn, hd, 0, source, 0.)
    while size:
        u, du, size = _partC_rev1_heap_pop(hn, hd, size)
        if settled[u] or du != distance[u]:
            continue
        settled[u] = 1
        for i in range(indptr[u], indptr[u + 1]):
            a = order[i]
            v = head[a]
            nd = du + cost[a]
            if not settled[v] and nd < distance[v]:
                distance[v] = nd
                pred[v] = a
                size = _partC_rev1_heap_push(hn, hd, size, v, nd)
    return distance, pred


@njit(cache=True, nogil=True)
def internal_all_pairs(tail, head, costs, owner, node_pos, sizes, offsets):
    """Packed per-B Floyd-Warshall, preserving the cheapest physical edge ID."""
    distances = np.full(offsets[-1], np.inf)
    first_arc = np.full(offsets[-1], -1, np.int64)
    for p in range(len(sizes)):
        n = sizes[p]
        base = offsets[p]
        for j in range(n):
            distances[base + j * n + j] = 0.
    for a in range(len(tail)):
        p = owner[a]
        if p >= 0:
            i = offsets[p] + node_pos[tail[a]] * sizes[p] + node_pos[head[a]]
            if costs[a] < distances[i]:
                distances[i] = costs[a]
                first_arc[i] = a
    for p in range(len(sizes)):
        n = sizes[p]
        base = offsets[p]
        for k in range(n):
            for i in range(n):
                ik = distances[base + i * n + k]
                if not np.isfinite(ik):
                    continue
                for j in range(n):
                    ij = base + i * n + j
                    alt = ik + distances[base + k * n + j]
                    if alt < distances[ij]:
                        distances[ij] = alt
                        first_arc[ij] = first_arc[base + i * n + k]
    return distances, first_arc


@njit(cache=True, nogil=True)
def transition_cost(u, v, node_b, node_pos, sizes, offsets, internal):
    if u == v:
        return 0.
    p = node_b[u]
    if p < 0 or node_b[v] != p:
        return np.inf
    return internal[offsets[p] + node_pos[u] * sizes[p] + node_pos[v]]


@njit(cache=True, nogil=True)
def make_lift_layers(path, op, ref, origin, destination, tail, head,
                     bundle_ptr, bundle_arcs, macro_ptr, macro_bundles):
    choices = 2
    layers = 2
    for edge in path:
        kind, value = op[edge], ref[edge]
        if kind == 1 or kind == 4:
            choices += 1; layers += 1
        elif kind == 2:
            choices += bundle_ptr[value+1]-bundle_ptr[value];layers += 1
        elif kind == 3:
            for j in range(macro_ptr[value],macro_ptr[value+1]):
                p=macro_bundles[j];choices += bundle_ptr[p+1]-bundle_ptr[p];layers += 1
    starts=np.empty(choices,np.int64);ends=np.empty(choices,np.int64);arcs=np.empty(choices,np.int64)
    ptr=np.empty(layers+1,np.int64);ptr[0]=0;ptr[1]=1
    starts[0]=origin;ends[0]=origin;arcs[0]=-1
    pos=1;layer=1
    for edge in path:
        kind,value=op[edge],ref[edge]
        if kind==0:continue
        if kind==1 or kind==4:
            starts[pos]=tail[value] if kind==1 else value
            ends[pos]=head[value] if kind==1 else value
            arcs[pos]=value if kind==1 else -1
            pos+=1;layer+=1;ptr[layer]=pos
        elif kind==2:
            for j in range(bundle_ptr[value],bundle_ptr[value+1]):
                a=bundle_arcs[j];starts[pos]=tail[a];ends[pos]=head[a];arcs[pos]=a;pos+=1
            layer+=1;ptr[layer]=pos
        elif kind==3:
            for j in range(macro_ptr[value],macro_ptr[value+1]):
                p=macro_bundles[j]
                for k in range(bundle_ptr[p],bundle_ptr[p+1]):
                    a=bundle_arcs[k];starts[pos]=tail[a];ends[pos]=head[a];arcs[pos]=a;pos+=1
                layer+=1;ptr[layer]=pos
        else:
            raise ValueError('Invalid lift operation')
    starts[pos]=destination;ends[pos]=destination;arcs[pos]=-1;ptr[layer+1]=pos+1
    return ptr,starts,ends,arcs


@njit(cache=True, nogil=True)
def lift_dp(ptr, starts, ends, arcs, physical_cost, node_b, node_pos,
            sizes, offsets, internal):
    """DP over prescribed physical-edge/bundle/anchor layers, with exact B paths."""
    values = np.full(len(starts), np.inf)
    prev = np.full(len(starts), -1, np.int64)
    values[0] = 0.
    failed_layer = -1
    for layer in range(1, len(ptr) - 1):
        reachable = False
        for j in range(ptr[layer], ptr[layer + 1]):
            best = np.inf
            parent = -1
            for i in range(ptr[layer - 1], ptr[layer]):
                if np.isfinite(values[i]):
                    candidate = values[i] + transition_cost(
                        ends[i], starts[j], node_b, node_pos, sizes, offsets, internal)
                    if candidate < best:
                        best, parent = candidate, i
            if np.isfinite(best):
                values[j] = best + (physical_cost[arcs[j]] if arcs[j] >= 0 else 0.)
                prev[j] = parent
                reachable = True
        if not reachable:
            failed_layer = layer
            break
    return values, prev, failed_layer


@njit(cache=True, nogil=True)
def recover_lift(chosen, starts, ends, arcs, physical_head,
                 node_b, node_pos, sizes, offsets, first_arc):
    result = []
    u = ends[chosen[0]]
    for j in chosen[1:]:
        target = starts[j]
        hops = 0
        while u != target:
            p = node_b[u]
            if p < 0 or node_b[target] != p:
                raise ValueError("Non-B internal movement in lift reconstruction")
            a = first_arc[offsets[p] + node_pos[u] * sizes[p] + node_pos[target]]
            if a < 0 or hops > sizes[p]:
                raise ValueError("Missing/cyclic directed internal witness")
            result.append(a)
            u = physical_head[a]
            hops += 1
        if arcs[j] >= 0:
            result.append(arcs[j])
        u = ends[j]
    return np.asarray(result, dtype=np.int64)
