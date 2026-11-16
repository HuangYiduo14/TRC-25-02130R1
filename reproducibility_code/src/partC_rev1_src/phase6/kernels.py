"""Compiled regional pricing and feasible sparse path-pool operations.

The indexed heap is reused from Phase 4. All prices use owned physical arcs;
multi-source roots are initial conditions, never zero-cost routing hubs.
"""
import numpy as np
from numba import njit

from partC_rev1_src.phase4.kernels import _heap_pop, _heap_push_or_decrease


@njit(cache=True, nogil=True)
def trees(indptr, head, cost, blocked_through, source_offsets, source_nodes):
    n = len(indptr) - 1
    groups = len(source_offsets) - 1
    ds = np.full((groups, n), np.inf)
    ps = np.full((groups, n), -1, np.int64)
    roots = np.full((groups, n), -1, np.int64)
    for group in range(groups):
        heap_nodes = np.empty(n, np.int64)
        heap_keys = np.empty(n, np.float64)
        positions = np.full(n, -1, np.int64)
        settled = np.zeros(n, np.bool_)
        source_mask = np.zeros(n, np.bool_)
        size = 0
        for j in range(source_offsets[group], source_offsets[group + 1]):
            s = source_nodes[j]
            ds[group, s] = 0.; roots[group, s] = s; source_mask[s] = True
            size = _heap_push_or_decrease(heap_nodes, heap_keys, positions, size, s, 0.)
        while size:
            u, du, size = _heap_pop(heap_nodes, heap_keys, positions, size)
            if settled[u]: continue
            settled[u] = True
            if blocked_through[u] and not source_mask[u]: continue
            for a in range(indptr[u], indptr[u + 1]):
                v = head[a]
                candidate = du + cost[a]
                if not settled[v] and candidate < ds[group, v]:
                    ds[group, v] = candidate; ps[group, v] = a
                    roots[group, v] = roots[group, u]
                    size = _heap_push_or_decrease(heap_nodes, heap_keys, positions, size, v, candidate)
    return ds, ps, roots


@njit(cache=True, nogil=True)
def choose_paths(tail, distance, predecessor, roots, choice_offsets, choice_group, choice_target):
    m = len(choice_offsets) - 1
    tau = np.full(m, np.inf)
    source = np.full(m, -1, np.int64); target = source.copy(); group = source.copy()
    offsets = np.zeros(m + 1, np.int64)
    for k in range(m):
        for j in range(choice_offsets[k], choice_offsets[k + 1]):
            g, t = choice_group[j], choice_target[j]
            value = distance[g, t]; s = roots[g, t]
            if value < tau[k] or (value == tau[k] and value < np.inf and
                                  (s < source[k] or (s == source[k] and t < target[k]))):
                tau[k] = value; source[k] = s; target[k] = t; group[k] = g
        length = 0
        if group[k] >= 0:
            v = target[k]
            while v != source[k]:
                a = predecessor[group[k], v]
                if a < 0 or length >= distance.shape[1]: raise ValueError("Invalid predecessor")
                length += 1; v = tail[a]
        offsets[k + 1] = offsets[k] + length
    arcs = np.empty(offsets[-1], np.int64)
    for k in range(m):
        v = target[k]; pos = offsets[k + 1]
        if group[k] < 0: continue
        while v != source[k]:
            a = predecessor[group[k], v]; pos -= 1; arcs[pos] = a; v = tail[a]
    return tau, source, target, offsets, arcs


@njit(cache=True, nogil=True)
def bpr(z, t0, capacity, alpha, beta):
    c = np.empty(len(z), np.float64); value = 0.
    for a in range(len(z)):
        if z[a] < 0 or not np.isfinite(z[a]): raise ValueError("Invalid physical flow")
        term = alpha[a] * (z[a] / capacity[a]) ** beta[a]
        c[a] = t0[a] * (1 + term)
        value += t0[a] * z[a] * (1 + term / (beta[a] + 1))
        if not np.isfinite(c[a]) or not np.isfinite(value): raise ValueError("Nonfinite BPR")
    return c, value


@njit(cache=True, nogil=True)
def load_routes(demand, offsets, arcs, arc_count):
    z = np.zeros(arc_count)
    for k in range(len(demand)):
        for j in range(offsets[k], offsets[k + 1]): z[arcs[j]] += demand[k]
    return z


@njit(cache=True, nogil=True)
def register_routes(table, paths, pairs, sources, targets, source, target, offsets, arcs):
    ids = np.empty(len(source), np.int64)
    for k in range(len(source)):
        route = arcs[offsets[k]:offsets[k + 1]]
        h = np.uint64(14695981039346656037)
        for a in route:
            h ^= np.uint64(a + 1); h *= np.uint64(1099511628211)
        key = (np.int64(k), np.int64(source[k]), np.int64(target[k]), np.int64(h))
        if key in table:
            idx = table[key]
            old = paths[idx]
            if len(old) != len(route): raise ValueError("Path hash collision")
            for j in range(len(old)):
                if old[j] != route[j]: raise ValueError("Path hash collision")
        else:
            idx = len(paths); table[key] = idx
            paths.append(route.copy()); pairs.append(np.int64(k))
            sources.append(np.int64(source[k])); targets.append(np.int64(target[k]))
        ids[k] = idx
    return ids


@njit(cache=True, nogil=True)
def import_catalog(table, paths, pairs, sources, targets, ids, pair, source, target, offsets, arcs):
    for j in range(len(ids)):
        route = arcs[offsets[j]:offsets[j + 1]]
        idx = ids[j]
        if idx < len(paths):
            if pairs[idx] != pair[j] or sources[idx] != source[j] or targets[idx] != target[j]:
                raise ValueError("Catalog identity changed")
            old = paths[idx]
            if len(old) != len(route): raise ValueError("Catalog path changed")
            for q in range(len(route)):
                if old[q] != route[q]: raise ValueError("Catalog arc changed")
            continue
        if idx != len(paths): raise ValueError("Catalog path IDs have a gap")
        h = np.uint64(14695981039346656037)
        for arc in route:
            h ^= np.uint64(arc + 1); h *= np.uint64(1099511628211)
        key = (np.int64(pair[j]), np.int64(source[j]), np.int64(target[j]), np.int64(h))
        if key in table: raise ValueError("Duplicate or colliding restored path")
        table[key] = idx
        paths.append(route.copy()); pairs.append(np.int64(pair[j]))
        sources.append(np.int64(source[j])); targets.append(np.int64(target[j]))


@njit(cache=True, nogil=True)
def register_selected(table, paths, pairs, sources, targets, pair, source, target, offsets, arcs):
    result = np.empty(len(pair), np.int64)
    for j in range(len(pair)):
        route = arcs[offsets[j]:offsets[j + 1]]
        h = np.uint64(14695981039346656037)
        for arc in route:
            h ^= np.uint64(arc + 1); h *= np.uint64(1099511628211)
        key = (np.int64(pair[j]), np.int64(source[j]), np.int64(target[j]), np.int64(h))
        if key in table:
            idx = table[key]
            old = paths[idx]
            if len(old) != len(route): raise ValueError("Path collision")
            for q in range(len(route)):
                if route[q] != old[q]: raise ValueError("Path collision")
        else:
            idx = len(paths); table[key] = idx
            paths.append(route.copy()); pairs.append(np.int64(pair[j]))
            sources.append(np.int64(source[j])); targets.append(np.int64(target[j]))
        result[j] = idx
    return result


@njit(cache=True, nogil=True)
def pool_flow(h, paths, arc_count):
    z = np.zeros(arc_count)
    for p in range(len(h)):
        if h[p] < 0 or not np.isfinite(h[p]): raise ValueError("Invalid path weight")
        if h[p] == 0: continue
        for a in paths[p]: z[a] += h[p]
    return z


@njit(cache=True, nogil=True)
def derivative(z, direction, step, t0, cap, alpha, beta):
    result = 0.
    for a in range(len(z)):
        x = z[a] + step * direction[a]
        # Endpoint cancellation is covered by reconstruction from nonnegative paths.
        if x < -1e-12 * max(1., z[a]): raise ValueError("Infeasible line search")
        if x < 0: x = 0.
        result += t0[a] * (1 + alpha[a] * (x / cap[a]) ** beta[a]) * direction[a]
    return result


@njit(cache=True, nogil=True)
def line_search(z, direction, maximum, t0, cap, alpha, beta):
    if derivative(z, direction, 0., t0, cap, alpha, beta) >= 0: return 0.
    if derivative(z, direction, maximum, t0, cap, alpha, beta) <= 0: return maximum
    lo = 0.; hi = maximum
    for _ in range(64):
        mid = .5 * (lo + hi)
        if derivative(z, direction, mid, t0, cap, alpha, beta) > 0: hi = mid
        else: lo = mid
        if hi - lo <= max(1e-12 * max(1., maximum), 4 * np.spacing(hi)): break
    return .5 * (lo + hi)


@njit(cache=True, nogil=True)
def polish(h, paths, pair_order, pair_offsets, z, t0, cap, alpha, beta, sweeps):
    exchanges = 0
    for _ in range(sweeps):
        for k in range(len(pair_offsets) - 1):
            cost, _ = bpr(z, t0, cap, alpha, beta)
            best = -1; worst = -1; best_cost = np.inf; worst_cost = -np.inf
            for j in range(pair_offsets[k], pair_offsets[k + 1]):
                p = pair_order[j]; pc = 0.
                for a in paths[p]: pc += cost[a]
                if pc < best_cost: best = p; best_cost = pc
                if h[p] > 0 and pc > worst_cost: worst = p; worst_cost = pc
            if best < 0 or worst < 0 or best == worst or best_cost >= worst_cost: continue
            direction = np.zeros(len(z))
            for a in paths[best]: direction[a] += 1
            for a in paths[worst]: direction[a] -= 1
            step = line_search(z, direction, h[worst], t0, cap, alpha, beta)
            if step > 0:
                h[best] += step; h[worst] -= step
                # Always reconstruct from nonnegative path weights; no drift clipping.
                z = pool_flow(h, paths, len(z)); exchanges += 1
    return h, z, exchanges
