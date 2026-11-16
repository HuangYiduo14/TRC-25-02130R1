"""Certified active-coordinate regional value with endogenous fine disaggregation."""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np
from numba import types
from numba.typed import Dict, List

from . import kernels as K
from .common import arrays_hash


@dataclass
class Result:
    demand: np.ndarray
    z: np.ndarray
    tau: np.ndarray
    path_ids: np.ndarray
    weights: np.ndarray
    value_upper: float
    value_lower: float
    tangent_lower_at_sample: float
    supporting_intercept: float
    gap: float
    raw_gap: float
    roundoff_allowance: float
    relative_gap: float
    iterations: int
    seconds: float
    status: str
    quality: str
    exchanges: int
    lmo_calls: int

    def scalars(self):
        return {k: getattr(self, k) for k in self.__dataclass_fields__
                if not isinstance(getattr(self, k), np.ndarray)}


class Oracle:
    def __init__(self, arrays):
        self.a = {k: np.asarray(v) for k, v in arrays.items()}
        a = self.a
        self.m = len(a["choice_offsets"]) - 1
        if self.m == 0:
            raise ValueError("NO_ACTIVE_LOW_PAIRS: optimizer construction prohibited")
        n = len(a["indptr"]) - 1; arcs = len(a["tail"])
        if n < 1 or a["indptr"][0] != 0 or a["indptr"][-1] != arcs:
            raise ValueError("Invalid local CSR")
        if not np.array_equal(np.repeat(np.arange(n), np.diff(a["indptr"])), a["tail"]):
            raise ValueError("Arc order must be CSR order")
        if np.any(a["head"] < 0) or np.any(a["head"] >= n): raise ValueError("Bad local endpoint")
        for name in ["t0", "capacity", "alpha", "beta"]:
            if a[name].shape != (arcs,) or not np.isfinite(a[name]).all():
                raise ValueError("Invalid BPR array: " + name)
        if (a["capacity"] <= 0).any() or (a["t0"] < 0).any() or (a["alpha"] < 0).any() or (a["beta"] < 1).any():
            raise ValueError("Invalid monotone physical costs")
        self.table = Dict.empty(types.UniTuple(types.int64, 4), types.int64)
        self.paths = List.empty_list(types.int64[:])
        self.pairs = List.empty_list(types.int64)
        self.sources = List.empty_list(types.int64)
        self.targets = List.empty_list(types.int64)
        self.lmo_count = 0
        self.c0, source, target, offsets, paths = self.lmo(a["t0"])
        self.free_ids = self.register(source, target, offsets, paths)
        self.free_offsets = offsets; self.free_arcs = paths

    def lmo(self, cost):
        a = self.a
        if cost.shape != a["t0"].shape or not np.isfinite(cost).all() or (cost < 0).any():
            raise ValueError("Invalid pricing cost")
        d, p, r = K.trees(a["indptr"], a["head"], cost, a["blocked_through"],
                          a["source_offsets"], a["source_nodes"])
        result = K.choose_paths(a["tail"], d, p, r, a["choice_offsets"],
                                a["choice_group"], a["choice_target"])
        if not np.isfinite(result[0]).all():
            raise ValueError("Supported pair has no admissible physical path")
        self.lmo_count += 1
        return result

    def register(self, source, target, offsets, arcs):
        return K.register_routes(self.table, self.paths, self.pairs, self.sources, self.targets,
                                 source, target, offsets, arcs)

    def pool_arrays(self):
        return (np.asarray(self.pairs, dtype=np.int64), np.asarray(self.sources, dtype=np.int64),
                np.asarray(self.targets, dtype=np.int64))

    def import_catalog(self, catalog):
        K.import_catalog(self.table, self.paths, self.pairs, self.sources, self.targets,
            catalog["path_id"], catalog["pair"], catalog["source"], catalog["target"],
            catalog["offsets"], catalog["arcs"])

    def import_fixed_warm(self, other, result, fine_to_coarse, demand):
        ids = result.path_ids
        pair, source, target = other.pool_arrays()
        paths = [other.paths[int(p)] for p in ids]
        off = np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int64)
        arcs = np.concatenate(paths) if paths else np.empty(0, np.int64)
        new_ids = K.register_selected(self.table, self.paths, self.pairs, self.sources, self.targets,
            fine_to_coarse[pair[ids]], source[ids], target[ids], off, arcs)
        unique, inv = np.unique(new_ids, return_inverse=True)
        weights = np.bincount(inv, weights=result.weights, minlength=len(unique))
        self.validate(demand, unique, weights)
        return replace(result, demand=np.asarray(demand).copy(), path_ids=unique, weights=weights)

    def validate(self, demand, ids, weights):
        a = self.a; totals = np.zeros(self.m)
        if len(ids) != len(weights) or (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError("Invalid sparse path weights")
        for p, weight in zip(ids, weights):
            p = int(p); k = self.pairs[p]; s = self.sources[p]; t = self.targets[p]
            allowed = False
            for j in range(a["choice_offsets"][k], a["choice_offsets"][k + 1]):
                group = a["choice_group"][j]
                if t == a["choice_target"][j] and s in a["source_nodes"][a["source_offsets"][group]:a["source_offsets"][group + 1]]:
                    allowed = True; break
            if not allowed: raise ValueError("Forbidden fine fiber")
            u = s
            for arc in self.paths[p]:
                if a["tail"][arc] != u: raise ValueError("Discontinuous path")
                if u != s and a["blocked_through"][u]: raise ValueError("Centroid used as transit")
                u = int(a["head"][arc])
            if u != t: raise ValueError("Wrong path endpoint")
            totals[k] += weight
        error = np.abs(totals - demand)
        if np.any(error > 1e-12 + 2e-11 * np.abs(demand)):
            raise ValueError("Path demand conservation failed")
        return float(error.max(initial=0.))

    def certify(self, demand, h, iteration, started, status, exchanges, initial_lmo):
        a = self.a
        z = K.pool_flow(h, self.paths, len(a["tail"]))
        cost, U = K.bpr(z, a["t0"], a["capacity"], a["alpha"], a["beta"])
        tau, _, _, _, _ = self.lmo(cost)
        T = float(z @ cost); S = float(demand @ tau)
        raw = T - S; allowance = 1e-10 * max(1., abs(T), abs(S))
        if raw < -allowance: raise ValueError("Negative refreshed FW certificate")
        gap = max(raw, 0.); relative = gap / max(1., abs(S))
        ids = np.flatnonzero(h > 0); self.validate(demand, ids, h[ids])
        ell = U - gap; intercept = U - T
        lower = max(ell, float(self.c0 @ demand))
        if lower > U + allowance: raise ValueError("Inverted value interval")
        quality = "A" if relative <= 1e-6 else "B" if relative <= 1e-5 else "C" if relative <= 1e-4 else "QUARANTINED"
        return Result(demand.copy(), z, tau, ids, h[ids].copy(), float(U), lower, ell, intercept,
                      gap, raw, allowance, relative, iteration, time.perf_counter() - started,
                      status, quality, exchanges, self.lmo_count - initial_lmo)

    def solve(self, demand, *, target=1e-5, max_iter=2000, seconds=300, warm=None,
              polish_sweeps=0, progress=None, stop=None):
        a = self.a; demand = np.asarray(demand, dtype=np.float64)
        if demand.shape != (self.m,) or (demand < 0).any() or not np.isfinite(demand).all():
            raise ValueError("Invalid active demand")
        started = time.perf_counter(); initial_lmo = self.lmo_count; exchanges = 0
        h = np.zeros(len(self.paths))
        if warm is not None:
            for p, value in zip(warm.path_ids, warm.weights):
                k = self.pairs[int(p)]
                if warm.demand[k] > 0 and demand[k] > 0:
                    h[p] = value * (demand[k] / warm.demand[k])
        for k, p in enumerate(self.free_ids):
            if warm is None or warm.demand[k] == 0: h[p] += demand[k]
        self.validate(demand, np.flatnonzero(h > 0), h[h > 0])
        z = K.pool_flow(h, self.paths, len(a["tail"]))
        history = []
        for iteration in range(max_iter + 1):
            cost, U = K.bpr(z, a["t0"], a["capacity"], a["alpha"], a["beta"])
            tau, source, target_node, offsets, arcs = self.lmo(cost)
            T = float(z @ cost); S = float(demand @ tau); raw = T - S
            if raw < -1e-10 * max(1., abs(T), abs(S)): raise ValueError("Negative FW gap")
            relative = max(0., raw) / max(1., abs(S)); history.append(relative)
            reached_time = time.perf_counter() - started >= seconds
            stop_requested = stop is not None and stop()
            plateau = (len(history) > 100 and min(history[-25:]) >= .999 * min(history[-100:-75]))
            if iteration % 25 == 0 and progress:
                progress({"iteration": iteration, "gap": relative, "value": float(U),
                          "path_count": len(h), "seconds": time.perf_counter() - started})
            if relative <= target or iteration == max_iter or reached_time or stop_requested or (plateau and iteration >= 100):
                status = ("CONVERGED" if relative <= target else "TIME_LIMIT" if reached_time or stop_requested
                          else "PLATEAU" if plateau else "ITERATION_LIMIT")
                return self.certify(demand, h, iteration, started, status, exchanges, initial_lmo)
            route_ids = self.register(source, target_node, offsets, arcs)
            if len(h) < len(self.paths): h = np.pad(h, (0, len(self.paths) - len(h)))
            y = K.load_routes(demand, offsets, arcs, len(z)); direction = y - z
            step = K.line_search(z, direction, 1., a["t0"], a["capacity"], a["alpha"], a["beta"])
            h *= (1 - step); np.add.at(h, route_ids, step * demand)
            z = K.pool_flow(h, self.paths, len(z))
            if polish_sweeps:
                pairs = np.asarray(self.pairs, dtype=np.int64)
                order = np.argsort(pairs, kind="stable")
                poff = np.r_[0, np.cumsum(np.bincount(pairs, minlength=self.m))]
                h, z, count = K.polish(h, self.paths, order, poff, z,
                    a["t0"], a["capacity"], a["alpha"], a["beta"], polish_sweeps)
                exchanges += count
        raise AssertionError("Unreachable solver state")

    def catalog(self, start=0):
        paths = list(self.paths)[start:]
        offsets = np.r_[0, np.cumsum([len(p) for p in paths])].astype(np.int64)
        pairs, sources, targets = self.pool_arrays()
        return dict(path_id=np.arange(start, len(self.paths), dtype=np.int64),
                    pair=pairs[start:], source=sources[start:], target=targets[start:],
                    offsets=offsets, arcs=np.concatenate(paths) if paths else np.empty(0, np.int64))


def from_fibers(n, tail, head, t0, capacity, fibers, alpha=None, beta=None, blocked=None):
    """Exact fine-source fallback, also used for the fixed-fine comparator."""
    tail = np.asarray(tail, dtype=np.int64); head = np.asarray(head, dtype=np.int64)
    if len(tail) and np.any(tail[1:] < tail[:-1]): raise ValueError("Fixture arcs must be in CSR order")
    sources = sorted(set(s for fiber in fibers for s, _ in fiber))
    smap = {s: j for j, s in enumerate(sources)}
    choices = [sorted(set(f)) for f in fibers]
    if any(not f for f in choices): raise ValueError("Empty allowed fiber")
    a = dict(tail=tail, head=head, indptr=np.r_[0, np.cumsum(np.bincount(tail, minlength=n))],
             t0=np.asarray(t0, float), capacity=np.asarray(capacity, float),
             alpha=np.full(len(tail), .15) if alpha is None else np.asarray(alpha, float),
             beta=np.full(len(tail), 4.) if beta is None else np.asarray(beta, float),
             blocked_through=np.zeros(n, bool) if blocked is None else np.asarray(blocked, bool),
             source_offsets=np.arange(len(sources) + 1, dtype=np.int64), source_nodes=np.asarray(sources, np.int64),
             choice_offsets=np.r_[0, np.cumsum([len(f) for f in choices])],
             choice_group=np.asarray([smap[s] for f in choices for s, t in f], np.int64),
             choice_target=np.asarray([t for f in choices for s, t in f], np.int64))
    return Oracle(a)


def normalized(result, scales, c0, vscale):
    if np.any(scales <= 0) or vscale <= 0: raise ValueError("Invalid scales")
    baseline = float(c0 @ result.demand)
    return {"input": result.demand / scales,
            "value_upper": (result.value_upper - baseline) / vscale,
            "value_lower": (result.value_lower - baseline) / vscale,
            "gradient": scales * (result.tau - c0) / vscale,
            "gap": result.gap / vscale}
