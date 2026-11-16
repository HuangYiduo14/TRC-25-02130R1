"""Fixed sparse composite potential, gradient and full regional HVP."""
from __future__ import annotations

import time
import numpy as np
from partC_rev1_src.phase67.portable import predict_arrays
from .common import array_hash, digest


KINDS = {"FREEFLOW_LINEAR", "OLD_POWER4_NNLS", "NEW_POWER4_RELATIVE_NNLS",
         "OLD_ICNN_FROZEN", "LEGACY_ICNN_CONTINUED", "POWER4_SKIP_ICNN32", "POWER4_SKIP_ICNN64"}


class UnsafeCost(RuntimeError):
    def __init__(self, reason, diagnostic, flow=None):
        super().__init__(reason)
        self.reason = reason
        self.diagnostic = diagnostic
        self.flow = None if flow is None else flow.copy()


def validate_arrays(kind, a):
    if kind not in KINDS:
        raise ValueError("Unsupported exact model kind: " + kind)
    if any(np.asarray(x).dtype.kind not in "biuf" or not np.isfinite(x).all() for x in a.values()):
        raise ValueError("Invalid model parameter dtype/finiteness")
    c0 = a["c0"]
    if c0.ndim != 1 or not len(c0) or (c0 < 0).any():
        raise ValueError("Invalid ordered FFTT")
    m = len(c0)
    if kind == "FREEFLOW_LINEAR":
        return
    if a["scale"].shape != (m,) or (a["scale"] <= 0).any():
        raise ValueError("Invalid positive native normalization scale")
    if "ICNN" not in kind:
        if a["eta"].shape != (m,) or (a["eta"] < 0).any():
            raise ValueError("Invalid analytic coefficients")
        return
    h, g = len(a["b1"]), len(a["b2"])
    for k, shape in {"A1": (h, m), "A2": (g, m), "W2": (g, h), "v": (g,),
                     "b1": (h,), "b2": (g,), "anchor_gradient": (m,)}.items():
        if a[k].shape != shape:
            raise ValueError("Invalid coupled model shape: " + k)
    if (a["W2"] < 0).any() or (a["v"] < 0).any():
        raise ValueError("Negative constrained ICNN weight")
    if "SKIP_ICNN" in kind:
        if int(a.get("head_power", 1)) not in (1, 5):
            raise ValueError("Unsupported exact activation head")
        if a["coef"].shape != (m,) or (a["coef"] < 0).any() or (a["eta_unit"] < 0).any():
            raise ValueError("Invalid skip coefficients")
        amplitude = float(a["residual_unit"]*a["gate"])
    else:
        amplitude = float(a["Vscale"]*a["gain"])
    if amplitude < 0:
        raise ValueError("Negative anchored convex residual amplitude")
    zero = predict_arrays(kind, a, np.zeros(m))
    allowance = 1e-8 + 1e-10*max(1., float(c0.max()))
    if not np.isfinite(zero[0]) or abs(float(zero[0])) > allowance or np.max(np.abs(zero[1]-c0)) > allowance:
        raise ValueError("Native zero anchor is inconsistent")


class Composite:
    def __init__(self, suite, entries, registry, guard=None):
        self.suite = suite
        self.packed = suite["packed"]
        self.np = suite["packed_count"]
        self.c0 = suite["pairs"].c0_seconds.to_numpy(float)
        self.m = len(self.c0)
        self.nzero = len(suite["graph"].tail)-self.np-self.m
        if self.nzero < 0:
            raise ValueError("Composite logical-edge count mismatch")
        self.entries = entries
        self.registry = registry
        self.guard = guard
        self.scales = np.ones(self.m)
        self.normalized = np.zeros(self.m, bool)
        self.eta = np.zeros(self.m)
        self.neural = []
        covered = np.zeros(self.m, bool)
        self.region_index = np.full(self.m, -1, np.int64)
        for number, e in enumerate(entries):
            kind, a = e["model_kind"], e["arrays"]
            validate_arrays(kind, a)
            sl = slice(e["start"], e["stop"])
            if covered[sl].any() or not np.array_equal(self.c0[sl], a["c0"]):
                raise ValueError("Invalid sparse regional coordinate ownership")
            covered[sl] = True
            self.region_index[sl] = number
            if kind != "FREEFLOW_LINEAR":
                self.scales[sl] = a["scale"]
                self.normalized[sl] = True
            if kind in {"OLD_POWER4_NNLS", "NEW_POWER4_RELATIVE_NNLS"}:
                self.eta[sl] = a["eta"]
            if "ICNN" in kind:
                self.neural.append(e)
        if entries and not covered.all():
            raise ValueError("Uncovered active coordinates")
        self.identity = digest(dict(graph=suite["graph"].identity,
                                     registry=registry["identity"], support=suite["support_identity"],
                                     packed=array_hash(self.packed.arrays), semantics="FIXED_NATIVE_COMPOSITE_V1"))
        self.timings = []
        self.last_diagnostic = None

    def evaluate(self, flow, direction=None, *, enforce_safety=True):
        started = time.perf_counter()
        f = np.asarray(flow, np.float64)
        if f.shape != (self.np+self.m+self.nzero,) or not np.isfinite(f).all() or (f < 0).any():
            raise ValueError("One finite nonnegative flow per logical edge required")
        d = None if direction is None else np.asarray(direction, np.float64)
        if d is not None and (d.shape != f.shape or not np.isfinite(d).all()):
            raise ValueError("Invalid HVP direction")
        u = f[self.np:self.np+self.m]
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                (pc, pv, pd), flags = self.packed.evaluate(f[:self.np], "combined")
                x = u/self.scales
                rc = self.c0+self.eta*x**4
                rv = u*self.c0+self.eta*self.scales*x**5/5.
                rh = None if d is None else 4*self.eta*x**3*d[self.np:self.np+self.m]/self.scales
                # Each neural block receives the entire ordered regional vector.
                # No full Hessian or regional equilibrium is constructed.
                for number, e in enumerate(self.neural):
                    if self.guard and number % 32 == 0:
                        self.guard()
                    sl = slice(e["start"], e["stop"])
                    result = predict_arrays(e["model_kind"], e["arrays"], u[sl],
                                            None if d is None else d[self.np:][sl])
                    rc[sl] = result[1]
                    rv[sl] = 0.
                    rv[e["start"]] = result[0]
                    if rh is not None:
                        rh[sl] = result[2]
                value = float(np.sum(pv)+np.sum(rv))
                cost = np.r_[pc, rc, np.zeros(self.nzero)]
                hvp = None if d is None else np.r_[pd*d[:self.np], rh, np.zeros(self.nzero)]
        except (FloatingPointError, OverflowError) as exc:
            raise UnsafeCost("STOP_NONFINITE", dict(error=str(exc), input_state_hash=array_hash({"flow": f})), f) from exc
        if not np.isfinite(value) or not np.isfinite(cost).all() or (hvp is not None and not np.isfinite(hvp).all()):
            raise UnsafeCost("STOP_NONFINITE", dict(input_state_hash=array_hash({"flow": f})), f)
        deficit = np.maximum(self.c0-rc, 0.)
        floor = deficit > 1e-8
        ratios = u/self.scales
        domains = self.packed.arrays["domain_Q"]
        valid_domain = self.packed.arrays["is_fitted"] & (domains > 0)
        diag = dict(minimum_cost_seconds=float(cost.min(initial=np.inf)),
                    below_fftt_coordinates=int(floor.sum()), maximum_below_fftt_seconds=float(deficit.max(initial=0)),
                    maximum_below_fftt_fraction=float(np.max(deficit/np.maximum(self.c0, 1.), initial=0.)),
                    below_fftt_traversal_flow=float(u[floor].sum()),
                    flow_weighted_below_fftt_deficit=float(u@deficit),
                    maximum_native_scale_ratio=(float(ratios[self.normalized].max()) if self.normalized.any() else None),
                    native_scale_ratio_coordinates=int(self.normalized.sum()),
                    packed_extrapolated_links=int(flags.sum()),
                    packed_max_domain_ratio=float(np.max(f[:self.np][valid_domain]/domains[valid_domain], initial=0.)),
                    global_neural_nonnegativity_certified=False if self.neural else True)
        self.last_diagnostic = diag
        self.timings.append(dict(operation="VALUE_GRAD_HVP" if d is not None else "VALUE_GRAD",
                                 seconds=time.perf_counter()-started))
        if enforce_safety and (cost < 0).any():
            edge = int(np.argmin(cost))
            diag = dict(diag, edge_index=edge, edge_id=str(self.suite["graph"].edge_ids[edge]),
                        cost_seconds=float(cost[edge]), input_state_hash=array_hash({"flow": f}))
            if edge >= self.np:
                coord = edge-self.np
                i = self.region_index[coord]
                if i >= 0:
                    e = self.entries[i]
                    diag.update(region_id=e["region_id"], model_kind=e["model_kind"],
                                portable_sha256=e["portable_sha256"], portable_path=e["portable_path"],
                                native_coordinate=coord-e["start"], domain_ratio=float(ratios[coord]))
            raise UnsafeCost("STOP_NEGATIVE_COST", diag, f)
        return value, cost, hvp, diag
