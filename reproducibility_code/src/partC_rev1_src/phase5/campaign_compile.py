"""Low-only campaign compilation using the unchanged certified v4 primitives."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import time

import numpy as np
import pandas as pd

from . import bundle_oracle as oracle
from .bundle_oracle import BundleSpec, exact_power_terms, series_power_terms
from .experiments import (MacroExperiment, activation_flows, build_oracle_grid,
    calibration_request_key, calibrated_model_key, dynamic_expression_key)
from .runtime_functions import (PowerSumRuntime, fit_one_term, pilot_weights,
    potential_gradient_audit)
from .campaign_runtime import independent_validation_grid, stable_id

METHOD = "P5_CAMPAIGN_POWER4_LOW_TRAIN_AND_INDEPENDENT_V1"


def campaign_weights(flow,low_flow,activation):
    weights=pilot_weights(flow,low_flow,activation)
    # The authority explicitly gives qL weight eight even when qL=0. Its
    # residual is exactly zero because the dynamic intercept is fixed.
    weights[np.asarray(flow)==float(low_flow)]=8.0
    return weights


def spec_from_row(row):
    return BundleSpec(tuple(f"{row.canonical_primitive_key}:{i}" for i in range(row.member_count)),
        *[np.asarray(json.loads(getattr(row, "member_"+name+"_values_json")), dtype=float)
          for name in ("fftt", "capacity", "alpha", "beta")])


class Population:
    def __init__(self, source, suites):
        self.source = source
        self.chains = pd.read_parquet(source/"topology/chain_candidates_r3.parquet")
        self.bundles = pd.read_parquet(source/"bundle_oracle/bundle_parameter_occurrences.parquet")
        if set(self.chains.suite_short_id) != set(suites) or set(self.bundles.suite_short_id) != set(suites):
            raise ValueError("Incomplete source population")
        fields = ["member_"+n+"_values_json" for n in ("fftt", "capacity", "alpha", "beta")]
        # Deduplicate before constructing numerical objects, while checking all payloads.
        payloads = self.bundles[["canonical_primitive_key", *fields]].drop_duplicates()
        if payloads.canonical_primitive_key.duplicated().any():
            raise ValueError("Primitive identity collision")
        unique = self.bundles.drop_duplicates("canonical_primitive_key").sort_values("canonical_primitive_key")
        self.specs = {r.canonical_primitive_key: spec_from_row(r) for r in unique.itertuples(index=False)}
        self.exact = {k: exact_power_terms(v) for k, v in self.specs.items()}
        if any(not np.all(v.beta == 4.0) for v in self.specs.values()):
            raise ValueError("Unexpected physical beta in authenticated population")
        self.by_suite = {s: f.set_index("bundle_id") for s, f in self.bundles.groupby("suite_short_id")}
        self.bundle_rows = {(r.suite_short_id, r.bundle_id): r for r in self.bundles.itertuples(index=False)}
        self.chain_rows = {r.chain_id: r for r in self.chains.itertuples(index=False)}
        self.experiments = {}
        for r in self.chains.loc[~self.chains.protected_or_blocked].itertuples(index=False):
            ids = json.loads(r.primitive_bundle_ids_json)
            ex = self.experiment(r.suite_short_id, ids, r.chain_id)
            if ex.canonical_signature != r.calibration_request_key:
                raise ValueError("Source low-only request identity mismatch: "+r.chain_id)
            if ex.dynamic_expression_key != r.dynamic_expression_key:
                raise ValueError("Source dynamic expression identity mismatch")
            self.experiments.setdefault(ex.canonical_signature, ex)
        if not self.experiments:
            raise ValueError("Zero requests is not a campaign")

    def experiment(self, suite, primitive_ids, occurrence):
        rows = [self.bundle_rows[(suite, b)] for b in primitive_ids]
        specs = tuple(self.specs[r.canonical_primitive_key] for r in rows)
        low = min(r.low_current_flow for r in rows)
        high = min(r.high_current_flow for r in rows)
        for name in ("low_current_flow", "high_current_flow"):
            v = np.asarray([getattr(r, name) for r in rows])
            if np.ptp(v) > 1e-8+1e-10*max(1.0, float(np.abs(v).max())):
                raise ValueError("CURRENT-inconsistent source corridor: "+occurrence)
        aon = max(r.low_aon_flow for r in rows)
        key = calibration_request_key(specs, low, aon)
        return MacroExperiment(key, occurrence, suite, specs, low, aon, high,
            len(specs), False, dynamic_expression_key=dynamic_expression_key(specs),
            calibration_request_key=key)

    def counts(self):
        n_exact = sum(series_power_terms(e.primitive_bundles) is not None for e in self.experiments.values())
        return {"suite_count": int(self.chains.suite_short_id.nunique()),
            "candidate_occurrences": len(self.chains), "unblocked_occurrences": int((~self.chains.protected_or_blocked).sum()),
            "calibration_requests": len(self.experiments), "exact_macro_requests": n_exact,
            "heterogeneous_macro_requests": len(self.experiments)-n_exact,
            "primitive_occurrences": len(self.bundles), "primitive_expressions": len(self.specs),
            "physical_interB_member_occurrences": int(self.bundles.member_count.sum())}


class PilotGridReuse:
    """Use certified dynamic training grids only after exact method/hash checks."""
    def __init__(self, source):
        self.domain = pd.read_csv(source/"runtime_fits/pilot_domain_summary.csv").set_index("canonical_macro_signature")
        cols = ["canonical_macro_signature", "query_flow", "query_role", "exact_base_cost",
                "exact_base_potential", "certified", "allocation_residual", "kkt_residual",
                "fw_gap_per_flow_s", "primitive_count"]
        self.frame = pd.read_parquet(source/"bundle_oracle/oracle_sample_results.parquet", columns=cols)
        self.frame = self.frame.loc[~self.frame.query_role.str.contains("HIGH")].reset_index(drop=True)
        self.index = self.frame.groupby("canonical_macro_signature", sort=False).indices

    def get(self, ex):
        key = ex.canonical_signature
        if key not in self.index or key not in self.domain.index:
            return None
        d = self.domain.loc[key]
        f = self.frame.iloc[self.index[key]].sort_values("query_flow")
        if f.query_flow.duplicated().any():
            raise ValueError("Ambiguous pilot training query")
        q = f.query_flow.to_numpy(dtype=np.float64)
        roles = f.query_role.astype(str).tolist()
        weights = pilot_weights(q, ex.low_flow, ["ACTIVATION" in r for r in roles])
        if not np.array_equal(weights,campaign_weights(q,ex.low_flow,["ACTIVATION" in r for r in roles])):
            return None
        expected = calibrated_model_key(ex, q, roles, weights, float(d.flow_domain_max))
        if (expected != d.calibrated_model_key or dynamic_expression_key(ex.primitive_bundles) != d.dynamic_expression_key
            or hashlib.sha256(q.tobytes()).hexdigest() != d.calibration_grid_sha256
            or hashlib.sha256(weights.tobytes()).hexdigest() != d.calibration_weights_sha256):
            return None  # A different algorithm/grid is never treated as compatible reuse.
        if not f.certified.all() or np.any(f.allocation_residual > 1e-8+1e-10*np.maximum(1,q)):
            raise ArithmeticError("Reused source grid lacks certified flow closure")
        for col in ("kkt_residual", "fw_gap_per_flow_s"):
            if np.any(f[col].to_numpy() > 1e-7+1e-10*np.maximum(1,np.abs(f.exact_base_cost))):
                raise ArithmeticError("Reused source certificate exceeds hard gate")
        metadata = {k: d[k] for k in ("flow_domain_max", "q_ratio4_status", "domain_status",
            "structural_flow_scale", "calibration_grid_sha256", "calibration_weights_sha256", "calibrated_model_key")}
        return q, roles, f.exact_base_cost.to_numpy(), f.exact_base_potential.to_numpy(), metadata


@contextmanager
def certificate_journal(request_id, records):
    original = oracle.exact_bundle_oracle
    seen = set()
    keys = {}
    def certified(spec, q, **kwargs):
        result = original(spec, q, **kwargs)
        if not result.certified:
            raise ArithmeticError("Uncertified primitive query")
        if id(spec) not in keys:
            keys[id(spec)] = dynamic_expression_key((spec,))
        identity = keys[id(spec)]
        key = (identity, float(q).hex())
        if key not in seen:
            seen.add(key)
            records.append({"request_id": request_id, "primitive_expression_id": identity,
                "query_flow": float(q), "cost_s": result.marginal_cost_s,
                "potential": result.potential_flow_seconds,
                "flow_residual": result.allocation_residual,
                "raw_flow_residual": result.raw_allocation_residual,
                "allocation_correction_l1": result.allocation_correction_l1,
                "kkt_s": result.kkt_residual_s, "gap_per_flow_s": result.fw_gap_per_flow_s,
                "certified": result.certified, "root_iterations": result.root_iteration_count,
                "certificate_role": "OFFLINE_DOMAIN_TRAIN_OR_INDEPENDENT"})
        return result
    oracle.exact_bundle_oracle = certified
    try:
        yield
    finally:
        oracle.exact_bundle_oracle = original


def compile_request(ex, reuse=None):
    started = time.perf_counter()
    certificates = []
    probes = []
    exact = series_power_terms(ex.primitive_bundles)
    row = {"request_id": ex.canonical_signature, "expression_id": ex.dynamic_expression_key,
        "low_current_flow": ex.low_flow, "low_aon_domain_flow": ex.low_aon_flow,
        "method": METHOD, "H_in_identity": False, "high_used": False,
        "coefficient_source": "LOW_ONLY", "source_occurrence_id": ex.occurrence_id,
        "source_suite": ex.suite_id, "is_exact": exact is not None}
    with certificate_journal(ex.canonical_signature, certificates):
        if exact is not None:
            tau, terms = exact
            if any(beta != 4.0 for beta, _ in terms):
                raise ValueError("Unexpected exact exponent; source reconciliation required")
            k = sum(v for _, v in terms)
            function = PowerSumRuntime(tau, np.array([4.]), np.array([k]))
            structural = max(1., sum(float(s.capacity.sum()) for s in ex.primitive_bundles))
            q = np.unique([0., ex.low_flow, ex.low_aon_flow, structural])
            values = [oracle.exact_series_oracle(ex.primitive_bundles, float(v)) for v in q]
            costs = np.array([v.marginal_cost_s for v in values])
            potentials = np.array([v.potential_flow_seconds for v in values])
            row.update(runtime_class="EXACT_POWER_SUM", tau_s=tau, k4=k, eta_s=0., scale_flow=1.,
                domain_Q=float("inf"), domain_policy="ALGEBRAIC_EXACT_ALL_NONNEGATIVE",
                train_count=0, independent_count=len(q), grid_hash="EXACT_FORMULA",
                weights_hash="NO_FIT", source_grid_reused=False, pilot_model_key="")
            roles = ["INDEPENDENT_EXACT_FORMULA"]*len(q)
        else:
            reused = reuse.get(ex) if reuse is not None else None
            if reused is None:
                qtrain, rtrain, cache, metadata = build_oracle_grid(ex)
                ctr = np.array([cache[float(v)].marginal_cost_s for v in qtrain])
                ptr = np.array([cache[float(v)].potential_flow_seconds for v in qtrain])
            else:
                qtrain, rtrain, ctr, ptr, metadata = reused
            weights = campaign_weights(qtrain, ex.low_flow, ["ACTIVATION" in r for r in rtrain])
            domain = float(metadata["flow_domain_max"])
            function = fit_one_term(qtrain, ctr, intercept_s=float(ctr[0]), exponent=4.,
                scale_flow=max(domain,1.), weights=weights)
            independent = independent_validation_grid(qtrain, ex.primitive_bundles, domain, ex.low_flow, activation_flows)
            values = [oracle.exact_series_oracle(ex.primitive_bundles, float(v)) for v in independent]
            q = np.r_[qtrain, independent]
            costs = np.r_[ctr, [v.marginal_cost_s for v in values]]
            potentials = np.r_[ptr, [v.potential_flow_seconds for v in values]]
            roles = ["TRAIN_REUSED" if reused else "TRAIN_NEW"]*len(qtrain)+["INDEPENDENT_LOW_ONLY"]*len(independent)
            unique,first,inverse=np.unique(q,return_index=True,return_inverse=True)
            merged_roles=[set() for _ in unique]
            for i,role in enumerate(roles):merged_roles[inverse[i]].add(role)
            q=unique;costs=costs[first];potentials=potentials[first]
            roles=[";".join(sorted(r)) for r in merged_roles]
            row.update(runtime_class="FIT_POWER4", tau_s=function.intercept_s,
                k4=function.eta_s/function.scale_flow**4, eta_s=function.eta_s,
                scale_flow=function.scale_flow, domain_Q=domain, domain_policy=str(metadata["domain_status"]),
                q_ratio4_status=str(metadata["q_ratio4_status"]), train_count=len(qtrain),
                independent_count=len(independent), grid_hash=hashlib.sha256(qtrain.tobytes()).hexdigest(),
                weights_hash=hashlib.sha256(weights.tobytes()).hexdigest(),
                source_grid_reused=reused is not None, pilot_model_key=str(metadata["calibrated_model_key"]))
        predicted_cost = function.cost(q)
        predicted_potential = function.potential(q)
        ec = (predicted_cost-costs)/np.maximum(1., np.abs(costs))
        ep = (predicted_potential-potentials)/np.maximum(1., np.abs(potentials))
        if not np.isfinite(predicted_cost).all() or not np.isfinite(predicted_potential).all() or (function.derivative(q) < 0).any():
            raise ArithmeticError("Nonfinite or nonmonotone runtime")
        audit = potential_gradient_audit(function, q)
        # v4's knot-aware identities return independent integral/local evidence.
        identity_error = float(audit["max_corrected_local_relative_error"])
        if identity_error > 1e-6:
            raise ArithmeticError("Potential-gradient identity failed")
        mc, mp = float(np.abs(ec).max()), float(np.abs(ep).max())
        if exact is not None and max(mc,mp) > 1e-10:
            raise ArithmeticError("Analytic formula disagrees with certified oracle")
        passed = max(mc,mp) <= .01
        row.update(status="ACCEPT" if passed else "REJECT_FIT_RETAIN_OR_ONE_SPLIT",
            cost_error=mc, potential_error=mp, worst_cost_flow=float(q[np.abs(ec).argmax()]),
            worst_potential_flow=float(q[np.abs(ep).argmax()]), potential_gradient_error=identity_error,
            raw_potential_gradient_error=float(audit["max_raw_local_relative_error"]),
            certificate_count=len(certificates), elapsed_seconds=time.perf_counter()-started,
            validation_grid_hash=hashlib.sha256(q.tobytes()).hexdigest())
        row["model_id"] = stable_id("p5_model_", row["expression_id"], METHOD,
            row["tau_s"], row["k4"], row["grid_hash"], row["validation_grid_hash"])
        for i in range(len(q)):
            probes.append({"request_id": ex.canonical_signature, "query_flow": float(q[i]),
                "role": roles[i], "exact_cost_s": float(costs[i]), "exact_potential": float(potentials[i]),
                "signed_cost_relative_error": float(ec[i]), "signed_potential_relative_error": float(ep[i])})
    return row, probes, certificates
