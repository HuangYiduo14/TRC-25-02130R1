"""Bounded regional jobs, shared single writer, persistent certificates."""
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd

from .common import (read_json, read_npz, write_json, write_npz, write_frame, event, input_hash,
                     arrays_hash, digest, elapsed, reserve, can_admit, WRITE_LOCK,sha)
from .registry import region_dir
from .storage import Store
from .oracle import Oracle, Result
from . import kernels as K
from .resume_policy import enabled, assess_comparison, native_premise, comparison_ready, LocalRegionError


def solve_sample(c, store, x, metadata, target, scales=None, warm=None, *, retry=False):
    metadata={**metadata,"phase_id":c["phase_id"],"step_id":c["step_id"],"run_id":c["run_id"],
              "execution_head":c.get("execution_head"),"config_sha256":c.get("execution_config_sha256"),
              "original_task_started_unix":c["task_started_unix"],"target_gap":target,
              "validation_policy_id":c.get("validation_policy_id"),"governance_policy_id":c.get("governance_policy_id")}
    ih = input_hash(store.desc["model_hash"], store.desc["support_hash"], x)
    if store.exists(ih):
        return store.get(ih)[0], "REUSED_VALID_CURRENT_MODEL"
    out = Path(c["output_root"])
    candidate_path=store.root/"quarantine"/(ih+"_normal.json")
    if enabled(c) and candidate_path.exists() and candidate_path.with_suffix(".npz").exists():
        prior=read_json(candidate_path)
        if prior["scalars"]["quality"] in {"A","B","C"}:
            saved=read_npz(candidate_path.with_suffix(".npz"))
            if not np.array_equal(saved["demand"],x):raise ValueError("Preserved candidate input changed")
            original=Oracle(store.oracle.a)
            original.import_catalog({k.removeprefix("catalog_"):v for k,v in saved.items() if k.startswith("catalog_")})
            h=np.zeros(len(original.paths));h[saved["path_ids"]]=saved["weights"]
            certified=original.certify(x,h,prior["scalars"]["iterations"],time.perf_counter(),"PRESERVED_CANDIDATE_REOPEN",0,original.lmo_count)
            if certified.quality not in {"A","B","C"}:raise ValueError("Preserved candidate quality changed")
            imported=store.oracle.import_fixed_warm(original,certified,np.arange(store.oracle.m,dtype=np.int64),x)
            h=np.zeros(len(store.oracle.paths));h[imported.path_ids]=imported.weights
            result=store.oracle.certify(x,h,certified.iterations,time.perf_counter(),"REUSED_PRESERVED_FINAL_CANDIDATE",0,store.oracle.lmo_count)
            metadata.update(reused_preserved_candidate=True,preserved_candidate_path=str(candidate_path),
                preserved_candidate_sha256=sha(candidate_path.with_suffix(".npz")),
                native_input_origin_run_id=prior.get("metadata",{}).get("run_id"),new_optimizer_solve=False,
                achieved_relative_gap=result.relative_gap,requested_target_reached=result.relative_gap<=target,
                anchor_downgrade=bool(target<=1e-6 and result.relative_gap>target),
                finite_tolerance_fallback="PRESERVED_VALID_ABC_CANDIDATE_UNDER_CURRENT_EXPLICIT_POLICY")
            with WRITE_LOCK:store.publish(result,metadata,scales)
            event(out/"control/attempts.jsonl",dict(region_id=store.desc["region_id"],input_hash=ih,
                namespace=store.root.name,disposition="PRESERVED_CANDIDATE_PROMOTED",metadata=metadata,scalars=result.scalars()))
            return result,"REUSED_PRESERVED_CANDIDATE"
    forecast=c.get("sample_admission_forecast_seconds",1200 if retry else 300)
    if not can_admit(c, forecast): return None, "DEFERRED_DEADLINE"
    log = out / "control/sample_progress.jsonl"
    startgap = []
    def progress(row):
        if not startgap: startgap.append(row["gap"])
        event(log, dict(region_id=store.desc["region_id"], input_hash=ih,
                        namespace=store.root.name, retry=retry, **row))
    stop = lambda: elapsed(c) >= 86400-reserve(c) or (out / "control/STOP_SCIENCE.json").exists()
    previous = store.root / "quarantine" / (ih + "_normal.json")
    previous_meta = read_json(previous) if retry and previous.exists() else None
    progressed = bool(previous_meta and previous_meta["scalars"]["iterations"] > 0 and
                      previous_meta["scalars"]["relative_gap"] < .99 * previous_meta["initial_gap"])
    seconds=1200 if retry and progressed else 300
    if enabled(c):seconds=max(.001,min(seconds,86400-reserve(c)-elapsed(c)-.5))
    result = store.oracle.solve(x, target=target, max_iter=8000 if retry else 2000,
        seconds=seconds, polish_sweeps=3, warm=None if retry else warm,
        progress=progress, stop=stop)
    attempt = dict(region_id=store.desc["region_id"], input_hash=ih, namespace=store.root.name,
        retry=retry, initial_gap=startgap[0] if startgap else None, scalars=result.scalars(),
        metadata=metadata, target=target)
    event(out / "control/attempts.jsonl", attempt)
    if retry and previous_meta and previous_meta["scalars"]["relative_gap"] < result.relative_gap:
        pa = read_npz(previous.with_suffix(".npz"))
        candidate_oracle = Oracle(store.oracle.a)
        candidate_oracle.import_catalog({k.removeprefix("catalog_"):v for k,v in pa.items() if k.startswith("catalog_")})
        candidate = Result(**{**previous_meta["scalars"], **{k:pa[k] for k in ["demand","z","tau","path_ids","weights"]}})
        result = store.oracle.import_fixed_warm(candidate_oracle, candidate, np.arange(store.oracle.m,dtype=np.int64), x)
        # Mapping modifies IDs only; recompute the final certificate in the current catalog.
        h = np.zeros(len(store.oracle.paths)); h[result.path_ids] = result.weights
        result = store.oracle.certify(x, h, candidate.iterations, time.perf_counter(),
                                     "BEST_NORMAL_AFTER_COLD_RETRY", candidate.exchanges, store.oracle.lmo_count)
    if result.quality in {"A", "B", "C"} and (result.relative_gap <= target or retry or enabled(c)):
        if result.relative_gap>target:
            metadata={**metadata,"requested_target_reached":False,"achieved_relative_gap":result.relative_gap,
                "finite_tolerance_fallback":"VALID_FINAL_CERTIFICATE_WITHIN_DECLARED_ABC_ENVELOPE",
                "anchor_downgrade":target<=1e-6,"fallback_does_not_change_requested_target":True}
        with WRITE_LOCK: store.publish(result, metadata, scales)
        return result, "NEW"
    name = "retry" if retry else "normal"
    qp = store.root / "quarantine" / (ih + "_" + name + ".json")
    if not qp.exists():
        write_json(qp, attempt)
        write_npz(qp.with_suffix(".npz"), demand=result.demand, z=result.z, tau=result.tau,
                  path_ids=result.path_ids, weights=result.weights, **{"catalog_"+k:v for k,v in store.oracle.catalog().items()})
    return result, "COLD_RETRY_REQUIRED" if not retry else "QUARANTINED"


def fixed_low(c, desc, a, low, pairs, *, retry=False, generate_missing=True):
    rd = region_dir(c, desc["region_id"])
    fine = pd.read_parquet(rd / "LOW_native_fine_trace.parquet")
    # Exact included native fine amounts; outside coarse excursions remain in source ledger.
    fine = fine.loc[fine.coarse_key.isin(pairs.coarse_key) & fine.amount.gt(0)].copy()
    fine = fine.sort_values(["coarse_key", "fine_key"]).reset_index(drop=True)
    nodes = a["global_node_index"]
    source = np.searchsorted(nodes, fine.source_global_node.to_numpy(np.int64))
    target = np.searchsorted(nodes, fine.target_global_node.to_numpy(np.int64))
    uniques, inverse = np.unique(source, return_inverse=True)
    fa = dict(a)
    fa.update(source_offsets=np.arange(len(uniques)+1, dtype=np.int64), source_nodes=uniques,
        choice_offsets=np.arange(len(fine)+1, dtype=np.int64), choice_group=inverse, choice_target=target)
    fine_to_coarse = pd.Index(pairs.coarse_key).get_indexer(fine.coarse_key)
    demand = fine.amount.to_numpy(np.float64)
    mapped = np.bincount(fine_to_coarse, weights=demand, minlength=len(low))
    if not np.allclose(mapped, low, rtol=1e-12, atol=1e-9):
        raise LocalRegionError("EXACT_INCLUDED_FIXED_FINE_Q_CLOSURE",evidence={"region_id":desc["region_id"]})
    fd = {**desc, "model_class": "FIXED_INCLUDED_NATIVE_FINE_COMPARATOR",
          "model_hash": digest({"parent_model":desc["model_hash"], "fine_input":arrays_hash({"source":source,"target":target,"demand":demand})})}
    store = Store(rd, fd, fa, namespace="fixed_fine")
    if not (store.root / "definition.npz").exists():
        if not generate_missing:return store,None,fine_to_coarse,"PENDING_FIXED_FINE"
        write_npz(store.root / "definition.npz", **fa, native_demand=demand, fine_to_coarse=fine_to_coarse)
        write_json(store.root / "descriptor.json", fd)
        write_frame(store.root / "included_native_fine.parquet", fine)
    if not generate_missing:
        ih=input_hash(fd["model_hash"],fd["support_hash"],demand)
        return store,store.get(ih)[0] if store.exists(ih) else None,fine_to_coarse,"REUSED_FIXED_FINE" if store.exists(ih) else "PENDING_FIXED_FINE"
    r, disposition = solve_sample(c, store, demand,
        dict(role="FIXED_INCLUDED_NATIVE_FINE_LOW", split="DIAGNOSTIC", outside_policy="EXACT_INCLUDED_ONLY"),
        1e-6, retry=retry)
    return store, r, fine_to_coarse, disposition


def baseline_region(c, rid, *, roles=None, retry=False, generate_missing=True):
    rd = region_dir(c, rid); desc = read_json(rd / "descriptor.json")
    if not desc["m_active"]:
        return dict(region_id=rid, status="NO_ACTIVE_LOW_PAIRS", optimizer_calls=0, model_constructions=0)
    started = time.perf_counter()
    a = read_npz(rd / "local_graph.npz"); low = read_npz(rd / "support_map.npz")["LOW_ACTIVE"]
    pairs = pd.read_parquet(rd / "active_pairs.parquet"); scales = read_npz(rd / "scales.npz")
    xs = read_npz(rd / "baseline_inputs.npz")["x"]; plan = pd.read_parquet(rd / "baseline_plan.parquet")
    store = Store(rd, desc, a)
    fixed, fr, fine_to_coarse, fstatus = fixed_low(c, desc, a, low, pairs, retry=retry,generate_missing=generate_missing)
    warm = None
    if fr is not None:
        warm = store.oracle.import_fixed_warm(fixed.oracle, fr, fine_to_coarse, low)
    selected = plan if roles is None else plan.loc[plan.role_index.isin(roles)]
    # Fixed-fine initializes LOW before other planned inputs; aliases share the strictest target.
    selected = selected.copy()
    selected["dispatch_order"] = np.where(selected.family.eq("LOW"), -2,
                                          np.where(selected.family.eq("ZERO"), -1, selected.role_index))
    selected = selected.sort_values(["dispatch_order", "role_index"])
    done = []; deferred = []
    for row in selected.itertuples():
        target = float(plan.loc[plan.input_hash.eq(row.input_hash), "target_gap"].min())
        metadata = {"role_index":int(row.role_index), "split":str(row.owner_split), "family":str(row.family),
                    "round_index":0, "seed":str(row.seed), "baseline":True}
        if not generate_missing and not store.exists(row.input_hash):
            deferred.append(dict(input_hash=row.input_hash,role_index=int(row.role_index),disposition="PENDING_UNPROCESSED"));continue
        result, disposition = solve_sample(c, store, xs[row.input_index], metadata, target, scales, warm, retry=retry)
        if result is not None and store.exists(row.input_hash):
            warm = result; done.append(row.input_hash)
        else:
            deferred.append(dict(input_hash=row.input_hash, role_index=int(row.role_index), disposition=disposition))
            if enabled(c) and result is not None:
                raise LocalRegionError("MANDATORY_ACCURACY_UNMET",evidence=dict(region_id=rid,input_hash=row.input_hash,
                    namespace="labels",role_index=int(row.role_index),family=row.family,seed=str(row.seed),
                    target=target,achieved_gap=result.relative_gap,final_native_certificate_valid=True,
                    candidate_json=str(store.root/"quarantine"/(row.input_hash+"_normal.json")),
                    candidate_npz=str(store.root/"quarantine"/(row.input_hash+"_normal.npz")),scalars=result.scalars()))
        if result is None: break
    hashes = store.hashes()
    missing = sorted(set(plan.input_hash) - hashes)
    lowhash = input_hash(desc["model_hash"], desc["support_hash"], low)
    hierarchy = {"status":"PENDING"}
    fixedhash = input_hash(fixed.desc["model_hash"], fixed.desc["support_hash"], fr.demand) if fr is not None else None
    if store.exists(lowhash) and fr is not None and fixed.exists(fixedhash):
        lr, _ = store.get(lowhash)
        mapped = read_npz(rd / "LOW_mapped_physical.npz")["z"]
        _, mappedU = K.bpr(mapped, a["t0"], a["capacity"], a["alpha"], a["beta"])
        allowance = lr.roundoff_allowance + fr.roundoff_allowance + 1e-10 * max(1.,mappedU)
        if enabled(c):
            premise,evidence=native_premise(c,desc,a,fixed,fr,mapped,low,fine_to_coarse)
            hierarchy=assess_comparison(lr,fr,float(mappedU),allowance,premise=premise,evidence=evidence)
            hierarchy["outside_LOW_exposure"]=desc["outside_LOW_exposure"]
        else:
            okay = lr.value_lower <= fr.value_upper + allowance and fr.value_lower <= mappedU + allowance
            hierarchy = dict(status="PASS" if okay else "FAIL", coarse_lower=lr.value_lower, coarse_upper=lr.value_upper,
                fixed_lower=fr.value_lower, fixed_upper=fr.value_upper, mapped_physical_potential=float(mappedU),
                allowance=allowance, comparator="EXACT_INCLUDED_NATIVE_FINE_DEMANDS", outside_LOW_exposure=desc["outside_LOW_exposure"])
            if not okay: raise ValueError("Interval-compatible LOW hierarchy failed: " + str(hierarchy))
    ready=comparison_ready(hierarchy) if enabled(c) else hierarchy["status"]=="PASS"
    result = dict(region_id=rid, status="BASELINE_LABELS_COMPLETE" if not missing and ready else "BASELINE_PARTIAL",
        planned_roles=len(plan), planned_unique_inputs=plan.input_hash.nunique(), completed_unique_inputs=len(hashes.intersection(set(plan.input_hash))),
        missing_input_hashes=missing, deferred=deferred, fixed_fine_disposition=fstatus, hierarchy=hierarchy,
        seconds=time.perf_counter()-started, processed_role_subset=None if roles is None else roles,
        partial_label_state_is_not_baseline_acceptance=True)
    out = rd / "baseline_status.json"
    write_json(out, result, new=False)
    event(Path(c["output_root"])/"control/region_progress.jsonl", result)
    if enabled(c) and hierarchy.get("policy_decision")=="LOCAL_ERROR":
        raise LocalRegionError("LOCAL_BENCHMARK_CERTIFICATE_CONTRADICTION",evidence=hierarchy)
    if enabled(c) and generate_missing and fr is not None and fixedhash and not fixed.exists(fixedhash):
        raise LocalRegionError("FIXED_FINE_ACCURACY_UNMET",evidence=dict(region_id=rid,input_hash=fixedhash,
            namespace="fixed_fine",target=1e-6,achieved_gap=fr.relative_gap,final_native_certificate_valid=True,
            candidate_json=str(fixed.root/"quarantine"/(fixedhash+"_normal.json")),
            candidate_npz=str(fixed.root/"quarantine"/(fixedhash+"_normal.npz")),scalars=fr.scalars()))
    return result
