"""Conditional benchmark applicability and durable, scoped regional quarantine."""
from pathlib import Path
from functools import lru_cache
import math
import time
import traceback
import numpy as np
import pandas as pd

from .common import read_json,read_npz,write_json,event,sha,digest,WRITE_LOCK

POLICY="CONDITIONAL_MAPPED_COMPARATOR_V1"
READY="HEALTHY_BASELINE_READY"
QUARANTINED="QUARANTINED_LOCAL"
STATES={"PENDING","RUNNING",READY,QUARANTINED,"RESOURCE_DEFERRED"}


class LocalRegionError(RuntimeError):
    """A known region/input scoped check, with evidence; unknown errors are not this type."""
    def __init__(self,code,*,evidence=None):
        super().__init__(code);self.code=code;self.evidence=evidence or {}


class SharedIntegrityError(RuntimeError):pass
class FailureThreshold(RuntimeError):pass
class ResourceStop(RuntimeError):pass
class UnavailableRegion(ValueError):pass


def enabled(c):return c.get("validation_policy_id")==POLICY


def assess_comparison(coarse,fixed,mapped_potential,allowance,*,premise,evidence=None,q_closure=True,same_model=True):
    """Assess verified native certificates; this function does not manufacture premise proof."""
    if premise not in {"CERTIFIED_FEASIBLE","CERTIFIED_DOMINATING_SUBFLOW","SOURCE_PATH_MISMATCH","NOT_ESTABLISHED"}:
        raise SharedIntegrityError("Unknown mapped premise")
    if not q_closure or not same_model:raise LocalRegionError("FINE_COMPARATOR_EMBEDDING_FAILED")
    values=[coarse.value_lower,coarse.value_upper,fixed.value_lower,fixed.value_upper,mapped_potential,allowance]
    if not all(math.isfinite(float(x)) for x in values) or allowance<0:raise SharedIntegrityError("Nonfinite comparison certificate")
    if coarse.value_lower>coarse.value_upper+allowance or fixed.value_lower>fixed.value_upper+allowance:
        raise LocalRegionError("INVALID_CERTIFICATE_INTERVAL")
    if premise!="NOT_ESTABLISHED" and not evidence:raise SharedIntegrityError("Claimed premise has no proof provenance")
    same=coarse.value_lower<=fixed.value_upper+allowance
    applicable=premise in {"CERTIFIED_FEASIBLE","CERTIFIED_DOMINATING_SUBFLOW"}
    mapped_ok=fixed.value_lower<=mapped_potential+allowance
    mapped_status=("APPLICABLE_INTERVAL_COMPATIBLE" if mapped_ok else "APPLICABLE_CERTIFICATE_CONTRADICTION") if applicable else (
        "NOT_APPLICABLE_SOURCE_PATH_MISMATCH" if premise=="SOURCE_PATH_MISMATCH" else "NOT_ESTABLISHED_NONBLOCKING")
    okay=same and (not applicable or mapped_ok)
    return dict(status="ELIGIBLE_LOCAL_DATA" if okay else "LOCAL_ERROR",policy_decision="ELIGIBLE_LOCAL_DATA" if okay else "LOCAL_ERROR",
        validation_policy_id=POLICY,local_label_certificate="PASS",
        same_model_coarse_fixed="INTERVAL_COMPATIBLE" if same else "CONTRADICTION",
        mapped_premise=premise,mapped_comparison=mapped_status,inequality_certified_applicable=applicable,
        coarse_lower=float(coarse.value_lower),coarse_upper=float(coarse.value_upper),
        fixed_lower=float(fixed.value_lower),fixed_upper=float(fixed.value_upper),mapped_physical_potential=float(mapped_potential),
        allowance=float(allowance),signed_fixed_lower_excess=float(fixed.value_lower-mapped_potential),
        signed_coarse_lower_excess=float(coarse.value_lower-mapped_potential),
        signed_fixed_upper_difference=float(fixed.value_upper-mapped_potential),
        signed_coarse_upper_difference=float(coarse.value_upper-mapped_potential),
        comparator="EXACT_INCLUDED_NATIVE_FINE_DEMANDS",premise_evidence=evidence)


@lru_cache(maxsize=4)
def source_mismatches(parent_root):
    p=Path(parent_root)/"validation/mandatory_hierarchy_defect/centroid_flow_and_endpoint_ledger.parquet"
    if sha(p)!="63ef75e804c0b7a86a055d66e5b3a3de2dc6d0cb3d317fe2085e5e8c5171fb2e":
        raise SharedIntegrityError("Reviewed source-path premise ledger hash changed")
    frame=pd.read_parquet(p); selected=frame.loc[frame.positive_nonendpoint_transit]
    ids=set(selected.region_id)
    if len(ids)!=9:raise SharedIntegrityError("Source nine-region warning ledger changed")
    return {rid:dict(path=str(p),sha256=sha(p),witnesses=selected.loc[selected.region_id.eq(rid)].to_dict("records")) for rid in ids}


def native_premise(c,desc,arrays,fixed_store,fixed_result,mapped,low,fine_to_coarse):
    """Reuse an existing exact fine path witness when it already dominates the mapped flow."""
    warnings=source_mismatches(c["parent_output_root"])
    binding=dict(region_id=desc["region_id"],model_hash=desc["model_hash"],support_hash=desc["support_hash"],
        graph_hash=desc.get("graph_hash"),fiber_hash=desc.get("fiber_hash"),fixed_model_hash=fixed_store.desc["model_hash"])
    mapped_demand=np.bincount(fine_to_coarse,weights=fixed_result.demand,minlength=len(low))
    if not np.allclose(mapped_demand,low,rtol=1e-12,atol=1e-9):raise LocalRegionError("EXACT_FINE_Q_CLOSURE_FAILED",evidence=binding)
    for key in ["tail","head","indptr","blocked_through","t0","capacity","alpha","beta"]:
        if not np.array_equal(arrays[key],fixed_store.oracle.a[key]):raise SharedIntegrityError("Fine comparator changed graph/cost/path rule")
    if desc["region_id"] in warnings:
        return "SOURCE_PATH_MISMATCH",{**binding,**warnings[desc["region_id"]]}
    tolerance=1e-12+2e-11*np.abs(mapped)
    if np.all(fixed_result.z<=mapped+tolerance):
        # The solver/get-recertification validates every saved path under this same fine model.
        fixed_store.oracle.validate(fixed_result.demand,fixed_result.path_ids,fixed_result.weights)
        return "CERTIFIED_DOMINATING_SUBFLOW",{**binding, "namespace":"fixed_fine",
            "proof_kind":"EXISTING_CERTIFIED_FIXED_FINE_PATH_FLOW_COMPONENTWISE_DOMINATION",
            "max_flow_excess":float(np.max(fixed_result.z-mapped,initial=0.)),
            "max_flow_allowance":float(np.max(tolerance,initial=0.)),
            "catalog_index_path":str(fixed_store.index_file),"catalog_index_sha256":sha(fixed_store.index_file),
            "input_hashes":sorted(fixed_store.hashes())}
    return "NOT_ESTABLISHED",{**binding,"reason":"No readily available admissible mapped decomposition or dominating retained subflow; aggregate balance is insufficient"}


def comparison_ready(record):
    return (record.get("validation_policy_id")==POLICY and record.get("local_label_certificate")=="PASS"
            and record.get("same_model_coarse_fixed")=="INTERVAL_COMPATIBLE"
            and record.get("policy_decision")=="ELIGIBLE_LOCAL_DATA"
            and record.get("mapped_comparison") in {"APPLICABLE_INTERVAL_COMPATIBLE","NOT_APPLICABLE_SOURCE_PATH_MISMATCH","NOT_ESTABLISHED_NONBLOCKING"})


class RegionLedger:
    def __init__(self,root,active_ids):
        self.root=Path(root); ids=list(active_ids)
        if not ids or len(ids)!=len(set(ids)):raise SharedIntegrityError("Invalid frozen active denominator")
        self.active=frozenset(ids);self.path=self.root/"control/region_states.json"
        if self.path.exists():
            record=read_json(self.path)
            if set(record["states"])!=self.active:raise SharedIntegrityError("Failure denominator changed")
            self.states=record["states"]
        else:self.states={rid:dict(state="PENDING",reason="UNPROCESSED_IS_NOT_FAILURE") for rid in ids}
        self._save()

    def _save(self):
        write_json(self.path,dict(policy_id="REGIONAL_QUARANTINE_LT5_ACTIVE_V1",N=len(self.active),states=self.states),new=False)

    @property
    def failed(self):return {rid for rid,x in self.states.items() if x["state"]==QUARANTINED}
    @property
    def within_limit(self):return 20*len(self.failed)<len(self.active)

    def set(self,rid,state,reason,**evidence):
        with WRITE_LOCK:
            if rid not in self.active or state not in STATES:raise SharedIntegrityError("Invalid regional state transition")
            old=self.states[rid]["state"]
            if old==QUARANTINED and state!=QUARANTINED and not evidence.get("all_required_revalidated"):
                raise SharedIntegrityError("Quarantine recovery requires complete revalidation")
            self.states[rid]=dict(state=state,reason=reason,updated_unix=time.time(),**evidence)
            event(self.root/"control/region_state_history.jsonl",dict(region_id=rid,old=old,**self.states[rid]))
            self._save()

    def quarantine(self,c,rid,exc,stage):
        if not isinstance(exc,LocalRegionError):raise SharedIntegrityError("Unknown error scope cannot consume regional allowance")
        with WRITE_LOCK:
            rd=self.root/"regions"/rid.removeprefix("partC_p2_region_")
            desc=read_json(rd/"descriptor.json")
            paths=[p for p in [rd/"local_graph.npz",rd/"support_map.npz",rd/"baseline_inputs.npz",rd/"baseline_plan.parquet",rd/"labels/catalog_index.json"] if p.exists()]
            incident=dict(region_id=rid,stage=stage,code=exc.code,error=repr(exc),traceback=traceback.format_exc(),
                scope="TYPED_REGION_LOCAL_CHECK",evidence=exc.evidence,model_hash=desc["model_hash"],support_hash=desc["support_hash"],
                native_sources=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in paths],
                retry_requires="One targeted replay only after proved fix, tests, commit and complete local revalidation",
                retry_command=f'{c["python_executable"]} -X utf8 -m partC_rev1_src.phase6.run --config {c["effective_config_path"]} --stage resume-replay --region {rid} --fix-record APPROVED_FIX_RECORD.json')
            incident_id=digest([rid,stage,exc.code,time.time()])[:20]
            path=self.root/"failures"/rid.removeprefix("partC_p2_region_")/(incident_id+".json")
            write_json(path,incident)
            event(self.root/"failures/region_failure_ledger.jsonl",dict(**incident,incident_id=incident_id,incident_path=str(path)))
            self.set(rid,QUARANTINED,exc.code,incident_path=str(path))
            fingerprint=digest([stage,exc.code])
            occurrences={r for r,v in self.states.items() if v["state"]==QUARANTINED and v["reason"]==exc.code}
            if len(occurrences)>=3:
                # Persist scope evidence and stop affected dispatch for bounded common-cause triage.
                # A numeric failure count alone is never permission to hide a shared defect.
                probe=dict(fingerprint=fingerprint,regions=sorted(occurrences),code=exc.code,
                    status="BOUNDED_COMMON_CAUSE_TRIAGE_REQUIRED",first_incident=str(path))
                probe_path=self.root/"failures"/("common_cause_"+fingerprint+".json")
                prior=read_json(probe_path) if probe_path.exists() else None
                if prior and prior.get("status")=="ISOLATED_FINITE_BUDGET_MISSES_VERIFIED":
                    probe={**prior,"regions":sorted(occurrences)}
                elif exc.code in {"MANDATORY_ACCURACY_UNMET","FIXED_FINE_ACCURACY_UNMET"}:
                    probe.update(probe_nonconvergence(self.root,[self.states[r]["incident_path"] for r in sorted(occurrences)[:3]]))
                write_json(probe_path,probe,new=False)
                if probe["status"]!="ISOLATED_FINITE_BUDGET_MISSES_VERIFIED":
                    raise SharedIntegrityError("Three-region novel signature requires source/code scope triage: "+exc.code)
            if not self.within_limit:raise FailureThreshold("79 unresolved active regional failures; stop new dispatch")
            return dict(region_id=rid,status=QUARANTINED,incident_path=str(path),failed_active_count=len(self.failed))

    def summary(self,*,global_integrity=True,inactive_valid=True):
        healthy=sorted(r for r,v in self.states.items() if v["state"]==READY)
        failed=sorted(self.failed);pending=sorted(self.active-set(healthy)-set(failed))
        ready=not pending and self.within_limit and global_integrity and inactive_valid
        return dict(N=len(self.active),F=len(failed),healthy_ids=healthy,quarantined_ids=failed,pending_ids=pending,
            failed_active_count=len(failed),healthy_active_count=len(healthy),pending_active_count=len(pending),
            failure_fraction=len(failed)/len(self.active),strict_limit_pass=self.within_limit,
            healthy_scope_complete=ready,all_regions_dataset_complete=ready and not failed,
            global_integrity_pass=global_integrity,inactive_records_valid=inactive_valid)


@lru_cache(maxsize=2)
def ledger(root):
    frame=pd.read_parquet(Path(root)/"registry/regions.parquet")
    ids=frame.loc[frame.m_active.gt(0),"region_id"].tolist()
    if len(frame)!=1847 or len(ids)!=1566:raise SharedIntegrityError("Frozen region denominator differs")
    return RegionLedger(root,ids)


def require_readiness(root,region_id=None):
    root=Path(root); p=root/"validation/BASELINE_SCOPE_READINESS.json"
    if (root/"control/STOP_SCIENCE.json").exists():raise UnavailableRegion("UNAVAILABLE_REGION: campaign integrity/resource stop")
    if not p.exists():raise UnavailableRegion("UNAVAILABLE_REGION: healthy baseline readiness is pending")
    gate=read_json(p)
    if not gate.get("global_integrity_pass") or not gate.get("healthy_scope_complete") or not gate.get("strict_limit_pass"):
        raise UnavailableRegion("UNAVAILABLE_REGION: global/healthy baseline gate is not ready")
    if not (root/"validation/review_resolution.json").exists():raise SharedIntegrityError("No current hash-bound review resolution")
    resolution=read_json(root/"validation/review_resolution.json")
    if resolution.get("validation_policy_id")!=POLICY or resolution.get("status")!="VERIFIED_RESOLVED":
        raise SharedIntegrityError("Unverified policy resolution")
    if gate.get("review_resolution_sha256")!=sha(root/"validation/review_resolution.json"):
        raise SharedIntegrityError("Readiness policy binding changed")
    if gate.get("region_status_path") and sha(gate["region_status_path"])!=gate["region_status_sha256"]:
        raise SharedIntegrityError("Readiness region manifest changed")
    states=read_json(root/"control/region_states.json")["states"]
    if len(states)!=1566:raise SharedIntegrityError("Readiness denominator changed")
    failed={rid for rid,v in states.items() if v["state"]==QUARANTINED}
    if 20*len(failed)>=1566:raise UnavailableRegion("UNAVAILABLE_REGION: failure threshold reached")
    if region_id is not None and (region_id not in states or states[region_id]["state"]!=READY):
        reason=states.get(region_id,{}).get("reason","unknown active region")
        raise UnavailableRegion("UNAVAILABLE_REGION: "+reason)
    return gate


def probe_nonconvergence(root,incident_paths):
    """Bounded native replays distinguish three accuracy misses from bad certificates."""
    from .oracle import Oracle
    results=[];started=time.perf_counter()
    for incident_path in incident_paths:
        incident=read_json(incident_path);e=incident["evidence"]
        rd=Path(root)/"regions"/incident["region_id"].removeprefix("partC_p2_region_")
        p=rd/"local_graph.npz" if e["namespace"]=="labels" else rd/"fixed_fine/definition.npz"
        a=read_npz(p);a.pop("native_demand",None);a.pop("fine_to_coarse",None)
        candidate=read_npz(e["candidate_npz"]);o=Oracle(a)
        o.import_catalog({k.removeprefix("catalog_"):v for k,v in candidate.items() if k.startswith("catalog_")})
        h=np.zeros(len(o.paths));h[candidate["path_ids"]]=candidate["weights"]
        result=o.certify(candidate["demand"],h,0,time.perf_counter(),"COMMON_CAUSE_NATIVE_REOPEN",0,o.lmo_count)
        if not np.allclose(result.z,candidate["z"],rtol=2e-12,atol=1e-10):
            raise SharedIntegrityError("Nonconvergence probe found inconsistent native paths")
        if not np.allclose(result.tau,candidate["tau"],rtol=2e-12,atol=1e-10):
            raise SharedIntegrityError("Nonconvergence probe found inconsistent refreshed pricing")
        results.append(dict(region_id=incident["region_id"],certificate_valid=True,quality=result.quality,
            gap=result.relative_gap,candidate_sha256=sha(e["candidate_npz"])))
        if time.perf_counter()-started>60:raise SharedIntegrityError("Bounded common-cause probe time cap")
    return dict(status="ISOLATED_FINITE_BUDGET_MISSES_VERIFIED",scope="Three independent native path/pricing reopens show valid certificates; the required accuracy was not reached within fixed local limits",native_replays=results)
