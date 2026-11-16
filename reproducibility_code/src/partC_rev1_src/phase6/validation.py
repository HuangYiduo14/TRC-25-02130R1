"""Gap-aware population checks and independent certificate reopens."""
from pathlib import Path
import time
import numpy as np
import pandas as pd

from .common import read_json, read_npz, write_json, write_frame, input_hash, arrays_hash, digest
from .registry import region_dir
from .storage import Store
from .sampling import rng_for
from .resume_policy import enabled,comparison_ready,LocalRegionError


def pair_check(a, b):
    allowance = a.roundoff_allowance + b.roundoff_allowance
    p = float(b.value_upper - a.supporting_intercept - a.tau @ b.demand)
    q = float(a.value_upper - b.supporting_intercept - b.tau @ a.demand)
    mono = float((a.tau-b.tau) @ (a.demand-b.demand) + a.gap+b.gap)
    return dict(plane_ij_slack=p, plane_ji_slack=q, monotonicity_slack=mono,
                allowance=allowance, passed=min(p,q,mono) >= -allowance)


def cycle_check(results):
    lhs = sum(float(a.tau @ (b.demand-a.demand)) for a,b in zip(results,results[1:]+results[:1]))
    bound = sum(a.gap+a.roundoff_allowance for a in results)
    return dict(lhs=lhs, gap_and_roundoff_bound=bound, passed=lhs<=bound)


def baseline_audit(c, rid, *, fresh=True):
    rd = region_dir(c, rid); d = read_json(rd / "descriptor.json")
    if not d["m_active"]:
        forbidden = [p.name for p in rd.iterdir() if p.name in {"labels", "fixed_fine", "scales.npz", "baseline_inputs.npz"}]
        if forbidden: raise ValueError("Inactive region acquired optimization artifacts")
        return dict(region_id=rid, status="PASS", region_status="NO_ACTIVE_LOW_PAIRS", optimizer_calls=0)
    a = read_npz(rd / "local_graph.npz"); scales = read_npz(rd / "scales.npz")
    store = Store(rd,d,a)
    plan = pd.read_parquet(rd / "baseline_plan.parquet"); xs = read_npz(rd / "baseline_inputs.npz")["x"]
    missing = set(plan.input_hash)-store.hashes()
    if missing: return dict(region_id=rid,status="MISSING_BASELINE",missing=len(missing))
    results = {}; qualities = {}; maxerror=0.
    for ih in sorted(set(plan.input_hash)):
        r, receipt = store.get(ih,recertify=fresh)
        results[ih]=r; qualities[r.quality]=qualities.get(r.quality,0)+1
        rows=plan.loc[plan.input_hash.eq(ih)]
        if any(not np.array_equal(xs[j],r.demand) for j in rows.input_index): raise ValueError("Planned native input differs")
        payload=read_npz(store.root/receipt["payload_path"])
        if not np.allclose(payload["normalized_input"],r.demand/scales["input_scale"],rtol=0,atol=0): raise ValueError("Normalized input differs")
        expected=scales["input_scale"]*(r.tau-scales["c0"])/float(scales["Vscale"])
        if not np.array_equal(expected,payload["normalized_gradient"]): raise ValueError("Normalized chain rule differs")
        totals=np.bincount(payload["fine_pair"],weights=payload["fine_amount"],minlength=d["m_active"])
        if not np.allclose(totals,r.demand,rtol=2e-11,atol=1e-12): raise ValueError("Stored fine allocation differs")
        maxerror=max(maxerror,float(np.abs(totals-r.demand).max(initial=0.)))
    hashes=sorted(results)
    rng,_=rng_for(rid,0,"AUDIT_PAIRS",0,support_hash=d["support_hash"])
    pairs=[]
    while len(pairs)<min(32,len(hashes)*(len(hashes)-1)) and len(hashes)>1:
        i,j=rng.choice(len(hashes),2,replace=False)
        key=(int(i),int(j))
        if key not in pairs: pairs.append(key)
    checks=[dict(kind="DETERMINISTIC_PAIR",input_i=hashes[i],input_j=hashes[j],**pair_check(results[hashes[i]],results[hashes[j]])) for i,j in pairs]
    radial=plan.loc[plan.family.isin(["ZERO","LOW","RADIAL_TEMPLATE","RADIAL_STRESS"])].copy()
    radial["total"]=[results[k].demand.sum() for k in radial.input_hash]
    radial=radial.sort_values("total")
    for left,right in zip(radial.input_hash,radial.input_hash.iloc[1:]):
        checks.append(dict(kind="RADIAL_NEIGHBOR",input_i=left,input_j=right,**pair_check(results[left],results[right])))
    cycles=[]
    for j in range(8):
        length=min(3+j%4,len(hashes))
        if length<3: break
        ids=rng.choice(len(hashes),length,replace=False)
        cycles.append(dict(cycle=j,length=length,**cycle_check([results[hashes[k]] for k in ids])))
    lowrow=plan.loc[plan.family.eq("LOW")].iloc[0]; low=results[lowrow.input_hash]
    fd=[]
    for group,rows in plan.loc[plan.family.eq("FINITE_DIFFERENCE")].groupby("direction_index"):
        rows=rows.sort_values("signed_step"); direction=np.asarray(rows.iloc[0].direction,dtype=float)
        slope=float(low.tau@direction); right=rows.iloc[-1]; h=float(right.signed_step)
        upper=(results[right.input_hash].value_upper-low.tangent_lower_at_sample)/h
        tol=(low.roundoff_allowance+results[right.input_hash].roundoff_allowance)/h
        left=rows.iloc[0]; lower=None
        if float(left.signed_step)<0:
            lower=(low.tangent_lower_at_sample-results[left.input_hash].value_upper)/(-float(left.signed_step))
            tol+=(low.roundoff_allowance+results[left.input_hash].roundoff_allowance)/(-float(left.signed_step))
        passed=slope<=upper+tol and (lower is None or slope>=lower-tol)
        width=None if lower is None else upper-lower
        fd.append(dict(direction_index=float(group),central=lower is not None,slope=slope,lower=lower,upper=upper,
            allowance=tol,passed=passed,status="NOISE_DOMINATED" if width is not None and width>.1*max(1.,abs(slope)) else "RESOLVED_BOUND"))
    if not all(v["passed"] for v in checks+cycles+fd):
        raise LocalRegionError("GAP_AWARE_REGIONAL_AUDIT_FAILED",evidence=dict(region_id=rid,
            failed_checks=[v for v in checks+cycles+fd if not v["passed"]],fresh_local_certificates_verified=fresh))
    st=read_json(rd/"baseline_status.json")
    ready=comparison_ready(st["hierarchy"]) if enabled(c) else st["hierarchy"]["status"]=="PASS"
    if not ready:return dict(region_id=rid,status="MISSING_FIXED_FINE")
    if (rd/"BASELINE_AUDIT_PASS.json").exists():
        previous=read_json(rd/"BASELINE_AUDIT_PASS.json")
        if previous.get("validation_policy_id")==c.get("validation_policy_id"):
            return {**previous,"fresh_recertification":fresh,"rechecked_existing_audit":True}
    write_frame(rd/"baseline_pair_audit.parquet",pd.DataFrame(checks))
    write_frame(rd/"baseline_cycle_audit.parquet",pd.DataFrame(cycles))
    write_frame(rd/"baseline_finite_difference_audit.parquet",pd.DataFrame(fd))
    result=dict(region_id=rid,status="PASS",baseline_unique=len(results),qualities=qualities,
        validation_policy_id=c.get("validation_policy_id"),
        pair_checks=len(checks),cycle_checks=len(cycles),finite_difference_directions=len(fd),
        finite_difference_noise_dominated=sum(x["status"]=="NOISE_DOMINATED" for x in fd),
        max_fine_mass_error=maxerror,fresh_recertification=fresh,split_aliases=int(plan.alias.sum()),
        independent_heldout=int((plan.split.ne("TRAIN") & plan.independent_role).sum()))
    write_json(rd/"BASELINE_AUDIT_PASS.json",result)
    return result
