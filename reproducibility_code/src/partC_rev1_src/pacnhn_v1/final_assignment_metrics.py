"""Unused until admitted after export: independent native A01--A06 audit.

No models, assignment solvers, shortest-path kernels or production arrays are
loaded on import. Mathematical reductions below do not call metrics.py.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd

from .control import Context, array_hash, write_json
from .final_training_limits import Reader, Budget, require, _seal

B_COUNT = 87582
KEY_COUNT = 200
SCENARIOS = {"open": "OPEN", "i95": "REPRESENTED_I95"}
ADMITTED = {"CONVERGED", "TIME_LIMIT", "STAGNATION"}
UNAVAILABLE = {"UNRUN_STAGE_TIME_LIMIT", "UNAVAILABLE_INCOMPLETE_PURE_FAMILY",
               "UNAVAILABLE_CURRENT_RAM", "TIME_LIMIT_DURING_INPUT_LOAD", "FAILED"}
GEO = ["tail_longitude", "tail_latitude", "head_longitude", "head_latitude"]
TASKS = {"VERIFY24_CASE_LEDGER", "VERIFY_NATIVE_STATE", "RECOMPUTE_METRICS", "VERIFY_CLOSED_COMPARISONS"}
LIMITATIONS = [
    "Saved fresh certificate/state/served evidence is independently checked; no AON, cost-field/model inference or convergence rerun occurs.",
    "Native backbone coordinates are compared directly; no physical-flow projection or recovery is performed.",
    "The represented I95 mask omits hidden physical backbone-interior movements; it is not a full physical I95 closure.",
    "Open-to-closed traversal changes mix lost served demand with redistribution of retained demand; they are not pure rerouting.",
    "Original OD multiplicity/demand comes from the authenticated prior exact membership audit; current served masks are checked independently.",
]
TIMING_BOUNDARY = ('Before graph/model load through final flow/trace/support exports; '
                  'excludes final RESULT/STAGES/HISTORY writes and post-run admission')


def method_id(case):
    return ('P/REFERENCE' if case['assignment_role']=='REFERENCE_SELF' else
            'P/'+case['solver'] if case['network']=='P' else case['suite_id']+'/'+case['family'])


def admitted(result):
    return result.get('status') in ADMITTED and result.get('final_certificate_fresh') is True


def constant(frame, field, value):
    """Check exported identities without accepting missing rows or coercing booleans."""
    require(len(frame)>0 and field in frame, 'Missing exported metadata: '+field)
    values=frame[field].to_numpy()
    if isinstance(value,bool): require(np.all(booleans(values)==value),'Exported boolean differs: '+field)
    elif value is None: require(pd.isna(values).all(),'Expected absent metadata: '+field)
    elif isinstance(value,str): require(np.all(values==value),'Exported identity differs: '+field)
    else: near(values,np.full(len(values),value),field)


def finite(value, name, *, nonnegative=True):
    require(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            and (not nonnegative or value >= 0), "Invalid " + name)
    return float(value)


def near(actual, expected, name, *, rtol=1e-10, atol=1e-10):
    aa = np.asarray(actual, dtype=float); ee = np.asarray(expected, dtype=float)
    require(aa.shape == ee.shape and np.allclose(aa, ee, rtol=rtol, atol=atol, equal_nan=True), "Mismatch: " + name)


def vector(value):
    a = np.asarray(value)
    require(a.dtype == np.dtype("float64") and a.ndim == 1 and len(a) > 0
            and np.isfinite(a).all() and np.all(a >= 0), "Expected native finite nonnegative float64 vector")
    return a


def booleans(value):
    values = np.asarray(value)
    if values.dtype == bool:
        return values
    strings = values.astype(str)
    require(np.all(np.isin(strings, ["True", "False"])), "Invalid exported boolean")
    return strings == "True"


def planned_cases(matrix, outcomes):
    expected = {}
    for solver in ("CFW", "FW", "MSA"):
        for scenario in SCENARIOS:
            key = f"P_{solver}_{scenario}"
            expected[key] = dict(case_id=key, network="P", suite_id=None, family=None,
                                 solver=solver, scenario=scenario, priority="PRIMARY")
    for suite in ("s03", "s04", "s02", "s05", "s06"):
        for family in (("FFTT", "POWER4", "XGBOOST", "ICNN") if suite == "s03" else ("FFTT",)):
            for scenario in SCENARIOS:
                key = f"AC_{suite}_{family}_{scenario}"
                expected[key] = dict(case_id=key,network="AC",suite_id=suite,family=family,
                    solver="MSA" if family == "XGBOOST" else "FW",scenario=scenario,
                    priority="PRIMARY" if suite == "s03" else "ADDITIONAL")
    require(len(matrix) == 22 and {r["case_id"]:r for r in matrix} == expected, "Frozen22case matrix differs")
    actual = {r["case"]["case_id"]:r for r in outcomes}
    require(len(outcomes) == len(actual) == 22 and set(actual) == set(expected), "Missing/duplicate terminal matrix outcome")
    for key, row in actual.items():
        require(row["case"] == expected[key] and row["status"] in ADMITTED | UNAVAILABLE, "Changed/nonterminal planned outcome")
    priorities = [r["priority"] for r in matrix]
    require(priorities[:14] == ["PRIMARY"]*14 and priorities[14:] == ["ADDITIONAL"]*8,
            "Additional jobs precede planned primary priority")
    return expected, actual


def independent_metrics(reference, predicted, key_mask, base, k4):
    """Direct ratio-of-sums reductions; retain every reference-zero link."""
    r, p = vector(reference), vector(predicted)
    key = np.asarray(key_mask)
    require(p.shape == r.shape and key.dtype == bool and key.shape == r.shape and key.any(), "Metric population mismatch")
    base, k4 = vector(base), np.asarray(k4)
    require(base.shape == r.shape and k4.dtype == np.dtype("float64") and k4.shape == r.shape
            and np.isfinite(k4).all() and np.all(k4 >= 0), "Invalid fixed B costs")
    d = p-r; absolute = np.abs(d); maximum = np.maximum(r,p)
    per = dict(signed_error=d, absolute_error=absolute,
        signed_bounded_error=np.divide(d,maximum,out=np.zeros_like(d),where=maximum>0),
        signed_relative_error=np.divide(d,r,out=np.full_like(d,np.nan),where=r>0),
        absolute_relative_error=np.divide(absolute,r,out=np.full_like(d,np.nan),where=r>0))
    total = math.fsum(r)
    per["backbone_wmape_contribution_percent"] = absolute*100/total if total else np.full_like(d,np.nan)
    populations = {}
    for label, mask in (("BACKBONE",np.ones(len(r),bool)),("KEY_LINKS",key)):
        rr, pp, dd = r[mask], p[mask], d[mask]
        numerator = math.fsum(np.abs(dd)); denominator = math.fsum(rr)
        scale = float(np.max(np.abs(dd)))
        rmse = scale*float(np.sqrt(np.mean((dd/scale)**2))) if scale else 0.
        with np.errstate(over="raise", invalid="raise"):
            rc = float(rr@(base[mask]+k4[mask]*rr**4)); pc = float(pp@(base[mask]+k4[mask]*pp**4))
        populations[label] = dict(link_count=len(rr), reference_flow_sum=denominator, absolute_error_sum=numerator,
            wmape_percent=100*numerator/denominator if denominator else None,
            wmape_status="DEFINED" if denominator else "UNDEFINED_ZERO_REFERENCE_TOTAL", mae=numerator/len(rr),
            rmse=rmse,mean_signed_error=math.fsum(dd)/len(rr),reference_zero_link_count=int(np.count_nonzero(rr==0)),
            reference_zero_positive_prediction_count=int(np.count_nonzero((rr==0)&(pp>0))),
            reference_backbone_cost=rc,predicted_backbone_cost=pc,
            backbone_cost_error_percent=100*abs(pc-rc)/rc if rc else None,
            backbone_cost_status="DEFINED" if rc else "UNDEFINED_ZERO_REFERENCE_COST")
    return populations, per


def verify_certificate(result, flow, *, graph_identity, scenario_OD_identity, case, stage3_clock=None):
    flow = vector(flow); cert = result["certificate"]
    require(result["status"] in ADMITTED and result.get("final_certificate_fresh") is True and cert.get("fresh") is True,
            "Uncertified terminal state")
    require(result.get("reported_state") == "LAST_FRESH_CERTIFIED_STATE"
            and isinstance(result["reported_state_iteration"],int) and result["reported_state_iteration"] >= 0,
            "Missing certified reported iteration")
    require(result["graph_identity"] == graph_identity and result["OD_identity"] == scenario_OD_identity
            and cert["state_hash"] == array_hash({"flow":flow}), "Native flow/graph/OD certificate identity differs")
    tstt=finite(cert["TSTT"],"TSTT"); sptt=finite(cert["SPTT"],"SPTT")
    gap=finite(cert["raw_signed_gap"],"signed gap",nonnegative=False); allowance=finite(cert["arithmetic_allowance"],"allowance")
    require(gap >= -allowance, "Materially negative gap")
    near(gap,tstt-sptt,"signed gap",atol=max(1e-12,allowance))
    near(allowance,1e-10*max(1.,abs(tstt),abs(sptt)),"gap allowance",rtol=1e-12,atol=1e-12)
    if sptt:
        finite(cert["RG_SPTT"],"RG_SPTT",nonnegative=False)
        near(cert["RG_SPTT"],gap/sptt,"relative gap",rtol=1e-9,atol=1e-12)
    else:
        require(cert["RG_SPTT"] is None, "Zero SPTT must preserve undefined relative gap")
    has_potential = case.get("family") != "XGBOOST"
    require(result["objective_has_potential"] is has_potential, "Solver/objective family mismatch")
    require(cert["interpretation"] == ("CONVEX_POTENTIAL_FW_GAP" if has_potential else "POINTWISE_VI_RESIDUAL_NO_POTENTIAL"),
            "FW-gap/VI interpretation differs")
    if has_potential: finite(cert["value"],"potential")
    else: require(cert["value"] is None,"Cost-only XGBoost must not invent a potential")
    is_reference = case["assignment_role"] == "REFERENCE_SELF"
    tolerance = 1e-4 if is_reference else 1e-3
    require(result["nominal_tolerance"] == tolerance and result["method"] == case["solver"], "Method/nominal tolerance differs")
    if result["status"] == "CONVERGED":
        require(cert["RG_SPTT"] <= tolerance if sptt else abs(gap)<=allowance,"Convergence exceeds fresh tolerance")
    start=finite(result["complete_query_start_epoch"],"query start"); end=finite(result["deadline_epoch"],"query deadline")
    elapsed=finite(result["elapsed_seconds"],"complete elapsed")
    require(0 < end-start <= (10800 if is_reference else 2700)+1e-6,"Declared complete-query cap exceeds authority")
    if not is_reference:
        require(stage3_clock is not None and start >= stage3_clock["original_stage_start_epoch"]
                and end <= stage3_clock["deadline_epoch"],"Query outside original Stage3 window")
        require(result["intent"]["initialization"] == "COLD_FEASIBLE_AON"
                and result["intent"]["complete_query_start_epoch"] == start
                and result["intent"]["deadline_epoch"] == end,"Cold start/timing boundary differs")
        require(result["independent_cold_control"] is (case["network"] == "P"),"Reference/control role differs")
    return dict(RG_SPTT=cert["RG_SPTT"],solver=result["method"],residual_kind="FW_GAP" if has_potential else "VI_RESIDUAL",
                declared_cap_seconds=end-start,complete_elapsed_seconds=elapsed,observed_complete_wall_overrun_seconds=max(0.,elapsed-(end-start)))


def delta_statistics(open_flow, closed_flow):
    oo, cc=vector(open_flow),vector(closed_flow);require(oo.shape==cc.shape,"Paired native coordinates differ")
    dd=cc-oo
    return dict(positive_traversal_change=math.fsum(dd[dd>0]),negative_traversal_change_magnitude=math.fsum(-dd[dd<0]),
        net_traversal_change=math.fsum(dd),absolute_traversal_change=math.fsum(np.abs(dd)),
        open_traversal_sum=math.fsum(oo),closed_traversal_sum=math.fsum(cc),flow_units="ANNUAL_KTON",
        interpretation="Traversal differences include changed served demand; not original OD loss or pure rerouting")


class Sources:
    def __init__(self, reader, rows):
        self.reader=reader;self.files={}
        for row in rows:
            p=str(Path(row["path"]).resolve())
            require(p not in self.files or self.files[p]["sha256"]==row["sha256"],"Conflicting explicit source bindings")
            self.files[p]=row
    def binding(self,path):
        key=str(Path(path).resolve())
        require(key in self.files,"Missing explicit source binding: "+key)
        return self.files[key]
    def path(self,path):
        b=self.binding(path);return self.reader.bind(b["path"],b["sha256"])
    def json(self,path,maximum=32*1024**2):return self.reader.json(self.binding(path),maximum=maximum)
    def npz(self,path,fields):
        p=self.path(path)
        with zipfile.ZipFile(p) as archive:
            sizes={i.filename:i.file_size for i in archive.infolist()}
            require(all(k+'.npy' in sizes for k in fields) and
                    sum(sizes[k+'.npy'] for k in fields)<=384*1024**2,'Requested NPZ payload exceeds bounded reader')
        with np.load(p,allow_pickle=False) as data: result={k:data[k] for k in fields}
        self.reader.check();return result
    def csv(self,path,**kwargs):return pd.read_csv(self.path(path),float_precision="round_trip",**kwargs)


def _row(frame, **keys):
    mask=np.ones(len(frame),bool)
    for key,value in keys.items():mask &= frame[key].astype(str).to_numpy()==str(value)
    found=frame.loc[mask]
    require(len(found)==1,"Missing/duplicate exported row: "+str(keys))
    return found.iloc[0]


def _compare_population(exported, expected):
    for key,value in expected.items():
        actual=exported[key]
        if value is None:require(pd.isna(actual),"Undefined metric was filled: "+key)
        elif isinstance(value,str):require(actual==value,"Population semantic mismatch: "+key)
        elif isinstance(value,int):require(actual==value,"Population count mismatch: "+key)
        else:near(actual,value,key)


def validate_training_coverage(training, support, cases):
    require(training['status']=='STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE','Training is not terminal')
    rows=training['coverage'];lookup={(r['suite_id'],r['family']):r for r in rows}
    suites={r['suite_id']:r['active_regions'] for r in support['suites']}
    require(len(rows)==len(lookup)==20 and set(lookup)=={(s,f) for s in suites for f in ('FFTT','POWER4','ICNN','XGBOOST')},'Training coverage inventory differs')
    for (sid,family),r in lookup.items():
        require(all(isinstance(r[k],int) and not isinstance(r[k],bool) for k in ('expected','available','qualified')),'Noninteger training coverage')
        require(r['expected']==suites[sid] and 0<=r['qualified']<=r['available']<=r['expected']
                and r['complete_family'] is (r['available']==r['expected']),'Invalid complete-family coverage')
        if sid!='s03' and family!='FFTT':require(r['available']==r['qualified']==0,'Unexpected later-resolution training')
    return rows


def verify_method_row(exported, case, result, reference_result, cert, statistics):
    frame=pd.DataFrame([exported])
    fields=dict(method_id=method_id(case),network=case['network'],suite_id=case['suite_id'],scenario=SCENARIOS[case['scenario']],
        solver=case['solver'],assignment_role=case['assignment_role'],reference_case_id='P_REFERENCE_'+case['scenario'],
        certificate_fresh=True,residual_kind=cert['residual_kind'],nominal_tolerance=result['nominal_tolerance'],
        query_cap_seconds=cert['declared_cap_seconds'],timing_boundary=TIMING_BOUNDARY,reference_status=reference_result['status'],
        served_OD_membership_equal=True,reported_state_iteration=result['reported_state_iteration'],
        elapsed_seconds=result['elapsed_seconds'],reference_zero_link_count=statistics['BACKBONE']['reference_zero_link_count'],
        reference_zero_positive_prediction_count=statistics['BACKBONE']['reference_zero_positive_prediction_count'])
    for field,value in fields.items():constant(frame,field,value)
    for field,value in [('RG_SPTT',result['certificate']['RG_SPTT']),('reference_RG_SPTT',reference_result['certificate']['RG_SPTT'])]:
        constant(frame,field,value)


def verify_comparison_metadata(frame,case,ids,closed):
    fields=dict(scenario=SCENARIOS[case['scenario']],reference_case_id='P_REFERENCE_'+case['scenario'],
        case_id=case['case_id'],method_id=method_id(case),assignment_role=case['assignment_role'],
        backbone_order_identity=array_hash({'link_id':ids}),served_OD_membership_equal=True,
        comparison_space='COMMON_NATIVE_BACKBONE',key_origin_scenario='OPEN',closed_backbone_count=int(closed.sum()))
    for field,value in fields.items():constant(frame,field,value)


def audit_clock_window(clock,*,root,binding,run_id,stage4_binding,window,now):
    require(Path(binding['path']).resolve()==root/'final_audits/ASSIGNMENT_METRICS_CLOCK.json','Canonical cumulative metrics clock required')
    require(clock['schema']=='PACNHN_FINAL_ASSIGNMENT_AUDIT_CLOCK_V1' and clock['phase_id']=='PACNHN_V1'
            and clock['run_id']==run_id and clock['stage4_clock_sha256']==stage4_binding['sha256'],'Audit clock authority differs')
    start=finite(clock['start_epoch'],'audit clock start');deadline=finite(clock['deadline_epoch'],'audit clock deadline')
    require(window['stage_start_epoch']<=start<=now and deadline==min(start+1200,window['stage_deadline_epoch']),
            'Audit clock was reset or widened')
    require(now<deadline,'Cumulative metrics audit clock expired')
    return deadline


def run(config_path,*,admission_path,admission_sha256):
    started=time.time(); plain=Reader();config_path=Path(config_path).resolve(strict=True)
    config_sha=hashlib.sha256(config_path.read_bytes()).hexdigest()
    cfg=plain.json(dict(path=str(config_path),sha256=config_sha))
    require(cfg["schema"]=="PACNHN_FINAL_ASSIGNMENT_METRICS_CONFIG_V1" and cfg["phase_id"]=="PACNHN_V1"
            and cfg["step_id"]=="final_assignment_metrics" and cfg["run_id"],"Explicit final metrics contract required")
    require(TASKS <= set(cfg["tasks"]) and cfg["input_versions"] and cfg["expected_outputs"] and cfg["acceptance_tests"],"Incomplete execution contract")
    resources=cfg["resources"]
    require(15<resources["wall_seconds"]<=1200 and 0<resources["single_process_rss_bytes"]<=1024**3
            and 0<resources["tree_rss_bytes"]<=24*1024**3 and resources["minimum_available_bytes"]>=3*1024**3,"Resource limit relaxed")
    for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):require(os.environ.get(name)=="1","Single-thread environment required")
    campaign=plain.json(cfg["campaign_config"]);require(campaign["run_id"]==cfg["run_id"],"Run mismatch")
    ctx=Context(cfg["campaign_config"]["path"],step_id="publication");ctx.require_action("analysis")
    from .handoff import publication_window
    window=publication_window(campaign,campaign_config_sha256=cfg["campaign_config"]["sha256"],
        stage4_clock=cfg["stage4_clock"],clock_amendment=cfg["clock_amendment"],deadline_epoch=cfg["deadline_epoch"])
    audit_clock=plain.json(cfg['audit_clock'])
    original_end=audit_clock_window(audit_clock,root=ctx.root.resolve(),binding=cfg['audit_clock'],run_id=cfg['run_id'],
        stage4_binding=cfg['stage4_clock'],window=window,now=time.time())
    end=min(window["deadline_epoch"],original_end,started+resources["wall_seconds"]);check=Budget(ctx,end-15,resources);check()
    reader=Reader(check);reader.bindings.update(plain.bindings)
    inventory=reader.json(cfg["source_inventory"])
    required={str((Path(__file__).parent/n).resolve()) for n in ("final_assignment_metrics.py","final_training_limits.py",
        "control.py","continuation.py","handoff.py","metrics.py","analysis_export.py","topology.py")}
    test=Path(__file__).parent/"tests/test_final_assignment_metrics.py";required.add(str(test.resolve()))
    code={str(Path(r["path"]).resolve()):r for r in inventory["files"]}
    require(required<=set(code),"Incomplete metric audit source closure")
    for row in code.values():reader.bind(row["path"],row["sha256"])
    admission=reader.json(dict(path=admission_path,sha256=admission_sha256))
    for key,value in dict(schema="PACNHN_FINAL_ASSIGNMENT_METRICS_ADMISSION_V1",decision="ADMIT_FINAL_ASSIGNMENT_METRICS_AUDIT",
        phase_id="PACNHN_V1",run_id=cfg["run_id"],config_sha256=config_sha,source_inventory_sha256=cfg["source_inventory"]["sha256"],
        stage4_clock_sha256=cfg["stage4_clock"]["sha256"],audit_clock_sha256=cfg['audit_clock']['sha256']).items():require(admission.get(key)==value,"Admission mismatch: "+key)
    tests=reader.json(admission["test_receipt"])
    require(tests.get("status")=="PASS" and tests.get("exit_code")==0
            and tests.get("source_sha256")==code[str(Path(__file__).resolve())]["sha256"]
            and tests.get("test_sha256")==code[str(test.resolve())]["sha256"],"Tests bind other code or failed")
    source_index=reader.json(cfg["input_inventory"])
    source=Sources(reader,source_index["files"]);root=ctx.root.resolve()
    stage3=source.json(root/"STAGE3_RESULT.json");clock=source.json(root/"STAGE3_CLOCK.json")
    require(stage3["status"]=="COMPLETE_WITH_RECORDED_OUTCOMES" and stage3["clock"]==clock
            and stage3.get('rows')==22 and stage3.get('primary_rows')==14
            and clock["original_stage_start_epoch"]+finite(stage3['elapsed_seconds'],'Stage3 elapsed')<=window["stage_start_epoch"],"Stage3 not terminal before Stage4")
    provenance=source.json(root/"analysis/PROVENANCE.json")
    require(provenance["status"]=="PASS" and provenance["schema"]=="PACNHN_ANALYSIS_V1","Completed analysis export required")
    for row in provenance["inputs"]+provenance["outputs"]:
        if str(Path(row["path"]).resolve()) in source.files:
            require(source.binding(row["path"])["sha256"]==row["sha256"],"Analysis/source inventory disagrees")
    # Every exported file consumed below must be in the sealed export manifest.
    exports={str(Path(r["path"]).resolve()) for r in provenance["outputs"]}
    names=("methods.csv","metric_populations.csv","native_backbone_vectors.csv","backbone_links.csv","keys.csv","I95.csv","I95_OD_COVERAGE.json")
    require({str(root/"analysis"/n) for n in names}<=exports,"Missing exported metric sources")
    expected,outcomes=planned_cases(source.json(root/"ASSIGNMENT_MATRIX.json"),source.json(root/"ASSIGNMENT_RESULTS.json"))
    cases=[dict(case_id="P_REFERENCE_"+s,network="P",suite_id=None,family=None,solver="CFW",scenario=s,
                priority="REFERENCE",assignment_role="REFERENCE_SELF") for s in SCENARIOS]
    cases += [dict(row,assignment_role="INDEPENDENT_CONTROL" if row["network"]=="P" else "SURROGATE") for row in expected.values()]
    output=Path(cfg["output_dir"]).resolve();require(output.is_relative_to(root/"audits") and not output.exists(),"New owned output required")
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"INTENT.json",dict(config=cfg,config_sha256=config_sha,admission=admission,started_epoch=started,
        work_deadline_epoch=end-15,audit_deadline_epoch=end,no_model_or_solver_inference=True))
    ledger={c["case_id"]:dict(case_id=c["case_id"],disposition="NOT_CHECKED",assignment_role=c["assignment_role"],
        status=None,served_membership_equal=None,full_B_links=None,key_links=None,wmape_backbone_percent=None,
        wmape_key_percent=None,complete_wall_overrun_seconds=None) for c in cases}
    status="FAIL";error=None;current=None;native={};statistics={};deltas={};references={};reference_results={};training_coverage=[]
    try:
        proof=reader.json(cfg["reference_membership_proof"])
        proof_index=reader.json(cfg["reference_membership_sources"])
        require(proof["status"]=="PASS" and proof["schema"]=="PACNHN_FROZEN_REFERENCE_I95_MEMBERSHIP_V1"
                and proof["run_id"]==cfg["run_id"] and proof["demand_renormalized"] is False,"Invalid prior exact OD proof")
        proof_sources={str(Path(r["path"]).resolve()):r for r in proof_index["bindings"]}
        for p in [root/"topology/common_od.npz"]+[root/"references"/s/n for s in SCENARIOS for n in ("RESULT.json","coverage.npz","final_flow.npz")]:
            require(source.binding(p)["sha256"]==proof_sources[str(p)]["sha256"],"Reference OD proof source changed")
            source.path(p)
        topo=source.json(root/"topology/RESULT.json");require(topo["status"]=="PASS" and topo["OD_identity"]==proof["common_OD_identity"],"Common OD/topology mismatch")
        backbone=pd.read_parquet(source.path(root/"topology/backbone.parquet"),columns=["link_id"]+GEO)
        require(len(backbone)==B_COUNT and backbone.link_id.nunique()==B_COUNT,"Common native B population differs")
        xy=backbone[GEO].to_numpy(float)
        require(np.isfinite(xy).all() and np.all(abs(xy[:,[0,2]])<=180) and np.all(abs(xy[:,[1,3]])<=90),'Invalid native geography/units')
        ids=backbone.link_id.to_numpy(dtype=str)
        common=source.npz(root/"topology/common_graph.npz",["edge_ids","closed_mask","base","k4"])
        require(np.array_equal(common["edge_ids"][:B_COUNT],ids) and common["closed_mask"].dtype==bool
                and common['closed_mask'].shape==common['edge_ids'].shape,"Frozen native B identity/mask mismatch")
        common['edge_count']=len(common['edge_ids']);common['edge_ids']=common['edge_ids'][:B_COUNT].copy()
        for field in ('base','k4'):common[field]=common[field][:B_COUNT].copy()
        graphs={"P":(common,topo["graph_identity"])}
        support=source.json(root/"SUPPORT_FREEZE.json",maximum=16*1024**2)
        require(support["status"]=="PASS","Frozen support is not PASS")
        witness=source.json(root/'COMPLETE_WITNESS_AUDIT.json')
        require(witness['status']=='PASS' and witness['support_freeze_sha256']==source.binding(root/'SUPPORT_FREEZE.json')['sha256'],'Complete witness/support binding differs')
        training=source.json(root/'training/RESULT.json')
        training_coverage=validate_training_coverage(training,support,cases)
        coverage_by_family={(r['suite_id'],r['family']):r for r in training_coverage}
        for sid in ("s02","s03","s04","s05","s06"):
            report=source.json(root/"acnhn"/sid/"REPORT.json")
            require([s for s in support["suites"] if s["suite_id"]==sid]==[report] and report['OD_identity']==topo['OD_identity'],"AC report differs from frozen support/common OD")
            path=root/"acnhn"/sid/"graph.npz";require(source.binding(path)["sha256"]==report["graph_sha256"],"Frozen AC hash differs")
            graph=source.npz(path,["edge_ids","closed_mask","base","k4"])
            for key in graph:require(np.array_equal(graph[key][:B_COUNT],common[key][:B_COUNT]),"Fixed B differs across resolution: "+key)
            require(graph['closed_mask'].dtype==bool and graph['closed_mask'].shape==graph['edge_ids'].shape,'Invalid complete AC mask')
            graphs[sid]=(dict(edge_count=len(graph['edge_ids']),closed_mask=graph['closed_mask']),report["graph_identity"])
            del graph
        keymeta=source.json(root/"metrics/keys/KEY_SELECTION_FREEZE.json")
        frozenkeys=source.csv(root/"metrics/keys/key_links_200.csv",dtype={"link_id":str})
        require(source.binding(root/"metrics/keys/key_links_200.csv")["sha256"]==keymeta["key_file_sha256"]
                and keymeta["key_ids"]==frozenkeys.link_id.tolist() and len(frozenkeys)==KEY_COUNT
                and frozenkeys.link_id.nunique()==KEY_COUNT and keymeta["prediction_errors_used"] is False,"Frozen200selection changed")
        require(keymeta['schema']=='pacnhn.fixed_keys.v1' and keymeta['reference_id']=='P_REFERENCE_open'
                and keymeta['reference_scenario']=='OPEN' and keymeta['target']==KEY_COUNT and keymeta['backbone_count']==B_COUNT
                and keymeta['key_selection_role']=='REFERENCE_ONLY' and keymeta['full_backbone_population_filtered'] is False,'Key selection source/scope changed')
        for binding in keymeta['source_bindings']:
            require(source.binding(binding['path'])['sha256']==binding['sha256'],'Key source binding differs');source.path(binding['path'])
        reference_keys=source.csv(root/'metrics/keys/open_backbone_reference.csv',dtype={'link_id':str})
        require(source.binding(root/'metrics/keys/open_backbone_reference.csv')['sha256']==keymeta['reference_file_sha256']
                and np.array_equal(reference_keys.link_id.to_numpy(),ids),'Key reference population differs')
        keymask=np.isin(ids,frozenkeys.link_id.to_numpy())
        require(np.count_nonzero(keymask)==KEY_COUNT and keymeta["backbone_order_identity"]==array_hash({"link_id":ids}),"Key/native join differs")
        wide=source.csv(root/"analysis/native_backbone_vectors.csv",dtype={"link_id":str})
        methods=source.csv(root/"analysis/methods.csv");populations=source.csv(root/"analysis/metric_populations.csv")
        require(len(methods)==24 and set(methods.case_id)==set(ledger),"Exported24case ledger differs")
        require(len(wide)==B_COUNT and np.array_equal(wide.link_id.to_numpy(),ids)
                and np.array_equal(wide.native_index,np.arange(B_COUNT)) and np.array_equal(booleans(wide.is_key),keymask)
                and np.array_equal(booleans(wide.represented_I95_closed),common["closed_mask"][:B_COUNT]),"Wide native coordinates/keys/mask differ")
        near(wide[GEO].to_numpy(),backbone[GEO].to_numpy(),"native geometry",rtol=0,atol=0)
        exportedkeys=source.csv(root/"analysis/keys.csv",dtype={"link_id":str})
        require(exportedkeys.equals(frozenkeys),"Exported key coordinates/metadata/order differ")
        for case in cases:
            check();current=case["case_id"];row=ledger[current];scenario=case["scenario"]
            result=source.json(root/("references" if case["assignment_role"]=="REFERENCE_SELF" else "assignments")/
                               (scenario if case["assignment_role"]=="REFERENCE_SELF" else current)/"RESULT.json") if (
                    case["assignment_role"]=="REFERENCE_SELF" or outcomes[current]["status"] not in {"UNRUN_STAGE_TIME_LIMIT","UNAVAILABLE_INCOMPLETE_PURE_FAMILY"}) else outcomes[current]
            if case["assignment_role"]!="REFERENCE_SELF":require(result==outcomes[current],"Case result/outcome differs")
            exported=_row(methods,case_id=current);require(exported.status==result["status"],"Published case status differs")
            for field,value in dict(method_id=method_id(case),scenario=SCENARIOS[scenario],network=case['network'],suite_id=case['suite_id'],
                solver=case['solver'],assignment_role=case['assignment_role'],reference_case_id='P_REFERENCE_'+scenario).items():
                constant(pd.DataFrame([exported]),field,value)
            row["status"]=result["status"]
            if not admitted(result):
                require(case['assignment_role']!='REFERENCE_SELF','Mandatory P reference is not certified')
                require("flow__"+current not in wide and populations.loc[populations.case_id==current].empty,"Unavailable case has invented native measurements")
                for field in ("wmape_backbone_percent","wmape_key_percent"):
                    require(field not in exported or pd.isna(exported[field]),"Unavailable case has invented accuracy")
                constant(pd.DataFrame([exported]),'certificate_fresh',False)
                row["disposition"]="VERIFIED_UNAVAILABLE_OR_UNRUN";continue
            if case['network']=='AC':require(coverage_by_family[case['suite_id'],case['family']]['complete_family'],'Admitted pure family is incomplete')
            if case['assignment_role']!='REFERENCE_SELF':
                intent=source.json(root/'intents'/f'{current}.json')
                require(intent==result['intent'] and intent['registry_sha256']==training['registry_sha256']
                        and intent['phase_id']==cfg['phase_id'] and intent['run_id']==cfg['run_id']
                        and intent['OD_identity']==topo['OD_identity'],'Case intent/registry/common OD differs')
            directory=root/("references" if case["assignment_role"]=="REFERENCE_SELF" else "assignments")/(scenario if case["assignment_role"]=="REFERENCE_SELF" else current)
            flow=source.npz(directory/"final_flow.npz",["flow"])["flow"]
            graph,graph_identity=graphs["P" if case["network"]=="P" else case["suite_id"]]
            require(flow.shape==(graph['edge_count'],),"Full native flow dimension differs")
            cert=verify_certificate(result,flow,graph_identity=graph_identity,scenario_OD_identity=proof["scenario_OD_identities"][scenario],case=case,stage3_clock=clock)
            history=source.json(directory/"HISTORY.json")
            require(history and history[-1]["iteration"]==result["reported_state_iteration"]
                    and history[-1]["state_hash"]==result["certificate"]["state_hash"],"History/final state differs")
            serve=source.npz(directory/"coverage.npz",["serve"])["serve"]
            require(serve.dtype==bool and serve.shape==(proof["aggregate_OD_rows"],),"Served mask population/order differs")
            if scenario=="i95":require(np.all(flow[graph["closed_mask"]]==0),"Nonzero closed native flow")
            b=flow[:B_COUNT].copy();del flow
            if case["assignment_role"]=="REFERENCE_SELF":
                references[scenario]=(b,serve);reference_results[scenario]=result
                if scenario=='open':
                    require(keymeta['open_reference_flow_identity']==array_hash({'flow':b}),'Frozen key reference state differs')
                    near(reference_keys.lambda_reference.to_numpy(),b,'Frozen open reference',rtol=0,atol=0)
                    indices=frozenkeys.native_index.to_numpy()
                    require(np.issubdtype(indices.dtype,np.integer) and np.all((indices>=0)&(indices<B_COUNT))
                            and np.array_equal(ids[indices],frozenkeys.link_id.to_numpy()),'Key coordinates differ')
                    near(frozenkeys.lambda_reference.to_numpy(),b[indices],'Frozen key flow',rtol=0,atol=0)
                    require(np.all(b[indices]>0),'Frozen OPEN key pool has zero-reference entries')
            reference,reference_serve=references[scenario]
            require(np.array_equal(serve,reference_serve),"Within-scenario exact served membership changed")
            stats,per=independent_metrics(reference,b,keymask,common["base"][:B_COUNT],common["k4"][:B_COUNT])
            near(wide["flow__"+current].to_numpy(float),b,"Exported native vector",rtol=0,atol=0)
            for population,values in stats.items():_compare_population(_row(populations,case_id=current,population=population),values)
            population_rows=populations.loc[populations.case_id==current]
            closed=common['closed_mask'][:B_COUNT] if scenario=='i95' else np.zeros(B_COUNT,bool)
            verify_comparison_metadata(population_rows,case,ids,closed)
            constant(population_rows,'flow_units','ANNUAL_KTON');constant(population_rows,'cost_units','ANNUAL_KTON_SECONDS')
            verify_method_row(exported,case,result,reference_results[scenario],cert,stats)
            for field,pop in (("wmape_backbone_percent","BACKBONE"),("wmape_key_percent","KEY_LINKS")):
                value=stats[pop]["wmape_percent"]
                if value is None:require(pd.isna(exported[field]),"Undefined method denominator filled")
                else:near(exported[field],value,field)
            native[current]=b;statistics[current]=stats
            row.update(disposition="VERIFIED_NATIVE_COMPARISON",served_membership_equal=True,full_B_links=B_COUNT,key_links=KEY_COUNT,
                wmape_backbone_percent=stats["BACKBONE"]["wmape_percent"],wmape_key_percent=stats["KEY_LINKS"]["wmape_percent"],
                complete_wall_overrun_seconds=cert["observed_complete_wall_overrun_seconds"])
            pd.DataFrame(ledger.values()).to_csv(output/"CASE_LEDGER.csv",index=False)
        require(set(wide.columns[wide.columns.str.startswith("flow__")])=={"flow__"+key for key in native},"Wide matrix has missing/extra cases")
        require(len(populations)==2*len(native) and set(zip(populations.case_id,populations.population))=={(c,p) for c in native for p in ('BACKBONE','KEY_LINKS')},'Population ledger has extra/missing cases')
        _verify_long(source,root,ids,backbone,keymask,common,native,references,cases,statistics,check)
        deltas=_verify_i95(source,root,ids,backbone,common,native,cases,proof,references,check)
        # Native graph/source masks are the exact ones bound by the prior OD audit.
        reader.unchanged();check();status="COMPLETE_WITH_SCOPE_LIMITATIONS"
    except (TimeoutError,MemoryError) as exc:
        status="PARTIAL_BUDGET_OR_RESOURCE";error=dict(type=type(exc).__name__,message=str(exc))
    except Exception as exc:
        status="FAIL";error=dict(type=type(exc).__name__,message=str(exc))
    finally:
        if error and current and ledger[current]["disposition"]=="NOT_CHECKED":ledger[current]["disposition"]="FAILED_OR_INCOMPLETE"
        pd.DataFrame(ledger.values()).to_csv(output/"CASE_LEDGER.csv",index=False)
        write_json(output/"NATIVE_METRICS.json",statistics);write_json(output/"PAIRED_CHANGES.json",deltas)
        write_json(output/"SOURCE_BINDINGS.json",list(reader.bindings.values()))
        summary=dict(status=status,planned_rows=24,case_dispositions={k:v["disposition"] for k,v in ledger.items()},
            admitted_native_cases=len(native),full_B_count=B_COUNT,frozen_key_count=KEY_COUNT,error=error,limitations=LIMITATIONS,
            frozen_training_coverage=training_coverage,
            started_epoch=started,finished_epoch=time.time(),audit_deadline_epoch=end,
            recorded_wall_overruns={k:r["complete_wall_overrun_seconds"] for k,r in ledger.items() if r["complete_wall_overrun_seconds"] and r["complete_wall_overrun_seconds"]>0},
            cap_interpretation="Declared caps verified; any measured complete-wall overrun remains an explicit A02 qualification, never silently subtracted.",
            new_assignment_or_model_inference=False,resource=dict(peak_sampled_RSS_bytes=check.peak_rss,minimum_sampled_available_bytes=check.minimum_available))
        write_json(output/"RESULT.json",summary);write_json(output/"STATUS.json",dict(status=status,requirements=["A01","A02","A03","A04","A05","A06"]))
        write_json(output/'VALIDATION.json',dict(status=status,planned_rows=24,completed_native_comparisons=len(native),
            full_population=B_COUNT,frozen_key_population=KEY_COUNT,source_bindings=len(reader.bindings),
            current_source_coverage='COMPLETED_PREFIX_ONLY' if error else 'ALL_CONSUMED_INPUTS',error=error))
        (output/"REPORT.md").write_text("# Independent native assignment metrics audit\n\n"+json.dumps(summary,indent=2)+"\n",encoding="utf-8")
        try:_seal(output,Budget(ctx,end,resources))
        except Exception as exc:
            summary.update(status="PARTIAL_OR_FAILED_HANDOFF",handoff_verified=False,handoff_error=str(exc))
            write_json(output/"SEAL_INVALIDATED.json",dict(verified=False,error=str(exc),observed_epoch=time.time()))
            write_json(output/"RESULT.json",summary);write_json(output/"STATUS.json",dict(status=summary["status"]))
    return summary


def _verify_long(source,root,ids,backbone,keymask,common,native,references,cases,statistics,check):
    byid={c["case_id"]:c for c in cases};seen=set()
    for frame in source.csv(root/"analysis/backbone_links.csv",chunksize=B_COUNT,dtype={"link_id":str}):
        check();require(len(frame)==B_COUNT and frame.case_id.nunique()==1,"Long native CSV has partial/mixed case block")
        key=frame.case_id.iloc[0];require(key in native and key not in seen,"Extra/duplicate long case block");seen.add(key)
        case=byid[key];reference=references[case["scenario"]][0];pred=native[key]
        require(np.array_equal(frame.link_id.to_numpy(),ids) and np.array_equal(frame.native_index,np.arange(B_COUNT))
                and np.array_equal(booleans(frame.is_key),keymask),"Long native identity/key order differs")
        near(frame.reference_flow.to_numpy(),reference,"Long reference vector",rtol=0,atol=0)
        near(frame.predicted_flow.to_numpy(),pred,"Long prediction vector",rtol=0,atol=0)
        near(frame[GEO].to_numpy(),backbone[GEO].to_numpy(),"Long native geometry",rtol=0,atol=0)
        _,per=independent_metrics(reference,pred,keymask,common["base"][:B_COUNT],common["k4"][:B_COUNT])
        for field,value in per.items():near(frame[field].to_numpy(),value,"Long per-link "+field)
        closed=common["closed_mask"][:B_COUNT] if case["scenario"]=="i95" else np.zeros(B_COUNT,bool)
        require(np.array_equal(booleans(frame.is_closed),closed),"Long closure mask differs")
        verify_comparison_metadata(frame,case,ids,closed)
    require(seen==set(native),"Missing long native case block")


def _verify_i95(source,root,ids,backbone,common,native,cases,proof,references,check):
    oo,cc=references["open"][1],references["i95"][1]
    masks={"COMMON_SERVED":oo&cc,"LOST_CLOSED_VS_OPEN":oo&~cc,"GAINED_CLOSED_VS_OPEN":~oo&cc,"NEITHER_SERVED":~oo&~cc}
    known={r["population"]:r for r in proof["populations"]}
    for name,mask in masks.items():
        require(int(np.count_nonzero(mask))==known[name]["aggregate_OD_rows"]
                and array_hash({"membership":mask})==known[name]["membership_mask_identity"],"Cross-scenario membership proof differs")
    require(not masks["GAINED_CLOSED_VS_OPEN"].any() and proof["closed_served_subset_of_open"] is True,"Closed served is not a subset")
    coverage=source.json(root/"analysis/I95_OD_COVERAGE.json")
    for key,value in dict(open_served_OD_rows=int(oo.sum()),closed_served_OD_rows=int(cc.sum()),
        lost_OD_rows=known["LOST_CLOSED_VS_OPEN"]["aggregate_OD_rows"],gained_OD_rows=0).items():require(coverage[key]==value,"OD count mismatch")
    near(coverage["lost_OD_demand"],known["LOST_CLOSED_VS_OPEN"]["demand"],"Original lost demand",atol=1e-7)
    near(coverage["gained_OD_demand"],0.,"Original gained demand",atol=0)
    require(coverage["demand_units"]=="ANNUAL_KTON" and coverage["demand_renormalized"] is False,"OD units/renormalization changed")
    require(coverage['open_closed_OD_membership_equal'] is bool(np.array_equal(oo,cc)),'Cross-scenario equality label differs')
    pairs={}
    for case in cases:
        if case["case_id"] not in native:continue
        method=("P/REFERENCE" if case["assignment_role"]=="REFERENCE_SELF" else "P/"+case["solver"] if case["network"]=="P"
                else case["suite_id"]+"/"+case["family"])
        pairs.setdefault(method,{})[case["scenario"]]=native[case["case_id"]]
    pairs={k:v for k,v in pairs.items() if set(v)==set(SCENARIOS)};seen=set();result={}
    for frame in source.csv(root/"analysis/I95.csv",chunksize=B_COUNT,dtype={"link_id":str}):
        check();require(len(frame)==B_COUNT and frame.method_id.nunique()==1,"Mixed/partial I95 method block")
        method=frame.method_id.iloc[0];require(method in pairs and method not in seen,"Extra/duplicate I95 pair");seen.add(method)
        pair=pairs[method];require(np.array_equal(frame.link_id.to_numpy(),ids),"I95 native identities differ")
        near(frame[GEO].to_numpy(),backbone[GEO].to_numpy(),'I95 geography',rtol=0,atol=0)
        for scenario,column in (("open","open_flow"),("i95","closed_flow")):near(frame[column].to_numpy(),pair[scenario],column,rtol=0,atol=0)
        near(frame.delta.to_numpy(),pair["i95"]-pair["open"],"closed-minus-open delta",rtol=0,atol=0)
        require(np.array_equal(booleans(frame.is_closed),common["closed_mask"][:B_COUNT]),"I95 mask differs")
        for field in ('open_closed_OD_membership_equal','lost_OD_rows','gained_OD_rows','lost_OD_demand','gained_OD_demand'):
            constant(frame,field,coverage[field])
        result[method]=delta_statistics(pair["open"],pair["i95"])
    require(seen==set(pairs),"Missing paired method")
    result["ORIGINAL_OD_MEMBERSHIP_PROOF"]=dict(populations=proof["populations"],represented_closure=proof["represented_closure"],
        demand_units="ANNUAL_KTON",demand_renormalized=False,source="Authenticated prior exact audit; current masks independently matched")
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",required=True);parser.add_argument("--admission",required=True)
    parser.add_argument("--admission-sha256",required=True);args=parser.parse_args()
    result=run(args.config,admission_path=args.admission,admission_sha256=args.admission_sha256)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result["status"]=="COMPLETE_WITH_SCOPE_LIMITATIONS" else 2)
