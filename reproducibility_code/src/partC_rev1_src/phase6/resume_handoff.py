"""One compact cumulative handoff with honest scope and fresh extracted verification."""
from pathlib import Path
from collections import deque
import json
import os
import shutil
import subprocess
import time
import zipfile
import xml.etree.ElementTree as ET
import pandas as pd
import psutil

from .common import read_json,write_json,write_frame,sha,git,tree_bytes
from .handoff import copy,text_write,dependency_snapshot,example
from .artifacts import inventory,verify_witness,describe
from .registry import region_dir
from .resume_policy import ledger,READY,QUARANTINED
from .resume_review_evidence import optional_docket,compact_external_evidence,reopen_rejected,privacy_and_format_scan


def campaign_status(scope,reopen_pass,stop):
    if not scope["strict_limit_pass"]:return "PHASE6_ACTIVE_CAMPAIGN_BLOCKED_FAILURE_THRESHOLD"
    resource=stop.get("error_type")=="ResourceStop" or stop.get("reason") in {"ORIGINAL_WHOLE_CAMPAIGN_DEADLINE","WHOLE_TASK_DEADLINE","RESOURCE_LIMIT","DISK_RESERVE"}
    if (stop and not resource) or not reopen_pass:return "PHASE6_ACTIVE_CAMPAIGN_BLOCKED_SHARED_INTEGRITY"
    if not scope["healthy_scope_complete"]:return "PHASE6_ACTIVE_CAMPAIGN_PARTIAL_TIME_OR_RESOURCE_LIMIT"
    if scope["F"]:return "PHASE6_ACTIVE_CAMPAIGN_COMPLETE_WITH_QUARANTINED_REGIONS_WAITING_EXTERNAL_REVIEW"
    return "PHASE6_ACTIVE_CAMPAIGN_COMPLETE_WAITING_EXTERNAL_REVIEW"


def failure_docket(c,regions):
    root=Path(c["output_root"]);book=ledger(str(root));out=root/"failures";out.mkdir(exist_ok=True)
    active=regions.loc[regions.m_active.gt(0)];failed=active.loc[active.region_id.isin(book.failed)]
    shares=dict(active_regions=1566,failed_active_regions=len(failed),active_pair_denominator=80316,
        failed_active_pairs=int(failed.m_active.sum()),failed_active_pair_share=float(failed.m_active.sum()/80316),
        LOW_traversal_exposure_denominator=float(active.low_exposure.sum()),
        failed_LOW_traversal_exposure=float(failed.low_exposure.sum()),
        failed_LOW_traversal_exposure_share=float(failed.low_exposure.sum()/active.low_exposure.sum()),
        exposure_is_not_deduplicated_national_OD=True,topology_edges_removed=0)
    summaries=[];repair=[]
    for row in regions.itertuples():
        rid=row.region_id;rd=region_dir(c,rid)
        if not row.m_active:continue
        state=book.states[rid];plan=pd.read_parquet(rd/"baseline_plan.parquet")
        existing={p.stem for p in (rd/"labels/receipts").glob("*.json")}
        incident=read_json(state["incident_path"]) if state.get("incident_path") else {}
        if rid in book.failed:
            summaries.append(dict(region_id=rid,m_active=int(row.m_active),physical_nodes=int(row.physical_nodes),
                physical_arcs=int(row.physical_arcs),LOW_traversal_exposure=float(row.low_exposure),
                code=state["reason"],incident_path=state.get("incident_path"),incident_sha256=sha(state["incident_path"]),
                stage=incident.get("stage"),scope=incident.get("scope"),evidence_json=json.dumps(incident.get("evidence",{}),sort_keys=True),
                model_hash=row.model_hash,support_hash=row.support_hash,missing_unique=len(set(plan.input_hash)-existing)))
        for ih in sorted(set(plan.input_hash)-existing):
            slots=plan.loc[plan.input_hash.eq(ih)]
            repair.append(dict(region_id=rid,kind="MISSING_MANDATORY_INPUT",input_hash=ih,model_hash=row.model_hash,
                support_hash=row.support_hash,state=state["state"],role_indices=[int(x) for x in slots.role_index],
                family=slots.family.tolist(),seed=slots.seed.astype(str).tolist(),input_index=int(slots.iloc[0].input_index),
                input_path=str(rd/"baseline_inputs.npz"),plan_path=str(rd/"baseline_plan.parquet"),
                graph_path=str(rd/"local_graph.npz"),catalog_index_path=str(rd/"labels/catalog_index.json"),
                input_file_sha256=sha(rd/"baseline_inputs.npz"),baseline_plan_sha256=sha(rd/"baseline_plan.parquet"),
                replay_command=incident.get("retry_command"),authorization="Original budget only; after seal requires new hash-bound review"))
        if state["state"]!=READY:
            repair.append(dict(region_id=rid,kind="INCOMPLETE_OR_FAILED_REGION_AUDITS",state=state["state"],
                incident_path=state.get("incident_path"),native_region_path=str(rd),model_hash=row.model_hash,support_hash=row.support_hash,
                baseline_audit_record=str(rd/"BASELINE_AUDIT_PASS.json"),fixed_fine_definition=str(rd/"fixed_fine/definition.npz"),
                replay_command=incident.get("retry_command"),last_checkpoint=git(c["repository_root"],"rev-parse","HEAD")))
    nonlabels,optional_repairs,optional_examples=optional_docket(c,regions,book.states)
    repair.extend(optional_repairs)
    text_write(out/"optional_nonlabel_slots.jsonl","".join(json.dumps(r,sort_keys=True)+"\n" for r in nonlabels))
    write_json(out/"OPTIONAL_CANDIDATE_REVIEW.json",dict(nonlabel_slots=len(nonlabels),
        rejected_candidates=len(optional_examples),deferred_inputs=sum(r["kind"]=="OPTIONAL_DEFERRED_DEADLINE" for r in nonlabels),
        no_unique_inputs=sum(r["kind"]=="OPTIONAL_NO_UNIQUE_INPUT" for r in nonlabels),
        affected_region_ids=sorted({r["region_id"] for r in nonlabels}),regional_failures_added=0,
        rejected_candidate_examples=optional_examples,optimizer_solves=0))
    shares.update(optional_rejected_candidates=len(optional_examples),optional_nonlabel_slots=len(nonlabels))
    frame=pd.DataFrame(summaries,columns=["region_id","m_active","physical_nodes","physical_arcs","LOW_traversal_exposure",
        "code","incident_path","incident_sha256","stage","scope","evidence_json","model_hash","support_hash","missing_unique"])
    frame.to_csv(out/"region_failure_summary.csv",index=False);write_frame(out/"region_failure_summary.parquet",frame)
    text_write(out/"repair_manifest.jsonl","".join(json.dumps(r,sort_keys=True)+"\n" for r in repair))
    if not (out/"region_failure_ledger.jsonl").exists():text_write(out/"region_failure_ledger.jsonl","")
    write_json(out/"failure_population_shares.json",shares)
    bindings=pd.read_parquet(root/"validation/source_manifest.parquet")
    nodes_binding=bindings.loc[bindings.absolute_path.str.endswith("acnhn_nodes.parquet")]
    if len(nodes_binding)!=1:raise ValueError("Regional geographic source identity is ambiguous")
    binding=nodes_binding.iloc[0]
    if sha(binding.absolute_path)!=binding.sha256:raise ValueError("Geographic source changed")
    nodes=pd.read_parquet(binding.absolute_path,columns=["node_id","longitude","latitude"])
    terminals=pd.read_parquet(root/"registry/coarse_terminals.parquet",columns=["region_id","terminal_id"])
    joined=terminals.merge(nodes,left_on="terminal_id",right_on="node_id",how="left",validate="many_to_one")
    geo=joined.groupby("region_id").agg(terminal_longitude_mean=("longitude","mean"),terminal_latitude_mean=("latitude","mean"),
        terminal_longitude_min=("longitude","min"),terminal_longitude_max=("longitude","max"),
        terminal_latitude_min=("latitude","min"),terminal_latitude_max=("latitude","max"),located_terminals=("longitude","count")).reset_index()
    geo=regions[["region_id","m_active","physical_nodes","physical_arcs","low_exposure"]].merge(geo,on="region_id",how="left",validate="one_to_one")
    geo["state"]=[book.states[rid]["state"] if rid in book.active else "NO_ACTIVE_LOW_PAIRS" for rid in geo.region_id]
    geo["size_bin"]=pd.cut(geo.m_active,[-1,0,8,32,128,512,float("inf")],labels=["inactive","1-8","9-32","33-128","129-512","513+"]).astype(str)
    geo.to_csv(out/"region_geographic_and_size_distribution.csv",index=False)
    geo.groupby(["size_bin","state"],observed=True).agg(regions=("region_id","size"),active_pairs=("m_active","sum"),LOW_traversal_exposure=("low_exposure","sum")).reset_index().to_csv(out/"failure_size_distribution.csv",index=False)
    write_json(out/"GEOGRAPHY_PROVENANCE.json",dict(source_path=binding.absolute_path,sha256=binding.sha256,
        definition="Mean and bounds of accepted regional terminal representative coordinates; not a physical-region polygon or geometric centroid",
        unmatched_terminals=int(joined.longitude.isna().sum()),missing_geometry_is_not_invented=True))
    text_write(out/"FAILURE_REVIEW.md",f"# Cumulative regional failure docket\n\n{len(failed)} of 1,566 active regions are quarantined; the strict test is 20*F < 1566. "
        "The nine source-path warnings and unprocessed regions are excluded from F. Native logical links remain unchanged.\n\n"
        "region_failure_summary.csv links each exception, stack, native input/model/support identity and scoped incident. "
        "Per-incident evidence carries target/achieved gap and candidate paths where applicable; no unmeasured diagnosis is asserted. "
        "The external output index binds all attempt/progress/resource logs, payloads and catalog segments. "
        "repair_manifest.jsonl enumerates missing mandatory inputs, audit roles and rejected/deferred optional inputs. "
        "optional_nonlabel_slots.jsonl also preserves degenerate draws. Saved rejected candidates are freshly repriced without optimization; "
        "their actual gap, seed, family, batch input index and graph/catalog/source hashes are included. "
        "No exact optional-input retry CLI is authorized or implemented: the docket explicitly records this missing repair entrypoint, "
        "rather than inventing a command that would restart or redraw science. "
        "A quarantined-region replay requires a proved fix, tests, committed code, one explicit fix record and complete local revalidation. "
        "Pending work needs approved continuation after this final seal; no individual failure is silently retried.\n\n"
        f"Pair and LOW traversal exposure shares: {json.dumps(shares,sort_keys=True)}\n")
    return shares


def population(c):
    root=Path(c["output_root"]);regions=pd.read_parquet(root/"registry/regions.parquet");book=ledger(str(root));rows=[];quality={}
    for rid in regions.region_id:
        rd=region_dir(c,rid);labels=[]
        for p in (rd/"labels/receipts").glob("*.json"):
            receipt=read_json(p);meta=receipt["metadata"];s=receipt["scalars"]
            labels.append(dict(region_id=rid,input_hash=receipt["input_hash"],quality=s["quality"],relative_gap=s["relative_gap"],
                baseline=meta.get("baseline",True),split=meta.get("split"),family=meta.get("family"),seed=meta.get("seed"),
                round_index=meta.get("round_index",0),origin_run_id=meta.get("run_id"),model_hash=receipt["model_hash"],
                support_hash=receipt["support_hash"],payload_sha256=receipt["payload_sha256"],payload_bytes=receipt["payload_bytes"],
                target_gap=meta.get("target_gap"),finite_tolerance_fallback=meta.get("finite_tolerance_fallback"),
                eligible_AB=rid in book.states and book.states[rid]["state"]==READY and s["quality"] in {"A","B"}))
        if labels:
            f=pd.DataFrame(labels);write_frame(root/"reports/label_population"/(rid.removeprefix("partC_p2_region_")+".parquet"),f)
            rows.extend(labels)
    f=pd.DataFrame(rows)
    if not len(f):raise ValueError("No-op cumulative dataset is failure")
    for key in ["quality","split","origin_run_id"]:quality[key]=f.groupby(key,dropna=False).size().to_dict()
    quality={k:{str(a):int(b) for a,b in v.items()} for k,v in quality.items()}
    result=dict(total_published_coarse=len(f),baseline_unique=int(f.baseline.sum()),extra_unique=int((~f.baseline).sum()),
        default_eligible_AB=int(f.eligible_AB.sum()),grades_splits_origins=quality,actual_fallbacks=int(f.finite_tolerance_fallback.notna().sum()))
    write_json(root/"reports/POPULATION.json",result);return regions,result


def finish(c,config_path):
    root=Path(c["output_root"]);repo=Path(c["repository_root"]);work=Path(c["work_records_root"])
    sealed=root/"manifests/RESUME_HANDOFF_SEAL_RESULT.json"
    if sealed.exists():return read_json(sealed)
    if (root/"control/writer.lock").exists():raise ValueError("A scientific stage is still writing")
    if git(repo,"status","--porcelain","--","partC_rev1_src/phase6","tests/test_phase6*.py"):
        raise ValueError("Commit tested implementation before finalization")
    before=git(repo,"rev-parse","HEAD");root.joinpath("reports").mkdir(exist_ok=True)
    from .resume_campaign import region_tables
    tables=region_tables(c,"FINAL");regions,pop=population(c);shares=failure_docket(c,regions)
    gate=read_json(root/"validation/BASELINE_SCOPE_READINESS.json") if (root/"validation/BASELINE_SCOPE_READINESS.json").exists() else {}
    reopen=read_json(root/"validation/R5_REOPEN_RESULT.json") if (root/"validation/R5_REOPEN_RESULT.json").exists() else {}
    stop=read_json(root/"control/STOP_SCIENCE.json") if (root/"control/STOP_SCIENCE.json").exists() else {}
    scope=ledger(str(root)).summary(global_integrity=bool(gate.get("global_integrity_pass") and reopen.get("status")=="PASS" and not stop))
    extra=read_json(root/"samples/ENRICHMENT_RESULT.json") if (root/"samples/ENRICHMENT_RESULT.json").exists() else {}
    state=campaign_status(scope,reopen.get("status")=="PASS",stop)
    result=dict(campaign_status=state,status=state,lifecycle_status="WAITING_FOR_EXTERNAL_REVIEW",phase_id=c["phase_id"],step_id=c["step_id"],run_id=c["run_id"],
        **scope,with_quarantined_regions=bool(scope["F"]),enrichment_performed=pop["extra_unique"]>0,population=pop,enrichment=extra,
        phase7_loader_ready_for_healthy_subset=scope["healthy_scope_complete"],phase7_training_authorized=False,
        national_physical_fidelity_validated=False,upstream_physical_model_readiness="MODEL_REVISION_REQUIRED",
        model_class=c["model_class"],validation_policy_id=c["validation_policy_id"],active_pairs=80316,static_links=143603,
        inactive_regions=281,all_regions=1847,retained_nodes=72724,regional_physical_arcs=327757,
        failure_population_shares=shares,region_tables=tables,source_reopen=reopen,controlled_stop=stop,
        authority_sha256=c["authority_zip_sha256"],launch_sha256=c["launch_sha256"],source_handoff_sha256=sha(c["source_handoff_path"]),
        config_sha256=sha(config_path),execution_snapshot_head=before,
        scientific_execution_head=read_json(root/"validation/review_resolution.json")["execution_head"],
        finalization_implementation_head=before,original_start_unix=c["task_started_unix"],
        hard_deadline_unix=c["hard_deadline_unix"],timer_reset=False,elapsed_seconds_before_seal=time.time()-c["task_started_unix"],
        engineering_epochs_used=read_json(root/"control/engineering_epochs.json")["used"],
        prohibited_actions_performed=[],ICNN_constructions=0,ICNN_training_runs=0,national_assignments=0,
        next_step_authorized=False,next_step_recommendation="External review of this cumulative handoff; address the failure docket under a new hash-bound decision before further science.")
    summary=(f"# Phase 6 active s03 cumulative result\n\n{state}. WAITING_FOR_EXTERNAL_REVIEW.\n\n"
        f"Healthy baseline: {scope['healthy_active_count']}; quarantined: {scope['F']}; pending: {scope['pending_active_count']}; inactive skips: 281. "
        f"All-region dataset complete: {scope['all_regions_dataset_complete']}; healthy scope complete: {scope['healthy_scope_complete']}.\n\n"
        f"Published coarse labels: {pop['total_published_coarse']:,}; baseline unique: {pop['baseline_unique']:,}; fresh-seed extras: {pop['extra_unique']:,}. "
        f"Population split/grade/origin counts and partial round quotas are included. {pop['actual_fallbacks']} labels explicitly record a valid finite-tolerance fallback.\n\n"
        "The user-approved LOW-active model, all 80,316 active links, 143,603 static links, graph costs and centroid path rules remain unchanged. "
        "Only active coordinates exist; 281 zero-active regions skip local data and all ICNN construction. "
        "The same-model coarse/fixed interval check remains required. The mapped inequality is checked only with a proved admissible premise; "
        "nine source-path mismatches are inapplicable warnings, and unproved premises are explicitly nonblocking unknowns, never PASS.\n\n"
        f"Original clock: 2026-09-07 03:29:13 UTC through 2026-09-08 03:29:13 UTC. It was never reset; external-review wall time counts. "
        f"Preseal elapsed: {result['elapsed_seconds_before_seal']/3600:.3f} hours. Remaining science used measured finalization reserve.\n\n"
        "Every current artifact is hash-bound in the external inventory; preserved parent files and catalog prefixes are independently checked. "
        "R5 receipts state exactly which labels were freshly recertified; quarantined partial data are retained for explicit audit, excluded from default eligibility. "
        "No positive-demand label was truncated into a reduced-model label. No B restoration, fine-R expansion, inactive fallback, model training, other-suite generation or national assignment was performed.\n\n"
        "Review failures/FAILURE_REVIEW.md, all-region status and benchmark tables, original/new label populations, native witnesses, tests, source invariance and the companion. "
        "National physical fidelity remains unvalidated and upstream readiness remains MODEL_REVISION_REQUIRED. No further scientific step is self-approved.\n")
    write_json(root/"reports/FINAL_RESULT.json",result);text_write(root/"reports/FINAL_REPORT.md",summary)
    write_json(work/"FINAL_RESULT.json",result);text_write(work/"FINAL_REPORT.md",summary)
    resource_path=root/"control/resource_history.jsonl"
    resources=pd.DataFrame([json.loads(x) for x in resource_path.read_text().splitlines()]) if resource_path.exists() else pd.DataFrame()
    resources.to_csv(root/"reports/RESOURCE_USAGE.csv",index=False)
    for name in ["00_READ_FIRST.md","PROJECT_STATUS.md"]:
        p=repo/"partC_doc_icnn"/name
        p.write_text(f"# Phase 6 active resume: cumulative final result\n\n{state}. See phase_6_work/s03_active_resume_0907/FINAL_REPORT.md. External review only; no next step authorized.\n\n---\n\n"+p.read_text(encoding="utf-8"),encoding="utf-8")
    doc_paths=[str(work/"FINAL_RESULT.json"),str(work/"FINAL_REPORT.md"),"partC_doc_icnn/00_READ_FIRST.md","partC_doc_icnn/PROJECT_STATUS.md"]
    git(repo,"add","--",*doc_paths);git(repo,"commit","-m","phase6: record cumulative active resume result before sealing")
    head=git(repo,"rev-parse","HEAD");hid="active_resume_0907_"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"_final_001"
    parent=repo/"partC_doc_icnn/handoffs/phase_06/step_s03_active_resume_0907";source=parent/hid
    archives=repo/"artifacts_icnn/handoff_bundles/phase_06";archives.mkdir(parents=True,exist_ok=True)
    archive=archives/("phase06_steps03_active_resume_0907_"+hid+"_SYNC.zip")
    if source.exists() or archive.exists():raise FileExistsError("A sealed handoff cannot be overwritten")
    source.mkdir(parents=True)
    write_json(root/"control/status.json",result,new=False)
    external=inventory(root,workers=4,prefix="resume_output",exclude=["control/driver_events.jsonl"])
    for name in ["governing_docs","configs","code_snapshot","code_diff","summary_data","figures","geospatial_summary","log_excerpts","external_artifact_index"]:(source/name).mkdir()
    for p in Path(c["authority_root"]).rglob("*"):
        if p.is_file() and p.suffix not in {".zip",".pyc"} and "__pycache__" not in p.parts:copy(p,source/"governing_docs/authority"/p.relative_to(c["authority_root"]))
    for name in ["00_READ_FIRST.md","PROJECT_STATUS.md","12_DECISION_LOG.md","20_HANDOFF_SYNC_PROTOCOL.md","21_REVIEW_APPROVAL_GATES.md","23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md","24_CODE_AND_DATA_SNAPSHOT_POLICY.md"]:copy(repo/"partC_doc_icnn"/name,source/"governing_docs"/name)
    copy(repo/"AGENTS.md",source/"governing_docs/AGENTS.md");copy(config_path,source/"configs/effective_execution.json")
    snapshot=dependency_snapshot(repo)+list((repo/"tests").glob("test_phase6*.py"))
    for p in snapshot:copy(p,source/"code_snapshot"/p.relative_to(repo))
    patch=subprocess.check_output(["git","-C",str(repo),"-c","core.longpaths=true","diff",c["required_ancestor"],head,"--","partC_rev1_src/phase6","tests/test_phase6*.py"])
    (source/"code_diff/resume.patch").write_bytes(patch)
    text_write(source/"code_diff/commits.txt",git(repo,"log","--oneline",c["required_ancestor"]+".."+head)+"\n")
    external_shards,compact=compact_external_evidence(root,source/"external_artifact_index")
    write_json(source/"external_artifact_index/COMPACT_REVIEW_POLICY.json",compact)
    copy(root/"manifests/resume_output_INDEX.json",source/"external_artifact_index/output_INDEX.json")
    # Full multi-round row populations stay externally bound, with compact samples and complete region summaries.
    for folder in ["reports","failures","validation","dataset"]:
        for p in (root/folder).rglob("*"):
            if not p.is_file() or p.stat().st_size>25*2**20:continue
            if p.resolve() in external_shards:continue
            if p.suffix in {".parquet",".json",".csv",".md",".xml",".jsonl"}:
                copy(p,source/"summary_data"/p.relative_to(root))
    for name in ["registry/REGISTRY_COMPLETE.json","registry/regions.parquet","registry/high_projection_loss.parquet","samples/BASELINE_PLAN_FROZEN.json","samples/baseline_plan.parquet","samples/enrichment_quotas.parquet","samples/ENRICHMENT_RESULT.json"]:
        if (root/name).exists():copy(root/name,source/"summary_data"/name)
    for p in (root/"control").glob("*"):
        if p.suffix in {".json",".jsonl",".log"}:
            if p.stat().st_size<2**20:copy(p,source/"log_excerpts/control"/p.name)
            else:
                with p.open(encoding="utf-8",errors="replace") as stream:tail=deque(stream,maxlen=60)
                text_write(source/"log_excerpts/control"/(p.name+".excerpt.txt"),"Full source is hash-bound externally. Last 60 lines:\n"+"".join(tail))
    for p in work.glob("*"):
        if p.is_file() and p.stat().st_size<4*2**20:copy(p,source/"summary_data/work_records"/p.name)
    valid=regions.loc[regions.region_id.isin(scope["healthy_ids"])].sort_values(["m_active","region_id"])
    examples=[regions.loc[regions.m_active.eq(0),"region_id"].iloc[0],"partC_p2_region_9567843a01be0d75fd9b"]
    if len(valid):examples += [valid.iloc[0].region_id,valid.iloc[-1].region_id]
    for rid in dict.fromkeys(examples):example(c,rid,source/"summary_data/native_examples"/rid.removeprefix("partC_p2_region_"))
    for rid in scope["quarantined_ids"][:3]:
        incident=read_json(ledger(str(root)).states[rid]["incident_path"]);candidate=incident.get("evidence",{}).get("candidate_npz")
        if candidate and Path(candidate).exists() and Path(candidate).stat().st_size<4*2**20:copy(candidate,source/"summary_data/failing_native_candidates"/rid/(Path(candidate).name))
    rejected=read_json(root/"failures/OPTIONAL_CANDIDATE_REVIEW.json")["rejected_candidate_examples"]
    included_rejected=[]
    for rec in rejected:
        if len(included_rejected)>=3:break
        native=[Path(rec[k]["absolute_path"]) for k in ["candidate_json","candidate_npz"]]
        if any(p.stat().st_size>25*2**20 for p in native):continue
        target=source/"summary_data/rejected_optional_examples"/rec["input_hash"]
        for p in native:copy(p,target/p.name)
        copy(rec["native_sources"]["graph"]["absolute_path"],target/"local_graph.npz")
        write_json(target/"review_record.json",rec);included_rejected.append(rec["input_hash"])
    write_json(source/"summary_data/rejected_optional_examples_SELECTION.json",dict(
        included_input_hashes=included_rejected,total_rejected=len(rejected),
        selection="First three durable batch-ordered candidates with each member under 25 MiB; every other candidate remains hash-bound in the complete docket."))
    write_json(source/"OUTPUT_NAME_MAP.json",dict(
        stage_events="log_excerpts/control/driver_events.jsonl and stage_history.jsonl",
        region_status="summary_data/validation/region_status_FINAL.parquet and .csv; exact latest paths also in FINAL_RESULT.region_tables",
        benchmark_comparison_by_region="summary_data/validation/benchmark_comparison_by_region_FINAL.parquet and .csv; R2/R3/R4/R5 retained",
        all_input_seed_quality_alias_ledgers="external_artifact_index/FULL_LOCAL_SHARD_BINDINGS.json and output_INDEX.json bind complete local population, plan, batch and native receipt files",
        complete_region_population="external_artifact_index/complete_region_label_population_summary.parquet",
        failed_and_missing_input_roles="summary_data/failures/repair_manifest.jsonl",
        optional_nonlabels="summary_data/failures/optional_nonlabel_slots.jsonl"))
    text_write(source/"figures/README.md","Region-level size, quality, exposure and failure tables are the authoritative comparison. No new scientific figure was required.\n")
    text_write(source/"geospatial_summary/README.md","No geometry was changed. summary_data/failures/region_geographic_and_size_distribution.csv reports every region's accepted terminal representative coordinates and bounds, with explicit source hashes and missing-coordinate counts. These are terminal summaries, not invented region footprints. Size and failure exposure distributions accompany the complete docket.\n")
    core={"00_HANDOFF_README.md":"Read 01_STEP_SUMMARY.md, 02_HANDOFF_STATUS.json and summary_data/failures/FAILURE_REVIEW.md. Upload this single cumulative ZIP with its SHA and companion.\n",
        "01_STEP_SUMMARY.md":summary,"03_DECISIONS_AND_OPEN_QUESTIONS.md":"Current user authority resolves only the conditional mapped comparison. Model and path rules remain frozen. Review the complete regional failure docket; no new scientific decision is inferred.\n",
        "04_VALIDATION_RESULTS.md":"R1 JUnit, saved counterexample, R2 preservation, R3 remaining work, R4 local/global diagnostics and R5 fresh-process receipts are in summary_data/validation. Absence or failure is explicit in terminal status; tests do not replace science.\n",
        "09_NEXT_STEP_RECOMMENDATION.md":result["next_step_recommendation"]+"\n","10_UPLOAD_THIS_FILE.txt":str(archive)+"\n",
        "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md":"Cumulative continuation of reviewed handoff aa0dd1b73a5268991d1169e411e387f10ec18cba579e4fe3bc1118bd7b38e11a. That ZIP and parent root remain immutable. Current comparison policy, regional quarantine, completed work and source preservation are separately recorded.\n"}
    for name,value in core.items():text_write(source/name,value)
    write_json(source/"02_HANDOFF_STATUS.json",{**result,"snapshot_head":head,"handoff_id":hid})
    copy(root/"reports/RESOURCE_USAGE.csv",source/"05_RESOURCE_USAGE.csv")
    pd.DataFrame([dict(relative_path=p.relative_to(repo).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size) for p in snapshot]).to_csv(source/"07_CODE_CHANGE_INDEX.csv",index=False)
    text_write(source/"08_COMMAND_LOG.md",f"# Explicit execution\n\n`{c['python_executable']} -X utf8 -m partC_rev1_src.phase6.run --config {config_path} --stage resume`\n\n"
        "Exact stage commands, PIDs, code/config hashes and timestamps are in log_excerpts/control/driver_events.jsonl and stage_history.jsonl. Native threads are one per worker, two scientific workers. R1 tests use all tests/test_phase6*.py and both applicable authority reference suites.\n")
    write_json(source/"log_excerpts/privacy_and_format_scan.json",privacy_and_format_scan(source))
    records=[]
    for p in sorted(source.rglob("*")):
        if p.is_file():
            if p.stat().st_size>25*2**20:raise ValueError("Oversize handoff member: "+str(p))
            describe(p);records.append(dict(relative_path=p.relative_to(source).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size))
    pd.DataFrame(records).to_csv(source/"06_ARTIFACT_MANIFEST.csv",index=False)
    checks=records+[dict(relative_path="06_ARTIFACT_MANIFEST.csv",sha256=sha(source/"06_ARTIFACT_MANIFEST.csv"))]
    text_write(source/"CHECKSUMS.sha256","".join(r["sha256"]+"  "+r["relative_path"]+"\n" for r in checks))
    temp=archive.with_suffix(".zip.building")
    with zipfile.ZipFile(temp,"x",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(source.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(source).as_posix())
    if temp.stat().st_size>200_000_000:raise ValueError("Compact handoff exceeds 200 MB hard limit")
    extraction=root/"manifests"/(hid+"_fresh_extraction")
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None:raise ValueError("CRC failure")
        for name in z.namelist():
            if not (extraction/name).resolve().is_relative_to(extraction.resolve()):raise ValueError("Unsafe member path")
        z.extractall(extraction)
    for r in checks:
        if sha(extraction/r["relative_path"])!=r["sha256"]:raise ValueError("Extracted checksum mismatch")
    witnesses=[dict(region_id=p.name,**verify_witness(p)) for p in (extraction/"summary_data/native_examples").iterdir() if (p/"label_payload.npz").exists()]
    rejected_witnesses=[]
    for ih in included_rejected:
        target=extraction/"summary_data/rejected_optional_examples"/ih;rec=read_json(target/"review_record.json")
        rejected_witnesses.append(dict(input_hash=ih,**reopen_rejected(target/(ih+"_normal.json"),
            target/"local_graph.npz",rec["model_hash"],rec["support_hash"])))
    xml=root/"manifests"/(hid+"_snapshot_tests.xml");environment=dict(os.environ)
    environment.update(PYTHONPATH=str(extraction/"code_snapshot"),NUMBA_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_CACHE_DIR=str(root/"manifests/snapshot_numba_cache"))
    test=subprocess.run([c["python_executable"],"-X","utf8","-m","pytest","-q",str(extraction/"code_snapshot/tests"),str(extraction/"governing_docs/authority/tests"),"--junitxml="+str(xml)],cwd=extraction/"code_snapshot",env=environment,capture_output=True,text=True,encoding="utf-8")
    text_write(xml.with_suffix(".log"),test.stdout+test.stderr)
    if test.returncode:raise ValueError("Extracted snapshot tests failed: "+test.stdout+test.stderr)
    counts={k:sum(int(s.attrib.get(k,0)) for s in ET.parse(xml).getroot().iter("testsuite")) for k in ["tests","failures","errors","skipped"]}
    if time.time()>=c["hard_deadline_unix"]:raise TimeoutError("Original hard deadline reached during seal; preserve unsealed evidence")
    digest=sha(temp);size=temp.stat().st_size;temp.rename(archive)
    publication=dict(zip_path=str(archive),zip_sha256=digest,zip_bytes=size,snapshot_head=head,campaign_status=state,status="WAITING_FOR_EXTERNAL_REVIEW")
    write_json(work/"HANDOFF_PUBLICATION.json",publication)
    git(repo,"add","--",str(work/"HANDOFF_PUBLICATION.json"));git(repo,"commit","-m","phase6: bind final cumulative handoff and await external review")
    ending_head=git(repo,"rev-parse","HEAD")
    companion=dict(**publication,ending_head=ending_head,branch=c["branch"],source_folder=str(source),authority_sha256=c["authority_zip_sha256"],config_sha256=sha(config_path),
        snapshot_head_explanation="Final Git HEAD adds only the publication binding after the reviewed source/result snapshot; ZIP bytes remain immutable.",
        scientific_execution_head=result["scientific_execution_head"],finalization_implementation_head=before,
        verified_checksums=len(checks),CRC_test="PASS",fresh_extraction="PASS",fresh_snapshot_tests=counts,compact_native_witnesses=witnesses,
        rejected_native_witnesses=rejected_witnesses,compact_review=compact,
        target_100_MB_met=size<=100_000_000,hard_200_MB_met=size<=200_000_000,
        snapshot_test_xml_sha256=sha(xml),whole_campaign_elapsed_seconds=time.time()-c["task_started_unix"],timer_reset=False,
        process_rss_at_seal_bytes=psutil.Process().memory_info().rss,measured_run_handoff_added_bytes=tree_bytes(root)+tree_bytes(source)+size,
        external_manifest=external,next_step_authorized=False,one_final_zip=True)
    write_json(archive.with_suffix(".companion.json"),companion);text_write(archive.with_suffix(".sha256"),digest+"  "+archive.name+"\n")
    text_write(parent/"LATEST_HANDOFF.txt",json.dumps(companion,indent=2)+"\n");write_json(sealed,companion)
    return companion
