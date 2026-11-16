"""One compact externally reviewable campaign handoff and exact sidecars."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd

from .campaign_data import PRIMARY
from .campaign_io import (binding_hash, code_binding, file_record, frame_write, git,
                         sha, verify_receipt, write_json)


def prepare_final(ctx):
    out=ctx.output
    coverage=json.loads((out/"validation/complete_coverage.json").read_text())
    limitations=json.loads((out/"validation/model_limitations.json").read_text())
    gates=pd.read_csv(out/"validation/acceptance_results_preseal.csv")
    if not gates.loc[gates.gate_id.ne("G19"),"status"].eq("PASS").all():raise ValueError("Preseal hard technical gates incomplete")
    # Preserve the C4 core byte-for-byte. Final evidence is appended only to a
    # new final SQLite copy, avoiding mutation of an upstream PASS artifact.
    database=out/"database/phase5_campaign.sqlite"
    if database.exists():raise FileExistsError("Final database is write-new only")
    shutil.copy2(out/"database/phase5_compilation_core.sqlite",database)
    with sqlite3.connect(database) as db:
        for table,path in (("source_bindings","validation/source_bindings.parquet"),
            ("class_runtime_benchmarks","validation/runtime_benchmark.csv"),("resource_history","validation/resource_history.csv"),
            ("path_fidelity","validation/path_fidelity.parquet"),("racn_witness_arcs","registries/racn_witness_arcs.parquet")):
            frame=pd.read_parquet(out/path) if path.endswith("parquet") else pd.read_csv(out/path)
            frame.to_sql(table,db,index=False,if_exists="fail",chunksize=10000)
        preflight=json.loads((out/"reports/preflight_decision.json").read_text())
        pd.DataFrame([{"run_id":ctx.config["run_id"],"decision":preflight["decision"],"payload_json":json.dumps(preflight,sort_keys=True)}]).to_sql("preflight_receipt",db,index=False)
        interface=json.loads((out/"registries/regional_interface_contract.json").read_text())
        pd.DataFrame([{"suite_id":r["suite_id"],"phase6_authorized":False,"payload_json":json.dumps(r,sort_keys=True)} for r in interface["suites"]]).to_sql("regional_interface_contract",db,index=False)
        pd.DataFrame([{"limitation_key":k,"value_json":json.dumps(v,sort_keys=True)} for k,v in limitations.items()]).to_sql("model_limitations",db,index=False)
        receipts=[];artifacts=[]
        for p in sorted((out/"stage_receipts").glob("*.json")):
            r=json.loads(p.read_text());receipts.append({"stage_id":r["stage_id"],"status":r["status"],"elapsed_seconds":r["elapsed_seconds"],"dependency_hash":r["dependency_hash"],"receipt_sha256":sha(p)})
            artifacts.extend({"stage_id":r["stage_id"],**a} for a in r["artifacts"])
        pd.DataFrame(receipts).to_sql("stage_receipts",db,index=False)
        pd.DataFrame(artifacts).to_sql("stage_receipt_artifacts",db,index=False)
        pd.DataFrame([{"run_id":ctx.config["run_id"],"C7_binding":"Final C7 receipt and complete run MANIFEST are outside this database to avoid cyclic self hashes",
            "source_sha256":ctx.config["source_pilot_sha256"],"authority_sha256":ctx.config["authority_zip_sha256"],"resource_snapshot_unix":time.time()}]).to_sql("run_manifest",db,index=False)
        if db.execute("PRAGMA integrity_check").fetchone()[0]!="ok":raise ValueError("Final evidence database integrity failure")
    payload={"run_id":ctx.config["run_id"],"status":"C0_C6_COMPLETE_C7_SEAL_PENDING","compilation_status":"COMPLETE",
        "model_readiness":"MODEL_LIMITATIONS_REMAIN","production_accepted":False,"phase6_authorized":False,"training_authorized":False,
        "national_assignment_performed":False,"national_assignment_authorized":False,"coverage":coverage,
        "model_limitations":limitations,"next_action":"Commit explicit code/tests/docs result paths, reopen and seal one final ZIP"}
    write_json(out/"reports/preseal_result.json",payload)
    return [out/"reports/preseal_result.json",database]


def _describe(path):
    if path.suffix==".parquet":
        import pyarrow.parquet as pq
        f=pq.ParquetFile(path)
        return f.metadata.num_rows,str(f.schema_arrow)
    if path.suffix==".npz":
        with np.load(path,allow_pickle=False) as z:
            return len(z.files),json.dumps({n:{"shape":z[n].shape,"dtype":z[n].dtype.str} for n in z.files})
    if path.suffix==".sqlite":
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro",uri=True) as db:
            names=[r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            info={n:db.execute('SELECT COUNT(*) FROM "'+n+'"').fetchone()[0] for n in names}
            return sum(info.values()),json.dumps(info)
    return 1,path.suffix


def external_manifest(out):
    records=[]
    for p in sorted(out.rglob("*")):
        if not p.is_file() or p.is_relative_to(out/"manifests") or p.name.endswith(".tmp"):continue
        count,schema=_describe(p)
        records.append({**file_record(out,p),"absolute_path":str(p),"format":p.suffix,"row_or_object_count":count,
            "schema_or_shape":schema,"handoff_policy":"EXTERNAL_FULL_BOUND_BY_HASH"})
    frame=pd.DataFrame(records);frame_write(out/"manifests/MANIFEST.csv",frame)
    checks="".join(r["sha256"]+"  "+r["relative_path"]+"\n" for r in records)
    (out/"manifests/CHECKSUMS.sha256").write_text(checks,encoding="utf-8")
    return frame


def balanced_path_sample(config,destination):
    source=Path(config["source_pilot_output_root"])
    original=pd.read_parquet(source/"path_validation/path_cost_sample.parquet")
    original["path_id"]=["p5_path_"+hashlib.sha256(json.dumps((r.evaluation_role,str(r.sample_od_id)),sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest() for r in original.itertuples(index=False)]
    worst=set(original.sort_values("absolute_relative_cost_error",ascending=False).head(20).path_id)
    selected=[]
    for role,g in original.groupby("evaluation_role"):
        mandatory=g.loc[g.path_id.isin(worst)]
        rest=g.loc[~g.path_id.isin(worst)].copy();rest["sample_rank"]=rest.path_id.map(lambda s:hashlib.sha256(s.encode()).hexdigest())
        selected.append(pd.concat([mandatory,rest.sort_values("sample_rank").head(40-len(mandatory))]))
    selected=pd.concat(selected,ignore_index=True)
    if selected.path_id.nunique()!=120 or not worst.issubset(set(selected.path_id)):raise ValueError("Balanced/worst path review sampling failed")
    jsoncolumns=[c for c in selected if c.endswith("_json")]
    simple=selected.drop(columns=jsoncolumns+["sample_rank"],errors="ignore")
    frame_write(destination/"legacy_path_review_scalar_fields.parquet",simple)
    # All original source fields remain available via normalized related tables.
    normalized=[]
    for r in selected.itertuples(index=False):
        for col in jsoncolumns:
            value=getattr(r,col)
            if not isinstance(value,str):continue
            obj=json.loads(value)
            items=obj.items() if isinstance(obj,dict) else enumerate(obj) if isinstance(obj,list) else [("value",obj)]
            for key,item in items:
                normalized.append({"path_id":r.path_id,"source_field":col,"key":str(key),"value_json":json.dumps(item,sort_keys=True,separators=(",", ":"))})
    frame=pd.DataFrame(normalized)
    for i in range(0,len(frame),100000):frame_write(destination/f"legacy_path_related_fields_{i//100000:03d}.parquet",frame.iloc[i:i+100000])
    out=Path(config["output_root"])
    for name in ("path_fidelity","path_physical_arcs","path_node_half_incidence","path_original_static_occurrences"):
        f=pd.read_parquet(out/"validation"/(name+".parquet"));f=f.loc[f.path_id.isin(selected.path_id)]
        for i in range(0,len(f),100000):frame_write(destination/f"{name}_{i//100000:03d}.parquet",f.iloc[i:i+100000])
    write_json(destination/"path_review_sample_receipt.json",{"records":120,"roles":selected.evaluation_role.value_counts().to_dict(),
        "all_worst20_included":True,"source_population":600,"full_local_source_preserved":True,"sample_method":"40 per source role; all 20 worst accepted v4 cases; remaining stable SHA256 rank",
        "original_fields":list(original.columns),"normalized_json_fields":jsoncolumns,"path_ids":selected.path_id.tolist()})


def seal_final(config_path,cfg):
    out=Path(cfg["output_root"]);root=Path(cfg["repository_root"])
    from .campaign_cli import dependencies
    for stage in ("C0","C1","C2","C3","C4","C5","C6","C7"):
        verify_receipt(out,out/"stage_receipts"/(stage+".json"),dependencies(cfg,config_path,stage))
    final_head=git(root,"rev-parse","HEAD")
    changed=git(root,"status","--porcelain","--","partC_rev1_src/phase5","tests/test_phase5*",cfg["review_gate_relative"],cfg["work_records_relative"],"partC_doc_icnn/00_READ_FIRST.md","partC_doc_icnn/PROJECT_STATUS.md","partC_doc_icnn/12_DECISION_LOG.md")
    if changed:raise ValueError("Current campaign code/tests/docs must be explicitly committed before sealing")
    from .campaign_validation import validate_serialization
    reopened=validate_serialization(cfg)
    coverage=json.loads((out/"validation/complete_coverage.json").read_text())
    limitations=json.loads((out/"validation/model_limitations.json").read_text())
    resources=pd.read_csv(out/"validation/resource_history.csv")
    elapsed=time.time()-cfg["task_started_unix"]
    stage_rows=[]
    for stage in ("C0","C1","C2","C3","C4","C5","C6","C7"):
        r=json.loads((out/"stage_receipts"/(stage+".json")).read_text());stage_rows.append({"stage":stage,"seconds":r["elapsed_seconds"],"status":r["status"]})
    frame_write(out/"validation/stage_timing.csv",pd.DataFrame(stage_rows))
    status={"campaign_id":cfg["campaign_id"],"run_id":cfg["run_id"],"status":"PHASE5_CAMPAIGN_COMPLETE_WAITING_EXTERNAL_REVIEW",
        "technical_status":"PASS","compilation_status":"COMPLETE","model_readiness":"MODEL_LIMITATIONS_REMAIN",
        "production_accepted":False,"phase6_authorized":False,"training_authorized":False,"national_assignment_performed":False,
        "national_assignment_authorized":False,"physical_benchmark_rerun_performed":False,"source_sha256":cfg["source_pilot_sha256"],
        "ending_head":final_head,"blockers":[],"model_limitations":list(limitations.keys()),"next_step_authorized":False,
        "authority_sha256":cfg["authority_zip_sha256"],"root_free_runtime_coverage":1.,"proxy_count":0,"online_root_calls":0,
        "coverage":coverage,"whole_task_elapsed_seconds_before_seal":elapsed,
        "peak_recorded_process_tree_rss_bytes":int(resources.process_tree_rss_bytes.max()),"added_disk_bytes_before_seal":int(resources.added_disk_bytes.max()),
        "resource_sampling":"30-second supervisor process-tree samples; initial coding wall time included from user request",
        "stage_timing":stage_rows,"engineering_epochs_used":len(list((out/"engineering_epochs").glob("epoch_*/repair.json")))}
    import jsonschema
    jsonschema.validate(status,json.loads((Path(cfg["authority_root"])/"config/status.schema.json").read_text()))
    write_json(out/"reports/campaign_status.json",status)
    gates=pd.read_csv(out/"validation/acceptance_results_preseal.csv");gates.loc[gates.gate_id.eq("G19"),"status"]="PASS_SUBJECT_TO_ATOMIC_SEAL_VERIFICATION"
    gates.loc[gates.gate_id.eq("G19"),"evidence"]="Adjacent final companion records actual fresh extraction/hash/size checks; ZIP is not promoted on failure"
    frame_write(out/"validation/acceptance_results.csv",gates)
    summary=("# Phase 5 full compilation-and-validation campaign\n\n"
        "C0-C7 compilation and validation completed for s02-s06. Technical status PASS; complete native runtime coverage; model readiness MODEL_LIMITATIONS_REMAIN. Production is not accepted.\n\n"
        f"Authority SHA-256 `{cfg['authority_zip_sha256']}`; source v4 ZIP `{cfg['source_pilot_sha256']}`; final repository HEAD `{final_head}` on `{cfg['branch']}`. The v4 pilot is externally complete with notes and retains its adverse model disposition.\n\n"
        f"Actual emitted links: {coverage['emitted_links']:,}; distinct dynamic models: {coverage['dynamic_models']:,}. Online roots and proxies: zero. All observed physical member CURRENT flows remain distinct; AON is separate domain evidence.\n\n"
        "Exact functions and low-only beta-4 fits were validated against certified scalar references. Rejected corridors received the one authorized reciprocal-compatible split, otherwise original primitives were retained with explicit physical-member fallback. Regional interfaces and accepted scientific inputs remain unchanged.\n\n"
        "The full 600 original path witnesses were reevaluated under baseline and structural charges. Signed error components and paired rankings remain explicit. Fixed-charge/path errors, free-mixing endpoint state, unknown counterfactual R transitions and global liftability remain external scientific questions. Compiler-only timing does not establish national assignment speedup.\n\n"
        f"Whole-task time before seal: {elapsed/3600:.3f} h. Peak sampled process-tree RSS: {status['peak_recorded_process_tree_rss_bytes']/2**30:.3f} GiB. Added disk before seal: {status['added_disk_bytes_before_seal']/2**30:.3f} GiB. Exact per-stage and final sealing resources are in the machine records.\n\n"
        "Recommendation: externally review the complete research compiler and adverse findings. Do not infer production readiness. National assignment, Phase 6, training and physical benchmark reruns remain unauthorized.\n")
    (out/"reports/campaign_report.md").write_text(summary,encoding="utf-8")
    required=pd.read_csv(cfg["expected_outputs_file"])
    for pattern in required.relative_path:
        names=[pattern.format(suite=s) for s in cfg["suites"]] if "{suite}" in pattern else [pattern]
        for name in names:
            if name.startswith("manifests/"):continue
            if not (out/name).is_file():raise FileNotFoundError("Required campaign output missing: "+name)
    manifest=external_manifest(out)
    handoff_id="campaign_v1_0905_"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"_final_001"
    parent=root/"partC_doc_icnn/handoffs/phase_05/step_campaign_v1_0905"
    source=parent/handoff_id;staging=parent/(handoff_id+".building")
    archive_root=root/"artifacts_icnn/handoff_bundles/phase_05"
    archive=archive_root/("phase05_stepcampaign_v1_0905_"+handoff_id+"_SYNC.zip")
    if source.exists() or staging.exists() or archive.exists():raise FileExistsError("Handoff is write-new only")
    staging.mkdir(parents=True);archive_root.mkdir(parents=True,exist_ok=True)
    def copy(p,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    sections=("governing_docs","configs","code_snapshot","code_diff","summary_data","figures","geospatial_summary","log_excerpts","external_artifact_index")
    for section in sections:(staging/section).mkdir()
    for p in sorted(Path(cfg["authority_root"]).rglob("*")):
        if p.is_file() and p.suffix!=".zip":copy(p,staging/"governing_docs/authority"/p.relative_to(cfg["authority_root"]))
    copy(config_path,staging/"configs/campaign_execution.json")
    for name in ("00_READ_FIRST.md","20_HANDOFF_SYNC_PROTOCOL.md","21_REVIEW_APPROVAL_GATES.md","23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md","24_CODE_AND_DATA_SNAPSHOT_POLICY.md","PROJECT_STATUS.md","12_DECISION_LOG.md"):
        copy(root/"partC_doc_icnn"/name,staging/"governing_docs"/name)
    for p in sorted((root/cfg["work_records_relative"]).rglob("*")):
        if p.is_file():copy(p,staging/"summary_data/work_records"/p.relative_to(root/cfg["work_records_relative"]))
    snapshot=list(code_binding(root))
    snapshot += ["partC_rev1_src/phase4/inputs.py","partC_rev1_src/phase4/models.py","partC_rev1_src/phase4/tracing.py",
                 "partC_rev1_src/partC_rev1_shortest_paths.py"]
    for name in snapshot:copy(root/name,staging/"code_snapshot"/name)
    patch=subprocess.check_output(["git","-C",str(root),"-c","core.longpaths=true","diff",cfg["required_ancestor"],final_head,"--",
        "partC_rev1_src/phase5","tests/test_phase5*",cfg["review_gate_relative"],cfg["work_records_relative"],"partC_doc_icnn/00_READ_FIRST.md","partC_doc_icnn/PROJECT_STATUS.md","partC_doc_icnn/12_DECISION_LOG.md"])
    (staging/"code_diff/campaign.patch").write_bytes(patch)
    copy(out/"manifests/MANIFEST.csv",staging/"external_artifact_index/full_output_MANIFEST.csv")
    copy(out/"manifests/CHECKSUMS.sha256",staging/"external_artifact_index/full_output_CHECKSUMS.sha256")
    write_json(staging/"external_artifact_index/source_archives.json",{"authority":{"path":cfg["authority_zip_path"],"sha256":cfg["authority_zip_sha256"]},
        "pilot":{"path":cfg["source_pilot_zip"],"sha256":cfg["source_pilot_sha256"]},"recursive_archive_embedding":False})
    for directory in ("reports","validation","preflight","figures","stage_receipts","failures"):
        for p in sorted((out/directory).glob("*")):
            if p.is_file() and p.stat().st_size<=2*2**20 and p.suffix in (".json",".md",".csv",".log",".xml",".png",".jsonl"):
                copy(p,staging/("figures" if directory=="figures" else "summary_data"/Path(directory))/p.name)
    # Compact balanced decision/model/ownership samples, preserving rare actions.
    for name,key in (("corridor_decisions","action"),("calibration_requests","status"),("split_decisions","group_action"),
                     ("dynamic_models","runtime_class"),("physical_link_ownership","ownership_role"),("charges","model_variant")):
        frame=pd.read_parquet(out/"registries"/(name+".parquet"))
        sample=pd.concat([g.head(300) for _,g in frame.groupby(key,dropna=False)],ignore_index=True)
        frame_write(staging/"summary_data/registry_samples"/(name+".parquet"),sample)
    balanced_path_sample(cfg,staging/"summary_data/path_review")
    (staging/"geospatial_summary/NO_NEW_GEOSPATIAL_PRODUCT.md").write_text("No new spatial model was introduced. Accepted Phase 3 geometry/interface bindings remain external and unchanged. Actual post-fit/expanded counts are plotted.\n",encoding="utf-8")
    headers={"00_HANDOFF_README.md":"Upload this single campaign ZIP and its adjacent companion. The source is a complete research compiler, with model limitations remaining.\n",
        "01_STEP_SUMMARY.md":summary,"03_DECISIONS_AND_OPEN_QUESTIONS.md":json.dumps(limitations,indent=2),
        "04_VALIDATION_RESULTS.md":"All C0-C6 technical/coverage gates pass. C7 promotes the ZIP only after fresh extraction, complete checksums, size and snapshot checks. See acceptance/results and companion.\n",
        "08_COMMAND_LOG.md":"Exact commands are recorded in summary_data/validation/command_log.jsonl and code snapshot campaign_cli.py.\n",
        "09_NEXT_STEP_RECOMMENDATION.md":"External review of compiler products and scientific model limitations. No downstream execution is authorized.\n",
        "10_UPLOAD_THIS_FILE.txt":str(archive)+"\n",
        "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md":"New full campaign following externally reviewed v4 pilot. Full request coverage, independent validation, one deterministic reciprocal split, explicit physical-member fallback, complete current mappings and no-proxy ablation. Old pilot/archive products are preserved, hash-bound and not recursively embedded.\n"}
    for name,text in headers.items():(staging/name).write_text(text,encoding="utf-8")
    write_json(staging/"02_HANDOFF_STATUS.json",status)
    copy(out/"validation/resource_history.csv",staging/"05_RESOURCE_USAGE.csv")
    frame_write(staging/"07_CODE_CHANGE_INDEX.csv",pd.DataFrame([{"relative_path":p,"sha256":sha(root/p)} for p in snapshot]))
    # Scan current evidence using the established repository scanner.
    from .handoff import _privacy_scan
    scan=_privacy_scan(staging)
    # The historical text scanner predates the authorized normalized Parquet
    # review tables. Independently reopen those typed binary payloads and XML
    # test receipts before classifying their format as reviewed.
    resolved=[];remaining=[]
    for finding in scan["findings"]:
        p=staging/finding["path"]
        if finding["pattern"]=="unreviewed_binary_type" and p.suffix==".parquet":
            frame=pd.read_parquet(p)
            if frame.empty:raise ValueError("Empty review sample")
            resolved.append({**finding,"disposition":"TYPED_PARQUET_REOPEN_PASS","rows":len(frame)})
        elif finding["pattern"]=="unreviewed_binary_type" and p.suffix==".xml":
            import xml.etree.ElementTree as ET
            ET.fromstring(p.read_text(encoding="utf-8"));resolved.append({**finding,"disposition":"UTF8_XML_PARSE_PASS"})
        else:remaining.append(finding)
    scan.update(findings=remaining,format_reopen_evidence=resolved,status="PASS" if not remaining else "FAIL")
    write_json(staging/"log_excerpts/privacy_scan.json",scan)
    if remaining:raise ValueError("Handoff text/privacy scan failed: "+json.dumps(remaining))
    payload=[]
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            if p.stat().st_size>cfg["handoff"]["member_limit_bytes"]:raise ValueError("Handoff member exceeds 25 MiB: "+str(p))
            payload.append(file_record(staging,p))
    frame_write(staging/"06_ARTIFACT_MANIFEST.csv",pd.DataFrame(payload))
    checks=payload+[file_record(staging,staging/"06_ARTIFACT_MANIFEST.csv")]
    (staging/"CHECKSUMS.sha256").write_text("".join(r["sha256"]+"  "+r["relative_path"]+"\n" for r in checks),encoding="utf-8")
    tmp=archive.with_suffix(".zip.building")
    with zipfile.ZipFile(tmp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(staging.rglob("*")):
            if p.is_file():z.write(p,p.relative_to(staging).as_posix())
    if tmp.stat().st_size>cfg["handoff"]["hard_limit_bytes"]:raise ValueError("Handoff exceeds hard 100 MiB")
    # A write-new fresh extraction is preserved as sealing evidence until verified.
    verify_root=out.parent/"seal_verification"/handoff_id
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise ValueError("ZIP CRC failure")
        for n in z.namelist():
            if not (verify_root/n).resolve().is_relative_to(verify_root.resolve()):raise ValueError("Unsafe ZIP member")
        z.extractall(verify_root)
    for r in checks:
        p=verify_root/r["relative_path"]
        if p.stat().st_size!=r["size_bytes"] or sha(p)!=r["sha256"]:raise ValueError("Fresh extraction checksum/size mismatch")
    expected={r["relative_path"] for r in checks}|{"CHECKSUMS.sha256"}
    actual={p.relative_to(verify_root).as_posix() for p in verify_root.rglob("*") if p.is_file()}
    if actual!=expected:raise ValueError("Fresh extraction manifest closure failure")
    added=sum(p.stat().st_size for p in out.parent.rglob("*") if p.is_file())+sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())+tmp.stat().st_size
    for sub in (cfg["review_gate_relative"],cfg["work_records_relative"]):
        added+=sum(p.stat().st_size for p in (root/sub).rglob("*") if p.is_file())
    if added>=cfg["resources"]["added_disk_limit_bytes"] or time.time()-cfg["task_started_unix"]>=cfg["resources"]["base_wall_seconds"]:
        raise ValueError("Final sealing resource limit exceeded before ZIP promotion")
    zip_hash=sha(tmp);tmp.rename(archive);staging.rename(source)
    sha_path=Path(str(archive)+".sha256");sha_path.write_text(zip_hash+"  "+archive.name+"\n",encoding="utf-8")
    companion_path=Path(str(archive)+".companion.json")
    total_seconds=time.time()-cfg["task_started_unix"]
    companion={"campaign_id":cfg["campaign_id"],"run_id":cfg["run_id"],"handoff_id":handoff_id,"status":status["status"],
        "zip_path":str(archive),"zip_size_bytes":archive.stat().st_size,"zip_sha256":zip_hash,"sha256_path":str(sha_path),
        "companion_path":str(companion_path),"source_folder":str(source),"ending_head":final_head,
        "config_sha256":sha(config_path),"authority_sha256":cfg["authority_zip_sha256"],"source_pilot_sha256":cfg["source_pilot_sha256"],
        "fresh_extraction_checksum_entries":len(checks),"fresh_extraction_manifest_closure":"PASS","size_gate":"PASS",
        "all_19_acceptance_gates":"PASS","technical_status":"PASS","compilation_status":"COMPLETE","model_readiness":"MODEL_LIMITATIONS_REMAIN",
        "production_accepted":False,"next_step_authorized":False,"phase6_authorized":False,"training_authorized":False,"national_assignment_authorized":False,
        "whole_task_elapsed_seconds":total_seconds,"peak_recorded_process_tree_rss_bytes":status["peak_recorded_process_tree_rss_bytes"],"added_disk_bytes_including_handoff_and_verification":added,
        "coverage":coverage,"stage_timing":stage_rows,"external_decisions_required":["Review complete compiler research products","Assess B/R charge and state fidelity before any model release","Separately authorize any later phase"]}
    if added>=cfg["resources"]["added_disk_limit_bytes"] or total_seconds>=cfg["resources"]["base_wall_seconds"]:raise ValueError("Final sealing resource limit exceeded")
    write_json(companion_path,companion)
    (parent/"LATEST_HANDOFF.txt").write_text(json.dumps({"source_folder":str(source),"zip_path":str(archive),"sha256":zip_hash,"companion":str(companion_path),"status":status["status"]},indent=2)+"\n",encoding="utf-8")
    return companion
