"""Authenticated write-new adoption and byte-preserving continuation copies."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse
from .stack_sampler import StackSampler
import hashlib
import io
import json
import os
import shutil
import time
import zipfile
import pandas as pd

from .common import sha, git, read_json, write_json, write_frame, event, load_config, Monitor

AUTHORITY_SHA = "09ead1aec6f380c5ab32e5f484a50f20daefe9adb1884dce989be3350c384217"
LAUNCH_SHA = "b32eb068579b445b185a75bc5827e25283b0e1d7e155e2f4ebf3073403d7f019"


def verify_zip(raw):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        if z.testzip() is not None: raise ValueError("Authority CRC failure")
        checks=[]
        for line in z.read("CHECKSUMS.sha256").decode().splitlines():
            if not line.strip(): continue
            digest,name=line.split(None,1); name=name.lstrip("*")
            if hashlib.sha256(z.read(name)).hexdigest()!=digest: raise ValueError("Authority checksum: "+name)
            checks.append(name)
        return len(checks)


def install(authority, launch):
    authority=Path(authority); launch=Path(launch)
    if sha(authority)!=AUTHORITY_SHA or sha(launch)!=LAUNCH_SHA: raise ValueError("Wrong current authority/launch")
    raw=authority.read_bytes(); counts={"outer":verify_zip(raw)}
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        c=json.loads(z.read("config/effective_execution.json"))
        repo=Path(c["repository_root"]); root=Path(c["output_root"])
        auth=repo/c["authority_install_relative"]/"authority"
        work=repo/c["work_records_relative"]
        if any(p.exists() for p in [root,auth,work]): raise FileExistsError("Adoption is write-new; inspect matching live continuation")
        if git(repo,"branch","--show-current")!=c["branch"]: raise ValueError("Wrong branch")
        git(repo,"merge-base","--is-ancestor",c["required_ancestor"],"HEAD")
        parent=Path(c["parent_output_root"])
        if (parent/"control/writer.lock").exists(): raise ValueError("Parent has a writer lock; verify/stop only its live handle")
        for member,key in [("input/blocked_active_campaign_SYNC.zip","current_reviewed_handoff"),("input/parent_active_authority.zip","current_parent")]:
            data=z.read(member); identity=c["input_versions"][key]
            if len(data)!=identity["bytes"] or hashlib.sha256(data).hexdigest()!=identity["sha256"]: raise ValueError("Embedded identity differs")
            counts[member]=verify_zip(data)
        latest=json.loads((repo/"partC_doc_icnn/handoffs/phase_06/step_s03_active_0906/LATEST_HANDOFF.txt").read_text())
        if sha(latest["zip_path"])!=c["input_versions"]["current_reviewed_handoff"]["sha256"]: raise ValueError("Live reviewed handoff changed")
        source_code=[]
        with zipfile.ZipFile(io.BytesIO(z.read("input/blocked_active_campaign_SYNC.zip"))) as old:
            for n in old.namelist():
                prefix="code_snapshot/partC_rev1_src/phase6/"
                if n.startswith(prefix) and n.endswith(".py"):
                    p=repo/n.removeprefix("code_snapshot/")
                    source_code.append(dict(path=str(p),reviewed_sha256=hashlib.sha256(old.read(n)).hexdigest(),live_sha256=sha(p)))
        root.mkdir(parents=True); auth.mkdir(parents=True); work.mkdir(parents=True)
        for name in ["control","validation","manifests","history","failures","reports","dataset","cache/numba","authority_archives"]:
            (root/name).mkdir(parents=True,exist_ok=True)
        mapping=[]
        for info in z.infolist():
            if info.is_dir(): continue
            target=(root/"authority_archives"/Path(info.filename).name if info.filename.startswith("input/") else auth/info.filename)
            allowed=root/"authority_archives" if info.filename.startswith("input/") else auth
            if not target.resolve().is_relative_to(allowed.resolve()): raise ValueError("Unsafe authority member")
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open("xb") as f: f.write(z.read(info.filename))
            mapping.append(dict(member=info.filename,path=str(target),sha256=sha(target),bytes=target.stat().st_size))
        shutil.copyfile(authority,root/"authority_archives/current_authority.zip")
        shutil.copyfile(launch,auth.parent/launch.name)
        adopted=time.time()
        clock=dict(original_start_unix=c["task_started_unix"],hard_deadline_unix=c["hard_deadline_unix"],
            science_cutoff_unix=c["science_deadline_unix"],adopted_unix=adopted,
            remaining_whole_seconds=c["hard_deadline_unix"]-adopted,remaining_science_seconds=c["science_deadline_unix"]-adopted,
            external_review_wall_time_included=True,new_clock_authorized=False)
        c.update(authority_root=str(auth),authority_zip_sha256=AUTHORITY_SHA,launch_sha256=LAUNCH_SHA,
            authority_archive_path=str(root/"authority_archives/current_authority.zip"),work_records_root=str(work),
            parent_authority_root=str(repo/"partC_doc_icnn/review_gates/phase_06/s03_active_0906/authority"),
            expected_outputs_document=str(auth/"07_OUTPUTS_ACCEPTANCE_AND_HANDOFF.md"),
            acceptance_tests_config=str(auth/"config/acceptance_and_tests.json"),
            comparison_policy_file=str(auth/"config/benchmark_comparison_policy.json"),
            failure_policy_file=str(auth/"config/region_failure_policy.json"),
            adopted_unix=adopted,source_handoff_path=latest["zip_path"],source_code_at_adoption=source_code,
            parent_added_bytes_at_adoption=read_json(parent/"manifests/HANDOFF_SEAL_RESULT.json")["measured_added_bytes_in_run_root_and_this_handoff"])
        config=work/"effective_execution.json";write_json(config,c)
        write_json(root/"control/clock_adoption.json",clock)
        write_json(root/"control/resume_identity.json",dict(run_id=c["run_id"],authority_sha256=AUTHORITY_SHA,
            config_sha256=sha(config),parent_run_id=c["parent_run_id"],parent_output_root=c["parent_output_root"],
            source_handoff_sha256=latest["zip_sha256"],adoption_head=git(repo,"rev-parse","HEAD"),clock=clock))
        evidence=dict(checksum_counts=counts,installed_files=mapping,source_code=source_code,clock=clock)
        write_json(work/"AUTHORITY_VERIFICATION.json",evidence)
        decision=json.loads(z.read("config/review_decision.json"))
        write_json(auth.parent/"APPROVAL.json",dict(**decision,decision="REPAIR_AND_RESUME_UNDER_EXPLICIT_USER_AUTHORITY",
            current_authority_sha256=AUTHORITY_SHA,current_launch_sha256=LAUNCH_SHA,
            reviewed_sync_zip=latest["zip_path"],reviewed_sync_zip_sha256=latest["zip_sha256"],
            next_authorized_step="R0 through cumulative R5 handoff within original deadline",recorded_at_unix=adopted))
        (auth.parent/"REVIEW_DECISION.md").write_text("# Explicit user review resolution\n\n"+z.read("01_REVIEW_AND_AUTHORIZATION.md").decode(),encoding="utf-8")
        (work/"EXECUTION_PLAN.md").write_text("# Resume execution plan\n\nR0 authenticates and independently copies all live scientific files. R1 repairs conditional comparison, typed quarantine and all downstream gates, tests and commits before dispatch. R2 freshly reopens copied native labels and reconciles every frozen region. R3 fills healthy missing baseline inputs, with explicit isolated-failure dockets. R4 completes healthy audits, precision, real restriction and loaders. E continues fair proportional fresh seeds until measured reserve. R5 reopens cumulative data and seals one compact review handoff. No original file is changed, no clock resets, and current authority supersedes intermediate external stops only within this scope.\n",encoding="utf-8")
        for name in ["00_READ_FIRST.md","PROJECT_STATUS.md"]:
            p=repo/"partC_doc_icnn"/name
            lead="# Current authority: Phase 6 active repair and resume, 0907\n\nExplicit user authority P6_ACTIVE_RESUME_0907_V1 resolves the conditional mapped-comparison defect and authorizes <5% isolated regional quarantine with healthy continuation. Read review_gates/phase_06/s03_active_resume_0907/authority/00_READ_FIRST.md and current configs. Execution config: phase_6_work/s03_active_resume_0907/effective_execution.json. Parent data and handoff remain immutable. Original deadline is 2026-09-08T03:29:13Z. One cumulative final handoff, no intermediate regional approvals.\n\n---\n\n"
            p.write_text(lead+p.read_text(encoding="utf-8"),encoding="utf-8")
        p=repo/"partC_doc_icnn/12_DECISION_LOG.md"
        p.write_text("# D-048 — Explicit conditional comparison and regional quarantine resume authority\n\nThe user authorizes START_P6_ACTIVE_RESUME_0907.md under authority SHA-256 "+AUTHORITY_SHA+". Reviewed source handoff "+latest["zip_sha256"]+" at 29780f6c7b24db386651a8682e16693baeaa62a5 remains preserved. The mapped physical upper comparison is conditional on certified same-demand path feasibility. Nine source warnings are nonblocking; genuine isolated regional failures are quarantined while 20*F<1566. Healthy baseline then proportional enrichment use the unchanged model, labels/support and original clock. Three of six engineering epochs remain. No new scientific model, path rule, training or national assignment is authorized. Exact review and config are under review_gates/phase_06/s03_active_resume_0907.\n\n---\n\n"+p.read_text(encoding="utf-8"),encoding="utf-8")
    return dict(config=str(config),clock=clock,verified=counts)


def copy_once(c):
    root=Path(c["output_root"]); parent=Path(c["parent_output_root"])
    if (root/"validation/COPY_COMPLETE.json").exists(): return read_json(root/"validation/COPY_COMPLETE.json")
    tasks=[]
    for name in ["registry","samples","regions"]:
        for p in sorted((parent/name).rglob("*")):
            if p.is_file(): tasks.append((p,root/p.relative_to(parent),True))
    for name in ["validation","control","reports"]:
        for p in sorted((parent/name).rglob("*")):
            if p.is_file() and p.name!="writer.lock":tasks.append((p,root/"history/parent_run"/p.relative_to(parent),False))
    tasks.append((parent/"validation/source_manifest.parquet",root/"validation/source_manifest.parquet",False))
    def one(task):
        src,dst,scientific=task; raw=src.read_bytes(); digest=hashlib.sha256(raw).hexdigest()
        dst.parent.mkdir(parents=True,exist_ok=True)
        if dst.exists():
            if sha(dst)!=digest: raise ValueError("Continuation copy collision: "+str(dst))
        else:
            with dst.open("xb") as f:f.write(raw)
        if sha(dst)!=digest or os.path.samefile(src,dst):raise ValueError("Copy bytes/independence failure")
        return dict(source_path=str(src),destination_path=str(dst),relative_path=src.relative_to(parent).as_posix(),
            sha256=digest,size_bytes=len(raw),scientific=scientific,independent_copy=True)
    shards=[]; total=0; size=0
    with Monitor(c,"R0_VERIFIED_COPY") as monitor,ThreadPoolExecutor(max_workers=4,thread_name_prefix="copy") as pool:
        for start in range(0,len(tasks),6000):
            monitor.check(); rows=[]
            for offset in range(start,min(start+6000,len(tasks)),256):
                rows.extend(pool.map(one,tasks[offset:min(offset+256,start+6000,len(tasks))]))
            frame=pd.DataFrame(rows); path=root/"manifests"/f"source_copy_{len(shards):04d}.parquet"
            if path.exists():
                if not pd.read_parquet(path).equals(frame):raise ValueError("Changed copy resume shard")
            else:write_frame(path,frame)
            shards.append(dict(path=str(path),sha256=sha(path),rows=len(rows)))
            total+=len(rows);size+=int(frame.size_bytes.sum())
            print("R0 COPY",total,"/",len(tasks),flush=True)
    result=dict(status="PASS",files=total,bytes=size,manifest_shards=shards,source_immutable=True,
        mutable_catalogs_independent=True,new_optimizer_calls=0,run_id=c["run_id"],source_root=str(parent))
    write_json(root/"validation/COPY_COMPLETE.json",result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--operation",required=True,choices=["install","copy"])
    p.add_argument("--authority");p.add_argument("--launch");p.add_argument("--config")
    a=p.parse_args()
    if a.operation=="install": result=install(a.authority,a.launch)
    else:
        c=load_config(a.config)
        if git(c["repository_root"],"status","--porcelain","--","partC_rev1_src/phase6/resume_adoption.py"):
            raise ValueError("Commit adoption implementation before copy execution")
        root=Path(c["output_root"]); lock=root/"control/writer.lock"
        with lock.open("x") as f:f.write(str(os.getpid()))
        identity=dict(stage="R0_VERIFIED_COPY",pid=os.getpid(),run_id=c["run_id"],
            code_head=git(c["repository_root"],"rev-parse","HEAD"),config_sha256=sha(a.config))
        try:
            with (root/"control/R0_stacks.log").open("a") as stack,StackSampler(stack):
                event(root/"control/stage_events.jsonl",dict(**identity,status="STARTED"))
                result=copy_once(c)
                event(root/"control/stage_events.jsonl",dict(**identity,status="PASS",result=result))
        finally:
            lock.unlink(missing_ok=True)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":main()
