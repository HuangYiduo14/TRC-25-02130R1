"""One new campaign identity; serial supervised C0-C7 with resume/status.

No legacy pilot command is dispatched. Scientific stages run in independent
children; one supervisor owns resource history, receipts and the command log.
"""
from __future__ import annotations

import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

import psutil

from .campaign_io import (append_event,binding_hash,code_binding,file_record,git,
                         publish_receipt,sha,verify_receipt,write_json)

STAGES=tuple("C"+str(i) for i in range(8))


def load_config(path):
    path=Path(path).resolve();cfg=json.loads(path.read_text(encoding="utf-8"))
    for name,value in (("campaign_id","phase5_campaign_v1_0905"),("run_id","phase5_campaign_v1_0905_run_001"),
        ("phase_id","phase_05"),("step_id","campaign_v1_0905")):
        if cfg.get(name)!=value:raise ValueError("Explicit campaign identity mismatch: "+name)
    if cfg.get("tasks")!=list(STAGES) or not all(cfg.get(k) for k in ("expected_outputs_file","acceptance_tests_file","input_versions_file","resources")):
        raise ValueError("Explicit tasks, inputs, outputs, tests and budget are mandatory")
    if cfg["suites"]!=["s02","s03","s04","s05","s06"]:raise ValueError("Frozen suite population changed")
    if any(cfg["authorization"][k] for k in ("national_assignment_authorized","phase6_authorized","training_authorized","physical_benchmark_rerun_authorized")):
        raise ValueError("Forbidden authorization flag")
    authority=json.loads((Path(cfg["authority_root"])/"config/campaign.json").read_text())
    for k,v in authority.items():
        if cfg[k]!=v:raise ValueError("Execution changed controlling authority field: "+k)
    return path,cfg


def dependencies(cfg,config_path,stage):
    root=Path(cfg["repository_root"])
    output=Path(cfg["output_root"])
    upstream={}
    for predecessor in STAGES[:STAGES.index(stage)]:
        p=output/"stage_receipts"/(predecessor+".json")
        if not p.exists():raise ValueError("Missing upstream stage: "+predecessor)
        upstream[predecessor]=sha(p)
    return binding_hash({"config_sha256":sha(config_path),"code":code_binding(root),"upstream":upstream,
                         "authority":cfg["authority_zip_sha256"],"source":cfg["source_pilot_sha256"]})


class Context:
    def __init__(self,config_path,cfg,stage):
        self.config_path=config_path;self.config=cfg;self.stage=stage
        self.output=Path(cfg["output_root"]);self.dependency_hash=dependencies(cfg,config_path,stage)
    def mark(self,message):
        write_json(self.output/"validation/current_progress.json",{"stage":self.stage,"message":message,"unix":time.time(),"pid":os.getpid()})
        append_event(self.output/"validation/stage_events.jsonl",{"stage":self.stage,"event":"PROGRESS","message":message})
        print(message,flush=True)


def worker(config_path,cfg,stage):
    ctx=Context(config_path,cfg,stage);started=time.time()
    path=ctx.output/"stage_receipts"/(stage+".json")
    if path.exists():raise FileExistsError("Stage already has immutable PASS receipt")
    stacks=ctx.output/"logs"/(stage+"_python_stacks.log")
    stacks.parent.mkdir(parents=True,exist_ok=True)
    with stacks.open("a",encoding="utf-8") as f:
        faulthandler.enable(file=f,all_threads=True)
        faulthandler.dump_traceback_later(cfg["resources"]["long_step_stack_seconds"],repeat=True,file=f)
        stop=threading.Event()
        def requested_stack():
            request=ctx.output/"validation/request_stack.txt"
            while not stop.wait(1):
                if request.exists():
                    faulthandler.dump_traceback(file=f,all_threads=True);f.flush();request.unlink()
        monitor=threading.Thread(target=requested_stack,daemon=True);monitor.start()
        try:
            ctx.mark(stage+" started")
            if stage in STAGES[:5]:
                from . import campaign_stages as stages
                paths=getattr(stages,stage.lower())(ctx)
            elif stage=="C5":
                from .campaign_validation import c5
                paths=c5(ctx)
            elif stage=="C6":
                from .campaign_validation import c6
                paths=c6(ctx)
            else:
                from .campaign_handoff import prepare_final
                paths=prepare_final(ctx)
            publish_receipt(ctx.output,path,stage,ctx.dependency_hash,paths,started,code_head=git(Path(cfg["repository_root"]),"rev-parse","HEAD"))
            ctx.mark(stage+" PASS")
        except BaseException as error:
            failure={"stage":stage,"status":"FAIL","error":repr(error),"traceback":traceback.format_exc(),
                     "reproducer":[sys.executable,"-m","partC_rev1_src.phase5.campaign_cli","worker","--config",str(config_path),"--stage",stage],
                     "started_unix":started,"elapsed_seconds":time.time()-started,"code_head":git(Path(cfg["repository_root"]),"rev-parse","HEAD")}
            target=ctx.output/"failures"/(stage+"_"+str(time.time_ns())+".json");write_json(target,failure)
            append_event(ctx.output/"validation/stage_events.jsonl",{"stage":stage,"event":"FAILED","failure":str(target)})
            traceback.print_exc();return 1
        finally:
            stop.set();monitor.join(timeout=2);faulthandler.cancel_dump_traceback_later()
    return 0


def resource_sample(cfg,child=None):
    current=psutil.Process();processes=[current]+current.children(recursive=True)
    rss=0
    for p in processes:
        try:rss+=p.memory_info().rss
        except psutil.Error:pass
    root=Path(cfg["output_root"]).parent
    added=sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    for sub in (cfg["review_gate_relative"],cfg["work_records_relative"]):
        added+=sum(p.stat().st_size for p in (Path(cfg["repository_root"])/sub).rglob("*") if p.is_file())
    return {"unix":time.time(),"task_elapsed_seconds":time.time()-cfg["task_started_unix"],
            "process_tree_rss_bytes":rss,"added_disk_bytes":added,"child_pid":child.pid if child else 0}


def engineering_repair(config_path,cfg,reason):
    """Preserve all invalidated evidence before a dependency-bound restart.

    C0 binds the full current source/test snapshot. A code epoch conservatively
    invalidates that integrated technical receipt and its C1-C7 descendants.
    Accepted upstream pilot/Phase3/Phase4 assets never enter this operation.
    """
    if not reason or len(reason.strip())<20:raise ValueError("An actual evidenced defect/reproducer is required")
    out=Path(cfg["output_root"]).resolve();root=Path(cfg["repository_root"])
    if not (out/"run_identity.json").is_file():raise ValueError("No matching campaign to repair")
    if not list((out/"failures").glob("*.json")):raise ValueError("No actual failure evidence to repair")
    epochs=out/"engineering_epochs";count=len(list(epochs.glob("epoch_*/repair.json")))
    if count>=cfg["resources"]["engineering_epochs_max"]:raise ValueError("Authorized engineering epoch limit exhausted")
    if git(root,"status","--porcelain","--","partC_rev1_src/phase5","tests/test_phase5*"):
        raise ValueError("Regression and minimal code repair must be committed before invalidation")
    target=epochs/f"epoch_{count+1:03d}";target.mkdir(parents=True,exist_ok=False)
    snapshot=[]
    # All source/execution snapshots are immutable under the epoch archive.
    for name in ("compilation","registries","preflight","reports","suites","database","baseline","figures","stage_receipts"):
        p=(out/name).resolve();destination=(target/name).resolve()
        if not p.is_relative_to(out) or not destination.is_relative_to(out):raise ValueError("Repair archive escaped campaign root")
        if p.exists():
            snapshot.extend(file_record(out,f) for f in p.rglob("*") if f.is_file())
            p.rename(destination)
    for p in list((out/"validation").iterdir()):
        if p.is_file() and p.name not in {"stage_events.jsonl","resource_history.csv","heartbeats.jsonl","command_log.jsonl","current_progress.json"}:
            destination=target/"validation"/p.name;destination.parent.mkdir(exist_ok=True)
            snapshot.append(file_record(out,p));p.rename(destination)
    record={"epoch":count+1,"authority_sha256":cfg["authority_zip_sha256"],"authorization":"User-authorized bounded engineering repair epoch",
        "defect_and_reproducer":reason,"repair_code_head":git(root,"rev-parse","HEAD"),"snapshot":snapshot,
        "invalidated_stages":list(STAGES),"dependency_reason":"C0 integrates complete current source/tests; dependent scientific products preserved and revalidated under new epoch",
        "scientific_thresholds_changed":False,"accepted_inputs_changed":False,"recorded_unix":time.time()}
    write_json(target/"repair.json",record)
    append_event(out/"validation/stage_events.jsonl",{"event":"ENGINEERING_EPOCH","epoch":count+1,"reason":reason,"record":str(target/"repair.json")})
    print(json.dumps(record,indent=2));return 0


def monitor_child(cfg,stage,process):
    out=Path(cfg["output_root"]);history=out/"validation/resource_history.csv"
    start=time.time();last_sample=0.;last_print=0.;diagnosed=False
    while process.poll() is None:
        now=time.time()
        if now-last_sample>=cfg["resources"]["resource_sample_seconds"]:
            sample=resource_sample(cfg,process);sample["stage"]=stage
            with history.open("a",encoding="utf-8",newline="") as f:
                import csv
                writer=csv.DictWriter(f,fieldnames=list(sample))
                if f.tell()==0:writer.writeheader()
                writer.writerow(sample)
            last_sample=now
            progresspath=out/"validation/current_progress.json"
            progress=json.loads(progresspath.read_text()) if progresspath.exists() else {"unix":start,"message":"starting"}
            append_event(out/"validation/heartbeats.jsonl",{**sample,"progress":progress["message"]})
            print(json.dumps({"stage":stage,"elapsed_s":round(sample["task_elapsed_seconds"]),"rss_GiB":round(sample["process_tree_rss_bytes"]/2**30,3),"disk_GiB":round(sample["added_disk_bytes"]/2**30,3),"progress":progress["message"]}),flush=True)
            budget=cfg["resources"]["base_wall_seconds"]-(0 if stage=="C7" else cfg["resources"]["sealing_reserve_seconds"])
            violation=None
            if sample["process_tree_rss_bytes"]>=cfg["resources"]["rss_limit_bytes"]:violation="Process-tree RSS guard"
            if sample["added_disk_bytes"]>=cfg["resources"]["added_disk_limit_bytes"]:violation="Added disk guard"
            if sample["task_elapsed_seconds"]>=budget:violation="Protected whole-task wall guard"
            if stage=="C0" and now-start>cfg["resources"]["c0_budget_seconds"]:violation="C0 two-hour guard"
            idle=now-progress["unix"]
            if idle>=cfg["resources"]["no_progress_diagnosis_seconds"]:
                (out/"validation/request_stack.txt").write_text("Real stack requested after no progress",encoding="utf-8")
                if diagnosed:violation="No progress for 15 minutes; real stack diagnosis requested and captured"
                diagnosed=True
            else:diagnosed=False
            if violation:
                for p in psutil.Process(process.pid).children(recursive=True):
                    try:p.terminate()
                    except psutil.Error:pass
                process.terminate();process.wait(timeout=30)
                write_json(out/"failures"/(stage+"_supervisor_"+str(time.time_ns())+".json"),{"stage":stage,"error":violation,"resource_state":sample,"progress":progress})
                raise RuntimeError(violation)
        time.sleep(.5)
    return process.returncode


def run(config_path,cfg,resume=False,until=None):
    out=Path(cfg["output_root"]);root=Path(cfg["repository_root"])
    if git(root,"branch","--show-current")!=cfg["branch"]:raise ValueError("Wrong branch")
    git(root,"merge-base","--is-ancestor",cfg["required_ancestor"],"HEAD")
    if resume:
        identity=json.loads((out/"run_identity.json").read_text())
        if identity["config_sha256"]!=sha(config_path) or identity["run_id"]!=cfg["run_id"]:
            raise ValueError("Cannot resume different run/config identity")
    else:
        if out.exists():raise FileExistsError("New output must be absent at first launch; use matching resume")
        out.mkdir(parents=True)
        write_json(out/"run_identity.json",{"run_id":cfg["run_id"],"campaign_id":cfg["campaign_id"],"config_sha256":sha(config_path),
            "authority_sha256":cfg["authority_zip_sha256"],"starting_head":git(root,"rev-parse","HEAD"),
            "task_started_unix":cfg["task_started_unix"],"first_scientific_launch_unix":time.time(),"initial_output_absent":True})
    (out/"validation").mkdir(exist_ok=True);(out/"logs").mkdir(exist_ok=True)
    for stage in STAGES:
        p=out/"stage_receipts"/(stage+".json")
        dep=dependencies(cfg,config_path,stage)
        if p.exists():
            verify_receipt(out,p,dep)
        else:
            command=[cfg["python_executable"],"-u","-m","partC_rev1_src.phase5.campaign_cli","worker","--config",str(config_path),"--stage",stage]
            append_event(out/"validation/command_log.jsonl",{"stage":stage,"command":command,"cwd":str(root)})
            env=os.environ.copy();env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_NUM_THREADS="1")
            with (out/"logs"/(stage+"_"+str(time.time_ns())+".log")).open("w",encoding="utf-8") as log:
                child=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,env=env)
                rc=monitor_child(cfg,stage,child)
            if rc:raise RuntimeError(stage+" failed; preserve outputs and inspect exact failure before engineering repair")
            verify_receipt(out,p,dep)
        if until==stage:break
    return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["run","resume","status","worker","validate","seal","repair"])
    parser.add_argument("--config",required=True)
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--stage",choices=STAGES)
    parser.add_argument("--until",choices=STAGES)
    parser.add_argument("--reason")
    args=parser.parse_args();path,cfg=load_config(args.config)
    if args.dry_run:
        print(json.dumps({"status":"EXPLICIT_DRY_RUN_NO_SCIENCE","run_id":cfg["run_id"],"tasks":cfg["tasks"],
            "expected_outputs":cfg["expected_outputs_file"],"input_versions":cfg["input_versions_file"],"acceptance_tests":cfg["acceptance_tests_file"],"budget":cfg["resources"]},indent=2));return 0
    if args.command=="worker":
        if not args.stage:parser.error("worker requires --stage")
        return worker(path,cfg,args.stage)
    if args.command=="repair":return engineering_repair(path,cfg,args.reason)
    if args.command=="status":
        out=Path(cfg["output_root"])
        print(json.dumps({"run_id":cfg["run_id"],"initialized":out.exists(),"pass_stages":[s for s in STAGES if (out/"stage_receipts"/(s+".json")).exists()]},indent=2));return 0
    if args.command=="validate":
        from .campaign_validation import validate_serialization
        print(json.dumps(validate_serialization(cfg),indent=2));return 0
    if args.command=="seal":
        from .campaign_handoff import seal_final
        print(json.dumps(seal_final(path,cfg),indent=2));return 0
    return run(path,cfg,resume=args.command=="resume",until=args.until)


if __name__=="__main__":
    raise SystemExit(main())
