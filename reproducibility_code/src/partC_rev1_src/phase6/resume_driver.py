"""One supervised continuation, separate fresh-process stages, and one cumulative seal."""
from pathlib import Path
import os
import subprocess
import time
import json
from .stack_sampler import StackSampler

from .common import read_json,write_json,event,sha,git,Monitor
from .resume_policy import POLICY

STAGES=[("resume-reconcile","validation/R2_RECONCILE_RESULT.json"),
        ("resume-baseline","validation/R3_BASELINE_RESULT.json"),
        ("resume-audits","validation/R4_RESULT.json"),
        ("resume-enrichment","samples/ENRICHMENT_RESULT.json")]


def next_stages(c,exists,read,now):
    """Pure routing used by the actual supervisor, including partial and expired cases."""
    stages=[]
    if now<c["science_deadline_unix"]:
        for stage,receipt in STAGES:
            if not exists(receipt):stages.append(stage)
    if not exists("validation/R5_REOPEN_RESULT.json"):stages.append("resume-reopen")
    stages.append("resume-seal")
    return stages


def child(c,config_path,stage):
    root=Path(c["output_root"]);stamp=time.time_ns();path=root/"control"/(stage+"_"+str(stamp)+".log")
    command=[c["python_executable"],"-X","utf8","-m","partC_rev1_src.phase6.run","--config",str(config_path),"--stage",stage]
    environment=dict(os.environ)
    environment.update(NUMBA_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
        NUMBA_CACHE_DIR=str(root/"cache/numba"),PYTHONUNBUFFERED="1")
    with path.open("x",encoding="utf-8") as log:
        proc=subprocess.Popen(command,cwd=c["repository_root"],env=environment,stdout=log,stderr=subprocess.STDOUT)
        info=dict(stage=stage,pid=proc.pid,command=command,log_path=str(path),started_unix=time.time(),run_id=c["run_id"])
        write_json(root/"control/active_process.json",info,new=False)
        event(root/"control/driver_events.jsonl",dict(**info,status="STARTED"))
        print("START",stage,"PID",proc.pid,flush=True)
        try:
            while proc.poll() is None:
                time.sleep(1)
                if time.time()>c["hard_deadline_unix"]:
                    write_json(root/"control/STOP_SCIENCE.json",dict(reason="ORIGINAL_WHOLE_CAMPAIGN_DEADLINE",stage=stage),new=False)
        except BaseException:
            write_json(root/"control/STOP_SCIENCE.json",dict(reason="OPERATOR_OR_SUPERVISOR_STOP",stage=stage),new=False)
            # Do not kill unrelated processes or restart a worker merely because a poll timed out.
            proc.wait(timeout=305)
            raise
    code=proc.returncode
    event(root/"control/driver_events.jsonl",dict(**info,status="FINISHED",exit_code=code,ended_unix=time.time()))
    write_json(root/"control/active_process.json",dict(**info,status="TERMINAL",exit_code=code),new=False)
    print("END",stage,"EXIT",code,flush=True)
    return code


def run(c,config_path,*,run_child=child,seal_function=None):
    root=Path(c["output_root"])
    if c.get("validation_policy_id")!=POLICY:raise ValueError("Explicit current resume policy required")
    if (root/"manifests/RESUME_HANDOFF_SEAL_RESULT.json").exists():
        return read_json(root/"manifests/RESUME_HANDOFF_SEAL_RESULT.json")
    if (root/"control/writer.lock").exists():raise ValueError("A stage writer already owns this run")
    lock=root/"control/driver.lock"
    with lock.open("x") as f:f.write(str(os.getpid()))
    c["effective_config_path"]=str(Path(config_path).resolve())
    try:
        if not (root/"validation/review_resolution.json").exists():
            if run_child(c,config_path,"resume-witness"):
                raise ValueError("Resume witness failed before dispatch")
        resolution=read_json(root/"validation/review_resolution.json")
        for name,bound in resolution["implementation_sha256"].items():
            if sha(Path(__file__).with_name(name))!=bound:raise ValueError("Revalidate changed R1 dependency: "+name)
        stages=next_stages(c,lambda p:(root/p).exists(),lambda p:read_json(root/p),time.time())
        with (root/"control/driver_stacks.log").open("a") as stack,Monitor(c,"RESUME_SUPERVISOR"),StackSampler(stack):
            try:
                for stage in stages:
                    if stage=="resume-seal":break
                    if stage=="resume-enrichment":
                        p=root/"validation/BASELINE_SCOPE_READINESS.json"
                        if not p.exists() or not read_json(p).get("healthy_scope_complete"):
                            event(root/"control/driver_events.jsonl",dict(stage=stage,status="NOT_DISPATCHED",reason="HEALTHY_BASELINE_NOT_READY"));continue
                    if stage!="resume-reopen" and ((root/"control/STOP_SCIENCE.json").exists() or time.time()>=c["science_deadline_unix"]):
                        event(root/"control/driver_events.jsonl",dict(stage=stage,status="NOT_DISPATCHED",reason="CONTROLLED_STOP_OR_ORIGINAL_CUTOFF"));continue
                    code=run_child(c,config_path,stage)
                    if code and stage=="resume-reopen":break
                    # A stopped stage is preserved. The next science stage is skipped;
                    # fresh reopening and final handoff still run without a new approval.
                    if code and not (root/"control/STOP_SCIENCE.json").exists():
                        write_json(root/"control/STOP_SCIENCE.json",dict(stage=stage,reason="STAGE_FAILED_WITH_UNKNOWN_SCOPE",exit_code=code),new=False)
                    if code and stage!="resume-reopen":
                        stop=read_json(root/"control/STOP_SCIENCE.json")
                        epochs=read_json(root/"control/engineering_epochs.json")
                        if stop.get("error_type") not in {"ResourceStop","FailureThreshold"} and epochs["used"]<6 and time.time()<c["science_deadline_unix"]-1200:
                            write_json(root/"control/ENGINEERING_ATTENTION.json",dict(stage=stage,stop_sha256=sha(root/"control/STOP_SCIENCE.json"),
                                engineering_epochs_used=epochs["used"],status="BOUNDED_REPAIR_BEFORE_CONTINUATION",original_clock_unchanged=True),new=False)
                            raise RuntimeError("Bounded engineering triage required; preserve this process result and test/commit any repair before explicit stage continuation")
            finally:pass
        # The supervisor heartbeat is closed before the immutable output inventory.
        event(root/"control/driver_events.jsonl",dict(stage="resume-seal",status="FINALIZATION_START",ended_unix=time.time()))
        if seal_function is None:
            from .resume_handoff import finish
            seal_function=finish
        result=seal_function(c,config_path)
        return result
    finally:lock.unlink(missing_ok=True)
