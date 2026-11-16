"""External single-child watchdog; termination is limited to its own process tree."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

from .common import read_json, write_json


def terminate_tree(pid):
    try:
        p = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    children = p.children(recursive=True)
    members = children+[p]
    ids = [x.pid for x in members]
    for x in reversed(members):
        try:
            x.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(members, timeout=3)
    for x in alive:
        try:
            x.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(alive, timeout=3)
    return ids


def run_supervised(command, env, log, *, deadline, rss_cap, available_floor=3*1024**3,
                   on_start=None, sample=None):
    started = time.time()
    with Path(log).open("x", encoding="utf-8") as out:
        p = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT, env=env,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if on_start:
            on_start(p.pid)
        reason = "CHILD_EXIT"
        peak = 0
        samples = 0
        terminated = []
        while p.poll() is None:
            try:
                process = psutil.Process(p.pid)
                family = [process]+process.children(recursive=True)
                rss = sum(x.memory_info().rss for x in family if x.is_running())
                peak = max(peak, rss)
                available = psutil.virtual_memory().available
            except psutil.NoSuchProcess:
                continue
            now = time.time()
            if sample and samples % 10 == 0:
                sample(dict(epoch=now, child_pid=p.pid, process_tree_rss=rss,
                            system_available_memory=available, elapsed_seconds=now-started))
            samples += 1
            if now >= deadline:
                reason = "WATCHDOG_TIME_LIMIT"
            elif rss >= rss_cap or available < available_floor:
                reason = "WATCHDOG_RESOURCE_LIMIT"
            if reason != "CHILD_EXIT":
                terminated = terminate_tree(p.pid)
                break
            time.sleep(.25)
        p.wait(timeout=10)
    return dict(pid=p.pid, returncode=p.returncode, reason=reason, peak_process_tree_rss=peak,
                started_epoch=started, ended_epoch=time.time(), elapsed_seconds=time.time()-started,
                deadline_epoch=deadline, terminated_owned_pids=terminated,
                no_orphan_child=not psutil.pid_exists(p.pid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cap-seconds", required=True, type=float)
    parser.add_argument("--suite")
    parser.add_argument("--scenario")
    parser.add_argument("--micro", action="store_true")
    args = parser.parse_args()
    c = read_json(args.config)
    if args.run_id != c["run_id"] or args.cap_seconds <= 0:
        raise ValueError("Explicit valid original round/job identity required")
    root = Path(c["output_root"])
    lock = root/"NUMERIC_OWNER.json"
    if lock.exists():
        old = read_json(lock)
        if old.get("status") == "ACTIVE" and psutil.pid_exists(old["supervisor_pid"]):
            raise RuntimeError("Another controlled numeric job already owns this round")
        if old.get("status") == "ACTIVE" and psutil.pid_exists(old.get("child_pid", -1)):
            raise RuntimeError("Prior live child requires original-job reconciliation")
    job = root/"receipts"/(args.job_id+"_JOB.json")
    if job.exists():
        raise FileExistsError("Existing job receipt: preserve original clock; do not relaunch")
    start = time.time()
    deadline = min(start+args.cap_seconds, c["compute_cutoff_epoch"])
    if deadline-start <= 1:
        raise TimeoutError("No numerical admission budget")
    env = os.environ.copy()
    for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        env[k] = "1"
    for k in ["NUMBA_CACHE_DIR", "MPLCONFIGDIR", "XDG_CACHE_HOME"]:
        env[k] = str(root/"cache"/k.lower())
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = c["repository"]
    command = [c["python_executable"], "-B", "-m", "partC_rev1_src.phase_n8.cli",
               "--config", args.config, "--run-id", args.run_id, "--task", args.task,
               "--job-id", args.job_id, "--job-start", str(start), "--job-deadline", str(deadline)]
    if args.suite:
        command += ["--suite", args.suite]
    if args.scenario:
        command += ["--scenario", args.scenario]
    if args.micro:
        command += ["--micro"]
    payload = dict(job_id=args.job_id, task=args.task, run_id=args.run_id, cap_seconds=args.cap_seconds,
                   start_epoch=start, deadline_epoch=deadline, original_R2_start=c["start_epoch"],
                   command=command, environment={k:env[k] for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS", "NUMBA_CACHE_DIR"]},
                   suite=args.suite, scenario=args.scenario, micro=args.micro,
                   supervisor_pid=os.getpid(), status="ACTIVE", workers=1)
    write_json(job, payload)
    write_json(lock, payload)
    def began(pid):
        payload["child_pid"] = pid
        write_json(job, payload)
        write_json(lock, payload)
    def sampled(row):
        with (root/"logs"/(args.job_id+"_resources.jsonl")).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row)+"\n")
    result = run_supervised(command, env, root/"logs"/(args.job_id+".log"), deadline=deadline,
                            rss_cap=min(10*1024**3, c["max_process_tree_RSS_bytes"]), on_start=began, sample=sampled)
    payload.update(status="FINISHED", result=result)
    write_json(job, payload)
    write_json(lock, payload)
    print(json.dumps(payload, indent=2))
    return result["returncode"]


if __name__ == "__main__":
    raise SystemExit(main())
