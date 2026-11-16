"""Explicit, guarded Phase 6 stage entrypoint; stages never default to science."""
from __future__ import annotations

import argparse
import faulthandler
import os
from pathlib import Path
import time
import traceback

from .common import load_config, write_json, event, git, sha, digest
from .stack_sampler import StackSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=["registry", "low-plan", "source-audit", "high-plan", "startup", "baseline", "blocked-audit", "preserve",
        "resume","resume-witness","resume-reconcile","resume-baseline","resume-audits","resume-enrichment","resume-reopen","resume-seal","resume-replay"])
    parser.add_argument("--region")
    parser.add_argument("--fix-record")
    args = parser.parse_args()
    c = load_config(args.config); out = Path(c["output_root"])
    repo = Path(c["repository_root"])
    if git(repo, "branch", "--show-current") != c["branch"]: raise ValueError("Wrong branch")
    if git(repo, "status", "--porcelain", "--", "partC_rev1_src/phase6", "tests/test_phase6*.py"):
        raise ValueError("Scientific implementation must be checkpoint-committed before dispatch")
    c["effective_config_path"]=str(Path(args.config).resolve())
    if args.stage=="resume":
        from .resume_driver import run
        print(run(c,args.config),flush=True);return
    if args.stage=="resume-seal":
        from .resume_handoff import finish
        print(finish(c,args.config),flush=True);return
    lock = out / "control/writer.lock"
    with lock.open("x", encoding="utf-8") as f: f.write(str(os.getpid()))
    started = time.time()
    identity = {"stage": args.stage, "pid": os.getpid(), "started_unix": started,
                "original_task_started_unix": c["task_started_unix"], "code_head": git(repo, "rev-parse", "HEAD"),
                "config_sha256": sha(args.config), "run_id": c["run_id"]}
    c["execution_head"]=identity["code_head"]
    c["execution_config_sha256"]=identity["config_sha256"]
    log = out / "control" / f"{args.stage}_{int(started)}_stacks.log"
    try:
        with log.open("w", encoding="utf-8") as stack,StackSampler(stack):
            faulthandler.enable(stack)
            event(out / "control/stage_history.jsonl", {**identity, "status": "STARTED"})
            if args.stage.startswith("resume-"):
                from . import resume_campaign as continuation
                dispatch={"resume-witness":continuation.known_witness,"resume-reconcile":continuation.reconcile,
                    "resume-baseline":continuation.remaining_baseline,"resume-audits":continuation.healthy_gate,
                    "resume-reopen":continuation.cumulative_reopen}
                if args.stage=="resume-enrichment":
                    from .enrichment import run_enrichment
                    result=run_enrichment(c)
                elif args.stage=="resume-replay":
                    from .resume_replay import targeted_replay
                    result=targeted_replay(c,args.region,args.fix_record)
                else:result=dispatch[args.stage](c)
            elif args.stage == "registry":
                from .registry import build
                result = build(c)
            elif args.stage == "low-plan":
                from .sampling import prepare_all_low
                result = prepare_all_low(c)
            elif args.stage == "source-audit":
                from .source_audit import audit
                result = audit(c)
            elif args.stage == "high-plan":
                from .projection import project_high
                result = project_high(c)
            elif args.stage == "startup":
                from .campaign import startup
                result = startup(c)
            elif args.stage == "baseline":
                from .campaign import baseline_campaign
                result = baseline_campaign(c)
            elif args.stage == "blocked-audit":
                from .blocking_audit import run_blocking_audit
                result = run_blocking_audit(c)
            elif args.stage == "preserve":
                from .preservation import preserve
                result = preserve(c)
            receipt_name=f"{args.stage}_PASS_{int(started)}.json" if args.stage.startswith("resume-") else f"{args.stage}_PASS.json"
            write_json(out / "control" / receipt_name, {**identity,
                "status": "PASS", "ended_unix": time.time(), "result": result})
            event(out / "control/stage_history.jsonl", {**identity, "status": "PASS", "result": result})
            write_json(out / "control/status.json", {**identity,"status":"EXECUTING",
                "last_completed_stage":args.stage,"last_result":result,"updated_unix":time.time()},new=False)
            print(result, flush=True)
    except BaseException as exc:
        event(out / "control/stage_history.jsonl", {**identity, "status": "FAILED", "error": repr(exc)})
        write_json(out / "control" / f"{args.stage}_failure_{int(started)}.json", {**identity,
            "status": "FAILED", "error": repr(exc), "traceback": traceback.format_exc()})
        raise
    finally:
        faulthandler.disable()
        lock.unlink(missing_ok=True)


if __name__ == "__main__": main()
