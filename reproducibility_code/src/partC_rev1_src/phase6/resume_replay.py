"""One explicit engineering-fix replay, never a blanket regional retry sweep."""
from pathlib import Path
from .common import read_json,write_json,sha,git,can_admit,event
from .resume_policy import ledger,QUARANTINED,READY,SharedIntegrityError,LocalRegionError


def targeted_replay(c,rid,fix_record):
    if not rid or not fix_record:raise ValueError("Exact region and tested fix record required")
    root=Path(c["output_root"]);book=ledger(str(root));fix=read_json(fix_record)
    if rid not in book.failed:raise ValueError("Only an explicitly quarantined region may be replayed")
    incident=Path(book.states[rid]["incident_path"])
    if fix.get("region_id")!=rid or fix.get("incident_sha256")!=sha(incident):raise ValueError("Fix does not bind this failure")
    if fix.get("tested_commit")!=git(c["repository_root"],"rev-parse","HEAD"):raise ValueError("Fix checkpoint differs")
    if not fix.get("proved_cause") or not fix.get("tests_passed") or not 4<=fix.get("engineering_epoch",99)<=6:
        raise ValueError("Proved cause, passing tests and remaining engineering epoch required")
    for r in fix.get("test_artifacts",[]):
        if sha(r["path"])!=r["sha256"]:raise ValueError("Fix test evidence changed")
    if not fix.get("test_artifacts"):raise ValueError("Machine-readable test evidence required")
    replay=root/"failures"/rid.removeprefix("partC_p2_region_")/"targeted_replay.json"
    if replay.exists():raise ValueError("The single targeted replay has already been used")
    if not can_admit(c,1200):raise ValueError("Original campaign has no targeted replay budget")
    from .worker import baseline_region
    from .validation import baseline_audit
    from .preservation import reopen_existing_region
    event(root/"failures/repair_events.jsonl",dict(region_id=rid,fix_path=str(fix_record),fix_sha256=sha(fix_record),status="TARGETED_REPLAY_STARTED"))
    try:
        baseline=baseline_region(c,rid)
        if baseline["status"]!="BASELINE_LABELS_COMPLETE":raise LocalRegionError("TARGETED_REPLAY_STILL_INCOMPLETE")
        audit=baseline_audit(c,rid);reopened,_=reopen_existing_region(c,rid)
    except Exception as exc:
        write_json(replay,dict(status="QUARANTINE_RETAINED",error=repr(exc),fix_sha256=sha(fix_record)))
        raise
    result=dict(status="COMPLETE_LOCAL_REVALIDATION_PASS",baseline=baseline,audit=audit,reopened=reopened,fix_sha256=sha(fix_record))
    write_json(replay,result);book.set(rid,READY,"ONE_PROVED_FIX_REPLAY_AND_ALL_LOCAL_GATES_PASS",repair_record=str(replay),all_required_revalidated=True)
    return result
