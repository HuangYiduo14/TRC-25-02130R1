"""Authenticated continuation segments preserve every earlier clock and history.

The numerical solver writes only into the new segment. A continuation never
overlays an earlier run or extends its deadline. Campaign promotion is separate
and is intentionally not implied by this function.
"""
from pathlib import Path
import time

from .common import read_json, write_json, sha
from .solver import reopen_checkpoint, solve


def resume_segment(graph, od, objective, balance, previous_root, segment_root, *,
                   identity, clocks, ctx, tol=.001):
    previous_root = Path(previous_root).resolve(strict=True)
    segment_root = Path(segment_root).resolve()
    if segment_root.exists():
        raise FileExistsError("A restart requires a new, exclusive continuation segment")
    if time.time() >= clocks["run_deadline_epoch"]:
        raise TimeoutError("Original run deadline has expired; restart cannot reset it")
    ctx.guard(deadline=clocks["run_deadline_epoch"])
    _, meta = reopen_checkpoint(previous_root, identity, clocks)
    if tol != identity.get("tolerance", tol):
        raise ValueError("Restart tolerance differs from the authenticated run")
    inherited_history = read_json(previous_root/"HISTORY.json") if (previous_root/"HISTORY.json").exists() else []
    segment_root.mkdir(parents=True, exist_ok=False)
    write_json(segment_root/"RESTART_PROVENANCE.json", dict(
        previous_root=str(previous_root), previous_pointer_sha256=sha(previous_root/"LATEST_CHECKPOINT.json"),
        previous_history_sha256=sha(previous_root/"HISTORY.json") if (previous_root/"HISTORY.json").exists() else None,
        identity=identity, clocks=clocks, inherited_iteration=meta["accepted_iteration"],
        restart_module_sha256=sha(__file__), original_clock_preserved=True,
        deadline_extension_authorized=False))
    write_json(segment_root/"LATEST_CHECKPOINT.json", read_json(previous_root/"LATEST_CHECKPOINT.json"))
    result = solve(graph, od, objective, balance, segment_root, identity=identity,
                   clocks=clocks, ctx=ctx, tol=tol, resume=True)
    new_history = read_json(segment_root/"HISTORY.json")
    write_json(segment_root/"CUMULATIVE_HISTORY.json", inherited_history+new_history)
    return result
