"""Phase 6 explicit identities, atomic artifacts, and original-clock resources."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import numpy as np
import psutil

from . import MODEL_CLASS

WRITE_LOCK = threading.RLock()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def arrays_hash(arrays):
    h = hashlib.sha256()
    for k, v in sorted(arrays.items()):
        a = np.asarray(v)
        if a.dtype.kind == "O":
            raise ValueError("Object arrays are not authoritative")
        h.update(canonical([k, a.dtype.str, list(a.shape)]))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def input_hash(model_hash, support_hash, x):
    a = np.asarray(x, dtype="<f8")
    if a.ndim != 1 or not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("Invalid native input")
    # Signed zero is mathematically the same demand. Planners emit positive zero.
    a = a.copy(); a[a == 0] = 0.
    h = hashlib.sha256(canonical([model_hash, support_hash, len(a)]))
    h.update(a.tobytes())
    return h.hexdigest()


def _atomic_bytes(path, data, *, new=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if new and path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + f".pending.{os.getpid()}")
    with tmp.open("wb") as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    for attempt in range(12):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 11: raise
            time.sleep(.05 * (attempt + 1))


def atomic_bytes(path, data, *, new=True):
    with WRITE_LOCK:
        return _atomic_bytes(path, data, new=new)


def write_json(path, value, *, new=True):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                ensure_ascii=False, allow_nan=False).encode() + b"\n", new=new)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_npz(path, **arrays):
    import io
    if any(np.asarray(v).dtype.kind == "O" for v in arrays.values()):
        raise ValueError("Pickle arrays prohibited")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    atomic_bytes(path, stream.getvalue())


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def write_frame(path, frame):
    import io
    stream = io.BytesIO()
    frame.to_parquet(stream, index=False, compression="zstd")
    atomic_bytes(path, stream.getvalue())


def event(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(canonical({"unix": time.time(), **value}).decode() + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), "-c", "core.longpaths=true", *args],
                                   text=True, encoding="utf-8").strip()


def load_config(path):
    c = read_json(path)
    required = ["phase_id", "step_id", "run_id", "tasks", "input_versions", "resources",
                "expected_outputs_document", "acceptance_tests_config", "task_started_unix"]
    if any(not c.get(k) for k in required):
        raise ValueError("Every execution requires explicit identity, tasks, inputs, outputs, tests and budget")
    if c["model_class"] != MODEL_CLASS or c["suites"] != ["s03"]:
        raise ValueError("Wrong model or suite")
    if c["resources"]["hard_whole_campaign_seconds"] != 86400:
        raise ValueError("Original 24-hour clock required")
    if c.get("validation_policy_id") == "CONDITIONAL_MAPPED_COMPARATOR_V1":
        if (c["task_started_unix"],c["hard_deadline_unix"],c["science_deadline_unix"]) != (1788751753,1788838153,1788832753):
            raise ValueError("Resume authority does not reset the original clock")
    for k in ["network", "package_install", "icnn_training", "icnn_instantiation", "national_assignment",
              "B_restoration", "fine_R_expansion", "inactive_pair_fallback", "other_suites"]:
        if c["permissions"][k]: raise ValueError("Prohibited permission: " + k)
    return c


def elapsed(c):
    return time.time() - c["task_started_unix"]


def reserve(c, finalization_forecast=3600):
    return max(c["resources"]["sealing_reserve_seconds"], 1.5 * finalization_forecast)


def can_admit(c, seconds, used_bytes=0, new_bytes=0, outstanding_bytes=0,
              finalization_forecast=3600):
    vals = [seconds, used_bytes, new_bytes, outstanding_bytes, finalization_forecast]
    if any(not np.isfinite(v) or v < 0 for v in vals): raise ValueError("Invalid forecast")
    return (elapsed(c) + 2 * seconds < 86400 - reserve(c, finalization_forecast)
            and used_bytes + new_bytes + outstanding_bytes <= c["resources"]["added_disk_bytes"] - 2 * 2**30)


def tree_bytes(root):
    total = 0
    for folder, _, files in os.walk(root):
        for name in files:
            try: total += (Path(folder) / name).stat().st_size
            except FileNotFoundError: pass
    return total


class Monitor:
    def __init__(self, c, stage):
        self.c = c; self.stage = stage; self.root = Path(c["output_root"])
        self.stop_event = threading.Event(); self.failure = None
        self.last = {}; self.thread = None

    def sample(self):
        proc = psutil.Process()
        processes = [proc] + proc.children(recursive=True)
        rss = 0
        for p in processes:
            try: rss += p.memory_info().rss
            except psutil.NoSuchProcess: pass
        continuation_size = tree_bytes(self.root)
        prior_size = self.c.get("parent_added_bytes_at_adoption",0)
        size = continuation_size + prior_size
        record = {"stage": self.stage, "pid": os.getpid(), "elapsed_seconds": elapsed(self.c),
                  "process_tree_rss_bytes": rss, "added_output_bytes": size,
                  "continuation_output_bytes":continuation_size,"preserved_prior_campaign_bytes":prior_size,
                  "free_disk_bytes": shutil.disk_usage(self.root).free,
                  "hard_deadline_unix": self.c["task_started_unix"] + 86400}
        self.last = record
        event(self.root / "control/resource_history.jsonl", record)
        if rss >= self.c["resources"]["rss_bytes"] or size >= self.c["resources"]["added_disk_bytes"]:
            self.failure = "RESOURCE_LIMIT"
        if record["free_disk_bytes"] < 2 * 2**30: self.failure = "DISK_RESERVE"
        if elapsed(self.c) >= 86400: self.failure = "WHOLE_TASK_DEADLINE"
        if self.failure:
            write_json(self.root / "control/STOP_SCIENCE.json", {"reason": self.failure}, new=False)

    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop_event.wait(30):
                try: self.sample()
                except Exception as e:
                    self.failure = repr(e)
        self.thread = threading.Thread(target=loop, daemon=True); self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set(); self.thread.join(2); self.sample()

    def check(self):
        if self.failure:
            if self.c.get("validation_policy_id")=="CONDITIONAL_MAPPED_COMPARATOR_V1":
                from .resume_policy import ResourceStop,SharedIntegrityError
                if self.failure in {"RESOURCE_LIMIT","DISK_RESERVE","WHOLE_TASK_DEADLINE"}:raise ResourceStop(self.failure)
                raise SharedIntegrityError("Resource monitor failed: "+self.failure)
            raise RuntimeError(self.failure)
        if self.c.get("validation_policy_id")=="CONDITIONAL_MAPPED_COMPARATOR_V1":
            # Resume dispatch and sample admission reserve finalization explicitly;
            # the final audit itself must be allowed to use that protected time.
            return
        if elapsed(self.c) >= 86400 - reserve(self.c):
            raise TimeoutError("Protected finalization reserve")
