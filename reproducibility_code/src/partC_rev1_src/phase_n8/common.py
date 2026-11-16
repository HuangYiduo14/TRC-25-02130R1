"""Explicit original-clock context and authenticated, atomic owned outputs."""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import time
import uuid

import numpy as np
import psutil


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.asarray(value)
        if a.dtype.kind == "O":
            raise ValueError("Object arrays cannot define native identity")
        h.update(canonical([key, a.dtype.str, list(a.shape)]))
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_bytes(path, content):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".pending." + uuid.uuid4().hex[:12])
    with tmp.open("xb") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())
    for attempt in range(20):
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.025 * (attempt + 1))


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, indent=2, sort_keys=True,
                                  allow_nan=False).encode() + b"\n")


def save_npz(path, arrays):
    if any(np.asarray(v).dtype.kind == "O" for v in arrays.values()):
        raise ValueError("Pickle/object payload prohibited")
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    atomic_bytes(path, b.getvalue())


def read_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


class ResourceLimit(RuntimeError):
    pass


class Context:
    def __init__(self, config, step, run_id):
        self.config_path = Path(config).resolve(strict=True)
        self.c = read_json(self.config_path)
        c = self.c
        required = ("phase_id", "step_id", "run_id", "tasks", "input_versions",
                    "expected_outputs", "acceptance_tests", "start_epoch")
        if any(not c.get(k) for k in required):
            raise ValueError("Explicit complete execution contract required")
        if c["phase_id"] != "N8" or c["step_id"] != "N8R2" or c["run_id"] != run_id:
            raise ValueError("Wrong explicit phase/step/run")
        if c["deadline_epoch"] != c["start_epoch"] + 28800:
            raise ValueError("Original R2 deadline changed")
        self.repo = Path(c["repository"]).resolve(strict=True)
        self.root = Path(c["output_root"]).resolve(strict=True)
        if self.root == self.repo or self.repo in self.root.parents:
            raise ValueError("Bulk results must be external")
        if step not in {"INTAKE", "BUILD", "NATIVE_TESTS", "PILOT", "TIMING", "REPORT", "VERIFY"}:
            raise ValueError("Unknown explicit step")
        self.step = step
        self.started = time.time()
        self.monotonic_started = time.monotonic()
        self.bindings = {}
        self.peak_rss = 0
        self.last_resource = 0
        self.guard()

    def guard(self, deadline=None):
        cutoff = self.c["deadline_epoch"] if self.step in {"REPORT", "VERIFY"} else self.c["compute_cutoff_epoch"]
        if deadline is not None:
            cutoff = min(cutoff, deadline)
        now = max(time.time(), self.started + time.monotonic() - self.monotonic_started)
        if now >= cutoff:
            raise TimeoutError("Original task/run deadline or reporting reserve reached")
        p = psutil.Process()
        rss = p.memory_info().rss
        for child in p.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        self.peak_rss = max(self.peak_rss, rss)
        # Current available memory requires a substantially stricter local ceiling.
        cap = min(self.c["max_process_tree_RSS_bytes"], 10 * 1024**3)
        available = psutil.virtual_memory().available
        if rss >= cap or available < 3 * 1024**3:
            raise ResourceLimit("RSS/free-system-memory headroom guard")
        if now - self.last_resource >= 15:
            added = sum(x.stat().st_size for x in self.root.rglob("*") if x.is_file())
            if added >= self.c["R2_added_disk_bytes"]:
                raise ResourceLimit("R2 added-disk guard")
            record = dict(epoch=now, elapsed_since_R2_start=now-self.c["start_epoch"],
                          pid=os.getpid(), rss_bytes=rss, system_available_memory=available,
                          disk_added_bytes=added, cpu_percent=p.cpu_percent(),
                          gpu_memory_bytes=None, current_task=self.step)
            with (self.root/"logs/resource_usage.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            self.last_resource = now

    def bind(self, path, role, expected=None):
        p = Path(path).resolve(strict=True)
        key = str(p)
        if key not in self.bindings:
            st = p.stat()
            self.bindings[key] = dict(native_path=key, role=role, sha256=sha(p),
                                      size_bytes=st.st_size, mtime_ns=st.st_mtime_ns)
        r = self.bindings[key]
        if expected is not None and r["sha256"] != expected:
            raise ValueError("Immutable source hash mismatch: " + key)
        return r["sha256"]

    def finish(self, name, result):
        self.guard()
        for r in self.bindings.values():
            p = Path(r["native_path"])
            if p.stat().st_mtime_ns != r["mtime_ns"] and sha(p) != r["sha256"]:
                raise ValueError("Source changed while in use: " + str(p))
        result.update(step=self.step, started_epoch=self.started, ended_epoch=time.time(),
                      elapsed_seconds=time.time()-self.started, peak_rss_bytes=self.peak_rss,
                      source_mutations=0)
        write_json(self.root/f"receipts/{name}_RESULT.json", result)
        write_json(self.root/f"receipts/{name}_SOURCE_BINDINGS.json", list(self.bindings.values()))
        print(json.dumps(result, allow_nan=False), flush=True)


def code_identity(repo):
    root = Path(repo)
    paths = [root/"partC_rev1_src/phase_n8"/s for s in
             ["common.py", "native.py", "objective.py", "oracle.py", "solver.py", "cli.py", "supervise.py"]]
    paths += [root/"partC_rev1_src"/s for s in
              ["phase5/campaign_runtime.py", "phase5/scc_reachability.py", "phase67/portable.py", "phase7/portable.py", "phase4/kernels.py"]]
    records = {str(p.relative_to(root)).replace("\\", "/"): sha(p) for p in paths}
    return digest(records), records
