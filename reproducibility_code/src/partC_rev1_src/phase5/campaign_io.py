"""Campaign identity, immutable receipts and compact numerical serialization."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(json_safe(value), sort_keys=True, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    os.replace(tmp,path)


def frame_write(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    if path.suffix == ".parquet":
        frame.to_parquet(tmp, index=False, compression="zstd")
    else:
        frame.to_csv(tmp, index=False)
    os.replace(tmp,path)


def append_event(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a",encoding="utf-8") as f:
        f.write(json.dumps(json_safe({"unix":time.time(), **value}), sort_keys=True)+"\n")


def git(root,*args):
    return subprocess.check_output(["git","-C",str(root),"-c","core.longpaths=true",*args],text=True).strip()


def code_binding(root):
    paths = sorted((root/"partC_rev1_src/phase5").glob("*.py"))
    paths += sorted((root/"tests").glob("test_phase5*.py"))
    return {p.relative_to(root).as_posix():sha(p) for p in paths}


def binding_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",", ":")).encode()).hexdigest()


def file_record(root,path):
    return {"relative_path":Path(path).relative_to(root).as_posix(),
        "size_bytes":Path(path).stat().st_size,"sha256":sha(path)}


def verify_receipt(root,path,dependency_hash=None):
    receipt = json.loads(Path(path).read_text(encoding="utf-8"))
    if receipt["status"] != "PASS" or not receipt["artifacts"]:
        raise ValueError("Incomplete/no-op stage receipt")
    if dependency_hash is not None and receipt["dependency_hash"] != dependency_hash:
        raise ValueError("Changed dependency requires evidenced engineering invalidation")
    for r in receipt["artifacts"]:
        p = (root/r["relative_path"]).resolve()
        if not p.is_relative_to(root.resolve()) or p.stat().st_size != r["size_bytes"] or sha(p)!=r["sha256"]:
            raise ValueError("Stage artifact changed: "+str(p))
    return receipt


def publish_receipt(root,path,stage,dependency_hash,paths,started,**extra):
    if path.exists():
        raise FileExistsError("PASS receipts are immutable: "+str(path))
    records = [file_record(root,p) for p in paths]
    if not records:
        raise ValueError("No-op stage is a failure")
    write_json(path,{"stage_id":stage,"status":"PASS","dependency_hash":dependency_hash,
        "started_unix":started,"ended_unix":time.time(),"elapsed_seconds":time.time()-started,
        "artifacts":records,**extra})


def npz_write(path, arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    if any(np.asarray(v).dtype.kind == "O" for v in arrays.values()):
        raise ValueError("Object/pickle arrays prohibited")
    with tmp.open("wb") as f:
        np.savez_compressed(f,**arrays)
    os.replace(tmp,path)


def array_schema(path, arrays, link_ids_hash, node_ids_hash):
    return {"format":"NPZ","allow_pickle":False,"float_precision":"float64",
        "byte_order":"little-endian numeric Windows host","path":str(path),"sha256":sha(path),
        "link_id_order_sha256":link_ids_hash,"node_id_order_sha256":node_ids_hash,
        "units":{"time":"seconds","flow":"accepted assignment units (no relabeling)","potential":"flow-seconds"},
        "arrays":{n:{"shape":list(a.shape),"dtype":a.dtype.str} for n,a in arrays.items()}}
