"""Streaming external inventories and safe compact native-label witnesses."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import zipfile
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import sha,read_json,read_npz,write_json,write_npz,write_frame


def npz_schema(path):
    result={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0): shape,fortran,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0): shape,fortran,dtype=np.lib.format.read_array_header_2_0(stream)
                else: raise ValueError("Unreviewed NumPy header version")
            if dtype.hasobject: raise ValueError("Object/pickle arrays prohibited")
            result[name.removesuffix(".npy")]=dict(shape=list(shape),dtype=dtype.str,fortran_order=fortran)
    return result


def describe(path):
    if path.suffix==".npz":
        schema=npz_schema(path)
        return len(schema),json.dumps(schema,sort_keys=True,separators=(",",":"))
    if path.suffix==".parquet":
        with pq.ParquetFile(path) as f:return f.metadata.num_rows,str(f.schema_arrow)
    if path.suffix in {".json",".jsonl",".csv",".md",".txt",".xml",".log"}: return -1,"UTF8 "+path.suffix
    if path.suffix==".zip": return -1,"Authenticated authority archive; indexed only, no recursive embedding"
    return -1,"Non-scientific cache or auxiliary "+path.suffix


def inventory(root,*,subtrees=None,prefix="output",chunk_size=12000,workers=1,exclude=()):
    root=Path(root).resolve(); dest=root/"manifests"; dest.mkdir(parents=True,exist_ok=True)
    if workers not in {1,2,4}:raise ValueError("Bounded inventory reader count")
    if subtrees is None:
        files=sorted(p for p in root.rglob("*") if p.is_file() and not p.is_relative_to(dest)
                     and p.name not in {"writer.lock","driver.lock"} and ".pending." not in p.name
                     and p.relative_to(root).as_posix() not in exclude)
    else:
        files=sorted(p for subtree in subtrees for p in (root/subtree).rglob("*") if p.is_file())
    shards=[]; size=0; count=0
    def record(p):
        rows,schema=describe(p); stat=p.stat(); rel=p.relative_to(root).as_posix()
        digest=sha(p)
        if p.stat().st_size!=stat.st_size or p.stat().st_mtime_ns!=stat.st_mtime_ns:raise ValueError("Inventory file mutated: "+rel)
        return dict(relative_path=rel,absolute_path=str(p),size_bytes=stat.st_size,
            modified_unix_ns=stat.st_mtime_ns,sha256=digest,format=p.suffix,row_or_object_count=rows,
            schema_or_shape=schema,producer="partC_rev1_src.phase6 or authenticated unchanged source copy",
            consumer="phase6.preservation, explicit review, gated untrained data interface",
            copy_policy="EXTERNAL_FULL_HASH_BOUND",authoritative_scientific=not rel.startswith("cache/"))
    with ThreadPoolExecutor(max_workers=workers) if workers>1 else nullcontext(None) as pool:
        for start in range(0,len(files),chunk_size):
            subset=files[start:start+chunk_size]
            records=list(pool.map(record,subset)) if pool else [record(p) for p in subset]
            size+=sum(r["size_bytes"] for r in records);count+=len(records)
            target=dest/f"{prefix}_{len(shards):04d}.parquet"
            write_frame(target,pd.DataFrame(records)); shards.append(dict(path=str(target),sha256=sha(target),rows=len(records),bytes=target.stat().st_size))
            print("INVENTORY",count,"files",flush=True)
    result=dict(files=count,bytes=size,shards=shards,scope="all run outputs except self-referential manifests and transient writer lock",
                checksum_algorithm="SHA256",numeric_npz_headers_reopened=True,inventory_readers=workers,
                additional_mutable_seal_controls_excluded=list(exclude))
    write_json(dest/(prefix+"_INDEX.json"),result)
    return result


def verify_witness(example):
    """Reconstruct a compact path witness without needing unrelated catalog paths."""
    from .oracle import Oracle
    from . import kernels as K
    example=Path(example)
    graph=read_npz(example/"local_graph.npz")
    label=read_npz(example/"label_payload.npz"); paths=read_npz(example/"used_path_catalog.npz")
    receipt=read_json(example/"label_receipt.json")
    if not np.array_equal(label["path_ids"],paths["path_id"]): raise ValueError("Partial witness ID order differs")
    if sha(example/"label_payload.npz")!=receipt["payload_sha256"]: raise ValueError("Native witness label changed")
    o=Oracle(graph)
    newids=K.register_selected(o.table,o.paths,o.pairs,o.sources,o.targets,paths["pair"],paths["source"],paths["target"],paths["offsets"],paths["arcs"])
    h=np.bincount(newids,weights=label["weights"],minlength=len(o.paths))
    import time
    r=o.certify(label["demand"],h,0,time.perf_counter(),"EXTRACTED_WITNESS",0,o.lmo_count)
    if not np.allclose(label["z"],r.z,rtol=2e-11,atol=1e-10) or not np.allclose(label["tau"],r.tau,rtol=2e-11,atol=1e-8): raise ValueError("Extracted witness recertification differs")
    if not np.isclose(r.value_upper,receipt["scalars"]["value_upper"],rtol=2e-11,atol=1e-7): raise ValueError("Extracted witness potential differs")
    return dict(status="PASS",m_active=len(r.demand),quality=r.quality,relative_gap=r.relative_gap)
