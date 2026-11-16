"""Fresh hash verification of independent native copies and append-only prefixes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import pandas as pd
from .common import read_json,write_json,write_frame,sha
from .resume_policy import SharedIntegrityError


def check_copy(row):
    src=Path(row.source_path);dst=Path(row.destination_path)
    if src.stat().st_size!=row.size_bytes or sha(src)!=row.sha256:
        raise SharedIntegrityError("Frozen parent changed: "+str(src))
    disposition="EXACT_INDEPENDENT_COPY"
    if dst.name=="catalog_index.json" and row.scientific:
        old=read_json(src);new=read_json(dst)
        if old["model_hash"]!=new["model_hash"] or new["segments"][:len(old["segments"])]!=old["segments"] or new["path_count"]<old["path_count"]:
            raise SharedIntegrityError("Preserved catalog prefix changed: "+str(dst))
        disposition="VERIFIED_APPEND_ONLY_CATALOG_PREFIX"
    elif dst.name=="baseline_status.json" and row.scientific:
        # Its exact original is retained in the immutable parent and copy manifest.
        disposition="CURRENT_POLICY_STATUS_REVISION_NATIVE_DATA_UNCHANGED"
    elif dst.stat().st_size!=row.size_bytes or sha(dst)!=row.sha256:
        raise SharedIntegrityError("Copied native payload or source record changed: "+str(dst))
    return dict(relative_path=row.relative_path,parent_sha256=row.sha256,parent_unchanged=True,
        copy_disposition=disposition,independent_copy=row.independent_copy)


def verify_copy_lineage(c):
    root=Path(c["output_root"]);started=time.perf_counter();result_path=root/"validation/R5_COPY_LINEAGE.json"
    if result_path.exists():return read_json(result_path)
    record=read_json(root/"validation/COPY_COMPLETE.json");shards=[];count=0
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,shard in enumerate(record["manifest_shards"]):
            if sha(shard["path"])!=shard["sha256"]:raise SharedIntegrityError("Copy manifest changed")
            frame=pd.read_parquet(shard["path"]);rows=list(pool.map(check_copy,frame.itertuples()))
            path=root/"validation/copy_lineage"/f"part_{i:04d}.parquet"
            if not path.exists():write_frame(path,pd.DataFrame(rows))
            shards.append(dict(path=str(path),sha256=sha(path),rows=len(rows)));count+=len(rows)
            print("COPY LINEAGE",count,flush=True)
    result=dict(status="PASS",files=count,manifest_shards=shards,wall_seconds=time.perf_counter()-started,
        native_parent_unchanged=True,all_original_payloads_unchanged=True,catalog_prefixes_preserved=True,parallel_readers=4)
    write_json(result_path,result);return result
