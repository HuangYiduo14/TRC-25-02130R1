"""One write-new blocked-result handoff, with complete external artifact evidence."""
from pathlib import Path
import argparse
import ast
import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import os
import shutil
import subprocess
import time
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import psutil

from .common import load_config,read_json,read_npz,write_json,write_npz,write_frame,sha,git,tree_bytes
from .registry import region_dir
from .storage import Store
from .artifacts import inventory,npz_schema,verify_witness


def text_write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): raise FileExistsError(path)
    path.write_text(value,encoding="utf-8")


def prepare(c):
    root=Path(c["output_root"]); repo=Path(c["repository_root"]); work=Path(c["work_records_root"])
    preserved=read_json(root/"validation/preservation/PRESERVATION_PASS.json")
    blocked=read_json(root/"validation/BLOCKING_DIAGNOSIS.json")
    registry=read_json(root/"registry/REGISTRY_COMPLETE.json")
    baseline=read_json(root/"samples/BASELINE_PLAN_FROZEN.json")
    resources=pd.DataFrame([json.loads(line) for line in (root/"control/resource_history.jsonl").read_text().splitlines()])
    junit=ET.parse(root/"validation/final_all_tests.xml").getroot()
    suites=list(junit.iter("testsuite"))
    unit_tests={name:sum(int(s.attrib.get(name,0)) for s in suites) for name in ["tests","failures","errors","skipped"]}
    code=git(repo,"rev-parse","HEAD")
    report=dict(phase_id=c["phase_id"],step_id=c["step_id"],run_id=c["run_id"],model_class=c["model_class"],
        status="BLOCKED_WITH_EXACT_INTEGRITY_DEFECT",lifecycle_status="WAITING_FOR_EXTERNAL_REVIEW",
        result_checkpoint_before_document_commit=code,reviewed_authority_sha256=c["authority_zip_sha256"],
        launch_sha256=c["launch_sha256"],required_ancestor=c["required_ancestor"],registry=registry,
        current_unit_tests=unit_tests,
        baseline_plan=baseline,preservation=preserved,blocking_diagnosis=blocked,
        original_start_unix=c["task_started_unix"],hard_deadline_unix=c["hard_deadline_unix"],timer_reset=False,
        elapsed_seconds_before_seal=time.time()-c["task_started_unix"],
        maximum_sampled_process_rss_bytes=int(resources.process_tree_rss_bytes.max()),
        output_bytes_before_inventory=tree_bytes(root),engineering_epochs_used=3,
        baseline_complete=False,optional_training_data_rounds=0,phase7_training_performed=False,
        ICNN_constructions=0,national_assignment_performed=False,national_physical_fidelity_validated=False,
        B_restoration_performed=False,fine_R_expansion_performed=False,other_suite_generation_performed=False,
        physical_benchmark_rerun_performed=False,capacity_multiplier_tuning_performed=False,
        inactive_pair_fallback_performed=False,regional_physical_arc_deletion_performed=False,
        regional_pair_support_pruned=True,low_active_support_policy_authorized=True,
        source_model_readiness="MODEL_REVISION_REQUIRED",next_step_authorized=False,
        no_source_arrays_or_existing_labels_rewritten=True,
        next_action="External scientific review of the mandatory LOW source/path-contract conflict; no further generation under this terminal handoff")
    report["unperformed_required_work"]=["Remaining mandatory baseline labels and cold retries","Complete baseline finite-difference audits",
        "12-region precision/cold-warm audit","Real full-to-active zero-embedding native probe",
        "Completed-dataset Phase7 views","All proportional fresh-seed rounds"]
    n=preserved["published_coarse_labels"]; missing=preserved["missing_baseline_unique_inputs"]
    witness=blocked["certificate"]; source=blocked["source_audit"]
    summary=f"""# Phase 6 s03 LOW-active campaign: blocked result

The authorized campaign adopted LOW CURRENT active-only regional connectivity and
generated a substantial part of the mandatory baseline. It stopped at a required
LOW value-hierarchy check. The baseline is incomplete and enrichment did not start.
Status: **BLOCKED_WITH_EXACT_INTEGRITY_DEFECT**; **WAITING_FOR_EXTERNAL_REVIEW**.
This is a result report and recommendation, not approval of a scientific change.

## Verified scope and outputs

- 1,847 regions; 80,316 active regional pairs in 1,566 regions; 281 no-active regions.
- 808,471 zero-LOW conceptual pairs have no edge and no fallback. All 327,757 owned
  physical arcs remain. The 143,603 static links and 72,724 retained nodes are unchanged.
  Total logical links: 223,919.
- 682 original positive LOW keys are explicitly outside the accepted interface:
  677 same-terminal exclusions and five independently proved centroid exclusions.
  No original demand was silently discarded or made an inactive coordinate.
- The plan freezes 76,734 roles / 75,276 unique native inputs. {n:,} coarse labels
  and {preserved['fixed_fine_labels']:,} fixed-fine comparison labels were produced.
  {missing:,} mandatory unique coarse inputs remain missing.
- {preserved['regions_with_all_mandatory_unique_labels']:,} active regions have all
  required unique coarse labels; this does not establish baseline acceptance because
  the shared mandatory source/path gate failed. All 281 inactive regions skipped data,
  optimizer and ICNN construction.
- Every existing published label was reopened in a fresh process: native payload,
  path/fine/arc feasibility, BPR potential, full active LMO, value bounds and normalized
  chain rule passed. Quality counts: {json.dumps(preserved['quality_counts'],sort_keys=True)}.
  These labels remain conditional on their exact saved regional model. The default
  dataset loader rejects the blocked campaign.
- Projected HIGH retains {baseline['projected_HIGH_exposure']:,.8f} regional traversal
  exposure; {baseline['excluded_HIGH_exposure']:,.8f} is explicitly excluded, without
  redistribution or background demand. These are not deduplicated national OD totals.

## Exact blocker

For region `{witness['region_id']}`, an independent replay of the original Phase 3
oracle reprices the saved feasible LOW iterate and yields a lower bound
{witness['independent_lower_bound']:.8f} flow-seconds. The mapped physical potential is
{witness['mapped_physical_potential']:.8f}. The lower bound exceeds it by
{witness['lower_exceeds_mapped_by']:.8f}, with a rounding allowance of only
{witness['roundoff_allowance']:.8f}. There is no outside-interface LOW exposure in this
region. Thus the required comparison W(u_LOW) <= V_fixedfine <= B(z_mapped) cannot pass
under the inherited centroid nontransit path rule.

All native regional node balances pass (maximum error
{source['maximum_node_balance_error']:.12g}). Nine active regions have positive
nonendpoint through-flow at a prohibited centroid; this is a source/path-contract
mismatch, not a net-flow balance error. At physical centroid 19241 in the witness
region, the excess is about 4,872.145054 native flow units. The supplied ledger records
all nine cases. Its summed node exposure is not an OD-demand total, and the ledger is
not a reconstruction of historical path mixtures. A cheaper physical path crossing
the prohibited centroid is retained only as a diagnostic witness.

## Implementation, preservation and limits

The existing native loaders, regional CSR products and indexed heap were reused.
New Phase 6 code provides exact active pricing, feasible path-pool FW with bounded
path exchanges, atomic native labels/catalogs, source/model/seed hashes and guarded
untrained loaders. {unit_tests['tests']} focused/reference unit tests pass. Three engineering epochs resolved source-ledger classification,
immutable resume and the native retained-node flag; none changed the scientific model.
Prepared precision/enrichment code has unit tests but did not execute those native
stages. Its presence is not evidence that those stages completed.

There was no prior Phase 6 dataset to migrate. No old positive-demand label was
truncated or relabeled as a reduced-model value. Existing new labels and source
arrays remain unchanged. {preserved['authoritative_source_files_rehashed']:,} original
source files were rehashed; all matched. Static Phase 5 sources were also rehashed.

The current authority SHA-256 is `{c['authority_zip_sha256']}`; launch SHA-256 is
`{c['launch_sha256']}`. The starting/reviewed ancestor is `{c['required_ancestor']}`.
The exact final snapshot HEAD, ZIP/hash/size and sealing checks are bound by the
adjacent companion. The effective config and authority hierarchy are included.

The original clock began 2026-09-07 03:29:13 UTC, with hard deadline
2026-09-08 03:29:13 UTC. It was never reset. Work stopped early because of the
confirmed hard gate, not a quota or resource limit. Preseal elapsed time is
{report['elapsed_seconds_before_seal']/3600:.4f} h; maximum sampled process RSS is
{report['maximum_sampled_process_rss_bytes']/2**30:.4f} GiB. Final elapsed/disk/RSS
evidence is in the companion and resource records. RSS figures are sampled values.

No ICNN construction/training, B restoration, fine-R expansion, inactive fallback,
other-suite generation, national assignment, capacity tuning, physical benchmark
rerun or manuscript numerical-result change was performed. Upstream national
physical-model readiness remains MODEL_REVISION_REQUIRED.

## Recommendation and exact next step

Recommend **BLOCK pending external scientific review**. Resolve the compatibility
of the inherited centroid path restriction, the native mapped physical reference,
and the mandatory comparison. Do not silently relax the path rule, alter demands,
regenerate physical benchmarks or waive the gate. Any authorized model change must
identify affected model hashes and preserve/recompute or explicitly recertify work
under an approved migration policy. No next scientific step is authorized by this
report. An explicit review decision must bind the final ZIP SHA-256 and code/config.
"""
    write_json(root/"reports/FINAL_RESULT.json",report)
    text_write(root/"reports/FINAL_REPORT.md",summary)
    write_json(work/"FINAL_RESULT.json",report)
    text_write(work/"FINAL_REPORT.md",summary)
    resources.to_csv(root/"reports/RESOURCE_USAGE.csv",index=False)
    # Current state is prepended; older authority/result records stay untouched below it.
    update=("# Phase 6 s03 active campaign: blocked pending external review\n\n"
        f"Mandatory LOW value comparison failed and was independently repriced. {n:,} coarse labels are preserved and freshly recertified under their exact regional model; {missing:,} mandatory inputs remain missing. Baseline and enrichment are incomplete.\n\n"
        "See phase_6_work/s03_active_0906/FINAL_REPORT.md and FINAL_RESULT.json. The single final ZIP and companion are referenced by handoffs/phase_06/step_s03_active_0906/LATEST_HANDOFF.txt after sealing. Next step: external review only; no model/path-rule change is self-approved.\n\n---\n\n")
    for name in ["00_READ_FIRST.md","PROJECT_STATUS.md"]:
        p=repo/"partC_doc_icnn"/name
        p.write_text(update+p.read_text(encoding="utf-8"),encoding="utf-8")
    p=repo/"partC_doc_icnn/12_DECISION_LOG.md"
    entry=("# D-047 — Phase 6 mandatory LOW hierarchy gate stop (2026-09-07)\n\n"
        "Recorded result, not an approval: original Phase 3 pricing confirms a local lower bound above mapped physical potential in region 9567843a01be0d75fd9b. Nine regions have native centroid through-flow inconsistent with the inherited nontransit reference. Preserve source and certified regional labels, disable default dataset use, produce one final handoff and await an explicit hash-bound review. Do not change mathematical/path semantics or rerun physical benchmarks by inference.\n\n---\n\n")
    p.write_text(entry+p.read_text(encoding="utf-8"),encoding="utf-8")
    return report


def copy(source,target):
    target=Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)


def dependency_snapshot(repo):
    """Resolve the import closure of actual Phase 6 modules, including package initializers."""
    seeds=list((repo/"partC_rev1_src/phase6").glob("*.py"))
    found=set(); queue=list(seeds)
    while queue:
        p=queue.pop()
        if p in found or not p.exists(): continue
        found.add(p)
        rel=p.relative_to(repo).with_suffix("")
        module=".".join(rel.parts)
        package=module.removesuffix(".__init__") if p.name=="__init__.py" else module.rpartition(".")[0]
        for parent in p.parents:
            if parent==repo: break
            init=parent/"__init__.py"
            if init.exists() and init not in found: queue.append(init)
        tree=ast.parse(p.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import): names=[n.name for n in node.names]
            if isinstance(node,ast.ImportFrom):
                name=importlib.util.resolve_name("."*node.level+(node.module or ""),package) if node.level else (node.module or "")
                names=[name]+[name+"."+n.name for n in node.names if n.name!="*"]
            for name in names:
                if name.startswith("partC_rev1_src"):
                    candidate=repo/Path(*name.split("."))
                    for target in [candidate.with_suffix(".py"),candidate/"__init__.py"]:
                        if target.exists() and target not in found: queue.append(target)
    return sorted(found)


def example(c,rid,target):
    rd=region_dir(c,rid); d=read_json(rd/"descriptor.json")
    for name in ["descriptor.json","local_graph.npz","support_map.npz","fine_members.npz","active_pairs.parquet"]:
        copy(rd/name,target/name)
    if not d["m_active"]:
        text_write(target/"NO_ACTIVE_PAIRS.md","No labels, optimizer or ICNN were constructed for this region.\n")
        return
    plan=pd.read_parquet(rd/"baseline_plan.parquet")
    ih=plan.loc[plan.family.eq("LOW"),"input_hash"].iloc[0]
    s=Store(rd,d,read_npz(rd/"local_graph.npz"))
    if not s.exists(ih): raise ValueError("Chosen review example has no LOW label")
    r,receipt=s.get(ih)
    copy(s.root/"receipts"/(ih+".json"),target/"label_receipt.json")
    copy(s.root/receipt["payload_path"],target/"label_payload.npz")
    paths=[s.oracle.paths[int(k)] for k in r.path_ids]
    pair,source,dest=s.oracle.pool_arrays()
    write_npz(target/"used_path_catalog.npz",path_id=r.path_ids,pair=pair[r.path_ids],source=source[r.path_ids],
        target=dest[r.path_ids],offsets=np.r_[0,np.cumsum([len(p) for p in paths])].astype(np.int64),
        arcs=np.concatenate(paths) if paths else np.empty(0,np.int64))
    write_json(target/"WITNESS_FORMAT.json",dict(original_path_ids_preserved=True,partial_catalog=True,
        reconstruction="phase6.artifacts.verify_witness",scope="Only paths used by the included native LOW label"))


def seal(c,config_path):
    root=Path(c["output_root"]); repo=Path(c["repository_root"])
    if (root/"control/writer.lock").exists(): raise ValueError("Scientific writer still active")
    scoped=git(repo,"status","--porcelain","--","partC_rev1_src/phase6","tests/test_phase6*",str(Path(c["work_records_root"])),
        "partC_doc_icnn/00_READ_FIRST.md","partC_doc_icnn/PROJECT_STATUS.md","partC_doc_icnn/12_DECISION_LOG.md")
    if scoped: raise ValueError("Commit current code/tests/result documents before sealing")
    result=read_json(root/"reports/FINAL_RESULT.json"); head=git(repo,"rev-parse","HEAD")
    if result["status"]!="BLOCKED_WITH_EXACT_INTEGRITY_DEFECT": raise ValueError("Wrong terminal report")
    if time.time()>=c["hard_deadline_unix"]: raise TimeoutError("Original campaign deadline")
    hid="active_0906_"+time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())+"_blocked_final_001"
    parent=repo/"partC_doc_icnn/handoffs/phase_06/step_s03_active_0906"
    source=parent/hid; staging=parent/(hid+".building")
    archive_root=repo/"artifacts_icnn/handoff_bundles/phase_06"
    archive=archive_root/("phase06_steps03_active_0906_"+hid+"_SYNC.zip")
    for p in [source,staging]:
        if not p.resolve().is_relative_to(parent.resolve()): raise ValueError("Handoff path escapes")
    if not archive.resolve().is_relative_to(archive_root.resolve()): raise ValueError("Archive path escapes")
    if source.exists() or staging.exists() or archive.exists(): raise FileExistsError("Never replace a handoff")
    staging.mkdir(parents=True); archive_root.mkdir(parents=True,exist_ok=True)
    status={**result,"ending_head":head,"handoff_id":hid,"next_step_authorized":False,
        "lifecycle_status":"WAITING_FOR_EXTERNAL_REVIEW","whole_elapsed_seconds_before_inventory":time.time()-c["task_started_unix"]}
    write_json(root/"reports/FINAL_SEAL_STATUS.json",status)
    write_json(root/"control/status.json",status,new=False)
    external=inventory(root)
    sections=["governing_docs","configs","code_snapshot","code_diff","summary_data","figures","geospatial_summary","log_excerpts","external_artifact_index"]
    for name in sections: (staging/name).mkdir()
    for p in sorted(Path(c["authority_root"]).rglob("*")):
        if p.is_file() and p.suffix not in {".zip",".pyc"} and "__pycache__" not in p.parts:
            copy(p,staging/"governing_docs/authority"/p.relative_to(c["authority_root"]))
    for name in ["00_READ_FIRST.md","PROJECT_STATUS.md","12_DECISION_LOG.md","20_HANDOFF_SYNC_PROTOCOL.md","21_REVIEW_APPROVAL_GATES.md","23_SYNC_BUNDLE_CONTENT_AND_SIZE_POLICY.md","24_CODE_AND_DATA_SNAPSHOT_POLICY.md"]:
        copy(repo/"partC_doc_icnn"/name,staging/"governing_docs"/name)
    copy(repo/"AGENTS.md",staging/"governing_docs/AGENTS.md")
    copy(config_path,staging/"configs/resolved_effective_config.json")
    for name in ["native_source_binding_file","phase4_graph_config_file"]: copy(c[name],staging/"configs"/(name+".json"))
    launch=Path(c["authority_root"]).parent/"START_P6_S03_ACTIVE_24H_0906.md"
    if not launch.exists(): raise FileNotFoundError("Bound launch document missing")
    if sha(launch)!=c["launch_sha256"]: raise ValueError("Launch document changed")
    copy(launch,staging/"governing_docs/START_P6_S03_ACTIVE_24H_0906.md")
    snapshot=dependency_snapshot(repo)+list((repo/"tests").glob("test_phase6*.py"))
    changed=git(repo,"diff","--name-only",c["required_ancestor"],head).splitlines()
    for name in changed:
        p=repo/name
        if p.is_file() and p.suffix in {".py",".json",".toml",".yaml",".yml"} and p not in snapshot: snapshot.append(p)
    for p in snapshot: copy(p,staging/"code_snapshot"/p.relative_to(repo))
    for p in sorted(Path(c["work_records_root"]).glob("*")):
        if p.is_file(): copy(p,staging/"summary_data/work_records"/p.name)
    patch=subprocess.check_output(["git","-C",str(repo),"-c","core.longpaths=true","diff",c["required_ancestor"],head])
    (staging/"code_diff/phase6.patch").write_bytes(patch)
    text_write(staging/"code_diff/git_status.txt",git(repo,"status","--short")+"\n")
    text_write(staging/"code_diff/git_diff_stat.txt",git(repo,"diff","--stat",c["required_ancestor"],head)+"\n")
    text_write(staging/"code_diff/commits.txt",git(repo,"log","--oneline",c["required_ancestor"]+".."+head)+"\n")
    for p in sorted((root/"manifests").glob("output*")): copy(p,staging/"external_artifact_index"/p.name)
    for name in ["registry/REGISTRY_COMPLETE.json","registry/regions.parquet","registry/runtime_adjacency.json",
                 "registry/outside_interface_trace_ledger.parquet","registry/accepted_centroid_exclusion_proofs.json",
                 "registry/high_projection_loss.parquet","samples/BASELINE_PLAN_FROZEN.json"]:
        copy(root/name,staging/"summary_data"/name)
    for folder in ["mandatory_hierarchy_defect","preservation","source_support_audit"]:
        for p in sorted((root/"validation"/folder).glob("*")):
            if p.is_file() and p.stat().st_size<25*2**20: copy(p,staging/"summary_data/validation"/folder/p.name)
    for p in sorted((root/"validation").glob("*")):
        if p.is_file() and p.stat().st_size<4*2**20: copy(p,staging/"summary_data/validation"/p.name)
    for p in sorted((root/"control").glob("*")):
        if not p.is_file(): continue
        if p.stat().st_size<=2**20: copy(p,staging/"log_excerpts/control"/p.name)
        elif p.suffix in {".jsonl",".log"}:
            from collections import deque
            tail=deque(maxlen=40)
            with p.open(encoding="utf-8") as stream:
                first=[]
                for i,line in enumerate(stream):
                    if i<10: first.append(line)
                    tail.append(line)
            text_write(staging/"log_excerpts/control"/(p.name+".excerpt.txt"),"".join(first)+"\n[Full original file is SHA-bound in external index; final 40 lines follow.]\n"+"".join(tail))
    selected=read_json(root/"validation/P2_selection.json")
    examples=[selected["region_ids"][0],selected["region_ids"][3],selected["inactive_region_id"],"partC_p2_region_9567843a01be0d75fd9b"]
    for rid in examples: example(c,rid,staging/"summary_data/native_examples"/rid.removeprefix("partC_p2_region_"))
    text_write(staging/"figures/NO_FIGURES.md","The exact value-bound and centroid-flow tables are the direct diagnostic evidence; no new figures were generated.\n")
    text_write(staging/"geospatial_summary/NO_NEW_GEOSPATIAL_PRODUCT.md","No geometry was changed or regenerated; accepted native graph IDs and source hashes remain authoritative.\n")
    summary=(root/"reports/FINAL_REPORT.md").read_text()
    core={"00_HANDOFF_README.md":"Upload this single ZIP. Read 01_STEP_SUMMARY.md and the exact mandatory_hierarchy_defect evidence. This is BLOCKED, not a completed baseline. The adjacent companion binds final ZIP/hash/size and extraction checks.\n",
        "01_STEP_SUMMARY.md":summary,
        "03_DECISIONS_AND_OPEN_QUESTIONS.md":"Verified facts and the exact counterexample are in the summary. External review must resolve the native physical reference versus inherited centroid nontransit path contract. No path-rule, demand, cost, comparison or acceptance change was approved by Codex. Existing labels remain tied to their exact restricted model; future reuse requires explicit model/migration authority.\n",
        "04_VALIDATION_RESULTS.md":"Registry, strict LOW selection, immutable source identities, frozen HIGH projection, 8-region startup, all existing label reopens and runtime closure pass. Mandatory LOW hierarchy FAILS and was independently repriced. Remaining baseline, prescribed precision and enrichment were not executed to completion. XML receipts record unit tests; none substitutes for the failed native gate.\n",
        "09_NEXT_STEP_RECOMMENDATION.md":"Recommend BLOCK pending an explicit scientific review decision bound to this ZIP SHA-256 and final HEAD. Do not resume generation, change centroid semantics, rerun physical references, train ICNNs or start another phase without that decision. Preserve this output root and original clock.\n",
        "10_UPLOAD_THIS_FILE.txt":str(archive)+"\n",
        "11_CHANGELOG_FROM_PREVIOUS_HANDOFF.md":"First and only Phase 6 active-model handoff for this Goal. It follows the explicitly supplied active/enrichment/base authority chain and reviewed Phase 5F evidence. P0-P2 and partial P3 are recorded; the independent LOW hierarchy blocker and full preservation audit are new. Previous phase artifacts and all original source arrays remain immutable. No earlier Phase 6 ZIP is superseded.\n"}
    for name,value in core.items(): text_write(staging/name,value)
    write_json(staging/"02_HANDOFF_STATUS.json",status)
    copy(root/"reports/RESOURCE_USAGE.csv",staging/"05_RESOURCE_USAGE.csv")
    pd.DataFrame([dict(relative_path=p.relative_to(repo).as_posix(),sha256=sha(p),size_bytes=p.stat().st_size,
                       changed_by_task=p.relative_to(repo).as_posix() in changed) for p in snapshot]).to_csv(staging/"07_CODE_CHANGE_INDEX.csv",index=False)
    stages=[json.loads(line) for line in (root/"control/stage_history.jsonl").read_text().splitlines()]
    commands=["# Recorded execution commands\n\nNative threads: NUMBA_NUM_THREADS=OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1. NUMBA_CACHE_DIR is the external run cache.\n\n"]
    for s in stages:
        if s["status"]=="STARTED": commands.append(f"- HEAD {s['code_head']}; config {s['config_sha256']}; started {s['started_unix']}: `{c['python_executable']} -X utf8 -m partC_rev1_src.phase6.run --config {config_path} --stage {s['stage']}`\n")
    commands.append("\nUnit test commands target every tests/test_phase6*.py and current authority/tests with pytest -q and the included JUnit output paths. Preliminary read-only shell inspections are summarized in work records; they are not an exhaustive shell transcript. All persisted numerical diagnostics are reproducible through the versioned entrypoints.\n")
    commands.append(f"\nFinal packaging: `{c['python_executable']} -X utf8 -m partC_rev1_src.phase6.handoff --config {config_path} --operation seal`\n")
    text_write(staging/"08_COMMAND_LOG.md","".join(commands))
    versions={name:importlib.metadata.version(name) for name in ["numpy","pandas","scipy","numba","pyarrow","psutil","pytest"]}
    write_json(staging/"configs/runtime_versions.json",dict(python=os.sys.version,packages=versions,network_used=False,packages_installed=False))
    from partC_rev1_src.phase5.handoff import _privacy_scan
    scan=_privacy_scan(staging); remaining=[]; checked=[]
    for finding in scan["findings"]:
        p=staging/finding["path"]
        if finding["pattern"]=="unreviewed_binary_type" and p.suffix==".parquet":
            import pyarrow.parquet as pq
            f=pq.ParquetFile(p); checked.append({**finding,"rows":f.metadata.num_rows,"typed_reopen":"PASS"})
        elif finding["pattern"]=="unreviewed_binary_type" and p.suffix==".npz":
            npz_schema(p); checked.append({**finding,"typed_reopen":"NO_PICKLE_NPZ_PASS"})
        elif finding["pattern"]=="unreviewed_binary_type" and p.suffix==".xml":
            ET.parse(p); checked.append({**finding,"typed_reopen":"XML_PASS"})
        else: remaining.append(finding)
    scan.update(findings=remaining,status="PASS" if not remaining else "FAIL",typed_format_reopens=checked)
    write_json(staging/"log_excerpts/privacy_and_format_scan.json",scan)
    if remaining: raise ValueError("Package scan failed: "+str(remaining))
    records=[]
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            if p.stat().st_size>25*2**20: raise ValueError("Member exceeds 25 MiB: "+str(p))
            records.append(dict(relative_path=p.relative_to(staging).as_posix(),size_bytes=p.stat().st_size,sha256=sha(p)))
    pd.DataFrame(records).to_csv(staging/"06_ARTIFACT_MANIFEST.csv",index=False)
    checks=records+[dict(relative_path="06_ARTIFACT_MANIFEST.csv",sha256=sha(staging/"06_ARTIFACT_MANIFEST.csv"))]
    text_write(staging/"CHECKSUMS.sha256","".join(r["sha256"]+"  "+r["relative_path"]+"\n" for r in checks))
    temp=archive.with_suffix(".zip.building")
    with zipfile.ZipFile(temp,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(staging.rglob("*")):
            if p.is_file(): z.write(p,p.relative_to(staging).as_posix())
    if temp.stat().st_size>100*2**20: raise ValueError("Compact ZIP exceeds 100 MiB target")
    extract=root/"manifests"/(hid+"_fresh_extraction")
    if extract.exists(): raise FileExistsError(extract)
    with zipfile.ZipFile(temp) as z:
        if z.testzip() is not None: raise ValueError("ZIP CRC failure")
        for name in z.namelist():
            if not (extract/name).resolve().is_relative_to(extract.resolve()): raise ValueError("Unsafe archive member")
        z.extractall(extract)
    for row in checks:
        if sha(extract/row["relative_path"])!=row["sha256"]: raise ValueError("Fresh extracted hash differs")
    witness_results=[]
    for p in sorted((extract/"summary_data/native_examples").iterdir()):
        if (p/"label_payload.npz").exists(): witness_results.append(dict(example=p.name,**verify_witness(p)))
    snapshot_xml=root/"manifests"/(hid+"_snapshot_tests.xml")
    environment=dict(os.environ)
    environment.update(PYTHONPATH=str(extract/"code_snapshot"),NUMBA_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",NUMBA_CACHE_DIR=str(root/"manifests/snapshot_numba_cache"))
    check=subprocess.run([c["python_executable"],"-X","utf8","-m","pytest","-q",
        str(extract/"code_snapshot/tests"),str(extract/"governing_docs/authority/tests"),"--junitxml="+str(snapshot_xml)],
        cwd=extract/"code_snapshot",env=environment,text=True,encoding="utf-8",capture_output=True)
    text_write(root/"manifests"/(hid+"_snapshot_tests.log"),check.stdout+check.stderr)
    if check.returncode: raise ValueError("Fresh extracted unit tests failed: "+check.stdout+check.stderr)
    snapshot_suites=list(ET.parse(snapshot_xml).getroot().iter("testsuite"))
    extracted_tests={name:sum(int(s.attrib.get(name,0)) for s in snapshot_suites) for name in ["tests","failures","errors","skipped"]}
    elapsed=time.time()-c["task_started_unix"]
    if elapsed>=86400: raise TimeoutError("Whole-task hard deadline")
    digest=sha(temp); size=temp.stat().st_size
    # Both source directory targets were resolved and contained above; no overwrite.
    staging.rename(source); temp.rename(archive)
    companion=dict(status="VERIFIED_HANDOFF_WAITING_FOR_EXTERNAL_REVIEW",scientific_status=status["status"],
        lifecycle_status="WAITING_FOR_EXTERNAL_REVIEW",handoff_id=hid,zip_path=str(archive),zip_sha256=digest,
        zip_bytes=size,source_folder=str(source),ending_head=head,branch=c["branch"],next_step_authorized=False,
        authority_sha256=c["authority_zip_sha256"],config_sha256=sha(config_path),
        internal_member_count=len(checks)+1,verified_checksums=len(checks),CRC_test="PASS",fresh_extraction="PASS",
        compact_native_witnesses=witness_results,source_and_existing_data_reopens="PASS_CONDITIONAL_EXACT_MODEL",
        fresh_snapshot_unit_tests=extracted_tests,snapshot_unit_test_xml_sha256=sha(snapshot_xml),
        mandatory_LOW_comparison="FAIL_CONFIRMED",whole_campaign_elapsed_seconds=elapsed,
        unused_original_budget_seconds=86400-elapsed,early_stop_reason="EXACT_MANDATORY_SOURCE_PATH_CONTRACT_DEFECT",
        process_rss_at_seal_bytes=psutil.Process().memory_info().rss,
        measured_added_bytes_in_run_root_and_this_handoff=tree_bytes(root)+tree_bytes(source)+size,
        full_output_manifest=external,one_final_zip=True,no_next_stage_dispatched=True)
    write_json(archive.with_suffix(".companion.json"),companion)
    text_write(archive.with_suffix(".sha256"),digest+"  "+archive.name+"\n")
    latest=dict(source_folder=str(source),zip_path=str(archive),zip_sha256=digest,zip_bytes=size,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),status="WAITING_FOR_EXTERNAL_REVIEW",scientific_status=status["status"],ending_head=head,
        companion_path=str(archive.with_suffix(".companion.json")))
    text_write(parent/"LATEST_HANDOFF.txt",json.dumps(latest,indent=2)+"\n")
    write_json(root/"manifests/HANDOFF_SEAL_RESULT.json",companion)
    return companion


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--operation",required=True,choices=["prepare","seal"])
    args=p.parse_args(); c=load_config(args.config)
    result=prepare(c) if args.operation=="prepare" else seal(c,args.config)
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__": main()
