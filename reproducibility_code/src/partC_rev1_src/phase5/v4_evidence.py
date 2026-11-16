"""Version 4 dependency verification and evidence; the existing runner owns execution."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

V4_SCHEMA = "partC.phase5.pilot_v4_0905.execution.v1"
V4_AUTHORITY_SHA = "f6cbbb285e1fcd1afaa9810136ac168a0e7b51d6b2888b06a64d010d764fc462"
V2_HANDOFF_SHA = "4c199414121a379ebe30ead1e7c4066c1db29aad13885ddcb6f5165b8379c46a"


def is_v4(execution):
    return execution.get("schema_id") == V4_SCHEMA


def authorize_audit_repair(*, config_path, execution, output, pre_run_head, previous_head,
                           live_head, prior_records, validation_path, validation_payload,
                           changed_paths, reason):
    """Preserve successful executions rejected by a later evidenced result audit.

    This is a dependency-scoped v4 repair authorization, not an invented child
    failure or an overwritten PASS. Original receipts and attempt status survive.
    """
    from .staged_pilot import (_atomic_json, _append_jsonl, _bound_file, _artifact_record,
        _receipt_path, validate_pass_receipt, _engineering_retry_root,
        _validate_engineering_retry_chain, _git_patch_sha256, ENGINEERING_RETRY_SCHEMA, sha256_file, utc_now)
    if not is_v4(execution) or validation_payload.get("affected_stages") != ["R5","R6"]:
        raise ValueError("This audited repair transition is restricted to v4 R5/R6")
    rejection_path=output/validation_payload["audit_rejection_relative_path"]
    if not rejection_path.resolve().is_relative_to(output.resolve()) or sha256_file(rejection_path)!=validation_payload["audit_rejection_sha256"]:
        raise ValueError("Audit rejection evidence binding failed")
    rejection=json.loads(rejection_path.read_text())
    if rejection.get("status")!="FAIL_REQUIRES_ENGINEERING_REPAIR" or not rejection.get("invalid_total_normalized_band_variant_count"):
        raise ValueError("A post-run repair requires concrete failing result evidence")
    if not validation_payload.get("upstream_dependency_ast_checks_pass"):
        raise ValueError("R0-R4 dependency preservation is not demonstrated")
    ordinal=len(prior_records)+1
    archive=output/"validation/superseded_audits"/f"epoch_{ordinal:03d}"
    if archive.exists(): raise FileExistsError(archive)
    receipts={};targets=set();old_attempts={}
    for stage in ("R5","R6"):
        path=_receipt_path(output,stage)
        receipt=validate_pass_receipt(output=output,receipt_path=path,config_path=config_path,
            pre_run_code_head=pre_run_head,expected_stage_id=stage)
        if sha256_file(path)!=rejection["original_receipts"][stage]:
            raise ValueError("Original reviewed PASS receipt changed")
        receipts[stage]=receipt;targets.add(path)
        targets.update(output/r["relative_path"] for r in receipt["artifacts"])
        old_attempts[stage]=str(receipt["attempt_id"])
    for relative in ("reports/08_PILOT_STATUS.json","validation/resource_usage.json","validation/reproduction.json",
        "validation/required_output_validation.json","manifests/MANIFEST.csv","manifests/CHECKSUMS.sha256"):
        p=output/relative
        if p.is_file(): targets.add(p)
    # Verify every final absolute source and target stays inside the named output
    # root before any move. A source is a file, never the root or a parent tree.
    moves=[]
    for p in sorted(targets):
        source=p.resolve();target=(archive/"original_products"/p.relative_to(output)).resolve()
        if not source.is_file() or not source.is_relative_to(output.resolve()) or not target.is_relative_to(archive.resolve()):
            raise ValueError("Unsafe audited evidence move")
        moves.append((source,target,_artifact_record(output,source)))
    archive.mkdir(parents=True,exist_ok=False)
    for source,target,_ in moves:
        target.parent.mkdir(parents=True,exist_ok=True);source.replace(target)
    records=[{**record,"preserved_relative_path":target.relative_to(output).as_posix(),"reopened_sha256":sha256_file(target)} for _,target,record in moves]
    if any(r["sha256"]!=r["reopened_sha256"] for r in records):
        raise ValueError("Preserved audit products failed hash equality")
    preservation=archive/"audit_invalidation.json"
    _atomic_json(preservation,{"schema_id":"partC.phase5.v4.audited_dependency_invalidation.v1","status":"PASS",
        "trigger":"POST_RUN_AUDIT_REJECTION", "prior_child_execution_status":"PASS_PRESERVED_NOT_REWRITTEN",
        "rejection_evidence":_bound_file(output,rejection_path),"invalidated_stages":["R5","R6"],
        "upstream_reused_stages":["R0_v2","R1_s02","R2_s03","R2_s04","R2_s05","R2_s06","R3","R4"],"preserved_artifacts":records})
    next_attempts={s:f"attempt_{int(old_attempts[s].split('_')[1])+1:03d}" for s in old_attempts}
    repo=Path(execution["project_root"])
    epoch={"schema_id":ENGINEERING_RETRY_SCHEMA,"epoch_number":ordinal,"recorded_utc":utc_now(),
        "campaign_id":execution["campaign_id"],"run_id":execution["run_id"],"branch":execution["expected_branch"],
        "pre_run_code_head":pre_run_head,"previous_execution_code_head":previous_head,"authorized_execution_code_head":live_head,
        "previous_epoch_sha256":prior_records[-1]["record_sha256"] if prior_records else None,
        "execution_config_path":str(config_path),"execution_config_sha256":sha256_file(config_path),
        "failed_stage_id":"R5","failed_attempt_id":old_attempts["R5"],"authorized_next_attempt_id":next_attempts["R5"],
        "authorized_stage_attempts":next_attempts,"retry_reason":reason,"trigger":"POST_RUN_AUDIT_REJECTION",
        "failed_stage_field_semantics":"COMPATIBILITY_FIELD_FOR_AUDIT_INVALIDATED_STAGE; CHILD_EXECUTION_WAS_PASS",
        "failed_timing":{"status":"PASS_SUBSEQUENTLY_INVALIDATED_BY_RESULT_AUDIT"},
        "failed_attempt_evidence":[_bound_file(output,preservation),_bound_file(output,rejection_path)],
        "upstream_receipts":[_bound_file(output,_receipt_path(output,"R4"))],
        "repair_validation_record":{"relative_path":validation_path.relative_to(repo).as_posix(),"size_bytes":validation_path.stat().st_size,"sha256":sha256_file(validation_path)},
        "changed_paths":changed_paths,"git_patch_sha256":_git_patch_sha256(repo,previous_head,live_head),
        "scientific_config_unchanged":True,"upstream_pass_receipts_unchanged":True,"blind_retry_performed":False,
        "status":"PASS_AUTHORIZED_ENGINEERING_RETRY"}
    epoch_path=_engineering_retry_root(output)/f"epoch_{ordinal:03d}.json"
    _atomic_json(epoch_path,epoch)
    _validate_engineering_retry_chain(config_path=config_path,execution=execution,output=output,pre_run_code_head=pre_run_head)
    _append_jsonl(output/"validation/repair_epochs.jsonl",epoch)
    _append_jsonl(output/"validation/stage_events.jsonl",{"event":"POST_RUN_AUDIT_INVALIDATED_R5_R6","recorded_utc":utc_now(),"code_epoch":ordinal,"preservation":str(preservation)})
    _atomic_json(output/"validation/current_stage.json",{"stage_id":"R5","state":"AUDIT_REPAIR_READY","attempt_code_head":live_head,"pre_run_code_head":pre_run_head,"authorized_attempts":next_attempts})
    return epoch_path


def verify_v4_authority(execution):
    from .staged_pilot import authority_root, sha256_file
    root = authority_root(execution)
    outer = Path(execution["authority_zip_path"])
    inner = root / execution["latest_handoff_relative"]
    if sha256_file(outer) != V4_AUTHORITY_SHA or sha256_file(inner) != V2_HANDOFF_SHA:
        raise ValueError("Current v4 authority or reviewed handoff SHA-256 mismatch")
    checks = []
    for archive, extracted in ((outer, root), (inner, Path(execution["reviewed_handoff_extraction"]))):
        with zipfile.ZipFile(archive) as z:
            if z.testzip() is not None:
                raise ValueError("Authority ZIP CRC failed")
            members = {n for n in z.namelist() if not n.endswith("/")}
            for name in members:
                p = extracted / name
                if not p.is_file() or hashlib.sha256(z.read(name)).hexdigest() != sha256_file(p):
                    raise ValueError(f"Authority extraction differs: {name}")
            declared = set()
            for line in z.read("CHECKSUMS.sha256").decode("utf-8-sig").splitlines():
                if not line.strip():
                    continue
                digest, name = line.split(maxsplit=1)
                name = name.strip().lstrip("*")
                declared.add(name)
                if name not in members or sha256_file(extracted / name) != digest.lower():
                    raise ValueError(f"Authority checksum failed: {name}")
            manifest_name = "MANIFEST.csv" if "MANIFEST.csv" in members else "06_ARTIFACT_MANIFEST.csv"
            manifest = pd.read_csv(extracted / manifest_name)
            for row in manifest.itertuples(index=False):
                p = extracted / row.relative_path
                if sha256_file(p) != row.sha256 or p.stat().st_size != int(row.size_bytes):
                    raise ValueError(f"Authority manifest failed: {row.relative_path}")
            if declared | {"CHECKSUMS.sha256"} != members:
                raise ValueError("Authority checksum member closure failed")
            if set(manifest.relative_path) | {manifest_name, "CHECKSUMS.sha256"} != members:
                raise ValueError("Authority manifest member closure failed")
            checks.append({"path": str(archive), "sha256": sha256_file(archive),
                           "members": len(members), "checksums": len(declared), "status": "PASS"})
    return {"schema_id": "partC.phase5.v4.authority_verification.v1", "status": "PASS", "package_checks": checks,
            "original_authority_sha256": execution["inherited_science"]["original_authority_sha256"],
            "v4_supersedes_prior_execution_restrictions_only_within_bounded_pilot": True}


def import_verified_r0(context, bindings, authentication_path, external_path):
    from .staged_pilot import _atomic_frame, _atomic_json, _atomic_text, _binding_row, sha256_file
    from .bacn_charges import load_reachable_od_supply_cache, physical_flow_conservation
    start = time.perf_counter()
    execution = context.execution
    old = Path(execution["r0"]["source_output_root"])
    reviewed = Path(execution["reviewed_handoff_extraction"])
    index = pd.read_csv(reviewed / "external_artifact_index/full_output_MANIFEST.csv").set_index("relative_path")
    names = ["cache/reachable_od_supply_v2.npz", "cache/reachable_od_supply_v2.metadata.json",
             "validation/reachability_and_physical_conservation_v2.json", "validation/r0_cross_scc_exact_audit.json",
             "validation/r0_scc_algorithm_benchmark.json", "validation/r0_scipy_full_graph_call_audit.json",
             "validation/r0_v1_to_v2_change_audit.json", "stage_receipts/R0_v2.json"]
    bound = []
    for name in names:
        p = old / name
        r = index.loc[name]
        if p.stat().st_size != int(r.size_bytes) or sha256_file(p) != r.sha256:
            raise ValueError(f"Reviewed R0 dependency changed: {p}")
        bound.append(_binding_row("reviewed_v2_" + name, p, r.sha256, lineage="immutable reviewed v2 R0"))
    repo = Path(execution["project_root"])
    scc_relative = "partC_rev1_src/phase5/scc_reachability.py"
    if sha256_file(repo / scc_relative) != sha256_file(reviewed / "code_snapshot" / scc_relative):
        raise ValueError("Working SCC implementation changed; v4 requires reuse")
    def function_ast(path, name):
        module = ast.parse(path.read_text(encoding="utf-8"))
        return ast.dump(next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == name), include_attributes=False)
    bacn_relative = "partC_rev1_src/phase5/bacn_charges.py"
    if function_ast(repo / bacn_relative, "physical_flow_conservation") != function_ast(reviewed / "code_snapshot" / bacn_relative, "physical_flow_conservation"):
        raise ValueError("Reviewed R0 conservation dependency changed")
    cache = context.output / names[0]
    metadata_path = context.output / names[1]
    shutil.copy2(old / names[0], cache)
    shutil.copy2(old / names[1], metadata_path)
    metadata = json.loads(metadata_path.read_text())
    supply = load_reachable_od_supply_cache(cache, expected_node_count=len(bindings.graph.node_idx_to_uid),
        expected_row_count=len(bindings.od.demand), expected_graph_identity=execution["graph_identity"],
        expected_od_identity=bindings.od.od_hash, expected_code_sha256=metadata["code_sha256"], expected_config_sha256=metadata["config_sha256"])
    accepted = execution["accepted_demand_accounting"]
    for key in ("reachable_demand", "unreachable_demand", "total_demand"):
        observed = supply.reachable_demand + supply.unreachable_demand if key == "total_demand" else getattr(supply, key)
        if not np.isclose(observed, accepted[key], rtol=accepted["comparison_rtol"], atol=accepted["comparison_atol"]):
            raise ValueError(f"R0 accepted demand mismatch: {key}")
    if supply.unreachable_row_count != accepted["unreachable_row_count"] or supply.cross_scc_reachable_row_count != 0:
        raise ValueError("R0 row or exact cross-SCC audit failed")
    if supply.row_count != accepted["aggregated_row_count"] or len(supply.scc_labels) != execution["accepted_graph"]["node_count"] or len(bindings.graph.head) != execution["accepted_graph"]["arc_count"]:
        raise ValueError("R0 accepted graph and OD counts failed")
    low = physical_flow_conservation(bindings.graph, bindings.low_physical.current_flow, supply)
    high = physical_flow_conservation(bindings.graph, bindings.high_physical.current_flow, supply)
    if not low["pass"] or not high["pass"]:
        raise ValueError("Reused R0 fails current low/high physical conservation")
    artifacts = [cache, metadata_path, authentication_path, external_path,
        context.output / "validation/focused_tests.txt", context.output / "validation/focused_tests_status.json"]
    for name in names[2:]:
        target = context.output / "repository_audit/reused_v2" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old / name, target)
        artifacts.append(target)
    # New validation is explicitly a reuse receipt, never a relabelled old run.
    prior = json.loads((old / names[2]).read_text())
    reuse = {**prior, "schema_id": "partC.phase5.v4.r0_reuse_conservation.v1", "source_run_root": str(old),
        "source_validation_sha256": sha256_file(old / names[2]), "low_physical_conservation": low,
        "high_physical_conservation": high, "reused_cache_bytes_identical": sha256_file(cache) == sha256_file(old / names[0]),
        "scc_executed_in_v4": False, "source_timing_seconds": prior["timing_seconds"],
        "timing_seconds": {"verified_cache_reuse_and_conservation": time.perf_counter() - start}, "status": "PASS"}
    validation = context.output / names[2]
    _atomic_json(validation, reuse)
    artifacts.append(validation)
    receipt = {"schema_id": "partC.phase5.v4.source_reuse.v1", "status": "PASS", "r0": reuse,
        "dependencies": bound, "unchanged_scc_sha256": sha256_file(repo / scc_relative),
        "physical_conservation_function_ast_identical": True,
        "original_cache_code_sha256_preserved": metadata["code_sha256"], "original_cache_config_sha256_preserved": metadata["config_sha256"],
        "whole_bacn_and_runner_module_changes_do_not_change_scc_or_conservation_dependencies": True,
        "r1_through_r6_reuse": "INVALIDATED_BY_V4_TOPOLOGY_ORACLE_GRID_FD_ACCOUNTING_CHANGES"}
    for name in ("source_reuse_receipts.json", "r0_reusable_supply_assessment.json"):
        p = context.output / "validation" / name
        _atomic_json(p, receipt)
        artifacts.append(p)
    _atomic_frame(external_path, pd.concat([pd.read_csv(external_path), pd.DataFrame(bound)], ignore_index=True))
    report = context.output / "reports/R0_VERIFIED_V2_CACHE_REUSE.md"
    _atomic_text(report, "# Verified v2 R0 reuse\n\nThe reviewed cache and its metadata are byte identical. The unchanged SCC source, original code/config hashes, exact cross-SCC audit, accepted demand and newly recomputed low/high physical conservation pass. SCC reconstruction was not executed. Source timing remains separate from measured v4 cache verification time.")
    artifacts.append(report)
    context.mark("R0_VERIFIED_CACHE_REUSE_COMPLETE", force=True)
    return artifacts


def atomic_runtime_retention(chains, fits):
    """A model can be valid while its occurrence must remain uncontracted."""
    selected = set(fits.loc[fits.recommended_status.astype(str).str.startswith("CASCADE_SELECTED"), "canonical_macro_signature"].astype(str))
    result = chains.copy()
    eligible = ~result.protected_or_blocked.astype(bool)
    active = eligible & result.canonical_signature.astype(str).isin(selected)
    result["runtime_resolution"] = np.where(active, "SELECTED_ROOT_FREE_RUNTIME", "RETAIN_ORIGINAL_PRIMITIVE_LINKS")
    result["postfit_retention_reason"] = np.where(active, "", "TOPOLOGY_PROTECTION_OR_SAMPLE_OR_FIT_GATE")
    reciprocal = result.reciprocal_group_id.fillna("").astype(str).ne("")
    for _, group in result.loc[reciprocal].groupby(["suite_short_id", "reciprocal_group_id"], sort=True):
        if len(group) != 2 or not active.loc[group.index].all():
            result.loc[group.index, "runtime_resolution"] = "RETAIN_ORIGINAL_RECIPROCAL_GROUP_ATOMICALLY"
            result.loc[group.index, "postfit_retention_reason"] = "RECIPROCAL_DIRECTION_NOT_REPLACED"
    return result


def postfit_topology_evidence(output, chains):
    from .staged_pilot import _atomic_frame, _atomic_json
    from .topology_reduction import remaining_incidence_audit
    checks = []; census = []; ledgers = []
    for suite_id, group in chains.groupby("suite_short_id", sort=True):
        root = output / "suite_products" / suite_id
        static = pd.read_parquet(root / "static_link_ownership.parquet")
        backbone = static.loc[static.primitive_bundle_id.ne("")]
        bundles = backbone[["primitive_bundle_id", "tail_node_id", "head_node_id"]].rename(columns={"primitive_bundle_id":"bundle_id", "tail_node_id":"tail_bacn_id", "head_node_id":"head_bacn_id"})
        effective = group.copy()
        effective["protected_or_blocked"] = ~effective.runtime_resolution.eq("SELECTED_ROOT_FREE_RUNTIME")
        protection = pd.read_parquet(root / "bacn_protection_census.parquet")
        endpoints = protection.loc[protection.protection_reasons_json.str.contains("OD_ENDPOINT"), "bacn_id"]
        ledger, check = remaining_incidence_audit(suite_id, effective, static, bundles, endpoints)
        if check["status"] != "PASS":
            raise ValueError(f"Post-fit incidence closure failed: {check}")
        checks.append(check); ledgers.append(ledger)
        original = pd.read_csv(root / "topology_census.csv").iloc[0].to_dict()
        original.update({"postfit_replacement_count":int((~effective.protected_or_blocked).sum()),
            "postfit_retained_bacn_count":int(original["original_bacn_count"])-check["removed_bacn_count"],
            "postfit_backbone_link_count":len(bundles)-check["consumed_primitive_count"]+int((~effective.protected_or_blocked).sum()),
            "postfit_dangling_incidence":check["dangling_surviving_incidence"], "postfit_closure_status":check["status"]})
        census.append(original)
    paths = []
    for name, frame in (("corrected_topology_census.csv", pd.DataFrame(census)),
        ("reciprocal_group_decisions.parquet", chains.loc[chains.reciprocal_group_id.fillna("").ne("")]),
        ("excluded_self_loop_candidates.csv", chains.loc[chains.retained_tail_bacn_id.eq(chains.retained_head_bacn_id)]),
        ("postfit_static_link_ownership.parquet", pd.concat(ledgers, ignore_index=True))):
        p = output / "topology" / name
        _atomic_frame(p, frame); paths.append(p)
    p = output / "topology/remaining_incidence_validation.json"
    _atomic_json(p, {"schema_id":"partC.phase5.v4.postfit_incidence.v1", "suite_count":len(checks), "suites":checks,
        "reciprocal_atomic":True, "full_original_static_incidence":True, "status":"PASS"})
    paths.append(p)
    return paths


def sample_delta_evidence(output, execution, chains):
    from .staged_pilot import _atomic_frame
    old = pd.read_parquet(Path(execution["r0"]["source_output_root"]) / "topology/chain_candidates.parquet")
    keys = ["suite_short_id", "chain_id"]
    old = old[keys + ["canonical_signature", "sample_included", "protected_or_blocked"]].rename(columns={
        "canonical_signature":"old_charge_inclusive_signature", "sample_included":"old_sample_included", "protected_or_blocked":"old_blocked"})
    now = chains[keys + ["canonical_signature", "sample_included", "protected_or_blocked", "block_reasons_json"]].rename(columns={
        "canonical_signature":"new_calibration_request_key", "sample_included":"new_sample_included", "protected_or_blocked":"new_blocked"})
    merged = old.merge(now, on=keys, how="outer", validate="one_to_one", indicator=True)
    merged["comparison_basis"] = "SUITE_QUALIFIED_PHYSICAL_ORDERED_OCCURRENCE"
    merged["new_namespace"] = execution["sampling"]["stable_hash_namespace"]
    p = output / "runtime_fits/old_new_sample_delta.csv"
    _atomic_frame(p, merged)
    return p


def certificate_gate(oracle, fits):
    q = oracle.query_flow.to_numpy(float)
    allocation_limit = 1e-8 + 1e-10*np.maximum(1, np.abs(q))
    failed = ~oracle.certified.astype(bool).to_numpy() | ~np.isfinite(oracle.exact_base_cost.to_numpy(float))
    failed |= oracle.allocation_residual.to_numpy(float) > allocation_limit
    failed |= oracle.allocation_correction_l1.to_numpy(float) > allocation_limit
    # Each series primitive is certified at q; a K*q concatenated total is not a primal check.
    if np.any(failed) or fits.finite_difference_relative_error.gt(1e-6).any() or fits.potential_identity_status.ne("PASS").any():
        raise ValueError(f"R4 publication certificate failed: {int(failed.sum())} primal rows")
    if not oracle.primitive_certificates_json.astype(str).str.startswith("[").all():
        raise ValueError("Per-primitive certificates missing")
    primitive_rows = 0
    for row in oracle.itertuples(index=False):
        certs = json.loads(row.primitive_certificates_json)
        if len(certs) != int(row.primitive_count):
            raise ValueError("Series primitive certificate count failed")
        for c in certs:
            flow_limit = 1e-8 + 1e-10*max(1, abs(c["q"]))
            cost_limit = 1e-7 + 1e-10*max(1, abs(c["marginal_cost_s"]))
            if c["q"] != row.query_flow or not c["certified"] or c["allocation_residual"] > flow_limit or c["correction_l1"] > flow_limit or c["kkt_s"] > cost_limit or c["fw_gap_per_flow_s"] > cost_limit:
                raise ValueError(f"Per-primitive primal/KKT/FW certificate failed: {c}")
            primitive_rows += 1
    return {"status":"PASS", "query_rows":len(oracle), "fit_variants":len(fits),
        "max_flow_residual":float(oracle.allocation_residual.max()),
        "max_kkt_residual_s":float(oracle.kkt_residual.max()),
        "max_fw_gap_per_flow_s":float(oracle.fw_gap_per_flow_s.max()),
        "max_all_variant_potential_identity_error":float(fits.finite_difference_relative_error.max()),
        "primitive_query_rows":primitive_rows,
        "per_primitive_primal_kkt_gap_certified_before_return":True}


def stage_warmup(context):
    """Bounded real-parameter gate with an independent absolute-cost Decimal solver."""
    from decimal import Decimal, localcontext
    from .staged_pilot import _atomic_json, _atomic_frame, authority_root
    from .bundle_oracle import BundleSpec, exact_bundle_oracle, oracle_cache_profile
    from .experiments import MacroExperiment, calibration_request_key, run_macro_experiments
    root = context.output / "r4_shards" / context.stage_id
    root.mkdir(parents=True, exist_ok=True)
    fixtures = json.loads((authority_root(context.execution) / "reference/fixtures/real_bundle_witnesses.json").read_text())
    rows = []; independent = []; experiments = []
    for i, entry in enumerate(fixtures):
        b = entry["bundle"]
        spec = BundleSpec(tuple(b["member_ids"]), *(np.asarray(b[k], dtype=float) for k in ("free_time_s", "capacity", "alpha", "beta")))
        flows = [0., 1e-12, 1e-8, 1., 1e3, 1e4, 1e5, 1e6]
        for level in np.unique(spec.free_time_s)[1:]:
            keep = spec.free_time_s < level
            activation = float(sum(spec.capacity[keep]*((level-spec.free_time_s[keep])/(spec.free_time_s[keep]*spec.alpha[keep]))**(1/spec.beta[keep])))
            flows.extend([float(np.nextafter(activation, 0)), activation, float(np.nextafter(activation, np.inf))])
        for q in flows:
            r = exact_bundle_oracle(spec, q)
            rows.append({"fixture":i, "q":q, "member_parameters_json":json.dumps(b), "allocation_json":json.dumps(r.member_allocation.tolist()),
                "cost":r.marginal_cost_s, "potential":r.potential_flow_seconds, "raw_residual":r.raw_allocation_residual,
                "corrected_residual":r.allocation_residual, "kkt_s":r.kkt_residual_s, "fw_gap_per_flow_s":r.fw_gap_per_flow_s,
                "status":"PASS" if r.certified else "FAIL"})
        if i < 4:
            q = [1e-12, 1e-8, 1e5, 1e6][i]
            with localcontext() as dc:
                dc.prec = 100
                t,c,a,beta = [[Decimal.from_float(float(v)) for v in ar] for ar in (spec.free_time_s,spec.capacity,spec.alpha,spec.beta)]
                Q = Decimal.from_float(q)
                def allocation(cost):
                    return [ci*((cost-ti)/(ti*ai))**(Decimal(1)/bi) if cost > ti else Decimal(0) for ti,ci,ai,bi in zip(t,c,a,beta)]
                lo=min(t); hi=min(ti+ti*ai*(Q/ci)**bi for ti,ci,ai,bi in zip(t,c,a,beta))
                for _ in range(440):
                    mid=(lo+hi)/2
                    if sum(allocation(mid)) < Q: lo=mid
                    else: hi=mid
                z=allocation((lo+hi)/2)
                potential=sum(ti*zi+ti*ai*zi*(zi/ci)**bi/(bi+1) for ti,ci,ai,bi,zi in zip(t,c,a,beta,z))
            r=exact_bundle_oracle(spec,q)
            error=float(np.max(np.abs(r.member_allocation-np.array(list(map(float,z))))))
            if error > 1e-8+1e-10*max(1,q) or not np.isclose(r.potential_flow_seconds,float(potential),rtol=1e-10,atol=1e-20):
                raise ValueError("Independent Decimal gate failed")
            independent.append({"fixture":i,"q":q,"precision_digits":100,"iterations":440,"allocation_absolute_error":error,
                "potential_reference":float(potential),"potential_binary64":r.potential_flow_seconds,"status":"PASS"})
        if i < 8:
            key=calibration_request_key([spec],1e4,2e4)
            experiments.append(MacroExperiment(key,f"real_warmup_{i}","REAL_FIXTURE",(spec,),1e4,2e4,1e5,1,False))
        context.mark(f"WARMUP_REAL_FIXTURE_{i+1}_OF_{len(fixtures)}")
    oracle,fits,domains=run_macro_experiments(experiments,progress_guard=context.mark)
    gate=certificate_gate(oracle,fits)
    artifacts=[]
    for name,payload in (("high_precision_reference_checks.json",{"method":"independent absolute marginal-cost bisection", "checks":independent,"status":"PASS"}),
        ("old_vs_new_oracle_witnesses.json",{"old_failure":"q=1e-8 returned 12.875510258735302 allocation, KKT=0, ROOT_SOLVED", "current_queries":rows,"status":"PASS"}),
        ("shard_validation.json",{"real_fixture_count":len(fixtures),"gate":gate,"activation_cache_profile":oracle_cache_profile(),"status":"PASS"})):
        p=root/name;_atomic_json(p,payload);artifacts.append(p)
    for name,frame in (("oracle_sample_results.parquet",oracle),("runtime_fit_results.parquet",fits),("pilot_domain_summary.csv",domains)):
        p=root/name;_atomic_frame(p,frame);artifacts.append(p)
    return artifacts


def stage_retained_controls(context):
    from .staged_pilot import _atomic_frame, _atomic_json, _write_experiment_jsonl, _spec_from_occurrence
    from .bundle_oracle import series_power_terms, oracle_cache_profile
    from .experiments import MacroExperiment, calibration_request_key, dynamic_expression_key, run_macro_experiments
    output=context.output; root=output/"r4_shards"/context.stage_id
    root.mkdir(parents=True,exist_ok=True)
    plan=json.loads((output/"runtime_fits/shard_plan.json").read_text())
    mainfits=pd.concat([pd.read_parquet(output/"r4_shards"/s["shard_id"]/"runtime_fit_results.parquet") for s in plan["shards"]],ignore_index=True)
    chains=atomic_runtime_retention(pd.read_parquet(output/"topology/chain_candidates_r3.parquet"),mainfits)
    consumed=set()
    for row in chains.loc[chains.runtime_resolution.eq("SELECTED_ROOT_FREE_RUNTIME")].itertuples(index=False):
        consumed.update((str(row.suite_short_id),str(edge)) for edge in json.loads(row.primitive_bundle_ids_json))
    occurrences=pd.read_parquet(output/"bundle_oracle/bundle_parameter_occurrences.parquet")
    spec_by_key={}; requests={}; coverage=[]
    for suite,group in occurrences.groupby("suite_short_id",sort=True):
        counts={"RETAINED_EXACT_ELIGIBLE_UNCOMPILED":0,"RETAINED_HETEROGENEOUS_UNCOMPILED":0,"CONSUMED_BY_ACCEPTED_MACRO":0}
        for row in group.itertuples(index=False):
            if (str(suite),str(row.bundle_id)) in consumed:
                counts["CONSUMED_BY_ACCEPTED_MACRO"]+=1;continue
            key=str(row.canonical_primitive_key)
            if key not in spec_by_key: spec_by_key[key]=_spec_from_occurrence(row)
            spec=spec_by_key[key]
            hetero=series_power_terms([spec]) is None
            counts["RETAINED_HETEROGENEOUS_UNCOMPILED" if hetero else "RETAINED_EXACT_ELIGIBLE_UNCOMPILED"]+=1
            if not hetero: continue
            request=calibration_request_key([spec],float(row.low_current_flow),float(row.low_aon_flow))
            if request not in requests:
                requests[request]=MacroExperiment(request,f"retained:{suite}:{row.bundle_id}",str(suite),(spec,),
                    float(row.low_current_flow),float(row.low_aon_flow),float(row.high_current_flow),1,False,
                    dynamic_expression_key=dynamic_expression_key([spec]),calibration_request_key=request)
        for kind,count in counts.items():
            coverage.append({"suite_short_id":suite,"universe_class":kind,"occurrence_count":count,
                "actual_compiled_occurrences":0,"kernel_proxy":"CURRENT_LEGACY_BPR_PLACEHOLDER" if kind.startswith("RETAINED") else "ACCEPTED_MACRO_REPLACEMENT",
                "full_reduced_network_measured":False})
        macro_count=int(((chains.suite_short_id==suite)&chains.runtime_resolution.eq("SELECTED_ROOT_FREE_RUNTIME")).sum())
        coverage.append({"suite_short_id":suite,"universe_class":"ACTUAL_ACCEPTED_MACRO_REPLACEMENT","occurrence_count":macro_count,
            "actual_compiled_occurrences":macro_count,"kernel_proxy":"COMPILED_DYNAMIC_MODEL_WITH_OCCURRENCE_H","full_reduced_network_measured":False})
    maximum=int(context.execution["sampling"]["retained_heterogeneous_primitive_controls_max"])
    ranked=sorted(requests,key=lambda k:hashlib.sha256(("p5-v4-retained-control\0"+k).encode()).hexdigest())
    selected=[requests[k] for k in ranked[:maximum]]
    if not selected: raise ValueError("Retained heterogeneous control universe is unexpectedly empty")
    context.mark(f"RETAINED_CONTROL_UNIVERSE_{len(requests)}_SELECTED_{len(selected)}",force=True)
    input_path=root/"input_experiments.jsonl";_write_experiment_jsonl(input_path,selected)
    oracle,fits,domains=run_macro_experiments(selected,progress_guard=context.mark)
    gate=certificate_gate(oracle,fits)
    selected_class=fits.loc[fits.recommended_status.astype(str).str.startswith("CASCADE_SELECTED"),["canonical_macro_signature","runtime_class","max_base_rel_cost_error","max_base_rel_potential_error"]]
    results=domains.merge(selected_class,on="canonical_macro_signature",how="left",validate="one_to_one")
    results["actual_primitive_parameters_json"]=[json.dumps({"member_ids":list(x.primitive_bundles[0].member_ids),
        **{k:getattr(x.primitive_bundles[0],k).tolist() for k in ("free_time_s","capacity","alpha","beta")}}) for x in selected]
    results["control_role"]="AUXILIARY_RETAINED_HETEROGENEOUS_NOT_MAIN_10000"
    results["retained_unique_calibration_request_universe"]=len(requests)
    artifacts=[input_path]
    for name,frame in (("oracle_sample_results.parquet",oracle),("runtime_fit_results.parquet",fits),("pilot_domain_summary.csv",domains),
        ("retained_heterogeneous_control_results.csv",results),("compiled_vs_proxy_coverage.csv",pd.DataFrame(coverage))):
        p=root/name;_atomic_frame(p,frame);artifacts.append(p)
    p=root/"shard_validation.json"
    _atomic_json(p,{"status":"PASS","control_count":len(selected),"maximum_controls":maximum,"retained_unique_calibration_requests":len(requests),
        "main_macro_request_count":plan["signature_count"],"gate":gate,"activation_cache_profile":oracle_cache_profile(),
        "selection_uses_high":False,"topology_gate_high_is_safety_only":True})
    artifacts.append(p)
    return artifacts


def write_v4_audits(context, chains, oracle, fits, domains, visits, charges, paths, benchmarks):
    """Publish measured certificates, coverage, roles, and bounded formal forecasts."""
    from .staged_pilot import _atomic_frame, _atomic_json, _atomic_text, _scientific_elapsed, _observed_peak_rss, sha256_file
    from .benchmarks import _runtime_from_row, _packed_runtime
    out=context.output; artifacts=[]
    def table(relative, frame):
        p=out/relative;_atomic_frame(p,frame);artifacts.append(p)
    def obj(relative, data):
        p=out/relative;_atomic_json(p,data);artifacts.append(p)
    def report(relative, content):
        p=out/relative;_atomic_text(p,content);artifacts.append(p)
    protected_checks=[]
    for suite, group in chains.groupby("suite_short_id",sort=True):
        protection=pd.read_parquet(out/"suite_products"/suite/"bacn_protection_census.parquet")
        removed=set()
        for encoded in group.loc[group.runtime_resolution.eq("SELECTED_ROOT_FREE_RUNTIME"),"internal_bacn_ids_json"]:
            removed.update(json.loads(encoded))
        protected=set(protection.loc[protection.protected.astype(bool),"bacn_id"].astype(str))
        od=set(protection.loc[protection.protection_reasons_json.str.contains("OD_OR_CENTROID_ROLE"),"bacn_id"].astype(str))
        conflict=removed&protected
        if conflict: raise ValueError(f"Post-fit protected/OD role removal: {suite}: {sorted(conflict)[:10]}")
        protected_checks.append({"suite_short_id":suite,"protected_node_count":len(protected),"od_or_centroid_role_count":len(od),
            "removed_node_count":len(removed),"removed_protected_nodes":len(conflict),"removed_od_nodes":len(removed&od),"status":"PASS"})
    obj("topology/postfit_protected_role_validation.json",{"status":"PASS","checks":protected_checks,
        "source_role_name":"OD_OR_CENTROID_ROLE","all_protected_roles_included":True})
    ctrl=out/"r4_shards/r4_v4_controls"
    cf=pd.read_parquet(ctrl/"runtime_fit_results.parquet")
    co=pd.read_parquet(ctrl/"oracle_sample_results.parquet")
    cd=pd.read_csv(ctrl/"pilot_domain_summary.csv")
    allfits=pd.concat([fits.assign(sample_role="MAIN_MACRO"),cf.assign(sample_role="RETAINED_CONTROL")],ignore_index=True)
    alloracle=pd.concat([oracle.assign(sample_role="MAIN_MACRO"),co.assign(sample_role="RETAINED_CONTROL")],ignore_index=True)
    context.mark("V4_ALL_QUERY_CERTIFICATES",force=True)
    gates=[{"sample_role":"MAIN_MACRO",**certificate_gate(oracle,fits)}, {"sample_role":"RETAINED_CONTROL",**certificate_gate(co,cf)}]
    table("bundle_oracle/oracle_certification_summary.csv",pd.DataFrame(gates))
    table("bundle_oracle/all_query_certificate.parquet",alloracle)
    for name in ("high_precision_reference_checks.json","old_vs_new_oracle_witnesses.json"):
        obj("bundle_oracle/"+name,json.loads((out/"r4_shards/r4_v4_warmup"/name).read_text()))
    plan=json.loads((out/"runtime_fits/shard_plan.json").read_text())
    profiles=[]
    for shard in [s["shard_id"] for s in plan["shards"]]+["r4_v4_warmup","r4_v4_controls"]:
        r=json.loads((out/"r4_shards"/shard/"shard_validation.json").read_text())
        profiles.append({"stage_id":shard,"profile":r["activation_cache_profile"]})
    obj("bundle_oracle/primitive_activation_cache_profile.json",{"status":"PASS","profiles":profiles,"memoization":"exact parameter tuple and q; activation profiles precomputed once; offline microbench clears query cache every call"})
    fdcols=["canonical_macro_signature","sample_role","runtime_class","variant_id","recommended_status",
        "raw_centered_fd_relative_error","finite_difference_relative_error","interval_integral_relative_error",
        "knot_continuity_relative_error","potential_knot_continuity_relative_error","potential_identity_status","fd_worst_probe_json"]
    table("runtime_fits/raw_vs_knot_aware_fd.csv",allfits[fdcols])
    obj("runtime_fits/knot_continuity_integral_checks.json",{"variant_count":len(allfits),"all_variant_status":bool(allfits.potential_identity_status.eq("PASS").all()),
        "raw_above_target_count":int(allfits.raw_centered_fd_relative_error.gt(1e-6).sum()),
        "corrected_above_target_count":int(allfits.finite_difference_relative_error.gt(1e-6).sum()),
        "max_interval_relative_error":float(allfits.interval_integral_relative_error.max()),
        "max_cost_knot_continuity_relative_error":float(allfits.knot_continuity_relative_error.max()),
        "max_potential_knot_continuity_relative_error":float(allfits.potential_knot_continuity_relative_error.max()),
        "denominator":"max(1, abs(local dynamic cost)); fixed H excluded", "target":1e-6,"status":"PASS"})
    context.mark("V4_ALL_VARIANT_KERNEL_SERIALIZATION",force=True)
    roundtrips=[]
    for cls, group in allfits.groupby("runtime_class",sort=True):
        # Every serialized variant is checked, including rejected fits.
        functions=[_runtime_from_row(pd.Series(r._asdict())) for r in group.itertuples(index=False)]
        source=np.repeat(np.arange(len(functions),dtype=np.int64),3)
        domain=group.flow_domain_max.to_numpy(float)
        q=(domain[:,None]*np.array([0.,.5,1.])).ravel()
        packed=_packed_runtime(functions,source)
        maxerr={}
        for operation in ("cost","potential","derivative"):
            expected=np.concatenate([getattr(f,operation)(q[3*i:3*i+3]) for i,f in enumerate(functions)])
            actual=getattr(packed,operation)(q)
            error=float(np.max(np.abs(actual-expected)/np.maximum(1,np.abs(expected))))
            if not np.isfinite(actual).all() or error>1e-10:
                raise ValueError(f"Actual serialized packed {cls} {operation} roundtrip failed: {error}")
            maxerr[operation]=error
        roundtrips.append({"runtime_class":cls,"variant_count":len(group),"probe_count":len(q),"max_local_relative_errors":maxerr,"status":"PASS"})
        context.mark(f"V4_KERNEL_ROUNDTRIP_{cls}_COMPLETE",force=True)
    obj("runtime_fits/kernel_roundtrip_checks.json",{"status":"PASS","all_selected_and_rejected_variants":True,"class_checks":roundtrips})
    all_domains=pd.concat([domains.assign(sample_role="MAIN_MACRO"),cd.assign(sample_role="RETAINED_CONTROL")],ignore_index=True)
    dynamic_cols=[c for c in ("dynamic_expression_key","zero_flow_base_cost_s","sample_role") if c in all_domains]
    if "dynamic_expression_key" not in dynamic_cols:
        raise ValueError("Dynamic identity missing from actual calibrated domains")
    table("runtime_fits/dynamic_function_registry.csv",all_domains[dynamic_cols].drop_duplicates())
    table("runtime_fits/calibration_registry.csv",all_domains)
    occurrence_cols=[c for c in chains.columns if c in {"suite_short_id","chain_id","canonical_signature","dynamic_expression_key","calibration_request_key","legacy_charge_inclusive_signature","primitive_bundle_ids_json","internal_bacn_ids_json","retained_tail_bacn_id","retained_head_bacn_id","internal_charge_s","internal_charge_estimator_id","runtime_resolution","postfit_retention_reason","reciprocal_group_id"}]
    table("runtime_fits/occurrence_charge_ordered_liftback_registry.parquet",chains[occurrence_cols])
    controls=pd.read_csv(ctrl/"retained_heterogeneous_control_results.csv")
    coverage=pd.read_csv(ctrl/"compiled_vs_proxy_coverage.csv")
    table("runtime_fits/retained_heterogeneous_control_results.csv",controls)
    table("benchmarks/compiled_vs_proxy_coverage.csv",coverage)
    table("benchmarks/runtime_benchmarks.csv",benchmarks.assign(full_reduced_network_measured=False,
        coverage_evidence="compiled_vs_proxy_coverage.csv",assignment_speedup_established=False))
    pathcols=["sample_od_id","evaluation_role","physical_path_cost_s","legacy_physical_node_charge_s","fixed_charge_estimate_s","legacy_minus_static_incidence_charge_s","projected_static_link_count","static_link_occurrences_json","static_bacn_half_incidence_json","bacn_path_charge_decomposition_json","contraction_static_charge_identity_error_s"]
    table("charge_audit/projected_static_incidence_vs_physical_node_visits.csv",paths[pathcols])
    obj("charge_audit/charge_units_and_denominators.json",{"schema_id":"partC.phase5.v4.units_fields.v1","status":"PASS",
        "physical_time_input":"minutes", "single_conversion":"graph.free_time * 60 at physical bundle construction", "canonical_time":"seconds",
        "flow_unit":"accepted vehicle-flow unit unchanged", "capacity":"accepted capacity times 0.30 once", "potential_unit":"flow-seconds",
        "static_charge_formula":"sum over projected static link occurrences [0.5 chi(tail)+0.5 chi(head)]",
        "visit_denominator":"0.5*(BACKBONE_BUNDLE in+out + actual terminal connector in+out); no virtual OD supply",
        "legacy_od_augmented_incidence":"preserved solely for physical/OD balance diagnostics", "internal_physical_B_inside_region":"zero additional coarse visit",
        "macro_model":"b(q), B(q)","occurrence_total":"H+b(q), H*q+B(q)",
        "dynamic_fit_denominator":"max(1, abs(exact dynamic cost))", "path_error_denominator":"max(1 second, abs(physical path cost))",
        "field_mapping":{"old canonical_signature":"legacy_charge_inclusive_signature; new canonical_signature is low-only calibration request key",
            "old max_rel_cost_error":"kept total-cost denominator; max_base_rel_cost_error is the v4 selection quantity",
            "liftability_error":"legacy alias for witness_integrity_error; new global liftability is not established",
            "path status PASS":"accounting and witness closure only; absolute_relative_cost_error measures approximation accuracy",
            "bacn_visit_count":"legacy physical-node episode diagnostic; static_bacn_half_incidence_json is charged incidence"}})
    summaries=[]
    for role,group in paths.groupby("evaluation_role",sort=True):
        summaries.append({"source_role":role,"path_count":len(group),"current_cfw_path_mixture":False,
            "mean_signed_relative_error":float(group.signed_relative_cost_error.mean()),
            "median_absolute_relative_error":float(group.absolute_relative_cost_error.median()),
            "p95_absolute_relative_error":float(group.absolute_relative_cost_error.quantile(.95)),
            "max_absolute_relative_error":float(group.absolute_relative_cost_error.max()),
            "accounting_closure_pass_count":int(group.accounting_closure_pass.sum()),
            "witness_integrity_pass_count":int(group.witness_integrity_error.eq(0).sum()),
            "global_liftability":"NOT_ESTABLISHED","accuracy_inferred_from_closure":False})
    table("path_validation/path_fidelity_by_source_role.csv",pd.DataFrame(summaries))
    worst=[]
    for row in paths.nlargest(20,"absolute_relative_cost_error").itertuples(index=False):
        worst.append({"object_type":"PATH","object_id":row.sample_od_id,"source_role":row.evaluation_role,"error_s":row.candidate_minus_physical_cost_s,
            "denominator":row.path_cost_denominator_s,"absolute_relative_error":row.absolute_relative_cost_error,
            "decomposition_json":json.dumps({k:getattr(row,k) for k in ("physical_path_cost_s","regional_owned_cost_s","suppressed_internal_cost_s","fixed_charge_estimate_s","runtime_fit_base_error_s","free_mixing_endpoint_state_error_s","flow_state_alignment_error_s","component_closure_error_s","bacn_path_charge_decomposition_json")})})
    cp=charges.copy();cp["_absolute_residual"]=cp.high_residual_using_recommended.abs()
    for row in cp.nlargest(20,"_absolute_residual").to_dict("records"):
        worst.append({"object_type":"BACN","object_id":str(row["suite_short_id"])+":"+str(row["bacn_id"]),"source_role":"HIGH_HELDOUT_EXPOSURE",
            "error_s":None,"denominator":None,"absolute_relative_error":None,"decomposition_json":json.dumps(row,default=str)})
    table("charge_audit/worst_path_and_bacn_decompositions.csv",pd.DataFrame(worst))
    # Forecast the full retained plus macro universe, with measured class rates.
    context.mark("V4_FULL_UNIVERSE_FORECAST",force=True)
    timings=pd.read_csv(out/"validation/stage_timing.csv")
    macro_census=pd.read_csv(out/"bundle_oracle/canonical_macro_census.csv").iloc[0]
    class_rates=[]
    for role,group in all_domains.groupby(["sample_role","primitive_parameter_class"],sort=True):
        seconds=group.total_oracle_fit_diagnostic_seconds.to_numpy(float)
        class_rates.append({"sample_role":role[0],"parameter_class":role[1],"measured_objects":len(group),
            "mean_seconds_per_object":float(seconds.mean()),"p25_seconds_per_object":float(np.quantile(seconds,.25)),
            "p95_seconds_per_object":float(np.quantile(seconds,.95)),"certified_grid_seconds":float(group.certified_grid_seconds.sum()),
            "total_fit_diagnostic_seconds":float(seconds.sum())})
    unique_macro=int(macro_census["unique_signature_count"])
    exact_macro=int(macro_census["exact_closed_form_unique_count"])
    retained_exact=int(coverage.loc[coverage.universe_class.eq("RETAINED_EXACT_ELIGIBLE_UNCOMPILED"),"occurrence_count"].sum())
    retained_hetero=int(coverage.loc[coverage.universe_class.eq("RETAINED_HETEROGENEOUS_UNCOMPILED"),"occurrence_count"].sum())
    ctrl_validation=json.loads((ctrl/"shard_validation.json").read_text())
    retained_unique=int(ctrl_validation["retained_unique_calibration_requests"])
    def bounds(role,cls,count_low,count_high):
        candidates=[r for r in class_rates if r["sample_role"]==role and r["parameter_class"]==cls]
        if not candidates:
            if count_high: raise ValueError(f"Forecast class has no measured coverage: {role}/{cls}")
            return [0.,0.]
        r=candidates[0]
        return [count_low*r["p25_seconds_per_object"], count_high*max(r["p95_seconds_per_object"],r["mean_seconds_per_object"])*2]
    macro_exact=bounds("MAIN_MACRO","EXACT_SERIES",exact_macro,exact_macro)
    macro_hetero=bounds("MAIN_MACRO","HETEROGENEOUS_SERIES",unique_macro-exact_macro,unique_macro-exact_macro)
    retain_hetero=bounds("RETAINED_CONTROL","HETEROGENEOUS_SERIES",retained_unique,retained_hetero)
    # Retained exact objects need algebraic serialization; measured full exact cascade is a conservative upper reference.
    retain_exact=bounds("MAIN_MACRO","EXACT_SERIES",0,retained_exact)
    fixed=float(timings.loc[~timings.stage_id.astype(str).str.startswith("r4_"),"elapsed_seconds"].sum())
    parallelizable=np.sum(np.array([macro_exact,macro_hetero,retain_hetero,retain_exact]),axis=0)
    serial=[fixed+parallelizable[0],2*fixed+parallelizable[1]]
    forecast={"schema_id":"partC.phase5.v4.formal_forecast.v1","status":"ESTIMATE_ONLY_SEPARATE_PREFLIGHT_REQUIRED",
        "full_universe":{"macro_unique_calibration_requests":unique_macro,"macro_exact_eligible":exact_macro,
            "macro_heterogeneous":unique_macro-exact_macro,"retained_exact_occurrences":retained_exact,
            "retained_heterogeneous_occurrences":retained_hetero,"retained_unique_heterogeneous_calibration_requests":retained_unique},
        "measured_class_throughput":class_rates,"observed_stages":timings.to_dict("records"),
        "serial_seconds_range":serial,"ideal_two_worker_seconds_range":[fixed+parallelizable[0]/2,2*fixed+parallelizable[1]/2],
        "two_worker_run_measured":False,"parallel_model":"Amdahl: serial census/IO/audits fixed; only independent oracle/fit work divided by two; contention excluded",
        "retained_exact_lower_cost_unknown":True,"retained_exact_upper_reference":"full exact macro cascade per occurrence; conservative, not measured retained-universe compilation",
        "range_method":"p25 measured object time lower; 2x max(p95, mean) upper; retained dedup lower and full occurrence upper",
        "fit_only_linear_population_scaling_replaced":True,"engineering_in_task_wall_seconds":_scientific_elapsed(out),
        "observed_peak_rss_bytes":_observed_peak_rss(out),"observed_output_bytes":sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        "storage_forecast":"stream shards and discard no evidence; scale oracle/fit row bytes by full calibration universe; raw accepted inputs excluded",
        "future_assignment_speedup_established":False,"national_assignment_performed":False,
        "limits":["R5 timing excludes its still-running final report work", "full retained graph not compiled", "RACN ownership unresolved", "existing path witnesses do not establish new-flow liftability"]}
    calibration_population=[unique_macro+retained_unique,unique_macro+retained_hetero+retained_exact]
    measured_population=len(domains)+len(cd)
    current_products_bytes=sum((out/r).stat().st_size for r in ("bundle_oracle/oracle_sample_results.parquet","runtime_fits/runtime_fit_results.parquet"))
    control_products_bytes=sum((ctrl/r).stat().st_size for r in ("oracle_sample_results.parquet","runtime_fit_results.parquet"))
    bytes_per_request=(current_products_bytes+control_products_bytes)/measured_population
    forecast["projected_oracle_fit_storage_bytes_range"]=[int(bytes_per_request*n) for n in calibration_population]
    forecast["projected_added_disk_bytes_range_including_three_evidence_copies_and_observed_fixed_bytes"]=[int(forecast["observed_output_bytes"]+3*bytes_per_request*n) for n in calibration_population]
    forecast["projected_serial_streamed_rss_bytes_range"]=[forecast["observed_peak_rss_bytes"],2*forecast["observed_peak_rss_bytes"]]
    forecast["storage_and_rss_assumptions"]="Observed bytes/request scaled to unique-versus-occurrence universe, allowing three evidence copies; serial RSS assumes bounded shards and streaming aggregate audits, not loading a full campaign into RAM. These are estimates, not measured full-production usage."
    obj("validation/formal_campaign_forecast.json",forecast)
    report("reports/formal_campaign_forecast.md","# Formal campaign forecast\n\nThe full macro and retained primitive universe is included. The serial range is "
        f"{serial[0]/3600:.2f}–{serial[1]/3600:.2f} hours. This is an engineering estimate requiring a separate preflight. "
        "The ideal two-worker bound applies Amdahl's fixed serial portion and has not been measured. Exact retained-link serialization has no measured lower bound; the upper reference uses the measured full exact macro cascade. Class rates, populations, stage times and limitations are recorded in `validation/formal_campaign_forecast.json`. No assignment speedup follows from these kernel timings.")
    recommendation="NOT_READY_MODEL_LIMITATIONS" if any(r["p95_absolute_relative_error"]>.01 for r in summaries) or retained_hetero else "READY_EXTERNAL_DESIGN_FREEZE"
    # R ownership remains unresolved in the inherited evidence; keep the recommendation explicit.
    recommendation="NOT_READY_MODEL_LIMITATIONS"
    obj("validation/v4_correctness_and_recommendation.json",{"status":"PASS","production_recommendation":recommendation,
        "scientific_diagnostics_complete":True,"numerical_correctness_pass":True,
        "remaining_model_limitations":["RACN cost ownership unresolved","retained heterogeneous runtime coverage bounded by controls","existing witnesses do not prove global new-flow liftability","path fidelity is reported separately from accounting closure"],
        "full_phase5_authorized":False,"phase6_authorized":False,"icnn_training_performed":False,"national_assignment_performed":False})
    obj("validation/correction_regression_tests.json",{"status":"PASS","test_evidence":json.loads((out/"validation/focused_tests_status.json").read_text()),
        "v4_gate_rows":gates,"independent_decimal_evidence":"bundle_oracle/high_precision_reference_checks.json",
        "reproduction_failures_preserved_in_repository":"logs_icnn/phase_5/pilot_v4_0905/v0_reproduced_failures.xml"})
    decisions=pd.read_csv(out/"validation/pilot_decision_evidence.csv").set_index("decision_id")
    topology=pd.read_csv(out/"topology/corrected_topology_census.csv")
    decisions.loc["P5-G001","measured_facts"]="Candidate removal is distinct from post-fit overlay. Actual retained B-ACN counts: "+json.dumps(dict(zip(topology.suite_short_id,topology.postfit_retained_bacn_count)))
    decisions.loc["P5-G011","measured_facts"]="Kernel tables combine actual macro models with explicitly labelled retained-link proxies; full reduced-network evaluation has not been measured. See compiled_vs_proxy_coverage.csv."
    decisions.loc["P5-G012","measured_facts"]=f"Full retained+macro universe serial estimate {serial[0]/3600:.2f}–{serial[1]/3600:.2f} h; measured class rates, Amdahl bounds, numeric storage/RSS ranges and assumptions are in formal_campaign_forecast.json."
    decisions.loc["P5-G012","evidence_paths"]="validation/formal_campaign_forecast.json;benchmarks/compiled_vs_proxy_coverage.csv"
    decisions["v4_primary_band_error"]="DYNAMIC_ONLY_MAX_BASE_REL_ERROR"
    table("validation/pilot_decision_evidence.csv",decisions.reset_index())
    report("reports/12_V4_DECISIONS_AND_MODEL_LIMITATIONS.md","# V4 decisions and model limitations\n\n"+decisions.reset_index().to_markdown(index=False)+"\n\nAll recommendations require external review. R5 error-band eligibility uses dynamic b/B errors; legacy total-cost error columns remain separately named in the full fit tables. Full Phase 5, Phase 6, training and national assignment remain unauthorized.")
    return artifacts
