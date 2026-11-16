"""Portable fresh-data entry points for the final P-ACNHN manuscript.

Scientific numerical primitives are imported from the shipped, unchanged source.
This adapter replaces historical campaign orchestration, clocks and path bindings.
It never searches for old labels, checkpoints, authority ZIPs or source runs.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import psutil
from partC_rev1_src.pacnhn_v1.control import array_hash, read_json, sha, write_json, save_npz

SUITES = ("s02", "s03", "s04", "s05", "s06")
FAMILIES = ("FFTT", "POWER4", "ICNN", "XGBOOST")


def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (PACKAGE / p).resolve()


class Run:
    """Explicit fresh-run clock, hash binding and process-tree resource guard."""
    def __init__(self, config, action):
        self.path = Path(config).resolve(strict=True)
        self.c = read_json(self.path)
        if "REPRODUCTION_RUN_ID" in os.environ:
            self.c["run_id"] = os.environ["REPRODUCTION_RUN_ID"]
        required = ("phase_id", "run_id", "tasks", "input_versions", "expected_outputs", "acceptance_tests", "resources", "run_root")
        if any(not self.c.get(k) for k in required) or action not in self.c["tasks"]:
            raise ValueError("Explicit nonempty configuration and authorized task required")
        self.root = resolve(self.c["run_root"])
        self.root.mkdir(parents=True, exist_ok=True)
        self.action = action
        self.started = time.time()
        self.bindings = {}
        self.peak_rss = 0
        self.minimum_available = 2**63 - 1
        clock = self.root / "REPRODUCTION_CLOCK.json"
        if not clock.exists():
            write_json(clock, dict(phase_id=self.c["phase_id"], run_id=self.c["run_id"],
                start_epoch=self.started, deadline_epoch=self.started+self.c["resources"]["total_wall_seconds"],
                baseline_free_bytes=shutil.disk_usage(self.root).free))
        self.clock = read_json(clock)
        if self.clock["run_id"] != self.c["run_id"] or self.clock["phase_id"] != self.c["phase_id"]:
            raise ValueError("Run identity changed: choose a new run_root and run_id")
        self.deadline = min(self.clock["deadline_epoch"], self.started+self.c["action_seconds"][action])
        # Native support and witness construction accept this minimal context.
        self.c["maximum_cumulative_stage_deadlines"] = [self.deadline]
        self.c["topology"] = dict(expected_backbone_count=self.c["expected_backbone_count"])
        self.c["resources"]["regional_workers"] = self.c["workers"]
        self.c["suite_ids"] = list(SUITES)
        self.bind(self.path, "CURRENT_CONFIG")
        write_json(self.root / "reproduction_logs" / action / "INTENT.json", dict(
            phase_id=self.c["phase_id"], step_id=action, run_id=self.c["run_id"],
            tasks=[action], config_sha256=sha(self.path), start_epoch=self.started,
            deadline_epoch=self.deadline, input_versions=self.c["input_versions"],
            expected_outputs=self.c["expected_outputs"], acceptance_tests=self.c["acceptance_tests"],
            resources=self.c["resources"]))

    def bind(self, path, role, expected=None):
        p = Path(path).resolve(strict=True)
        digest = sha(p)
        if expected is not None and digest != expected:
            raise ValueError("Current input hash differs: " + str(p))
        prior = self.bindings.get(str(p))
        if prior and prior["sha256"] != digest:
            raise ValueError("Bound input changed during action")
        self.bindings[str(p)] = dict(path=str(p), sha256=digest, bytes=p.stat().st_size, role=role)
        return digest

    def guard(self, deadline=None):
        now = time.time()
        if now >= min(self.deadline, deadline if deadline is not None else self.deadline):
            raise TimeoutError("Explicit fresh-run wall budget reached")
        process = psutil.Process()
        rss = process.memory_info().rss
        for child in process.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        available = psutil.virtual_memory().available
        self.peak_rss = max(self.peak_rss, rss)
        self.minimum_available = min(self.minimum_available, available)
        r = self.c["resources"]
        if rss > r["max_rss_bytes"] or available < r["min_available_bytes"]:
            raise MemoryError("Declared RSS/free-RAM resource guard reached")
        free = shutil.disk_usage(self.root).free
        if free < r["min_free_disk_bytes"] or self.clock["baseline_free_bytes"]-free > r["max_added_disk_bytes"]:
            raise OSError("Declared free-disk/added-disk resource guard reached")

    def flush_bindings(self):
        write_json(self.root / "reproduction_logs" / self.action / "INPUT_MANIFEST.json", list(self.bindings.values()))

    def finish(self, result):
        self.flush_bindings()
        payload = dict(result, phase_id=self.c["phase_id"], step_id=self.action, run_id=self.c["run_id"],
            elapsed_seconds=time.time()-self.started, peak_process_tree_RSS_bytes=self.peak_rss,
            minimum_available_RAM_bytes=self.minimum_available,
            measured_disk_free_decrease_bytes=max(0, self.clock["baseline_free_bytes"]-shutil.disk_usage(self.root).free),
            next_step_authorized=False)
        write_json(self.root / "reproduction_logs" / self.action / "RESULT.json", payload)
        return payload


def support_rows(ctx):
    path = ctx.root / "SUPPORT_FREEZE.json"
    ctx.bind(path, "CURRENT_SUPPORT_FREEZE")
    support = read_json(path)
    if support.get("status") != "PASS" or {x["suite_id"] for x in support["suites"]} != set(SUITES):
        raise ValueError("Fresh five-suite support freeze required")
    rows = sorted((r for s in support["suites"] for r in s["region_records"]),
                  key=lambda r: (r["suite_id"] != "s03", r["suite_id"], r["region_index"]))
    if not rows:
        raise ValueError("No-op support is forbidden")
    return support, rows


def reference(ctx):
    from partC_rev1_src.pacnhn_v1.topology import load_common
    from partC_rev1_src.pacnhn_v1.solver import PolynomialObjective, solve
    graph, od, a, p = load_common(ctx.root)
    for filename in ("common_graph.npz", "common_od.npz", "partitions.npz", "RESULT.json"):
        ctx.bind(ctx.root / "topology" / filename, "CURRENT_COMMON_P_TOPOLOGY")
    reports = []
    for scenario in ("open", "i95"):
        ctx.guard()
        target = ctx.root / "references" / scenario
        if target.exists():
            raise FileExistsError("Reference already exists; use a new run root, preserve original")
        started = time.time()
        result = solve(graph, od, PolynomialObjective(a["base"], a["k4"]), target,
            method="CFW", run_start_epoch=started,
            deadline_epoch=min(ctx.deadline, started+ctx.c["reference"]["per_scenario_seconds"]),
            tolerance=ctx.c["reference"]["relative_gap"], ctx=ctx,
            enabled=~a["closed_mask"] if scenario == "i95" else np.ones(len(graph.tail), bool),
            edge_regions=p["edge_region"], workers=ctx.c["workers"], capture_witness=True)
        reports.append(dict(scenario=scenario, **result))
        if not result["final_certificate_fresh"] or result["status"] != "CONVERGED":
            raise ValueError("P reference fails fresh certificate or required 1e-4 accuracy gate; partial evidence retained")
    return dict(status="COMPLETE", reports=reports,
        accuracy_targets_met=all(x["status"] == "CONVERGED" for x in reports))


def support(ctx):
    from partC_rev1_src.pacnhn_v1.support import build
    from partC_rev1_src.pacnhn_v1.witnesses import audit
    suites = build(ctx)
    witness = audit(ctx)
    return dict(status="COMPLETE", suites=[dict(suite_id=s["suite_id"],
        regions=s["active_regions"], movements=s["active_pairs"]) for s in suites], witness=witness)


def region_root(ctx, record):
    return ctx.root / "training" / "regions" / record["suite_id"] / f'{record["region_index"]:06d}'


def load_graph(ctx, record):
    from partC_rev1_src.pacnhn_v1.regional import load_region_graph
    ctx.bind(record["path"], "CURRENT_REGION_GRAPH", record["sha256"])
    return load_region_graph(record["path"], expected_sha256=record["sha256"])


def scale_for(region, record):
    from partC_rev1_src.pacnhn_v1.training import native_scale
    return native_scale(region, record, dict(native_scale_positive_median_fraction=.01))[0]


def labels(ctx):
    from partC_rev1_src.pacnhn_v1.regional import sample_requests, generate_labels
    _, records = support_rows(ctx)
    records=[r for r in records if r["suite_id"]=="s03"]
    label_cfg = ctx.c["labels"]
    rows = []
    for record in records:
        ctx.guard()
        region = load_graph(ctx, record)
        root = region_root(ctx, record) / "labels"
        if (root / "REQUEST_INDEX.json").exists():
            raise FileExistsError("Labels already exist; use retry-labels to preserve immutable roles")
        requests = []
        existing = set()
        for role in ("TRAIN", "VALID", "TEST"):
            added = sample_requests(region, role=role, count=label_cfg["counts"][role], seed=label_cfg["seed"],
                reference_scale=scale_for(region, record), existing_input_ids=existing)
            requests.extend(added)
            existing.update(r["input_id"] for r in added)
        # generate_labels freezes EVERY role/input before accessing its first label.
        config = dict(phase_id="PACNHN_V1", run_id=ctx.c["run_id"],
            region_id=f'{record["suite_id"]}:{record["region_index"]:06d}', graph_path=record["path"],
            graph_sha256=record["sha256"], deadline_epoch=ctx.deadline,
            per_label_seconds=label_cfg["attempt_seconds"][0], max_iterations=20000,
            relative_gap_tolerance=1e-6, solver_method="CFW", retry_solver_method="CFW",
            cfw_reset_on_bound=True, retry_cfw_reset_on_bound=True,
            iterate_selection="BEST_CERTIFIED_RG", retry_iterate_selection="BEST_CERTIFIED_RG",
            max_attempts_per_input=len(label_cfg["attempt_seconds"]))
        summary = generate_labels(config, root, requests=requests, guard=ctx.guard)
        if any(summary["admitted_by_role"][r] < 1 for r in ("TRAIN", "VALID", "TEST")):
            raise ValueError("Required nonempty admitted regional role is absent; preserved requests/failed attempts")
        rows.append(dict(suite_id=record["suite_id"], region_index=record["region_index"], **summary))
        write_json(ctx.root / "training" / "LABEL_DISPOSITIONS.json", rows)
    return dict(status="COMPLETE_WITH_LIMITATIONS" if any(sum(r["admitted_by_role"].values())<r["request_count"] for r in rows) else "COMPLETE", regions=len(rows),
        admitted=sum(sum(r["admitted_by_role"].values()) for r in rows),
        requested=sum(r["request_count"] for r in rows), missing_labels_preserved=True)


def retry_labels(ctx):
    from partC_rev1_src.pacnhn_v1.regional import generate_labels
    _, records = support_rows(ctx)
    records=[r for r in records if r["suite_id"]=="s03"]
    outcomes = []
    requested = admitted = pending = 0
    attempts = ctx.c["labels"]["attempt_seconds"]
    for record in records:
        root = region_root(ctx, record) / "labels"
        index = read_json(root / "REQUEST_INDEX.json")
        for attempt_number, seconds in enumerate(attempts[1:], start=2):
            requests = []
            for row in index.values():
                if row["selected_label"] is None:
                    metadata = read_json(row["request_metadata"])
                    with np.load(row["input_file"], allow_pickle=False) as z:
                        requests.append(dict(metadata, u=z["u"].copy()))
            if not requests:
                break
            ctx.guard()
            cfg = dict(phase_id="PACNHN_V1", run_id=ctx.c["run_id"],
                region_id=f'{record["suite_id"]}:{record["region_index"]:06d}', graph_path=record["path"],
                graph_sha256=record["sha256"], deadline_epoch=ctx.deadline,
                per_label_seconds=seconds, max_iterations=20000, relative_gap_tolerance=1e-6,
                solver_method="CFW", retry_solver_method="CFW", cfw_reset_on_bound=True,
                retry_cfw_reset_on_bound=True, iterate_selection="BEST_CERTIFIED_RG",
                retry_iterate_selection="BEST_CERTIFIED_RG", max_attempts_per_input=len(attempts))
            result = generate_labels(cfg, root, requests=requests, guard=ctx.guard)
            outcomes.append(dict(suite_id=record["suite_id"], region_index=record["region_index"], **result))
            index = read_json(root / "REQUEST_INDEX.json")
        requested += len(index)
        admitted += sum(row["selected_label"] is not None for row in index.values())
        pending += sum(row["selected_label"] is None for row in index.values())
    return dict(status="COMPLETE_WITH_LIMITATIONS" if pending else "COMPLETE", attempts=len(outcomes),
        requested=requested,admitted=admitted,pending=pending,no_pending_inputs=pending==0,
        reports=outcomes,existing_role_index_validation=True)


def cohorts(ctx, record):
    from partC_rev1_src.pacnhn_v1.regional import load_admitted_dataset
    root = region_root(ctx, record) / "labels"
    ctx.bind(root / "REQUEST_INDEX.json", "CURRENT_ADMITTED_TRAIN_VALID_ROLES")
    train = load_admitted_dataset(root, role="TRAIN")
    valid = load_admitted_dataset(root, role="VALID")
    return train, valid


def metric(model, dataset, ctx):
    from partC_rev1_src.pacnhn_v1.training import _evaluate
    return _evaluate(model, dataset, ctx.c["training"]["audit_batch_rows"], ctx.guard)


def eligible_key(row):
    return row["TRAIN"]["WMAPE_percent"] >= 1., row["VALID"]["WMAPE_percent"], row["candidate_index"]


def native_record(ctx, record, family, candidate, train, valid, region):
    from partC_rev1_src.pacnhn_v1.models import load_model
    from partC_rev1_src.pacnhn_recovery24.numerical_admission import audit_power4, audit_xgboost, structural_zeros
    from partC_rev1_src.pacnhn_recovery24.labels import native_json
    from partC_rev1_src.pacnhn_v1.final_icnn_guarantees import audit_model
    model = load_model(candidate["model"]["path"])
    if family == "ICNN":
        prop = audit_model(model, guard=ctx.guard)
    elif family == "XGBOOST":
        prop = audit_xgboost(model, structural_zero_mask=structural_zeros(region, ctx.guard), guard=ctx.guard)
    else:
        prop = audit_power4(model, guard=ctx.guard, fftt=family == "FFTT")
    prop_path = region_root(ctx, record) / "models" / f"{family}_PROPERTY.json"
    write_json(prop_path, native_json(prop))
    info = candidate["model"]
    with zipfile.ZipFile(info["path"]) as z:
        raw = sum(i.file_size for i in z.infolist())
    return dict(suite_id=record["suite_id"], region_index=record["region_index"], pairs=record["pairs"],
        family=family, status="AVAILABLE", model_path=info["path"], model_sha256=info["sha256"], model_kind=info["kind"],
        artifact_hashes={info["path"]:info["sha256"], str(prop_path):sha(prop_path)},
        uncompressed_model_bytes=raw, uncompressed_booster_bytes=0 if model.has_potential else len(model.booster.save_raw(raw_format="ubj")) if model.booster else 0,
        forecast_resident_booster_bytes=0 if model.has_potential else 3*raw,
        graph_sha256=record["sha256"], pair_keys_identity=array_hash({"pair_keys":region.pair_keys}),
        train_wmape_percent=candidate["TRAIN"]["WMAPE_percent"], valid_wmape_percent=candidate["VALID"]["WMAPE_percent"],
        current_train_dataset_identity=train["dataset_identity"], current_valid_dataset_identity=valid["dataset_identity"],
        audit_train_dataset_identity=train["dataset_identity"], audit_valid_dataset_identity=valid["dataset_identity"],
        audit_model_sha256=info["sha256"], property_status=prop["status"], property_model_sha256=info["sha256"])


def train_region(ctx, record):
    import torch
    from partC_rev1_src.pacnhn_v1.models import (Power4Model, fit_power4_cost, fit_xgboost_cost,
        update_icnn, save_model, ICNN_SCALED, XGB_CENTERED)
    from partC_rev1_src.pacnhn_v1.training import _initialize_icnn, _structural_zeros
    torch.set_num_threads(1)
    train, valid = cohorts(ctx, record)
    region = load_graph(ctx, record)
    scale = scale_for(region, record)
    folder = region_root(ctx, record) / "models"
    folder.mkdir(parents=True, exist_ok=False)
    c = ctx.c["training"]
    end = min(ctx.deadline-20, time.time()+c["region_seconds"])
    candidates = {f:[] for f in FAMILIES}
    def audit_save(family, model, **extra):
        ctx.guard()
        index = len(candidates[family])
        info = save_model(model, folder/f'{family}_{index:04d}.npz', metadata=dict(TEST_used=False, **extra))
        row = dict(candidate_index=index, model=info, TRAIN=metric(model, train, ctx), VALID=metric(model, valid, ctx), **extra)
        candidates[family].append(row)
        write_json(folder / "CANDIDATES.json", candidates)
        return row
    audit_save("FFTT", Power4Model(train["c0"], scale, np.zeros(region.m)))
    power = fit_power4_cost(train["u"], train["tau"], train["c0"], scale)
    audit_save("POWER4", power)
    for width in c["icnn_widths"]:
        if time.time() >= end:
            break
        model = _initialize_icnn(train, scale, dict(seed=c["seed"], icnn_coefficient_parameterization=ICNN_SCALED,
            icnn_initial_neural_cost_share=.001), "cpu", width)
        opt = torch.optim.Adam(model.parameters(), lr=.001)
        generator = torch.Generator().manual_seed(c["seed"]+width)
        updates = 0
        stop_width = False
        for rate in c["icnn_learning_rates"]:
            for group in opt.param_groups:
                group["lr"] = rate
            before = min(candidates["ICNN"], key=eligible_key) if candidates["ICNN"] else None
            extra = 0
            for block in range(c["initial_blocks"]+c["extra_blocks"]):
                if block >= c["initial_blocks"]+extra or time.time() >= end:
                    break
                for _ in range(c["updates_per_block"]):
                    ctx.guard()
                    if time.time() >= end:
                        break
                    ix = torch.randint(len(train["u"]), (min(c["batch_rows"], len(train["u"])),), generator=generator).numpy()
                    update_icnn(model, opt, train["u"][ix], train["tau"][ix], train_cost_mean=train["train_cost_mean"])
                    updates += 1
                row = audit_save("ICNN", model.export_model(), width=width, learning_rate=rate, updates=updates)
                # Persist optimizer separately; fresh runs never consume historical moments.
                torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), generator=generator.get_state(),
                    updates=updates, scale=torch.tensor(scale), TRAIN_identity=train["dataset_identity"], VALID_identity=valid["dataset_identity"]),
                    folder/f'ICNN_OPT_{width}_{updates:06d}.pt')
                if row["TRAIN"]["WMAPE_percent"] < 1.:
                    stop_width = True
                    break
                if block+1 == c["initial_blocks"] and before is not None:
                    tr_gain = before["TRAIN"]["WMAPE_percent"]-row["TRAIN"]["WMAPE_percent"]
                    va_gain = before["VALID"]["WMAPE_percent"]-row["VALID"]["WMAPE_percent"]
                    productive = tr_gain >= max(.01, .01*before["TRAIN"]["WMAPE_percent"]) or va_gain >= max(.01, .01*before["VALID"]["WMAPE_percent"])
                    extra = c["extra_blocks"] if productive else 0
                elif block+1 == c["initial_blocks"] and before is None:
                    extra = c["extra_blocks"]
            if stop_width:
                break
        if stop_width:
            break
    zero = _structural_zeros(region)
    xgb = None
    quota = record.get("xgb_raw_quota_bytes", c["xgb_region_raw_bytes"])
    for rounds in c["xgb_round_increments"]:
        if time.time() >= end:
            break
        xgb = fit_xgboost_cost(train["u"], train["tau"], train["c0"], scale,
            seed=c["seed"], n_estimators=rounds, max_depth=3, learning_rate=.05, n_jobs=1,
            structural_zero_mask=zero, structural_zero_evidence="FROZEN_GRAPH_ZERO_COST_DIRECTED_REACHABILITY",
            initial_model=xgb, deadline_epoch=end, guard=ctx.guard,
            log_center_policy=XGB_CENTERED, train_dataset_identity=train["dataset_identity"])
        raw = 0 if xgb.booster is None else len(xgb.booster.save_raw(raw_format="ubj"))
        if raw > quota:
            raise MemoryError("Explicit per-region XGBoost raw-model envelope exceeded")
        row = audit_save("XGBOOST", xgb, boosting_rounds=xgb.metadata["boosting_rounds_total"])
        if row["TRAIN"]["WMAPE_percent"] < 1.:
            break
    selected = []
    for family in FAMILIES:
        if candidates[family]:
            chosen = min(candidates[family], key=eligible_key)
            selected.append(native_record(ctx, record, family, chosen, train, valid, region))
        else:
            selected.append(dict(suite_id=record["suite_id"], region_index=record["region_index"], pairs=record["pairs"],
                family=family, status="UNAVAILABLE_WITHIN_BUDGET", graph_sha256=record["sha256"]))
    write_json(folder/"SELECTED.json", selected)
    return selected


def training(ctx):
    _, records = support_rows(ctx)
    records=[r for r in records if r["suite_id"]=="s03"]
    if (ctx.root / "portfolios" / "PORTFOLIOS.json").exists():
        raise PermissionError("Portfolio already frozen; no training after TEST selection freeze")
    rows, failures = [], []
    # Preserve the final source's 4GiB complete-suite raw-booster envelope.
    for sid in SUITES:
        group=[r for r in records if r["suite_id"]==sid]
        count=len(group);pairs=sum(r["pairs"] for r in group);minimum=192*1024
        cap=4*2**30
        if count*minimum>cap:
            raise MemoryError("Complete-suite minimum XGBoost allocation exceeds original envelope")
        for record in group:
            record["xgb_raw_quota_bytes"]=min(ctx.c["training"]["xgb_region_raw_bytes"], minimum+(cap-count*minimum)*record["pairs"]//pairs)
    for record in records:
        ctx.guard()
        try:
            selected = train_region(ctx, record)
        except Exception as exc:
            if isinstance(exc, TimeoutError):
                raise
            failures.append(dict(suite_id=record["suite_id"], region_index=record["region_index"],
                error_type=type(exc).__name__, error=str(exc)))
            selected = [dict(suite_id=record["suite_id"], region_index=record["region_index"],
                pairs=record["pairs"], family=f, status="FAILED_RETAINED_EVIDENCE", graph_sha256=record["sha256"]) for f in FAMILIES]
        rows.extend(selected)
        write_json(ctx.root/"training/NORMALIZED_RECORDS.json", rows)
        write_json(ctx.root/"training/FAILURES.json", failures)
    if failures or any(not any(r["status"]=="AVAILABLE" and r["family"]=="POWER4" and r["suite_id"]==record["suite_id"] and r["region_index"]==record["region_index"] for r in rows) for record in records):
        raise ValueError("Mandatory regional training/baseline task failed; complete failure ledger preserved")
    return dict(status="COMPLETE_WITH_LIMITATIONS" if any(r["status"]!="AVAILABLE" for r in rows) else "COMPLETE", regions=len(records), available=sum(r["status"]=="AVAILABLE" for r in rows),
        failures=failures, recipe="FROZEN_MODEL_MATH_FRESH_WIDTH_RATE_BLOCK_SEARCH", TEST_used=False)


def freeze(ctx):
    import copy
    from collections import Counter
    from partC_rev1_src.pacnhn_recovery24.portfolio import make_portfolio,model_eligible,_digest
    support, records = support_rows(ctx)
    target = ctx.root / "portfolios/PORTFOLIOS.json"
    if target.exists():
        raise FileExistsError("One-way portfolio freeze already exists")
    path = ctx.root / "training/NORMALIZED_RECORDS.json"
    ctx.bind(path, "CURRENT_TRAIN_VALID_SELECTION_ONLY")
    rows = read_json(path)
    portfolios = make_portfolio(support, rows)
    qualified=copy.deepcopy(portfolios)
    write_json(ctx.root/"portfolios/PORTFOLIOS_ACCURACY_QUALIFIED.json", dict(schema="PORTABLE_PACNHN_PORTFOLIOS_V1",
        policy="ACCURACY_QUALIFIED",portfolios=qualified,selection_records_sha256=sha(path),TEST_used=False,
        fixed_across_scenarios=True,dynamic_fallback=False,created_epoch=time.time()))
    index={(r["suite_id"],r["region_index"],r["family"]):r for r in rows}
    for portfolio in portfolios:
        if portfolio["requested_family"]!="ICNN":
            continue
        for component in portfolio["regions"]:
            rec=index.get((portfolio["suite_id"],component["region_index"],"ICNN"))
            if not model_eligible(rec,require_train=False):
                continue
            component["component"]="ICNN";component["reason"]="RETAINED_FINITE_CURRENT_MODEL_PROPERTY_PASS_WITHOUT_TRAIN_QUALIFICATION"
            component["model"]={k:rec[k] for k in ("model_path","model_sha256","artifact_hashes","model_kind","pair_keys_identity","graph_sha256",
                "property_status","property_model_sha256","uncompressed_model_bytes","uncompressed_booster_bytes","forecast_resident_booster_bytes") if k in rec}
        portfolio["component_counts"]=dict(Counter(r["component"] for r in portfolio["regions"]))
        portfolio["portfolio_identity"]=_digest(dict(suite_id=portfolio["suite_id"],requested_family="ICNN",regions=portfolio["regions"],policy="RETAINED_MODEL"))
    write_json(target, dict(schema="PORTABLE_PACNHN_PORTFOLIOS_V1", policy="RETAINED_MODEL_ICNN_AND_ACCURACY_QUALIFIED_XGB", portfolios=portfolios,
        selection_records_sha256=sha(path), support_sha256=sha(ctx.root/"SUPPORT_FREEZE.json"), TEST_used=False,
        label_indexes={f'{r["suite_id"]}:{r["region_index"]:06d}':dict(path=str(region_root(ctx,r)/"labels/REQUEST_INDEX.json"),
            sha256=sha(region_root(ctx,r)/"labels/REQUEST_INDEX.json")) for r in records if r["suite_id"]=="s03"},
        fixed_across_scenarios=True, dynamic_fallback=False, created_epoch=time.time()))
    return dict(status="FROZEN", portfolios=len(portfolios), TEST_used=False)


def assignments(ctx,qualified=False):
    from partC_rev1_src.pacnhn_v1.topology import load_common
    from partC_rev1_src.pacnhn_v1.oracle import Network, Demand
    from partC_rev1_src.pacnhn_v1.models import load_model
    from partC_rev1_src.pacnhn_v1.solver import solve
    from partC_rev1_src.pacnhn_recovery24.portfolio import FrozenComposite
    path = ctx.root / "portfolios" / ("PORTFOLIOS_ACCURACY_QUALIFIED.json" if qualified else "PORTFOLIOS.json")
    ctx.bind(path, "IMMUTABLE_PRE_TEST_PORTFOLIOS")
    frozen = read_json(path)
    rows = []
    for portfolio in frozen["portfolios"]:
        sid, family = portfolio["suite_id"], portfolio["requested_family"]
        if sid!="s03" and family!="FFTT":
            continue
        if qualified and family!="ICNN":
            continue
        for scenario in ("open", "i95"):
            ctx.guard()
            # Complete-query timing includes current graph/model/OD input loading.
            started = time.time()
            query_end=min(ctx.deadline,started+ctx.c["assignment"]["per_query_seconds"])
            _, od, _, _ = load_common(ctx.root)
            gp = ctx.root / "acnhn" / sid / "graph.npz"
            ctx.bind(gp, "CURRENT_ABSTRACT_GRAPH", portfolio["graph_sha256"])
            with np.load(gp, allow_pickle=False) as z:
                a = {k:z[k].copy() for k in z.files}
            graph = Network(a["tail"], a["head"], a["blocked"], a["edge_ids"], a["node_ids"])
            bc = int(a["backbone_count"])
            blocks = []
            for component in portfolio["regions"]:
                ctx.guard(deadline=query_end)
                if component["component"] == "FFTT":
                    continue
                info = component["model"]
                ctx.bind(info["model_path"], "EXACT_FROZEN_REGION_MODEL", info["model_sha256"])
                indices = bc+np.flatnonzero(a["pair_region"] == component["region_index"])
                blocks.append((indices, load_model(info["model_path"])))
            objective = FrozenComposite(a["base"], a["k4"], bc+np.arange(len(a["pair_keys"])), blocks, cost_only=family=="XGBOOST")
            with np.load(ctx.root/"references"/scenario/"coverage.npz", allow_pickle=False) as z:
                reference_serve = z["serve"].copy()
            fixed_od = Demand(od.origins, od.offsets, od.destinations, od.demand, reference_serve)
            root = ctx.root / ("assignments_accuracy_qualified" if qualified else "assignments") / sid / family / scenario
            result = solve(graph, fixed_od, objective, root, method="MSA" if family=="XGBOOST" else "FW",
                run_start_epoch=started, deadline_epoch=query_end,
                tolerance=ctx.c["assignment"]["relative_gap"], ctx=ctx, enabled=~a["closed_mask"] if scenario=="i95" else None,
                workers=ctx.c["workers"])
            with np.load(root/"coverage.npz", allow_pickle=False) as z:
                if not np.array_equal(z["serve"], reference_serve):
                    raise ArithmeticError("Abstract scenario served membership differs from P reference")
            if not result["final_certificate_fresh"] or (result["accepted_updates"]==0 and result["status"]!="CONVERGED"):
                raise ValueError("Abstract assignment has no meaningful certified progress")
            rows.append(dict(suite_id=sid, family=family, scenario=scenario, **result))
            write_json(ctx.root/("assignments_accuracy_qualified" if qualified else "assignments")/"RESULTS.json", rows)
            del graph,od,a,objective,blocks
    return dict(status="ALL_ASSIGNMENT_DISPOSITIONS_RECORDED", cases=len(rows), converged=sum(r["status"]=="CONVERGED" for r in rows),
        portfolios_sha256=sha(path), convergence_not_inferred_from_timer=True)


def load_test(ctx, record):
    freeze_path=ctx.root / "portfolios/PORTFOLIOS.json"
    if not freeze_path.exists():
        raise PermissionError("TEST evaluation requires persisted model/portfolio freeze")
    frozen=read_json(freeze_path)
    ctx.bind(freeze_path,"IMMUTABLE_PRE_TEST_SELECTION_FREEZE")
    ctx.bind(ctx.root/"training/NORMALIZED_RECORDS.json","EXACT_FROZEN_SELECTION_RECORDS",frozen["selection_records_sha256"])
    ctx.bind(ctx.root/"SUPPORT_FREEZE.json","EXACT_FROZEN_SUPPORT",frozen["support_sha256"])
    root = region_root(ctx, record)/"labels"
    key=f'{record["suite_id"]}:{record["region_index"]:06d}'
    binding=frozen["label_indexes"][key]
    if Path(binding["path"]).resolve()!=(root/"REQUEST_INDEX.json").resolve():
        raise ValueError("Frozen TEST index path differs from this region")
    ctx.bind(root/"REQUEST_INDEX.json","IMMUTABLE_PRE_TEST_ROLE_AND_LABEL_INDEX",binding["sha256"])
    region=load_graph(ctx,record)
    index = read_json(root/"REQUEST_INDEX.json")
    inputs, costs, c0 = [], [], None
    for input_id, row in index.items():
        selected = row["selected_label"]
        if row["role"] != "TEST" or selected is None:
            continue
        ctx.bind(selected["path"], "BLIND_TEST_AFTER_FROZEN_SELECTION", selected["sha256"])
        ctx.bind(selected["receipt"], "BLIND_TEST_CERTIFICATE", selected["receipt_sha256"])
        receipt = read_json(selected["receipt"])
        if (receipt["role"] != "TEST" or receipt["input_identity"] != input_id or not receipt["admitted"]
            or receipt["graph_sha256"]!=record["sha256"] or receipt["support_identity"]!=region.support_identity):
            raise ValueError("Invalid TEST label identity/role")
        with np.load(selected["path"], allow_pickle=False) as z:
            inputs.append(z["u"].copy()); costs.append(z["tau"].copy()); current_c0=z["c0"].copy()
        if array_hash({"u":inputs[-1]})!=input_id or not np.array_equal(current_c0,region.c0):
            raise ValueError("TEST native input/free-flow order differs from frozen graph")
        if c0 is not None and not np.array_equal(c0,current_c0):
            raise ValueError("TEST free-flow vector changed")
        c0=current_c0
    if not inputs:
        raise ValueError("No admitted TEST inputs")
    return dict(u=np.stack(inputs), tau=np.stack(costs), c0=c0, role="TEST", unique_rows=len(inputs))


def evaluate(ctx):
    from partC_rev1_src.pacnhn_v1.models import load_model
    _, records = support_rows(ctx)
    records=[r for r in records if r["suite_id"]=="s03"]
    frozen=read_json(ctx.root/"portfolios/PORTFOLIOS.json")
    ctx.bind(ctx.root/"training/NORMALIZED_RECORDS.json","EXACT_FROZEN_SELECTION_RECORDS",frozen["selection_records_sha256"])
    selected = read_json(ctx.root/"training/NORMALIZED_RECORDS.json")
    rows = []
    for record in records:
        ctx.guard()
        try:
            test = load_test(ctx, record)
            for row in selected:
                if row["suite_id"] == record["suite_id"] and row["region_index"] == record["region_index"] and row["status"] == "AVAILABLE":
                    ctx.bind(row["model_path"], "FROZEN_EVALUATED_MODEL", row["model_sha256"])
                    rows.append(dict(suite_id=row["suite_id"],region_index=row["region_index"],family=row["family"],
                        TEST=metric(load_model(row["model_path"]),test,ctx)))
        except ValueError as exc:
            rows.append(dict(suite_id=record["suite_id"],region_index=record["region_index"],status="NO_ADMITTED_TEST",error=str(exc)))
        write_json(ctx.root/"evaluation/REGIONAL_TEST.json",rows)
    return dict(status="TEST_DISPOSITIONS_RECORDED",rows=len(rows),TEST_used_for_selection=False)


def sample_fractions(ctx):
    from experiments import run
    return run(ctx,enriched=False)


def enrichment(ctx):
    from experiments import run
    return run(ctx,enriched=True)


COMMANDS = {"references":reference,"support":support,"labels":labels,"retry-labels":retry_labels,
    "train":training,"freeze":freeze,"assign":assignments,"assign-qualified":lambda ctx:assignments(ctx,qualified=True),
    "evaluate":evaluate,"sample-fractions":sample_fractions,"enrichment":enrichment}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=COMMANDS)
    parser.add_argument("--config",required=True)
    parser.add_argument("--run-id",help="Override configured identity for an optional direct action in an existing supervised campaign")
    args=parser.parse_args()
    if args.run_id is not None:
        os.environ["REPRODUCTION_RUN_ID"] = args.run_id
    ctx=Run(args.config,args.command)
    try:
        result=COMMANDS[args.command](ctx)
    except Exception as exc:
        result=dict(status="TIME_LIMIT" if isinstance(exc,TimeoutError) else "FAILED",error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),outputs_preserved=True)
        ctx.finish(result)
        raise
    print(json.dumps(ctx.finish(result),allow_nan=False))


if __name__=="__main__":
    main()
