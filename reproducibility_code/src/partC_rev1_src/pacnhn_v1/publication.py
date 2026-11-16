"""Explicit, source-bound PACNHN_V1 publication extraction and reproduction.

This module never assigns traffic, trains a model or discovers historical data.
The parent exporter supplies nine hash-bound CSVs. Execution requires an explicit
publication configuration and the applicable authenticated campaign window.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .control import Context, read_json, sha, write_json
from .publication_templates.csv_support import CSV_PART_COLUMNS, csv_leaf, csv_logical_names, csv_expected_files, read_logical_csv

TEMPLATES = Path(__file__).with_name("publication_templates")
GEO = ["tail_longitude", "tail_latitude", "head_longitude", "head_latitude"]
SCHEMAS = {
    "network": "suite_id resolution nodes edges backbone_links regional_links regions active_regions active_pairs served_rows_open served_rows_closed".split(),
    "methods": "case_id method_id network suite_id scenario status solver RG_SPTT elapsed_seconds wmape_backbone_percent wmape_key_percent reference_case_id assignment_role".split(),
    "backbone_links": "case_id scenario link_id native_index reference_flow predicted_flow signed_bounded_error is_key".split()+GEO+["is_closed"],
    "learning": "suite_id region_index family TRAIN_WMAPE_percent VALID_WMAPE_percent TEST_WMAPE_percent target_met status train_rows parameters neural_amplitude".split(),
    "convergence": "case_id iteration elapsed_seconds RG_SPTT".split(),
    "training_curves": "suite_id region_index family block elapsed_seconds train_wmape valid_wmape".split(),
    "timing": "task category seconds status timing_boundary".split(),
    "I95": ["method_id", "link_id", "open_flow", "closed_flow", "delta"]+GEO+["is_closed"],
    "keys": ["link_id"],
}
OPTIONAL = {
    "network": ["status", "note"],
    "methods": "certificate_fresh residual_kind nominal_tolerance query_cap_seconds timing_boundary reference_status reference_RG_SPTT served_OD_membership_equal note".split(),
    "backbone_links": [],
    "learning": ("valid_rows test_rows test_opened TEST_status selected_checkpoint selected_is_initialization selected_optimizer_updates selected_boosting_rounds "
        "observed_increment_absolute_sum maximum_native_cost_increment nonzero_prediction_coordinates structurally_active "
        "neural_contribution_train_dataset_identity train_dataset_identity pairs label_origin note "
        "TEST_absolute_error_sum TEST_true_cost_sum TEST_coordinates TEST_inputs").split(),
    "convergence": ["residual_kind"],
    "training_curves": "candidate_id train_dataset_identity valid_dataset_identity evaluation_kind".split(),
    "timing": "start_epoch end_epoch accounting_role parent_task time_kind cap_seconds note".split(),
    "I95": ["case_id", "suite_id", "reference_case_id", "open_closed_OD_membership_equal", "lost_OD_rows", "gained_OD_rows", "lost_OD_demand", "gained_OD_demand"],
    "keys": GEO+["native_index", "key_order", "lambda_reference", "longitude", "latitude", "x", "y", "cell_x", "cell_y"],
}
IDENTITIES = {"suite_id", "case_id", "method_id", "reference_case_id", "link_id", "scenario", "family", "status", "network", "assignment_role", "solver"}
PANEL_COLUMNS = "panel_id title kind source x y group filter_column filter_value xlabel ylabel xscale yscale xlinthresh ylinthresh symmetric point_markers colorbar threshold note".split()
TITLES = {
    "figure3": "Native backbone accuracy and the frozen 200-link subset",
    "figure4": "Regional cost prediction and training trajectories",
    "figure5": "Resolution coverage on one fixed backbone",
    "figure6": "Represented I-95 closure and native backbone redistribution",
    "figure7": "State-consistent assignment convergence diagnostics",
    "figure8": "Recorded query and offline timing",
    "table1": "Common-network and regional-resolution inventory",
    "table2": "Open-scenario native backbone comparison",
    "table3": "Regional model accuracy and held-out coverage",
    "table4": "Resolution-specific assignment coverage",
    "table5": "Represented I-95 redistribution and closed-reference accuracy",
    "table6": "Timing boundaries, accounting roles and censoring",
}
CAPTIONS = {
    "table1": "The consolidated backbone is fixed across resolutions. Counts describe the supplied export; missing support or reachability counts remain N/A. Logical OD roles are nontransit. Regional centroid physical identities are retained; residual partitions do not promote new backbone arcs.",
    "table2": "WMAPE = 100 sum|prediction-reference|/sum(reference), with every native B coordinate retained, including zero-reference links. The same frozen 200 IDs are used in all cases. Reference-self rows are separate from independent controls. TIME_LIMIT gives a censored query result, not convergence. FW gaps and cost-only VI residuals have distinct semantics. N/A is undefined or unavailable.",
    "table3": "Regional medians weight finite regional errors equally. Pooled TEST WMAPE uses summed native cost errors/truth over explicitly opened rows; coverage counts distinguish these populations. TRAIN <1% does not ensure backbone accuracy. Neural sum/max/nnz measure cost increments over each ICNN's own power4 component on its named full TRAIN, not improvement over separately fitted power4 or trip-weighted travel time. Amplitude, structural activity and measured nonzero increments are distinct. Checkpoint IDs, optimizer updates and boosting rounds differ. All underlying observations remain in CSV.",
    "table4": "Every available resolution shares the same native backbone and OD map. Only exported jobs are measured. Additional learned training depends on complete s03 TRAIN qualification in both ICNN and XGBoost. Analytic FFTT is available for all five frozen-support suites; its additional assignments are separate. Missing, invalid and skipped methods retain their dispositions; no alternate regional family is substituted under a pure-family label.",
    "table5": "Gain/loss/net/absolute change sum native link traversals (annual thousand tons); they are not OD demand losses. The represented B closure omits hidden physical interiors. Closed errors use closed P and open-frozen keys. Within-scenario AC=P membership is distinct from open-versus-closed membership; traversal changes may include newly unserved OD demand. Reference-self is separate from controls.",
    "table6": "Selected complete queries, whole-model calls, stage/worker spans and historical context retain exact boundaries, caps and status. All timing observations and the display-selection ledger remain in CSV. TIME_LIMIT censors completion. Only checked DISJOINT_INTERVAL measured-wall rows add; nested/worker rows are separate. Reference preparation, labels, training and publication are offline costs. No amortized speedup is inferred from unmatched scopes.",
    "figure3": "Maps show native endpoint chords in longitude/latitude, cropped to the contiguous US for display only. Error populations retain full native B, including zero-reference coordinates; the 200-key subset is frozen from open P. Signed bounded error is (prediction-reference)/max(prediction,reference), with zero/zero defined as zero.",
    "figure4": "CDFs weight each supplied regional model equally, with finite n shown per series; they are not pooled-coordinate WMAPE. Blind TEST requires explicit opening metadata. Curves use each candidate's first recorded evaluation, ordered within region/family. Coverage and historical TRAIN/admitted VALID identities can change. All observations and the selection ledger remain available; the ordinal is not an optimizer-step count.",
    "figure5": "Native B is held fixed as residual regional resolution changes. Error points represent only available jobs and do not interpolate missing runs. Resolution coverage and the s03-first qualification gate remain separate from accuracy.",
    "figure6": "Maps show native longitude/latitude endpoint chords, hidden axes and a contiguous-US display crop. Closed-minus-open traversal changes can combine rerouting and newly unserved OD demand. The closure covers represented B coordinates only. Traversal sums are not OD-demand losses or a full physical I-95 closure.",
    "figure7": "Each curve belongs to one recorded assignment case. FW/CFW potential gaps and cost-only XGBoost VI residuals are shown separately. Requested tolerances are not achieved-status claims. Final fresh certificates and TIME_LIMIT dispositions are reported in the method tables.",
    "figure8": "Bars show recorded seconds, not projected time to convergence. TIME_LIMIT is marked as censored. Offline measured-wall rows are plotted separately from nested worker timings; only validated disjoint intervals may be added. No break-even or amortized speedup is claimed.",
}


def nums(series):
    return pd.to_numeric(series, errors="coerce")


def truth(series):
    return series.astype(str).str.lower().isin(["true", "1", "1.0"])


def finite_text(value, digits=4):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "N/A"
    return f"{value:.{digits}g}" if math.isfinite(value) else "N/A"


def text_value(value):
    return str(value) if pd.notna(value) and str(value).strip() else "N/A"


def field(row, name):
    return text_value(row.get(name, ""))


def case_labels(methods):
    data = methods.copy()
    data["label"] = data.method_id.astype(str)
    duplicate = data.duplicated(["method_id", "scenario"], keep=False)
    data.loc[duplicate, "label"] += " / "+data.loc[duplicate, "case_id"].astype(str)
    qualified = data.status.astype(str).ne("CONVERGED")
    data.loc[qualified, "label"] += " ["+data.loc[qualified, "status"].astype(str)+"]"
    return data


def checkpoint_counts(frame):
    selected = frame.selected_checkpoint.astype(str) if "selected_checkpoint" in frame else pd.Series("", index=frame.index)
    known = selected.str.strip().ne("") & ~selected.str.lower().isin(["nan", "none"])
    if "selected_is_initialization" in frame:
        values = frame.selected_is_initialization.astype(str).str.lower()
        classified = values.isin(["true", "false", "0", "1", "0.0", "1.0"])
        initial = truth(frame.selected_is_initialization) & classified
        return int(known.sum()), int(initial.sum()) if classified.any() else None, int(classified.sum())
    if "selected_optimizer_updates" in frame:
        updates = nums(frame.selected_optimizer_updates)
        return int(known.sum()), int(updates.eq(0).sum()) if updates.notna().any() else None, int(updates.notna().sum())
    return int(known.sum()), None, 0


def selected_counter_text(frame, column, label):
    values = nums(frame[column]) if column in frame else pd.Series(dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return label+": N/A"
    if (values < 0).any() or not np.equal(values, np.floor(values)).all():
        raise ValueError("Selected optimizer/tree counters must be nonnegative integers")
    return f"{label}: {int(values.min())}–{int(values.max())} (n={len(values)})"


def neural_summary(frame):
    """Summarize measured increments against each ICNN's OWN power4 part."""
    selected = frame.loc[frame.family.eq("ICNN")].copy()
    fields = ["observed_increment_absolute_sum", "maximum_native_cost_increment", "nonzero_prediction_coordinates"]
    result = {"eligible_rows": len(selected), "measured_regions": 0, "increment_absolute_sum": None,
        "maximum_native_increment": None, "nonzero_coordinates": None, "structurally_active": 0,
        "activity_known": 0, "positive_amplitude": 0, "amplitude_known": 0, "dataset_identity_count": 0,
        "prediction_coordinates": None, "nonzero_coordinate_fraction": None}
    amplitude = nums(selected.neural_amplitude)
    result["positive_amplitude"] = int(amplitude.gt(0).sum()); result["amplitude_known"] = int(np.isfinite(amplitude).sum())
    if "structurally_active" in selected:
        known = selected.structurally_active.astype(str).str.lower().isin(["true", "false", "0", "1", "0.0", "1.0"])
        result["activity_known"] = int(known.sum())
        result["structurally_active"] = int((truth(selected.structurally_active) & known).sum())
    identity_columns = ["neural_contribution_train_dataset_identity", "train_dataset_identity"]
    if not set(fields+identity_columns).issubset(selected):
        return result
    measured = np.isfinite(selected[fields].apply(pd.to_numeric, errors="coerce").to_numpy()).all(axis=1)
    any_measured = selected.loc[measured]
    if len(any_measured):
        identities = any_measured[identity_columns].astype(str)
        if identities.eq("").any(axis=None) or not identities.iloc[:, 0].eq(identities.iloc[:, 1]).all():
            raise ValueError("Measured neural increment is not bound to the current full TRAIN identity")
        numbers = any_measured[fields].apply(pd.to_numeric, errors="raise")
        if (numbers.to_numpy() < 0).any() or not np.equal(numbers.nonzero_prediction_coordinates, np.floor(numbers.nonzero_prediction_coordinates)).all():
            raise ValueError("Neural increment sums/maxima/counts must be finite nonnegative observations")
        result.update(measured_regions=len(any_measured), increment_absolute_sum=float(numbers.observed_increment_absolute_sum.sum()),
            maximum_native_increment=float(numbers.maximum_native_cost_increment.max()),
            nonzero_coordinates=int(numbers.nonzero_prediction_coordinates.sum()),
            dataset_identity_count=int(identities.iloc[:, 0].nunique()))
        if {"train_rows", "pairs"}.issubset(any_measured):
            coordinates = nums(any_measured.train_rows)*nums(any_measured.pairs)
            if not np.isfinite(coordinates).all() or (coordinates <= 0).any() or not np.equal(coordinates, np.floor(coordinates)).all():
                raise ValueError("Measured neural audit needs positive full-TRAIN coordinate denominators")
            if (numbers.nonzero_prediction_coordinates.to_numpy() > coordinates.to_numpy()).any():
                raise ValueError("Neural nonzero count exceeds the full TRAIN coordinate denominator")
            denominator = int(coordinates.sum())
            result["prediction_coordinates"] = denominator
            result["nonzero_coordinate_fraction"] = result["nonzero_coordinates"]/denominator
    return result


def neural_narrative(frame):
    summary = neural_summary(frame)
    structural = (f"Structural activity is recorded for {summary['activity_known']} ICNN rows, with {summary['structurally_active']} active branches; "
        f"positive neural amplitude occurs in {summary['positive_amplitude']} of {summary['amplitude_known']} amplitude records. ")
    if not summary["measured_regions"]:
        return structural+"No dataset-bound full-TRAIN native neural increment audit is available in this export. Structural activity and amplitude alone do not establish a measured nonzero neural contribution."
    denominator = summary["prediction_coordinates"]
    count_text = f"{summary['nonzero_coordinates']:,} strictly nonzero floating-point increments"
    if denominator is not None:
        count_text += f" among {denominator:,} TRAIN prediction coordinates ({100*summary['nonzero_coordinate_fraction']:.6g}%)"
    return (f"Measured neural increment audits cover {summary['measured_regions']} ICNN regional models on their exact current full unique TRAIN datasets. "
        f"The sum of absolute native cost increments is {summary['increment_absolute_sum']:.8g} seconds across prediction coordinates, "
        f"with maximum increment {summary['maximum_native_increment']:.8g} seconds and {count_text}. "
        +structural+"These increments compare each selected ICNN with its own power4 component, not with the separately fitted power4 model. "
        "They measure neural cost contribution, not prediction-error improvement or trip-weighted total travel time; structural-zero coordinates remain in the coordinate denominator.")


TEST_AGGREGATE_FIELDS = ["TEST_absolute_error_sum", "TEST_true_cost_sum", "TEST_coordinates", "TEST_inputs"]


def pooled_test_summary(frame):
    """Native-cost ratio of supplied TEST sums; never reconstruct from percentages."""
    opened = truth(frame.test_opened) if "test_opened" in frame else pd.Series(False, index=frame.index)
    available = frame.status.eq("AVAILABLE") if "status" in frame else pd.Series(False, index=frame.index)
    eligible = opened & available
    result = dict(recorded_regions=len(frame), opened_available_regions=int(eligible.sum()),
        unavailable_regions=int((~available).sum()), contributing_regions=0,
        excluded_regions=len(frame), absolute_error_sum=None, true_cost_sum=None,
        coordinates=None, inputs=None, wmape_percent=None, status="UNAVAILABLE_NO_EXPLICIT_SUMS")
    if not set(TEST_AGGREGATE_FIELDS).issubset(frame):
        return result
    values = frame[TEST_AGGREGATE_FIELDS].apply(pd.to_numeric, errors="coerce")
    # Missing evidence remains excluded; supplied invalid numerical evidence fails closed.
    for column in TEST_AGGREGATE_FIELDS:
        supplied = frame[column].notna() & frame[column].astype(str).str.strip().ne("")
        if (eligible & supplied & values[column].isna()).any():
            raise ValueError("Supplied TEST aggregate evidence is nonnumeric: "+column)
        observed = values.loc[eligible, column].dropna()
        if not np.isfinite(observed).all() or (observed < 0).any():
            raise ValueError("TEST aggregate evidence must be finite and nonnegative: "+column)
        if column in {"TEST_coordinates", "TEST_inputs"} and not np.equal(observed, np.floor(observed)).all():
            raise ValueError("TEST aggregate counts must be integers: "+column)
    complete = eligible & values.notna().all(axis=1)
    valid = values.loc[complete]
    if len(valid):
        positive_counts = valid.TEST_coordinates.gt(0) & valid.TEST_inputs.gt(0)
        if ((~positive_counts) & (valid.TEST_absolute_error_sum.gt(0) | valid.TEST_true_cost_sum.gt(0))).any():
            raise ValueError("Nonzero TEST cost sums require positive input and coordinate counts")
        if (valid.TEST_coordinates < valid.TEST_inputs).any():
            raise ValueError("TEST coordinates cannot be fewer than TEST inputs")
        valid = valid.loc[positive_counts]
    if not len(valid):
        result["status"] = "UNAVAILABLE_NO_COMPLETE_OPENED_OBSERVATIONS"
        return result
    error = math.fsum(valid.TEST_absolute_error_sum)
    actual = math.fsum(valid.TEST_true_cost_sum)
    if not math.isfinite(error) or not math.isfinite(actual):
        raise ValueError("TEST pooled sums overflowed")
    result.update(contributing_regions=len(valid), excluded_regions=len(frame)-len(valid),
        absolute_error_sum=error, true_cost_sum=actual,
        coordinates=sum(int(v) for v in valid.TEST_coordinates), inputs=sum(int(v) for v in valid.TEST_inputs),
        wmape_percent=100*(error/actual) if actual > 0 else None,
        status="AVAILABLE" if actual > 0 else "UNDEFINED_ZERO_TRUE_COST")
    if result["wmape_percent"] is not None and not math.isfinite(result["wmape_percent"]):
        raise ValueError("TEST pooled WMAPE overflowed")
    return result


def pooled_test_narrative(frame):
    measured = []; unavailable = []
    for (suite, family), group in frame.groupby(["suite_id", "family"], sort=True):
        summary = pooled_test_summary(group)
        identity = f"{suite}/{family}"
        if summary["wmape_percent"] is None:
            unavailable.append(f"{identity} ({summary['status']}; {summary['excluded_regions']} excluded records)")
            continue
        opened = truth(group.test_opened)
        regional = nums(group.loc[opened & group.status.eq("AVAILABLE"), "TEST_WMAPE_percent"])
        regional = regional[np.isfinite(regional)]
        measured.append(f"{identity}: {summary['wmape_percent']:.4g}% pooled versus {finite_text(regional.median())}% unweighted regional median "
            f"(pooled n={summary['contributing_regions']}/{summary['recorded_regions']} regions, {summary['inputs']:,} region-input observations, "
            f"{summary['coordinates']:,} cost coordinates; median n={len(regional)}; {summary['excluded_regions']} excluded, "
            f"{summary['unavailable_regions']} unavailable model records). The regional TEST maximum is {finite_text(regional.max())}% "
            f"and {int(regional.gt(100).sum())}/{len(regional)} finite regional errors exceed 100%")
    if not measured:
        text = "Pooled native-cost TEST WMAPE is N/A: no group has an explicitly opened, available positive-denominator population with the four required aggregate fields. Regional medians are not used to approximate it."
    else:
        text = "The sealed TEST aggregate evidence gives "+"; ".join(measured)+". Pooled error uses the sum of native absolute cost errors divided by the sum of true native costs; it is not trip weighted and is not an arithmetic mean of regional percentages. Large regional errors and their contribution to the pooled statistic remain visible rather than being summarized by the median alone."
    if unavailable:
        text += " Groups without a defined pooled result: "+"; ".join(unavailable)+". Missing or unavailable records are excluded, never assigned zero error."
    return text


def i95_coverage(frame):
    fields = ["open_closed_OD_membership_equal", "lost_OD_rows", "gained_OD_rows", "lost_OD_demand", "gained_OD_demand"]
    if not len(frame) or not set(fields).issubset(frame):
        return None
    result = {}
    for name in fields:
        values = frame[name].astype(str).unique()
        if len(values) != 1 or not values[0].strip():
            raise ValueError("Open/closed OD coverage must be one consistent exact ledger: "+name)
        if name == "open_closed_OD_membership_equal":
            if values[0].lower() not in {"true", "false", "1", "0", "1.0", "0.0"}:
                raise ValueError("Invalid cross-scenario membership boolean")
            result[name] = bool(truth(pd.Series(values)).iloc[0])
        else:
            result[name] = float(values[0])
            if not math.isfinite(result[name]) or result[name] < 0:
                raise ValueError("Invalid nonnegative cross-scenario OD coverage")
    if result["open_closed_OD_membership_equal"] and any(result[name] != 0 for name in fields[1:]):
        raise ValueError("Equal cross-scenario membership contradicts lost/gained OD evidence")
    return result


def i95_coverage_note(frame):
    coverage = i95_coverage(frame)
    if coverage is None:
        return "Open-versus-closed served-OD loss/gain evidence is unavailable; native changes cannot be attributed solely to rerouting."
    return (f"Open-to-closed OD: {coverage['lost_OD_rows']:,.0f} rows lost ({coverage['lost_OD_demand']:,.8g} annual thousand tons); "
        f"{coverage['gained_OD_rows']:,.0f} rows gained ({coverage['gained_OD_demand']:,.8g} annual thousand tons). "
        "No demand renormalization is applied.")


def primary(methods, scenario="OPEN"):
    physical = methods.network.astype(str).str.upper().isin(["P", "P-ACNHN", "P_ACNHN", "PACNHN"])
    return methods.loc[(physical | methods.suite_id.eq("s03")) & methods.scenario.eq(scenario)].copy()


def read_contract(path):
    cfg = read_json(path)
    needed = {"phase_id", "step_id", "run_id", "campaign_config", "inputs", "output_root", "tasks", "expected_outputs", "acceptance_tests", "resource_budget", "deadline_epoch", "manuscript_evidence"}
    if needed-set(cfg) or cfg.get("phase_id") != "PACNHN_V1" or cfg.get("step_id") != "publication":
        raise ValueError("Explicit complete PACNHN_V1 publication contract required")
    for key in ["tasks", "expected_outputs", "acceptance_tests", "resource_budget"]:
        if not cfg[key]:
            raise ValueError("Empty publication execution field: "+key)
    ctx = Context(cfg["campaign_config"], step_id="publication")
    from .continuation import validate_continuation
    from .handoff import publication_window
    if ctx.c["original_start_epoch"] != 1790304598 or ctx.c["hard_deadline_epoch"] != 1790304598+52*3600:
        raise ValueError("Original PACNHN clock fields must remain immutable")
    if cfg["run_id"] != ctx.c["run_id"]:
        raise ValueError("Publication identity differs from original campaign")
    authority = validate_continuation(ctx.c, action="publication")
    ctx.publication_continuation_window = None
    if authority is not None:
        from .gates import require_component
        require_component(ctx, "publication")
        ctx.publication_continuation_window = publication_window(ctx.c,
            campaign_config_sha256=sha(cfg["campaign_config"]), stage4_clock=cfg.get("stage4_clock"), clock_amendment=cfg.get("clock_amendment"),
            deadline_epoch=cfg["deadline_epoch"])
        for label in ("stage4_clock", "clock_amendment"):
            binding = ctx.publication_continuation_window[label]
            ctx.bind(binding["path"], label.upper(), binding["sha256"])
    elif cfg["deadline_epoch"] > ctx.c["hard_deadline_epoch"]:
        raise ValueError("Publication deadline exceeds original campaign without authority")
    root = Path(cfg["output_root"]).resolve()
    if not root.is_relative_to(ctx.root.resolve()) or root == ctx.root.resolve():
        raise ValueError("Publication output must be a new child of the campaign output")
    ctx.bind(path, "explicit_publication_configuration")
    ctx.guard(cfg["deadline_epoch"])
    return cfg, ctx, root


def publication_code_paths():
    return [Path(__file__), Path(__file__).with_name("publication_evidence.py"),
        Path(__file__).with_name("publication_results.py"), Path(__file__).with_name("publication_training_curves.py"),
        *sorted(TEMPLATES.glob("*.py"))]


def load_manuscript_evidence(cfg, ctx):
    from .publication_evidence import load_evidence
    pins = ctx.c.get("continuation")
    if not pins:
        raise ValueError("This supplemental publication contract requires the authenticated continuation pins")
    result = load_evidence(cfg["manuscript_evidence"],
        training_result_sha256=pins["training_result_sha256"], registry_sha256=pins["registry_sha256"],
        guard=lambda: ctx.guard(cfg["deadline_epoch"]))
    for key, binding in result["bindings"].items():
        ctx.bind(binding["path"], "manuscript_metadata_"+key, binding["sha256"])
    return result


def load_inputs(cfg, ctx):
    data = {}; records = []
    if set(cfg["inputs"]) != set(SCHEMAS):
        raise ValueError("Exactly the nine documented input datasets must be bound")
    for name, required in SCHEMAS.items():
        ctx.guard(cfg["deadline_epoch"])
        binding = cfg["inputs"][name]
        if not binding.get("sha256"):
            raise ValueError("Every publication input needs a frozen SHA-256")
        path = Path(binding["path"]).resolve(strict=True)
        digest = ctx.bind(path, "analysis_"+name, binding["sha256"])
        frame = pd.read_csv(path, dtype={c: str for c in IDENTITIES}, keep_default_na=False)
        if set(required)-set(frame):
            raise ValueError(name+" missing columns: "+str(sorted(set(required)-set(frame))))
        keep = required+[c for c in OPTIONAL[name] if c in frame and c not in required]
        frame = frame.loc[:, keep].copy()
        if "suite_id" in frame and frame.suite_id.eq("s01").any():
            raise ValueError("Excluded s01 cannot enter publication data")
        # Public inputs contain identifiers, not machine-local data paths.
        for col in frame.select_dtypes(include=["object", "string"]).columns:
            if frame[col].astype(str).str.contains(r"[A-Za-z]:[\\/]|file://|\\\\[^\\]+\\", regex=True).any():
                raise ValueError("Private filesystem path in public column: "+name+"."+col)
        data[name] = frame
        records.append({"dataset": name, "source_basename": path.name, "sha256": digest,
            "bytes": path.stat().st_size, "rows": len(frame), "columns": keep,
            "omitted_private_or_unused_columns": sorted(set(pd.read_csv(path, nrows=0))-set(keep))})
    validate_inputs(data)
    return data, records


def validate_inputs(data):
    n, m, links, learning, i95, keys = [data[k] for k in ["network", "methods", "backbone_links", "learning", "I95", "keys"]]
    if not len(n) or n.suite_id.duplicated().any():
        raise ValueError("Network inventory must have actual, unique suite rows")
    b = nums(n.backbone_links).dropna()
    if not len(b) or not b.eq(87582).all():
        raise ValueError("Native backbone must have the approved 87582 coordinates")
    if m.case_id.duplicated().any():
        raise ValueError("Method case_id is not unique")
    if len(keys) != 200 or keys.link_id.duplicated().any():
        raise ValueError("Publication requires the exact frozen 200-key inventory")
    if links.duplicated(["case_id", "link_id"]).any():
        raise ValueError("Repeated native link observation")
    if set(links.case_id)-set(m.case_id):
        raise ValueError("Link observations have no method disposition")
    if learning.duplicated(["suite_id", "region_index", "family"]).any():
        raise ValueError("Learning export must select one checkpoint per regional family")
    neural_summary(learning)
    pooled_test_summary(learning)
    native_ids = None; native_geo = None; scenario_masks = {}
    for case, frame in links.groupby("case_id", sort=False):
        if len(frame) != 87582 or not np.array_equal(np.sort(nums(frame.native_index)), np.arange(87582)):
            raise ValueError("Each admitted case needs every native B coordinate: "+str(case))
        ordered = frame.assign(_native=nums(frame.native_index)).sort_values("_native")
        geometry = ordered[GEO].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        if not np.isfinite(geometry).all() or (np.abs(geometry[:, [0, 2]]) > 180).any() or (np.abs(geometry[:, [1, 3]]) > 90).any():
            raise ValueError("Invalid native longitude/latitude coordinates")
        if native_ids is None:
            native_ids = ordered.link_id.to_numpy(); native_geo = geometry
        elif not np.array_equal(ordered.link_id, native_ids) or not np.array_equal(geometry, native_geo):
            raise ValueError("Native ordered IDs/geometry differ across cases")
        scenarios = frame.scenario.unique()
        if len(scenarios) != 1 or scenarios[0] not in {"OPEN", "REPRESENTED_I95"}:
            raise ValueError("Each case requires one canonical scenario")
        scenario, mask = scenarios[0], truth(ordered.is_closed).to_numpy()
        if scenario == "OPEN" and mask.any():
            raise ValueError("OPEN native observation declares closed edges")
        if scenario in scenario_masks and not np.array_equal(mask, scenario_masks[scenario]):
            raise ValueError("Scenario's native closure mask differs across methods")
        scenario_masks[scenario] = mask
        if set(frame.loc[truth(frame.is_key), "link_id"]) != set(keys.link_id):
            raise ValueError("Frozen keys changed for case "+str(case))
        ref, pred = nums(frame.reference_flow).to_numpy(), nums(frame.predicted_flow).to_numpy()
        if not np.isfinite(ref).all() or not np.isfinite(pred).all() or np.any(ref < 0) or np.any(pred < 0):
            raise ValueError("Invalid native flow observation")
        denominator = np.maximum(ref, pred)
        expected = np.divide(pred-ref, denominator, out=np.zeros_like(ref), where=denominator>0)
        if not np.allclose(nums(frame.signed_bounded_error), expected, atol=1e-12, rtol=1e-10):
            raise ValueError("Signed bounded error disagrees with native flows")
        method = m.loc[m.case_id.eq(case)].iloc[0]
        if method.status not in {"CONVERGED", "TIME_LIMIT", "STAGNATION"}:
            raise ValueError("Unadmitted method has measured link observations: "+str(case))
        if "certificate_fresh" not in m or not truth(pd.Series([method.certificate_fresh])).iloc[0]:
            raise ValueError("Measured observations require an explicitly fresh certificate: "+str(case))
        if "served_OD_membership_equal" not in m or not truth(pd.Series([method.served_OD_membership_equal])).iloc[0]:
            raise ValueError("Measured comparison requires exact served-OD membership evidence: "+str(case))
        if method.get("residual_kind") not in {"FW_GAP", "VI_RESIDUAL"} or not np.isfinite(pd.to_numeric(method.RG_SPTT, errors="coerce")):
            raise ValueError("Measured comparison has missing final residual semantics")
        for mask, col in [(np.ones(len(frame), bool), "wmape_backbone_percent"), (truth(frame.is_key).to_numpy(), "wmape_key_percent")]:
            denominator = ref[mask].sum()
            expected = 100*np.abs(pred[mask]-ref[mask]).sum()/denominator if denominator > 0 else np.nan
            supplied = float(pd.to_numeric(method[col], errors="coerce"))
            if not (np.isnan(expected) and np.isnan(supplied)) and not np.isclose(expected, supplied, atol=1e-8, rtol=1e-8):
                raise ValueError("Method WMAPE disagrees with source observations: "+str(case)+"/"+col)
    if len(i95):
        i95_coverage(i95)
        if i95.duplicated(["method_id", "link_id"]).any():
            raise ValueError("I95 method identifiers must be unique across suite/role; duplicate link observations")
        for method, frame in i95.groupby("method_id", sort=False):
            if len(frame) != 87582:
                raise ValueError("I95 change population is incomplete: "+str(method))
            if native_ids is not None and set(frame.link_id) != set(native_ids):
                raise ValueError("I95 change IDs differ from native backbone")
            if not np.isfinite(frame[["open_flow", "closed_flow", "delta"]].apply(pd.to_numeric, errors="coerce").to_numpy()).all():
                raise ValueError("I95 change observation is nonfinite")
            if not np.allclose(nums(frame.closed_flow)-nums(frame.open_flow), nums(frame.delta), atol=1e-9, rtol=1e-10):
                raise ValueError("I95 delta is not closed minus open")
            if not np.allclose(nums(frame.loc[truth(frame.is_closed), "closed_flow"]), 0., atol=1e-12, rtol=0):
                raise ValueError("Masked backbone links carry closed flow")
    timing = data["timing"]
    if {"accounting_role", "time_kind", "start_epoch", "end_epoch"}.issubset(timing):
        disjoint = timing.loc[timing.accounting_role.eq("DISJOINT_INTERVAL") & timing.time_kind.eq("MEASURED_WALL")].copy()
        disjoint["start_epoch"] = nums(disjoint.start_epoch); disjoint["end_epoch"] = nums(disjoint.end_epoch)
        disjoint = disjoint.sort_values("start_epoch")
        if len(disjoint):
            bounds = disjoint[["start_epoch", "end_epoch"]].to_numpy(float)
            if not np.isfinite(bounds).all() or (bounds[:, 1] < bounds[:, 0]).any() or (bounds[1:, 0] < bounds[:-1, 1]-1e-6).any():
                raise ValueError("Declared disjoint timing intervals overlap or have missing bounds")
            if not np.allclose(bounds[:, 1]-bounds[:, 0], nums(disjoint.seconds), rtol=1e-6, atol=.02):
                raise ValueError("Declared timing seconds differ from interval widths")


def write_csv_parts(path, frame, *, target_bytes=20_000_000, batch_rows=10_000, guard=lambda: None):
    """Lossless CSV row partitioning with bounded serialization buffers.

    No observation is sampled, rounded beyond the original 17-digit CSV format,
    regrouped or reordered. Batches that do not fit are bisected by row index.
    Every part is a valid CSV with the same header. A lone part keeps the original
    name; multiple parts use deterministic 1-based names and contiguous offsets.
    """
    path = Path(path); csv_leaf(path.name)
    if path.name == "csv_parts.csv" or path.exists():
        raise ValueError("Logical CSV path is reserved or already exists")
    if not isinstance(target_bytes, int) or isinstance(target_bytes, bool) or not 0 < target_bytes <= 20_000_000 or not isinstance(batch_rows, int) or batch_rows <= 0:
        raise ValueError("CSV part bounds must be positive and within the 20 MB target")
    if not len(frame.columns) or frame.columns.duplicated().any() or any(not isinstance(c, str) or "\n" in c or "\r" in c for c in frame):
        raise ValueError("Logical CSV requires distinct nonempty single-line column names")
    header = frame.iloc[:0].to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
    if len(header) > target_bytes:
        raise ValueError("CSV header alone exceeds the part limit")
    records = []; stream = None; position = 0; count = 0; size = 0; checksum = None; part = None

    def batches(first, last):
        guard()
        data = frame.iloc[first:last].to_csv(index=False, header=False, float_format="%.17g", lineterminator="\n").encode("utf-8")
        guard()
        if len(data)+len(header) <= target_bytes:
            yield first, last, data
        elif last-first <= 1:
            raise ValueError("A complete CSV observation exceeds the part limit; explicit storage redesign required, never truncate")
        else:
            middle = (first+last)//2
            yield from batches(first, middle); yield from batches(middle, last)

    def finish():
        nonlocal stream, position, count, size
        if stream is None:
            return
        stream.close(); stream = None
        records.append(dict(logical_file=path.name, part_file=part.name, part_index=len(records)+1,
            row_start=position, row_count=count, total_rows=len(frame), size_bytes=size,
            sha256=checksum.hexdigest(), header_bytes=len(header), header_sha256=hashlib.sha256(header).hexdigest(),
            columns_json=json.dumps(list(frame), ensure_ascii=False, separators=(",", ":")),
            text_columns_json=json.dumps([c for c in frame if frame[c].dtype.kind in "OUS" or isinstance(frame[c].dtype, pd.CategoricalDtype)], ensure_ascii=False, separators=(",", ":"))))
        position += count; count = 0; size = 0

    def begin():
        nonlocal stream, part, checksum, size
        guard()
        part = path.with_name(path.stem+f".part{len(records)+1:05d}.csv")
        stream = part.open("xb"); checksum = hashlib.sha256(); stream.write(header); checksum.update(header); size = len(header)

    try:
        begin()
        for start in range(0, len(frame), batch_rows):
            for first, last, data in batches(start, min(start+batch_rows, len(frame))):
                if count and size+len(data) > target_bytes:
                    finish(); begin()
                stream.write(data); checksum.update(data); size += len(data); count += last-first
        finish()
    finally:
        if stream is not None:
            stream.close()
    if len(records) == 1:
        path.with_name(records[0]["part_file"]).rename(path)
        records[0]["part_file"] = path.name
    if position != len(frame):
        raise ValueError("CSV part writer did not retain every source row")
    return records


def public_csv(path, frame, *, part_registry=None, target_bytes=20_000_000, guard=lambda: None):
    if part_registry is None:
        frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")
    else:
        part_registry.extend(write_csv_parts(path, frame, target_bytes=target_bytes, guard=guard))


def dictionary(frame, name):
    rows = []
    exact = {
        "RG_SPTT": ("dimensionless", "Fresh or recorded normalized assignment residual; residual_kind distinguishes potential FW gap from empirical cost-only VI residual."),
        "signed_bounded_error": ("dimensionless", "(predicted_flow-reference_flow)/max(predicted_flow,reference_flow); zero/zero=0."),
        "is_key": ("boolean", "Membership in the one 200-native-link subset frozen from open P."),
        "is_closed": ("boolean", "Coordinate is closed in the named scenario; OPEN link observations are all false. Paired I95 data records the represented consolidated-backbone mask."),
        "delta": ("annual thousand tons", "closed_flow minus open_flow on the same native directed backbone coordinate."),
        "neural_amplitude": ("model parameter", "Recorded scalar neural amplitude; not a measured fractional neural contribution."),
        "target_met": ("boolean", "Recorded per-region/family TRAIN <1% qualification; does not imply backbone accuracy."),
        "resolution": ("dimensionless", "Approved residual-region resolution with fixed backbone."),
        "test_opened": ("boolean", "Explicit record that the frozen blind TEST was opened after selection."),
        "TEST_absolute_error_sum": ("seconds summed over cost coordinates", "Sealed blind TEST sum of absolute native predicted-minus-true cost errors; not inferred from a regional percentage and not trip weighted."),
        "TEST_true_cost_sum": ("seconds summed over cost coordinates", "Sealed blind TEST sum of true native costs; pooled WMAPE denominator is the sum of these values over admitted regional rows."),
        "TEST_coordinates": ("cost-coordinate count", "Actual sealed TEST cost coordinates contributing to the native sums; includes structural-zero coordinates."),
        "TEST_inputs": ("region-input count", "Actual sealed TEST inputs for this regional model; sums across regions count region-input observations, not distinct national OD vectors."),
        "absolute_error_sum": ("seconds summed over cost coordinates", "Sum of explicit sealed TEST absolute-error sums over this suite/family's contributing regional rows."),
        "true_cost_sum": ("seconds summed over cost coordinates", "Sum of explicit sealed TEST true-cost sums over exactly the same contributing regional rows."),
        "coordinates": ("cost-coordinate count", "Pooled TEST cost-coordinate population; includes structural-zero coordinates."),
        "inputs": ("region-input count", "Sum of TEST inputs over contributing regional models, not distinct national inputs; repeated across model families."),
        "contributing_regions": ("regional-model count", "Opened AVAILABLE rows with all four finite TEST aggregates and positive input/coordinate counts."),
        "excluded_regions": ("regional-model count", "Recorded regional-model rows omitted from the pooled statistic because required admission or aggregate evidence is absent."),
        "selected_checkpoint": ("identifier", "Actually selected candidate ID; do not interpret a candidate-ID string as an optimizer update count."),
        "selected_is_initialization": ("boolean", "Explicit producer evidence that the selected checkpoint is the initialization."),
        "selected_optimizer_updates": ("count", "Actual optimizer updates represented by the selected checkpoint; zero is meaningful."),
        "selected_boosting_rounds": ("count", "Actual boosting rounds represented by the selected XGBoost model; not a candidate-ID interpretation."),
        "observed_increment_absolute_sum": ("seconds summed over prediction coordinates", "Measured sum of absolute native ICNN cost increments versus its OWN power4 component on the named full unique TRAIN dataset; not improvement over separately fitted power4 or trip-weighted total travel time."),
        "maximum_native_cost_increment": ("seconds", "Maximum native cost increment over the selected ICNN's own power4 component on the named full TRAIN coordinates."),
        "nonzero_prediction_coordinates": ("prediction-coordinate count", "Full unique TRAIN coordinates with strictly nonzero floating-point neural cost increment; denominator train_rows*pairs includes structural-zero coordinates."),
        "structurally_active": ("boolean", "Recorded structural neural-branch activity; distinct from observed nonzero cost increment and positive scalar amplitude."),
        "neural_contribution_train_dataset_identity": ("dataset hash", "Exact current full unique TRAIN dataset underlying measured neural branch increments; must match selected model train_dataset_identity."),
        "evaluation_kind": ("category", "Recorded producer tag: NEW_CHECKPOINT means TRAIN identity matches the candidate's training identity; multiple VALID identities can reuse this tag. It is not a unique creation event."),
        "checkpoint_ordinal": ("ordinal", "Chronological first recorded NEW_CHECKPOINT evaluation per distinct candidate within region/family; not optimizer updates or original block index."),
        "source_observation_index": ("zero-based row", "Position in the unchanged canonical training-curve observations supplied to publication."),
        "selection_reason": ("category", "Explicit plotted/retained-only chronology disposition; no raw observation is removed."),
        "open_closed_OD_membership_equal": ("boolean", "Exact OPEN-versus-I95 served-OD membership equality; distinct from within-scenario AC-versus-P equality."),
        "lost_OD_rows": ("OD row count", "Exact rows served OPEN and unserved represented-I95, without renormalization."),
        "gained_OD_rows": ("OD row count", "Exact rows unserved OPEN and served represented-I95."),
        "lost_OD_demand": ("annual thousand tons", "Original demand on rows served OPEN but unserved represented-I95; not a sum of link-flow changes."),
        "gained_OD_demand": ("annual thousand tons", "Original demand on rows newly served represented-I95."),
        "train_wmape": ("percent", "Full current TRAIN cost ratio-of-sums WMAPE at this audit; dataset can grow between blocks."),
        "valid_wmape": ("percent", "Fixed VALID cost ratio-of-sums WMAPE at this audit."),
        "assignment_role": ("category", "REFERENCE_SELF, independent control or surrogate; self-zero is not an independent solver result."),
        "accounting_role": ("category", "DISJOINT_INTERVAL or NESTED_DETAIL; overlapping and worker times must not be added to wall time."),
    }
    for col in frame:
        if col in exact:
            unit, meaning = exact[col]
        elif "longitude" in col:
            unit, meaning = "degrees east", "Actual native endpoint longitude; maps use endpoint chords, not full road polylines."
        elif "latitude" in col:
            unit, meaning = "degrees north", "Actual native endpoint latitude; contiguous-US crop is display only."
        elif "WMAPE" in col or "wmape" in col:
            unit, meaning = ("percent" if "percent" in col or "WMAPE" in col else "as supplied; see exporter"), "Ratio-of-sums absolute error under the named TRAIN/VALID/TEST or native-link population; not mean percentage error."
        elif col.endswith("flow"):
            unit, meaning = "annual thousand tons", "Native directed-link traversal flow, retaining zero coordinates."
        elif "seconds" in col or col == "seconds":
            unit, meaning = "seconds", "Recorded elapsed quantity with scope given by timing_boundary and status."
        elif col.endswith("epoch"):
            unit, meaning = "Unix seconds", "Recorded wall-clock boundary; does not reset the original campaign clock."
        else:
            unit, meaning = "identifier/category/count or display text", "Source/export field "+col+"; blanks mean unavailable, never an imputed zero."
        rows.append(dict(file=name, column=col, units=unit, meaning=meaning))
    return rows


def asset_folder(root, asset_id, frames, records, *, table=None, columns=None, panels=None, caption=None, guard=lambda: None):
    folder = root/"numerical_data"/asset_id; folder.mkdir(parents=True, exist_ok=False)
    parts = []
    def local_csv(name, frame):
        public_csv(folder/name, frame, part_registry=parts, guard=guard)
    caption = caption or CAPTIONS[asset_id]
    metadata = pd.DataFrame([dict(asset_id=asset_id, title=TITLES[asset_id], caption=caption)])
    local_csv("asset.csv", metadata)
    dictionaries = dictionary(metadata, "asset.csv")
    for name, frame in frames.items():
        local_csv(name, frame); dictionaries.extend(dictionary(frame, name))
    if table is not None:
        local_csv("table.csv", table)
        column_frame = pd.DataFrame(columns, columns=["column", "label", "width"])
        local_csv("columns.csv", column_frame)
        dictionaries.extend(dictionary(column_frame, "columns.csv"))
        dictionaries.extend(dictionary(table, "table.csv"))
        script = "build_table.py"
    else:
        panel_frame = pd.DataFrame(panels, columns=PANEL_COLUMNS).fillna("")
        local_csv("panels.csv", panel_frame)
        dictionaries.extend(dictionary(panel_frame, "panels.csv"))
        script = "plot.py"
    local_csv("data_dictionary.csv", pd.DataFrame(dictionaries))
    public_csv(folder/"csv_parts.csv", pd.DataFrame(parts, columns=CSV_PART_COLUMNS))
    if (folder/"csv_parts.csv").stat().st_size > 20_000_000:
        raise ValueError("CSV part ledger exceeds the bounded asset metadata limit")
    template = (TEMPLATES/script).read_text(encoding="utf-8")
    marker = "# EMBED_LOCAL_LOGICAL_CSV_SUPPORT_HERE"
    if template.count(marker) != 1:
        raise ValueError("Standalone logical-CSV support insertion marker is missing or duplicated")
    (folder/script).write_text(template.replace(marker, (TEMPLATES/"csv_support.py").read_text(encoding="utf-8")), encoding="utf-8")
    write_json(folder/"provenance.json", {"schema": "pacnhn.public_asset.v1", "asset_id": asset_id,
        "source_datasets": records, "input_rule": "Only local CSV scientific inputs; no machine-local paths",
        "script_sha256": sha(folder/script), "csv_parts_sha256": sha(folder/"csv_parts.csv"),
        "logical_csv_count": len({p["logical_file"] for p in parts}), "physical_csv_parts": len(parts),
        "csv_part_target_bytes": 20_000_000, "rows_retained_without_sampling": True,
        "visual_qa": "PENDING_ACTUAL_RENDER_AND_INSPECTION"})
    (folder/"README.md").write_text("# "+TITLES[asset_id]+"\n\n"+caption+"\n\n"
        "Run `python "+script+"` from any working directory. Dependencies: Python, NumPy, pandas, Matplotlib. "
        "All scientific inputs are CSV files in this folder. Large logical CSV datasets are losslessly divided into ordered parts of at most 20,000,000 bytes; csv_parts.csv binds each part's row range, schema, size and SHA-256. "
        "A single embedded CSV reader reconstructs the complete logical dataset without external code or paths. Keep the ledger and every part. "
        "The renderer writes PNG/PDF, a CSV read/hash audit and a render receipt; tables also write a TeX fragment. "
        "No network, model, repository, external geometry or cached scientific array is needed. Underlying observation rows are retained. "
        "No unavailable measurement is replaced by zero. Missing figure panels produce explicit dispositions, not empty plotted boxes. "
        "A render receipt is not visual-QA approval.\n", encoding="utf-8")
    return folder


def table_frame(rows, columns):
    if not rows:
        rows = [{col: "N/A" for col, _, _ in columns}]
        if "status" in rows[0]:
            rows[0]["status"] = "NO_MEASUREMENT_IN_EXPORT"
        if "note" in rows[0]:
            rows[0]["note"] = "No observation rows were supplied."
    return pd.DataFrame(rows).reindex(columns=[c[0] for c in columns]).fillna("N/A")


def build_tables(root, d, records, guard=lambda: None):
    n, m, learning, timing = [d[k] for k in ["network", "methods", "learning", "timing"]]
    columns = [("suite", "Suite / resolution", 1.2), ("nodes", "Nodes", .7), ("edges", "Edges", .7), ("B", "Native B", .7),
        ("R", "Regional arcs", .85), ("regions", "Regions / active", 1), ("pairs", "Active pairs", .8), ("served", "Served OD rows open / closed", 1.5), ("status", "Status", 1.3)]
    rows = [dict(suite=f"{r.suite_id} / {finite_text(r.resolution)}", nodes=finite_text(r.nodes, 9), edges=finite_text(r.edges, 9),
        B=finite_text(r.backbone_links, 9), R=finite_text(r.regional_links, 9), regions=finite_text(r.regions, 8)+" / "+finite_text(r.active_regions, 8),
        pairs=finite_text(r.active_pairs, 8), served=finite_text(r.served_rows_open, 10)+" / "+finite_text(r.served_rows_closed, 10),
        status=field(r, "status")) for _, r in n.iterrows()]
    asset_folder(root, "table1", {"observations_network.csv": n}, records, table=table_frame(rows, columns), columns=columns, guard=guard)

    columns = [("case", "Case / role", 1.65), ("solver", "Solver / residual kind", 1.15), ("B", "Full B WMAPE (%)", .7),
        ("key", "200-key WMAPE (%)", .7), ("residual", "Fresh residual", .8), ("time", "Recorded s / cap s", 1.1),
        ("status", "Status", 1.1), ("reference", "Reference / quality / served membership", 1.9)]
    rows = []
    for _, r in primary(m).iterrows():
        rows.append(dict(case=str(r.case_id)+" / "+str(r.assignment_role), solver=str(r.solver)+" / "+field(r, "residual_kind"),
            B=finite_text(r.wmape_backbone_percent), key=finite_text(r.wmape_key_percent),
            residual=finite_text(r.RG_SPTT)+"; fresh="+field(r, "certificate_fresh"), time=finite_text(r.elapsed_seconds, 6)+" / "+finite_text(r.get("query_cap_seconds"), 6),
            status=r.status, reference=str(r.reference_case_id)+" / "+field(r, "reference_status")+" RG="+finite_text(r.get("reference_RG_SPTT"))+" / equal="+field(r, "served_OD_membership_equal")))
    asset_folder(root, "table2", {"observations_methods.csv": primary(m), "observations_links.csv": d["backbone_links"].loc[d["backbone_links"].case_id.isin(primary(m).case_id)], "observations_keys.csv": d["keys"]}, records, table=table_frame(rows, columns), columns=columns, guard=guard)

    columns = [("group", "Suite / family", 1), ("coverage", "Recorded regions / TRAIN target met", 1.05),
        ("TRAIN", "TRAIN median % (n)", .85), ("VALID", "VALID median % (n)", .85), ("TEST", "Opened TEST median % (n)", .95), ("pooled_TEST", "Pooled TEST % (regions)", 1.0),
        ("checkpoint", "Known selected IDs; init / known; actual counters", 1.45),
        ("neural", "TRAIN neural sum / max (s); nonzero coords; structural / amplitude", 2.2),
        ("status", "Status counts / label origins", 1.7)]
    rows = []; pooled_rows = []
    for (suite, family), frame in learning.groupby(["suite_id", "family"], sort=True):
        result = dict(group=f"{suite} / {family}", coverage=f"{len(frame)} / {int(truth(frame.target_met).sum())}")
        for split in ["TRAIN", "VALID", "TEST"]:
            values = nums(frame[split+"_WMAPE_percent"])
            if split == "TEST":
                opened = truth(frame.test_opened) if "test_opened" in frame else pd.Series(False, index=frame.index)
                values = values.loc[opened]
            values = values[np.isfinite(values)]
            result[split] = finite_text(values.median())+f" ({len(values)})"
        pooled = pooled_test_summary(frame)
        pooled_rows.append(dict(suite_id=suite, family=family, **pooled))
        result["pooled_TEST"] = finite_text(pooled["wmape_percent"])+f" ({pooled['contributing_regions']}/{pooled['recorded_regions']})"
        selected, initialized, classified = checkpoint_counts(frame)
        result["checkpoint"] = f"IDs {selected}"
        if family == "ICNN":
            result["checkpoint"] += f"; init {initialized if initialized is not None else 'N/A'}/{classified}; "+selected_counter_text(frame, "selected_optimizer_updates", "updates")
            neural = neural_summary(frame)
            result["neural"] = (f"n={neural['measured_regions']}; sum {finite_text(neural['increment_absolute_sum'])}; max {finite_text(neural['maximum_native_increment'])}; "
                f"nnz {neural['nonzero_coordinates'] if neural['nonzero_coordinates'] is not None else 'N/A'}/{neural['prediction_coordinates'] if neural['prediction_coordinates'] is not None else 'N/A'}; "
                f"active {neural['structurally_active']}/{neural['activity_known']}; amp+ {neural['positive_amplitude']}/{neural['amplitude_known']}")
        else:
            result["neural"] = "N/A (non-ICNN)"
            if family == "XGBOOST":
                result["checkpoint"] += "; "+selected_counter_text(frame, "selected_boosting_rounds", "rounds")
        result["status"] = "; ".join(f"{key}:{count}" for key, count in frame.status.value_counts(dropna=False).items())
        result["status"] += " / labels="+(";".join(sorted(frame.label_origin.astype(str).unique())) if "label_origin" in frame else "UNAVAILABLE")
        rows.append(result)
    asset_folder(root, "table3", {"observations_learning.csv": learning, "pooled_test_summary.csv": pd.DataFrame(pooled_rows, columns=["suite_id", "family"]+list(pooled_test_summary(learning)))}, records, table=table_frame(rows, columns), columns=columns, guard=guard)

    columns = [("suite", "Suite / resolution", 1.15), ("case", "Case / scenario", 1.8), ("status", "Disposition", 1.3),
        ("pairs", "Active pairs", .8), ("B", "Full B WMAPE (%)", .8), ("key", "200-key WMAPE (%)", .8), ("note", "Coverage", 1.8)]
    rows = []
    for _, network in n.iterrows():
        physical = m.network.astype(str).str.upper().isin(["P", "P-ACNHN", "P_ACNHN", "PACNHN"])
        cases = m.loc[physical] if str(network.suite_id).upper() == "P" else m.loc[m.suite_id.eq(network.suite_id)]
        if not len(cases):
            rows.append(dict(suite=str(network.suite_id)+" / "+finite_text(network.resolution), case="N/A", status="NO_METHOD_ROW_IN_EXPORT", pairs=finite_text(network.active_pairs, 9), B="N/A", key="N/A", note="No job is inferred from network existence."))
        for _, r in cases.iterrows():
            rows.append(dict(suite=str(network.suite_id)+" / "+finite_text(network.resolution), case=str(r.case_id)+" / "+str(r.scenario), status=r.status,
                pairs=finite_text(network.active_pairs, 9), B=finite_text(r.wmape_backbone_percent), key=finite_text(r.wmape_key_percent), note=field(r, "note")))
    asset_folder(root, "table4", {"observations_network.csv": n, "observations_methods.csv": m,
        "observations_links.csv": d["backbone_links"]}, records, table=table_frame(rows, columns), columns=columns, guard=guard)

    columns = [("method", "Method / case", 1.45), ("gain", "Gain", .75), ("loss", "Loss magnitude", .85), ("net", "Net / absolute change", 1.1),
        ("accuracy", "Closed full B / key WMAPE (%)", 1.2), ("membership", "Within-scenario AC=P served OD", 1.05), ("status", "Closed status / reference", 1.8)]
    rows = []; closed = m.loc[m.scenario.eq("REPRESENTED_I95")]
    for method, frame in d["I95"].groupby("method_id", sort=True):
        values = nums(frame.delta).to_numpy(float)
        candidates = closed.loc[closed.method_id.eq(method)]
        if "case_id" in frame and len(frame.case_id.astype(str).unique()) == 1:
            candidates = closed.loc[closed.case_id.eq(str(frame.case_id.iloc[0]))]
        if len(candidates) != 1:
            accuracy, membership, status = "N/A", "N/A", "CLOSED_METHOD_LINK_AMBIGUOUS_OR_UNAVAILABLE"
        else:
            r = candidates.iloc[0]
            accuracy = finite_text(r.wmape_backbone_percent)+" / "+finite_text(r.wmape_key_percent)
            membership = field(r, "served_OD_membership_equal")
            status = str(r.status)+" / "+str(r.reference_case_id)+" "+field(r, "reference_status")+" RG="+finite_text(r.get("reference_RG_SPTT"))
        rows.append(dict(method=str(method), gain=finite_text(values[values>0].sum(), 6), loss=finite_text(-values[values<0].sum(), 6),
            net=finite_text(values.sum(), 6)+" / "+finite_text(np.abs(values).sum(), 6), accuracy=accuracy, membership=membership, status=status))
    # A closed method may exist without a paired open/closed link export.
    represented = set(d["I95"].method_id)
    for _, r in closed.loc[~closed.method_id.isin(represented)].iterrows():
        rows.append(dict(method=r.case_id, gain="N/A", loss="N/A", net="N/A", accuracy=finite_text(r.wmape_backbone_percent)+" / "+finite_text(r.wmape_key_percent),
            membership=field(r, "served_OD_membership_equal"), status=str(r.status)+"; no paired change export"))
    asset_folder(root, "table5", {"observations_I95.csv": d["I95"], "observations_closed_methods.csv": closed,
        "observations_closed_links.csv": d["backbone_links"].loc[d["backbone_links"].scenario.eq("REPRESENTED_I95")],
        "observations_keys.csv": d["keys"]}, records, table=table_frame(rows, columns), columns=columns, guard=guard,
        caption=CAPTIONS["table5"]+" "+i95_coverage_note(d["I95"]))

    columns = [("task", "Task / category", 1.5), ("seconds", "Recorded s / cap s", 1), ("status", "Status", 1),
        ("kind", "Time kind / accounting role", 1.5), ("boundary", "Exact timing boundary", 2.4), ("parent", "Parent / note", 1.5)]
    display = timing.category.isin(["COMPLETE_QUERY", "ONLINE_ASSIGNMENT_COMPLETE_QUERY", "WHOLE_MODEL_INFERENCE",
        "OFFLINE_TRAINING_STAGE_WALL", "OFFLINE_STAGE1_BROAD_WALL", "ONLINE_ASSIGNMENT_STAGE3_BROAD_WALL",
        "WORKER_COMPLETE_SLICE"]) | timing.category.astype(str).str.startswith("HISTORICAL_")
    display_ledger = timing[["task", "category"]].copy()
    display_ledger["displayed"] = display
    display_ledger["policy"] = "Complete query/call/stage/worker spans and historical context; nested components retained in observations_timing.csv"
    rows = [dict(task=str(r.task)+" / "+str(r.category), seconds=finite_text(r.seconds, 7)+" / "+finite_text(r.get("cap_seconds"), 7), status=r.status,
        kind=field(r, "time_kind")+" / "+field(r, "accounting_role"), boundary=text_value(r.timing_boundary), parent=field(r, "parent_task")+" / "+field(r, "note")) for _, r in timing.loc[display].iterrows()]
    asset_folder(root, "table6", {"observations_timing.csv": timing, "observations_display_policy.csv": display_ledger}, records,
        table=table_frame(rows, columns), columns=columns, guard=guard)


def panel(letter, title, kind, source, **kwargs):
    return dict(panel_id=letter, title=f"({letter}) "+title, kind=kind, source=source, **kwargs)


def build_figures(root, d, records, guard=lambda: None):
    methods = case_labels(d["methods"])
    method_columns = ["case_id", "method_id", "suite_id", "label", "status", "assignment_role"]
    links = d["backbone_links"].merge(methods[method_columns], on="case_id", validate="many_to_one")
    p = primary(methods)
    independent_ids = p.loc[~p.assignment_role.eq("REFERENCE_SELF"), "case_id"]
    displayed = links.loc[links.case_id.isin(independent_ids)].copy()
    # A label identifies the actual case, not a family assumed to have run.
    displayed["key_filter"] = np.where(truth(displayed.is_key), "KEY", "OTHER")
    keys = d["keys"].copy()
    if not set(GEO).issubset(keys):
        reference = links.loc[links.case_id.eq("P_REFERENCE_open"), ["link_id"]+GEO]
        if len(reference) != 87582:
            raise ValueError("Frozen key geometry requires explicit columns or the open reference native rows")
        keys = keys.merge(reference, on="link_id", validate="one_to_one")
    panels = [
        panel("a", "Frozen 200-key native links", "map", "keys.csv", point_markers="true", note="The open-P-only selection is reused unchanged for all methods, resolutions and scenarios. Native endpoint chords with midpoint markers for visibility; contiguous-US display crop only."),
        panel("b", "Full-backbone signed bounded error", "ecdf", "links.csv", x="signed_bounded_error", group="label", xlabel="Signed bounded error", ylabel="Fraction of native backbone links", note="Each series retains all native B coordinates, including zero-reference and zero/zero links. Reference-self is excluded. Status and reference quality are in Table 2."),
        panel("c", "Frozen 200-key signed bounded error", "ecdf", "links.csv", x="signed_bounded_error", group="label", filter_column="key_filter", filter_value="KEY", xlabel="Signed bounded error", ylabel="Fraction of the 200 frozen keys", note="The denominator is the same 200 native IDs for each available case. This is a geographically dispersed, high-flow subset, not a random representative sample."),
        panel("d", "Native flow on the 200 frozen keys", "scatter", "links.csv", x="reference_flow", y="predicted_flow", group="label", filter_column="key_filter", filter_value="KEY", xlabel="Open P reference (annual thousand tons)", ylabel="Compared native flow (annual thousand tons)", xscale="symlog", yscale="symlog", note="Signed native flow is nonnegative; symmetric-log axes preserve zero. Reference-self is excluded. Approximate-reference error remains part of this comparison."),
    ]
    asset_folder(root, "figure3", {"links.csv": displayed, "keys.csv": keys, "observations_methods.csv": p}, records, panels=panels, guard=guard)

    learning = d["learning"]; learning_rows = []
    for split in ["TRAIN", "VALID", "TEST"]:
        frame = learning.copy()
        if split == "TEST":
            frame = frame.loc[truth(frame.test_opened)] if "test_opened" in frame else frame.iloc[:0]
        frame["split"] = split; frame["error_percent"] = nums(frame[split+"_WMAPE_percent"])
        frame["label"] = frame.suite_id.astype(str)+" / "+frame.family.astype(str)
        learning_rows.append(frame)
    long_learning = pd.concat(learning_rows, ignore_index=True)
    curves = d["training_curves"].copy()
    if "evaluation_kind" in curves:
        from .publication_training_curves import select_first_candidate_evaluations
        plotted_curves, curve_selection = select_first_candidate_evaluations(curves)
        selection_rows = pd.DataFrame(curve_selection["observations"])
    else:
        plotted_curves = curves.iloc[:0].copy()
        plotted_curves["checkpoint_ordinal"] = pd.Series(dtype=float)
        selection_rows = pd.DataFrame(columns=["source_observation_index", "plotted", "selection_reason"])
    curve_rows = []
    for split in ["train", "valid"]:
        frame = plotted_curves.copy(); frame["error_percent"] = nums(frame[split+"_wmape"])
        frame["label"] = frame.suite_id.astype(str)+" / "+frame.family.astype(str)+" / "+split.upper()
        curve_rows.append(frame)
    long_curves = pd.concat(curve_rows, ignore_index=True)
    panels = [panel(letter, split+" regional cost error", "ecdf", "learning.csv", x="error_percent", group="label", filter_column="split", filter_value=split,
        xlabel=split+" regional WMAPE (%)", ylabel="Fraction of recorded regional models", xscale="symlog", xlinthresh="1",
        note="Unweighted regional CDF; finite n is displayed for each series. TRAIN <1% is a per-region/family target, not a backbone-flow guarantee. TEST uses only explicitly opened records.")
        for letter, split in zip("abc", ["TRAIN", "VALID", "TEST"])]
    panels.append(panel("d", "First recorded candidate evaluations", "median_curve", "training_curves.csv", x="checkpoint_ordinal", y="error_percent", group="label", xlabel="Candidate evaluation ordinal within region/family", ylabel="Median regional cost WMAPE (%)", yscale="symlog", ylinthresh="1",
        note="One earliest saved NEW_CHECKPOINT evaluation per candidate; repeated evaluations and all original records are retained. This ordinal is not an optimizer-step count. Region coverage and historical TRAIN/admitted VALID identities can change; this is not a fixed-data ablation."))
    asset_folder(root, "figure4", {"learning.csv": long_learning, "training_curves.csv": long_curves,
        "observations_learning.csv": learning, "observations_training_curves.csv": curves,
        "observations_curve_selection.csv": selection_rows}, records, panels=panels, guard=guard)

    network = d["network"].copy()
    network["label"] = network.suite_id.astype(str)+" (r="+network.resolution.map(finite_text)+")"
    resolution = methods.loc[methods.scenario.eq("OPEN") & methods.suite_id.isin(network.suite_id)].copy()
    resolution["family"] = resolution.method_id.astype(str).str.split("/").str[-1]
    # Keep suite on the categorical x axis; values are the approved resolutions.
    resolution = resolution.merge(network[["suite_id", "resolution"]], on="suite_id", validate="many_to_one")
    resolution["resolution_label"] = resolution.suite_id.astype(str)+" / "+resolution.resolution.map(finite_text)
    panels = [panel("a", "Fixed native-backbone coordinates", "bar", "network.csv", x="label", y="backbone_links", xlabel="Native directed B coordinates", note="Identical coordinate count alone does not establish identity; exact ID, endpoint, coefficient and mask invariance is a separate campaign gate."),
        panel("b", "Regional active-pair coverage", "bar", "network.csv", x="label", y="active_pairs", xlabel="Active regional pairs", note="P has detailed regional arcs rather than learned pair coordinates. Missing entries remain absent, not zero. Network existence does not imply training or assignment completion."),
        panel("c", "Available open native-backbone errors", "points", "methods.csv", x="resolution_label", y="wmape_backbone_percent", group="family", xlabel="Suite / residual-region resolution", ylabel="Full native B WMAPE (%)", note="Only available measurements are plotted; missing resolutions are not connected or filled. All errors use one fixed native B and the frozen open P reference."),
        panel("d", "Available open 200-key errors", "points", "methods.csv", x="resolution_label", y="wmape_key_percent", group="family", xlabel="Suite / residual-region resolution", ylabel="Frozen 200-key WMAPE (%)", note="Every measurement uses the same key IDs frozen before candidate-error inspection. The per-region TRAIN qualification gate and skipped jobs are reported separately.")]
    asset_folder(root, "figure5", {"network.csv": network, "methods.csv": resolution, "observations_methods.csv": methods}, records, panels=panels, guard=guard)

    changes = d["I95"].copy()
    mask = changes.loc[truth(changes.is_closed)].drop_duplicates("link_id")
    panels = [panel("a", "Represented consolidated-backbone closure", "map", "mask.csv", note="Only the represented native B mask is shown. Hidden physical interiors and unrepresented physical I-95 links are omitted. Directed native endpoint chords; contiguous-US display crop only.")]
    # The primary s03 ICNN/XGBoost maps have explicit IDs; missing data is a
    # disposition rather than substitution with an analytic family or reference.
    for letter, family in [("b", "ICNN"), ("c", "XGBOOST")]:
        panels.append(panel(letter, family+" closed-minus-open redistribution", "map", "changes.csv", y="delta", filter_column="method_id", filter_value="s03/"+family, symmetric="true",
            colorbar="Native flow change (annual thousand tons)", note="Closed minus open flow on the same native B coordinates. Positive and negative changes count link traversals; they are not OD demand creation or loss. No missing model family is substituted."))
    panels.append(panel("d", "Native link-traversal gain and loss", "gain_loss", "changes.csv", y="delta", group="method_id", xlabel="Magnitude (annual thousand tons)",
        note="Native changes combine rerouting and any newly unserved OD demand; they are not pure rerouting at unchanged served demand. "+i95_coverage_note(changes)))
    asset_folder(root, "figure6", {"changes.csv": changes, "mask.csv": mask, "observations_methods.csv": methods.loc[methods.scenario.eq("REPRESENTED_I95")]}, records, panels=panels, guard=guard,
        caption=CAPTIONS["figure6"]+" "+i95_coverage_note(changes))

    c = d["convergence"].copy()
    # Residual semantics come from the admitted method certificate, not solver
    # name guessing. A missing kind cannot be promoted to a potential claim.
    if "residual_kind" in c:
        c = c.rename(columns={"residual_kind": "history_residual_kind"})
    method_cols = ["case_id", "label", "scenario", "status"]+[col for col in ["residual_kind", "nominal_tolerance"] if col in methods]
    c = c.merge(methods[method_cols], on="case_id", validate="many_to_one")
    if "residual_kind" not in c:
        c["residual_kind"] = "UNAVAILABLE"
    c["panel_group"] = c.scenario.astype(str)+" / "+c.residual_kind.astype(str)
    panels = []
    for letter, (scenario, kind) in zip("abcd", [("OPEN", "FW_GAP"), ("OPEN", "VI_RESIDUAL"), ("REPRESENTED_I95", "FW_GAP"), ("REPRESENTED_I95", "VI_RESIDUAL")]):
        subset = c.loc[c.panel_group.eq(scenario+" / "+kind)]
        tolerances = nums(subset.nominal_tolerance).dropna().unique() if "nominal_tolerance" in subset else []
        panels.append(panel(letter, scenario+" / "+kind, "line", "convergence.csv", x="elapsed_seconds", y="RG_SPTT", group="label", filter_column="panel_group", filter_value=scenario+" / "+kind,
            xlabel="Recorded elapsed time (s)", ylabel="Normalized "+("FW gap" if kind == "FW_GAP" else "cost-only VI residual"), yscale="symlog", ylinthresh="0.000001", threshold=str(tolerances[0]) if len(tolerances)==1 else "",
            note="Final statuses and fresh state certificates remain authoritative. TIME_LIMIT is not convergence. Reference and independent controls are distinct; a VI residual is not a Beckmann potential gap or a convergence guarantee."))
    asset_folder(root, "figure7", {"convergence.csv": c, "observations_methods.csv": methods}, records, panels=panels, guard=guard)

    timing = d["timing"].copy()
    if {"time_kind", "accounting_role"}.issubset(timing):
        offline = timing.loc[timing.time_kind.eq("MEASURED_WALL") & timing.accounting_role.eq("DISJOINT_INTERVAL") & ~timing.category.astype(str).str.upper().str.contains("ONLINE|ASSIGNMENT")].copy()
    else:
        offline = timing.iloc[:0].copy()
    offline["label"] = offline.task.astype(str).str.replace("_", " ", regex=False)
    panels = [panel("a", "Open recorded assignment time", "bar", "methods.csv", x="label", y="elapsed_seconds", filter_column="scenario", filter_value="OPEN", xlabel="Recorded query wall seconds", note="TIME_LIMIT bars are censored completion times. The recorded boundary excludes only the explicitly identified post-query writes/admission. Reference preparation and independent cold-query controls have distinct roles."),
        panel("b", "Represented-I95 recorded assignment time", "bar", "methods.csv", x="label", y="elapsed_seconds", filter_column="scenario", filter_value="REPRESENTED_I95", xlabel="Recorded query wall seconds", note="Actual recorded times and caps, including approximate references. Unequal precision or different methods cannot establish a like-for-like converged speedup. Exact scopes are retained in Table 6."),
        panel("c", "Disjoint measured offline intervals", "bar", "offline.csv", x="label", y="seconds", xlabel="Measured wall seconds", note="Only explicitly declared, nonoverlapping measured-wall intervals are shown. Worker totals, nested details and estimates are excluded. No offline/online amortization threshold is inferred.")]
    asset_folder(root, "figure8", {"methods.csv": methods, "offline.csv": offline, "observations_timing.csv": timing}, records, panels=panels, guard=guard)


def scientific_supplements(supplements):
    """Prepare paper text while leaving full adapter evidence in provenance."""
    if supplements is None:
        return dict(learning=[], validation=[], resolution=[], timing=[])
    facts = supplements["facts"]
    groups = {key: list(value) for key, value in supplements["paragraphs"].items()}
    tests = facts.get("test")
    if tests:
        # Condense only suites with no metric and no requested/admitted input.
        # Any requested-but-missing population retains its explicit coverage row.
        empty = {suite: row for suite, row in tests["suite_coverage"].items()
            if row["regions_with_metrics"] == 0 and row["requested_inputs"] == 0 and row["admitted_inputs"] == 0}
        groups["learning"] = [text for text in groups["learning"]
            if not any(text.startswith(suite+" held-out coverage:") for suite in empty)]
        if empty:
            count = sum(row["regions"] for row in empty.values())
            groups["learning"].append(
                f"The other {len(empty)} resolution suites contain {count:,} regional populations without held-out model metrics; "
                "no TEST inputs were requested or admitted there. They are not zero-error observations.")
    training = facts.get("training")
    groups["resolution"] = []
    if training:
        other = [row for row in training["coverage"] if row["suite_id"] != "s03"]
        learned = [row for row in other if row["family"] != "FFTT"]
        fftt = [row for row in other if row["family"] == "FFTT"]
        if learned and all(row["available"] == 0 for row in learned):
            populations = {}
            for row in learned:
                old = populations.setdefault(row["suite_id"], row["expected"])
                if old != row["expected"]:
                    raise ValueError("Inconsistent cross-family resolution population")
            regions = sum(populations.values())
            groups["resolution"].append(
                f"Learned models were unavailable for the {regions:,} regional populations across the other {len(populations)} resolutions. "
                "Consequently, their learned-family accuracy is unavailable rather than zero. "
                "The additional-training condition requires every eligible s03 region to qualify in both ICNN and XGBoost.")
        elif learned:
            groups["resolution"].append("Other-resolution learned-model availability was "+"; ".join(
                f"{row['suite_id']}/{row['family']}: {row['available']:,}/{row['expected']:,}" for row in learned)+".")
        if fftt:
            groups["resolution"].append(
                f"Analytic FFTT was available for {sum(row['available'] for row in fftt):,}/{sum(row['expected'] for row in fftt):,} "
                "regional populations in those resolutions. FFTT assignments do not require the learned-training milestone; "
                "Table 4 records their actual outcomes separately.")
    groups["validation"] = []
    audits = facts.get("audits", {})
    properties = audits.get("properties")
    if properties:
        if properties.get("documentary_recovery"):
            groups["validation"].append(
                f"Only documentary closure was completed for the all-selected ICNN numerical audit: "
                f"{properties['completed']:,}/{properties['expected']:,} numerical model checks completed. "
                "An operational serialization failure preceded expiry of the original 600-second allowance; "
                "no numerical work was executed during documentary recovery and no replacement clock was opened. "
                f"The {properties['uncompleted']:,} uncompleted checks do not establish a model defect or a numerical guarantee.")
        groups["validation"].append(
            f"The all-selected ICNN implementation audit recorded {str(properties['status']).lower().replace('_', ' ')} "
            f"for {properties['expected']:,} expected models, with "+"; ".join(
                f"{str(key).lower().replace('_', ' ')}: {value:,}" for key, value in sorted(properties['dispositions'].items()))+". "
            "The planned finite boundary/stress, derivative and Hessian-vector checks are distinct from the global mathematical guarantees "
            "for the constrained intended model. They do not prove global floating-point behavior, fitting accuracy or generalization; "
            "inconclusive numerical checks remain inconclusive.")
    else:
        groups["validation"].append("Final all-selected ICNN numerical-check outcomes are unavailable. "
            "The mathematical proof and synthetic tests do not establish completed numerical validation of every selected model.")
    limits = audits.get("training_limits")
    if limits:
        maximum = "unavailable" if limits["maximum_bound_percent"] is None else f"{limits['maximum_bound_percent']:.8g}%"
        groups["validation"].append(
            f"The current-TRAIN monotonicity screen recorded {str(limits['status']).lower().replace('_', ' ')} and covered "
            f"{limits['completed']:,}/{limits['expected']:,} regions: {limits['reused']:,} exact-current reused screens and "
            f"{limits['refreshed']:,} refreshed screens. Positive fixed-label lower bounds occurred in {limits['positive_bound_regions']:,} "
            f"regions, with {limits['above_1_percent_regions']:,} above 1% and a maximum of {maximum}. "
            "The finite screened pairs do not establish absence of further conflicts, attainable accuracy, exact physical truth "
            "or coordinatewise marginal-cost error bounds. The screen does not establish global label-uncertainty or zero distributions.")
    else:
        groups["validation"].append("Final current-TRAIN conflict and uncertainty results are unavailable; "
            "neither screen completeness nor attainable error is inferred.")
    return groups


def manuscript_sections(d, *, clock_disclosure=None, supplements=None):
    from .publication_results import scientific_results
    methods, learning = d["methods"], d["learning"]
    measured = methods.loc[nums(methods.RG_SPTT).notna()]
    statuses = "; ".join(f"{key}: {value}" for key, value in methods.status.value_counts(dropna=False).items()) or "no case rows supplied"
    opened = truth(learning.test_opened) if "test_opened" in learning else pd.Series(False, index=learning.index)
    finite_test = np.isfinite(nums(learning.TEST_WMAPE_percent)) & opened
    selected, initialized, classified = checkpoint_counts(learning)
    supplementary = scientific_supplements(supplements)
    result_methods = methods.copy()
    for column in ("certificate_fresh", "served_OD_membership_equal"):
        if column not in result_methods:
            result_methods[column] = False
    results = scientific_results(result_methods, d["I95"])
    test_paragraphs = supplementary["learning"]
    if supplements is None or "test" not in supplements["facts"]:
        test_paragraphs = test_paragraphs + [pooled_test_narrative(learning)]
    clock_text = ("The original 52-hour delivery window was not met. Completion continued under a separately approved "
        "24-hour extension. The original 24-hour regional data-and-training window was not restarted or extended."
        if clock_disclosure else "The original 52-hour campaign clock includes all stages and is never restarted.")
    sections = [
        ("Common benchmark, populations and invariants", [
            "The benchmark is a partial-physical P-ACNHN: the frozen s03 consolidated backbone and its fitted coefficients and internal charges, combined with detailed residual regional arcs. Every resolution retains the same 87,582 ordered backbone coordinates. A common OD map preserves distinct regional centroid physical identities and nontransit logical origin and destination roles. This comparison isolates regional approximation conditional on that backbone; it does not establish that earlier error was caused solely by consolidation. Table 1 reports network and population counts.",
            "Directed witness projection and lifting establish feasibility on the represented support, not equality with unrestricted physical route choice. Costs and capacity preparation are inherited once from the frozen source. Native flow is annual thousand tons and cost is seconds. These are modeled quantities, not independently observed truck counts. Maps use native longitude and latitude; the contiguous-US crop affects display only.",
            "For each native population, flow WMAPE is 100 times the sum of absolute prediction-minus-reference errors divided by the sum of reference flows. Full-backbone statistics retain zero-reference links. Signed bounded link error divides prediction minus reference by their maximum, with zero divided by zero defined as zero. One geographically dispersed high-flow set of 200 links was frozen from open P before inspecting candidate errors and reused across methods, resolutions and scenarios. It is not a random representative sample. Figure 3 distinguishes these populations."]),
        ("Direct-cost learning, selection and held-out evidence", [
            "ICNN learns a scalar potential whose native-flow gradient predicts the cost vector. Nonnegative input, recurrent and output weights, softplus hidden activations, a linear scalar head, tangent anchoring and the nonnegative fifth-power potential skip give the intended model a cost floor, convexity and nonnegative cross-partials on nonnegative demand. These strong monotonicity restrictions can exclude physical rerouting responses. Power-4 and XGBoost instead fit the cost vector directly from the complete ordered flow vector. Power-4 has an analytic integral used only as a solver companion. XGBoost fits centered log costs and applies the exponential inverse; it has no learned potential.",
            "The qualification target is regional, family-specific TRAIN cost WMAPE below 1% over every unique current TRAIN coordinate. It does not imply blind generalization or 1% backbone-flow error. TRAIN, requested VALID and blind TEST roles were separated before model selection. Qualified checkpoints were selected by the current admitted VALID error; if none qualified, the best valid checkpoint was retained with its target shortfall. Requested VALID roles were fixed, while historical admitted label populations and dataset identities could change during repair and label admission. All learned families used the same admitted TRAIN inputs.",
            "Detailed-regional labels retained their actual solver and certificate provenance. The final policy used FW for the first attempt and safeguarded CFW for retries, with the best fully certified retry state selected by finite relative gap. Whole flow, cost and certificate states were retained together; a smaller equilibrium gap is not a marginal-cost accuracy guarantee. Historical label-origin tags do not replace these solver receipts.",
            f"The export contains {len(learning):,} regional-family records, including {int(finite_test.sum()):,} explicitly opened finite TEST errors. The other records are not treated as TEST observations. Selected checkpoint identities are known for {selected:,} records; identities, optimizer updates and boosting rounds are distinct. Explicit initialization classification covers {classified:,} records, of which {initialized if initialized is not None else 'an unavailable number of'} are initializations. Table 3 retains these counters. Figure 4 plots each candidate's earliest recorded NEW_CHECKPOINT evaluation by chronology ordinal within region/family; the producer can reuse that tag when TRAIN matches but VALID changes. Every original observation and its plotting disposition are retained. Historical re-evaluations are not additional optimizer blocks or a fixed-data ablation.",
            neural_narrative(learning)] + test_paragraphs + supplementary["validation"]),
        ("Independent assignments and native accuracy", [
            "Table 2 compares independent P-CFW, P-FW and P-MSA controls with the s03 FFTT, power-4, XGBoost and ICNN families for the open scenario. Reference-self rows are separate from independent controls. Pure-family assignments retain unavailable or invalid-model dispositions; another family is never substituted silently. Available models above the TRAIN target retain their measured shortfalls.",
            "Potential-based FW and CFW use fresh state-consistent normalized gaps. Cost-only XGBoost uses empirical MSA with a fresh variational-inequality residual, without a potential or convergence theorem. The P references used a nominal tolerance of 1e-4 and three-hour caps; primary comparisons used 1e-3 and 45-minute complete-query caps. Reference quality and independent-control variation limit interpretation. Figure 7 separates these residual semantics.",
            f"The method ledger contains {len(methods):,} rows, of which {len(measured):,} have a numerical final residual. Recorded dispositions are {statuses}. Numerical values alone do not establish convergence or certificate freshness. A time-limited row describes its achieved state, not the eventual equilibrium, and missing values are never imputed as zero."] + results["assignments"]),
        ("Fixed-backbone resolution coverage", [
            "The residual resolutions are s02=0.3107232505953859, s03=1, s04=1.6, s05=4 and s06=12. The excluded s01 suite contributes no experiment. Backbone costs and the OD map are fixed across all five suites. Table 4 and Figure 5 distinguish constructed networks, available models and measured assignments; unavailable accuracy is not interpolated.",
            "Training prioritized s03, and additional learned training required complete s03 qualification in both ICNN and XGBoost. Analytic FFTT assignments were eligible independently of that milestone. Network construction and isolated successful regions therefore do not establish full learned-model coverage."] + supplementary["resolution"]),
        ("Represented I-95 scenario and redistribution", [
            "The closed scenario masks the same represented consolidated-backbone IDs in P and each ACNHN. Hidden physical interiors are omitted, so this is not a full physical I-95 closure. Open and closed witnesses determine the frozen active support. Within-scenario AC-versus-P served-OD membership is checked separately from open-versus-closed membership changes; equal row counts alone establish neither.",
            i95_coverage_note(d["I95"])+" Newly unserved demand is measured on the original OD rows without renormalizing remaining demand. Thus traversal changes can combine rerouting with lost served demand.",
            "Closed errors use closed P and the same 200 link IDs frozen from open P. Redistribution is closed-minus-open native backbone flow. Gain and loss sum positive and negative link-traversal changes; their net or absolute sum is not lost OD demand. Table 5 preserves closed-reference accuracy, and Figure 6 shows endpoint chords in longitude and latitude."] + results["i95"]),
        ("Timing, censoring and limits of interpretation", [
            "Table 6 preserves timing boundaries, accounting roles, caps and dispositions. Query times use the declared complete-query boundary for loading, initialization, solution and final-certificate work. Queue-level validation and registry admission precede per-case clocks and are included in the Stage3 elapsed wall time; receipt writes or other overhead outside each query boundary remain separate. Offline topology, path-tracing references, regional labels, fitting, evaluation and publication are distinct from online queries. Nested worker sums are not added to elapsed campaign wall time.",
            "Figure 8 shows measured times with time-limited completion censored. Only nonoverlapping measured-wall intervals may be totaled. Different residual targets, reference duties and missing work prevent an unqualified converged-speedup comparison. No amortization threshold or end-to-end speedup is inferred from unmatched scopes. "+clock_text] + supplementary["timing"] + results["timing"] + [
            "Conclusions are conditional on the fixed backbone, witness-supported regional routes, admitted label accuracy, achieved model coverage and measured scenarios. TRAIN qualification is not a deployment guarantee, and modeled annual flows do not supply independent truck-count validation."]),
    ]
    return sections


def tex_escape(value):
    replacements = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(replacements.get(ch, ch) for ch in str(value))


def write_manuscript(root, data, *, rendered=False, clock_disclosure=None, supplements=None):
    sections = manuscript_sections(data, clock_disclosure=clock_disclosure, supplements=supplements)
    markdown = ["# 5. Numerical experiments", "", "Regional approximation on a common partial-physical benchmark.", ""]
    tex = [r"\documentclass[10pt,letterpaper]{article}", r"\usepackage[margin=0.7in]{geometry}",
        r"\usepackage[T1]{fontenc}", r"\usepackage{lmodern,amsmath,amssymb,graphicx,longtable,booktabs,array,pdflscape,xcolor,hyperref}",
        r"\hypersetup{hidelinks}", r"\setlength{\parskip}{5pt}", r"\setlength{\parindent}{0pt}",
        r"\begin{document}", r"\title{P-ACNHN: numerical experiments}", r"\author{}\date{}\maketitle",
        r"\setcounter{section}{4}\section{Numerical experiments}"]
    for index, (title, paragraphs) in enumerate(sections, 1):
        markdown.extend([f"## 5.{index} {title}", ""])
        tex.append(r"\subsection{"+tex_escape(title)+"}")
        for paragraph in paragraphs:
            markdown.extend([paragraph, ""]); tex.extend([tex_escape(paragraph), ""])
    # Landscape table captions identify the section without an orphan divider.
    tex.append(r"\clearpage")
    for number in range(1, 7):
        asset = "table"+str(number); path = root/"numerical_data"/asset/"table.tex"
        if rendered and path.exists():
            tex.append(r"\input{numerical_data/"+asset+"/table.tex}")
        else:
            tex.append(r"\paragraph{"+tex_escape(TITLES[asset])+r"}Table rendering pending; no numerical panel is fabricated.")
    # Every panel starts a float page; a separate heading would occupy a blank page.
    tex.append(r"\clearpage")
    for number in range(3, 9):
        asset = "figure"+str(number); folder = root/"numerical_data"/asset
        status = pd.read_csv(folder/"panel_status.csv", keep_default_na=False) if rendered and (folder/"panel_status.csv").exists() else pd.DataFrame()
        rendered_rows = status.loc[status.status.eq("RENDERED")] if len(status) else pd.DataFrame()
        if not len(rendered_rows):
            disposition = "No finite observations are available for rendering." if rendered else "Rendering pending."
            tex.append(r"\paragraph{Figure "+str(number)+". "+tex_escape(TITLES[asset])+"}"+tex_escape(disposition))
            continue
        for j, row in enumerate(rendered_rows.itertuples(index=False)):
            tex.extend([r"\clearpage", r"\begin{figure}[p]\centering",
                r"\includegraphics[width=\textwidth,height=0.78\textheight,keepaspectratio]{numerical_data/"+asset+"/panel_"+row.panel_id+".pdf}"])
            if j == 0:
                tex.extend([r"\setcounter{figure}{"+str(number-1)+"}", r"\caption{"+tex_escape(TITLES[asset]+". "+CAPTIONS[asset])+r"}\label{fig:"+asset+"}"])
            else:
                tex.append(r"\par\small Figure "+str(number)+" (continued): "+tex_escape(row.title)+r"\par")
            tex.append(r"\end{figure}")
        missing = status.loc[~status.status.eq("RENDERED")]
        if len(missing):
            tex.append(r"\clearpage\noindent "+tex_escape("Figure "+str(number)+" omitted panels: "+"; ".join(missing.title+" — "+missing.status))+r"\par")
    tex.append(r"\end{document}")
    (root/"numerical.md").write_text("\n".join(markdown)+"\n", encoding="utf-8")
    (root/"numerical.tex").write_text("\n".join(tex)+"\n", encoding="utf-8")


def write_index(root):
    rows = []
    for asset in ["figure"+str(i) for i in range(3, 9)]+["table"+str(i) for i in range(1, 7)]:
        folder = root/"numerical_data"/asset
        result = read_json(folder/"RENDER_RESULT.json") if (folder/"RENDER_RESULT.json").exists() else {"status": "EXTRACTED_RENDER_PENDING", "visual_qa": "PENDING"}
        rows.append(dict(asset_id=asset, title=TITLES[asset], folder="numerical_data/"+asset,
            status=result["status"], visual_qa=result.get("visual_qa", "PENDING"), csv_files=len(list(folder.glob("*.csv")))))
    public_csv(root/"index.csv", pd.DataFrame(rows))
    parts = ["<!doctype html><html lang='en'><meta charset='utf-8'><title>P-ACNHN numerical assets</title>",
        "<style>body{font:16px system-ui;max-width:1100px;margin:3rem auto;padding:1rem}td,th{padding:.7rem;border-bottom:1px solid #ddd;text-align:left}a{color:#075985}</style>",
        "<h1>P-ACNHN numerical assets</h1><p>Figures and tables use nine canonical local CSV datasets; manuscript metadata supplements are separately authenticated and indexed in manuscript_evidence/PROVENANCE.json. Recorded render status is separate from required visual QA. Native flow units are annual thousand tons.</p><table><tr><th>Asset</th><th>Title</th><th>Disposition</th><th>Visual QA</th></tr>"]
    for row in rows:
        parts.append("<tr><td><a href='"+row["folder"]+"/README.md'>"+row["asset_id"]+"</a></td><td>"+html.escape(row["title"])+"</td><td>"+row["status"]+"</td><td>"+row["visual_qa"]+"</td></tr>")
    parts.append("</table></html>"); (root/"index.html").write_text("\n".join(parts), encoding="utf-8")


def build(config):
    cfg, ctx, root = read_contract(config)
    if "extract" not in cfg["tasks"]:
        raise ValueError("Extraction is not an explicit publication task")
    if root.exists():
        raise FileExistsError("Preserve existing publication bundle; use a new versioned output_root")
    extraction_code = {p.name: sha(p) for p in publication_code_paths()}
    data, records = load_inputs(cfg, ctx)
    supplements = load_manuscript_evidence(cfg, ctx)
    from .publication_evidence import snapshot_evidence, verify_snapshot
    root.mkdir(parents=True, exist_ok=False)
    supplement_binding = snapshot_evidence(root, supplements, guard=lambda: ctx.guard(cfg["deadline_epoch"]))
    write_json(root/"SOURCE_INPUTS.json", records)
    write_json(root/"MANUSCRIPT_PROVENANCE.json", dict(canonical_csv_inputs=records, metadata=supplement_binding, code_sha256=extraction_code,
        clock_disclosure=(ctx.publication_continuation_window or {}).get("disclosure")))
    build_tables(root, data, records, guard=lambda: ctx.guard(cfg["deadline_epoch"])); ctx.guard(cfg["deadline_epoch"])
    build_figures(root, data, records, guard=lambda: ctx.guard(cfg["deadline_epoch"])); ctx.guard(cfg["deadline_epoch"])
    write_manuscript(root, data, clock_disclosure=(ctx.publication_continuation_window or {}).get("disclosure"), supplements=supplements); write_index(root)
    verify_snapshot(root, supplement_binding, evidence=load_manuscript_evidence(cfg, ctx), guard=lambda: ctx.guard(cfg["deadline_epoch"]))
    for binding in ctx.bindings.values():
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Source changed during publication extraction")
    if extraction_code != {p.name: sha(p) for p in publication_code_paths()}:
        raise ValueError("Publication source changed during extraction")
    manifest = {"status": "EXTRACTED_RENDER_PENDING", "phase_id": cfg["phase_id"], "run_id": cfg["run_id"],
        "original_start_epoch": ctx.c["original_start_epoch"], "created_epoch": time.time(),
        "assets": list(TITLES), "source_inputs": records, "config_sha256": sha(config),
        "manuscript_evidence": supplement_binding, "code_sha256": extraction_code,
        "files": [{"relative_path": p.relative_to(root).as_posix(), "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(root.rglob("*")) if p.is_file()]}
    write_json(root/"EXTRACTION_RESULT.json", manifest)
    write_json(ctx.root/"manifests"/(root.name+"_publication_inputs.json"), list(ctx.bindings.values()))
    return manifest


def static_renderer_check(path):
    """Reject project imports and alternative scientific-input channels."""
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    allowed = {"pathlib", "csv", "hashlib", "io", "json", "math", "textwrap", "numpy", "pandas", "matplotlib"}
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("Standalone renderer cannot import relative project modules")
            imports.append((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec", "__import__", "compile"}:
            raise ValueError("Dynamic renderer code is not allowed")
    if set(imports)-allowed:
        raise ValueError("Disallowed standalone imports: "+str(sorted(set(imports)-allowed)))
    forbidden = ["read_parquet", "read_pickle", "read_sql", "read_excel", "np.load(", "numpy.load(", "http://", "https://", "socket", "subprocess", "os.system", "C:/", "C:\\\\"]
    found = [s for s in forbidden if s in source]
    if found:
        raise ValueError("Non-CSV scientific input or external access in renderer: "+str(found))
    return {"status": "PASS", "imports": sorted(set(imports)), "sha256": sha(path)}


def run_process(command, cwd, environment, ctx, deadline, log_stem, cap_seconds):
    """Poll the child under the original deadline and process-tree RAM guard."""
    log_stem = Path(log_stem); log_stem.parent.mkdir(parents=True, exist_ok=True)
    started = time.time(); limit = min(deadline, started+cap_seconds)
    with log_stem.with_suffix(".stdout.log").open("wb") as stdout, log_stem.with_suffix(".stderr.log").open("wb") as stderr:
        process = subprocess.Popen(command, cwd=cwd, env=environment, stdout=stdout, stderr=stderr)
        try:
            while process.poll() is None:
                ctx.guard(limit)
                time.sleep(.25)
            if process.returncode:
                raise RuntimeError(f"Publication child exit {process.returncode}; inspect {log_stem}")
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=10)
    return {"elapsed_seconds": time.time()-started, "returncode": process.returncode,
            "command": [str(x) for x in command]}


def render_one(folder, cfg, ctx, evidence, label):
    folder = Path(folder); evidence = Path(evidence); evidence.mkdir(parents=True, exist_ok=False)
    script = folder/("build_table.py" if folder.name.startswith("table") else "plot.py")
    check = static_renderer_check(script)
    inputs = {p.name: sha(p) for p in folder.glob("*.csv")}
    before = set(folder.iterdir())
    for path in [folder/".mplconfig", folder/".tmp"]:
        path.mkdir(exist_ok=False)
    env = os.environ.copy(); env.update(MPLCONFIGDIR=str(folder/".mplconfig"), TEMP=str(folder/".tmp"), TMP=str(folder/".tmp"), PYTHONDONTWRITEBYTECODE="1")
    # The unrelated CWD catches accidental relative-path lookup. The runtime
    # audit rejects access to any external scientific file, including this repo.
    cwd = evidence/"unrelated_cwd"; cwd.mkdir()
    command = [cfg["python_path"], "-I", "-B", str(TEMPLATES/"audit_runner.py"), str(script), str(folder/"FILE_ACCESS_AUDIT.json")]
    run = run_process(command, cwd, env, ctx, cfg["deadline_epoch"], evidence/label,
        float(cfg["resource_budget"].get("per_asset_seconds", 180)))
    for name, digest in inputs.items():
        if sha(folder/name) != digest:
            raise ValueError("Renderer changed a source CSV: "+name)
    result = read_json(folder/"RENDER_RESULT.json")
    allowed_status = {"RENDERED", "UNAVAILABLE_NO_RENDERABLE_OBSERVATIONS"}
    if result["status"] not in allowed_status:
        raise ValueError("Unrecognized render disposition")
    outputs = result["outputs"]
    if result["status"] == "RENDERED" and not outputs:
        raise ValueError("No-op renderer cannot pass")
    for name in outputs:
        if Path(name).name != name or not (folder/name).is_file() or (folder/name).stat().st_size == 0:
            raise ValueError("Missing or unsafe declared render output: "+name)
    read_audit = pd.read_csv(folder/"read_csv_hashes.csv", dtype=str)
    if read_audit.file.duplicated().any():
        raise ValueError("Repeated file in CSV read audit")
    for row in read_audit.itertuples(index=False):
        if row.file not in inputs or row.sha256 != inputs[row.file]:
            raise ValueError("CSV read hash was not an original local input")
    expected = {"asset.csv", "table.csv", "columns.csv"} if script.name == "build_table.py" else {"asset.csv", "panels.csv"}|set(read_logical_csv(folder, "panels.csv", {}).source)
    expected |= set(csv_logical_names(folder, "observations_"))
    expected = csv_expected_files(folder, expected)
    if set(read_audit.file) != expected:
        raise ValueError("Renderer did not read every declared scientific input")
    file_audit = read_json(folder/"FILE_ACCESS_AUDIT.json")
    if file_audit["status"] != "PASS" or not all(row["allowed"] for row in file_audit["access"]):
        raise ValueError("Renderer file-access audit failed")
    return {"status": result["status"], "static": check, "run": run, "inputs": inputs,
        "csv_read_sha256": sha(folder/"read_csv_hashes.csv"),
        "outputs": {name: sha(folder/name) for name in outputs},
        "new_files": sorted(p.name for p in folder.iterdir() if p not in before and p.is_file())}


def render(config):
    cfg, ctx, root = read_contract(config)
    if "render" not in cfg["tasks"] or not cfg.get("python_path"):
        raise ValueError("Explicit render task and Python runtime are required")
    extraction = read_json(root/"EXTRACTION_RESULT.json")
    if extraction["config_sha256"] != sha(config):
        raise ValueError("Publication configuration changed after extraction")
    for item in extraction["files"]:
        if sha(root/item["relative_path"]) != item["sha256"]:
            raise ValueError("Extracted source changed before render: "+item["relative_path"])
    from .publication_evidence import verify_snapshot
    supplements = load_manuscript_evidence(cfg, ctx)
    verify_snapshot(root, extraction["manuscript_evidence"], evidence=supplements, guard=lambda: ctx.guard(cfg["deadline_epoch"]))
    if extraction["code_sha256"] != {p.name: sha(p) for p in publication_code_paths()}:
        raise ValueError("Publication source changed after extraction")
    evidence = root/"validation"/"asset_reproduction"
    if evidence.exists():
        raise FileExistsError("Preserve prior rendering evidence; rebuild in a new publication version")
    evidence.mkdir(parents=True)
    records = []
    for asset in ["figure"+str(i) for i in range(3, 9)]+["table"+str(i) for i in range(1, 7)]:
        ctx.guard(cfg["deadline_epoch"])
        folder = root/"numerical_data"/asset
        # Snapshot every supplied file before any rendering. Both independent
        # copies run without source outputs or shared Matplotlib caches.
        copies = []
        for run_id in ["A", "B"]:
            isolated = evidence/run_id/asset
            shutil.copytree(folder, isolated)
            copies.append(isolated)
        native = render_one(folder, cfg, ctx, evidence/"native"/asset, "native")
        replicas = [render_one(copy, cfg, ctx, evidence/("audit_"+run_id)/asset, run_id) for copy, run_id in zip(copies, ["A", "B"])]
        for replica in replicas:
            if replica["inputs"] != native["inputs"] or replica["csv_read_sha256"] != native["csv_read_sha256"]:
                raise ValueError("Isolated CSV input/read audit differs: "+asset)
            if replica["status"] != native["status"] or set(replica["outputs"]) != set(native["outputs"]):
                raise ValueError("Isolated output inventory/disposition differs: "+asset)
            for name, digest in native["outputs"].items():
                # PNG and TeX must be byte-identical. PDF encoding may carry
                # backend metadata; PDF content is additionally visually checked.
                if name.endswith((".png", ".tex")) and replica["outputs"][name] != digest:
                    raise ValueError("Independent render differs: "+asset+"/"+name)
        records.append({"asset_id": asset, "status": "PASS" if native["status"] == "RENDERED" else "QUALIFIED_UNAVAILABLE",
                        "native": native, "independent_A": replicas[0], "independent_B": replicas[1]})
        print("ASSET_ISOLATION_PASS "+asset, flush=True)
    data, _ = load_inputs(cfg, ctx)
    write_manuscript(root, data, rendered=True, clock_disclosure=(ctx.publication_continuation_window or {}).get("disclosure"), supplements=supplements); write_index(root)
    verify_snapshot(root, extraction["manuscript_evidence"], evidence=load_manuscript_evidence(cfg, ctx), guard=lambda: ctx.guard(cfg["deadline_epoch"]))
    if extraction["code_sha256"] != {p.name: sha(p) for p in publication_code_paths()}:
        raise ValueError("Publication source changed during rendering")
    provenance_rows = [r for r in extraction["files"] if r["relative_path"] == "MANUSCRIPT_PROVENANCE.json"]
    if len(provenance_rows) != 1 or provenance_rows[0]["sha256"] != sha(root/"MANUSCRIPT_PROVENANCE.json"):
        raise ValueError("Manuscript provenance changed during rendering")
    result = {"status": "PASS", "scope": "12 local-CSV asset folders independently reproduced twice",
        "assets": records, "config_sha256": sha(config), "manuscript_sha256": sha(root/"numerical.tex"),
        "code_sha256": {p.name: sha(p) for p in publication_code_paths()},
        "manuscript_evidence": extraction["manuscript_evidence"], "manuscript_provenance_sha256": sha(root/"MANUSCRIPT_PROVENANCE.json"),
        "visual_qa": "PENDING_ACTUAL_EVERY_PANEL_AND_TABLE_PAGE_INSPECTION"}
    write_json(root/"REPRODUCTION_RESULT.json", result)
    return result


def compile_pdf(config):
    cfg, ctx, root = read_contract(config)
    required = ["compiler_path", "compiler_cache_source", "pdftoppm_path"]
    if "compile" not in cfg["tasks"] or any(not cfg.get(k) for k in required):
        raise ValueError("Explicit compile task, portable compiler/cache and Poppler are required")
    reproduction = read_json(root/"REPRODUCTION_RESULT.json")
    if reproduction["status"] != "PASS":
        raise ValueError("Independent CSV asset reproduction must pass before PDF compilation")
    if reproduction["config_sha256"] != sha(config) or reproduction["manuscript_sha256"] != sha(root/"numerical.tex"):
        raise ValueError("Publication configuration or manuscript changed after reproduction")
    from .publication_evidence import verify_snapshot
    verify_snapshot(root, reproduction["manuscript_evidence"], evidence=load_manuscript_evidence(cfg, ctx), guard=lambda: ctx.guard(cfg["deadline_epoch"]))
    if reproduction["manuscript_provenance_sha256"] != sha(root/"MANUSCRIPT_PROVENANCE.json"):
        raise ValueError("Manuscript provenance changed after reproduction")
    for path in publication_code_paths():
        if reproduction["code_sha256"].get(path.name) != sha(path):
            raise ValueError("Publication code changed after reproduction: "+path.name)
    for asset in reproduction["assets"]:
        folder = root/"numerical_data"/asset["asset_id"]
        for name, digest in {**asset["native"]["inputs"], **asset["native"]["outputs"]}.items():
            if sha(folder/name) != digest:
                raise ValueError("Reproduced asset changed before compilation: "+asset["asset_id"]+"/"+name)
        script = folder/("build_table.py" if folder.name.startswith("table") else "plot.py")
        if sha(script) != asset["native"]["static"]["sha256"]:
            raise ValueError("Reproduced standalone script changed")
    compiler = Path(cfg["compiler_path"]).resolve(strict=True)
    cache_source = Path(cfg["compiler_cache_source"]).resolve(strict=True)
    evidence = root/"validation"/"numerical_pdf"
    if evidence.exists():
        raise FileExistsError("Preserve existing compile evidence; use a new versioned bundle")
    evidence.mkdir(parents=True)
    cache = evidence/"cache"; shutil.copytree(cache_source, cache)
    compiler_digest = sha(compiler)
    env = os.environ.copy(); env.update(TECTONIC_CACHE_DIR=str(cache), XDG_CACHE_HOME=str(cache),
        LOCALAPPDATA=str(evidence/"localappdata"), APPDATA=str(evidence/"appdata"))
    commands = []
    # Compilation is deliberately cache-only: a PDF build cannot introduce an
    # unrecorded network/toolchain dependency after scientific results freeze.
    command = [str(compiler), "-X", "compile", "--only-cached", "--keep-logs", "--outdir", str(evidence), str(root/"numerical.tex")]
    commands.append(run_process(command, root, env, ctx, cfg["deadline_epoch"], evidence/"tectonic", float(cfg["resource_budget"].get("compile_seconds", 240))))
    produced = evidence/"numerical.pdf"
    if not produced.exists() or produced.stat().st_size == 0:
        raise ValueError("Compiler produced no PDF")
    if sha(compiler) != compiler_digest:
        raise ValueError("Read-only compiler changed")
    shutil.copyfile(produced, root/"numerical.pdf")
    pages = evidence/"pages"; pages.mkdir()
    command = [cfg["pdftoppm_path"], "-r", "135", "-png", str(produced), str(pages/"page")]
    commands.append(run_process(command, root, env, ctx, cfg["deadline_epoch"], evidence/"pdftoppm", float(cfg["resource_budget"].get("pdf_render_seconds", 240))))
    images = sorted(pages.glob("page-*.png"))
    if not images:
        raise ValueError("No PDF pages rendered for mandatory visual inspection")
    # pypdf is an optional verification dependency of the pipeline only; the
    # standalone figure/table scripts remain NumPy/pandas/Matplotlib+stdlib.
    from pypdf import PdfReader
    reader = PdfReader(produced)
    if len(images) != len(reader.pages):
        raise ValueError("PDF page render count mismatch")
    fonts = []
    for page_no, page in enumerate(reader.pages, 1):
        resource = page.get("/Resources", {})
        if hasattr(resource, "get_object"):
            resource = resource.get_object()
        font_map = resource.get("/Font", {})
        if hasattr(font_map, "get_object"):
            font_map = font_map.get_object()
        for name, reference in font_map.items():
            font = reference.get_object(); descendants = font.get("/DescendantFonts", [])
            targets = [child.get_object() for child in descendants] or [font]
            embedded = True
            for target in targets:
                descriptor = target.get("/FontDescriptor")
                descriptor = descriptor.get_object() if descriptor else {}
                embedded &= any(key in descriptor for key in ["/FontFile", "/FontFile2", "/FontFile3"])
            fonts.append(dict(page=page_no, font=str(font.get("/BaseFont", name)), embedded=bool(embedded)))
    if fonts and not all(item["embedded"] for item in fonts):
        raise ValueError("PDF contains unembedded fonts")
    log = (evidence/"numerical.log").read_text(encoding="utf-8", errors="replace") if (evidence/"numerical.log").exists() else ""
    warnings = [line for line in log.splitlines() if any(term in line for term in ["Overfull", "Underfull", "Missing character", "undefined", "multiply defined"])]
    verify_snapshot(root, reproduction["manuscript_evidence"], evidence=load_manuscript_evidence(cfg, ctx), guard=lambda: ctx.guard(cfg["deadline_epoch"]))
    if reproduction["code_sha256"] != {p.name: sha(p) for p in publication_code_paths()}:
        raise ValueError("Publication source changed during compilation")
    if reproduction["manuscript_sha256"] != sha(root/"numerical.tex") or reproduction["manuscript_provenance_sha256"] != sha(root/"MANUSCRIPT_PROVENANCE.json"):
        raise ValueError("Manuscript or provenance changed during compilation")
    result = {"status": "COMPILED_VISUAL_QA_PENDING", "manuscript_evidence": reproduction["manuscript_evidence"], "manuscript_provenance_sha256": sha(root/"MANUSCRIPT_PROVENANCE.json"), "pdf_sha256": sha(produced), "bytes": produced.stat().st_size,
        "pages": len(images), "page_images": [{"file": p.relative_to(root).as_posix(), "sha256": sha(p)} for p in images],
        "compiler_sha256": compiler_digest, "cache_only": True, "fonts": fonts, "layout_or_reference_warnings": warnings,
        "commands": commands, "visual_qa": "PENDING_ACTUAL_EVERY_PAGE_INSPECTION"}
    write_json(root/"PDF_BUILD_RESULT.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--action", required=True, choices=["extract", "render", "compile"])
    args = parser.parse_args()
    result = {"extract": build, "render": render, "compile": compile_pdf}[args.action](args.config)
    print(json.dumps({k: result[k] for k in ["status", "pages", "visual_qa"] if k in result}, indent=2))


if __name__ == "__main__":
    main()
