"""Deferred T08 audit of frozen TRAIN labels; never fits, labels or opens TEST.

Production requires an admitted config and existing Stage3/Stage4/shared-audit
clocks. Only two explicitly identified stale regions are replayed. Other regions
reuse authenticated, final-identity-aligned receipts of the same non-exhaustive
screen. No model, solver or training module is imported by this module.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import zipfile

import numpy as np
import psutil

from .control import Context, array_hash, write_json

SCREEN = "DISJOINT_ADJACENT_SORTED_QUERY_PAIRS_NOT_EXHAUSTIVE"
EXPECTED_REGIONS = 1499
FRESH_KEYS = frozenset({"s03:000936", "s03:000939"})
REQUIRED_TASKS = {"AUTHENTICATE_FINAL_TRAIN", "REUSE_CURRENT_SCREENS",
                  "REFRESH_TWO_STALE_SCREENS", "REPORT_LIMITATIONS"}
SOURCE_NAMES = {"final_training_limits.py", "control.py", "continuation.py",
                "handoff.py", "regional.py", "training.py"}
LIMITATIONS = [
    "Complete regional coverage is coverage of a disjoint-adjacent screen, not exhaustive comparable-pair coverage.",
    "Zero lower bound or no ordered comparisons does not establish attainability of the 1% TRAIN target.",
    "Bounds concern recorded approximate TRAIN targets, not exact physical equilibrium costs.",
    "FW potential-gap intervals do not bound individual marginal-cost errors.",
    "Existing receipt order/decrease assertions are reused, not independently replayed for1497 regions.",
    "Observed zero targets do not by themselves prove structural zeros for all inputs.",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _reject_constant(value):
    raise ValueError("Nonfinite JSON constant: " + value)


def _object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _state(path):
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


class Reader:
    """Hash complete bytes; retain small projections, never whole histories."""
    def __init__(self, check=lambda: None):
        self.check = check
        self.bindings = {}

    def bind(self, path, expected, *, maximum=None):
        self.check()
        p = Path(path).resolve(strict=True)
        before = _state(p)
        require(re.fullmatch(r"[0-9a-f]{64}", expected or "") is not None,
                "Explicit SHA256 required: " + str(p))
        if maximum is not None:
            require(before[0] <= maximum, "Input exceeds admitted byte bound: " + str(p))
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                h.update(block)
                self.check()
        require(h.hexdigest() == expected and _state(p) == before,
                "Source bytes changed: " + str(p))
        prior = self.bindings.get(str(p))
        require(prior is None or prior["sha256"] == expected, "Conflicting source identity")
        self.bindings[str(p)] = dict(path=str(p), sha256=expected, bytes=before[0],
                                    mtime_ns=before[1], ctime_ns=before[2])
        return p

    def json(self, binding, *, maximum=32*1024**2):
        p = self.bind(binding["path"], binding["sha256"], maximum=maximum)
        before = _state(p)
        with p.open(encoding="utf-8-sig") as stream:
            value = json.load(stream, parse_constant=_reject_constant, object_pairs_hook=_object)
        self.check()
        require(_state(p) == before, "JSON changed during read")
        return value

    def unchanged(self):
        self.check()
        for row in self.bindings.values():
            require(_state(row["path"]) == (row["bytes"], row["mtime_ns"], row["ctime_ns"]),
                    "Admitted input changed during audit: " + row["path"])


def _number(value, name, *, nonnegative=True):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and (not nonnegative or value >= 0), "Invalid " + name)
    return float(value)


def _integer(value, name, *, minimum=0):
    require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, "Invalid " + name)
    return value


def _close(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)


def validate_screen(receipt, identity, rows):
    require(receipt["train_dataset_identity"] == identity and receipt["unique_rows"] == rows,
            "Screen differs from final TRAIN identity/count")
    require(receipt["screen"] == SCREEN, "Different conflict screen")
    comparisons = _integer(receipt["ordered_comparisons"], "ordered comparison count")
    violations = receipt["violations"]
    require(isinstance(violations, list) and comparisons <= rows//2
            and len(violations) == receipt["violating_disjoint_pairs"] <= comparisons,
            "Invalid disjoint comparison coverage")
    used = set()
    total = []
    for row in violations:
        i = _integer(row["lower_row"], "lower row")
        j = _integer(row["upper_row"], "upper row")
        require(i != j and i < rows and j < rows and i not in used and j not in used,
                "Violating query pairs are not disjoint")
        used.update((i, j))
        keys, decreases = row["pair_keys"], row["native_cost_decrease"]
        require(len(keys) == len(decreases) > 0 and len(set(keys)) == len(keys), "Invalid violating coordinates")
        for decrease in decreases:
            require(_number(decrease, "cost decrease") > 0, "Nonpositive violation")
        for field in ("lower_input_id", "upper_input_id"):
            require(re.fullmatch(r"[0-9a-f]{64}", row[field]) is not None, "Invalid violation input identity")
        total.extend(decreases)
    lower = _number(receipt["fixed_label_L1_error_lower_bound"], "lower bound")
    denom = _number(receipt["denominator_native_cost_sum"], "cost denominator")
    require(_close(lower, math.fsum(total)), "Lower-bound arithmetic mismatch")
    wmape = receipt["fixed_label_WMAPE_percent_lower_bound"]
    if denom == 0:
        require(lower == 0 and wmape is None, "Undefined zero-denominator bound must be null")
    else:
        require(_close(_number(wmape, "WMAPE bound"), 100*lower/denom), "WMAPE bound arithmetic mismatch")
    return receipt


def monotonic_screen(train, scale, pair_keys):
    """Exact production disjoint-adjacent algorithm, without training imports."""
    u, tau = train["u"], train["tau"]
    require(u.ndim == 2 and u.shape == tau.shape and len(u) > 0, "Empty/misaligned TRAIN")
    require(scale.shape == (u.shape[1],) and np.isfinite(scale).all() and np.all(scale > 0), "Invalid native scale")
    order = np.argsort(np.sum(u/scale, axis=1), kind="stable")
    violations = []; lower = 0.; comparisons = 0
    for at in range(0, len(order)-1, 2):
        i, j = int(order[at]), int(order[at+1])
        if not np.all(u[i] <= u[j]):
            continue
        comparisons += 1
        differences = tau[i]-tau[j]
        bad = np.flatnonzero(differences > 0.)
        if len(bad):
            lower += float(differences[bad].sum())
            violations.append(dict(lower_row=i, upper_row=j, pair_keys=[str(pair_keys[k]) for k in bad],
                native_cost_decrease=differences[bad].tolist(), lower_input_id=train["records"][i]["input_identity"],
                upper_input_id=train["records"][j]["input_identity"]))
    denom = float(tau.sum())
    return dict(train_dataset_identity=train["dataset_identity"], screen=SCREEN, unique_rows=len(u),
        ordered_comparisons=comparisons, violating_disjoint_pairs=len(violations),
        fixed_label_L1_error_lower_bound=lower,
        fixed_label_WMAPE_percent_lower_bound=100*lower/denom if denom > 0 else None,
        denominator_native_cost_sum=denom, violations=violations,
        scope="Lower bound for fitting these recorded targets; no coordinatewise true-equilibrium label-error certificate")


def _inside(path, root):
    p = Path(path).resolve(strict=True)
    require(p.is_relative_to(Path(root).resolve(strict=True)), "Input outside admitted regional directory")
    return p


def _npz(reader, path, expected, fields, *, maximum=64*1024**2):
    p = reader.bind(path, expected, maximum=maximum)
    # Check decompressed size before NumPy allocates; never deserialize pickle.
    with zipfile.ZipFile(p) as archive:
        members = archive.infolist()
        require(len({m.filename for m in members}) == len(members), "Duplicate NPZ members")
        require(sum(m.file_size for m in members) <= maximum, "NPZ expansion exceeds memory budget")
    with np.load(p, allow_pickle=False) as data:
        result = {key: data[key].copy() for key in fields}
    reader.check()
    return result


def load_graph_identity(reader, record, expected_identity):
    fields = ("tail", "head", "node_ids", "edge_ids", "base", "k4", "pair_source", "pair_target",
              "pair_keys", "baseline_open", "baseline_i95", "c0")
    a = _npz(reader, record["path"], record["sha256"], fields)
    identity = array_hash({k: a[k] for k in fields if k != "c0"})
    support = array_hash({k: a[k] for k in ("pair_source", "pair_target", "pair_keys")})
    require(identity == expected_identity["graph_identity"] and support == expected_identity["support_identity"]
            and record["sha256"] == expected_identity["graph_sha256"], "Frozen native graph/support identity mismatch")
    m = record["pairs"]
    require(a["pair_keys"].dtype == np.dtype("uint64") and len(np.unique(a["pair_keys"])) == m,
            "Invalid frozen pair identities")
    for key in ("pair_source", "pair_target", "pair_keys", "baseline_open", "baseline_i95", "c0"):
        require(a[key].shape == (m,), "Frozen coordinate order/shape mismatch")
    for key in ("baseline_open", "baseline_i95", "c0"):
        require(np.isfinite(a[key]).all() and np.all(a[key] >= 0), "Invalid frozen native values")
    return a


def native_scale(graph, record, saved, fraction):
    require(fraction == .01 and saved["fraction"] == fraction, "Frozen native scale policy differs")
    baseline = np.maximum(graph["baseline_open"], graph["baseline_i95"])
    positive = baseline[baseline > 0]
    if len(positive):
        reference = float(np.median(positive))
    else:
        reference = _number(record.get("physical_capacity_reference"), "capacity fallback")
        require(reference > 0 and record.get("physical_capacity_reference_provenance"), "Unproven scale fallback")
    scale = np.maximum(baseline, fraction*reference)
    require(saved["graph_sha256"] == record["sha256"] and np.array_equal(np.asarray(saved["scale"]), scale),
            "Saved native scale differs from exact original construction")
    return scale


def label_metadata(record, input_id, identity, selected, pair_count):
    """Validate selected certificate/admission; drop history before retention."""
    require(record.get("admitted") is True and record.get("role") == "TRAIN"
            and record.get("input_identity") == input_id, "Selected label role/admission/input mismatch")
    for field in ("graph_identity", "graph_sha256", "support_identity"):
        require(record[field] == identity[field], "Selected label graph/support mismatch")
    require(record["pair_count"] == pair_count and record["label_sha256"] == selected["sha256"]
            and record["fresh_certificate"] is True, "Selected label identity/freshness mismatch")
    cert = record["certificate"]
    require(cert.get("fresh") is True and cert.get("interpretation") == "CONVEX_POTENTIAL_FW_GAP",
            "Selected certificate is not a fresh potential certificate")
    rg = cert["RG_SPTT"]
    gap = _number(cert["raw_signed_gap"], "signed gap", nonnegative=False)
    allowance = _number(cert["arithmetic_allowance"], "gap allowance")
    tstt = _number(cert["TSTT"], "TSTT")
    sptt = _number(cert["SPTT"], "SPTT")
    require(_close(gap, tstt-sptt) and _close(allowance, 1e-10*max(1., abs(tstt), abs(sptt))),
            "Certificate gap/allowance arithmetic mismatch")
    if rg is not None:
        rg = _number(rg, "relative gap", nonnegative=False)
        require(sptt > 0 and _close(rg, gap/sptt), "Relative-gap arithmetic mismatch")
    else:
        require(sptt == 0, "Missing relative gap on positive SPTT")
    quality = "A" if ((rg is not None and rg <= 1e-6) or (rg is None and abs(gap) <= allowance)) else (
        "B" if rg is not None and rg <= 1e-5 else "C" if rg is not None and rg <= 1e-4 else "QUARANTINE")
    require(quality in {"A", "B"} and quality == record["quality"], "Frozen label fails original A/B admission")
    require(gap >= -allowance, "Certificate has a materially negative gap")
    bound = _number(record["potential_error_bound"], "potential error bound")
    lower = _number(record["potential_lower"], "potential lower")
    upper = _number(record["potential_upper"], "potential upper")
    require(_close(bound, max(0., gap)+allowance) and lower <= upper+allowance
            and _close(upper, cert["value"]), "Invalid selected potential interval")
    require(record["marginal_cost_error_bound_available"] is False, "Unexpected coordinatewise error interpretation")
    require(re.fullmatch(r"[0-9a-f]{64}", cert["state_hash"]) is not None, "Missing state identity")
    if "selected_state_hash" in record:
        require(record["selected_state_hash"] == cert["state_hash"], "Selected/last certificate confused")
    change = record.get("last_cost_change")
    if change is not None:
        _number(change, "selected cost change")
    return dict(input_identity=input_id, quality=quality, RG_SPTT=rg,
        potential_error_bound=bound, potential_lower=lower, potential_upper=upper,
        last_cost_change=change, fresh_certificate=True, state_hash=cert["state_hash"],
        marginal_cost_error_bound_available=False, selected_state_iteration=record.get("selected_state_iteration",
        record.get("reported_state_iteration")), iterate_selection=record.get("iterate_selection", "LEGACY_LAST_CERTIFIED"))


def load_train(reader, label_root, bindings, graph, expected, *, run_id):
    """Read only admitted TRAIN, one full receipt at a time, discard its history."""
    root = Path(label_root).resolve(strict=True)
    identity = reader.json(bindings["region_identity"], maximum=1024**2)
    require(Path(bindings["region_identity"]["path"]).resolve() == root/"REGION_IDENTITY.json"
            and identity["phase_id"] == "PACNHN_V1" and identity["run_id"] == run_id
            and identity["region_id"] == expected["key"], "Regional identity/path mismatch")
    index = reader.json(bindings["request_index"], maximum=8*1024**2)
    require(Path(bindings["request_index"]["path"]).resolve() == root/"REQUEST_INDEX.json", "Index path mismatch")
    require(all(row.get("role") in {"TRAIN", "VALID", "TEST"} for row in index.values()), "Unknown sample role")
    u = []; tau = []; records = []; m = expected["pairs"]; saved_c0 = None
    for input_id, row in index.items():
        if row["role"] != "TRAIN" or row.get("selected_label") is None:
            continue
        reader.check()
        selected = row["selected_label"]
        for key in ("path", "receipt"):
            _inside(selected[key], root)
        full = reader.json(dict(path=selected["receipt"], sha256=selected["receipt_sha256"]))
        small = label_metadata(full, input_id, identity, selected, m)
        del full  # Never retain 291MB of complete histories in records.
        a = _npz(reader, selected["path"], selected["sha256"], ("u", "tau", "c0", "pair_keys", "flow"))
        for key in ("u", "tau", "c0"):
            require(a[key].dtype == np.dtype("float64") and a[key].shape == (m,)
                    and np.isfinite(a[key]).all() and np.all(a[key] >= 0), "Invalid native selected label array")
        require(array_hash({"u": a["u"]}) == input_id and np.allclose(a["c0"], graph["c0"], rtol=1e-12, atol=1e-10)
                and np.array_equal(a["pair_keys"], graph["pair_keys"]), "Label input/c0/pair ordering mismatch")
        # Match producer graph tolerance, but never replace the exact saved c0
        # payload when reconstructing the frozen TRAIN dataset identity.
        require(saved_c0 is None or np.array_equal(saved_c0, a["c0"]), "Saved c0 changed within TRAIN population")
        if saved_c0 is None:
            saved_c0 = a["c0"].copy()
        expected_lower = max(0., small["potential_upper"]-small["potential_error_bound"], float(a["c0"]@a["u"]))
        require(_close(small["potential_lower"], expected_lower), "Potential interval does not bind this input/c0")
        require(array_hash({"flow": a["flow"]}) == small["state_hash"], "Certificate does not bind selected saved flow")
        require(a["flow"].dtype == np.dtype("float64") and a["flow"].shape == graph["tail"].shape
                and np.isfinite(a["flow"]).all() and np.all(a["flow"] >= 0), "Invalid certified saved flow")
        u.append(a["u"]); tau.append(a["tau"]); records.append(small)
        del a
    require(len(u) == expected["rows"] > 0, "Current admitted TRAIN row count changed")
    data = dict(u=np.stack(u), tau=np.stack(tau), c0=saved_c0, records=records)
    data["dataset_identity"] = array_hash({k: data[k] for k in ("u", "tau", "c0")})
    require(data["dataset_identity"] == expected["identity"], "TRAIN differs from final current dataset identity")
    reader.bind(bindings["request_index"]["path"], bindings["request_index"]["sha256"])
    return data


def summarize_fresh(data):
    records = data["records"]
    gaps = [row["RG_SPTT"] for row in records if row["RG_SPTT"] is not None]
    changes = [row["last_cost_change"] for row in records if row["last_cost_change"] is not None]
    return dict(TRAIN_inputs=len(records), quality_counts={q: sum(r["quality"] == q for r in records) for q in ("A", "B")},
        RG_SPTT_min=min(gaps) if gaps else None, RG_SPTT_max=max(gaps) if gaps else None,
        exact_zero_denominator_certificate_count=len(records)-len(gaps),
        maximum_selected_cost_change=max(changes) if changes else None,
        maximum_potential_error_bound=max(r["potential_error_bound"] for r in records),
        observed_zero_target_coordinates=int(np.count_nonzero(data["tau"] == 0)),
        coordinates_zero_in_every_observed_TRAIN_row=int(np.count_nonzero(np.all(data["tau"] == 0, axis=0))),
        structural_zero_proof="NOT_REPLAYED; observed zeros are not a structural proof",
        coordinatewise_marginal_cost_error_bound_available=False,
        scope="Only two refreshed TRAIN populations; not a global uncertainty/zero distribution")


def final_population(reader, binding):
    path = reader.bind(binding["path"], binding["sha256"], maximum=32*1024**2)
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["suite_id"] != "s03" or row["family"] != "ICNN":
                continue
            require(re.fullmatch(r"[0-9]+",row['region_index']) is not None, "Invalid final region index")
            key = f"s03:{int(row['region_index']):06d}"
            require(key not in result, "Duplicate final region")
            result[key] = dict(key=key, identity=row["current_train_dataset_identity"], rows=int(row["train_rows"]),
                               pairs=int(row["pairs"]), graph_sha256=row["graph_sha256"])
    require(len(result) == EXPECTED_REGIONS, "Final s03 coverage is not exactly1499")
    require(all(r["rows"] > 0 and r["pairs"] > 0 and re.fullmatch(r"[0-9a-f]{64}", r["identity"]) for r in result.values()),
            "Missing final TRAIN identity/count")
    return result


def supported_regions(support):
    """Active IDs are a sparse subset of the1847 frozen s03 region IDs."""
    suite=[s for s in support['suites'] if s['suite_id']=='s03']
    require(len(suite)==1 and suite[0]['active_regions']==EXPECTED_REGIONS
            and suite[0]['support_frozen_before_labels'] is True,"Incomplete frozen s03 support")
    records=suite[0]['region_records']
    require(all(isinstance(r['region_index'],int) and not isinstance(r['region_index'],bool)
                and r['region_index']>=0 for r in records),"Invalid frozen region index")
    regional={f"s03:{r['region_index']:06d}":r for r in records}
    require(len(records)==len(regional)==EXPECTED_REGIONS,"Duplicate/missing frozen active regions")
    return regional


def shared_window(clock, stage4_sha, run_id, now, stage4_start, stage4_deadline):
    require(clock.get("schema") == "PACNHN_FINAL_AUDIT_CLOCK_V1" and clock.get("phase_id") == "PACNHN_V1"
            and clock.get("run_id") == run_id and clock.get("stage4_clock_sha256") == stage4_sha,
            "Shared audit clock identity mismatch")
    start = _number(clock["start_epoch"], "shared audit start")
    end = _number(clock["deadline_epoch"], "shared audit deadline")
    require(stage4_start <= start <= now < end == min(start+1800, stage4_deadline),
            "Shared audit window expired/reset/over1800s")
    return end


class Budget:
    def __init__(self, ctx, deadline, resources):
        self.ctx, self.deadline, self.resources = ctx, deadline, resources
        self.peak_rss = 0; self.peak_tree_rss = 0; self.minimum_available = 2**63-1

    def __call__(self):
        self.ctx.guard(deadline=self.deadline)
        if time.time() >= self.deadline:
            raise TimeoutError("T08/shared Stage4 audit deadline reached")
        process = psutil.Process(); rss = process.memory_info().rss; tree = rss
        for child in process.children(recursive=True):
            try:
                tree += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        available = psutil.virtual_memory().available
        self.peak_rss = max(self.peak_rss, rss); self.peak_tree_rss = max(self.peak_tree_rss, tree)
        self.minimum_available = min(self.minimum_available, available)
        if rss > self.resources["single_process_rss_bytes"] or tree > self.resources["tree_rss_bytes"]:
            raise MemoryError("T08 RSS budget reached")
        if available < self.resources["minimum_available_bytes"]:
            raise MemoryError("T08 system available RAM guard")


def _csv(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _stream_digest(path, guard):
    guard(); h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            h.update(block); guard()
    return h.hexdigest()


def _seal(output, guard):
    artifacts = []
    for p in sorted(output.rglob("*")):
        guard()
        if p.is_file():
            artifacts.append(dict(relative_path=p.relative_to(output).as_posix(), bytes=p.stat().st_size,
                                  sha256=_stream_digest(p, guard)))
    guard()
    write_json(output/"ARTIFACT_MANIFEST.json", dict(artifacts=artifacts))
    with zipfile.ZipFile(output/"SYNC.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for row in artifacts + [dict(relative_path="ARTIFACT_MANIFEST.json")]:
            guard()
            with (output/row["relative_path"]).open("rb") as source, archive.open(row["relative_path"], "w") as target:
                for block in iter(lambda: source.read(1024**2), b""):
                    target.write(block); guard()
    with zipfile.ZipFile(output/"SYNC.zip") as archive:
        for row in artifacts:
            guard(); h = hashlib.sha256()
            with archive.open(row["relative_path"]) as stream:
                for block in iter(lambda: stream.read(1024**2), b""):
                    h.update(block); guard()
            require(h.hexdigest() == row["sha256"], "Archive digest mismatch")
        with archive.open("ARTIFACT_MANIFEST.json") as stream:
            archived_manifest = stream.read()
        require(hashlib.sha256(archived_manifest).hexdigest() == _stream_digest(output/"ARTIFACT_MANIFEST.json", guard),
                "Archived manifest digest mismatch")
    digest = _stream_digest(output/"SYNC.zip", guard)
    guard()
    write_json(output/"SEAL.json", dict(zip_sha256=digest,
        zip_bytes=(output/"SYNC.zip").stat().st_size, verified=True, artifacts=len(artifacts), verified_epoch=time.time()))
    guard()


def run(config_path, *, admission_path, admission_sha256):
    """Explicit production entry; no defaults, clock creation or automatic admission."""
    started = time.time(); plain = Reader()
    config_path = Path(config_path).resolve(strict=True)
    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    cfg = plain.json(dict(path=str(config_path), sha256=config_sha), maximum=2*1024**2)
    require(cfg["schema"] == "PACNHN_FINAL_TRAIN_LIMITS_CONFIG_V1" and cfg["phase_id"] == "PACNHN_V1"
            and cfg["step_id"] == "final_training_limits" and cfg["run_id"], "Explicit T08 config required")
    require(REQUIRED_TASKS <= set(cfg["tasks"]) and cfg["expected_outputs"] and cfg["acceptance_tests"]
            and cfg["input_versions"], "Incomplete T08 execution contract")
    require(set(cfg["fresh_regions"]) == FRESH_KEYS and len(cfg["fresh_regions"]) == 2,
            "Only the two approved stale regions may be replayed")
    resources = cfg["resources"]
    require(0 < resources["wall_seconds"] <= 1800 and 0 < resources["single_process_rss_bytes"] <= 1024**3
            and 0 < resources["tree_rss_bytes"] <= 24*1024**3 and resources["minimum_available_bytes"] >= 3*1024**3,
            "T08 resource guard may not be relaxed")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        require(os.environ.get(name) == "1", "Explicit one-thread environment required: " + name)
    campaign = plain.json(cfg["campaign_config"])
    require(campaign["run_id"] == cfg["run_id"], "Campaign/run mismatch")
    ctx = Context(cfg["campaign_config"]["path"], step_id="publication")
    ctx.require_action("analysis")
    from .handoff import publication_window
    window = publication_window(campaign, campaign_config_sha256=cfg["campaign_config"]["sha256"],
        stage4_clock=cfg["stage4_clock"], clock_amendment=cfg["clock_amendment"], deadline_epoch=cfg["deadline_epoch"])
    shared = plain.json(cfg["shared_audit_clock"])
    require(Path(cfg["shared_audit_clock"]["path"]).resolve() == ctx.root.resolve()/"final_audits/AUDIT_CLOCK.json",
            "Canonical shared audit clock required; alternate clocks cannot reset the audit")
    shared_end = shared_window(shared, cfg["stage4_clock"]["sha256"], cfg["run_id"], time.time(),
                               window["stage_start_epoch"], window["stage_deadline_epoch"])
    audit_end = min(window["deadline_epoch"], shared_end, started+resources["wall_seconds"])
    # Stop array/receipt processing early enough to persist a bounded handoff.
    deadline = audit_end-15.
    check = Budget(ctx, deadline, resources); check()
    reader = Reader(check); reader.bindings.update(plain.bindings)
    stage3 = reader.json(cfg["stage3_result"])
    require(Path(cfg["stage3_result"]["path"]).resolve() == ctx.root.resolve()/"STAGE3_RESULT.json"
            and stage3["status"] == "COMPLETE_WITH_RECORDED_OUTCOMES" and stage3["primary_rows"] == 14,
            "Actual terminal Stage3 outcomes required before T08")
    stage3_clock = reader.json(cfg["stage3_clock"])
    require(Path(cfg["stage3_clock"]["path"]).resolve() == ctx.root.resolve()/"STAGE3_CLOCK.json"
            and stage3.get("clock") == stage3_clock
            and stage3_clock["original_stage_start_epoch"]+stage3["elapsed_seconds"] <= window["stage_start_epoch"],
            "Stage4 precedes actual recorded Stage3 completion or its original clock differs")
    for clock_binding in (cfg["stage4_clock"], cfg["clock_amendment"]):
        reader.bind(clock_binding["path"], clock_binding["sha256"])
    inventory = reader.json(cfg["source_inventory"])
    source_rows = inventory["files"]
    require(len({str(Path(r["path"]).resolve()) for r in source_rows}) == len(source_rows), "Duplicate source inventory")
    expected = {str((Path(__file__).parent/name).resolve()) for name in SOURCE_NAMES}
    test_path = Path(__file__).parent/"tests/test_final_training_limits.py"
    expected.add(str(test_path.resolve()))
    require(expected <= {str(Path(r["path"]).resolve()) for r in source_rows}, "Incomplete T08 source closure")
    for row in source_rows:
        reader.bind(row["path"], row["sha256"], maximum=2*1024**2)
    admission = reader.json(dict(path=admission_path, sha256=admission_sha256))
    for key, value in dict(schema="PACNHN_FINAL_TRAIN_LIMITS_ADMISSION_V1", decision="ADMIT_FINAL_TRAIN_LIMITS_AUDIT",
            phase_id="PACNHN_V1", run_id=cfg["run_id"], config_sha256=config_sha,
            source_inventory_sha256=cfg["source_inventory"]["sha256"], stage4_clock_sha256=cfg["stage4_clock"]["sha256"],
            shared_audit_clock_sha256=cfg["shared_audit_clock"]["sha256"], stage3_result_sha256=cfg["stage3_result"]["sha256"]).items():
        require(admission.get(key) == value, "T08 admission mismatch: " + key)
    require(admission["tests_passed"] is True, "T08 requires current admitted tests")
    test_receipt = reader.json(admission["test_receipt"])
    require(test_receipt.get("status") == "PASS" and test_receipt.get("exit_code") == 0,
            "Admitted technical tests did not pass")
    source_hashes = {str(Path(row["path"]).resolve()): row["sha256"] for row in source_rows}
    require(test_receipt.get("source_sha256") == source_hashes[str(Path(__file__).resolve())]
            and test_receipt.get("test_sha256") == source_hashes[str(test_path.resolve())],
            "Technical tests bind different audit source/tests")
    output = Path(cfg["output_dir"]).resolve()
    require(output.is_relative_to(ctx.root.resolve()/"audits") and not output.exists(), "New owned audit directory required")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"INTENT.json", dict(config=cfg, config_sha256=config_sha, admission=admission,
        started_epoch=started, array_and_receipt_deadline_epoch=deadline, audit_deadline_epoch=audit_end,
        finalization_reserve_seconds=15., pid=os.getpid(), action="analysis", scientific_training=False))
    ledger = {}
    status = "FAIL"; error = None; screens = {}; uncertainty = {}; active_key = None
    try:
        support = reader.json(cfg["support_freeze"], maximum=16*1024**2)
        require(Path(cfg["support_freeze"]["path"]).resolve() == ctx.root.resolve()/"SUPPORT_FREEZE.json"
                and support["status"] == "PASS", "Frozen support is not canonical/PASS")
        regional=supported_regions(support)
        ledger={key:dict(region_key=key,disposition="NOT_STARTED",TRAIN_identity=None,TRAIN_rows=None,
            ordered_comparisons=None,violating_disjoint_pairs=None,fixed_label_WMAPE_percent_lower_bound=None,
            receipt_sha256=None) for key in regional}
        result = reader.json(cfg["training_result"])
        require(cfg["training_result"]["sha256"] == campaign["continuation"]["training_result_sha256"]
                and Path(cfg["training_result"]["path"]).resolve() == ctx.root.resolve()/"training/RESULT.json"
                and result["status"] == "STAGE2_FINISHED_WITH_EXPLICIT_COVERAGE", "Different terminal training")
        require(all(row["available"] == 0 for row in result["coverage"]
                    if row["suite_id"] != "s03" and row["family"] != "FFTT"),
                "Other-suite trained availability differs from admitted T08 scope")
        learning = dict(path=str(ctx.root/"training/LEARNING_RESULTS.csv"), sha256=result["learning_csv_sha256"])
        population = final_population(reader, learning)
        require(set(regional) == set(population), "Support/final population mismatch")
        prep = reader.json(cfg["preparation_manifest"])
        prepdir = Path(cfg["preparation_manifest"]["path"]).resolve().parent
        indexed = {r["relative_path"]: r for r in prep["artifacts"]}
        csvrow = indexed["CURRENT_RECEIPT_IDENTITIES.csv"]
        p = reader.bind(prepdir/csvrow["relative_path"], csvrow["sha256"])
        with p.open(encoding="utf-8-sig", newline="") as stream:
            old = list(csv.DictReader(stream))
        require(len(old) == EXPECTED_REGIONS and {r["region_key"] for r in old} == set(population), "Reuse inventory coverage mismatch")
        stale = set()
        for i, row in enumerate(old):
            check(); key = row["region_key"]; pop = population[key]
            active_key = key
            require(row["final_TRAIN_identity"] == pop["identity"] and int(row["final_TRAIN_rows"]) == pop["rows"]
                    and regional[key]["sha256"] == pop["graph_sha256"], "Reuse inventory/final CSV mismatch")
            ledger[key].update(TRAIN_identity=pop["identity"], TRAIN_rows=pop["rows"])
            path = Path(row["receipt_path"])
            expected_path = ctx.root/"training/regions/s03"/key.split(":")[1]/"monotonic_conflicts"/(row["saved_TRAIN_identity"]+".json")
            require(path.resolve() == expected_path.resolve(), "Reuse receipt path mismatch")
            receipt = reader.json(dict(path=str(path), sha256=row["receipt_sha256"]), maximum=2*1024**2)
            validate_screen(receipt, row["saved_TRAIN_identity"], int(row["saved_TRAIN_rows"]))
            if row["saved_TRAIN_identity"] != pop["identity"]:
                stale.add(key); ledger[key]["disposition"] = "PENDING_FRESH_SCREEN"
            else:
                require(key not in FRESH_KEYS, "Approved stale set changed; re-admission required")
                screens[key] = receipt
                ledger[key].update(disposition="REUSED_EXACT_CURRENT_SCREEN", ordered_comparisons=receipt["ordered_comparisons"],
                    violating_disjoint_pairs=receipt["violating_disjoint_pairs"],
                    fixed_label_WMAPE_percent_lower_bound=receipt["fixed_label_WMAPE_percent_lower_bound"], receipt_sha256=row["receipt_sha256"])
            if i % 50 == 0:
                _csv(output/"REGION_COVERAGE.csv", list(ledger.values()))
        require(stale == FRESH_KEYS, "Fresh set differs from approved two-region inventory")
        require(set(cfg["fresh_inputs"]) == FRESH_KEYS, "Missing fresh-region source bindings")
        for key in sorted(FRESH_KEYS):
            check(); binding = cfg["fresh_inputs"][key]; pop = population[key]; record = regional[key]
            active_key = key; ledger[key]["disposition"] = "RUNNING_FRESH_SCREEN"
            labelroot = ctx.root/"training/regions/s03"/key.split(":")[1]/"labels"
            identity = reader.json(binding["region_identity"], maximum=1024**2)
            graph = load_graph_identity(reader, record, identity)
            scale = native_scale(graph, record, reader.json(binding["native_scale"], maximum=1024**2), .01)
            data = load_train(reader, labelroot, binding, graph, pop, run_id=cfg["run_id"])
            check(); receipt = validate_screen(monotonic_screen(data, scale, graph["pair_keys"]), pop["identity"], pop["rows"])
            screens[key] = receipt; uncertainty[key] = summarize_fresh(data)
            destination = output/(key.replace(":", "_")+"_CURRENT_SCREEN.json")
            write_json(destination, receipt)
            write_json(output/(key.replace(":", "_")+"_LABEL_METADATA.json"), data["records"])
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            ledger[key].update(disposition="REFRESHED_EXACT_CURRENT_SCREEN", ordered_comparisons=receipt["ordered_comparisons"],
                violating_disjoint_pairs=receipt["violating_disjoint_pairs"],
                fixed_label_WMAPE_percent_lower_bound=receipt["fixed_label_WMAPE_percent_lower_bound"], receipt_sha256=digest)
            _csv(output/"REGION_COVERAGE.csv", list(ledger.values()))
            del data, graph
        require(len(screens) == EXPECTED_REGIONS, "Incomplete final1499screen coverage")
        reader.unchanged(); check(); status = "COMPLETE_WITH_SCOPE_LIMITATIONS"
    except (TimeoutError, MemoryError) as exc:
        status = "PARTIAL_BUDGET_OR_RESOURCE"; error = dict(type=type(exc).__name__, message=str(exc))
        if active_key is not None and active_key not in screens:
            ledger[active_key]["disposition"] = "INCOMPLETE_BUDGET_OR_RESOURCE"
    except Exception as exc:
        status = "FAIL"; error = dict(type=type(exc).__name__, message=str(exc))
        if active_key is not None and active_key not in screens:
            ledger[active_key]["disposition"] = "FAILED_VALIDATION"
    finally:
        _csv(output/"REGION_COVERAGE.csv", list(ledger.values()))
        write_json(output/"SOURCE_BINDINGS.json", list(reader.bindings.values()))
        write_json(output/"UNCERTAINTY_AND_ZEROS.json", dict(refreshed_regions=uncertainty,
            global_structural_zero_proofs="NOT_REPLAYED_OR_NEWLY_ESTABLISHED", global_label_uncertainty_distribution="NOT_RECOMPUTED",
            coordinatewise_marginal_cost_error_bound_available=False, limitations=LIMITATIONS))
        write_json(output/"SCREENS.json", screens)
        values = [r["fixed_label_WMAPE_percent_lower_bound"] for r in screens.values()
                  if r["fixed_label_WMAPE_percent_lower_bound"] is not None]
        summary = dict(status=status, expected_regions=EXPECTED_REGIONS, completed_regions=len(screens),
            reused_regions=sum(r["disposition"] == "REUSED_EXACT_CURRENT_SCREEN" for r in ledger.values()),
            refreshed_regions=sum(r["disposition"] == "REFRESHED_EXACT_CURRENT_SCREEN" for r in ledger.values()),
            TRAIN_inputs=sum(r["unique_rows"] for r in screens.values()), ordered_comparisons=sum(r["ordered_comparisons"] for r in screens.values()),
            positive_bound_regions=sum(v > 0 for v in values), above_1_percent_regions=sum(v > 1 for v in values),
            maximum_bound_percent=max(values) if values else None, error=error, limitations=LIMITATIONS,
            untrained_suites=["s02", "s04", "s05", "s06"], other_suite_conflicts="UNAVAILABLE_NO_TRAINED_CURRENT_TRAIN_POPULATION",
            started_epoch=started, ended_epoch=time.time(), deadline_epoch=deadline,
            audit_deadline_epoch=audit_end, finalization_reserve_seconds=15.,
            resource=dict(peak_sampled_process_RSS_bytes=check.peak_rss, peak_sampled_tree_RSS_bytes=check.peak_tree_rss,
                          minimum_sampled_available_bytes=check.minimum_available), TEST_targets_opened=False,
            new_labels_or_fits=False, selected_models_changed=False, exact_physical_cost_truth_claim=False)
        write_json(output/"RESULT.json", summary)
        write_json(output/"STATUS.json", dict(status=status, complete_region_coverage=len(screens) == EXPECTED_REGIONS,
            next_action="Review exact TRAIN screen and limitations; no retraining/TEST/target changes authorized"))
        write_json(output/"VALIDATION.json", dict(status=status, full_planned_ledger_rows=len(ledger),
            completed_screen_count=len(screens), current_identity_checks_scope="COMPLETED_PREFIX_ONLY" if error else "ALL1499",
            source_bindings=len(reader.bindings), error=error))
        (output/"REPORT.md").write_text("# Final frozen TRAIN limitations audit\n\n"+json.dumps(summary, indent=2)+
            "\n\nBounds apply to recorded targets and the named non-exhaustive screen. Other suites remain unavailable. "
            "All planned1499regions remain in REGION_COVERAGE.csv, including any missing/failed rows.\n", encoding="utf-8")
        seal_check = Budget(ctx, audit_end, resources)
        try:
            _seal(output, seal_check)
        except Exception as exc:
            seal_status = ("PARTIAL_HANDOFF_DEADLINE_OR_RESOURCE" if isinstance(exc, (TimeoutError, MemoryError))
                           else "FAIL_HANDOFF_INTEGRITY")
            summary.update(status=seal_status, handoff_verified=False,
                           handoff_error=dict(type=type(exc).__name__, message=str(exc)))
            # Preserve any incomplete archive. Recovery metadata is not a new
            # analysis run and must never certify an expired attempt as sealed.
            write_json(output/"SEAL_INVALIDATED.json", dict(verified=False, status=summary["status"],
                error=summary["handoff_error"], observed_epoch=time.time(), original_audit_deadline_epoch=audit_end))
            write_json(output/"STATUS.json", dict(status=summary["status"], handoff_verified=False))
            write_json(output/"RESULT.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--admission", required=True)
    parser.add_argument("--admission-sha256", required=True)
    args = parser.parse_args()
    result = run(args.config, admission_path=args.admission, admission_sha256=args.admission_sha256)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "COMPLETE_WITH_SCOPE_LIMITATIONS" else 2)
