"""Native fixed-backbone metrics and reference-only, immutable key selection.

No discovery, numerical run or file write occurs on import. Callers supply exact
case identities and paths after the reference gate. The first B coordinates are
compared directly; physical projection/recovery is not part of this module.
"""
from __future__ import annotations

import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .control import array_hash, read_json, sha, write_json

BACKBONE_COUNT = 87582
KEY_COUNT = 200
SCENARIOS = {"OPEN", "REPRESENTED_I95"}
ROLES = {"REFERENCE_SELF", "INDEPENDENT_CONTROL", "SURROGATE"}
GEO_COLUMNS = ["tail_longitude", "tail_latitude", "head_longitude", "head_latitude"]


def _vector(value, name, *, nonnegative=True):
    raw = np.asarray(value)
    if not np.isrealobj(raw):
        raise ValueError(f"{name}: real values required")
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError(f"{name}: nonempty finite vector required")
    if nonnegative and np.any(a < 0):
        raise ValueError(f"{name}: negative value")
    return a


def _bool_vector(value, name, length):
    a = np.asarray(value)
    if a.dtype != np.dtype(bool) or a.shape != (length,):
        raise ValueError(f"{name}: exact boolean vector/order required")
    return a


def _ids(values, name):
    s = pd.Series(values, copy=False)
    if s.isna().any():
        raise ValueError(f"{name}: missing IDs")
    ids = s.astype(str).to_numpy(dtype=str)
    if not len(ids) or np.any(np.char.str_len(ids) == 0) or len(np.unique(ids)) != len(ids):
        raise ValueError(f"{name}: nonempty unique IDs required")
    return ids


def backbone_frame(backbone, *, expected_count=BACKBONE_COUNT):
    """Validate the recorded native order without sorting/relabeling coordinates."""
    missing = set(["link_id"] + GEO_COLUMNS) - set(backbone.columns)
    if missing:
        raise ValueError("Missing backbone fields: " + ",".join(sorted(missing)))
    f = backbone.copy().reset_index(drop=True)
    if len(f) != expected_count or expected_count <= 0:
        raise ValueError("Frozen backbone count differs")
    f["link_id"] = _ids(f.link_id, "backbone")
    native = np.arange(len(f), dtype=np.int64)
    if "native_index" in f and not np.array_equal(f.native_index.to_numpy(), native):
        raise ValueError("Backbone native order differs")
    f["native_index"] = native
    xy = f[GEO_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(xy).all():
        raise ValueError("Nonfinite backbone geography")
    if np.any(abs(xy[:, [0, 2]]) > 180) or np.any(abs(xy[:, [1, 3]]) > 90):
        raise ValueError("Invalid longitude/latitude units")
    return f


def backbone_order_identity(backbone):
    return array_hash({"link_id": _ids(backbone.link_id, "backbone")})


def extract_backbone(flow, edge_ids, backbone, *, expected_count=BACKBONE_COUNT):
    """Require the full graph's first B IDs to match the common topology exactly."""
    f = backbone_frame(backbone, expected_count=expected_count)
    x = _vector(flow, "native flow")
    ids = _ids(edge_ids, "native graph")
    if len(x) != len(ids) or len(x) < len(f) or not np.array_equal(ids[:len(f)], f.link_id):
        raise ValueError("Native flow/backbone prefix identity differs")
    return x[:len(f)].copy()


def select_keys(backbone, open_reference_flow, *, reference_id,
                reference_scenario, target=KEY_COUNT, expected_count=BACKBONE_COUNT):
    """Prior paper selection rule, freshly applied only to the open P reference.

    Full-B science is unfiltered. The explicitly defined key sampling pool is
    positive-reference CONUS representative chords. EPSG:4269 -> EPSG:5070 and
    250 km cells preserve the prior geodispersed/high-flow sampling semantics.
    Candidate predictions/errors are not inputs and cannot affect selection.
    """
    if reference_scenario != "OPEN" or not reference_id:
        raise ValueError("Key selection requires the identified OPEN P reference")
    if not isinstance(target, (int, np.integer)) or target <= 0:
        raise ValueError("Positive integer key count required")
    frame = backbone_frame(backbone, expected_count=expected_count)
    ref = _vector(open_reference_flow, "open reference")
    if ref.shape != (len(frame),):
        raise ValueError("Open reference is not the native B vector")
    # Import only when selection is explicitly called; no dependency activation
    # or source access on module import.
    from pyproj import Transformer
    transform = Transformer.from_crs("EPSG:4269", "EPSG:5070", always_xy=True)
    for end in ("tail", "head"):
        frame[end+"_x"], frame[end+"_y"] = transform.transform(
            frame[end+"_longitude"].to_numpy(), frame[end+"_latitude"].to_numpy())
    for name in ("longitude", "latitude", "x", "y"):
        frame[name] = (frame["tail_"+name] + frame["head_"+name])/2
    if not np.isfinite(frame[["x", "y"]].to_numpy()).all():
        raise ValueError("Backbone geographic projection failed")
    frame["lambda_reference"] = ref
    frame["reference_id"] = str(reference_id)
    eligible = (frame.lambda_reference.gt(0) & frame.longitude.between(-125, -66)
                & frame.latitude.between(24, 50))
    pool = frame.loc[eligible].sort_values(
        ["lambda_reference", "link_id"], ascending=[False, True], kind="stable").copy()
    if len(pool) < target:
        raise ValueError("Insufficient positive geographically located reference links")
    fraction = next(f for f in (.2, .25, .5, 1.) if math.ceil(len(pool)*f) >= target)
    candidates = pool.head(math.ceil(len(pool)*fraction)).copy()
    cells = np.floor(candidates[["x", "y"]].to_numpy()/250000).astype(np.int64)
    candidates["cell_id"] = pd.Series(cells[:, 0], index=candidates.index).astype(str) + ":" + pd.Series(cells[:, 1], index=candidates.index).astype(str)
    # This vectorized rank ordering is the same cell-by-cell round-robin as the
    # old code, without nested loops over the candidate observations.
    cell_heads = candidates.drop_duplicates("cell_id").sort_values(
        ["lambda_reference", "cell_id"], ascending=[False, True], kind="stable")
    order = {cell: i for i, cell in enumerate(cell_heads.cell_id)}
    candidates["within_cell_rank"] = candidates.groupby("cell_id", sort=False).cumcount()
    candidates["cell_rank"] = candidates.cell_id.map(order)
    keys = candidates.sort_values(["within_cell_rank", "cell_rank"], kind="stable").head(target).copy()
    keys["sample_rank"] = np.arange(1, target+1)
    keys = keys.reset_index(drop=True)
    meta = dict(schema="pacnhn.fixed_keys.v1", target=int(target), reference_id=str(reference_id),
        reference_scenario="OPEN", backbone_count=len(frame),
        backbone_order_identity=backbone_order_identity(frame),
        open_reference_flow_identity=array_hash({"flow": ref}),
        candidate_fraction=fraction, candidate_count=len(candidates),
        positive_CONUS_count=len(pool), geographic_or_zero_exclusions=int((~eligible).sum()),
        full_backbone_population_filtered=False,
        selection_pool="POSITIVE_OPEN_P_FLOW_AND_CONUS_REPRESENTATIVE_MIDPOINT",
        projection="EPSG:4269_TO_EPSG:5070", grid_metres=250000,
        candidate_minimum_reference_flow=float(candidates.lambda_reference.min()),
        selected_reference_flow_share=math.fsum(keys.lambda_reference)/math.fsum(ref),
        occupied_candidate_cells=len(cell_heads), occupied_selected_cells=int(keys.cell_id.nunique()),
        ordering="CELL_BEST_REFERENCE_DESC_THEN_CELL_ID; WITHIN_CELL_REFERENCE_DESC_THEN_LINK_ID; ROUND_ROBIN",
        random_seed="NONE_DETERMINISTIC", prediction_errors_used=False,
        key_selection_role="REFERENCE_ONLY", key_ids=keys.link_id.tolist())
    frame["is_key_link"] = frame.link_id.isin(keys.link_id)
    return keys, frame, meta


def freeze_keys(output_root, backbone, open_reference_flow, *, reference_id,
                reference_scenario, source_bindings, target=KEY_COUNT,
                expected_count=BACKBONE_COUNT):
    """Create a new immutable key folder; never replace an existing freeze."""
    if not source_bindings or any(not r.get("path") or not r.get("sha256") for r in source_bindings):
        raise ValueError("Exact reference/topology source bindings required")
    for binding in source_bindings:
        if sha(binding["path"]) != binding["sha256"]:
            raise ValueError("Key-selection input source changed: " + str(binding["path"]))
    keys, full, meta = select_keys(backbone, open_reference_flow, reference_id=reference_id,
        reference_scenario=reference_scenario, target=target, expected_count=expected_count)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=False)
    keys.to_csv(root/"key_links_200.csv", index=False, float_format="%.17g")
    full.to_csv(root/"open_backbone_reference.csv", index=False, float_format="%.17g")
    meta.update(frozen_epoch=time.time(), source_bindings=source_bindings,
        key_file_sha256=sha(root/"key_links_200.csv"),
        reference_file_sha256=sha(root/"open_backbone_reference.csv"))
    write_json(root/"KEY_SELECTION_FREEZE.json", meta)
    return meta


def load_frozen_keys(root, backbone, open_reference_flow, *, reference_id,
                     expected_count=BACKBONE_COUNT, expected_key_count=KEY_COUNT):
    root = Path(root); meta = read_json(root/"KEY_SELECTION_FREEZE.json")
    frame = backbone_frame(backbone, expected_count=expected_count)
    ref = _vector(open_reference_flow, "open reference")
    expected = (meta.get("schema") == "pacnhn.fixed_keys.v1" and meta.get("reference_scenario") == "OPEN"
        and meta.get("target") == expected_key_count and meta.get("reference_id") == reference_id
        and meta.get("backbone_count") == len(frame)
        and meta.get("backbone_order_identity") == backbone_order_identity(frame)
        and meta.get("open_reference_flow_identity") == array_hash({"flow": ref}))
    if not expected or ref.shape != (len(frame),):
        raise ValueError("Key freeze/reference/backbone identity differs")
    for name, field in (("key_links_200.csv", "key_file_sha256"),
                        ("open_backbone_reference.csv", "reference_file_sha256")):
        if sha(root/name) != meta.get(field):
            raise ValueError("Frozen key/reference file changed: " + name)
    keys = pd.read_csv(root/"key_links_200.csv", dtype={"link_id": str}, float_precision="round_trip")
    ids = _ids(keys.link_id, "frozen keys")
    if len(ids) != expected_key_count or ids.tolist() != meta.get("key_ids"):
        raise ValueError("Frozen key membership/order differs")
    key_index = pd.Index(frame.link_id).get_indexer(ids)
    if np.any(key_index < 0) or not np.array_equal(key_index, keys.native_index.to_numpy()):
        raise ValueError("Frozen keys do not join the native backbone")
    if not np.array_equal(ref[key_index], keys.lambda_reference.to_numpy()):
        raise ValueError("Frozen key reference values differ")
    return keys, meta


def compare_vectors(reference, predicted, keys, base=None, k4=None):
    """Prior paper ratio-of-sums and signed bounded per-link definitions.

    Both-reference-and-prediction zero has bounded error 0; relative error at
    zero reference is undefined. No mask removes reference-zero coordinates.
    Optional base/k4 are once-owned native cost coefficients, never refitted.
    """
    ref = _vector(reference, "reference"); pred = _vector(predicted, "prediction")
    if ref.shape != pred.shape:
        raise ValueError("Comparison coordinate shape differs")
    key = _bool_vector(keys, "keys", len(ref))
    if not key.any():
        raise ValueError("Empty key comparison population")
    if (base is None) != (k4 is None):
        raise ValueError("Both backbone cost coefficient vectors are required")
    if base is not None:
        base = _vector(base, "base"); k4 = _vector(k4, "k4")
        if base.shape != ref.shape or k4.shape != ref.shape:
            raise ValueError("Cost coefficient native shape differs")
    difference = pred-ref; absolute = abs(difference); scale = np.maximum(ref, pred)
    per = pd.DataFrame(dict(lambda_reference=ref, lambda_estimate=pred,
        signed_error=difference, absolute_error=absolute,
        signed_bounded_error=np.divide(difference, scale, out=np.zeros_like(scale), where=scale > 0),
        signed_relative_error=np.divide(difference, ref, out=np.full(len(ref), np.nan), where=ref > 0),
        absolute_relative_error=np.divide(absolute, ref, out=np.full(len(ref), np.nan), where=ref > 0),
        is_key_link=key))
    denominator_all = math.fsum(ref)
    per["backbone_wmape_contribution_percent"] = 100*(absolute/denominator_all) if denominator_all else np.nan
    rows = []
    for population, mask in (("BACKBONE", np.ones(len(ref), bool)), ("KEY_LINKS", key)):
        r, p = ref[mask], pred[mask]; d = p-r
        numerator = math.fsum(abs(d)); denominator = math.fsum(r)
        largest = float(np.max(abs(d)))
        rmse = largest*float(np.sqrt(np.mean((d/largest)**2))) if largest else 0.
        rc = pc = None
        if base is not None:
            with np.errstate(over="ignore", invalid="ignore"):
                rc = float(r@(base[mask]+k4[mask]*r**4))
                pc = float(p@(base[mask]+k4[mask]*p**4))
            if not math.isfinite(rc) or not math.isfinite(pc):
                raise FloatingPointError("Nonfinite native backbone cost metric")
        rows.append(dict(population=population, link_count=len(r), reference_flow_sum=denominator,
            absolute_error_sum=numerator, wmape_percent=100*(numerator/denominator) if denominator else None,
            wmape_status="DEFINED" if denominator else "UNDEFINED_ZERO_REFERENCE_TOTAL",
            mae=numerator/len(r), rmse=rmse, mean_signed_error=math.fsum(d)/len(r),
            reference_zero_link_count=int((r == 0).sum()),
            reference_zero_positive_prediction_count=int(((r == 0) & (p > 0)).sum()),
            reference_backbone_cost=rc, predicted_backbone_cost=pc,
            backbone_cost_error_percent=100*abs(pc-rc)/rc if rc else None,
            backbone_cost_status=("UNAVAILABLE_COEFFICIENTS" if rc is None else
                                  "DEFINED" if rc else "UNDEFINED_ZERO_REFERENCE_COST"),
            flow_units="ANNUAL_KTON", cost_units="ANNUAL_KTON_SECONDS"))
    return rows, per


def compare_native(backbone, reference, predicted, frozen_keys, *, scenario,
                   reference_case_id, case_id, method_id, assignment_role,
                   reference_serve, predicted_serve, closed_mask=None,
                   base=None, k4=None, expected_count=BACKBONE_COUNT,
                   expected_key_count=KEY_COUNT):
    """Compare already admitted states with exact served membership and native IDs.

    The caller supplies this scenario's P reference. Closed comparisons retain
    the OPEN-frozen IDs even when their CLOSED reference flows become zero.
    Independent CFW controls are ordinary comparisons; no solver label forces 0.
    """
    if scenario not in SCENARIOS or assignment_role not in ROLES:
        raise ValueError("Explicit scenario and assignment role required")
    if not all(isinstance(x, str) and x for x in (reference_case_id, case_id, method_id)):
        raise ValueError("Nonempty case/method identities required")
    if (assignment_role == "REFERENCE_SELF") != (reference_case_id == case_id):
        raise ValueError("Reference-self role must be explicit and distinct from independent cases")
    frame = backbone_frame(backbone, expected_count=expected_count)
    ref = _vector(reference, "reference"); pred = _vector(predicted, "prediction")
    if ref.shape != (len(frame),) or pred.shape != ref.shape:
        raise ValueError("Comparison requires exact full native B vectors")
    sr = np.asarray(reference_serve); sp = np.asarray(predicted_serve)
    if sr.ndim != 1 or not len(sr) or sr.dtype != bool or sp.dtype != bool or not np.array_equal(sr, sp):
        raise ValueError("Exact served OD membership differs or is unavailable")
    if assignment_role == "REFERENCE_SELF" and not np.array_equal(ref, pred):
        raise ValueError("Reference-self data are not identical")
    ids = _ids(frozen_keys.link_id, "frozen keys")
    if len(ids) != expected_key_count or not pd.Index(ids).isin(frame.link_id).all():
        raise ValueError("Frozen key IDs/count differ from common B")
    key = frame.link_id.isin(ids).to_numpy()
    if scenario == "REPRESENTED_I95":
        closed = _bool_vector(closed_mask, "represented I95 B mask", len(frame))
        if np.any(ref[closed] != 0) or np.any(pred[closed] != 0):
            raise ValueError("Nonzero represented closed-backbone flow")
    else:
        closed = np.zeros(len(frame), bool)
    rows, per = compare_vectors(ref, pred, key, base, k4)
    identity = dict(scenario=scenario, reference_case_id=reference_case_id, case_id=case_id,
        method_id=method_id, assignment_role=assignment_role,
        backbone_order_identity=backbone_order_identity(frame),
        served_OD_membership_equal=True, comparison_space="COMMON_NATIVE_BACKBONE",
        key_origin_scenario="OPEN", closed_backbone_count=int(closed.sum()))
    for row in rows:
        row.update(identity)
    per.insert(0, "link_id", frame.link_id.to_numpy())
    per.insert(1, "native_index", np.arange(len(frame)))
    per["is_closed"] = closed
    for k, v in identity.items():
        per[k] = v
    return rows, per


def admit_assignment(root, *, expected_graph_identity, expected_edge_ids,
                     expected_OD_identity=None):
    """Read current solver.py output only when explicitly called after a run.

    Return its complete certified native flow, original result, served membership
    and source hashes. Qualified TIME_LIMIT/STAGNATION states stay qualified;
    pilot-update-limit and uncertified states are rejected for final comparisons.
    """
    root = Path(root); result = read_json(root/"RESULT.json")
    if result.get("status") not in {"CONVERGED", "TIME_LIMIT", "STAGNATION"}:
        raise ValueError("Assignment status is not admitted for final comparison")
    cert = result.get("certificate")
    if result.get("final_certificate_fresh") is not True or not isinstance(cert, dict) or cert.get("fresh") is not True:
        raise ValueError("Missing fresh final-state certificate")
    if result.get("reported_state") != "LAST_FRESH_CERTIFIED_STATE" or not isinstance(result.get("reported_state_iteration"), int) or result["reported_state_iteration"] < 0:
        raise ValueError("Reported state is not an identified certified iteration")
    if not expected_graph_identity or result.get("graph_identity") != expected_graph_identity:
        raise ValueError("Assignment graph identity differs")
    if expected_OD_identity is not None and result.get("OD_identity") != expected_OD_identity:
        raise ValueError("Assignment OD identity differs")
    with np.load(root/"final_flow.npz", allow_pickle=False) as z:
        flow = _vector(z["flow"], "certified final flow").copy()
    edge_ids = _ids(expected_edge_ids, "expected graph")
    if len(flow) != len(edge_ids) or cert.get("state_hash") != array_hash({"flow": flow}):
        raise ValueError("Final flow/certificate state identity differs")
    numbers = {k: float(cert[k]) for k in ("TSTT", "SPTT", "raw_signed_gap", "arithmetic_allowance")}
    if not all(math.isfinite(v) for v in numbers.values()) or min(numbers["TSTT"], numbers["SPTT"], numbers["arithmetic_allowance"]) < 0:
        raise ValueError("Invalid signed certificate values")
    gap, sptt, allowance = numbers["raw_signed_gap"], numbers["SPTT"], numbers["arithmetic_allowance"]
    if gap < -allowance or not math.isclose(gap, numbers["TSTT"]-sptt, rel_tol=1e-12, abs_tol=max(1e-12, allowance)):
        raise ValueError("Inconsistent or materially negative signed gap")
    if sptt > 0:
        rg = cert.get("RG_SPTT")
        if rg is None or not math.isfinite(float(rg)) or not math.isclose(float(rg), gap/sptt, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("Relative and raw certificate gaps differ")
    elif cert.get("RG_SPTT") is not None:
        raise ValueError("Zero SPTT must retain an undefined relative gap")
    has_potential = result.get("objective_has_potential")
    if not isinstance(has_potential, bool):
        raise ValueError("Missing objective-interface identity")
    interpretation = "CONVEX_POTENTIAL_FW_GAP" if has_potential else "POINTWISE_VI_RESIDUAL_NO_POTENTIAL"
    if cert.get("interpretation") != interpretation or (not has_potential and result.get("method") != "MSA"):
        raise ValueError("Certificate interpretation and solver interface differ")
    if has_potential:
        if cert.get("value") is None or not math.isfinite(float(cert["value"])):
            raise ValueError("Missing finite potential certificate")
    elif cert.get("value") is not None:
        raise ValueError("Cost-only state must not invent a potential")
    if result["status"] == "CONVERGED":
        tolerance = float(result["nominal_tolerance"])
        if not math.isfinite(tolerance) or tolerance <= 0 or (sptt > 0 and float(cert["RG_SPTT"]) > tolerance) or (sptt == 0 and abs(gap) > allowance):
            raise ValueError("Convergence label exceeds its stated tolerance")
    with np.load(root/"coverage.npz", allow_pickle=False) as z:
        serve = np.asarray(z["serve"]).copy()
    if serve.ndim != 1 or serve.dtype != bool or not len(serve) or not serve.any():
        raise ValueError("Missing nonempty served-OD membership")
    bindings = [{"path": str(root/name), "sha256": sha(root/name), "bytes": (root/name).stat().st_size}
                for name in ("RESULT.json", "final_flow.npz", "coverage.npz")]
    return flow, result, serve, bindings
