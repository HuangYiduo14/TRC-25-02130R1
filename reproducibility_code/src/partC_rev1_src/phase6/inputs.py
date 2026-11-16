"""Read-only native source authentication and explicit trace-role reconciliation."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from partC_rev1_src.phase4.inputs import load_phase4_graph
from partC_rev1_src.phase4.tracing import load_suite_trace_maps
from partC_rev1_src.phase5.inputs import _physical_state, load_suite_trace_arrays
from .common import read_json, sha, write_json, write_frame


def load_native(c, *, verify=True):
    p4 = read_json(c["phase4_graph_config_file"])
    sources = read_json(c["native_source_binding_file"])
    g = load_phase4_graph(p4)
    t = load_suite_trace_maps(p4, g, suite_ids=["s03"], verify_directory_digests=verify)[0]
    if t.directory_digest != c["expected_interface"]["phase3_s03_directory_digest"]:
        raise ValueError("Wrong s03 source directory identity")
    if g.graph_hash != sources["graph"]["identity"]:
        raise ValueError("Wrong physical graph")
    frames = {name: pd.read_parquet(t.root / (name + ".parquet")) for name in
              ["regional_pair_support_summary", "consolidated_terminals", "terminal_fine_members",
               "terminal_connectors", "acnhn_nodes"]}
    return g, t, frames, sources, p4


def read_tier(c, g, t, sources, tier):
    if tier not in {"low_benchmark", "high_benchmark"}: raise ValueError("Wrong tier")
    state = _physical_state(sources[tier], .3, load_aon=False)
    if not np.array_equal(state.used_capacity, g.capacity * .3):
        raise ValueError("Accepted capacity must be scaled exactly once")
    expected = g.free_time * 60 * (1 + g.alpha * (state.current_flow / state.used_capacity) ** g.beta)
    if not np.allclose(state.current_cost_s, expected, rtol=2e-15, atol=1e-7):
        raise ValueError("Native minutes-to-seconds BPR identity failed")
    arrays = load_suite_trace_arrays(sources, "s03", tier, trace_map=t)
    for k, v in arrays.items():
        if k.startswith("current__") and v.dtype.kind == "f":
            if not np.isfinite(v).all() or (v < 0).any():
                raise ValueError("Nonfinite/negative CURRENT, no clipping allowed: " + k)
    return state, arrays


def reconcile_fine(g, t, arrays):
    """Recover only role splits uniquely determined by both native marginals.

    Fine trace keys omit endpoint-vs-boundary role. Most fine keys have one
    possible positive coarse key; the remainder must have a unique nonnegative
    solution to the original fine AND coarse marginal equations. No proportional
    disaggregation or new demand is inferred. Report this serialization limitation.
    """
    fk = arrays["current__fine_regional_pair_flow__keys"]
    fv = arrays["current__fine_regional_pair_flow__values"]
    ck = arrays["current__coarse_regional_pair_flow__keys"]
    cv = arrays["current__coarse_regional_pair_flow__values"]
    region = (fk >> np.uint64(38)).astype(np.int64)
    sn = ((fk >> np.uint64(19)) & np.uint64((1 << 19) - 1)).astype(np.int64)
    tn = (fk & np.uint64((1 << 19) - 1)).astype(np.int64)
    n = len(g.node_idx_to_uid)
    si = np.searchsorted(t.terminal_region_node_keys, region * n + sn)
    ti = np.searchsorted(t.terminal_region_node_keys, region * n + tn)
    if np.any(si >= len(t.terminal_region_node_keys)) or np.any(ti >= len(t.terminal_region_node_keys)):
        raise ValueError("Unknown fine endpoint")
    if not np.array_equal(t.terminal_region_node_keys[si], region * n + sn) or not np.array_equal(t.terminal_region_node_keys[ti], region * n + tn):
        raise ValueError("Unknown fine endpoint role")
    possibilities = []
    for sm in [0, 2]:
        for tm in [1, 3]:
            source = t.terminal_region_node_values[sm, si]
            target = t.terminal_region_node_values[tm, ti]
            key = (source.astype(np.uint64) << np.uint64(32)) | target.astype(np.uint64)
            ix = np.searchsorted(ck, key)
            safe = np.minimum(ix, len(ck) - 1)
            good = (source >= 0) & (target >= 0) & (ix < len(ck)) & (ck[safe] == key)
            possibilities.append(np.column_stack([np.flatnonzero(good), ix[good]]))
    edges = np.unique(np.concatenate(possibilities), axis=0)
    counts = np.bincount(edges[:, 0], minlength=len(fk))
    if np.any(counts == 0): raise ValueError("Fine CURRENT key has no corresponding native coarse role")
    singleton = counts[edges[:, 0]] == 1
    amounts = np.zeros(len(edges)); amounts[singleton] = fv[edges[singleton, 0]]
    assigned = np.bincount(edges[singleton, 1], weights=amounts[singleton], minlength=len(ck))
    residual = cv - assigned
    unresolved = edges[~singleton]
    rank = 0; singular_values = []
    if len(unresolved):
        fine_rows = np.unique(unresolved[:, 0]); coarse_rows = np.unique(unresolved[:, 1])
        A = np.zeros((len(fine_rows) + len(coarse_rows), len(unresolved)))
        cols = np.arange(len(unresolved))
        A[np.searchsorted(fine_rows, unresolved[:, 0]), cols] = 1.
        A[len(fine_rows) + np.searchsorted(coarse_rows, unresolved[:, 1]), cols] = 1.
        rhs = np.r_[fv[fine_rows], residual[coarse_rows]]
        solution, _, rank, singular = np.linalg.lstsq(A, rhs, rcond=None)
        singular_values = singular.tolist()
        if rank != len(unresolved):
            raise ValueError(f"Native role split is underdetermined: {len(unresolved)} variables, rank {rank}")
        allowance = 1e-9 + 1e-12 * np.max(np.abs(rhs), initial=0.)
        if np.min(solution, initial=0.) < -allowance or np.max(np.abs(A @ solution - rhs), initial=0.) > allowance:
            raise ValueError("Native fine/coarse marginals inconsistent")
        if (solution < 0).any():
            # Roundoff in reconstructed derived role allocation, not source clipping.
            solution[solution < 0] = 0.
        amounts[~singleton] = solution
    fine_sum = np.bincount(edges[:, 0], weights=amounts, minlength=len(fk))
    coarse_sum = np.bincount(edges[:, 1], weights=amounts, minlength=len(ck))
    if not np.allclose(fine_sum, fv, rtol=1e-12, atol=1e-9) or not np.allclose(coarse_sum, cv, rtol=1e-12, atol=1e-9):
        raise ValueError("Complete native Q-fine/coarse closure failed")
    key = ck[edges[:, 1]]
    frame = pd.DataFrame({"fine_key": fk[edges[:, 0]], "coarse_key": key,
        "region_index": region[edges[:, 0]], "source_global_node": sn[edges[:, 0]],
        "target_global_node": tn[edges[:, 0]], "amount": amounts,
        "role_split_reconstructed_from_unique_native_marginals": ~singleton})
    report = dict(coarse_keys=len(ck), fine_keys=len(fk), ambiguous_fine_keys=int((counts > 1).sum()),
                  derived_role_variables=len(unresolved), derived_role_system_rank=int(rank),
                  fine_max_absolute_error=float(np.abs(fine_sum - fv).max(initial=0.)),
                  coarse_max_absolute_error=float(np.abs(coarse_sum - cv).max(initial=0.)),
                  total_coarse_exposure=float(cv.sum()), total_fine_exposure=float(fv.sum()),
                  source_values_modified=False, unique_role_reconstruction=True)
    return frame, report


def source_manifest(c, t, sources, p4):
    files = [Path(c["phase4_graph_config_file"]), Path(c["native_source_binding_file"]),
             Path(p4["input_versions"]["graph_npz"]), Path(p4["input_versions"]["graph_metadata_json"])]
    files.extend(p for p in t.root.rglob("*") if p.is_file())
    for tier in ["low_benchmark", "high_benchmark"]:
        files += [Path(sources[tier]["physical_arrays"]), Path(sources[tier]["suite_trace_template"].format(suite="s03"))]
    rows = [dict(absolute_path=str(p), size_bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(set(files))]
    # Authenticated inherited bundle supplies hashes for immutable physical/trace files.
    evidence = Path(c["authority_root"]) / "inherited/base/evidence/external_artifact_bindings.csv"
    bindings = pd.read_csv(evidence)
    expected = {str(Path(r.absolute_path).resolve()): r for r in bindings.itertuples()}
    for row in rows:
        e = expected.get(str(Path(row["absolute_path"]).resolve()))
        if e is not None and e.format != "Parquet directory":
            if row["size_bytes"] != int(e.size_bytes) or row["sha256"] != e.sha256_or_digest:
                raise ValueError("External source binding differs: " + row["absolute_path"])
    return pd.DataFrame(rows)
