"""All-region active connectivity overlay built from immutable directed support."""
from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import pandas as pd

from partC_rev1_src.phase3a.schemas import stable_object_id
from partC_rev1_src.phase3a.regional_pairs import LazyRegionalPairOracle, build_region_support
from . import MODEL_CLASS, kernels as K
from .common import (read_json, read_npz, write_json, write_npz, write_frame, arrays_hash,
                     digest, sha, event, Monitor)
from .inputs import load_native, read_tier


def proxy(m):
    if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m < 0: raise ValueError("Bad dimension")
    return 65 * int(m) + 1121 if m else 0


def quota_target(m):
    return 32 * max(2, (proxy(m) + 1023) // 1024) if m else 0


def region_dir(c, rid):
    return Path(c["output_root"]) / "regions" / rid.removeprefix("partC_p2_region_")


def make_choices(pair_source, pair_target, members, *, source_ids=None):
    sources = sorted(set(pair_source.tolist())) if source_ids is None else list(source_ids)
    source_map = {s: i for i, s in enumerate(sources)}
    sn = [members[s] for s in sources]
    offsets = np.r_[0, np.cumsum([len(v) for v in sn])].astype(np.int64)
    lengths = np.array([len(members[t]) for t in pair_target], np.int64)
    coff = np.r_[0, np.cumsum(lengths)].astype(np.int64)
    cg = np.repeat(np.array([source_map[s] for s in pair_source], np.int64), lengths)
    ct = np.concatenate([members[t] for t in pair_target]) if len(pair_target) else np.empty(0, np.int64)
    return dict(source_offsets=offsets, source_nodes=np.concatenate(sn) if sn else np.empty(0, np.int64),
                choice_offsets=coff, choice_group=cg, choice_target=ct)


def native_lookup(keys, values, wanted):
    ix = np.searchsorted(keys, wanted)
    result = np.zeros(len(wanted))
    good = ix < len(keys)
    good[good] &= keys[ix[good]] == wanted[good]
    result[good] = values[ix[good]]
    return result


def save_snapshot_frame(path, frame):
    if path.exists():
        if not pd.read_parquet(path).equals(frame.reset_index(drop=True)):
            raise ValueError("Immutable partial registry differs: " + str(path))
    else:
        write_frame(path, frame)


def save_snapshot_npz(path, **arrays):
    if path.exists():
        prior = read_npz(path)
        if set(prior) != set(arrays) or any(not np.array_equal(prior[k], v) for k, v in arrays.items()):
            raise ValueError("Immutable partial registry differs: " + str(path))
    else:
        write_npz(path, **arrays)


def retained_static_nodes(nodes, links):
    if nodes.retained.dtype != bool or nodes.retained.isna().any():
        raise ValueError("Native retained-node mask is not explicit boolean")
    retained = nodes.loc[nodes.retained].copy()
    if retained.node_id.duplicated().any(): raise ValueError("Duplicate static node")
    if not links.tail_node_id.isin(retained.node_id).all() or not links.head_node_id.isin(retained.node_id).all():
        raise ValueError("Static link touches removed or unknown node")
    return retained


def prove_centroid_exclusion(g, t, rid, terms, local_members, nodes, arcs, source, target):
    """Replay accepted oracle and identify its sole additional reachability rule."""
    centroid_terms = terms.loc[terms.terminal_type.isin(["ORIGIN_CENTROID", "DESTINATION_CENTROID"]), "terminal_index"]
    centroids = frozenset(int(v) for k in centroid_terms for v in nodes[local_members[k]])
    support = build_region_support(region_id=rid, tail=g.tail[arcs], head=g.head[arcs],
        fftt_s=g.free_time[arcs] * 60., arc_ids=g.arc_idx_to_directed_arc_id[arcs],
        region_arc_mask=np.ones(len(arcs), bool))
    oracle = LazyRegionalPairOracle(suite_id=t.phase2_suite_id, candidate_id=t.candidate_id,
        support=support, terminals=terms,
        fine_members_global={t.terminal_ids[k]: tuple(int(v) for v in nodes[local_members[k]]) for k in local_members},
        physical_arc_ids=g.arc_idx_to_directed_arc_id[arcs], centroid_global_nodes=centroids)
    result = oracle.query(t.terminal_ids[source])
    if t.terminal_ids[target] in {p.target_terminal_id for p in result}:
        raise ValueError("Adapter omitted an accepted pair")
    oracle.centroid_global_nodes = frozenset()
    oracle.cache.clear()
    unrestricted = oracle.query(t.terminal_ids[source])
    matches = [p for p in unrestricted if p.target_terminal_id == t.terminal_ids[target]]
    if len(matches) != 1: raise ValueError("Unknown positive LOW pair has no proved centroid exclusion")
    p = matches[0]
    pathnodes = [p.source_node_idx] + [int(g.head[arcs[a]]) for a in p.arc_indices]
    through = [int(v) for v in pathnodes[1:-1] if v in centroids]
    if not through: raise ValueError("Diagnostic route lacks prohibited centroid transit")
    return dict(region_id=rid, coarse_key=(int(source) << 32) | int(target),
        source_terminal_id=t.terminal_ids[source], target_terminal_id=t.terminal_ids[target],
        original_oracle_excludes=True, directional_roles_valid=True,
        unrestricted_route_for_diagnostic_only=True, path_arc_ids=list(p.arc_ids),
        prohibited_transit_node_uids=[str(g.node_idx_to_uid[v]) for v in through],
        policy="LEDGER_AND_QUANTIFY_ONLY_WHEN_EXCLUSION_IS_PROVED_BY_ACCEPTED_SUPPORT")


def build(c):
    root = Path(c["output_root"])
    if (root / "registry/REGISTRY_COMPLETE.json").exists():
        raise FileExistsError("Registry already frozen; use exact saved products")
    with Monitor(c, "P1_ACTIVE_REGISTRIES") as monitor:
        g, t, frames, sources, p4 = load_native(c)
        state, low = read_tier(c, g, t, sources, "low_benchmark")
        terminals = frames["consolidated_terminals"].copy()
        members = frames["terminal_fine_members"].copy()
        terminal_ids = pd.Index(t.terminal_ids)
        terminals["terminal_index"] = terminal_ids.get_indexer(terminals.terminal_id)
        members["terminal_index"] = terminal_ids.get_indexer(members.terminal_id)
        members["global_node_index"] = pd.Index(g.node_idx_to_uid.astype(str)).get_indexer(members.partC_rev1_node_uid.astype(str))
        if (members.global_node_index < 0).any(): raise ValueError("Unknown fine member")
        terminal_groups = terminals.groupby("region_id", sort=True).indices
        member_groups = members.groupby("region_id", sort=True).indices
        fine_by_terminal = {int(k): v.global_node_index.to_numpy(np.int64)
                            for k, v in members.groupby("terminal_index", sort=True)}
        fine_role = pd.read_parquet(root / "validation/low_benchmark_native_role_reconciliation.parquet")
        fine_role_groups = fine_role.groupby("region_index", sort=True).indices
        summaries = frames["regional_pair_support_summary"].set_index("region_id")
        connectors = frames["terminal_connectors"]
        source_roles = set(connectors.loc[connectors.direction_role.eq("parent_to_region"), "head_node_id"])
        target_roles = set(connectors.loc[connectors.direction_role.eq("region_to_parent"), "tail_node_id"])
        source_roles.update(terminals.loc[terminals.terminal_type.eq("ORIGIN_CENTROID"), "terminal_id"])
        target_roles.update(terminals.loc[terminals.terminal_type.eq("DESTINATION_CENTROID"), "terminal_id"])
        if set(terminals.loc[terminals.can_exit_region, "terminal_id"]) != source_roles:
            raise ValueError("Source roles do not equal directional connectors/centroids")
        if set(terminals.loc[terminals.can_enter_region, "terminal_id"]) != target_roles:
            raise ValueError("Target roles do not equal directional connectors/centroids")
        ck = low["current__coarse_regional_pair_flow__keys"]
        cv = low["current__coarse_regional_pair_flow__values"]
        csource = (ck >> np.uint64(32)).astype(np.int64)
        ctarget = (ck & np.uint64(0xffffffff)).astype(np.int64)
        if not np.array_equal(t.terminal_region_local[csource], t.terminal_region_local[ctarget]):
            raise ValueError("Coarse trace crosses regional owners")
        trace_region = t.terminal_region_local[csource]
        active_parts = []; excluded_parts = []; outside_parts = []; region_rows = []; ownership_parts = []; exclusion_proofs = []
        resumed = 0
        for ri, rid in enumerate(t.region_ids):
            monitor.check()
            dest = region_dir(c, rid); dest.mkdir(parents=True, exist_ok=True)
            resume = (dest / "descriptor.json").exists()
            native = read_npz(t.root / "regional_support" / (rid + ".npz"))
            nodes = native["node_ids"].astype(np.int64)
            arc = native["arc_index"].astype(np.int64)
            tail = np.repeat(np.arange(len(nodes), dtype=np.int64), np.diff(native["indptr"]))
            if not np.array_equal(nodes[tail], g.tail[arc]) or not np.array_equal(nodes[native["head_local"]], g.head[arc]):
                raise ValueError("Local CSR changes physical topology")
            if not np.allclose(native["cost_s"], g.free_time[arc] * 60, rtol=2e-15, atol=1e-10):
                raise ValueError("Local freeflow native units mismatch")
            terms = terminals.iloc[terminal_groups[rid]].sort_values("terminal_id")
            local_members = {}
            for row in terms.itertuples():
                values = fine_by_terminal[row.terminal_index]
                local = np.searchsorted(nodes, values)
                if (local >= len(nodes)).any() or not np.array_equal(nodes[local], values):
                    raise ValueError("Fine terminal outside owned graph")
                local_members[row.terminal_index] = np.sort(local).astype(np.int64)
            centroid_terms = terms.loc[terms.terminal_type.isin(["ORIGIN_CENTROID", "DESTINATION_CENTROID"]), "terminal_index"]
            blocked = np.zeros(len(nodes), bool)
            for k in centroid_terms: blocked[local_members[k]] = True
            source_indices = terms.loc[terms.can_exit_region, "terminal_index"].to_numpy(np.int64)
            target_indices = terms.loc[terms.can_enter_region, "terminal_index"].to_numpy(np.int64)
            # Enumerate accepted directional reachability, never assume a Cartesian graph.
            choices = make_choices(source_indices, source_indices, local_members, source_ids=source_indices)
            ds, _, _ = K.trees(native["indptr"], native["head_local"], native["cost_s"], blocked,
                                choices["source_offsets"], choices["source_nodes"])
            full_source = []; full_target = []
            for si, sid in enumerate(source_indices):
                eligible = [int(tid) for tid in target_indices
                            if sid != tid and np.isfinite(ds[si, local_members[tid]]).any()]
                full_source.extend([int(sid)] * len(eligible)); full_target.extend(eligible)
            fs = np.asarray(full_source, np.int64); ft = np.asarray(full_target, np.int64)
            expected_D = int(summaries.loc[rid, "supported_pair_count"])
            if len(fs) != expected_D:
                raise ValueError(f"Exact accepted support count mismatch {rid}: {len(fs)} != {expected_D}")
            keys = (fs.astype(np.uint64) << np.uint64(32)) | ft.astype(np.uint64)
            values = native_lookup(ck, cv, keys); mask = values > 0; m = int(mask.sum())
            current_region_keys = ck[trace_region == ri]
            missing = np.setdiff1d(current_region_keys, keys)
            outside_s = (missing >> np.uint64(32)).astype(np.int64)
            outside_t = (missing & np.uint64(0xffffffff)).astype(np.int64)
            reasons = []
            for sid, tid in zip(outside_s, outside_t):
                if sid == tid:
                    reasons.append("OUTSIDE_COARSE_INTERFACE_SAME_TERMINAL")
                else:
                    if sid not in source_indices or tid not in target_indices:
                        raise ValueError("Positive trace has wrong directional role")
                    exclusion_proofs.append(prove_centroid_exclusion(g, t, rid, terms, local_members, nodes, arc, int(sid), int(tid)))
                    reasons.append("OUTSIDE_COARSE_INTERFACE_ACCEPTED_CENTROID_NONTRANSIT")
            outside = pd.DataFrame({"region_id": rid, "region_index": ri, "coarse_key": missing,
                "amount": native_lookup(ck, cv, missing), "reason": reasons})
            outside_parts.append(outside)
            full_ids = [stable_object_id("regional_pair", suite_id=t.phase2_suite_id, candidate_id=t.candidate_id,
                members=[t.terminal_ids[s], t.terminal_ids[v]], ordered_members=True,
                signature={"region_id": rid, "cost_version": "direct_fftt_v1"}) for s, v in zip(fs, ft)]
            full = pd.DataFrame({"region_id": rid, "region_index": ri, "full_coordinate": np.arange(len(fs)),
                "pair_id": full_ids, "coarse_key": keys, "source_terminal_id": np.asarray(t.terminal_ids)[fs],
                "target_terminal_id": np.asarray(t.terminal_ids)[ft], "source_terminal_index": fs,
                "target_terminal_index": ft, "LOW_CURRENT": values, "runtime_connected": mask})
            full["active_coordinate"] = np.where(mask, np.cumsum(mask) - 1, -1)
            full["source_benchmark"] = "physical_mu030_low_rg1e3"; full["source_flow_role"] = "CURRENT_X45"
            selected = full.loc[mask].copy(); excluded = full.loc[~mask].copy()
            excluded["reason"] = "UNOBSERVED_IN_LOW_CURRENT"
            fiber_hash = arrays_hash({str(k): v for k, v in local_members.items()})
            selected["fine_fiber_registry_hash"] = fiber_hash
            support_hash = arrays_hash({"full_keys": keys, "active_mask": mask, "LOW_CURRENT": values})
            a = dict(tail=tail, head=native["head_local"].astype(np.int64), indptr=native["indptr"].astype(np.int64),
                     t0=g.free_time[arc] * 60., capacity=state.used_capacity[arc], alpha=g.alpha[arc], beta=g.beta[arc],
                     blocked_through=blocked, global_node_index=nodes, global_arc_index=arc,
                     physical_node_id=g.node_idx_to_uid[nodes], physical_arc_id=g.arc_idx_to_directed_arc_id[arc])
            a.update(make_choices(fs[mask], ft[mask], local_members))
            graph_hash = arrays_hash({k: a[k] for k in ["tail", "head", "indptr", "t0", "capacity", "alpha", "beta", "blocked_through", "global_node_index", "global_arc_index"]})
            model_hash = digest({"model": MODEL_CLASS, "graph": graph_hash, "fibers": fiber_hash,
                                 "active_support": support_hash, "source_interface": t.directory_digest})
            selected["model_hash"] = model_hash; selected["support_hash"] = support_hash
            save_snapshot_frame(dest / "full_pair_audit.parquet", full)
            save_snapshot_frame(dest / "active_pairs.parquet", selected)
            save_snapshot_npz(dest / "local_graph.npz", **a)
            save_snapshot_npz(dest / "support_map.npz", full_keys=keys, active_indices=np.flatnonzero(mask),
                      full_to_active=full.active_coordinate.to_numpy(np.int64), LOW_ACTIVE=values[mask])
            member_ids = np.asarray(sorted(local_members), np.int64)
            save_snapshot_npz(dest / "fine_members.npz", terminal_indices=member_ids,
                offsets=np.r_[0, np.cumsum([len(local_members[k]) for k in member_ids])],
                local_nodes=np.concatenate([local_members[k] for k in member_ids]) if len(member_ids) else np.empty(0, np.int64))
            source_fine = fine_role.iloc[fine_role_groups.get(ri, [])].copy()
            save_snapshot_frame(dest / "LOW_native_fine_trace.parquet", source_fine)
            save_snapshot_npz(dest / "LOW_mapped_physical.npz", z=state.current_flow[arc])
            desc = dict(region_id=rid, region_index=ri, D_full=len(fs), m_active=m,
                n_full=len(terms), S_full=len(source_indices), T_full=len(target_indices),
                n_active=len(set(fs[mask]) | set(ft[mask])), S_active=len(np.unique(fs[mask])),
                T_active=len(np.unique(ft[mask])), physical_nodes=len(nodes), physical_arcs=len(arc),
                P_proxy=proxy(m), G_target=quota_target(m), hidden_width=32, hidden_layers=2,
                architecture_budget_only=True, phase7_architecture_frozen=False, train_required=m > 0,
                region_status="ACTIVE_BASELINE_PENDING" if m else "NO_ACTIVE_LOW_PAIRS",
                model_class=MODEL_CLASS, model_hash=model_hash, support_hash=support_hash, graph_hash=graph_hash,
                fiber_hash=fiber_hash, low_exposure=float(values.sum()),
                outside_LOW_exposure=float(outside.amount.sum()), outside_LOW_keys=len(outside),
                source_interface_hash=t.directory_digest, local_optimizer_calls=0 if not m else None,
                source_snapshot_sha256=sha(t.root / "regional_support" / (rid + ".npz")))
            if resume:
                if read_json(dest / "descriptor.json") != desc: raise ValueError("Immutable descriptor differs")
                resumed += 1
            else:
                write_json(dest / "descriptor.json", desc)
            region_rows.append(desc)
            active_parts.append(selected); excluded_parts.append(excluded)
            ownership_parts.append(pd.DataFrame({"region_id": rid, "region_index": ri, "physical_arc_index": arc,
                "physical_arc_id": g.arc_idx_to_directed_arc_id[arc]}))
            if (ri + 1) % 25 == 0:
                event(root / "control/progress.jsonl", {"stage": "P1", "regions": ri + 1,
                      "active_pairs": sum(r["m_active"] for r in region_rows)})
                print(f"P1 regions={ri + 1}/{len(t.region_ids)}", flush=True)
        active = pd.concat(active_parts, ignore_index=True); excluded = pd.concat(excluded_parts, ignore_index=True)
        regions = pd.DataFrame(region_rows); outside = pd.concat(outside_parts, ignore_index=True)
        owned = pd.concat(ownership_parts, ignore_index=True)
        if len(regions) != 1847 or len(active) + len(excluded) != 888787 or len(owned) != 327757:
            raise ValueError("All-region source closure failed")
        if owned.physical_arc_index.duplicated().any() or active.pair_id.duplicated().any():
            raise ValueError("Duplicate physical ownership or active pair")
        if len(active) + len(outside) != len(ck): raise ValueError("LOW included/outside key closure failed")
        save_snapshot_frame(root / "registry/regions.parquet", regions)
        save_snapshot_frame(root / "registry/modeled_active_pairs.parquet", active)
        save_snapshot_frame(root / "registry/excluded_zero_low_pairs.parquet", excluded)
        save_snapshot_frame(root / "registry/outside_interface_trace_ledger.parquet", outside)
        save_snapshot_frame(root / "registry/owned_arcs.parquet", owned)
        save_snapshot_frame(root / "registry/coarse_terminals.parquet", terminals)
        save_snapshot_frame(root / "registry/terminal_fine_members.parquet", members)
        proof_record = dict(
            proofs=exclusion_proofs, count=len(exclusion_proofs), no_source_or_support_change=True,
            original_oracle_source_sha256=sha(Path(__file__).parents[1] / "phase3a/regional_pairs.py"),
            fixed_fine_comparator="EXACT_INCLUDED_NATIVE_FINE_DEMANDS_ONLY; outside demands retained in ledger")
        proofpath = root / "registry/accepted_centroid_exclusion_proofs.json"
        if proofpath.exists():
            if read_json(proofpath) != proof_record: raise ValueError("Exclusion proof changed")
        else: write_json(proofpath, proof_record)
        static_root = Path(c["input_versions"]["phase5_source_root"]) / "suites/s03"
        links = pd.read_parquet(static_root / "runtime_links.parquet")
        source_nodes = pd.read_parquet(static_root / "suite_nodes.parquet")
        nodes = retained_static_nodes(source_nodes, links)
        counts = read_json(static_root / "counts.json")
        static_bindings = {p.name: {"absolute_path": str(p), "size_bytes": p.stat().st_size, "sha256": sha(p)}
                           for p in [static_root / "runtime_links.parquet", static_root / "suite_nodes.parquet",
                                     static_root / "runtime_arrays.npz", static_root / "regional_interface_contract.json"]}
        if len(links) != 143603 or len(nodes) != 72724:
            raise ValueError("Frozen Phase 5 static node/link counts differ")
        write_json(root / "registry/runtime_adjacency.json", dict(model_class=MODEL_CLASS,
            active_pairs_path="modeled_active_pairs.parquet", active_pairs_sha256=sha(root / "registry/modeled_active_pairs.parquet"),
            static_bindings=static_bindings, static_counts=counts, static_links=len(links), online_nodes=len(nodes),
            native_node_filter="suite_nodes.retained == true", original_node_rows=len(source_nodes),
            regional_logical_links=len(active), total_logical_links=143603 + len(active),
            excluded_pair_policy="NO_EDGE_NO_FALLBACK", per_sample_reselection=False,
            adjacency_hash=digest({"active": arrays_hash({"keys": active.coarse_key.to_numpy()}), "static": static_bindings})))
        record = dict(status="PASS", regions=1847, active_regions=int((regions.m_active > 0).sum()),
            inactive_regions=int((regions.m_active == 0).sum()), full_conceptual_pairs=888787,
            active_pairs=len(active), excluded_zero_low_pairs=len(excluded), outside_low_keys=len(outside),
            native_low_keys=len(ck), regional_physical_arcs=327757, logical_links=143603 + len(active),
            source_registry_hash=arrays_hash({"active_keys": active.coarse_key.to_numpy()}),
            LOW_included_exposure=float(active.LOW_CURRENT.sum()), LOW_outside_exposure=float(outside.amount.sum()),
            all_no_active_optimizer_calls_zero=True, graph_and_fibers_preserved=True,
            resumed_immutable_region_snapshots=resumed, proved_centroid_exclusions=len(exclusion_proofs))
        write_json(root / "registry/REGISTRY_COMPLETE.json", record)
        return record
