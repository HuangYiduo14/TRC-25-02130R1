"""Bounded LOW trace/support diagnostic; does not authorize or publish labels."""
from pathlib import Path
import numpy as np
import pandas as pd

from . import kernels as K
from .common import Monitor, read_npz, write_frame, write_json, sha
from .inputs import load_native, read_tier
from .registry import make_choices, native_lookup


def classify_pair(same_terminal, accepted, source_allowed, target_allowed,
                  unrestricted_reachable):
    if not source_allowed or not target_allowed:
        return "WRONG_DIRECTIONAL_ROLE"
    if same_terminal:
        return "ACCEPTED_SAME_TERMINAL_EXCLUSION"
    if accepted:
        return "IN_ACCEPTED_SUPPORT"
    if unrestricted_reachable:
        return "ACCEPTED_CENTROID_NONTRANSIT_EXCLUSION"
    return "NO_OWNED_PHYSICAL_PATH"


def audit(c):
    root = Path(c["output_root"])
    out = root / "validation/source_support_audit"
    out.mkdir(parents=True, exist_ok=True)
    with Monitor(c, "P1_SOURCE_SUPPORT_AUDIT") as monitor:
        g, t, frames, sources, p4 = load_native(c)
        state, low = read_tier(c, g, t, sources, "low_benchmark")
        terms = frames["consolidated_terminals"].copy()
        members = frames["terminal_fine_members"].copy()
        ids = pd.Index(t.terminal_ids)
        terms["terminal_index"] = ids.get_indexer(terms.terminal_id)
        members["terminal_index"] = ids.get_indexer(members.terminal_id)
        members["global_node_index"] = pd.Index(g.node_idx_to_uid.astype(str)).get_indexer(members.partC_rev1_node_uid.astype(str))
        if (members.global_node_index < 0).any(): raise ValueError("Unknown member")
        tg = terms.groupby("region_id", sort=True).indices
        byterm = {int(k): v.global_node_index.to_numpy(np.int64) for k, v in members.groupby("terminal_index")}
        fine = pd.read_parquet(root / "validation/low_benchmark_native_role_reconciliation.parquet")
        fg = fine.groupby("region_index").indices
        ck = low["current__coarse_regional_pair_flow__keys"]
        cv = low["current__coarse_regional_pair_flow__values"]
        cs = (ck >> np.uint64(32)).astype(np.int64)
        ct = (ck & np.uint64(0xffffffff)).astype(np.int64)
        cr = t.terminal_region_local[cs]
        if not np.array_equal(cr, t.terminal_region_local[ct]): raise ValueError("Cross-region trace")
        expected = frames["regional_pair_support_summary"].set_index("region_id")
        coarse_rows = []; region_rows = []; fine_parts = []; witnesses = []; mismatches = []
        for ri, rid in enumerate(t.region_ids):
            monitor.check()
            a = read_npz(t.root / "regional_support" / (rid + ".npz"))
            nodes = a["node_ids"].astype(np.int64)
            arcs = a["arc_index"].astype(np.int64)
            tr = terms.iloc[tg[rid]].sort_values("terminal_id")
            lm = {int(row.terminal_index): np.searchsorted(nodes, byterm[row.terminal_index]).astype(np.int64)
                  for row in tr.itertuples()}
            for k, v in lm.items():
                if (v >= len(nodes)).any() or not np.array_equal(nodes[v], byterm[k]): raise ValueError("Member mapping")
            sids = tr.loc[tr.can_exit_region, "terminal_index"].to_numpy(np.int64)
            tids = tr.loc[tr.can_enter_region, "terminal_index"].to_numpy(np.int64)
            sset, tset = set(sids), set(tids)
            cen = tr.loc[tr.terminal_type.isin(["ORIGIN_CENTROID", "DESTINATION_CENTROID"]), "terminal_index"]
            blocked = np.zeros(len(nodes), bool)
            for k in cen: blocked[lm[k]] = True
            ch = make_choices(sids, sids, lm, source_ids=sids)
            ds, _, _ = K.trees(a["indptr"], a["head_local"], a["cost_s"], blocked, ch["source_offsets"], ch["source_nodes"])
            accepted = set()
            for si, sid in enumerate(sids):
                accepted.update((int(sid) << 32) | int(tid) for tid in tids
                                if sid != tid and np.isfinite(ds[si, lm[tid]]).any())
            if len(accepted) != int(expected.loc[rid, "supported_pair_count"]):
                mismatches.append({"region_id": rid, "actual": len(accepted), "expected": int(expected.loc[rid, "supported_pair_count"])})
            rows = np.flatnonzero(cr == ri)
            missing = [j for j in rows if int(ck[j]) not in accepted and cs[j] != ct[j]]
            unres = None
            if missing:
                unres, _, _ = K.trees(a["indptr"], a["head_local"], a["cost_s"], np.zeros(len(nodes), bool), ch["source_offsets"], ch["source_nodes"])
            srcindex = {int(s): i for i, s in enumerate(sids)}
            for j in rows:
                source, target, key = int(cs[j]), int(ct[j]), int(ck[j])
                reachable = bool(unres is not None and source in srcindex and np.isfinite(unres[srcindex[source], lm[target]]).any())
                reason = classify_pair(source == target, key in accepted, source in sset, target in tset, reachable)
                row = {"region_id": rid, "region_index": ri, "coarse_key": key,
                       "source_terminal_id": t.terminal_ids[source], "target_terminal_id": t.terminal_ids[target],
                       "amount": float(cv[j]), "classification": reason,
                       "accepted": key in accepted, "source_role_allowed": source in sset, "target_role_allowed": target in tset}
                coarse_rows.append(row)
            fr = fine.iloc[fg.get(ri, [])].copy()
            fr["region_id"] = rid
            fr["coarse_pair_accepted"] = np.isin(fr.coarse_key.to_numpy(np.uint64), np.array(sorted(accepted), np.uint64))
            fr["fine_path_allowed"] = False
            fr["unrestricted_path_exists"] = False
            if len(fr):
                src = np.searchsorted(nodes, fr.source_global_node.to_numpy(np.int64))
                dst = np.searchsorted(nodes, fr.target_global_node.to_numpy(np.int64))
                if (src >= len(nodes)).any() or (dst >= len(nodes)).any(): raise ValueError("Fine node outside region")
                uniques, inv = np.unique(src, return_inverse=True)
                fd, fp, _ = K.trees(a["indptr"], a["head_local"], a["cost_s"], blocked,
                                    np.arange(len(uniques) + 1, dtype=np.int64), uniques)
                allowed = np.isfinite(fd[inv, dst])
                fr["fine_path_allowed"] = allowed
                fr["unrestricted_path_exists"] = allowed
                if (~allowed).any():
                    ud, up, _ = K.trees(a["indptr"], a["head_local"], a["cost_s"], np.zeros(len(nodes), bool),
                                        np.arange(len(uniques) + 1, dtype=np.int64), uniques)
                    fr["unrestricted_path_exists"] = np.isfinite(ud[inv, dst])
                    tails = np.repeat(np.arange(len(nodes)), np.diff(a["indptr"]))
                    for j in np.flatnonzero(~allowed):
                        if not np.isfinite(ud[inv[j], dst[j]]): continue
                        route = []; cursor = int(dst[j]); localnodes = [cursor]
                        while cursor != src[j]:
                            arc = int(up[inv[j], cursor])
                            if arc < 0 or len(route) >= len(nodes): raise ValueError("Diagnostic path error")
                            route.append(arc); cursor = int(tails[arc]); localnodes.append(cursor)
                        route.reverse(); localnodes.reverse()
                        transit = [v for v in localnodes[1:-1] if blocked[v]]
                        if not transit: raise ValueError("Nontransit failure lacks centroid witness")
                        witnesses.append({"region_id": rid, "fine_key": int(fr.iloc[j].fine_key),
                            "coarse_key": int(fr.iloc[j].coarse_key), "amount": float(fr.iloc[j].amount),
                            "coarse_pair_accepted": bool(fr.iloc[j].coarse_pair_accepted),
                            "source_node_uid": str(g.node_idx_to_uid[nodes[src[j]]]),
                            "target_node_uid": str(g.node_idx_to_uid[nodes[dst[j]]]),
                            "unrestricted_fftt_s": float(ud[inv[j], dst[j]]),
                            "path_arc_ids": [str(v) for v in g.arc_idx_to_directed_arc_id[arcs[route]]],
                            "path_node_uids": [str(v) for v in g.node_idx_to_uid[nodes[localnodes]]],
                            "prohibited_transit_node_uids": [str(v) for v in g.node_idx_to_uid[nodes[transit]]],
                            "regional_source_sha256": sha(t.root / "regional_support" / (rid + ".npz"))})
            fine_parts.append(fr)
            active = sum(int(ck[j]) in accepted for j in rows)
            region_rows.append({"region_id": rid, "region_index": ri, "D_full": len(accepted), "m_active": active,
                "physical_nodes": len(nodes), "physical_arcs": len(arcs), "native_low_keys": len(rows),
                "outside_low_keys": len(rows) - active, "unreachable_fine_rows": int((~fr.fine_path_allowed & fr.amount.gt(0)).sum()),
                "included_unreachable_fine_rows": int((fr.coarse_pair_accepted & ~fr.fine_path_allowed & fr.amount.gt(0)).sum()),
                "local_optimizer_calls": 0, "ICNN_constructions": 0,
                "region_status": "AUDIT_ACTIVE_NOT_DISPATCHED" if active else "NO_ACTIVE_LOW_PAIRS"})
            if (ri + 1) % 200 == 0: print(f"SUPPORT AUDIT {ri+1}/1847", flush=True)
        coarse = pd.DataFrame(coarse_rows); regions = pd.DataFrame(region_rows); fine = pd.concat(fine_parts, ignore_index=True)
        if len(regions) != 1847 or regions.D_full.sum() != 888787 or regions.physical_arcs.sum() != 327757 or mismatches:
            raise ValueError(f"Source support closure failed: {mismatches}")
        if len(coarse) != len(ck) or not np.isclose(coarse.amount.sum(), cv.sum(), rtol=1e-13, atol=1e-9):
            raise ValueError("Native trace audit conservation failed")
        write_frame(out / "coarse_LOW_audit.parquet", coarse)
        write_frame(out / "fine_LOW_audit.parquet", fine)
        write_frame(out / "all_region_audit.parquet", regions)
        write_json(out / "centroid_nontransit_witnesses.json", witnesses)
        detail = [{"classification": str(k), "keys": len(v), "exposure": float(v.amount.sum())}
                  for k, v in coarse.groupby("classification", sort=True)]
        bad = fine.loc[~fine.fine_path_allowed & fine.amount.gt(0)]
        result = dict(status="AUDIT_COMPLETE_NO_LABELS_PUBLISHED", regions=len(regions),
            full_conceptual_pairs=int(regions.D_full.sum()), active_pairs=int(regions.m_active.sum()),
            active_regions=int(regions.m_active.gt(0).sum()), inactive_regions=int(regions.m_active.eq(0).sum()),
            owned_physical_arcs=int(regions.physical_arcs.sum()), native_LOW_keys=len(coarse),
            coarse_classifications=detail, positive_unreachable_fine_rows=len(bad),
            unreachable_fine_exposure=float(bad.amount.sum()),
            included_positive_unreachable_fine_rows=int(bad.coarse_pair_accepted.sum()),
            included_unreachable_fine_exposure=float(bad.loc[bad.coarse_pair_accepted, "amount"].sum()),
            unknown_physical_paths=int((~fine.unrestricted_path_exists & fine.amount.gt(0)).sum()),
            all_centroid_rules_preserved=True, local_optimizer_calls=0, source_arrays_modified=False,
            candidate_registry_only=True, runtime_adjacency_published=False)
        write_json(out / "RESULT.json", result)
        return result
