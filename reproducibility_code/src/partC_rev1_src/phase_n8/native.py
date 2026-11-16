"""Read-only native identities and sparse adapters for the accepted five suites."""
from __future__ import annotations

from pathlib import Path
import json
import time
import numpy as np
import pandas as pd

from partC_rev1_src.phase5.campaign_runtime import PackedRuntime, pack_runtime
from partC_rev1_src.phase5.scc_reachability import load_reachable_od_supply_cache
from .common import read_json, read_npz, write_json, save_npz, sha, digest, array_hash
from .oracle import Network, Demand, aon, conservation


SUITES = ("s02", "s03", "s04", "s05", "s06")
COUNTS = dict(s02=85155, s03=80316, s04=77444, s05=72779, s06=69947)


class Sources:
    def __init__(self, ctx):
        self.ctx = ctx
        self.repo = ctx.repo
        self.base = self.repo.parent/(self.repo.name+"_outputs")
        self.r1 = Path(ctx.c["input_versions"]["current_R1_native_root"])
        self.p5 = self.base/"phase_05/campaign_v1_0905/run_001"
        self.p3 = self.base/"phase_03/all_suite_acnhn/phase3_all_suites_geo_gc_1000m_20260831_v1_run_001"
        self.p6 = self.base/"phase_06/s03_active_0906/run_001"
        self.authority = Path(ctx.c["authority_root"])
        p = self.r1/"tables/source_bindings.parquet"
        ctx.bind(p, "R1_NATIVE_SOURCE_INDEX", ctx.c["input_versions"]["R1_source_index_sha256"])
        f = pd.read_parquet(p, columns=["native_path", "sha256"])
        self.expected = {str(Path(p).resolve()): h for p, h in zip(f.native_path, f.sha256)}
        p = self.p5/"manifests/MANIFEST.csv"
        self.bind(p, "P5_OUTPUT_MANIFEST")
        f = pd.read_csv(p, usecols=["absolute_path", "sha256"])
        for p, h in zip(f.absolute_path, f.sha256):
            key = str(Path(p).resolve())
            if key in self.expected and self.expected[key] != h:
                raise ValueError("Conflicting accepted P5/R1 source identity")
            self.expected[key] = h
        p = self.r1/"derived/INDEX.json"
        self.bind(p, "R1_DERIVED_INDEX", sha(self.authority/"references/r1/derived/INDEX.json"))
        self.derived = {v["suite_id"]: v for v in read_json(p)}
        p = self.repo/"partC_doc_icnn/phase_4_work/phase4_dual_accuracy_benchmark_execution_config.json"
        self.bind(p, "P4_ACCEPTED_INPUT_IDENTITIES")
        self.p4 = read_json(p)

    def bind(self, path, role, expected=None):
        p = Path(path).resolve(strict=True)
        return self.ctx.bind(p, role, expected or self.expected.get(str(p)))

    def frame(self, path, role, expected=None):
        self.bind(path, role, expected)
        return pd.read_parquet(path)

    def arrays(self, path, role, expected=None):
        self.bind(path, role, expected)
        return read_npz(path)

    def suite(self, sid):
        if sid not in SUITES:
            raise ValueError("Suite outside the explicit active matrix")
        started = time.perf_counter()
        p5 = self.p5/"suites"/sid
        p3 = self.p3/"suites"/sid/"parquet"
        self.bind(p3/"artifact_manifest.csv", sid+"_P3_MANIFEST")
        for row in pd.read_csv(p3/"artifact_manifest.csv").itertuples():
            self.expected[str((p3/row.relative_path).resolve())] = row.sha256
        links = self.frame(p5/"runtime_links.parquet", sid+"_PACKED_LINKS")
        nodes = self.frame(p5/"retained_nodes.parquet", sid+"_PACKED_NODES")
        a = self.arrays(p5/"runtime_arrays.npz", sid+"_PACKED_ARRAYS")
        # Exact regeneration is a validation of packing, not a compiler/model run.
        check = pack_runtime(links, nodes).arrays
        for key in a:
            if not np.array_equal(a[key], check[key], equal_nan=True):
                raise ValueError("Packed registry/array order or coefficient mismatch: " + key)
        d = self.derived[sid]
        pairs = self.frame(d["path"], sid+"_ACCEPTED_R1_ACTIVE_FFTT", d["sha256"])
        if len(pairs) != COUNTS[sid] or pairs.pair_id.duplicated().any():
            raise ValueError("Accepted coordinate coverage changed")
        if not pairs.LOW_CURRENT.gt(0).all() or not np.isfinite(pairs.c0_seconds).all() or pairs.c0_seconds.lt(0).any():
            raise ValueError("Invalid accepted active support/FFTT")
        sorted_pairs = pairs.sort_values(["region_id", "native_index"], kind="stable")
        if not np.array_equal(pairs.pair_id, sorted_pairs.pair_id):
            raise ValueError("R1 region/pair order differs; do not silently reorder")
        for _, rows in pairs.groupby("region_id", sort=False):
            if not np.array_equal(rows.native_index, np.arange(len(rows))):
                raise ValueError("Region coordinate gap")
        terms = self.frame(p3/"consolidated_terminals.parquet", sid+"_TERMINAL_ROLES")
        terms = terms.sort_values("terminal_id", kind="stable").reset_index(drop=True)
        src = pairs.source_terminal_index.to_numpy(np.int64)
        dst = pairs.target_terminal_index.to_numpy(np.int64)
        if (src == dst).any() or not terms.can_exit_region.to_numpy()[src].all() or not terms.can_enter_region.to_numpy()[dst].all():
            raise ValueError("Invalid accepted regional endpoint roles")
        if not np.array_equal(terms.terminal_id.to_numpy()[src], pairs.source_terminal_id) or not np.array_equal(terms.terminal_id.to_numpy()[dst], pairs.target_terminal_id):
            raise ValueError("Terminal order/key mismatch")
        if not np.array_equal(terms.region_id.to_numpy()[src], pairs.region_id) or not np.array_equal(terms.region_id.to_numpy()[dst], pairs.region_id):
            raise ValueError("Regional edge crosses ownership")
        ni = pd.Index(nodes.node_id)
        rt = ni.get_indexer(pairs.source_terminal_id).astype(np.int64)
        rh = ni.get_indexer(pairs.target_terminal_id).astype(np.int64)
        if min(rt.min(), rh.min()) < 0:
            raise ValueError("Regional endpoint absent from retained graph")
        blocked = nodes.terminal_type.isin(["ORIGIN_CENTROID", "DESTINATION_CENTROID"]).to_numpy()
        g = Network(np.r_[a["tail"], rt], np.r_[a["head"], rh], blocked,
                    np.r_[links.link_id.to_numpy(str), pairs.pair_id.to_numpy(str)], nodes.node_id.to_numpy(str))
        fixed = np.r_[links.low_current_flow.to_numpy(float), pairs.LOW_CURRENT.to_numpy(float)]
        return dict(suite=sid, graph=g, packed=PackedRuntime(a), packed_count=len(links),
                    nodes=nodes, links=links, pairs=pairs, terms=terms, fixed_x45=fixed,
                    support_identity=d["sha256"], load_seconds=time.perf_counter()-started)

    def original_od(self):
        inp = self.p4["input_versions"]
        root = Path(inp["accepted_aggregated_od_root"])/"derived_inputs"
        self.bind(root/"aggregated_od_metadata.json", "ACCEPTED_AGGREGATED_OD_METADATA")
        meta = read_json(root/"aggregated_od_metadata.json")
        self.bind(root/"original_to_aggregated_od_index.npz", "EXACT_ORIGINAL_ROW_MAP", meta["mapping_npz_sha256"])
        a = self.arrays(root/"aggregated_od.npz", "EXACT_FULL_AGGREGATED_OD", meta["aggregate_npz_sha256"])
        if meta["od_hash"] != inp["od_identity"] or len(a["demand"]) != meta["aggregated_row_count"]:
            raise ValueError("Original OD identity/count mismatch")
        if not np.isfinite(a["demand"]).all() or (a["demand"] <= 0).any() or np.sum(a["original_row_multiplicity"]) != inp["od_row_count"]:
            raise ValueError("Original OD values/multiplicity invalid")
        if abs(float(a["demand"].sum())-inp["total_demand"]) > 1e-10*inp["total_demand"]:
            raise ValueError("Original OD total differs")
        self.bind(inp["graph_npz"], "PHYSICAL_NODE_ORDER")
        self.bind(inp["graph_metadata_json"], "PHYSICAL_GRAPH_IDENTITY")
        with np.load(inp["graph_npz"], allow_pickle=False) as z:
            uids = z["node_idx_to_uid"].astype(str)
        cache = self.base/"phase_05/pilot_v4_0905/run_001/cache/reachable_od_supply_v2.npz"
        self.bind(cache.with_suffix(".metadata.json"), "PHYSICAL_REACHABILITY_METADATA")
        sm = read_json(cache.with_suffix(".metadata.json"))
        self.bind(cache, "PHYSICAL_REACHABILITY_CACHE", sm["cache_sha256"])
        if sm["aggregated_od_sha256"] != meta["aggregate_npz_sha256"] or sm["cross_scc_reachable_row_count"] != 0:
            raise ValueError("Inherited exact physical reachability proof mismatch")
        supply = load_reachable_od_supply_cache(cache, expected_graph_identity=inp["graph_identity"],
                    expected_od_identity=inp["od_identity"], expected_node_count=len(uids),
                    expected_row_count=len(a["demand"]))
        groups = np.repeat(np.arange(len(a["origin_group_node_idx"])), np.diff(a["origin_offsets"]))
        physical_reachable = supply.scc_labels[a["origin_group_node_idx"]][groups] == supply.scc_labels[a["destination_node_idx"]]
        if int((~physical_reachable).sum()) != supply.unreachable_row_count:
            raise ValueError("Physical reachability membership does not reproduce")
        return a, uids, physical_reachable, meta

    def endpoints(self, suite, uids, original):
        sid = suite["suite"]
        p3 = self.p3/"suites"/sid/"parquet"
        nodes = pd.Index(suite["nodes"].node_id)
        ui = pd.Index(uids)
        endpoints = np.unique(np.r_[original["origin_group_node_idx"], original["destination_node_idx"]])
        needed = np.zeros(len(uids), bool)
        needed[endpoints] = True
        fine = self.frame(p3/"terminal_fine_members.parquet", sid+"_FINE_FIBERS")
        fine = fine.merge(suite["terms"][["terminal_id", "terminal_type"]], on="terminal_id", validate="many_to_one")
        fine["physical"] = ui.get_indexer(fine.partC_rev1_node_uid.astype(str))
        if (fine.physical < 0).any():
            raise ValueError("Invalid fine-member physical ID")
        bacn = self.frame(p3/"bacn_membership.parquet", sid+"_BACN_MEMBERSHIP")
        racn = self.frame(p3/"racn_membership.parquet", sid+"_RACN_MEMBERSHIP")
        self.bind(self.repo/"partC_rev1_src/phase5_fidelity/graph.py", "INHERITED_MULTIPLE_OD_ENDPOINT_ATTACHMENT_RULE")
        port_ids, edge_ids, tails, heads, multiple = [], [], [], [], []
        maps = []
        n = len(nodes)
        for role in ["ORIGIN_CENTROID", "DESTINATION_CENTROID"]:
            candidates = {int(i): set() for i in endpoints}
            for table, uid_column, node_column in [(bacn, "partC_rev1_node_uid", "bacn_id")]:
                pidx = ui.get_indexer(table[uid_column].astype(str))
                nidx = nodes.get_indexer(table[node_column].astype(str))
                if (pidx < 0).any():
                    raise ValueError("Invalid native physical membership")
                mask = needed[pidx]
                if (nidx[mask] < 0).any():
                    raise ValueError("Protected OD B node removed by source compilation")
                for physical, target in zip(pidx[mask], nidx[mask]):
                    candidates[int(physical)].add(int(target))
            c = fine.loc[fine.terminal_type.eq(role) & needed[fine.physical.to_numpy()]]
            targets = nodes.get_indexer(c.terminal_id)
            if (targets < 0).any():
                raise ValueError("Native centroid endpoint not retained")
            for physical, target in zip(c.physical, targets):
                candidates[int(physical)].add(int(target))
            pidx = ui.get_indexer(racn.physical_portal_node_uid.astype(str))
            nidx = nodes.get_indexer(racn.racn_id)
            if (pidx < 0).any():
                raise ValueError("Invalid native R physical membership")
            for physical, target in zip(pidx[needed[pidx]], nidx[needed[pidx]]):
                if not candidates[int(physical)]:
                    if target < 0:
                        raise ValueError("Protected R endpoint not retained")
                    candidates[int(physical)].add(int(target))
            mapping = np.full(len(uids), -1, np.int64)
            for physical in endpoints:
                targets = sorted(candidates[int(physical)])
                if not targets:
                    raise ValueError("INVALID_MAPPING: original OD endpoint has no native role")
                prefix = "origin" if role == "ORIGIN_CENTROID" else "destination"
                name = "n8_od_"+prefix+":"+str(uids[physical])
                port = n+len(port_ids)
                port_ids.append(name)
                mapping[physical] = port
                if len(targets)>1:
                    multiple.append(dict(physical_uid=str(uids[physical]), role=role,
                                         native_node_ids=[str(nodes[i]) for i in targets]))
                for target in targets:
                    edge_ids.append("n8_attachment:"+prefix+":"+str(uids[physical])+":"+str(nodes[target]))
                    tails.append(port if prefix=="origin" else target)
                    heads.append(target if prefix=="origin" else port)
            maps.append(mapping)
        ports = dict(port_node_ids=np.asarray(port_ids), port_edge_ids=np.asarray(edge_ids),
                     port_tail=np.asarray(tails, np.int64), port_head=np.asarray(heads, np.int64))
        suite["multiple_endpoint_attachments"] = multiple
        attach_roles(suite, ports)
        return maps, ports


def attach_roles(suite, ports):
    """Represent original OD endpoint choice with one-way zero-cost ports.

    This does not connect native centroid roles to one another. Every attachment
    is to an exact existing member, as in the inherited fidelity endpoint rule.
    """
    g = suite["graph"]
    is_origin = suite["nodes"].terminal_type.eq("ORIGIN_CENTROID").to_numpy()
    is_destination = suite["nodes"].terminal_type.eq("DESTINATION_CENTROID").to_numpy()
    indegree = np.bincount(g.head, minlength=len(g.node_ids))
    outdegree = np.bincount(g.tail, minlength=len(g.node_ids))
    if np.any(indegree[is_origin]) or np.any(outdegree[is_destination]):
        raise ValueError("Native centroid one-way role violated before OD attachment")
    # Native origin centroids become reachable only from their own origin port.
    # Destinations have no outgoing edges except their own absorbing OD port.
    # The wrapper ports themselves cannot be traversed by another commodity.
    blocked = np.r_[np.zeros(len(g.node_ids), bool), np.ones(len(ports["port_node_ids"]), bool)]
    suite["base_graph_identity"] = g.identity
    suite["graph"] = Network(np.r_[g.tail, ports["port_tail"]], np.r_[g.head, ports["port_head"]],
        blocked, np.r_[g.edge_ids, ports["port_edge_ids"]], np.r_[g.node_ids, ports["port_node_ids"]])
    suite["fixed_x45"] = np.r_[suite["fixed_x45"], np.zeros(len(ports["port_edge_ids"]))]
    suite["od_attachment_count"] = len(ports["port_edge_ids"])


def build_coverage(ctx, sources, sid, original, uids, physical_reachable):
    start = time.time()
    s = sources.suite(sid)
    (omap, dmap), ports = sources.endpoints(s, uids, original)
    od = mapped_demand(original, omap, dmap, physical_reachable)
    zero = np.zeros(len(s["graph"].tail))
    cost = np.r_[s["packed"].evaluate(zero[:s["packed_count"]], "cost")[0], s["pairs"].c0_seconds.to_numpy(float), np.zeros(s["od_attachment_count"])]
    loaded = aon(s["graph"], od, cost, guard=ctx.guard, return_distances=True)
    represented = physical_reachable & np.isfinite(loaded["distances"])
    classes = np.where(~physical_reachable, 1, np.where(represented, 2, 3)).astype(np.uint8)
    final_od = Demand(od.origins, od.offsets, od.destinations, od.demand, represented)
    balance = final_od.balance(len(s["graph"].node_ids))
    residual = conservation(s["graph"], loaded["flow"], balance, final_od.total)
    root = ctx.root/"graphs"/sid
    root.mkdir(parents=True, exist_ok=False)
    save_npz(root/"mapping.npz", dict(origin_map=omap, destination_map=dmap, classification=classes,
              balance=balance, **ports))
    rows = []
    for code, label in [(1, "INHERITED_PHYSICAL_UNREACHABLE"), (2, "REPRESENTED_REACHABLE"),
                        (3, "ADDITIONAL_ACTIVE_GRAPH_UNREACHABLE")]:
        mask = classes == code
        rows.append(dict(suite=sid, classification=label, aggregated_od_count=int(mask.sum()),
            original_od_rows=int(original["original_row_multiplicity"][mask].sum()),
            demand=float(original["demand"][mask].sum()),
            fraction_of_full_original_demand=float(original["demand"][mask].sum()/original["demand"].sum())))
    if not represented.any() or final_od.total <= 0:
        raise ValueError("No represented original demand; no-op is failure")
    report = dict(suite=sid, graph_identity=s["graph"].identity, support_identity=s["support_identity"],
                  demand_identity=final_od.identity, mapping_sha256=sha(root/"mapping.npz"),
                  nodes=len(s["graph"].node_ids), packed_links=s["packed_count"],
                  endpoint_attachment_links=s["od_attachment_count"],
                  multiple_native_endpoint_choices=s["multiple_endpoint_attachments"],
                  regional_links=len(s["pairs"]), active_regions=int(s["pairs"].region_id.nunique()),
                  original_origins=len(od.origins), represented_origins=int(np.count_nonzero(
                    np.add.reduceat(represented.astype(np.int64), od.offsets[:-1]))),
                  invalid_mapping_rows=0, full_original_od_coverage_ledger=True,
                  represented_demand=final_od.total, coverage=rows, conservation=residual,
                  full_loading_seconds=loaded["elapsed_seconds"], cold_load_seconds=s["load_seconds"],
                  zero_cost_full_demand_sptt=loaded["sptt"], sptt_difference=loaded["sptt_difference"],
                  started_epoch=start, elapsed_seconds=time.time()-start,
                  timing_role="FULL_ORIGIN_PROFILE_AT_SCENARIO_ZERO_INPUT", outcome_comparison=False,
                  endpoint_mapping_version="EXACT_NATIVE_ROLES_WITH_EMPTY_SELF_OD_V2",
                  mapping_file="mapping.npz")
    write_json(root/"IDENTITY_AND_COVERAGE.json", report)
    print(json.dumps(report), flush=True)
    return report


def reopen_demand(ctx, s, original):
    root = ctx.root/"graphs"/s["suite"]
    metapath = root/"IDENTITY_AND_COVERAGE_SELF_OD_V2.json"
    if not metapath.exists():
        metapath = root/"IDENTITY_AND_COVERAGE.json"
    meta = read_json(metapath)
    mapping = root/meta.get("mapping_file", "mapping.npz")
    if sha(mapping) != meta["mapping_sha256"]:
        raise ValueError("Frozen graph/mapping changed")
    a = read_npz(mapping)
    attach_roles(s, a)
    if s["graph"].identity != meta["graph_identity"]:
        raise ValueError("Frozen graph identity changed")
    od = mapped_demand(original, a["origin_map"], a["destination_map"], a["classification"] == 2)
    if od.identity != meta["demand_identity"]:
        raise ValueError("Frozen represented population changed")
    return od, a["balance"], meta


def mapped_demand(original, origin_map, destination_map, serve):
    origins = origin_map[original["origin_group_node_idx"]]
    destinations = destination_map[original["destination_node_idx"]].copy()
    # An OD with EXACTLY identical physical IDs has the empty path, just as in
    # Phase 4. This is per original row, never inferred from a collapsed B/R ID.
    # Aliasing its sink to its own source adds no edge or regional coordinate.
    for group, source in enumerate(original["origin_group_node_idx"]):
        begin, end = original["origin_offsets"][group:group+2]
        same = original["destination_node_idx"][begin:end] == source
        destinations[begin:end][same] = origins[group]
    return Demand(origins, original["origin_offsets"], destinations, original["demand"], serve)


def load_registry(sources, suite, registry_id):
    """Frozen membership and original eligibility; no score-based selection here."""
    if suite["suite"] != "s03":
        if not registry_id.startswith("R1_FFTT_"):
            raise ValueError("Only one suite has trained regional functions")
        return [], dict(registry_id=registry_id, identity=suite["support_identity"], entries=0)
    p = sources.r1/"proposals"/(registry_id+".json")
    source_copy = sources.authority/"references/r1/proposals"/(registry_id+".json")
    ph = sources.bind(p, "FROZEN_REGISTRY_"+registry_id, sha(source_copy))
    registry = read_json(p)
    pairs = suite["pairs"]
    groups = pairs.groupby("region_id", sort=False).indices
    if len(registry["rows"]) != len(groups) or {r["region_id"] for r in registry["rows"]} != set(groups):
        raise ValueError("Registry does not cover exactly the active regions")
    entries = []
    for number, row in enumerate(registry["rows"]):
        if number % 32 == 0:
            sources.ctx.guard()
        kind = row["model_kind"]
        if not kind:
            raise ValueError("Unavailable fixed registry candidate")
        indices = groups[row["region_id"]]
        native = pairs.iloc[indices]
        if len(indices) != indices[-1]-indices[0]+1:
            raise ValueError("Noncontiguous native regional vector")
        if digest(native.pair_id.tolist()) != row["pair_order_hash"] or set(native.support_hash) != {row["support_hash"]}:
            raise ValueError("Frozen pair order/support mismatch")
        path = Path(row["portable_path"])
        mf = path.parent/"manifest.json"
        sources.bind(mf, "ORIGINAL_FROZEN_CANDIDATE_MANIFEST", row["source_manifest_sha256"])
        meta = read_json(mf)
        if not meta["metrics"][kind]["eligible"] or meta["candidate_math"][kind]["status"] != "PASS":
            raise ValueError("Original math/selection ineligible entry cannot be promoted")
        if meta["pair_order_hash"] != row["pair_order_hash"] or meta["support_hash"] != row["support_hash"]:
            raise ValueError("Manifest order/support mismatch")
        binding = meta["files"][kind]
        if binding["sha256"] != row["portable_sha256"] or (mf.parents[3]/binding["path"]).resolve() != path.resolve():
            raise ValueError("Frozen model path/hash mismatch")
        a = sources.arrays(path, "EXACT_FROZEN_MODEL_"+kind, row["portable_sha256"])
        if not np.array_equal(a["c0"], native.c0_seconds.to_numpy(float)):
            raise ValueError("Model c0 is not the exact accepted ordered FFTT")
        entries.append(dict(**row, start=int(indices[0]), stop=int(indices[-1])+1, arrays=a))
    return entries, dict(registry_id=registry_id, identity=ph, entries=len(entries),
                         source_registry_unchanged=True, authorized_by="ICNN_N8R2_20260915_v1_AUTO")
