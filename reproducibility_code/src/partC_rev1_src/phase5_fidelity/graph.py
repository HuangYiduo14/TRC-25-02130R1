"""Diagnostic graphs retain accepted link identities and detailed regional arcs."""
from dataclasses import dataclass
from collections import deque
import hashlib
import json
import numpy as np
import pandas as pd
from .kernels import shortest_tree

NONE, ARC, BUNDLE, MACRO, ANCHOR = range(5)


@dataclass
class Graph:
    nodes: np.ndarray
    edges: pd.DataFrame
    metadata: dict

    def __post_init__(self):
        self.edges = self.edges.sort_values('edge_id', kind='stable').reset_index(drop=True)
        if pd.Index(self.nodes).has_duplicates or self.edges.edge_id.duplicated().any():
            raise ValueError('Duplicate stable graph identity')
        ix = pd.Index(self.nodes)
        self.tail = ix.get_indexer(self.edges['tail'])
        self.head = ix.get_indexer(self.edges['head'])
        if (self.tail < 0).any() or (self.head < 0).any():
            raise ValueError('Dangling graph endpoint')
        self.cost = self.edges.cost_s.to_numpy(float)
        if not np.isfinite(self.cost).all() or (self.cost < 0).any():
            raise ValueError('Nonfinite/negative frozen cost')
        self.order = np.argsort(self.tail, kind='stable').astype(np.int64)
        self.indptr = np.r_[0, np.cumsum(np.bincount(self.tail, minlength=len(ix)))].astype(np.int64)
        self.node_index = {str(v): i for i, v in enumerate(self.nodes)}
        self.edge_index = {str(v): i for i, v in enumerate(self.edges.edge_id)}
        self.op = self.edges.op.to_numpy(np.int8)
        self.ref = self.edges.ref.to_numpy(np.int64)
        h = hashlib.sha256()
        for x in [self.nodes, self.edges.edge_id.to_numpy(str), self.tail, self.head,
                  self.cost, self.op, self.ref,
                  self.edges.fixed_s.to_numpy(float), self.edges.fit_delta_s.to_numpy(float),
                  self.edges.out_of_domain.to_numpy(bool), self.edges.role.to_numpy(str)]:
            h.update(np.ascontiguousarray(x).tobytes())
        h.update(json.dumps(self.metadata, sort_keys=True).encode())
        self.fingerprint = h.hexdigest()

    def tree(self, origin):
        return shortest_tree(len(self.nodes), self.indptr, self.order, self.head,
                             self.cost, int(origin))

    def path(self, pred, origin, destination):
        result = []
        node = int(destination)
        for _ in range(len(self.nodes) + 1):
            if node == origin:
                return np.asarray(result[::-1], dtype=np.int64)
            edge = int(pred[node])
            if edge < 0:
                return np.empty(0, np.int64)
            if self.head[edge] != node:
                raise ValueError('Predecessor edge does not end at child')
            result.append(edge)
            node = int(self.tail[edge])
        raise ValueError('Cyclic predecessor path')

    def virtual_bridge(self, origin, destination):
        if origin == destination:
            return []
        pred = {int(origin): -1}
        queue = deque([int(origin)])
        while queue:
            u = queue.popleft()
            for a in self.order[self.indptr[u]:self.indptr[u + 1]]:
                if self.op[a] not in (NONE, ANCHOR):
                    continue
                v = int(self.head[a])
                if v in pred:
                    continue
                pred[v] = int(a)
                if v == destination:
                    result = []
                    while v != origin:
                        e = pred[v]; result.append(e); v = int(self.tail[e])
                    return result[::-1]
                queue.append(v)
        raise ValueError(f'No directional virtual projection bridge: {self.nodes[origin]} -> {self.nodes[destination]}')


def physical_graph(b, state):
    g = b.graph
    cost = (b.low_physical if state == 'LOW' else b.high_physical).current_cost_s
    nodes = np.asarray(['p:' + str(v) for v in g.node_idx_to_uid])
    edges = pd.DataFrame(dict(edge_id=np.char.add('physical:', g.arc_idx_to_directed_arc_id.astype(str)),
        tail=nodes[g.tail], head=nodes[g.head], cost_s=cost, fixed_s=0., fit_delta_s=0.,
        op=ARC, ref=np.arange(len(g.tail)), role='PHYSICAL', out_of_domain=False))
    return Graph(nodes, edges, dict(variant='PHYSICAL', state=state, suite='ALL',
        graph_hash=g.graph_hash, units='seconds', demand_loading=False))


def build_hybrid(b, t, frames, links, sequences, charges, sample, state,
                 variant, charge_variant, shadow_cost=None, restored=()):
    """Direction-separated terminal ports prohibit unpaid fine-state switches.

    A terminal's entry port is the head of its accepted parent->region connector;
    its exit port is the tail of region->parent. Their stable semantic terminal ID
    remains in the ledger. This is a diagnostic representation, not a new overlay.
    """
    g = b.graph
    nphys = len(g.node_idx_to_uid)
    cost = (b.low_physical if state == 'LOW' else b.high_physical).current_cost_s
    restored = set(restored)
    member = variant.startswith('MEMBER') or variant.startswith('RESTORED')
    fine_r = not variant.endswith('COARSE')
    use_shadow = variant.startswith('EXACT_DYNAMIC_SHADOW')
    c = charges.set_index('bacn_id').chi_s.to_dict()
    if not restored.issubset(set(t.bacn_ids)):
        raise ValueError('Restoration outside original B registry')
    bid = np.asarray(t.bacn_ids)
    bidx = pd.Index(t.bacn_ids)
    restored_local = bidx.get_indexer(sorted(restored))
    restore_mask = np.zeros(len(t.bacn_ids), bool)
    restore_mask[restored_local] = True
    pnames = np.asarray(['p:' + str(v) for v in g.node_idx_to_uid])
    nodes = set()
    parts = []
    ledger = []

    def add(edge_id, tail, head, weights, role, op=NONE, ref=-1, fixed=0., fit=0., ood=False):
        frame = pd.DataFrame(dict(edge_id=edge_id, tail=tail, head=head, cost_s=weights,
            role=role, op=op, ref=ref, fixed_s=fixed, fit_delta_s=fit, out_of_domain=ood))
        parts.append(frame)
        nodes.update(frame['tail']); nodes.update(frame['head'])

    # Region copies are keyed by accepted region index and physical node index.
    ra = np.flatnonzero(t.arc_region_local >= 0)
    reg = t.arc_region_local[ra].astype(np.int64)
    keys = np.unique(np.r_[reg * nphys + g.tail[ra], reg * nphys + g.head[ra]])
    rnames = np.asarray(['r:' + str(int(k // nphys)) + ':' + str(g.node_idx_to_uid[int(k % nphys)]) for k in keys])
    def region_nodes(region, physical):
        k = np.asarray(region, np.int64) * nphys + np.asarray(physical, np.int64)
        ix = np.searchsorted(keys, k)
        if (ix >= len(keys)).any() or np.any(keys[ix] != k):
            raise ValueError('Terminal fine member absent from detailed regional arcs')
        return rnames[ix]
    add(np.char.add('regional:', g.arc_idx_to_directed_arc_id[ra].astype(str)),
        region_nodes(reg, g.tail[ra]), region_nodes(reg, g.head[ra]), cost[ra],
        'REGIONAL_PHYSICAL', ARC, ra)

    # Every uncompressed inter-B physical edge keeps its own CURRENT flow/cost.
    if member:
        ma = np.flatnonzero(t.arc_to_bundle_local >= 0)
        bt, bh = t.node_to_bacn_local[g.tail[ma]], t.node_to_bacn_local[g.head[ma]]
        if (bt < 0).any() or (bh < 0).any():
            raise ValueError('Inter-B member endpoint lacks B owner')
        tail = np.where(restore_mask[bt], pnames[g.tail[ma]], bid[bt])
        head = np.where(restore_mask[bh], pnames[g.head[ma]], bid[bh])
        cv = np.asarray([c[v] for v in t.bacn_ids])
        fixed = .5 * (np.where(restore_mask[bt], 0., cv[bt]) + np.where(restore_mask[bh], 0., cv[bh]))
        add(np.char.add('member:', g.arc_idx_to_directed_arc_id[ma].astype(str)), tail, head,
            cost[ma] + fixed, 'INTER_B_MEMBER', ARC, ma, fixed)
        ia = np.flatnonzero((t.arc_internal_bacn_local >= 0) &
            np.isin(t.arc_internal_bacn_local, restored_local))
        if len(ia):
            add(np.char.add('restored:', g.arc_idx_to_directed_arc_id[ia].astype(str)),
                pnames[g.tail[ia]], pnames[g.head[ia]], cost[ia], 'RESTORED_B_PHYSICAL', ARC, ia)
    else:
        native = links.loc[~links.role.eq('TERMINAL_CONNECTOR')].copy()
        q = native.low_current_flow.to_numpy(float) if state == 'LOW' else native.high_current_flow.to_numpy(float)
        dynamic = native.tau_s.to_numpy(float) + native.k4.to_numpy(float) * q ** 4
        fixed = native.structural_fixed_s.to_numpy(float) if charge_variant == 'STRUCTURAL' else (
            native.fixed_internal_H_s + native.tail_half_charge_s + native.head_half_charge_s).to_numpy(float)
        exact = dynamic.copy()
        fitmask = native.runtime_class.eq('FIT_POWER4').to_numpy()
        if shadow_cost is None and fitmask.any():
            raise ValueError('Missing offline fitted-link exact weights')
        if fitmask.any():
            exact[fitmask] = native.loc[fitmask, 'link_id'].map(shadow_cost).to_numpy(float)
        ops = np.where(native.role.eq('MACRO'), MACRO, np.where(native.role.eq('PHYSICAL_MEMBER'), ARC, BUNDLE))
        refs = np.where(ops == ARC, native.member_arc_idx.to_numpy(),
            np.where(ops == BUNDLE, pd.Index(t.bundle_ids).get_indexer(native.primitive_id), np.arange(len(native))))
        # MACRO refs are the index in this deterministic native row table.
        add(native.link_id, native.tail_node_id, native.head_node_id,
            (exact if use_shadow else dynamic) + fixed, native.role, ops, refs, fixed,
            np.zeros(len(native)) if use_shadow else dynamic - exact,
            fitmask & (q > native.domain_Q.to_numpy(float)))

    fine = frames['terminal_fine_members'].merge(frames['consolidated_terminals'], on=['candidate_id', 'region_id', 'terminal_id'], validate='many_to_one')
    fine['physical'] = pd.Index(g.node_idx_to_uid.astype(str)).get_indexer(fine.partC_rev1_node_uid.astype(str))
    fine['region'] = pd.Index(t.region_ids).get_indexer(fine.region_id.astype(str))
    if (fine.physical < 0).any() or (fine.region < 0).any():
        raise ValueError('Fine terminal key is not accepted')
    fine['rnode'] = region_nodes(fine.region, fine.physical)
    terminals = frames['consolidated_terminals'].set_index('terminal_id')
    # Original static connectors are retained unless fine R or restored B replaces them.
    connector = frames['terminal_connectors'].copy()
    entry = connector.direction_role.eq('parent_to_region')
    connector['terminal_id'] = np.where(entry, connector.head_node_id, connector.tail_node_id)
    connector['parent_id'] = connector.terminal_id.map(terminals.parent_object_id)
    types = connector.terminal_id.map(terminals.terminal_type)
    replaced = (types.eq('RACN') & fine_r) | connector.parent_id.isin(restored)
    ledger = connector[['link_id', 'terminal_id', 'parent_id', 'direction_role']].rename(columns={'link_id': 'original_link_id'})
    ledger['disposition'] = np.where(replaced, 'REPLACED_EXACT_FINE', 'PRESERVED_DIRECTIONAL_PORT')
    preserved = connector.loc[~replaced]
    entry = preserved.direction_role.eq('parent_to_region')
    port = preserved.terminal_id + np.where(entry, ':in', ':out')
    fixed = preserved.parent_id.map(c).fillna(0.).to_numpy() / 2.
    add(preserved.link_id, np.where(entry, preserved.parent_id, port),
        np.where(entry, port, preserved.parent_id), preserved.direct_base_fftt_s.to_numpy() + fixed,
        'STATIC_CONNECTOR', fixed=fixed)
    # Fine portal attachments always carry the selected physical-node anchor.
    for entry, allowed, suffix in [(True, 'can_exit_region', ':in'), (False, 'can_enter_region', ':out')]:
        active = fine.loc[fine[allowed] & fine.terminal_type.isin(['BACN', 'RACN'])].copy()
        replaced = (active.terminal_type.eq('RACN') & fine_r) | active.parent_object_id.isin(restored)
        hub = np.where(replaced, pnames[active.physical], active.terminal_id + suffix)
        rn = active.rnode.to_numpy()
        add('attach:' + active.terminal_id + suffix + ':' + active.partC_rev1_node_uid.astype(str),
            hub if entry else rn, rn if entry else hub, np.zeros(len(active)), 'FINE_ATTACHMENT',
            ANCHOR, active.physical.to_numpy())

    # OD supernodes are one-way roles, never through-centroid connectors.
    for endpoint, prefix, centroid in [('origin_idx', 'od_o:', 'ORIGIN_CENTROID'), ('destination_idx', 'od_d:', 'DESTINATION_CENTROID')]:
        for u in sorted(sample[endpoint].unique()):
            u = int(u); name = prefix + str(g.node_idx_to_uid[u]); targets = set()
            p = int(t.node_to_bacn_local[u])
            if p >= 0:
                targets.add(pnames[u] if restore_mask[p] else str(bid[p]))
            rows = fine.loc[fine.physical.eq(u) & fine.terminal_type.eq(centroid)]
            targets.update(rows.rnode)
            if not targets:
                # Non-centroid R endpoints keep their exact fine physical portal.
                rf = fine.loc[fine.physical.eq(u) & fine.terminal_type.eq('RACN')]
                if len(rf):
                    targets.add(pnames[u] if fine_r else str(rf.iloc[0].parent_object_id))
            if not targets:
                raise ValueError('Accepted OD has no authorized endpoint attachment: ' + name)
            for target in sorted(targets):
                entering = endpoint == 'origin_idx'
                add(['endpoint:' + name + ':' + target], [name if entering else target],
                    [target if entering else name], [0.], 'OD_ROLE', ANCHOR, u)
    metadata = dict(variant=variant, suite=t.suite_id, state=state, charge_variant=charge_variant,
        restored_B=sorted(restored), source_interface_hash=t.directory_digest,
        regional_treatment='DETAILED_PHYSICAL_ARCS_AT_CURRENT', units='seconds',
        R_treatment='EXACT_FINE_PORTAL' if fine_r else 'APPROVED_COARSE_PARENT',
        terminal_direction_policy='SEPARATE_ENTRY_EXIT_PORTS_NO_UNPAID_FINE_SWITCH',
        production_adoption=False, online_oracle_calls=0)
    result = Graph(np.asarray(sorted(nodes)), pd.concat(parts, ignore_index=True), metadata)
    actual_regional = result.edges.loc[result.edges.role.eq('REGIONAL_PHYSICAL'), 'ref'].to_numpy()
    if not np.array_equal(np.sort(actual_regional), ra):
        raise ValueError('Regional ownership lost or duplicated')
    if restored == set(t.bacn_ids):
        owned = result.edges.loc[result.edges.op.eq(ARC), 'ref'].to_numpy()
        if not np.array_equal(np.sort(owned), np.arange(len(g.tail))):
            raise ValueError('All-restored physical arc ownership does not close exactly once')
        if np.any(result.edges.fixed_s != 0):
            raise ValueError('Restored graph retains scalar charge')
    result.connector_ledger = ledger
    result.native_rows = None if member else native
    return result
