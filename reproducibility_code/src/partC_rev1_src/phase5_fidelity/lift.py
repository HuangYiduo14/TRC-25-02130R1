"""Exact state-consistent physical lifts and original-witness projections."""
from dataclasses import dataclass
import json
import numpy as np
import pandas as pd
from .graph import NONE, ARC, BUNDLE, MACRO, ANCHOR
from .kernels import internal_all_pairs, lift_dp, recover_lift, make_lift_layers


class PhysicalLifter:
    def __init__(self, b, trace, frames, links, sequences, state):
        self.b, self.t, self.frames = b, trace, frames
        self.g = b.graph
        self.cost = (b.low_physical if state == 'LOW' else b.high_physical).current_cost_s
        self.sizes = np.bincount(trace.node_to_bacn_local[trace.node_to_bacn_local >= 0], minlength=len(trace.bacn_ids))
        self.offsets = np.r_[0, np.cumsum(self.sizes * self.sizes)].astype(np.int64)
        if self.offsets[-1] * 16 > 4 * 2 ** 30:
            raise MemoryError('Internal packed all-pairs exceeds 4-GiB preflight guard')
        self.pos = np.full(len(trace.node_to_bacn_local), -1, np.int64)
        order = np.argsort(trace.node_to_bacn_local, kind='stable')
        order = order[trace.node_to_bacn_local[order] >= 0]
        self.pos[order] = np.arange(len(order)) - np.repeat(np.r_[0, np.cumsum(self.sizes)[:-1]], self.sizes)
        mask = trace.arc_internal_bacn_local >= 0
        if np.any(trace.node_to_bacn_local[self.g.tail[mask]] != trace.arc_internal_bacn_local[mask]) or np.any(trace.node_to_bacn_local[self.g.head[mask]] != trace.arc_internal_bacn_local[mask]):
            raise ValueError('Suppressed arc endpoints disagree with physical B ownership')
        self.internal, self.first = internal_all_pairs(self.g.tail, self.g.head, self.cost,
            trace.arc_internal_bacn_local, self.pos, self.sizes, self.offsets)
        idx = np.flatnonzero(trace.arc_to_bundle_local >= 0)
        groups = pd.DataFrame({'bundle': trace.arc_to_bundle_local[idx], 'arc': idx}).groupby('bundle', sort=False)
        self.members = {int(k): v.arc.to_numpy(np.int64) for k, v in groups}
        bi = {v: i for i, v in enumerate(trace.bundle_ids)}
        self.macros = {str(k): [bi[x] for x in v.sort_values('sequence_position').primitive_id]
                       for k, v in sequences.groupby('corridor_id', sort=False)}
        self.native = links.loc[~links.role.eq('TERMINAL_CONNECTOR')].reset_index(drop=True)
        order=np.argsort(trace.arc_to_bundle_local[idx],kind='stable')
        self.bundle_arcs=idx[order]
        self.bundle_ptr=np.r_[0,np.cumsum(np.bincount(trace.arc_to_bundle_local[idx],minlength=len(trace.bundle_ids)))].astype(np.int64)
        seq=sequences.copy()
        if len(seq):
            seq['native']=pd.Index(self.native.link_id).get_indexer(seq.corridor_id)
            if (seq.native<0).any():raise ValueError('Macro sequence is absent from runtime registry')
            seq=seq.sort_values(['native','sequence_position'],kind='stable')
            counts=np.bincount(seq.native.to_numpy(),minlength=len(self.native))
            self.macro_bundles=pd.Index(trace.bundle_ids).get_indexer(seq.primitive_id).astype(np.int64)
        else:
            counts=np.zeros(len(self.native),np.int64);self.macro_bundles=np.empty(0,np.int64)
        self.macro_ptr=np.r_[0,np.cumsum(counts)].astype(np.int64)

    def layers(self, graph, path, origin, destination):
        starts = [int(origin)]; ends = [int(origin)]; arcs = [-1]; ptr = [0, 1]
        def append_members(members):
            starts.extend(self.g.tail[members]); ends.extend(self.g.head[members]); arcs.extend(members)
            ptr.append(len(starts))
        def anchor(node):
            starts.append(int(node)); ends.append(int(node)); arcs.append(-1); ptr.append(len(starts))
        for a in path:
            op, ref = int(graph.op[a]), int(graph.ref[a])
            if op == ARC:
                append_members(np.array([ref], np.int64))
            elif op == BUNDLE:
                append_members(self.members[ref])
            elif op == MACRO:
                lid = str(graph.edges.iloc[a].edge_id)
                for primitive in self.macros[lid]:
                    append_members(self.members[primitive])
            elif op == ANCHOR:
                anchor(ref)
            elif op != NONE:
                raise ValueError('Unrecognized native path operation')
        anchor(destination)
        return [np.asarray(a, dtype=np.int64) for a in (ptr, starts, ends, arcs)]

    def lift(self, graph, path, origin, destination, reference_distance):
        ptr, starts, ends, arcs = make_lift_layers(path,graph.op,graph.ref,int(origin),int(destination),
            self.g.tail,self.g.head,self.bundle_ptr,self.bundle_arcs,self.macro_ptr,self.macro_bundles)
        values, parents, failure = lift_dp(ptr, starts, ends, arcs, self.cost,
            self.t.node_to_bacn_local, self.pos, self.sizes, self.offsets, self.internal)
        if failure >= 0:
            previous = np.arange(ptr[failure - 1], ptr[failure])
            previous = previous[np.isfinite(values[previous])]
            targets = np.arange(ptr[failure], ptr[failure + 1])
            from_nodes, to_nodes = np.unique(ends[previous]), np.unique(starts[targets])
            from_b, to_b = self.t.node_to_bacn_local[from_nodes], self.t.node_to_bacn_local[to_nodes]
            bad_b = sorted(set(from_b[from_b >= 0]) & set(to_b[to_b >= 0]))
            return dict(lift_status='UNLIFTABLE', D_lift_s=None, regret=None,
                failure_layer=int(failure), failed_B_ids=[self.t.bacn_ids[i] for i in bad_b],
                failure_from_physical_nodes=from_nodes.tolist(), failure_to_physical_nodes=to_nodes.tolist()), np.empty(0, np.int64)
        j = int(ptr[-2] + np.argmin(values[ptr[-2]:ptr[-1]]))
        best = float(values[j]); chosen = [j]
        while parents[j] >= 0:
            j = int(parents[j]); chosen.append(j)
        chosen = np.asarray(chosen[::-1], np.int64)
        physical = recover_lift(chosen, starts, ends, arcs, self.g.head,
            self.t.node_to_bacn_local, self.pos, self.sizes, self.offsets, self.first)
        if len(physical):
            if self.g.tail[physical[0]] != origin or self.g.head[physical[-1]] != destination or np.any(self.g.head[physical[:-1]] != self.g.tail[physical[1:]]):
                raise ValueError('Discontinuous physical lift')
        elif origin != destination:
            raise ValueError('Empty nontrivial physical lift')
        realized = float(self.cost[physical].sum())
        tolerance = 1e-7 + 1e-10 * max(1., abs(reference_distance))
        if abs(realized - best) > tolerance or realized < reference_distance - tolerance:
            raise ValueError('Physical lift cost closure or nonnegative regret failed')
        owner = self.t.arc_internal_bacn_local[physical]
        bundle = self.t.arc_to_bundle_local[physical]
        region = self.t.arc_region_local[physical]
        return dict(lift_status='VERIFIED', D_lift_s=realized,
            regret=(realized - reference_distance) / max(1., reference_distance),
            lift_B_internal_s=float(self.cost[physical[owner >= 0]].sum()),
            lift_inter_B_s=float(self.cost[physical[bundle >= 0]].sum()),
            lift_regional_s=float(self.cost[physical[region >= 0]].sum()),
            failure_layer=-1, failed_B_ids=[], failure_from_physical_nodes=[], failure_to_physical_nodes=[]), physical


def project_witness(graph, lifter, arcs, origin, destination):
    """Project every required physical arc and bridge using virtual edges only."""
    t, g = lifter.t, lifter.g
    member = graph.metadata['variant'].startswith(('MEMBER', 'RESTORED'))
    restored = set(graph.metadata['restored_B'])
    # Ordered required physical/bundle events; unrefined internal arcs collapse.
    events = []
    for a in arcs:
        a = int(a)
        if t.arc_region_local[a] >= 0:
            events.append(('edge', 'regional:' + str(g.arc_idx_to_directed_arc_id[a])))
        elif t.arc_to_bundle_local[a] >= 0:
            if member:
                events.append(('edge', 'member:' + str(g.arc_idx_to_directed_arc_id[a])))
            else:
                events.append(('bundle', (int(t.arc_to_bundle_local[a]), a)))
        elif t.bacn_ids[t.arc_internal_bacn_local[a]] in restored:
            events.append(('edge', 'restored:' + str(g.arc_idx_to_directed_arc_id[a])))
    first = {ids[0]: (lid, ids) for lid, ids in lifter.macros.items()}
    native_member = {int(r.member_arc_idx): str(r.link_id) for r in lifter.native.loc[lifter.native.role.eq('PHYSICAL_MEMBER')].itertuples()}
    required = []
    i = 0
    while i < len(events):
        kind, value = events[i]
        if kind == 'edge':
            lid = value
        else:
            bundle, arc = value
            if bundle in first:
                lid, ids = first[bundle]
                actual = events[i:i + len(ids)]
                if len(actual) != len(ids) or any(k != 'bundle' or v[0] != p for (k, v), p in zip(actual, ids)):
                    raise ValueError('Physical witness partially traverses a removed macro')
                i += len(ids) - 1
            else:
                lid = native_member.get(arc, t.bundle_ids[bundle])
        if lid not in graph.edge_index:
            raise ValueError('Projected physical link absent from hybrid: ' + lid)
        required.append(graph.edge_index[lid]); i += 1
    source = graph.node_index['od_o:' + str(g.node_idx_to_uid[origin])]
    target = graph.node_index['od_d:' + str(g.node_idx_to_uid[destination])]
    path = []; current = source
    for edge in required:
        path.extend(graph.virtual_bridge(current, int(graph.tail[edge])))
        path.append(edge); current = int(graph.head[edge])
    path.extend(graph.virtual_bridge(current, target))
    return np.asarray(path, np.int64)
