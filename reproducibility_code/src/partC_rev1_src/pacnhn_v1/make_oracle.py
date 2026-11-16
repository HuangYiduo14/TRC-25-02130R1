"""One-time trace extension of the inspected immutable N8 multigraph kernel."""
from pathlib import Path

def main():
    source=Path(__file__).resolve().parents[1]/'phase_n8/oracle.py'
    text=source.read_text(encoding='utf-8')
    text=text[text.index('@njit(cache=True, nogil=True)'):text.index('\n\n@dataclass')]
    text=text.replace('def load_range(', 'def trace_range(')
    text=text.replace('serve, begin, end):','serve, begin, end, enabled, edge_regions, hash_capacity, keep_forest):')
    text=text.replace('    flow = np.zeros(len(tail))','''    flow = np.zeros(len(tail))
    suites = len(edge_regions)
    pair_keys = np.full((suites, hash_capacity), UINT64_SENTINEL, np.uint64)
    pair_values = np.zeros((suites, hash_capacity))
    forest = np.full((end-begin if keep_forest else 0, n), -2, np.int32)
    active_region = np.empty(suites, np.int64)
    exit_node = np.empty(suites, np.int64)
    traced_rows = 0
    trace_segments = np.zeros(suites, np.int64)''')
    text=text.replace('                child = head[edge]', '''                if not enabled[edge]:
                    continue
                child = head[edge]''')
    text=text.replace('        nsettled += settled_count','''        # Walk exact predecessor paths after loading. Regional traces are
        # keyed by the actual P vertices; no coarse portal selector is used.
        if keep_forest:
            forest[group-begin, source] = -1
            for k in range(settled_count):
                node = settlement[k]
                if node != source and subtree[node] > 0:
                    forest[group-begin, node] = pred[node]
        for row in range(offsets[group], offsets[group+1]):
            if not serve[row] or settled[destinations[row]] != epoch:
                continue
            traced_rows += 1
            node = destinations[row]
            amount = demand[row]
            active_region[:] = -1
            exit_node[:] = -1
            hops = 0
            while node != source:
                edge = pred[node]
                if edge < 0 or head[edge] != node or not enabled[edge]:
                    raise ValueError("Invalid traced predecessor")
                for suite in range(suites):
                    region = edge_regions[suite, edge]
                    if region != active_region[suite]:
                        if active_region[suite] >= 0:
                            key = (np.uint64(active_region[suite])*np.uint64(n)+np.uint64(node))*np.uint64(n)+np.uint64(exit_node[suite])
                            if not _hash_add(pair_keys[suite], pair_values[suite], key, amount):
                                raise RuntimeError("Trace hash capacity exhausted")
                            trace_segments[suite] += 1
                        active_region[suite] = region
                        exit_node[suite] = node
                node = tail[edge]
                hops += 1
                if hops > n:
                    raise ValueError("Predecessor cycle")
            for suite in range(suites):
                if active_region[suite] >= 0:
                    key = (np.uint64(active_region[suite])*np.uint64(n)+np.uint64(source))*np.uint64(n)+np.uint64(exit_node[suite])
                    if not _hash_add(pair_keys[suite], pair_values[suite], key, amount):
                        raise RuntimeError("Trace hash capacity exhausted")
                    trace_segments[suite] += 1
        nsettled += settled_count''')
    text=text.replace('return flow, distances, sptt, reachable, nsettled', 'return flow, distances, sptt, reachable, nsettled, pair_keys, pair_values, forest, traced_rows, trace_segments')
    header='''"""Settlement-safe P-ACNHN loader with exact regional segment traces.

Adapted from phase_n8/oracle.py, retaining audited heap/tie/settlement rules.
Only the new file is modified. Parallel arcs remain distinct coordinates.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import time
import numpy as np
from numba import njit
from partC_rev1_src.phase4.kernels import _heap_push_or_decrease, _heap_pop, _hash_add, UINT64_SENTINEL
from partC_rev1_src.phase_n8.oracle import Network, Demand, conservation
from partC_rev1_src.phase4.models import SparseFlowVector, linear_combine_sparse
from .control import save_npz, write_json, array_hash, sha

'''
    dest=Path(__file__).with_name('oracle.py')
    if dest.exists():raise FileExistsError(dest)
    dest.write_text(header+text+'\n',encoding='utf-8')
    print(dest)

if __name__=='__main__': main()
