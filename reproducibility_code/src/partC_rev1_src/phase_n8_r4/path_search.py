"""Native N8 tie/centroid policy with a retained predecessor for bounded diagnostics."""
import numpy as np
from numba import njit
from partC_rev1_src.phase4.kernels import _heap_push_or_decrease,_heap_pop
from partC_rev1_src.phase5.campaign_runtime import recover_edge_path


@njit(cache=True,nogil=True)
def target_tree(n,head,indptr,order,ranks,blocked,cost,source,target):
    distance=np.full(n,np.inf);pred=np.full(n,-1,np.int64);settled=np.zeros(n,np.uint8)
    heapnodes=np.empty(n,np.int64);heapkeys=np.empty(n);heappos=np.full(n,-1,np.int64)
    distance[source]=0.;size=_heap_push_or_decrease(heapnodes,heapkeys,heappos,0,source,0.)
    while size:
        node,d,size=_heap_pop(heapnodes,heapkeys,heappos,size)
        if settled[node]:continue
        settled[node]=1
        if node==target:break
        if blocked[node] and node!=source:continue
        for k in range(indptr[node],indptr[node+1]):
            edge=order[k];child=head[edge]
            if settled[child]:continue
            candidate=d+cost[edge]
            if candidate<distance[child]:
                distance[child]=candidate;pred[child]=edge
                size=_heap_push_or_decrease(heapnodes,heapkeys,heappos,size,child,candidate)
            elif candidate==distance[child] and pred[child]>=0 and ranks[edge]<ranks[pred[child]]:
                pred[child]=edge
    return distance[target],pred


def query(graph,cost,origin,destination):
    c=np.asarray(cost,float)
    if c.shape!=graph.tail.shape or not np.isfinite(c).all() or (c<0).any():raise ValueError('Invalid native diagnostic costs')
    d,p=target_tree(len(graph.node_ids),graph.head,graph.indptr,graph.order,graph.ranks,graph.blocked,c,int(origin),int(destination))
    path=recover_edge_path(p,graph.tail,graph.head,origin,destination) if np.isfinite(d) else np.empty(0,np.int64)
    if np.isfinite(d) and not np.isclose(c[path].sum(),d,rtol=1e-12,atol=1e-7):raise ArithmeticError('Diagnostic predecessor path/cost mismatch')
    if len(path)>1 and np.any(graph.blocked[graph.head[path[:-1]]]):raise ValueError('Native diagnostic used forbidden centroid transit')
    return float(d),path
