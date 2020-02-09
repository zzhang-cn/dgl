"""Neighbor sampling APIs"""

from .._ffi.function import _init_api
from .. import backend as F
from ..base import DGLError
from ..heterograph import DGLHeteroGraph
from .. import ndarray as nd
from .. import utils

__all__ = ['sample_neighbors', 'sample_neighbors_topk']

def sample_neighbors(g, nodes, fanout, edge_dir='in', p=None, replace=True):
    """Sample from the neighbors of the given nodes and return the induced subgraph.
   
    When sampling with replacement, the sampled subgraph could have parallel edges.
   
    For sampling without replace, if fanout > the number of neighbors, all the
    neighbors are sampled.

    Node/edge features are not preserved.
   
    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id tensor if
        the given graph g has only one type of nodes.
    fanout : int or list[int]
        The number of sampled neighbors for each node on each edge type. Provide a list
        to specify different fanout values for each edge type.
    edge_dir : str, optional
        Edge direction ('in' or 'out'). If is 'in', sample from in edges. Otherwise,
        sample from out edges.
    p : str, optional
        Feature name used as the probabilities associated with each neighbor of a node.
        Its shape should be compatible with a scalar edge feature tensor.
    replace : bool, optional
        If True, sample with replacement.
       
    Returns
    -------
    DGLHeteroGraph
        A sampled subgraph containing only the sampled neighbor edges from
        ``nodes``. The sampled subgraph has the same metagraph as the original
        one.
    """
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(utils.toindex(nodes[ntype]).todgltensor())
        else:
            nodes_all_types.append(nd.array([], ctx=nd.cpu()))

    if not isinstance(fanout, list):
        fanout = [int(fanout)] * len(g.etypes)
    if len(fanout) != len(g.etypes):
        raise DGLError('Fan-out must be specified for each edge type '
                       'if a list is provided.')

    if p is None:
        prob = [nd.array([], ctx=nd.cpu())] * len(g.etypes)
    else:
        prob = []
        for etype in g.canonical_etypes:
            if p in g.edges[etype].data:
                prob.append(F.zerocopy_to_dgl_ndarray(g.edges[etype].data[p]))
            else:
                prob.append(nd.array([], ctx=nd.cpu()))

    subgidx = _CAPI_DGLSampleNeighbors(g._graph, nodes_all_types, fanout,
                                       edge_dir, prob, replace)
    return DGLHeteroGraph(subgidx, g.ntypes, g.etypes)

def sample_neighbors_topk(g, nodes, k, edge_weights, edge_dir='in'):
    """Sample neighbors by top-k.

    If k > the number of neighbors, all the neighbors are sampled.

    Parameters
    ----------
    g : DGLHeteroGraph
        Full graph structure.
    nodes : tensor or dict
        Node ids to sample neighbors from. The allowed types
        are dictionary of node types to node id tensors, or simply node id
        tensor if the given graph g has only one type of nodes.
    k : int
        The K value.
    edge_weights : str
        Feature name of the weights associated with each edge. Its shape should be
        compatible with a scalar edge feature tensor.
    edge_dir : str, optional
        Edge direction ('in' or 'out'). If is 'in', sample from in edges.
        Otherwise, sample from out edges.

    Returns
    -------
    DGLGraph
        A sampled subgraph by top k criterion.
    """
    if not isinstance(nodes, dict):
        if len(g.ntypes) > 1:
            raise DGLError("Must specify node type when the graph is not homogeneous.")
        nodes = {g.ntypes[0] : nodes}
    nodes_all_types = []
    for ntype in g.ntypes:
        if ntype in nodes:
            nodes_all_types.append(utils.toindex(nodes[ntype]).todgltensor())
        else:
            nodes_all_types.append(nd.array([], ctx=nd.cpu()))

    if not isinstance(k, list):
        k = [int(k)] * len(g.etypes)
    if len(k) != len(g.etypes):
        raise DGLError('K value must be specified for each edge type '
                       'if a list is provided.')

    ew = []
    for etype in g.canonical_etypes:
        if edge_weights in g.edges[etype].data:
            ew.append(F.zerocopy_to_dgl_ndarray(g.edges[etype].data[edge_weights]))
        else:
            raise DGLError('Edge weights "{}" do not exist for relation graph "{}".'.format(
                edge_weights, etype))

    subgidx = _CAPI_DGLSampleNeighborsTopk(g._graph, nodes_all_types, k, edge_dir, ew)
    return DGLHeteroGraph(subgidx, g.ntypes, g.etypes)

_init_api('dgl.sampling.neighbor', __name__)
