"""Module for graph transformation utilities."""

from collections.abc import Iterable, Mapping
from collections import defaultdict
import numpy as np
from scipy import sparse

from ._ffi.function import _init_api
from .base import dgl_warning, DGLError
from . import convert
from .heterograph import DGLHeteroGraph, DGLBlock
from .frame import Frame
from . import ndarray as nd
from . import backend as F
from . import utils, batch
from .partition import metis_partition_assignment
from .partition import partition_graph_with_halo
from .partition import metis_partition

# TO BE DEPRECATED
from ._deprecate.graph import DGLGraph as DGLGraphStale

__all__ = [
    'line_graph',
    'khop_adj',
    'khop_graph',
    'reverse',
    'to_bidirected',
    'to_bidirected_stale',
    'add_reverse_edges',
    'laplacian_lambda_max',
    'knn_graph',
    'segmented_knn_graph',
    'add_edges',
    'add_nodes',
    'remove_edges',
    'remove_nodes',
    'add_self_loop',
    'remove_self_loop',
    'metapath_reachable_graph',
    'compact_graphs',
    'to_block',
    'to_simple',
    'to_simple_graph',
    'as_immutable_graph',
    'metis_partition_assignment',
    'partition_graph_with_halo',
    'metis_partition',
    'as_heterograph']


def pairwise_squared_distance(x):
    """
    x : (n_samples, n_points, dims)
    return : (n_samples, n_points, n_points)
    """
    x2s = F.sum(x * x, -1, True)
    # assuming that __matmul__ is always implemented (true for PyTorch, MXNet and Chainer)
    return x2s + F.swapaxes(x2s, -1, -2) - 2 * x @ F.swapaxes(x, -1, -2)

#pylint: disable=invalid-name
def knn_graph(x, k, algorithm='topk'):
    """Construct a graph from a set of points according to k-nearest-neighbor (KNN)
    and return.

    The function transforms the coordinates/features of a point set
    into a directed homogeneous graph. The coordinates of the point
    set is specified as a matrix whose rows correspond to points and
    columns correspond to coordinate/feature dimensions.

    The nodes of the returned graph correspond to the points, where the predecessors
    of each point are its k-nearest neighbors measured by the Euclidean distance.

    If :attr:`x` is a 3D tensor, then each submatrix will be transformed
    into a separate graph. DGL then composes the graphs into a large
    graph of multiple connected components.

    Parameters
    ----------
    x : Tensor
        The point coordinates. It can be either on CPU or GPU.

        * If is 2D, ``x[i]`` corresponds to the i-th node in the KNN graph.

        * If is 3D, ``x[i]`` corresponds to the i-th KNN graph and
          ``x[i][j]`` corresponds to the j-th node in the i-th KNN graph.
    k : int
        The number of nearest neighbors per node.
    algorithm : str, optional
        Algorithm used to compute the k-nearest neighbors.

        * 'topk' will use topk algorithm (quick-select or sorting,
          depending on backend implementation)
        * 'kd-tree' will use kd-tree algorithm (only on cpu)

        (default: 'topk')

    Returns
    -------
    DGLGraph
        The constructred graph. The node IDs are in the same order as :attr:`x`.

        The returned graph is on CPU, regardless of the context of input :attr:`x`.

    Examples
    --------

    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch

    When :attr:`x` is a 2D tensor, a single KNN graph is constructed.

    >>> x = torch.tensor([[0.0, 0.0, 1.0],
    ...                   [1.0, 0.5, 0.5],
    ...                   [0.5, 0.2, 0.2],
    ...                   [0.3, 0.2, 0.4]])
    >>> knn_g = dgl.knn_graph(x, 2)  # Each node has two predecessors
    >>> knn_g.edges()
    >>> (tensor([0, 1, 2, 2, 2, 3, 3, 3]), tensor([0, 1, 1, 2, 3, 0, 2, 3]))

    When :attr:`x` is a 3D tensor, DGL constructs multiple KNN graphs and
    and then composes them into a graph of multiple connected components.

    >>> x1 = torch.tensor([[0.0, 0.0, 1.0],
    ...                    [1.0, 0.5, 0.5],
    ...                    [0.5, 0.2, 0.2],
    ...                    [0.3, 0.2, 0.4]])
    >>> x2 = torch.tensor([[0.0, 1.0, 1.0],
    ...                    [0.3, 0.3, 0.3],
    ...                    [0.4, 0.4, 1.0],
    ...                    [0.3, 0.8, 0.2]])
    >>> x = torch.stack([x1, x2], dim=0)
    >>> knn_g = dgl.knn_graph(x, 2)  # Each node has two predecessors
    >>> knn_g.edges()
    (tensor([0, 1, 2, 2, 2, 3, 3, 3, 4, 5, 5, 5, 6, 6, 7, 7]),
     tensor([0, 1, 1, 2, 3, 0, 2, 3, 4, 5, 6, 7, 4, 6, 5, 7]))
    """
    if algorithm == 'topk':
        return _knn_graph_topk(x, k)
    else:
        if F.ndim(x) == 3:
            x_size = tuple(F.shape(x))
            x = F.reshape(x, (x_size[0] * x_size[1], x_size[2]))
            x_seg = x_size[0] * [x_size[1]]
        else:
            x_seg = [F.shape(x)[0]]
        out = knn(x, x_seg, x, x_seg, k, algorithm=algorithm)
        row, col = out[1], out[0]
        return convert.graph((row, col))

def _knn_graph_topk(x, k):
    """Construct a graph from a set of points according to k-nearest-neighbor (KNN)
    via topk method.

    Parameters
    ----------
    x : Tensor
        The point coordinates. It can be either on CPU or GPU.

        * If is 2D, ``x[i]`` corresponds to the i-th node in the KNN graph.

        * If is 3D, ``x[i]`` corresponds to the i-th KNN graph and
          ``x[i][j]`` corresponds to the j-th node in the i-th KNN graph.
    k : int
        The number of nearest neighbors per node.
    """
    if F.ndim(x) == 2:
        x = F.unsqueeze(x, 0)
    n_samples, n_points, _ = F.shape(x)

    dist = pairwise_squared_distance(x)
    k_indices = F.argtopk(dist, k, 2, descending=False)
    dst = F.copy_to(k_indices, F.cpu())

    src = F.zeros_like(dst) + F.reshape(F.arange(0, n_points), (1, -1, 1))

    per_sample_offset = F.reshape(F.arange(0, n_samples) * n_points, (-1, 1, 1))
    dst += per_sample_offset
    src += per_sample_offset
    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix(
        (F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))),
        shape=(n_samples * n_points, n_samples * n_points))
    return convert.from_scipy(adj)

#pylint: disable=invalid-name
def segmented_knn_graph(x, k, segs, algorithm='topk'):
    """Construct multiple graphs from multiple sets of points according to
    k-nearest-neighbor (KNN) and return.

    Compared with :func:`dgl.knn_graph`, this allows multiple point sets with
    different capacity. The points from different sets are stored contiguously
    in the :attr:`x` tensor.
    :attr:`segs` specifies the number of points in each point set. The
    function constructs a KNN graph for each point set, where the predecessors
    of each point are its k-nearest neighbors measured by the Euclidean distance.
    DGL then composes all KNN graphs
    into a graph with multiple connected components.

    Parameters
    ----------
    x : Tensor
        Coordinates/features of points. Must be 2D. It can be either on CPU or GPU.
    k : int
        The number of nearest neighbors per node.
    segs : list[int]
        Number of points in each point set. The numbers in :attr:`segs`
        must sum up to the number of rows in :attr:`x`.
    algorithm : str, optional
        Algorithm used to compute the k-nearest neighbors.

        * 'topk' will use topk algorithm (quick-select or sorting,
          depending on backend implementation)
        * 'kd-tree' will use kd-tree algorithm (only on cpu)

        (default: 'topk')

    Returns
    -------
    DGLGraph
        The graph. The node IDs are in the same order as :attr:`x`.

        The returned graph is on CPU, regardless of the context of input :attr:`x`.

    Examples
    --------

    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch

    In the example below, the first point set has three points
    and the second point set has four points.

    >>> # Features/coordinates of the first point set
    >>> x1 = torch.tensor([[0.0, 0.5, 0.2],
    ...                    [0.1, 0.3, 0.2],
    ...                    [0.4, 0.2, 0.2]])
    >>> # Features/coordinates of the second point set
    >>> x2 = torch.tensor([[0.3, 0.2, 0.1],
    ...                    [0.5, 0.2, 0.3],
    ...                    [0.1, 0.1, 0.2],
    ...                    [0.6, 0.3, 0.3]])
    >>> x = torch.cat([x1, x2], dim=0)
    >>> segs = [x1.shape[0], x2.shape[0]]
    >>> knn_g = dgl.segmented_knn_graph(x, 2, segs)
    >>> knn_g.edges()
    (tensor([0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6]),
     tensor([0, 1, 0, 1, 2, 2, 3, 5, 4, 6, 3, 5, 4, 6]))
    """
    if algorithm == 'topk':
        return _segmented_knn_graph_topk(x, k, segs)
    else:
        out = knn(x, segs, x, segs, k, algorithm=algorithm)
        row, col = out[1], out[0]
        return convert.graph((row, col))

def _segmented_knn_graph_topk(x, k, segs):
    """Construct multiple graphs from multiple sets of points according to
    k-nearest-neighbor (KNN) via topk method.

    Parameters
    ----------
    x : Tensor
        Coordinates/features of points. Must be 2D. It can be either on CPU or GPU.
    k : int
        The number of nearest neighbors per node.
    segs : list[int]
        Number of points in each point set. The numbers in :attr:`segs`
        must sum up to the number of rows in :attr:`x`.
    """
    n_total_points, _ = F.shape(x)
    offset = np.insert(np.cumsum(segs), 0, 0)

    h_list = F.split(x, segs, 0)
    dst = [
        F.argtopk(pairwise_squared_distance(h_g), k, 1, descending=False) +
        int(offset[i])
        for i, h_g in enumerate(h_list)]
    dst = F.cat(dst, 0)
    src = F.arange(0, n_total_points).unsqueeze(1).expand(n_total_points, k)

    dst = F.reshape(dst, (-1,))
    src = F.reshape(src, (-1,))
    adj = sparse.csr_matrix((F.asnumpy(F.zeros_like(dst) + 1), (F.asnumpy(dst), F.asnumpy(src))))

    return convert.from_scipy(adj)

def knn(x, x_segs, y, y_segs, k, algorithm='kd-tree', dist='euclidean'):
    r"""For each element in each segment in :attr:`y`, find :attr:`k` nearest
    points in the same segment in :attr:`x`.

    This function allows multiple point sets with different capacity. The points
    from different sets are stored contiguously in the :attr:`x` and :attr:`y` tensor.
    :attr:`x_segs` and :attr:`y_segs` specifies the number of points in each point set.

    Parameters
    ----------
    x : Tensor
        The point coordinates in x. It can be either on CPU or GPU (must be the
        same as :attr:`y`). Must be 2D.
    x_segs : Union[List[int], Tensor]
        Number of points in each point set in :attr:`x`. The numbers in :attr:`x_segs`
        must sum up to the number of rows in :attr:`x`.
    y : Tensor
        The point coordinates in y. It can be either on CPU or GPU (must be the
        same as :attr:`x`). Must be 2D.
    y_segs : Union[List[int], Tensor]
        Number of points in each point set in :attr:`y`. The numbers in :attr:`y_segs`
        must sum up to the number of rows in :attr:`y`.
    k : int
        The number of nearest neighbors per node.
    algorithm : str, optional
        Algorithm used to compute the k-nearest neighbors.
        Currently only cpu version kdtree is supported.
        (default: 'kd-tree')
    dist : str, optional
        The distance metric used to compute distance between points. It can be the following
        metrics:
        * 'euclidean': Use Euclidean distance (L2 norm) :math:`\sqrt{\sum_{i} (x_{i} - y_{i})^{2}}`.
        * 'cosine': Use cosine distance.
        (Default: "euclidean")

    Returns
    -------
    Tensor
        Tensor with size `(2, k * num_points(y))`
        The first subtensor contains point indexs in :attr:`y`. The second subtensor contains
        point indexs in :attr:`x`
    """
    # currently only cpu implementation is supported.
    if (F.context(x) != F.cpu() or F.context(y) != F.cpu()):
        dgl_warning("Currently only cpu implementation is supported," \
            "copy input tensors to cpu.")
    x = F.copy_to(x, F.cpu())
    y = F.copy_to(y, F.cpu())
    if isinstance(x_segs, (tuple, list)):
        x_segs = F.tensor(x_segs)
    if isinstance(y_segs, (tuple, list)):
        y_segs = F.tensor(y_segs)
    x_segs = F.copy_to(x_segs, F.context(x))
    y_segs = F.copy_to(y_segs, F.context(y))

    # supported algorithms
    algorithm_list = ['kd-tree']
    if algorithm not in algorithm_list:
        raise DGLError("only {} algorithms are supported, get '{}'".format(
            algorithm_list, algorithm))

    # k must less than or equal to min(x_segs)
    if k > F.min(x_segs, dim=0):
        raise DGLError("'k' must be less than or equal to the number of points in 'x'"
                       "expect k <= {}, got k = {}".format(F.min(x_segs, dim=0), k))
    dist = dist.lower()
    dist_metric_list = ['euclidean', 'cosine']
    if dist not in dist_metric_list:
        raise DGLError('Only {} are supported for distance'
                       'computation, got {}'.format(dist_metric_list, dist))

    x_offset = F.zeros((F.shape(x_segs)[0] + 1,), F.dtype(x_segs), F.context(x_segs))
    x_offset[1:] = F.cumsum(x_segs, dim=0)
    y_offset = F.zeros((F.shape(y_segs)[0] + 1,), F.dtype(y_segs), F.context(y_segs))
    y_offset[1:] = F.cumsum(y_segs, dim=0)

    out = F.zeros((2, F.shape(y)[0] * k), F.dtype(x_segs), F.context(x_segs))

    # if use cosine distance, normalize input points first
    # thus we can use euclidean distance to find knn equivalently.
    if dist == 'cosine':
        l2_norm = lambda v: F.sqrt(F.sum(v * v, dim=1, keepdims=True))
        x = x / (l2_norm(x) + 1e-5)
        y = y / (l2_norm(y) + 1e-5)

    _CAPI_DGLKNN(F.to_dgl_nd(x), F.to_dgl_nd(x_offset),
                 F.to_dgl_nd(y), F.to_dgl_nd(y_offset),
                 k, F.zerocopy_to_dgl_ndarray_for_write(out),
                 algorithm)
    return out

def to_bidirected(g, copy_ndata=False, readonly=None):
    r"""Convert the graph to a bi-directional simple graph and return.

    For an input graph :math:`G`, return a new graph :math:`G'` such that an edge
    :math:`(u, v)\in G'` if and only if there exists an edge :math:`(u, v)\in G` or
    an edge :math:`(v, u)\in G`. The resulting graph :math:`G'` is a simple graph,
    meaning there is no parallel edge.

    The operation only works for edges whose two endpoints belong to the same node type.
    DGL will raise error if the input graph is heterogeneous and contains edges
    with different types of endpoints.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    copy_ndata: bool, optional
        If True, the node features of the bidirected graph are copied from the
        original graph. If False, the bidirected graph will not have any node features.
        (Default: False)
    readonly : bool
        **DEPRECATED**.

    Returns
    -------
    DGLGraph
        The bidirected graph

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following examples use PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 0])))
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 2, 1, 2, 0]), tensor([1, 2, 0, 0, 1, 2]))

    The graph already have i->j and j->i

    >>> g = dgl.graph((th.tensor([0, 1, 2, 0]), th.tensor([1, 2, 0, 2])))
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges()
    (tensor([0, 1, 2, 1, 2, 0]), tensor([1, 2, 0, 0, 1, 2]))

    **Heterogeneous graphs with Multiple Edge Types**

    >>> g = dgl.heterograph({
    ...     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2]), th.tensor([1, 1, 2, 0])),
    ...     ('user', 'follows', 'user'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    ... })
    >>> bg1 = dgl.to_bidirected(g)
    >>> bg1.edges(etype='wins')
    (tensor([0, 0, 1, 1, 2, 2]), tensor([1, 2, 0, 2, 0, 1]))
    >>> bg1.edges(etype='follows')
    (tensor([1, 1, 2]), tensor([1, 2, 1]))
    """
    if readonly is not None:
        dgl_warning("Parameter readonly is deprecated" \
            "There will be no difference between readonly and non-readonly DGLGraph")

    for c_etype in g.canonical_etypes:
        if c_etype[0] != c_etype[2]:
            assert False, "to_bidirected is not well defined for " \
                "unidirectional bipartite graphs" \
                ", but {} is unidirectional bipartite".format(c_etype)

    assert g.is_multigraph is False, "to_bidirected only support simple graph"

    g = add_reverse_edges(g, copy_ndata=copy_ndata, copy_edata=False)
    g = to_simple(g, return_counts=None, copy_ndata=copy_ndata, copy_edata=False)
    return g

def add_reverse_edges(g, readonly=None, copy_ndata=True,
                      copy_edata=False, ignore_bipartite=False):
    r"""Add an reversed edge for each edge in the input graph and return a new graph.

    For a graph with edges :math:`(i_1, j_1), \cdots, (i_n, j_n)`, this
    function creates a new graph with edges
    :math:`(i_1, j_1), \cdots, (i_n, j_n), (j_1, i_1), \cdots, (j_n, i_n)`.

    The operation only works for edges whose two endpoints belong to the same node type.
    DGL will raise error if the input graph is heterogeneous and contains edges
    with different types of endpoints. If :attr:`ignore_bipartite` is true, DGL will
    ignore those edges instead.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    readonly : bool, default to be True
        Deprecated. There will be no difference between readonly and non-readonly
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from
        the original graph. If False, the new graph will not have any
        node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the features of the reversed edges will be identical to
        the original ones."

        If False, the new graph will not have any edge features.

        (Default: False)
    ignore_bipartite: bool, optional
        If True, unidirectional bipartite graphs are ignored and
        no error is raised. If False, an error  will be raised if
        an edge type of the input heterogeneous graph is for a unidirectional
        bipartite graph.

    Returns
    -------
    DGLGraph
        The graph with reversed edges added.

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs. On the contrary, the two graphs do not share
    the same edge feature storage.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    **Homogeneous graphs**

    >>> g = dgl.graph((th.tensor([0, 0]), th.tensor([0, 1])))
    >>> bg1 = dgl.add_reverse_edges(g)
    >>> bg1.edges()
    (tensor([0, 0, 0, 1]), tensor([0, 1, 0, 0]))

    **Heterogeneous graphs**

    >>> g = dgl.heterograph({
    >>>     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2, 2]), th.tensor([1, 1, 2, 1, 0])),
    >>>     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1])),
    >>>     ('user', 'follows', 'user'): (th.tensor([1, 2, 1), th.tensor([0, 0, 0]))
    >>> })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['wins'].data['h'] = th.tensor([0, 1, 2, 3, 4])

    The :func:`add_reverse_edges` operation is applied to the edge type
    ``('user', 'wins', 'user')`` and the edge type ``('user', 'follows', 'user')``.
    The edge type ``('user', 'plays', 'game')`` is ignored.  Both the node features and
    edge features are shared.

    >>> bg = dgl.add_reverse_edges(g, copy_ndata=True,
                               copy_edata=True, ignore_bipartite=True)
    >>> bg.edges(('user', 'wins', 'user'))
    (tensor([0, 2, 0, 2, 2, 1, 1, 2, 1, 0]), tensor([1, 1, 2, 1, 0, 0, 2, 0, 2, 2]))
    >>> bg.edges(('user', 'follows', 'user'))
    (tensor([1, 2, 1, 0, 0, 0]), tensor([0, 0, 0, 1, 2, 1]))
    >>> bg.edges(('user', 'plays', 'game'))
    (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    >>> bg.nodes['game'].data['hv']
    tensor([0, 0, 0])
    >>> bg.edges[('user', 'wins', 'user')].data['h']
    th.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
    """
    if readonly is not None:
        dgl_warning("Parameter readonly is deprecated" \
            "There will be no difference between readonly and non-readonly DGLGraph")

    # get node cnt for each ntype
    num_nodes_dict = {}
    for ntype in g.ntypes:
        num_nodes_dict[ntype] = g.number_of_nodes(ntype)

    canonical_etypes = g.canonical_etypes
    num_nodes_dict = {ntype: g.number_of_nodes(ntype) for ntype in g.ntypes}
    # fast path
    if ignore_bipartite is False:
        subgs = {}
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                assert False, "add_reverse_edges is not well defined for " \
                    "unidirectional bipartite graphs" \
                    ", but {} is unidirectional bipartite".format(c_etype)

            u, v = g.edges(form='uv', order='eid', etype=c_etype)
            subgs[c_etype] = (F.cat([u, v], dim=0), F.cat([v, u], dim=0))

        new_g = convert.heterograph(subgs, num_nodes_dict=num_nodes_dict)
    else:
        subgs = {}
        for c_etype in canonical_etypes:
            if c_etype[0] != c_etype[2]:
                u, v = g.edges(form='uv', order='eid', etype=c_etype)
                subgs[c_etype] = (u, v)
            else:
                u, v = g.edges(form='uv', order='eid', etype=c_etype)
                subgs[c_etype] = (F.cat([u, v], dim=0), F.cat([v, u], dim=0))

        new_g = convert.heterograph(subgs, num_nodes_dict=num_nodes_dict)

    # handle features
    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(new_g, node_frames=node_frames)

    if copy_edata:
        # find indices
        eids = []
        for c_etype in canonical_etypes:
            eid = F.copy_to(F.arange(0, g.number_of_edges(c_etype)), new_g.device)
            if c_etype[0] != c_etype[2]:
                eids.append(eid)
            else:
                eids.append(F.cat([eid, eid], 0))

        edge_frames = utils.extract_edge_subframes(g, eids)
        utils.set_new_frames(new_g, edge_frames=edge_frames)

    return new_g

def line_graph(g, backtracking=True, shared=False):
    """Return the line graph of this graph.

    The line graph ``L(G)`` of a given graph ``G`` is defined as another graph where
    the nodes in ``L(G)`` maps to the edges in ``G``.  For any pair of edges ``(u, v)``
    and ``(v, w)`` in ``G``, the corresponding node of edge ``(u, v)`` in ``L(G)`` will
    have an edge connecting to the corresponding node of edge ``(v, w)``.

    Parameters
    ----------
    g : DGLGraph
        Input graph.  Must be homogeneous.
    backtracking : bool, optional
        If False, the line graph node corresponding to edge ``(u, v)`` will not have
        an edge connecting to the line graph node corresponding to edge ``(v, u)``.

        Default: True.
    shared : bool, optional
        Whether to copy the edge features of the original graph as the node features
        of the result line graph.

    Returns
    -------
    G : DGLGraph
        The line graph of this graph.

    Notes
    -----
    * If :attr:`shared` is True, the node features of the resulting graph share the same
      storage with the edge features of the input graph. Hence, users should try to
      avoid in-place operations which will be visible to both graphs.
    * The function supports input graph on GPU but copies it to CPU during computation.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------
    Assume that the graph has the following adjacency matrix: ::

       A = [[0, 0, 1],
            [1, 0, 1],
            [1, 1, 0]]

    >>> g = dgl.graph(([0, 1, 1, 2, 2],[2, 0, 2, 0, 1]), 'user', 'follows')
    >>> lg = g.line_graph()
    >>> lg
    Graph(num_nodes=5, num_edges=8,
    ndata_schemes={}
    edata_schemes={})
    >>> lg.edges()
    (tensor([0, 0, 1, 2, 2, 3, 4, 4]), tensor([3, 4, 0, 3, 4, 0, 1, 2]))
    >>> lg = g.line_graph(backtracking=False)
    >>> lg
    Graph(num_nodes=5, num_edges=4,
    ndata_schemes={}
    edata_schemes={})
    >>> lg.edges()
    (tensor([0, 1, 2, 4]), tensor([4, 0, 3, 1]))
    """
    assert g.is_homogeneous, \
        'only homogeneous graph is supported'

    dev = g.device
    lg = DGLHeteroGraph(_CAPI_DGLHeteroLineGraph(g._graph.copy_to(nd.cpu()), backtracking))
    lg = lg.to(dev)
    if shared:
        new_frames = utils.extract_edge_subframes(g, None)
        utils.set_new_frames(lg, node_frames=new_frames)

    return lg

DGLHeteroGraph.line_graph = utils.alias_func(line_graph)

def khop_adj(g, k):
    """Return the matrix of :math:`A^k` where :math:`A` is the adjacency matrix of the graph
    :math:`g`.

    The returned matrix is a 32-bit float dense matrix on CPU. The graph must be homogeneous.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    k : int
        The :math:`k` in :math:`A^k`.

    Returns
    -------
    Tensor
        The returned tensor.

    Examples
    --------
    >>> import dgl
    >>> g = dgl.graph(([0,1,2,3,4,0,1,2,3,4], [0,1,2,3,4,1,2,3,4,0]))
    >>> dgl.khop_adj(g, 1)
    tensor([[1., 0., 0., 0., 1.],
            [1., 1., 0., 0., 0.],
            [0., 1., 1., 0., 0.],
            [0., 0., 1., 1., 0.],
            [0., 0., 0., 1., 1.]])
    >>> dgl.khop_adj(g, 3)
    tensor([[1., 0., 1., 3., 3.],
            [3., 1., 0., 1., 3.],
            [3., 3., 1., 0., 1.],
            [1., 3., 3., 1., 0.],
            [0., 1., 3., 3., 1.]])
    """
    assert g.is_homogeneous, \
        'only homogeneous graph is supported'
    adj_k = g.adj(scipy_fmt=g.formats()['created'][0]) ** k
    return F.tensor(adj_k.todense().astype(np.float32))

def khop_graph(g, k, copy_ndata=True):
    """Return the graph whose edges connect the :attr:`k`-hop neighbors of the original graph.

    More specifically, an edge from node ``u`` and node ``v`` exists in the new graph if
    and only if a path with length :attr:`k` exists from node ``u`` to node ``v`` in the
    original graph.

    The adjacency matrix of the returned graph is :math:`A^k`
    (where :math:`A` is the adjacency matrix of :math:`g`).

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    k : int
        The :math:`k` in `k`-hop graph.
    copy_ndata: bool, optional
        If True, the node features of the new graph are copied from the
        original graph.

        If False, the new graph will not have any node features.

        (Default: True)

    Returns
    -------
    DGLGraph
        The returned graph.

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------

    Below gives an easy example:

    >>> import dgl
    >>> g = dgl.graph(([0, 1], [1, 2]))
    >>> g_2 = dgl.transform.khop_graph(g, 2)
    >>> print(g_2.edges())
    (tensor([0]), tensor([2]))

    A more complicated example:

    >>> import dgl
    >>> g = dgl.graph(([0,1,2,3,4,0,1,2,3,4], [0,1,2,3,4,1,2,3,4,0]))
    >>> dgl.khop_graph(g, 1)
    DGLGraph(num_nodes=5, num_edges=10,
             ndata_schemes={}
             edata_schemes={})
    >>> dgl.khop_graph(g, 3)
    DGLGraph(num_nodes=5, num_edges=40,
             ndata_schemes={}
             edata_schemes={})
    """
    assert g.is_homogeneous, \
        'only homogeneous graph is supported'
    n = g.number_of_nodes()
    adj_k = g.adj(transpose=True, scipy_fmt=g.formats()['created'][0]) ** k
    adj_k = adj_k.tocoo()
    multiplicity = adj_k.data
    row = np.repeat(adj_k.row, multiplicity)
    col = np.repeat(adj_k.col, multiplicity)
    # TODO(zihao): we should support creating multi-graph from scipy sparse matrix
    # in the future.
    new_g = convert.graph((row, col), num_nodes=n)

    # handle ndata
    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(new_g, node_frames=node_frames)

    return new_g

def reverse(g, copy_ndata=True, copy_edata=False, *, share_ndata=None, share_edata=None):
    r"""Return a new graph with every edges being the reverse ones in the input graph.

    The reverse (also called converse, transpose) of a graph with edges
    :math:`(i_1, j_1), (i_2, j_2), \cdots` of type ``(U, E, V)`` is a new graph with edges
    :math:`(j_1, i_1), (j_2, i_2), \cdots` of type ``(V, E, U)``.

    The returned graph shares the data structure with the original graph, i.e. dgl.reverse
    will not create extra storage for the reversed graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    copy_ndata: bool, optional
        If True, the node features of the reversed graph are copied from the
        original graph. If False, the reversed graph will not have any node features.
        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the reversed graph are copied from the
        original graph. If False, the reversed graph will not have any edge features.
        (Default: False)

    Return
    ------
    DGLGraph
        The reversed graph.

    Notes
    -----
    If :attr:`copy_ndata` or :attr:`copy_edata` is True,
    the resulting graph will share the node or edge feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    **Homogeneous graphs**

    Create a graph to reverse.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 0])))
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.]])

    Reverse the graph.

    >>> rg = dgl.reverse(g, copy_edata=True)
    >>> rg.ndata['h']
    tensor([[0.],
            [1.],
            [2.]])

    The i-th edge in the reversed graph corresponds to the i-th edge in the
    original graph. When :attr:`copy_edata` is True, they have the same features.

    >>> rg.edges()
    (tensor([1, 2, 0]), tensor([0, 1, 2]))
    >>> rg.edata['h']
    tensor([[3.],
            [4.],
            [5.]])

    **Heterogenenous graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (th.tensor([0, 2]), th.tensor([1, 2])),
    ...     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    ... })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['plays'].data['he'] = th.zeros(3, 1)

    The resulting graph will have edge types
    ``('user', 'follows', 'user)`` and ``('game', 'plays', 'user')``.

    >>> rg = dgl.reverse(g, copy_ndata=True)
    >>> rg
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'follows', 'user'): 2, ('game', 'plays', 'user'): 3},
          metagraph=[('user', 'user'), ('game', 'user')])
    >>> rg.edges(etype='follows')
    (tensor([1, 2]), tensor([0, 2]))
    >>> rg.edges(etype='plays')
    (tensor([2, 1, 1]), tensor([1, 2, 1]))
    >>> rg.nodes['game'].data['hv']
    tensor([[1.],
            [1.],
            [1.]])
    >>> rg.edges['plays'].data
    {}
    """
    if share_ndata is not None:
        dgl_warning('share_ndata argument has been renamed to copy_ndata.')
        copy_ndata = share_ndata
    if share_edata is not None:
        dgl_warning('share_edata argument has been renamed to copy_edata.')
        copy_edata = share_edata
    if g.is_block:
        # TODO(0.5 release, xiangsx) need to handle BLOCK
        # currently reversing a block results in undefined behavior
        raise DGLError('Reversing a block graph is not supported.')
    gidx = g._graph.reverse()
    new_g = DGLHeteroGraph(gidx, g.ntypes, g.etypes)

    # handle ndata
    if copy_ndata:
        # for each ntype
        for ntype in g.ntypes:
            new_g.nodes[ntype].data.update(g.nodes[ntype].data)

    # handle edata
    if copy_edata:
        # for each etype
        for utype, etype, vtype in g.canonical_etypes:
            new_g.edges[vtype, etype, utype].data.update(
                g.edges[utype, etype, vtype].data)

    return new_g

DGLHeteroGraph.reverse = utils.alias_func(reverse)

def to_simple_graph(g):
    """Convert the graph to a simple graph with no multi-edge.

    DEPRECATED: renamed to dgl.to_simple

    Parameters
    ----------
    g : DGLGraph
        The input graph.

    Returns
    -------
    DGLGraph
        A simple graph.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.
    """
    dgl_warning('dgl.to_simple_graph is renamed to dgl.to_simple in v0.5.')
    return to_simple(g)

def to_bidirected_stale(g, readonly=True):
    """NOTE: this function only works on the deprecated
    :class:`dgl.DGLGraphStale` object.

    Convert the graph to a bidirected graph.

    The function generates a new graph with no node/edge feature.
    If g has an edge for ``(u, v)`` but no edge for ``(v, u)``, then the
    returned graph will have both ``(u, v)`` and ``(v, u)``.

    If the input graph is a multigraph (there are multiple edges from node u to node v),
    the returned graph isn't well defined.

    Parameters
    ----------
    g : DGLGraphStale
        The input graph.
    readonly : bool
        Whether the returned bidirected graph is readonly or not.

        (Default: True)

    Notes
    -----
    Please make sure g is a simple graph, otherwise the return value is undefined.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Returns
    -------
    DGLGraph

    Examples
    --------
    The following two examples use PyTorch backend, one for non-multi graph
    and one for multi-graph.

    >>> g = dgl._deprecate.graph.DGLGraph()
    >>> g.add_nodes(2)
    >>> g.add_edges([0, 0], [0, 1])
    >>> bg1 = dgl.to_bidirected_stale(g)
    >>> bg1.edges()
    (tensor([0, 1, 0]), tensor([0, 0, 1]))
    """
    if readonly:
        newgidx = _CAPI_DGLToBidirectedImmutableGraph(g._graph)
    else:
        newgidx = _CAPI_DGLToBidirectedMutableGraph(g._graph)
    return DGLGraphStale(newgidx)

def laplacian_lambda_max(g):
    """Return the largest eigenvalue of the normalized symmetric Laplacian of a graph.

    If the graph is batched from multiple graphs, return the list of the largest eigenvalue
    for each graph instead.

    Parameters
    ----------
    g : DGLGraph
        The input graph, it must be a bi-directed homogeneous graph, i.e., every edge
        should have an accompanied reverse edge in the graph.
        The graph can be batched from multiple graphs.

    Returns
    -------
    list[float]
        A list where the i-th item indicates the largest eigenvalue
        of i-th graph in :attr:`g`.

        In the case where the function takes a single graph, it will return a list
        consisting of a single element.

    Examples
    --------
    >>> import dgl
    >>> g = dgl.graph(([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], [1, 2, 3, 4, 0, 4, 0, 1, 2, 3]))
    >>> dgl.laplacian_lambda_max(g)
    [1.809016994374948]
    """
    g_arr = batch.unbatch(g)
    rst = []
    for g_i in g_arr:
        n = g_i.number_of_nodes()
        adj = g_i.adj(scipy_fmt=g_i.formats()['created'][0]).astype(float)
        norm = sparse.diags(F.asnumpy(g_i.in_degrees()).clip(1) ** -0.5, dtype=float)
        laplacian = sparse.eye(n) - norm * adj * norm
        rst.append(sparse.linalg.eigs(laplacian, 1, which='LM',
                                      return_eigenvectors=False)[0].real)
    return rst

def metapath_reachable_graph(g, metapath):
    """Return a graph where the successors of any node ``u`` are nodes reachable from ``u`` by
    the given metapath.

    If the beginning node type ``s`` and ending node type ``t`` are the same, it will return
    a homogeneous graph with node type ``s = t``.  Otherwise, a unidirectional bipartite graph
    with source node type ``s`` and destination node type ``t`` is returned.

    In both cases, two nodes ``u`` and ``v`` will be connected with an edge ``(u, v)`` if
    there exists one path matching the metapath from ``u`` to ``v``.

    The result graph keeps the node set of type ``s`` and ``t`` in the original graph even if
    they might have no neighbor.

    The features of the source/destination node type in the original graph would be copied to
    the new graph.

    Parameters
    ----------
    g : DGLGraph
        The input graph
    metapath : list[str or tuple of str]
        Metapath in the form of a list of edge types

    Returns
    -------
    DGLGraph
        A homogeneous or unidirectional bipartite graph. It will be on CPU regardless of
        whether the input graph is on CPU or GPU.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    >>> g = dgl.heterograph({
    ...     ('A', 'AB', 'B'): ([0, 1, 2], [1, 2, 3]),
    ...     ('B', 'BA', 'A'): ([1, 2, 3], [0, 1, 2])})
    >>> new_g = dgl.metapath_reachable_graph(g, ['AB', 'BA'])
    >>> new_g.edges(order='eid')
    (tensor([0, 1, 2]), tensor([0, 1, 2]))
    """
    adj = 1
    for etype in metapath:
        adj = adj * g.adj(etype=etype, scipy_fmt='csr', transpose=True)

    adj = (adj != 0).tocsr()
    srctype = g.to_canonical_etype(metapath[0])[0]
    dsttype = g.to_canonical_etype(metapath[-1])[2]
    new_g = convert.heterograph({(srctype, '_E', dsttype): adj.nonzero()},
                                {srctype: adj.shape[0], dsttype: adj.shape[1]},
                                idtype=g.idtype, device=g.device)

    # copy srcnode features
    new_g.nodes[srctype].data.update(g.nodes[srctype].data)
    # copy dstnode features
    if srctype != dsttype:
        new_g.nodes[dsttype].data.update(g.nodes[dsttype].data)

    return new_g

def add_nodes(g, num, data=None, ntype=None):
    r"""Add the given number of nodes to the graph and return a new graph.

    The new nodes will have IDs starting from ``g.num_nodes(ntype)``.

    Parameters
    ----------
    num : int
        The number of nodes to add.
    data : dict[str, Tensor], optional
        Feature data of the added nodes. The keys are feature names
        while the values are feature data.
    ntype : str, optional
        The node type name. Can be omitted if there is
        only one type of nodes in the graph.

    Return
    ------
    DGLGraph
        The graph with newly added nodes.

    Notes
    -----
    * For features in :attr:`g` but not in :attr:`data`,
      DGL assigns zero features for the newly added nodes.
    * For feature in :attr:`data` but not in :attr:`g`, DGL assigns zero features
      for the existing nodes in the graph.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> g.num_nodes()
    3
    >>> g = dgl.add_nodes(g, 2)
    >>> g.num_nodes()
    5

    If the graph has some node features and new nodes are added without
    features, their features will be filled with zeros.

    >>> g.ndata['h'] = torch.ones(5, 1)
    >>> g = dgl.add_nodes(g, 1)
    >>> g.ndata['h']
    tensor([[1.], [1.], [1.], [1.], [1.], [0.]])

    Assign features for the new nodes.

    >>> g = dgl.add_nodes(g, 1, {'h': torch.ones(1, 1), 'w': torch.ones(1, 1)})
    >>> g.ndata['h']
    tensor([[1.], [1.], [1.], [1.], [1.], [0.], [1.]])

    Since :attr:`data` contains new feature fields, the features for existing nodes
    will be filled with zeros.

    >>> g.ndata['w']
    tensor([[0.], [0.], [0.], [0.], [0.], [0.], [1.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g.num_nodes('user')
    3
    >>> g = dgl.add_nodes(g, 2, ntype='user')
    >>> g.num_nodes('user')
    5

    See Also
    --------
    remove_nodes
    add_edges
    remove_edges
    """
    g = g.clone()
    g.add_nodes(num, data=data, ntype=ntype)
    return g

def add_edges(g, u, v, data=None, etype=None):
    r"""Add the edges to the graph and return a new graph.

    The i-th new edge will be from ``u[i]`` to ``v[i]``.  The IDs of the new
    edges will start from ``g.num_edges(etype)``.

    Parameters
    ----------
    u : int, Tensor or iterable[int]
        Source node IDs, ``u[i]`` gives the source node for the i-th new edge.
    v : int, Tensor or iterable[int]
        Destination node IDs, ``v[i]`` gives the destination node for the i-th new edge.
    data : dict[str, Tensor], optional
        Feature data of the added edges. The keys are feature names
        while the values are feature data.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Return
    ------
    DGLGraph
        The graph with newly added edges.

    Notes
    -----
    * If the end nodes of the given edges do not exist in :attr:`g`,
      :func:`dgl.add_nodes` is invoked to add those nodes.
      The node features of the new nodes will be filled with zeros.
    * For features in :attr:`g` but not in :attr:`data`,
      DGL assigns zero features for the newly added nodes.
    * For feature in :attr:`data` but not in :attr:`g`, DGL assigns zero features
      for the existing nodes in the graph.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
    >>> g.num_edges()
    2
    >>> g = dgl.add_edges(g, torch.tensor([1, 3]), torch.tensor([0, 1]))
    >>> g.num_edges()
    4

    Since ``u`` or ``v`` contains a non-existing node ID, the nodes are
    added implicitly.

    >>> g.num_nodes()
    4

    If the graph has some edge features and new edges are added without
    features, their features will be filled with zeros.

    >>> g.edata['h'] = torch.ones(4, 1)
    >>> g = dgl.add_edges(g, torch.tensor([1]), torch.tensor([1]))
    >>> g.edata['h']
    tensor([[1.], [1.], [1.], [1.], [0.]])

    You can also assign features for the new edges in adding new edges.

    >>> g = dgl.add_edges(g, torch.tensor([0, 0]), torch.tensor([2, 2]),
    ...                   {'h': torch.tensor([[1.], [2.]]), 'w': torch.ones(2, 1)})
    >>> g.edata['h']
    tensor([[1.], [1.], [1.], [1.], [0.], [1.], [2.]])

    Since :attr:`data` contains new feature fields, the features for old edges
    will be filled with zeros.

    >>> g.edata['w']
    tensor([[0.], [0.], [0.], [0.], [0.], [1.], [1.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g.number_of_edges('plays')
    4
    >>> g = dgl.add_edges(g, torch.tensor([3]), torch.tensor([3]), etype='plays')
    >>> g.number_of_edges('plays')
    5

    See Also
    --------
    add_nodes
    remove_nodes
    remove_edges
    """
    g = g.clone()
    g.add_edges(u, v, data=data, etype=etype)
    return g

def remove_edges(g, eids, etype=None, store_ids=False):
    r"""Remove the specified edges and return a new graph.

    Also delete the features of the edges. The edges must exist in the graph.
    The resulting graph has the same number of the nodes as the input one,
    even if some nodes become isolated after the the edge removal.

    Parameters
    ----------
    eids : int, Tensor, iterable[int]
        The IDs of the edges to remove.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted nodes and edges in the ``ndata``
        and ``edata`` of the resulting graph under name ``dgl.NID`` and ``dgl.EID``,
        respectively.

    Return
    ------
    DGLGraph
        The graph with edges deleted.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.remove_edges(g, torch.tensor([0, 1]))
    >>> g
    Graph(num_nodes=3, num_edges=1,
        ndata_schemes={}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.edges('all')
    (tensor([2]), tensor([2]), tensor([0]))
    >>> g.edata['he']
    tensor([[2.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g = dgl.remove_edges(g, torch.tensor([0, 1]), 'plays')
    >>> g.edges('all', etype='plays')
    (tensor([0, 1]), tensor([0, 0]), tensor([0, 1]))

    See Also
    --------
    add_nodes
    add_edges
    remove_nodes
    """
    g = g.clone()
    g.remove_edges(eids, etype=etype, store_ids=store_ids)
    return g


def remove_nodes(g, nids, ntype=None, store_ids=False):
    r"""Remove the specified nodes and return a new graph.

    Also delete the features. Edges that connect from/to the nodes will be
    removed as well. After the removal, DGL re-labels the remaining nodes and edges
    with IDs from 0.

    Parameters
    ----------
    nids : int, Tensor, iterable[int]
        The nodes to be removed.
    ntype : str, optional
        The type of the nodes to remove. Can be omitted if there is
        only one node type in the graph.
    store_ids : bool, optional
        If True, it will store the raw IDs of the extracted nodes and edges in the ``ndata``
        and ``edata`` of the resulting graph under name ``dgl.NID`` and ``dgl.EID``,
        respectively.

    Return
    ------
    DGLGraph
        The graph with nodes deleted.

    Notes
    -----

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
    >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.remove_nodes(g, torch.tensor([0, 1]))
    >>> g
    Graph(num_nodes=1, num_edges=1,
        ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.ndata['hv']
    tensor([[2.]])
    >>> g.edata['he']
    tensor([[2.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
    ...                                 torch.tensor([0, 0, 1, 1])),
    ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
    ...                                         torch.tensor([0, 1]))
    ...     })
    >>> g = dgl.remove_nodes(g, torch.tensor([0, 1]), ntype='game')
    >>> g.num_nodes('user')
    3
    >>> g.num_nodes('game')
    0
    >>> g.num_edges('plays')
    0

    See Also
    --------
    add_nodes
    add_edges
    remove_edges
    """
    g = g.clone()
    g.remove_nodes(nids, ntype=ntype, store_ids=store_ids)
    return g

def add_self_loop(g, etype=None):
    r"""Add self-loops for each node in the graph and return a new graph.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Return
    ------
    DGLGraph
        The graph with self-loops.

    Notes
    -----
    * The function only supports homogeneous graphs or heterogeneous graphs but
      the relation graph specified by the :attr:`etype` argument is homogeneous.
    * The function adds self-loops regardless of whether they already exist or not.
      If one wishes to have exactly one self-loop for every node,
      call :func:`remove_self_loop` before invoking :func:`add_self_loop`.
    * Features of the new edges (self-loop edges) will be filled with zeros.
    * This function discards the batch information. Please use
      :func:`dgl.DGLGraph.set_batch_num_nodes`
      and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
      to maintain the information.

    Examples
    --------
    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([2, 1, 0])))
    >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
    >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
    >>> g = dgl.add_self_loop(g)
    >>> g
    Graph(num_nodes=3, num_edges=6,
        ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
        edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
    >>> g.edata['he']
    tensor([[0.],
            [1.],
            [2.],
            [0.],
            [0.],
            [0.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (torch.tensor([1, 2]),
    ...                                   torch.tensor([0, 1])),
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]),
    ...                                 torch.tensor([0, 1]))})
    >>> g = dgl.add_self_loop(g, etype='follows')
    >>> g
    Graph(num_nodes={'user': 3, 'game': 2},
        num_edges={('user', 'plays', 'game'): 2, ('user', 'follows', 'user'): 5},
        metagraph=[('user', 'user'), ('user', 'game')])
    """
    etype = g.to_canonical_etype(etype)
    if etype[0] != etype[2]:
        raise DGLError(
            'add_self_loop does not support unidirectional bipartite graphs: {}.' \
            'Please make sure the types of head node and tail node are identical.' \
            ''.format(etype))
    nodes = g.nodes(etype[0])
    new_g = add_edges(g, nodes, nodes, etype=etype)
    return new_g

DGLHeteroGraph.add_self_loop = utils.alias_func(add_self_loop)

def remove_self_loop(g, etype=None):
    r""" Remove self-loops for each node in the graph and return a new graph.

    Parameters
    ----------
    g : DGLGraph
        The graph.
    etype : str or (str, str, str), optional
        The type names of the edges. The allowed type name formats are:

        * ``(str, str, str)`` for source node type, edge type and destination node type.
        * or one ``str`` edge type name if the name can uniquely identify a
          triplet format in the graph.

        Can be omitted if the graph has only one type of edges.

    Notes
    -----
    If a node has multiple self-loops, remove them all. Do nothing for nodes without
    self-loops.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    ---------

    >>> import dgl
    >>> import torch

    **Homogeneous Graphs**

    >>> g = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([1, 0, 0, 2])))
    >>> g.edata['he'] = torch.arange(4).float().reshape(-1, 1)
    >>> g = dgl.remove_self_loop(g)
    >>> g
    Graph(num_nodes=3, num_edges=2,
        edata_schemes={'he': Scheme(shape=(2,), dtype=torch.float32)})
    >>> g.edata['he']
    tensor([[0.],[3.]])

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'follows', 'user'): (torch.tensor([0, 1, 1, 1, 2]),
    ...                                   torch.tensor([0, 0, 1, 1, 1])),
    ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]),
    ...                                 torch.tensor([0, 1]))
    ...     })
    >>> g = dgl.remove_self_loop(g, etype='follows')
    >>> g.num_nodes('user')
    3
    >>> g.num_nodes('game')
    2
    >>> g.num_edges('follows')
    2
    >>> g.num_edges('plays')
    2

    See Also
    --------
    add_self_loop
    """
    etype = g.to_canonical_etype(etype)
    if etype[0] != etype[2]:
        raise DGLError(
            'remove_self_loop does not support unidirectional bipartite graphs: {}.' \
            'Please make sure the types of head node and tail node are identical.' \
            ''.format(etype))
    u, v = g.edges(form='uv', order='eid', etype=etype)
    self_loop_eids = F.tensor(F.nonzero_1d(u == v), dtype=F.dtype(u))
    new_g = remove_edges(g, self_loop_eids, etype=etype)
    return new_g

DGLHeteroGraph.remove_self_loop = utils.alias_func(remove_self_loop)

def compact_graphs(graphs, always_preserve=None, copy_ndata=True, copy_edata=True):
    """Given a list of graphs with the same set of nodes, find and eliminate the common
    isolated nodes across all graphs.

    This function requires the graphs to have the same set of nodes (i.e. the node types
    must be the same, and the number of nodes of each node type must be the same).  The
    metagraph does not have to be the same.

    It finds all the nodes that have zero in-degree and zero out-degree in all the given
    graphs, and eliminates them from all the graphs.

    Useful for graph sampling where you have a giant graph but you only wish to perform
    message passing on a smaller graph with a (tiny) subset of nodes.

    Parameters
    ----------
    graphs : DGLGraph or list[DGLGraph]
        The graph, or list of graphs.

        All graphs must be on CPU.

        All graphs must have the same set of nodes.
    always_preserve : Tensor or dict[str, Tensor], optional
        If a dict of node types and node ID tensors is given, the nodes of given
        node types would not be removed, regardless of whether they are isolated.

        If a Tensor is given, DGL assumes that all the graphs have one (same) node type.
    copy_ndata: bool, optional
        If True, the node features of the returned graphs are copied from the
        original graphs.

        If False, the returned graphs will not have any node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the reversed graph are copied from the
        original graph.

        If False, the reversed graph will not have any edge features.

        (Default: True)

    Returns
    -------
    DGLGraph or list[DGLGraph]
        The compacted graph or list of compacted graphs.

        Each returned graph would have a feature ``dgl.NID`` containing the mapping
        of node IDs for each type from the compacted graph(s) to the original graph(s).
        Note that the mapping is the same for all the compacted graphs.

        All the returned graphs are on CPU.

    Notes
    -----
    This function currently requires that the same node type of all graphs should have
    the same node type ID, i.e. the node types are *ordered* the same.

    If :attr:`copy_edata` is True, the resulting graph will share the edge feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    The following code constructs a bipartite graph with 20 users and 10 games, but
    only user #1 and #3, as well as game #3 and #5, have connections:

    >>> g = dgl.heterograph({('user', 'plays', 'game'): ([1, 3], [3, 5])},
    >>>                      {'user': 20, 'game': 10})

    The following would compact the graph above to another bipartite graph with only
    two users and two games.

    >>> new_g, induced_nodes = dgl.compact_graphs(g)
    >>> induced_nodes
    {'user': tensor([1, 3]), 'game': tensor([3, 5])}

    The mapping tells us that only user #1 and #3 as well as game #3 and #5 are kept.
    Furthermore, the first user and second user in the compacted graph maps to
    user #1 and #3 in the original graph.  Games are similar.

    One can verify that the edge connections are kept the same in the compacted graph.

    >>> new_g.edges(form='all', order='eid', etype='plays')
    (tensor([0, 1]), tensor([0, 1]), tensor([0, 1]))

    When compacting multiple graphs, nodes that do not have any connections in any
    of the given graphs are removed.  So if you compact ``g`` and the following ``g2``
    graphs together:

    >>> g2 = dgl.heterograph({('user', 'plays', 'game'): ([1, 6], [6, 8])},
    >>>                      {'user': 20, 'game': 10})
    >>> (new_g, new_g2), induced_nodes = dgl.compact_graphs([g, g2])
    >>> induced_nodes
    {'user': tensor([1, 3, 6]), 'game': tensor([3, 5, 6, 8])}

    Then one can see that user #1 from both graphs, users #3 from the first graph, as
    well as user #6 from the second graph, are kept.  Games are similar.

    Similarly, one can also verify the connections:

    >>> new_g.edges(form='all', order='eid', etype='plays')
    (tensor([0, 1]), tensor([0, 1]), tensor([0, 1]))
    >>> new_g2.edges(form='all', order='eid', etype='plays')
    (tensor([0, 2]), tensor([2, 3]), tensor([0, 1]))
    """
    return_single = False
    if not isinstance(graphs, Iterable):
        graphs = [graphs]
        return_single = True
    if len(graphs) == 0:
        return []
    if graphs[0].is_block:
        raise DGLError('Compacting a block graph is not allowed.')
    assert all(g.device == F.cpu() for g in graphs), 'all the graphs must be on CPU'

    # Ensure the node types are ordered the same.
    # TODO(BarclayII): we ideally need to remove this constraint.
    ntypes = graphs[0].ntypes
    idtype = graphs[0].idtype
    device = graphs[0].device
    for g in graphs:
        assert ntypes == g.ntypes, \
            ("All graphs should have the same node types in the same order, got %s and %s" %
             ntypes, g.ntypes)
        assert idtype == g.idtype, "Expect graph data type to be {}, but got {}".format(
            idtype, g.idtype)
        assert device == g.device, "Expect graph device to be {}, but got {}".format(
            device, g.device)

    # Process the dictionary or tensor of "always preserve" nodes
    if always_preserve is None:
        always_preserve = {}
    elif not isinstance(always_preserve, Mapping):
        if len(ntypes) > 1:
            raise ValueError("Node type must be given if multiple node types exist.")
        always_preserve = {ntypes[0]: always_preserve}

    always_preserve = utils.prepare_tensor_dict(graphs[0], always_preserve, 'always_preserve')
    always_preserve_nd = []
    for ntype in ntypes:
        nodes = always_preserve.get(ntype, None)
        if nodes is None:
            nodes = F.copy_to(F.tensor([], idtype), device)
        always_preserve_nd.append(F.to_dgl_nd(nodes))

    # Compact and construct heterographs
    new_graph_indexes, induced_nodes = _CAPI_DGLCompactGraphs(
        [g._graph for g in graphs], always_preserve_nd)
    induced_nodes = [F.from_dgl_nd(nodes) for nodes in induced_nodes]

    new_graphs = [
        DGLHeteroGraph(new_graph_index, graph.ntypes, graph.etypes)
        for new_graph_index, graph in zip(new_graph_indexes, graphs)]

    if copy_ndata:
        for g, new_g in zip(graphs, new_graphs):
            node_frames = utils.extract_node_subframes(g, induced_nodes)
            utils.set_new_frames(new_g, node_frames=node_frames)
    if copy_edata:
        for g, new_g in zip(graphs, new_graphs):
            edge_frames = utils.extract_edge_subframes(g, None)
            utils.set_new_frames(new_g, edge_frames=edge_frames)

    if return_single:
        new_graphs = new_graphs[0]

    return new_graphs

def to_block(g, dst_nodes=None, include_dst_in_src=True):
    """Convert a graph into a bipartite-structured *block* for message passing.

    A block is a graph consisting of two sets of nodes: the
    *source* nodes and *destination* nodes.  The source and destination nodes can have multiple
    node types.  All the edges connect from source nodes to destination nodes.

    Specifically, the source nodes and destination nodes will have the same node types as the
    ones in the original graph.  DGL maps each edge ``(u, v)`` with edge type
    ``(utype, etype, vtype)`` in the original graph to the edge with type
    ``etype`` connecting from node ID ``u`` of type ``utype`` in the source side to node
    ID ``v`` of type ``vtype`` in the destination side.

    For blocks returned by :func:`to_block`, the destination nodes of the block will only
    contain the nodes that have at least one inbound edge of any type.  The source nodes
    of the block will only contain the nodes that appear in the destination nodes, as well
    as the nodes that have at least one outbound edge connecting to one of the destination nodes.

    The destination nodes are specified by the :attr:`dst_nodes` argument if it is not None.

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    dst_nodes : Tensor or dict[str, Tensor], optional
        The list of destination nodes.

        If a tensor is given, the graph must have only one node type.

        If given, it must be a superset of all the nodes that have at least one inbound
        edge.  An error will be raised otherwise.
    include_dst_in_src : bool
        If False, do not include destination nodes in source nodes.

        (Default: True)

    Returns
    -------
    DGLBlock
        The new graph describing the block.

        The node IDs induced for each type in both sides would be stored in feature
        ``dgl.NID``.

        The edge IDs induced for each type would be stored in feature ``dgl.EID``.

    Raises
    ------
    DGLError
        If :attr:`dst_nodes` is specified but it is not a superset of all the nodes that
        have at least one inbound edge.

        If :attr:`dst_nodes` is not None, and :attr:`g` and :attr:`dst_nodes`
        are not in the same context.

    Notes
    -----
    :func:`to_block` is most commonly used in customizing neighborhood sampling
    for stochastic training on a large graph.  Please refer to the user guide
    :ref:`guide-minibatch` for a more thorough discussion about the methodology
    of stochastic training.

    See also :func:`create_block` for more flexible construction of blocks.

    Examples
    --------
    Converting a homogeneous graph to a block as described above:
    >>> g = dgl.graph(([1, 2], [2, 3]))
    >>> block = dgl.to_block(g, torch.LongTensor([3, 2]))

    The destination nodes would be exactly the same as the ones given: [3, 2].

    >>> induced_dst = block.dstdata[dgl.NID]
    >>> induced_dst
    tensor([3, 2])

    The first few source nodes would also be exactly the same as
    the ones given.  The rest of the nodes are the ones necessary for message passing
    into nodes 3, 2.  This means that the node 1 would be included.

    >>> induced_src = block.srcdata[dgl.NID]
    >>> induced_src
    tensor([3, 2, 1])

    You can notice that the first two nodes are identical to the given nodes as well as
    the destination nodes.

    The induced edges can also be obtained by the following:

    >>> block.edata[dgl.EID]
    tensor([2, 1])

    This indicates that edge (2, 3) and (1, 2) are included in the result graph.  You can
    verify that the first edge in the block indeed maps to the edge (2, 3), and the
    second edge in the block indeed maps to the edge (1, 2):

    >>> src, dst = block.edges(order='eid')
    >>> induced_src[src], induced_dst[dst]
    (tensor([2, 1]), tensor([3, 2]))

    The destination nodes specified must be a superset of the nodes that have edges connecting
    to them.  For example, the following will raise an error since the destination nodes
    does not contain node 3, which has an edge connecting to it.

    >>> g = dgl.graph(([1, 2], [2, 3]))
    >>> dgl.to_block(g, torch.LongTensor([2]))     # error

    Converting a heterogeneous graph to a block is similar, except that when specifying
    the destination nodes, you have to give a dict:

    >>> g = dgl.heterograph({('A', '_E', 'B'): ([1, 2], [2, 3])})

    If you don't specify any node of type A on the destination side, the node type ``A``
    in the block would have zero nodes on the destination side.

    >>> block = dgl.to_block(g, {'B': torch.LongTensor([3, 2])})
    >>> block.number_of_dst_nodes('A')
    0
    >>> block.number_of_dst_nodes('B')
    2
    >>> block.dstnodes['B'].data[dgl.NID]
    tensor([3, 2])

    The source side would contain all the nodes on the destination side:

    >>> block.srcnodes['B'].data[dgl.NID]
    tensor([3, 2])

    As well as all the nodes that have connections to the nodes on the destination side:

    >>> block.srcnodes['A'].data[dgl.NID]
    tensor([2, 1])

    See also
    --------
    create_block
    """
    if dst_nodes is None:
        # Find all nodes that appeared as destinations
        dst_nodes = defaultdict(list)
        for etype in g.canonical_etypes:
            _, dst = g.edges(etype=etype)
            dst_nodes[etype[2]].append(dst)
        dst_nodes = {ntype: F.unique(F.cat(values, 0)) for ntype, values in dst_nodes.items()}
    elif not isinstance(dst_nodes, Mapping):
        # dst_nodes is a Tensor, check if the g has only one type.
        if len(g.ntypes) > 1:
            raise DGLError(
                'Graph has more than one node type; please specify a dict for dst_nodes.')
        dst_nodes = {g.ntypes[0]: dst_nodes}

    dst_node_ids = [
        utils.toindex(dst_nodes.get(ntype, []), g._idtype_str).tousertensor(
            ctx=F.to_backend_ctx(g._graph.ctx))
        for ntype in g.ntypes]
    dst_node_ids_nd = [F.to_dgl_nd(nodes) for nodes in dst_node_ids]

    for d in dst_node_ids_nd:
        if g._graph.ctx != d.ctx:
            raise ValueError('g and dst_nodes need to have the same context.')

    new_graph_index, src_nodes_nd, induced_edges_nd = _CAPI_DGLToBlock(
        g._graph, dst_node_ids_nd, include_dst_in_src)

    # The new graph duplicates the original node types to SRC and DST sets.
    new_ntypes = (g.ntypes, g.ntypes)
    new_graph = DGLBlock(new_graph_index, new_ntypes, g.etypes)
    assert new_graph.is_unibipartite  # sanity check

    src_node_ids = [F.from_dgl_nd(src) for src in src_nodes_nd]
    edge_ids = [F.from_dgl_nd(eid) for eid in induced_edges_nd]

    node_frames = utils.extract_node_subframes_for_block(g, src_node_ids, dst_node_ids)
    edge_frames = utils.extract_edge_subframes(g, edge_ids)
    utils.set_new_frames(new_graph, node_frames=node_frames, edge_frames=edge_frames)

    return new_graph

def _coalesce_edge_frame(g, edge_maps, counts, aggregator):
    r"""Coalesce edge features of duplicate edges via given aggregator in g.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    edge_maps : List[Tensor]
        The edge mapping corresponding to each edge type in g.
    counts : List[Tensor]
        The number of duplicated edges from the original graph for each edge type.
    aggregator : str
        Indicates how to coalesce edge features, could be ``arbitrary``, ``sum``
        or ``mean``.

    Returns
    -------
    List[Frame]
        The frames corresponding to each edge type.
    """
    if aggregator == 'arbitrary':
        eids = []
        for i in range(len(g.canonical_etypes)):
            feat_idx = F.asnumpy(edge_maps[i])
            _, indices = np.unique(feat_idx, return_index=True)
            eids.append(F.zerocopy_from_numpy(indices))

        edge_frames = utils.extract_edge_subframes(g, eids)
    elif aggregator in ['sum', 'mean']:
        edge_frames = []
        for i in range(len(g.canonical_etypes)):
            feat_idx = edge_maps[i]
            _, indices = np.unique(F.asnumpy(feat_idx), return_index=True)
            _num_rows = len(indices)
            _data = {}
            for key, col in g._edge_frames[i]._columns.items():
                data = col.data
                new_data = F.scatter_add(data, feat_idx, _num_rows)
                if aggregator == 'mean':
                    norm = F.astype(counts[i], F.dtype(data))
                    norm = F.reshape(norm, (F.shape(norm)[0],) + (1,) * (F.ndim(data) - 1))
                    new_data /= norm
                _data[key] = new_data

            newf = Frame(data=_data, num_rows=_num_rows)
            edge_frames.append(newf)
    else:
        raise DGLError("Aggregator {} not regonized, cannot coalesce edge feature in the "
                       "specified way".format(aggregator))
    return edge_frames

def to_simple(g,
              return_counts='count',
              writeback_mapping=False,
              copy_ndata=True,
              copy_edata=False,
              aggregator='arbitrary'):
    r"""Convert a graph to a simple graph without parallel edges and return.

    For a heterogeneous graph with multiple edge types, DGL treats edges with the same
    edge type and endpoints as parallel edges and removes them.
    Optionally, one can get the the number of parallel edges by specifying the
    :attr:`return_counts` argument. To get the a mapping from the edge IDs in the
    input graph to the edge IDs in the resulting graph, set :attr:`writeback_mapping`
    to true.

    Parameters
    ----------
    g : DGLGraph
        The input graph.  Must be on CPU.
    return_counts : str, optional
        If given, the count of each edge in the original graph
        will be stored as edge features under the name
        ``return_counts``.  The old features with the same name will be replaced.

        (Default: "count")
    writeback_mapping: bool, optional
        If True, return an extra write-back mapping for each edge
        type.  The write-back mapping is a tensor recording
        the mapping from the edge IDs in the input graph to
        the edge IDs in the result graph. If the graph is
        heterogeneous, DGL returns a dictionary of edge types and such
        tensors.

        If False, only the simple graph is returned.

        (Default: False)
    copy_ndata: bool, optional
        If True, the node features of the simple graph are copied
        from the original graph.

        If False, the simple graph will not have any node features.

        (Default: True)
    copy_edata: bool, optional
        If True, the edge features of the simple graph are copied
        from the original graph. If there exists duplicate edges between
        two nodes (u, v), the feature of the edge is the aggregation
        of edge feature of duplicate edges.

        If False, the simple graph will not have any edge features.

        (Default: False)
    aggregator: str, optional
        Indicate how to coalesce edge feature of duplicate edges.
        If ``arbitrary``, select one of the duplicate edges' feature.
        If ``sum``, compute the summation of duplicate edges' feature.
        If ``mean``, compute the average of duplicate edges' feature.

        (Default: ``arbitrary``)

    Returns
    -------
    DGLGraph
        The graph.
    tensor or dict of tensor
        The writeback mapping. Only when ``writeback_mapping`` is True.

    Notes
    -----
    If :attr:`copy_ndata` is True, the resulting graph will share the node feature
    tensors with the input graph. Hence, users should try to avoid in-place operations
    which will be visible to both graphs.

    This function discards the batch information. Please use
    :func:`dgl.DGLGraph.set_batch_num_nodes`
    and :func:`dgl.DGLGraph.set_batch_num_edges` on the transformed graph
    to maintain the information.

    Examples
    --------
    **Homogeneous Graphs**

    Create a graph for demonstrating to_simple API.
    In the original graph, there are multiple edges between 1 and 2.

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph((th.tensor([0, 1, 2, 1]), th.tensor([1, 2, 0, 2])))
    >>> g.ndata['h'] = th.tensor([[0.], [1.], [2.]])
    >>> g.edata['h'] = th.tensor([[3.], [4.], [5.], [6.]])

    Convert the graph to a simple graph. The return counts is
    stored in the edge feature 'cnt' and the writeback mapping
    is returned in a tensor.

    >>> sg, wm = dgl.to_simple(g, return_counts='cnt', writeback_mapping=True)
    >>> sg.ndata['h']
    tensor([[0.],
            [1.],
            [2.]])
    >>> u, v, eid = sg.edges(form='all')
    >>> u
    tensor([0, 1, 2])
    >>> v
    tensor([1, 2, 0])
    >>> eid
    tensor([0, 1, 2])
    >>> sg.edata['cnt']
    tensor([1, 2, 1])
    >>> wm
    tensor([0, 1, 2, 1])
    >>> 'h' in g.edata
    False

    **Heterogeneous Graphs**

    >>> g = dgl.heterograph({
    ...     ('user', 'wins', 'user'): (th.tensor([0, 2, 0, 2, 2]), th.tensor([1, 1, 2, 1, 0])),
    ...     ('user', 'plays', 'game'): (th.tensor([1, 2, 1]), th.tensor([2, 1, 1]))
    ... })
    >>> g.nodes['game'].data['hv'] = th.ones(3, 1)
    >>> g.edges['plays'].data['he'] = th.zeros(3, 1)

    The return counts is stored in the default edge feature 'count' for each edge type.

    >>> sg, wm = dgl.to_simple(g, copy_ndata=False, writeback_mapping=True)
    >>> sg
    Graph(num_nodes={'game': 3, 'user': 3},
          num_edges={('user', 'wins', 'user'): 4, ('game', 'plays', 'user'): 3},
          metagraph=[('user', 'user'), ('game', 'user')])
    >>> sg.edges(etype='wins')
    (tensor([0, 2, 0, 2]), tensor([1, 1, 2, 0]))
    >>> wm[('user', 'wins', 'user')]
    tensor([0, 1, 2, 1, 3])
    >>> sg.edges(etype='plays')
    (tensor([2, 1, 1]), tensor([1, 2, 1]))
    >>> wm[('user', 'plays', 'game')]
    tensor([0, 1, 2])
    >>> 'hv' in sg.nodes['game'].data
    False
    >>> 'he' in sg.edges['plays'].data
    False
    >>> sg.edata['count']
    {('user', 'wins', 'user'): tensor([1, 2, 1, 1])
     ('user', 'plays', 'game'): tensor([1, 1, 1])}
    """
    assert g.device == F.cpu(), 'the graph must be on CPU'
    if g.is_block:
        raise DGLError('Cannot convert a block graph to a simple graph.')
    simple_graph_index, counts, edge_maps = _CAPI_DGLToSimpleHetero(g._graph)
    simple_graph = DGLHeteroGraph(simple_graph_index, g.ntypes, g.etypes)
    counts = [F.from_dgl_nd(count) for count in counts]
    edge_maps = [F.from_dgl_nd(edge_map) for edge_map in edge_maps]

    if copy_ndata:
        node_frames = utils.extract_node_subframes(g, None)
        utils.set_new_frames(simple_graph, node_frames=node_frames)
    if copy_edata:
        new_edge_frames = _coalesce_edge_frame(g, edge_maps, counts, aggregator)
        utils.set_new_frames(simple_graph, edge_frames=new_edge_frames)

    if return_counts is not None:
        for count, canonical_etype in zip(counts, g.canonical_etypes):
            simple_graph.edges[canonical_etype].data[return_counts] = count

    if writeback_mapping:
        # single edge type
        if len(edge_maps) == 1:
            return simple_graph, edge_maps[0]
        # multiple edge type
        else:
            wb_map = {}
            for edge_map, canonical_etype in zip(edge_maps, g.canonical_etypes):
                wb_map[canonical_etype] = edge_map
            return simple_graph, wb_map

    return simple_graph

DGLHeteroGraph.to_simple = utils.alias_func(to_simple)

def as_heterograph(g, ntype='_U', etype='_E'):  # pylint: disable=unused-argument
    """Convert a DGLGraph to a DGLHeteroGraph with one node and edge type.

    DEPRECATED: DGLGraph and DGLHeteroGraph have been merged. This function will
                do nothing and can be removed safely in all cases.
    """
    dgl_warning('DEPRECATED: DGLGraph and DGLHeteroGraph have been merged in v0.5.\n'
                '\tdgl.as_heterograph will do nothing and can be removed safely in all cases.')
    return g

def as_immutable_graph(hg):
    """Convert a DGLHeteroGraph with one node and edge type into a DGLGraph.

    DEPRECATED: DGLGraph and DGLHeteroGraph have been merged. This function will
                do nothing and can be removed safely in all cases.
    """
    dgl_warning('DEPRECATED: DGLGraph and DGLHeteroGraph have been merged in v0.5.\n'
                '\tdgl.as_immutable_graph will do nothing and can be removed safely in all cases.')
    return hg

_init_api("dgl.transform")
