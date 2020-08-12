"""Module for converting graph from/to other object."""
# pylint: disable=dangerous-default-value
from collections import defaultdict, Iterable
from scipy.sparse import spmatrix
import numpy as np
import networkx as nx

from . import backend as F
from . import heterograph_index
from .heterograph import DGLHeteroGraph, combine_frames
from . import graph_index
from . import utils
from .base import NTYPE, ETYPE, NID, EID, DGLError, dgl_warning

__all__ = [
    'graph',
    'bipartite',
    'hetero_from_relations',
    'hetero_from_shared_memory',
    'heterograph',
    'to_heterogeneous',
    'to_hetero',
    'to_homogeneous',
    'to_homo',
    'from_scipy',
    'bipartite_from_scipy',
    'from_networkx',
    'bipartite_from_networkx',
    'to_networkx',
]

def graph(data,
          ntype=None, etype=None,
          num_nodes=None,
          idtype=None,
          device=None,
          **deprecated_kwargs):
    """Create a graph.

    Parameters
    ----------
    data : graph data
        The data for constructing a graph, which takes the form of :math:`(U, V)`.
        :math:`(U[i], V[i])` forms an edge and is given edge ID :math:`i` in the graph.
        The allowed data formats are:

        - (Tensor, Tensor): Each tensor must be a 1D tensor containing node IDs.
          DGL calls this format "tuple of node-tensors". The tensors should have the same
          data type of int32/int64 and device context (see below the descriptions of
          :attr:`idtype` and :attr:`device`).
        - (iterable[int], iterable[int]): Similar to the tuple of node-tensors
          format, but stores node IDs in two sequences (e.g. list, tuple, numpy.ndarray).
    ntype : str, optional
        Deprecated. To construct a graph with named node types, see :func:`dgl.heterograph`.
    etype : str, optional
        Deprecated. To construct a graph with named edge types, see :func:`dgl.heterograph`.
    num_nodes : int, optional
        The number of nodes in the graph. If not given, this will be the largest node ID
        plus 1 from the :attr:`data` argument. If given and the value is no greater than
        the largest node ID from the :attr:`data` argument, DGL will raise an error.
    idtype : int32 or int64, optional
        The data type for storing the structure-related graph information such as node and
        edge IDs. It should be a framework-specific data type object (e.g., torch.int32).
        If not given (default), DGL infers the ID type from the :attr:`data` argument.
        See "Notes" for more details.
    device : device context, optional
        The device of the returned graph, which should be a framework-specific device object
        (e.g., torch.device). If ``None`` (default), DGL uses the device of the tensors of
        the :attr:`data` argument. If :attr:`data` is not a tuple of node-tensors, the
        returned graph is on CPU.  If the specified :attr:`device` differs from that of the
        provided tensors, it casts the given tensors to the specified device first.

    Returns
    -------
    DGLGraph
        The created graph.

    Notes
    -----
    1. If the :attr:`idtype` argument is not given then:

       - in the case of the tuple of node-tensor format, DGL uses
         the data type of the tensors for storing node/edge IDs.
       - in the case of the tuple of sequence format, DGL uses int64.

       Once the graph has been created, you can change the data type by using
       :func:`dgl.DGLGraph.long` or :func:`dgl.DGLGraph.int`.

       If the specified :attr:`idtype` argument differs from the data type of the provided
       tensors, it casts the given tensors to the specified data type first.
    2. The most efficient construction approach is to provide a tuple of node tensors without
       specifying :attr:`idtype` and :attr:`device`. This is because the returned graph shares
       the storage with the input node-tensors in this case.
    3. DGL internally maintains multiple copies of the graph structure in different sparse
       formats and chooses the most efficient one depending on the computation invoked.
       If memory usage becomes an issue in the case of large graphs, use
       :func:`dgl.DGLGraph.formats` to restrict the allowed formats.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Create a small three-edge graph.

    >>> # Source nodes for edges (2, 1), (3, 2), (4, 3)
    >>> src_ids = torch.tensor([2, 3, 4])
    >>> # Destination nodes for edges (2, 1), (3, 2), (4, 3)
    >>> dst_ids = torch.tensor([1, 2, 3])
    >>> g = dgl.graph((src_ids, dst_ids))

    Explicitly specify the number of nodes in the graph.

    >>> g = dgl.graph((src_ids, dst_ids), num_nodes=100)

    Create a graph on the first GPU card with data type int32.

    >>> g = dgl.graph((src_ids, dst_ids), idtype=torch.int32, device='cuda:0')

    See Also
    --------
    from_scipy
    from_networkx
    """
    # Sanity check
    utils.check_type(data, tuple, 'data', skip_none=False)
    if len(data) != 2:
        raise DGLError('Expect data to have length 2, got {:d}'.format(len(data)))

    src_type = type(data[0])
    dst_type = type(data[1])
    if src_type != dst_type:
        raise DGLError('Expect the source and destination node IDs to have the same type, ' \
                       'got {} and {}'.format(src_type, dst_type))

    if len(data[0]) != len(data[1]):
        raise DGLError('Expect the source and destination node IDs to have the same length, ' \
                       'got {:d} and {:d}'.format(len(data[0]), len(data[1])))

    if F.is_tensor(data[0]):
        src_dtype = F.dtype(data[0])
        dst_dtype = F.dtype(data[1])
        if src_dtype != dst_dtype:
            raise DGLError('Expect the source and destination node tensors to have the same ' \
                           'data type, got {} and {}'.format(src_dtype, dst_dtype))
        if src_dtype not in [F.int32, F.int64]:
            raise DGLError('Expect the node-tensors to have data type int32 or int64, '
                           'got {}'.format(src_dtype))
        src_ctx = F.context(data[0])
        dst_ctx = F.context(data[1])
        if src_ctx != dst_ctx:
            raise DGLError('Expect the source and destination node tensors to have the same ' \
                           'context, got {} and {}'.format(src_ctx, dst_ctx))
    elif isinstance(data[0], Iterable):
        # NaN/Inf values cannot appear in int32/int64 tensors
        utils.detect_nan_in_iterable(data[0], 'the source node IDs')
        utils.detect_inf_in_iterable(data[0], 'the source node IDs')
        utils.detect_nan_in_iterable(data[1], 'the destination node IDs')
        utils.detect_inf_in_iterable(data[1], 'the destination node IDs')
    else:
        raise DGLError('Expect sequences (e.g., list, numpy.ndarray) or tensors for data, ' \
                       'got {}'.format(type(data[0])))

    if ntype is not None:
        raise DGLError('The ntype argument is deprecated for dgl.graph. To construct ' \
                       'a graph with named node types, use dgl.heterograph.')
    if etype is not None:
        raise DGLError('The etype argument is deprecated for dgl.graph. To construct ' \
                       'a graph with named edge types, use dgl.heterograph.')

    if num_nodes is not None and not (isinstance(num_nodes, int) and num_nodes >= 0):
        raise DGLError('Expect num_nodes to be a positive integer, got {}'.format(num_nodes))

    utils.check_valid_idtype(idtype)

    # Deprecation handling
    deprecated_kwargs = set(deprecated_kwargs.keys())
    nx_deprecated_kwargs = deprecated_kwargs & {
        'edge_id_attr_name', 'node_attrs', 'edge_attrs'}
    other_deprecated_kwargs = deprecated_kwargs & {
        'card', 'validate', 'restrict_format'}

    # Handle the deprecated arguments related to SciPy
    if isinstance(data, spmatrix):
        raise DGLError("dgl.graph no longer supports graph construction from a SciPy "
                       "sparse matrix, use dgl.from_scipy instead.")

    # Handle the deprecated arguments related to NetworkX
    if isinstance(data, nx.Graph) or len(nx_deprecated_kwargs) > 0:
        raise DGLError("dgl.graph no longer supports graph construction from a NetworkX "
                       "graph, use dgl.from_networkx instead.")

    # Handle the deprecation of other input arguments
    if len(other_deprecated_kwargs) > 0:
        raise DGLError("Arguments {} have been deprecated.".format(other_deprecated_kwargs))

    u, v, urange, vrange = utils.graphdata2tensors(data, idtype)
    if num_nodes is not None:  # override the number of nodes
        urange, vrange = num_nodes, num_nodes
    if len(u) > 0:
        utils.assert_nonnegative_iterable(u, 'the source node IDs')
        utils.assert_nonnegative_iterable(v, 'the destination node IDs')

    g = create_from_edges(u, v, '_N', '_E', '_N', urange, vrange)

    return g.to(device)

def bipartite(data,
              utype='_U', etype='_E', vtype='_V',
              num_nodes=None,
              card=None,
              validate=True,
              restrict_format='any',
              **kwargs):
    """DEPRECATED: use dgl.heterograph instead."""
    raise DGLError('dgl.bipartite is deprecated.\n\n'
                   'Use dgl.heterograph instead.')

def hetero_from_relations(rel_graphs, num_nodes_per_type=None):
    """DEPRECATED: use dgl.heterograph instead."""
    raise DGLError('dgl.hetero_from_relations is deprecated.\n\n'
                   'Use dgl.heterograph instead.')

def hetero_from_shared_memory(name):
    """Create a heterograph from shared memory with the given name.

    The newly created graph will have the same node types and edge types as the original graph.
    But it does not have node features or edges features.

    Paramaters
    ----------
    name : str
        The name of the share memory

    Returns
    -------
    HeteroGraph (in shared memory)
    """
    g, ntypes, etypes = heterograph_index.create_heterograph_from_shared_memory(name)
    return DGLHeteroGraph(g, ntypes, etypes)

def heterograph(data_dict,
                num_nodes_dict=None,
                idtype=None,
                device=None):
    """Create a heterogeneous graph.

    Parameters
    ----------
    data_dict : graph data
        The dictionary data for constructing a heterogeneous graph. The keys are in the form of
        string triplet :math:`(src_type, edge_type, dst_type)`, specifying the source node,
        edge, and destination node types. The values are graph data in the form of
        :math:`(U, V)`, where :math:`(U[i], V[i])` forms an edge and is given edge ID :math:`i`.
        The allowed graph data formats are:

        - (Tensor, Tensor): Each tensor must be a 1D tensor containing node IDs. DGL calls this
          format "tuple of node-tensors". The tensors should have the same data type of
          int32/int64 and device context (see below the descriptions of :attr:`idtype` and
          :attr:`device`).
        - (iterable[int], iterable[int]): Similar to the tuple of node-tensors
          format, but stores node IDs in two sequences (e.g. list, tuple, numpy.ndarray).
    num_nodes_dict : dict[str, int], optional
        The number of nodes for each node type. If not given (default), for each node type
        :math:`T`, DGL finds the largest ID appeared in *every* graph data whose source or
        destination node type is :math:`T`, and sets the number of nodes to be that ID plus one.
        If given and the value is no greater than the largest ID, DGL will raise an error.
    idtype : int32 or int64, optional
        The data type for storing the structure-related graph information such as node and
        edge IDs. It should be a framework-specific data type object (e.g., torch.int32).
        If not given (default), DGL infers the ID type from the :attr:`data_dict` argument.
    device : device context, optional
        The device of the returned graph, which should be a framework-specific device object
        (e.g., torch.device). If ``None`` (default), DGL uses the device of the tensors of
        the :attr:`data` argument. If :attr:`data` is not a tuple of node-tensors, the
        returned graph is on CPU.  If the specified :attr:`device` differs from that of the
        provided tensors, it casts the given tensors to the specified device first.

    Returns
    -------
    DGLGraph
        The created graph.

    Notes
    -----
    1. If the :attr:`idtype` argument is not given then:

       - in the case of the tuple of node-tensor format, DGL uses
         the data type of the tensors for storing node/edge IDs.
       - in the case of the tuple of sequence format, DGL uses int64.

       Once the graph has been created, you can change the data type by using
       :func:`dgl.DGLGraph.long` or :func:`dgl.DGLGraph.int`.

       If the specified :attr:`idtype` argument differs from the data type of the provided
       tensors, it casts the given tensors to the specified data type first.
    2. The most efficient construction approach is to provide a tuple of node tensors without
       specifying :attr:`idtype` and :attr:`device`. This is because the returned graph shares
       the storage with the input node-tensors in this case.
    3. DGL internally maintains multiple copies of the graph structure in different sparse
       formats and chooses the most efficient one depending on the computation invoked.
       If memory usage becomes an issue in the case of large graphs, use
       :func:`dgl.DGLGraph.formats` to restrict the allowed formats.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch

    Create a heterograph with three canonical edge types.

    >>> data_dict = {
    >>>     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    >>>     ('user', 'follows', 'topic'): (torch.tensor([1, 1]), torch.tensor([1, 2])),
    >>>     ('user', 'plays', 'game'): (torch.tensor([0, 3]), torch.tensor([3, 4]))
    >>> }
    >>> g = dgl.heterograph(data_dict)
    >>> g
    Graph(num_nodes={'game': 5, 'topic': 3, 'user': 4},
          num_edges={('user', 'follows', 'user'): 2, ('user', 'follows', 'topic'): 2,
                     ('user', 'plays', 'game'): 2},
          metagraph=[('user', 'user', 'follows'), ('user', 'topic', 'follows'),
                     ('user', 'game', 'plays')])

    Explicitly specify the number of nodes for each node type in the graph.

    >>> num_nodes_dict = {'user': 4, 'topic': 4, 'game': 6}
    >>> g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)

    Create a graph on the first GPU card with data type int32.

    >>> g = dgl.heterograph(data_dict, idtype=torch.int32, device='cuda:0')
    """
    # Sanity check
    utils.check_type(data_dict, dict, 'data_dict', skip_none=False)
    # Record the type of all graph data for consistency check
    data_types = []
    # Record the data type/device of node-tensors for consistency check
    data_dtypes = []
    data_devices = []

    # Check type, length, dtype of the arguments
    for key, data in data_dict.items():
        # Check for key
        utils.check_type(key, tuple, 'a key of data_dict', skip_none=False)
        if len(key) != 3:
            raise DGLError('Expect a key of data_dict to have length 3, '
                           'got {:d}'.format(len(key)))
        for typ in key:
            if not isinstance(typ, str):
                raise DGLError('Expect a key of data_dict to contain str only, '
                               'got {}'.format(type(typ)))

        # Check for value
        utils.check_type(data, tuple, 'data_dict[{}]'.format(key), skip_none=False)
        if len(data) != 2:
            raise DGLError('Expect data_dict[{}] to have length 2, '
                           'got {:d}'.format(key, len(data)))
        src_type = type(data[0])
        dst_type = type(data[1])
        if src_type != dst_type:
            raise DGLError('Expect the source and destination node IDs for {} to have '
                           'the same type, got {} and {}'.format(key, src_type, dst_type))
        data_types.append(src_type)

        if F.is_tensor(data[0]):
            src_dtype = F.dtype(data[0])
            dst_dtype = F.dtype(data[1])
            if src_dtype != dst_dtype:
                raise DGLError('Expect the source and destination node tensors for {} to have the '
                               'same data type, got {} and {}'.format(key, src_dtype, dst_dtype))
            data_dtypes.append(src_dtype)
            src_ctx = F.context(data[0])
            dst_ctx = F.context(data[1])
            if src_ctx != dst_ctx:
                raise DGLError('Expect the source and destination node tensors for {} to have the '
                               'same context, got {} and {}'.format(key, src_ctx, dst_ctx))
            data_devices.append(src_ctx)
        elif isinstance(data[0], Iterable):
            # NaN/Inf values cannot appear in int32/int64 tensors
            utils.detect_nan_in_iterable(data[0], 'the source node IDs for {}'.format(key))
            utils.detect_inf_in_iterable(data[0], 'the source node IDs for {}'.format(key))
            utils.detect_nan_in_iterable(data[1], 'the destination node IDs for {}'.format(key))
            utils.detect_inf_in_iterable(data[1], 'the destination node IDs for {}'.format(key))
        else:
            raise DGLError('Expect sequences (e.g., list, numpy.ndarray) or tensors for '
                           'data_dict[{}], got {}'.format(key, type(data[0])))

    data_types = set(data_types)
    if len(data_types) != 1:
        raise DGLError('Expect the node IDs to have a same type, got {}'.format(data_types))
    # Skip the check for sequences
    if len(data_dtypes) > 0:
        data_dtypes = set(data_dtypes)
        if len(data_dtypes) != 1:
            raise DGLError('Expect the node tensors to have a same data type, '
                           'got {}'.format(data_dtypes))
        data_devices = set(data_devices)
        if len(data_devices) != 1:
            raise DGLError('Expect the node tensors to be on a same device, '
                           'got {}'.format(data_devices))

    utils.check_valid_idtype(idtype)
    if idtype is None:
        if len(data_dtypes) > 0:
            idtype = list(data_dtypes)[0]
        else:
            idtype = F.int64

    # Convert all data to node tensors first
    data_dict = {(sty, ety, dty) : utils.graphdata2tensors(data, idtype, bipartite=(sty != dty))
                 for (sty, ety, dty), data in data_dict.items()}

    # Sanity check for the node IDs and the number of nodes
    num_nodes_dict_ = defaultdict(int)
    for (srctype, etype, dsttype), data in data_dict.items():
        relation = (srctype, etype, dsttype)
        src, dst, nsrc, ndst = data
        if len(src) > 0:
            utils.assert_nonnegative_iterable(
                src, 'the source node IDs of canonical edge type {}'.format(relation))
            utils.assert_nonnegative_iterable(
                dst, 'the destination node IDs of canonical edge type {}'.format(relation))
        num_nodes_dict_[srctype] = max(num_nodes_dict_[srctype], nsrc)
        num_nodes_dict_[dsttype] = max(num_nodes_dict_[dsttype], ndst)

    if num_nodes_dict is None:
        num_nodes_dict = num_nodes_dict_
    else:
        utils.check_type(num_nodes_dict, dict, 'num_nodes_dict', skip_none=False)
        for nty in num_nodes_dict_:
            if nty not in num_nodes_dict:
                raise DGLError('Missing node type {} for num_nodes_dict'.format(nty))
            if num_nodes_dict[nty] < num_nodes_dict_[nty]:
                raise DGLError('Expect the number of nodes to be at least {:d} for type {}, '
                               'got {:d}'.format(num_nodes_dict_[nty], nty, num_nodes_dict[nty]))

    # Graph creation
    # TODO(BarclayII): I'm keeping the node type names sorted because even if
    # the metagraph is the same, the same node type name in different graphs may
    # map to different node type IDs.
    # In the future, we need to lower the type names into C++.
    ntypes = list(sorted(num_nodes_dict.keys()))
    num_nodes_per_type = utils.toindex([num_nodes_dict[ntype] for ntype in ntypes], "int64")
    ntype_dict = {ntype: i for i, ntype in enumerate(ntypes)}

    meta_edges_src = []
    meta_edges_dst = []
    etypes = []
    rel_graphs = []
    for (srctype, etype, dsttype), data in data_dict.items():
        meta_edges_src.append(ntype_dict[srctype])
        meta_edges_dst.append(ntype_dict[dsttype])
        etypes.append(etype)
        src, dst, _, _ = data
        g = create_from_edges(src, dst, srctype, etype, dsttype,
                              num_nodes_dict[srctype], num_nodes_dict[dsttype])
        rel_graphs.append(g)

    # metagraph is DGLGraph, currently still using int64 as index dtype
    metagraph = graph_index.from_coo(len(ntypes), meta_edges_src, meta_edges_dst, True)
    # create graph index
    hgidx = heterograph_index.create_heterograph_from_relations(
        metagraph, [rgrh._graph for rgrh in rel_graphs], num_nodes_per_type)
    retg = DGLHeteroGraph(hgidx, ntypes, etypes)

    return retg.to(device)

def to_heterogeneous(G, ntypes, etypes, ntype_field=NTYPE,
                     etype_field=ETYPE, metagraph=None):
    """Convert the given homogeneous graph to a heterogeneous graph.

    The input graph should have only one type of nodes and edges. Each node and edge
    stores an integer feature (under ``ntype_field`` and ``etype_field``), representing
    the type id, which can be used to retrieve the type names stored
    in the given ``ntypes`` and ``etypes`` arguments.

    The function will automatically distinguish edge types that have the same given
    type IDs but different src and dst type IDs. For example, we allow both edges A and B
    to have the same type ID 0, but one has (0, 1) and the other as (2, 3) as the
    (src, dst) type IDs. In this case, the function will "split" edge type 0 into two types:
    (0, ty_A, 1) and (2, ty_B, 3). In another word, these two edges share the same edge
    type name, but can be distinguished by a canonical edge type tuple.

    Parameters
    ----------
    G : DGLGraph
        The homogeneous graph.
    ntypes : list of str
        The node type names.
    etypes : list of str
        The edge type names.
    ntype_field : str, optional
        The feature field used to store node type. (Default: ``dgl.NTYPE``)
    etype_field : str, optional
        The feature field used to store edge type. (Default: ``dgl.ETYPE``)
    metagraph : networkx MultiDiGraph, optional
        Metagraph of the returned heterograph.
        If provided, DGL assumes that G can indeed be described with the given metagraph.
        If None, DGL will infer the metagraph from the given inputs, which would be
        potentially slower for large graphs.

    Returns
    -------
    DGLGraph
        A heterogeneous graph. The parent node and edge ID are stored in the column
        ``dgl.NID`` and ``dgl.EID`` respectively for all node/edge types.

    Notes
    -----
    The returned node and edge types may not necessarily be in the same order as
    ``ntypes`` and ``etypes``.  And edge types may be duplicated if the source
    and destination types differ.

    The node IDs of a single type in the returned heterogeneous graph is ordered
    the same as the nodes with the same ``ntype_field`` feature. Edge IDs of
    a single type is similar.

    Examples
    --------

    >>> import dgl
    >>> hg = dgl.heterograph({
    >>>     ('user', 'develops', 'activity'): ([0, 1], [1, 2]),
    >>>     ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    >>> })
    >>> print(hg)
    Graph(num_nodes={'user': 2, 'activity': 3, 'developer': 2, 'game': 2},
          num_edges={('user', 'develops', 'activity'): 2, ('developer', 'develops', 'game'): 2},
          metagraph=[('user', 'activity'), ('developer', 'game')])

    We first convert the heterogeneous graph to a homogeneous graph.

    >>> g = dgl.to_homogeneous(hg)
    >>> print(g)
    Graph(num_nodes=9, num_edges=4,
          ndata_schemes={'_TYPE': Scheme(shape=(), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_TYPE': Scheme(shape=(), dtype=torch.int64),
                         '_ID': Scheme(shape=(), dtype=torch.int64)})
    >>> g.ndata
    {'_TYPE': tensor([0, 0, 1, 1, 1, 2, 2, 3, 3]), '_ID': tensor([0, 1, 0, 1, 2, 0, 1, 0, 1])}
    Nodes 0, 1 for 'user', 2, 3, 4 for 'activity', 5, 6 for 'developer', 7, 8 for 'game'
    >>> g.edata
    {'_TYPE': tensor([0, 0, 1, 1]), '_ID': tensor([0, 1, 0, 1])}
    Edges 0, 1 for ('user', 'develops', 'activity'), 2, 3 for ('developer', 'develops', 'game')

    Now convert the homogeneous graph back to a heterogeneous graph.

    >>> hg_2 = dgl.to_heterogeneous(g, hg.ntypes, hg.etypes)
    >>> print(hg_2)
    Graph(num_nodes={'user': 2, 'activity': 3, 'developer': 2, 'game': 2},
          num_edges={('user', 'develops', 'activity'): 2, ('developer', 'develops', 'game'): 2},
          metagraph=[('user', 'activity'), ('developer', 'game')])

    See Also
    --------
    to_homogeneous
    """
    # TODO(minjie): use hasattr to support DGLGraph input; should be fixed once
    #  DGLGraph is merged with DGLHeteroGraph
    if (hasattr(G, 'ntypes') and len(G.ntypes) > 1
            or hasattr(G, 'etypes') and len(G.etypes) > 1):
        raise DGLError('The input graph should be homogeneous and have only one '
                       ' type of nodes and edges.')

    num_ntypes = len(ntypes)
    idtype = G.idtype
    device = G.device

    ntype_ids = F.asnumpy(G.ndata[ntype_field])
    etype_ids = F.asnumpy(G.edata[etype_field])

    # relabel nodes to per-type local IDs
    ntype_count = np.bincount(ntype_ids, minlength=num_ntypes)
    ntype_offset = np.insert(np.cumsum(ntype_count), 0, 0)
    ntype_ids_sortidx = np.argsort(ntype_ids)
    ntype_local_ids = np.zeros_like(ntype_ids)
    node_groups = []
    for i in range(num_ntypes):
        node_group = ntype_ids_sortidx[ntype_offset[i]:ntype_offset[i+1]]
        node_groups.append(node_group)
        ntype_local_ids[node_group] = np.arange(ntype_count[i])

    src, dst = G.all_edges(order='eid')
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    src_local = ntype_local_ids[src]
    dst_local = ntype_local_ids[dst]
    # a 2D tensor of shape (E, 3). Each row represents the (stid, etid, dtid) tuple.
    edge_ctids = np.stack([ntype_ids[src], etype_ids, ntype_ids[dst]], 1)

    # infer metagraph and canonical edge types
    # No matter which branch it takes, the code will generate a 2D tensor of shape (E_m, 3),
    # E_m is the set of all possible canonical edge tuples. Each row represents the
    # (stid, dtid, dtid) tuple. We then compute a 2D tensor of shape (E, E_m) using the
    # above ``edge_ctids`` matrix. Each element i,j indicates whether the edge i is of the
    # canonical edge type j. We can then group the edges of the same type together.
    if metagraph is None:
        canonical_etids, _, etype_remapped = \
                utils.make_invmap(list(tuple(_) for _ in edge_ctids), False)
        etype_mask = (etype_remapped[None, :] == np.arange(len(canonical_etids))[:, None])
    else:
        ntypes_invmap = {nt: i for i, nt in enumerate(ntypes)}
        etypes_invmap = {et: i for i, et in enumerate(etypes)}
        canonical_etids = []
        for i, (srctype, dsttype, etype) in enumerate(metagraph.edges(keys=True)):
            srctype_id = ntypes_invmap[srctype]
            etype_id = etypes_invmap[etype]
            dsttype_id = ntypes_invmap[dsttype]
            canonical_etids.append((srctype_id, etype_id, dsttype_id))
        canonical_etids = np.asarray(canonical_etids)
        etype_mask = (edge_ctids[None, :] == canonical_etids[:, None]).all(2)
    edge_groups = [etype_mask[i].nonzero()[0] for i in range(len(canonical_etids))]

    data_dict = dict()
    for i, (stid, etid, dtid) in enumerate(canonical_etids):
        src_of_etype = src_local[edge_groups[i]]
        dst_of_etype = dst_local[edge_groups[i]]
        data_dict[(ntypes[stid], etypes[etid], ntypes[dtid])] = \
            (src_of_etype, dst_of_etype)
    hg = heterograph(data_dict,
                     {ntype: count for ntype, count in zip(ntypes, ntype_count)},
                     idtype=idtype, device=device)

    ntype2ngrp = {ntype : node_groups[ntid] for ntid, ntype in enumerate(ntypes)}

    # features
    for key, data in G.ndata.items():
        for ntid, ntype in enumerate(hg.ntypes):
            rows = F.copy_to(F.tensor(ntype2ngrp[ntype]), F.context(data))
            hg._node_frames[ntid][key] = F.gather_row(data, rows)
    for key, data in G.edata.items():
        for etid in range(len(hg.canonical_etypes)):
            rows = F.copy_to(F.tensor(edge_groups[etid]), F.context(data))
            hg._edge_frames[etid][key] = F.gather_row(data, rows)

    for ntid, ntype in enumerate(hg.ntypes):
        hg._node_frames[ntid][NID] = F.tensor(ntype2ngrp[ntype])

    for etid in range(len(hg.canonical_etypes)):
        hg._edge_frames[etid][EID] = F.tensor(edge_groups[etid])

    return hg

def to_hetero(G, ntypes, etypes, ntype_field=NTYPE, etype_field=ETYPE,
              metagraph=None):
    """Convert the given homogeneous graph to a heterogeneous graph.

    DEPRECATED: Please use to_heterogeneous
    """
    dgl_warning("dgl.to_hetero is deprecated. Please use dgl.to_heterogeneous")
    return to_heterogeneous(G, ntypes, etypes, ntype_field=ntype_field,
                            etype_field=etype_field, metagraph=metagraph)

def to_homogeneous(G):
    """Convert the given heterogeneous graph to a homogeneous graph.

    The returned graph has only one type of nodes and edges.

    Node and edge types are stored as features in the returned graph. Each feature
    is an integer representing the type id, which can be used to retrieve the type
    names stored in ``G.ntypes`` and ``G.etypes`` arguments.

    Parameters
    ----------
    G : DGLGraph
        The heterogeneous graph.

    Returns
    -------
    DGLGraph
        A homogeneous graph. The parent node and edge type/ID are stored in
        columns ``dgl.NTYPE/dgl.NID`` and ``dgl.ETYPE/dgl.EID`` respectively.

    Examples
    --------

    >>> hg = dgl.heterograph({
    >>>     ('user', 'follows', 'user'): [[0, 1], [1, 2]],
    >>>     ('developer', 'develops', 'game'): [[0, 1], [0, 1]]
    >>> })
    >>> g = dgl.to_homogeneous(hg)
    >>> g.ndata
    {'_TYPE': tensor([0, 0, 0, 1, 1, 2, 2]), '_ID': tensor([0, 1, 2, 0, 1, 0, 1])}
    First three nodes for 'user', next two for 'developer' and the last two for 'game'
    >>> g.edata
    {'_TYPE': tensor([0, 0, 1, 1]), '_ID': tensor([0, 1, 0, 1])}
    First two edges for 'follows', next two for 'develops'

    See Also
    --------
    to_heterogeneous
    """
    num_nodes_per_ntype = [G.number_of_nodes(ntype) for ntype in G.ntypes]
    offset_per_ntype = np.insert(np.cumsum(num_nodes_per_ntype), 0, 0)
    srcs = []
    dsts = []
    etype_ids = []
    eids = []
    ntype_ids = []
    nids = []
    total_num_nodes = 0

    for ntype_id, ntype in enumerate(G.ntypes):
        num_nodes = G.number_of_nodes(ntype)
        total_num_nodes += num_nodes
        # Type ID is always in int64
        ntype_ids.append(F.full_1d(num_nodes, ntype_id, F.int64, F.cpu()))
        nids.append(F.arange(0, num_nodes, G.idtype))

    for etype_id, etype in enumerate(G.canonical_etypes):
        srctype, _, dsttype = etype
        src, dst = G.all_edges(etype=etype, order='eid')
        num_edges = len(src)
        srcs.append(src + int(offset_per_ntype[G.get_ntype_id(srctype)]))
        dsts.append(dst + int(offset_per_ntype[G.get_ntype_id(dsttype)]))
        # Type ID is always in int64
        etype_ids.append(F.full_1d(num_edges, etype_id, F.int64, F.cpu()))
        eids.append(F.arange(0, num_edges, G.idtype))

    retg = graph((F.cat(srcs, 0), F.cat(dsts, 0)), num_nodes=total_num_nodes,
                 idtype=G.idtype, device=G.device)

    # copy features
    comb_nf = combine_frames(G._node_frames, range(len(G.ntypes)))
    comb_ef = combine_frames(G._edge_frames, range(len(G.etypes)))
    if comb_nf is not None:
        retg.ndata.update(comb_nf)
    if comb_ef is not None:
        retg.edata.update(comb_ef)

    # assign node type and id mapping field.
    retg.ndata[NTYPE] = F.copy_to(F.cat(ntype_ids, 0), G.device)
    retg.ndata[NID] = F.copy_to(F.cat(nids, 0), G.device)
    retg.edata[ETYPE] = F.copy_to(F.cat(etype_ids, 0), G.device)
    retg.edata[EID] = F.copy_to(F.cat(eids, 0), G.device)

    return retg

def to_homo(G):
    """Convert the given heterogeneous graph to a homogeneous graph.

    DEPRECATED: Please use to_homogeneous
    """
    dgl_warning("dgl.to_homo is deprecated. Please use dgl.to_homogeneous")
    return to_homogeneous(G)

def from_scipy(sp_mat,
               eweight_name=None,
               idtype=None,
               device=None):
    """Create a graph from a SciPy sparse matrix.

    Parameters
    ----------
    sp_mat : scipy.sparse.spmatrix
        The graph adjacency matrix. Each nonzero entry ``sp_mat[i, j]`` represents an edge from
        node ``i`` to ``j``. The matrix must have square shape ``(N, N)``, where ``N`` is the
        number of nodes in the graph.
    eweight_name : str, optional
        The edata name for storing the nonzero values of :attr:`sp_mat`. If given, DGL will
        store the nonzero values of :attr:`sp_mat` in ``edata[eweight_name]`` of the returned
        graph.
    idtype : int32 or int64, optional
        The data type for storing the structure-related graph information such as node and
        edge IDs. It should be a framework-specific data type object (e.g., torch.int32).
        By default, DGL uses int64.
    device : device context, optional
        The device of the resulting graph. It should be a framework-specific device object
        (e.g., torch.device). By default, DGL stores the graph on CPU.

    Returns
    -------
    DGLGraph
        The created graph.

    Notes
    -----
    1. The function supports all kinds of SciPy sparse matrix classes (e.g.,
       :class:`scipy.sparse.csr.csr_matrix`). It converts the input matrix to the COOrdinate
       format using :func:`scipy.sparse.spmatrix.tocoo` before creates a :class:`DGLGraph`.
       Creating from a :class:`scipy.sparse.coo.coo_matrix` is hence the most efficient way.
    2. DGL internally maintains multiple copies of the graph structure in different sparse
       formats and chooses the most efficient one depending on the computation invoked.
       If memory usage becomes an issue in the case of large graphs, use
       :func:`dgl.DGLGraph.formats` to restrict the allowed formats.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import numpy as np
    >>> import torch
    >>> from scipy.sparse import coo_matrix

    Create a small three-edge graph.

    >>> # Source nodes for edges (2, 1), (3, 2), (4, 3)
    >>> src_ids = np.array([2, 3, 4])
    >>> # Destination nodes for edges (2, 1), (3, 2), (4, 3)
    >>> dst_ids = np.array([1, 2, 3])
    >>> # Weight for edges (2, 1), (3, 2), (4, 3)
    >>> eweight = np.array([0.2, 0.3, 0.5])
    >>> sp_mat = coo_matrix((eweight, (src_ids, dst_ids)), shape=(5, 5))
    >>> g = dgl.from_scipy(sp_mat)

    Retrieve the edge weights.

    >>> g = dgl.from_scipy(sp_mat, eweight_name='w')
    >>> g.edata['w']
    tensor([0.2000, 0.3000, 0.5000], dtype=torch.float64)

    Create a graph on the first GPU card with data type int32.

    >>> g = dgl.from_scipy(sp_mat, idtype=torch.int32, device='cuda:0')

    See Also
    --------
    graph
    from_networkx
    """
    # Sanity check
    utils.check_type(sp_mat, spmatrix, 'sp_mat', skip_none=False)
    num_rows = sp_mat.shape[0]
    num_cols = sp_mat.shape[1]
    if num_rows != num_cols:
        raise DGLError('Expect the number of rows to be the same as the number of columns for '
                       'sp_mat, got {:d} and {:d}.'.format(num_rows, num_cols))
    utils.check_valid_idtype(idtype)

    u, v, urange, vrange = utils.graphdata2tensors(sp_mat, idtype)
    g = create_from_edges(u, v, '_N', '_E', '_N', urange, vrange)
    if eweight_name is not None:
        g.edata[eweight_name] = F.tensor(sp_mat.data)
    return g.to(device)

def bipartite_from_scipy(sp_mat,
                         eweight_name=None,
                         idtype=None,
                         device=None):
    """Create a unidirectional bipartite graph from a SciPy sparse matrix.

    A bipartite graph has two types of nodes ``"SRC"`` and ``"DST"`` and
    there are only edges between nodes of different types. By "unidirectional",
    there are only edges from ``"SRC"`` nodes to ``"DST"`` nodes.

    Parameters
    ----------
    sp_mat : scipy.sparse.spmatrix
        The graph adjacency matrix. Each nonzero entry ``sp_mat[i, j]``
        represents an edge from node ``i`` of type ``"SRC"`` to ``j`` of type ``"DST"``.
        Let the matrix shape be ``(N, M)``. There will be ``N`` ``"SRC"``-type nodes
        and ``M`` ``"DST"``-type nodes in the resulting graph.
    eweight_name : str, optional
        The edata name for storing the nonzero values of :attr:`sp_mat`.
        If given, DGL will store the nonzero values of :attr:`sp_mat` in ``edata[eweight_name]``
        of the returned graph.
    idtype : int32 or int64, optional
        The data type for storing the structure-related graph information such as node and
        edge IDs. It should be a framework-specific data type object (e.g., torch.int32).
        By default, DGL uses int64.
    device : device context, optional
        The device of the resulting graph. It should be a framework-specific device object
        (e.g., torch.device). By default, DGL stores the graph on CPU.

    Returns
    -------
    DGLGraph
        The created graph.

    Notes
    -----
    1. The function supports all kinds of SciPy sparse matrix classes (e.g.,
       :class:`scipy.sparse.csr.csr_matrix`). It converts the input matrix to the COOrdinate
       format using :func:`scipy.sparse.spmatrix.tocoo` before creates a :class:`DGLGraph`.
       Creating from a :class:`scipy.sparse.coo.coo_matrix` is hence the most efficient way.
    2. DGL internally maintains multiple copies of the graph structure in different sparse
       formats and chooses the most efficient one depending on the computation invoked.
       If memory usage becomes an issue in the case of large graphs, use
       :func:`dgl.DGLGraph.formats` to restrict the allowed formats.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import numpy as np
    >>> import torch
    >>> from scipy.sparse import coo_matrix

    Create a small three-edge graph.

    >>> # Source nodes for edges (2, 1), (3, 2), (4, 3)
    >>> src_ids = np.array([2, 3, 4])
    >>> # Destination nodes for edges (2, 1), (3, 2), (4, 3)
    >>> dst_ids = np.array([1, 2, 3])
    >>> # Weight for edges (2, 1), (3, 2), (4, 3)
    >>> eweight = np.array([0.2, 0.3, 0.5])
    >>> sp_mat = coo_matrix((eweight, (src_ids, dst_ids)))
    >>> g = dgl.bipartite_from_scipy(sp_mat)

    Retrieve the edge weights.

    >>> g = dgl.bipartite_from_scipy(sp_mat, eweight_name='w')
    >>> g.edata['w']
    tensor([0.2000, 0.3000, 0.5000], dtype=torch.float64)

    Create a graph on the first GPU card with data type int32.

    >>> g = dgl.bipartite_from_scipy(sp_mat, idtype=torch.int32, device='cuda:0')

    See Also
    --------
    heterograph
    bipartite_from_networkx
    """
    # Sanity check
    utils.check_type(sp_mat, spmatrix, 'sp_mat', skip_none=False)
    utils.check_valid_idtype(idtype)

    u, v, urange, vrange = utils.graphdata2tensors(sp_mat, idtype, bipartite=True)
    g = create_from_edges(u, v, '_U', '_E', '_V', urange, vrange)
    if eweight_name is not None:
        g.edata[eweight_name] = F.tensor(sp_mat.data)
    return g.to(device)

def from_networkx(nx_graph,
                  node_attrs=None,
                  edge_attrs=None,
                  edge_id_attr_name=None,
                  idtype=None,
                  device=None):
    """Create a graph from a NetworkX graph.

    Creating a DGLGraph from a NetworkX graph is not fast especially for large scales.
    It is recommended to first convert a NetworkX graph into a tuple of node-tensors
    and then construct a DGLGraph with :func:`dgl.graph`.

    Parameters
    ----------
    nx_graph : networkx.Graph
        The NetworkX graph holding the graph structure and the node/edge attributes.
        DGL will relabel the nodes using consecutive integers starting from zero if it is
        not the case. If the input graph is undirected, DGL converts it to a directed graph
        by :func:`networkx.Graph.to_directed`.
    node_attrs : list[str], optional
        The names of the node attributes to retrieve from the NetworkX graph. If given, DGL
        stores the retrieved node attributes in ``ndata`` of the returned graph using their
        original names. The attribute data must be convertible to Tensor type (e.g., scalar,
        numpy.ndarray, list, etc.).
    edge_attrs : list[str], optional
        The names of the edge attributes to retrieve from the NetworkX graph. If given, DGL
        stores the retrieved edge attributes in ``edata`` of the returned graph using their
        original names. The attribute data must be convertible to Tensor type (e.g., scalar,
        numpy.ndarray, list, etc.). It must be None if :attr:`nx_graph` is undirected.
    edge_id_attr_name : str, optional
        The name of the edge attribute that stores the edge IDs. If given, DGL will assign edge
        IDs accordingly when creating the graph, so the attribute must be valid IDs, i.e.
        consecutive integers starting from zero. By default, the edge IDs of the returned graph
        can be arbitrary. It must be None if :attr:`nx_graph` is undirected.
    idtype : int32 or int64, optional
        The data type for storing the structure-related graph information such as node and
        edge IDs. It should be a framework-specific data type object (e.g., torch.int32).
        By default, DGL uses int64.
    device : device context, optional
        The device of the resulting graph. It should be a framework-specific device object
        (e.g., torch.device). By default, DGL stores the graph on CPU.

    Returns
    -------
    DGLGraph
        The created graph.

    Notes
    -----
    DGL internally maintains multiple copies of the graph structure in different sparse
    formats and chooses the most efficient one depending on the computation invoked.
    If memory usage becomes an issue in the case of large graphs, use
    :func:`dgl.DGLGraph.formats` to restrict the allowed formats.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import networkx as nx
    >>> import numpy as np
    >>> import torch

    Create a 2-edge NetworkX graph

    >>> nx_g = nx.DiGraph()
    >>> # Add 3 nodes and two features for them
    >>> nx_g.add_nodes_from([0, 1, 2], feat1=np.zeros((3, 1)), feat2=np.ones((3, 1)))
    >>> # Add 2 edges (1, 2) and (2, 1) with two features, one being edge IDs
    >>> nx_g.add_edge(1, 2, weight=np.ones((1, 1)), eid=np.array([1]))
    >>> nx_g.add_edge(2, 1, weight=np.ones((1, 1)), eid=np.array([0]))

    Convert it into a DGLGraph with structure only.

    >>> g = dgl.from_networkx(nx_g)

    Retrieve the node/edge features of the graph.

    >>> g = dgl.from_networkx(nx_g, node_attrs=['feat1', 'feat2'], edge_attrs=['weight'])

    Use a pre-specified ordering of the edges.

    >>> g.edges()
    (tensor([1, 2]), tensor([2, 1]))
    >>> g = dgl.from_networkx(nx_g, edge_id_attr_name='eid')
    (tensor([2, 1]), tensor([1, 2]))

    Create a graph on the first GPU card with data type int32.

    >>> g = dgl.from_networkx(nx_g, idtype=torch.int32, device='cuda:0')

    See Also
    --------
    graph
    from_scipy
    """
    # Sanity check
    utils.check_type(nx_graph, nx.Graph, 'nx_graph', skip_none=False)
    utils.check_all_same_type(node_attrs, str, 'node_attrs', skip_none=True)
    utils.check_all_same_type(edge_attrs, str, 'edge_attrs', skip_none=True)
    utils.check_type(edge_id_attr_name, str, 'edge_id_attr_name', skip_none=True)
    if edge_id_attr_name is not None and \
            edge_id_attr_name not in next(iter(nx_graph.edges(data=True)))[-1]:
        raise DGLError('Failed to find the pre-specified edge IDs in the edge features of '
                       'the NetworkX graph with name {}'.format(edge_id_attr_name))
    utils.check_valid_idtype(idtype)

    if not nx_graph.is_directed() and not (edge_id_attr_name is None and edge_attrs is None):
        raise DGLError('Expect edge_id_attr_name and edge_attrs to be None when nx_graph is '
                       'undirected, got {} and {}'.format(edge_id_attr_name, edge_attrs))

    # Relabel nodes using consecutive integers starting from 0
    nx_graph = nx.convert_node_labels_to_integers(nx_graph, ordering='sorted')
    if not nx_graph.is_directed():
        nx_graph = nx_graph.to_directed()

    u, v, urange, vrange = utils.graphdata2tensors(
        nx_graph, idtype, edge_id_attr_name=edge_id_attr_name)

    g = create_from_edges(u, v, '_N', '_E', '_N', urange, vrange)

    # nx_graph.edges(data=True) returns src, dst, attr_dict
    has_edge_id = nx_graph.number_of_edges() > 0 and edge_id_attr_name is not None

    # handle features
    # copy attributes
    def _batcher(lst):
        if F.is_tensor(lst[0]):
            return F.cat([F.unsqueeze(x, 0) for x in lst], dim=0)
        else:
            return F.tensor(lst)
    if node_attrs is not None:
        # mapping from feature name to a list of tensors to be concatenated
        attr_dict = defaultdict(list)
        for nid in range(g.number_of_nodes()):
            for attr in node_attrs:
                attr_dict[attr].append(nx_graph.nodes[nid][attr])
        for attr in node_attrs:
            g.ndata[attr] = F.copy_to(_batcher(attr_dict[attr]), g.device)

    if edge_attrs is not None:
        # mapping from feature name to a list of tensors to be concatenated
        attr_dict = defaultdict(lambda: [None] * g.number_of_edges())
        # each defaultdict value is initialized to be a list of None
        # None here serves as placeholder to be replaced by feature with
        # corresponding edge id
        if has_edge_id:
            num_edges = g.number_of_edges()
            for _, _, attrs in nx_graph.edges(data=True):
                if attrs[edge_id_attr_name] >= num_edges:
                    raise DGLError('Expect the pre-specified edge ids to be'
                                   ' smaller than the number of edges --'
                                   ' {}, got {}.'.format(num_edges, attrs['id']))
                for key in edge_attrs:
                    attr_dict[key][attrs[edge_id_attr_name]] = attrs[key]
        else:
            # XXX: assuming networkx iteration order is deterministic
            #      so the order is the same as graph_index.from_networkx
            for eid, (_, _, attrs) in enumerate(nx_graph.edges(data=True)):
                for key in edge_attrs:
                    attr_dict[key][eid] = attrs[key]
        for attr in edge_attrs:
            for val in attr_dict[attr]:
                if val is None:
                    raise DGLError('Not all edges have attribute {}.'.format(attr))
            g.edata[attr] = F.copy_to(_batcher(attr_dict[attr]), g.device)

    return g.to(device)

def bipartite_from_networkx(nx_graph,
                            src_attrs=None,
                            edge_attrs=None,
                            dst_attrs=None,
                            edge_id_attr_name=None,
                            idtype=None,
                            device=None):
    """Create a unidirectional bipartite graph from a NetworkX graph.

    A bipartite graph has two types of nodes ``"SRC"`` and ``"DST"`` and
    there are only edges between nodes of different types. By "unidirectional",
    there are only edges from ``"SRC"`` nodes to ``"DST"`` nodes.

    Creating a DGLGraph from a NetworkX graph is not fast especially for large scales.
    It is recommended to first convert a NetworkX graph into a tuple of node-tensors
    and then construct a DGLGraph with :func:`dgl.heterograph`.

    Parameters
    ----------
    nx_graph : networkx.DiGraph
        The NetworkX graph holding the graph structure and the node/edge attributes.
        DGL will relabel the nodes using consecutive integers starting from zero if it is
        not the case. The graph must follow `NetworkX's bipartite graph convention
        <https://networkx.github.io/documentation/stable/reference/algorithms/bipartite.html>`_,
        and furthermore the edges must be from nodes with attribute `bipartite=0` to nodes
        with attribute `bipartite=1`.
    src_attrs : list[str], optional
        The names of the ``"SRC"`` node attributes to retrieve from the NetworkX graph. If given,
        DGL stores the retrieved node attributes in ``srcdata`` of the returned graph using their
        original names. The attribute data must be convertible to Tensor type (e.g., scalar,
        numpy.array, list, etc.).
    edge_attrs : list[str], optional
        The names of the edge attributes to retrieve from the NetworkX graph. If given, DGL
        stores the retrieved edge attributes in ``edata`` of the returned graph using their
        original names. The attribute data must be convertible to Tensor type (e.g., scalar,
        numpy.ndarray, list, etc.).
    dst_attrs : list[str], optional
        The names of the ``"DST"`` node attributes to retrieve from the NetworkX graph. If given,
        DGL stores the retrieved node attributes in ``dstdata`` of the returned graph using their
        original names. The attribute data must be convertible to Tensor type (e.g., scalar,
        numpy.array, list, etc.).
    edge_id_attr_name : str, optional
        The name of the edge attribute that stores the edge IDs. If given, DGL will assign edge
        IDs accordingly when creating the graph, so the attribute must be valid IDs, i.e.
        consecutive integers starting from zero. By default, the edge IDs of the returned graph
        can be arbitrary.
    idtype : int32 or int64, optional
        The data type for storing the structure-related graph information such as node and
        edge IDs. It should be a framework-specific data type object (e.g., torch.int32).
        By default, DGL uses int64.
    device : device context, optional
        The device of the resulting graph. It should be a framework-specific device object
        (e.g., torch.device). By default, DGL stores the graph on CPU.

    Returns
    -------
    DGLGraph
        The created graph.

    Examples
    --------

    The following example uses PyTorch backend.

    >>> import dgl
    >>> import networkx as nx
    >>> import numpy as np
    >>> import torch

    Create a 2-edge unidirectional bipartite graph.

    >>> nx_g = nx.DiGraph()
    >>> # Add nodes for the source type
    >>> nx_g.add_nodes_from([1, 3], bipartite=0, feat1=np.zeros((2, 1)), feat2=np.ones((2, 1)))
    >>> # Add nodes for the destination type
    >>> nx_g.add_nodes_from([2, 4, 5], bipartite=1, feat3=np.zeros((3, 1)))
    >>> nx_g.add_edge(1, 4, weight=np.ones((1, 1)), eid=np.array([1]))
    >>> nx_g.add_edge(3, 5, weight=np.ones((1, 1)), eid=np.array([0]))

    Convert it into a DGLGraph with structure only.

    >>> g = dgl.bipartite_from_networkx(nx_g)

    Retrieve the node/edge features of the graph.

    >>> g = dgl.bipartite_from_networkx(nx_g, src_attrs=['feat1', 'feat2'],
    >>>                                 edge_attrs=['weight'], dst_attrs=['feat3'])

    Use a pre-specified ordering of the edges.

    >>> g.edges()
    (tensor([0, 1]), tensor([1, 2]))
    >>> g = dgl.bipartite_from_networkx(nx_g, edge_id_attr_name='eid')
    (tensor([1, 0]), tensor([2, 1]))

    Create a graph on the first GPU card with data type int32.

    >>> g = dgl.bipartite_from_networkx(nx_g, idtype=torch.int32, device='cuda:0')

    See Also
    --------
    heterograph
    bipartite_from_scipy
    """
    if not nx_graph.is_directed():
        raise DGLError('Expect nx_graph to be a directed NetworkX graph.')
    utils.check_all_same_type(src_attrs, str, 'src_attrs', skip_none=True)
    utils.check_all_same_type(edge_attrs, str, 'edge_attrs', skip_none=True)
    utils.check_all_same_type(dst_attrs, str, 'dst_attrs', skip_none=True)
    utils.check_type(edge_id_attr_name, str, 'edge_id_attr_name', skip_none=True)
    if edge_id_attr_name is not None and \
            not edge_id_attr_name in next(iter(nx_graph.edges(data=True)))[-1]:
        raise DGLError('Failed to find the pre-specified edge IDs in the edge features '
                       'of the NetworkX graph with name {}'.format(edge_id_attr_name))
    utils.check_valid_idtype(idtype)

    # Get the source and destination node sets
    top_nodes = set()
    bottom_nodes = set()
    for n, ndata in nx_graph.nodes(data=True):
        if 'bipartite' not in ndata:
            raise DGLError('Expect the node {} to have attribute bipartite'.format(n))
        if ndata['bipartite'] == 0:
            top_nodes.add(n)
        elif ndata['bipartite'] == 1:
            bottom_nodes.add(n)
        else:
            raise ValueError('Expect the bipartite attribute of the node {} to be 0 or 1, '
                             'got {}'.format(n, ndata['bipartite']))

    # Separately relabel the source and destination nodes.
    top_nodes = sorted(top_nodes)
    bottom_nodes = sorted(bottom_nodes)
    top_map = {n : i for i, n in enumerate(top_nodes)}
    bottom_map = {n : i for i, n in enumerate(bottom_nodes)}

    # Get the node tensors and the number of nodes
    u, v, urange, vrange = utils.graphdata2tensors(
        nx_graph, idtype, bipartite=True,
        edge_id_attr_name=edge_id_attr_name,
        top_map=top_map, bottom_map=bottom_map)

    g = create_from_edges(u, v, '_U', '_E', '_V', urange, vrange)

    # nx_graph.edges(data=True) returns src, dst, attr_dict
    has_edge_id = nx_graph.number_of_edges() > 0 and edge_id_attr_name is not None

    # handle features
    # copy attributes
    def _batcher(lst):
        if F.is_tensor(lst[0]):
            return F.cat([F.unsqueeze(x, 0) for x in lst], dim=0)
        else:
            return F.tensor(lst)

    if src_attrs is not None:
        # mapping from feature name to a list of tensors to be concatenated
        src_attr_dict = defaultdict(list)
        for nid in top_map.keys():
            for attr in src_attrs:
                src_attr_dict[attr].append(nx_graph.nodes[nid][attr])
        for attr in src_attrs:
            g.srcdata[attr] = F.copy_to(_batcher(src_attr_dict[attr]), g.device)

    if dst_attrs is not None:
        # mapping from feature name to a list of tensors to be concatenated
        dst_attr_dict = defaultdict(list)
        for nid in bottom_map.keys():
            for attr in dst_attrs:
                dst_attr_dict[attr].append(nx_graph.nodes[nid][attr])
        for attr in dst_attrs:
            g.dstdata[attr] = F.copy_to(_batcher(dst_attr_dict[attr]), g.device)

    if edge_attrs is not None:
        # mapping from feature name to a list of tensors to be concatenated
        attr_dict = defaultdict(lambda: [None] * g.number_of_edges())
        # each defaultdict value is initialized to be a list of None
        # None here serves as placeholder to be replaced by feature with
        # corresponding edge id
        if has_edge_id:
            for _, _, attrs in nx_graph.edges(data=True):
                for key in edge_attrs:
                    attr_dict[key][attrs[edge_id_attr_name]] = attrs[key]
        else:
            # XXX: assuming networkx iteration order is deterministic
            #      so the order is the same as graph_index.from_networkx
            for eid, (_, _, attrs) in enumerate(nx_graph.edges(data=True)):
                for key in edge_attrs:
                    attr_dict[key][eid] = attrs[key]
        for attr in edge_attrs:
            for val in attr_dict[attr]:
                if val is None:
                    raise DGLError('Not all edges have attribute {}.'.format(attr))
            g.edata[attr] = F.copy_to(_batcher(attr_dict[attr]), g.device)

    return g.to(device)

def to_networkx(g, node_attrs=None, edge_attrs=None):
    """Convert to networkx graph.

    The edge id will be saved as the 'id' edge attribute.

    Parameters
    ----------
    g : DGLGraph or DGLHeteroGraph
        For DGLHeteroGraphs, we currently only support the
        case of one node type and one edge type.
    node_attrs : iterable of str, optional
        The node attributes to be copied. (Default: None)
    edge_attrs : iterable of str, optional
        The edge attributes to be copied. (Default: None)

    Returns
    -------
    networkx.DiGraph
        The nx graph
    """
    if g.device != F.cpu():
        raise DGLError('Cannot convert a CUDA graph to networkx. Call g.cpu() first.')
    if not g.is_homogeneous():
        raise DGLError('dgl.to_networkx only supports homogeneous graphs.')
    src, dst = g.edges()
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    # xiangsx: Always treat graph as multigraph
    nx_graph = nx.MultiDiGraph()
    nx_graph.add_nodes_from(range(g.number_of_nodes()))
    for eid, (u, v) in enumerate(zip(src, dst)):
        nx_graph.add_edge(u, v, id=eid)

    if node_attrs is not None:
        for nid, attr in nx_graph.nodes(data=True):
            feat_dict = g._get_n_repr(0, nid)
            attr.update({key: F.squeeze(feat_dict[key], 0) for key in node_attrs})
    if edge_attrs is not None:
        for _, _, attr in nx_graph.edges(data=True):
            eid = attr['id']
            feat_dict = g._get_e_repr(0, eid)
            attr.update({key: F.squeeze(feat_dict[key], 0) for key in edge_attrs})
    return nx_graph

DGLHeteroGraph.to_networkx = to_networkx

############################################################
# Internal APIs
############################################################

def create_from_edges(u, v,
                      utype, etype, vtype,
                      urange, vrange,
                      validate=True,
                      formats=['coo', 'csr', 'csc']):
    """Internal function to create a graph from incident nodes with types.

    utype could be equal to vtype

    Parameters
    ----------
    u : Tensor
        Source node IDs.
    v : Tensor
        Dest node IDs.
    utype : str
        Source node type name.
    etype : str
        Edge type name.
    vtype : str
        Destination node type name.
    urange : int, optional
        The source node ID range. If None, the value is the maximum
        of the source node IDs in the edge list plus 1. (Default: None)
    vrange : int, optional
        The destination node ID range. If None, the value is the
        maximum of the destination node IDs in the edge list plus 1. (Default: None)
    validate : bool, optional
        If True, checks if node IDs are within range.
    formats : str or list of str
        It can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of them,
        Force the storage formats.  Default: ``['coo', 'csr', 'csc']``.

    Returns
    -------
    DGLHeteroGraph
    """
    if validate:
        if urange is not None and len(u) > 0 and \
            urange <= F.as_scalar(F.max(u, dim=0)):
            raise DGLError('Invalid node id {} (should be less than cardinality {}).'.format(
                urange, F.as_scalar(F.max(u, dim=0))))
        if vrange is not None and len(v) > 0 and \
            vrange <= F.as_scalar(F.max(v, dim=0)):
            raise DGLError('Invalid node id {} (should be less than cardinality {}).'.format(
                vrange, F.as_scalar(F.max(v, dim=0))))

    if utype == vtype:
        num_ntypes = 1
    else:
        num_ntypes = 2

    if 'coo' in formats:
        hgidx = heterograph_index.create_unitgraph_from_coo(
            num_ntypes, urange, vrange, u, v, formats)
    else:
        hgidx = heterograph_index.create_unitgraph_from_coo(
            num_ntypes, urange, vrange, u, v, ['coo']).formats(formats)
    if utype == vtype:
        return DGLHeteroGraph(hgidx, [utype], [etype])
    else:
        return DGLHeteroGraph(hgidx, [utype, vtype], [etype])
