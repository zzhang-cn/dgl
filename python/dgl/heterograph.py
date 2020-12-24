"""Classes for heterogeneous graphs."""
#pylint: disable= too-many-lines
from collections import defaultdict
from collections.abc import Mapping, Iterable
from contextlib import contextmanager
import copy
import numbers
import networkx as nx
import numpy as np

from ._ffi.function import _init_api
from .base import ALL, SLICE_FULL, NTYPE, NID, ETYPE, EID, is_all, DGLError, dgl_warning
from . import core
from . import graph_index
from . import heterograph_index
from . import utils
from . import backend as F
from .frame import Frame
from .view import HeteroNodeView, HeteroNodeDataView, HeteroEdgeView, HeteroEdgeDataView

__all__ = ['DGLHeteroGraph', 'combine_names']

class DGLHeteroGraph(object):
    """Class for storing graph structure and node/edge feature data.

    There are a few ways to create a DGLGraph:

    * To create a homogeneous graph from Tensor data, use :func:`dgl.graph`.
    * To create a heterogeneous graph from Tensor data, use :func:`dgl.heterograph`.
    * To create a graph from other data sources, use ``dgl.*`` create ops. See
      :ref:`api-graph-create-ops`.

    Read the user guide chapter :ref:`guide-graph` for an in-depth explanation about its
    usage.
    """
    is_block = False

    # pylint: disable=unused-argument, dangerous-default-value
    def __init__(self,
                 gidx=[],
                 ntypes=['_U'],
                 etypes=['_V'],
                 node_frames=None,
                 edge_frames=None,
                 **deprecate_kwargs):
        """Internal constructor for creating a DGLGraph.

        Parameters
        ----------
        gidx : HeteroGraphIndex
            Graph index object.
        ntypes : list of str, pair of list of str
            Node type list. ``ntypes[i]`` stores the name of node type i.
            If a pair is given, the graph created is a uni-directional bipartite graph,
            and its SRC node types and DST node types are given as in the pair.
        etypes : list of str
            Edge type list. ``etypes[i]`` stores the name of edge type i.
        node_frames : list[Frame], optional
            Node feature storage. If None, empty frame is created.
            Otherwise, ``node_frames[i]`` stores the node features
            of node type i. (default: None)
        edge_frames : list[Frame], optional
            Edge feature storage. If None, empty frame is created.
            Otherwise, ``edge_frames[i]`` stores the edge features
            of edge type i. (default: None)
        """
        if isinstance(gidx, DGLHeteroGraph):
            raise DGLError('The input is already a DGLGraph. No need to create it again.')
        if not isinstance(gidx, heterograph_index.HeteroGraphIndex):
            dgl_warning('Recommend creating graphs by `dgl.graph(data)`'
                        ' instead of `dgl.DGLGraph(data)`.')
            u, v, num_src, num_dst = utils.graphdata2tensors(gidx)
            gidx = heterograph_index.create_unitgraph_from_coo(
                1, num_src, num_dst, u, v, ['coo', 'csr', 'csc'])
        if len(deprecate_kwargs) != 0:
            dgl_warning('Keyword arguments {} are deprecated in v0.5, and can be safely'
                        ' removed in all cases.'.format(list(deprecate_kwargs.keys())))
        self._init(gidx, ntypes, etypes, node_frames, edge_frames)

    def _init(self, gidx, ntypes, etypes, node_frames, edge_frames):
        """Init internal states."""
        self._graph = gidx
        self._canonical_etypes = None
        self._batch_num_nodes = None
        self._batch_num_edges = None

        # Handle node types
        if isinstance(ntypes, tuple):
            if len(ntypes) != 2:
                errmsg = 'Invalid input. Expect a pair (srctypes, dsttypes) but got {}'.format(
                    ntypes)
                raise TypeError(errmsg)
            if not is_unibipartite(self._graph.metagraph):
                raise ValueError('Invalid input. The metagraph must be a uni-directional'
                                 ' bipartite graph.')
            self._ntypes = ntypes[0] + ntypes[1]
            self._srctypes_invmap = {t : i for i, t in enumerate(ntypes[0])}
            self._dsttypes_invmap = {t : i + len(ntypes[0]) for i, t in enumerate(ntypes[1])}
            self._is_unibipartite = True
            if len(ntypes[0]) == 1 and len(ntypes[1]) == 1 and len(etypes) == 1:
                self._canonical_etypes = [(ntypes[0][0], etypes[0], ntypes[1][0])]
        else:
            self._ntypes = ntypes
            if len(ntypes) == 1:
                src_dst_map = None
            else:
                src_dst_map = find_src_dst_ntypes(self._ntypes, self._graph.metagraph)
            self._is_unibipartite = (src_dst_map is not None)
            if self._is_unibipartite:
                self._srctypes_invmap, self._dsttypes_invmap = src_dst_map
            else:
                self._srctypes_invmap = {t : i for i, t in enumerate(self._ntypes)}
                self._dsttypes_invmap = self._srctypes_invmap

        # Handle edge types
        self._etypes = etypes
        if self._canonical_etypes is None:
            if (len(etypes) == 1 and len(ntypes) == 1):
                self._canonical_etypes = [(ntypes[0], etypes[0], ntypes[0])]
            else:
                self._canonical_etypes = make_canonical_etypes(
                    self._etypes, self._ntypes, self._graph.metagraph)

        # An internal map from etype to canonical etype tuple.
        # If two etypes have the same name, an empty tuple is stored instead to indicate
        # ambiguity.
        self._etype2canonical = {}
        for i, ety in enumerate(self._etypes):
            if ety in self._etype2canonical:
                self._etype2canonical[ety] = tuple()
            else:
                self._etype2canonical[ety] = self._canonical_etypes[i]
        self._etypes_invmap = {t : i for i, t in enumerate(self._canonical_etypes)}

        # node and edge frame
        if node_frames is None:
            node_frames = [None] * len(self._ntypes)
        node_frames = [Frame(num_rows=self._graph.number_of_nodes(i))
                       if frame is None else frame
                       for i, frame in enumerate(node_frames)]
        self._node_frames = node_frames

        if edge_frames is None:
            edge_frames = [None] * len(self._etypes)
        edge_frames = [Frame(num_rows=self._graph.number_of_edges(i))
                       if frame is None else frame
                       for i, frame in enumerate(edge_frames)]
        self._edge_frames = edge_frames

    def __setstate__(self, state):
        # Compatibility check
        # TODO: version the storage
        if isinstance(state, dict):
            # Since 0.5 we use the default __dict__ method
            self.__dict__.update(state)
        elif isinstance(state, tuple) and len(state) == 5:
            # DGL == 0.4.3
            dgl_warning("The object is pickled with DGL == 0.4.3.  "
                        "Some of the original attributes are ignored.")
            self._init(*state)
        elif isinstance(state, dict):
            # DGL <= 0.4.2
            dgl_warning("The object is pickled with DGL <= 0.4.2.  "
                        "Some of the original attributes are ignored.")
            self._init(state['_graph'], state['_ntypes'], state['_etypes'], state['_node_frames'],
                       state['_edge_frames'])
        else:
            raise IOError("Unrecognized pickle format.")

    def __repr__(self):
        if len(self.ntypes) == 1 and len(self.etypes) == 1:
            ret = ('Graph(num_nodes={node}, num_edges={edge},\n'
                   '      ndata_schemes={ndata}\n'
                   '      edata_schemes={edata})')
            return ret.format(node=self.number_of_nodes(), edge=self.number_of_edges(),
                              ndata=str(self.node_attr_schemes()),
                              edata=str(self.edge_attr_schemes()))
        else:
            ret = ('Graph(num_nodes={node},\n'
                   '      num_edges={edge},\n'
                   '      metagraph={meta})')
            nnode_dict = {self.ntypes[i] : self._graph.number_of_nodes(i)
                          for i in range(len(self.ntypes))}
            nedge_dict = {self.canonical_etypes[i] : self._graph.number_of_edges(i)
                          for i in range(len(self.etypes))}
            meta = str(self.metagraph().edges(keys=True))
            return ret.format(node=nnode_dict, edge=nedge_dict, meta=meta)

    def __copy__(self):
        """Shallow copy implementation."""
        #TODO(minjie): too many states in python; should clean up and lower to C
        cls = type(self)
        obj = cls.__new__(cls)
        obj.__dict__.update(self.__dict__)
        return obj

    #################################################################
    # Mutation operations
    #################################################################

    def add_nodes(self, num, data=None, ntype=None):
        r"""Add new nodes of the same node type

        Parameters
        ----------
        num : int
            Number of nodes to add.
        data : dict, optional
            Feature data of the added nodes.
        ntype : str, optional
            The type of the new nodes. Can be omitted if there is
            only one node type in the graph.

        Notes
        -----

        * Inplace update is applied to the current graph.
        * If the key of ``data`` does not contain some existing feature fields,
        those features for the new nodes will be created by initializers
        defined with :func:`set_n_initializer` (default initializer fills zeros).
        * If the key of ``data`` contains new feature fields, those features for
        the old nodes will be created by initializers defined with
        :func:`set_n_initializer` (default initializer fills zeros).

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.num_nodes()
        3
        >>> g.add_nodes(2)
        >>> g.num_nodes()
        5

        If the graph has some node features and new nodes are added without
        features, their features will be created by initializers defined
        with :func:`set_n_initializer`.

        >>> g.ndata['h'] = torch.ones(5, 1)
        >>> g.add_nodes(1)
        >>> g.ndata['h']
        tensor([[1.], [1.], [1.], [1.], [1.], [0.]])

        We can also assign features for the new nodes in adding new nodes.

        >>> g.add_nodes(1, {'h': torch.ones(1, 1), 'w': torch.ones(1, 1)})
        >>> g.ndata['h']
        tensor([[1.], [1.], [1.], [1.], [1.], [0.], [1.]])

        Since ``data`` contains new feature fields, the features for old nodes
        will be created by initializers defined with :func:`set_n_initializer`.

        >>> g.ndata['w']
        tensor([[0.], [0.], [0.], [0.], [0.], [0.], [1.]])


        **Heterogeneous Graphs with Multiple Node Types**

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.add_nodes(2)
        DGLError: Node type name must be specified
        if there are more than one node types.
        >>> g.num_nodes('user')
        3
        >>> g.add_nodes(2, ntype='user')
        >>> g.num_nodes('user')
        5

        See Also
        --------
        remove_nodes
        add_edges
        remove_edges
        """
        # TODO(xiangsx): block do not support add_nodes
        if ntype is None:
            if self._graph.number_of_ntypes() != 1:
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')

        # nothing happen
        if num == 0:
            return

        assert num > 0, 'Number of new nodes should be larger than one.'
        ntid = self.get_ntype_id(ntype)
        # update graph idx
        metagraph = self._graph.metagraph
        num_nodes_per_type = []
        for c_ntype in self.ntypes:
            if self.get_ntype_id(c_ntype) == ntid:
                num_nodes_per_type.append(self.number_of_nodes(c_ntype) + num)
            else:
                num_nodes_per_type.append(self.number_of_nodes(c_ntype))

        relation_graphs = []
        for c_etype in self.canonical_etypes:
            # src or dst == ntype, update the relation graph
            if self.get_ntype_id(c_etype[0]) == ntid or self.get_ntype_id(c_etype[2]) == ntid:
                u, v = self.edges(form='uv', order='eid', etype=c_etype)
                hgidx = heterograph_index.create_unitgraph_from_coo(
                    1 if c_etype[0] == c_etype[2] else 2,
                    self.number_of_nodes(c_etype[0]) + \
                        (num if self.get_ntype_id(c_etype[0]) == ntid else 0),
                    self.number_of_nodes(c_etype[2]) + \
                        (num if self.get_ntype_id(c_etype[2]) == ntid else 0),
                    u,
                    v,
                    ['coo', 'csr', 'csc'])
                relation_graphs.append(hgidx)
            else:
                # do nothing
                relation_graphs.append(self._graph.get_relation_graph(self.get_etype_id(c_etype)))
        hgidx = heterograph_index.create_heterograph_from_relations(
            metagraph, relation_graphs, utils.toindex(num_nodes_per_type, "int64"))
        self._graph = hgidx

        # update data frames
        if data is None:
            # Initialize feature with :func:`set_n_initializer`
            self._node_frames[ntid].add_rows(num)
        else:
            self._node_frames[ntid].append(data)
        self._reset_cached_info()

    def add_edge(self, u, v, data=None, etype=None):
        """Add one edge to the graph.

        DEPRECATED: please use ``add_edges``.
        """
        dgl_warning("DGLGraph.add_edge is deprecated. Please use DGLGraph.add_edges")
        self.add_edges(u, v, data, etype)

    def add_edges(self, u, v, data=None, etype=None):
        r"""Add multiple new edges for the specified edge type

        The i-th new edge will be from ``u[i]`` to ``v[i]``.

        Parameters
        ----------
        u : int, tensor, numpy.ndarray, list
            Source node IDs, ``u[i]`` gives the source node for the i-th new edge.
        v : int, tensor, numpy.ndarray, list
            Destination node IDs, ``v[i]`` gives the destination node for the i-th new edge.
        data : dict, optional
            Feature data of the added edges. The i-th row of the feature data
            corresponds to the i-th new edge.
        etype : str or tuple of str, optional
            The type of the new edges. Can be omitted if there is
            only one edge type in the graph.

        Notes
        -----

        * Inplace update is applied to the current graph.
        * If end nodes of adding edges does not exists, add_nodes is invoked
        to add new nodes. The node features of the new nodes will be created
        by initializers defined with :func:`set_n_initializer` (default
        initializer fills zeros). In certain cases, it is recommanded to
        add_nodes first and then add_edges.
        * If the key of ``data`` does not contain some existing feature fields,
        those features for the new edges will be created by initializers
        defined with :func:`set_n_initializer` (default initializer fills zeros).
        * If the key of ``data`` contains new feature fields, those features for
        the old edges will be created by initializers defined with
        :func:`set_n_initializer` (default initializer fills zeros).

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Edge Type**

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.num_edges()
        2
        >>> g.add_edges(torch.tensor([1, 3]), torch.tensor([0, 1]))
        >>> g.num_edges()
        4

        Since ``u`` or ``v`` contains a non-existing node ID, the nodes are
        added implicitly.
        >>> g.num_nodes()
        4

        If the graph has some edge features and new edges are added without
        features, their features will be created by initializers defined
        with :func:`set_n_initializer`.

        >>> g.edata['h'] = torch.ones(4, 1)
        >>> g.add_edges(torch.tensor([1]), torch.tensor([1]))
        >>> g.edata['h']
        tensor([[1.], [1.], [1.], [1.], [0.]])

        We can also assign features for the new edges in adding new edges.

        >>> g.add_edges(torch.tensor([0, 0]), torch.tensor([2, 2]),
        ...             {'h': torch.tensor([[1.], [2.]]), 'w': torch.ones(2, 1)})
        >>> g.edata['h']
        tensor([[1.], [1.], [1.], [1.], [0.], [1.], [2.]])

        Since ``data`` contains new feature fields, the features for old edges
        will be created by initializers defined with :func:`set_n_initializer`.

        >>> g.edata['w']
        tensor([[0.], [0.], [0.], [0.], [0.], [1.], [1.]])

        **Heterogeneous Graphs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.add_edges(torch.tensor([3]), torch.tensor([3]))
        DGLError: Edge type name must be specified
        if there are more than one edge types.
        >>> g.number_of_edges('plays')
        4
        >>> g.add_edges(torch.tensor([3]), torch.tensor([3]), etype='plays')
        >>> g.number_of_edges('plays')
        5

        See Also
        --------
        add_nodes
        remove_nodes
        remove_edges
        """
        # TODO(xiangsx): block do not support add_edges
        u = utils.prepare_tensor(self, u, 'u')
        v = utils.prepare_tensor(self, v, 'v')

        if etype is None:
            if self._graph.number_of_etypes() != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')

        # nothing changed
        if len(u) == 0 or len(v) == 0:
            return

        assert len(u) == len(v) or len(u) == 1 or len(v) == 1, \
            'The number of source nodes and the number of destination nodes should be same, ' \
            'or either the number of source nodes or the number of destination nodes is 1.'

        if len(u) == 1 and len(v) > 1:
            u = F.full_1d(len(v), F.as_scalar(u), dtype=F.dtype(u), ctx=F.context(u))
        if len(v) == 1 and len(u) > 1:
            v = F.full_1d(len(u), F.as_scalar(v), dtype=F.dtype(v), ctx=F.context(v))

        u_type, e_type, v_type = self.to_canonical_etype(etype)
        # if end nodes of adding edges does not exists
        # use add_nodes to add new nodes first.
        num_of_u = self.number_of_nodes(u_type)
        num_of_v = self.number_of_nodes(v_type)
        u_max = F.as_scalar(F.max(u, dim=0)) + 1
        v_max = F.as_scalar(F.max(v, dim=0)) + 1

        if u_type == v_type:
            num_nodes = max(u_max, v_max)
            if num_nodes > num_of_u:
                self.add_nodes(num_nodes - num_of_u, ntype=u_type)
        else:
            if u_max > num_of_u:
                self.add_nodes(u_max - num_of_u, ntype=u_type)
            if v_max > num_of_v:
                self.add_nodes(v_max - num_of_v, ntype=v_type)

        # metagraph is not changed
        metagraph = self._graph.metagraph
        num_nodes_per_type = []
        for ntype in self.ntypes:
            num_nodes_per_type.append(self.number_of_nodes(ntype))
        # update graph idx
        relation_graphs = []
        for c_etype in self.canonical_etypes:
            # the target edge type
            if c_etype == (u_type, e_type, v_type):
                old_u, old_v = self.edges(form='uv', order='eid', etype=c_etype)
                hgidx = heterograph_index.create_unitgraph_from_coo(
                    1 if u_type == v_type else 2,
                    self.number_of_nodes(u_type),
                    self.number_of_nodes(v_type),
                    F.cat([old_u, u], dim=0),
                    F.cat([old_v, v], dim=0),
                    ['coo', 'csr', 'csc'])
                relation_graphs.append(hgidx)
            else:
                # do nothing
                # Note: node range change has been handled in add_nodes()
                relation_graphs.append(self._graph.get_relation_graph(self.get_etype_id(c_etype)))

        hgidx = heterograph_index.create_heterograph_from_relations(
            metagraph, relation_graphs, utils.toindex(num_nodes_per_type, "int64"))
        self._graph = hgidx

        # handle data
        etid = self.get_etype_id(etype)
        if data is None:
            self._edge_frames[etid].add_rows(len(u))
        else:
            self._edge_frames[etid].append(data)
        self._reset_cached_info()

    def remove_edges(self, eids, etype=None):
        r"""Remove multiple edges with the specified edge type

        Nodes will not be removed. After removing edges, the rest
        edges will be re-indexed using consecutive integers from 0,
        with their relative order preserved.

        The features for the removed edges will be removed accordingly.

        Parameters
        ----------
        eids : int, tensor, numpy.ndarray, list
            IDs for the edges to remove.
        etype : str or tuple of str, optional
            The type of the edges to remove. Can be omitted if there is
            only one edge type in the graph.

        Examples
        --------

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Edge Type**

        >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
        >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
        >>> g.remove_edges(torch.tensor([0, 1]))
        >>> g
        Graph(num_nodes=3, num_edges=1,
            ndata_schemes={}
            edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
        >>> g.edges('all')
        (tensor([2]), tensor([2]), tensor([0]))
        >>> g.edata['he']
        tensor([[2.]])

        **Heterogeneous Graphs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.remove_edges(torch.tensor([0, 1]))
        DGLError: Edge type name must be specified
        if there are more than one edge types.
        >>> g.remove_edges(torch.tensor([0, 1]), 'plays')
        >>> g.edges('all', etype='plays')
        (tensor([0, 1]), tensor([0, 0]), tensor([0, 1]))

        See Also
        --------
        add_nodes
        add_edges
        remove_nodes
        """
        # TODO(xiangsx): block do not support remove_edges
        if etype is None:
            if self._graph.number_of_etypes() != 1:
                raise DGLError('Edge type name must be specified if there are more than one ' \
                               'edge types.')
        eids = utils.prepare_tensor(self, eids, 'u')
        if len(eids) == 0:
            # no edge to delete
            return
        assert self.number_of_edges(etype) > F.as_scalar(F.max(eids, dim=0)), \
            'The input eid {} is out of the range [0:{})'.format(
                F.as_scalar(F.max(eids, dim=0)), self.number_of_edges(etype))

        # edge_subgraph
        edges = {}
        u_type, e_type, v_type = self.to_canonical_etype(etype)
        for c_etype in self.canonical_etypes:
            # the target edge type
            if c_etype == (u_type, e_type, v_type):
                origin_eids = self.edges(form='eid', order='eid', etype=c_etype)
                edges[c_etype] = utils.compensate(eids, origin_eids)
            else:
                edges[c_etype] = self.edges(form='eid', order='eid', etype=c_etype)

        sub_g = self.edge_subgraph(edges, preserve_nodes=True)
        self._graph = sub_g._graph
        self._node_frames = sub_g._node_frames
        self._edge_frames = sub_g._edge_frames

    def remove_nodes(self, nids, ntype=None):
        r"""Remove multiple nodes with the specified node type

        Edges that connect to the nodes will be removed as well. After removing
        nodes and edges, the rest nodes and edges will be re-indexed using
        consecutive integers from 0, with their relative order preserved.

        The features for the removed nodes/edges will be removed accordingly.

        Parameters
        ----------
        nids : int, tensor, numpy.ndarray, list
            Nodes to remove.
        ntype : str, optional
            The type of the nodes to remove. Can be omitted if there is
            only one node type in the graph.

        Examples
        --------

        >>> import dgl
        >>> import torch

        **Homogeneous Graphs or Heterogeneous Graphs with A Single Node Type**

        >>> g = dgl.graph((torch.tensor([0, 0, 2]), torch.tensor([0, 1, 2])))
        >>> g.ndata['hv'] = torch.arange(3).float().reshape(-1, 1)
        >>> g.edata['he'] = torch.arange(3).float().reshape(-1, 1)
        >>> g.remove_nodes(torch.tensor([0, 1]))
        >>> g
        Graph(num_nodes=1, num_edges=1,
            ndata_schemes={'hv': Scheme(shape=(1,), dtype=torch.float32)}
            edata_schemes={'he': Scheme(shape=(1,), dtype=torch.float32)})
        >>> g.ndata['hv']
        tensor([[2.]])
        >>> g.edata['he']
        tensor([[2.]])

        **Heterogeneous Graphs with Multiple Node Types**

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.remove_nodes(torch.tensor([0, 1]))
        DGLError: Node type name must be specified
        if there are more than one node types.
        >>> g.remove_nodes(torch.tensor([0, 1]), ntype='game')
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
        # TODO(xiangsx): block do not support remove_nodes
        if ntype is None:
            if self._graph.number_of_ntypes() != 1:
                raise DGLError('Node type name must be specified if there are more than one ' \
                               'node types.')

        nids = utils.prepare_tensor(self, nids, 'u')
        if len(nids) == 0:
            # no node to delete
            return
        assert self.number_of_nodes(ntype) > F.as_scalar(F.max(nids, dim=0)), \
            'The input nids {} is out of the range [0:{})'.format(
                F.as_scalar(F.max(nids, dim=0)), self.number_of_nodes(ntype))

        ntid = self.get_ntype_id(ntype)
        nodes = {}
        for c_ntype in self.ntypes:
            if self.get_ntype_id(c_ntype) == ntid:
                original_nids = self.nodes(c_ntype)
                nodes[c_ntype] = utils.compensate(nids, original_nids)
            else:
                nodes[c_ntype] = self.nodes(c_ntype)

        # node_subgraph
        sub_g = self.subgraph(nodes)
        self._graph = sub_g._graph
        self._node_frames = sub_g._node_frames
        self._edge_frames = sub_g._edge_frames

    def _reset_cached_info(self):
        """Some info like batch_num_nodes may be stale after mutation
        Clean these cached info
        """
        self._batch_num_nodes = None
        self._batch_num_edges = None


    #################################################################
    # Metagraph query
    #################################################################

    @property
    def is_unibipartite(self):
        """Return whether the graph is a uni-bipartite graph.

        A uni-bipartite heterograph can further divide its node types into two sets:
        SRC and DST. All edges are from nodes in SRC to nodes in DST. The following APIs
        can be used to get the type, data, and nodes that belong to SRC and DST sets:

        * :func:`srctype` and :func:`dsttype`
        * :func:`srcdata` and :func:`dstdata`
        * :func:`srcnodes` and :func:`dstnodes`

        Note that we allow two node types to have the same name as long as one
        belongs to SRC while the other belongs to DST. To distinguish them, prepend
        the name with ``"SRC/"`` or ``"DST/"`` when specifying a node type.
        """
        return self._is_unibipartite

    @property
    def ntypes(self):
        """Return all the node type names in the graph.

        Returns
        -------
        list[str]
            All the node type names in a list.

        Notes
        -----
        DGL internally assigns an integer ID for each node type. The returned
        node type names are sorted according to their IDs.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> g.ntypes
        ['game', 'user']
        """
        return self._ntypes

    @property
    def etypes(self):
        """Return all the edge type names in the graph.

        Returns
        -------
        list[str]
            All the edge type names in a list.

        Notes
        -----
        DGL internally assigns an integer ID for each edge type. The returned
        edge type names are sorted according to their IDs.

        The complete format to specify an relation is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type. DGL calls this
        format *canonical edge type*. An edge type can appear in multiple canonical edge types.
        For example, ``'interacts'`` can appear in two canonical edge types
        ``('drug', 'interacts', 'drug')`` and ``('protein', 'interacts', 'protein')``.

        See Also
        --------
        canonical_etypes

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> g.etypes
        ['follows', 'follows', 'plays']
        """
        return self._etypes

    @property
    def canonical_etypes(self):
        """Return all the canonical edge types in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.

        Returns
        -------
        list[(str, str, str)]
            All the canonical edge type triplets in a list.

        Notes
        -----
        DGL internally assigns an integer ID for each edge type. The returned
        edge type names are sorted according to their IDs.

        See Also
        --------
        etypes

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> g.canonical_etypes
        [('user', 'follows', 'user'),
         ('user', 'follows', 'game'),
         ('user', 'plays', 'game')]
        """
        return self._canonical_etypes

    @property
    def srctypes(self):
        """Return all the source node type names in this graph.

        If the graph can further divide its node types into two subsets A and B where
        all the edeges are from nodes of types in A to nodes of types in B, we call
        this graph a *uni-bipartite* graph and the nodes in A being the *source*
        nodes and the ones in B being the *destination* nodes. If the graph is not
        uni-bipartite, the source and destination nodes are just the entire set of
        nodes in the graph.

        Returns
        -------
        list[str]
            All the source node type names in a list.

        See Also
        --------
        dsttypes
        is_unibipartite

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for a uni-bipartite graph.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })
        >>> g.srctypes
        ['developer', 'user']

        Query for a graph that is not uni-bipartite.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })
        >>> g.srctypes
        ['developer', 'game', 'user']
        """
        if self.is_unibipartite:
            return sorted(list(self._srctypes_invmap.keys()))
        else:
            return self.ntypes

    @property
    def dsttypes(self):
        """Return all the destination node type names in this graph.

        If the graph can further divide its node types into two subsets A and B where
        all the edeges are from nodes of types in A to nodes of types in B, we call
        this graph a *uni-bipartite* graph and the nodes in A being the *source*
        nodes and the ones in B being the *destination* nodes. If the graph is not
        uni-bipartite, the source and destination nodes are just the entire set of
        nodes in the graph.

        Returns
        -------
        list[str]
            All the destination node type names in a list.

        See Also
        --------
        srctypes
        is_unibipartite

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for a uni-bipartite graph.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })
        >>> g.dsttypes
        ['game']

        Query for a graph that is not uni-bipartite.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })
        >>> g.dsttypes
        ['developer', 'game', 'user']
        """
        if self.is_unibipartite:
            return sorted(list(self._dsttypes_invmap.keys()))
        else:
            return self.ntypes

    def metagraph(self):
        """Return the metagraph of the heterograph.

        The metagraph (or network schema) of a heterogeneous network specifies type constraints
        on the sets of nodes and edges between the nodes. For a formal definition, refer to
        `Yizhou et al. <https://www.kdd.org/exploration_files/V14-02-03-Sun.pdf>`_.

        Returns
        -------
        networkx.MultiDiGraph
            The metagraph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> meta_g = g.metagraph()
        >>> meta_g.nodes()
        NodeView(('user', 'game'))
        >>> meta_g.edges()
        OutMultiEdgeDataView([('user', 'user'), ('user', 'game'), ('user', 'game')])
        """
        nx_graph = self._graph.metagraph.to_networkx()
        nx_metagraph = nx.MultiDiGraph()
        for u_v in nx_graph.edges:
            srctype, etype, dsttype = self.canonical_etypes[nx_graph.edges[u_v]['id']]
            nx_metagraph.add_edge(srctype, dsttype, etype)
        return nx_metagraph

    def to_canonical_etype(self, etype):
        """Convert an edge type to the corresponding canonical edge type in the graph.

        A canonical edge type is a string triplet ``(str, str, str)``
        for source node type, edge type and destination node type.

        The function expects the given edge type name can uniquely identify a canonical edge
        type. DGL will raise error if this is not the case.

        Parameters
        ----------
        etype : str or (str, str, str)
            If :attr:`etype` is an edge type (str), it returns the corresponding canonical edge
            type in the graph. If :attr:`etype` is already a canonical edge type,
            it directly returns the input unchanged.

        Returns
        -------
        (str, str, str)
            The canonical edge type corresponding to the edge type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a heterograph.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ...     ('developer', 'follows', 'game'): ([0, 1], [0, 1])
        ... })

        Map an edge type to its corresponding canonical edge type.

        >>> g.to_canonical_etype('plays')
        ('user', 'plays', 'game')
        >>> g.to_canonical_etype(('user', 'plays', 'game'))
        ('user', 'plays', 'game')

        See Also
        --------
        canonical_etypes
        """
        if etype is None:
            if len(self.etypes) != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')
            etype = self.etypes[0]
        if isinstance(etype, tuple):
            return etype
        else:
            ret = self._etype2canonical.get(etype, None)
            if ret is None:
                raise DGLError('Edge type "{}" does not exist.'.format(etype))
            if len(ret) == 0:
                raise DGLError('Edge type "%s" is ambiguous. Please use canonical edge type '
                               'in the form of (srctype, etype, dsttype)' % etype)
            return ret

    def get_ntype_id(self, ntype):
        """Return the ID of the given node type.

        ntype can also be None. If so, there should be only one node type in the
        graph.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if self.is_unibipartite and ntype is not None:
            # Only check 'SRC/' and 'DST/' prefix when is_unibipartite graph is True.
            if ntype.startswith('SRC/'):
                return self.get_ntype_id_from_src(ntype[4:])
            elif ntype.startswith('DST/'):
                return self.get_ntype_id_from_dst(ntype[4:])
            # If there is no prefix, fallback to normal lookup.

        # Lookup both SRC and DST
        if ntype is None:
            if self.is_unibipartite or len(self._srctypes_invmap) != 1:
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')
            return 0
        ntid = self._srctypes_invmap.get(ntype, self._dsttypes_invmap.get(ntype, None))
        if ntid is None:
            raise DGLError('Node type "{}" does not exist.'.format(ntype))
        return ntid

    def get_ntype_id_from_src(self, ntype):
        """Internal function to return the ID of the given SRC node type.

        ntype can also be None. If so, there should be only one node type in the
        SRC category. Callable even when the self graph is not uni-bipartite.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if ntype is None:
            if len(self._srctypes_invmap) != 1:
                raise DGLError('SRC node type name must be specified if there are more than one '
                               'SRC node types.')
            return next(iter(self._srctypes_invmap.values()))
        ntid = self._srctypes_invmap.get(ntype, None)
        if ntid is None:
            raise DGLError('SRC node type "{}" does not exist.'.format(ntype))
        return ntid

    def get_ntype_id_from_dst(self, ntype):
        """Internal function to return the ID of the given DST node type.

        ntype can also be None. If so, there should be only one node type in the
        DST category. Callable even when the self graph is not uni-bipartite.

        Parameters
        ----------
        ntype : str
            Node type

        Returns
        -------
        int
        """
        if ntype is None:
            if len(self._dsttypes_invmap) != 1:
                raise DGLError('DST node type name must be specified if there are more than one '
                               'DST node types.')
            return next(iter(self._dsttypes_invmap.values()))
        ntid = self._dsttypes_invmap.get(ntype, None)
        if ntid is None:
            raise DGLError('DST node type "{}" does not exist.'.format(ntype))
        return ntid

    def get_etype_id(self, etype):
        """Return the id of the given edge type.

        etype can also be None. If so, there should be only one edge type in the
        graph.

        Parameters
        ----------
        etype : str or tuple of str
            Edge type

        Returns
        -------
        int
        """
        if etype is None:
            if self._graph.number_of_etypes() != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')
            return 0
        etid = self._etypes_invmap.get(self.to_canonical_etype(etype), None)
        if etid is None:
            raise DGLError('Edge type "{}" does not exist.'.format(etype))
        return etid

    #################################################################
    # Batching
    #################################################################
    @property
    def batch_size(self):
        """Return the number of graphs in the batched graph.

        Returns
        -------
        int
            The Number of graphs in the batch. If the graph is not a batched one,
            it will return 1.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for homogeneous graphs.

        >>> g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
        >>> g1.batch_size
        1
        >>> g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
        >>> bg = dgl.batch([g1, g2])
        >>> bg.batch_size
        2

        Query for heterogeneous graphs.

        >>> hg1 = dgl.heterograph({
        ...       ('user', 'plays', 'game') : (torch.tensor([0, 1]), torch.tensor([0, 0]))})
        >>> hg1.batch_size
        1
        >>> hg2 = dgl.heterograph({
        ...       ('user', 'plays', 'game') : (torch.tensor([0, 0]), torch.tensor([1, 0]))})
        >>> bg = dgl.batch([hg1, hg2])
        >>> bg.batch_size
        2
        """
        return len(self.batch_num_nodes(self.ntypes[0]))

    def batch_num_nodes(self, ntype=None):
        """Return the number of nodes for each graph in the batch with the specified node type.

        Parameters
        ----------
        ntype : str, optional
            The node type for query. If the graph has multiple node types, one must
            specify the argument. Otherwise, it can be omitted. If the graph is not a batched
            one, it will return a list of length 1 that holds the number of nodes in the graph.

        Returns
        -------
        Tensor
            The number of nodes with the specified type for each graph in the batch. The i-th
            element of it is the number of nodes with the specified type for the i-th graph.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for homogeneous graphs.

        >>> g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
        >>> g1.batch_num_nodes()
        tensor([4])
        >>> g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
        >>> bg = dgl.batch([g1, g2])
        >>> bg.batch_num_nodes()
        tensor([4, 3])

        Query for heterogeneous graphs.

        >>> hg1 = dgl.heterograph({
        ...       ('user', 'plays', 'game') : (torch.tensor([0, 1]), torch.tensor([0, 0]))})
        >>> hg2 = dgl.heterograph({
        ...       ('user', 'plays', 'game') : (torch.tensor([0, 0]), torch.tensor([1, 0]))})
        >>> bg = dgl.batch([hg1, hg2])
        >>> bg.batch_num_nodes('user')
        tensor([2, 1])
        """
        if ntype is not None and ntype not in self.ntypes:
            raise DGLError('Expect ntype in {}, got {}'.format(self.ntypes, ntype))

        if self._batch_num_nodes is None:
            self._batch_num_nodes = {}
            for ty in self.ntypes:
                bnn = F.copy_to(F.tensor([self.number_of_nodes(ty)], F.int64), self.device)
                self._batch_num_nodes[ty] = bnn
        if ntype is None:
            if len(self.ntypes) != 1:
                raise DGLError('Node type name must be specified if there are more than one '
                               'node types.')
            ntype = self.ntypes[0]
        return self._batch_num_nodes[ntype]

    def set_batch_num_nodes(self, val):
        """TBD"""
        if not isinstance(val, Mapping):
            if len(self.ntypes) != 1:
                raise DGLError('Must provide a dictionary when there are multiple node types.')
            val = {self.ntypes[0] : val}
        self._batch_num_nodes = val

    def batch_num_edges(self, etype=None):
        """Return the number of edges for each graph in the batch with the specified edge type.

        Parameters
        ----------
        etype : str or tuple of str, optional
            The edge type for query, which can be an edge type (str) or a canonical edge type
            (3-tuple of str). When an edge type appears in multiple canonical edge types, one
            must use a canonical edge type. If the graph has multiple edge types, one must
            specify the argument. Otherwise, it can be omitted.

        Returns
        -------
        Tensor
            The number of edges with the specified type for each graph in the batch. The i-th
            element of it is the number of edges with the specified type for the i-th graph.
            If the graph is not a batched one, it will return a list of length 1 that holds
            the number of edges in the graph.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for homogeneous graphs.

        >>> g1 = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
        >>> g1.batch_num_edges()
        tensor([3])
        >>> g2 = dgl.graph((torch.tensor([0, 0, 0, 1]), torch.tensor([0, 1, 2, 0])))
        >>> bg = dgl.batch([g1, g2])
        >>> bg.batch_num_edges()
        tensor([3, 4])

        Query for heterogeneous graphs.

        >>> hg1 = dgl.heterograph({
        ...       ('user', 'plays', 'game') : (torch.tensor([0, 1]), torch.tensor([0, 0]))})
        >>> hg2 = dgl.heterograph({
        ...       ('user', 'plays', 'game') : (torch.tensor([0, 0]), torch.tensor([1, 0]))})
        >>> bg = dgl.batch([hg1, hg2])
        >>> bg.batch_num_edges('plays')
        tensor([2, 2])
        """
        if self._batch_num_edges is None:
            self._batch_num_edges = {}
            for ty in self.canonical_etypes:
                bne = F.copy_to(F.tensor([self.number_of_edges(ty)], F.int64), self.device)
                self._batch_num_edges[ty] = bne
        if etype is None:
            if len(self.etypes) != 1:
                raise DGLError('Edge type name must be specified if there are more than one '
                               'edge types.')
            etype = self.canonical_etypes[0]
        else:
            etype = self.to_canonical_etype(etype)
        return self._batch_num_edges[etype]

    def set_batch_num_edges(self, val):
        """TBD"""
        if not isinstance(val, Mapping):
            if len(self.etypes) != 1:
                raise DGLError('Must provide a dictionary when there are multiple edge types.')
            val = {self.canonical_etypes[0] : val}
        self._batch_num_edges = val

    #################################################################
    # View
    #################################################################

    @property
    def nodes(self):
        """Return a node view

        One can use it for:

        1. Getting the node IDs for a single node type.
        2. Setting/getting features for all nodes of a single node type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph and a heterogeneous graph of two node types.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })

        Get the node IDs of the homogeneous graph.

        >>> g.nodes()
        tensor([0, 1, 2])

        Get the node IDs of the heterogeneous graph. With multiple node types introduced,
        one needs to specify the node type for query.

        >>> hg.nodes('user')
        tensor([0, 1, 2, 3, 4])

        Set and get a feature 'h' for all nodes of a single type in the heterogeneous graph.

        >>> hg.nodes['user'].data['h'] = torch.ones(5, 1)
        >>> hg.nodes['user'].data['h']
        tensor([[1.], [1.], [1.], [1.], [1.]])

        To set node features for a graph with a single node type, use :func:`DGLGraph.ndata`.

        See Also
        --------
        ndata
        """
        # Todo (Mufei) Replace the syntax g.nodes[...].ndata[...] with g.nodes[...][...]
        return HeteroNodeView(self, self.get_ntype_id)

    @property
    def srcnodes(self):
        """Return a node view for source nodes

        If the graph is a uni-bipartite graph (see :func:`is_unibipartite` for reference),
        this is :func:`nodes` restricted to source node types. Otherwise, it is an alias
        for :func:`nodes`.

        One can use it for:

        1. Getting the node IDs for a single node type.
        2. Setting/getting features for all nodes of a single node type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a uni-bipartite graph.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })

        Get the node IDs for source node types.

        >>> g.srcnodes('user')
        tensor([0])
        >>> g.srcnodes('developer')
        tensor([0, 1])

        Set/get features for source node types.

        >>> g.srcnodes['user'].data['h'] = torch.ones(1, 1)
        >>> g.srcnodes['user'].data['h']
        tensor([[1.]])

        Create a graph that is not uni-bipartite.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })

        :func:`dgl.DGLGraph.srcnodes` falls back to :func:`dgl.DGLGraph.nodes` and one can
        get the node IDs for both source and destination node types.

        >>> g.srcnodes('game')
        tensor([0, 1, 2])

        One can also set/get features for destination node types in this case.

        >>> g.srcnodes['game'].data['h'] = torch.ones(3, 1)
        >>> g.srcnodes['game'].data['h']
        tensor([[1.],
                [1.],
                [1.]])

        See Also
        --------
        srcdata
        """
        return HeteroNodeView(self, self.get_ntype_id_from_src)

    @property
    def dstnodes(self):
        """Return a node view for destination nodes

        If the graph is a uni-bipartite graph (see :func:`is_unibipartite` for reference),
        this is :func:`nodes` restricted to destination node types. Otherwise, it is an alias
        for :func:`nodes`.

        One can use it for:

        1. Getting the node IDs for a single node type.
        2. Setting/getting features for all nodes of a single node type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a uni-bipartite graph.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })

        Get the node IDs for destination node types.

        >>> g.dstnodes('game')
        tensor([0, 1, 2])

        Set/get features for destination node types.

        >>> g.dstnodes['game'].data['h'] = torch.ones(3, 1)
        >>> g.dstnodes['game'].data['h']
        tensor([[1.],
                [1.],
                [1.]])

        Create a graph that is not uni-bipartite.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0]), torch.tensor([1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([1]), torch.tensor([2]))
        ... })

        :func:`dgl.DGLGraph.dstnodes` falls back to :func:`dgl.DGLGraph.nodes` and one can
        get the node IDs for both source and destination node types.

        >>> g.dstnodes('developer')
        tensor([0, 1])

        One can also set/get features for source node types in this case.

        >>> g.dstnodes['developer'].data['h'] = torch.ones(2, 1)
        >>> g.dstnodes['developer'].data['h']
        tensor([[1.],
                [1.]])

        See Also
        --------
        dstdata
        """
        return HeteroNodeView(self, self.get_ntype_id_from_dst)

    @property
    def ndata(self):
        """Return a node data view for setting/getting node features

        Let ``g`` be a DGLGraph. If ``g`` is a graph of a single node type, ``g.ndata[feat]``
        returns the node feature associated with the name ``feat``. One can also set a node
        feature associated with the name ``feat`` by setting ``g.ndata[feat]`` to a tensor.

        If ``g`` is a graph of multiple node types, ``g.ndata[feat]`` returns a
        dict[str, Tensor] mapping node types to the node features associated with the name
        ``feat`` for the corresponding type. One can also set a node feature associated
        with the name ``feat`` for some node type(s) by setting ``g.ndata[feat]`` to a
        dictionary as described.

        Notes
        -----
        For setting features, the device of the features must be the same as the device
        of the graph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Set and get feature 'h' for a graph of a single node type.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.ndata['h'] = torch.ones(3, 1)
        >>> g.ndata['h']
        tensor([[1.],
                [1.],
                [1.]])

        Set and get feature 'h' for a graph of multiple node types.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([1, 2]), torch.tensor([3, 4])),
        ...     ('player', 'plays', 'game'): (torch.tensor([2, 2]), torch.tensor([1, 1]))
        ... })
        >>> g.ndata['h'] = {'game': torch.zeros(2, 1), 'player': torch.ones(3, 1)}
        >>> g.ndata['h']
        {'game': tensor([[0.], [0.]]),
         'player': tensor([[1.], [1.], [1.]])}
        >>> g.ndata['h'] = {'game': torch.ones(2, 1)}
        >>> g.ndata['h']
        {'game': tensor([[1.], [1.]]),
         'player': tensor([[1.], [1.], [1.]])}

        See Also
        --------
        nodes
        """
        if len(self.ntypes) == 1:
            ntid = self.get_ntype_id(None)
            ntype = self.ntypes[0]
            return HeteroNodeDataView(self, ntype, ntid, ALL)
        else:
            ntids = [self.get_ntype_id(ntype) for ntype in self.ntypes]
            ntypes = self.ntypes
            return HeteroNodeDataView(self, ntypes, ntids, ALL)

    @property
    def srcdata(self):
        """Return a node data view for setting/getting source node features.

        Let ``g`` be a DGLGraph. If ``g`` is a graph of a single source node type,
        ``g.srcdata[feat]`` returns the source node feature associated with the name ``feat``.
        One can also set a source node feature associated with the name ``feat`` by
        setting ``g.srcdata[feat]`` to a tensor.

        If ``g`` is a graph of multiple source node types, ``g.srcdata[feat]`` returns a
        dict[str, Tensor] mapping source node types to the node features associated with
        the name ``feat`` for the corresponding type. One can also set a node feature
        associated with the name ``feat`` for some source node type(s) by setting
        ``g.srcdata[feat]`` to a dictionary as described.

        Notes
        -----
        For setting features, the device of the features must be the same as the device
        of the graph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Set and get feature 'h' for a graph of a single source node type.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]), torch.tensor([1, 2]))})
        >>> g.srcdata['h'] = torch.ones(2, 1)
        >>> g.srcdata['h']
        tensor([[1.],
                [1.]])

        Set and get feature 'h' for a graph of multiple source node types.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 2]), torch.tensor([3, 4])),
        ...     ('player', 'plays', 'game'): (torch.tensor([2, 2]), torch.tensor([1, 1]))
        ... })
        >>> g.srcdata['h'] = {'user': torch.zeros(3, 1), 'player': torch.ones(3, 1)}
        >>> g.srcdata['h']
        {'player': tensor([[1.], [1.], [1.]]),
         'user': tensor([[0.], [0.], [0.]])}
        >>> g.srcdata['h'] = {'user': torch.ones(3, 1)}
        >>> g.srcdata['h']
        {'player': tensor([[1.], [1.], [1.]]),
         'user': tensor([[1.], [1.], [1.]])}

        See Also
        --------
        nodes
        ndata
        srcnodes
        """
        if len(self.srctypes) == 1:
            ntype = self.srctypes[0]
            ntid = self.get_ntype_id_from_src(ntype)
            return HeteroNodeDataView(self, ntype, ntid, ALL)
        else:
            ntypes = self.srctypes
            ntids = [self.get_ntype_id_from_src(ntype) for ntype in ntypes]
            return HeteroNodeDataView(self, ntypes, ntids, ALL)

    @property
    def dstdata(self):
        """Return a node data view for setting/getting destination node features.

        Let ``g`` be a DGLGraph. If ``g`` is a graph of a single destination node type,
        ``g.dstdata[feat]`` returns the destination node feature associated with the name
        ``feat``. One can also set a destination node feature associated with the name
        ``feat`` by setting ``g.dstdata[feat]`` to a tensor.

        If ``g`` is a graph of multiple destination node types, ``g.dstdata[feat]`` returns a
        dict[str, Tensor] mapping destination node types to the node features associated with
        the name ``feat`` for the corresponding type. One can also set a node feature
        associated with the name ``feat`` for some destination node type(s) by setting
        ``g.dstdata[feat]`` to a dictionary as described.

        Notes
        -----
        For setting features, the device of the features must be the same as the device
        of the graph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Set and get feature 'h' for a graph of a single destination node type.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]), torch.tensor([1, 2]))})
        >>> g.dstdata['h'] = torch.ones(3, 1)
        >>> g.dstdata['h']
        tensor([[1.],
                [1.],
                [1.]])

        Set and get feature 'h' for a graph of multiple destination node types.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 2]), torch.tensor([1, 2])),
        ...     ('user', 'watches', 'movie'): (torch.tensor([2, 2]), torch.tensor([1, 1]))
        ... })
        >>> g.dstdata['h'] = {'game': torch.zeros(3, 1), 'movie': torch.ones(2, 1)}
        >>> g.dstdata['h']
        {'game': tensor([[0.], [0.], [0.]]),
         'movie': tensor([[1.], [1.]])}
        >>> g.dstdata['h'] = {'game': torch.ones(3, 1)}
        >>> g.dstdata['h']
        {'game': tensor([[1.], [1.], [1.]]),
         'movie': tensor([[1.], [1.]])}

        See Also
        --------
        nodes
        ndata
        dstnodes
        """
        if len(self.dsttypes) == 1:
            ntype = self.dsttypes[0]
            ntid = self.get_ntype_id_from_dst(ntype)
            return HeteroNodeDataView(self, ntype, ntid, ALL)
        else:
            ntypes = self.dsttypes
            ntids = [self.get_ntype_id_from_dst(ntype) for ntype in ntypes]
            return HeteroNodeDataView(self, ntypes, ntids, ALL)

    @property
    def edges(self):
        """Return an edge view

        One can use it for:

        1. Getting the edges for a single edge type. In this case, it can take the
           following optional arguments:

            - form : str, optional
                  The return form, which can be one of the following:

                  - ``'uv'`` (default): The returned result is a 2-tuple of 1D tensors
                    :math:`(U, V)`, representing the source and destination nodes of all edges.
                    For each :math:`i`, :math:`(U[i], V[i])` forms an edge.
                  - ``'eid'``: The returned result is a 1D tensor :math:`EID`, representing
                    the IDs of all edges.
                  - ``'all'``: The returned result is a 3-tuple of 1D tensors :math:`(U, V, EID)`,
                    representing the source nodes, destination nodes and IDs of all edges.
                    For each :math:`i`, :math:`(U[i], V[i])` forms an edge with ID :math:`EID[i]`.
            - order : str, optional
                  The order of the returned edges, which can be one of the following:

                  - ``'eid'`` (default): The edges are sorted by their IDs.
                  - ``'srcdst'``: The edges are sorted first by their source node IDs and then
                    by their destination node IDs to break ties.
            - etype : str or tuple of str, optional
                  The edge type for query, which can be an edge type (str) or a canonical edge
                  type (3-tuple of str). When an edge type appears in multiple canonical edge
                  types, one must use a canonical edge type. If the graph has multiple edge
                  types, one must specify the argument. Otherwise, it can be omitted.
        2. Setting/getting features for all edges of a single edge type. To set/get a feature
           ``feat`` for edges of type ``etype`` in a graph ``g``, one can use
           ``g.edges[etype].data[feat]``.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Get the Edges for a Single Edge Type**

        Create a graph with a single edge type.

        >>> g = dgl.graph((torch.tensor([1, 0, 0]), torch.tensor([1, 1, 0])))
        >>> g.edges()
        (tensor([1, 0, 0]), tensor([1, 1, 0]))

        Specify a different value for :attr:`form` and :attr:`order`.

        >>> g.edges(form='all', order='srcdst')
        (tensor([0, 0, 1]), tensor([0, 1, 1]), tensor([2, 1, 0]))

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.edges(etype='plays')
        (tensor([3, 4]), tensor([5, 6]))

        **Set/get Features for All Edges of a Single Edge Type**

        Create a heterogeneous graph of two edge types.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })

        Set and get a feature 'h' for all edges of a single type in the heterogeneous graph.

        >>> hg.edges['follows'].data['h'] = torch.ones(2, 1)
        >>> hg.edges['follows'].data['h']
        tensor([[1.], [1.]])

        To set edge features for a graph with a single edge type, use :func:`DGLGraph.edata`.

        See Also
        --------
        edata
        """
        # TODO(Mufei): Replace the syntax g.edges[...].edata[...] with g.edges[...][...]
        return HeteroEdgeView(self)

    @property
    def edata(self):
        """Return an edge data view for setting/getting edge features.

        Let ``g`` be a DGLGraph. If ``g`` is a graph of a single edge type, ``g.edata[feat]``
        returns the edge feature associated with the name ``feat``. One can also set an
        edge feature associated with the name ``feat`` by setting ``g.edata[feat]`` to a tensor.

        If ``g`` is a graph of multiple edge types, ``g.edata[feat]`` returns a
        dict[str, Tensor] mapping canonical edge types to the edge features associated with
        the name ``feat`` for the corresponding type. One can also set an edge feature
        associated with the name ``feat`` for some edge type(s) by setting
        ``g.edata[feat]`` to a dictionary as described.

        Notes
        -----
        For setting features, the device of the features must be the same as the device
        of the graph.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Set and get feature 'h' for a graph of a single edge type.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.edata['h'] = torch.ones(2, 1)
        >>> g.edata['h']
        tensor([[1.],
                [1.]])

        Set and get feature 'h' for a graph of multiple edge types.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([1, 2]), torch.tensor([3, 4])),
        ...     ('user', 'plays', 'user'): (torch.tensor([2, 2]), torch.tensor([1, 1])),
        ...     ('player', 'plays', 'game'): (torch.tensor([2, 2]), torch.tensor([1, 1]))
        ... })
        >>> g.edata['h'] = {('user', 'follows', 'user'): torch.zeros(2, 1),
        ...                 ('user', 'plays', 'user'): torch.ones(2, 1)}
        >>> g.edata['h']
        {('user', 'follows', 'user'): tensor([[0.], [0.]]),
         ('user', 'plays', 'user'): tensor([[1.], [1.]])}
        >>> g.edata['h'] = {('user', 'follows', 'user'): torch.ones(2, 1)}
        >>> g.edata['h']
        {('user', 'follows', 'user'): tensor([[1.], [1.]]),
         ('user', 'plays', 'user'): tensor([[1.], [1.]])}

        See Also
        --------
        edges
        """
        if len(self.canonical_etypes) == 1:
            return HeteroEdgeDataView(self, None, ALL)
        else:
            return HeteroEdgeDataView(self, self.canonical_etypes, ALL)

    def _find_etypes(self, key):
        etypes = [
            i for i, (srctype, etype, dsttype) in enumerate(self._canonical_etypes) if
            (key[0] == SLICE_FULL or key[0] == srctype) and
            (key[1] == SLICE_FULL or key[1] == etype) and
            (key[2] == SLICE_FULL or key[2] == dsttype)]
        return etypes

    def __getitem__(self, key):
        """Return the relation slice of this graph.

        You can get a relation slice with ``self[srctype, etype, dsttype]``, where
        ``srctype``, ``etype``, and ``dsttype`` can be either a string or a full
        slice (``:``) representing wildcard (i.e. any source/edge/destination type).

        A relation slice is a homogeneous (with one node type and one edge type) or
        bipartite (with two node types and one edge type) graph, transformed from
        the original heterogeneous graph.

        If there is only one canonical edge type found, then the returned relation
        slice would be a subgraph induced from the original graph.  That is, it is
        equivalent to ``self.edge_type_subgraph(etype)``.  The node and edge features
        of the returned graph would be shared with thew original graph.

        If there are multiple canonical edge types found, then the source/edge/destination
        node types would be a *concatenation* of original node/edge types.  The
        new source/destination node type would have the concatenation determined by
        :func:`dgl.combine_names() <dgl.combine_names>` called on original source/destination
        types as its name.  The source/destination node would be formed by concatenating the
        common features of the original source/destination types.  Therefore they are not
        shared with the original graph.  Edge type is similar.

        Parameters
        ----------
        key : str or tuple
            Either a string representing the edge type name, or a tuple in the form of
            ``(srctype, etype, dsttype)`` where ``srctype``, ``etype``, ``dsttype`` can be either
            strings representing type names or a full slice object (`:`).

        Returns
        -------
        DGLGraph
            The relation slice.

        Notes
        -----
        This function returns a new graph.  Changing the content of this graph does not reflect
        onto the original graph.

        If the graph combines multiple node types or edge types together, it will have the
        mapping of node/edge types and IDs from the new graph to the original graph.
        The mappings have the name ``dgl.NTYPE``, ``dgl.NID``, ``dgl.ETYPE`` and ``dgl.EID``,
        similar to the function :func:`dgl.to_homogenenous`.

        Examples
        --------
        >>> g = dgl.heterograph({
        ...     ('A1', 'AB1', 'B'): ([0, 1, 2], [1, 2, 3]),
        ...     ('A1', 'AB2', 'B'): ([1, 2, 3], [3, 4, 5]),
        ...     ('A2', 'AB2', 'B'): ([1, 3, 5], [2, 4, 6])})
        >>> new_g = g['A1', :, 'B']         # combines all edge types between A1 and B
        >>> new_g
        Graph(num_nodes={'A1': 4, 'B': 7},
              num_edges={('A1', 'AB1+AB2', 'B'): 6},
              metagraph=[('A1', 'B', 'AB1+AB2')])
        >>> new_g.edges()
        (tensor([0, 1, 2, 1, 2, 3]), tensor([1, 2, 3, 3, 4, 5]))
        >>> new_g2 = g[:, 'AB2', 'B']        # combines all node types that are source of AB2
        >>> new_g2
        Graph(num_nodes={'A1+A2': 10, 'B': 7},
              num_edges={('A1+A2', 'AB2+AB2', 'B'): 6},
              metagraph=[('A1+A2', 'B', 'AB2+AB2')])
        >>> new_g2.edges()
        (tensor([1, 2, 3, 5, 7, 9]), tensor([3, 4, 5, 2, 4, 6]))

        If a combination of multiple node types and edge types occur, one can find
        the mapping to the original node type and IDs like the following:

        >>> new_g1.edges['AB1+AB2'].data[dgl.EID]
        tensor([0, 1, 2, 0, 1, 2])
        >>> new_g1.edges['AB1+AB2'].data[dgl.ETYPE]
        tensor([0, 0, 0, 1, 1, 1])
        >>> new_g2.nodes['A1+A2'].data[dgl.NID]
        tensor([0, 1, 2, 3, 0, 1, 2, 3, 4, 5])
        >>> new_g2.nodes['A1+A2'].data[dgl.NTYPE]
        tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        """
        err_msg = "Invalid slice syntax. Use G['etype'] or G['srctype', 'etype', 'dsttype'] " +\
                  "to get view of one relation type. Use : to slice multiple types (e.g. " +\
                  "G['srctype', :, 'dsttype'])."

        orig_key = key
        if not isinstance(key, tuple):
            key = (SLICE_FULL, key, SLICE_FULL)

        if len(key) != 3:
            raise DGLError(err_msg)

        etypes = self._find_etypes(key)

        if len(etypes) == 0:
            raise DGLError('Invalid key "{}". Must be one of the edge types.'.format(orig_key))

        if len(etypes) == 1:
            # no ambiguity: return the unitgraph itself
            srctype, etype, dsttype = self._canonical_etypes[etypes[0]]
            stid = self.get_ntype_id_from_src(srctype)
            etid = self.get_etype_id((srctype, etype, dsttype))
            dtid = self.get_ntype_id_from_dst(dsttype)
            new_g = self._graph.get_relation_graph(etid)

            if stid == dtid:
                new_ntypes = [srctype]
                new_nframes = [self._node_frames[stid]]
            else:
                new_ntypes = ([srctype], [dsttype])
                new_nframes = [self._node_frames[stid], self._node_frames[dtid]]
            new_etypes = [etype]
            new_eframes = [self._edge_frames[etid]]

            return self.__class__(new_g, new_ntypes, new_etypes, new_nframes, new_eframes)
        else:
            flat = self._graph.flatten_relations(etypes)
            new_g = flat.graph

            # merge frames
            stids = flat.induced_srctype_set.asnumpy()
            dtids = flat.induced_dsttype_set.asnumpy()
            etids = flat.induced_etype_set.asnumpy()
            new_ntypes = [combine_names(self.ntypes, stids)]
            if new_g.number_of_ntypes() == 2:
                new_ntypes.append(combine_names(self.ntypes, dtids))
                new_nframes = [
                    combine_frames(self._node_frames, stids),
                    combine_frames(self._node_frames, dtids)]
            else:
                assert np.array_equal(stids, dtids)
                new_nframes = [combine_frames(self._node_frames, stids)]
            new_etypes = [combine_names(self.etypes, etids)]
            new_eframes = [combine_frames(self._edge_frames, etids)]

            # create new heterograph
            new_hg = self.__class__(new_g, new_ntypes, new_etypes, new_nframes, new_eframes)

            src = new_ntypes[0]
            dst = new_ntypes[1] if new_g.number_of_ntypes() == 2 else src
            # put the parent node/edge type and IDs
            new_hg.nodes[src].data[NTYPE] = F.zerocopy_from_dgl_ndarray(flat.induced_srctype)
            new_hg.nodes[src].data[NID] = F.zerocopy_from_dgl_ndarray(flat.induced_srcid)
            new_hg.nodes[dst].data[NTYPE] = F.zerocopy_from_dgl_ndarray(flat.induced_dsttype)
            new_hg.nodes[dst].data[NID] = F.zerocopy_from_dgl_ndarray(flat.induced_dstid)
            new_hg.edata[ETYPE] = F.zerocopy_from_dgl_ndarray(flat.induced_etype)
            new_hg.edata[EID] = F.zerocopy_from_dgl_ndarray(flat.induced_eid)

            return new_hg

    #################################################################
    # Graph query
    #################################################################

    def number_of_nodes(self, ntype=None):
        """Alias of :meth:`num_nodes`"""
        return self.num_nodes(ntype)

    def num_nodes(self, ntype=None):
        """Return the number of nodes in the graph.

        Parameters
        ----------
        ntype : str, optional
            The node type name. If given, it returns the number of nodes of the
            type. If not given (default), it returns the total number of nodes of all types.

        Returns
        -------
        int
            The number of nodes.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a graph with two node types -- 'user' and 'game'.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })

        Query for the number of nodes.

        >>> g.num_nodes('user')
        5
        >>> g.num_nodes('game')
        7
        >>> g.num_nodes()
        12
        """
        if ntype is None:
            return sum([self._graph.number_of_nodes(ntid) for ntid in range(len(self.ntypes))])
        else:
            return self._graph.number_of_nodes(self.get_ntype_id(ntype))

    def number_of_src_nodes(self, ntype=None):
        """Alias of :meth:`num_src_nodes`"""
        return self.num_src_nodes(ntype)

    def num_src_nodes(self, ntype=None):
        """Return the number of source nodes in the graph.

        If the graph can further divide its node types into two subsets A and B where
        all the edeges are from nodes of types in A to nodes of types in B, we call
        this graph a *uni-bipartite* graph and the nodes in A being the *source*
        nodes and the ones in B being the *destination* nodes. If the graph is not
        uni-bipartite, the source and destination nodes are just the entire set of
        nodes in the graph.

        Parameters
        ----------
        ntype : str, optional
            The source node type name. If given, it returns the number of nodes for
            the source node type. If not given (default), it returns the number of
            nodes summed over all source node types.

        Returns
        -------
        int
            The number of nodes

        See Also
        --------
        num_dst_nodes
        is_unibipartite

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph for query.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.num_src_nodes()
        3

        Create a heterogeneous graph with two source node types -- 'developer' and 'user'.

        >>> g = dgl.heterograph({
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })

        Query for the number of nodes.

        >>> g.num_src_nodes('developer')
        2
        >>> g.num_src_nodes('user')
        5
        >>> g.num_src_nodes()
        7
        """
        if ntype is None:
            return sum([self._graph.number_of_nodes(self.get_ntype_id_from_src(nty))
                        for nty in self.srctypes])
        else:
            return self._graph.number_of_nodes(self.get_ntype_id_from_src(ntype))

    def number_of_dst_nodes(self, ntype=None):
        """Alias of :func:`num_dst_nodes`"""
        return self.num_dst_nodes(ntype)

    def num_dst_nodes(self, ntype=None):
        """Return the number of destination nodes in the graph.

        If the graph can further divide its node types into two subsets A and B where
        all the edeges are from nodes of types in A to nodes of types in B, we call
        this graph a *uni-bipartite* graph and the nodes in A being the *source*
        nodes and the ones in B being the *destination* nodes. If the graph is not
        uni-bipartite, the source and destination nodes are just the entire set of
        nodes in the graph.

        Parameters
        ----------
        ntype : str, optional
            The destination node type name. If given, it returns the number of nodes of
            the destination node type. If not given (default), it returns the number of
            nodes summed over all the destination node types.

        Returns
        -------
        int
            The number of nodes

        See Also
        --------
        num_src_nodes
        is_unibipartite

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph for query.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.num_dst_nodes()
        3

        Create a heterogeneous graph with two destination node types -- 'user' and 'game'.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })

        Query for the number of nodes.

        >>> g.num_dst_nodes('user')
        5
        >>> g.num_dst_nodes('game')
        7
        >>> g.num_dst_nodes()
        12
        """
        if ntype is None:
            return sum([self._graph.number_of_nodes(self.get_ntype_id_from_dst(nty))
                        for nty in self.dsttypes])
        else:
            return self._graph.number_of_nodes(self.get_ntype_id_from_dst(ntype))

    def number_of_edges(self, etype=None):
        """Alias of :func:`num_edges`"""
        return self.num_edges(etype)

    def num_edges(self, etype=None):
        """Return the number of edges in the graph.

        Parameters
        ----------
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            If not provided, return the total number of edges regardless of the types
            in the graph.

        Returns
        -------
        int
            The number of edges.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a graph with three canonical edge types.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })

        Query for the number of edges.

        >>> g.num_edges('plays')
        2
        >>> g.num_edges()
        7

        Use a canonical edge type instead when there is ambiguity for an edge type.

        >>> g.num_edges(('user', 'follows', 'user'))
        2
        >>> g.num_edges(('user', 'follows', 'game'))
        3
        """
        if etype is None:
            return sum([self._graph.number_of_edges(etid)
                        for etid in range(len(self.canonical_etypes))])
        else:
            return self._graph.number_of_edges(self.get_etype_id(etype))

    def __len__(self):
        """Deprecated: please directly call :func:`number_of_nodes`
        """
        dgl_warning('DGLGraph.__len__ is deprecated.'
                    'Please directly call DGLGraph.number_of_nodes.')
        return self.number_of_nodes()

    @property
    def is_multigraph(self):
        """Return whether the graph is a multigraph with parallel edges.

        A multigraph has more than one edges between the same pair of nodes, called
        *parallel edges*.  For heterogeneous graphs, parallel edge further requires
        the canonical edge type to be the same (see :meth:`canonical_etypes` for the
        definition).

        Returns
        -------
        bool
            True if the graph is a multigraph.

        Notes
        -----
        Checking whether the graph is a multigraph could be expensive for a large one.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Check for homogeneous graphs.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 3])))
        >>> g.is_multigraph
        False
        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([1, 3, 3])))
        >>> g.is_multigraph
        True

        Check for heterogeneous graphs.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
        ... })
        >>> g.is_multigraph
        False
        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1, 1]), torch.tensor([1, 2, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))
        ... })
        >>> g.is_multigraph
        True
        """
        return self._graph.is_multigraph()

    @property
    def is_homogeneous(self):
        """Return whether the graph is a homogeneous graph.

        A homogeneous graph only has one node type and one edge type.

        Returns
        -------
        bool
            True if the graph is a homogeneous graph.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph for check.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))
        >>> g.is_homogeneous
        True

        Create a heterogeneous graph for check.

        If the graph has multiple edge types, one need to specify the edge type.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3]))})
        >>> g.is_homogeneous
        False
        """
        return len(self.ntypes) == 1 and len(self.etypes) == 1

    @property
    def is_readonly(self):
        """**DEPRECATED**: DGLGraph will always be mutable.

        Returns
        -------
        bool
            True if the graph is readonly, False otherwise.
        """
        dgl_warning('DGLGraph.is_readonly is deprecated in v0.5.\n'
                    'DGLGraph now always supports mutable operations like add_nodes'
                    ' and add_edges.')
        return False

    @property
    def idtype(self):
        """The data type for storing the structure-related graph information
        such as node and edge IDs.

        Returns
        -------
        Framework-specific device object
            For example, this can be ``torch.int32`` or ``torch.int64`` for PyTorch.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> src_ids = torch.tensor([0, 0, 1])
        >>> dst_ids = torch.tensor([1, 2, 2])
        >>> g = dgl.graph((src_ids, dst_ids))
        >>> g.idtype
        torch.int64
        >>> g = dgl.graph((src_ids, dst_ids), idtype=torch.int32)
        >>> g.idtype
        torch.int32

        See Also
        --------
        long
        int
        """
        return getattr(F, self._graph.dtype)

    @property
    def _idtype_str(self):
        """The dtype of graph index

        Returns
        -------
        backend dtype object
            th.int32/th.int64 or tf.int32/tf.int64 etc.
        """
        return self._graph.dtype

    def __contains__(self, vid):
        """**DEPRECATED**: please directly call :func:`has_nodes`."""
        dgl_warning('DGLGraph.__contains__ is deprecated.'
                    ' Please directly call has_nodes.')
        return self.has_nodes(vid)

    def has_nodes(self, vid, ntype=None):
        """Return whether the graph contains the given nodes.

        Parameters
        ----------
        vid : node ID(s)
            The nodes IDs. The allowed nodes ID formats are:

            * ``int``: The ID of a single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

        ntype : str, optional
            The node type name. Can be omitted if there is
            only one type of nodes in the graph.

        Returns
        -------
        bool or bool Tensor
            A tensor of bool flags where each element is True if the node is in the graph.
            If the input is a single node, return one bool value.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a graph with two node types -- 'user' and 'game'.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([0, 1]))
        ... })

        Query for the nodes.

        >>> g.has_nodes(0, 'user')
        True
        >>> g.has_nodes(3, 'game')
        False
        >>> g.has_nodes(torch.tensor([3, 0, 1]), 'game')
        tensor([False,  True,  True])
        """
        vid_tensor = utils.prepare_tensor(self, vid, "vid")
        if len(vid_tensor) > 0 and F.as_scalar(F.min(vid_tensor, 0)) < 0 < len(vid_tensor):
            raise DGLError('All IDs must be non-negative integers.')
        ret = self._graph.has_nodes(
            self.get_ntype_id(ntype), vid_tensor)
        if isinstance(vid, numbers.Integral):
            return bool(F.as_scalar(ret))
        else:
            return F.astype(ret, F.bool)

    def has_node(self, vid, ntype=None):
        """Whether the graph has a particular node of a given type.

        **DEPRECATED**: see :func:`~DGLGraph.has_nodes`
        """
        dgl_warning("DGLGraph.has_node is deprecated. Please use DGLGraph.has_nodes")
        return self.has_nodes(vid, ntype)

    def has_edges_between(self, u, v, etype=None):
        """Return whether the graph contains the given edges.

        Parameters
        ----------
        u : node IDs
            The source node IDs of the edges. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

        v : node IDs
            The destination node IDs of the edges. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.


        Returns
        -------
        bool or bool Tensor
            A tensor of bool flags where each element is True if the node is in the graph.
            If the input is a single node, return one bool value.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))

        Query for the edges.

        >>> g.has_edges_between(1, 2)
        True
        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]))
        tensor([ True, False])

        If the graph has multiple edge types, one need to specify the edge type.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]), 'plays')
        tensor([ True, False])

        Use a canonical edge type instead when there is ambiguity for an edge type.

        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]),
        ...                     ('user', 'follows', 'user'))
        tensor([ True, False])
        >>> g.has_edges_between(torch.tensor([1, 2]), torch.tensor([2, 3]),
        ...                     ('user', 'follows', 'game'))
        tensor([True, True])
        """
        srctype, _, dsttype = self.to_canonical_etype(etype)
        u_tensor = utils.prepare_tensor(self, u, 'u')
        if F.as_scalar(F.sum(self.has_nodes(u_tensor, ntype=srctype), dim=0)) != len(u_tensor):
            raise DGLError('u contains invalid node IDs')
        v_tensor = utils.prepare_tensor(self, v, 'v')
        if F.as_scalar(F.sum(self.has_nodes(v_tensor, ntype=dsttype), dim=0)) != len(v_tensor):
            raise DGLError('v contains invalid node IDs')
        ret = self._graph.has_edges_between(
            self.get_etype_id(etype),
            u_tensor, v_tensor)
        if isinstance(u, numbers.Integral) and isinstance(v, numbers.Integral):
            return bool(F.as_scalar(ret))
        else:
            return F.astype(ret, F.bool)

    def has_edge_between(self, u, v, etype=None):
        """Whether the graph has edges of type ``etype``.

        **DEPRECATED**: please use :func:`~DGLGraph.has_edge_between`.
        """
        dgl_warning("DGLGraph.has_edge_between is deprecated. "
                    "Please use DGLGraph.has_edges_between")
        return self.has_edges_between(u, v, etype)

    def predecessors(self, v, etype=None):
        """Return the predecessor(s) of a particular node with the specified edge type.

        Node ``u`` is a predecessor of node ``v`` if there is an edge ``(u, v)`` with type
        ``etype`` in the graph.

        Parameters
        ----------
        v : int
            The node ID. If the graph has multiple edge types, the ID is for the destination
            type corresponding to the edge type.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.


        Returns
        -------
        Tensor
            The predecessors of :attr:`v` with the specified edge type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 2, 3])))

        Query for node 1.

        >>> g.predecessors(1)
        tensor([0, 0])

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.predecessors(1, etype='follows')
        tensor([0])

        See Also
        --------
        successors
        """
        if not self.has_nodes(v, self.to_canonical_etype(etype)[-1]):
            raise DGLError('Non-existing node ID {}'.format(v))
        return self._graph.predecessors(self.get_etype_id(etype), v)

    def successors(self, v, etype=None):
        """Return the successor(s) of a particular node with the specified edge type.

        Node ``u`` is a successor of node ``v`` if there is an edge ``(v, u)`` with type
        ``etype`` in the graph.

        Parameters
        ----------
        v : int
            The node ID. If the graph has multiple edge types, the ID is for the source
            type corresponding to the edge type.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor
            The successors of :attr:`v` with the specified edge type.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 2, 3])))

        Query for node 1.

        >>> g.successors(1)
        tensor([2, 3])

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.successors(1, etype='follows')
        tensor([2])

        See Also
        --------
        predecessors
        """
        if not self.has_nodes(v, self.to_canonical_etype(etype)[0]):
            raise DGLError('Non-existing node ID {}'.format(v))
        return self._graph.successors(self.get_etype_id(etype), v)

    def edge_id(self, u, v, force_multi=None, return_uv=False, etype=None):
        """Return the edge ID, or an array of edge IDs, between source node
        `u` and destination node `v`, with the specified edge type

        **DEPRECATED**: See edge_ids
        """
        dgl_warning("DGLGraph.edge_id is deprecated. Please use DGLGraph.edge_ids.")
        return self.edge_ids(u, v, force_multi=force_multi,
                             return_uv=return_uv, etype=etype)

    def edge_ids(self, u, v, force_multi=None, return_uv=False, etype=None):
        """Return the edge ID(s) given the two endpoints of the edge(s).

        Parameters
        ----------
        u : node IDs
            The source node IDs of the edges. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

        v : node IDs
            The destination node IDs of the edges. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
        force_multi : bool, optional
            **DEPRECATED**, use :attr:`return_uv` instead. Whether to allow the graph to be a
            multigraph, i.e. there can be multiple edges from one node to another.
        return_uv : bool, optional
            Whether to return the source and destination node IDs along with the edges. If
            False (default), it assumes that the graph is a simple graph and there is only
            one edge from one node to another. If True, there can be multiple edges found
            from one node to another.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor, or (Tensor, Tensor, Tensor)

            * If ``return_uv=False``, it returns the edge IDs in a tensor, where the i-th
              element is the ID of the edge ``(u[i], v[i])``.
            * If ``return_uv=True``, it returns a tuple of three 1D tensors ``(eu, ev, e)``.
              ``e[i]`` is the ID of an edge from ``eu[i]`` to ``ev[i]``. It returns all edges
              (including parallel edges) from ``eu[i]`` to ``ev[i]`` in this case.

        Notes
        -----
        If the graph is a simple graph, ``return_uv=False``, and there are no edges
        between some pairs of node(s), it will raise an error.

        If the graph is a multigraph, ``return_uv=False``, and there are multiple edges
        between some pairs of node(s), it returns an arbitrary one from them.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1, 1]), torch.tensor([1, 0, 2, 3, 2])))

        Query for the edges.

        >>> g.edge_ids(0, 0)
        1
        >>> g.edge_ids(torch.tensor([1, 0]), torch.tensor([3, 1]))
        tensor([3, 0])

        Get all edges for pairs of nodes.

        >>> g.edge_ids(torch.tensor([1, 0]), torch.tensor([3, 1]), return_uv=True)
        (tensor([1, 0]), tensor([3, 1]), tensor([3, 0]))

        If the graph has multiple edge types, one need to specify the edge type.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'follows', 'game'): (torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])),
        ...     ('user', 'plays', 'game'): (torch.tensor([1, 3]), torch.tensor([2, 3]))
        ... })
        >>> g.edge_ids(torch.tensor([1]), torch.tensor([2]), etype='plays')
        tensor([0])

        Use a canonical edge type instead when there is ambiguity for an edge type.

        >>> g.edge_ids(torch.tensor([0, 1]), torch.tensor([1, 2]),
        ...            etype=('user', 'follows', 'user'))
        tensor([0, 1])
        >>> g.edge_ids(torch.tensor([1, 2]), torch.tensor([2, 3]),
        ...            etype=('user', 'follows', 'game'))
        tensor([1, 2])
        """
        is_int = isinstance(u, numbers.Integral) and isinstance(v, numbers.Integral)
        srctype, _, dsttype = self.to_canonical_etype(etype)
        u = utils.prepare_tensor(self, u, 'u')
        if F.as_scalar(F.sum(self.has_nodes(u, ntype=srctype), dim=0)) != len(u):
            raise DGLError('u contains invalid node IDs')
        v = utils.prepare_tensor(self, v, 'v')
        if F.as_scalar(F.sum(self.has_nodes(v, ntype=dsttype), dim=0)) != len(v):
            raise DGLError('v contains invalid node IDs')
        if force_multi is not None:
            dgl_warning("force_multi will be deprecated, " \
                        "Please use return_uv instead")
            return_uv = force_multi

        if return_uv:
            return self._graph.edge_ids_all(self.get_etype_id(etype), u, v)
        else:
            eid = self._graph.edge_ids_one(self.get_etype_id(etype), u, v)
            is_neg_one = F.equal(eid, -1)
            if F.as_scalar(F.sum(is_neg_one, 0)):
                # Raise error since some (u, v) pair is not a valid edge.
                idx = F.nonzero_1d(is_neg_one)
                raise DGLError("Error: (%d, %d) does not form a valid edge." % (
                    F.as_scalar(F.gather_row(u, idx)),
                    F.as_scalar(F.gather_row(v, idx))))
            return F.as_scalar(eid) if is_int else eid

    def find_edges(self, eid, etype=None):
        """Return the source and destination node ID(s) given the edge ID(s).

        Parameters
        ----------
        eid : edge ID(s)
            The edge IDs. The allowed formats are:

            * ``int``: A single ID.
            * Int Tensor: Each element is an ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is an ID.

        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor
            The source node IDs of the edges. The i-th element is the source node ID of
            the i-th edge.
        Tensor
            The destination node IDs of the edges. The i-th element is the destination node
            ID of the i-th edge.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))

        Find edges of IDs 0 and 2.

        >>> g.find_edges(torch.tensor([0, 2]))
        (tensor([0, 1]), tensor([1, 2]))

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.find_edges(torch.tensor([1, 0]), 'plays')
        (tensor([4, 3]), tensor([6, 5]))
        """
        eid = utils.prepare_tensor(self, eid, 'eid')
        if len(eid) > 0:
            min_eid = F.as_scalar(F.min(eid, 0))
            if min_eid < 0:
                raise DGLError('Invalid edge ID {:d}'.format(min_eid))
            max_eid = F.as_scalar(F.max(eid, 0))
            if max_eid >= self.num_edges(etype):
                raise DGLError('Invalid edge ID {:d}'.format(max_eid))

        if len(eid) == 0:
            empty = F.copy_to(F.tensor([], self.idtype), self.device)
            return empty, empty
        src, dst, _ = self._graph.find_edges(self.get_etype_id(etype), eid)
        return src, dst

    def in_edges(self, v, form='uv', etype=None):
        """Return the incoming edges of the given nodes.

        Parameters
        ----------
        v : node ID(s)
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
        form : str, optional
            The result format, which can be one of the following:

            - ``'eid'``: The returned result is a 1D tensor :math:`EID`, representing
              the IDs of all edges.
            - ``'uv'`` (default): The returned result is a 2-tuple of 1D tensors :math:`(U, V)`,
              representing the source and destination nodes of all edges. For each :math:`i`,
              :math:`(U[i], V[i])` forms an edge.
            - ``'all'``: The returned result is a 3-tuple of 1D tensors :math:`(U, V, EID)`,
              representing the source nodes, destination nodes and IDs of all edges.
              For each :math:`i`, :math:`(U[i], V[i])` forms an edge with ID :math:`EID[i]`.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor or (Tensor, Tensor) or (Tensor, Tensor, Tensor)
            All incoming edges of the nodes with the specified type. For a description of the
            returned result, see the description of :attr:`form`.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))

        Query for the nodes 1 and 0.

        >>> g.in_edges(torch.tensor([1, 0]))
        (tensor([0, 0]), tensor([1, 0]))

        Specify a different value for :attr:`form`.

        >>> g.in_edges(torch.tensor([1, 0]), form='all')
        (tensor([0, 0]), tensor([1, 0]), tensor([0, 1]))

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.in_edges(torch.tensor([1, 0]), etype='follows')
        (tensor([0]), tensor([1]))

        See Also
        --------
        edges
        out_edges
        """
        v = utils.prepare_tensor(self, v, 'v')
        src, dst, eid = self._graph.in_edges(self.get_etype_id(etype), v)
        if form == 'all':
            return src, dst, eid
        elif form == 'uv':
            return src, dst
        elif form == 'eid':
            return eid
        else:
            raise DGLError('Invalid form: {}. Must be "all", "uv" or "eid".'.format(form))

    def out_edges(self, u, form='uv', etype=None):
        """Return the outgoing edges of the given nodes.

        Parameters
        ----------
        u : node ID(s)
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.
        form : str, optional
            The return form, which can be one of the following:

            - ``'eid'``: The returned result is a 1D tensor :math:`EID`, representing
              the IDs of all edges.
            - ``'uv'`` (default): The returned result is a 2-tuple of 1D tensors :math:`(U, V)`,
              representing the source and destination nodes of all edges. For each :math:`i`,
              :math:`(U[i], V[i])` forms an edge.
            - ``'all'``: The returned result is a 3-tuple of 1D tensors :math:`(U, V, EID)`,
              representing the source nodes, destination nodes and IDs of all edges.
              For each :math:`i`, :math:`(U[i], V[i])` forms an edge with ID :math:`EID[i]`.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor or (Tensor, Tensor) or (Tensor, Tensor, Tensor)
            All outgoing edges of the nodes with the specified type. For a description of the
            returned result, see the description of :attr:`form`.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))

        Query for the nodes 1 and 2.

        >>> g.out_edges(torch.tensor([1, 2]))
        (tensor([1, 1]), tensor([2, 3]))

        Specify a different value for :attr:`form`.

        >>> g.out_edges(torch.tensor([1, 2]), form='all')
        (tensor([1, 1]), tensor([2, 3]), tensor([2, 3]))

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.out_edges(torch.tensor([1, 2]), etype='follows')
        (tensor([1]), tensor([2]))

        See Also
        --------
        edges
        in_edges
        """
        u = utils.prepare_tensor(self, u, 'u')
        srctype, _, _ = self.to_canonical_etype(etype)
        if F.as_scalar(F.sum(self.has_nodes(u, ntype=srctype), dim=0)) != len(u):
            raise DGLError('u contains invalid node IDs')
        src, dst, eid = self._graph.out_edges(self.get_etype_id(etype), u)
        if form == 'all':
            return src, dst, eid
        elif form == 'uv':
            return src, dst
        elif form == 'eid':
            return eid
        else:
            raise DGLError('Invalid form: {}. Must be "all", "uv" or "eid".'.format(form))

    def all_edges(self, form='uv', order='eid', etype=None):
        """Return all edges with the specified edge type.

        Parameters
        ----------
        form : str, optional
            The return form, which can be one of the following:

            - ``'eid'``: The returned result is a 1D tensor :math:`EID`, representing
              the IDs of all edges.
            - ``'uv'`` (default): The returned result is a 2-tuple of 1D tensors :math:`(U, V)`,
              representing the source and destination nodes of all edges. For each :math:`i`,
              :math:`(U[i], V[i])` forms an edge.
            - ``'all'``: The returned result is a 3-tuple of 1D tensors :math:`(U, V, EID)`,
              representing the source nodes, destination nodes and IDs of all edges.
              For each :math:`i`, :math:`(U[i], V[i])` forms an edge with ID :math:`EID[i]`.
        order : str, optional
            The order of the returned edges, which can be one of the following:

            - ``'srcdst'``: The edges are sorted first by their source node IDs and then
              by their destination node IDs to break ties.
            - ``'eid'`` (default): The edges are sorted by their IDs.
        etype : str or tuple of str, optional
            The edge type for query, which can be an edge type (str) or a canonical edge type
            (3-tuple of str). When an edge type appears in multiple canonical edge types, one
            must use a canonical edge type. If the graph has multiple edge types, one must
            specify the argument. Otherwise, it can be omitted.

        Returns
        -------
        Tensor or (Tensor, Tensor) or (Tensor, Tensor, Tensor)
            All edges of the specified edge type. For a description of the returned result,
            see the description of :attr:`form`.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 0, 2, 3])))

        Query for edges.

        >>> g.all_edges()
        (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))

        Specify a different value for :attr:`form` and :attr:`order`.

        >>> g.all_edges(form='all', order='srcdst')
        (tensor([0, 0, 1, 1]), tensor([0, 1, 2, 3]), tensor([1, 0, 2, 3]))

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.all_edges(etype='plays')
        (tensor([3, 4]), tensor([5, 6]))

        See Also
        --------
        edges
        in_edges
        out_edges
        """
        src, dst, eid = self._graph.edges(self.get_etype_id(etype), order)
        if form == 'all':
            return src, dst, eid
        elif form == 'uv':
            return src, dst
        elif form == 'eid':
            return eid
        else:
            raise DGLError('Invalid form: {}. Must be "all", "uv" or "eid".'.format(form))

    def in_degree(self, v, etype=None):
        """Return the in-degree of node ``v`` with edges of type ``etype``.

        **DEPRECATED**: Please use in_degrees
        """
        dgl_warning("DGLGraph.in_degree is deprecated. Please use DGLGraph.in_degrees")
        return self.in_degrees(v, etype)

    def in_degrees(self, v=ALL, etype=None):
        """Return the in-degree(s) of the given nodes.

        It computes the in-degree(s) w.r.t. to the edges of the given edge type.

        Parameters
        ----------
        v : node IDs
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

            If not given, return the in-degrees of all the nodes.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        int or Tensor
            The in-degree(s) of the node(s) in a Tensor. The i-th element is the in-degree
            of the i-th input node. If :attr:`v` is an ``int``, return an ``int`` too.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 2, 3])))

        Query for all nodes.

        >>> g.in_degrees()
        tensor([0, 2, 1, 1])

        Query for nodes 1 and 2.

        >>> g.in_degrees(torch.tensor([1, 2]))
        tensor([2, 1])

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.in_degrees(torch.tensor([1, 0]), etype='follows')
        tensor([1, 0])

        See Also
        --------
        out_degrees
        """
        dsttype = self.to_canonical_etype(etype)[2]
        etid = self.get_etype_id(etype)
        if is_all(v):
            v = self.dstnodes(dsttype)
        
        v_tensor = utils.prepare_tensor(self, v, 'v')
        deg = self._graph.in_degrees(etid, v_tensor)
        if isinstance(v, numbers.Integral):
            return F.as_scalar(deg)
        else:
            return deg

    def out_degree(self, u, etype=None):
        """Return the out-degree of node `u` with edges of type ``etype``.

        DEPRECATED: please use DGL.out_degrees
        """
        dgl_warning("DGLGraph.out_degree is deprecated. Please use DGLGraph.out_degrees")
        return self.out_degrees(u, etype)

    def out_degrees(self, u=ALL, etype=None):
        """Return the out-degree(s) of the given nodes.

        It computes the out-degree(s) w.r.t. to the edges of the given edge type.

        Parameters
        ----------
        u : node IDs
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

            If not given, return the in-degrees of all the nodes.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        int or Tensor
            The out-degree(s) of the node(s) in a Tensor. The i-th element is the out-degree
            of the i-th input node. If :attr:`v` is an ``int``, return an ``int`` too.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 2, 3])))

        Query for all nodes.

        >>> g.out_degrees()
        tensor([2, 2, 0, 0])

        Query for nodes 1 and 2.

        >>> g.out_degrees(torch.tensor([1, 2]))
        tensor([2, 0])

        For a graph of multiple edge types, it is required to specify the edge type in query.

        >>> hg = dgl.heterograph({
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
        ...     ('user', 'plays', 'game'): (torch.tensor([3, 4]), torch.tensor([5, 6]))
        ... })
        >>> hg.out_degrees(torch.tensor([1, 0]), etype='follows')
        tensor([1, 1])

        See Also
        --------
        in_degrees
        """
        srctype = self.to_canonical_etype(etype)[0]
        etid = self.get_etype_id(etype)
        if is_all(u):
            u = self.srcnodes(srctype)
        u_tensor = utils.prepare_tensor(self, u, 'u')
        if F.as_scalar(F.sum(self.has_nodes(u_tensor, ntype=srctype), dim=0)) != len(u_tensor):
            raise DGLError('u contains invalid node IDs')
        deg = self._graph.out_degrees(etid, utils.prepare_tensor(self, u, 'u'))
        if isinstance(u, numbers.Integral):
            return F.as_scalar(deg)
        else:
            return deg

    def adjacency_matrix(self, transpose=True, ctx=F.cpu(), scipy_fmt=None, etype=None):
        """Alias of :meth:`adj`"""
        return self.adj(transpose, ctx, scipy_fmt, etype)

    def adj(self, transpose=True, ctx=F.cpu(), scipy_fmt=None, etype=None):
        """Return the adjacency matrix of edges of the given edge type.

        By default, a row of returned adjacency matrix represents the
        destination of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column
        represents a destination.

        Parameters
        ----------
        transpose : bool, optional
            A flag to transpose the returned adjacency matrix. (Default: True)
        ctx : context, optional
            The context of returned adjacency matrix. (Default: cpu)
        scipy_fmt : str, optional
            If specified, return a scipy sparse matrix in the given format.
            Otherwise, return a backend dependent sparse tensor. (Default: None)
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.


        Returns
        -------
        SparseTensor or scipy.sparse.spmatrix
            Adjacency matrix.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Instantiate a heterogeneous graph.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): ([0, 1], [0, 1]),
        ...     ('developer', 'develops', 'game'): ([0, 1], [0, 2])
        ... })

        Get a backend dependent sparse tensor. Here we use PyTorch for example.

        >>> g.adj(etype='develops')
        tensor(indices=tensor([[0, 1],
                               [0, 2]]),
               values=tensor([1., 1.]),
               size=(2, 3), nnz=2, layout=torch.sparse_coo)

        Get a scipy coo sparse matrix.

        >>> g.adj(scipy_fmt='coo', etype='develops')
        <2x3 sparse matrix of type '<class 'numpy.int64'>'
	    with 2 stored elements in COOrdinate format>
        """
        etid = self.get_etype_id(etype)
        if scipy_fmt is None:
            return self._graph.adjacency_matrix(etid, transpose, ctx)[0]
        else:
            return self._graph.adjacency_matrix_scipy(etid, transpose, scipy_fmt, False)


    def adjacency_matrix_scipy(self, transpose=True, fmt='csr', return_edge_ids=None):
        """DEPRECATED: please use ``dgl.adjacency_matrix(transpose, scipy_fmt=fmt)``.
        """
        dgl_warning('DGLGraph.adjacency_matrix_scipy is deprecated. '
                    'Please replace it with:\n\n\t'
                    'DGLGraph.adjacency_matrix(transpose, scipy_fmt="{}").\n'.format(fmt))

        return self.adjacency_matrix(transpose=transpose, scipy_fmt=fmt)

    def inc(self, typestr, ctx=F.cpu(), etype=None):
        """Return the incidence matrix representation of edges with the given
        edge type.

        An incidence matrix is an n-by-m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of incidence matrices :math:`I`:

        * ``in``:

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`
              (or :math:`v` is the dst node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``out``:

            - :math:`I[v, e] = 1` if :math:`e` is the out-edge of :math:`v`
              (or :math:`v` is the src node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``both`` (only if source and destination node type are the same):

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`;
            - :math:`I[v, e] = -1` if :math:`e` is the out-edge of :math:`v`;
            - :math:`I[v, e] = 0` otherwise (including self-loop).

        Parameters
        ----------
        typestr : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional
            The context of returned incidence matrix. (Default: cpu)
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Framework SparseTensor
            The incidence matrix.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl

        >>> g = dgl.graph(([0, 1], [0, 2]))
        >>> g.inc('in')
        tensor(indices=tensor([[0, 2],
                               [0, 1]]),
               values=tensor([1., 1.]),
               size=(3, 2), nnz=2, layout=torch.sparse_coo)
        >>> g.inc('out')
        tensor(indices=tensor([[0, 1],
                               [0, 1]]),
               values=tensor([1., 1.]),
               size=(3, 2), nnz=2, layout=torch.sparse_coo)
        >>> g.inc('both')
        tensor(indices=tensor([[1, 2],
                               [1, 1]]),
               values=tensor([-1.,  1.]),
               size=(3, 2), nnz=2, layout=torch.sparse_coo)
        """
        etid = self.get_etype_id(etype)
        return self._graph.incidence_matrix(etid, typestr, ctx)[0]

    incidence_matrix = inc

    #################################################################
    # Features
    #################################################################

    def node_attr_schemes(self, ntype=None):
        """Return the node feature schemes for the specified type.

        The scheme of a feature describes the shape and data type of it.

        Parameters
        ----------
        ntype : str, optional
            The node type name. Can be omitted if there is only one type of nodes
            in the graph.

        Returns
        -------
        dict[str, Scheme]
            A dictionary mapping a feature name to its associated feature scheme.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.ndata['h1'] = torch.randn(3, 1)
        >>> g.ndata['h2'] = torch.randn(3, 2)
        >>> g.node_attr_schemes()
        {'h1': Scheme(shape=(1,), dtype=torch.float32),
         'h2': Scheme(shape=(2,), dtype=torch.float32)}

        Query for a heterogeneous graph of multiple node types.

        >>> g = dgl.heterograph({('user', 'plays', 'game'):
        ...                      (torch.tensor([1, 2]), torch.tensor([3, 4]))})
        >>> g.nodes['user'].data['h1'] = torch.randn(3, 1)
        >>> g.nodes['user'].data['h2'] = torch.randn(3, 2)
        >>> g.node_attr_schemes('user')
        {'h1': Scheme(shape=(1,), dtype=torch.float32),
         'h2': Scheme(shape=(2,), dtype=torch.float32)}

        See Also
        --------
        edge_attr_schemes
        """
        return self._node_frames[self.get_ntype_id(ntype)].schemes

    def edge_attr_schemes(self, etype=None):
        """Return the edge feature schemes for the specified type.

        The scheme of a feature describes the shape and data type of it.

        Parameters
        ----------
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.


        Returns
        -------
        dict[str, Scheme]
            A dictionary mapping a feature name to its associated feature scheme.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Query for a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> g.edata['h1'] = torch.randn(2, 1)
        >>> g.edata['h2'] = torch.randn(2, 2)
        >>> g.edge_attr_schemes()
        {'h1': Scheme(shape=(1,), dtype=torch.float32),
         'h2': Scheme(shape=(2,), dtype=torch.float32)}

        Query for a heterogeneous graph of multiple edge types.

        >>> g = dgl.heterograph({('user', 'plays', 'game'):
        ...                      (torch.tensor([1, 2]), torch.tensor([3, 4])),
        ...                      ('user', 'follows', 'user'):
        ...                      (torch.tensor([3, 4]), torch.tensor([5, 6]))})
        >>> g.edges['plays'].data['h1'] = torch.randn(2, 1)
        >>> g.edges['plays'].data['h2'] = torch.randn(2, 2)
        >>> g.edge_attr_schemes('plays')
        {'h1': Scheme(shape=(1,), dtype=torch.float32),
         'h2': Scheme(shape=(2,), dtype=torch.float32)}

        See Also
        --------
        node_attr_schemes
        """
        return self._edge_frames[self.get_etype_id(etype)].schemes

    def set_n_initializer(self, initializer, field=None, ntype=None):
        """Set the initializer for node features.

        When only part of the nodes have a feature (e.g. new nodes are added,
        features are set for a subset of nodes), the initializer initializes
        features for the rest nodes.

        Parameters
        ----------
        initializer : callable
            A function of signature ``func(shape, dtype, ctx, id_range) -> Tensor``.
            The tensor will be the initialized features. The arguments are:

            - ``shape``: The shape of the tensor to return, which is a tuple of int.
              The first dimension is the number of nodes for feature initialization.
            - ``dtype``: The data type of the tensor to return, which is a
              framework-specific data type object.
            - ``ctx``: The device of the tensor to return, which is a framework-specific
              device object.
            - ``id_range``: The start and end ID of the nodes for feature initialization,
              which is a slice.
        field : str, optional
            The name of the feature that the initializer applies. If not given, the
            initializer applies to all features.
        ntype : str, optional
            The type name of the nodes. Can be omitted if the graph has only one type of nodes.

        Notes
        -----
        Without setting a node feature initializer, zero tensors are generated
        for nodes without a feature.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Define a function for initializer.

        >>> def init_feats(shape, dtype, device, id_range):
        ...     return torch.ones(shape, dtype=dtype, device=device)

        An example for a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0]), torch.tensor([1])))
        >>> g.ndata['h1'] = torch.zeros(2, 2)
        >>> g.ndata['h2'] = torch.ones(2, 1)
        >>> # Apply the initializer to feature 'h2' only.
        >>> g.set_n_initializer(init_feats, field='h2')
        >>> g.add_nodes(1)
        >>> print(g.ndata['h1'])
        tensor([[0., 0.],
                [0., 0.],
                [0., 0.]])
        >>> print(g.ndata['h2'])
        tensor([[1.], [1.], [1.]])

        An example for a heterogeneous graph of multiple node types.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.nodes['user'].data['h'] = torch.zeros(3, 2)
        >>> g.nodes['game'].data['w'] = torch.ones(2, 2)
        >>> g.set_n_initializer(init_feats, ntype='game')
        >>> g.add_nodes(1, ntype='user')
        >>> # Initializer not set for 'user', use zero tensors by default
        >>> g.nodes['user'].data['h']
        tensor([[0., 0.],
                [0., 0.],
                [0., 0.],
                [0., 0.]])
        >>> # Initializer set for 'game'
        >>> g.add_nodes(1, ntype='game')
        >>> g.nodes['game'].data['w']
        tensor([[1., 1.],
                [1., 1.],
                [1., 1.]])
        """
        ntid = self.get_ntype_id(ntype)
        self._node_frames[ntid].set_initializer(initializer, field)

    def set_e_initializer(self, initializer, field=None, etype=None):
        """Set the initializer for edge features.

        When only part of the edges have a feature (e.g. new edges are added,
        features are set for a subset of edges), the initializer initializes
        features for the rest edges.

        Parameters
        ----------
        initializer : callable
            A function of signature ``func(shape, dtype, ctx, id_range) -> Tensor``.
            The tensor will be the initialized features. The arguments are:

            - ``shape``: The shape of the tensor to return, which is a tuple of int.
              The first dimension is the number of edges for feature initialization.
            - ``dtype``: The data type of the tensor to return, which is a
              framework-specific data type object.
            - ``ctx``: The device of the tensor to return, which is a framework-specific
              device object.
            - ``id_range``: The start and end ID of the edges for feature initialization,
              which is a slice.
        field : str, optional
            The name of the feature that the initializer applies. If not given, the
            initializer applies to all features.
        etype : str or (str, str, str), optional
            The type names of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.


        Notes
        -----
        Without setting an edge feature initializer, zero tensors are generated
        for edges without a feature.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Define a function for initializer.

        >>> def init_feats(shape, dtype, device, id_range):
        ...     return torch.ones(shape, dtype=dtype, device=device)

        An example for a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0]), torch.tensor([1])))
        >>> g.edata['h1'] = torch.zeros(1, 2)
        >>> g.edata['h2'] = torch.ones(1, 1)
        >>> # Apply the initializer to feature 'h2' only.
        >>> g.set_e_initializer(init_feats, field='h2')
        >>> g.add_edges(torch.tensor([1]), torch.tensor([1]))
        >>> print(g.edata['h1'])
        tensor([[0., 0.],
                [0., 0.]])
        >>> print(g.edata['h2'])
        tensor([[1.], [1.]])

        An example for a heterogeneous graph of multiple edge types.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1]),
        ...                                 torch.tensor([0, 0])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.edges['plays'].data['h'] = torch.zeros(2, 2)
        >>> g.edges['develops'].data['w'] = torch.ones(2, 2)
        >>> g.set_e_initializer(init_feats, etype='plays')
        >>> # Initializer not set for 'develops', use zero tensors by default
        >>> g.add_edges(torch.tensor([1]), torch.tensor([1]), etype='develops')
        >>> g.edges['develops'].data['w']
        tensor([[1., 1.],
                [1., 1.],
                [0., 0.]])
        >>> # Initializer set for 'plays'
        >>> g.add_edges(torch.tensor([1]), torch.tensor([1]), etype='plays')
        >>> g.edges['plays'].data['h']
        tensor([[0., 0.],
                [0., 0.],
                [1., 1.]])
        """
        etid = self.get_etype_id(etype)
        self._edge_frames[etid].set_initializer(initializer, field)

    def _set_n_repr(self, ntid, u, data):
        """Internal API to set node features.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        All updates will be done out of place to work with autograd.

        Parameters
        ----------
        ntid : int
            Node type id.
        u : node, container or tensor
            The node(s).
        data : dict of tensor
            Node representation.
        """
        if is_all(u):
            num_nodes = self._graph.number_of_nodes(ntid)
        else:
            u = utils.prepare_tensor(self, u, 'u')
            num_nodes = len(u)
        for key, val in data.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_nodes:
                raise DGLError('Expect number of features to match number of nodes (len(u)).'
                               ' Got %d and %d instead.' % (nfeats, num_nodes))
            if F.context(val) != self.device:
                raise DGLError('Cannot assign node feature "{}" on device {} to a graph on'
                               ' device {}. Call DGLGraph.to() to copy the graph to the'
                               ' same device.'.format(key, F.context(val), self.device))

        if is_all(u):
            self._node_frames[ntid].update(data)
        else:
            self._node_frames[ntid].update_row(u, data)

    def _get_n_repr(self, ntid, u):
        """Get node(s) representation of a single node type.

        The returned feature tensor batches multiple node features on the first dimension.

        Parameters
        ----------
        ntid : int
            Node type id.
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict from feature name to feature tensor.
        """
        if is_all(u):
            return self._node_frames[ntid]
        else:
            u = utils.prepare_tensor(self, u, 'u')
            return self._node_frames[ntid].subframe(u)

    def _pop_n_repr(self, ntid, key):
        """Internal API to get and remove the specified node feature.

        Parameters
        ----------
        ntid : int
            Node type id.
        key : str
            The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        return self._node_frames[ntid].pop(key)

    def _set_e_repr(self, etid, edges, data):
        """Internal API to set edge(s) features.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out of place to work with autograd.

        Parameters
        ----------
        etid : int
            Edge type id.
        edges : edges
            Edges can be either

            * A pair of endpoint nodes (u, v), where u is the node ID of source
              node type and v is that of destination node type.
            * A tensor of edge ids of the given type.

            The default value is all the edges.
        data : tensor or dict of tensor
            Edge representation.
        """
        # parse argument
        if not is_all(edges):
            eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')

        # sanity check
        if not utils.is_dict_like(data):
            raise DGLError('Expect dictionary type for feature data.'
                           ' Got "%s" instead.' % type(data))

        if is_all(edges):
            num_edges = self._graph.number_of_edges(etid)
        else:
            num_edges = len(eid)
        for key, val in data.items():
            nfeats = F.shape(val)[0]
            if nfeats != num_edges:
                raise DGLError('Expect number of features to match number of edges.'
                               ' Got %d and %d instead.' % (nfeats, num_edges))
            if F.context(val) != self.device:
                raise DGLError('Cannot assign edge feature "{}" on device {} to a graph on'
                               ' device {}. Call DGLGraph.to() to copy the graph to the'
                               ' same device.'.format(key, F.context(val), self.device))

        # set
        if is_all(edges):
            self._edge_frames[etid].update(data)
        else:
            self._edge_frames[etid].update_row(eid, data)

    def _get_e_repr(self, etid, edges):
        """Internal API to get edge features.

        Parameters
        ----------
        etid : int
            Edge type id.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        dict
            Representation dict
        """
        # parse argument
        if is_all(edges):
            return dict(self._edge_frames[etid])
        else:
            eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')
            return self._edge_frames[etid].subframe(eid)

    def _pop_e_repr(self, etid, key):
        """Get and remove the specified edge repr of a single edge type.

        Parameters
        ----------
        etid : int
            Edge type id.
        key : str
          The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        self._edge_frames[etid].pop(key)

    #################################################################
    # Message passing
    #################################################################

    def apply_nodes(self, func, v=ALL, ntype=None, inplace=False):
        """Update the features of the specified nodes by the provided function.

        Parameters
        ----------
        func : callable
            The function to update node features. It must be
            a :ref:`apiudf`.
        v : node IDs
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

            If not given (default), use all the nodes in the graph.
        ntype : str, optional
            The node type name. Can be omitted if there is
            only one type of nodes in the graph.
        inplace : bool, optional
            **DEPRECATED**.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homogeneous graph**

        >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
        >>> g.ndata['h'] = torch.ones(5, 2)
        >>> g.apply_nodes(lambda nodes: {'x' : nodes.data['h'] * 2})
        >>> g.ndata['x']
        tensor([[2., 2.],
                [2., 2.],
                [2., 2.],
                [2., 2.],
                [2., 2.]])

        **Heterogeneous graph**

        >>> g = dgl.heterograph({('user', 'follows', 'user'): ([0, 1], [1, 2])})
        >>> g.nodes['user'].data['h'] = torch.ones(3, 5)
        >>> g.apply_nodes(lambda nodes: {'h': nodes.data['h'] * 2}, ntype='user')
        >>> g.nodes['user'].data['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])

        See Also
        --------
        apply_edges
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        ntid = self.get_ntype_id(ntype)
        ntype = self.ntypes[ntid]
        if is_all(v):
            v_id = self.nodes(ntype)
        else:
            v_id = utils.prepare_tensor(self, v, 'v')
        ndata = core.invoke_node_udf(self, v_id, ntype, func, orig_nid=v_id)
        self._set_n_repr(ntid, v, ndata)

    def apply_edges(self, func, edges=ALL, etype=None, inplace=False):
        """Update the features of the specified edges by the provided function.

        Parameters
        ----------
        func : dgl.function.BuiltinFunction or callable
            The function to generate new edge features. It must be either
            a :ref:`api-built-in` or a :ref:`apiudf`.
        edges : edges
            The edges to update features on. The allowed input formats are:

            * ``int``: A single edge ID.
            * Int Tensor: Each element is an edge ID.  The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * (Tensor, Tensor): The node-tensors format where the i-th elements
              of the two tensors specify an edge.
            * (iterable[int], iterable[int]): Similar to the node-tensors format but
              stores edge endpoints in python iterables.

            Default value specifies all the edges in the graph.

        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        inplace: bool, optional
            **DEPRECATED**.

        Notes
        -----
        DGL recommends using DGL's bulit-in function for the :attr:`func` argument,
        because DGL will invoke efficient kernels that avoids copying node features to
        edge features in this case.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homogeneous graph**

        >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
        >>> g.ndata['h'] = torch.ones(5, 2)
        >>> g.apply_edges(lambda edges: {'x' : edges.src['h'] + edges.dst['h']})
        >>> g.edata['x']
        tensor([[2., 2.],
                [2., 2.],
                [2., 2.],
                [2., 2.]])

        Use built-in function

        >>> import dgl.function as fn
        >>> g.apply_edges(fn.u_add_v('h', 'h', 'x'))
        >>> g.edata['x']
        tensor([[2., 2.],
                [2., 2.],
                [2., 2.],
                [2., 2.]])

        **Heterogeneous graph**

        >>> g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 2, 1])})
        >>> g.edges[('user', 'plays', 'game')].data['h'] = torch.ones(4, 5)
        >>> g.apply_edges(lambda edges: {'h': edges.data['h'] * 2})
        >>> g.edges[('user', 'plays', 'game')].data['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])

        See Also
        --------
        apply_nodes
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        etid = self.get_etype_id(etype)
        etype = self.canonical_etypes[etid]
        g = self if etype is None else self[etype]
        if is_all(edges):
            eid = ALL
        else:
            eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')
        if core.is_builtin(func):
            if not is_all(eid):
                g = g.edge_subgraph(eid, preserve_nodes=True)
            edata = core.invoke_gsddmm(g, func)
        else:
            edata = core.invoke_edge_udf(g, eid, etype, func)
        self._set_e_repr(etid, eid, edata)

    def send_and_recv(self,
                      edges,
                      message_func,
                      reduce_func,
                      apply_node_func=None,
                      etype=None,
                      inplace=False):
        """Send messages along the specified edges and reduce them on
        the destination nodes to update their features.

        Parameters
        ----------
        edges : edges
            The edges to send and receive messages on. The allowed input formats are:

            * ``int``: A single edge ID.
            * Int Tensor: Each element is an edge ID.  The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * (Tensor, Tensor): The node-tensors format where the i-th elements
              of the two tensors specify an edge.
            * (iterable[int], iterable[int]): Similar to the node-tensors format but
              stores edge endpoints in python iterables.

        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        reduce_func : dgl.function.BuiltinFunction or callable
            The reduce function to aggregate the messages.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction. It must be a :ref:`apiudf`.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        inplace: bool, optional
            **DEPRECATED**.

        Notes
        -----
        DGL recommends using DGL's bulit-in function for the :attr:`message_func`
        and the :attr:`reduce_func` arguments,
        because DGL will invoke efficient kernels that avoids copying node features to
        edge features in this case.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        **Homogeneous graph**

        >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
        >>> g.ndata['x'] = torch.ones(5, 2)
        >>> # Specify edges using (Tensor, Tensor).
        >>> g.send_and_recv(([1, 2], [2, 3]), fn.copy_u('x', 'm'), fn.sum('m', 'h'))
        >>> g.ndata['h']
        tensor([[0., 0.],
                [0., 0.],
                [1., 1.],
                [1., 1.],
                [0., 0.]])
        >>> # Specify edges using IDs.
        >>> g.send_and_recv([0, 2, 3], fn.copy_u('x', 'm'), fn.sum('m', 'h'))
        >>> g.ndata['h']
        tensor([[0., 0.],
                [1., 1.],
                [0., 0.],
                [1., 1.],
                [1., 1.]])

        **Heterogeneous graph**

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ...     ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1])
        ... })
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])
        >>> g.send_and_recv(g['follows'].edges(), fn.copy_src('h', 'm'),
        ...                 fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [0.],
                [1.]])
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        # edge type
        etid = self.get_etype_id(etype)
        _, dtid = self._graph.metagraph.find_edge(etid)
        etype = self.canonical_etypes[etid]
        # edge IDs
        eid = utils.parse_edges_arg_to_eid(self, edges, etid, 'edges')
        if len(eid) == 0:
            # no computation
            return
        u, v = self.find_edges(eid, etype=etype)
        # call message passing onsubgraph
        g = self if etype is None else self[etype]
        ndata = core.message_passing(_create_compute_graph(g, u, v, eid),
                                     message_func, reduce_func, apply_node_func)
        dstnodes = F.unique(v)
        self._set_n_repr(dtid, dstnodes, ndata)

    def pull(self,
             v,
             message_func,
             reduce_func,
             apply_node_func=None,
             etype=None,
             inplace=False):
        """Pull messages from the specified node(s)' predecessors along the
        specified edge type, aggregate them to update the node features.

        Parameters
        ----------
        v : node IDs
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        reduce_func : dgl.function.BuiltinFunction or callable
            The reduce function to aggregate the messages.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction. It must be a :ref:`apiudf`.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        inplace: bool, optional
            **DEPRECATED**.

        Notes
        -----
        * If some of the given nodes :attr:`v` has no in-edges, DGL does not invoke
          message and reduce functions for these nodes and fill their aggregated messages
          with zero. Users can control the filled values via :meth:`set_n_initializer`.
          DGL still invokes :attr:`apply_node_func` if provided.
        * DGL recommends using DGL's bulit-in function for the :attr:`message_func`
          and the :attr:`reduce_func` arguments,
          because DGL will invoke efficient kernels that avoids copying node features to
          edge features in this case.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        **Homogeneous graph**

        >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
        >>> g.ndata['x'] = torch.ones(5, 2)
        >>> g.pull([0, 3, 4], fn.copy_u('x', 'm'), fn.sum('m', 'h'))
        >>> g.ndata['h']
        tensor([[0., 0.],
                [0., 0.],
                [0., 0.],
                [1., 1.],
                [1., 1.]])

        **Heterogeneous graph**

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ...     ('user', 'plays', 'game'): ([0, 2], [0, 1])
        ... })
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])

        Pull.

        >>> g['follows'].pull(2, fn.copy_src('h', 'm'), fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [1.],
                [1.]])
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        v = utils.prepare_tensor(self, v, 'v')
        if len(v) == 0:
            # no computation
            return
        etid = self.get_etype_id(etype)
        _, dtid = self._graph.metagraph.find_edge(etid)
        etype = self.canonical_etypes[etid]
        g = self if etype is None else self[etype]
        # call message passing on subgraph
        src, dst, eid = g.in_edges(v, form='all')
        ndata = core.message_passing(_create_compute_graph(g, src, dst, eid, v),
                                     message_func, reduce_func, apply_node_func)
        self._set_n_repr(dtid, v, ndata)

    def push(self,
             u,
             message_func,
             reduce_func,
             apply_node_func=None,
             etype=None,
             inplace=False):
        """Send message from the specified node(s) to their successors
        along the specified edge type and update their node features.

        Parameters
        ----------
        v : node IDs
            The node IDs. The allowed formats are:

            * ``int``: A single node.
            * Int Tensor: Each element is a node ID. The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is a node ID.

        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        reduce_func : dgl.function.BuiltinFunction or callable
            The reduce function to aggregate the messages.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction. It must be a :ref:`apiudf`.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        inplace: bool, optional
            **DEPRECATED**.

        Notes
        -----
        DGL recommends using DGL's bulit-in function for the :attr:`message_func`
        and the :attr:`reduce_func` arguments,
        because DGL will invoke efficient kernels that avoids copying node features to
        edge features in this case.

        Examples
        --------

        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        **Homogeneous graph**

        >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
        >>> g.ndata['x'] = torch.ones(5, 2)
        >>> g.push([0, 1], fn.copy_u('x', 'm'), fn.sum('m', 'h'))
        >>> g.ndata['h']
        tensor([[0., 0.],
                [1., 1.],
                [1., 1.],
                [0., 0.],
                [0., 0.]])

        **Heterogeneous graph**

        >>> g = dgl.heterograph({('user', 'follows', 'user'): ([0, 0], [1, 2])})
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])

        Push.

        >>> g['follows'].push(0, fn.copy_src('h', 'm'), fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [0.],
                [0.]])
        """
        if inplace:
            raise DGLError('The `inplace` option is removed in v0.5.')
        edges = self.out_edges(u, form='eid', etype=etype)
        self.send_and_recv(edges, message_func, reduce_func, apply_node_func, etype=etype)

    def update_all(self,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        """Send messages along all the edges of the specified type
        and update all the nodes of the corresponding destination type.

        Parameters
        ----------
        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        reduce_func : dgl.function.BuiltinFunction or callable
            The reduce function to aggregate the messages.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction. It must be a :ref:`apiudf`.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Notes
        -----
        * If some of the nodes in the graph has no in-edges, DGL does not invoke
          message and reduce functions for these nodes and fill their aggregated messages
          with zero. Users can control the filled values via :meth:`set_n_initializer`.
          DGL still invokes :attr:`apply_node_func` if provided.
        * DGL recommends using DGL's bulit-in function for the :attr:`message_func`
          and the :attr:`reduce_func` arguments,
          because DGL will invoke efficient kernels that avoids copying node features to
          edge features in this case.

        Examples
        --------
        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        **Homogeneous graph**

        >>> g = dgl.graph(([0, 1, 2, 3], [1, 2, 3, 4]))
        >>> g.ndata['x'] = torch.ones(5, 2)
        >>> g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'h'))
        >>> g.ndata['h']
        tensor([[0., 0.],
                [1., 1.],
                [1., 1.],
                [1., 1.],
                [1., 1.]])

        **Heterogeneous graph**

        >>> g = dgl.heterograph({('user', 'follows', 'user'): ([0, 1, 2], [1, 2, 2])})

        Update all.

        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [2.]])
        >>> g['follows'].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [0.],
                [3.]])
        """
        etid = self.get_etype_id(etype)
        etype = self.canonical_etypes[etid]
        _, dtid = self._graph.metagraph.find_edge(etid)
        g = self if etype is None else self[etype]
        ndata = core.message_passing(g, message_func, reduce_func, apply_node_func)
        self._set_n_repr(dtid, ALL, ndata)

    #################################################################
    # Message passing on heterograph
    #################################################################

    def multi_update_all(self, etype_dict, cross_reducer, apply_node_func=None):
        r"""Send messages along all the edges, reduce them by first type-wisely
        then across different types, and then update the node features of all
        the nodes.

        Parameters
        ----------
        etype_dict : dict
            Arguments for edge-type-wise message passing. The keys are edge types
            while the values are message passing arguments.

            The allowed key formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            The value must be a tuple ``(message_func, reduce_func, [apply_node_func])``, where

            * message_func : dgl.function.BuiltinFunction or callable
                The message function to generate messages along the edges.
                It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
            * reduce_func : dgl.function.BuiltinFunction or callable
                The reduce function to aggregate the messages.
                It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
            * apply_node_func : callable, optional
                An optional apply function to further update the node features
                after the message reduction. It must be a :ref:`apiudf`.

        cross_reducer : str
            Cross type reducer. One of ``"sum"``, ``"min"``, ``"max"``, ``"mean"``, ``"stack"``.
        apply_node_func : callable, optional
            An optional apply function after the messages are reduced both
            type-wisely and across different types.
            It must be a :ref:`apiudf`.

        Notes
        -----
        DGL recommends using DGL's bulit-in function for the message_func
        and the reduce_func in the type-wise message passing arguments,
        because DGL will invoke efficient kernels that avoids copying node features to
        edge features in this case.


        Examples
        --------
        >>> import dgl
        >>> import dgl.function as fn
        >>> import torch

        Instantiate a heterograph.

        >>> g = dgl.heterograph({
        ...     ('user', 'follows', 'user'): ([0, 1], [1, 1]),
        ...     ('game', 'attracts', 'user'): ([0], [1])
        ... })
        >>> g.nodes['user'].data['h'] = torch.tensor([[1.], [2.]])
        >>> g.nodes['game'].data['h'] = torch.tensor([[1.]])

        Update all.

        >>> g.multi_update_all(
        ...     {'follows': (fn.copy_src('h', 'm'), fn.sum('m', 'h')),
        ...      'attracts': (fn.copy_src('h', 'm'), fn.sum('m', 'h'))},
        ... "sum")
        >>> g.nodes['user'].data['h']
        tensor([[0.],
                [4.]])
        """
        all_out = defaultdict(list)
        merge_order = defaultdict(list)
        for etype, args in etype_dict.items():
            etid = self.get_etype_id(etype)
            _, dtid = self._graph.metagraph.find_edge(etid)
            args = pad_tuple(args, 3)
            if args is None:
                raise DGLError('Invalid arguments for edge type "{}". Should be '
                               '(msg_func, reduce_func, [apply_node_func])'.format(etype))
            mfunc, rfunc, afunc = args
            g = self if etype is None else self[etype]
            all_out[dtid].append(core.message_passing(g, mfunc, rfunc, afunc))
            merge_order[dtid].append(etid)  # use edge type id as merge order hint
        for dtid, frames in all_out.items():
            # merge by cross_reducer
            self._node_frames[dtid].update(
                reduce_dict_data(frames, cross_reducer, merge_order[dtid]))
            # apply
            if apply_node_func is not None:
                self.apply_nodes(apply_node_func, ALL, self.ntypes[dtid])

    #################################################################
    # Message propagation
    #################################################################

    def prop_nodes(self,
                   nodes_generator,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        """Propagate messages using graph traversal by sequentially triggering
        :func:`pull()` on nodes.

        The traversal order is specified by the ``nodes_generator``. It generates
        node frontiers, which is a list or a tensor of nodes. The nodes in the
        same frontier will be triggered together, while nodes in different frontiers
        will be triggered according to the generating order.

        Parameters
        ----------
        nodes_generator : iterable[node IDs]
            The generator of node frontiers. Each frontier is a set of node IDs
            stored in Tensor or python iterables.
            It specifies which nodes perform :func:`pull` at each step.
        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        reduce_func : dgl.function.BuiltinFunction or callable
            The reduce function to aggregate the messages.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction. It must be a :ref:`apiudf`.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Examples
        --------
        >>> import torch
        >>> import dgl
        >>> import dgl.function as fn

        Instantiate a heterogrph and perform multiple rounds of message passing.

        >>> g = dgl.heterograph({('user', 'follows', 'user'): ([0, 1, 2, 3], [2, 3, 4, 4])})
        >>> g.nodes['user'].data['h'] = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
        >>> g['follows'].prop_nodes([[2, 3], [4]], fn.copy_src('h', 'm'),
        ...                         fn.sum('m', 'h'), etype='follows')
        tensor([[1.],
                [2.],
                [1.],
                [2.],
                [3.]])

        See Also
        --------
        prop_edges
        """
        for node_frontier in nodes_generator:
            self.pull(node_frontier, message_func, reduce_func, apply_node_func, etype=etype)

    def prop_edges(self,
                   edges_generator,
                   message_func,
                   reduce_func,
                   apply_node_func=None,
                   etype=None):
        """Propagate messages using graph traversal by sequentially triggering
        :func:`send_and_recv()` on edges.

        The traversal order is specified by the ``edges_generator``. It generates
        edge frontiers. The edge frontiers should be of *valid edges type*.
        See :func:`send` for more details.

        Edges in the same frontier will be triggered together, and edges in
        different frontiers will be triggered according to the generating order.

        Parameters
        ----------
        edges_generator : generator
            The generator of edge frontiers.
        message_func : dgl.function.BuiltinFunction or callable
            The message function to generate messages along the edges.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        reduce_func : dgl.function.BuiltinFunction or callable
            The reduce function to aggregate the messages.
            It must be either a :ref:`api-built-in` or a :ref:`apiudf`.
        apply_node_func : callable, optional
            An optional apply function to further update the node features
            after the message reduction. It must be a :ref:`apiudf`.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Examples
        --------
        >>> import torch
        >>> import dgl
        >>> import dgl.function as fn

        Instantiate a heterogrph and perform multiple rounds of message passing.

        >>> g = dgl.heterograph({('user', 'follows', 'user'): ([0, 1, 2, 3], [2, 3, 4, 4])})
        >>> g.nodes['user'].data['h'] = torch.tensor([[1.], [2.], [3.], [4.], [5.]])
        >>> g['follows'].prop_edges([[0, 1], [2, 3]], fn.copy_src('h', 'm'),
        ...                         fn.sum('m', 'h'), etype='follows')
        >>> g.nodes['user'].data['h']
        tensor([[1.],
                [2.],
                [1.],
                [2.],
                [3.]])

        See Also
        --------
        prop_nodes
        """
        for edge_frontier in edges_generator:
            self.send_and_recv(edge_frontier, message_func, reduce_func,
                               apply_node_func, etype=etype)

    #################################################################
    # Misc
    #################################################################

    def filter_nodes(self, predicate, nodes=ALL, ntype=None):
        """Return the IDs of the nodes with the given node type that satisfy
        the given predicate.

        Parameters
        ----------
        predicate : callable
            A function of signature ``func(nodes) -> Tensor``.
            ``nodes`` are :class:`dgl.NodeBatch` objects.
            Its output tensor should be a 1D boolean tensor with
            each element indicating whether the corresponding node in
            the batch satisfies the predicate.
        nodes : node ID(s), optional
            The node(s) for query. The allowed formats are:

            - Tensor: A 1D tensor that contains the node(s) for query, whose data type
              and device should be the same as the :py:attr:`idtype` and device of the graph.
            - iterable[int] : Similar to the tensor, but stores node IDs in a sequence
              (e.g. list, tuple, numpy.ndarray).

            By default, it considers all nodes.
        ntype : str, optional
            The node type for query. If the graph has multiple node types, one must
            specify the argument. Otherwise, it can be omitted.

        Returns
        -------
        Tensor
            A 1D tensor that contains the ID(s) of the node(s) that satisfy the predicate.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Define a predicate function.

        >>> def nodes_with_feature_one(nodes):
        ...     # Whether a node has feature 1
        ...     return (nodes.data['h'] == 1.).squeeze(1)

        Filter nodes for a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
        >>> g.ndata['h'] = torch.tensor([[0.], [1.], [1.], [0.]])
        >>> print(g.filter_nodes(nodes_with_feature_one))
        tensor([1, 2])

        Filter on nodes with IDs 0 and 1

        >>> print(g.filter_nodes(nodes_with_feature_one, nodes=torch.tensor([0, 1])))
        tensor([1])

        Filter nodes for a heterogeneous graph.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1]))})
        >>> g.nodes['user'].data['h'] = torch.tensor([[0.], [1.], [1.]])
        >>> g.nodes['game'].data['h'] = torch.tensor([[0.], [1.]])
        >>> # Filter for 'user' nodes
        >>> print(g.filter_nodes(nodes_with_feature_one, ntype='user'))
        tensor([1, 2])
        """
        if is_all(nodes):
            nodes = self.nodes(ntype)
        v = utils.prepare_tensor(self, nodes, 'nodes')
        if F.as_scalar(F.sum(self.has_nodes(v, ntype=ntype), dim=0)) != len(v):
            raise DGLError('v contains invalid node IDs')

        with self.local_scope():
            self.apply_nodes(lambda nbatch: {'_mask' : predicate(nbatch)}, nodes, ntype)
            ntype = self.ntypes[0] if ntype is None else ntype
            mask = self.nodes[ntype].data['_mask']
            if is_all(nodes):
                return F.nonzero_1d(mask)
            else:
                return F.boolean_mask(v, F.gather_row(mask, v))

    def filter_edges(self, predicate, edges=ALL, etype=None):
        """Return the IDs of the edges with the given edge type that satisfy
        the given predicate.

        Parameters
        ----------
        predicate : callable
            A function of signature ``func(edges) -> Tensor``.
            ``edges`` are :class:`dgl.EdgeBatch` objects.
            Its output tensor should be a 1D boolean tensor with
            each element indicating whether the corresponding edge in
            the batch satisfies the predicate.
        edges : edges
            The edges to send and receive messages on. The allowed input formats are:

            * ``int``: A single edge ID.
            * Int Tensor: Each element is an edge ID.  The tensor must have the same device type
              and ID data type as the graph's.
            * iterable[int]: Each element is an edge ID.
            * (Tensor, Tensor): The node-tensors format where the i-th elements
              of the two tensors specify an edge.
            * (iterable[int], iterable[int]): Similar to the node-tensors format but
              stores edge endpoints in python iterables.

            By default, it considers all the edges.
        etype : str or (str, str, str), optional
            The type name of the edges. The allowed type name formats are:

            * ``(str, str, str)`` for source node type, edge type and destination node type.
            * or one ``str`` edge type name if the name can uniquely identify a
              triplet format in the graph.

            Can be omitted if the graph has only one type of edges.

        Returns
        -------
        Tensor
            A 1D tensor that contains the ID(s) of the edge(s) that satisfy the predicate.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Define a predicate function.

        >>> def edges_with_feature_one(edges):
        ...     # Whether an edge has feature 1
        ...     return (edges.data['h'] == 1.).squeeze(1)

        Filter edges for a homogeneous graph.

        >>> g = dgl.graph((torch.tensor([0, 1, 2]), torch.tensor([1, 2, 3])))
        >>> g.edata['h'] = torch.tensor([[0.], [1.], [1.]])
        >>> print(g.filter_edges(edges_with_feature_one))
        tensor([1, 2])

        Filter on edges with IDs 0 and 1

        >>> print(g.filter_edges(edges_with_feature_one, edges=torch.tensor([0, 1])))
        tensor([1])

        Filter edges for a heterogeneous graph.

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2]))})
        >>> g.edges['plays'].data['h'] = torch.tensor([[0.], [1.], [1.], [0.]])
        >>> # Filter for 'plays' nodes
        >>> print(g.filter_edges(edges_with_feature_one, etype='plays'))
        tensor([1, 2])
        """
        if is_all(edges):
            pass
        elif isinstance(edges, tuple):
            u, v = edges
            srctype, _, dsttype = self.to_canonical_etype(etype)
            u = utils.prepare_tensor(self, u, 'u')
            if F.as_scalar(F.sum(self.has_nodes(u, ntype=srctype), dim=0)) != len(u):
                raise DGLError('edges[0] contains invalid node IDs')
            v = utils.prepare_tensor(self, v, 'v')
            if F.as_scalar(F.sum(self.has_nodes(v, ntype=dsttype), dim=0)) != len(v):
                raise DGLError('edges[1] contains invalid node IDs')
        elif isinstance(edges, Iterable) or F.is_tensor(edges):
            edges = utils.prepare_tensor(self, edges, 'edges')
            min_eid = F.as_scalar(F.min(edges, 0))
            if len(edges) > 0 > min_eid:
                raise DGLError('Invalid edge ID {:d}'.format(min_eid))
            max_eid = F.as_scalar(F.max(edges, 0))
            if len(edges) > 0 and max_eid >= self.num_edges(etype):
                raise DGLError('Invalid edge ID {:d}'.format(max_eid))
        else:
            raise ValueError('Unsupported type of edges:', type(edges))

        with self.local_scope():
            self.apply_edges(lambda ebatch: {'_mask' : predicate(ebatch)}, edges, etype)
            etype = self.canonical_etypes[0] if etype is None else etype
            mask = self.edges[etype].data['_mask']
            if is_all(edges):
                return F.nonzero_1d(mask)
            else:
                if isinstance(edges, tuple):
                    e = self.edge_ids(edges[0], edges[1], etype=etype)
                else:
                    e = utils.prepare_tensor(self, edges, 'edges')
                return F.boolean_mask(e, F.gather_row(mask, e))

    @property
    def device(self):
        """Get the device of the graph.

        Returns
        -------
        device context
            The device of the graph, which should be a framework-specific device object
            (e.g., ``torch.device``).

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a homogeneous graph for demonstration.

        >>> g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
        >>> print(g.device)
        device(type='cpu')

        The case of heterogeneous graphs is the same.
        """
        return F.to_backend_ctx(self._graph.ctx)

    def to(self, device, **kwargs):  # pylint: disable=invalid-name
        """Move ndata, edata and graph structure to the targeted device (cpu/gpu).

        If the graph is already on the specified device, the function directly returns it.
        Otherwise, it returns a cloned graph on the specified device.

        Parameters
        ----------
        device : Framework-specific device context object
            The context to move data to (e.g., ``torch.device``).
        kwargs : Key-word arguments.
            Key-word arguments fed to the framework copy function.

        Returns
        -------
        DGLGraph
            The graph on the specified device.

        Examples
        --------
        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        >>> g = dgl.graph((torch.tensor([1, 0]), torch.tensor([1, 2])))
        >>> g.ndata['h'] = torch.ones(3, 1)
        >>> g.edata['h'] = torch.zeros(2, 2)
        >>> g1 = g.to(torch.device('cuda:0'))
        >>> print(g1.device)
        device(type='cuda', index=0)
        >>> print(g1.ndata['h'].device)
        device(type='cuda', index=0)
        >>> print(g1.nodes().device)
        device(type='cuda', index=0)

        The original graph is still on CPU.

        >>> print(g.device)
        device(type='cpu')
        >>> print(g.ndata['h'].device)
        device(type='cpu')
        >>> print(g.nodes().device)
        device(type='cpu')

        The case of heterogeneous graphs is the same.
        """
        if device is None or self.device == device:
            return self

        ret = copy.copy(self)

        # 1. Copy graph structure
        ret._graph = self._graph.copy_to(utils.to_dgl_context(device))

        # 2. Copy features
        # TODO(minjie): handle initializer
        new_nframes = []
        for nframe in self._node_frames:
            new_nframes.append(nframe.to(device, **kwargs))
        ret._node_frames = new_nframes

        new_eframes = []
        for eframe in self._edge_frames:
            new_eframes.append(eframe.to(device, **kwargs))
        ret._edge_frames = new_eframes

        # 2. Copy misc info
        if self._batch_num_nodes is not None:
            new_bnn = {k : F.copy_to(num, device, **kwargs)
                       for k, num in self._batch_num_nodes.items()}
            ret._batch_num_nodes = new_bnn
        if self._batch_num_edges is not None:
            new_bne = {k : F.copy_to(num, device, **kwargs)
                       for k, num in self._batch_num_edges.items()}
            ret._batch_num_edges = new_bne

        return ret

    def cpu(self):
        """Return a new copy of this graph on CPU.

        Returns
        -------
        DGLHeteroGraph
            Graph on CPU.

        See Also
        --------
        to
        """
        return self.to(F.cpu())

    def clone(self):
        """Return a heterograph object that is a clone of current graph.

        Returns
        -------
        DGLHeteroGraph
            The graph object that is a clone of current graph.
        """
        # XXX(minjie): Do a shallow copy first to clone some internal metagraph information.
        #   Not a beautiful solution though.
        ret = copy.copy(self)

        # Clone the graph structure
        meta_edges = []
        for s_ntype, _, d_ntype in self.canonical_etypes:
            meta_edges.append((self.get_ntype_id(s_ntype), self.get_ntype_id(d_ntype)))

        metagraph = graph_index.from_edge_list(meta_edges, True)
        # rebuild graph idx
        num_nodes_per_type = [self.number_of_nodes(c_ntype) for c_ntype in self.ntypes]
        relation_graphs = [self._graph.get_relation_graph(self.get_etype_id(c_etype))
                           for c_etype in self.canonical_etypes]
        ret._graph = heterograph_index.create_heterograph_from_relations(
            metagraph, relation_graphs, utils.toindex(num_nodes_per_type, "int64"))

        # Clone the frames
        ret._node_frames = [fr.clone() for fr in self._node_frames]
        ret._edge_frames = [fr.clone() for fr in self._edge_frames]

        return ret

    def local_var(self):
        """Return a graph object for usage in a local function scope.

        The returned graph object shares the feature data and graph structure of this graph.
        However, any out-place mutation to the feature data will not reflect to this graph,
        thus making it easier to use in a function scope (e.g. forward computation of a model).

        If set, the local graph object will use same initializers for node features and
        edge features.

        Returns
        -------
        DGLGraph
            The graph object for a local variable.

        Notes
        -----
        Inplace operations do reflect to the original graph. This function also has little
        overhead when the number of feature tensors in this graph is small.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a function for computation on graphs.

        >>> def foo(g):
        ...     g = g.local_var()
        ...     g.edata['h'] = torch.ones((g.num_edges(), 3))
        ...     g.edata['h2'] = torch.ones((g.num_edges(), 3))
        ...     return g.edata['h']

        ``local_var`` avoids changing the graph features when exiting the function.

        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([0, 0, 2])))
        >>> g.edata['h'] = torch.zeros((g.num_edges(), 3))
        >>> newh = foo(g)
        >>> print(g.edata['h'])  # still get tensor of all zeros
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
        >>> 'h2' in g.edata      # new feature set in the function scope is not found
        False

        In-place operations will still reflect to the original graph.

        >>> def foo(g):
        ...     g = g.local_var()
        ...     # in-place operation
        ...     g.edata['h'] += 1
        ...     return g.edata['h']

        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([0, 0, 2])))
        >>> g.edata['h'] = torch.zeros((g.num_edges(), 1))
        >>> newh = foo(g)
        >>> print(g.edata['h'])  # the result changes
        tensor([[1.],
                [1.],
                [1.]])

        See Also
        --------
        local_scope
        """
        ret = copy.copy(self)
        ret._node_frames = [fr.clone() for fr in self._node_frames]
        ret._edge_frames = [fr.clone() for fr in self._edge_frames]
        return ret

    @contextmanager
    def local_scope(self):
        """Enter a local scope context for the graph.

        By entering a local scope, any out-place mutation to the feature data will
        not reflect to the original graph, thus making it easier to use in a function scope
        (e.g. forward computation of a model).

        If set, the local scope will use same initializers for node features and
        edge features.

        Notes
        -----
        Inplace operations do reflect to the original graph. This function also has little
        overhead when the number of feature tensors in this graph is small.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a function for computation on graphs.

        >>> def foo(g):
        ...     with g.local_scope():
        ...         g.edata['h'] = torch.ones((g.num_edges(), 3))
        ...         g.edata['h2'] = torch.ones((g.num_edges(), 3))
        ...         return g.edata['h']

        ``local_scope`` avoids changing the graph features when exiting the function.

        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([0, 0, 2])))
        >>> g.edata['h'] = torch.zeros((g.num_edges(), 3))
        >>> newh = foo(g)
        >>> print(g.edata['h'])  # still get tensor of all zeros
        tensor([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]])
        >>> 'h2' in g.edata      # new feature set in the function scope is not found
        False

        In-place operations will still reflect to the original graph.

        >>> def foo(g):
        ...     with g.local_scope():
        ...         # in-place operation
        ...         g.edata['h'] += 1
        ...         return g.edata['h']

        >>> g = dgl.graph((torch.tensor([0, 1, 1]), torch.tensor([0, 0, 2])))
        >>> g.edata['h'] = torch.zeros((g.num_edges(), 1))
        >>> newh = foo(g)
        >>> print(g.edata['h'])  # the result changes
        tensor([[1.],
                [1.],
                [1.]])

        See Also
        --------
        local_var
        """
        old_nframes = self._node_frames
        old_eframes = self._edge_frames
        self._node_frames = [fr.clone() for fr in self._node_frames]
        self._edge_frames = [fr.clone() for fr in self._edge_frames]
        yield
        self._node_frames = old_nframes
        self._edge_frames = old_eframes

    def formats(self, formats=None):
        r"""Get a cloned graph with the specified sparse format(s) or query
        for the usage status of sparse formats

        The API copies both the graph structure and the features.

        If the input graph has multiple edge types, they will have the same
        sparse format.

        Parameters
        ----------
        formats : str or list of str or None

            * If formats is None, return the usage status of sparse formats
            * Otherwise, it can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of
            them, specifying the sparse formats to use.

        Returns
        -------
        dict or DGLGraph

            * If formats is None, the result will be a dict recording the usage
              status of sparse formats.
            * Otherwise, a DGLGraph will be returned, which is a clone of the
              original graph with the specified sparse format(s) ``formats``.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homographs or Heterographs with A Single Edge Type**

        >>> g = dgl.graph(([0, 0, 1], [2, 3, 2]))
        >>> g.ndata['h'] = torch.ones(4, 1)
        >>> # Check status of format usage
        >>> g.formats()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> # Get a clone of the graph with 'csr' format
        >>> csr_g = g.formats('csr')
        >>> # Only allowed formats will be displayed in the status query
        >>> csr_g.formats()
        {'created': ['csr'], 'not created': []}
        >>> # Features are copied as well
        >>> csr_g.ndata['h']
        tensor([[1.],
                [1.],
                [1.],
                [1.]])

        **Heterographs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.formats()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> # Get a clone of the graph with 'csr' format
        >>> csr_g = g.formats('csr')
        >>> # Only allowed formats will be displayed in the status query
        >>> csr_g.formats()
        {'created': ['csr'], 'not created': []}
        """
        if formats is None:
            # Return the format information
            return self._graph.formats()
        else:
            # Convert the graph to use another format
            ret = copy.copy(self)
            ret._graph = self._graph.formats(formats)
            return ret

    def create_formats_(self):
        r"""Create all sparse matrices allowed for the graph.

        By default, we create sparse matrices for a graph only when necessary.
        In some cases we may want to create them immediately (e.g. in a
        multi-process data loader), which can be achieved via this API.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        **Homographs or Heterographs with A Single Edge Type**

        >>> g = dgl.graph(([0, 0, 1], [2, 3, 2]))
        >>> g.format()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> g.create_formats_()
        >>> g.format()
        {'created': ['coo', 'csr', 'csc'], 'not created': []}

        **Heterographs with Multiple Edge Types**

        >>> g = dgl.heterograph({
        ...     ('user', 'plays', 'game'): (torch.tensor([0, 1, 1, 2]),
        ...                                 torch.tensor([0, 0, 1, 1])),
        ...     ('developer', 'develops', 'game'): (torch.tensor([0, 1]),
        ...                                         torch.tensor([0, 1]))
        ...     })
        >>> g.format()
        {'created': ['coo'], 'not created': ['csr', 'csc']}
        >>> g.create_formats_()
        >>> g.format()
        {'created': ['coo', 'csr', 'csc'], 'not created': []}
        """
        return self._graph.create_formats_()

    def astype(self, idtype):
        """Cast this graph to use another ID type.

        Features are copied (shallow copy) to the new graph.

        Parameters
        ----------
        idtype : Data type object.
            New ID type. Can only be int32 or int64.

        Returns
        -------
        DGLHeteroGraph
            Graph in the new ID type.
        """
        if idtype is None:
            return self
        utils.check_valid_idtype(idtype)
        if self.idtype == idtype:
            return self
        bits = 32 if idtype == F.int32 else 64
        ret = copy.copy(self)
        ret._graph = self._graph.asbits(bits)
        return ret

    # TODO: Formats should not be specified, just saving all the materialized formats
    def shared_memory(self, name, formats=('coo', 'csr', 'csc')):
        """Return a copy of this graph in shared memory, without node data or edge data.

        It moves the graph index to shared memory and returns a DGLHeterograph object which
        has the same graph structure, node types and edge types but does not contain node data
        or edge data.

        Parameters
        ----------
        name : str
            The name of the shared memory.
        formats : str or a list of str (optional)
            Desired formats to be materialized.

        Returns
        -------
        HeteroGraph
            The graph in shared memory
        """
        assert len(name) > 0, "The name of shared memory cannot be empty"
        assert len(formats) > 0
        if isinstance(formats, str):
            formats = [formats]
        for fmt in formats:
            assert fmt in ("coo", "csr", "csc"), '{} is not coo, csr or csc'.format(fmt)
        gidx = self._graph.shared_memory(name, self.ntypes, self.etypes, formats)
        return DGLHeteroGraph(gidx, self.ntypes, self.etypes)


    def long(self):
        """Cast the graph to one with idtype int64

        If the graph already has idtype int64, the function directly returns it. Otherwise,
        it returns a cloned graph of idtype int64 with features copied (shallow copy).

        Returns
        -------
        DGLGraph
            The graph of idtype int64.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a graph of idtype int32.

        >>> # (0, 1), (0, 2), (1, 2)
        >>> g = dgl.graph((torch.tensor([0, 0, 1]).int(), torch.tensor([1, 2, 2]).int()))
        >>> g.ndata['feat'] = torch.ones(3, 1)
        >>> g.idtype
        torch.int32

        Cast the graph to one of idtype int64.

        >>> # A cloned graph with an idtype of int64
        >>> g_long = g.long()
        >>> g_long.idtype
        torch.int64
        >>> # The idtype of the original graph does not change.
        >>> g.idtype
        torch.int32
        >>> g_long.edges()
        (tensor([0, 0, 1]), tensor([1, 2, 2]))
        >>> g_long.ndata
        {'feat': tensor([[1.],
                         [1.],
                         [1.]])}

        See Also
        --------
        int
        idtype
        """
        return self.astype(F.int64)

    def int(self):
        """Cast the graph to one with idtype int32

        If the graph already has idtype int32, the function directly returns it. Otherwise,
        it returns a cloned graph of idtype int32 with features copied (shallow copy).

        Returns
        -------
        DGLGraph
            The graph of idtype int32.

        Examples
        --------

        The following example uses PyTorch backend.

        >>> import dgl
        >>> import torch

        Create a graph of idtype int64.

        >>> # (0, 1), (0, 2), (1, 2)
        >>> g = dgl.graph((torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2])))
        >>> g.ndata['feat'] = torch.ones(3, 1)
        >>> g.idtype
        torch.int64

        Cast the graph to one of idtype int32.

        >>> # A cloned graph with an idtype of int32
        >>> g_int = g.int()
        >>> g_int.idtype
        torch.int32
        >>> # The idtype of the original graph does not change.
        >>> g.idtype
        torch.int64
        >>> g_int.edges()
        (tensor([0, 0, 1], dtype=torch.int32), tensor([1, 2, 2], dtype=torch.int32))
        >>> g_int.ndata
        {'feat': tensor([[1.],
                         [1.],
                         [1.]])}

        See Also
        --------
        long
        idtype
        """
        return self.astype(F.int32)

    #################################################################
    # DEPRECATED: from the old DGLGraph
    #################################################################

    def from_networkx(self, nx_graph, node_attrs=None, edge_attrs=None):
        """DEPRECATED: please use

            ``dgl.from_networkx(nx_graph, node_attrs, edge_attrs)``

        which will return a new graph created from the networkx graph.
        """
        raise DGLError('DGLGraph.from_networkx is deprecated. Please call the following\n\n'
                       '\t dgl.from_networkx(nx_graph, node_attrs, edge_attrs)\n\n'
                       ', which creates a new DGLGraph from the networkx graph.')

    def from_scipy_sparse_matrix(self, spmat, multigraph=None):
        """DEPRECATED: please use

            ``dgl.from_scipy(spmat)``

        which will return a new graph created from the scipy matrix.
        """
        raise DGLError('DGLGraph.from_scipy_sparse_matrix is deprecated. '
                       'Please call the following\n\n'
                       '\t dgl.from_scipy(spmat)\n\n'
                       ', which creates a new DGLGraph from the scipy matrix.')

    def register_apply_node_func(self, func):
        """Deprecated: please directly call :func:`apply_nodes` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_apply_node_func is deprecated.'
                       ' Please directly call apply_nodes with func as the argument.')

    def register_apply_edge_func(self, func):
        """Deprecated: please directly call :func:`apply_edges` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_apply_edge_func is deprecated.'
                       ' Please directly call apply_edges with func as the argument.')

    def register_message_func(self, func):
        """Deprecated: please directly call :func:`update_all` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_message_func is deprecated.'
                       ' Please directly call update_all with func as the argument.')

    def register_reduce_func(self, func):
        """Deprecated: please directly call :func:`update_all` with ``func``
        as argument.
        """
        raise DGLError('DGLGraph.register_reduce_func is deprecated.'
                       ' Please directly call update_all with func as the argument.')

    def group_apply_edges(self, group_by, func, edges=ALL, etype=None, inplace=False):
        """**DEPRECATED**: The API is removed in 0.5."""
        raise DGLError('DGLGraph.group_apply_edges is removed in 0.5.')

    def send(self, edges, message_func, etype=None):
        """Send messages along the given edges with the same edge type.

        DEPRECATE: please use send_and_recv, update_all.
        """
        raise DGLError('DGLGraph.send is deprecated. As a replacement, use DGLGraph.apply_edges\n'
                       ' API to compute messages as edge data. Then use DGLGraph.send_and_recv\n'
                       ' and set the message function as dgl.function.copy_e to conduct message\n'
                       ' aggregation.')

    def recv(self, v, reduce_func, apply_node_func=None, etype=None, inplace=False):
        r"""Receive and reduce incoming messages and update the features of node(s) :math:`v`.

        DEPRECATE: please use send_and_recv, update_all.
        """
        raise DGLError('DGLGraph.recv is deprecated. As a replacement, use DGLGraph.apply_edges\n'
                       ' API to compute messages as edge data. Then use DGLGraph.send_and_recv\n'
                       ' and set the message function as dgl.function.copy_e to conduct message\n'
                       ' aggregation.')

    def multi_recv(self, v, reducer_dict, cross_reducer, apply_node_func=None, inplace=False):
        r"""Receive messages from multiple edge types and perform aggregation.

        DEPRECATE: please use multi_send_and_recv, multi_update_all.
        """
        raise DGLError('DGLGraph.multi_recv is deprecated. As a replacement,\n'
                       ' use DGLGraph.apply_edges API to compute messages as edge data.\n'
                       ' Then use DGLGraph.multi_send_and_recv and set the message function\n'
                       ' as dgl.function.copy_e to conduct message aggregation.')

    def multi_send_and_recv(self, etype_dict, cross_reducer, apply_node_func=None, inplace=False):
        r"""**DEPRECATED**: The API is removed in v0.5."""
        raise DGLError('DGLGraph.multi_pull is removed in v0.5. As a replacement,\n'
                       ' use DGLGraph.edge_subgraph to extract the subgraph first \n'
                       ' and then call DGLGraph.multi_update_all.')

    def multi_pull(self, v, etype_dict, cross_reducer, apply_node_func=None, inplace=False):
        r"""**DEPRECATED**: The API is removed in v0.5."""
        raise DGLError('DGLGraph.multi_pull is removed in v0.5. As a replacement,\n'
                       ' use DGLGraph.edge_subgraph to extract the subgraph first \n'
                       ' and then call DGLGraph.multi_update_all.')

    def readonly(self, readonly_state=True):
        """Deprecated: DGLGraph will always be mutable."""
        dgl_warning('DGLGraph.readonly is deprecated in v0.5.\n'
                    'DGLGraph now always supports mutable operations like add_nodes'
                    ' and add_edges.')

############################################################
# Internal APIs
############################################################

def make_canonical_etypes(etypes, ntypes, metagraph):
    """Internal function to convert etype name to (srctype, etype, dsttype)

    Parameters
    ----------
    etypes : list of str
        Edge type list
    ntypes : list of str
        Node type list
    metagraph : GraphIndex
        Meta graph.

    Returns
    -------
    list of tuples (srctype, etype, dsttype)
    """
    # sanity check
    if len(etypes) != metagraph.number_of_edges():
        raise DGLError('Length of edge type list must match the number of '
                       'edges in the metagraph. {} vs {}'.format(
                           len(etypes), metagraph.number_of_edges()))
    if len(ntypes) != metagraph.number_of_nodes():
        raise DGLError('Length of nodes type list must match the number of '
                       'nodes in the metagraph. {} vs {}'.format(
                           len(ntypes), metagraph.number_of_nodes()))
    if (len(etypes) == 1 and len(ntypes) == 1):
        return [(ntypes[0], etypes[0], ntypes[0])]
    src, dst, eid = metagraph.edges(order="eid")
    rst = [(ntypes[sid], etypes[eid], ntypes[did]) for sid, did, eid in zip(src, dst, eid)]
    return rst

def is_unibipartite(graph):
    """Internal function that returns whether the given graph is a uni-directional
    bipartite graph.

    Parameters
    ----------
    graph : GraphIndex
        Input graph

    Returns
    -------
    bool
        True if the graph is a uni-bipartite.
    """
    src, dst, _ = graph.edges()
    return set(src.tonumpy()).isdisjoint(set(dst.tonumpy()))

def find_src_dst_ntypes(ntypes, metagraph):
    """Internal function to split ntypes into SRC and DST categories.

    If the metagraph is not a uni-bipartite graph (so that the SRC and DST categories
    are not well-defined), return None.

    For node types that are isolated (i.e, no relation is associated with it), they
    are assigned to the SRC category.

    Parameters
    ----------
    ntypes : list of str
        Node type list
    metagraph : GraphIndex
        Meta graph.

    Returns
    -------
    (dict[int, str], dict[int, str]) or None
        Node types belonging to SRC and DST categories. Types are stored in
        a dictionary from type name to type id. Return None if the graph is
        not uni-bipartite.
    """
    ret = _CAPI_DGLFindSrcDstNtypes(metagraph)
    if ret is None:
        return None
    else:
        src, dst = ret
        srctypes = {ntypes[tid] : tid for tid in src}
        dsttypes = {ntypes[tid] : tid for tid in dst}
        return srctypes, dsttypes

def pad_tuple(tup, length, pad_val=None):
    """Pad the given tuple to the given length.

    If the input is not a tuple, convert it to a tuple of length one.
    Return None if pad fails.
    """
    if not isinstance(tup, tuple):
        tup = (tup, )
    if len(tup) > length:
        return None
    elif len(tup) == length:
        return tup
    else:
        return tup + (pad_val,) * (length - len(tup))

def reduce_dict_data(frames, reducer, order=None):
    """Merge tensor dictionaries into one. Resolve conflict fields using reducer.

    Parameters
    ----------
    frames : list[dict[str, Tensor]]
        Input tensor dictionaries
    reducer : str
        One of "sum", "max", "min", "mean", "stack"
    order : list[Int], optional
        Merge order hint. Useful for "stack" reducer.
        If provided, each integer indicates the relative order
        of the ``frames`` list. Frames are sorted according to this list
        in ascending order. Tie is not handled so make sure the order values
        are distinct.

    Returns
    -------
    dict[str, Tensor]
        Merged frame
    """
    if len(frames) == 1 and reducer != 'stack':
        # Directly return the only one input. Stack reducer requires
        # modifying tensor shape.
        return frames[0]
    if reducer == 'stack':
        # Stack order does not matter. However, it must be consistent!
        if order:
            assert len(order) == len(frames)
            sorted_with_key = sorted(zip(frames, order), key=lambda x: x[1])
            frames = list(zip(*sorted_with_key))[0]
        def merger(flist):
            return F.stack(flist, 1)
    else:
        redfn = getattr(F, reducer, None)
        if redfn is None:
            raise DGLError('Invalid cross type reducer. Must be one of '
                           '"sum", "max", "min", "mean" or "stack".')
        def merger(flist):
            return redfn(F.stack(flist, 0), 0) if len(flist) > 1 else flist[0]
    keys = set()
    for frm in frames:
        keys.update(frm.keys())
    ret = {}
    for k in keys:
        flist = []
        for frm in frames:
            if k in frm:
                flist.append(frm[k])
        ret[k] = merger(flist)
    return ret

def combine_frames(frames, ids, col_names=None):
    """Merge the frames into one frame, taking the common columns.

    Return None if there is no common columns.

    Parameters
    ----------
    frames : List[Frame]
        List of frames
    ids : List[int]
        List of frame IDs
    col_names : List[str], optional
        Column names to consider. If not given, it considers all columns.

    Returns
    -------
    Frame
        The resulting frame
    """
    # find common columns and check if their schemes match
    if col_names is None:
        schemes = {key: scheme for key, scheme in frames[ids[0]].schemes.items()}
    else:
        schemes = {key: frames[ids[0]].schemes[key] for key in col_names}
    for frame_id in ids:
        frame = frames[frame_id]
        for key, scheme in list(schemes.items()):
            if key in frame.schemes:
                if frame.schemes[key] != scheme:
                    raise DGLError('Cannot concatenate column %s with shape %s and shape %s' %
                                   (key, frame.schemes[key], scheme))
            else:
                del schemes[key]

    if len(schemes) == 0:
        return None

    # concatenate the columns
    to_cat = lambda key: [frames[i][key] for i in ids if frames[i].num_rows > 0]
    cols = {key: F.cat(to_cat(key), dim=0) for key in schemes}
    return Frame(cols)

def combine_names(names, ids=None):
    """Combine the selected names into one new name.

    Parameters
    ----------
    names : list of str
        String names
    ids : numpy.ndarray, optional
        Selected index

    Returns
    -------
    str
    """
    if ids is None:
        return '+'.join(sorted(names))
    else:
        selected = sorted([names[i] for i in ids])
        return '+'.join(selected)

class DGLBlock(DGLHeteroGraph):
    """Subclass that signifies the graph is a block created from
    :func:`dgl.to_block`.
    """
    # (BarclayII) I'm making a subclass because I don't want to make another version of
    # serialization that contains the is_block flag.
    is_block = True

    def __repr__(self):
        if len(self.srctypes) == 1 and len(self.dsttypes) == 1 and len(self.etypes) == 1:
            ret = 'Block(num_src_nodes={srcnode}, num_dst_nodes={dstnode}, num_edges={edge})'
            return ret.format(
                srcnode=self.number_of_src_nodes(),
                dstnode=self.number_of_dst_nodes(),
                edge=self.number_of_edges())
        else:
            ret = ('Block(num_src_nodes={srcnode},\n'
                   '      num_dst_nodes={dstnode},\n'
                   '      num_edges={edge},\n'
                   '      metagraph={meta})')
            nsrcnode_dict = {ntype : self.number_of_src_nodes(ntype)
                             for ntype in self.srctypes}
            ndstnode_dict = {ntype : self.number_of_dst_nodes(ntype)
                             for ntype in self.dsttypes}
            nedge_dict = {etype : self.number_of_edges(etype)
                          for etype in self.canonical_etypes}
            meta = str(self.metagraph().edges(keys=True))
            return ret.format(
                srcnode=nsrcnode_dict, dstnode=ndstnode_dict, edge=nedge_dict, meta=meta)


def _create_compute_graph(graph, u, v, eid, recv_nodes=None):
    """Create a computation graph from the given edges.

    The compute graph is a uni-directional bipartite graph with only
    one edge type. Similar to subgraph extraction, it stores the original node IDs
    in the srcdata[NID] and dstdata[NID] and extracts features accordingly.
    Edges are not relabeled.

    This function is typically used during message passing to generate
    a graph that contains only the active set of edges.

    Parameters
    ----------
    graph : DGLGraph
        The input graph.
    u : Tensor
        Src nodes.
    v : Tensor
        Dst nodes.
    eid : Tensor
        Edge IDs.
    recv_nodes : Tensor
        Nodes that receive messages. If None, it is equal to unique(v).
        Otherwise, it must be a superset of v and can contain nodes
        that have no incoming edges.

    Returns
    -------
    DGLGraph
        A computation graph.
    """
    if len(u) == 0:
        # The computation graph has no edge and will not trigger message
        # passing. However, because of the apply node phase, we still construct
        # an empty graph to continue.
        unique_src = new_u = new_v = u
        assert recv_nodes is not None
        unique_dst, _ = utils.relabel(recv_nodes)
    else:
        # relabel u and v to starting from 0
        unique_src, src_map = utils.relabel(u)
        if recv_nodes is None:
            unique_dst, dst_map = utils.relabel(v)
        else:
            unique_dst, dst_map = utils.relabel(recv_nodes)
        new_u = F.gather_row(src_map, u)
        new_v = F.gather_row(dst_map, v)

    srctype, etype, dsttype = graph.canonical_etypes[0]
    # create graph
    hgidx = heterograph_index.create_unitgraph_from_coo(
        2, len(unique_src), len(unique_dst), new_u, new_v, ['coo', 'csr', 'csc'])
    # create frame
    srcframe = graph._node_frames[graph.get_ntype_id(srctype)].subframe(unique_src)
    srcframe[NID] = unique_src
    dstframe = graph._node_frames[graph.get_ntype_id(dsttype)].subframe(unique_dst)
    dstframe[NID] = unique_dst
    eframe = graph._edge_frames[0].subframe(eid)
    eframe[EID] = eid

    return DGLHeteroGraph(hgidx, ([srctype], [dsttype]), [etype],
                          node_frames=[srcframe, dstframe],
                          edge_frames=[eframe])

_init_api("dgl.heterograph")
