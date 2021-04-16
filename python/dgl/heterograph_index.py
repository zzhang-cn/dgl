"""Module for heterogeneous graph index class definition."""
from __future__ import absolute_import

import itertools
import numpy as np
import scipy

from ._ffi.object import register_object, ObjectBase
from ._ffi.function import _init_api
from .base import DGLError, dgl_warning
from . import backend as F
from . import utils

@register_object('graph.HeteroGraph')
class HeteroGraphIndex(ObjectBase):
    """HeteroGraph index object.

    Note
    ----
    Do not create GraphIndex directly.
    """
    def __new__(cls):
        obj = ObjectBase.__new__(cls)
        obj._cache = {}
        return obj

    def __getstate__(self):
        """Issue: https://github.com/pytorch/pytorch/issues/32351
           Need to set the tensor created in the __getstate__ function
            as object attribute to avoid potential bugs
        """
        self._pk_state = _CAPI_DGLHeteroPickle(self)
        return self._pk_state

    def __setstate__(self, state):
        self._cache = {}

        # Pickle compatibility check
        # TODO: we should store a storage version number in later releases.
        if isinstance(state, HeteroPickleStates):
            # post-0.4.3
            self.__init_handle_by_constructor__(_CAPI_DGLHeteroUnpickle, state)
        elif isinstance(state, tuple) and len(state) == 3:
            # pre-0.4.2
            metagraph, number_of_nodes, edges = state

            self._cache = {}
            # loop over etypes and recover unit graphs
            rel_graphs = []
            for i, edges_per_type in enumerate(edges):
                src_ntype, dst_ntype = metagraph.find_edge(i)
                num_src = number_of_nodes[src_ntype]
                num_dst = number_of_nodes[dst_ntype]
                src_id, dst_id, _ = edges_per_type
                rel_graphs.append(create_unitgraph_from_coo(
                    1 if src_ntype == dst_ntype else 2, num_src, num_dst, src_id, dst_id,
                    ['coo', 'csr', ' csc']))
            self.__init_handle_by_constructor__(
                _CAPI_DGLHeteroCreateHeteroGraph, metagraph, rel_graphs)

    @property
    def metagraph(self):
        """Meta graph

        Returns
        -------
        GraphIndex
            The meta graph.
        """
        return _CAPI_DGLHeteroGetMetaGraph(self)

    def number_of_ntypes(self):
        """Return number of node types."""
        return self.metagraph.number_of_nodes()

    def number_of_etypes(self):
        """Return number of edge types."""
        return self.metagraph.number_of_edges()

    def get_relation_graph(self, etype):
        """Get the unitgraph graph of the given edge/relation type.

        Parameters
        ----------
        etype : int
            The edge/relation type.

        Returns
        -------
        HeteroGraphIndex
            The unitgraph graph.
        """
        return _CAPI_DGLHeteroGetRelationGraph(self, int(etype))

    def flatten_relations(self, etypes):
        """Convert the list of requested unitgraph graphs into a single unitgraph
        graph.

        Parameters
        ----------
        etypes : list[int]
            The edge/relation types.

        Returns
        -------
        FlattenedHeteroGraph
            A flattened heterograph object
        """
        return _CAPI_DGLHeteroGetFlattenedGraph(self, etypes)

    def add_nodes(self, ntype, num):
        """Add nodes.

        Parameters
        ----------
        ntype : int
            Node type
        num : int
            Number of nodes to be added.
        """
        _CAPI_DGLHeteroAddVertices(self, int(ntype), int(num))
        self.clear_cache()

    def add_edge(self, etype, u, v):
        """Add one edge.

        Parameters
        ----------
        etype : int
            Edge type
        u : int
            The src node.
        v : int
            The dst node.
        """
        _CAPI_DGLHeteroAddEdge(self, int(etype), int(u), int(v))
        self.clear_cache()

    def add_edges(self, etype, u, v):
        """Add many edges.

        Parameters
        ----------
        etype : int
            Edge type
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.
        """
        _CAPI_DGLHeteroAddEdges(self, int(etype), u.todgltensor(), v.todgltensor())
        self.clear_cache()

    def clear(self):
        """Clear the graph."""
        _CAPI_DGLHeteroClear(self)
        self._cache.clear()

    @property
    def dtype(self):
        """Return the data type of this graph index.

        Returns
        -------
        DGLDataType
            The data type of the graph.
        """
        return _CAPI_DGLHeteroDataType(self)

    @property
    def ctx(self):
        """Return the context of this graph index.

        Returns
        -------
        DGLContext
            The context of the graph.
        """
        return _CAPI_DGLHeteroContext(self)

    def bits_needed(self, etype):
        """Return the number of integer bits needed to represent the unitgraph graph.

        Parameters
        ----------
        etype : int
            The edge type.

        Returns
        -------
        int
            The number of bits needed.
        """
        stype, dtype = self.metagraph.find_edge(etype)
        if (self.number_of_edges(etype) >= 0x80000000 or
                self.number_of_nodes(stype) >= 0x80000000 or
                self.number_of_nodes(dtype) >= 0x80000000):
            return 64
        else:
            return 32

    def asbits(self, bits):
        """Transform the graph to a new one with the given number of bits storage.

        NOTE: this method only works for immutable graph index

        Parameters
        ----------
        bits : int
            The number of integer bits (32 or 64)

        Returns
        -------
        HeteroGraphIndex
            The graph index stored using the given number of bits.
        """
        return _CAPI_DGLHeteroAsNumBits(self, int(bits))

    def copy_to(self, ctx):
        """Copy this immutable graph index to the given device context.

        NOTE: this method only works for immutable graph index

        Parameters
        ----------
        ctx : DGLContext
            The target device context.

        Returns
        -------
        HeteroGraphIndex
            The graph index on the given device context.
        """
        return _CAPI_DGLHeteroCopyTo(self, ctx.device_type, ctx.device_id)

    def shared_memory(self, name, ntypes=None, etypes=None, formats=('coo', 'csr', 'csc')):
        """Return a copy of this graph in shared memory

        Parameters
        ----------
        name : str
            The name of the shared memory.
        ntypes : list of str
            Name of node types
        etypes : list of str
            Name of edge types
        format : list of str
            Desired formats to be materialized.

        Returns
        -------
        HeteroGraphIndex
            The graph index in shared memory
        """
        assert len(name) > 0, "The name of shared memory cannot be empty"
        assert len(formats) > 0
        for fmt in formats:
            assert fmt in ("coo", "csr", "csc")
        ntypes = [] if ntypes is None else ntypes
        etypes = [] if etypes is None else etypes
        return _CAPI_DGLHeteroCopyToSharedMem(self, name, ntypes, etypes, formats)

    def is_multigraph(self):
        """Return whether the graph is a multigraph
        The time cost will be O(E)

        Returns
        -------
        bool
            True if it is a multigraph, False otherwise.
        """
        return bool(_CAPI_DGLHeteroIsMultigraph(self))

    def is_readonly(self):
        """Return whether the graph index is read-only.

        Returns
        -------
        bool
            True if it is a read-only graph, False otherwise.
        """
        return bool(_CAPI_DGLHeteroIsReadonly(self))

    def number_of_nodes(self, ntype):
        """Return the number of nodes.

        Parameters
        ----------
        ntype : int
            Node type

        Returns
        -------
        int
            The number of nodes
        """
        return _CAPI_DGLHeteroNumVertices(self, int(ntype))

    def number_of_edges(self, etype):
        """Return the number of edges.

        Parameters
        ----------
        etype : int
            Edge type

        Returns
        -------
        int
            The number of edges
        """
        return _CAPI_DGLHeteroNumEdges(self, int(etype))

    def has_nodes(self, ntype, vids):
        """Return true if the nodes exist.

        Parameters
        ----------
        ntype : int
            Node type
        vid : Tensor
            Node IDs

        Returns
        -------
        Tensor
            0-1 array indicating existence
        """
        return F.from_dgl_nd(_CAPI_DGLHeteroHasVertices(
            self, int(ntype), F.to_dgl_nd(vids)))

    def has_edges_between(self, etype, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        etype : int
            Edge type
        u : Tensor
            Src node Ids.
        v : Tensor
            Dst node Ids.

        Returns
        -------
        Tensor
            0-1 array indicating existence
        """
        return F.from_dgl_nd(_CAPI_DGLHeteroHasEdgesBetween(
            self, int(etype), F.to_dgl_nd(u), F.to_dgl_nd(v)))

    def predecessors(self, etype, v):
        """Return the predecessors of the node.

        Assume that node_type(v) == dst_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : int
            The node.

        Returns
        -------
        Tensor
            Array of predecessors
        """
        return F.from_dgl_nd(_CAPI_DGLHeteroPredecessors(
            self, int(etype), int(v)))

    def successors(self, etype, v):
        """Return the successors of the node.

        Assume that node_type(v) == src_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : int
            The node.

        Returns
        -------
        Tensor
            Array of successors
        """
        return F.from_dgl_nd(_CAPI_DGLHeteroSuccessors(
            self, int(etype), int(v)))

    def edge_ids_all(self, etype, u, v):
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        etype : int
            Edge type
        u : Tensor
            The src nodes.
        v : Tensor
            The dst nodes.

        Returns
        -------
        Tensor
            The src nodes.
        Tensor
            The dst nodes.
        Tensor
            The edge ids.
        """
        edge_array = _CAPI_DGLHeteroEdgeIdsAll(
            self, int(etype), F.to_dgl_nd(u), F.to_dgl_nd(v))

        src = F.from_dgl_nd(edge_array(0))
        dst = F.from_dgl_nd(edge_array(1))
        eid = F.from_dgl_nd(edge_array(2))

        return src, dst, eid

    def edge_ids_one(self, etype, u, v):
        """Return an arrays of edge IDs.

        Parameters
        ----------
        etype : int
            Edge type
        u : Tensor
            The src nodes.
        v : Tensor
            The dst nodes.

        Returns
        -------
        Tensor
            The edge ids.
        """
        eid = F.from_dgl_nd(_CAPI_DGLHeteroEdgeIdsOne(
            self, int(etype), F.to_dgl_nd(u), F.to_dgl_nd(v)))
        return eid

    def find_edges(self, etype, eid):
        """Return a triplet of arrays that contains the edge IDs.

        Parameters
        ----------
        etype : int
            Edge type
        eid : Tensor
            Edge ids.

        Returns
        -------
        Tensor
            The src nodes.
        Tensor
            The dst nodes.
        Tensor
            The edge ids.
        """
        edge_array = _CAPI_DGLHeteroFindEdges(
            self, int(etype), F.to_dgl_nd(eid))

        src = F.from_dgl_nd(edge_array(0))
        dst = F.from_dgl_nd(edge_array(1))
        eid = F.from_dgl_nd(edge_array(2))

        return src, dst, eid

    def in_edges(self, etype, v):
        """Return the in edges of the node(s).

        Assume that node_type(v) == dst_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : Tensor
            Node IDs.

        Returns
        -------
        Tensor
            The src nodes.
        Tensor
            The dst nodes.
        Tensor
            The edge ids.
        """
        edge_array = _CAPI_DGLHeteroInEdges_2(self, int(etype), F.to_dgl_nd(v))
        src = F.from_dgl_nd(edge_array(0))
        dst = F.from_dgl_nd(edge_array(1))
        eid = F.from_dgl_nd(edge_array(2))
        return src, dst, eid

    def out_edges(self, etype, v):
        """Return the out edges of the node(s).

        Assume that node_type(v) == src_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : Tensor
            Node IDs.

        Returns
        -------
        Tensor
            The src nodes.
        Tensor
            The dst nodes.
        Tensor
            The edge ids.
        """
        edge_array = _CAPI_DGLHeteroOutEdges_2(self, int(etype), F.to_dgl_nd(v))
        src = F.from_dgl_nd(edge_array(0))
        dst = F.from_dgl_nd(edge_array(1))
        eid = F.from_dgl_nd(edge_array(2))
        return src, dst, eid

    def edges(self, etype, order=None):
        """Return all the edges

        Parameters
        ----------
        etype : int
            Edge type
        order : string
            The order of the returned edges. Currently support:

            - 'srcdst' : sorted by their src and dst ids.
            - 'eid'    : sorted by edge Ids.
            - None     : the arbitrary order.

        Returns
        -------
        Tensor
            The src nodes.
        Tensor
            The dst nodes.
        Tensor
            The edge ids.
        """
        if order is None:
            order = ""
        elif order not in ['srcdst', 'eid']:
            raise DGLError("Expect order to be one of None, 'srcdst', 'eid', "
                           "got {}".format(order))
        edge_array = _CAPI_DGLHeteroEdges(self, int(etype), order)
        src = F.from_dgl_nd(edge_array(0))
        dst = F.from_dgl_nd(edge_array(1))
        eid = F.from_dgl_nd(edge_array(2))
        return src, dst, eid

    def in_degrees(self, etype, v):
        """Return the in degrees of the nodes.

        Assume that node_type(v) == dst_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : Tensor
            The nodes.

        Returns
        -------
        Tensor
            The in degree array.
        """
        return F.from_dgl_nd(_CAPI_DGLHeteroInDegrees(
            self, int(etype), F.to_dgl_nd(v)))

    def out_degrees(self, etype, v):
        """Return the out degrees of the nodes.

        Assume that node_type(v) == src_type(etype). Thus, the ntype argument is omitted.

        Parameters
        ----------
        etype : int
            Edge type
        v : Tensor
            The nodes.

        Returns
        -------
        Tensor
            The out degree array.
        """
        return F.from_dgl_nd(_CAPI_DGLHeteroOutDegrees(
            self, int(etype), F.to_dgl_nd(v)))

    def adjacency_matrix(self, etype, transpose, ctx):
        """Return the adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        etype : int
            Edge type
        transpose : bool
            A flag to transpose the returned adjacency matrix.
        ctx : context
            The context of the returned matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        Tensor
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        if not isinstance(transpose, bool):
            raise DGLError('Expect bool value for "transpose" arg,'
                           ' but got %s.' % (type(transpose)))
        fmt = F.get_preferred_sparse_format()
        rst = _CAPI_DGLHeteroGetAdj(self, int(etype), transpose, fmt)
        # convert to framework-specific sparse matrix
        srctype, dsttype = self.metagraph.find_edge(etype)
        nrows = self.number_of_nodes(srctype) if transpose else self.number_of_nodes(dsttype)
        ncols = self.number_of_nodes(dsttype) if transpose else self.number_of_nodes(srctype)
        nnz = self.number_of_edges(etype)
        if fmt == "csr":
            indptr = F.copy_to(F.from_dgl_nd(rst(0)), ctx)
            indices = F.copy_to(F.from_dgl_nd(rst(1)), ctx)
            shuffle = F.copy_to(F.from_dgl_nd(rst(2)), ctx)
            dat = F.ones(nnz, dtype=F.float32, ctx=ctx)  # FIXME(minjie): data type
            spmat = F.sparse_matrix(dat, ('csr', indices, indptr), (nrows, ncols))[0]
            return spmat, shuffle
        elif fmt == "coo":
            idx = F.copy_to(F.from_dgl_nd(rst(0)), ctx)
            idx = F.reshape(idx, (2, nnz))
            dat = F.ones((nnz,), dtype=F.float32, ctx=ctx)
            adj, shuffle_idx = F.sparse_matrix(
                dat, ('coo', idx), (nrows, ncols))
            return adj, shuffle_idx
        else:
            raise Exception("unknown format")

    def adjacency_matrix_scipy(self, etype, transpose, fmt, return_edge_ids=None):
        """Return the scipy adjacency matrix representation of this graph.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        Parameters
        ----------
        etype : int
            Edge type
        transpose : bool
            A flag to transpose the returned adjacency matrix.
        fmt : str
            Indicates the format of returned adjacency matrix.
        return_edge_ids : bool
            Indicates whether to return edge IDs or 1 as elements.

        Returns
        -------
        scipy.sparse.spmatrix
            The scipy representation of adjacency matrix.
        """
        if not isinstance(transpose, bool):
            raise DGLError('Expect bool value for "transpose" arg,'
                           ' but got %s.' % (type(transpose)))

        if return_edge_ids is None:
            dgl_warning(
                "Adjacency matrix by default currently returns edge IDs."
                "  As a result there is one 0 entry which is not eliminated."
                "  In the next release it will return 1s by default,"
                " and 0 will be eliminated otherwise.",
                FutureWarning)
            return_edge_ids = True

        rst = _CAPI_DGLHeteroGetAdj(self, int(etype), transpose, fmt)
        srctype, dsttype = self.metagraph.find_edge(etype)
        nrows = self.number_of_nodes(srctype) if transpose else self.number_of_nodes(dsttype)
        ncols = self.number_of_nodes(dsttype) if transpose else self.number_of_nodes(srctype)
        nnz = self.number_of_edges(etype)
        if fmt == "csr":
            indptr = utils.toindex(rst(0), self.dtype).tonumpy()
            indices = utils.toindex(rst(1), self.dtype).tonumpy()
            data = utils.toindex(rst(2)).tonumpy() if return_edge_ids else np.ones_like(indices)
            return scipy.sparse.csr_matrix((data, indices, indptr), shape=(nrows, ncols))
        elif fmt == 'coo':
            idx = utils.toindex(rst(0), self.dtype).tonumpy()
            row, col = np.reshape(idx, (2, nnz))
            data = np.arange(0, nnz) if return_edge_ids else np.ones_like(row)
            return scipy.sparse.coo_matrix((data, (row, col)), shape=(nrows, ncols))
        else:
            raise Exception("unknown format")

    def incidence_matrix(self, etype, typestr, ctx):
        """Return the incidence matrix representation of this graph.

        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of an incidence matrix `I`:
        * "in":
          - I[v, e] = 1 if e is the in-edge of v (or v is the dst node of e);
          - I[v, e] = 0 otherwise.
        * "out":
          - I[v, e] = 1 if e is the out-edge of v (or v is the src node of e);
          - I[v, e] = 0 otherwise.
        * "both":
          - I[v, e] = 1 if e is the in-edge of v;
          - I[v, e] = -1 if e is the out-edge of v;
          - I[v, e] = 0 otherwise (including self-loop).

        Parameters
        ----------
        etype : int
            Edge type
        typestr : str
            Can be either "in", "out" or "both"
        ctx : context
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        utils.Index
            A index for data shuffling due to sparse format change. Return None
            if shuffle is not required.
        """
        src, dst, eid = self.edges(etype)
        srctype, dsttype = self.metagraph.find_edge(etype)

        m = self.number_of_edges(etype)
        if typestr == 'in':
            n = self.number_of_nodes(dsttype)
            row = F.unsqueeze(dst, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.copy_to(F.cat([row, col], dim=0), ctx)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        elif typestr == 'out':
            n = self.number_of_nodes(srctype)
            row = F.unsqueeze(src, 0)
            col = F.unsqueeze(eid, 0)
            idx = F.copy_to(F.cat([row, col], dim=0), ctx)
            # FIXME(minjie): data type
            dat = F.ones((m,), dtype=F.float32, ctx=ctx)
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        elif typestr == 'both':
            assert srctype == dsttype, \
                    "'both' is supported only if source and destination type are the same"
            n = self.number_of_nodes(srctype)
            # first remove entries for self loops
            mask = F.logical_not(F.equal(src, dst))
            src = F.boolean_mask(src, mask)
            dst = F.boolean_mask(dst, mask)
            eid = F.boolean_mask(eid, mask)
            n_entries = F.shape(src)[0]
            # create index
            row = F.unsqueeze(F.cat([src, dst], dim=0), 0)
            col = F.unsqueeze(F.cat([eid, eid], dim=0), 0)
            idx = F.copy_to(F.cat([row, col], dim=0), ctx)
            # FIXME(minjie): data type
            x = -F.ones((n_entries,), dtype=F.float32, ctx=ctx)
            y = F.ones((n_entries,), dtype=F.float32, ctx=ctx)
            dat = F.cat([x, y], dim=0)
            inc, shuffle_idx = F.sparse_matrix(dat, ('coo', idx), (n, m))
        else:
            raise DGLError('Invalid incidence matrix type: %s' % str(typestr))
        return inc, shuffle_idx

    def node_subgraph(self, induced_nodes):
        """Return the induced node subgraph.

        Parameters
        ----------
        induced_nodes : list of utils.Index
            Induced nodes. The length should be equal to the number of
            node types in this heterograph.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        """
        vids = [F.to_dgl_nd(nodes) for nodes in induced_nodes]
        return _CAPI_DGLHeteroVertexSubgraph(self, vids)

    def edge_subgraph(self, induced_edges, preserve_nodes):
        """Return the induced edge subgraph.

        Parameters
        ----------
        induced_edges : list of utils.Index
            Induced edges. The length should be equal to the number of
            edge types in this heterograph.
        preserve_nodes : bool
            Indicates whether to preserve all nodes or not.
            If true, keep the nodes which have no edge connected in the subgraph;
            If false, all nodes without edge connected to it would be removed.

        Returns
        -------
        SubgraphIndex
            The subgraph index.
        """
        eids = [F.to_dgl_nd(edges) for edges in induced_edges]
        return _CAPI_DGLHeteroEdgeSubgraph(self, eids, preserve_nodes)

    def get_unitgraph(self, etype, ctx):
        """Create a unitgraph graph from given edge type and copy to the given device
        context.

        Note: this internal function is for DGL scheduler use only

        Parameters
        ----------
        etype : int
            If the graph index is a Bipartite graph index, this argument must be None.
            Otherwise, it represents the edge type.
        ctx : DGLContext
            The context of the returned graph.

        Returns
        -------
        HeteroGraphIndex
        """
        g = self.get_relation_graph(etype)
        return g.copy_to(ctx).asbits(self.bits_needed(etype or 0))

    def get_csr_shuffle_order(self, etype):
        """Return the edge shuffling order when a coo graph is converted to csr format

        Parameters
        ----------
        etype : int
            The edge type

        Returns
        -------
        tuple of two utils.Index
            The first element of the tuple is the shuffle order for outward graph
            The second element of the tuple is the shuffle order for inward graph
        """
        csr = _CAPI_DGLHeteroGetAdj(self, int(etype), True, "csr")
        order = csr(2)
        rev_csr = _CAPI_DGLHeteroGetAdj(self, int(etype), False, "csr")
        rev_order = rev_csr(2)
        return utils.toindex(order, self.dtype), utils.toindex(rev_order, self.dtype)

    def formats(self, formats=None):
        """Get a graph index with the specified sparse format(s) or query
        for the usage status of sparse formats

        If the graph has multiple edge types, they will have the same
        sparse format.

        Parameters
        ----------
        formats : str or list of str or None

            * If formats is None, return the usage status of sparse formats
            * Otherwise, it can be ``'coo'``/``'csr'``/``'csc'`` or a sublist of
            them, specifying the sparse formats to use.

        Returns
        -------
        dict or GraphIndex

            * If formats is None, the result will be a dict recording the usage
              status of sparse formats.
            * Otherwise, a GraphIndex will be returned, which is a clone of the
              original graph with the specified sparse format(s) ``formats``.

        """
        formats_allowed = _CAPI_DGLHeteroGetAllowedFormats(self)
        formats_created = _CAPI_DGLHeteroGetCreatedFormats(self)
        created = []
        not_created = []
        if formats is None:
            for fmt in ['coo', 'csr', 'csc']:
                if fmt in formats_allowed:
                    if fmt in formats_created:
                        created.append(fmt)
                    else:
                        not_created.append(fmt)
            return {
                'created': created,
                'not created': not_created
            }
        else:
            if isinstance(formats, str):
                formats = [formats]
            return _CAPI_DGLHeteroGetFormatGraph(self, formats)

    def create_formats_(self):
        """Create all sparse matrices allowed for the graph."""
        return _CAPI_DGLHeteroCreateFormat(self)

    def reverse(self):
        """Reverse the heterogeneous graph adjacency

        The node types and edge types are not changed.

        Returns
        -------
        A new graph index.
        """
        return _CAPI_DGLHeteroReverse(self)

@register_object('graph.HeteroSubgraph')
class HeteroSubgraphIndex(ObjectBase):
    """Hetero-subgraph data structure"""
    @property
    def graph(self):
        """The subgraph structure

        Returns
        -------
        HeteroGraphIndex
            The subgraph
        """
        return _CAPI_DGLHeteroSubgraphGetGraph(self)

    @property
    def induced_nodes(self):
        """Induced nodes for each node type. The return list
        length should be equal to the number of node types.

        Returns
        -------
        list of utils.Index
            Induced nodes
        """
        ret = _CAPI_DGLHeteroSubgraphGetInducedVertices(self)
        return [F.from_dgl_nd(v) for v in ret]

    @property
    def induced_edges(self):
        """Induced edges for each edge type. The return list
        length should be equal to the number of edge types.

        Returns
        -------
        list of utils.Index
            Induced edges
        """
        ret = _CAPI_DGLHeteroSubgraphGetInducedEdges(self)
        return [F.from_dgl_nd(v) for v in ret]


#################################################################
# Creators
#################################################################

def create_unitgraph_from_coo(num_ntypes, num_src, num_dst, row, col,
                              formats, row_sorted=False, col_sorted=False):
    """Create a unitgraph graph index from COO format

    Parameters
    ----------
    num_ntypes : int
        Number of node types (must be 1 or 2).
    num_src : int
        Number of nodes in the src type.
    num_dst : int
        Number of nodes in the dst type.
    row : utils.Index
        Row index.
    col : utils.Index
        Col index.
    formats : list of str.
        Restrict the storage formats allowed for the unit graph.
    row_sorted : bool, optional
        Whether or not the rows of the COO are in ascending order.
    col_sorted : bool, optional
        Whether or not the columns of the COO are in ascending order within
        each row. This only has an effect when ``row_sorted`` is True.

    Returns
    -------
    HeteroGraphIndex
    """
    if isinstance(formats, str):
        formats = [formats]
    return _CAPI_DGLHeteroCreateUnitGraphFromCOO(
        int(num_ntypes), int(num_src), int(num_dst),
        F.to_dgl_nd(row), F.to_dgl_nd(col),
        formats, row_sorted, col_sorted)

def create_unitgraph_from_csr(num_ntypes, num_src, num_dst, indptr, indices, edge_ids,
                              formats):
    """Create a unitgraph graph index from CSR format

    Parameters
    ----------
    num_ntypes : int
        Number of node types (must be 1 or 2).
    num_src : int
        Number of nodes in the src type.
    num_dst : int
        Number of nodes in the dst type.
    indptr : utils.Index
        CSR indptr.
    indices : utils.Index
        CSR indices.
    edge_ids : utils.Index
        Edge shuffle id.
    formats : str
        Restrict the storage formats allowed for the unit graph.

    Returns
    -------
    HeteroGraphIndex
    """
    if isinstance(formats, str):
        formats = [formats]
    return _CAPI_DGLHeteroCreateUnitGraphFromCSR(
        int(num_ntypes), int(num_src), int(num_dst),
        F.to_dgl_nd(indptr), F.to_dgl_nd(indices), F.to_dgl_nd(edge_ids),
        formats)

def create_heterograph_from_relations(metagraph, rel_graphs, num_nodes_per_type):
    """Create a heterograph from metagraph and graphs of every relation.

    Parameters
    ----------
    metagraph : GraphIndex
        Meta-graph.
    rel_graphs : list of HeteroGraphIndex
        Bipartite graph of each relation.
    num_nodes_per_type : utils.Index, optional
        Number of nodes per node type

    Returns
    -------
    HeteroGraphIndex
    """
    if num_nodes_per_type is None:
        return _CAPI_DGLHeteroCreateHeteroGraph(metagraph, rel_graphs)
    else:
        return _CAPI_DGLHeteroCreateHeteroGraphWithNumNodes(
            metagraph, rel_graphs, num_nodes_per_type.todgltensor())

def create_heterograph_from_shared_memory(name):
    """Create a heterograph from shared memory with the given name.

    Paramaters
    ----------
    name : str
        The name of the share memory

    Returns
    -------
    HeteroGraphIndex (in shared memory)
    ntypes : list of str
        Names of node types
    etypes : list of str
        Names of edge types
    """
    g, ntypes, etypes = _CAPI_DGLHeteroCreateFromSharedMem(name)
    return g, list(ntypes), list(etypes)

def joint_union(metagraph, gidx_list):
    """Return a joint union of the input heterographs.

    Parameters
    ----------
    metagraph : GraphIndex
        Meta-graph.
    gidx_list : list of HeteroGraphIndex
        Heterographs to be joint_unioned.

    Returns
    -------
    HeteroGraphIndex
        joint_unioned Heterograph.
    """
    return _CAPI_DGLHeteroJointUnion(metagraph, gidx_list)

def disjoint_union(metagraph, graphs):
    """Return a disjoint union of the input heterographs.

    Parameters
    ----------
    metagraph : GraphIndex
        Meta-graph.
    graphs : list of HeteroGraphIndex
        Heterographs to be batched.

    Returns
    -------
    HeteroGraphIndex
        Batched Heterograph.
    """
    return _CAPI_DGLHeteroDisjointUnion_v2(metagraph, graphs)

def disjoint_partition(graph, bnn_all_types, bne_all_types):
    """Partition the graph disjointly.

    Parameters
    ----------
    graph : HeteroGraphIndex
        The graph to be partitioned.
    bnn_all_types : list of list of int
        bnn_all_types[t] gives the number of nodes with t-th type in the batch.
    bne_all_types : list of list of int
        bne_all_types[t] gives the number of edges with t-th type in the batch.

    Returns
    --------
    list of HeteroGraphIndex
        Heterographs unbatched.
    """
    bnn_all_types = utils.toindex(list(itertools.chain.from_iterable(bnn_all_types)))
    bne_all_types = utils.toindex(list(itertools.chain.from_iterable(bne_all_types)))
    return _CAPI_DGLHeteroDisjointPartitionBySizes_v2(
        graph, bnn_all_types.todgltensor(), bne_all_types.todgltensor())

#################################################################
# Data structure used by C APIs
#################################################################

@register_object("graph.FlattenedHeteroGraph")
class FlattenedHeteroGraph(ObjectBase):
    """FlattenedHeteroGraph object class in C++ backend."""

@register_object("graph.HeteroPickleStates")
class HeteroPickleStates(ObjectBase):
    """Pickle states object class in C++ backend."""
    @property
    def version(self):
        """Version number

        Returns
        -------
        int
            version number
        """
        return _CAPI_DGLHeteroPickleStatesGetVersion(self)

    @property
    def meta(self):
        """Meta info

        Returns
        -------
        bytearray
            Serialized meta info
        """
        return bytearray(_CAPI_DGLHeteroPickleStatesGetMeta(self))

    @property
    def arrays(self):
        """Arrays representing the graph structure (COO or CSR)

        Returns
        -------
        list of dgl.ndarray.NDArray
            Arrays
        """
        num_arr = _CAPI_DGLHeteroPickleStatesGetArraysNum(self)
        arr_func = _CAPI_DGLHeteroPickleStatesGetArrays(self)
        return [arr_func(i) for i in range(num_arr)]

    def __getstate__(self):
        """Issue: https://github.com/pytorch/pytorch/issues/32351
           Need to set the tensor created in the __getstate__ function
            as object attribute to avoid potential bugs
        """
        self._pk_arrays = [F.zerocopy_from_dgl_ndarray(arr) for arr in self.arrays]
        return self.version, self.meta, self._pk_arrays

    def __setstate__(self, state):
        if isinstance(state[0], int):
            _, meta, arrays = state
            arrays = [F.zerocopy_to_dgl_ndarray(arr) for arr in arrays]
            self.__init_handle_by_constructor__(
                _CAPI_DGLCreateHeteroPickleStates, meta, arrays)
        else:
            metagraph, num_nodes_per_type, adjs = state
            num_nodes_per_type = F.zerocopy_to_dgl_ndarray(num_nodes_per_type)
            self.__init_handle_by_constructor__(
                _CAPI_DGLCreateHeteroPickleStatesOld, metagraph, num_nodes_per_type, adjs)

_init_api("dgl.heterograph_index")
