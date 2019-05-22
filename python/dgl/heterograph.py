"""Classes for heterogeneous graphs."""

class DGLBaseHeteroGraph(object):
    """Base Heterogeneous graph class.

    A Heterogeneous graph is defined as a graph with node types and edge
    types.

    If two edges share the same edge type, then their source nodes, as well
    as their destination nodes, also have the same type (the source node
    types don't have to be the same as the destination node types).

    Parameters
    ----------
    metagraph : NetworkX MultiGraph or compatible data structure
        The set of node types and edge types, as well as the
        source/destination node type of each edge type is specified in the
        metagraph.
        The edge types are specified as edge keys on the NetworkX MultiGraph.
        The node types and edge types must be strings.
    number_of_nodes_by_type : dict[str, int]
        Number of nodes for each node type.
    edge_connections_by_type : dict
        Specifies how edges would connect nodes of the source type to nodes of
        the destination type in the following form:

            {edge_type: edge_specifier}

        where edge_specifier can be either of the following:

        * (source_node_id_tensor, destination_node_id_tensor)
            * ``source_node_id_tensor`` and ``destination_node_id_tensor`` are
              IDs within the source and destination node type respectively.
            * ``edge_type`` is a triplet of

                  (source_node_type_name,
                   destination_node_type_name,
                   edge_type_name)

        * scipy.sparse.matrix
          By default, the rows represent the destination of an edge, and the
          column represents the source.

    Examples
    --------
    Suppose that we want to construct the following heterogeneous graph:

    .. graphviz::

       digraph G {
           Alice -> Bob [label=follows]
           Bob -> Carol [label=follows]
           Alice -> Tetris [label=plays]
           Bob -> Tetris [label=plays]
           Bob -> Minecraft [label=plays]
           Carol -> Minecraft [label=plays]
           Nintendo -> Tetris [label=develops]
           Mojang -> Minecraft [label=develops]
           {rank=source; Alice; Bob; Carol}
           {rank=sink; Nintendo; Mojang}
       }

    One can analyze the graph and figure out the metagraph as follows:

    .. graphviz::

       digraph G {
           User -> User [label=follows]
           User -> Game [label=plays]
           Developer -> Game [label=develops]
       }

    Suppose that one maps the users, games and developers to the following
    IDs:

        User name   Alice   Bob     Carol
        User ID     0       1       2

        Game name   Tetris  Minecraft
        Game ID     0       1

        Developer name  Nintendo    Mojang
        Developer ID    0           1

    One can construct the graph as follows:

    >>> import networkx as nx
    >>> metagraph = nx.MultiGraph([
    ...     ('user', 'user', 'follows'),
    ...     ('user', 'game', 'plays'),
    ...     ('developer', 'game', 'develops')])
    >>> g = DGLBaseHeteroGraph(
    ...     metagraph=metagraph,
    ...     number_of_nodes_by_type={'user': 4, 'game': 2, 'developer': 2},
    ...     edge_connections_by_type={
    ...         # Alice follows Bob and Bob follows Carol
    ...         ('user', 'user', 'follows'): ([0, 1], [1, 2]),
    ...         # Alice and Bob play Tetris and Bob and Carol play Minecraft
    ...         ('user', 'game', 'plays'): ([0, 1, 1, 2], [0, 0, 1, 1]),
    ...         # Nintendo develops Tetris and Mojang develops Minecraft
    ...         ('developer', 'game', 'develops'): ([0, 1], [0, 1])})
    """

    # pylint: disable=unused-argument
    def __init__(
            self,
            metagraph,
            number_of_nodes_by_type,
            edge_connections_by_type):
        super(DGLBaseHeteroGraph, self).__init__()

    def __getitem__(self, key):
        """Returns a view on the heterogeneous graph with given node/edge
        type:

        * If ``key`` is a str, it returns a heterogeneous subgraph induced
          from nodes of type ``key``.
        * If ``key`` is a pair of str (type_A, type_B), it returns a
          heterogeneous subgraph induced from the union of both node types.
        * If ``key`` is a triplet of str

              (src_type_name, dst_type_name, edge_type_name)

          It returns a heterogeneous subgraph induced from the edges with
          source type name ``src_type_name``, destination type name
          ``dst_type_name``, and edge type name ``edge_type_name``.

        The view would share the frames with the parent graph; any
        modifications on one's frames would reflect on the other.

        Note that the subgraph itself is not materialized until someone
        queries the subgraph structure.  This implies that calling computation
        methods such as

            g['user'].update_all(...)

        would not create a subgraph of users.

        Parameters
        ----------
        key : str or tuple
            See above

        Returns
        -------
        DGLBaseHeteroGraphView
            The induced subgraph view.
        """
        pass


class DGLBaseHeteroGraphView(object):
    """View on a heterogeneous graph, constructed from
    DGLBaseHeteroGraph.__getitem__().

    It is semantically the same as a subgraph, except that

    * The subgraph itself is not materialized until the user explicitly
    queries the subgraph structure (e.g. calling ``in_edges``, but not
    ``update_all``).
    """
    pass


class DGLHeteroGraph(DGLBaseHeteroGraph):
    """Base heterogeneous graph class.

    The graph stores nodes, edges and also their (type-specific) features.

    Heterogeneous graphs are by default multigraphs.

    Parameters
    ----------
    metagraph, number_of_nodes_by_type, edge_connections_by_type :
        See DGLBaseHeteroGraph
    node_frame : dict[str, FrameRef], optional
        Node feature storage per type
    edge_frame : dict[str, FrameRef], optional
        Edge feature storage per type
    readonly : bool, optional
        Whether the graph structure is read-only (default: False)
    """
    # pylint: disable=unused-argument
    def __init__(
            self,
            metagraph,
            number_of_nodes_by_type,
            edge_connections_by_type,
            node_frame=None,
            edge_frame=None,
            readonly=False):
        super(DGLHeteroGraph, self).__init__(
            metagraph, number_of_nodes_by_type, edge_connections_by_type)

    # TODO: REVIEW
    def add_nodes(self, num, node_type, data=None):
        """Add multiple new nodes of the same node type

        Parameters
        ----------
        num : int
            Number of nodes to be added.
        node_type : str
            Type of the added nodes.  Must appear in the metagraph.
        data : dict, optional
            Feature data of the added nodes.

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['game'].number_of_nodes()
        2
        >>> g.add_nodes(3, 'game')  # add 3 new games
        >>> g['game'].number_of_nodes()
        5
        """
        pass

    # TODO: REVIEW
    def add_edge(self, u, v, utype, vtype, etype, data=None):
        """Add an edge of ``etype`` between u of type ``utype`` and v of type
        ``vtype``.

        Parameters
        ----------
        u : int
            The source node ID of type ``utype``.  Must exist in the graph.
        v : int
            The destination node ID of type ``vtype``.  Must exist in the
            graph.
        utype : str
            The source node type name.  Must exist in the metagraph.
        vtype : str
            The destination node type name.  Must exist in the metagraph.
        etype : str
            The edge type name.  Must exist in the metagraph.
        data : dict, optional
            Feature data of the added edge.

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['user', 'game', 'plays'].number_of_edges()
        4
        >>> g.add_edge(2, 0, 'user', 'game', 'plays')
        >>> g['user', 'game', 'plays'].number_of_edges()
        5
        """
        pass

    def add_edges(self, u, v, utype, vtype, etype, data=None):
        """Add multiple edges of ``etype`` between list of source nodes ``u``
        of type ``utype`` and list of destination nodes ``v`` of type
        ``vtype``.  A single edge is added between every pair of ``u[i]`` and
        ``v[i]``.

        Parameters
        ----------
        u : list, tensor
            The source node IDs of type ``utype``.  Must exist in the graph.
        v : list, tensor
            The destination node IDs of type ``vtype``.  Must exist in the
            graph.
        utype : str
            The source node type name.  Must exist in the metagraph.
        vtype : str
            The destination node type name.  Must exist in the metagraph.
        etype : str
            The edge type name.  Must exist in the metagraph.
        data : dict, optional
            Feature data of the added edge.

        Examples
        --------
        The variable ``g`` is constructed from the example in
        DGLBaseHeteroGraph.

        >>> g['user', 'game', 'plays'].number_of_edges()
        4
        >>> g.add_edges([0, 2], [1, 0], 'user', 'game', 'plays')
        >>> g['user', 'game', 'plays'].number_of_edges()
        6
        """
        pass

    def node_attr_schemes(self, ntype):
        """Return the node feature schemes for a given node type.

        Each feature scheme is a named tuple that stores the shape and data type
        of the node feature

        Parameters
        ----------
        ntype : str
            The node type

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.
        """
        pass

    def edge_attr_schemes(self, etype):
        """Return the edge feature schemes for a given edge type.

        Each feature scheme is a named tuple that stores the shape and data type
        of the edge feature

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.

        Returns
        -------
        dict of str to schemes
            The schemes of node feature columns.
        """
        pass

    def set_n_initializer(self, ntype, initializer, field=None):
        """Set the initializer for empty node features of given type.

        Initializer is a callable that returns a tensor given the shape, data type
        and device context.

        When a subset of the nodes are assigned a new feature, initializer is
        used to create feature for rest of the nodes.

        Parameters
        ----------
        ntype : str
            The node type name.
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.
        """
        pass

    def set_e_initializer(self, etype, initializer, field=None):
        """Set the initializer for empty edge features of given type.

        Initializer is a callable that returns a tensor given the shape, data
        type and device context.

        When a subset of the edges are assigned a new feature, initializer is
        used to create feature for rest of the edges.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        initializer : callable
            The initializer.
        field : str, optional
            The feature field name. Default is set an initializer for all the
            feature fields.
        """
        pass

    @property
    def nodes(self):
        """Return a node view that can used to set/get feature data of a
        single node type.

        Notes
        -----
        An error is raised if the graph contains multiple node types.  Use

            g[ntype]

        to select nodes with type ``ntype``.
        """
        pass

    @property
    def ndata(self):
        """Return the data view of all the nodes of a single node type.

        Notes
        -----
        An error is raised if the graph contains multiple node types.  Use

            g[ntype]

        to select nodes with type ``ntype``.
        """
        pass

    @property
    def edges(self):
        """Return an edges view that can used to set/get feature data of a
        single edge type.

        Notes
        -----
        An error is raised if the graph contains multiple edge types.  Use

            g[src_type, dst_type, edge_type]

        to select edges with type ``(src_type, dst_type, edge_type)``.
        """
        pass

    @property
    def edata(self):
        """Return the data view of all the edges of a single edge type.

        Notes
        -----
        An error is raised if the graph contains multiple edge types.  Use

            g[src_type, dst_type, edge_type]

        to select edges with type ``(src_type, dst_type, edge_type)``.
        """
        pass

    def set_n_repr(self, data, ntype, u=ALL, inplace=False):
        """Set node(s) representation of a single node type.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of nodes to be updated,
        and (D1, D2, ...) be the shape of the node representation tensor. The
        length of the given node ids must match B (i.e, len(u) == B).

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        data : dict of tensor
            Node representation.
        ntype : str
            Node type.
        u : node, container or tensor
            The node(s).
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        pass

    def get_n_repr(self, ntype, u=ALL):
        """Get node(s) representation of a single node type.

        The returned feature tensor batches multiple node features on the first dimension.

        Parameters
        ----------
        ntype : str
            Node type.
        u : node, container or tensor
            The node(s).

        Returns
        -------
        dict
            Representation dict from feature name to feature tensor.
        """
        pass

    def pop_n_repr(self, ntype, key):
        """Get and remove the specified node repr of a given node type.

        Parameters
        ----------
        ntype : str
            The node type.
        key : str
            The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        pass

    def set_e_repr(self, data, etype, edges=ALL, inplace=False):
        """Set edge(s) representation of a single edge type.

        `data` is a dictionary from the feature name to feature tensor. Each tensor
        is of shape (B, D1, D2, ...), where B is the number of edges to be updated,
        and (D1, D2, ...) be the shape of the edge representation tensor.

        All update will be done out of place to work with autograd unless the
        inplace flag is true.

        Parameters
        ----------
        data : tensor or dict of tensor
            Edge representation.
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.
        inplace : bool
            If True, update will be done in place, but autograd will break.
        """
        pass

    def get_e_repr(self, etype, edges=ALL):
        """Get edge(s) representation.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        edges : edges
            Edges can be a pair of endpoint nodes (u, v), or a
            tensor of edge ids. The default value is all the edges.

        Returns
        -------
        dict
            Representation dict
        """
        pass

    def pop_e_repr(self, etype, key):
        """Get and remove the specified edge repr of a single edge type.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        key : str
          The attribute name.

        Returns
        -------
        Tensor
            The popped representation
        """
        pass

    def apply_nodes(self, func, v=ALL, inplace=False):
        """Apply the function on the nodes with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable, dict[str, callable], or None
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``func`` is not a
            dict, it will throw an error.
        v : int, iterable of int, tensor, dict, optional
            The (type-specific) node (ids) on which to apply ``func``.

            If ``func`` is not a dict, then ``v`` must not be a dict.
            If ``func`` is a dict, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in ``func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``func``, indicating the nodes to be updated for each type.
        inplace : bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g['user'].ndata['h'] = torch.ones(3, 5)
        >>> g['user'].apply_nodes(lambda x: {'h': x * 2})
        >>> g['user'].ndata['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        >>> g.apply_nodes({'user': lambda x: {'h': x * 2}})
        >>> g['user'].ndata['h']
        tensor([[4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.]])
        """
        pass

    def apply_edges(self, func, edges=ALL, inplace=False):
        """Apply the function on the edges with the same type to update their
        features.

        If None is provided for ``func``, nothing will happen.

        Parameters
        ----------
        func : callable, dict[etype, callable], or None
            Apply function on the edge. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``func`` is not a
            dict, it will throw an error.
        edges : any valid edge specification, dict, optional
            Edges on which to apply ``func``. See :func:`send` for valid
            edge specification.

            If ``func`` is not a dict, then ``edges`` must not be a dict.
            If ``func`` is a dict, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in ``func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``func``, indicating the edges to be updated for each type.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Examples
        --------
        >>> g['user', 'game', 'plays'].edata['h'] = torch.ones(3, 5)
        >>> g['user', 'game', 'plays'].apply_edges(lambda x: {'h': x * 2})
        >>> g['user', 'game', 'plays'].edata['h']
        tensor([[2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.],
                [2., 2., 2., 2., 2.]])
        >>> g.apply_edges({('user', 'game', 'plays'): lambda x: {'h': x * 2}})
        tensor([[4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.],
                [4., 4., 4., 4., 4.]])
        """
        pass

    def group_apply_edges(self, group_by, func, edges=ALL, inplace=False):
        """Group the edges by nodes and apply the function of the grouped
        edges to update their features.  The edges are of the same edge type
        (hence having the same source and destination node type).

        Parameters
        ----------
        group_by : str
            Specify how to group edges. Expected to be either 'src' or 'dst'
        func : callable, dict[etype, callable]
            Apply function on the edge.  The function should be
            an :mod:`Edge UDF <dgl.udf>`. The input of `Edge UDF` should
            be (bucket_size, degrees, *feature_shape), and
            return the dict with values of the same shapes.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``func`` is not a
            dict, it will throw an error.
        edges : valid edges type, dict, optional
            Edges on which to group and apply ``func``. See :func:`send` for valid
            edges type. Default is all the edges.

            If ``func`` is not a dict, then ``edges`` must not be a dict.
            If ``func`` is a dict, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in ``func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``func``, indicating the edges to be updated for each type.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.
        """
        pass

    # TODO: REVIEW
    def send(self, edges=ALL, message_func=None):
        """Send messages along the given edges with the same edge type.

        ``edges`` can be any of the following types:

        * ``int`` : Specify one edge using its edge id.
        * ``pair of int`` : Specify one edge using its endpoints.
        * ``int iterable`` / ``tensor`` : Specify multiple edges using their edge ids.
        * ``pair of int iterable`` / ``pair of tensors`` :
          Specify multiple edges using their endpoints.

        The UDF returns messages on the edges and can be later fetched in
        the destination node's ``mailbox``. Receiving will consume the messages.
        See :func:`recv` for example.

        If multiple ``send`` are triggered on the same edge without ``recv``. Messages
        generated by the later ``send`` will overwrite previous messages.

        Parameters
        ----------
        edges : valid edges type, dict, optional
            Edges on which to apply ``message_func``. Default is sending along all
            the edges.

            If ``message_func`` is not a dict, then ``edges`` must not be a dict.
            If ``message_func`` is a dict, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in
              ``message_func``.
            * a dict of int, iterable of int, or tensors, with the same keys
              as ``message_func``, indicating the edges to be updated for each
              type.
        message_func : callable, dict[etype, callable]
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.

        Notes
        -----
        On multigraphs, if :math:`u` and :math:`v` are specified, then the messages will be sent
        along all edges between :math:`u` and :math:`v`.
        """
        pass

    def recv(self,
             v=ALL,
             reduce_func=None,
             apply_node_func=None,
             inplace=False):
        """Receive and reduce incoming messages and update the features of node(s) :math:`v`.

        Optionally, apply a function to update the node features after receive.

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        The node features will be updated by the result of the ``reduce_func``.

        Messages are consumed once received.

        The provided UDF maybe called multiple times so it is recommended to provide
        function with no side effect.

        Parameters
        ----------
        v : int, container or tensor, dict, optional
            The node(s) to be updated. Default is receiving all the nodes.

            If ``apply_node_func`` is not a dict, then ``v`` must not be a
            dict.
            If ``apply_node_func`` is a dict, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in
              ``apply_node_func``.
            * a dict of int, iterable of int, or tensors, indicating the nodes
              to be updated for each type.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable]
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types in ``v``, the node types in ``apply_node_func``,
          and the destination types in ``reduce_func`` must be the same.
        """
        pass

    def send_and_recv(self,
                      edges,
                      message_func="default",
                      reduce_func="default",
                      apply_node_func="default",
                      inplace=False):
        """Send messages along edges with the same edge type, and let destinations
        receive them.

        Optionally, apply a function to update the node features after receive.

        This is a convenient combination for performing
        ``send(self, self.edges, message_func)`` and
        ``recv(self, dst, reduce_func, apply_node_func)``, where ``dst``
        are the destinations of the ``edges``.

        Parameters
        ----------
        edges : valid edges type
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type.

            If the functions are not dicts, then ``edges`` must not be a dict.
            If the functions are dicts, then ``edges`` must either be
            * ALL: for computing on all edges with the given types in the
              functions.
            * a dict of int, iterable of int, or tensors, indicating the edges
              to be updated for each type.
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the destination type of ``edges``, the node types in
          ``apply_node_func``, and the destination types in ``reduce_func``
          must be the same.
        * the edge type of ``edges``, ``message_func`` and ``reduce_func``
          must also be the same.
        """
        pass

    def pull(self,
             v,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Pull messages from the node(s)' predecessors and then update their features.

        Optionally, apply a function to update the node features after receive.

        * `reduce_func` will be skipped for nodes with no incoming message.
        * If all ``v`` have no incoming message, this will downgrade to an :func:`apply_nodes`.
        * If some ``v`` have no incoming message, their new feature value will be calculated
          by the column initializer (see :func:`set_n_initializer`). The feature shapes and
          dtypes will be inferred.

        Parameters
        ----------
        v : int, container or tensor, dict, optional
            The node(s) to be updated. Default is receiving all the nodes.

            If the functions are not dicts, then ``v`` must not be a dict.
            If the functions are dicts, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in the
              functions.
            * a dict of int, iterable of int, or tensors, indicating the nodes
              to be updated for each type.
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types of ``v``, the node types in ``apply_node_func``,
          and the destination types in ``reduce_func`` must be the same.
        * the edge type of ``message_func`` and ``reduce_func`` must also be
          the same.
        """
        pass

    def push(self,
             u,
             message_func="default",
             reduce_func="default",
             apply_node_func="default",
             inplace=False):
        """Send message from the node(s) to their successors and update them.

        Optionally, apply a function to update the node features after receive.

        Parameters
        ----------
        u : int, container or tensor, dict
            The node(s) to push messages out.

            If the functions are not dicts, then ``v`` must not be a dict.
            If the functions are dicts, then ``v`` must either be
            * ALL: for computing on all nodes with the given types in the
              functions.
            * a dict of int, iterable of int, or tensors, indicating the nodes
              to be updated for each type.
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.
        inplace: bool, optional
            If True, update will be done in place, but autograd will break.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types in ``apply_node_func`` and the destination types in
          ``reduce_func`` must be the same.
        * the source types of ``message_func`` and the node types of ``u`` must
          be the same.
        * the edge type of ``message_func`` and ``reduce_func`` must also be
          the same.
        """
        pass

    def update_all(self,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Send messages through all edges and update all nodes.

        Optionally, apply a function to update the node features after receive.

        This is a convenient combination for performing
        ``send(self, self.edges(), message_func)`` and
        ``recv(self, self.nodes(), reduce_func, apply_node_func)``.

        Parameters
        ----------
        message_func : callable, dict[etype, callable], optional
            Message function on the edges. The function should be
            an :mod:`Edge UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            edge type.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``message_func`` is
            not a dict, it will throw an error.
        reduce_func : callable, dict[etype, callable], optional
            Reduce function on the node. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the messages will be aggregated onto the
            nodes by the edge type of the message.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.
            If the graph has more than one edge type and ``reduce_func`` is not
            a dict, it will throw an error.
        apply_node_func : callable, dict[str, callable], optional
            Apply function on the nodes. The function should be
            a :mod:`Node UDF <dgl.udf>`.

            If a dict is provided, the functions will be applied according to
            node type.
            If the graph has more than one node type and ``apply_func`` is not
            a dict, it will throw an error.

        Notes
        -----
        If the graph is heterogeneous (i.e. having more than one node/edge
        type),
        * the node types in ``apply_node_func`` and the destination types in
          ``reduce_func`` must be the same.
        * the edge type of ``message_func`` and ``reduce_func`` must also be
          the same.
        """
        pass

    # TODO should we support this?
    def prop_nodes(self,
                   nodes_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Node propagation in heterogeneous graph is not supported.
        """
        raise NotImplementedError('not supported')

    # TODO should we support this?
    def prop_edges(self,
                   edges_generator,
                   message_func="default",
                   reduce_func="default",
                   apply_node_func="default"):
        """Edge propagation in heterogeneous graph is not supported.
        """
        raise NotImplementedError('not supported')

    def subgraph(self, nodes):
        """Return the subgraph induced on given nodes.

        Parameters
        ----------
        nodes : dict[str, list or iterable]
            A dictionary of node types to node ID array to construct
            subgraph.
            All nodes must exist in the graph.

        Returns
        -------
        G : DGLHeteroSubGraph
            The subgraph.
            The nodes are relabeled so that node `i` of type `t` in the
            subgraph is mapped to the ``nodes[i]`` of type `t` in the
            original graph.
            The edges are also relabeled.
            One can retrieve the mapping from subgraph node/edge ID to parent
            node/edge ID via `parent_nid` and `parent_eid` properties of the
            subgraph.
        """
        pass

    def subgraphs(self, nodes):
        """Return a list of subgraphs, each induced in the corresponding given
        nodes in the list.

        Equivalent to
        ``[self.subgraph(nodes_list) for nodes_list in nodes]``

        Parameters
        ----------
        nodes : a list of dict[str, list or iterable]
            A list of type-ID dictionaries to construct corresponding
            subgraphs.  The dictionaries are of the same form as
            :func:`subgraph`.
            All nodes in all the list items must exist in the graph.

        Returns
        -------
        G : A list of DGLHeteroSubGraph
            The subgraphs.
        """
        pass

    def edge_subgraph(self, edges):
        """Return the subgraph induced on given edges.

        Parameters
        ----------
        edges : dict[etype, list or iterable]
            A dictionary of edge types to edge ID array to construct
            subgraph.
            All edges must exist in the subgraph.
            The edge type is characterized by a triplet of source type name,
            destination type name, and edge type name.

        Returns
        -------
        G : DGLHeteroSubGraph
            The subgraph.
            The edges are relabeled so that edge `i` of type `t` in the
            subgraph is mapped to the ``edges[i]`` of type `t` in the
            original graph.
            One can retrieve the mapping from subgraph node/edge ID to parent
            node/edge ID via `parent_nid` and `parent_eid` properties of the
            subgraph.
        """
        pass

    def adjacency_matrix_scipy(self, etype, transpose=False, fmt='csr'):
        """Return the scipy adjacency matrix representation of edges with the
        given edge type.

        By default, a row of returned adjacency matrix represents the destination
        of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column represents
        a destination.

        The elements in the adajency matrix are edge ids.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        transpose : bool, optional (default=False)
            A flag to transpose the returned adjacency matrix.
        fmt : str, optional (default='csr')
            Indicates the format of returned adjacency matrix.

        Returns
        -------
        scipy.sparse.spmatrix
            The scipy representation of adjacency matrix.
        """
        pass

    def adjacency_matrix(self, etype, transpose=False, ctx=F.cpu()):
        """Return the adjacency matrix representation of edges with the
        given edge type.

        By default, a row of returned adjacency matrix represents the
        destination of an edge and the column represents the source.

        When transpose is True, a row represents the source and a column
        represents a destination.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        transpose : bool, optional (default=False)
            A flag to transpose the returned adjacency matrix.
        ctx : context, optional (default=cpu)
            The context of returned adjacency matrix.

        Returns
        -------
        SparseTensor
            The adjacency matrix.
        """
        pass

    def incidence_matrix(self, etype, typestr, ctx=F.cpu()):
        """Return the incidence matrix representation of edges with the given
        edge type.

        An incidence matrix is an n x m sparse matrix, where n is
        the number of nodes and m is the number of edges. Each nnz
        value indicating whether the edge is incident to the node
        or not.

        There are three types of an incidence matrix :math:`I`:

        * ``in``:

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`
              (or :math:`v` is the dst node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``out``:

            - :math:`I[v, e] = 1` if :math:`e` is the out-edge of :math:`v`
              (or :math:`v` is the src node of :math:`e`);
            - :math:`I[v, e] = 0` otherwise.

        * ``both``:

            - :math:`I[v, e] = 1` if :math:`e` is the in-edge of :math:`v`;
            - :math:`I[v, e] = -1` if :math:`e` is the out-edge of :math:`v`;
            - :math:`I[v, e] = 0` otherwise (including self-loop).

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        typestr : str
            Can be either ``in``, ``out`` or ``both``
        ctx : context, optional (default=cpu)
            The context of returned incidence matrix.

        Returns
        -------
        SparseTensor
            The incidence matrix.
        """
        pass

    def filter_nodes(self, ntype, predicate, nodes=ALL):
        """Return a tensor of node IDs with the given node type that satisfy
        the given predicate.

        Parameters
        ----------
        ntype : str
            The node type.
        predicate : callable
            A function of signature ``func(nodes) -> tensor``.
            ``nodes`` are :class:`NodeBatch` objects as in :mod:`~dgl.udf`.
            The ``tensor`` returned should be a 1-D boolean tensor with
            each element indicating whether the corresponding node in
            the batch satisfies the predicate.
        nodes : int, iterable or tensor of ints
            The nodes to filter on. Default value is all the nodes.

        Returns
        -------
        tensor
            The filtered nodes.
        """
        pass

    def filter_edges(self, etype, predicate, edges=ALL):
        """Return a tensor of edge IDs with the given edge type that satisfy
        the given predicate.

        Parameters
        ----------
        etype : tuple[str, str, str]
            The edge type, characterized by a triplet of source type name,
            destination type name, and edge type name.
        predicate : callable
            A function of signature ``func(edges) -> tensor``.
            ``edges`` are :class:`EdgeBatch` objects as in :mod:`~dgl.udf`.
            The ``tensor`` returned should be a 1-D boolean tensor with
            each element indicating whether the corresponding edge in
            the batch satisfies the predicate.
        edges : valid edges type
            Edges on which to apply ``func``. See :func:`send` for valid
            edges type. Default value is all the edges.

        Returns
        -------
        tensor
            The filtered edges represented by their ids.
        """
        pass

    def readonly(self, readonly_state=True):
        """Set this graph's readonly state in-place.

        Parameters
        ----------
        readonly_state : bool, optional
            New readonly state of the graph, defaults to True.
        """
        pass

    def __repr__(self):
        pass

# pylint: disable=abstract-method
class DGLHeteroSubGraph(DGLHeteroGraph):
    """
    Parameters
    ----------
    parent : DGLHeteroGraph
        The parent graph.
    parent_nid : dict[str, utils.Index]
        The type-specific parent node IDs for each type.
    parent_eid : dict[etype, utils.Index]
        The type-specific parent edge IDs for each type.
    graph_idx : GraphIndex
        The graph index
    shared : bool, optional
        Whether the subgraph shares node/edge features with the parent graph
    """
    # pylint: disable=unused-argument, super-init-not-called
    def __init__(
            self,
            parent,
            parent_nid,
            parent_eid,
            graph_idx,
            shared=False):
        pass

    @property
    def parent_nid(self):
        """Get the parent node ids.

        The returned tensor dictionary can be used as a map from the node id
        in this subgraph to the node id in the parent graph.

        Returns
        -------
        dict[str, Tensor]
            The parent node id array for each type.
        """
        pass

    @property
    def parent_eid(self):
        """Get the parent edge ids.

        The returned tensor dictionary can be used as a map from the edge id
        in this subgraph to the edge id in the parent graph.

        Returns
        -------
        dict[etype, Tensor]
            The parent edge id array for each type.
            The edge types are characterized by a triplet of source type
            name, destination type name, and edge type name.
        """
        pass

    def copy_to_parent(self, inplace=False):
        """Write node/edge features to the parent graph.

        Parameters
        ----------
        inplace : bool
            If true, use inplace write (no gradient but faster)
        """
        pass

    def copy_from_parent(self):
        """Copy node/edge features from the parent graph.

        All old features will be removed.
        """
        pass

    def map_to_subgraph_nid(self, parent_vids):
        """Map the node IDs in the parent graph to the node IDs in the
        subgraph.

        Parameters
        ----------
        parent_vids : dict[str, list or tensor]
            The dictionary of node types to parent node ID array.

        Returns
        -------
        dict[str, tensor]
            The node ID array in the subgraph of each node type.
        """
        pass
