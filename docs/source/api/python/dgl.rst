.. _apidgl:

dgl
=============================

.. currentmodule:: dgl
.. automodule:: dgl

.. _api-graph-create-ops:

Graph Create Ops
-------------------------

Operators for constructing :class:`DGLGraph` from raw data formats.

.. autosummary::
    :toctree: ../../generated/

    graph
    heterograph
    from_scipy
    from_networkx
    bipartite_from_scipy
    bipartite_from_networkx
    rand_graph
    rand_bipartite
    knn_graph
    segmented_knn_graph
    create_block
    block_to_graph

.. _api-subgraph-extraction:

Subgraph Extraction Ops
-------------------------------------

Operators for extracting and returning subgraphs.

.. autosummary::
    :toctree: ../../generated/

    node_subgraph
    edge_subgraph
    node_type_subgraph
    edge_type_subgraph
    in_subgraph
    out_subgraph

.. _api-transform:

Graph Transform Ops
----------------------------------

Operators for generating new graphs by manipulating the structure of the existing ones.

.. autosummary::
    :toctree: ../../generated/

    add_nodes
    add_edges
    remove_nodes
    remove_edges
    add_self_loop
    remove_self_loop
    add_reverse_edges
    reverse
    to_bidirected
    to_simple
    to_block
    compact_graphs
    to_heterogeneous
    to_homogeneous
    to_networkx
    line_graph
    khop_graph
    metapath_reachable_graph
    adj_product_graph
    adj_sum_graph

.. _api-batch:

Batching and Reading Out Ops
-------------------------------

Operators for batching multiple graphs into one for batch processing and
operators for computing graph-level representation for both single and batched graphs.

.. autosummary::
    :toctree: ../../generated/

    batch
    unbatch
    readout_nodes
    readout_edges
    sum_nodes
    sum_edges
    mean_nodes
    mean_edges
    max_nodes
    max_edges
    softmax_nodes
    softmax_edges
    broadcast_nodes
    broadcast_edges
    topk_nodes
    topk_edges

Adjacency Related Utilities
-------------------------------

Utilities for computing adjacency matrix and Lapacian matrix.

.. autosummary::
    :toctree: ../../generated/

    khop_adj
    laplacian_lambda_max

Graph Traversal & Message Propagation
------------------------------------------

DGL implements graph traversal algorithms implemented as python generators,
which returns the visited set of nodes or edges (in ID tensor) at each iteration.
The naming convention is ``<algorithm>_[nodes|edges]_generator``.
An example usage is as follows.

.. code:: python

    g = ...  # some DGLGraph
    for nodes in dgl.bfs_nodes_generator(g, 0):
        do_something(nodes)

.. autosummary::
    :toctree: ../../generated/

    bfs_nodes_generator
    bfs_edges_generator
    topological_nodes_generator
    dfs_edges_generator
    dfs_labeled_edges_generator

DGL provides APIs to perform message passing following graph traversal order. ``prop_nodes_XXX``
calls traversal algorithm ``XXX`` and triggers :func:`~DGLGraph.pull()` on the visited node
set at each iteration. ``prop_edges_YYY`` applies traversal algorithm ``YYY`` and triggers
:func:`~DGLGraph.send_and_recv()` on the visited edge set at each iteration.

.. autosummary::
    :toctree: ../../generated/

    prop_nodes
    prop_nodes_bfs
    prop_nodes_topo
    prop_edges
    prop_edges_dfs

Utilities
-----------------------------------------------

Other utilities for controlling randomness, saving and loading graphs, etc.

.. autosummary::
    :toctree: ../../generated/

    seed
    save_graphs
    load_graphs
