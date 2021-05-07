import os
import scipy.sparse as sp
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

import numpy as np
import torch as th
import dgl 

class MetricLogger(object):
    def __init__(self, save_dir, log_interval):
        self.save_dir = save_dir
        self.log_interval = log_interval

    def log(self, info, model, optimizer):
        epoch, train_loss, test_rmse = info['epoch'], info['train_loss'], info['test_rmse']
        with open(os.path.join(self.save_dir, 'log.txt'), 'a') as f:
            f.write('Epoch {}, train loss {:.4f}, test rmse {:.6f}\n'.format(
                epoch, train_loss, test_rmse))
        # if type(epoch) == int and epoch % self.log_interval == 0:
        #     print('Saving model states...')
        #     model_name = os.path.join(self.save_dir, 'model_checkpoint{}.pth'.format(epoch))
        #     optimizer_name = os.path.join(
        #         self.save_dir, 'optimizer_checkpoint{}.pth'.format(epoch)
        #     )
        #     if model is not None:
        #         th.save(model.state_dict(), model_name)
        #     if optimizer is not None:
        #         th.save(optimizer.state_dict(), optimizer_name)

def torch_total_param_num(net):
    return sum([np.prod(p.shape) for p in net.parameters()])

def torch_net_info(net, save_path=None):
    info_str = 'Total Param Number: {}\n'.format(torch_total_param_num(net)) +\
               'Params:\n'
    for k, v in net.named_parameters():
        info_str += '\t{}: {}, {}\n'.format(k, v.shape, np.prod(v.shape))
    info_str += str(net)
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(info_str)
    return info_str

def one_hot(idx, length):
    x = th.zeros([len(idx), length])
    x[th.arange(len(idx)), idx] = 1.0
    return x

def cal_dist(csr_graph, node_to_remove):
    # cal dist to node 0, with target edge nodes 0/1 removed
    nodes = list(set(range(csr_graph.shape[1])) - set([node_to_remove]))
    csr_graph = csr_graph[nodes, :][:, nodes]
    dists = np.clip(sp.csgraph.dijkstra(
                        csr_graph, indices=0, directed=False, unweighted=True, limit=1e6
                    )[1:], 0, 1e7)
    return dists.astype(np.int64)

def subgraph_extraction_labeling(graph, edge, h=1, sample_ratio=1.0, max_nodes_per_hop=200):
    """Extract the h-hop sampled enclosing subgraph around given edge with hops as node labels.

    DGL constructs an ``h``-hop sampled enclosing subgraph as follows.

    1. DGL selects the incident nodes of the target edge and sets them as the active set.
    2. DGL then samples a subset of nodes from the neighbors of the active set. The size of the 
       subset is a given ratio of the entire neighborhood, clipped by the given maximum number of
       nodes at each hop.
    3. DGL sets the sampled nodes as the active set, and repeats steps 2 and 3 ``h`` times in total.
    4. DGL induces a subgraph from all the nodes that appeared in the active set at least once.

    DGL assigns to each node an integer ``i`` if the node appears in the active set at the ``i``-th
    hop.

    Parameters
    ----------
    graph: DGLGraph
        The graph to extract subgraph from. Must be homogeneous. 
    edge: tuple of (int, int), ``(u, v)``
        The target edge for extracting subgraph.
    h: int, default 1
        The number of hop.
    sample_ratio: float, default 1.0
        The sampling ratio of neighbor nodes at each hop. 
    max_nodes_per_hop: int, default 200
        Max number of node to be sampled at each hop.

    Returns
    -------
    subgraph: DGLGraph
        The subgraph extracted with extra node attributes ``nlabel``.
    
    Examples
    --------
    The following example uses PyTorch backend.

    >>> edges = (th.tensor([0, 1, 1, 2, 2, 2]), 
    ...          th.tensor([3, 3, 4, 3, 4, 5]))
    >>> g = dgl.graph((th.cat([edges[0], edges[1]]), th.cat([edges[1], edges[0]])))
    >>> edge = (g.edges()[0][1], g.edges()[1][1]) # edge = (tensor(1), tensor(3))
    >>> sg = subgraph_extraction_labeling(g, edge)
    >>> sg.ndata[dgl.NID], sg.ndata['nlabel']
    (tensor([1, 3, 0, 2, 4]), tensor([0, 0, 1, 1, 1]))

    """

    # 1. neighbor nodes sampling
    dist = 0
    nodes = th.as_tensor(edge)
    dists = th.zeros_like(nodes) 
    visited = th.unique(nodes)
    fringe = th.unique(nodes)

    for dist in range(1, h+1):
        fringe = graph.in_edges(fringe)[0]    
        fringe = th.from_numpy(np.setdiff1d(fringe.numpy(), visited.numpy()))
        visited = th.unique(th.cat([visited, fringe]))

        if sample_ratio < 1.0:
            shuffled_idx = th.randperm(len(fringe))
            fringe = fringe[shuffled_idx[:int(sample_ratio*len(fringe))]]
        if max_nodes_per_hop is not None and max_nodes_per_hop < len(fringe):
            shuffled_idx = th.randperm(len(fringe))
            fringe = fringe[shuffled_idx[:max_nodes_per_hop]]
        if len(fringe) == 0:
            break
        nodes = th.cat([nodes, fringe])
        dists = th.cat([dists, th.full((len(fringe), ), dist, dtype=th.int64)])
    
    # 2. node labeling
    node_labels = dists
    
    # 3. extract subgraph with sampled nodes
    subgraph = graph.subgraph(nodes)
    subgraph.ndata['nlabel'] = node_labels

    return subgraph

def igmc_subgraph_extraction_labeling(graph, edge, h=1, sample_ratio=1.0, max_nodes_per_hop=200):
    """Extract the h-hop sampled enclosing subgraph around given edge with node labels in IGMC model.

    DGL induces an enclosing subgraph for a node pair ``(u, v)`` by the union of ``u`` and ``v``'s sampled 
    neighbors up to ``h`` hops. DGL assigns node with labels :math:`2 \\times i + j`, if the node appears
    at the :math:`i`-th hop, where :math:`j` is 0 if the node has the same type as node ``u``, otherwise 1.

    For more information, please see paper 
    `Inductive Matrix Completion Based on Graph Neural Networks <https://arxiv.org/abs/1904.12058>`__

    Parameters
    ----------
    graph: DGLGraph.
        The graph to extract subgraph from. Must be homogeneous graph with bipartite structure.
    edge: tuple of (int, int), ``(u, v)``
        The target edge for extracting subgraph.
    h: int, default 1
        The number of hop.
    sample_ratio: float, default 1.0
        The sampling ratio of neighbor nodes at each hop. 
    max_nodes_per_hop: int, default 200
        Max number of node to be sampled at each hop.

    Returns
    -------
    subgraph: DGLGraph
        The subgraph extracted with extra node attributes ``nlabel``.
    
    Examples
    --------
    The following example uses PyTorch backend.

    >>> u = th.tensor([0, 1, 1, 2, 2, 2])
    >>> v = th.tensor([3, 3, 4, 3, 4, 5])
    >>> g = dgl.graph((th.cat([u, v]), th.cat([v, u])))
    >>> edge = (g.edges()[0][1], g.edges()[1][1]) # edge = (tensor(1), tensor(3))
    >>> sg = igmc_subgraph_extraction_labeling(g, edge)
    >>> sg.ndata[dgl.NID], sg.ndata['nlabel']
    (tensor([1, 0, 2, 3, 4]), tensor([0, 2, 2, 1, 3]))

    """

    # 1. neighbor nodes sampling
    dist = 0
    u_nodes, v_nodes = th.as_tensor([edge[0]]), th.as_tensor([edge[1]])
    u_dist, v_dist = th.tensor([0]), th.tensor([0])
    u_visited, v_visited = th.unique(u_nodes), th.unique(v_nodes)
    u_fringe, v_fringe = th.unique(u_nodes), th.unique(v_nodes)

    for dist in range(1, h+1):
        # sample neigh alternately
        u_fringe, v_fringe = graph.in_edges(v_fringe)[0], graph.in_edges(u_fringe)[0]
        u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
        v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
        u_visited = th.unique(th.cat([u_visited, u_fringe]))
        v_visited = th.unique(th.cat([v_visited, v_fringe]))

        if sample_ratio < 1.0:
            shuffled_idx = th.randperm(len(u_fringe))
            u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
            shuffled_idx = th.randperm(len(v_fringe))
            v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
            if max_nodes_per_hop < len(v_fringe):
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = th.cat([u_nodes, u_fringe])
        v_nodes = th.cat([v_nodes, v_fringe])
        u_dist = th.cat([u_dist, th.full((len(u_fringe), ), dist, dtype=th.int64)])
        v_dist = th.cat([v_dist, th.full((len(v_fringe), ), dist, dtype=th.int64)])

    nodes = th.cat([u_nodes, v_nodes])

    # 2. node labeling
    u_node_labels = th.stack([x*2 for x in u_dist])
    v_node_labels = th.stack([x*2+1 for x in v_dist])
    node_labels = th.cat([u_node_labels, v_node_labels])
    
    # 3. extract subgraph with sampled nodes
    subgraph = graph.subgraph(nodes)
    subgraph.ndata['nlabel'] = node_labels

    return subgraph

def grail_subgraph_extraction_labeling(graph, edge, h=1, sample_ratio=1.0, max_nodes_per_hop=200):
    """Extract the h-hop sampled enclosing subgraph around given edge with with node labels in GraIL model.

    DGL removes edges between the input node pair ``(u, v)`` in the graph. DGL then assigns label to 
    sampled node ``i`` as the concatenation of distance from the node ``i`` to each ``u`` and ``v`` 
    in the pruned graph, which is :math:`[dist(i, u), dist(i, v)]`.

    For more information, please see paper 
    `Inductive Relation Prediction by Subgraph Reasoning <https://arxiv.org/abs/1911.06962>`__

    Parameters
    ----------
    graph: DGLGraph
        The graph to extract subgraph from. Must be homogeneous. 
    edge: tuple of (int, int), ``(u, v)``
        The target edge for extracting subgraph.
    h: int, default 1
        The number of hop.
    sample_ratio: float, default 1.0
        The sampling ratio of neighbor nodes at each hop. 
    max_nodes_per_hop: int, default 200
        Max number of node to be sampled at each hop.

    Returns
    -------
    subgraph: DGLGraph
        The subgraph extracted with extra node attributes ``nlabel``.
    
    Examples
    --------
    The following example uses PyTorch backend.

    >>> edges = (th.tensor([0, 1, 1, 2, 2, 2, 3, 4]),  
    ...          th.tensor([2, 2, 3, 3, 4, 5, 5, 5]))  
    >>> g = dgl.graph((th.cat([edges[0], edges[1]]), th.cat([edges[1], edges[0]])))
    >>> edge = (g.edges()[0][3], g.edges()[1][3]) # (tensor(2), tensor(3))
    >>> sg = grail_subgraph_extraction_labeling(g, edge, h=2)
    >>> sg.ndata[dgl.NID]
    tensor([2, 3, 1, 4, 5])
    >>> sg.ndata['nlabel']
    tensor([[0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
            [1, 1]])
    
    """
    # 1. neighbor nodes sampling
    # make sure target nodes not in uv.
    u_nodes, v_nodes = th.tensor([], dtype=th.long), th.tensor([], dtype=th.long)
    # u_dist, v_dist = th.tensor([0]), th.tensor([0])
    u_visited, v_visited = th.tensor([edge[0]]), th.tensor([edge[1]])
    u_fringe, v_fringe = th.tensor([edge[0]]), th.tensor([edge[1]])

    for dist in range(1, h+1):
        # sample neigh separately
        u_fringe = graph.in_edges(u_fringe)[0]
        v_fringe = graph.in_edges(v_fringe)[0]

        u_fringe = th.from_numpy(np.setdiff1d(u_fringe.numpy(), u_visited.numpy()))
        v_fringe = th.from_numpy(np.setdiff1d(v_fringe.numpy(), v_visited.numpy()))
        u_visited = th.unique(th.cat([u_visited, u_fringe]))
        v_visited = th.unique(th.cat([v_visited, v_fringe]))

        if sample_ratio < 1.0:
            shuffled_idx = th.randperm(len(u_fringe))
            u_fringe = u_fringe[shuffled_idx[:int(sample_ratio*len(u_fringe))]]
            shuffled_idx = th.randperm(len(v_fringe))
            v_fringe = v_fringe[shuffled_idx[:int(sample_ratio*len(v_fringe))]]
        if max_nodes_per_hop is not None:
            if max_nodes_per_hop < len(u_fringe):
                shuffled_idx = th.randperm(len(u_fringe))
                u_fringe = u_fringe[shuffled_idx[:max_nodes_per_hop]]
            if max_nodes_per_hop < len(v_fringe):
                shuffled_idx = th.randperm(len(v_fringe))
                v_fringe = v_fringe[shuffled_idx[:max_nodes_per_hop]]
        if len(u_fringe) == 0 and len(v_fringe) == 0:
            break
        u_nodes = th.cat([u_nodes, u_fringe])
        v_nodes = th.cat([v_nodes, v_fringe])

    nodes = th.from_numpy(np.intersect1d(u_nodes.numpy(), v_nodes.numpy()))
    # concatenate target nodes to the front.
    nodes = th.cat([th.tensor(edge), nodes])
    
    # 2. node labeling
    csr_subgraph = graph.subgraph(nodes).adjacency_matrix(scipy_fmt="csr")
    dists = th.stack([th.tensor(cal_dist(csr_subgraph, 1)), 
                      th.tensor(cal_dist(csr_subgraph, 0))], axis=1)
    edge_labels = th.tensor([[0, 1], [1, 0]])
    node_labels = th.cat([edge_labels, dists]) if dists.size() else edge_labels

    # 3. prune nodes that are at a distance greater than hop from neigh of the target nodes
    pruned_mask = th.max(node_labels, axis=1)[0] <= h
    nodes, node_labels = nodes[pruned_mask], node_labels[pruned_mask]

    # 4. extract subgraph with sampled nodes
    subgraph = graph.subgraph(nodes)
    subgraph.ndata['nlabel'] = node_labels

    return subgraph

if __name__ == "__main__":
    import time
    from data import MovieLens
    movielens = MovieLens("ml-100k", testing=True)

    train_edges = movielens.train_rating_pairs
    train_graph = movielens.train_graph

    idx = 0
    u, v = train_edges[0][idx], train_edges[1][idx]
    subgraph = subgraph_extraction_labeling(
                    (u, v), train_graph, 
                    hop=1, sample_ratio=1.0, max_nodes_per_hop=200)
    pass
