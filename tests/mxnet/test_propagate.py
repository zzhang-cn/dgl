import os
os.environ['DGLBACKEND'] = 'mxnet'
import dgl
import networkx as nx
import numpy as np
import mxnet as mx

def mfunc(edges):
    return {'m' : edges.src['x']}

def rfunc(nodes):
    msg = mx.nd.sum(nodes.mailbox['m'], 1)
    return {'x' : nodes.data['x'] + msg}

def test_prop_nodes_bfs():
    g = dgl.DGLGraph(nx.path_graph(5))
    g.ndata['x'] = mx.nd.ones(shape=(5, 2))
    g.register_message_func(mfunc)
    g.register_reduce_func(rfunc)

    dgl.prop_nodes_bfs(g, 0)
    # pull nodes using bfs order will result in a cumsum[i] + data[i] + data[i+1]
    assert np.allclose(g.ndata['x'].asnumpy(),
            np.array([[2., 2.], [4., 4.], [6., 6.], [8., 8.], [9., 9.]]))

def test_prop_edges_dfs():
    g = dgl.DGLGraph(nx.path_graph(5))
    g.register_message_func(mfunc)
    g.register_reduce_func(rfunc)

    g.ndata['x'] = mx.nd.ones(shape=(5, 2))
    dgl.prop_edges_dfs(g, 0)
    # snr using dfs results in a cumsum
    assert np.allclose(g.ndata['x'].asnumpy(),
            np.array([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]]))

    g.ndata['x'] = mx.nd.ones(shape=(5, 2))
    dgl.prop_edges_dfs(g, 0, has_reverse_edge=True)
    # result is cumsum[i] + cumsum[i-1]
    assert np.allclose(g.ndata['x'].asnumpy(),
            np.array([[1., 1.], [3., 3.], [5., 5.], [7., 7.], [9., 9.]]))

    g.ndata['x'] = mx.nd.ones(shape=(5, 2))
    dgl.prop_edges_dfs(g, 0, has_nontree_edge=True)
    # result is cumsum[i] + cumsum[i+1]
    assert np.allclose(g.ndata['x'].asnumpy(),
            np.array([[3., 3.], [5., 5.], [7., 7.], [9., 9.], [5., 5.]]))

def test_prop_nodes_topo():
    # bi-directional chain
    g = dgl.DGLGraph(nx.path_graph(5))

    # tree
    tree = dgl.DGLGraph()
    tree.add_nodes(5)
    tree.add_edge(1, 0)
    tree.add_edge(2, 0)
    tree.add_edge(3, 2)
    tree.add_edge(4, 2)
    tree.register_message_func(mfunc)
    tree.register_reduce_func(rfunc)
    # init node feature data
    tree.ndata['x'] = mx.nd.zeros(shape=(5, 2))
    # set all leaf nodes to be ones
    tree.nodes[[1, 3, 4]].data['x'] = mx.nd.ones(shape=(3, 2))
    dgl.prop_nodes_topo(tree)
    # root node get the sum
    assert np.allclose(tree.nodes[0].data['x'].asnumpy(), np.array([[3., 3.]]))

if __name__ == '__main__':
    test_prop_nodes_bfs()
    test_prop_edges_dfs()
    test_prop_nodes_topo()
