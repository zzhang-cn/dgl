import dgl
import backend as F
import unittest
from test_utils import parametrize_dtype

def tree1(idtype):
    """Generate a tree
         0
        / \
       1   2
      / \
     3   4
    Edges are from leaves to root.
    """
    g = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g.add_nodes(5)
    g.add_edge(3, 1)
    g.add_edge(4, 1)
    g.add_edge(1, 0)
    g.add_edge(2, 0)
    g.ndata['h'] = F.tensor([0, 1, 2, 3, 4])
    g.edata['h'] = F.randn((4, 10))
    return g

def tree2(idtype):
    """Generate a tree
         1
        / \
       4   3
      / \
     2   0
    Edges are from leaves to root.
    """
    g = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g.add_nodes(5)
    g.add_edge(2, 4)
    g.add_edge(0, 4)
    g.add_edge(4, 1)
    g.add_edge(3, 1)
    g.ndata['h'] = F.tensor([0, 1, 2, 3, 4])
    g.edata['h'] = F.randn((4, 10))
    return g

@parametrize_dtype
def test_batch_unbatch(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)

    bg = dgl.batch([t1, t2])
    assert bg.number_of_nodes() == 10
    assert bg.number_of_edges() == 8
    assert bg.batch_size == 2
    assert F.allclose(bg.batch_num_nodes(), F.tensor([5, 5]))
    assert F.allclose(bg.batch_num_edges(), F.tensor([4, 4]))

    tt1, tt2 = dgl.unbatch(bg)
    assert F.allclose(t1.ndata['h'], tt1.ndata['h'])
    assert F.allclose(t1.edata['h'], tt1.edata['h'])
    assert F.allclose(t2.ndata['h'], tt2.ndata['h'])
    assert F.allclose(t2.edata['h'], tt2.edata['h'])

@parametrize_dtype
def test_batch_unbatch1(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)
    b1 = dgl.batch([t1, t2])
    b2 = dgl.batch([t2, b1])
    assert b2.number_of_nodes() == 15
    assert b2.number_of_edges() == 12
    assert b2.batch_size == 3
    assert F.allclose(b2.batch_num_nodes(), F.tensor([5, 5, 5]))
    assert F.allclose(b2.batch_num_edges(), F.tensor([4, 4, 4]))

    s1, s2, s3 = dgl.unbatch(b2)
    assert F.allclose(t2.ndata['h'], s1.ndata['h'])
    assert F.allclose(t2.edata['h'], s1.edata['h'])
    assert F.allclose(t1.ndata['h'], s2.ndata['h'])
    assert F.allclose(t1.edata['h'], s2.edata['h'])
    assert F.allclose(t2.ndata['h'], s3.ndata['h'])
    assert F.allclose(t2.edata['h'], s3.edata['h'])

@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow" or dgl.backend.backend_name == "jax",
    reason="TF and JAX don't support inplace update"
)
@parametrize_dtype
def test_batch_unbatch_frame(idtype):
    """Test module of node/edge frames of batched/unbatched DGLGraphs.
    Also address the bug mentioned in https://github.com/dmlc/dgl/issues/1475.
    """
    t1 = tree1(idtype)
    t2 = tree2(idtype)
    N1 = t1.number_of_nodes()
    E1 = t1.number_of_edges()
    N2 = t2.number_of_nodes()
    E2 = t2.number_of_edges()
    D = 10
    t1.ndata['h'] = F.randn((N1, D))
    t1.edata['h'] = F.randn((E1, D))
    t2.ndata['h'] = F.randn((N2, D))
    t2.edata['h'] = F.randn((E2, D))

    b1 = dgl.batch([t1, t2])
    b2 = dgl.batch([t2])
    b1.ndata['h'][:N1] = F.zeros((N1, D))
    b1.edata['h'][:E1] = F.zeros((E1, D))
    b2.ndata['h'][:N2] = F.zeros((N2, D))
    b2.edata['h'][:E2] = F.zeros((E2, D))
    assert not F.allclose(t1.ndata['h'], F.zeros((N1, D)))
    assert not F.allclose(t1.edata['h'], F.zeros((E1, D)))
    assert not F.allclose(t2.ndata['h'], F.zeros((N2, D)))
    assert not F.allclose(t2.edata['h'], F.zeros((E2, D)))

    g1, g2 = dgl.unbatch(b1)
    _g2, = dgl.unbatch(b2)
    assert F.allclose(g1.ndata['h'], F.zeros((N1, D)))
    assert F.allclose(g1.edata['h'], F.zeros((E1, D)))
    assert F.allclose(g2.ndata['h'], t2.ndata['h'])
    assert F.allclose(g2.edata['h'], t2.edata['h'])
    assert F.allclose(_g2.ndata['h'], F.zeros((N2, D)))
    assert F.allclose(_g2.edata['h'], F.zeros((E2, D)))

@parametrize_dtype
def test_batch_unbatch2(idtype):
    # test setting/getting features after batch
    a = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    a.add_nodes(4)
    a.add_edges(0, [1, 2, 3])
    b = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    b.add_nodes(3)
    b.add_edges(0, [1, 2])
    c = dgl.batch([a, b])
    c.ndata['h'] = F.ones((7, 1))
    c.edata['w'] = F.ones((5, 1))
    assert F.allclose(c.ndata['h'], F.ones((7, 1)))
    assert F.allclose(c.edata['w'], F.ones((5, 1)))

@parametrize_dtype
def test_batch_send_and_recv(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)

    bg = dgl.batch([t1, t2])
    _mfunc = lambda edges: {'m' : edges.src['h']}
    _rfunc = lambda nodes: {'h' : F.sum(nodes.mailbox['m'], 1)}
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]

    bg.send_and_recv((u, v), _mfunc, _rfunc)

    t1, t2 = dgl.unbatch(bg)
    assert F.asnumpy(t1.ndata['h'][1]) == 7
    assert F.asnumpy(t2.ndata['h'][4]) == 2

@parametrize_dtype
def test_batch_propagate(idtype):
    t1 = tree1(idtype)
    t2 = tree2(idtype)

    bg = dgl.batch([t1, t2])
    _mfunc = lambda edges: {'m' : edges.src['h']}
    _rfunc = lambda nodes: {'h' : F.sum(nodes.mailbox['m'], 1)}
    # get leaves.

    order = []

    # step 1
    u = [3, 4, 2 + 5, 0 + 5]
    v = [1, 1, 4 + 5, 4 + 5]
    order.append((u, v))

    # step 2
    u = [1, 2, 4 + 5, 3 + 5]
    v = [0, 0, 1 + 5, 1 + 5]
    order.append((u, v))

    bg.prop_edges(order, _mfunc, _rfunc)
    t1, t2 = dgl.unbatch(bg)

    assert F.asnumpy(t1.ndata['h'][0]) == 9
    assert F.asnumpy(t2.ndata['h'][1]) == 5

@parametrize_dtype
def test_batched_edge_ordering(idtype):
    g1 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    e1 = F.randn((5, 10))
    g1.edata['h'] = e1
    g2 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g2.add_nodes(6)
    g2.add_edges([0, 1 ,2 ,5, 4 ,5], [1, 2, 3, 4, 3, 0])
    e2 = F.randn((6, 10))
    g2.edata['h'] = e2
    g = dgl.batch([g1, g2])
    r1 = g.edata['h'][g.edge_id(4, 5)]
    r2 = g1.edata['h'][g1.edge_id(4, 5)]
    assert F.array_equal(r1, r2)

@parametrize_dtype
def test_batch_no_edge(idtype):
    g1 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g1.add_nodes(6)
    g1.add_edges([4, 4, 2, 2, 0], [5, 3, 3, 1, 1])
    g2 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g2.add_nodes(6)
    g2.add_edges([0, 1, 2, 5, 4, 5], [1 ,2 ,3, 4, 3, 0])
    g3 = dgl.graph(([], [])).astype(idtype).to(F.ctx())
    g3.add_nodes(1)  # no edges
    g = dgl.batch([g1, g3, g2]) # should not throw an error

if __name__ == '__main__':
    test_batch_unbatch()
    test_batch_unbatch1()
    test_batch_unbatch_frame()
    #test_batch_unbatch2()
    #test_batched_edge_ordering()
    #test_batch_send_then_recv()
    #test_batch_send_and_recv()
    #test_batch_propagate()
    #test_batch_no_edge()
