import math
import numbers
import numpy as np
import scipy.sparse as sp
import networkx as nx
import dgl
import backend as F
from dgl import DGLError
import pytest

# graph generation: a random graph with 10 nodes
#  and 20 edges.
#  - has self loop
#  - no multi edge
def edge_pair_input(sort=False):
    if sort:
        src = [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 5, 5, 6, 7, 7, 7, 9]
        dst = [4, 6, 9, 3, 5, 3, 7, 5, 8, 1, 3, 4, 9, 1, 9, 6, 2, 8, 9, 2]
        return src, dst
    else:
        src = [0, 0, 4, 5, 0, 4, 7, 4, 4, 3, 2, 7, 7, 5, 3, 2, 1, 9, 6, 1]
        dst = [9, 6, 3, 9, 4, 4, 9, 9, 1, 8, 3, 2, 8, 1, 5, 7, 3, 2, 6, 5]
        return src, dst

def nx_input():
    g = nx.DiGraph()
    src, dst = edge_pair_input()
    for i, e in enumerate(zip(src, dst)):
        g.add_edge(*e, id=i)
    return g

def elist_input():
    src, dst = edge_pair_input()
    return list(zip(src, dst))

def scipy_coo_input():
    src, dst = edge_pair_input()
    return sp.coo_matrix((np.ones((20,)), (src, dst)), shape=(10,10))

def scipy_csr_input():
    src, dst = edge_pair_input()
    csr = sp.coo_matrix((np.ones((20,)), (src, dst)), shape=(10,10)).tocsr()
    csr.sort_indices()
    # src = [0 0 0 1 1 2 2 3 3 4 4 4 4 5 5 6 7 7 7 9]
    # dst = [4 6 9 3 5 3 7 5 8 1 3 4 9 1 9 6 2 8 9 2]
    return csr

def gen_by_mutation():
    g = dgl.DGLGraph()
    src, dst = edge_pair_input()
    g.add_nodes(10)
    g.add_edges(src, dst)
    return g

def gen_from_data(data, readonly, sort):
    return dgl.DGLGraph(data, readonly=readonly, sort_csr=True)

def test_query():
    def _test_one(g):
        assert g.number_of_nodes() == 10
        assert g.number_of_edges() == 20
        assert len(g) == 10

        for i in range(10):
            assert g.has_node(i)
            assert i in g
        assert not g.has_node(11)
        assert not 11 in g
        assert F.allclose(g.has_nodes([0,2,10,11]), F.tensor([1,1,0,0]))

        src, dst = edge_pair_input()
        for u, v in zip(src, dst):
            assert g.has_edge_between(u, v)
        assert not g.has_edge_between(0, 0)
        assert F.allclose(g.has_edges_between([0, 0, 3], [0, 9, 8]), F.tensor([0,1,1]))
        assert set(F.asnumpy(g.predecessors(9))) == set([0,5,7,4])
        assert set(F.asnumpy(g.successors(2))) == set([7,3])

        assert g.edge_id(4,4) == 5
        assert F.allclose(g.edge_ids([4,0], [4,9]), F.tensor([5,0]))

        src, dst = g.find_edges([3, 6, 5])
        assert F.allclose(src, F.tensor([5, 7, 4]))
        assert F.allclose(dst, F.tensor([9, 9, 4]))

        src, dst, eid = g.in_edges(9, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(5,9,3),(7,9,6),(4,9,7)])
        src, dst, eid = g.in_edges([9,0,8], form='all')  # test node#0 has no in edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(5,9,3),(7,9,6),(4,9,7),(3,8,9),(7,8,12)])

        src, dst, eid = g.out_edges(0, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(0,6,1),(0,4,4)])
        src, dst, eid = g.out_edges([0,4,8], form='all')  # test node#8 has no out edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,0),(0,6,1),(0,4,4),(4,3,2),(4,4,5),(4,9,7),(4,1,8)])

        src, dst, eid = g.edges('all', 'eid')
        t_src, t_dst = edge_pair_input()
        t_tup = list(zip(t_src, t_dst, list(range(20))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(eid)) == list(range(20))

        src, dst, eid = g.edges('all', 'srcdst')
        t_src, t_dst = edge_pair_input()
        t_tup = list(zip(t_src, t_dst, list(range(20))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(src)) == sorted(list(F.asnumpy(src)))

        assert g.in_degree(0) == 0
        assert g.in_degree(9) == 4
        assert F.allclose(g.in_degrees([0, 9]), F.tensor([0, 4]))
        assert g.out_degree(8) == 0
        assert g.out_degree(9) == 1
        assert F.allclose(g.out_degrees([8, 9]), F.tensor([0, 1]))

        assert np.array_equal(
                F.sparse_to_numpy(g.adjacency_matrix(transpose=False)), scipy_coo_input().toarray().T)
        assert np.array_equal(
                F.sparse_to_numpy(g.adjacency_matrix(transpose=True)), scipy_coo_input().toarray())

    def _test(g):
        # test twice to see whether the cached format works or not
        _test_one(g)
        _test_one(g)

    def _test_csr_one(g):
        assert g.number_of_nodes() == 10
        assert g.number_of_edges() == 20
        assert len(g) == 10

        for i in range(10):
            assert g.has_node(i)
            assert i in g
        assert not g.has_node(11)
        assert not 11 in g
        assert F.allclose(g.has_nodes([0,2,10,11]), F.tensor([1,1,0,0]))

        src, dst = edge_pair_input(sort=True)
        for u, v in zip(src, dst):
            assert g.has_edge_between(u, v)
        assert not g.has_edge_between(0, 0)
        assert F.allclose(g.has_edges_between([0, 0, 3], [0, 9, 8]), F.tensor([0,1,1]))
        assert set(F.asnumpy(g.predecessors(9))) == set([0,5,7,4])
        assert set(F.asnumpy(g.successors(2))) == set([7,3])

        # src = [0 0 0 1 1 2 2 3 3 4 4 4 4 5 5 6 7 7 7 9]
        # dst = [4 6 9 3 5 3 7 5 8 1 3 4 9 1 9 6 2 8 9 2]
        # eid = [0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9]
        assert g.edge_id(4,4) == 11
        assert F.allclose(g.edge_ids([4,0], [4,9]), F.tensor([11,2]))

        src, dst = g.find_edges([3, 6, 5])
        assert F.allclose(src, F.tensor([1, 2, 2]))
        assert F.allclose(dst, F.tensor([3, 7, 3]))

        src, dst, eid = g.in_edges(9, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(5,9,14),(7,9,18),(4,9,12)])
        src, dst, eid = g.in_edges([9,0,8], form='all')  # test node#0 has no in edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(5,9,14),(7,9,18),(4,9,12),(3,8,8),(7,8,17)])

        src, dst, eid = g.out_edges(0, form='all')
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(0,6,1),(0,4,0)])
        src, dst, eid = g.out_edges([0,4,8], form='all')  # test node#8 has no out edges
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set([(0,9,2),(0,6,1),(0,4,0),(4,3,10),(4,4,11),(4,9,12),(4,1,9)])

        src, dst, eid = g.edges('all', 'eid')
        t_src, t_dst = edge_pair_input(sort=True)
        t_tup = list(zip(t_src, t_dst, list(range(20))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(eid)) == list(range(20))

        src, dst, eid = g.edges('all', 'srcdst')
        t_src, t_dst = edge_pair_input(sort=True)
        t_tup = list(zip(t_src, t_dst, list(range(20))))
        tup = list(zip(F.asnumpy(src), F.asnumpy(dst), F.asnumpy(eid)))
        assert set(tup) == set(t_tup)
        assert list(F.asnumpy(src)) == sorted(list(F.asnumpy(src)))

        assert g.in_degree(0) == 0
        assert g.in_degree(9) == 4
        assert F.allclose(g.in_degrees([0, 9]), F.tensor([0, 4]))
        assert g.out_degree(8) == 0
        assert g.out_degree(9) == 1
        assert F.allclose(g.out_degrees([8, 9]), F.tensor([0, 1]))

        assert np.array_equal(
                F.sparse_to_numpy(g.adjacency_matrix(transpose=False)), scipy_coo_input().toarray().T)
        assert np.array_equal(
                F.sparse_to_numpy(g.adjacency_matrix(transpose=True)), scipy_coo_input().toarray())

    def _test_csr(g):
        # test twice to see whether the cached format works or not
        _test_csr_one(g)
        _test_csr_one(g)

    def _test_edge_ids():
        g = gen_by_mutation()
        eids = g.edge_ids([4,0], [4,9])
        assert eids.shape[0] == 2
        eid = g.edge_id(4, 4)
        assert isinstance(eid, numbers.Number)
        with pytest.raises(DGLError):
            eids = g.edge_ids([9,0], [4,9])

        with pytest.raises(DGLError):
            eid = g.edge_id(4, 5)

        g.add_edge(0, 4)
        eids = g.edge_ids([0,0], [4,9])
        eid = g.edge_id(0, 4)

    _test(gen_by_mutation())
    _test(gen_from_data(elist_input(), False, False))
    _test(gen_from_data(elist_input(), True, False))
    _test(gen_from_data(elist_input(), True, True))
    _test(gen_from_data(scipy_coo_input(), False, False))
    _test(gen_from_data(scipy_coo_input(), True, False))

    _test_csr(gen_from_data(scipy_csr_input(), False, False))
    _test_csr(gen_from_data(scipy_csr_input(), True, False))
    _test_edge_ids()

def test_mutation():
    g = dgl.DGLGraph()
    g = g.to(F.ctx())
    # test add nodes with data
    g.add_nodes(5)
    g.add_nodes(5, {'h' : F.ones((5, 2))})
    ans = F.cat([F.zeros((5, 2)), F.ones((5, 2))], 0)
    assert F.allclose(ans, g.ndata['h'])
    g.ndata['w'] = 2 * F.ones((10, 2))
    assert F.allclose(2 * F.ones((10, 2)), g.ndata['w'])
    # test add edges with data
    g.add_edges([2, 3], [3, 4])
    g.add_edges([0, 1], [1, 2], {'m' : F.ones((2, 2))})
    ans = F.cat([F.zeros((2, 2)), F.ones((2, 2))], 0)
    assert F.allclose(ans, g.edata['m'])

def test_scipy_adjmat():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    g.add_edges(range(9), range(1, 10))

    adj_0 = g.adj(scipy_fmt='csr')
    adj_1 = g.adj(scipy_fmt='coo')
    assert np.array_equal(adj_0.toarray(), adj_1.toarray())

    adj_t0 = g.adj(transpose=True, scipy_fmt='csr')
    adj_t_1 = g.adj(transpose=True, scipy_fmt='coo')
    assert np.array_equal(adj_0.toarray(), adj_1.toarray())

def test_incmat():
    g = dgl.DGLGraph()
    g.add_nodes(4)
    g.add_edge(0, 1) # 0
    g.add_edge(0, 2) # 1
    g.add_edge(0, 3) # 2
    g.add_edge(2, 3) # 3
    g.add_edge(1, 1) # 4
    inc_in = F.sparse_to_numpy(g.incidence_matrix('in'))
    inc_out = F.sparse_to_numpy(g.incidence_matrix('out'))
    inc_both = F.sparse_to_numpy(g.incidence_matrix('both'))
    print(inc_in)
    print(inc_out)
    print(inc_both)
    assert np.allclose(
            inc_in,
            np.array([[0., 0., 0., 0., 0.],
                      [1., 0., 0., 0., 1.],
                      [0., 1., 0., 0., 0.],
                      [0., 0., 1., 1., 0.]]))
    assert np.allclose(
            inc_out,
            np.array([[1., 1., 1., 0., 0.],
                      [0., 0., 0., 0., 1.],
                      [0., 0., 0., 1., 0.],
                      [0., 0., 0., 0., 0.]]))
    assert np.allclose(
            inc_both,
            np.array([[-1., -1., -1., 0., 0.],
                      [1., 0., 0., 0., 0.],
                      [0., 1., 0., -1., 0.],
                      [0., 0., 1., 1., 0.]]))

def test_find_edges():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    g.add_edges(range(9), range(1, 10))
    e = g.find_edges([1, 3, 2, 4])
    assert F.asnumpy(e[0][0]) == 1 and F.asnumpy(e[0][1]) == 3 and F.asnumpy(e[0][2]) == 2 and F.asnumpy(e[0][3]) == 4
    assert F.asnumpy(e[1][0]) == 2 and F.asnumpy(e[1][1]) == 4 and F.asnumpy(e[1][2]) == 3 and F.asnumpy(e[1][3]) == 5

    try:
        g.find_edges([10])
        fail = False
    except DGLError:
        fail = True
    finally:
        assert fail

def test_ismultigraph():
    g = dgl.DGLGraph()
    g.add_nodes(10)
    assert g.is_multigraph == False
    g.add_edges([0], [0])
    assert g.is_multigraph == False
    g.add_edges([1], [2])
    assert g.is_multigraph == False
    g.add_edges([0, 2], [0, 3])
    assert g.is_multigraph == True

def test_hypersparse_query():
    g = dgl.DGLGraph()
    g = g.to(F.ctx())
    g.add_nodes(1000001)
    g.add_edges([0], [1])
    for i in range(10):
        assert g.has_node(i)
        assert i in g
    assert not g.has_node(1000002)
    assert g.edge_id(0, 1) == 0
    src, dst = g.find_edges([0])
    src, dst, eid = g.in_edges(1, form='all')
    src, dst, eid = g.out_edges(0, form='all')
    src, dst = g.edges()
    assert g.in_degree(0) == 0
    assert g.in_degree(1) == 1
    assert g.out_degree(0) == 1
    assert g.out_degree(1) == 0

def test_empty_data_initialized():
    g = dgl.DGLGraph()
    g = g.to(F.ctx())
    g.ndata["ha"] = F.tensor([])
    g.add_nodes(1, {"hb": F.tensor([1])})
    assert "ha" in g.ndata
    assert len(g.ndata["ha"]) == 1

def test_is_sorted():
   u_src, u_dst = edge_pair_input(False) 
   s_src, s_dst = edge_pair_input(True) 

   u_src = F.tensor(u_src, dtype=F.int32)
   u_dst = F.tensor(u_dst, dtype=F.int32)
   s_src = F.tensor(s_src, dtype=F.int32)
   s_dst = F.tensor(s_dst, dtype=F.int32)

   src_sorted, dst_sorted = dgl.utils.is_sorted_srcdst(u_src, u_dst)
   assert src_sorted == False
   assert dst_sorted == False

   src_sorted, dst_sorted = dgl.utils.is_sorted_srcdst(s_src, s_dst)
   assert src_sorted == True
   assert dst_sorted == True

   src_sorted, dst_sorted = dgl.utils.is_sorted_srcdst(u_src, u_dst)
   assert src_sorted == False
   assert dst_sorted == False

   src_sorted, dst_sorted = dgl.utils.is_sorted_srcdst(s_src, u_dst)
   assert src_sorted == True
   assert dst_sorted == False

if __name__ == '__main__':
    test_query()
    test_mutation()
    test_scipy_adjmat()
    test_incmat()
    test_find_edges()
    test_hypersparse_query()
    test_is_sorted()
