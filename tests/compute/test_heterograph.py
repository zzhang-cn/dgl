import dgl
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import itertools
import backend as F
import networkx as nx
import unittest, pytest
from dgl import DGLError
import test_utils
from test_utils import parametrize_dtype, get_cases
from scipy.sparse import rand

def create_test_heterograph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
    }, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g

def create_test_heterograph1(idtype):
    edges = []
    edges.extend([(0, 1), (1, 2)])  # follows
    edges.extend([(0, 3), (1, 3), (2, 4), (1, 4)])  # plays
    edges.extend([(0, 4), (2, 3)])  # wishes
    edges.extend([(5, 3), (6, 4)])  # develops
    edges = tuple(zip(*edges))
    ntypes = F.tensor([0, 0, 0, 1, 1, 2, 2])
    etypes = F.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
    g0 = dgl.graph(edges, idtype=idtype, device=F.ctx())
    g0.ndata[dgl.NTYPE] = ntypes
    g0.edata[dgl.ETYPE] = etypes
    return dgl.to_heterogeneous(g0, ['user', 'game', 'developer'],
                                ['follows', 'plays', 'wishes', 'develops'])

def create_test_heterograph2(idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
        }, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g

def create_test_heterograph3(idtype):
    g = dgl.heterograph({
        ('user', 'plays', 'game'): (F.tensor([0, 1, 1, 2], dtype=idtype),
                                    F.tensor([0, 0, 1, 1], dtype=idtype)),
        ('developer', 'develops', 'game'): (F.tensor([0, 1], dtype=idtype),
                                            F.tensor([0, 1], dtype=idtype))},
        idtype=idtype, device=F.ctx())

    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())
    g.nodes['developer'].data['h'] = F.copy_to(F.tensor([3, 3], dtype=idtype), ctx=F.ctx())
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 1, 1, 1], dtype=idtype), ctx=F.ctx())
    return g

def create_test_heterograph4(idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): (F.tensor([0, 1, 1, 2, 2, 2], dtype=idtype),
                                      F.tensor([0, 0, 1, 1, 2, 2], dtype=idtype)),
        ('user', 'plays', 'game'): (F.tensor([0, 1], dtype=idtype),
                                    F.tensor([0, 1], dtype=idtype))},
        idtype=idtype, device=F.ctx())
    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())
    g.edges['follows'].data['h'] = F.copy_to(F.tensor([1, 2, 3, 4, 5, 6], dtype=idtype), ctx=F.ctx())
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    return g

def create_test_heterograph5(idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): (F.tensor([1, 2], dtype=idtype),
                                      F.tensor([0, 1], dtype=idtype)),
        ('user', 'plays', 'game'): (F.tensor([0, 1], dtype=idtype),
                                    F.tensor([0, 1], dtype=idtype))},
        idtype=idtype, device=F.ctx())
    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())
    g.edges['follows'].data['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    return g

def get_redfn(name):
    return getattr(F, name)

@parametrize_dtype
def test_create(idtype):
    device = F.ctx()
    g0 = create_test_heterograph(idtype)
    g1 = create_test_heterograph1(idtype)
    g2 = create_test_heterograph2(idtype)
    assert set(g0.ntypes) == set(g1.ntypes) == set(g2.ntypes)
    assert set(g0.canonical_etypes) == set(g1.canonical_etypes) == set(g2.canonical_etypes)

    # Create a bipartite graph from a SciPy matrix
    src_ids = np.array([2, 3, 4])
    dst_ids = np.array([1, 2, 3])
    eweight = np.array([0.2, 0.3, 0.5])
    sp_mat = ssp.coo_matrix((eweight, (src_ids, dst_ids)))
    g = dgl.bipartite_from_scipy(sp_mat, utype='user', etype='plays',
                                 vtype='game', idtype=idtype, device=device)
    assert g.idtype == idtype
    assert g.device == device
    assert g.num_src_nodes() == 5
    assert g.num_dst_nodes() == 4
    assert g.num_edges() == 3
    src, dst = g.edges()
    assert F.allclose(src, F.tensor([2, 3, 4], dtype=idtype))
    assert F.allclose(dst, F.tensor([1, 2, 3], dtype=idtype))
    g = dgl.bipartite_from_scipy(sp_mat, utype='_U', etype='_E', vtype='_V',
                                 eweight_name='w', idtype=idtype, device=device)
    assert F.allclose(g.edata['w'], F.tensor(eweight))

    # Create a bipartite graph from a NetworkX graph
    nx_g = nx.DiGraph()
    nx_g.add_nodes_from([1, 3], bipartite=0, feat1=np.zeros((2)), feat2=np.ones((2)))
    nx_g.add_nodes_from([2, 4, 5], bipartite=1, feat3=np.zeros((3)))
    nx_g.add_edge(1, 4, weight=np.ones((1)), eid=np.array([1]))
    nx_g.add_edge(3, 5, weight=np.ones((1)), eid=np.array([0]))
    g = dgl.bipartite_from_networkx(nx_g, utype='user', etype='plays',
                                    vtype='game', idtype=idtype, device=device)
    assert g.idtype == idtype
    assert g.device == device
    assert g.num_src_nodes() == 2
    assert g.num_dst_nodes() == 3
    assert g.num_edges() == 2
    src, dst = g.edges()
    assert F.allclose(src, F.tensor([0, 1], dtype=idtype))
    assert F.allclose(dst, F.tensor([1, 2], dtype=idtype))
    g = dgl.bipartite_from_networkx(nx_g, utype='_U', etype='_E', vtype='V',
                                    u_attrs=['feat1', 'feat2'],
                                    e_attrs = ['weight'], v_attrs = ['feat3'])
    assert F.allclose(g.srcdata['feat1'], F.tensor(np.zeros((2, 2))))
    assert F.allclose(g.srcdata['feat2'], F.tensor(np.ones((2, 2))))
    assert F.allclose(g.dstdata['feat3'], F.tensor(np.zeros((3, 3))))
    assert F.allclose(g.edata['weight'], F.tensor(np.ones((2, 1))))
    g = dgl.bipartite_from_networkx(nx_g, utype='_U', etype='_E', vtype='V',
                                    edge_id_attr_name='eid', idtype=idtype, device=device)
    src, dst = g.edges()
    assert F.allclose(src, F.tensor([1, 0], dtype=idtype))
    assert F.allclose(dst, F.tensor([2, 1], dtype=idtype))

    # create from scipy
    spmat = ssp.coo_matrix(([1,1,1], ([0, 0, 1], [2, 3, 2])), shape=(4, 4))
    g = dgl.from_scipy(spmat, idtype=idtype, device=device)
    assert g.num_nodes() == 4
    assert g.num_edges() == 3
    assert g.idtype == idtype
    assert g.device == device

    # test inferring number of nodes for heterograph
    g = dgl.heterograph({
        ('l0', 'e0', 'l1'): ([0, 0], [1, 2]),
        ('l0', 'e1', 'l2'): ([2], [2]),
        ('l2', 'e2', 'l2'): ([1, 3], [1, 3])
        }, idtype=idtype, device=device)
    assert g.num_nodes('l0') == 3
    assert g.num_nodes('l1') == 3
    assert g.num_nodes('l2') == 4
    assert g.idtype == idtype
    assert g.device == device

    # test if validate flag works
    # homo graph
    with pytest.raises(DGLError):
        g = dgl.graph(
            ([0, 0, 0, 1, 1, 2], [0, 1, 2, 0, 1, 2]),
            num_nodes=2,
            idtype=idtype, device=device
        )
    # bipartite graph
    def _test_validate_bipartite(card):
        with pytest.raises(DGLError):
            g = dgl.heterograph({
                ('_U', '_E', '_V'): ([0, 0, 1, 1, 2], [1, 1, 2, 2, 3])
            }, {'_U': card[0], '_V': card[1]}, idtype=idtype, device=device)

    _test_validate_bipartite((3, 3))
    _test_validate_bipartite((2, 4))

    # test from_scipy
    num_nodes = 10
    density = 0.25
    for fmt in ['csr', 'coo', 'csc']:
        adj = rand(num_nodes, num_nodes, density=density, format=fmt)
        g = dgl.from_scipy(adj, eweight_name='w', idtype=idtype)
        assert g.idtype == idtype
        assert g.device == F.cpu()
        assert F.array_equal(g.edata['w'], F.copy_to(F.tensor(adj.data), F.cpu()))

@parametrize_dtype
def test_query(idtype):
    g = create_test_heterograph(idtype)

    ntypes = ['user', 'game', 'developer']
    canonical_etypes = [
        ('user', 'follows', 'user'),
        ('user', 'plays', 'game'),
        ('user', 'wishes', 'game'),
        ('developer', 'develops', 'game')]
    etypes = ['follows', 'plays', 'wishes', 'develops']

    # node & edge types
    assert set(ntypes) == set(g.ntypes)
    assert set(etypes) == set(g.etypes)
    assert set(canonical_etypes) == set(g.canonical_etypes)

    # metagraph
    mg = g.metagraph()
    assert set(g.ntypes) == set(mg.nodes)
    etype_triplets = [(u, v, e) for u, v, e in mg.edges(keys=True)]
    assert set([
        ('user', 'user', 'follows'),
        ('user', 'game', 'plays'),
        ('user', 'game', 'wishes'),
        ('developer', 'game', 'develops')]) == set(etype_triplets)
    for i in range(len(etypes)):
        assert g.to_canonical_etype(etypes[i]) == canonical_etypes[i]

    def _test(g):
        # number of nodes
        assert [g.num_nodes(ntype) for ntype in ntypes] == [3, 2, 2]

        # number of edges
        assert [g.num_edges(etype) for etype in etypes] == [2, 4, 2, 2]

        # has_node & has_nodes
        for ntype in ntypes:
            n = g.number_of_nodes(ntype)
            for i in range(n):
                assert g.has_node(i, ntype)
            assert not g.has_node(n, ntype)
            assert np.array_equal(
                F.asnumpy(g.has_nodes([0, n], ntype)).astype('int32'), [1, 0])

        assert not g.is_multigraph

        for etype in etypes:
            srcs, dsts = edges[etype]
            for src, dst in zip(srcs, dsts):
                assert g.has_edges_between(src, dst, etype)
            assert F.asnumpy(g.has_edges_between(srcs, dsts, etype)).all()

            srcs, dsts = negative_edges[etype]
            for src, dst in zip(srcs, dsts):
                assert not g.has_edges_between(src, dst, etype)
            assert not F.asnumpy(g.has_edges_between(srcs, dsts, etype)).any()

            srcs, dsts = edges[etype]
            n_edges = len(srcs)

            # predecessors & in_edges & in_degree
            pred = [s for s, d in zip(srcs, dsts) if d == 0]
            assert set(F.asnumpy(g.predecessors(0, etype)).tolist()) == set(pred)
            u, v = g.in_edges([0], etype=etype)
            assert F.asnumpy(v).tolist() == [0] * len(pred)
            assert set(F.asnumpy(u).tolist()) == set(pred)
            assert g.in_degrees(0, etype) == len(pred)

            # successors & out_edges & out_degree
            succ = [d for s, d in zip(srcs, dsts) if s == 0]
            assert set(F.asnumpy(g.successors(0, etype)).tolist()) == set(succ)
            u, v = g.out_edges([0], etype=etype)
            assert F.asnumpy(u).tolist() == [0] * len(succ)
            assert set(F.asnumpy(v).tolist()) == set(succ)
            assert g.out_degrees(0, etype) == len(succ)

            # edge_id & edge_ids
            for i, (src, dst) in enumerate(zip(srcs, dsts)):
                assert g.edge_ids(src, dst, etype=etype) == i
                _, _, eid = g.edge_ids(src, dst, etype=etype, return_uv=True)
                assert eid == i
            assert F.asnumpy(g.edge_ids(srcs, dsts, etype=etype)).tolist() == list(range(n_edges))
            u, v, e = g.edge_ids(srcs, dsts, etype=etype, return_uv=True)
            u, v, e = F.asnumpy(u), F.asnumpy(v), F.asnumpy(e)
            assert u[e].tolist() == srcs
            assert v[e].tolist() == dsts

            # find_edges
            for eid in [list(range(n_edges)), np.arange(n_edges), F.astype(F.arange(0, n_edges), g.idtype)]:
                u, v = g.find_edges(eid, etype)
                assert F.asnumpy(u).tolist() == srcs
                assert F.asnumpy(v).tolist() == dsts

            # all_edges.
            for order in ['eid']:
                u, v, e = g.edges('all', order, etype)
                assert F.asnumpy(u).tolist() == srcs
                assert F.asnumpy(v).tolist() == dsts
                assert F.asnumpy(e).tolist() == list(range(n_edges))

            # in_degrees & out_degrees
            in_degrees = F.asnumpy(g.in_degrees(etype=etype))
            out_degrees = F.asnumpy(g.out_degrees(etype=etype))
            src_count = Counter(srcs)
            dst_count = Counter(dsts)
            utype, _, vtype = g.to_canonical_etype(etype)
            for i in range(g.number_of_nodes(utype)):
                assert out_degrees[i] == src_count[i]
            for i in range(g.number_of_nodes(vtype)):
                assert in_degrees[i] == dst_count[i]

    edges = {
        'follows': ([0, 1], [1, 2]),
        'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
        'wishes': ([0, 2], [1, 0]),
        'develops': ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        'follows': ([0, 1], [0, 1]),
        'plays': ([0, 2], [1, 0]),
        'wishes': ([0, 1], [0, 1]),
        'develops': ([0, 1], [1, 0]),
    }
    g = create_test_heterograph(idtype)
    _test(g)
    g = create_test_heterograph1(idtype)
    _test(g)
    if F._default_context_str != 'gpu':
        # XXX: CUDA COO operators have not been live yet.
        g = create_test_heterograph2(idtype)
        _test(g)

    etypes = canonical_etypes
    edges = {
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        ('user', 'follows', 'user'): ([0, 1], [0, 1]),
        ('user', 'plays', 'game'): ([0, 2], [1, 0]),
        ('user', 'wishes', 'game'): ([0, 1], [0, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [1, 0]),
        }
    g = create_test_heterograph(idtype)
    _test(g)
    g = create_test_heterograph1(idtype)
    _test(g)
    if F._default_context_str != 'gpu':
        # XXX: CUDA COO operators have not been live yet.
        g = create_test_heterograph2(idtype)
        _test(g)

    # test repr
    print(g)

@parametrize_dtype
def test_empty_query(idtype):
    g = dgl.graph(([1, 2, 3], [0, 4, 5]), idtype=idtype, device=F.ctx())
    g.add_nodes(0)
    g.add_edges([], [])
    g.remove_edges([])
    g.remove_nodes([])
    assert F.shape(g.has_nodes([])) == (0,)
    assert F.shape(g.has_edges_between([], [])) == (0,)
    g.edge_ids([], [])
    g.edge_ids([], [], return_uv=True)
    g.find_edges([])

    assert F.shape(g.in_edges([], form='eid')) == (0,)
    u, v = g.in_edges([], form='uv')
    assert F.shape(u) == (0,)
    assert F.shape(v) == (0,)
    u, v, e = g.in_edges([], form='all')
    assert F.shape(u) == (0,)
    assert F.shape(v) == (0,)
    assert F.shape(e) == (0,)

    assert F.shape(g.out_edges([], form='eid')) == (0,)
    u, v = g.out_edges([], form='uv')
    assert F.shape(u) == (0,)
    assert F.shape(v) == (0,)
    u, v, e = g.out_edges([], form='all')
    assert F.shape(u) == (0,)
    assert F.shape(v) == (0,)
    assert F.shape(e) == (0,)

    assert F.shape(g.in_degrees([])) == (0,)
    assert F.shape(g.out_degrees([])) == (0,)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU does not have COO impl.")
def _test_hypersparse():
    N1 = 1 << 50        # should crash if allocated a CSR
    N2 = 1 << 48

    g = dgl.heterograph({
        ('user', 'follows', 'user'): (F.tensor([0], F.int64), F.tensor([1], F.int64)),
        ('user', 'plays', 'game'): (F.tensor([0], F.int64), F.tensor([N2], F.int64))},
        {'user': N1, 'game': N1},
        device=F.ctx())
    assert g.number_of_nodes('user') == N1
    assert g.number_of_nodes('game') == N1
    assert g.number_of_edges('follows') == 1
    assert g.number_of_edges('plays') == 1

    assert g.has_edges_between(0, 1, 'follows')
    assert not g.has_edges_between(0, 0, 'follows')
    mask = F.asnumpy(g.has_edges_between([0, 0], [0, 1], 'follows')).tolist()
    assert mask == [0, 1]

    assert g.has_edges_between(0, N2, 'plays')
    assert not g.has_edges_between(0, 0, 'plays')
    mask = F.asnumpy(g.has_edges_between([0, 0], [0, N2], 'plays')).tolist()
    assert mask == [0, 1]

    assert F.asnumpy(g.predecessors(0, 'follows')).tolist() == []
    assert F.asnumpy(g.successors(0, 'follows')).tolist() == [1]
    assert F.asnumpy(g.predecessors(1, 'follows')).tolist() == [0]
    assert F.asnumpy(g.successors(1, 'follows')).tolist() == []

    assert F.asnumpy(g.predecessors(0, 'plays')).tolist() == []
    assert F.asnumpy(g.successors(0, 'plays')).tolist() == [N2]
    assert F.asnumpy(g.predecessors(N2, 'plays')).tolist() == [0]
    assert F.asnumpy(g.successors(N2, 'plays')).tolist() == []

    assert g.edge_ids(0, 1, etype='follows') == 0
    assert g.edge_ids(0, N2, etype='plays') == 0

    u, v = g.find_edges([0], 'follows')
    assert F.asnumpy(u).tolist() == [0]
    assert F.asnumpy(v).tolist() == [1]
    u, v = g.find_edges([0], 'plays')
    assert F.asnumpy(u).tolist() == [0]
    assert F.asnumpy(v).tolist() == [N2]
    u, v, e = g.all_edges('all', 'eid', 'follows')
    assert F.asnumpy(u).tolist() == [0]
    assert F.asnumpy(v).tolist() == [1]
    assert F.asnumpy(e).tolist() == [0]
    u, v, e = g.all_edges('all', 'eid', 'plays')
    assert F.asnumpy(u).tolist() == [0]
    assert F.asnumpy(v).tolist() == [N2]
    assert F.asnumpy(e).tolist() == [0]

    assert g.in_degrees(0, 'follows') == 0
    assert g.in_degrees(1, 'follows') == 1
    assert F.asnumpy(g.in_degrees([0, 1], 'follows')).tolist() == [0, 1]
    assert g.in_degrees(0, 'plays') == 0
    assert g.in_degrees(N2, 'plays') == 1
    assert F.asnumpy(g.in_degrees([0, N2], 'plays')).tolist() == [0, 1]
    assert g.out_degrees(0, 'follows') == 1
    assert g.out_degrees(1, 'follows') == 0
    assert F.asnumpy(g.out_degrees([0, 1], 'follows')).tolist() == [1, 0]
    assert g.out_degrees(0, 'plays') == 1
    assert g.out_degrees(N2, 'plays') == 0
    assert F.asnumpy(g.out_degrees([0, N2], 'plays')).tolist() == [1, 0]

def _test_edge_ids():
    N1 = 1 << 50        # should crash if allocated a CSR
    N2 = 1 << 48

    g = dgl.heterograph({
        ('user', 'follows', 'user'): (F.tensor([0], F.int64), F.tensor([1], F.int64)),
        ('user', 'plays', 'game'): (F.tensor([0], F.int64), F.tensor([N2], F.int64))},
        {'user': N1, 'game': N1})
    with pytest.raises(DGLError):
        eid = g.edge_ids(0, 0, etype='follows')

    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): (F.tensor([0, 0], F.int64), F.tensor([1, 1], F.int64)),
        ('user', 'plays', 'game'): (F.tensor([0], F.int64), F.tensor([N2], F.int64))},
        {'user': N1, 'game': N1}, device=F.cpu())

    eid = g2.edge_ids(0, 1, etype='follows')
    assert eid == 0

@parametrize_dtype
def test_adj(idtype):
    g = create_test_heterograph(idtype)
    adj = F.sparse_to_numpy(g.adj(transpose=False, etype='follows'))
    assert np.allclose(
            adj,
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = F.sparse_to_numpy(g.adj(transpose=True, etype='follows'))
    assert np.allclose(
            adj,
            np.array([[0., 1., 0.],
                      [0., 0., 1.],
                      [0., 0., 0.]]))
    adj = F.sparse_to_numpy(g.adj(transpose=False, etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = F.sparse_to_numpy(g.adj(transpose=True, etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 0.],
                      [1., 1.],
                      [0., 1.]]))

    adj = g.adj(transpose=False, scipy_fmt='csr', etype='follows')
    assert np.allclose(
            adj.todense(),
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = g.adj(transpose=False, scipy_fmt='coo', etype='follows')
    assert np.allclose(
            adj.todense(),
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = g.adj(transpose=False, scipy_fmt='csr', etype='plays')
    assert np.allclose(
            adj.todense(),
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = g.adj(transpose=False, scipy_fmt='coo', etype='plays')
    assert np.allclose(
            adj.todense(),
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = F.sparse_to_numpy(g['follows'].adj(transpose=False))
    assert np.allclose(
            adj,
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))

@parametrize_dtype
def test_inc(idtype):
    g = create_test_heterograph(idtype)
    adj = F.sparse_to_numpy(g['follows'].inc('in'))
    assert np.allclose(
            adj,
            np.array([[0., 0.],
                      [1., 0.],
                      [0., 1.]]))
    adj = F.sparse_to_numpy(g['follows'].inc('out'))
    assert np.allclose(
            adj,
            np.array([[1., 0.],
                      [0., 1.],
                      [0., 0.]]))
    adj = F.sparse_to_numpy(g['follows'].inc('both'))
    assert np.allclose(
            adj,
            np.array([[-1., 0.],
                      [1., -1.],
                      [0., 1.]]))
    adj = F.sparse_to_numpy(g.inc('in', etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 1., 0., 0.],
                      [0., 0., 1., 1.]]))
    adj = F.sparse_to_numpy(g.inc('out', etype='plays'))
    assert np.allclose(
            adj,
            np.array([[1., 0., 0., 0.],
                      [0., 1., 0., 1.],
                      [0., 0., 1., 0.]]))
    adj = F.sparse_to_numpy(g.inc('both', etype='follows'))
    assert np.allclose(
            adj,
            np.array([[-1., 0.],
                      [1., -1.],
                      [0., 1.]]))

@parametrize_dtype
def test_view(idtype):
    # test single node type
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2])
    }, idtype=idtype, device=F.ctx())
    f1 = F.randn((3, 6))
    g.ndata['h'] = f1
    f2 = g.nodes['user'].data['h']
    assert F.array_equal(f1, f2)
    fail = False
    try:
        g.ndata['h'] = {'user' : f1}
    except Exception:
        fail = True
    assert fail

    # test single edge type
    f3 = F.randn((2, 4))
    g.edata['h'] = f3
    f4 = g.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    fail = False
    try:
        g.edata['h'] = {'follows' : f3}
    except Exception:
        fail = True
    assert fail

    # test data view
    g = create_test_heterograph(idtype)

    f1 = F.randn((3, 6))
    g.nodes['user'].data['h'] = f1       # ok
    f2 = g.nodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(g.nodes('user'), F.arange(0, 3, idtype))
    g.nodes['user'].data.pop('h')

    # multi type ndata
    f1 = F.randn((3, 6))
    f2 = F.randn((2, 6))
    fail = False
    try:
        g.ndata['h'] = f1
    except Exception:
        fail = True
    assert fail

    f3 = F.randn((2, 4))
    g.edges['user', 'follows', 'user'].data['h'] = f3
    f4 = g.edges['user', 'follows', 'user'].data['h']
    f5 = g.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    assert F.array_equal(f3, f5)
    assert F.array_equal(g.edges(etype='follows', form='eid'), F.arange(0, 2, idtype))
    g.edges['follows'].data.pop('h')

    f3 = F.randn((2, 4))
    fail = False
    try:
        g.edata['h'] = f3
    except Exception:
        fail = True
    assert fail

    # test srcdata
    f1 = F.randn((3, 6))
    g.srcnodes['user'].data['h'] = f1       # ok
    f2 = g.srcnodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(g.srcnodes('user'), F.arange(0, 3, idtype))
    g.srcnodes['user'].data.pop('h')

    # test dstdata
    f1 = F.randn((3, 6))
    g.dstnodes['user'].data['h'] = f1       # ok
    f2 = g.dstnodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(g.dstnodes('user'), F.arange(0, 3, idtype))
    g.dstnodes['user'].data.pop('h')

@parametrize_dtype
def test_view1(idtype):
    # test relation view
    HG = create_test_heterograph(idtype)
    ntypes = ['user', 'game', 'developer']
    canonical_etypes = [
        ('user', 'follows', 'user'),
        ('user', 'plays', 'game'),
        ('user', 'wishes', 'game'),
        ('developer', 'develops', 'game')]
    etypes = ['follows', 'plays', 'wishes', 'develops']

    def _test_query():
        for etype in etypes:
            utype, _, vtype = HG.to_canonical_etype(etype)
            g = HG[etype]
            srcs, dsts = edges[etype]
            for src, dst in zip(srcs, dsts):
                assert g.has_edges_between(src, dst)
            assert F.asnumpy(g.has_edges_between(srcs, dsts)).all()

            srcs, dsts = negative_edges[etype]
            for src, dst in zip(srcs, dsts):
                assert not g.has_edges_between(src, dst)
            assert not F.asnumpy(g.has_edges_between(srcs, dsts)).any()

            srcs, dsts = edges[etype]
            n_edges = len(srcs)

            # predecessors & in_edges & in_degree
            pred = [s for s, d in zip(srcs, dsts) if d == 0]
            assert set(F.asnumpy(g.predecessors(0)).tolist()) == set(pred)
            u, v = g.in_edges([0])
            assert F.asnumpy(v).tolist() == [0] * len(pred)
            assert set(F.asnumpy(u).tolist()) == set(pred)
            assert g.in_degrees(0) == len(pred)

            # successors & out_edges & out_degree
            succ = [d for s, d in zip(srcs, dsts) if s == 0]
            assert set(F.asnumpy(g.successors(0)).tolist()) == set(succ)
            u, v = g.out_edges([0])
            assert F.asnumpy(u).tolist() == [0] * len(succ)
            assert set(F.asnumpy(v).tolist()) == set(succ)
            assert g.out_degrees(0) == len(succ)

            # edge_id & edge_ids
            for i, (src, dst) in enumerate(zip(srcs, dsts)):
                assert g.edge_ids(src, dst, etype=etype) == i
                _, _, eid = g.edge_ids(src, dst, etype=etype, return_uv=True)
                assert eid == i
            assert F.asnumpy(g.edge_ids(srcs, dsts)).tolist() == list(range(n_edges))
            u, v, e = g.edge_ids(srcs, dsts, return_uv=True)
            u, v, e = F.asnumpy(u), F.asnumpy(v), F.asnumpy(e)
            assert u[e].tolist() == srcs
            assert v[e].tolist() == dsts

            # find_edges
            u, v = g.find_edges(list(range(n_edges)))
            assert F.asnumpy(u).tolist() == srcs
            assert F.asnumpy(v).tolist() == dsts

            # all_edges.
            for order in ['eid']:
                u, v, e = g.all_edges(form='all', order=order)
                assert F.asnumpy(u).tolist() == srcs
                assert F.asnumpy(v).tolist() == dsts
                assert F.asnumpy(e).tolist() == list(range(n_edges))

            # in_degrees & out_degrees
            in_degrees = F.asnumpy(g.in_degrees())
            out_degrees = F.asnumpy(g.out_degrees())
            src_count = Counter(srcs)
            dst_count = Counter(dsts)
            for i in range(g.number_of_nodes(utype)):
                assert out_degrees[i] == src_count[i]
            for i in range(g.number_of_nodes(vtype)):
                assert in_degrees[i] == dst_count[i]

    edges = {
        'follows': ([0, 1], [1, 2]),
        'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
        'wishes': ([0, 2], [1, 0]),
        'develops': ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        'follows': ([0, 1], [0, 1]),
        'plays': ([0, 2], [1, 0]),
        'wishes': ([0, 1], [0, 1]),
        'develops': ([0, 1], [1, 0]),
    }
    _test_query()
    etypes = canonical_etypes
    edges = {
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 2], [1, 0]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1]),
    }
    # edges that does not exist in the graph
    negative_edges = {
        ('user', 'follows', 'user'): ([0, 1], [0, 1]),
        ('user', 'plays', 'game'): ([0, 2], [1, 0]),
        ('user', 'wishes', 'game'): ([0, 1], [0, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [1, 0]),
        }
    _test_query()

    # test features
    HG.nodes['user'].data['h'] = F.ones((HG.number_of_nodes('user'), 5))
    HG.nodes['game'].data['m'] = F.ones((HG.number_of_nodes('game'), 3)) * 2

    # test only one node type
    g = HG['follows']
    assert g.number_of_nodes() == 3

    # test ndata and edata
    f1 = F.randn((3, 6))
    g.ndata['h'] = f1       # ok
    f2 = HG.nodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(g.nodes(), F.arange(0, 3, g.idtype))

    f3 = F.randn((2, 4))
    g.edata['h'] = f3
    f4 = HG.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    assert F.array_equal(g.edges(form='eid'), F.arange(0, 2, g.idtype))

@parametrize_dtype
def test_flatten(idtype):
    def check_mapping(g, fg):
        if len(fg.ntypes) == 1:
            SRC = DST = fg.ntypes[0]
        else:
            SRC = fg.ntypes[0]
            DST = fg.ntypes[1]

        etypes = F.asnumpy(fg.edata[dgl.ETYPE]).tolist()
        eids = F.asnumpy(fg.edata[dgl.EID]).tolist()

        for i, (etype, eid) in enumerate(zip(etypes, eids)):
            src_g, dst_g = g.find_edges([eid], g.canonical_etypes[etype])
            src_fg, dst_fg = fg.find_edges([i])
            # TODO(gq): I feel this code is quite redundant; can we just add new members (like
            # "induced_srcid") to returned heterograph object and not store them as features?
            assert F.asnumpy(src_g) == F.asnumpy(F.gather_row(fg.nodes[SRC].data[dgl.NID], src_fg)[0])
            tid = F.asnumpy(F.gather_row(fg.nodes[SRC].data[dgl.NTYPE], src_fg)).item()
            assert g.canonical_etypes[etype][0] == g.ntypes[tid]
            assert F.asnumpy(dst_g) == F.asnumpy(F.gather_row(fg.nodes[DST].data[dgl.NID], dst_fg)[0])
            tid = F.asnumpy(F.gather_row(fg.nodes[DST].data[dgl.NTYPE], dst_fg)).item()
            assert g.canonical_etypes[etype][2] == g.ntypes[tid]

    # check for wildcard slices
    g = create_test_heterograph(idtype)
    g.nodes['user'].data['h'] = F.ones((3, 5))
    g.nodes['game'].data['i'] = F.ones((2, 5))
    g.edges['plays'].data['e'] = F.ones((4, 4))
    g.edges['wishes'].data['e'] = F.ones((2, 4))
    g.edges['wishes'].data['f'] = F.ones((2, 4))

    fg = g['user', :, 'game']   # user--plays->game and user--wishes->game
    assert len(fg.ntypes) == 2
    assert fg.ntypes == ['user', 'game']
    assert fg.etypes == ['plays+wishes']
    assert fg.idtype == g.idtype
    assert fg.device == g.device
    etype = fg.etypes[0]
    assert fg[etype] is not None        # Issue #2166

    assert F.array_equal(fg.nodes['user'].data['h'], F.ones((3, 5)))
    assert F.array_equal(fg.nodes['game'].data['i'], F.ones((2, 5)))
    assert F.array_equal(fg.edata['e'], F.ones((6, 4)))
    assert 'f' not in fg.edata

    etypes = F.asnumpy(fg.edata[dgl.ETYPE]).tolist()
    eids = F.asnumpy(fg.edata[dgl.EID]).tolist()
    assert set(zip(etypes, eids)) == set([(3, 0), (3, 1), (2, 1), (2, 0), (2, 3), (2, 2)])

    check_mapping(g, fg)

    fg = g['user', :, 'user']
    assert fg.idtype == g.idtype
    assert fg.device == g.device
    # NOTE(gq): The node/edge types from the parent graph is returned if there is only one
    # node/edge type.  This differs from the behavior above.
    assert fg.ntypes == ['user']
    assert fg.etypes == ['follows']
    u1, v1 = g.edges(etype='follows', order='eid')
    u2, v2 = fg.edges(etype='follows', order='eid')
    assert F.array_equal(u1, u2)
    assert F.array_equal(v1, v2)

    fg = g['developer', :, 'game']
    assert fg.idtype == g.idtype
    assert fg.device == g.device
    assert fg.ntypes == ['developer', 'game']
    assert fg.etypes == ['develops']
    u1, v1 = g.edges(etype='develops', order='eid')
    u2, v2 = fg.edges(etype='develops', order='eid')
    assert F.array_equal(u1, u2)
    assert F.array_equal(v1, v2)

    fg = g[:, :, :]
    assert fg.idtype == g.idtype
    assert fg.device == g.device
    assert fg.ntypes == ['developer+user', 'game+user']
    assert fg.etypes == ['develops+follows+plays+wishes']
    check_mapping(g, fg)

    # Test another heterograph
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1, 2], [1, 2, 3]),
        ('user', 'knows', 'user'): ([0, 2], [2, 3])
    }, idtype=idtype, device=F.ctx())
    g.nodes['user'].data['h'] = F.randn((4, 3))
    g.edges['follows'].data['w'] = F.randn((3, 2))
    g.nodes['user'].data['hh'] = F.randn((4, 5))
    g.edges['knows'].data['ww'] = F.randn((2, 10))

    fg = g['user', :, 'user']
    assert fg.idtype == g.idtype
    assert fg.device == g.device
    assert fg.ntypes == ['user']
    assert fg.etypes == ['follows+knows']
    check_mapping(g, fg)

    fg = g['user', :, :]
    assert fg.idtype == g.idtype
    assert fg.device == g.device
    assert fg.ntypes == ['user']
    assert fg.etypes == ['follows+knows']
    check_mapping(g, fg)

@unittest.skipIf(F._default_context_str == 'cpu', reason="Need gpu for this test")
@parametrize_dtype
def test_to_device(idtype):
    # TODO: rewrite this test case to accept different graphs so we
    #  can test reverse graph and batched graph
    g = create_test_heterograph(idtype)
    g.nodes['user'].data['h'] = F.ones((3, 5))
    g.nodes['game'].data['i'] = F.ones((2, 5))
    g.edges['plays'].data['e'] = F.ones((4, 4))
    assert g.device == F.ctx()
    g = g.to(F.cpu())
    assert g.device == F.cpu()
    assert F.context(g.nodes['user'].data['h']) == F.cpu()
    assert F.context(g.nodes['game'].data['i']) == F.cpu()
    assert F.context(g.edges['plays'].data['e']) == F.cpu()
    for ntype in g.ntypes:
        assert F.context(g.batch_num_nodes(ntype)) == F.cpu()
    for etype in g.canonical_etypes:
        assert F.context(g.batch_num_edges(etype)) == F.cpu()

    if F.is_cuda_available():
        g1 = g.to(F.cuda())
        assert g1.device == F.cuda()
        assert F.context(g1.nodes['user'].data['h']) == F.cuda()
        assert F.context(g1.nodes['game'].data['i']) == F.cuda()
        assert F.context(g1.edges['plays'].data['e']) == F.cuda()
        for ntype in g1.ntypes:
            assert F.context(g1.batch_num_nodes(ntype)) == F.cuda()
        for etype in g1.canonical_etypes:
            assert F.context(g1.batch_num_edges(etype)) == F.cuda()
        assert F.context(g.nodes['user'].data['h']) == F.cpu()
        assert F.context(g.nodes['game'].data['i']) == F.cpu()
        assert F.context(g.edges['plays'].data['e']) == F.cpu()
        for ntype in g.ntypes:
            assert F.context(g.batch_num_nodes(ntype)) == F.cpu()
        for etype in g.canonical_etypes:
            assert F.context(g.batch_num_edges(etype)) == F.cpu()
        with pytest.raises(DGLError):
            g1.nodes['user'].data['h'] = F.copy_to(F.ones((3, 5)), F.cpu())
        with pytest.raises(DGLError):
            g1.edges['plays'].data['e'] = F.copy_to(F.ones((4, 4)), F.cpu())

@unittest.skipIf(F._default_context_str == 'cpu', reason="Need gpu for this test")
@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['block']))
def test_to_device2(g, idtype):
    g = g.astype(idtype)
    g = g.to(F.cpu())
    assert g.device == F.cpu()
    if F.is_cuda_available():
        g1 = g.to(F.cuda())
        assert g1.device == F.cuda()
        assert g1.ntypes == g.ntypes
        assert g1.etypes == g.etypes
        assert g1.canonical_etypes == g.canonical_etypes

@parametrize_dtype
def test_convert_bound(idtype):
    def _test_bipartite_bound(data, card):
        with pytest.raises(DGLError):
            dgl.heterograph({
                ('_U', '_E', '_V'): data
            }, {'_U': card[0], '_V': card[1]}, idtype=idtype, device=F.ctx())

    def _test_graph_bound(data, card):
        with pytest.raises(DGLError):
            dgl.graph(data, num_nodes=card, idtype=idtype, device=F.ctx())

    _test_bipartite_bound(([1, 2], [1, 2]), (2, 3))
    _test_bipartite_bound(([0, 1], [1, 4]), (2, 3))
    _test_graph_bound(([1, 3], [1, 2]), 3)
    _test_graph_bound(([0, 1], [1, 3]), 3)


@parametrize_dtype
def test_convert(idtype):
    hg = create_test_heterograph(idtype)
    hs = []
    for ntype in hg.ntypes:
        h = F.randn((hg.number_of_nodes(ntype), 5))
        hg.nodes[ntype].data['h'] = h
        hs.append(h)
    hg.nodes['user'].data['x'] = F.randn((3, 3))
    ws = []
    for etype in hg.canonical_etypes:
        w = F.randn((hg.number_of_edges(etype), 5))
        hg.edges[etype].data['w'] = w
        ws.append(w)
    hg.edges['plays'].data['x'] = F.randn((4, 3))

    g = dgl.to_homogeneous(hg, ndata=['h'], edata=['w'])
    assert g.idtype == idtype
    assert g.device == hg.device
    assert F.array_equal(F.cat(hs, dim=0), g.ndata['h'])
    assert 'x' not in g.ndata
    assert F.array_equal(F.cat(ws, dim=0), g.edata['w'])
    assert 'x' not in g.edata

    src, dst = g.all_edges(order='eid')
    src = F.asnumpy(src)
    dst = F.asnumpy(dst)
    etype_id, eid = F.asnumpy(g.edata[dgl.ETYPE]), F.asnumpy(g.edata[dgl.EID])
    ntype_id, nid = F.asnumpy(g.ndata[dgl.NTYPE]), F.asnumpy(g.ndata[dgl.NID])
    for i in range(g.number_of_edges()):
        srctype = hg.ntypes[ntype_id[src[i]]]
        dsttype = hg.ntypes[ntype_id[dst[i]]]
        etype = hg.etypes[etype_id[i]]
        src_i, dst_i = hg.find_edges([eid[i]], (srctype, etype, dsttype))
        assert np.asscalar(F.asnumpy(src_i)) == nid[src[i]]
        assert np.asscalar(F.asnumpy(dst_i)) == nid[dst[i]]

    mg = nx.MultiDiGraph([
        ('user', 'user', 'follows'),
        ('user', 'game', 'plays'),
        ('user', 'game', 'wishes'),
        ('developer', 'game', 'develops')])

    for _mg in [None, mg]:
        hg2 = dgl.to_heterogeneous(
                g, hg.ntypes, hg.etypes,
                ntype_field=dgl.NTYPE, etype_field=dgl.ETYPE, metagraph=_mg)
        assert hg2.idtype == hg.idtype
        assert hg2.device == hg.device
        assert set(hg.ntypes) == set(hg2.ntypes)
        assert set(hg.canonical_etypes) == set(hg2.canonical_etypes)
        for ntype in hg.ntypes:
            assert hg.number_of_nodes(ntype) == hg2.number_of_nodes(ntype)
            assert F.array_equal(hg.nodes[ntype].data['h'], hg2.nodes[ntype].data['h'])
        for canonical_etype in hg.canonical_etypes:
            src, dst = hg.all_edges(etype=canonical_etype, order='eid')
            src2, dst2 = hg2.all_edges(etype=canonical_etype, order='eid')
            assert F.array_equal(src, src2)
            assert F.array_equal(dst, dst2)
            assert F.array_equal(hg.edges[canonical_etype].data['w'], hg2.edges[canonical_etype].data['w'])

    # hetero_from_homo test case 2
    g = dgl.graph(([0, 1, 2, 0], [2, 2, 3, 3]), idtype=idtype, device=F.ctx())
    g.ndata[dgl.NTYPE] = F.tensor([0, 0, 1, 2])
    g.edata[dgl.ETYPE] = F.tensor([0, 0, 1, 2])
    hg = dgl.to_heterogeneous(g, ['l0', 'l1', 'l2'], ['e0', 'e1', 'e2'])
    assert hg.idtype == idtype
    assert hg.device == g.device
    assert set(hg.canonical_etypes) == set(
        [('l0', 'e0', 'l1'), ('l1', 'e1', 'l2'), ('l0', 'e2', 'l2')])
    assert hg.number_of_nodes('l0') == 2
    assert hg.number_of_nodes('l1') == 1
    assert hg.number_of_nodes('l2') == 1
    assert hg.number_of_edges('e0') == 2
    assert hg.number_of_edges('e1') == 1
    assert hg.number_of_edges('e2') == 1
    assert F.array_equal(hg.ndata[dgl.NID]['l0'], F.tensor([0, 1], F.int64))
    assert F.array_equal(hg.ndata[dgl.NID]['l1'], F.tensor([2], F.int64))
    assert F.array_equal(hg.ndata[dgl.NID]['l2'], F.tensor([3], F.int64))
    assert F.array_equal(hg.edata[dgl.EID][('l0', 'e0', 'l1')], F.tensor([0, 1], F.int64))
    assert F.array_equal(hg.edata[dgl.EID][('l0', 'e2', 'l2')], F.tensor([3], F.int64))
    assert F.array_equal(hg.edata[dgl.EID][('l1', 'e1', 'l2')], F.tensor([2], F.int64))

    # hetero_from_homo test case 3
    mg = nx.MultiDiGraph([
        ('user', 'movie', 'watches'),
        ('user', 'TV', 'watches')])
    g = dgl.graph(((0, 0), (1, 2)), idtype=idtype, device=F.ctx())
    g.ndata[dgl.NTYPE] = F.tensor([0, 1, 2])
    g.edata[dgl.ETYPE] = F.tensor([0, 0])
    for _mg in [None, mg]:
        hg = dgl.to_heterogeneous(g, ['user', 'TV', 'movie'], ['watches'], metagraph=_mg)
        assert hg.idtype == g.idtype
        assert hg.device == g.device
        assert set(hg.canonical_etypes) == set(
            [('user', 'watches', 'movie'), ('user', 'watches', 'TV')])
        assert hg.number_of_nodes('user') == 1
        assert hg.number_of_nodes('TV') == 1
        assert hg.number_of_nodes('movie') == 1
        assert hg.number_of_edges(('user', 'watches', 'TV')) == 1
        assert hg.number_of_edges(('user', 'watches', 'movie')) == 1
        assert len(hg.etypes) == 2

    # hetero_to_homo test case 2
    hg = dgl.heterograph({
        ('_U', '_E', '_V'): ([0, 1], [0, 1])
    }, {'_U': 2, '_V': 3}, idtype=idtype, device=F.ctx())
    g = dgl.to_homogeneous(hg)
    assert hg.idtype == g.idtype
    assert hg.device == g.device
    assert g.number_of_nodes() == 5

@parametrize_dtype
def test_metagraph_reachable(idtype):
    g = create_test_heterograph(idtype)
    x = F.randn((3, 5))
    g.nodes['user'].data['h'] = x

    new_g = dgl.metapath_reachable_graph(g, ['follows', 'plays'])
    assert new_g.idtype == idtype
    assert new_g.ntypes == ['game', 'user']
    assert new_g.number_of_edges() == 3
    assert F.asnumpy(new_g.has_edges_between([0, 0, 1], [0, 1, 1])).all()

    new_g = dgl.metapath_reachable_graph(g, ['follows'])
    assert new_g.idtype == idtype
    assert new_g.ntypes == ['user']
    assert new_g.number_of_edges() == 2
    assert F.asnumpy(new_g.has_edges_between([0, 1], [1, 2])).all()

@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="MXNet doesn't support bool tensor")
@parametrize_dtype
def test_subgraph_mask(idtype):
    g = create_test_heterograph(idtype)
    g_graph = g['follows']
    g_bipartite = g['plays']

    x = F.randn((3, 5))
    y = F.randn((2, 4))
    g.nodes['user'].data['h'] = x
    g.edges['follows'].data['h'] = y

    def _check_subgraph(g, sg):
        assert sg.idtype == g.idtype
        assert sg.device == g.device
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes
        assert F.array_equal(F.tensor(sg.nodes['user'].data[dgl.NID]),
                             F.tensor([1, 2], idtype))
        assert F.array_equal(F.tensor(sg.nodes['game'].data[dgl.NID]),
                             F.tensor([0], idtype))
        assert F.array_equal(F.tensor(sg.edges['follows'].data[dgl.EID]),
                             F.tensor([1], idtype))
        assert F.array_equal(F.tensor(sg.edges['plays'].data[dgl.EID]),
                             F.tensor([1], idtype))
        assert F.array_equal(F.tensor(sg.edges['wishes'].data[dgl.EID]),
                             F.tensor([1], idtype))
        assert sg.number_of_nodes('developer') == 0
        assert sg.number_of_edges('develops') == 0
        assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'][1:3])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'][1:2])

    sg1 = g.subgraph({'user': F.tensor([False, True, True], dtype=F.bool),
                      'game': F.tensor([True, False, False, False], dtype=F.bool)})
    _check_subgraph(g, sg1)
    if F._default_context_str != 'gpu':
        # TODO(minjie): enable this later
        sg2 = g.edge_subgraph({'follows': F.tensor([False, True], dtype=F.bool),
                               'plays': F.tensor([False, True, False, False], dtype=F.bool),
                               'wishes': F.tensor([False, True], dtype=F.bool)})
        _check_subgraph(g, sg2)

@parametrize_dtype
def test_subgraph(idtype):
    g = create_test_heterograph(idtype)
    g_graph = g['follows']
    g_bipartite = g['plays']

    x = F.randn((3, 5))
    y = F.randn((2, 4))
    g.nodes['user'].data['h'] = x
    g.edges['follows'].data['h'] = y

    def _check_subgraph(g, sg):
        assert sg.idtype == g.idtype
        assert sg.device == g.device
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes
        assert F.array_equal(F.tensor(sg.nodes['user'].data[dgl.NID]),
                             F.tensor([1, 2], g.idtype))
        assert F.array_equal(F.tensor(sg.nodes['game'].data[dgl.NID]),
                             F.tensor([0], g.idtype))
        assert F.array_equal(F.tensor(sg.edges['follows'].data[dgl.EID]),
                             F.tensor([1], g.idtype))
        assert F.array_equal(F.tensor(sg.edges['plays'].data[dgl.EID]),
                             F.tensor([1], g.idtype))
        assert F.array_equal(F.tensor(sg.edges['wishes'].data[dgl.EID]),
                             F.tensor([1], g.idtype))
        assert sg.number_of_nodes('developer') == 0
        assert sg.number_of_edges('develops') == 0
        assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'][1:3])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'][1:2])

    sg1 = g.subgraph({'user': [1, 2], 'game': [0]})
    _check_subgraph(g, sg1)
    if F._default_context_str != 'gpu':
        # TODO(minjie): enable this later
        sg2 = g.edge_subgraph({'follows': [1], 'plays': [1], 'wishes': [1]})
        _check_subgraph(g, sg2)

    # backend tensor input
    sg1 = g.subgraph({'user': F.tensor([1, 2], dtype=idtype),
                      'game': F.tensor([0], dtype=idtype)})
    _check_subgraph(g, sg1)
    if F._default_context_str != 'gpu':
        # TODO(minjie): enable this later
        sg2 = g.edge_subgraph({'follows': F.tensor([1], dtype=idtype),
                               'plays': F.tensor([1], dtype=idtype),
                               'wishes': F.tensor([1], dtype=idtype)})
        _check_subgraph(g, sg2)

    # numpy input
    sg1 = g.subgraph({'user': np.array([1, 2]),
                      'game': np.array([0])})
    _check_subgraph(g, sg1)
    if F._default_context_str != 'gpu':
        # TODO(minjie): enable this later
        sg2 = g.edge_subgraph({'follows': np.array([1]),
                               'plays': np.array([1]),
                               'wishes': np.array([1])})
        _check_subgraph(g, sg2)

    def _check_subgraph_single_ntype(g, sg, preserve_nodes=False):
        assert sg.idtype == g.idtype
        assert sg.device == g.device
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes

        if not preserve_nodes:
            assert F.array_equal(F.tensor(sg.nodes['user'].data[dgl.NID]),
                                 F.tensor([1, 2], g.idtype))
        else:
            for ntype in sg.ntypes:
                assert g.number_of_nodes(ntype) == sg.number_of_nodes(ntype)

        assert F.array_equal(F.tensor(sg.edges['follows'].data[dgl.EID]),
                             F.tensor([1], g.idtype))

        if not preserve_nodes:
            assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'][1:3])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'][1:2])

    def _check_subgraph_single_etype(g, sg, preserve_nodes=False):
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes

        if not preserve_nodes:
            assert F.array_equal(F.tensor(sg.nodes['user'].data[dgl.NID]),
                                 F.tensor([0, 1], g.idtype))
            assert F.array_equal(F.tensor(sg.nodes['game'].data[dgl.NID]),
                                 F.tensor([0], g.idtype))
        else:
            for ntype in sg.ntypes:
                assert g.number_of_nodes(ntype) == sg.number_of_nodes(ntype)

        assert F.array_equal(F.tensor(sg.edges['plays'].data[dgl.EID]),
                             F.tensor([0, 1], g.idtype))

    sg1_graph = g_graph.subgraph([1, 2])
    _check_subgraph_single_ntype(g_graph, sg1_graph)
    if F._default_context_str != 'gpu':
        # TODO(minjie): enable this later
        sg1_graph = g_graph.edge_subgraph([1])
        _check_subgraph_single_ntype(g_graph, sg1_graph)
        sg1_graph = g_graph.edge_subgraph([1], preserve_nodes=True)
        _check_subgraph_single_ntype(g_graph, sg1_graph, True)
        sg2_bipartite = g_bipartite.edge_subgraph([0, 1])
        _check_subgraph_single_etype(g_bipartite, sg2_bipartite)
        sg2_bipartite = g_bipartite.edge_subgraph([0, 1], preserve_nodes=True)
        _check_subgraph_single_etype(g_bipartite, sg2_bipartite, True)

    def _check_typed_subgraph1(g, sg):
        assert g.idtype == sg.idtype
        assert g.device == sg.device
        assert set(sg.ntypes) == {'user', 'game'}
        assert set(sg.etypes) == {'follows', 'plays', 'wishes'}
        for ntype in sg.ntypes:
            assert sg.number_of_nodes(ntype) == g.number_of_nodes(ntype)
        for etype in sg.etypes:
            src_sg, dst_sg = sg.all_edges(etype=etype, order='eid')
            src_g, dst_g = g.all_edges(etype=etype, order='eid')
            assert F.array_equal(src_sg, src_g)
            assert F.array_equal(dst_sg, dst_g)
        assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'])
        g.nodes['user'].data['h'] = F.scatter_row(g.nodes['user'].data['h'], F.tensor([2]), F.randn((1, 5)))
        g.edges['follows'].data['h'] = F.scatter_row(g.edges['follows'].data['h'], F.tensor([1]), F.randn((1, 4)))
        assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'])

    def _check_typed_subgraph2(g, sg):
        assert set(sg.ntypes) == {'developer', 'game'}
        assert set(sg.etypes) == {'develops'}
        for ntype in sg.ntypes:
            assert sg.number_of_nodes(ntype) == g.number_of_nodes(ntype)
        for etype in sg.etypes:
            src_sg, dst_sg = sg.all_edges(etype=etype, order='eid')
            src_g, dst_g = g.all_edges(etype=etype, order='eid')
            assert F.array_equal(src_sg, src_g)
            assert F.array_equal(dst_sg, dst_g)

    sg3 = g.node_type_subgraph(['user', 'game'])
    _check_typed_subgraph1(g, sg3)
    sg4 = g.edge_type_subgraph(['develops'])
    _check_typed_subgraph2(g, sg4)
    sg5 = g.edge_type_subgraph(['follows', 'plays', 'wishes'])
    _check_typed_subgraph1(g, sg5)

@parametrize_dtype
def test_apply(idtype):
    def node_udf(nodes):
        return {'h': nodes.data['h'] * 2}
    def node_udf2(nodes):
        return {'h': F.sum(nodes.data['h'], dim=1, keepdims=True)}
    def edge_udf(edges):
        return {'h': edges.data['h'] * 2 + edges.src['h']}

    g = create_test_heterograph(idtype)
    g.nodes['user'].data['h'] = F.ones((3, 5))
    g.apply_nodes(node_udf, ntype='user')
    assert F.array_equal(g.nodes['user'].data['h'], F.ones((3, 5)) * 2)

    g['plays'].edata['h'] = F.ones((4, 5))
    g.apply_edges(edge_udf, etype=('user', 'plays', 'game'))
    assert F.array_equal(g['plays'].edata['h'], F.ones((4, 5)) * 4)

    # test apply on graph with only one type
    g['follows'].apply_nodes(node_udf)
    assert F.array_equal(g.nodes['user'].data['h'], F.ones((3, 5)) * 4)

    g['plays'].apply_edges(edge_udf)
    assert F.array_equal(g['plays'].edata['h'], F.ones((4, 5)) * 12)

    # Test the case that feature size changes
    g.nodes['user'].data['h'] = F.ones((3, 5))
    g.apply_nodes(node_udf2, ntype='user')
    assert F.array_equal(g.nodes['user'].data['h'], F.ones((3, 1)) * 5)

    # test fail case
    # fail due to multiple types
    with pytest.raises(DGLError):
        g.apply_nodes(node_udf)

    with pytest.raises(DGLError):
        g.apply_edges(edge_udf)

@parametrize_dtype
def test_level2(idtype):
    #edges = {
    #    'follows': ([0, 1], [1, 2]),
    #    'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
    #    'wishes': ([0, 2], [1, 0]),
    #    'develops': ([0, 1], [0, 1]),
    #}
    g = create_test_heterograph(idtype)
    def rfunc(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def rfunc2(nodes):
        return {'y': F.max(nodes.mailbox['m'], 1)}
    def mfunc(edges):
        return {'m': edges.src['h']}
    def afunc(nodes):
        return {'y' : nodes.data['y'] + 1}

    #############################################################
    #  send_and_recv
    #############################################################

    g.nodes['user'].data['h'] = F.ones((3, 2))
    g.send_and_recv([2, 3], mfunc, rfunc, etype='plays')
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))

    # only one type
    g['plays'].send_and_recv([2, 3], mfunc, rfunc)
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))

    # test fail case
    # fail due to multiple types
    with pytest.raises(DGLError):
        g.send_and_recv([2, 3], mfunc, rfunc)

    g.nodes['game'].data.clear()

    #############################################################
    #  pull
    #############################################################

    g.nodes['user'].data['h'] = F.ones((3, 2))
    g.pull(1, mfunc, rfunc, etype='plays')
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))

    # only one type
    g['plays'].pull(1, mfunc, rfunc)
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[0., 0.], [2., 2.]]))

    # test fail case
    with pytest.raises(DGLError):
        g.pull(1, mfunc, rfunc)

    g.nodes['game'].data.clear()

    #############################################################
    #  update_all
    #############################################################

    g.nodes['user'].data['h'] = F.ones((3, 2))
    g.update_all(mfunc, rfunc, etype='plays')
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[2., 2.], [2., 2.]]))

    # only one type
    g['plays'].update_all(mfunc, rfunc)
    y = g.nodes['game'].data['y']
    assert F.array_equal(y, F.tensor([[2., 2.], [2., 2.]]))

    # test fail case
    # fail due to multiple types
    with pytest.raises(DGLError):
        g.update_all(mfunc, rfunc)

    # test multi
    g.multi_update_all(
        {'plays' : (mfunc, rfunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum')
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[3., 3.], [3., 3.]]))

    # test multi
    g.multi_update_all(
        {'plays' : (mfunc, rfunc, afunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum', afunc)
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[5., 5.], [5., 5.]]))

    # test cross reducer
    g.nodes['user'].data['h'] = F.randn((3, 2))
    for cred in ['sum', 'max', 'min', 'mean', 'stack']:
        g.multi_update_all(
            {'plays' : (mfunc, rfunc, afunc),
             'wishes': (mfunc, rfunc2)},
            cred, afunc)
        y = g.nodes['game'].data['y']
        g['plays'].update_all(mfunc, rfunc, afunc)
        y1 = g.nodes['game'].data['y']
        g['wishes'].update_all(mfunc, rfunc2)
        y2 = g.nodes['game'].data['y']
        if cred == 'stack':
            # stack has an internal order by edge type id
            yy = F.stack([y1, y2], 1)
            yy = yy + 1  # final afunc
            assert F.array_equal(y, yy)
        else:
            yy = get_redfn(cred)(F.stack([y1, y2], 0), 0)
            yy = yy + 1  # final afunc
            assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    with pytest.raises(DGLError):
        g.update_all(
            {'plays' : (mfunc, rfunc),
             'follows': (mfunc, rfunc2)},
            'sum')

    g.nodes['game'].data.clear()

@parametrize_dtype
def test_updates(idtype):
    def msg_func(edges):
        return {'m': edges.src['h']}
    def reduce_func(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def apply_func(nodes):
        return {'y': nodes.data['y'] * 2}
    g = create_test_heterograph(idtype)
    x = F.randn((3, 5))
    g.nodes['user'].data['h'] = x

    for msg, red, apply in itertools.product(
            [fn.copy_u('h', 'm'), msg_func], [fn.sum('m', 'y'), reduce_func],
            [None, apply_func]):
        multiplier = 1 if apply is None else 2

        g['user', 'plays', 'game'].update_all(msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g.nodes['game'].data['y']

        g['user', 'plays', 'game'].send_and_recv(([0, 1, 2], [0, 1, 1]), msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        assert F.array_equal(y[1], (x[1] + x[2]) * multiplier)
        del g.nodes['game'].data['y']

        # pulls from destination (game) node 0
        g['user', 'plays', 'game'].pull(0, msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], (x[0] + x[1]) * multiplier)
        del g.nodes['game'].data['y']

        # pushes from source (user) node 0
        g['user', 'plays', 'game'].push(0, msg, red, apply)
        y = g.nodes['game'].data['y']
        assert F.array_equal(y[0], x[0] * multiplier)
        del g.nodes['game'].data['y']


@unittest.skipIf(F.backend_name == "jax", reason="Jax backward doesn't work like this")
@parametrize_dtype
def test_backward(idtype):
    g = create_test_heterograph(idtype)
    x = F.randn((3, 5))
    F.attach_grad(x)
    g.nodes['user'].data['h'] = x
    with F.record_grad():
        g.multi_update_all(
            {'plays' : (fn.copy_u('h', 'm'), fn.sum('m', 'y')),
             'wishes': (fn.copy_u('h', 'm'), fn.sum('m', 'y'))},
            'sum')
        y = g.nodes['game'].data['y']
        F.backward(y, F.ones(y.shape))
    print(F.grad(x))
    assert F.array_equal(F.grad(x), F.tensor([[2., 2., 2., 2., 2.],
                                              [2., 2., 2., 2., 2.],
                                              [2., 2., 2., 2., 2.]]))


@parametrize_dtype
def test_empty_heterograph(idtype):
    def assert_empty(g):
        assert g.number_of_nodes('user') == 0
        assert g.number_of_edges('plays') == 0
        assert g.number_of_nodes('game') == 0

    # empty src-dst pair
    assert_empty(dgl.heterograph({('user', 'plays', 'game'): ([], [])}))

    g = dgl.heterograph({('user', 'follows', 'user'): ([], [])}, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 0
    assert g.number_of_edges('follows') == 0

    # empty relation graph with others
    g = dgl.heterograph({('user', 'plays', 'game'): ([], []), ('developer', 'develops', 'game'):
        ([0, 1], [0, 1])}, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 0
    assert g.number_of_edges('plays') == 0
    assert g.number_of_nodes('game') == 2
    assert g.number_of_edges('develops') == 2
    assert g.number_of_nodes('developer') == 2

@parametrize_dtype
def test_types_in_function(idtype):
    def mfunc1(edges):
        assert edges.canonical_etype == ('user', 'follow', 'user')
        return {}

    def rfunc1(nodes):
        assert nodes.ntype == 'user'
        return {}

    def filter_nodes1(nodes):
        assert nodes.ntype == 'user'
        return F.zeros((3,))

    def filter_edges1(edges):
        assert edges.canonical_etype == ('user', 'follow', 'user')
        return F.zeros((2,))

    def mfunc2(edges):
        assert edges.canonical_etype == ('user', 'plays', 'game')
        return {}

    def rfunc2(nodes):
        assert nodes.ntype == 'game'
        return {}

    def filter_nodes2(nodes):
        assert nodes.ntype == 'game'
        return F.zeros((3,))

    def filter_edges2(edges):
        assert edges.canonical_etype == ('user', 'plays', 'game')
        return F.zeros((2,))

    g = dgl.heterograph({('user', 'follow', 'user'): ((0, 1), (1, 2))},
                        idtype=idtype, device=F.ctx())
    g.apply_nodes(rfunc1)
    g.apply_edges(mfunc1)
    g.update_all(mfunc1, rfunc1)
    g.send_and_recv([0, 1], mfunc1, rfunc1)
    g.push([0], mfunc1, rfunc1)
    g.pull([1], mfunc1, rfunc1)
    g.filter_nodes(filter_nodes1)
    g.filter_edges(filter_edges1)

    g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx())
    g.apply_nodes(rfunc2, ntype='game')
    g.apply_edges(mfunc2)
    g.update_all(mfunc2, rfunc2)
    g.send_and_recv([0, 1], mfunc2, rfunc2)
    g.push([0], mfunc2, rfunc2)
    g.pull([1], mfunc2, rfunc2)
    g.filter_nodes(filter_nodes2, ntype='game')
    g.filter_edges(filter_edges2)

@parametrize_dtype
def test_stack_reduce(idtype):
    #edges = {
    #    'follows': ([0, 1], [1, 2]),
    #    'plays': ([0, 1, 2, 1], [0, 0, 1, 1]),
    #    'wishes': ([0, 2], [1, 0]),
    #    'develops': ([0, 1], [0, 1]),
    #}
    g = create_test_heterograph(idtype)
    g.nodes['user'].data['h'] = F.randn((3, 200))
    def rfunc(nodes):
        return {'y': F.sum(nodes.mailbox['m'], 1)}
    def rfunc2(nodes):
        return {'y': F.max(nodes.mailbox['m'], 1)}
    def mfunc(edges):
        return {'m': edges.src['h']}
    g.multi_update_all(
            {'plays' : (mfunc, rfunc),
             'wishes': (mfunc, rfunc2)},
            'stack')
    assert g.nodes['game'].data['y'].shape == (g.number_of_nodes('game'), 2, 200)
    # only one type-wise update_all, stack still adds one dimension
    g.multi_update_all(
            {'plays' : (mfunc, rfunc)},
            'stack')
    assert g.nodes['game'].data['y'].shape == (g.number_of_nodes('game'), 1, 200)

@parametrize_dtype
def test_isolated_ntype(idtype):
    g = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 1, 2], [1, 2, 3])},
        num_nodes_dict={'A': 3, 'B': 4, 'C': 4},
        idtype=idtype, device=F.ctx())
    assert g.number_of_nodes('A') == 3
    assert g.number_of_nodes('B') == 4
    assert g.number_of_nodes('C') == 4

    g = dgl.heterograph({
        ('A', 'AC', 'C'): ([0, 1, 2], [1, 2, 3])},
        num_nodes_dict={'A': 3, 'B': 4, 'C': 4},
        idtype=idtype, device=F.ctx())
    assert g.number_of_nodes('A') == 3
    assert g.number_of_nodes('B') == 4
    assert g.number_of_nodes('C') == 4

    G = dgl.graph(([0, 1, 2], [4, 5, 6]), num_nodes=11, idtype=idtype, device=F.ctx())
    G.ndata[dgl.NTYPE] = F.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=F.int64)
    G.edata[dgl.ETYPE] = F.tensor([0, 0, 0], dtype=F.int64)
    g = dgl.to_heterogeneous(G, ['A', 'B', 'C'], ['AB'])
    assert g.number_of_nodes('A') == 3
    assert g.number_of_nodes('B') == 4
    assert g.number_of_nodes('C') == 4


@parametrize_dtype
def test_ismultigraph(idtype):
    g1 = dgl.heterograph({('A', 'AB', 'B'): ([0, 0, 1, 2], [1, 2, 5, 5])},
                         {'A': 6, 'B': 6}, idtype=idtype, device=F.ctx())
    assert g1.is_multigraph == False
    g2 = dgl.heterograph({('A', 'AC', 'C'): ([0, 0, 0, 1], [1, 1, 2, 5])},
                         {'A': 6, 'C': 6}, idtype=idtype, device=F.ctx())
    assert g2.is_multigraph == True
    g3 = dgl.graph(((0, 1), (1, 2)), num_nodes=6, idtype=idtype, device=F.ctx())
    assert g3.is_multigraph == False
    g4 = dgl.graph(([0, 0, 1], [1, 1, 2]), num_nodes=6, idtype=idtype, device=F.ctx())
    assert g4.is_multigraph == True
    g = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 0, 1, 2], [1, 2, 5, 5]),
        ('A', 'AA', 'A'): ([0, 1], [1, 2])},
        {'A': 6, 'B': 6}, idtype=idtype, device=F.ctx())
    assert g.is_multigraph == False
    g = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 0, 1, 2], [1, 2, 5, 5]),
        ('A', 'AC', 'C'): ([0, 0, 0, 1], [1, 1, 2, 5])},
        {'A': 6, 'B': 6, 'C': 6}, idtype=idtype, device=F.ctx())
    assert g.is_multigraph == True
    g = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 0, 1, 2], [1, 2, 5, 5]),
        ('A', 'AA', 'A'): ([0, 0, 1], [1, 1, 2])},
        {'A': 6, 'B': 6}, idtype=idtype, device=F.ctx())
    assert g.is_multigraph == True
    g = dgl.heterograph({
        ('A', 'AC', 'C'): ([0, 0, 0, 1], [1, 1, 2, 5]),
        ('A', 'AA', 'A'): ([0, 1], [1, 2])},
        {'A': 6, 'C': 6}, idtype=idtype, device=F.ctx())
    assert g.is_multigraph == True

@parametrize_dtype
def test_bipartite(idtype):
    g1 = dgl.heterograph({('A', 'AB', 'B'): ([0, 0, 1], [1, 2, 5])},
                         idtype=idtype, device=F.ctx())
    assert g1.is_unibipartite
    assert len(g1.ntypes) == 2
    assert g1.etypes == ['AB']
    assert g1.srctypes == ['A']
    assert g1.dsttypes == ['B']
    assert g1.number_of_nodes('A') == 2
    assert g1.number_of_nodes('B') == 6
    assert g1.number_of_src_nodes('A') == 2
    assert g1.number_of_src_nodes() == 2
    assert g1.number_of_dst_nodes('B') == 6
    assert g1.number_of_dst_nodes() == 6
    assert g1.number_of_edges() == 3
    g1.srcdata['h'] = F.randn((2, 5))
    assert F.array_equal(g1.srcnodes['A'].data['h'], g1.srcdata['h'])
    assert F.array_equal(g1.nodes['A'].data['h'], g1.srcdata['h'])
    assert F.array_equal(g1.nodes['SRC/A'].data['h'], g1.srcdata['h'])
    g1.dstdata['h'] = F.randn((6, 3))
    assert F.array_equal(g1.dstnodes['B'].data['h'], g1.dstdata['h'])
    assert F.array_equal(g1.nodes['B'].data['h'], g1.dstdata['h'])
    assert F.array_equal(g1.nodes['DST/B'].data['h'], g1.dstdata['h'])

    # more complicated bipartite
    g2 = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 0, 1], [1, 2, 5]),
        ('A', 'AC', 'C'): ([1, 0], [0, 0])
    }, idtype=idtype, device=F.ctx())

    assert g2.is_unibipartite
    assert g2.srctypes == ['A']
    assert set(g2.dsttypes) == {'B', 'C'}
    assert g2.number_of_nodes('A') == 2
    assert g2.number_of_nodes('B') == 6
    assert g2.number_of_nodes('C') == 1
    assert g2.number_of_src_nodes('A') == 2
    assert g2.number_of_src_nodes() == 2
    assert g2.number_of_dst_nodes('B') == 6
    assert g2.number_of_dst_nodes('C') == 1
    g2.srcdata['h'] = F.randn((2, 5))
    assert F.array_equal(g2.srcnodes['A'].data['h'], g2.srcdata['h'])
    assert F.array_equal(g2.nodes['A'].data['h'], g2.srcdata['h'])
    assert F.array_equal(g2.nodes['SRC/A'].data['h'], g2.srcdata['h'])

    g3 = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 0, 1], [1, 2, 5]),
        ('A', 'AC', 'C'): ([1, 0], [0, 0]),
        ('A', 'AA', 'A'): ([0, 1], [0, 1])
    }, idtype=idtype, device=F.ctx())
    assert not g3.is_unibipartite

    g4 = dgl.heterograph({
        ('A', 'AB', 'B'): ([0, 0, 1], [1, 2, 5]),
        ('C', 'CA', 'A'): ([1, 0], [0, 0])
    }, idtype=idtype, device=F.ctx())

    assert not g4.is_unibipartite

@parametrize_dtype
def test_dtype_cast(idtype):
    g = dgl.graph(([0, 1, 0, 2], [0, 1, 1, 0]), idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    g.ndata["feat"] = F.tensor([3, 4, 5])
    g.edata["h"] = F.tensor([3, 4, 5, 6])
    if idtype == "int32":
        g_cast = g.long()
        assert g_cast.idtype == F.int64
    else:
        g_cast = g.int()
        assert g_cast.idtype == F.int32
    test_utils.check_graph_equal(g, g_cast, check_idtype=False)

@parametrize_dtype
def test_format(idtype):
    # single relation
    g = dgl.graph(([0, 1, 0, 2], [0, 1, 1, 0]), idtype=idtype, device=F.ctx())
    assert g.formats()['created'] == ['coo']
    g1 = g.formats(['coo', 'csr', 'csc'])
    assert len(g1.formats()['created']) + len(g1.formats()['not created']) == 3
    g1.create_formats_()
    assert len(g1.formats()['created']) == 3
    assert g.formats()['created'] == ['coo']

    # multiple relation
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1], [1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        ('developer', 'develops', 'game'): ([0, 1], [0, 1])
        }, idtype=idtype, device=F.ctx())
    user_feat = F.randn((g['follows'].number_of_src_nodes(), 5))
    g['follows'].srcdata['h'] = user_feat
    g1 = g.formats('csc')
    # test frame
    assert F.array_equal(g1['follows'].srcdata['h'], user_feat)
    # test each relation graph
    assert g1.formats()['created'] == ['csc']
    assert len(g1.formats()['not created']) == 0

@parametrize_dtype
def test_edges_order(idtype):
    # (0, 2), (1, 2), (0, 1), (0, 1), (2, 1)
    g = dgl.graph((
        np.array([0, 1, 0, 0, 2]),
        np.array([2, 2, 1, 1, 1])
    ), idtype=idtype, device=F.ctx())

    print(g.formats())
    src, dst = g.all_edges(order='srcdst')
    assert F.array_equal(src, F.tensor([0, 0, 0, 1, 2], dtype=idtype))
    assert F.array_equal(dst, F.tensor([1, 1, 2, 2, 1], dtype=idtype))

@parametrize_dtype
def test_reverse(idtype):
    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1, 2, 4, 3 ,1, 3], [1, 2, 3, 2, 0, 0, 1]),
    }, idtype=idtype, device=F.ctx())
    gidx = g._graph
    r_gidx = gidx.reverse()

    assert gidx.number_of_nodes(0) == r_gidx.number_of_nodes(0)
    assert gidx.number_of_edges(0) == r_gidx.number_of_edges(0)
    g_s, g_d, _ = gidx.edges(0)
    rg_s, rg_d, _ = r_gidx.edges(0)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)

    # force to start with 'csr'
    gidx = gidx.formats('csr')
    gidx = gidx.formats(['coo', 'csr', 'csc'])
    r_gidx = gidx.reverse()
    assert 'csr' in gidx.formats()['created']
    assert 'csc' in r_gidx.formats()['created']
    assert gidx.number_of_nodes(0) == r_gidx.number_of_nodes(0)
    assert gidx.number_of_edges(0) == r_gidx.number_of_edges(0)
    g_s, g_d, _ = gidx.edges(0)
    rg_s, rg_d, _ = r_gidx.edges(0)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)

    # force to start with 'csc'
    gidx = gidx.formats('csc')
    gidx = gidx.formats(['coo', 'csr', 'csc'])
    r_gidx = gidx.reverse()
    assert 'csc' in gidx.formats()['created']
    assert 'csr' in r_gidx.formats()['created']
    assert gidx.number_of_nodes(0) == r_gidx.number_of_nodes(0)
    assert gidx.number_of_edges(0) == r_gidx.number_of_edges(0)
    g_s, g_d, _ = gidx.edges(0)
    rg_s, rg_d, _ = r_gidx.edges(0)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)

    g = dgl.heterograph({
        ('user', 'follows', 'user'): ([0, 1, 2, 4, 3 ,1, 3], [1, 2, 3, 2, 0, 0, 1]),
        ('user', 'plays', 'game'): ([0, 0, 2, 3, 3, 4, 1], [1, 0, 1, 0, 1, 0, 0]),
        ('developer', 'develops', 'game'): ([0, 1, 1, 2], [0, 0, 1, 1]),
        }, idtype=idtype, device=F.ctx())
    gidx = g._graph
    r_gidx = gidx.reverse()

    # metagraph
    mg = gidx.metagraph
    r_mg = r_gidx.metagraph
    for etype in range(3):
        assert mg.find_edge(etype) == r_mg.find_edge(etype)[::-1]

    # three node types and three edge types
    assert gidx.number_of_nodes(0) == r_gidx.number_of_nodes(0)
    assert gidx.number_of_nodes(1) == r_gidx.number_of_nodes(1)
    assert gidx.number_of_nodes(2) == r_gidx.number_of_nodes(2)
    assert gidx.number_of_edges(0) == r_gidx.number_of_edges(0)
    assert gidx.number_of_edges(1) == r_gidx.number_of_edges(1)
    assert gidx.number_of_edges(2) == r_gidx.number_of_edges(2)
    g_s, g_d, _ = gidx.edges(0)
    rg_s, rg_d, _ = r_gidx.edges(0)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)
    g_s, g_d, _ = gidx.edges(1)
    rg_s, rg_d, _ = r_gidx.edges(1)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)
    g_s, g_d, _ = gidx.edges(2)
    rg_s, rg_d, _ = r_gidx.edges(2)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)

    # force to start with 'csr'
    gidx = gidx.formats('csr')
    gidx = gidx.formats(['coo', 'csr', 'csc'])
    r_gidx = gidx.reverse()
    # three node types and three edge types
    assert 'csr' in gidx.formats()['created']
    assert 'csc' in r_gidx.formats()['created']
    assert gidx.number_of_nodes(0) == r_gidx.number_of_nodes(0)
    assert gidx.number_of_nodes(1) == r_gidx.number_of_nodes(1)
    assert gidx.number_of_nodes(2) == r_gidx.number_of_nodes(2)
    assert gidx.number_of_edges(0) == r_gidx.number_of_edges(0)
    assert gidx.number_of_edges(1) == r_gidx.number_of_edges(1)
    assert gidx.number_of_edges(2) == r_gidx.number_of_edges(2)
    g_s, g_d, _ = gidx.edges(0)
    rg_s, rg_d, _ = r_gidx.edges(0)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)
    g_s, g_d, _ = gidx.edges(1)
    rg_s, rg_d, _ = r_gidx.edges(1)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)
    g_s, g_d, _ = gidx.edges(2)
    rg_s, rg_d, _ = r_gidx.edges(2)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)

    # force to start with 'csc'
    gidx = gidx.formats('csc')
    gidx = gidx.formats(['coo', 'csr', 'csc'])
    r_gidx = gidx.reverse()
    # three node types and three edge types
    assert 'csc' in gidx.formats()['created']
    assert 'csr' in r_gidx.formats()['created']
    assert gidx.number_of_nodes(0) == r_gidx.number_of_nodes(0)
    assert gidx.number_of_nodes(1) == r_gidx.number_of_nodes(1)
    assert gidx.number_of_nodes(2) == r_gidx.number_of_nodes(2)
    assert gidx.number_of_edges(0) == r_gidx.number_of_edges(0)
    assert gidx.number_of_edges(1) == r_gidx.number_of_edges(1)
    assert gidx.number_of_edges(2) == r_gidx.number_of_edges(2)
    g_s, g_d, _ = gidx.edges(0)
    rg_s, rg_d, _ = r_gidx.edges(0)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)
    g_s, g_d, _ = gidx.edges(1)
    rg_s, rg_d, _ = r_gidx.edges(1)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)
    g_s, g_d, _ = gidx.edges(2)
    rg_s, rg_d, _ = r_gidx.edges(2)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)

@parametrize_dtype
def test_clone(idtype):
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.edata['h'] = F.copy_to(F.tensor([1, 1], dtype=idtype), ctx=F.ctx())

    new_g = g.clone()
    assert g.number_of_nodes() == new_g.number_of_nodes()
    assert g.number_of_edges() == new_g.number_of_edges()
    assert g.device == new_g.device
    assert g.idtype == new_g.idtype
    assert F.array_equal(g.ndata['h'], new_g.ndata['h'])
    assert F.array_equal(g.edata['h'], new_g.edata['h'])
    # data change
    new_g.ndata['h'] = F.copy_to(F.tensor([2, 2, 2], dtype=idtype), ctx=F.ctx())
    assert (F.array_equal(g.ndata['h'], new_g.ndata['h']) == False)
    g.edata['h'] = F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())
    assert (F.array_equal(g.edata['h'], new_g.edata['h']) == False)
    # graph structure change
    g.add_nodes(1)
    assert g.number_of_nodes() != new_g.number_of_nodes()
    new_g.add_edges(1, 1)
    assert g.number_of_edges() != new_g.number_of_edges()

    # zero data graph
    g = dgl.graph(([], []), num_nodes=0, idtype=idtype, device=F.ctx())
    new_g = g.clone()
    assert g.number_of_nodes() == new_g.number_of_nodes()
    assert g.number_of_edges() == new_g.number_of_edges()

    # heterograph
    g = create_test_heterograph3(idtype)
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 2, 3, 4], dtype=idtype), ctx=F.ctx())
    new_g = g.clone()
    assert g.number_of_nodes('user') == new_g.number_of_nodes('user')
    assert g.number_of_nodes('game') == new_g.number_of_nodes('game')
    assert g.number_of_nodes('developer') == new_g.number_of_nodes('developer')
    assert g.number_of_edges('plays') == new_g.number_of_edges('plays')
    assert g.number_of_edges('develops') == new_g.number_of_edges('develops')
    assert F.array_equal(g.nodes['user'].data['h'], new_g.nodes['user'].data['h'])
    assert F.array_equal(g.nodes['game'].data['h'], new_g.nodes['game'].data['h'])
    assert F.array_equal(g.edges['plays'].data['h'], new_g.edges['plays'].data['h'])
    assert g.device == new_g.device
    assert g.idtype == new_g.idtype
    u, v = g.edges(form='uv', order='eid', etype='plays')
    nu, nv = new_g.edges(form='uv', order='eid', etype='plays')
    assert F.array_equal(u, nu)
    assert F.array_equal(v, nv)
    # graph structure change
    u = F.tensor([0, 4], dtype=idtype)
    v = F.tensor([2, 6], dtype=idtype)
    g.add_edges(u, v, etype='plays')
    u, v = g.edges(form='uv', order='eid', etype='plays')
    assert u.shape[0] != nu.shape[0]
    assert v.shape[0] != nv.shape[0]
    assert g.nodes['user'].data['h'].shape[0] != new_g.nodes['user'].data['h'].shape[0]
    assert g.nodes['game'].data['h'].shape[0] != new_g.nodes['game'].data['h'].shape[0]
    assert g.edges['plays'].data['h'].shape[0] != new_g.edges['plays'].data['h'].shape[0]


@parametrize_dtype
def test_add_edges(idtype):
    # homogeneous graph
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    u = 0
    v = 1
    g.add_edges(u, v)
    assert g.device == F.ctx()
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    u = [0]
    v = [1]
    g.add_edges(u, v)
    assert g.device == F.ctx()
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 4
    u = F.tensor(u, dtype=idtype)
    v = F.tensor(v, dtype=idtype)
    g.add_edges(u, v)
    assert g.device == F.ctx()
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 5
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0, 1, 0, 0, 0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 1, 1, 1], dtype=idtype))

    # node id larger than current max node id
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    u = F.tensor([0, 1], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g.add_edges(u, v)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 4
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0, 1, 0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))

    # has data
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata['h'] = F.copy_to(F.tensor([1, 1, 1], dtype=idtype), ctx=F.ctx())
    g.edata['h'] = F.copy_to(F.tensor([1, 1], dtype=idtype), ctx=F.ctx())
    u = F.tensor([0, 1], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    e_feat = {'h' : F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
              'hh' : F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())}
    g.add_edges(u, v, e_feat)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 4
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0, 1, 0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))
    assert F.array_equal(g.ndata['h'], F.tensor([1, 1, 1, 0], dtype=idtype))
    assert F.array_equal(g.edata['h'], F.tensor([1, 1, 2, 2], dtype=idtype))
    assert F.array_equal(g.edata['hh'], F.tensor([0, 0, 2, 2], dtype=idtype))

    # zero data graph
    g = dgl.graph(([], []), num_nodes=0, idtype=idtype, device=F.ctx())
    u = F.tensor([0, 1], dtype=idtype)
    v = F.tensor([2, 2], dtype=idtype)
    e_feat = {'h' : F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
              'hh' : F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())}
    g.add_edges(u, v, e_feat)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2, 2], dtype=idtype))
    assert F.array_equal(g.edata['h'], F.tensor([2, 2], dtype=idtype))
    assert F.array_equal(g.edata['hh'], F.tensor([2, 2], dtype=idtype))

    # bipartite graph
    g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1], [1, 2])},
                        idtype=idtype, device=F.ctx())
    u = 0
    v = 1
    g.add_edges(u, v)
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 2
    assert g.number_of_nodes('game') == 3
    assert g.number_of_edges() == 3
    u = [0]
    v = [1]
    g.add_edges(u, v)
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 2
    assert g.number_of_nodes('game') == 3
    assert g.number_of_edges() == 4
    u = F.tensor(u, dtype=idtype)
    v = F.tensor(v, dtype=idtype)
    g.add_edges(u, v)
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 2
    assert g.number_of_nodes('game') == 3
    assert g.number_of_edges() == 5
    u, v = g.edges(form='uv')
    assert F.array_equal(u, F.tensor([0, 1, 0, 0, 0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 1, 1, 1], dtype=idtype))

    # node id larger than current max node id
    g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1], [1, 2])},
                        idtype=idtype, device=F.ctx())
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g.add_edges(u, v)
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 3
    assert g.number_of_nodes('game') == 4
    assert g.number_of_edges() == 4
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0, 1, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))

    # has data
    g = dgl.heterograph({
        ('user', 'plays', 'game'): ([0, 1], [1, 2])
    }, idtype=idtype, device=F.ctx())
    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2, 2], dtype=idtype), ctx=F.ctx())
    g.edata['h'] = F.copy_to(F.tensor([1, 1], dtype=idtype), ctx=F.ctx())
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    e_feat = {'h' : F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx()),
              'hh' : F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())}
    g.add_edges(u, v, e_feat)
    assert g.number_of_nodes('user') == 3
    assert g.number_of_nodes('game') == 4
    assert g.number_of_edges() == 4
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0, 1, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([1, 2, 2, 3], dtype=idtype))
    assert F.array_equal(g.nodes['user'].data['h'], F.tensor([1, 1, 0], dtype=idtype))
    assert F.array_equal(g.nodes['game'].data['h'], F.tensor([2, 2, 2, 0], dtype=idtype))
    assert F.array_equal(g.edata['h'], F.tensor([1, 1, 2, 2], dtype=idtype))
    assert F.array_equal(g.edata['hh'], F.tensor([0, 0, 2, 2], dtype=idtype))

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g.add_edges(u, v, etype='plays')
    assert g.number_of_nodes('user') == 3
    assert g.number_of_nodes('game') == 4
    assert g.number_of_nodes('developer') == 2
    assert g.number_of_edges('plays') == 6
    assert g.number_of_edges('develops') == 2
    u, v = g.edges(form='uv', order='eid', etype='plays')
    assert F.array_equal(u, F.tensor([0, 1, 1, 2, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 0, 1, 1, 2, 3], dtype=idtype))
    assert F.array_equal(g.nodes['user'].data['h'], F.tensor([1, 1, 1], dtype=idtype))
    assert F.array_equal(g.nodes['game'].data['h'], F.tensor([2, 2, 0, 0], dtype=idtype))
    assert F.array_equal(g.edges['plays'].data['h'], F.tensor([1, 1, 1, 1, 0, 0], dtype=idtype))

    # add with feature
    e_feat = {'h': F.copy_to(F.tensor([2, 2], dtype=idtype), ctx=F.ctx())}
    u = F.tensor([0, 2], dtype=idtype)
    v = F.tensor([2, 3], dtype=idtype)
    g.nodes['game'].data['h'] =  F.copy_to(F.tensor([2, 2, 1, 1], dtype=idtype), ctx=F.ctx())
    g.add_edges(u, v, data=e_feat, etype='develops')
    assert g.number_of_nodes('user') == 3
    assert g.number_of_nodes('game') == 4
    assert g.number_of_nodes('developer') == 3
    assert g.number_of_edges('plays') == 6
    assert g.number_of_edges('develops') == 4
    u, v = g.edges(form='uv', order='eid', etype='develops')
    assert F.array_equal(u, F.tensor([0, 1, 0, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1, 2, 3], dtype=idtype))
    assert F.array_equal(g.nodes['developer'].data['h'], F.tensor([3, 3, 0], dtype=idtype))
    assert F.array_equal(g.nodes['game'].data['h'], F.tensor([2, 2, 1, 1], dtype=idtype))
    assert F.array_equal(g.edges['develops'].data['h'], F.tensor([0, 0, 2, 2], dtype=idtype))

@parametrize_dtype
def test_add_nodes(idtype):
    # homogeneous Graphs
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata['h'] = F.copy_to(F.tensor([1,1,1], dtype=idtype), ctx=F.ctx())
    g.add_nodes(1)
    assert g.number_of_nodes() == 4
    assert F.array_equal(g.ndata['h'], F.tensor([1, 1, 1, 0], dtype=idtype))

    # zero node graph
    g = dgl.graph(([], []), num_nodes=3, idtype=idtype, device=F.ctx())
    g.ndata['h'] = F.copy_to(F.tensor([1,1,1], dtype=idtype), ctx=F.ctx())
    g.add_nodes(1, data={'h' : F.copy_to(F.tensor([2],  dtype=idtype), ctx=F.ctx())})
    assert g.number_of_nodes() == 4
    assert F.array_equal(g.ndata['h'], F.tensor([1, 1, 1, 2], dtype=idtype))

    # bipartite graph
    g = dgl.heterograph({('user', 'plays', 'game'): ([0, 1], [1, 2])},
                        idtype=idtype, device=F.ctx())
    g.add_nodes(2, data={'h' : F.copy_to(F.tensor([2, 2],  dtype=idtype), ctx=F.ctx())}, ntype='user')
    assert g.number_of_nodes('user') == 4
    assert F.array_equal(g.nodes['user'].data['h'], F.tensor([0, 0, 2, 2], dtype=idtype))
    g.add_nodes(2, ntype='game')
    assert g.number_of_nodes('game') == 5

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    g.add_nodes(1, ntype='user')
    g.add_nodes(2, data={'h' : F.copy_to(F.tensor([2, 2],  dtype=idtype), ctx=F.ctx())}, ntype='game')
    g.add_nodes(0, ntype='developer')
    assert g.number_of_nodes('user') == 4
    assert g.number_of_nodes('game') == 4
    assert g.number_of_nodes('developer') == 2
    assert F.array_equal(g.nodes['user'].data['h'], F.tensor([1, 1, 1, 0], dtype=idtype))
    assert F.array_equal(g.nodes['game'].data['h'], F.tensor([2, 2, 2, 2], dtype=idtype))

@unittest.skipIf(dgl.backend.backend_name == "mxnet", reason="MXNet has error with (0,) shape tensor.")
@parametrize_dtype
def test_remove_edges(idtype):
    # homogeneous Graphs
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    e = 0
    g.remove_edges(e)
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    e = [0]
    g.remove_edges(e)
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    e = F.tensor([0], dtype=idtype)
    g.remove_edges(e)
    assert g.number_of_edges() == 0

    # has node data
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.ndata['h'] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g.remove_edges(1)
    assert g.number_of_edges() == 1
    assert F.array_equal(g.ndata['h'], F.tensor([1, 2, 3], dtype=idtype))

    # has edge data
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    g.edata['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    g.remove_edges(0)
    assert g.number_of_edges() == 1
    assert F.array_equal(g.edata['h'], F.tensor([2], dtype=idtype))

    # invalid eid
    assert_fail = False
    try:
        g.remove_edges(1)
    except:
        assert_fail = True
    assert assert_fail

    # bipartite graph
    g = dgl.heterograph({
        ('user', 'plays', 'game'): ([0, 1], [1, 2])
    }, idtype=idtype, device=F.ctx())
    e = 0
    g.remove_edges(e)
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    g = dgl.heterograph(
        {('user', 'plays', 'game'): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx())
    e = [0]
    g.remove_edges(e)
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    e = F.tensor([0], dtype=idtype)
    g.remove_edges(e)
    assert g.number_of_edges() == 0

    # has data
    g = dgl.heterograph(
        {('user', 'plays', 'game'): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx())
    g.nodes['user'].data['h'] = F.copy_to(F.tensor([1, 1], dtype=idtype), ctx=F.ctx())
    g.nodes['game'].data['h'] = F.copy_to(F.tensor([2, 2, 2], dtype=idtype), ctx=F.ctx())
    g.edata['h'] = F.copy_to(F.tensor([1, 2], dtype=idtype), ctx=F.ctx())
    g.remove_edges(1)
    assert g.number_of_edges() == 1
    assert F.array_equal(g.nodes['user'].data['h'], F.tensor([1, 1], dtype=idtype))
    assert F.array_equal(g.nodes['game'].data['h'], F.tensor([2, 2, 2], dtype=idtype))
    assert F.array_equal(g.edata['h'], F.tensor([1], dtype=idtype))

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 2, 3, 4], dtype=idtype), ctx=F.ctx())
    g.remove_edges(1, etype='plays')
    assert g.number_of_edges('plays') == 3
    u, v = g.edges(form='uv', order='eid', etype='plays')
    assert F.array_equal(u, F.tensor([0, 1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 1, 1], dtype=idtype))
    assert F.array_equal(g.edges['plays'].data['h'], F.tensor([1, 3, 4], dtype=idtype))
    # remove all edges of 'develops'
    g.remove_edges([0, 1], etype='develops')
    assert g.number_of_edges('develops') == 0
    assert F.array_equal(g.nodes['user'].data['h'], F.tensor([1, 1, 1], dtype=idtype))
    assert F.array_equal(g.nodes['game'].data['h'], F.tensor([2, 2], dtype=idtype))
    assert F.array_equal(g.nodes['developer'].data['h'], F.tensor([3, 3], dtype=idtype))

@parametrize_dtype
def test_remove_nodes(idtype):
    # homogeneous Graphs
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    n = 0
    g.remove_nodes(n)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    n = [1]
    g.remove_nodes(n)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 0
    g = dgl.graph(([0, 1], [1, 2]), idtype=idtype, device=F.ctx())
    n = F.tensor([2], dtype=idtype)
    g.remove_nodes(n)
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))

    # invalid nid
    assert_fail = False
    try:
        g.remove_nodes(3)
    except:
        assert_fail = True
    assert assert_fail

    # has node and edge data
    g = dgl.graph(([0, 0, 2], [0, 1, 2]), idtype=idtype, device=F.ctx())
    g.ndata['hv'] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g.edata['he'] = F.copy_to(F.tensor([1, 2, 3], dtype=idtype), ctx=F.ctx())
    g.remove_nodes(F.tensor([0], dtype=idtype))
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))
    assert F.array_equal(g.ndata['hv'], F.tensor([2, 3], dtype=idtype))
    assert F.array_equal(g.edata['he'], F.tensor([3], dtype=idtype))

    # node id larger than current max node id
    g = dgl.heterograph(
        {('user', 'plays', 'game'): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx())
    n = 0
    g.remove_nodes(n, ntype='user')
    assert g.number_of_nodes('user') == 1
    assert g.number_of_nodes('game') == 3
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([2], dtype=idtype))
    g = dgl.heterograph(
        {('user', 'plays', 'game'): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx())
    n = [1]
    g.remove_nodes(n, ntype='user')
    assert g.number_of_nodes('user') == 1
    assert g.number_of_nodes('game') == 3
    assert g.number_of_edges() == 1
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0], dtype=idtype))
    assert F.array_equal(v, F.tensor([1], dtype=idtype))
    g = dgl.heterograph(
        {('user', 'plays', 'game'): ([0, 1], [1, 2])}, idtype=idtype, device=F.ctx())
    n = F.tensor([0], dtype=idtype)
    g.remove_nodes(n, ntype='game')
    assert g.number_of_nodes('user') == 2
    assert g.number_of_nodes('game') == 2
    assert g.number_of_edges() == 2
    u, v = g.edges(form='uv', order='eid')
    assert F.array_equal(u, F.tensor([0, 1], dtype=idtype))
    assert F.array_equal(v, F.tensor([0 ,1], dtype=idtype))

    # heterogeneous graph
    g = create_test_heterograph3(idtype)
    g.edges['plays'].data['h'] = F.copy_to(F.tensor([1, 2, 3, 4], dtype=idtype), ctx=F.ctx())
    g.remove_nodes(0, ntype='game')
    assert g.number_of_nodes('user') == 3
    assert g.number_of_nodes('game') == 1
    assert g.number_of_nodes('developer') == 2
    assert g.number_of_edges('plays') == 2
    assert g.number_of_edges('develops') == 1
    assert F.array_equal(g.nodes['user'].data['h'], F.tensor([1, 1, 1], dtype=idtype))
    assert F.array_equal(g.nodes['game'].data['h'], F.tensor([2], dtype=idtype))
    assert F.array_equal(g.nodes['developer'].data['h'], F.tensor([3, 3], dtype=idtype))
    u, v = g.edges(form='uv', order='eid', etype='plays')
    assert F.array_equal(u, F.tensor([1, 2], dtype=idtype))
    assert F.array_equal(v, F.tensor([0, 0], dtype=idtype))
    assert F.array_equal(g.edges['plays'].data['h'], F.tensor([3, 4], dtype=idtype))
    u, v = g.edges(form='uv', order='eid', etype='develops')
    assert F.array_equal(u, F.tensor([1], dtype=idtype))
    assert F.array_equal(v, F.tensor([0], dtype=idtype))

@parametrize_dtype
def test_frame(idtype):
    g = dgl.graph(([0, 1, 2], [1, 2, 3]), idtype=idtype, device=F.ctx())
    g.ndata['h'] = F.copy_to(F.tensor([0, 1, 2, 3], dtype=idtype), ctx=F.ctx())
    g.edata['h'] = F.copy_to(F.tensor([0, 1, 2], dtype=idtype), ctx=F.ctx())

    # remove nodes
    sg = dgl.remove_nodes(g, [3])
    # check for lazy update
    assert F.array_equal(sg._node_frames[0]._columns['h'].storage, g.ndata['h'])
    assert F.array_equal(sg._edge_frames[0]._columns['h'].storage, g.edata['h'])
    assert sg.ndata['h'].shape[0] == 3
    assert sg.edata['h'].shape[0] == 2
    # update after read
    assert F.array_equal(sg._node_frames[0]._columns['h'].storage, F.tensor([0, 1, 2], dtype=idtype))
    assert F.array_equal(sg._edge_frames[0]._columns['h'].storage, F.tensor([0, 1], dtype=idtype))

    ng = dgl.add_nodes(sg, 1)
    assert ng.ndata['h'].shape[0] == 4
    assert F.array_equal(ng._node_frames[0]._columns['h'].storage, F.tensor([0, 1, 2, 0], dtype=idtype))
    ng = dgl.add_edges(ng, [3], [1])
    assert ng.edata['h'].shape[0] == 3
    assert F.array_equal(ng._edge_frames[0]._columns['h'].storage, F.tensor([0, 1, 0], dtype=idtype))

    # multi level lazy update
    sg = dgl.remove_nodes(g, [3])
    assert F.array_equal(sg._node_frames[0]._columns['h'].storage, g.ndata['h'])
    assert F.array_equal(sg._edge_frames[0]._columns['h'].storage, g.edata['h'])
    ssg = dgl.remove_nodes(sg, [1])
    assert F.array_equal(ssg._node_frames[0]._columns['h'].storage, g.ndata['h'])
    assert F.array_equal(ssg._edge_frames[0]._columns['h'].storage, g.edata['h'])
    # ssg is changed
    assert ssg.ndata['h'].shape[0] == 2
    assert ssg.edata['h'].shape[0] == 0
    assert F.array_equal(ssg._node_frames[0]._columns['h'].storage, F.tensor([0, 2], dtype=idtype))
    # sg still in lazy model
    assert F.array_equal(sg._node_frames[0]._columns['h'].storage, g.ndata['h'])
    assert F.array_equal(sg._edge_frames[0]._columns['h'].storage, g.edata['h'])

@unittest.skipIf(dgl.backend.backend_name == "tensorflow", reason="TensorFlow always create a new tensor")
@unittest.skipIf(F._default_context_str == 'cpu', reason="cpu do not have context change problem")
@parametrize_dtype
def test_frame_device(idtype):
    g = dgl.graph(([0,1,2], [2,3,1]))
    g.ndata['h'] = F.copy_to(F.tensor([1,1,1,2], dtype=idtype), ctx=F.cpu())
    g.ndata['hh'] = F.copy_to(F.ones((4,3), dtype=idtype), ctx=F.cpu())
    g.edata['h'] = F.copy_to(F.tensor([1,2,3], dtype=idtype), ctx=F.cpu())

    g = g.to(F.ctx())
    # lazy device copy
    assert F.context(g._node_frames[0]._columns['h'].storage) == F.cpu()
    assert F.context(g._node_frames[0]._columns['hh'].storage) == F.cpu()
    print(g.ndata['h'])
    assert F.context(g._node_frames[0]._columns['h'].storage) == F.ctx()
    assert F.context(g._node_frames[0]._columns['hh'].storage) == F.cpu()
    assert F.context(g._edge_frames[0]._columns['h'].storage) == F.cpu()

    # lazy device copy in subgraph
    sg = dgl.node_subgraph(g, [0,1,2])
    assert F.context(sg._node_frames[0]._columns['h'].storage) == F.ctx()
    assert F.context(sg._node_frames[0]._columns['hh'].storage) == F.cpu()
    assert F.context(sg._edge_frames[0]._columns['h'].storage) == F.cpu()
    print(sg.ndata['hh'])
    assert F.context(sg._node_frames[0]._columns['hh'].storage) == F.ctx()
    assert F.context(sg._edge_frames[0]._columns['h'].storage) == F.cpu()

    # back to cpu
    sg = sg.to(F.cpu())
    assert F.context(sg._node_frames[0]._columns['h'].storage) == F.ctx()
    assert F.context(sg._node_frames[0]._columns['hh'].storage) == F.ctx()
    assert F.context(sg._edge_frames[0]._columns['h'].storage) == F.cpu()
    print(sg.ndata['h'])
    print(sg.ndata['hh'])
    print(sg.edata['h'])
    assert F.context(sg._node_frames[0]._columns['h'].storage) == F.cpu()
    assert F.context(sg._node_frames[0]._columns['hh'].storage) == F.cpu()
    assert F.context(sg._edge_frames[0]._columns['h'].storage) == F.cpu()

    # set some field
    sg = sg.to(F.ctx())
    assert F.context(sg._node_frames[0]._columns['h'].storage) == F.cpu()
    sg.ndata['h'][0] = 5
    assert F.context(sg._node_frames[0]._columns['h'].storage) == F.ctx()
    assert F.context(sg._node_frames[0]._columns['hh'].storage) == F.cpu()
    assert F.context(sg._edge_frames[0]._columns['h'].storage) == F.cpu()

    # add nodes
    ng = dgl.add_nodes(sg, 3)
    assert F.context(ng._node_frames[0]._columns['h'].storage) == F.ctx()
    assert F.context(ng._node_frames[0]._columns['hh'].storage) == F.ctx()
    assert F.context(ng._edge_frames[0]._columns['h'].storage) == F.cpu()



if __name__ == '__main__':
    # test_create()
    # test_query()
    # test_hypersparse()
    # test_adj("int32")
    # test_inc()
    # test_view("int32")
    # test_view1("int32")
    # test_flatten(F.int32)
    # test_convert_bound()
    # test_convert()
    # test_to_device("int32")
    # test_transform("int32")
    # test_subgraph("int32")
    # test_subgraph_mask("int32")
    # test_apply()
    # test_level1()
    # test_level2()
    # test_updates()
    # test_backward()
    # test_empty_heterograph('int32')
    # test_types_in_function()
    # test_stack_reduce()
    # test_isolated_ntype()
    # test_bipartite()
    # test_dtype_cast()
    # test_reverse("int32")
    # test_format()
    #test_add_edges(F.int32)
    #test_add_nodes(F.int32)
    #test_remove_edges(F.int32)
    #test_remove_nodes(F.int32)
    #test_clone(F.int32)
    #test_frame(F.int32)
    #test_frame_device(F.int32)
    #test_empty_query(F.int32)
    pass
