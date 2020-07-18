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
from utils import parametrize_dtype

def create_test_heterograph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    plays_spmat = ssp.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 1], [0, 0, 1, 1])))
    wishes_nx = nx.DiGraph()
    wishes_nx.add_nodes_from(['u0', 'u1', 'u2'], bipartite=0)
    wishes_nx.add_nodes_from(['g0', 'g1'], bipartite=1)
    wishes_nx.add_edge('u0', 'g1', id=0)
    wishes_nx.add_edge('u2', 'g0', id=1)

    follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows', idtype=idtype, device=F.ctx())
    plays_g = dgl.bipartite(plays_spmat, 'user', 'plays', 'game', idtype=idtype, device=F.ctx())
    wishes_g = dgl.bipartite(wishes_nx, 'user', 'wishes', 'game', idtype=idtype, device=F.ctx())
    develops_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game', idtype=idtype, device=F.ctx())
    assert follows_g.idtype == idtype
    assert plays_g.idtype == idtype
    assert wishes_g.idtype == idtype
    assert develops_g.idtype == idtype
    g = dgl.hetero_from_relations([follows_g, plays_g, wishes_g, develops_g])
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g

def create_test_heterograph1(idtype):
    edges = []
    edges.extend([(0,1), (1,2)])  # follows
    edges.extend([(0,3), (1,3), (2,4), (1,4)])  # plays
    edges.extend([(0,4), (2,3)])  # wishes
    edges.extend([(5,3), (6,4)])  # develops
    ntypes = F.tensor([0, 0, 0, 1, 1, 2, 2])
    etypes = F.tensor([0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
    g0 = dgl.graph(edges, idtype=idtype, device=F.ctx())
    g0.ndata[dgl.NTYPE] = ntypes
    g0.edata[dgl.ETYPE] = etypes
    return dgl.to_hetero(g0, ['user', 'game', 'developer'], ['follows', 'plays', 'wishes', 'develops'])

def create_test_heterograph2(idtype):
    plays_spmat = ssp.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 1], [0, 0, 1, 1])))
    wishes_nx = nx.DiGraph()
    wishes_nx.add_nodes_from(['u0', 'u1', 'u2'], bipartite=0)
    wishes_nx.add_nodes_from(['g0', 'g1'], bipartite=1)
    wishes_nx.add_edge('u0', 'g1', id=0)
    wishes_nx.add_edge('u2', 'g0', id=1)
    develops_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game')

    g = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): plays_spmat,
        ('user', 'wishes', 'game'): wishes_nx,
        ('developer', 'develops', 'game'): develops_g,
        }, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g

def create_test_heterograph3(idtype):
    device = F.ctx()
    plays_spmat = ssp.coo_matrix(([1, 1, 1, 1], ([0, 1, 2, 1], [0, 0, 1, 1])))
    wishes_nx = nx.DiGraph()
    wishes_nx.add_nodes_from(['u0', 'u1', 'u2'], bipartite=0)
    wishes_nx.add_nodes_from(['g0', 'g1'], bipartite=1)
    wishes_nx.add_edge('u0', 'g1', id=0)
    wishes_nx.add_edge('u2', 'g0', id=1)

    follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows',
            restrict_format='coo', idtype=idtype, device=device)
    plays_g = dgl.bipartite([(0, 0), (1, 0), (2, 1), (1, 1)], 'user', 'plays', 'game',
            restrict_format='coo', idtype=idtype, device=device)
    wishes_g = dgl.bipartite([(0, 1), (2, 0)], 'user', 'wishes', 'game',
            restrict_format='coo', idtype=idtype, device=device)
    develops_g = dgl.bipartite([(0, 0), (1, 1)], 'developer', 'develops', 'game',
            restrict_format='coo', idtype=idtype, device=device)
    g = dgl.hetero_from_relations([follows_g, plays_g, wishes_g, develops_g])
    assert g.idtype == idtype
    assert g.device == device
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

    # create from nx complete bipartite graph
    nxg = nx.complete_bipartite_graph(3, 4)
    g = dgl.bipartite(nxg, 'user', 'plays', 'game', idtype=idtype, device=device)
    assert g.ntypes == ['user', 'game']
    assert g.etypes == ['plays']
    assert g.number_of_edges() == 12
    assert g.idtype == idtype
    assert g.device == device

    # create from scipy
    spmat = ssp.coo_matrix(([1,1,1], ([0, 0, 1], [2, 3, 2])), shape=(4, 4))
    g = dgl.graph(spmat, idtype=idtype, device=device)
    assert g.number_of_nodes() == 4
    assert g.number_of_edges() == 3
    assert g.idtype == idtype
    assert g.device == device

    # test inferring number of nodes for heterograph
    g = dgl.heterograph({
        ('l0', 'e0', 'l1'): [(0, 1), (0, 2)],
        ('l0', 'e1', 'l2'): [(2, 2)],
        ('l2', 'e2', 'l2'): [(1, 1), (3, 3)],
        }, idtype=idtype, device=device)
    assert g.number_of_nodes('l0') == 3
    assert g.number_of_nodes('l1') == 3
    assert g.number_of_nodes('l2') == 4
    assert g.idtype == idtype
    assert g.device == device

    # test if validate flag works
    # homo graph
    with pytest.raises(DGLError):
        g = dgl.graph(
            ([0, 0, 0, 1, 1, 2], [0, 1, 2, 0, 1, 2]),
            num_nodes=2,
            validate=True,
            idtype=idtype, device=device
        )
    # bipartite graph
    def _test_validate_bipartite(card):
        with pytest.raises(DGLError):
            g = dgl.bipartite(
                ([0, 0, 1, 1, 2], [1, 1, 2, 2, 3]),
                num_nodes=card,
                validate=True,
                idtype=idtype, device=device
            )

    _test_validate_bipartite((3, 3))
    _test_validate_bipartite((2, 4))

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
    mg = g.metagraph
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
        assert [g.number_of_nodes(ntype) for ntype in ntypes] == [3, 2, 2]

        # number of edges
        assert [g.number_of_edges(etype) for etype in etypes] == [2, 4, 2, 2]

        # has_node & has_nodes
        for ntype in ntypes:
            n = g.number_of_nodes(ntype)
            for i in range(n):
                assert g.has_node(i, ntype)
            assert not g.has_node(n, ntype)
            assert np.array_equal(
                F.asnumpy(g.has_nodes([0, n], ntype)).astype('int32'), [1, 0])

        assert not g.is_multigraph
        assert g.is_readonly

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
        g = create_test_heterograph3(idtype)
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
        g = create_test_heterograph3(idtype)
        _test(g)

    # test repr
    print(g)

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU does not have COO impl.")
def test_hypersparse():
    N1 = 1 << 50        # should crash if allocated a CSR
    N2 = 1 << 48

    g = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1)],
        ('user', 'plays', 'game'): [(0, N2)]},
        {'user': N1, 'game': N1},
        idtype=F.int64, device=F.ctx())
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

def test_edge_ids():
    N1 = 1 << 50        # should crash if allocated a CSR
    N2 = 1 << 48

    g = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1)],
        ('user', 'plays', 'game'): [(0, N2)]},
        {'user': N1, 'game': N1})
    with pytest.raises(DGLError):
        eid = g.edge_ids(0, 0, etype='follows')

    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (0, 1)],
        ('user', 'plays', 'game'): [(0, N2)]},
        {'user': N1, 'game': N1})

    eid = g2.edge_ids(0, 1, etype='follows')
    assert eid == 0

@parametrize_dtype
def test_adj(idtype):
    g = create_test_heterograph(idtype)
    adj = F.sparse_to_numpy(g.adj(etype='follows'))
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
    adj = F.sparse_to_numpy(g.adj(etype='plays'))
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

    adj = g.adj(scipy_fmt='csr', etype='follows')
    assert np.allclose(
            adj.todense(),
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = g.adj(scipy_fmt='coo', etype='follows')
    assert np.allclose(
            adj.todense(),
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))
    adj = g.adj(scipy_fmt='csr', etype='plays')
    assert np.allclose(
            adj.todense(),
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = g.adj(scipy_fmt='coo', etype='plays')
    assert np.allclose(
            adj.todense(),
            np.array([[1., 1., 0.],
                      [0., 1., 1.]]))
    adj = F.sparse_to_numpy(g['follows'].adj())
    assert np.allclose(
            adj,
            np.array([[0., 0., 0.],
                      [1., 0., 0.],
                      [0., 1., 0.]]))

@parametrize_dtype
def test_inc(idtype):
    g = create_test_heterograph(idtype)
    #follows_g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows')
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
    g = dgl.graph([(0, 1), (1, 2)], 'user', 'follows', idtype=idtype, device=F.ctx())
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
    assert F.array_equal(F.tensor(g.nodes('user')), F.arange(0, 3))
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
    g.ndata['h'] = {'user' : f1,
                    'game' : f2}
    f3 = g.nodes['user'].data['h']
    f4 = g.nodes['game'].data['h']
    assert F.array_equal(f1, f3)
    assert F.array_equal(f2, f4)
    data = g.ndata['h']
    assert F.array_equal(f1, data['user'])
    assert F.array_equal(f2, data['game'])
    # test repr
    print(g.ndata)
    g.ndata.pop('h')
    # test repr
    print(g.ndata)

    f3 = F.randn((2, 4))
    g.edges['user', 'follows', 'user'].data['h'] = f3
    f4 = g.edges['user', 'follows', 'user'].data['h']
    f5 = g.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    assert F.array_equal(f3, f5)
    assert F.array_equal(F.tensor(g.edges(etype='follows', form='eid')), F.arange(0, 2))
    g.edges['follows'].data.pop('h')

    f3 = F.randn((2, 4))
    fail = False
    try:
        g.edata['h'] = f3
    except Exception:
        fail = True
    assert fail
    g.edata['h'] = {('user', 'follows', 'user') : f3}
    f4 = g.edges['user', 'follows', 'user'].data['h']
    f5 = g.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    assert F.array_equal(f3, f5)
    data = g.edata['h']
    assert F.array_equal(f3, data[('user', 'follows', 'user')])
    # test repr
    print(g.edata)
    g.edata.pop('h')
    # test repr
    print(g.edata)

    # test srcdata
    f1 = F.randn((3, 6))
    g.srcnodes['user'].data['h'] = f1       # ok
    f2 = g.srcnodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(F.tensor(g.srcnodes('user')), F.arange(0, 3))
    g.srcnodes['user'].data.pop('h')

    # multi type ndata
    f1 = F.randn((3, 6))
    f2 = F.randn((2, 6))
    fail = False
    try:
        g.srcdata['h'] = f1
    except Exception:
        fail = True
    assert fail
    g.srcdata['h'] = {'user' : f1,
                      'developer' : f2}
    f3 = g.srcnodes['user'].data['h']
    f4 = g.srcnodes['developer'].data['h']
    assert F.array_equal(f1, f3)
    assert F.array_equal(f2, f4)
    data = g.srcdata['h']
    assert F.array_equal(f1, data['user'])
    assert F.array_equal(f2, data['developer'])
    # test repr
    print(g.srcdata)
    g.srcdata.pop('h')

    # test dstdata
    f1 = F.randn((3, 6))
    g.dstnodes['user'].data['h'] = f1       # ok
    f2 = g.dstnodes['user'].data['h']
    assert F.array_equal(f1, f2)
    assert F.array_equal(F.tensor(g.dstnodes('user')), F.arange(0, 3))
    g.dstnodes['user'].data.pop('h')

    # multi type ndata
    f1 = F.randn((3, 6))
    f2 = F.randn((2, 6))
    fail = False
    try:
        g.dstdata['h'] = f1
    except Exception:
        fail = True
    assert fail
    g.dstdata['h'] = {'user' : f1,
                      'game' : f2}
    f3 = g.dstnodes['user'].data['h']
    f4 = g.dstnodes['game'].data['h']
    assert F.array_equal(f1, f3)
    assert F.array_equal(f2, f4)
    data = g.dstdata['h']
    assert F.array_equal(f1, data['user'])
    assert F.array_equal(f2, data['game'])
    # test repr
    print(g.dstdata)
    g.dstdata.pop('h')

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
    assert F.array_equal(F.tensor(g.nodes()), F.arange(0, 3))

    f3 = F.randn((2, 4))
    g.edata['h'] = f3
    f4 = HG.edges['follows'].data['h']
    assert F.array_equal(f3, f4)
    assert F.array_equal(F.tensor(g.edges(form='eid')), F.arange(0, 2))

    # multiple types
    ndata = HG.ndata['h']
    assert isinstance(ndata, dict)
    assert F.array_equal(ndata['user'], f2)
    
    edata = HG.edata['h']
    assert isinstance(edata, dict)
    assert F.array_equal(edata[('user', 'follows', 'user')], f4)

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

    assert F.array_equal(fg.nodes['user'].data['h'], F.ones((3, 5)))
    assert F.array_equal(fg.nodes['game'].data['i'], F.ones((2, 5)))
    assert F.array_equal(fg.edata['e'], F.ones((6, 4)))
    assert 'f' not in fg.edata

    etypes = F.asnumpy(fg.edata[dgl.ETYPE]).tolist()
    eids = F.asnumpy(fg.edata[dgl.EID]).tolist()
    assert set(zip(etypes, eids)) == set([(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1)])

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
    g_x = dgl.graph(([0, 1, 2], [1, 2, 3]), 'user', 'follows', idtype=idtype, device=F.ctx())
    g_y = dgl.graph(([0, 2], [2, 3]), 'user', 'knows', idtype=idtype, device=F.ctx())
    g_x.nodes['user'].data['h'] = F.randn((4, 3))
    g_x.edges['follows'].data['w'] = F.randn((3, 2))
    g_y.nodes['user'].data['hh'] = F.randn((4, 5))
    g_y.edges['knows'].data['ww'] = F.randn((2, 10))
    g = dgl.hetero_from_relations([g_x, g_y])

    assert F.array_equal(g.ndata['h'], g_x.ndata['h'])
    assert F.array_equal(g.ndata['hh'], g_y.ndata['hh'])
    assert F.array_equal(g.edges['follows'].data['w'], g_x.edata['w'])
    assert F.array_equal(g.edges['knows'].data['ww'], g_y.edata['ww'])

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
        assert F.context(g.batch_num_nodes[ntype]) == F.cpu()
    for etype in g.canonical_etypes:
        assert F.context(g.batch_num_edges[etype]) == F.cpu()

    if F.is_cuda_available():
        g1 = g.to(F.cuda())
        assert g1.device == F.cuda()
        assert F.context(g1.nodes['user'].data['h']) == F.cuda()
        assert F.context(g1.nodes['game'].data['i']) == F.cuda()
        assert F.context(g1.edges['plays'].data['e']) == F.cuda()
        for ntype in g1.ntypes:
            assert F.context(g1.batch_num_nodes[ntype]) == F.cuda()
        for etype in g1.canonical_etypes:
            assert F.context(g1.batch_num_edges[etype]) == F.cuda()
        assert F.context(g.nodes['user'].data['h']) == F.cpu()
        assert F.context(g.nodes['game'].data['i']) == F.cpu()
        assert F.context(g.edges['plays'].data['e']) == F.cpu()
        for ntype in g.ntypes:
            assert F.context(g.batch_num_nodes[ntype]) == F.cpu()
        for etype in g.canonical_etypes:
            assert F.context(g.batch_num_edges[etype]) == F.cpu()
        with pytest.raises(DGLError):
            g1.nodes['user'].data['h'] = F.copy_to(F.ones((3, 5)), F.cpu())
        with pytest.raises(DGLError):
            g1.edges['plays'].data['e'] = F.copy_to(F.ones((4, 4)), F.cpu())

@parametrize_dtype
def test_convert_bound(idtype):
    def _test_bipartite_bound(data, card):
        with pytest.raises(DGLError):
            dgl.bipartite(data, num_nodes=card, idtype=idtype, device=F.ctx())

    def _test_graph_bound(data, card):
        with pytest.raises(DGLError):
            dgl.graph(data, num_nodes=card, idtype=idtype, device=F.ctx())

    _test_bipartite_bound(([1,2],[1,2]),(2,3))
    _test_bipartite_bound(([0,1],[1,4]),(2,3))
    _test_graph_bound(([1,3],[1,2]), 3)
    _test_graph_bound(([0,1],[1,3]),3)


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

    g = dgl.to_homo(hg)
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
        hg2 = dgl.to_hetero(
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
    g = dgl.graph([(0, 2), (1, 2), (2, 3), (0, 3)], idtype=idtype, device=F.ctx())
    g.ndata[dgl.NTYPE] = F.tensor([0, 0, 1, 2])
    g.edata[dgl.ETYPE] = F.tensor([0, 0, 1, 2])
    hg = dgl.to_hetero(g, ['l0', 'l1', 'l2'], ['e0', 'e1', 'e2'])
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

    # hetero_from_homo test case 3
    mg = nx.MultiDiGraph([
        ('user', 'movie', 'watches'),
        ('user', 'TV', 'watches')])
    g = dgl.graph([(0, 1), (0, 2)], idtype=idtype, device=F.ctx())
    g.ndata[dgl.NTYPE] = F.tensor([0, 1, 2])
    g.edata[dgl.ETYPE] = F.tensor([0, 0])
    for _mg in [None, mg]:
        hg = dgl.to_hetero(g, ['user', 'TV', 'movie'], ['watches'], metagraph=_mg)
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
    hg = dgl.bipartite([(0, 0), (1, 1)], num_nodes=(2, 3), idtype=idtype, device=F.ctx())
    g = dgl.to_homo(hg)
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
    assert new_g.ntypes == ['user', 'game']
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
                             F.tensor([1, 2], F.int64))
        assert F.array_equal(F.tensor(sg.nodes['game'].data[dgl.NID]),
                             F.tensor([0], F.int64))
        assert F.array_equal(F.tensor(sg.edges['follows'].data[dgl.EID]),
                             F.tensor([1], F.int64))
        assert F.array_equal(F.tensor(sg.edges['plays'].data[dgl.EID]),
                             F.tensor([1], F.int64))
        assert F.array_equal(F.tensor(sg.edges['wishes'].data[dgl.EID]),
                             F.tensor([1], F.int64))
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
                             F.tensor([1, 2], F.int64))
        assert F.array_equal(F.tensor(sg.nodes['game'].data[dgl.NID]),
                             F.tensor([0], F.int64))
        assert F.array_equal(F.tensor(sg.edges['follows'].data[dgl.EID]),
                             F.tensor([1], F.int64))
        assert F.array_equal(F.tensor(sg.edges['plays'].data[dgl.EID]),
                             F.tensor([1], F.int64))
        assert F.array_equal(F.tensor(sg.edges['wishes'].data[dgl.EID]),
                             F.tensor([1], F.int64))
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
                                 F.tensor([1, 2], F.int64))
        else:
            for ntype in sg.ntypes:
                assert g.number_of_nodes(ntype) == sg.number_of_nodes(ntype)

        assert F.array_equal(F.tensor(sg.edges['follows'].data[dgl.EID]),
                             F.tensor([1], F.int64))

        if not preserve_nodes:
            assert F.array_equal(sg.nodes['user'].data['h'], g.nodes['user'].data['h'][1:3])
        assert F.array_equal(sg.edges['follows'].data['h'], g.edges['follows'].data['h'][1:2])

    def _check_subgraph_single_etype(g, sg, preserve_nodes=False):
        assert sg.ntypes == g.ntypes
        assert sg.etypes == g.etypes
        assert sg.canonical_etypes == g.canonical_etypes

        if not preserve_nodes:
            assert F.array_equal(F.tensor(sg.nodes['user'].data[dgl.NID]),
                                 F.tensor([0, 1], F.int64))
            assert F.array_equal(F.tensor(sg.nodes['game'].data[dgl.NID]),
                                 F.tensor([0], F.int64))
        else:
            for ntype in sg.ntypes:
                assert g.number_of_nodes(ntype) == sg.number_of_nodes(ntype)

        assert F.array_equal(F.tensor(sg.edges['plays'].data[dgl.EID]),
                             F.tensor([0, 1], F.int64))

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

    # Test for restricted format
    if F._default_context_str != 'gpu':
        # TODO(minjie): enable this later
        for fmt in ['csr', 'csc', 'coo']:
            g = dgl.graph([(0, 1), (1, 2)], restrict_format=fmt)
            sg = g.subgraph({g.ntypes[0]: [1, 0]})
            nids = F.asnumpy(sg.ndata[dgl.NID])
            assert np.array_equal(nids, np.array([1, 0]))
            src, dst = sg.edges(order='eid')
            src = F.asnumpy(src)
            dst = F.asnumpy(dst)
            assert np.array_equal(src, np.array([1]))
            assert np.array_equal(dst, np.array([0]))

@parametrize_dtype
def test_apply(idtype):
    def node_udf(nodes):
        return {'h': nodes.data['h'] * 2}
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

    # test multi
    g.multi_send_and_recv(
        {'plays' : (g.edges(etype='plays'), mfunc, rfunc),
         ('user', 'wishes', 'game'): (g.edges(etype='wishes'), mfunc, rfunc2)},
        'sum')
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[3., 3.], [3., 3.]]))

    # test multi
    g.multi_send_and_recv(
        {'plays' : (g.edges(etype='plays'), mfunc, rfunc, afunc),
         ('user', 'wishes', 'game'): (g.edges(etype='wishes'), mfunc, rfunc2)},
        'sum', afunc)
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[5., 5.], [5., 5.]]))

    # test cross reducer
    g.nodes['user'].data['h'] = F.randn((3, 2))
    for cred in ['sum', 'max', 'min', 'mean']:
        g.multi_send_and_recv(
            {'plays' : (g.edges(etype='plays'), mfunc, rfunc, afunc),
             'wishes': (g.edges(etype='wishes'), mfunc, rfunc2)},
            cred, afunc)
        y = g.nodes['game'].data['y']
        g['plays'].send_and_recv(g.edges(etype='plays'), mfunc, rfunc, afunc)
        y1 = g.nodes['game'].data['y']
        g['wishes'].send_and_recv(g.edges(etype='wishes'), mfunc, rfunc2)
        y2 = g.nodes['game'].data['y']
        yy = get_redfn(cred)(F.stack([y1, y2], 0), 0)
        yy = yy + 1  # final afunc
        assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    with pytest.raises(DGLError):
        g.multi_send_and_recv(
            {'plays' : (g.edges(etype='plays'), mfunc, rfunc),
             'follows': (g.edges(etype='follows'), mfunc, rfunc2)},
            'sum')

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

    # test multi
    g.multi_pull(
        1,
        {'plays' : (mfunc, rfunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum')
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[0., 0.], [3., 3.]]))

    # test multi
    g.multi_pull(
        1,
        {'plays' : (mfunc, rfunc, afunc),
         ('user', 'wishes', 'game'): (mfunc, rfunc2)},
        'sum', afunc)
    assert F.array_equal(g.nodes['game'].data['y'], F.tensor([[0., 0.], [5., 5.]]))

    # test cross reducer
    g.nodes['user'].data['h'] = F.randn((3, 2))
    for cred in ['sum', 'max', 'min', 'mean']:
        g.multi_pull(
            1,
            {'plays' : (mfunc, rfunc, afunc),
             'wishes': (mfunc, rfunc2)},
            cred, afunc)
        y = g.nodes['game'].data['y']
        g['plays'].pull(1, mfunc, rfunc, afunc)
        y1 = g.nodes['game'].data['y']
        g['wishes'].pull(1, mfunc, rfunc2)
        y2 = g.nodes['game'].data['y']
        g.nodes['game'].data['y'] = get_redfn(cred)(F.stack([y1, y2], 0), 0)
        g.apply_nodes(afunc, 1, ntype='game')
        yy = g.nodes['game'].data['y']
        assert F.array_equal(y, yy)

    # test fail case
    # fail because cannot infer ntype
    with pytest.raises(DGLError):
        g.multi_pull(
            1,
            {'plays' : (mfunc, rfunc),
             'follows': (mfunc, rfunc2)},
            'sum')

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

    # empty edge list
    assert_empty(dgl.heterograph({('user', 'plays', 'game'): []}))
    # empty src-dst pair
    assert_empty(dgl.heterograph({('user', 'plays', 'game'): ([], [])}))
    # empty sparse matrix
    assert_empty(dgl.heterograph({('user', 'plays', 'game'): ssp.coo_matrix((0, 0))}))
    # empty networkx graph
    assert_empty(dgl.heterograph({('user', 'plays', 'game'): nx.DiGraph()}))

    g = dgl.heterograph({('user', 'follows', 'user'): []}, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 0
    assert g.number_of_edges('follows') == 0

    # empty relation graph with others
    g = dgl.heterograph({('user', 'plays', 'game'): [], ('developer', 'develops', 'game'): [
                        (0, 0), (1, 1)]}, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    assert g.number_of_nodes('user') == 0
    assert g.number_of_edges('plays') == 0
    assert g.number_of_nodes('game') == 2
    assert g.number_of_edges('develops') == 2
    assert g.number_of_nodes('developer') == 2


def test_types_in_function():
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

    g = dgl.graph([(0, 1), (1, 2)], 'user', 'follow')
    g.apply_nodes(rfunc1)
    g.apply_edges(mfunc1)
    g.update_all(mfunc1, rfunc1)
    g.send_and_recv([0, 1], mfunc1, rfunc1)
    g.push([0], mfunc1, rfunc1)
    g.pull([1], mfunc1, rfunc1)
    g.filter_nodes(filter_nodes1)
    g.filter_edges(filter_edges1)

    g = dgl.bipartite([(0, 1), (1, 2)], 'user', 'plays', 'game')
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
        ('A', 'AB', 'B'): [(0, 1), (1, 2), (2, 3)]},
        num_nodes_dict={'A': 3, 'B': 4, 'C': 4},
        idtype=idtype, device=F.ctx())
    assert g.number_of_nodes('A') == 3
    assert g.number_of_nodes('B') == 4
    assert g.number_of_nodes('C') == 4

    g = dgl.heterograph({
        ('A', 'AC', 'C'): [(0, 1), (1, 2), (2, 3)]},
        num_nodes_dict={'A': 3, 'B': 4, 'C': 4},
        idtype=idtype, device=F.ctx())
    assert g.number_of_nodes('A') == 3
    assert g.number_of_nodes('B') == 4
    assert g.number_of_nodes('C') == 4

    G = dgl.graph(([0, 1, 2], [4, 5, 6]), num_nodes=11, idtype=idtype, device=F.ctx())
    G.ndata[dgl.NTYPE] = F.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=F.int64)
    G.edata[dgl.ETYPE] = F.tensor([0, 0, 0], dtype=F.int64)
    g = dgl.to_hetero(G, ['A', 'B', 'C'], ['AB'])
    assert g.number_of_nodes('A') == 3
    assert g.number_of_nodes('B') == 4
    assert g.number_of_nodes('C') == 4


@parametrize_dtype
def test_ismultigraph(idtype):
    g1 = dgl.bipartite([(0, 1), (0, 2), (1, 5), (2, 5)], 'A',
                       'AB', 'B', num_nodes=(6, 6), idtype=idtype, device=F.ctx())
    assert g1.is_multigraph == False
    g2 = dgl.bipartite([(0, 1), (0, 1), (0, 2), (1, 5)], 'A',
                       'AC', 'C', num_nodes=(6, 6), idtype=idtype, device=F.ctx())
    assert g2.is_multigraph == True
    g3 = dgl.graph([(0, 1), (1, 2)], 'A', 'AA',
                   num_nodes=6, idtype=idtype, device=F.ctx())
    assert g3.is_multigraph == False
    g4 = dgl.graph([(0, 1), (0, 1), (1, 2)], 'A', 'AA',
                   num_nodes=6, idtype=idtype, device=F.ctx())
    assert g4.is_multigraph == True
    g = dgl.hetero_from_relations([g1, g3])
    assert g.is_multigraph == False
    g = dgl.hetero_from_relations([g1, g2])
    assert g.is_multigraph == True
    g = dgl.hetero_from_relations([g1, g4])
    assert g.is_multigraph == True
    g = dgl.hetero_from_relations([g2, g4])
    assert g.is_multigraph == True

@parametrize_dtype
def test_bipartite(idtype):
    g1 = dgl.bipartite([(0, 1), (0, 2), (1, 5)], 'A', 'AB', 'B', idtype=idtype, device=F.ctx())
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
    g2 = dgl.bipartite([(1, 0), (0, 0)], 'A', 'AC', 'C', idtype=idtype, device=F.ctx())
    g3 = dgl.hetero_from_relations([g1, g2])
    assert g3.is_unibipartite
    assert g3.srctypes == ['A']
    assert set(g3.dsttypes) == {'B', 'C'}
    assert g3.number_of_nodes('A') == 2
    assert g3.number_of_nodes('B') == 6
    assert g3.number_of_nodes('C') == 1
    assert g3.number_of_src_nodes('A') == 2
    assert g3.number_of_src_nodes() == 2
    assert g3.number_of_dst_nodes('B') == 6
    assert g3.number_of_dst_nodes('C') == 1
    g3.srcdata['h'] = F.randn((2, 5))
    assert F.array_equal(g3.srcnodes['A'].data['h'], g3.srcdata['h'])
    assert F.array_equal(g3.nodes['A'].data['h'], g3.srcdata['h'])
    assert F.array_equal(g3.nodes['SRC/A'].data['h'], g3.srcdata['h'])

    g4 = dgl.graph([(0, 0), (1, 1)], 'A', 'AA', idtype=idtype, device=F.ctx())
    g5 = dgl.hetero_from_relations([g1, g2, g4])
    assert not g5.is_unibipartite

@parametrize_dtype
def test_dtype_cast(idtype):
    g = dgl.graph([(0, 0), (1, 1), (0, 1), (2, 0)], idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    g.ndata["feat"] = F.tensor([3, 4, 5])
    g.edata["h"] = F.tensor([3, 4, 5, 6])
    if idtype == "int32":
        g_cast = g.long()
        assert g_cast.idtype == F.int64
    else:
        g_cast = g.int()
        assert g_cast.idtype == F.int32
    assert "feat" in g_cast.ndata
    assert "h" in g_cast.edata
    assert F.array_equal(g.ndata["feat"], g_cast.ndata["feat"])
    assert F.array_equal(g.edata["h"], g_cast.edata["h"])

@parametrize_dtype
def test_format(idtype):
    # single relation
    g = dgl.graph([(0, 0), (1, 1), (0, 1), (2, 0)], restrict_format='coo', idtype=idtype, device=F.ctx())
    assert g.restrict_format() == 'coo'
    assert g.format_in_use() == ['coo']
    try:
        spmat = g.adjacency_matrix(scipy_fmt="csr")
    except:
        print('test passed, graph with restrict_format coo should not create csr matrix.')
    else:
        assert False, 'cannot create csr when restrict_format is coo'
    g1 = g.to_format('any')
    assert g1.restrict_format() == 'any'
    g1.request_format('coo')
    g1.request_format('csr')
    g1.request_format('csc')
    assert len(g1.format_in_use()) == 3
    assert g.restrict_format() == 'coo'
    assert g.format_in_use() == ['coo']

    # multiple relation
    g = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): [(0, 0), (1, 0), (1, 1), (2, 1)],
        ('developer', 'develops', 'game'): [(0, 0), (1, 1)],
        }, restrict_format='csr', idtype=idtype, device=F.ctx())
    user_feat = F.randn((g['follows'].number_of_src_nodes(), 5))
    g['follows'].srcdata['h'] = user_feat
    for rel_type in ['follows', 'plays', 'develops']:
        assert g.restrict_format(rel_type) == 'csr'
        assert g.format_in_use(rel_type) == ['csr']
        try:
            g[rel_type].request_format('coo')
        except:
            print('test passed, graph with restrict_format csr should not create coo matrix')
        else:
            assert False, 'cannot create coo when restrict_format is csr'

    g1 = g.to_format('csc')
    # test frame
    assert F.array_equal(g1['follows'].srcdata['h'], user_feat)
    # test each relation graph
    for rel_type in ['follows', 'plays', 'develops']:
        assert g1.restrict_format(rel_type) == 'csc'
        assert g1.format_in_use(rel_type) == ['csc']
        assert g.restrict_format(rel_type) == 'csr'
        assert g.format_in_use(rel_type) == ['csr']

@parametrize_dtype
def test_edges_order(idtype):
    # (0, 2), (1, 2), (0, 1), (0, 1), (2, 1)
    g = dgl.graph((
        np.array([0, 1, 0, 0, 2]),
        np.array([2, 2, 1, 1, 1])
    ), idtype=idtype, device=F.ctx())

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
    gidx = gidx.to_format('csr')
    gidx = gidx.to_format('any')
    r_gidx = gidx.reverse()
    assert gidx.format_in_use(0)[0] == 'csr'
    assert r_gidx.format_in_use(0)[0] == 'csc'
    assert gidx.number_of_nodes(0) == r_gidx.number_of_nodes(0)
    assert gidx.number_of_edges(0) == r_gidx.number_of_edges(0)
    g_s, g_d, _ = gidx.edges(0)
    rg_s, rg_d, _ = r_gidx.edges(0)
    assert F.array_equal(g_s, rg_d)
    assert F.array_equal(g_d, rg_s)

    # force to start with 'csc'
    gidx = gidx.to_format('csc')
    gidx = gidx.to_format('any')
    r_gidx = gidx.reverse()
    assert gidx.format_in_use(0)[0] == 'csc'
    assert r_gidx.format_in_use(0)[0] == 'csr'
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
    gidx = gidx.to_format('csr')
    gidx = gidx.to_format('any')
    r_gidx = gidx.reverse()
    # three node types and three edge types
    assert gidx.format_in_use(0)[0] == 'csr'
    assert r_gidx.format_in_use(0)[0] == 'csc'
    assert gidx.format_in_use(1)[0] == 'csr'
    assert r_gidx.format_in_use(1)[0] == 'csc'
    assert gidx.format_in_use(2)[0] == 'csr'
    assert r_gidx.format_in_use(2)[0] == 'csc'
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
    gidx = gidx.to_format('csc')
    gidx = gidx.to_format('any')
    r_gidx = gidx.reverse()
    # three node types and three edge types
    assert gidx.format_in_use(0)[0] == 'csc'
    assert r_gidx.format_in_use(0)[0] == 'csr'
    assert gidx.format_in_use(1)[0] == 'csc'
    assert r_gidx.format_in_use(1)[0] == 'csr'
    assert gidx.format_in_use(2)[0] == 'csc'
    assert r_gidx.format_in_use(2)[0] == 'csr'
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


if __name__ == '__main__':
    # test_create()
    # test_query()
    # test_hypersparse()
    # test_adj("int32")
    # test_inc()
    # test_view("int32")
    # test_view1("int32")
    test_flatten(F.int32)
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
    test_reverse("int32")
    test_format()
    pass
