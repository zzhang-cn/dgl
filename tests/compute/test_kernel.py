import dgl
import dgl.function as fn
import networkx as nx
import numpy as np
import backend as F
from itertools import product
from test_utils import parametrize_dtype, get_cases
import pytest

def udf_copy_src(edges):
    return {'m': edges.src['u']}

def udf_copy_edge(edges):
    return {'m': edges.data['e']}

def udf_mean(nodes):
    return {'r2': F.mean(nodes.mailbox['m'], 1)}

def udf_sum(nodes):
    return {'r2': F.sum(nodes.mailbox['m'], 1)}

def udf_max(nodes):
    return {'r2': F.max(nodes.mailbox['m'], 1)}


D1 = 5
D2 = 3
D3 = 4
D4 = 10 # NOTE(xiang): used to dot feature vector
builtin = {'sum': fn.sum, 'max': fn.max, 'mean': fn.mean}
udf_reduce = {'sum': udf_sum, 'max': udf_max, 'mean': udf_mean}
fill_value = {'sum': 0, 'max': float("-inf")}


def generate_feature(g, broadcast='none', binary_op='none'):
    """Create graph with src, edge, dst feature. broadcast can be 'u',
    'e', 'v', 'none'
    """
    np.random.seed(31)
    nv = g.number_of_nodes()
    ne = g.number_of_edges()
    if binary_op == 'dot':
        if broadcast == 'e':
            u = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3, D4)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D2, 1, D4)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3, D4)))
        elif broadcast == 'u':
            u = F.tensor(np.random.uniform(-1, 1, (nv, D2, 1, D4)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D1, D2, D3, D4)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3, D4)))
        elif broadcast == 'v':
            u = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3, D4)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D1, D2, D3, D4)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D2, 1, D4)))
        else:
            u = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3, D4)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D1, D2, D3, D4)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3, D4)))
    else:
        if broadcast == 'e':
            u = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D2, 1)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3)))
        elif broadcast == 'u':
            u = F.tensor(np.random.uniform(-1, 1, (nv, D2, 1)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D1, D2, D3)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3)))
        elif broadcast == 'v':
            u = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D1, D2, D3)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D2, 1)))
        else:
            u = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3)))
            e = F.tensor(np.random.uniform(-1, 1, (ne, D1, D2, D3)))
            v = F.tensor(np.random.uniform(-1, 1, (nv, D1, D2, D3)))
    return F.astype(u, F.float32), F.astype(v, F.float32), F.astype(e, F.float32)


def test_copy_src_reduce():
    def _test(red, partial):
        g = dgl.DGLGraph(nx.erdos_renyi_graph(100, 0.1))
        # NOTE(zihao): add self-loop to avoid zero-degree nodes.
        # https://github.com/dmlc/dgl/issues/761
        g.add_edges(g.nodes(), g.nodes())
        g = g.to(F.ctx())
        hu, hv, he = generate_feature(g, 'none', 'none')
        if partial:
            nid = F.tensor(list(range(0, 100, 2)), g.idtype)

        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            if partial:
                g.pull(nid, fn.copy_src(src='u', out='m'),
                       builtin[red](msg='m', out='r1'))
            else:
                g.update_all(fn.copy_src(src='u', out='m'),
                             builtin[red](msg='m', out='r1'))
            r1 = g.ndata['r1']
            F.backward(F.reduce_sum(r1))
            n_grad1 = F.grad(g.ndata['u'])

        # reset grad
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            if partial:
                g.pull(nid, udf_copy_src, udf_reduce[red])
            else:
                g.update_all(udf_copy_src, udf_reduce[red])
            r2 = g.ndata['r2']
            F.backward(F.reduce_sum(r2))
            n_grad2 = F.grad(g.ndata['u'])

        def _print_error(a, b):
            print("ERROR: Test copy_src_{} partial: {}".
                  format(red, partial))
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y):
                    print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)

        if F.backend_name != "jax":
            if not F.allclose(n_grad1, n_grad2):
                print('node grad')
                _print_error(n_grad1, n_grad2)
            assert(F.allclose(n_grad1, n_grad2))

    _test('sum', False)
    _test('max', False)
    _test('mean', False)
    _test('sum', True)
    _test('max', True)
    _test('mean', True)


def test_copy_edge_reduce():
    def _test(red, partial):
        g = dgl.DGLGraph(nx.erdos_renyi_graph(100, 0.1))
        # NOTE(zihao): add self-loop to avoid zero-degree nodes.
        g.add_edges(g.nodes(), g.nodes())
        g = g.to(F.ctx())
        hu, hv, he = generate_feature(g, 'none', 'none')
        if partial:
            nid = F.tensor(list(range(0, 100, 2)), g.idtype)

        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            if partial:
                g.pull(nid, fn.copy_edge(edge='e', out='m'),
                       builtin[red](msg='m', out='r1'))
            else:
                g.update_all(fn.copy_edge(edge='e', out='m'),
                             builtin[red](msg='m', out='r1'))
            r1 = g.ndata['r1']
            F.backward(F.reduce_sum(r1))
            e_grad1 = F.grad(g.edata['e'])

        # reset grad
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        with F.record_grad():
            if partial:
                g.pull(nid, udf_copy_edge, udf_reduce[red])
            else:
                g.update_all(udf_copy_edge, udf_reduce[red])
            r2 = g.ndata['r2']
            F.backward(F.reduce_sum(r2))
            e_grad2 = F.grad(g.edata['e'])

        def _print_error(a, b):
            print("ERROR: Test copy_edge_{} partial: {}".
                  format(red, partial))
            return
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y):
                    print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)

        if F.backend_name != "jax":
            if not F.allclose(e_grad1, e_grad2):
                print('edge gradient')
                _print_error(e_grad1, e_grad2)
            assert(F.allclose(e_grad1, e_grad2))

    _test('sum', False)
    _test('max', False)
    _test('mean', False)
    _test('sum', True)
    _test('max', True)
    _test('mean', True)


def test_all_binary_builtins():
    def _test(g, lhs, rhs, binary_op, reducer, partial, nid, broadcast='none'):
        # initialize node/edge features with uniform(-1, 1)
        hu, hv, he = generate_feature(g, broadcast, binary_op)
        if binary_op == 'div':
            # op = div
            # lhs range: [-1, 1]
            # rhs range: [1, 2]
            # result range: [-1, 1]
            if rhs == 'u':
                hu = (hu + 3) / 2
            elif rhs == 'v':
                hv = (hv + 3) / 2
            elif rhs == 'e':
                he = (he + 3) / 2

        if binary_op == 'add' or binary_op == 'sub':
            # op = add, sub
            # lhs range: [-1/2, 1/2]
            # rhs range: [-1/2, 1/2]
            # result range: [-1, 1]
            hu = hu / 2
            hv = hv / 2
            he = he / 2

        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        builtin_msg_name = "{}_{}_{}".format(lhs, binary_op, rhs)
        builtin_msg = getattr(fn, builtin_msg_name)
        builtin_red = getattr(fn, reducer)

        def target_feature_switch(g, target):
            if target == "u":
                return g.ndata["u"]
            elif target == "v":
                return g.ndata["v"]
            else:
                return g.edata["e"]

        with F.record_grad():
            if partial:
                g.pull(nid, builtin_msg(lhs, rhs, 'm'), builtin_red('m', 'r1'))
            else:
                g.update_all(builtin_msg(lhs, rhs, 'm'), builtin_red('m', 'r1'))
            r1 = g.ndata.pop('r1')
            F.backward(F.reduce_sum(r1))
            lhs_grad_1 = F.grad(target_feature_switch(g, lhs))
            rhs_grad_1 = F.grad(target_feature_switch(g, rhs))

        # reset grad
        g.ndata['u'] = F.attach_grad(F.clone(hu))
        g.ndata['v'] = F.attach_grad(F.clone(hv))
        g.edata['e'] = F.attach_grad(F.clone(he))

        def target_switch(edges, target):
            if target == "u":
                return edges.src
            elif target == "v":
                return edges.dst
            elif target == "e":
                return edges.data
            else:
                assert(0), "Unknown target {}".format(target)

        def mfunc(edges):
            op = getattr(F, binary_op)
            lhs_data = target_switch(edges, lhs)[lhs]
            rhs_data = target_switch(edges, rhs)[rhs]
            # NOTE(zihao): we need to do batched broadcast
            # e.g. (68, 3, 1) op (68, 5, 3, 4)
            while F.ndim(lhs_data) < F.ndim(rhs_data):
                lhs_data = F.unsqueeze(lhs_data, 1)
            while F.ndim(rhs_data) < F.ndim(lhs_data):
                rhs_data = F.unsqueeze(rhs_data, 1)
            return {"m": op(lhs_data, rhs_data)}

        def rfunc(nodes):
            op = getattr(F, reducer)
            return {"r2": op(nodes.mailbox['m'], 1)}

        with F.record_grad():
            if partial:
                g.pull(nid, mfunc, rfunc)
            else:
                g.update_all(mfunc, rfunc)
            r2 = g.ndata.pop('r2')
            F.backward(F.reduce_sum(r2), F.tensor([1.]))
            lhs_grad_2 = F.grad(target_feature_switch(g, lhs))
            rhs_grad_2 = F.grad(target_feature_switch(g, rhs))

        rtol = 1e-4
        atol = 1e-4

        def _print_error(a, b):
            print("ERROR: Test {}_{}_{}_{} broadcast: {} partial: {}".
                  format(lhs, binary_op, rhs, reducer, broadcast, partial))
            return
            if lhs == 'u':
                lhs_data = hu
            elif lhs == 'v':
                lhs_data = hv
            elif lhs == 'e':
                lhs_data = he

            if rhs == 'u':
                rhs_data = hu
            elif rhs == 'v':
                rhs_data = hv
            elif rhs == 'e':
                rhs_data = he
            print("lhs", F.asnumpy(lhs_data).tolist())
            print("rhs", F.asnumpy(rhs_data).tolist())
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y, rtol, atol):
                    print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2, rtol, atol):
            _print_error(r1, r2)
        assert F.allclose(r1, r2, rtol, atol)

        if F.backend_name != "jax":
            if not F.allclose(lhs_grad_1, lhs_grad_2, rtol, atol):
                print("left grad")
                _print_error(lhs_grad_1, lhs_grad_2)
            assert(F.allclose(lhs_grad_1, lhs_grad_2, rtol, atol))

            if not F.allclose(rhs_grad_1, rhs_grad_2, rtol, atol):
                print("right grad")
                _print_error(rhs_grad_1, rhs_grad_2)
            assert(F.allclose(rhs_grad_1, rhs_grad_2, rtol, atol))

    g = dgl.DGLGraph()
    g.add_nodes(20)
    # NOTE(zihao): add self-loop to avoid zero-degree nodes.
    g.add_edges(g.nodes(), g.nodes())
    for i in range(2, 18):
        g.add_edge(0, i)
        g.add_edge(1, i)
        g.add_edge(i, 18)
        g.add_edge(i, 19)
    g.add_edge(18, 0)
    g.add_edge(18, 1)
    g.add_edge(19, 0)
    g.add_edge(19, 1)
    g = g.to(F.ctx())
    nid = F.tensor([0, 1, 4, 5, 7, 12, 14, 15, 18, 19], g.idtype)
    target = ["u", "v", "e"]

    for lhs, rhs in product(target, target):
        if lhs == rhs:
            continue
        for binary_op in ["add", "sub", "mul", "div"]:
            for reducer in ["sum", "max", "min", "mean"]:
                for broadcast in ["none", lhs, rhs]:
                    for partial in [False, True]:
                        print(lhs, rhs, binary_op, reducer, broadcast, partial)
                        _test(g, lhs, rhs, binary_op, reducer, partial, nid,
                              broadcast=broadcast)

@parametrize_dtype
@pytest.mark.parametrize('g', get_cases(['homo-zero-degree']))
def test_mean_zero_degree(g, idtype):
    g = g.astype(idtype).to(F.ctx())
    g.ndata['h'] = F.ones((g.number_of_nodes(), 3))
    g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'x'))
    deg = F.asnumpy(g.in_degrees())
    v = F.tensor(np.where(deg == 0)[0])
    assert F.allclose(F.gather_row(g.ndata['x'], v), F.zeros((len(v), 3)))

if __name__ == '__main__':
    test_copy_src_reduce()
    test_copy_edge_reduce()
    test_all_binary_builtins()
