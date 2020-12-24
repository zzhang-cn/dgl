import pytest
import backend as F
import dgl
D = 5
def generate_graph(idtype=F.int32, grad=False):
    '''
    s, d, eid
    0, 1, 0
    1, 9, 1
    0, 2, 2
    2, 9, 3
    0, 3, 4
    3, 9, 5
    0, 4, 6
    4, 9, 7
    0, 5, 8
    5, 9, 9
    0, 6, 10
    6, 9, 11
    0, 7, 12
    7, 9, 13
    0, 8, 14
    8, 9, 15
    9, 0, 16
    '''
    u = F.tensor([0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 9])
    v = F.tensor([1, 9, 2, 9, 3, 9, 4, 9, 5, 9, 6, 9, 7, 9, 8, 9, 0])
    g = dgl.graph((u, v), idtype=idtype)
    assert g.device == F.ctx()
    ncol = F.randn((10, D))
    ecol = F.randn((17, D))
    if grad:
        ncol = F.attach_grad(ncol)
        ecol = F.attach_grad(ecol)

    g.ndata['h'] = ncol
    g.edata['w'] = ecol
    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)
    return g

def test_batch_setter_autograd(idtype=F.int32):
    g = generate_graph(idtype, grad=True)
    h1 = g.ndata['h']
    # partial set
    v = F.tensor([1, 2, 8], g.idtype)
    hh = F.zeros((len(v), D))
    print(hh)

    def f(hh, g=g):
        g.nodes[v].data['h'] = hh
        h2 = g.ndata['h']
        print(g.ndata)
        return h2

    import jax
    _, dy_dhh = jax.vjp(f, hh)

    assert F.array_equal(dy_dhh(F.ones((10, D)) * 2)[0][:,0], F.tensor([2., 2., 2.]))
