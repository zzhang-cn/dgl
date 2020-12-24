import pytest

def test_constructing_nn():
    import jax
    from jax import numpy as jnp
    import dgl
    from dgl.nn.jax import GraphConv
    import flax
    g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    g = dgl.add_self_loop(g)
    x = jnp.ones((6, 10))
    _gn = GraphConv.partial(in_feats=10, out_feats=5)
    ys, initial_params = _gn.init(jax.random.PRNGKey(0), g, x)
    gn = flax.nn.Model(_gn, initial_params)
    optimizer = flax.optim.Momentum(
          learning_rate=0.1, beta=0.9).create(gn)

    def loss_fn(gn):
        y = gn(g, x)
        return y.sum()

    d_loss_d_x = jax.grad(loss_fn)(optimizer.target)

    print(d_loss_d_x)
