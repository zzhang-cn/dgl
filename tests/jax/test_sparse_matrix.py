import pytest

def test_to_and_from_dense():
    import dgl
    import jax
    from jax import numpy as jnp
    from dgl.backend.jax.tensor import SparseMatrix2D
    key = jax.random.PRNGKey(2666)
    shape = (10, 5)
    x_dense = jax.random.normal(
        key=key,
        shape=shape,
        dtype=jnp.float32,
    )
    x = SparseMatrix2D.from_dense(x_dense)
    assert (x.to_dense() == x_dense).all()

def test_matmul():
    import dgl
    import jax
    from jax import numpy as jnp
    from dgl.backend.jax.tensor import SparseMatrix2D
    key = jax.random.PRNGKey(2666)
    x = jax.random.normal(
        key=key,
        shape=(10, 5),
        dtype=jnp.float32,
    )
    y = jax.random.normal(
        key=key,
        shape=(5, 4, 3, 2),
        dtype=jnp.float32,
    )
    x_sparse = SparseMatrix2D.from_dense(x)
    import numpy as onp

    assert onp.allclose(
        x_sparse @ y,
        jnp.einsum(
            'ab, bcde->acde',
            x, y
        )
    )
