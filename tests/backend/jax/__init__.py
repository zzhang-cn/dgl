from __future__ import absolute_import

import jax
from jax import numpy as jnp

def cuda():
    try:
        return jax.devices("gpu")[0]
    except:
        return None

def is_cuda_available():
    from jax.lib import xla_bridge
    return 'gpu' in str(xla_bridge.get_backend().platform)

def array_equal(a, b):
    if a is None or b is None:
        return False
    return (a == b).all()

def allclose(a, b, rtol=1e-4, atol=1e-4):
    return jnp.allclose(
        a, b, rtol=rtol, atol=atol
    )

def randn(shape):
    key = jax.random.PRNGKey(2666)
    return jax.random.normal(
        key=key,
        shape=shape,
        dtype=jnp.float32, # this is ridiculous
    )

def full(shape, fill_value, dtype, ctx):
    # TODO: not sure about device yet
    return jnp.full(
        shape=shape,
        fill_value=fill_value,
        dtype=dtype,
    )

def narrow_row_set(x, start, stop, new):
    x[start:stop] = new

def sparse_to_numpy(x):
    return x.to_dense()

def clone(x):
    return jnp.asarray(x)

def reduce_sum(x):
    return jnp.sum(x)

def softmax(x, dim):
    return jax.nn.softmax(x, axis=dim)

def spmm(x, y):
    return x @ y

def add(a, b):
    return a + b

def sub(a, b):
    return a - b

def mul(a, b):
    return a * b

def div(a, b):
    return a / b

def sum(x, dim, keepdims=False):
    return x.sum(dim, keepdims=keepdims)

def max(x, dim):
    return x.max(dim)

def min(x, dim):
    return x.min(dim)

def prod(x, dim):
    return x.prod(dim)

def matmul(a, b):
    return a @ b

def dot(a, b):
    return sum(mul(a, b), dim=-1)

def abs(a):
    return a.abs()
