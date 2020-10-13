import jax
from jax import numpy as jnp
from ...base import is_all, ALL
from ...sparse import _gspmm, _gsddmm


__all__ = ['gspmm', 'gsddmm', 'edge_softmax']


def _reduce_grad(grad, shape):
    """Reduce gradient on the broadcast dimension
    If there is broadcast in forward pass, gradients need to be reduced on
    broadcast dimension. This function checks the input tensor shape and
    gradient shape and perform the reduction.

    Parameters
    ----------
    grad: Tensor
        Gradient tensor
    shape: tuple
        Shape of input tensor

    Returns
    -------
    Tensor
    """
    grad_shape = grad.shape[1:]
    in_shape = shape[1:]
    if in_shape == grad_shape:
        # no need to reduce
        return grad
    num_to_squeeze = len(grad_shape) - len(in_shape)
    # pad inshape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = jnp.nonzero(jnp.tensor(grad_shape) - jnp.tensor(in_shape))
    reduce_idx += 1  # skip batch dim
    if len(reduce_idx) > 0:
        grad = grad.sum(dim=tuple(reduce_idx), keepdim=True)
    return grad.view(-1, *shape[1:])


def _need_reduce_last_dim(ufeat, efeat):
    """Indicates whether to reduce the last dimension on edges
    in the backward pass of spmm,
    if so, use dot instead of mul."""
    ushp = ufeat.shape
    eshp = efeat.shape
    return ushp[1:-1] == eshp[1:-1] and eshp[-1] == 1 and ushp[-1] > 1


def _muldiv(op, x):
    return 1. / x if op == 'div' else x


def _addsub(op, x):
    return -x if op == 'sub' else x


def _expand(x, shape):
    return x.expand(-1, *shape)

def gspmm(gidx, op, reduce_op, X, Y):
    out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)
    return out

def gsddmm(gidx, op, X, Y, lhs_target, rhs_target):
    out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)
    return out

def edge_softmax(gidx, logits, eids=ALL, norm_by='dst'):
    if not is_all(eids):
        gidx = gidx.edge_subgraph([eids], True).graph
    if norm_by == 'src':
        gidx = gidx.reverse()
    score_max = _gspmm(gidx, 'copy_rhs', 'max', None, logits)[0]
    score = jnp.exp(_gsddmm(gidx, 'sub', logits, score_max, 'e', 'v'))
    score_sum = _gspmm(gidx, 'copy_rhs', 'sum', None, score)[0]
    out = _gsddmm(gidx, 'div', score, score_sum, 'e', 'v')
    return out
