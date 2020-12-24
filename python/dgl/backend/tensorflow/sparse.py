import tensorflow as tf
import numpy as np
from .tensor import tensor, copy_to, context, asnumpy, zerocopy_from_numpy
from ...base import is_all, ALL
from ...sparse import _gspmm, _gsddmm, _segment_reduce, _bwd_segment_cmp

__all__ = ['gspmm', 'gsddmm', 'edge_softmax', 'segment_reduce']


def _scatter_nd(index, src, n_rows):
    assert index.shape == src.shape
    shp = index.shape
    ctx = context(src)
    ndim = index.ndim
    offsets = []
    stride = 1
    for i in reversed(range(1, ndim)):
        di = shp[i]
        offset_i = tf.range(di, dtype=index.dtype)
        offsets.append(
            tf.reshape((stride * offset_i), (1,) * i + (di,) + (1,) * (ndim - 1 - i)))
        stride *= di
    if ndim > 1:
        new_idx = index * stride + copy_to(sum(offsets), ctx)
    else:
        new_idx = index
    src = tf.reshape(src, (-1,))
    new_idx = tf.reshape(new_idx, (-1, 1))
    rst = tf.reshape(tf.scatter_nd(new_idx, src, (stride * n_rows,)), (n_rows, *shp[1:]))
    return rst


def _gather_nd(index, src):
    shp = index.shape
    ctx = context(src)
    ndim = index.ndim
    offsets = []
    stride = 1
    for i in reversed(range(1, ndim)):
        di = shp[i]
        offset_i = tf.range(di, dtype=index.dtype)
        offsets.append(
            tf.reshape((stride * offset_i), (1,) * i + (di,) + (1,) * (ndim - 1 - i)))
        stride *= di
    if ndim > 1:
        new_idx = index * stride + copy_to(sum(offsets), ctx)
    else:
        new_idx = index
    src = tf.reshape(src, (-1,))
    new_idx = tf.reshape(new_idx, (-1))
    rst = tf.reshape(tf.gather(src, new_idx), shp)
    return rst


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
    reduce_idx = np.asarray(np.nonzero(np.asarray(grad_shape) - np.asarray(in_shape)))
    reduce_idx += 1  # skip batch dim
    reduce_idx_tensor = tf.constant(tuple(
        reduce_idx.flatten().tolist()), dtype=tf.int32)
    grad = tf.reduce_sum(grad, axis=reduce_idx_tensor, keepdims=True)
    return tf.reshape(grad, shape)


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
    return tf.broadcast_to(x, (x.shape[0], *shape))


def gspmm_real(gidx, op, reduce_op, X, Y):
    out, (argX, argY) = _gspmm(gidx, op, reduce_op, X, Y)

    def grad(dZ):
        dZ = tensor(dZ)
        if op != 'copy_rhs':
            g_rev = gidx.reverse()
            if reduce_op == 'sum':
                if op in ['mul', 'div']:
                    dX = _gspmm(g_rev, 'mul', 'sum', dZ, _muldiv(op, Y))[0]
                elif op in ['add', 'sub']:
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, Y)[0]
                elif op == 'copy_lhs':
                    dX = _gspmm(g_rev, 'copy_lhs', 'sum', dZ, None)[0]
            else:
                if op in ['mul', 'div']:
                    dX = _scatter_nd(
                        argX,
                        _muldiv(op, _gather_nd(argY, _expand(Y, dZ.shape[1:]))) * dZ,
                        X.shape[0])
                elif op in ['add', 'sub', 'copy_lhs']:
                    dX = _scatter_nd(argX, dZ, X.shape[0])
            dX = _reduce_grad(dX, X.shape)
        else:
            dX = tf.zeros_like(X)
        if op != 'copy_lhs':
            if reduce_op == 'sum':
                if op == 'mul' and _need_reduce_last_dim(X, Y):
                    dY = _gsddmm(gidx, 'dot', X, dZ)
                elif op in ['mul', 'div']:
                    dY = _gsddmm(gidx, 'mul', X, dZ)
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _gsddmm(gidx, 'copy_rhs', X, _addsub(op, dZ))
            else:
                out_shp = (Y.shape[0],) + dZ.shape[1:]
                if op in ['mul',  'div']:
                    dY = _scatter_nd(
                        argY,
                        _gather_nd(argX, _expand(X, dZ.shape[1:])) * dZ,
                        Y.shape[0])
                    if op == 'div': dY = -dY / (Y ** 2)
                elif op in ['add', 'sub', 'copy_rhs']:
                    dY = _scatter_nd(argY, _addsub(op, dZ), Y.shape[0])
            dY = _reduce_grad(dY, Y.shape)
        else:
            dY = tf.zeros_like(Y)
        return dX, dY
    return out, grad


def gspmm(gidx, op, reduce_op, X, Y):
    @tf.custom_gradient
    def _lambda(X, Y):
        return gspmm_real(gidx, op, reduce_op, X, Y)
    if X is None:
        X = tf.zeros(())
    if Y is None:
        Y = tf.zeros(())
    return _lambda(X, Y)


def gsddmm_real(gidx, op, X, Y, lhs_target, rhs_target):
    out = _gsddmm(gidx, op, X, Y, lhs_target, rhs_target)

    def grad(dZ):
        if op != 'copy_rhs':
            if lhs_target in ['u', 'v']:
                _gidx = gidx if lhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0]
                else:  # mul, div, dot
                    if rhs_target == lhs_target:
                        dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0] * _muldiv(op, Y)
                    elif rhs_target == 'e':
                        dX = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * _muldiv(op, Y))[0]
                    else:  # rhs_target = !lhs_target
                        dX = _gspmm(_gidx, 'mul', 'sum', _muldiv(op, Y), dZ)[0]
            else:  # lhs_target == 'e'
                if op in ['add', 'sub', 'copy_lhs']:
                    dX = dZ
                else:  # mul, div, dot
                    dX = _gsddmm(gidx, 'mul', dZ, _muldiv(op, Y), 'e', rhs_target)
            dX = _reduce_grad(dX, X.shape)
        else:
            dX = tf.zeros_like(X)
        if op != 'copy_lhs':
            if rhs_target in ['u', 'v']:
                _gidx = gidx if rhs_target == 'v' else gidx.reverse()
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, _addsub(op, dZ))[0]
                else:  # mul, div, dot
                    if lhs_target == rhs_target:
                        dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ)[0] * X
                    elif lhs_target == 'e':
                        dY = _gspmm(_gidx, 'copy_rhs', 'sum', None, dZ * X)[0]
                    else:  # rhs_target = !lhs_target
                        dY = _gspmm(_gidx, 'mul', 'sum', X, dZ)[0]
                    if op == 'div':
                        dY = -dY / (Y ** 2)
            else:
                if op in ['add', 'sub', 'copy_rhs']:
                    dY = _addsub(op, dZ)
                else:  # mul, div, dot
                    dY = _gsddmm(gidx, 'mul', dZ, X, 'e', lhs_target)
                    if op == 'div':
                        dY = -dY / (Y ** 2)
            dY = _reduce_grad(dY, Y.shape)
        else:
            dY = tf.zeros_like(Y)
        return dX, dY
    return out, grad


def gsddmm(gidx, op, X, Y, lhs_target='u', rhs_target='v'):
    @tf.custom_gradient
    def _lambda(X, Y):
        return gsddmm_real(gidx, op, X, Y, lhs_target, rhs_target)
    if X is None:
        X = tf.zeros(())
    if Y is None:
        Y = tf.zeros(())
    return _lambda(X, Y)


def edge_softmax_real(gidx, score, eids=ALL, norm_by='dst'):
    if not is_all(eids):
        gidx = gidx.edge_subgraph([eids], True).graph
    if norm_by == 'src':
        gidx = gidx.reverse()
    score_max = _gspmm(gidx, 'copy_rhs', 'max', None, score)[0]
    score = tf.math.exp(_gsddmm(gidx, 'sub', score, score_max, 'e', 'v'))
    score_sum = _gspmm(gidx, 'copy_rhs', 'sum', None, score)[0]
    out = _gsddmm(gidx, 'div', score, score_sum, 'e', 'v')

    def edge_softmax_backward(grad_out):
        sds = out * grad_out
        accum = gspmm(gidx, 'copy_rhs', 'sum', None, sds)
        grad_score = sds - gsddmm(gidx, 'mul', out, accum, 'e', 'v')
        return grad_score

    return out, edge_softmax_backward


def edge_softmax(gidx, logits, eids=ALL, norm_by='dst'):
    @tf.custom_gradient
    def _lambda(logits):
        return edge_softmax_real(gidx, logits, eids, norm_by)
    return _lambda(logits)
