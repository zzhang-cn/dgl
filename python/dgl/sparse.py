"""Module for sparse matrix operators."""
# pylint: disable= invalid-name
from __future__ import absolute_import
from . import ndarray as nd
from ._ffi.function import _init_api
from .base import DGLError
from . import backend as F


def infer_broadcast_shape(op, shp1, shp2):
    r"""Check the shape validity, and infer the output shape given input shape and operator.
    Note the both :attr:`shp1`, :attr:`shp2` and the returned shape are feature
    shapes (i.e. we remove the first dimension, which correspond to graph statistics
    such as number of nodes, number of edges, etc.).

    We allow applying op on operands with different shapes, according to the
    broadcasting semantics of Numpy/Scipy:
    https://numpy.org/doc/stable/user/basics.broadcasting.html

    Parameters
    ----------
    op : str
        The binary op's name, could be `add`, `sub`, `mul`, `div`, `dot`, `copy_lhs`, `copy_rhs`.
    shp1 : tuple[int]
        The shape of lhs operand.
    shp2 : tuple[int]
        The shape of rhs operand.

    Returns
    -------
    tuple[int]
        shape after broadcasting
    """
    pad_shp1, pad_shp2 = shp1, shp2
    if op == "dot":
        if shp1[-1] != shp2[-1]:
            raise DGLError("Dot operator is only available for arrays with the "
                           "same size on last dimension, but got {} and {}."
                           .format(shp1, shp2))
    if op == "copy_lhs":
        return shp1
    if op == "copy_rhs":
        return shp2
    # operands are padded to have the same dimensionality with leading 1's.
    if len(shp1) > len(shp2):
        pad_shp2 = (1,) * (len(shp1) - len(shp2)) + shp2
    elif len(shp1) < len(shp2):
        pad_shp1 = (1,) * (len(shp2) - len(shp1)) + shp1
    for d1, d2 in zip(pad_shp1, pad_shp2):
        if d1 != d2 and d1 != 1 and d2 != 1:
            raise DGLError("Feature shapes {} and {} are not valid for broadcasting."
                           .format(shp1, shp2))
    rst = tuple(max(d1, d2) for d1, d2 in zip(pad_shp1, pad_shp2))
    return rst[:-1] + (1,) if op == "dot" else rst


def to_dgl_nd(x):
    """Convert framework-specific tensor/None to dgl ndarray."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray(x)


def to_dgl_nd_for_write(x):
    """Convert framework-specific tensor/None to dgl ndarray for write."""
    return nd.NULL['int64'] if x is None else F.zerocopy_to_dgl_ndarray_for_write(x)


target_mapping = {
    'u': 0,
    'e': 1,
    'v': 2,
    'src': 0,
    'edge': 1,
    'dst': 2
}


def _gspmm(gidx, op, reduce_op, u, e):
    r""" Generalized Sparse Matrix Multiplication interface. It takes the result of
    :attr:`op` on source node feature and edge feature, leads to a message on edge.
    Then aggregates the message by :attr:`reduce_op` on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e)\in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``, ``copy_lhs``,
        ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``.
    u : tensor or None
        The feature on source nodes, could be None if op is ``copy_rhs``.
    e : tensor or None
        The feature on edges, could be None if op is ``copy_lhs``.

    Returns
    -------
    tuple
        The returned tuple is composed of two elements:
        - The first element refers to the result tensor.
        - The second element refers to a tuple composed of arg_u and arg_e
          (which is useful when reducer is `min`/`max`).

    Notes
    -----
    This function does not handle gradients.
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gspmm on graph with one edge type")
    use_u = op != 'copy_rhs'
    use_e = op != 'copy_lhs'
    if use_u and use_e:
        if F.dtype(u) != F.dtype(e):
            raise DGLError("The node features' data type {} doesn't match edge"
                           " features' data type {}, please convert them to the"
                           " same type.".format(F.dtype(u), F.dtype(e)))
    # deal with scalar features.
    expand_u, expand_e = False, False
    if use_u:
        if F.ndim(u) == 1:
            u = F.unsqueeze(u, -1)
            expand_u = True
    if use_e:
        if F.ndim(e) == 1:
            e = F.unsqueeze(e, -1)
            expand_e = True

    ctx = F.context(u) if use_u else F.context(e)
    dtype = F.dtype(u) if use_u else F.dtype(e)
    u_shp = F.shape(u) if use_u else (0,)
    e_shp = F.shape(e) if use_e else (0,)
    _, dsttype = gidx.metagraph.find_edge(0)
    v_shp = (gidx.number_of_nodes(dsttype), ) +\
        infer_broadcast_shape(op, u_shp[1:], e_shp[1:])
    v = F.zeros(v_shp, dtype, ctx)
    use_cmp = reduce_op in ['max', 'min']
    arg_u, arg_e = None, None
    idtype = getattr(F, gidx.dtype)
    if use_cmp:
        if use_u:
            arg_u = F.zeros(v_shp, idtype, ctx)
        if use_e:
            arg_e = F.zeros(v_shp, idtype, ctx)
    arg_u_nd = to_dgl_nd_for_write(arg_u)
    arg_e_nd = to_dgl_nd_for_write(arg_e)
    if gidx.number_of_edges(0) > 0:
        _CAPI_DGLKernelSpMM(gidx, op, reduce_op,
                            to_dgl_nd(u if use_u else None),
                            to_dgl_nd(e if use_e else None),
                            to_dgl_nd_for_write(v),
                            arg_u_nd,
                            arg_e_nd)
    # NOTE(zihao): actually we can avoid the following step, because arg_*_nd
    # refers to the data that stores arg_*. After we call _CAPI_DGLKernelSpMM,
    # arg_* should have already been changed. But we found this doesn't work
    # under Tensorflow when index type is int32. (arg_u and arg_e would be
    # all zero).
    # The workaround is proposed by Jinjing, and we still need to investigate
    # where the problem is.
    arg_u = None if arg_u is None else F.zerocopy_from_dgl_ndarray(arg_u_nd)
    arg_e = None if arg_e is None else F.zerocopy_from_dgl_ndarray(arg_e_nd)
    # To deal with scalar node/edge features.
    if (expand_u or not use_u) and (expand_e or not use_e):
        v = F.squeeze(v, -1)
    if expand_u and use_cmp:
        arg_u = F.squeeze(arg_u, -1)
    if expand_e and use_cmp:
        arg_e = F.squeeze(arg_e, -1)
    return v, (arg_u, arg_e)


def _gsddmm(gidx, op, lhs, rhs, lhs_target='u', rhs_target='v'):
    r""" Generalized Sampled-Dense-Dense Matrix Multiplication interface. It
    takes the result of :attr:`op` on source node feature and destination node
    feature, leads to a feature on edge.

    .. math::
        x_{e} = \phi(x_u, x_e, x_v), \forall (u,e,v)\in \mathcal{G}

    where :math:`x_{e}` is the returned feature on edges and :math:`x_u`,
    :math:`x_v` refers to :attr:`u`, :attr:`v` respectively. :math:`\phi`
    is the binary operator :attr:`op`, and :math:`\mathcal{G}` is the graph
    we apply gsddmm on: :attr:`g`.

    Parameters
    ----------
    gidx : HeteroGraphIndex
        The input graph index.
    op : str
        Binary operator, could be ``add``, ``sub``, ``mul``, ``div``, ``dot``,
        ``copy_lhs``, ``copy_rhs``.
    lhs : tensor or None
        Left hand operand.
    rhs : tensor or None
        Right hand operand.
    lhs_target : str
        The target of left hand operand, could be ``src``, ``edge``, ``dst``
        or their alias ``u``, ``e``, ``v``.
    rhs_target : str
        The target of right hand operand, could be ``src``, ``edge``, ``dst``
        or their alias ``u``, ``e``, ``v``.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function does not handle gradients.
    """
    if gidx.number_of_etypes() != 1:
        raise DGLError("We only support gsddmm on graph with one edge type")
    use_lhs = op != 'copy_rhs'
    use_rhs = op != 'copy_lhs'
    if use_lhs and use_rhs:
        if F.dtype(lhs) != F.dtype(rhs):
            raise DGLError("The operands data type don't match: {} and {}, please convert them"
                           " to the same type.".format(F.dtype(lhs), F.dtype(rhs)))
    # deal with scalar features.
    expand_lhs, expand_rhs = False, False
    if use_lhs:
        if F.ndim(lhs) == 1:
            lhs = F.unsqueeze(lhs, -1)
            expand_lhs = True
    if use_rhs:
        if F.ndim(rhs) == 1:
            rhs = F.unsqueeze(rhs, -1)
            expand_rhs = True
    lhs_target = target_mapping[lhs_target]
    rhs_target = target_mapping[rhs_target]
    ctx = F.context(lhs) if use_lhs else F.context(rhs)
    dtype = F.dtype(lhs) if use_lhs else F.dtype(rhs)
    lhs_shp = F.shape(lhs) if use_lhs else (0,)
    rhs_shp = F.shape(rhs) if use_rhs else (0,)
    out_shp = (gidx.number_of_edges(0), ) +\
        infer_broadcast_shape(op, lhs_shp[1:], rhs_shp[1:])
    out = F.zeros(out_shp, dtype, ctx)
    if gidx.number_of_edges(0) > 0:
        _CAPI_DGLKernelSDDMM(gidx, op,
                             to_dgl_nd(lhs if use_lhs else None),
                             to_dgl_nd(rhs if use_rhs else None),
                             to_dgl_nd_for_write(out),
                             lhs_target, rhs_target)
    if (expand_lhs or not use_lhs) and (expand_rhs or not use_rhs):
        out = F.squeeze(out, -1)
    return out


def _segment_reduce(op, feat, offsets):
    r"""Segment reduction operator.

    It aggregates the value tensor along the first dimension by segments.
    The argument ``offsets`` specifies the start offset of each segment (and
    the upper bound of the last segment). Zero-length segments are allowed.

    .. math::
      y_i = \Phi_{j=\mathrm{offsets}_i}^{\mathrm{offsets}_{i+1}-1} x_j

    where :math:`\Phi` is the reduce operator.

    Parameters
    ----------
    op : str
        Aggregation method. Can be ``sum``, ``max``, ``min``.
    x : Tensor
        Value to aggregate.
    offsets : Tensor
        The start offsets of segments.

    Returns
    -------
    tuple(Tensor)
        The first tensor correspond to aggregated tensor of shape
        ``(len(seglen), value.shape[1:])``, and the second tensor records
        the argmin/max at each position for computing gradients.

    Notes
    -----
    This function does not handle gradients.
    """
    n = F.shape(offsets)[0] - 1
    out_shp = (n,) + F.shape(feat)[1:]
    ctx = F.context(feat)
    dtype = F.dtype(feat)
    idtype = F.dtype(offsets)
    out = F.zeros(out_shp, dtype, ctx)
    arg = None
    if op in ['min', 'max']:
        arg = F.zeros(out_shp, idtype, ctx)
    arg_nd = to_dgl_nd_for_write(arg)
    _CAPI_DGLKernelSegmentReduce(op,
                                 to_dgl_nd(feat),
                                 to_dgl_nd(offsets),
                                 to_dgl_nd_for_write(out),
                                 arg_nd)
    arg = None if arg is None else F.zerocopy_from_dgl_ndarray(arg_nd)
    return out, arg


def _scatter_add(x, idx, m):
    r""" Scatter add operator (on first dimension) implementation.

    Math: y[idx[i], *] += x[i, *]

    Parameters
    ----------
    x : Tensor
        The input feature.
    idx : Tensor
        The indices array.
    m : int
        The length of output.

    Returns
    -------
    Tensor
        The output tensor.
    """
    out_shp = (m,) + F.shape(x)[1:]
    ctx = F.context(x)
    dtype = F.dtype(x)
    out = F.zeros(out_shp, dtype, ctx)
    _CAPI_DGLKernelScatterAdd(to_dgl_nd(x),
                              to_dgl_nd(idx),
                              to_dgl_nd_for_write(out))
    return out


def _bwd_segment_cmp(feat, arg, m):
    r""" Backward phase of segment reduction (for 'min'/'max' reduction).

    It computes the gradient of input feature given output gradient of
    the segment reduction result.

    Parameters
    ----------
    feat : Tensor
        The output gradient
    arg : Tensor
        The ArgMin/Max tensor produced by segment_reduce op.
    m : int
        The length of input gradients' first dimension.

    Returns
    -------
    Tensor
        The input gradient.
    """
    out_shp = (m,) + F.shape(feat)[1:]
    ctx = F.context(feat)
    dtype = F.dtype(feat)
    out = F.zeros(out_shp, dtype, ctx)
    _CAPI_DGLKernelBwdSegmentCmp(to_dgl_nd(feat),
                                 to_dgl_nd(arg),
                                 to_dgl_nd_for_write(out))
    return out

def _csrmm(A, A_weights, B, B_weights, num_vtypes):
    """Return a graph whose adjacency matrix is the sparse matrix multiplication
    of those of two given graphs.

    Note that the edge weights of both graphs must be scalar, i.e. :attr:`A_weights`
    and :attr:`B_weights` must be 1D vectors.

    Parameters
    ----------
    A : HeteroGraphIndex
        The input graph index as left operand.
    A_weights : Tensor
        The edge weights of graph A as 1D tensor.
    B : HeteroGraphIndex
        The input graph index as right operand.
    B_weights : Tensor
        The edge weights of graph B as 1D tensor.
    num_vtypes : int
        The number of node types for the returned graph (must be either 1 or 2).

    Returns
    -------
    C : HeteroGraphIndex
        The output graph index.
    C_weights : Tensor
        The edge weights of the output graph.
    """
    C, C_weights = _CAPI_DGLCSRMM(
        A, F.to_dgl_nd(A_weights), B, F.to_dgl_nd(B_weights), num_vtypes)
    return C, F.from_dgl_nd(C_weights)

def _csrsum(As, A_weights):
    """Return a graph whose adjacency matrix is the sparse matrix summation
    of the given list of graphs.

    Note that the edge weights of all graphs must be scalar, i.e. the arrays in
    :attr:`A_weights` must be 1D vectors.

    Parameters
    ----------
    As : list[HeteroGraphIndex]
        The input graph indices.
    A_weights : list[Tensor]
        The edge weights of graph A as 1D tensor.

    Returns
    -------
    C : HeteroGraphIndex
        The output graph index.
    C_weights : Tensor
        The edge weights of the output graph.
    """
    C, C_weights = _CAPI_DGLCSRSum(As, [F.to_dgl_nd(w) for w in A_weights])
    return C, F.from_dgl_nd(C_weights)

def _csrmask(A, A_weights, B):
    """Return the weights of A at the locations identical to the sparsity pattern
    of B.

    If a non-zero entry in B does not exist in A, DGL returns 0 for that location
    instead.

    Note that the edge weights of the graph must be scalar, i.e. :attr:`A_weights`
    must be a 1D vector.

    In scipy notation this is identical to ``A[B != 0]``.

    Parameters
    ----------
    A : HeteroGraphIndex
        The input graph index as left operand.
    A_weights : Tensor
        The edge weights of graph A as 1D tensor.
    B : HeteroGraphIndex
        The input graph index as right operand.

    Returns
    -------
    B_weights : Tensor
        The output weights.
    """
    return F.from_dgl_nd(_CAPI_DGLCSRMask(A, F.to_dgl_nd(A_weights), B))

_init_api("dgl.sparse")
