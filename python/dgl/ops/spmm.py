"""dgl spmm operator module."""
import sys

from ..backend import gspmm as gspmm_internal
from .. import backend as F

__all__ = ['gspmm']


def gspmm(g, op, reduce_op, lhs_data, rhs_data):
    r""" Generalized Sparse Matrix Multiplication interface.
    It fuses two steps into one kernel.

    1. Computes messages by :attr:`op` source node and edge features.
    2. Aggregate the messages by :attr:`reduce_op` as the features on destination nodes.

    .. math::
        x_v = \psi_{(u, v, e)\in \mathcal{G}}(\rho(x_u, x_e))

    where :math:`x_v` is the returned feature on destination nodes, and :math:`x_u`,
    :math:`x_e` refers to :attr:`u`, :attr:`e` respectively. :math:`\rho` means binary
    operator :attr:`op` and :math:`\psi` means reduce operator :attr:`reduce_op`,
    :math:`\mathcal{G}` is the graph we apply gspmm on: :attr:`g`.

    Note that this function does not handle gradients.

    Parameters
    ----------
    g : DGLGraph
        The input graph.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``,
        ``copy_lhs``, ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``, ``mean``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by the op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by the op.

    Returns
    -------
    tensor
        The result tensor.
    """
    if op not in ['copy_lhs', 'copy_rhs']:
        # Expand dims so that there will be no broadcasting issues with different
        # number of dimensions. For example, given two shapes (N, 3, 1), (E, 5, 3, 4)
        # that are valid broadcastable shapes, change them to (N, 1, 3, 1) and
        # (E, 5, 3, 4)
        lhs_shape = F.shape(lhs_data)
        rhs_shape = F.shape(rhs_data)
        if len(lhs_shape) != len(rhs_shape):
            max_ndims = max(len(lhs_shape), len(rhs_shape))
            lhs_pad_ndims = max_ndims - len(lhs_shape)
            rhs_pad_ndims = max_ndims - len(rhs_shape)
            new_lhs_shape = (lhs_shape[0],) + (1,) * lhs_pad_ndims + lhs_shape[1:]
            new_rhs_shape = (rhs_shape[0],) + (1,) * rhs_pad_ndims + rhs_shape[1:]
            lhs_data = F.reshape(lhs_data, new_lhs_shape)
            rhs_data = F.reshape(rhs_data, new_rhs_shape)
    # With max and min reducers infinity will be returned for zero degree nodes
    ret = gspmm_internal(g._graph, op,
                         'sum' if reduce_op == 'mean' else reduce_op,
                         lhs_data, rhs_data)

    # Replace infinity with zero for isolated nodes when reducer is min/max
    if reduce_op in ['min', 'max']:
        ret = F.replace_inf_with_zero(ret)

    # divide in degrees for mean reducer.
    if reduce_op == 'mean':
        ret_shape = F.shape(ret)
        deg = g.in_degrees()
        deg = F.astype(F.clamp(deg, 1, g.number_of_edges()), F.dtype(ret))
        deg_shape = (ret_shape[0],) + (1,) * (len(ret_shape) - 1)
        return ret / F.reshape(deg, deg_shape)
    else:
        return ret


def _attach_zerodeg_note(docstring, reducer):
    note1 = """
    The {} function will return zero for nodes with no incoming messages.""".format(reducer)
    note2 = """
    This is implemented by replacing all {} values to zero.
    """.format("infinity" if reducer == "min" else "negative infinity")

    docstring = docstring + note1
    if reducer in ('min', 'max'):
        docstring = docstring + note2
    return docstring


def _gen_spmm_func(binary_op, reduce_op):
    name = "u_{}_e_{}".format(binary_op, reduce_op)
    docstring = """Generalized SpMM function.
    It fuses two steps into one kernel.

    1. Computes messages by {} source node and edge features.
    2. Aggregate the messages by {} as the features on destination nodes.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The source node features.
    y : tensor
        The edge features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient). If the
    feature shape of two input operands do not match, we first broadcasts the features to a unified
    shape (note that the memory usage will not increase accordingly) and then performs the operation.

    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.
    """.format(binary_op, reduce_op)
    docstring = _attach_zerodeg_note(docstring, reduce_op)

    def func(g, x, y):
        return gspmm(g, binary_op, reduce_op, x, y)
    func.__name__ = name
    func.__doc__ = docstring
    return func


def _gen_copy_reduce_func(binary_op, reduce_op):

    name = "{}_{}".format(binary_op, reduce_op)
    binary_str = {
        "copy_u": "It copies node feature to edge as the message.",
        'copy_e': "It regards edge feature as message."
    }
    x_str = {
        "copy_u": "source node",
        "copy_e": "edge"
    }
    docstring = lambda binary_op: _attach_zerodeg_note("""Generalized SpMM function. {}
    Then aggregates the message by {} on destination nodes.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The {} features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """.format(
        binary_str[binary_op],
        reduce_op,
        x_str[binary_op]), reduce_op)

    def func(g, x):
        if binary_op == 'copy_u':
            return gspmm(g, 'copy_lhs', reduce_op, x, None)
        else:
            return gspmm(g, 'copy_rhs', reduce_op, None, x)

    func.__name__ = name
    func.__doc__ = docstring(binary_op)
    return func


def _register_spmm_func():
    """Register spmm functions

    - Binary operation plus reduction between u and e: u_[]_e_[]
    - Copy u plus reduction: copy_u_[]
    - Copy e plus reduction: copy_e_[]
    """
    for binary_op in ["add", "sub", "mul", "div", "copy_u", "copy_e"]:
        for reduce_op in ["sum", "max", "min", "mean"]:
            if binary_op.startswith("copy"):
                func = _gen_copy_reduce_func(binary_op, reduce_op)
            else:
                func = _gen_spmm_func(binary_op, reduce_op)
            setattr(sys.modules[__name__], func.__name__, func)
            __all__.append(func.__name__)


_register_spmm_func()
