"""Implementation for core graph computation."""

from .base import DGLError
from . import function as fn
from .udf import NodeBatch, EdgeBatch
from . import ops

def is_builtin(func):
    return isinstance(func, fn.BuiltinFunction)

def invoke_node_udf(graph, nodes, ntype, func, ndata):
    nbatch = NodeBatch(graph, nodes, ndata, ntype)
    return func(nbatch)

def invoke_udf_reduce(graph, func, edata=None):
    pass

def invoke_edge_udf(graph, func):
    pass

def invoke_gsddmm(graph, func):
    pass

def invoke_gspmm(graph, mfunc, rfunc, *, srcdata=None, dstdata=None, edata=None):
    # sanity check
    if mfunc.out_field != rfunc.msg_field:
        raise DGLError('Invalid message ({}) and reduce ({}) function pairs.'
                       ' The output field of the message function must be equal to the'
                       ' message field of the reduce function.'.format(mfunc, rfunc))
    if edata is None:
        edata = graph.edata
    if srcdata is None:
        srcdata = graph.srcdata
    if dstdata is None:
        dstdata = graph.dstdata
    alldata = [srcdata, dstdata, edata]

    if isinstance(mfunc, fn.BinaryMessageFunction):
        x = alldata[mfunc.lhs][mfunc.lhs_field]
        y = alldata[mfunc.rhs][mfunc.rhs_field]
        op = getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name))
        z = op(graph, x, y)
    else:
        x = alldata[mfunc.target][mfunc.in_field]
        op = getattr(ops, '{}_{}'.format(mfunc.name, rfunc.name))
        z = op(graph, x)
    return {rfunc.out_field : z}

def message_passing(g, mfunc, rfunc, afunc):
    if g.number_of_edges() == 0:
        # No message passing is triggered.
        return
    if is_builtin(mfunc) and is_builtin(rfunc):
        ndata = invoke_gspmm(g, mfunc, rfunc)
    else:
        # message phase
        if is_builtin(mfunc):
            edata = invoke_gsddmm(g, mfunc)
        else:
            edata = invoke_edge_udf(g, mfunc)
        # reduce phase
        if is_builtin(rfunc):
            msg = rfunc.msg_field
            ndata = invoke_gspmm(g, fn.copy_e(msg, msg), rfunc, edata=edata)
        else:
            edata.update(g.edata)  # incorporate original edge features
            ndata = invoke_udf_reduce(g, rfunc, edata=edata)
    # apply phase
    if afunc is not None:
        ndata.update(g.dstdata)  # incorporate original node features
        ndata = invoke_node_udf(g, ALL, afunc, ndata)
    return ndata
