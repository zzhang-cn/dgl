"""Checking and logging utilities."""
# pylint: disable=invalid-name
from __future__ import absolute_import, division

from ..base import DGLError
from .._ffi.function import _init_api
from .. import backend as F

def prepare_tensor(g, data, name):
    """Convert the data to ID tensor and check its ID type and context.

    If the data is already in tensor type, raise error if its ID type
    and context does not match the graph's.
    Otherwise, convert it to tensor type of the graph's ID type and
    ctx and return.

    Parameters
    ----------
    g : DGLHeteroGraph
        Graph.
    data : int, iterable of int, tensor
        Data.
    name : str
        Name of the data.

    Returns
    -------
    Tensor
        Data in tensor object.
    """
    if F.is_tensor(data):
        if F.dtype(data) != g.idtype or F.context(data) != g.device:
            raise DGLError('Expect argument "{}" to have data type {} and device '
                           'context {}. But got {} and {}.'.format(
                               name, g.idtype, g.device, F.dtype(data), F.context(data)))
        ret = data
    else:
        data = F.tensor(data)
        if (not (F.ndim(data) > 0 and F.shape(data)[0] == 0) and        # empty tensor
                F.dtype(data) not in (F.int32, F.int64)):
            raise DGLError('Expect argument "{}" to have data type int32 or int64,'
                           ' but got {}.'.format(name, F.dtype(data)))
        ret = F.copy_to(F.astype(data, g.idtype), g.device)

    if F.ndim(ret) == 0:
        ret = F.unsqueeze(ret, 0)
    if F.ndim(ret) > 1:
        raise DGLError('Expect a 1-D tensor for argument "{}". But got {}.'.format(
            name, ret))
    return ret

def prepare_tensor_dict(g, data, name):
    """Convert a dictionary of data to a dictionary of ID tensors.

    If calls ``prepare_tensor`` on each key-value pair.

    Parameters
    ----------
    g : DGLHeteroGraph
        Graph.
    data : dict[str, (int, iterable of int, tensor)]
        Data dict.
    name : str
        Name of the data.

    Returns
    -------
    dict[str, tensor]
    """
    return {key : prepare_tensor(g, val, '{}["{}"]'.format(name, key))
            for key, val in data.items()}

def parse_edges_arg_to_eid(g, edges, etid, argname='edges'):
    """Parse the :attr:`edges` argument and return an edge ID tensor.

    The resulting edge ID tensor has the same ID type and device of :attr:`g`.

    Parameters
    ----------
    g : DGLGraph
        Graph
    edges : pair of Tensor, Tensor, iterable[int]
        Argument for specifying edges.
    etid : int
        Edge type ID.
    argname : str, optional
        Argument name.

    Returns
    -------
    Tensor
        Edge ID tensor
    """
    if isinstance(edges, tuple):
        u, v = edges
        u = prepare_tensor(g, u, '{}[0]'.format(argname))
        v = prepare_tensor(g, v, '{}[1]'.format(argname))
        eid = g.edge_ids(u, v, etype=g.canonical_etypes[etid])
    else:
        eid = prepare_tensor(g, edges, argname)
    return eid

def check_all_same_idtype(glist, name):
    """Check all the graphs have the same idtype."""
    if len(glist) == 0:
        return
    idtype = glist[0].idtype
    for i, g in enumerate(glist):
        if g.idtype != idtype:
            raise DGLError('Expect {}[{}] to have {} type ID, but got {}.'.format(
                name, i, idtype, g.idtype))

def check_all_same_device(glist, name):
    """Check all the graphs have the same device."""
    if len(glist) == 0:
        return
    device = glist[0].device
    for i, g in enumerate(glist):
        if g.device != device:
            raise DGLError('Expect {}[{}] to be on device {}, but got {}.'.format(
                name, i, device, g.device))

def check_all_same_schema(schemas, name):
    """Check the list of schemas are the same."""
    if len(schemas) == 0:
        return

    for i, schema in enumerate(schemas):
        if schema != schemas[0]:
            raise DGLError(
                'Expect all graphs to have the same schema on {}, '
                'but graph {} got\n\t{}\nwhich is different from\n\t{}.'.format(
                    name, i, schema, schemas[0]))

def check_all_same_schema_for_keys(schemas, keys, name):
    """Check the list of schemas are the same on the given keys."""
    if len(schemas) == 0:
        return

    head = None
    keys = set(keys)
    for i, schema in enumerate(schemas):
        if not keys.issubset(schema.keys()):
            raise DGLError(
                'Expect all graphs to have keys {} on {}, '
                'but graph {} got keys {}.'.format(
                    keys, name, i, schema.keys()))

        if head is None:
            head = {k: schema[k] for k in keys}
        else:
            target = {k: schema[k] for k in keys}
            if target != head:
                raise DGLError(
                    'Expect all graphs to have the same schema for keys {} on {}, '
                    'but graph {} got \n\t{}\n which is different from\n\t{}.'.format(
                        keys, name, i, target, head))

def check_valid_idtype(idtype):
    """Check whether the value of the idtype argument is valid (int32/int64)

    Parameters
    ----------
    idtype : data type
        The framework object of a data type.
    """
    if idtype not in [None, F.int32, F.int64]:
        raise DGLError('Expect idtype to be a framework object of int32/int64, '
                       'got {}'.format(idtype))

def is_sorted_srcdst(src, dst, num_src=None, num_dst=None):
    """Checks whether an edge list is in ascending src-major order (e.g., first
    sorted by ``src`` and then by ``dst``).

    Parameters
    ----------
    src : IdArray
        The tensor of source nodes for each edge.
    dst : IdArray
        The tensor of destination nodes for each edge.
    num_src : int, optional
        The number of source nodes.
    num_dst : int, optional
        The number of destination nodes.

    Returns
    -------
    bool, bool
        Whether ``src`` is in ascending order, and whether ``dst`` is
        in ascending order with respect to ``src``.
    """
    # for some versions of MXNET and TensorFlow, num_src and num_dst get
    # incorrectly marked as floats, so force them as integers here
    if num_src is None:
        num_src = int(F.as_scalar(F.max(src, dim=0)+1))
    if num_dst is None:
        num_dst = int(F.as_scalar(F.max(dst, dim=0)+1))

    src = F.zerocopy_to_dgl_ndarray(src)
    dst = F.zerocopy_to_dgl_ndarray(dst)
    sorted_status = _CAPI_DGLCOOIsSorted(src, dst, num_src, num_dst)

    row_sorted = sorted_status > 0
    col_sorted = sorted_status > 1

    return row_sorted, col_sorted

_init_api("dgl.utils.checks")
