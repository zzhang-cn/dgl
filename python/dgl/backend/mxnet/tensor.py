from __future__ import absolute_import

from distutils.version import LooseVersion

import os
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
import numbers
import builtins
from ... import ndarray as dglnd
from ..._deprecate import kernel as K
from ...function.base import TargetCode

MX_VERSION = LooseVersion(mx.__version__)
if MX_VERSION.version[0] == 1 and MX_VERSION.version[1] < 5:
    raise RuntimeError("DGL requires mxnet >= 1.5")

# After MXNet 1.5, empty tensors aren't supprted by default.
# After we turn on the numpy compatible flag, MXNet supports empty NDArray.
mx.set_np_shape(bool(os.environ.get('DGL_MXNET_SET_NP_SHAPE', True)))

def data_type_dict():
    return {'float16' : np.float16,
            'float32' : np.float32,
            'float64' : np.float64,
            'uint8'   : np.uint8,
            'int8'    : np.int8,
            'int16'   : np.int16,
            'int32'   : np.int32,
            'int64'   : np.int64,
            'bool'    : np.bool}  # mxnet does not support bool

def cpu():
    return mx.cpu()

def tensor(data, dtype=None):
    if dtype == np.bool:
        # mxnet doesn't support bool
        dtype =  np.int32
    if isinstance(data, nd.NDArray):
        if dtype is None or data.dtype == dtype:
            return data
        else:
            return data.astype(dtype)
    else:
        if isinstance(data, numbers.Number):
            data = [data]
        if dtype is None:
            if isinstance(data, np.ndarray):
                dtype = np.int32 if data.dtype == np.bool else data.dtype
            elif len(data) == 0:
                dtype = np.int64
            else:
                dtype = np.int64 if isinstance(data[0], numbers.Integral) else np.float32
        return nd.array(data, dtype=dtype)

def as_scalar(data):
    if data.size != 1:
        raise ValueError("The current array is not a scalar")
    if data.shape != (1,):
        data = data.expand_dims(axis=0)
    return data.asscalar()

def get_preferred_sparse_format():
    """Get the preferred sparse matrix format supported by the backend.

    Different backends have their preferred backend. This info is useful when
    constructing a sparse matrix.
    """
    return "csr"

def sparse_matrix(data, index, shape, force_format=False):
    fmt = index[0]
    if fmt == 'coo':
        if force_format:
            raise TypeError('MXNet backend only supports CSR format,'
                            ' but COO format is forced.')
        coord = index[1]
        # generate convert idx
        # FIXME: cannot use int64
        tmp_data = nd.arange(len(coord[0]), dtype=data.dtype, ctx=coord[0].context)
        tmp_spmat = nd.sparse.csr_matrix((tmp_data, (coord[0], coord[1])),
                tuple(shape), ctx=data.context)
        convert_idx = nd.cast(tmp_spmat.data, dtype='int64')
        # shuffle the data
        data = data[convert_idx]
        spmat = nd.sparse.csr_matrix((data, tmp_spmat.indices, tmp_spmat.indptr),
                tuple(shape), ctx=data.context)
        return spmat, convert_idx
    elif fmt == 'csr':
        indices = index[1]
        indptr = index[2]
        spmat = nd.sparse.csr_matrix((data, indices, indptr),
                tuple(shape), ctx=data.context)
        # No conversion is required.
        return spmat, None
    else:
        raise TypeError('Invalid format: %s.' % fmt)

def sparse_matrix_indices(spmat):
    return ('csr', spmat.indices, spmat.indptr)

def is_tensor(obj):
    return isinstance(obj, nd.NDArray)

def shape(input):
    # NOTE: the input cannot be a symbol
    return input.shape

def dtype(input):
    # NOTE: the input cannot be a symbol
    return input.dtype

def ndim(input):
    return input.ndim

def context(input):
    return input.context

def device_type(ctx):
    return ctx.device_type

def device_id(ctx):
    return ctx.device_id

def to_backend_ctx(dglctx):
    dev_type = dglctx.device_type
    if dev_type == 1:
        return mx.cpu()
    elif dev_type == 2:
        return mx.gpu(dglctx.device_id)
    else:
        raise ValueError('Unsupported DGL device context:', dglctx)

def astype(input, ty):
    if ty == np.bool:
        ty = np.int32
    return input.astype(ty)

def asnumpy(input):
    return input.asnumpy()

def copy_to(input, ctx, **kwargs):
    return input.as_in_context(ctx)

def sum(input, dim, keepdims=False):
    if len(input) == 0:
        return nd.array([0.], dtype=input.dtype, ctx=input.context)
    return nd.sum(input, axis=dim, keepdims=keepdims)

def reduce_sum(input):
    return input.sum()

def cumsum(input, dim):
    return nd.cumsum(input, axis=dim)

def mean(input, dim):
    return nd.mean(input, axis=dim)

def reduce_mean(input):
    return input.mean()

def max(input, dim):
    return nd.max(input, axis=dim)

def reduce_max(input):
    return input.max()

def min(input, dim):
    return nd.min(input, axis=dim)

def reduce_min(input):
    return input.min()

def topk(input, k, dim, descending=True):
    return nd.topk(input, axis=dim, k=k, ret_typ='value', is_ascend=not descending)

def argtopk(input, k, dim, descending=True):
    idx = nd.argsort(input, dim, is_ascend=not descending)
    return nd.slice_axis(input, dim, 0, k)

def argsort(input, dim, descending):
    idx = nd.argsort(input, dim, is_ascend=not descending)
    idx = nd.cast(idx, dtype='int64')
    return idx

def exp(input):
    return nd.exp(input)

def sqrt(input):
    return nd.sqrt(input)

def softmax(input, dim=-1):
    return nd.softmax(input, axis=dim)

def cat(seq, dim):
    return nd.concat(*seq, dim=dim)

def stack(seq, dim):
    return nd.stack(*seq, axis=dim)

def split(x, sizes_or_sections, dim):
    if isinstance(sizes_or_sections, list) and len(sizes_or_sections) == 1:
        assert len(x) == sizes_or_sections[0]
        return [x]

    if MX_VERSION.version[0] == 1 and MX_VERSION.version[1] >= 5:
        if isinstance(sizes_or_sections, (np.ndarray, list)):
            sizes_or_sections1 = tuple(np.cumsum(sizes_or_sections)[:-1])
        return nd.split_v2(x, sizes_or_sections1, axis=dim)

    if isinstance(sizes_or_sections, list) or isinstance(sizes_or_sections, np.ndarray):
        # Old MXNet doesn't support split with different section sizes.
        np_arr = x.asnumpy()
        indices = np.cumsum(sizes_or_sections)[:-1]
        res = np.split(np_arr, indices, axis=dim)
        return [tensor(arr, dtype=x.dtype) for arr in res]
    else:
        return nd.split(x, sizes_or_sections, axis=dim)

def repeat(input, repeats, dim):
    if isinstance(repeats, nd.NDArray):
        return nd.array(np.repeat(input.asnumpy(), repeats.asnumpy(), axis=dim),
                        ctx=input.context, dtype=input.dtype)
    else:
        return nd.repeat(input, repeats, axis=dim)

def gather_row(data, row_index):
    # MXNet workaround for empty row index
    if len(row_index) == 0:
        if data.shape[0] == 0:
            return data
        else:
            return data[0:0]

    if isinstance(row_index, nd.NDArray):
        return nd.take(data, row_index)
    else:
        return data[row_index,]

def slice_axis(data, axis, begin, end):
    dim = data.shape[axis]
    if begin < 0:
        begin += dim
    if end <= 0:
        end += dim
    return nd.slice_axis(data, axis, begin, end)

def take(data, indices, dim):
    return nd.take(data, indices, dim)

def narrow_row(data, start, stop):
    return data[start:stop]

def index_add_inplace(data, row_idx, value):
    raise NotImplementedError("MXNet doesn't support inplace index_add")

def scatter_row(data, row_index, value):
    return mx.nd.contrib.index_copy(data, row_index, value)

def scatter_row_inplace(data, row_index, value):
    data[row_index] = value

def squeeze(input, dim):
    return nd.squeeze(input, axis=dim)

def unsqueeze(input, dim):
    return nd.expand_dims(input, axis=dim)

def reshape(input, shape):
    # NOTE: the input cannot be a symbol
    return nd.reshape(input ,shape)

def swapaxes(input, axis1, axis2):
    return nd.swapaxes(input, axis1, axis2)

def zeros(shape, dtype, ctx):
    return nd.zeros(shape, dtype=dtype, ctx=ctx)

def zeros_like(input):
    return nd.zeros_like(input)

def ones(shape, dtype, ctx):
    return nd.ones(shape, dtype=dtype, ctx=ctx)

def uniform(shape, dtype, ctx, low, high):
    return nd.random.uniform(low, high, ctx=ctx, dtype=dtype, shape=shape)

def randint(shape, dtype, ctx, low, high):
    return nd.random.randint(low, high, ctx=ctx, dtype=dtype, shape=shape)

def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = input.shape
    if isinstance(lengths, nd.NDArray):
        lengths = list(lengths.asnumpy())
    max_len = builtins.max(lengths)

    if l_min is not None:
        max_len = builtins.max(max_len, l_min)

    batch_size = len(lengths)
    ctx = input.context
    dtype = input.dtype
    x = nd.full((batch_size * max_len, *old_shape[1:]), value, ctx=ctx, dtype=dtype)
    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = nd.array(index, ctx=ctx)
    return scatter_row(x, index, input).reshape(batch_size, max_len, *old_shape[1:])

def pack_padded_tensor(input, lengths):
    batch_size, max_len = input.shape[:2]
    ctx = input.context
    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = nd.array(index, ctx=ctx)
    return gather_row(input.reshape(batch_size * max_len, -1), index)

def boolean_mask(input, mask):
    return mx.contrib.nd.boolean_mask(input, mask)

def equal(x, y):
    return x == y

def logical_not(input):
    return nd.logical_not(input)

def logical_and(input1, input2):
    return nd.logical_and(input1, input2)

def clone(input):
    return input.copy()

def clamp(data, min_val, max_val):
    return nd.clip(data, min_val, max_val)

def replace_inf_with_zero(x):
    return nd.where(nd.abs(x) == np.inf, nd.zeros_like(x), x)

def unique(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.unique(tmp)
    return nd.array(tmp, ctx=input.context, dtype=input.dtype)

def full_1d(length, fill_value, dtype, ctx):
    return nd.full((length,), fill_value, dtype=dtype, ctx=ctx)

def nonzero_1d(input):
    # TODO: fallback to numpy is unfortunate
    tmp = input.asnumpy()
    tmp = np.nonzero(tmp)[0]
    r = nd.array(tmp, ctx=input.context, dtype=tmp.dtype)
    return r

def sort_1d(input):
    # TODO: this isn't an ideal implementation.
    val = nd.sort(input, axis=None, is_ascend=True)
    idx = nd.argsort(input, is_ascend=True)
    idx = nd.cast(idx, dtype='int64')
    return val, idx

def arange(start, stop, dtype=np.int64, ctx=None):
    if start >= stop:
        return nd.array([], dtype=dtype, ctx=ctx)
    else:
        return nd.arange(start, stop, dtype=dtype, ctx=ctx)

def rand_shuffle(arr):
    return mx.nd.random.shuffle(arr)

def zerocopy_to_dlpack(arr):
    return arr.to_dlpack_for_read()

def zerocopy_from_dlpack(dlpack_arr):
    return nd.from_dlpack(dlpack_arr)

def zerocopy_to_numpy(arr):
    # NOTE: not zerocopy
    return arr.asnumpy()

def zerocopy_from_numpy(np_data):
    return mx.nd.from_numpy(np_data, zero_copy=True)

def zerocopy_to_dgl_ndarray(arr):
    arr.to_dlpack_for_read()
    return dglnd.from_dlpack(arr.to_dlpack_for_read())

def zerocopy_to_dgl_ndarray_for_write(arr):
    return dglnd.from_dlpack(arr.to_dlpack_for_write())

def zerocopy_from_dgl_ndarray(arr):
    return nd.from_dlpack(arr.to_dlpack())


class BinaryReduce(mx.autograd.Function):
    def __init__(self, reducer, binary_op, graph, lhs, rhs, out_size, lhs_map,
                 rhs_map, out_map):
        super(BinaryReduce, self).__init__()
        self.reducer = reducer
        self.binary_op = binary_op
        self.graph = graph
        self.lhs = lhs
        self.rhs = rhs
        self.out_size = out_size
        self.lhs_map = lhs_map
        self.rhs_map = rhs_map
        self.out_map = out_map

    def forward(self, lhs_data, rhs_data):
        lhs_data_nd = zerocopy_to_dgl_ndarray(lhs_data)
        rhs_data_nd = zerocopy_to_dgl_ndarray(rhs_data)
        feat_shape = K.infer_binary_feature_shape(self.binary_op, lhs_data_nd, rhs_data_nd)
        out_shape = feat_shape
        if self.binary_op == 'dot':
            out_shape = feat_shape[:-1]
        out_data = nd.empty((self.out_size,) + out_shape,
                            ctx=lhs_data.context, dtype=lhs_data.dtype)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        K.binary_op_reduce(
            self.reducer if self.reducer != 'mean' else 'sum',
            self.binary_op, self.graph, self.lhs, self.rhs,
            lhs_data_nd, rhs_data_nd, out_data_nd, self.lhs_map[0],
            self.rhs_map[0], self.out_map[0])
        # normalize if mean reducer
        # NOTE(zihao): this is a temporary hack and we should have better solution in the future.
        if self.reducer == 'mean':
            degs = nd.empty((out_data.shape[0],),
                            ctx=out_data.context, dtype=out_data.dtype)
            degs_nd = zerocopy_to_dgl_ndarray(degs)
            if self.lhs != TargetCode.DST:
                target = self.lhs
                n = lhs_data.shape[0]
                in_map = self.lhs_map[0]
            else:
                target = self.rhs
                n = rhs_data.shape[0]
                in_map = self.rhs_map[0]
            in_ones = nd.ones((n,), ctx=lhs_data.context, dtype=lhs_data.dtype)
            in_ones_nd = zerocopy_to_dgl_ndarray(in_ones)
            K.copy_reduce(
                'sum', self.graph, target, in_ones_nd, degs_nd,
                in_map, self.out_map[0])
            # reshape
            degs = degs.reshape((out_data.shape[0],) + (1,) * (out_data.ndim - 1)).clip(1, float('inf'))
            out_data = out_data / degs
        else:
            degs = None
        self.save_for_backward(lhs_data_nd, rhs_data_nd, out_data_nd,
                               feat_shape, degs)
        return out_data

    def backward(self, grad_out):
        lhs_data_nd, rhs_data_nd, out_data_nd, feat_shape, degs = self.saved_tensors
        if self.reducer == 'mean':
            grad_out = grad_out / degs
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        grad_lhs = nd.empty((lhs_data_nd.shape[0],) + feat_shape,
                            ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_lhs_binary_op_reduce(
            self.reducer if self.reducer != 'mean' else 'sum',
            self.binary_op, self.graph, self.lhs, self.rhs,
            lhs_data_nd, rhs_data_nd, out_data_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_lhs), self.lhs_map[1],
            self.rhs_map[1], self.out_map[1])
        grad_lhs = _reduce_grad(grad_lhs, lhs_data_nd.shape)
        grad_rhs = nd.empty((rhs_data_nd.shape[0],) + feat_shape,
                             ctx=grad_out.context, dtype=grad_out.dtype)
        K.backward_rhs_binary_op_reduce(
            self.reducer if self.reducer != 'mean' else 'sum',
            self.binary_op, self.graph, self.lhs, self.rhs,
            lhs_data_nd, rhs_data_nd, out_data_nd, grad_out_nd,
            zerocopy_to_dgl_ndarray_for_write(grad_rhs), self.lhs_map[1],
            self.rhs_map[1], self.out_map[1])
        grad_rhs = _reduce_grad(grad_rhs, rhs_data_nd.shape)
        # clear saved tensors explicitly
        self.saved_tensors = None
        return grad_lhs, grad_rhs


def binary_reduce(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                  out_size, lhs_map=(None, None), rhs_map=(None, None), out_map=(None, None)):
    func = BinaryReduce(reducer, binary_op, graph, lhs, rhs, out_size, lhs_map,
                        rhs_map, out_map)
    return func(lhs_data, rhs_data)


class CopyReduce(mx.autograd.Function):
    def __init__(self, reducer, graph, target, out_size, in_map, out_map):
        super(CopyReduce, self).__init__()
        self.reducer = reducer
        self.graph = graph
        self.target = target
        self.out_size = out_size
        self.in_map = in_map
        self.out_map = out_map

    def forward(self, in_data):
        feat_shape = in_data.shape[1:]
        out_data = nd.empty((self.out_size,) + feat_shape,
                            ctx=in_data.context, dtype=in_data.dtype)
        in_data_nd = zerocopy_to_dgl_ndarray(in_data)
        out_data_nd = zerocopy_to_dgl_ndarray_for_write(out_data)
        K.copy_reduce(
            self.reducer if self.reducer != 'mean' else 'sum',
            self.graph, self.target, in_data_nd, out_data_nd,
            self.in_map[0], self.out_map[0])
        # normalize if mean reducer
        # NOTE(zihao): this is a temporary hack and we should have better solution in the future.
        if self.reducer == 'mean':
            in_ones = nd.ones((in_data.shape[0],),
                              ctx=in_data.context, dtype=in_data.dtype)
            degs = nd.empty((out_data.shape[0],),
                            ctx=out_data.context, dtype=out_data.dtype)
            in_ones_nd = zerocopy_to_dgl_ndarray(in_ones)
            degs_nd = zerocopy_to_dgl_ndarray(degs)
            K.copy_reduce(
                'sum', self.graph, self.target, in_ones_nd, degs_nd,
                self.in_map[0], self.out_map[0])
            # reshape
            degs = degs.reshape((out_data.shape[0],) + (1,) * (out_data.ndim - 1)).clip(1, float('inf'))
            out_data = out_data / degs
        else:
            degs = None
        self.save_for_backward(in_data_nd, out_data_nd, degs)
        return out_data

    def backward(self, grad_out):
        in_data_nd, out_data_nd, degs = self.saved_tensors
        grad_in = nd.empty(in_data_nd.shape, ctx=grad_out.context,
                            dtype=grad_out.dtype)
        if self.reducer == 'mean':
            grad_out = grad_out / degs
        grad_out_nd = zerocopy_to_dgl_ndarray(grad_out)
        K.backward_copy_reduce(
            self.reducer if self.reducer != 'mean' else 'sum',
            self.graph, self.target, in_data_nd, out_data_nd,
            grad_out_nd, zerocopy_to_dgl_ndarray_for_write(grad_in),
            self.in_map[1], self.out_map[1])
        # clear saved tensors explicitly
        self.saved_tensors = None
        return grad_in


def copy_reduce(reducer, graph, target, in_data, out_size, in_map=(None, None),
                out_map=(None, None)):
    func = CopyReduce(reducer, graph, target, out_size, in_map, out_map)
    return func(in_data)


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
    # pad in_shape
    in_shape = (1,) * num_to_squeeze + in_shape
    reduce_idx = np.nonzero(np.asarray(grad_shape) - np.asarray(in_shape))[0]
    reduce_idx += 1  # skip batch dim
    grad = grad.sum(axis=tuple(reduce_idx), keepdims=True)
    return grad.reshape(shape)

def sync():
    """Synchronize computation.

    In DL frameworks such as MXNet and TensorFlow, the computation in operators
    are done asynchronously. This is to synchronize computation and makes sure
    that all computation is complete after this function call.
    """
    mx.nd.waitall()

def attach_grad(tensor):
    tensor.attach_grad()
    return tensor

def backward(x, head_gradient=None):
    x.backward(head_gradient)

def grad(x):
    return x.grad

def is_no_grad(x):
    return (x != 0).sum() == 0

def is_recording():
    return mx.autograd.is_recording()

record_grad = mx.autograd.record

class no_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass
