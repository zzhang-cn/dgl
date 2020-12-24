from __future__ import absolute_import

import abc
import jax
from jax.config import config
config.update("jax_enable_x64", True)
from jax import numpy as jnp
from jax import xla
import numpy as onp
from jax import dlpack

import builtins
import numbers
# from torch.utils import dlpack


from ... import ndarray as nd
from ..._deprecate import kernel as K
from ...function.base import TargetCode
from ...base import dgl_warning

from functools import partial

def data_type_dict():
    return {'float16' : jnp.float16,
            'float32' : jnp.float32,
            'float64' : jnp.float64,
            'uint8'   : jnp.uint8,
            'int8'    : jnp.int8,
            'int16'   : jnp.int16,
            'int32'   : jnp.int32,
            'int64'   : jnp.int64,
            'bool'    : jnp.bool_}

def cpu():
    # TODO:
    # figure out whether multiple cpu is needed at any point
    return jax.devices('cpu')[0]

def tensor(data, dtype=None):
    if isinstance(data, numbers.Number):
        data = [data]
    if not is_tensor(data):
        data = jnp.array(data, dtype=dtype)
    # data.device_buffer.block_host_until_ready()
    return data

def as_scalar(data):
    return data.squeeze()


class SparseMatrix(abc.ABC):
    """ Base class for sparse matrix. """
    def __init__(self):
        super(SparseMatrix, self).__init__()
        # implement more general sparse matrix

# =============================================================================
# MODULE CLASS
# =============================================================================
class SparseMatrix2D(SparseMatrix):
    """ Two-dimensional sparse matrix. """
    def __init__(self, index=None, data=None, shape=None):
        super(SparseMatrix2D, self).__init__()
        self.index = jnp.atleast_2d(index)
        self.data = jnp.asarray(data)
        self._shape = shape

    def __hash__(self):
        return hash((self.index, self.data, self.shape))

    def to_dense(self):
        dense = jnp.zeros(self.shape, self.dtype)
        return dense.at[tuple(self.index)].add(self.data)

    @classmethod
    def from_dense(cls, x):
        x = jnp.asarray(x)
        nz = (x != 0)
        return cls(jnp.where(nz), x[nz], x.shape)

    @property
    def nnz(self):
        return self.data.shape[0]

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @staticmethod
    @partial(jax.jit, static_argnums=(3,))
    def _matmul(x, index, data, num_segments):
        # (n_entries, )
        rows = index[0, :]
        cols = index[1, :]

        # (n_entries, x_shape[1])
        in_ = x.take(cols, axis=0)

        # data: shape=(n_entries)
        prod = in_ * jax.lax.broadcast(data, in_.shape[1:][::-1]).transpose(
            list(range(len(in_.shape)))[::-1]
        )

        return jax.ops.segment_sum(prod, rows, num_segments)

    def __matmul__(self, x):
        if not self.ndim == 2:
            raise NotImplementedError

        assert self.shape[1] == x.shape[0]

        return self._matmul(x, self.index, self.data, self.shape[0])

    def __repr__(self):
        return 'Sparse Matrix with shape=%s, indices=%s, and data=%s' % (
            self.shape,
            self.index,
            self.data,
        )

    def __eq__(self, x):
        if not isinstance(x, type(self)):
            return False
        else:
            return self.to_dense() == x.to_dense()

def _flatten_SparseMatrix2D(x):
    return ((x.index, x.data, x.shape), None)

def _unflatten_SparseMatrix2D(aux_data, children):
    return SparseMatrix2D(*children)

jax.tree_util.register_pytree_node(
    SparseMatrix2D,
    _flatten_SparseMatrix2D,
    _unflatten_SparseMatrix2D
)

def get_preferred_sparse_format():
    return "coo"

def sparse_matrix(data, index, shape, force_format=False):
    if not len(shape) == 2:
        raise NotImplementedError

    fmt = index[0]
    if fmt != 'coo':
        raise TypeError(
            'JAX backend only supports COO format. But got %s.' % fmt
        )
    # spmat = th.sparse_coo_tensor(index[1], data, shape)
    spmat = SparseMatrix2D(
        index=index[1],
        data=data,
        shape=shape,
    )

    return spmat, None

def sparse_matrix_indices(spmat):
    return ('coo', spmat.index)


def cumsum(input, dim):
    return jnp.cumsum(input, axis=dim)


def is_tensor(obj):
    return isinstance(obj, jnp.ndarray) and hasattr(obj, 'device_buffer')

def shape(input):
    return input.shape

def dtype(input):
    return input.dtype

def ndim(input):
    return input.ndim

def context(input):
    try:
        return input.device_buffer.device()
    except:
        return jax.devices('cpu')[0]

def device_type(ctx):
    return ctx.platform

def device_id(ctx):
    return ctx.id

def to_backend_ctx(dglctx):
    dev_type = dglctx.device_type
    if dev_type == 1:
        return jax.devices('cpu')[0]
    elif dev_type == 2:
        return jax.devices('gpu')[0]
    else:
        raise ValueError('Unsupported DGL device context:', dglctx)

def astype(input, ty):
    return input.astype(ty)

def asnumpy(input):
    return onp.array(input)

def copy_to(input, ctx, **kwargs):
    # TODO:
    return jax.device_put(input, ctx)

def sum(input, dim, keepdims=False):
    return jnp.sum(input, axis=dim, keepdims=keepdims)

def reduce_sum(input):
    return input.sum()

def mean(input, dim):
    return jnp.mean(input, axis=dim)

def reduce_mean(input):
    return input.mean()

def max(input, dim):
    # NOTE: JAX handles the shape of `max` over multiple dimensions differently
    # from pytorch or tensorflow
    return jnp.max(input, axis=dim)

def reduce_max(input):
    return input.max()

def min(input, dim):
    # NOTE: the second argmin array is not returned
    return jnp.min(input, axis=dim)

def reduce_min(input):
    return input.min()

def argsort(input, dim, descending):
    _sorted = jnp.argsort(input, axis=dim)
    if descending is True:
        _sorted = jnp.flip(_sorted, axis=dim)
    return _sorted

def topk(input, k, dim, descending=True):
    _sorted = jnp.sort(input, axis=dim, descending=descending)
    return _sorted[:k]

def argtopk(input, k, dim, descending=True):
    _sorted = jnp.argsort(input, axis=dim, descending=descending)
    return _sorted[:k]

def exp(input):
    return jnp.exp(input)

def sqrt(input):
    return jnp.sqrt(input)

def softmax(input, dim=-1):
    return jax.nn.softmax(input, axis=dim)

def cat(seq, dim):
    return jnp.concatenate(seq, axis=dim)

def stack(seq, dim):
    return jnp.stack(seq, axis=dim)

def split(input, sizes_or_sections, dim):
    sizes_or_sections = jnp.cumsum(tensor(sizes_or_sections))
    return jnp.split(input, sizes_or_sections, axis=dim)[:-1]

def repeat(input, repeats, dim):
    return jnp.repeat(input, repeats=repeats, axis=dim)

def gather_row(data, row_index):
    return jnp.take(data, row_index, 0)

def slice_axis(data, axis, begin, end):
    return jnp.take(
        data,
        indices=jnp.arange(start=begin, stop=end),
        axis=axis,
    )

def take(data, indices, dim):
    return jnp.take(
        data,
        indices=indices,
        axis=dim,
    )

def narrow_row(x, start, stop):
    return x[start:stop]

def index_add_inplace(data, row_idx, value):
    raise NotImplementedError

def scatter_row(data, row_index, value):
    return jax.ops.index_update(
        data,
        row_index,
        value
    )

def scatter_row_inplace(data, row_index, value):
    raise NotImplementedError

def squeeze(input, dim):
    return jnp.squeeze(input, dim)

def unsqueeze(input, dim):
    return jnp.expand_dims(input, dim)

def reshape(input, shape):
    return jnp.reshape(input ,shape)

def swapaxes(input, axis1, axis2):
    return jnp.swapaxes(input, axis1, axis2)

def zeros(shape, dtype, ctx):
    # TODO: device
    return jnp.zeros(shape, dtype=dtype,) + 0 # eval lazy

def zeros_like(input):
    return jnp.zeros_like(input) + 0 # eval lazy

def ones(shape, dtype, ctx):
    # TODO: no device here
    return jnp.ones(shape, dtype=dtype,) + 0 # eval lazy

def uniform(shape, dtype, *, low, high):
    key = jax.random.PRNGKey(2666)
    return jax.random.uniform(
        key=key,
        shape=shape,
        minval=low,
        maxval=high,
        dtype=dtype,
    )

def randint(shape, dtype, *, low, high):
    key = jax.random.PRNGKey(2666)
    return jax.random.randint(
        key=key,
        shape=shape,
        minval=low,
        maxval=high,
        dtype=dtype,
    )

def pad_packed_tensor(input, lengths, value, l_min=None):
    old_shape = input.shape
    if isinstance(lengths, jnp.ndarray):
        max_len = as_scalar(lengths.max())
    else:
        max_len = builtins.max(lengths)

    if l_min is not None:
        max_len = builtins.max(max_len, l_min)

    batch_size = len(lengths)
    x = jnp.ones(
        shape=(batch_size * max_len, *old_shape[1:]),
        dtype=input.dtype,
    ) * value

    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = jnp.asarray(index)
    return scatter_row(x, index, input).reshape((batch_size, max_len, *old_shape[1:]))

def pack_padded_tensor(input, lengths):
    batch_size, max_len = input.shape[:2]
    index = []
    for i, l in enumerate(lengths):
        index.extend(range(i * max_len, i * max_len + l))
    index = tensor(index)
    return gather_row(input.reshape((batch_size * max_len, -1)), index)

def boolean_mask(input, mask):
    if 'bool' not in str(mask.dtype):
        mask = jnp.asarray(mask, dtype=jnp.bool_)
    return input[mask]

def equal(x, y):
    return x == y

def logical_not(input):
    return ~input

def logical_and(input1, input2):
    return input1 & input2

def clone(input):
    return input + 0

def clamp(data, min_val, max_val):
    return jnp.clip(data, min_val, max_val)

def replace_inf_with_zero(x):
    return jnp.nan_to_num(
        x,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

def unique(input):
    if input.dtype == jnp.bool_:
        input = input.type(jnp.int8)
    return jnp.unique(input)

def full_1d(length, fill_value, dtype, ctx):
    return jnp.full((length,), fill_value, dtype=dtype)

def nonzero_1d(input):
    x = (jnp.nonzero(input)[0]).squeeze()
    return x.flatten()

def sort_1d(input):
    idxs = jnp.argsort(input)
    return input[idxs], idxs

def arange(start, stop, dtype=jnp.int64, ctx=None):
    return jnp.arange(start, stop, dtype=dtype)

@jax.jit
def rand_shuffle(arr):
    key = jax.random.PRNGKey(2666)
    idx = jnp.random.permuataion(
        key,
        jnp.arrange(len(arr)),
    )
    return arr[idx]

def zerocopy_to_dlpack(input):
    return jax.dlpack.to_dlpack(input)

def zerocopy_from_dlpack(dlpack_tensor):
    return jax.dlpack.from_dlpack(dlpack_tensor)

def zerocopy_to_numpy(input):
    # NOTE: not zerocopy
    return asnumpy(input)

def zerocopy_from_numpy(np_array):
    return jnp.asarray(np_array)

def zerocopy_to_dgl_ndarray(data):
    if not is_tensor(data): data = tensor(data)
    return nd.from_dlpack(jax.dlpack.to_dlpack(data, take_ownership=False))

def zerocopy_to_dgl_ndarray_for_write(input):
    return nd.from_dlpack(jax.dlpack.to_dlpack(jnp.array(input), take_ownership=False))

def zerocopy_from_dgl_ndarray(data):
    if jax.devices()[0].platform == "gpu":
        if data.context.device_type == 1:
            x_cpu = jax.dlpack.from_dlpack(
                data.to_dlpack(),
                backend=jax.devices("cpu")[0].client,
            )

            return jax.device_put(
                x_cpu,
                jax.devices()[0],
            )

    return jax.dlpack.from_dlpack(data.to_dlpack())

def binary_reduce(reducer, binary_op, graph, lhs, rhs, lhs_data, rhs_data,
                  out_size, lhs_map=(None, None), rhs_map=(None, None), out_map=(None, None)):
    raise NotImplementedError

def copy_reduce(reducer, graph, target, in_data, out_size, in_map=(None, None),
                out_map=(None, None)):
    raise NotImplementedError

def _reduce_grad(grad, shape):
    raise NotImplementedError

def sync():
    # Pytorch performs computation synchronously, so no need for synchronization.
    pass

def attach_grad(x):
    # there is no concept of grad attribute in jax
    return x

def backward(x, head_gradient=None):
    # there is no concept of grad attribute in jax
    pass

def grad(x):
    # there is no concept of grad attribute in jax
    pass

def is_no_grad(x):
    return True

def is_recording():
    return True

class record_grad(object):
    def __init__(self):
        pass

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, exc_traceback):
        pass

def no_grad():
    raise NotImplementedError
