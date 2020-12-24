from dgl.backend import *
from dgl.nn import *
from . import backend_unittest
import os
import importlib
import sys
import numpy as np

mod = importlib.import_module('.%s' % backend_name, __name__)
thismod = sys.modules[__name__]

for api in backend_unittest.__dict__.keys():
    if api.startswith('__'):
        continue
    elif callable(mod.__dict__[api]):
        # Tensor APIs used in unit tests MUST be supported across all backends
        globals()[api] = mod.__dict__[api]

# Tensor creation with default dtype and context

_zeros = zeros
_ones = ones
_randn = randn
_tensor = tensor
_arange = arange
_full = full
_full_1d = full_1d
_softmax = softmax
_default_context_str = os.getenv('DGLTESTDEV', 'cpu')

if backend_name == "jax":
    import jax
    _default_context_str = jax.devices()[0].platform

_context_dict = {
        'cpu': cpu(),
        'gpu': cuda(),
        }
_default_context = _context_dict[_default_context_str]

def ctx():
    return _default_context

def gpu_ctx():
    return (_default_context_str == 'gpu')

def zeros(shape, dtype=float32, ctx=_default_context):
    return _zeros(shape, dtype, ctx)

def ones(shape, dtype=float32, ctx=_default_context):
    return _ones(shape, dtype, ctx)

def randn(shape):
    return copy_to(_randn(shape), _default_context)

def tensor(data, dtype=None):
    return copy_to(_tensor(data, dtype), _default_context)

def arange(start, stop, dtype=int64):
    return copy_to(_arange(start, stop, dtype), _default_context)

def full(shape, fill_value, dtype, ctx=_default_context):
    return _full(shape, fill_value, dtype, ctx)

def full_1d(length, fill_value, dtype, ctx=_default_context):
    return _full_1d(length, fill_value, dtype, ctx)

def softmax(x, dim):
    return _softmax(x, dim)
