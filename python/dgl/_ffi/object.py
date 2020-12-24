"""Object namespace"""
# pylint: disable=unused-import
from __future__ import absolute_import

import ctypes
import sys
from .. import _api_internal
from .object_generic import ObjectGeneric, convert_to_object
from .base import _LIB, check_call, c_str, py_str, _FFI_MODE

IMPORT_EXCEPT = RuntimeError if _FFI_MODE == "cython" else ImportError
try:
    # pylint: disable=wrong-import-position
    if _FFI_MODE == "ctypes":
        raise ImportError()
    if sys.version_info >= (3, 0):
        from ._cy3.core import _register_object, ObjectBase as _ObjectBase
    else:
        from ._cy2.core import _register_object, ObjectBase as _ObjectBase
except IMPORT_EXCEPT:
    # pylint: disable=wrong-import-position
    from ._ctypes.object import _register_object, ObjectBase as _ObjectBase


def _new_object(cls):
    """Helper function for pickle"""
    return cls.__new__(cls)


class ObjectBase(_ObjectBase):
    """ObjectBase is the base class of all DGL CAPI object.

    The core attribute is ``handle``, which is a C raw pointer.  It must be initialized
    via ``__init_handle_by_constructor__``.

    Note that the same handle **CANNOT** be shared across multiple ObjectBase instances.
    """
    def __dir__(self):
        plist = ctypes.POINTER(ctypes.c_char_p)()
        size = ctypes.c_uint()
        check_call(_LIB.DGLObjectListAttrNames(
            self.handle, ctypes.byref(size), ctypes.byref(plist)))
        names = []
        for i in range(size.value):
            names.append(py_str(plist[i]))
        return names

    # def __hash__(self):
    #     return _api_internal._raw_ptr(self)

    def __eq__(self, other):
        return self.same_as(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __reduce__(self):
        cls = type(self)
        return (_new_object, (cls, ), self.__getstate__())

    def __getstate__(self):
        # TODO(minjie): TVM assumes that a Node (Object in DGL) can be serialized
        #   to json. However, this is not true in DGL because DGL Object is meant
        #   for runtime API, so it could contain binary data such as NDArray.
        #   If this feature is required, please raise a RFC to DGL issue.
        raise RuntimeError("__getstate__ is not supported for object type")

    def __setstate__(self, state):
        # pylint: disable=assigning-non-slot
        # TODO(minjie): TVM assumes that a Node (Object in DGL) can be serialized
        #   to json. However, this is not true in DGL because DGL Object is meant
        #   for runtime API, so it could contain binary data such as NDArray.
        #   If this feature is required, please raise a RFC to DGL issue.
        raise RuntimeError("__setstate__ is not supported for object type")

    def same_as(self, other):
        """check object identity equality"""
        if not isinstance(other, ObjectBase):
            return False
        return self.__hash__() == other.__hash__()


def register_object(type_key=None):
    """Decorator used to register object type

    Examples
    --------
    >>> @register_object
    >>> class MyObject:
    >>> ... pass

    Parameters
    ----------
    type_key : str or cls
        The type key of the object
    """
    object_name = type_key if isinstance(type_key, str) else type_key.__name__

    def register(cls):
        """internal register function"""
        tindex = ctypes.c_int()
        ret = _LIB.DGLObjectTypeKey2Index(c_str(object_name), ctypes.byref(tindex))
        if ret == 0:
            _register_object(tindex.value, cls)
        return cls

    if isinstance(type_key, str):
        return register
    return register(type_key)
