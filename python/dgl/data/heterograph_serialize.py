"""For HeteroGraph Serialization"""
from __future__ import absolute_import
from ..heterograph import DGLHeteroGraph
from ..frame import Frame
from .._ffi.object import ObjectBase, register_object
from .._ffi.function import _init_api
from .. import backend as F
from ..container import convert_to_strmap

_init_api("dgl.data.heterograph_serialize")


def tensor_dict_to_ndarray_dict(tensor_dict):
    """Convert dict[str, tensor] to StrMap[NDArray]"""
    ndarray_dict = {}
    for key, value in tensor_dict.items():
        if F.backend_name == "jax":
            from jax import numpy as jnp
            if value.dtype == jnp.bool_:
                value = (value + 0).astype(jnp.int8)

        value = value + 0
        ndarray_dict[key] = F.zerocopy_to_dgl_ndarray(value)
    return convert_to_strmap(ndarray_dict)


def save_heterographs(filename, g_list, labels):
    """Save heterographs into file"""
    if labels is None:
        labels = {}
    if isinstance(g_list, DGLHeteroGraph):
        g_list = [g_list]
    assert all([type(g) == DGLHeteroGraph for g in g_list]), "Invalid DGLHeteroGraph in g_list argument"
    gdata_list = [HeteroGraphData.create(g) for g in g_list]
    _CAPI_SaveHeteroGraphData(filename, gdata_list, tensor_dict_to_ndarray_dict(labels))

@register_object("heterograph_serialize.HeteroGraphData")
class HeteroGraphData(ObjectBase):
    """Object to hold the data to be stored for DGLHeteroGraph"""

    @staticmethod
    def create(g):
        edata_list = []
        ndata_list = []
        for etype in g.canonical_etypes:
            edata_list.append(tensor_dict_to_ndarray_dict(g.edges[etype].data))
        for ntype in g.ntypes:
            ndata_list.append(tensor_dict_to_ndarray_dict(g.nodes[ntype].data))
        return _CAPI_MakeHeteroGraphData(g._graph, ndata_list, edata_list, g.ntypes, g.etypes)

    def get_graph(self):
        ntensor_list = list(_CAPI_GetNDataFromHeteroGraphData(self))
        etensor_list = list(_CAPI_GetEDataFromHeteroGraphData(self))
        ntype_names = list(_CAPI_GetNtypesFromHeteroGraphData(self))
        etype_names = list(_CAPI_GetEtypesFromHeteroGraphData(self))
        gidx = _CAPI_GetGindexFromHeteroGraphData(self)
        nframes = []
        eframes = []
        for ntid, ntensor in enumerate(ntensor_list):
            ndict = {ntensor[i]: F.zerocopy_from_dgl_ndarray(ntensor[i+1]) for i in range(0, len(ntensor), 2)}
            nframes.append(Frame(ndict, num_rows=gidx.number_of_nodes(ntid)))

        for etid, etensor in enumerate(etensor_list):
            edict = {etensor[i]: F.zerocopy_from_dgl_ndarray(etensor[i+1]) for i in range(0, len(etensor), 2)}
            eframes.append(Frame(edict, num_rows=gidx.number_of_edges(etid)))

        return DGLHeteroGraph(gidx, ntype_names, etype_names, nframes, eframes)
