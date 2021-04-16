/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/heterograph_capi.cc
 * \brief Heterograph CAPI bindings.
 */
#include <dgl/array.h>
#include <dgl/aten/coo.h>
#include <dgl/packed_func_ext.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <set>

#include "../c_api_common.h"
#include "./heterograph.h"
#include "unit_graph.h"

using namespace dgl::runtime;

namespace dgl {

///////////////////////// Unitgraph functions /////////////////////////

// XXX(minjie): Ideally, Unitgraph should be invisible to python side

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateUnitGraphFromCOO")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int64_t nvtypes = args[0];
    int64_t num_src = args[1];
    int64_t num_dst = args[2];
    IdArray row = args[3];
    IdArray col = args[4];
    List<Value> formats = args[5];
    bool row_sorted = args[6];
    bool col_sorted = args[7];
    std::vector<SparseFormat> formats_vec;
    for (Value val : formats) {
      std::string fmt = val->data;
      formats_vec.push_back(ParseSparseFormat(fmt));
    }
    const auto code = SparseFormatsToCode(formats_vec);
    auto hgptr = CreateFromCOO(nvtypes, num_src, num_dst, row, col,
        row_sorted, col_sorted, code);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateUnitGraphFromCSR")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int64_t nvtypes = args[0];
    int64_t num_src = args[1];
    int64_t num_dst = args[2];
    IdArray indptr = args[3];
    IdArray indices = args[4];
    IdArray edge_ids = args[5];
    List<Value> formats = args[6];
    std::vector<SparseFormat> formats_vec;
    for (Value val : formats) {
      std::string fmt = val->data;
      formats_vec.push_back(ParseSparseFormat(fmt));
    }
    const auto code = SparseFormatsToCode(formats_vec);
    auto hgptr = CreateFromCSR(nvtypes, num_src, num_dst, indptr, indices, edge_ids, code);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateHeteroGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef meta_graph = args[0];
    List<HeteroGraphRef> rel_graphs = args[1];
    std::vector<HeteroGraphPtr> rel_ptrs;
    rel_ptrs.reserve(rel_graphs.size());
    for (const auto& ref : rel_graphs) {
      rel_ptrs.push_back(ref.sptr());
    }
    auto hgptr = CreateHeteroGraph(meta_graph.sptr(), rel_ptrs);
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateHeteroGraphWithNumNodes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef meta_graph = args[0];
    List<HeteroGraphRef> rel_graphs = args[1];
    IdArray num_nodes_per_type = args[2];
    std::vector<HeteroGraphPtr> rel_ptrs;
    rel_ptrs.reserve(rel_graphs.size());
    for (const auto& ref : rel_graphs) {
      rel_ptrs.push_back(ref.sptr());
    }
    auto hgptr = CreateHeteroGraph(
        meta_graph.sptr(), rel_ptrs, num_nodes_per_type.ToVector<int64_t>());
    *rv = HeteroGraphRef(hgptr);
  });

///////////////////////// HeteroGraph member functions /////////////////////////

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetMetaGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = GraphRef(hg->meta_graph());
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetRelationGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    CHECK_LE(etype, hg->NumEdgeTypes()) << "invalid edge type " << etype;
    auto unit_graph = hg->GetRelationGraph(etype);
    auto meta_graph = unit_graph->meta_graph();
    auto hgptr = CreateHeteroGraph(
        meta_graph, {unit_graph}, unit_graph->NumVerticesPerType());
    *rv = HeteroGraphRef(hgptr);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetFlattenedGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> etypes = args[1];
    std::vector<dgl_id_t> etypes_vec;
    for (Value val : etypes) {
      // (gq) have to decompose it into two statements because of a weird MSVC internal error
      dgl_id_t id = val->data;
      etypes_vec.push_back(id);
    }

    *rv = FlattenedHeteroGraphRef(hg->Flatten(etypes_vec));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAddVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    int64_t num = args[2];
    hg->AddVertices(vtype, num);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAddEdge")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    dgl_id_t dst = args[3];
    hg->AddEdge(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAddEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray src = args[2];
    IdArray dst = args[3];
    hg->AddEdges(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroClear")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    hg->Clear();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroDataType")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->DataType();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->Context();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->NumBits();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroIsMultigraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->IsMultigraph();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroIsReadonly")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = hg->IsReadonly();
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroNumVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    *rv = static_cast<int64_t>(hg->NumVertices(vtype));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroNumEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    *rv = static_cast<int64_t>(hg->NumEdges(etype));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasVertex")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    dgl_id_t vid = args[2];
    *rv = hg->HasVertex(vtype, vid);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t vtype = args[1];
    IdArray vids = args[2];
    *rv = hg->HasVertices(vtype, vids);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasEdgeBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    dgl_id_t dst = args[3];
    *rv = hg->HasEdgeBetween(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroHasEdgesBetween")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray src = args[2];
    IdArray dst = args[3];
    *rv = hg->HasEdgesBetween(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroPredecessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t dst = args[2];
    *rv = hg->Predecessors(etype, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSuccessors")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    *rv = hg->Successors(etype, src);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdgeId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t src = args[2];
    dgl_id_t dst = args[3];
    *rv = hg->EdgeId(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdgeIdsAll")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray src = args[2];
    IdArray dst = args[3];
    const auto& ret = hg->EdgeIdsAll(etype, src, dst);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });


DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdgeIdsOne")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray src = args[2];
    IdArray dst = args[3];
    *rv = hg->EdgeIdsOne(etype, src, dst);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroFindEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray eids = args[2];
    const auto& ret = hg->FindEdges(etype, eids);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    const auto& ret = hg->InEdges(etype, vid);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    const auto& ret = hg->InEdges(etype, vids);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutEdges_1")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    const auto& ret = hg->OutEdges(etype, vid);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutEdges_2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    const auto& ret = hg->OutEdges(etype, vids);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    std::string order = args[2];
    const auto& ret = hg->Edges(etype, order);
    *rv = ConvertEdgeArrayToPackedFunc(ret);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    *rv = static_cast<int64_t>(hg->InDegree(etype, vid));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroInDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    *rv = hg->InDegrees(etype, vids);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutDegree")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    dgl_id_t vid = args[2];
    *rv = static_cast<int64_t>(hg->OutDegree(etype, vid));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroOutDegrees")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    IdArray vids = args[2];
    *rv = hg->OutDegrees(etype, vids);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetAdj")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_type_t etype = args[1];
    bool transpose = args[2];
    std::string fmt = args[3];
    *rv = ConvertNDArrayVectorToPackedFunc(
        hg->GetAdj(etype, transpose, fmt));
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroVertexSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> vids = args[1];
    std::vector<IdArray> vid_vec;
    vid_vec.reserve(vids.size());
    for (Value val : vids) {
      vid_vec.push_back(val->data);
    }
    std::shared_ptr<HeteroSubgraph> subg(
        new HeteroSubgraph(hg->VertexSubgraph(vid_vec)));
    *rv = HeteroSubgraphRef(subg);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroEdgeSubgraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> eids = args[1];
    bool preserve_nodes = args[2];
    std::vector<IdArray> eid_vec;
    eid_vec.reserve(eids.size());
    for (Value val : eids) {
      eid_vec.push_back(val->data);
    }
    std::shared_ptr<HeteroSubgraph> subg(
        new HeteroSubgraph(hg->EdgeSubgraph(eid_vec, preserve_nodes)));
    *rv = HeteroSubgraphRef(subg);
  });

///////////////////////// HeteroSubgraph members /////////////////////////

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSubgraphGetGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroSubgraphRef subg = args[0];
    *rv = HeteroGraphRef(subg->graph);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSubgraphGetInducedVertices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroSubgraphRef subg = args[0];
    List<Value> induced_verts;
    for (IdArray arr : subg->induced_vertices) {
      induced_verts.push_back(Value(MakeValue(arr)));
    }
    *rv = induced_verts;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroSubgraphGetInducedEdges")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroSubgraphRef subg = args[0];
    List<Value> induced_edges;
    for (IdArray arr : subg->induced_edges) {
      induced_edges.push_back(Value(MakeValue(arr)));
    }
    *rv = induced_edges;
  });

///////////////////////// Global functions and algorithms /////////////////////////

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroAsNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    int bits = args[1];
    HeteroGraphPtr bhg_ptr = hg.sptr();
    auto hg_ptr = std::dynamic_pointer_cast<HeteroGraph>(bhg_ptr);
    HeteroGraphPtr hg_new;
    if (hg_ptr) {
      hg_new = HeteroGraph::AsNumBits(hg_ptr, bits);
    } else {
      hg_new = UnitGraph::AsNumBits(bhg_ptr, bits);
    }
    *rv = HeteroGraphRef(hg_new);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCopyTo")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    int device_type = args[1];
    int device_id = args[2];
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(device_type);
    ctx.device_id = device_id;
    HeteroGraphPtr hg_new = HeteroGraph::CopyTo(hg.sptr(), ctx);
    *rv = HeteroGraphRef(hg_new);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCopyToSharedMem")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    std::string name = args[1];
    List<Value> ntypes = args[2];
    List<Value> etypes = args[3];
    List<Value> fmts = args[4];
    auto ntypes_vec = ListValueToVector<std::string>(ntypes);
    auto etypes_vec = ListValueToVector<std::string>(etypes);
    std::set<std::string> fmts_set;
    for (const auto &fmt : fmts) {
      std::string fmt_data = fmt->data;
      fmts_set.insert(fmt_data);
    }
    auto hg_share = HeteroGraph::CopyToSharedMem(
        hg.sptr(), name, ntypes_vec, etypes_vec, fmts_set);
    *rv = HeteroGraphRef(hg_share);
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateFromSharedMem")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    std::string name = args[0];
    HeteroGraphPtr hg;
    std::vector<std::string> ntypes;
    std::vector<std::string> etypes;
    std::tie(hg, ntypes, etypes) = HeteroGraph::CreateFromSharedMem(name);
    List<Value> ntypes_list;
    List<Value> etypes_list;
    for (const auto &ntype : ntypes)
      ntypes_list.push_back(Value(MakeValue(ntype)));
    for (const auto &etype : etypes)
      etypes_list.push_back(Value(MakeValue(etype)));
    List<ObjectRef> ret;
    ret.push_back(HeteroGraphRef(hg));
    ret.push_back(ntypes_list);
    ret.push_back(etypes_list);
    *rv = ret;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroJointUnion")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef meta_graph = args[0];
    List<HeteroGraphRef> component_graphs = args[1];
    CHECK(component_graphs.size() > 1)
      << "Expect graph list to have at least two graphs";
    std::vector<HeteroGraphPtr> component_ptrs;
    component_ptrs.reserve(component_graphs.size());
    const int64_t bits = component_graphs[0]->NumBits();
    const DLContext ctx = component_graphs[0]->Context();
    for (const auto& component : component_graphs) {
      component_ptrs.push_back(component.sptr());
      CHECK_EQ(component->NumBits(), bits)
        << "Expect graphs to joint union have the same index dtype(int" << bits
        << "), but got int" << component->NumBits();
      CHECK_EQ(component->Context(), ctx)
        << "Expect graphs to joint union have the same context" << ctx
        << "), but got " << component->Context();
    }

    auto hgptr = JointUnionHeteroGraph(meta_graph.sptr(), component_ptrs);
    *rv = HeteroGraphRef(hgptr);
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroDisjointUnion_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef meta_graph = args[0];
    List<HeteroGraphRef> component_graphs = args[1];
    CHECK(component_graphs.size() > 0)
      << "Expect graph list has at least one graph";
    std::vector<HeteroGraphPtr> component_ptrs;
    component_ptrs.reserve(component_graphs.size());
    const int64_t bits = component_graphs[0]->NumBits();
    const DLContext ctx = component_graphs[0]->Context();
    for (const auto& component : component_graphs) {
      component_ptrs.push_back(component.sptr());
      CHECK_EQ(component->NumBits(), bits)
        << "Expect graphs to batch have the same index dtype(int" << bits
        << "), but got int" << component->NumBits();
      CHECK_EQ(component->Context(), ctx)
        << "Expect graphs to batch have the same context" << ctx
        << "), but got " << component->Context();
    }

    auto hgptr = DisjointUnionHeteroGraph2(meta_graph.sptr(), component_ptrs);
    *rv = HeteroGraphRef(hgptr);
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroDisjointPartitionBySizes_v2")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    const IdArray vertex_sizes = args[1];
    const IdArray edge_sizes = args[2];
    std::vector<HeteroGraphPtr> ret;
    ret = DisjointPartitionHeteroBySizes2(hg->meta_graph(), hg.sptr(),
                                          vertex_sizes, edge_sizes);
    List<HeteroGraphRef> ret_list;
    for (HeteroGraphPtr hgptr : ret) {
      ret_list.push_back(HeteroGraphRef(hgptr));
    }
    *rv = ret_list;
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroDisjointPartitionBySizes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    const IdArray vertex_sizes = args[1];
    const IdArray edge_sizes = args[2];
    const int64_t bits = hg->NumBits();
    std::vector<HeteroGraphPtr> ret;
    ATEN_ID_BITS_SWITCH(bits, IdType, {
      ret = DisjointPartitionHeteroBySizes<IdType>(hg->meta_graph(), hg.sptr(),
                                                   vertex_sizes, edge_sizes);
    });
    List<HeteroGraphRef> ret_list;
    for (HeteroGraphPtr hgptr : ret) {
      ret_list.push_back(HeteroGraphRef(hgptr));
    }
    *rv = ret_list;
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetCreatedFormats")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> format_list;
    dgl_format_code_t code = hg->GetRelationGraph(0)->GetCreatedFormats();
    for (auto format : CodeToSparseFormats(code)) {
      format_list.push_back(
          Value(MakeValue(ToStringSparseFormat(format))));
    }
    *rv = format_list;
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetAllowedFormats")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> format_list;
    dgl_format_code_t code = hg->GetRelationGraph(0)->GetAllowedFormats();
    for (auto format : CodeToSparseFormats(code)) {
      format_list.push_back(
          Value(MakeValue(ToStringSparseFormat(format))));
    }
    *rv = format_list;
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroCreateFormat")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    dgl_format_code_t code = hg->GetRelationGraph(0)->GetAllowedFormats();
#if !defined(DGL_USE_CUDA)
#pragma omp parallel for
#endif
    for (int64_t etype = 0; etype < hg->NumEdgeTypes(); ++etype) {
      auto bg = std::dynamic_pointer_cast<UnitGraph>(hg->GetRelationGraph(etype));
      for (auto format : CodeToSparseFormats(code))
        bg->GetFormat(format);
    }
});

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroGetFormatGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    List<Value> formats = args[1];
    std::vector<SparseFormat> formats_vec;
    for (Value val : formats) {
      std::string fmt = val->data;
      formats_vec.push_back(ParseSparseFormat(fmt));
    }
    auto hgptr = hg->GetGraphInFormat(
        SparseFormatsToCode(formats_vec));
    *rv = HeteroGraphRef(hgptr);
});

DGL_REGISTER_GLOBAL("subgraph._CAPI_DGLInSubgraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    std::shared_ptr<HeteroSubgraph> ret(new HeteroSubgraph);
    *ret = InEdgeGraph(hg.sptr(), nodes);
    *rv = HeteroGraphRef(ret);
  });

DGL_REGISTER_GLOBAL("subgraph._CAPI_DGLOutSubgraph")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    std::shared_ptr<HeteroSubgraph> ret(new HeteroSubgraph);
    *ret = OutEdgeGraph(hg.sptr(), nodes);
    *rv = HeteroGraphRef(ret);
  });

DGL_REGISTER_GLOBAL("transform._CAPI_DGLAsImmutableGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    *rv = GraphRef(hg->AsImmutableGraph());
  });

DGL_REGISTER_GLOBAL("heterograph._CAPI_DGLFindSrcDstNtypes")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef metagraph = args[0];
    std::unordered_set<int64_t> dst_set;
    std::unordered_set<int64_t> src_set;

    for (int64_t eid = 0; eid < metagraph->NumEdges(); ++eid) {
      auto edge = metagraph->FindEdge(eid);
      auto src = edge.first;
      auto dst = edge.second;
      dst_set.insert(dst);
      src_set.insert(src);
    }

    List<Value> srclist, dstlist;
    List<List<Value>> ret_list;
    for (int64_t nid = 0; nid < metagraph->NumVertices(); ++nid) {
      auto is_dst = dst_set.count(nid);
      auto is_src = src_set.count(nid);
      if (is_dst && is_src)
        return;
      else if (is_dst)
        dstlist.push_back(Value(MakeValue(nid)));
      else
        // If a node type is isolated, put it in srctype as defined in the Python docstring.
        srclist.push_back(Value(MakeValue(nid)));
    }
    ret_list.push_back(srclist);
    ret_list.push_back(dstlist);
    *rv = ret_list;
  });

DGL_REGISTER_GLOBAL("heterograph_index._CAPI_DGLHeteroReverse")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    HeteroGraphRef hg = args[0];
    CHECK_GT(hg->NumEdgeTypes(), 0);
    auto g = std::dynamic_pointer_cast<HeteroGraph>(hg.sptr());
    std::vector<HeteroGraphPtr> rev_ugs;
    const auto &ugs = g->relation_graphs();
    rev_ugs.resize(ugs.size());

    for (size_t i = 0; i < ugs.size(); ++i) {
      const auto &rev_ug = ugs[i]->Reverse();
      rev_ugs[i] = rev_ug;
    }
    // node types are not changed
    const auto& num_nodes = g->NumVerticesPerType();
    const auto& meta_edges = hg->meta_graph()->Edges("eid");
    // reverse the metagraph
    const auto& rev_meta = ImmutableGraph::CreateFromCOO(hg->meta_graph()->NumVertices(),
                                                         meta_edges.dst, meta_edges.src);
    *rv = CreateHeteroGraph(rev_meta, rev_ugs, num_nodes);
  });
}  // namespace dgl
