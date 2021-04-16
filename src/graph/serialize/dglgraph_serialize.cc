/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/serialize/graph_serialize.cc
 * \brief Graph serialization implementation
 *
 * The storage structure is
 * {
 *   // MetaData Section
 *   uint64_t kDGLSerializeMagic
 *   uint64_t kVersion
 *   uint64_t GraphType
 *   ** Reserved Area till 4kB **
 *
 *   dgl_id_t num_graphs
 *   vector<dgl_id_t> graph_indices (start address of each graph)
 *   vector<dgl_id_t> nodes_num_list (list of number of nodes for each graph)
 *   vector<dgl_id_t> edges_num_list (list of number of edges for each graph)
 *
 *   vector<GraphData> graph_datas;
 *
 * }
 *
 * Storage of GraphData is
 * {
 *   // Everything uses in csr
 *   NDArray indptr
 *   NDArray indices
 *   NDArray edge_ids
 *   vector<pair<string, NDArray>> node_tensors;
 *   vector<pair<string, NDArray>> edge_tensors;
 * }
 *
 */
#include <dgl/aten/coo.h>
#include <dgl/graph_op.h>
#include <dgl/immutable_graph.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/object.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>

#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "graph_serialize.h"

using namespace dgl::runtime;

using dgl::COO;
using dgl::COOPtr;
using dgl::ImmutableGraph;
using dgl::runtime::NDArray;
using dgl::serialize::GraphData;
using dgl::serialize::GraphDataObject;
using dmlc::SeekStream;
using std::vector;

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, GraphDataObject, true);
}

namespace dgl {
namespace serialize {

bool SaveDGLGraphs(std::string filename, List<GraphData> graph_data,
                   std::vector<NamedTensor> labels_list) {
  auto fs = std::unique_ptr<SeekStream>(dynamic_cast<SeekStream *>(
    SeekStream::Create(filename.c_str(), "w", true)));
  CHECK(fs) << "File name " << filename << " is not a valid local file name";

  // Write DGL MetaData
  const uint64_t kVersion = 1;
  fs->Write(kDGLSerializeMagic);
  fs->Write(kVersion);
  fs->Write(GraphType::kImmutableGraph);
  fs->Seek(4096);

  // Write Graph Meta Data
  dgl_id_t num_graph = graph_data.size();

  std::vector<dgl_id_t> graph_indices(num_graph);
  std::vector<int64_t> nodes_num_list(num_graph);
  std::vector<int64_t> edges_num_list(num_graph);

  for (uint64_t i = 0; i < num_graph; ++i) {
    nodes_num_list[i] = graph_data[i]->gptr->NumVertices();
    edges_num_list[i] = graph_data[i]->gptr->NumEdges();
  }
  // Reserve spaces for graph indices
  fs->Write(num_graph);
  dgl_id_t indices_start_ptr = fs->Tell();
  fs->Write(graph_indices);
  fs->Write(nodes_num_list);
  fs->Write(edges_num_list);
  fs->Write(labels_list);

  // Write GraphData
  for (uint64_t i = 0; i < num_graph; ++i) {
    graph_indices[i] = fs->Tell();
    GraphDataObject gdata = *graph_data[i].as<GraphDataObject>();
    fs->Write(gdata);
  }

  fs->Seek(indices_start_ptr);
  fs->Write(graph_indices);

  return true;
}

StorageMetaData LoadDGLGraphs(const std::string &filename,
                              std::vector<dgl_id_t> idx_list, bool onlyMeta) {
  auto fs = std::unique_ptr<SeekStream>(
    SeekStream::CreateForRead(filename.c_str(), true));
  CHECK(fs) << "Filename is invalid";
  // Read DGL MetaData
  uint64_t magicNum, graphType, version;
  fs->Read(&magicNum);
  fs->Read(&version);
  fs->Read(&graphType);
  fs->Seek(4096);

  CHECK_EQ(magicNum, kDGLSerializeMagic) << "Invalid DGL files";
  CHECK_EQ(version, 1) << "Invalid DGL files";
  StorageMetaData metadata = StorageMetaData::Create();
  // Read Graph MetaData
  dgl_id_t num_graph;
  CHECK(fs->Read(&num_graph)) << "Invalid num of graph";
  std::vector<dgl_id_t> graph_indices;
  std::vector<int64_t> nodes_num_list;
  std::vector<int64_t> edges_num_list;
  std::vector<NamedTensor> labels_list;

  CHECK(fs->Read(&graph_indices)) << "Invalid graph indices";
  CHECK(fs->Read(&nodes_num_list)) << "Invalid node num list";
  CHECK(fs->Read(&edges_num_list)) << "Invalid edge num list";
  CHECK(fs->Read(&labels_list)) << "Invalid label list";

  metadata->SetMetaData(num_graph, nodes_num_list, edges_num_list, labels_list);

  std::vector<GraphData> gdata_refs;

  // Early Return
  if (onlyMeta) {
    return metadata;
  }

  if (idx_list.empty()) {
    // Read All Graphs
    gdata_refs.reserve(num_graph);
    for (uint64_t i = 0; i < num_graph; ++i) {
      GraphData gdata = GraphData::Create();
      GraphDataObject *gdata_ptr =
        const_cast<GraphDataObject *>(gdata.as<GraphDataObject>());
      fs->Read(gdata_ptr);
      gdata_refs.push_back(gdata);
    }
  } else {
    // Read Selected Graphss
    gdata_refs.reserve(idx_list.size());
    // Would be better if idx_list is sorted. However the returned the graphs
    // should be the same order as the idx_list
    for (uint64_t i = 0; i < idx_list.size(); ++i) {
      fs->Seek(graph_indices[idx_list[i]]);
      GraphData gdata = GraphData::Create();
      GraphDataObject *gdata_ptr =
        const_cast<GraphDataObject *>(gdata.as<GraphDataObject>());
      fs->Read(gdata_ptr);
      gdata_refs.push_back(gdata);
    }
  }
  metadata->SetGraphData(gdata_refs);
  return metadata;
}

void GraphDataObject::SetData(ImmutableGraphPtr gptr,
                              Map<std::string, Value> node_tensors,
                              Map<std::string, Value> edge_tensors) {
  this->gptr = gptr;

  for (auto kv : node_tensors) {
    std::string name = kv.first;
    Value v = kv.second;
    NDArray ndarray = static_cast<NDArray>(v->data);
    this->node_tensors.emplace_back(name, ndarray);
  }
  for (auto kv : edge_tensors) {
    std::string &name = kv.first;
    Value v = kv.second;
    const NDArray &ndarray = static_cast<NDArray>(v->data);
    this->edge_tensors.emplace_back(name, ndarray);
  }
}

void GraphDataObject::Save(dmlc::Stream *fs) const {
  // Using in csr for storage
  const CSRPtr g_csr = this->gptr->GetInCSR();
  fs->Write(g_csr->indptr());
  fs->Write(g_csr->indices());
  fs->Write(g_csr->edge_ids());
  fs->Write(node_tensors);
  fs->Write(edge_tensors);
}

bool GraphDataObject::Load(dmlc::Stream *fs) {
  NDArray indptr, indices, edge_ids;
  fs->Read(&indptr);
  fs->Read(&indices);
  fs->Read(&edge_ids);
  this->gptr = ImmutableGraph::CreateFromCSR(indptr, indices, edge_ids, "in");

  fs->Read(&this->node_tensors);
  fs->Read(&this->edge_tensors);
  return true;
}

ImmutableGraphPtr BatchLoadedGraphs(std::vector<GraphData> gdata_list) {
  std::vector<GraphPtr> gptrs;
  gptrs.reserve(gdata_list.size());
  for (auto gdata : gdata_list) {
    gptrs.push_back(static_cast<GraphPtr>(gdata->gptr));
  }
  ImmutableGraphPtr imGPtr =
    std::dynamic_pointer_cast<ImmutableGraph>(GraphOp::DisjointUnion(gptrs));
  return imGPtr;
}

ImmutableGraphPtr ToImmutableGraph(GraphPtr g) {
  ImmutableGraphPtr imgr = std::dynamic_pointer_cast<ImmutableGraph>(g);
  if (imgr) {
    return imgr;
  } else {
    MutableGraphPtr mgr = std::dynamic_pointer_cast<Graph>(g);
    CHECK(mgr) << "Invalid Graph Pointer";
    EdgeArray earray = mgr->Edges("eid");
    IdArray srcs_array = earray.src;
    IdArray dsts_array = earray.dst;

    bool row_sorted, col_sorted;
    std::tie(row_sorted, col_sorted) = COOIsSorted(
            aten::COOMatrix(mgr->NumVertices(), mgr->NumVertices(), srcs_array,
            dsts_array));

    ImmutableGraphPtr imgptr =
      ImmutableGraph::CreateFromCOO(mgr->NumVertices(), srcs_array, dsts_array,
            row_sorted, col_sorted);
    return imgptr;
  }
}

void StorageMetaDataObject::SetMetaData(dgl_id_t num_graph,
                                        std::vector<int64_t> nodes_num_list,
                                        std::vector<int64_t> edges_num_list,
                                        std::vector<NamedTensor> labels_list) {
  this->num_graph = num_graph;
  this->nodes_num_list = Value(MakeValue(aten::VecToIdArray(nodes_num_list)));
  this->edges_num_list = Value(MakeValue(aten::VecToIdArray(edges_num_list)));
  for (auto kv : labels_list) {
    this->labels_list.Set(kv.first, Value(MakeValue(kv.second)));
  }
}

void StorageMetaDataObject::SetGraphData(std::vector<GraphData> gdata) {
  this->graph_data = List<GraphData>(gdata);
}

}  // namespace serialize
}  // namespace dgl
