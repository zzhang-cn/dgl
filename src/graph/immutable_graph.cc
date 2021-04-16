/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/immutable_graph.cc
 * \brief DGL immutable graph index implementation
 */

#include <dgl/immutable_graph.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/smart_ptr_serializer.h>
#include <dgl/base_heterograph.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <string.h>
#include <bitset>
#include <numeric>
#include <tuple>

#include "../c_api_common.h"
#include "heterograph.h"
#include "unit_graph.h"

using namespace dgl::runtime;

namespace dgl {
namespace {
inline std::string GetSharedMemName(const std::string &name, const std::string &edge_dir) {
  return name + "_" + edge_dir;
}

/*
 * The metadata of a graph index that are needed for shared-memory graph.
 */
struct GraphIndexMetadata {
  int64_t num_nodes;
  int64_t num_edges;
  bool has_in_csr;
  bool has_out_csr;
  bool has_coo;
};

/*
 * Serialize the metadata of a graph index and place it in a shared-memory tensor.
 * In this way, another process can reconstruct a GraphIndex from a shared-memory tensor.
 */
NDArray SerializeMetadata(ImmutableGraphPtr gidx, const std::string &name) {
#ifndef _WIN32
  GraphIndexMetadata meta;
  meta.num_nodes = gidx->NumVertices();
  meta.num_edges = gidx->NumEdges();
  meta.has_in_csr = gidx->HasInCSR();
  meta.has_out_csr = gidx->HasOutCSR();
  meta.has_coo = false;

  NDArray meta_arr = NDArray::EmptyShared(name, {sizeof(meta)}, DLDataType{kDLInt, 8, 1},
                                          DLContext{kDLCPU, 0}, true);
  memcpy(meta_arr->data, &meta, sizeof(meta));
  return meta_arr;
#else
  LOG(FATAL) << "CSR graph doesn't support shared memory in Windows yet";
  return NDArray();
#endif  // _WIN32
}

/*
 * Deserialize the metadata of a graph index.
 */
GraphIndexMetadata DeserializeMetadata(const std::string &name) {
  GraphIndexMetadata meta;
#ifndef _WIN32
  NDArray meta_arr = NDArray::EmptyShared(name, {sizeof(meta)}, DLDataType{kDLInt, 8, 1},
                                          DLContext{kDLCPU, 0}, false);
  memcpy(&meta, meta_arr->data, sizeof(meta));
#else
  LOG(FATAL) << "CSR graph doesn't support shared memory in Windows yet";
#endif  // _WIN32
  return meta;
}

std::tuple<IdArray, IdArray, IdArray> MapFromSharedMemory(
  const std::string &shared_mem_name, int64_t num_verts, int64_t num_edges, bool is_create) {
#ifndef _WIN32
  const int64_t file_size = (num_verts + 1 + num_edges * 2) * sizeof(dgl_id_t);

  IdArray sm_array = IdArray::EmptyShared(
      shared_mem_name, {file_size}, DLDataType{kDLInt, 8, 1}, DLContext{kDLCPU, 0}, is_create);
  // Create views from the shared memory array. Note that we don't need to save
  //   the sm_array because the refcount is maintained by the view arrays.
  IdArray indptr = sm_array.CreateView({num_verts + 1}, DLDataType{kDLInt, 64, 1});
  IdArray indices = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1) * sizeof(dgl_id_t));
  IdArray edge_ids = sm_array.CreateView({num_edges}, DLDataType{kDLInt, 64, 1},
      (num_verts + 1 + num_edges) * sizeof(dgl_id_t));
  return std::make_tuple(indptr, indices, edge_ids);
#else
  LOG(FATAL) << "CSR graph doesn't support shared memory in Windows yet";
  return {};
#endif  // _WIN32
}
}  // namespace

//////////////////////////////////////////////////////////
//
// CSR graph implementation
//
//////////////////////////////////////////////////////////

CSR::CSR(int64_t num_vertices, int64_t num_edges) {
  CHECK(!(num_vertices == 0 && num_edges != 0));
  adj_ = aten::CSRMatrix{num_vertices, num_vertices,
                         aten::NewIdArray(num_vertices + 1),
                         aten::NewIdArray(num_edges),
                         aten::NewIdArray(num_edges)};
  adj_.sorted = false;
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids) {
  CHECK(aten::IsValidIdArray(indptr));
  CHECK(aten::IsValidIdArray(indices));
  CHECK(aten::IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t N = indptr->shape[0] - 1;
  adj_ = aten::CSRMatrix{N, N, indptr, indices, edge_ids};
  adj_.sorted = false;
}

CSR::CSR(IdArray indptr, IdArray indices, IdArray edge_ids,
         const std::string &shared_mem_name): shared_mem_name_(shared_mem_name) {
  CHECK(aten::IsValidIdArray(indptr));
  CHECK(aten::IsValidIdArray(indices));
  CHECK(aten::IsValidIdArray(edge_ids));
  CHECK_EQ(indices->shape[0], edge_ids->shape[0]);
  const int64_t num_verts = indptr->shape[0] - 1;
  const int64_t num_edges = indices->shape[0];
  adj_.num_rows = num_verts;
  adj_.num_cols = num_verts;
  std::tie(adj_.indptr, adj_.indices, adj_.data) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, true);
  // copy the given data into the shared memory arrays
  adj_.indptr.CopyFrom(indptr);
  adj_.indices.CopyFrom(indices);
  adj_.data.CopyFrom(edge_ids);
  adj_.sorted = false;
}

CSR::CSR(const std::string &shared_mem_name,
         int64_t num_verts, int64_t num_edges): shared_mem_name_(shared_mem_name) {
  CHECK(!(num_verts == 0 && num_edges != 0));
  adj_.num_rows = num_verts;
  adj_.num_cols = num_verts;
  std::tie(adj_.indptr, adj_.indices, adj_.data) = MapFromSharedMemory(
      shared_mem_name, num_verts, num_edges, false);
  adj_.sorted = false;
}

bool CSR::IsMultigraph() const {
  return aten::CSRHasDuplicate(adj_);
}

EdgeArray CSR::OutEdges(dgl_id_t vid) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  IdArray ret_dst = aten::CSRGetRowColumnIndices(adj_, vid);
  IdArray ret_eid = aten::CSRGetRowData(adj_, vid);
  IdArray ret_src = aten::Full(vid, ret_dst->shape[0], NumBits(), ret_dst->ctx);
  return EdgeArray{ret_src, ret_dst, ret_eid};
}

EdgeArray CSR::OutEdges(IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
  auto csrsubmat = aten::CSRSliceRows(adj_, vids);
  auto coosubmat = aten::CSRToCOO(csrsubmat, false);
  // Note that the row id in the csr submat is relabled, so
  // we need to recover it using an index select.
  auto row = aten::IndexSelect(vids, coosubmat.row);
  return EdgeArray{row, coosubmat.col, coosubmat.data};
}

DegreeArray CSR::OutDegrees(IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
  return aten::CSRGetRowNNZ(adj_, vids);
}

bool CSR::HasEdgeBetween(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "Invalid vertex id: " << src;
  CHECK(HasVertex(dst)) << "Invalid vertex id: " << dst;
  return aten::CSRIsNonZero(adj_, src, dst);
}

BoolArray CSR::HasEdgesBetween(IdArray src_ids, IdArray dst_ids) const {
  CHECK(aten::IsValidIdArray(src_ids)) << "Invalid vertex id array.";
  CHECK(aten::IsValidIdArray(dst_ids)) << "Invalid vertex id array.";
  return aten::CSRIsNonZero(adj_, src_ids, dst_ids);
}

IdArray CSR::Successors(dgl_id_t vid, uint64_t radius) const {
  CHECK(HasVertex(vid)) << "invalid vertex: " << vid;
  CHECK(radius == 1) << "invalid radius: " << radius;
  return aten::CSRGetRowColumnIndices(adj_, vid);
}

IdArray CSR::EdgeId(dgl_id_t src, dgl_id_t dst) const {
  CHECK(HasVertex(src)) << "invalid vertex: " << src;
  CHECK(HasVertex(dst)) << "invalid vertex: " << dst;
  return aten::CSRGetAllData(adj_, src, dst);
}

EdgeArray CSR::EdgeIds(IdArray src_ids, IdArray dst_ids) const {
  const auto& arrs = aten::CSRGetDataAndIndices(adj_, src_ids, dst_ids);
  return EdgeArray{arrs[0], arrs[1], arrs[2]};
}

EdgeArray CSR::Edges(const std::string &order) const {
  CHECK(order.empty() || order == std::string("srcdst"))
    << "CSR only support Edges of order \"srcdst\","
    << " but got \"" << order << "\".";
  const auto& coo = aten::CSRToCOO(adj_, false);
  return EdgeArray{coo.row, coo.col, coo.data};
}

Subgraph CSR::VertexSubgraph(IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
  const auto& submat = aten::CSRSliceMatrix(adj_, vids, vids);
  IdArray sub_eids = aten::Range(0, submat.data->shape[0], NumBits(), Context());
  CSRPtr subcsr(new CSR(submat.indptr, submat.indices, sub_eids));
  subcsr->adj_.sorted = this->adj_.sorted;
  Subgraph subg;
  subg.graph = subcsr;
  subg.induced_vertices = vids;
  subg.induced_edges = submat.data;
  return subg;
}

CSRPtr CSR::Transpose() const {
  const auto& trans = aten::CSRTranspose(adj_);
  return CSRPtr(new CSR(trans.indptr, trans.indices, trans.data));
}

COOPtr CSR::ToCOO() const {
  const auto& coo = aten::CSRToCOO(adj_, true);
  return COOPtr(new COO(NumVertices(), coo.row, coo.col));
}

CSR CSR::CopyTo(const DLContext& ctx) const {
  if (Context() == ctx) {
    return *this;
  } else {
    CSR ret(adj_.indptr.CopyTo(ctx),
            adj_.indices.CopyTo(ctx),
            adj_.data.CopyTo(ctx));
    return ret;
  }
}

CSR CSR::CopyToSharedMem(const std::string &name) const {
  if (IsSharedMem()) {
    CHECK(name == shared_mem_name_);
    return *this;
  } else {
    // TODO(zhengda) we need to set sorted_ properly.
    return CSR(adj_.indptr, adj_.indices, adj_.data, name);
  }
}

CSR CSR::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    CSR ret(aten::AsNumBits(adj_.indptr, bits),
            aten::AsNumBits(adj_.indices, bits),
            aten::AsNumBits(adj_.data, bits));
    return ret;
  }
}

DGLIdIters CSR::SuccVec(dgl_id_t vid) const {
  // TODO(minjie): This still assumes the data type and device context
  //   of this graph. Should fix later.
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
  const dgl_id_t* indices_data = static_cast<dgl_id_t*>(adj_.indices->data);
  const dgl_id_t start = indptr_data[vid];
  const dgl_id_t end = indptr_data[vid + 1];
  return DGLIdIters(indices_data + start, indices_data + end);
}

DGLIdIters CSR::OutEdgeVec(dgl_id_t vid) const {
  // TODO(minjie): This still assumes the data type and device context
  //   of this graph. Should fix later.
  const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
  const dgl_id_t* eid_data = static_cast<dgl_id_t*>(adj_.data->data);
  const dgl_id_t start = indptr_data[vid];
  const dgl_id_t end = indptr_data[vid + 1];
  return DGLIdIters(eid_data + start, eid_data + end);
}

bool CSR::Load(dmlc::Stream *fs) {
  fs->Read(const_cast<dgl::aten::CSRMatrix*>(&adj_));
  return true;
}

void CSR::Save(dmlc::Stream *fs) const {
  fs->Write(adj_);
}

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////
COO::COO(int64_t num_vertices, IdArray src, IdArray dst,
        bool row_sorted, bool col_sorted) {
  CHECK(aten::IsValidIdArray(src));
  CHECK(aten::IsValidIdArray(dst));
  CHECK_EQ(src->shape[0], dst->shape[0]);
  adj_ = aten::COOMatrix{num_vertices, num_vertices, src, dst,
                         aten::NullArray(), row_sorted, col_sorted};
}

bool COO::IsMultigraph() const {
  return aten::COOHasDuplicate(adj_);
}

std::pair<dgl_id_t, dgl_id_t> COO::FindEdge(dgl_id_t eid) const {
  CHECK(eid < NumEdges()) << "Invalid edge id: " << eid;
  const dgl_id_t src = aten::IndexSelect<dgl_id_t>(adj_.row, eid);
  const dgl_id_t dst = aten::IndexSelect<dgl_id_t>(adj_.col, eid);
  return std::pair<dgl_id_t, dgl_id_t>(src, dst);
}

EdgeArray COO::FindEdges(IdArray eids) const {
  CHECK(aten::IsValidIdArray(eids)) << "Invalid edge id array";
  BUG_ON(aten::IsNullArray(adj_.data)) <<
    "FindEdges requires the internal COO matrix not having EIDs.";
  return EdgeArray{aten::IndexSelect(adj_.row, eids),
                   aten::IndexSelect(adj_.col, eids),
                   eids};
}

EdgeArray COO::Edges(const std::string &order) const {
  CHECK(order.empty() || order == std::string("eid"))
    << "COO only support Edges of order \"eid\", but got \""
    << order << "\".";
  IdArray rst_eid = aten::Range(0, NumEdges(), NumBits(), Context());
  return EdgeArray{adj_.row, adj_.col, rst_eid};
}

Subgraph COO::EdgeSubgraph(IdArray eids, bool preserve_nodes) const {
  CHECK(aten::IsValidIdArray(eids)) << "Invalid edge id array.";
  COOPtr subcoo;
  IdArray induced_nodes;
  if (!preserve_nodes) {
    IdArray new_src = aten::IndexSelect(adj_.row, eids);
    IdArray new_dst = aten::IndexSelect(adj_.col, eids);
    induced_nodes = aten::Relabel_({new_src, new_dst});
    const auto new_nnodes = induced_nodes->shape[0];
    subcoo = COOPtr(new COO(new_nnodes, new_src, new_dst));
  } else {
    IdArray new_src = aten::IndexSelect(adj_.row, eids);
    IdArray new_dst = aten::IndexSelect(adj_.col, eids);
    induced_nodes = aten::Range(0, NumVertices(), NumBits(), Context());
    subcoo = COOPtr(new COO(NumVertices(), new_src, new_dst));
  }
  Subgraph subg;
  subg.graph = subcoo;
  subg.induced_vertices = induced_nodes;
  subg.induced_edges = eids;
  return subg;
}

CSRPtr COO::ToCSR() const {
  const auto& csr = aten::COOToCSR(adj_);
  return CSRPtr(new CSR(csr.indptr, csr.indices, csr.data));
}

COO COO::CopyTo(const DLContext& ctx) const {
  if (Context() == ctx) {
    return *this;
  } else {
    COO ret(NumVertices(),
            adj_.row.CopyTo(ctx),
            adj_.col.CopyTo(ctx));
    return ret;
  }
}

COO COO::CopyToSharedMem(const std::string &name) const {
  LOG(FATAL) << "COO doesn't supprt shared memory yet";
  return COO();
}

COO COO::AsNumBits(uint8_t bits) const {
  if (NumBits() == bits) {
    return *this;
  } else {
    COO ret(NumVertices(),
            aten::AsNumBits(adj_.row, bits),
            aten::AsNumBits(adj_.col, bits));
    return ret;
  }
}

//////////////////////////////////////////////////////////
//
// immutable graph implementation
//
//////////////////////////////////////////////////////////

BoolArray ImmutableGraph::HasVertices(IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid id array input";
  return aten::LT(vids, NumVertices());
}

CSRPtr ImmutableGraph::GetInCSR() const {
  if (!in_csr_) {
    if (out_csr_) {
      const_cast<ImmutableGraph*>(this)->in_csr_ = out_csr_->Transpose();
      if (out_csr_->IsSharedMem())
        LOG(WARNING) << "We just construct an in-CSR from a shared-memory out CSR. "
                     << "It may dramatically increase memory consumption.";
    } else {
      CHECK(coo_) << "None of CSR, COO exist";
      const_cast<ImmutableGraph*>(this)->in_csr_ = coo_->Transpose()->ToCSR();
    }
  }
  return in_csr_;
}

/* !\brief Return out csr. If not exist, transpose the other one.*/
CSRPtr ImmutableGraph::GetOutCSR() const {
  if (!out_csr_) {
    if (in_csr_) {
      const_cast<ImmutableGraph*>(this)->out_csr_ = in_csr_->Transpose();
      if (in_csr_->IsSharedMem())
        LOG(WARNING) << "We just construct an out-CSR from a shared-memory in CSR. "
                     << "It may dramatically increase memory consumption.";
    } else {
      CHECK(coo_) << "None of CSR, COO exist";
      const_cast<ImmutableGraph*>(this)->out_csr_ = coo_->ToCSR();
    }
  }
  return out_csr_;
}

/* !\brief Return coo. If not exist, create from csr.*/
COOPtr ImmutableGraph::GetCOO() const {
  if (!coo_) {
    if (in_csr_) {
      const_cast<ImmutableGraph*>(this)->coo_ = in_csr_->ToCOO()->Transpose();
    } else {
      CHECK(out_csr_) << "Both CSR are missing.";
      const_cast<ImmutableGraph*>(this)->coo_ = out_csr_->ToCOO();
    }
  }
  return coo_;
}

EdgeArray ImmutableGraph::Edges(const std::string &order) const {
  if (order.empty()) {
    // arbitrary order
    if (in_csr_) {
      // transpose
      const auto& edges = in_csr_->Edges(order);
      return EdgeArray{edges.dst, edges.src, edges.id};
    } else {
      return AnyGraph()->Edges(order);
    }
  } else if (order == std::string("srcdst")) {
    // TODO(minjie): CSR only guarantees "src" to be sorted.
    //   Maybe we should relax this requirement?
    return GetOutCSR()->Edges(order);
  } else if (order == std::string("eid")) {
    return GetCOO()->Edges(order);
  } else {
    LOG(FATAL) << "Unsupported order request: " << order;
  }
  return {};
}

Subgraph ImmutableGraph::VertexSubgraph(IdArray vids) const {
  // We prefer to generate a subgraph from out-csr.
  auto sg = GetOutCSR()->VertexSubgraph(vids);
  CSRPtr subcsr = std::dynamic_pointer_cast<CSR>(sg.graph);
  sg.graph = GraphPtr(new ImmutableGraph(subcsr));
  return sg;
}

Subgraph ImmutableGraph::EdgeSubgraph(IdArray eids, bool preserve_nodes) const {
  auto sg = GetCOO()->EdgeSubgraph(eids, preserve_nodes);
  COOPtr subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
  sg.graph = GraphPtr(new ImmutableGraph(subcoo));
  return sg;
}

std::vector<IdArray> ImmutableGraph::GetAdj(bool transpose, const std::string &fmt) const {
  // TODO(minjie): Our current semantics of adjacency matrix is row for dst nodes and col for
  //   src nodes. Therefore, we need to flip the transpose flag. For example, transpose=False
  //   is equal to in edge CSR.
  //   We have this behavior because previously we use framework's SPMM and we don't cache
  //   reverse adj. This is not intuitive and also not consistent with networkx's
  //   to_scipy_sparse_matrix. With the upcoming custom kernel change, we should change the
  //   behavior and make row for src and col for dst.
  if (fmt == std::string("csr")) {
    return transpose? GetOutCSR()->GetAdj(false, "csr") : GetInCSR()->GetAdj(false, "csr");
  } else if (fmt == std::string("coo")) {
    return GetCOO()->GetAdj(!transpose, fmt);
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format: " << fmt;
    return {};
  }
}

ImmutableGraphPtr ImmutableGraph::CreateFromCSR(
    IdArray indptr, IdArray indices, IdArray edge_ids, const std::string &edge_dir) {
  CSRPtr csr(new CSR(indptr, indices, edge_ids));
  if (edge_dir == "in") {
    return ImmutableGraphPtr(new ImmutableGraph(csr, nullptr));
  } else if (edge_dir == "out") {
    return ImmutableGraphPtr(new ImmutableGraph(nullptr, csr));
  } else {
    LOG(FATAL) << "Unknown edge direction: " << edge_dir;
    return ImmutableGraphPtr();
  }
}

ImmutableGraphPtr ImmutableGraph::CreateFromCSR(const std::string &name) {
  // If the shared memory graph index doesn't exist, we return null directly.
#ifndef _WIN32
  if (!SharedMemory::Exist(GetSharedMemName(name, "meta"))) {
    return nullptr;
  }
#endif  // _WIN32
  GraphIndexMetadata meta = DeserializeMetadata(GetSharedMemName(name, "meta"));
  CSRPtr in_csr, out_csr;
  if (meta.has_in_csr) {
    in_csr = CSRPtr(new CSR(GetSharedMemName(name, "in"), meta.num_nodes, meta.num_edges));
  }
  if (meta.has_out_csr) {
    out_csr = CSRPtr(new CSR(GetSharedMemName(name, "out"), meta.num_nodes, meta.num_edges));
  }
  return ImmutableGraphPtr(new ImmutableGraph(in_csr, out_csr, name));
}

ImmutableGraphPtr ImmutableGraph::CreateFromCOO(
    int64_t num_vertices, IdArray src, IdArray dst,
    bool row_sorted, bool col_sorted) {
  COOPtr coo(new COO(num_vertices, src, dst, row_sorted, col_sorted));
  return std::make_shared<ImmutableGraph>(coo);
}

ImmutableGraphPtr ImmutableGraph::ToImmutable(GraphPtr graph) {
  ImmutableGraphPtr ig = std::dynamic_pointer_cast<ImmutableGraph>(graph);
  if (ig) {
    return ig;
  } else {
    const auto& adj = graph->GetAdj(true, "csr");
    CSRPtr csr(new CSR(adj[0], adj[1], adj[2]));
    return ImmutableGraph::CreateFromCSR(adj[0], adj[1], adj[2], "out");
  }
}

ImmutableGraphPtr ImmutableGraph::CopyTo(ImmutableGraphPtr g, const DLContext& ctx) {
  if (ctx == g->Context()) {
    return g;
  }
  // TODO(minjie): since we don't have GPU implementation of COO<->CSR,
  //   we make sure that this graph (on CPU) has materialized CSR,
  //   and then copy them to other context (usually GPU). This should
  //   be fixed later.
  CSRPtr new_incsr = CSRPtr(new CSR(g->GetInCSR()->CopyTo(ctx)));
  CSRPtr new_outcsr = CSRPtr(new CSR(g->GetOutCSR()->CopyTo(ctx)));
  return ImmutableGraphPtr(new ImmutableGraph(new_incsr, new_outcsr));
}

ImmutableGraphPtr ImmutableGraph::CopyToSharedMem(ImmutableGraphPtr g, const std::string &name) {
  CSRPtr new_incsr, new_outcsr;
  std::string shared_mem_name = GetSharedMemName(name, "in");
  new_incsr = CSRPtr(new CSR(g->GetInCSR()->CopyToSharedMem(shared_mem_name)));

  shared_mem_name = GetSharedMemName(name, "out");
  new_outcsr = CSRPtr(new CSR(g->GetOutCSR()->CopyToSharedMem(shared_mem_name)));

  auto new_g = ImmutableGraphPtr(new ImmutableGraph(new_incsr, new_outcsr, name));
  new_g->serialized_shared_meta_ = SerializeMetadata(new_g, GetSharedMemName(name, "meta"));
  return new_g;
}

ImmutableGraphPtr ImmutableGraph::AsNumBits(ImmutableGraphPtr g, uint8_t bits) {
  if (g->NumBits() == bits) {
    return g;
  } else {
    // TODO(minjie): since we don't have int32 operations,
    //   we make sure that this graph (on CPU) has materialized CSR,
    //   and then copy them to other context (usually GPU). This should
    //   be fixed later.
    CSRPtr new_incsr = CSRPtr(new CSR(g->GetInCSR()->AsNumBits(bits)));
    CSRPtr new_outcsr = CSRPtr(new CSR(g->GetOutCSR()->AsNumBits(bits)));
    return ImmutableGraphPtr(new ImmutableGraph(new_incsr, new_outcsr));
  }
}

ImmutableGraphPtr ImmutableGraph::Reverse() const {
  if (coo_) {
    return ImmutableGraphPtr(new ImmutableGraph(
          out_csr_, in_csr_, coo_->Transpose()));
  } else {
    return ImmutableGraphPtr(new ImmutableGraph(out_csr_, in_csr_));
  }
}

constexpr uint64_t kDGLSerialize_ImGraph = 0xDD3c5FFE20046ABF;

/*! \return Load HeteroGraph from stream, using OutCSR Matrix*/
bool ImmutableGraph::Load(dmlc::Stream *fs) {
  uint64_t magicNum;
  aten::CSRMatrix out_csr_matrix;
  CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
  CHECK_EQ(magicNum, kDGLSerialize_ImGraph)
      << "Invalid ImmutableGraph Magic Number";
  CHECK(fs->Read(&out_csr_)) << "Invalid csr matrix";
  return true;
}

/*! \return Save HeteroGraph to stream, using OutCSR Matrix */
void ImmutableGraph::Save(dmlc::Stream *fs) const {
  fs->Write(kDGLSerialize_ImGraph);
  fs->Write(GetOutCSR());
}

HeteroGraphPtr ImmutableGraph::AsHeteroGraph() const {
  aten::CSRMatrix in_csr, out_csr;
  aten::COOMatrix coo;

  if (in_csr_)
    in_csr = GetInCSR()->ToCSRMatrix();
  if (out_csr_)
    out_csr = GetOutCSR()->ToCSRMatrix();
  if (coo_)
    coo = GetCOO()->ToCOOMatrix();

  auto g = UnitGraph::CreateHomographFrom(
      in_csr, out_csr, coo,
      in_csr_ != nullptr,
      out_csr_ != nullptr,
      coo_ != nullptr);
  return HeteroGraphPtr(new HeteroGraph(g->meta_graph(), {g}));
}

DGL_REGISTER_GLOBAL("transform._CAPI_DGLAsHeteroGraph")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    ImmutableGraphPtr ig = std::dynamic_pointer_cast<ImmutableGraph>(g.sptr());
    CHECK(ig) << "graph is not readonly";
    *rv = HeteroGraphRef(ig->AsHeteroGraph());
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLImmutableGraphCopyTo")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    const int device_type = args[1];
    const int device_id = args[2];
    DLContext ctx;
    ctx.device_type = static_cast<DLDeviceType>(device_type);
    ctx.device_id = device_id;
    ImmutableGraphPtr ig = CHECK_NOTNULL(std::dynamic_pointer_cast<ImmutableGraph>(g.sptr()));
    *rv = ImmutableGraph::CopyTo(ig, ctx);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLImmutableGraphCopyToSharedMem")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    std::string name = args[1];
    ImmutableGraphPtr ig = CHECK_NOTNULL(std::dynamic_pointer_cast<ImmutableGraph>(g.sptr()));
    *rv = ImmutableGraph::CopyToSharedMem(ig, name);
  });

DGL_REGISTER_GLOBAL("graph_index._CAPI_DGLImmutableGraphAsNumBits")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    GraphRef g = args[0];
    int bits = args[1];
    ImmutableGraphPtr ig = CHECK_NOTNULL(std::dynamic_pointer_cast<ImmutableGraph>(g.sptr()));
    *rv = ImmutableGraph::AsNumBits(ig, bits);
  });

}  // namespace dgl
