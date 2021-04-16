/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/unit_graph.cc
 * \brief UnitGraph graph implementation
 */
#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/immutable_graph.h>
#include <dgl/lazy.h>

#include "../c_api_common.h"
#include "./unit_graph.h"

namespace dgl {

namespace {

using namespace dgl::aten;

// create metagraph of one node type
inline GraphPtr CreateUnitGraphMetaGraph1() {
  // a self-loop edge 0->0
  std::vector<int64_t> row_vec(1, 0);
  std::vector<int64_t> col_vec(1, 0);
  IdArray row = aten::VecToIdArray(row_vec);
  IdArray col = aten::VecToIdArray(col_vec);
  GraphPtr g = ImmutableGraph::CreateFromCOO(1, row, col);
  return g;
}

// create metagraph of two node types
inline GraphPtr CreateUnitGraphMetaGraph2() {
  // an edge 0->1
  std::vector<int64_t> row_vec(1, 0);
  std::vector<int64_t> col_vec(1, 1);
  IdArray row = aten::VecToIdArray(row_vec);
  IdArray col = aten::VecToIdArray(col_vec);
  GraphPtr g = ImmutableGraph::CreateFromCOO(2, row, col);
  return g;
}

inline GraphPtr CreateUnitGraphMetaGraph(int num_vtypes) {
  static GraphPtr mg1 = CreateUnitGraphMetaGraph1();
  static GraphPtr mg2 = CreateUnitGraphMetaGraph2();
  if (num_vtypes == 1)
    return mg1;
  else if (num_vtypes == 2)
    return mg2;
  else
    LOG(FATAL) << "Invalid number of vertex types. Must be 1 or 2.";
  return {};
}

};  // namespace

//////////////////////////////////////////////////////////
//
// COO graph implementation
//
//////////////////////////////////////////////////////////

class UnitGraph::COO : public BaseHeteroGraph {
 public:
  COO(GraphPtr metagraph, int64_t num_src, int64_t num_dst, IdArray src,
      IdArray dst, bool row_sorted = false, bool col_sorted = false)
    : BaseHeteroGraph(metagraph) {
    CHECK(aten::IsValidIdArray(src));
    CHECK(aten::IsValidIdArray(dst));
    CHECK_EQ(src->shape[0], dst->shape[0]) << "Input arrays should have the same length.";
    adj_ = aten::COOMatrix{num_src, num_dst, src, dst,
        NullArray(),
        row_sorted, col_sorted};
  }

  COO(GraphPtr metagraph, const aten::COOMatrix& coo)
    : BaseHeteroGraph(metagraph), adj_(coo) {
    // Data index should not be inherited. Edges in COO format are always
    // assigned ids from 0 to num_edges - 1.
    CHECK(!COOHasData(coo)) << "[BUG] COO should not contain data.";
    adj_.data = aten::NullArray();
  }

  COO() {
    // set magic num_rows/num_cols to mark it as undefined
    // adj_.num_rows == 0 and adj_.num_cols == 0 means empty UnitGraph which is supported
    adj_.num_rows = -1;
    adj_.num_cols = -1;
  };

  bool defined() const {
    return (adj_.num_rows >= 0) && (adj_.num_cols >= 0);
  }

  inline dgl_type_t SrcType() const {
    return 0;
  }

  inline dgl_type_t DstType() const {
    return NumVertexTypes() == 1? 0 : 1;
  }

  inline dgl_type_t EdgeType() const {
    return 0;
  }

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "The method shouldn't be called for UnitGraph graph. "
      << "The relation graph is simply this graph itself.";
    return {};
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  DLDataType DataType() const override {
    return adj_.row->dtype;
  }

  DLContext Context() const override {
    return adj_.row->ctx;
  }

  uint8_t NumBits() const override {
    return adj_.row->dtype.bits;
  }

  COO AsNumBits(uint8_t bits) const {
    if (NumBits() == bits)
      return *this;

    COO ret(
        meta_graph_,
        adj_.num_rows, adj_.num_cols,
        aten::AsNumBits(adj_.row, bits),
        aten::AsNumBits(adj_.col, bits));
    return ret;
  }

  COO CopyTo(const DLContext& ctx) const {
    if (Context() == ctx)
      return *this;
    return COO(meta_graph_, adj_.CopyTo(ctx));
  }

  bool IsMultigraph() const override {
    return aten::COOHasDuplicate(adj_);
  }

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override {
    if (vtype == SrcType()) {
      return adj_.num_rows;
    } else if (vtype == DstType()) {
      return adj_.num_cols;
    } else {
      LOG(FATAL) << "Invalid vertex type: " << vtype;
      return 0;
    }
  }

  uint64_t NumEdges(dgl_type_t etype) const override {
    return adj_.row->shape[0];
  }

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override {
    return vid < NumVertices(vtype);
  }

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return {};
  }

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(DstType(), dst)) << "Invalid dst vertex id: " << dst;
    return aten::COOIsNonZero(adj_, src, dst);
  }

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override {
    CHECK(aten::IsValidIdArray(src_ids)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dst_ids)) << "Invalid vertex id array.";
    return aten::COOIsNonZero(adj_, src_ids, dst_ids);
  }

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override {
    CHECK(HasVertex(DstType(), dst)) << "Invalid dst vertex id: " << dst;
    return aten::COOGetRowDataAndIndices(aten::COOTranspose(adj_), dst).second;
  }

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    return aten::COOGetRowDataAndIndices(adj_, src).second;
  }

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(DstType(), dst)) << "Invalid dst vertex id: " << dst;
    return aten::COOGetAllData(adj_, src, dst);
  }

  EdgeArray EdgeIdsAll(dgl_type_t etype, IdArray src, IdArray dst) const override {
    CHECK(aten::IsValidIdArray(src)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dst)) << "Invalid vertex id array.";
    const auto& arrs = aten::COOGetDataAndIndices(adj_, src, dst);
    return EdgeArray{arrs[0], arrs[1], arrs[2]};
  }

  IdArray EdgeIdsOne(dgl_type_t etype, IdArray src, IdArray dst) const override {
    return aten::COOGetData(adj_, src, dst);
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override {
    CHECK(eid < NumEdges(etype)) << "Invalid edge id: " << eid;
    const dgl_id_t src = aten::IndexSelect<int64_t>(adj_.row, eid);
    const dgl_id_t dst = aten::IndexSelect<int64_t>(adj_.col, eid);
    return std::pair<dgl_id_t, dgl_id_t>(src, dst);
  }

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override {
    CHECK(aten::IsValidIdArray(eids)) << "Invalid edge id array";
    BUG_ON(aten::IsNullArray(adj_.data)) <<
      "FindEdges requires the internal COO matrix not having EIDs.";
    return EdgeArray{aten::IndexSelect(adj_.row, eids),
                     aten::IndexSelect(adj_.col, eids),
                     eids};
  }

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override {
    IdArray ret_src, ret_eid;
    std::tie(ret_eid, ret_src) = aten::COOGetRowDataAndIndices(
        aten::COOTranspose(adj_), vid);
    IdArray ret_dst = aten::Full(vid, ret_src->shape[0], NumBits(), ret_src->ctx);
    return EdgeArray{ret_src, ret_dst, ret_eid};
  }

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    auto coosubmat = aten::COOSliceRows(aten::COOTranspose(adj_), vids);
    auto row = aten::IndexSelect(vids, coosubmat.row);
    return EdgeArray{coosubmat.col, row, coosubmat.data};
  }

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override {
    IdArray ret_dst, ret_eid;
    std::tie(ret_eid, ret_dst) = aten::COOGetRowDataAndIndices(adj_, vid);
    IdArray ret_src = aten::Full(vid, ret_dst->shape[0], NumBits(), ret_dst->ctx);
    return EdgeArray{ret_src, ret_dst, ret_eid};
  }

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    auto coosubmat = aten::COOSliceRows(adj_, vids);
    auto row = aten::IndexSelect(vids, coosubmat.row);
    return EdgeArray{row, coosubmat.col, coosubmat.data};
  }

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override {
    CHECK(order.empty() || order == std::string("eid"))
      << "COO only support Edges of order \"eid\", but got \""
      << order << "\".";
    IdArray rst_eid = aten::Range(0, NumEdges(etype), NumBits(), Context());
    return EdgeArray{adj_.row, adj_.col, rst_eid};
  }

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override {
    CHECK(HasVertex(DstType(), vid)) << "Invalid dst vertex id: " << vid;
    return aten::COOGetRowNNZ(aten::COOTranspose(adj_), vid);
  }

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    return aten::COOGetRowNNZ(aten::COOTranspose(adj_), vids);
  }

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override {
    CHECK(HasVertex(SrcType(), vid)) << "Invalid src vertex id: " << vid;
    return aten::COOGetRowNNZ(adj_, vid);
  }

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    return aten::COOGetRowNNZ(adj_, vids);
  }

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(INFO) << "Not enabled for COO graph.";
    return {};
  }

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override {
    CHECK(fmt == "coo") << "Not valid adj format request.";
    if (transpose) {
      return {aten::HStack(adj_.col, adj_.row)};
    } else {
      return {aten::HStack(adj_.row, adj_.col)};
    }
  }

  aten::COOMatrix GetCOOMatrix(dgl_type_t etype) const override {
    return adj_;
  }

  aten::CSRMatrix GetCSCMatrix(dgl_type_t etype) const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return aten::CSRMatrix();
  }

  aten::CSRMatrix GetCSRMatrix(dgl_type_t etype) const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return aten::CSRMatrix();
  }

  SparseFormat SelectFormat(dgl_type_t etype, dgl_format_code_t preferred_formats) const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return SparseFormat::kCOO;
  }

  dgl_format_code_t GetAllowedFormats() const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return 0;
  }

  dgl_format_code_t GetCreatedFormats() const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return 0;
  }

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override {
    CHECK_EQ(vids.size(), NumVertexTypes()) << "Number of vertex types mismatch";
    auto srcvids = vids[SrcType()], dstvids = vids[DstType()];
    CHECK(aten::IsValidIdArray(srcvids)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dstvids)) << "Invalid vertex id array.";
    HeteroSubgraph subg;
    const auto& submat = aten::COOSliceMatrix(adj_, srcvids, dstvids);
    IdArray sub_eids = aten::Range(0, submat.data->shape[0], NumBits(), Context());
    subg.graph = std::make_shared<COO>(meta_graph(), submat.num_rows, submat.num_cols,
        submat.row, submat.col);
    subg.induced_vertices = vids;
    subg.induced_edges.emplace_back(submat.data);
    return subg;
  }

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override {
    CHECK_EQ(eids.size(), 1) << "Edge type number mismatch.";
    HeteroSubgraph subg;
    if (!preserve_nodes) {
      IdArray new_src = aten::IndexSelect(adj_.row, eids[0]);
      IdArray new_dst = aten::IndexSelect(adj_.col, eids[0]);
      subg.induced_vertices.emplace_back(aten::Relabel_({new_src}));
      subg.induced_vertices.emplace_back(aten::Relabel_({new_dst}));
      const auto new_nsrc = subg.induced_vertices[0]->shape[0];
      const auto new_ndst = subg.induced_vertices[1]->shape[0];
      subg.graph = std::make_shared<COO>(
          meta_graph(), new_nsrc, new_ndst, new_src, new_dst);
      subg.induced_edges = eids;
    } else {
      IdArray new_src = aten::IndexSelect(adj_.row, eids[0]);
      IdArray new_dst = aten::IndexSelect(adj_.col, eids[0]);
      subg.induced_vertices.emplace_back(
          aten::Range(0, NumVertices(SrcType()), NumBits(), Context()));
      subg.induced_vertices.emplace_back(
          aten::Range(0, NumVertices(DstType()), NumBits(), Context()));
      subg.graph = std::make_shared<COO>(
          meta_graph(), NumVertices(SrcType()), NumVertices(DstType()), new_src, new_dst);
      subg.induced_edges = eids;
    }
    return subg;
  }

  HeteroGraphPtr GetGraphInFormat(dgl_format_code_t formats) const override {
    LOG(FATAL) << "Not enabled for COO graph.";
    return nullptr;
  }

  aten::COOMatrix adj() const {
    return adj_;
  }

  /*!
   * \brief Determines whether the graph is "hypersparse", i.e. having significantly more
   * nodes than edges.
   */
  bool IsHypersparse() const {
    return (NumVertices(SrcType()) / 8 > NumEdges(EdgeType())) &&
           (NumVertices(SrcType()) > 1000000);
  }

  bool Load(dmlc::Stream* fs) {
    auto meta_imgraph = Serializer::make_shared<ImmutableGraph>();
    CHECK(fs->Read(&meta_imgraph)) << "Invalid meta graph";
    meta_graph_ = meta_imgraph;
    CHECK(fs->Read(&adj_)) << "Invalid adj matrix";
    return true;
  }
  void Save(dmlc::Stream* fs) const {
    auto meta_graph_ptr = ImmutableGraph::ToImmutable(meta_graph());
    fs->Write(meta_graph_ptr);
    fs->Write(adj_);
  }

 private:
  friend class Serializer;

  /*! \brief internal adjacency matrix. Data array is empty */
  aten::COOMatrix adj_;
};

//////////////////////////////////////////////////////////
//
// CSR graph implementation
//
//////////////////////////////////////////////////////////

/*! \brief CSR graph */
class UnitGraph::CSR : public BaseHeteroGraph {
 public:
  CSR(GraphPtr metagraph, int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids)
    : BaseHeteroGraph(metagraph) {
    CHECK(aten::IsValidIdArray(indptr));
    CHECK(aten::IsValidIdArray(indices));
    CHECK(aten::IsValidIdArray(edge_ids));
    CHECK_EQ(indices->shape[0], edge_ids->shape[0])
      << "indices and edge id arrays should have the same length";

    adj_ = aten::CSRMatrix{num_src, num_dst, indptr, indices, edge_ids};
  }

  CSR(GraphPtr metagraph, const aten::CSRMatrix& csr)
    : BaseHeteroGraph(metagraph), adj_(csr) {
  }

  CSR() {
    // set magic num_rows/num_cols to mark it as undefined
    // adj_.num_rows == 0 and adj_.num_cols == 0 means empty UnitGraph which is supported
    adj_.num_rows = -1;
    adj_.num_cols = -1;
  };

  bool defined() const {
    return (adj_.num_rows >= 0) || (adj_.num_cols >= 0);
  }

  inline dgl_type_t SrcType() const {
    return 0;
  }

  inline dgl_type_t DstType() const {
    return NumVertexTypes() == 1? 0 : 1;
  }

  inline dgl_type_t EdgeType() const {
    return 0;
  }

  HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const override {
    LOG(FATAL) << "The method shouldn't be called for UnitGraph graph. "
      << "The relation graph is simply this graph itself.";
    return {};
  }

  void AddVertices(dgl_type_t vtype, uint64_t num_vertices) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  void Clear() override {
    LOG(FATAL) << "UnitGraph graph is not mutable.";
  }

  DLDataType DataType() const override {
    return adj_.indices->dtype;
  }

  DLContext Context() const override {
    return adj_.indices->ctx;
  }

  uint8_t NumBits() const override {
    return adj_.indices->dtype.bits;
  }

  CSR AsNumBits(uint8_t bits) const {
    if (NumBits() == bits) {
      return *this;
    } else {
      CSR ret(
          meta_graph_,
          adj_.num_rows, adj_.num_cols,
          aten::AsNumBits(adj_.indptr, bits),
          aten::AsNumBits(adj_.indices, bits),
          aten::AsNumBits(adj_.data, bits));
      return ret;
    }
  }

  CSR CopyTo(const DLContext& ctx) const {
    if (Context() == ctx) {
      return *this;
    } else {
      return CSR(meta_graph_, adj_.CopyTo(ctx));
    }
  }

  bool IsMultigraph() const override {
    return aten::CSRHasDuplicate(adj_);
  }

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override {
    if (vtype == SrcType()) {
      return adj_.num_rows;
    } else if (vtype == DstType()) {
      return adj_.num_cols;
    } else {
      LOG(FATAL) << "Invalid vertex type: " << vtype;
      return 0;
    }
  }

  uint64_t NumEdges(dgl_type_t etype) const override {
    return adj_.indices->shape[0];
  }

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override {
    return vid < NumVertices(vtype);
  }

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return {};
  }

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(DstType(), dst)) << "Invalid dst vertex id: " << dst;
    return aten::CSRIsNonZero(adj_, src, dst);
  }

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override {
    CHECK(aten::IsValidIdArray(src_ids)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dst_ids)) << "Invalid vertex id array.";
    return aten::CSRIsNonZero(adj_, src_ids, dst_ids);
  }

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override {
    LOG(INFO) << "Not enabled for CSR graph.";
    return {};
  }

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    return aten::CSRGetRowColumnIndices(adj_, src);
  }

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override {
    CHECK(HasVertex(SrcType(), src)) << "Invalid src vertex id: " << src;
    CHECK(HasVertex(DstType(), dst)) << "Invalid dst vertex id: " << dst;
    return aten::CSRGetAllData(adj_, src, dst);
  }

  EdgeArray EdgeIdsAll(dgl_type_t etype, IdArray src, IdArray dst) const override {
    CHECK(aten::IsValidIdArray(src)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dst)) << "Invalid vertex id array.";
    const auto& arrs = aten::CSRGetDataAndIndices(adj_, src, dst);
    return EdgeArray{arrs[0], arrs[1], arrs[2]};
  }

  IdArray EdgeIdsOne(dgl_type_t etype, IdArray src, IdArray dst) const override {
    return aten::CSRGetData(adj_, src, dst);
  }

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override {
    CHECK(HasVertex(SrcType(), vid)) << "Invalid src vertex id: " << vid;
    IdArray ret_dst = aten::CSRGetRowColumnIndices(adj_, vid);
    IdArray ret_eid = aten::CSRGetRowData(adj_, vid);
    IdArray ret_src = aten::Full(vid, ret_dst->shape[0], NumBits(), ret_dst->ctx);
    return EdgeArray{ret_src, ret_dst, ret_eid};
  }

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    auto csrsubmat = aten::CSRSliceRows(adj_, vids);
    auto coosubmat = aten::CSRToCOO(csrsubmat, false);
    // Note that the row id in the csr submat is relabled, so
    // we need to recover it using an index select.
    auto row = aten::IndexSelect(vids, coosubmat.row);
    return EdgeArray{row, coosubmat.col, coosubmat.data};
  }

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override {
    CHECK(order.empty() || order == std::string("srcdst"))
      << "CSR only support Edges of order \"srcdst\","
      << " but got \"" << order << "\".";
    auto coo = aten::CSRToCOO(adj_, false);
    if (order == std::string("srcdst")) {
      // make sure the coo is sorted if an order is requested
      coo = aten::COOSort(coo, true);
    }
    return EdgeArray{coo.row, coo.col, coo.data};
  }

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override {
    CHECK(HasVertex(SrcType(), vid)) << "Invalid src vertex id: " << vid;
    return aten::CSRGetRowNNZ(adj_, vid);
  }

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override {
    CHECK(aten::IsValidIdArray(vids)) << "Invalid vertex id array.";
    return aten::CSRGetRowNNZ(adj_, vids);
  }

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override {
    // TODO(minjie): This still assumes the data type and device context
    //   of this graph. Should fix later.
    CHECK_EQ(NumBits(), 64);
    const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
    const dgl_id_t* indices_data = static_cast<dgl_id_t*>(adj_.indices->data);
    const dgl_id_t start = indptr_data[vid];
    const dgl_id_t end = indptr_data[vid + 1];
    return DGLIdIters(indices_data + start, indices_data + end);
  }

  DGLIdIters32 SuccVec32(dgl_type_t etype, dgl_id_t vid) {
    // TODO(minjie): This still assumes the data type and device context
    //   of this graph. Should fix later.
    const int32_t* indptr_data = static_cast<int32_t*>(adj_.indptr->data);
    const int32_t* indices_data = static_cast<int32_t*>(adj_.indices->data);
    const int32_t start = indptr_data[vid];
    const int32_t end = indptr_data[vid + 1];
    return DGLIdIters32(indices_data + start, indices_data + end);
  }

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    // TODO(minjie): This still assumes the data type and device context
    //   of this graph. Should fix later.
    CHECK_EQ(NumBits(), 64);
    const dgl_id_t* indptr_data = static_cast<dgl_id_t*>(adj_.indptr->data);
    const dgl_id_t* eid_data = static_cast<dgl_id_t*>(adj_.data->data);
    const dgl_id_t start = indptr_data[vid];
    const dgl_id_t end = indptr_data[vid + 1];
    return DGLIdIters(eid_data + start, eid_data + end);
  }

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override {
    CHECK(!transpose && fmt == "csr") << "Not valid adj format request.";
    return {adj_.indptr, adj_.indices, adj_.data};
  }

  aten::COOMatrix GetCOOMatrix(dgl_type_t etype) const override {
    LOG(FATAL) << "Not enabled for CSR graph";
    return aten::COOMatrix();
  }

  aten::CSRMatrix GetCSCMatrix(dgl_type_t etype) const override {
    LOG(FATAL) << "Not enabled for CSR graph";
    return aten::CSRMatrix();
  }

  aten::CSRMatrix GetCSRMatrix(dgl_type_t etype) const override {
    return adj_;
  }

  SparseFormat SelectFormat(dgl_type_t etype, dgl_format_code_t preferred_formats) const override {
    LOG(FATAL) << "Not enabled for CSR graph";
    return SparseFormat::kCSR;
  }

  dgl_format_code_t GetAllowedFormats() const override {
    LOG(FATAL) << "Not enabled for COO graph";
    return 0;
  }

  dgl_format_code_t GetCreatedFormats() const override {
    LOG(FATAL) << "Not enabled for CSR graph";
    return 0;
  }

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override {
    CHECK_EQ(vids.size(), NumVertexTypes()) << "Number of vertex types mismatch";
    auto srcvids = vids[SrcType()], dstvids = vids[DstType()];
    CHECK(aten::IsValidIdArray(srcvids)) << "Invalid vertex id array.";
    CHECK(aten::IsValidIdArray(dstvids)) << "Invalid vertex id array.";
    HeteroSubgraph subg;
    const auto& submat = aten::CSRSliceMatrix(adj_, srcvids, dstvids);
    IdArray sub_eids = aten::Range(0, submat.data->shape[0], NumBits(), Context());
    subg.graph = std::make_shared<CSR>(meta_graph(), submat.num_rows, submat.num_cols,
        submat.indptr, submat.indices, sub_eids);
    subg.induced_vertices = vids;
    subg.induced_edges.emplace_back(submat.data);
    return subg;
  }

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return {};
  }

  HeteroGraphPtr GetGraphInFormat(dgl_format_code_t formats) const override {
    LOG(FATAL) << "Not enabled for CSR graph.";
    return nullptr;
  }

  aten::CSRMatrix adj() const {
    return adj_;
  }

  bool Load(dmlc::Stream* fs) {
    auto meta_imgraph = Serializer::make_shared<ImmutableGraph>();
    CHECK(fs->Read(&meta_imgraph)) << "Invalid meta graph";
    meta_graph_ = meta_imgraph;
    CHECK(fs->Read(&adj_)) << "Invalid adj matrix";
    return true;
  }
  void Save(dmlc::Stream* fs) const {
    auto meta_graph_ptr = ImmutableGraph::ToImmutable(meta_graph());
    fs->Write(meta_graph_ptr);
    fs->Write(adj_);
  }

 private:
  friend class Serializer;

  /*! \brief internal adjacency matrix. Data array stores edge ids */
  aten::CSRMatrix adj_;
};

//////////////////////////////////////////////////////////
//
// unit graph implementation
//
//////////////////////////////////////////////////////////

DLDataType UnitGraph::DataType() const {
  return GetAny()->DataType();
}

DLContext UnitGraph::Context() const {
  return GetAny()->Context();
}

uint8_t UnitGraph::NumBits() const {
  return GetAny()->NumBits();
}

bool UnitGraph::IsMultigraph() const {
  const SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->IsMultigraph();
}

uint64_t UnitGraph::NumVertices(dgl_type_t vtype) const {
  const SparseFormat fmt = SelectFormat(ALL_CODE);
  const auto ptr = GetFormat(fmt);
  // TODO(BarclayII): we have a lot of special handling for CSC.
  // Need to have a UnitGraph::CSC backend instead.
  if (fmt == SparseFormat::kCSC)
    vtype = (vtype == SrcType()) ? DstType() : SrcType();
  return ptr->NumVertices(vtype);
}

uint64_t UnitGraph::NumEdges(dgl_type_t etype) const {
  return GetAny()->NumEdges(etype);
}

bool UnitGraph::HasVertex(dgl_type_t vtype, dgl_id_t vid) const {
  const SparseFormat fmt = SelectFormat(ALL_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    vtype = (vtype == SrcType()) ? DstType() : SrcType();
  return ptr->HasVertex(vtype, vid);
}

BoolArray UnitGraph::HasVertices(dgl_type_t vtype, IdArray vids) const {
  CHECK(aten::IsValidIdArray(vids)) << "Invalid id array input";
  return aten::LT(vids, NumVertices(vtype));
}

bool UnitGraph::HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const {
  const SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->HasEdgeBetween(etype, dst, src);
  else
    return ptr->HasEdgeBetween(etype, src, dst);
}

BoolArray UnitGraph::HasEdgesBetween(
    dgl_type_t etype, IdArray src, IdArray dst) const {
  const SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->HasEdgesBetween(etype, dst, src);
  else
    return ptr->HasEdgesBetween(etype, src, dst);
}

IdArray UnitGraph::Predecessors(dgl_type_t etype, dgl_id_t dst) const {
  const SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->Successors(etype, dst);
  else
    return ptr->Predecessors(etype, dst);
}

IdArray UnitGraph::Successors(dgl_type_t etype, dgl_id_t src) const {
  const SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->Successors(etype, src);
}

IdArray UnitGraph::EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const {
  const SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->EdgeId(etype, dst, src);
  else
    return ptr->EdgeId(etype, src, dst);
}

EdgeArray UnitGraph::EdgeIdsAll(dgl_type_t etype, IdArray src, IdArray dst) const {
  const SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC) {
    EdgeArray edges = ptr->EdgeIdsAll(etype, dst, src);
    return EdgeArray{edges.dst, edges.src, edges.id};
  } else {
    return ptr->EdgeIdsAll(etype, src, dst);
  }
}

IdArray UnitGraph::EdgeIdsOne(dgl_type_t etype, IdArray src, IdArray dst) const {
  const SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC) {
    return ptr->EdgeIdsOne(etype, dst, src);
  } else {
    return ptr->EdgeIdsOne(etype, src, dst);
  }
}

std::pair<dgl_id_t, dgl_id_t> UnitGraph::FindEdge(dgl_type_t etype, dgl_id_t eid) const {
  const SparseFormat fmt = SelectFormat(COO_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->FindEdge(etype, eid);
}

EdgeArray UnitGraph::FindEdges(dgl_type_t etype, IdArray eids) const {
  const SparseFormat fmt = SelectFormat(COO_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->FindEdges(etype, eids);
}

EdgeArray UnitGraph::InEdges(dgl_type_t etype, dgl_id_t vid) const {
  const SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC) {
    const EdgeArray& ret = ptr->OutEdges(etype, vid);
    return {ret.dst, ret.src, ret.id};
  } else {
    return ptr->InEdges(etype, vid);
  }
}

EdgeArray UnitGraph::InEdges(dgl_type_t etype, IdArray vids) const {
  const SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC) {
    const EdgeArray& ret = ptr->OutEdges(etype, vids);
    return {ret.dst, ret.src, ret.id};
  } else {
    return ptr->InEdges(etype, vids);
  }
}

EdgeArray UnitGraph::OutEdges(dgl_type_t etype, dgl_id_t vid) const {
  const SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->OutEdges(etype, vid);
}

EdgeArray UnitGraph::OutEdges(dgl_type_t etype, IdArray vids) const {
  const SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->OutEdges(etype, vids);
}

EdgeArray UnitGraph::Edges(dgl_type_t etype, const std::string &order) const {
  SparseFormat fmt;
  if (order == std::string("eid")) {
    fmt = SelectFormat(COO_CODE);
  } else if (order.empty()) {
    // arbitrary order
    fmt = SelectFormat(ALL_CODE);
  } else if (order == std::string("srcdst")) {
    fmt = SelectFormat(CSR_CODE);
  } else {
    LOG(FATAL) << "Unsupported order request: " << order;
    return {};
  }

  const auto& edges = GetFormat(fmt)->Edges(etype, order);
  if (fmt == SparseFormat::kCSC)
    return EdgeArray{edges.dst, edges.src, edges.id};
  else
    return edges;
}

uint64_t UnitGraph::InDegree(dgl_type_t etype, dgl_id_t vid) const {
  SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->OutDegree(etype, vid);
  else
    return ptr->InDegree(etype, vid);
}

DegreeArray UnitGraph::InDegrees(dgl_type_t etype, IdArray vids) const {
  SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->OutDegrees(etype, vids);
  else
    return ptr->InDegrees(etype, vids);
}

uint64_t UnitGraph::OutDegree(dgl_type_t etype, dgl_id_t vid) const {
  SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->OutDegree(etype, vid);
}

DegreeArray UnitGraph::OutDegrees(dgl_type_t etype, IdArray vids) const {
  SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->OutDegrees(etype, vids);
}

DGLIdIters UnitGraph::SuccVec(dgl_type_t etype, dgl_id_t vid) const {
  SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->SuccVec(etype, vid);
}

DGLIdIters32 UnitGraph::SuccVec32(dgl_type_t etype, dgl_id_t vid) const {
  SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = std::dynamic_pointer_cast<CSR>(GetFormat(fmt));
  CHECK_NOTNULL(ptr);
  return ptr->SuccVec32(etype, vid);
}

DGLIdIters UnitGraph::OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const {
  SparseFormat fmt = SelectFormat(CSR_CODE);
  const auto ptr = GetFormat(fmt);
  return ptr->OutEdgeVec(etype, vid);
}

DGLIdIters UnitGraph::PredVec(dgl_type_t etype, dgl_id_t vid) const {
  SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->SuccVec(etype, vid);
  else
    return ptr->PredVec(etype, vid);
}

DGLIdIters UnitGraph::InEdgeVec(dgl_type_t etype, dgl_id_t vid) const {
  SparseFormat fmt = SelectFormat(CSC_CODE);
  const auto ptr = GetFormat(fmt);
  if (fmt == SparseFormat::kCSC)
    return ptr->OutEdgeVec(etype, vid);
  else
    return ptr->InEdgeVec(etype, vid);
}

std::vector<IdArray> UnitGraph::GetAdj(
    dgl_type_t etype, bool transpose, const std::string &fmt) const {
  // TODO(minjie): Our current semantics of adjacency matrix is row for dst nodes and col for
  //   src nodes. Therefore, we need to flip the transpose flag. For example, transpose=False
  //   is equal to in edge CSR.
  //   We have this behavior because previously we use framework's SPMM and we don't cache
  //   reverse adj. This is not intuitive and also not consistent with networkx's
  //   to_scipy_sparse_matrix. With the upcoming custom kernel change, we should change the
  //   behavior and make row for src and col for dst.
  if (fmt == std::string("csr")) {
    return transpose? GetOutCSR()->GetAdj(etype, false, "csr")
      : GetInCSR()->GetAdj(etype, false, "csr");
  } else if (fmt == std::string("coo")) {
    return GetCOO()->GetAdj(etype, !transpose, fmt);
  } else {
    LOG(FATAL) << "unsupported adjacency matrix format: " << fmt;
    return {};
  }
}

HeteroSubgraph UnitGraph::VertexSubgraph(const std::vector<IdArray>& vids) const {
  // We prefer to generate a subgraph from out-csr.
  SparseFormat fmt = SelectFormat(CSR_CODE);
  HeteroSubgraph sg = GetFormat(fmt)->VertexSubgraph(vids);
  HeteroSubgraph ret;

  CSRPtr subcsr = nullptr;
  CSRPtr subcsc = nullptr;
  COOPtr subcoo = nullptr;
  switch (fmt) {
    case SparseFormat::kCSR:
      subcsr = std::dynamic_pointer_cast<CSR>(sg.graph);
      break;
    case SparseFormat::kCSC:
      subcsc = std::dynamic_pointer_cast<CSR>(sg.graph);
      break;
    case SparseFormat::kCOO:
      subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
      break;
    default:
      LOG(FATAL) << "[BUG] unsupported format " << static_cast<int>(fmt);
      return ret;
  }

  ret.graph = HeteroGraphPtr(new UnitGraph(meta_graph(), subcsc, subcsr, subcoo));
  ret.induced_vertices = std::move(sg.induced_vertices);
  ret.induced_edges = std::move(sg.induced_edges);
  return ret;
}

HeteroSubgraph UnitGraph::EdgeSubgraph(
    const std::vector<IdArray>& eids, bool preserve_nodes) const {
  SparseFormat fmt = SelectFormat(COO_CODE);
  auto sg = GetFormat(fmt)->EdgeSubgraph(eids, preserve_nodes);
  HeteroSubgraph ret;

  CSRPtr subcsr = nullptr;
  CSRPtr subcsc = nullptr;
  COOPtr subcoo = nullptr;
  switch (fmt) {
    case SparseFormat::kCSR:
      subcsr = std::dynamic_pointer_cast<CSR>(sg.graph);
      break;
    case SparseFormat::kCSC:
      subcsc = std::dynamic_pointer_cast<CSR>(sg.graph);
      break;
    case SparseFormat::kCOO:
      subcoo = std::dynamic_pointer_cast<COO>(sg.graph);
      break;
    default:
      LOG(FATAL) << "[BUG] unsupported format " << static_cast<int>(fmt);
      return ret;
  }

  ret.graph = HeteroGraphPtr(new UnitGraph(meta_graph(), subcsc, subcsr, subcoo));
  ret.induced_vertices = std::move(sg.induced_vertices);
  ret.induced_edges = std::move(sg.induced_edges);
  return ret;
}

HeteroGraphPtr UnitGraph::CreateFromCOO(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray row, IdArray col,
    bool row_sorted, bool col_sorted,
    dgl_format_code_t formats) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(num_src, num_dst);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  COOPtr coo(new COO(mg, num_src, num_dst, row, col,
      row_sorted, col_sorted));

  return HeteroGraphPtr(
      new UnitGraph(mg, nullptr, nullptr, coo, formats));
}

HeteroGraphPtr UnitGraph::CreateFromCOO(
    int64_t num_vtypes, const aten::COOMatrix& mat,
    dgl_format_code_t formats) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(mat.num_rows, mat.num_cols);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  COOPtr coo(new COO(mg, mat));

  return HeteroGraphPtr(
      new UnitGraph(mg, nullptr, nullptr, coo, formats));
}

HeteroGraphPtr UnitGraph::CreateFromCSR(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids, dgl_format_code_t formats) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(num_src, num_dst);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  CSRPtr csr(new CSR(mg, num_src, num_dst, indptr, indices, edge_ids));
  return HeteroGraphPtr(new UnitGraph(mg, nullptr, csr, nullptr, formats));
}

HeteroGraphPtr UnitGraph::CreateFromCSR(
    int64_t num_vtypes, const aten::CSRMatrix& mat,
    dgl_format_code_t formats) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(mat.num_rows, mat.num_cols);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  CSRPtr csr(new CSR(mg, mat));
  return HeteroGraphPtr(new UnitGraph(mg, nullptr, csr, nullptr, formats));
}

HeteroGraphPtr UnitGraph::CreateFromCSC(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids, dgl_format_code_t formats) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(num_src, num_dst);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  CSRPtr csc(new CSR(mg, num_src, num_dst, indptr, indices, edge_ids));
  return HeteroGraphPtr(new UnitGraph(mg, csc, nullptr, nullptr, formats));
}

HeteroGraphPtr UnitGraph::CreateFromCSC(
    int64_t num_vtypes, const aten::CSRMatrix& mat,
    dgl_format_code_t formats) {
  CHECK(num_vtypes == 1 || num_vtypes == 2);
  if (num_vtypes == 1)
    CHECK_EQ(mat.num_rows, mat.num_cols);
  auto mg = CreateUnitGraphMetaGraph(num_vtypes);
  CSRPtr csc(new CSR(mg, mat));
  return HeteroGraphPtr(new UnitGraph(mg, csc, nullptr, nullptr, formats));
}

HeteroGraphPtr UnitGraph::AsNumBits(HeteroGraphPtr g, uint8_t bits) {
  if (g->NumBits() == bits) {
    return g;
  } else {
    auto bg = std::dynamic_pointer_cast<UnitGraph>(g);
    CHECK_NOTNULL(bg);
    CSRPtr new_incsr =
      (bg->in_csr_->defined())? CSRPtr(new CSR(bg->in_csr_->AsNumBits(bits))) : nullptr;
    CSRPtr new_outcsr =
      (bg->out_csr_->defined())? CSRPtr(new CSR(bg->out_csr_->AsNumBits(bits))) : nullptr;
    COOPtr new_coo =
      (bg->coo_->defined())? COOPtr(new COO(bg->coo_->AsNumBits(bits))) : nullptr;
    return HeteroGraphPtr(
        new UnitGraph(g->meta_graph(), new_incsr, new_outcsr, new_coo, bg->formats_));
  }
}

HeteroGraphPtr UnitGraph::CopyTo(HeteroGraphPtr g, const DLContext& ctx) {
  if (ctx == g->Context()) {
    return g;
  } else {
    auto bg = std::dynamic_pointer_cast<UnitGraph>(g);
    CHECK_NOTNULL(bg);
    CSRPtr new_incsr =
      (bg->in_csr_->defined())? CSRPtr(new CSR(bg->in_csr_->CopyTo(ctx))) : nullptr;
    CSRPtr new_outcsr =
      (bg->out_csr_->defined())? CSRPtr(new CSR(bg->out_csr_->CopyTo(ctx))) : nullptr;
    COOPtr new_coo =
      (bg->coo_->defined())? COOPtr(new COO(bg->coo_->CopyTo(ctx))) : nullptr;
    return HeteroGraphPtr(
        new UnitGraph(g->meta_graph(), new_incsr, new_outcsr, new_coo, bg->formats_));
  }
}

void UnitGraph::InvalidateCSR() {
  this->out_csr_ = CSRPtr(new CSR());
}

void UnitGraph::InvalidateCSC() {
  this->in_csr_ = CSRPtr(new CSR());
}

void UnitGraph::InvalidateCOO() {
  this->coo_ = COOPtr(new COO());
}

UnitGraph::UnitGraph(GraphPtr metagraph, CSRPtr in_csr, CSRPtr out_csr, COOPtr coo,
                     dgl_format_code_t formats)
  : BaseHeteroGraph(metagraph), in_csr_(in_csr), out_csr_(out_csr), coo_(coo) {
  if (!in_csr_) {
    in_csr_ = CSRPtr(new CSR());
  }
  if (!out_csr_) {
    out_csr_ = CSRPtr(new CSR());
  }
  if (!coo_) {
    coo_ = COOPtr(new COO());
  }
  formats_ = formats;
  dgl_format_code_t created = GetCreatedFormats();
  if ((formats | created) != formats)
    LOG(FATAL) << "Graph created from formats: " << CodeToStr(created) <<
      ", which is not compatible with available formats: " << CodeToStr(formats);
  CHECK(GetAny()) << "At least one graph structure should exist.";
}

HeteroGraphPtr UnitGraph::CreateHomographFrom(
    const aten::CSRMatrix &in_csr,
    const aten::CSRMatrix &out_csr,
    const aten::COOMatrix &coo,
    bool has_in_csr,
    bool has_out_csr,
    bool has_coo,
    dgl_format_code_t formats) {
  auto mg = CreateUnitGraphMetaGraph1();

  CSRPtr in_csr_ptr = nullptr;
  CSRPtr out_csr_ptr = nullptr;
  COOPtr coo_ptr = nullptr;

  if (has_in_csr)
    in_csr_ptr = CSRPtr(new CSR(mg, in_csr));
  else
    in_csr_ptr = CSRPtr(new CSR());
  if (has_out_csr)
    out_csr_ptr = CSRPtr(new CSR(mg, out_csr));
  else
    out_csr_ptr = CSRPtr(new CSR());
  if (has_coo)
    coo_ptr = COOPtr(new COO(mg, coo));
  else
    coo_ptr = COOPtr(new COO());

  return HeteroGraphPtr(new UnitGraph(mg, in_csr_ptr, out_csr_ptr, coo_ptr, formats));
}

UnitGraph::CSRPtr UnitGraph::GetInCSR(bool inplace) const {
  if (inplace)
    if (!(formats_ & CSC_CODE))
      LOG(FATAL) << "The graph have restricted sparse format " <<
        CodeToStr(formats_) << ", cannot create CSC matrix.";
  CSRPtr ret = in_csr_;
  // Prefers converting from COO since it is parallelized.
  // TODO(BarclayII): need benchmarking.
  if (!in_csr_->defined()) {
    if (coo_->defined()) {
      const auto& newadj = aten::COOToCSR(
            aten::COOTranspose(coo_->adj()));

      if (inplace)
        *(const_cast<UnitGraph*>(this)->in_csr_) = CSR(meta_graph(), newadj);
      else
        ret = std::make_shared<CSR>(meta_graph(), newadj);
    } else {
      CHECK(out_csr_->defined()) << "None of CSR, COO exist";
      const auto& newadj = aten::CSRTranspose(out_csr_->adj());

      if (inplace)
        *(const_cast<UnitGraph*>(this)->in_csr_) = CSR(meta_graph(), newadj);
      else
        ret = std::make_shared<CSR>(meta_graph(), newadj);
    }
  }
  return ret;
}

/* !\brief Return out csr. If not exist, transpose the other one.*/
UnitGraph::CSRPtr UnitGraph::GetOutCSR(bool inplace) const {
  if (inplace)
    if (!(formats_ & CSR_CODE))
      LOG(FATAL) << "The graph have restricted sparse format " <<
        CodeToStr(formats_) << ", cannot create CSR matrix.";
  CSRPtr ret = out_csr_;
  // Prefers converting from COO since it is parallelized.
  // TODO(BarclayII): need benchmarking.
  if (!out_csr_->defined()) {
    if (coo_->defined()) {
      const auto& newadj = aten::COOToCSR(coo_->adj());

      if (inplace)
        *(const_cast<UnitGraph*>(this)->out_csr_) = CSR(meta_graph(), newadj);
      else
        ret = std::make_shared<CSR>(meta_graph(), newadj);
    } else {
      CHECK(in_csr_->defined()) << "None of CSR, COO exist";
      const auto& newadj = aten::CSRTranspose(in_csr_->adj());

      if (inplace)
        *(const_cast<UnitGraph*>(this)->out_csr_) = CSR(meta_graph(), newadj);
      else
        ret = std::make_shared<CSR>(meta_graph(), newadj);
    }
  }
  return ret;
}

/* !\brief Return coo. If not exist, create from csr.*/
UnitGraph::COOPtr UnitGraph::GetCOO(bool inplace) const {
  if (inplace)
    if (!(formats_ & COO_CODE))
      LOG(FATAL) << "The graph have restricted sparse format " <<
        CodeToStr(formats_) << ", cannot create COO matrix.";
  COOPtr ret = coo_;
  if (!coo_->defined()) {
    if (in_csr_->defined()) {
      const auto& newadj = aten::COOTranspose(aten::CSRToCOO(in_csr_->adj(), true));

      if (inplace)
        *(const_cast<UnitGraph*>(this)->coo_) = COO(meta_graph(), newadj);
      else
        ret = std::make_shared<COO>(meta_graph(), newadj);
    } else {
      CHECK(out_csr_->defined()) << "Both CSR are missing.";
      const auto& newadj = aten::CSRToCOO(out_csr_->adj(), true);

      if (inplace)
        *(const_cast<UnitGraph*>(this)->coo_) = COO(meta_graph(), newadj);
      else
        ret = std::make_shared<COO>(meta_graph(), newadj);
    }
  }
  return ret;
}

aten::CSRMatrix UnitGraph::GetCSCMatrix(dgl_type_t etype) const {
  return GetInCSR()->adj();
}

aten::CSRMatrix UnitGraph::GetCSRMatrix(dgl_type_t etype) const {
  return GetOutCSR()->adj();
}

aten::COOMatrix UnitGraph::GetCOOMatrix(dgl_type_t etype) const {
  return GetCOO()->adj();
}

HeteroGraphPtr UnitGraph::GetAny() const {
  if (in_csr_->defined()) {
    return in_csr_;
  } else if (out_csr_->defined()) {
    return out_csr_;
  } else {
    return coo_;
  }
}

dgl_format_code_t UnitGraph::GetCreatedFormats() const {
  dgl_format_code_t ret = 0;
  if (in_csr_->defined())
    ret |= CSC_CODE;
  if (out_csr_->defined())
    ret |= CSR_CODE;
  if (coo_->defined())
    ret |= COO_CODE;
  return ret;
}

dgl_format_code_t UnitGraph::GetAllowedFormats() const {
  return formats_;
}

HeteroGraphPtr UnitGraph::GetFormat(SparseFormat format) const {
  switch (format) {
  case SparseFormat::kCSR:
    return GetOutCSR();
  case SparseFormat::kCSC:
    return GetInCSR();
  default:
    return GetCOO();
  }
}

HeteroGraphPtr UnitGraph::GetGraphInFormat(dgl_format_code_t formats) const {
  if (formats == ALL_CODE)
    return HeteroGraphPtr(
        // TODO(xiangsx) Make it as graph storage.Clone()
        new UnitGraph(meta_graph_,
                      (in_csr_->defined())
                          ? CSRPtr(new CSR(*in_csr_))
                          : nullptr,
                      (out_csr_->defined())
                          ? CSRPtr(new CSR(*out_csr_))
                          : nullptr,
                      (coo_->defined())
                          ? COOPtr(new COO(*coo_))
                          : nullptr,
                      formats));
  int64_t num_vtypes = NumVertexTypes();
  if (formats & COO_CODE)
    return CreateFromCOO(num_vtypes, GetCOO(false)->adj(), formats);
  if (formats & CSR_CODE)
    return CreateFromCSR(num_vtypes, GetOutCSR(false)->adj(), formats);
  return CreateFromCSC(num_vtypes, GetInCSR(false)->adj(), formats);
}

SparseFormat UnitGraph::SelectFormat(dgl_format_code_t preferred_formats) const {
  dgl_format_code_t common = preferred_formats & formats_;
  dgl_format_code_t created = GetCreatedFormats();
  if (common & created)
    return DecodeFormat(common & created);

  // NOTE(zihao): hypersparse is currently disabled since many CUDA operators on COO have
  // not been implmented yet.
  // if (coo_->defined() && coo_->IsHypersparse())  // only allow coo for hypersparse graph.
  //   return SparseFormat::kCOO;
  if (common)
    return DecodeFormat(common);
  return DecodeFormat(created);
}

GraphPtr UnitGraph::AsImmutableGraph() const {
  CHECK(NumVertexTypes() == 1) << "not a homogeneous graph";
  dgl::CSRPtr in_csr_ptr = nullptr, out_csr_ptr = nullptr;
  dgl::COOPtr coo_ptr = nullptr;
  if (in_csr_->defined()) {
    aten::CSRMatrix csc = GetCSCMatrix(0);
    in_csr_ptr = dgl::CSRPtr(new dgl::CSR(csc.indptr, csc.indices, csc.data));
  }
  if (out_csr_->defined()) {
    aten::CSRMatrix csr = GetCSRMatrix(0);
    out_csr_ptr = dgl::CSRPtr(new dgl::CSR(csr.indptr, csr.indices, csr.data));
  }
  if (coo_->defined()) {
    aten::COOMatrix coo = GetCOOMatrix(0);
    if (!COOHasData(coo)) {
      coo_ptr = dgl::COOPtr(new dgl::COO(NumVertices(0), coo.row, coo.col));
    } else {
      IdArray new_src = Scatter(coo.row, coo.data);
      IdArray new_dst = Scatter(coo.col, coo.data);
      coo_ptr = dgl::COOPtr(new dgl::COO(NumVertices(0), new_src, new_dst));
    }
  }
  return GraphPtr(new dgl::ImmutableGraph(in_csr_ptr, out_csr_ptr, coo_ptr));
}

HeteroGraphPtr UnitGraph::LineGraph(bool backtracking) const {
  // TODO(xiangsx) currently we only support homogeneous graph
  auto fmt = SelectFormat(ALL_CODE);
  switch (fmt) {
    case SparseFormat::kCOO: {
      return CreateFromCOO(1, aten::COOLineGraph(coo_->adj(), backtracking));
    }
    case SparseFormat::kCSR: {
      const aten::CSRMatrix csr = GetCSRMatrix(0);
      const aten::COOMatrix coo = aten::COOLineGraph(aten::CSRToCOO(csr, true), backtracking);
      return CreateFromCOO(1, coo);
    }
    case SparseFormat::kCSC: {
      const aten::CSRMatrix csc = GetCSCMatrix(0);
      const aten::CSRMatrix csr = aten::CSRTranspose(csc);
      const aten::COOMatrix coo = aten::COOLineGraph(aten::CSRToCOO(csr, true), backtracking);
      return CreateFromCOO(1, coo);
    }
    default:
      LOG(FATAL) << "None of CSC, CSR, COO exist";
      break;
  }
  return nullptr;
}

constexpr uint64_t kDGLSerialize_UnitGraphMagic = 0xDD2E60F0F6B4A127;

bool UnitGraph::Load(dmlc::Stream* fs) {
  uint64_t magicNum;
  CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
  CHECK_EQ(magicNum, kDGLSerialize_UnitGraphMagic) << "Invalid UnitGraph Data";

  int64_t save_format_code, formats_code;
  CHECK(fs->Read(&save_format_code)) << "Invalid format";
  CHECK(fs->Read(&formats_code)) << "Invalid format";
  auto save_format = static_cast<SparseFormat>(save_format_code);
  if (formats_code >> 32) {
    formats_ = static_cast<dgl_format_code_t>(0xffffffff & formats_code);
  } else {
    // NOTE(zihao): to be compatible with old formats.
    switch (formats_code & 0xffffffff) {
    case 0:
      formats_ = ALL_CODE;
      break;
    case 1:
      formats_ = COO_CODE;
      break;
    case 2:
      formats_ = CSR_CODE;
      break;
    case 3:
      formats_ = CSC_CODE;
      break;
    default:
      LOG(FATAL) << "Load graph failed, formats code " << formats_code <<
        "not recognized.";
    }
  }

  switch (save_format) {
    case SparseFormat::kCOO:
      fs->Read(&coo_);
      break;
    case SparseFormat::kCSR:
      fs->Read(&out_csr_);
      break;
    case SparseFormat::kCSC:
      fs->Read(&in_csr_);
      break;
    default:
      LOG(FATAL) << "unsupported format code";
      break;
  }

  if (!in_csr_) {
    in_csr_ = CSRPtr(new CSR());
  }
  if (!out_csr_) {
    out_csr_ = CSRPtr(new CSR());
  }
  if (!coo_) {
    coo_ = COOPtr(new COO());
  }

  meta_graph_ = GetAny()->meta_graph();

  return true;
}


void UnitGraph::Save(dmlc::Stream* fs) const {
  fs->Write(kDGLSerialize_UnitGraphMagic);
  // Didn't write UnitGraph::meta_graph_, since it's included in the underlying
  // sparse matrix
  auto avail_fmt = SelectFormat(ALL_CODE);
  fs->Write(static_cast<int64_t>(avail_fmt));
  fs->Write(static_cast<int64_t>(formats_ | 0x100000000));
  switch (avail_fmt) {
    case SparseFormat::kCOO:
      fs->Write(GetCOO());
      break;
    case SparseFormat::kCSR:
      fs->Write(GetOutCSR());
      break;
    case SparseFormat::kCSC:
      fs->Write(GetInCSR());
      break;
    default:
      LOG(FATAL) << "unsupported format code";
      break;
  }
}

UnitGraphPtr UnitGraph::Reverse() const {
  CSRPtr new_incsr = out_csr_, new_outcsr = in_csr_;
  COOPtr new_coo = nullptr;
  if (coo_->defined()) {
    new_coo = COOPtr(new COO(coo_->meta_graph(), aten::COOTranspose(coo_->adj())));
  }

  return UnitGraphPtr(new UnitGraph(meta_graph(), new_incsr, new_outcsr, new_coo));
}

std::tuple<UnitGraphPtr, IdArray, IdArray>
UnitGraph::ToSimple() const {
  CSRPtr new_incsr = nullptr, new_outcsr = nullptr;
  COOPtr new_coo = nullptr;
  IdArray count;
  IdArray edge_map;

  auto avail_fmt = SelectFormat(ALL_CODE);
  switch (avail_fmt) {
    case SparseFormat::kCOO: {
      auto ret = aten::COOToSimple(GetCOO()->adj());
      count = std::get<1>(ret);
      edge_map = std::get<2>(ret);
      new_coo = COOPtr(new COO(meta_graph(), std::get<0>(ret)));
      break;
    }
    case SparseFormat::kCSR: {
      auto ret = aten::CSRToSimple(GetOutCSR()->adj());
      count = std::get<1>(ret);
      edge_map = std::get<2>(ret);
      new_outcsr = CSRPtr(new CSR(meta_graph(), std::get<0>(ret)));
      break;
    }
    case SparseFormat::kCSC: {
      auto ret = aten::CSRToSimple(GetInCSR()->adj());
      count = std::get<1>(ret);
      edge_map = std::get<2>(ret);
      new_incsr = CSRPtr(new CSR(meta_graph(), std::get<0>(ret)));
      break;
    }
    default:
      LOG(FATAL) << "At lease one of COO, CSR or CSC adj should exist.";
      break;
  }

  return std::make_tuple(UnitGraphPtr(new UnitGraph(meta_graph(), new_incsr, new_outcsr, new_coo)),
                         count,
                         edge_map);
}

}  // namespace dgl
