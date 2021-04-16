/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/unit_graph.h
 * \brief UnitGraph graph
 */

#ifndef DGL_GRAPH_UNIT_GRAPH_H_
#define DGL_GRAPH_UNIT_GRAPH_H_

#include <dgl/base_heterograph.h>
#include <dgl/lazy.h>
#include <dgl/array.h>
#include <dmlc/io.h>
#include <dmlc/type_traits.h>
#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <tuple>

#include "../c_api_common.h"

namespace dgl {

class HeteroGraph;
class UnitGraph;
typedef std::shared_ptr<UnitGraph> UnitGraphPtr;

/*!
 * \brief UnitGraph graph
 *
 * UnitGraph graph is a special type of heterograph which
 * (1) Have two types of nodes: "Src" and "Dst". All the edges are
 *     from "Src" type nodes to "Dst" type nodes, so there is no edge among
 *     nodes of the same type. Thus, its metagraph has two nodes and one edge
 *     between them.
 * (2) Have only one type of nodes and edges. Thus, its metagraph has one node
 *     and one self-loop edge.
 */
class UnitGraph : public BaseHeteroGraph {
 public:
  // internal data structure
  class COO;
  class CSR;
  typedef std::shared_ptr<COO> COOPtr;
  typedef std::shared_ptr<CSR> CSRPtr;

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

  DLDataType DataType() const override;

  DLContext Context() const override;

  uint8_t NumBits() const override;

  bool IsMultigraph() const override;

  bool IsReadonly() const override {
    return true;
  }

  uint64_t NumVertices(dgl_type_t vtype) const override;

  inline std::vector<int64_t> NumVerticesPerType() const override {
    std::vector<int64_t> num_nodes_per_type;
    for (dgl_type_t vtype = 0; vtype < NumVertexTypes(); ++vtype)
      num_nodes_per_type.push_back(NumVertices(vtype));
    return num_nodes_per_type;
  }

  uint64_t NumEdges(dgl_type_t etype) const override;

  bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const override;

  BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const override;

  bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override;

  BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const override;

  IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const override;

  IdArray Successors(dgl_type_t etype, dgl_id_t src) const override;

  IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const override;

  EdgeArray EdgeIdsAll(dgl_type_t etype, IdArray src, IdArray dst) const override;

  IdArray EdgeIdsOne(dgl_type_t etype, IdArray src, IdArray dst) const override;

  std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const override;

  EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const override;

  EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const override;

  EdgeArray InEdges(dgl_type_t etype, IdArray vids) const override;

  EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const override;

  EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const override;

  EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const override;

  uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const override;

  DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const override;

  uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const override;

  DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const override;

  DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const override;

  // 32bit version functions, patch for SuccVec
  DGLIdIters32 SuccVec32(dgl_type_t etype, dgl_id_t vid) const;

  DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const override;

  DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const override;

  DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const override;

  std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const override;

  HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const override;

  HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const override;

  // creators
  /*! \brief Create a graph with no edges */
  static HeteroGraphPtr Empty(
      int64_t num_vtypes, int64_t num_src, int64_t num_dst,
      DLDataType dtype, DLContext ctx) {
    IdArray row = IdArray::Empty({0}, dtype, ctx);
    IdArray col = IdArray::Empty({0}, dtype, ctx);
    return CreateFromCOO(num_vtypes, num_src, num_dst, row, col);
  }

  /*! \brief Create a graph from COO arrays */
  static HeteroGraphPtr CreateFromCOO(
      int64_t num_vtypes, int64_t num_src, int64_t num_dst,
      IdArray row, IdArray col, bool row_sorted = false,
      bool col_sorted = false, dgl_format_code_t formats = ALL_CODE);

  static HeteroGraphPtr CreateFromCOO(
      int64_t num_vtypes, const aten::COOMatrix& mat,
      dgl_format_code_t formats = ALL_CODE);

  /*! \brief Create a graph from (out) CSR arrays */
  static HeteroGraphPtr CreateFromCSR(
      int64_t num_vtypes, int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids,
      dgl_format_code_t formats = ALL_CODE);

  static HeteroGraphPtr CreateFromCSR(
      int64_t num_vtypes, const aten::CSRMatrix& mat,
      dgl_format_code_t formats = ALL_CODE);

  /*! \brief Create a graph from (in) CSC arrays */
  static HeteroGraphPtr CreateFromCSC(
      int64_t num_vtypes, int64_t num_src, int64_t num_dst,
      IdArray indptr, IdArray indices, IdArray edge_ids,
      dgl_format_code_t formats = ALL_CODE);

  static HeteroGraphPtr CreateFromCSC(
      int64_t num_vtypes, const aten::CSRMatrix& mat,
      dgl_format_code_t formats = ALL_CODE);

  /*! \brief Convert the graph to use the given number of bits for storage */
  static HeteroGraphPtr AsNumBits(HeteroGraphPtr g, uint8_t bits);

  /*! \brief Copy the data to another context */
  static HeteroGraphPtr CopyTo(HeteroGraphPtr g, const DLContext& ctx);

  /*! 
   * \brief Create in-edge CSR format of the unit graph.
   * \param inplace if true and the in-edge CSR format does not exist, the created
   *                format will be cached in this object unless the format is restricted.
   * \return Return the in-edge CSR format. Create from other format if not exist.
   */
  CSRPtr GetInCSR(bool inplace = true) const;

  /*! 
   * \brief Create out-edge CSR format of the unit graph.
   * \param inplace if true and the out-edge CSR format does not exist, the created
   *                format will be cached in this object unless the format is restricted.
   * \return Return the out-edge CSR format. Create from other format if not exist.
   */
  CSRPtr GetOutCSR(bool inplace = true) const;

  /*!
   * \brief Create COO format of the unit graph.
   * \param inplace if true and the COO format does not exist, the created
   *                format will be cached in this object unless the format is restricted.
   * \return Return the COO format. Create from other format if not exist.
   */
  COOPtr GetCOO(bool inplace = true) const;

  /*! \return Return the COO matrix form */
  aten::COOMatrix GetCOOMatrix(dgl_type_t etype) const override;

  /*! \return Return the in-edge CSC in the matrix form */
  aten::CSRMatrix GetCSCMatrix(dgl_type_t etype) const override;

  /*! \return Return the out-edge CSR in the matrix form */
  aten::CSRMatrix GetCSRMatrix(dgl_type_t etype) const override;

  SparseFormat SelectFormat(dgl_type_t etype, dgl_format_code_t preferred_formats) const override {
    return SelectFormat(preferred_formats);
  }

  /*!
   * \brief Return the graph in the given format. Perform format conversion if the
   * requested format does not exist.
   *
   * \return A graph in the requested format.
   */
  HeteroGraphPtr GetFormat(SparseFormat format) const;

  dgl_format_code_t GetCreatedFormats() const override;

  dgl_format_code_t GetAllowedFormats() const override;

  HeteroGraphPtr GetGraphInFormat(dgl_format_code_t formats) const override;

  /*! \return Load UnitGraph from stream, using CSRMatrix*/
  bool Load(dmlc::Stream* fs);

  /*! \return Save UnitGraph to stream, using CSRMatrix */
  void Save(dmlc::Stream* fs) const;

  /*! \brief Creat a LineGraph of self */
  HeteroGraphPtr LineGraph(bool backtracking) const;

  /*! \return the reversed graph */
  UnitGraphPtr Reverse() const;

  /*! \return the simpled (no-multi-edge) graph
   *          the count recording the number of duplicated edges from the original graph.
   *          the edge mapping from the edge IDs of original graph to those of the
   *          returned graph.
   */
  std::tuple<UnitGraphPtr, IdArray, IdArray>ToSimple() const;

  void InvalidateCSR();

  void InvalidateCSC();

  void InvalidateCOO();

 private:
  friend class Serializer;
  friend class HeteroGraph;
  friend class ImmutableGraph;

  // private empty constructor
  UnitGraph() {}

  /*!
   * \brief constructor
   * \param metagraph metagraph
   * \param in_csr in edge csr
   * \param out_csr out edge csr
   * \param coo coo
   */
  UnitGraph(GraphPtr metagraph, CSRPtr in_csr, CSRPtr out_csr, COOPtr coo,
            dgl_format_code_t formats = ALL_CODE);

  /*!
   * \brief constructor
   * \param metagraph metagraph
   * \param in_csr in edge csr
   * \param out_csr out edge csr
   * \param coo coo
   * \param has_in_csr whether in_csr is valid
   * \param has_out_csr whether out_csr is valid
   * \param has_coo whether coo is valid
   */
  static HeteroGraphPtr CreateHomographFrom(
      const aten::CSRMatrix &in_csr,
      const aten::CSRMatrix &out_csr,
      const aten::COOMatrix &coo,
      bool has_in_csr,
      bool has_out_csr,
      bool has_coo,
      dgl_format_code_t formats = ALL_CODE);

  /*! \return Return any existing format. */
  HeteroGraphPtr GetAny() const;

  /*!
   * \brief Determine which format to use with a preference.
   *
   * If the storage of unit graph is "locked", i.e. no conversion is allowed, then
   * it will return the locked format.
   *
   * Otherwise, it will return whatever DGL thinks is the most appropriate given
   * the arguments.
   */
  SparseFormat SelectFormat(dgl_format_code_t preferred_formats) const;

  /*! \return Whether the graph is hypersparse */
  bool IsHypersparse() const;

  GraphPtr AsImmutableGraph() const override;

  // Graph stored in different format. We use an on-demand strategy: the format is
  // only materialized if the operation that suitable for it is invoked.
  /*! \brief CSR graph that stores reverse edges */
  CSRPtr in_csr_;
  /*! \brief CSR representation */
  CSRPtr out_csr_;
  /*! \brief COO representation */
  COOPtr coo_;
  /*!
   * \brief Storage format restriction.
   */
  dgl_format_code_t formats_;
};

};  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::UnitGraph, true);
DMLC_DECLARE_TRAITS(has_saveload, dgl::UnitGraph::CSR, true);
DMLC_DECLARE_TRAITS(has_saveload, dgl::UnitGraph::COO, true);
}  // namespace dmlc

#endif  // DGL_GRAPH_UNIT_GRAPH_H_
