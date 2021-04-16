/*!
 *  Copyright (c) 2019 by Contributors
 * \file dgl/heterograph_interface.h
 * \brief DGL heterogeneous graph index class.
 */

#ifndef DGL_BASE_HETEROGRAPH_H_
#define DGL_BASE_HETEROGRAPH_H_

#include <string>
#include <vector>
#include <utility>
#include <algorithm>
#include <memory>

#include "./runtime/object.h"
#include "aten/spmat.h"
#include "aten/types.h"
#include "graph_interface.h"
#include "array.h"

namespace dgl {

// Forward declaration
class BaseHeteroGraph;
typedef std::shared_ptr<BaseHeteroGraph> HeteroGraphPtr;

struct FlattenedHeteroGraph;
typedef std::shared_ptr<FlattenedHeteroGraph> FlattenedHeteroGraphPtr;

struct HeteroSubgraph;

/*! \brief Enum class for edge direction */
enum class EdgeDir {
  kIn,  // in edge direction
  kOut  // out edge direction
};

/*!
 * \brief Base heterogenous graph.
 *
 * In heterograph, nodes represent entities and edges represent relations.
 * Nodes and edges are associated with types. The same pair of entity types
 * can have multiple relation types between them, but relation type **uniquely**
 * identifies the source and destination entity types.
 *
 * In a high-level, a heterograph is a data structure composed of:
 *  - A meta-graph that stores the entity-entity relation graph.
 *  - A dictionary of relation type to the bipartite graph representing the
 *    actual connections among entity nodes.
 */
class BaseHeteroGraph : public runtime::Object {
 public:
  explicit BaseHeteroGraph(GraphPtr meta_graph): meta_graph_(meta_graph) {}

  virtual ~BaseHeteroGraph() = default;

  ////////////////////////// query/operations on meta graph ////////////////////////

  /*! \return the number of vertex types */
  virtual uint64_t NumVertexTypes() const {
    return meta_graph_->NumVertices();
  }

  /*! \return the number of edge types */
  virtual uint64_t NumEdgeTypes() const {
    return meta_graph_->NumEdges();
  }

  /*! \return given the edge type, find the source type */
  virtual std::pair<dgl_type_t, dgl_type_t> GetEndpointTypes(dgl_type_t etype) const {
    return meta_graph_->FindEdge(etype);
  }

  /*! \return the meta graph */
  virtual GraphPtr meta_graph() const {
    return meta_graph_;
  }

  /*!
   * \brief Return the bipartite graph of the given edge type.
   * \param etype The edge type.
   * \return The bipartite graph.
   */
  virtual HeteroGraphPtr GetRelationGraph(dgl_type_t etype) const = 0;

  ////////////////////////// query/operations on realized graph ////////////////////////

  /*! \brief Add vertices to the given vertex type */
  virtual void AddVertices(dgl_type_t vtype, uint64_t num_vertices) = 0;

  /*! \brief Add one edge to the given edge type */
  virtual void AddEdge(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) = 0;

  /*! \brief Add edges to the given edge type */
  virtual void AddEdges(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) = 0;

  /*!
   * \brief Clear the graph. Remove all vertices/edges.
   */
  virtual void Clear() = 0;

  /*!
   * \brief Get the data type of node and edge IDs of this graph.
   */
  virtual DLDataType DataType() const = 0;

  /*!
   * \brief Get the device context of this graph.
   */
  virtual DLContext Context() const = 0;

  /*!
   * \brief Get the number of integer bits used to store node/edge ids (32 or 64).
   */
  // TODO(BarclayII) replace NumBits() calls to DataType() calls
  virtual uint8_t NumBits() const = 0;

  /*!
   * \return whether the graph is a multigraph
   */
  virtual bool IsMultigraph() const = 0;

  /*! \return whether the graph is read-only */
  virtual bool IsReadonly() const = 0;

  /*! \return the number of vertices in the graph.*/
  virtual uint64_t NumVertices(dgl_type_t vtype) const = 0;

  /*! \return the number of vertices for each type in the graph as a vector */
  inline virtual std::vector<int64_t> NumVerticesPerType() const {
    LOG(FATAL) << "[BUG] NumVerticesPerType() not supported on this object.";
    return {};
  }

  /*! \return the number of edges in the graph.*/
  virtual uint64_t NumEdges(dgl_type_t etype) const = 0;

  /*! \return true if the given vertex is in the graph.*/
  virtual bool HasVertex(dgl_type_t vtype, dgl_id_t vid) const = 0;

  /*! \return a 0-1 array indicating whether the given vertices are in the graph.*/
  virtual BoolArray HasVertices(dgl_type_t vtype, IdArray vids) const = 0;

  /*! \return true if the given edge is in the graph.*/
  virtual bool HasEdgeBetween(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const = 0;

  /*! \return a 0-1 array indicating whether the given edges are in the graph.*/
  virtual BoolArray HasEdgesBetween(dgl_type_t etype, IdArray src_ids, IdArray dst_ids) const = 0;

  /*!
   * \brief Find the predecessors of a vertex.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the predecessor id array.
   */
  virtual IdArray Predecessors(dgl_type_t etype, dgl_id_t dst) const = 0;

  /*!
   * \brief Find the successors of a vertex.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the successor id array.
   */
  virtual IdArray Successors(dgl_type_t etype, dgl_id_t src) const = 0;

  /*!
   * \brief Get all edge ids between the two given endpoints
   * \note The given src and dst vertices should belong to the source vertex type
   *       and the dest vertex type of the given edge type, respectively.
   * \param etype The edge type
   * \param src The source vertex.
   * \param dst The destination vertex.
   * \return the edge id array.
   */
  virtual IdArray EdgeId(dgl_type_t etype, dgl_id_t src, dgl_id_t dst) const = 0;

  /*!
   * \brief Get all edge ids between the given endpoint pairs.
   *
   * \param etype The edge type
   * \param src The src vertex ids.
   * \param dst The dst vertex ids.
   * \return EdgeArray containing all edges between all pairs.
   */
  virtual EdgeArray EdgeIdsAll(dgl_type_t etype, IdArray src, IdArray dst) const = 0;

  /*!
   * \brief Get edge ids between the given endpoint pairs.
   *
   * Only find one matched edge Ids even if there are multiple matches due to parallel
   * edges. The i^th Id in the returned array is for edge (src[i], dst[i]).
   *
   * \param etype The edge type
   * \param src The src vertex ids.
   * \param dst The dst vertex ids.
   * \return EdgeArray containing all edges between all pairs.
   */
  virtual IdArray EdgeIdsOne(dgl_type_t etype, IdArray src, IdArray dst) const = 0;

  /*!
   * \brief Find the edge ID and return the pair of endpoints
   * \param etype The edge type
   * \param eid The edge ID
   * \return a pair whose first element is the source and the second the destination.
   */
  virtual std::pair<dgl_id_t, dgl_id_t> FindEdge(dgl_type_t etype, dgl_id_t eid) const = 0;

  /*!
   * \brief Find the edge IDs and return their source and target node IDs.
   * \param etype The edge type
   * \param eids The edge ID array.
   * \return EdgeArray containing all edges with id in eid.  The order is preserved.
   */
  virtual EdgeArray FindEdges(dgl_type_t etype, IdArray eids) const = 0;

  /*!
   * \brief Get the in edges of the vertex.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the edges
   */
  virtual EdgeArray InEdges(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in edges of the vertices.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray InEdges(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get the out edges of the vertex.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray OutEdges(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out edges of the vertices.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vids The vertex id array.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray OutEdges(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get all the edges in the graph.
   * \note If order is "srcdst", the returned edges list is sorted by their src and
   *       dst ids. If order is "eid", they are in their edge id order.
   *       Otherwise, in the arbitrary order.
   * \param etype The edge type
   * \param order The order of the returned edge list.
   * \return the id arrays of the two endpoints of the edges.
   */
  virtual EdgeArray Edges(dgl_type_t etype, const std::string &order = "") const = 0;

  /*!
   * \brief Get the in degree of the given vertex.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the in degree
   */
  virtual uint64_t InDegree(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the in degrees of the given vertices.
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id array.
   * \return the in degree array
   */
  virtual DegreeArray InDegrees(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Get the out degree of the given vertex.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id.
   * \return the out degree
   */
  virtual uint64_t OutDegree(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the out degrees of the given vertices.
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param etype The edge type
   * \param vid The vertex id array.
   * \return the out degree array
   */
  virtual DegreeArray OutDegrees(dgl_type_t etype, IdArray vids) const = 0;

  /*!
   * \brief Return the successor vector
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the successor vector iterator pair.
   */
  virtual DGLIdIters SuccVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Return the out edge id vector
   * \note The given vertex should belong to the source vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the out edge id vector iterator pair.
   */
  virtual DGLIdIters OutEdgeVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Return the predecessor vector
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the predecessor vector iterator pair.
   */
  virtual DGLIdIters PredVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Return the in edge id vector
   * \note The given vertex should belong to the dest vertex type
   *       of the given edge type.
   * \param vid The vertex id.
   * \return the in edge id vector iterator pair.
   */
  virtual DGLIdIters InEdgeVec(dgl_type_t etype, dgl_id_t vid) const = 0;

  /*!
   * \brief Get the adjacency matrix of the graph.
   *
   * TODO(minjie): deprecate this interface; replace it with GetXXXMatrix.
   *
   * By default, a row of returned adjacency matrix represents the destination
   * of an edge and the column represents the source.
   *
   * If the fmt is 'csr', the function should return three arrays, representing
   *  indptr, indices and edge ids
   *
   * If the fmt is 'coo', the function should return one array of shape (2, nnz),
   * representing a horitonzal stack of row and col indices.
   *
   * \param transpose A flag to transpose the returned adjacency matrix.
   * \param fmt the format of the returned adjacency matrix.
   * \return a vector of IdArrays.
   */
  virtual std::vector<IdArray> GetAdj(
      dgl_type_t etype, bool transpose, const std::string &fmt) const = 0;

  /*!
   * \brief Determine which format to use with a preference.
   *

   * Otherwise, it will return whatever DGL thinks is the most appropriate given
   * the arguments.
   *
   * \param etype Edge type.
   * \param preferred_formats Preferred sparse formats.
   * \return Available sparse format.
   */
  virtual SparseFormat SelectFormat(
      dgl_type_t etype, dgl_format_code_t preferred_formats) const = 0;

  /*!
   * \brief Return sparse formats already created for the graph.
   *
   * \return a number of type dgl_format_code_t.
   */
  virtual dgl_format_code_t GetCreatedFormats() const = 0;

  /*!
   * \brief Return allowed sparse formats for the graph.
   *
   * \return a number of type dgl_format_code_t.
   */
  virtual dgl_format_code_t GetAllowedFormats() const = 0;

  /*!
   * \brief Return the graph in specified available formats.
   *
   * \return The new graph.
   */
  virtual HeteroGraphPtr GetGraphInFormat(dgl_format_code_t formats) const = 0;

  /*!
   * \brief Get adjacency matrix in COO format.
   * \param etype Edge type.
   * \return COO matrix.
   */
  virtual aten::COOMatrix GetCOOMatrix(dgl_type_t etype) const = 0;

  /*!
   * \brief Get adjacency matrix in CSR format.
   *
   * The row and column sizes are equal to the number of dsttype and srctype
   * nodes, respectively.
   *
   * \param etype Edge type.
   * \return CSR matrix.
   */
  virtual aten::CSRMatrix GetCSRMatrix(dgl_type_t etype) const = 0;

  /*!
   * \brief Get adjacency matrix in CSC format.
   *
   * A CSC matrix is equivalent to the transpose of a CSR matrix.
   * We reuse the CSRMatrix data structure as return value. The row and column
   * sizes are equal to the number of dsttype and srctype nodes, respectively.
   *
   * \param etype Edge type.
   * \return A CSR matrix.
   */
  virtual aten::CSRMatrix GetCSCMatrix(dgl_type_t etype) const = 0;

  /*!
   * \brief Extract the induced subgraph by the given vertices.
   *
   * The length of the given vector should be equal to the number of vertex types.
   * Empty arrays can be provided if no vertex is needed for the type. The result
   * subgraph has the same meta graph with the parent, but some types can have no
   * node/edge.
   *
   * \param vids the induced vertices per type.
   * \return the subgraph.
   */
  virtual HeteroSubgraph VertexSubgraph(const std::vector<IdArray>& vids) const = 0;

  /*!
   * \brief Extract the induced subgraph by the given edges.
   *
   * The length of the given vector should be equal to the number of edge types.
   * Empty arrays can be provided if no edge is needed for the type. The result
   * subgraph has the same meta graph with the parent, but some types can have no
   * node/edge.
   *
   * \param eids The edges in the subgraph.
   * \param preserve_nodes If true, the vertices will not be relabeled, so some vertices
   *                       may have no incident edges.
   * \return the subgraph.
   */
  virtual HeteroSubgraph EdgeSubgraph(
      const std::vector<IdArray>& eids, bool preserve_nodes = false) const = 0;

  /*!
   * \brief Convert the list of requested unitgraph graphs into a single unitgraph graph.
   *
   * \param etypes The list of edge type IDs.
   * \return The flattened graph, with induced source/edge/destination types/IDs.
   */
  virtual FlattenedHeteroGraphPtr Flatten(const std::vector<dgl_type_t>& etypes) const {
    LOG(FATAL) << "Flatten operation unsupported";
    return nullptr;
  }

  /*! \brief Cast this graph to immutable graph */
  virtual GraphPtr AsImmutableGraph() const {
    LOG(FATAL) << "AsImmutableGraph not supported.";
    return nullptr;
  }

  static constexpr const char* _type_key = "graph.HeteroGraph";
  DGL_DECLARE_OBJECT_TYPE_INFO(BaseHeteroGraph, runtime::Object);

 protected:
  /*! \brief meta graph */
  GraphPtr meta_graph_;

  // empty constructor
  BaseHeteroGraph(){}
};

// Define HeteroGraphRef
DGL_DEFINE_OBJECT_REF(HeteroGraphRef, BaseHeteroGraph);

/*!
 * \brief Hetero-subgraph data structure.
 *
 * This class can be used as arguments and return values of a C API.
 *
 * <code>
 *   DGL_REGISTER_GLOBAL("some_c_api")
 *   .set_body([] (DGLArgs args, DGLRetValue* rv) {
 *     HeteroSubgraphRef subg = args[0];
 *     std::shared_ptr<HeteroSubgraph> ret = do_something( ... );
 *     *rv = HeteroSubgraphRef(ret);
 *   });
 * </code>
 */
struct HeteroSubgraph : public runtime::Object {
  /*! \brief The heterograph. */
  HeteroGraphPtr graph;
  /*!
   * \brief The induced vertex ids of each entity type.
   * The vector length is equal to the number of vertex types in the parent graph.
   * Each array i has the same length as the number of vertices in type i.
   * Empty array is allowed if the mapping is identity.
   */
  std::vector<IdArray> induced_vertices;
  /*!
   * \brief The induced edge ids of each relation type.
   * The vector length is equal to the number of edge types in the parent graph.
   * Each array i has the same length as the number of edges in type i.
   * Empty array is allowed if the mapping is identity.
   */
  std::vector<IdArray> induced_edges;

  static constexpr const char* _type_key = "graph.HeteroSubgraph";
  DGL_DECLARE_OBJECT_TYPE_INFO(HeteroSubgraph, runtime::Object);
};

// Define HeteroSubgraphRef
DGL_DEFINE_OBJECT_REF(HeteroSubgraphRef, HeteroSubgraph);

/*! \brief The flattened heterograph */
struct FlattenedHeteroGraph : public runtime::Object {
  /*! \brief The graph */
  HeteroGraphRef graph;
  /*!
   * \brief Mapping from source node ID to node type in parent graph
   * \note The induced type array guarantees that the same type always appear contiguously.
   */
  IdArray induced_srctype;
  /*!
   * \brief The set of node types in parent graph appearing in source nodes.
   */
  IdArray induced_srctype_set;
  /*! \brief Mapping from source node ID to local node ID in parent graph */
  IdArray induced_srcid;
  /*!
   * \brief Mapping from edge ID to edge type in parent graph
   * \note The induced type array guarantees that the same type always appear contiguously.
   */
  IdArray induced_etype;
  /*!
   * \brief The set of edge types in parent graph appearing in edges.
   */
  IdArray induced_etype_set;
  /*! \brief Mapping from edge ID to local edge ID in parent graph */
  IdArray induced_eid;
  /*!
   * \brief Mapping from destination node ID to node type in parent graph
   * \note The induced type array guarantees that the same type always appear contiguously.
   */
  IdArray induced_dsttype;
  /*!
   * \brief The set of node types in parent graph appearing in destination nodes.
   */
  IdArray induced_dsttype_set;
  /*! \brief Mapping from destination node ID to local node ID in parent graph */
  IdArray induced_dstid;

  void VisitAttrs(runtime::AttrVisitor *v) final {
    v->Visit("graph", &graph);
    v->Visit("induced_srctype", &induced_srctype);
    v->Visit("induced_srctype_set", &induced_srctype_set);
    v->Visit("induced_srcid", &induced_srcid);
    v->Visit("induced_etype", &induced_etype);
    v->Visit("induced_etype_set", &induced_etype_set);
    v->Visit("induced_eid", &induced_eid);
    v->Visit("induced_dsttype", &induced_dsttype);
    v->Visit("induced_dsttype_set", &induced_dsttype_set);
    v->Visit("induced_dstid", &induced_dstid);
  }

  static constexpr const char* _type_key = "graph.FlattenedHeteroGraph";
  DGL_DECLARE_OBJECT_TYPE_INFO(FlattenedHeteroGraph, runtime::Object);
};
DGL_DEFINE_OBJECT_REF(FlattenedHeteroGraphRef, FlattenedHeteroGraph);

// Declarations of functions and algorithms

/*!
 * \brief Create a heterograph from meta graph and a list of bipartite graph,
 * additionally specifying number of nodes per type.
 */
HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph,
    const std::vector<HeteroGraphPtr> &rel_graphs,
    const std::vector<int64_t> &num_nodes_per_type = {});

/*!
 * \brief Create a heterograph from COO input.
 * \param num_vtypes Number of vertex types. Must be 1 or 2.
 * \param num_src Number of nodes in the source type.
 * \param num_dst Number of nodes in the destination type.
 * \param row Src node ids of the edges.
 * \param col Dst node ids of the edges.
 * \param row_sorted Whether the `row` array is in sorted ascending order.
 * \param col_sorted When `row_sorted` is true, whether the columns within each
 * row are also sorted. When `row_sorted` is false, this flag must also be
 * false.
 * \param formats Sparse formats used for storing this graph.
 * \return A heterograph pointer.
 */
HeteroGraphPtr CreateFromCOO(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray row, IdArray col, bool row_sorted = false, bool col_sorted = false,
    dgl_format_code_t formats = ALL_CODE);

/*!
 * \brief Create a heterograph from COO input.
 * \param num_vtypes Number of vertex types. Must be 1 or 2.
 * \param mat The COO matrix
 * \param formats Sparse formats used for storing this graph.
 * \return A heterograph pointer.
 */
HeteroGraphPtr CreateFromCOO(
    int64_t num_vtypes, const aten::COOMatrix& mat,
    dgl_format_code_t formats = ALL_CODE);

/*!
 * \brief Create a heterograph from CSR input.
 * \param num_vtypes Number of vertex types. Must be 1 or 2.
 * \param num_src Number of nodes in the source type.
 * \param num_dst Number of nodes in the destination type.
 * \param indptr Indptr array
 * \param indices Indices array
 * \param edge_ids Edge ids
 * \param formats Sparse formats for storing this graph.
 * \return A heterograph pointer.
 */
HeteroGraphPtr CreateFromCSR(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids,
    dgl_format_code_t formats = ALL_CODE);

/*!
 * \brief Create a heterograph from CSR input.
 * \param num_vtypes Number of vertex types. Must be 1 or 2.
 * \param mat The CSR matrix
 * \param formats Sparse formats for storing this graph.
 * \return A heterograph pointer.
 */
HeteroGraphPtr CreateFromCSR(
    int64_t num_vtypes, const aten::CSRMatrix& mat,
    dgl_format_code_t formats = ALL_CODE);

/*!
 * \brief Create a heterograph from CSC input.
 * \param num_vtypes Number of vertex types. Must be 1 or 2.
 * \param num_src Number of nodes in the source type.
 * \param num_dst Number of nodes in the destination type.
 * \param indptr Indptr array
 * \param indices Indices array
 * \param edge_ids Edge ids
 * \param formats Sparse formats used for storing this graph.
 * \return A heterograph pointer.
 */
HeteroGraphPtr CreateFromCSC(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids,
    dgl_format_code_t formats = ALL_CODE);

/*!
 * \brief Create a heterograph from CSC input.
 * \param num_vtypes Number of vertex types. Must be 1 or 2.
 * \param mat The CSC matrix
 * \param formats Sparse formats available for storing this graph.
 * \return A heterograph pointer.
 */
HeteroGraphPtr CreateFromCSC(
    int64_t num_vtypes, const aten::CSRMatrix& mat,
    dgl_format_code_t formats = ALL_CODE);

/*!
 * \brief Extract the subgraph of the in edges of the given nodes.
 * \param graph Graph
 * \param nodes Node IDs of each type
 * \return Subgraph containing only the in edges. The returned graph has the same
 *         schema as the original one.
 */
HeteroSubgraph InEdgeGraph(const HeteroGraphPtr graph, const std::vector<IdArray>& nodes);

/*!
 * \brief Extract the subgraph of the out edges of the given nodes.
 * \param graph Graph
 * \param nodes Node IDs of each type
 * \return Subgraph containing only the out edges. The returned graph has the same
 *         schema as the original one.
 */
HeteroSubgraph OutEdgeGraph(const HeteroGraphPtr graph, const std::vector<IdArray>& nodes);

/*!
 * \brief Joint union multiple graphs into one graph.
 *
 * All input graphs should have the same metagraph.
 *
 * TODO(xiangsx): remove the meta_graph argument
 *
 * \param meta_graph Metagraph of the inputs and result.
 * \param component_graphs Input graphs
 * \return One graph that unions all the components
 */
HeteroGraphPtr JointUnionHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs);

/*!
 * \brief Union multiple graphs into one with each input graph as one disjoint component.
 *
 * All input graphs should have the same metagraph.
 *
 * TODO(minjie): remove the meta_graph argument
 *
 * \tparam IdType Graph's index data type, can be int32_t or int64_t
 * \param meta_graph Metagraph of the inputs and result.
 * \param component_graphs Input graphs
 * \return One graph that unions all the components
 */
template <class IdType>
HeteroGraphPtr DisjointUnionHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs);

HeteroGraphPtr DisjointUnionHeteroGraph2(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs);

/*!
 * \brief Split a graph into multiple disjoin components.
 *
 * Edges across different components are ignored. All the result graphs have the same
 * metagraph as the input one.
 *
 * The `vertex_sizes` and `edge_sizes` arrays the concatenation of arrays of each
 * node/edge type. Suppose there are N vertex types, then the array length should
 * be B*N, where B is the number of components to split.
 *
 * TODO(minjie): remove the meta_graph argument; use vector<IdArray> for vertex_sizes
 *   and edge_sizes.
 *
 * \tparam IdType Graph's index data type, can be int32_t or int64_t
 * \param meta_graph Metagraph.
 * \param batched_graph Input graph.
 * \param vertex_sizes Number of vertices of each component.
 * \param edge_sizes Number of vertices of each component.
 * \return A list of graphs representing each disjoint components.
 */
template <class IdType>
std::vector<HeteroGraphPtr> DisjointPartitionHeteroBySizes(
    GraphPtr meta_graph,
    HeteroGraphPtr batched_graph,
    IdArray vertex_sizes,
    IdArray edge_sizes);


std::vector<HeteroGraphPtr> DisjointPartitionHeteroBySizes2(
    GraphPtr meta_graph,
    HeteroGraphPtr batched_graph,
    IdArray vertex_sizes,
    IdArray edge_sizes);

/*!
 * \brief Structure for pickle/unpickle.
 *
 * The design principle is to leverage the NDArray class as much as possible so
 * that when they are converted to backend-specific tensors, we could leverage
 * the efficient pickle/unpickle solutions from the backend framework.
 *
 * NOTE(minjie): This is a temporary solution before we support shared memory
 *   storage ourselves.
 *
 * This class can be used as arguments and return values of a C API.
 */
struct HeteroPickleStates : public runtime::Object {
  /*! \brief version number */
  int64_t version = 0;

  /*! \brief Metainformation
   *
   * metagraph, number of nodes per type, format, flags
   */
  std::string meta;

  /*! \brief Arrays representing graph structure (coo or csr) */
  std::vector<IdArray> arrays;

  /* To support backward compatibility, we have to retain fields in the old
   * version of HeteroPickleStates
   */

  /*! \brief Metagraph(64bits ImmutableGraph) */
  GraphPtr metagraph;

  /*! \brief Number of nodes per type */
  std::vector<int64_t> num_nodes_per_type;

  /*! \brief adjacency matrices of each relation graph */
  std::vector<std::shared_ptr<SparseMatrix> > adjs;

  static constexpr const char* _type_key = "graph.HeteroPickleStates";
  DGL_DECLARE_OBJECT_TYPE_INFO(HeteroPickleStates, runtime::Object);
};

// Define HeteroPickleStatesRef
DGL_DEFINE_OBJECT_REF(HeteroPickleStatesRef, HeteroPickleStates);

/*!
 * \brief Create a heterograph from pickling states.
 *
 * \param states Pickle states
 * \return A heterograph pointer
 */
HeteroGraphPtr HeteroUnpickle(const HeteroPickleStates& states);

/*!
 * \brief Get the pickling state of the relation graph structure in backend tensors.
 *
 * \return a HeteroPickleStates object
 */
HeteroPickleStates HeteroPickle(HeteroGraphPtr graph);

/*!
 * \brief Old version of HeteroUnpickle, for backward compatibility
 *
 * \param states Pickle states
 * \return A heterograph pointer
 */
HeteroGraphPtr HeteroUnpickleOld(const HeteroPickleStates& states);

#define FORMAT_HAS_CSC(format) \
  ((format) & CSC_CODE)

#define FORMAT_HAS_CSR(format) \
  ((format) & CSR_CODE)

#define FORMAT_HAS_COO(format) \
  ((format) & COO_CODE)

}  // namespace dgl

#endif  // DGL_BASE_HETEROGRAPH_H_
