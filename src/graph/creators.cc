/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/creators.cc
 * \brief Functions for constructing graphs.
 */
#include "./heterograph.h"
using namespace dgl::runtime;

namespace dgl {

// creator implementation
HeteroGraphPtr CreateHeteroGraph(
    GraphPtr meta_graph,
    const std::vector<HeteroGraphPtr>& rel_graphs,
    const std::vector<int64_t>& num_nodes_per_type) {
  return HeteroGraphPtr(new HeteroGraph(meta_graph, rel_graphs, num_nodes_per_type));
}

HeteroGraphPtr CreateFromCOO(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray row, IdArray col,
    bool row_sorted, bool col_sorted, dgl_format_code_t formats) {
  auto unit_g = UnitGraph::CreateFromCOO(
      num_vtypes, num_src, num_dst, row, col, row_sorted, col_sorted, formats);
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(), {unit_g}));
}

HeteroGraphPtr CreateFromCOO(
    int64_t num_vtypes, const aten::COOMatrix& mat,
    dgl_format_code_t formats) {
  auto unit_g = UnitGraph::CreateFromCOO(num_vtypes, mat, formats);
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(), {unit_g}));
}

HeteroGraphPtr CreateFromCSR(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids,
    dgl_format_code_t formats) {
  auto unit_g = UnitGraph::CreateFromCSR(
      num_vtypes, num_src, num_dst, indptr, indices, edge_ids, formats);
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(), {unit_g}));
}

HeteroGraphPtr CreateFromCSR(
    int64_t num_vtypes, const aten::CSRMatrix& mat,
    dgl_format_code_t formats) {
  auto unit_g = UnitGraph::CreateFromCSR(num_vtypes, mat, formats);
  auto ret = HeteroGraphPtr(new HeteroGraph(
      unit_g->meta_graph(),
      {unit_g}));
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(),
                                                       {unit_g}));
}

HeteroGraphPtr CreateFromCSC(
    int64_t num_vtypes, int64_t num_src, int64_t num_dst,
    IdArray indptr, IdArray indices, IdArray edge_ids,
    dgl_format_code_t formats) {
  auto unit_g = UnitGraph::CreateFromCSC(
      num_vtypes, num_src, num_dst, indptr, indices, edge_ids, formats);
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(), {unit_g}));
}

HeteroGraphPtr CreateFromCSC(
    int64_t num_vtypes, const aten::CSRMatrix& mat,
    dgl_format_code_t formats) {
  auto unit_g = UnitGraph::CreateFromCSC(num_vtypes, mat, formats);
  return HeteroGraphPtr(new HeteroGraph(unit_g->meta_graph(), {unit_g}));
}

}  // namespace dgl
