/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/union_partition.cc
 * \brief Functions for partition, union multiple graphs.
 */
#include "../heterograph.h"
using namespace dgl::runtime;

namespace dgl {

HeteroGraphPtr JointUnionHeteroGraph(
  GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs) {
  CHECK_GT(component_graphs.size(), 0) << "Input graph list has at least two graphs";
  std::vector<HeteroGraphPtr> rel_graphs(meta_graph->NumEdges());
  std::vector<int64_t> num_nodes_per_type(meta_graph->NumVertices(), 0);

  // Loop over all canonical etypes
  for (dgl_type_t etype = 0; etype < meta_graph->NumEdges(); ++etype) {
    auto pair = meta_graph->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    uint64_t num_src_v = component_graphs[0]->NumVertices(src_vtype);
    uint64_t num_dst_v = component_graphs[0]->NumVertices(dst_vtype);
    HeteroGraphPtr rgptr = nullptr;

    // ALL = CSC | CSR | COO
    dgl_format_code_t format = (1 << (SparseFormat2Code(SparseFormat::kCOO)-1)) |
                               (1 << (SparseFormat2Code(SparseFormat::kCSR)-1)) |
                               (1 << (SparseFormat2Code(SparseFormat::kCSC)-1));
    // get common format
    for (size_t i = 0; i < component_graphs.size(); ++i) {
      const auto& cg = component_graphs[i];
      CHECK_EQ(num_src_v, component_graphs[i]->NumVertices(src_vtype)) << "Input graph[" << i <<
        "] should have same number of src vertices as input graph[0]";
      CHECK_EQ(num_dst_v, component_graphs[i]->NumVertices(dst_vtype)) << "Input graph[" << i <<
        "] should have same number of dst vertices as input graph[0]";

      const std::string restrict_format = cg->GetRelationGraph(etype)->GetRestrictFormat();
      const SparseFormat curr_format = ParseSparseFormat(restrict_format);

      if (curr_format == SparseFormat::kCOO ||
          curr_format == SparseFormat::kCSR ||
          curr_format == SparseFormat::kCSC)
        format &=(1 << (SparseFormat2Code(curr_format)-1));
    }
    CHECK_GT(format, 0) << "The conjunction of restrict_format of the relation graphs under " <<
      etype << "should not be None.";

    // prefer COO
    if (FORMAT_HAS_COO(format)) {
      std::vector<aten::COOMatrix> coos;
      for (size_t i = 0; i < component_graphs.size(); ++i) {
        const auto& cg = component_graphs[i];
        aten::COOMatrix coo = cg->GetCOOMatrix(etype);
        coos.push_back(coo);
      }

      aten::COOMatrix res = aten::UnionCoo(coos);
      rgptr = UnitGraph::CreateFromCOO(
        (src_vtype == dst_vtype) ? 1 : 2, res,
        SparseFormat::kAny);
    } else if (FORMAT_HAS_CSR(format)) {
      std::vector<aten::CSRMatrix> csrs;
      for (size_t i = 0; i < component_graphs.size(); ++i) {
        const auto& cg = component_graphs[i];
        aten::CSRMatrix csr = cg->GetCSRMatrix(etype);
        csrs.push_back(csr);
      }

      aten::CSRMatrix res = aten::UnionCsr(csrs);
      rgptr = UnitGraph::CreateFromCSR(
        (src_vtype == dst_vtype) ? 1 : 2, res,
        SparseFormat::kAny);
    } else if (FORMAT_HAS_CSC(format)) {
      // CSR and CSC have the same storage format, i.e. CSRMatrix
      std::vector<aten::CSRMatrix> cscs;
      for (size_t i = 0; i < component_graphs.size(); ++i) {
        const auto& cg = component_graphs[i];
        aten::CSRMatrix csc = cg->GetCSCMatrix(etype);
        cscs.push_back(csc);
      }

      aten::CSRMatrix res = aten::UnionCsr(cscs);
      rgptr = UnitGraph::CreateFromCSC(
        (src_vtype == dst_vtype) ? 1 : 2, res,
        SparseFormat::kAny);
    }

    rel_graphs[etype] = rgptr;
    num_nodes_per_type[src_vtype] = num_src_v;
    num_nodes_per_type[dst_vtype] = num_dst_v;
  }

  return CreateHeteroGraph(meta_graph, rel_graphs, std::move(num_nodes_per_type));
}

HeteroGraphPtr DisjointUnionHeteroGraph2(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs) {
  CHECK_GT(component_graphs.size(), 0) << "Input graph list is empty";
  std::vector<HeteroGraphPtr> rel_graphs(meta_graph->NumEdges());
  std::vector<int64_t> num_nodes_per_type(meta_graph->NumVertices(), 0);

  // Loop over all canonical etypes
  for (dgl_type_t etype = 0; etype < meta_graph->NumEdges(); ++etype) {
    auto pair = meta_graph->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    uint64_t src_offset = 0, dst_offset = 0;
    HeteroGraphPtr rgptr = nullptr;

    // ALL = CSC | CSR | COO
    dgl_format_code_t format = (1 << (SparseFormat2Code(SparseFormat::kCOO)-1)) |
                               (1 << (SparseFormat2Code(SparseFormat::kCSR)-1)) |
                               (1 << (SparseFormat2Code(SparseFormat::kCSC)-1));
    // do some preprocess
    for (size_t i = 0; i < component_graphs.size(); ++i) {
      const auto& cg = component_graphs[i];
      const std::string restrict_format = cg->GetRelationGraph(etype)->GetRestrictFormat();
      const SparseFormat curr_format = ParseSparseFormat(restrict_format);

      if (curr_format == SparseFormat::kCOO ||
          curr_format == SparseFormat::kCSR ||
          curr_format == SparseFormat::kCSC)
        format &=(1 << (SparseFormat2Code(curr_format)-1));

      // Update offsets
      src_offset += cg->NumVertices(src_vtype);
      dst_offset += cg->NumVertices(dst_vtype);
    }
    CHECK_GT(format, 0) << "The conjunction of restrict_format of the relation graphs under " <<
      etype << "should not be None.";

    // prefer COO
    if (FORMAT_HAS_COO(format)) {
      std::vector<aten::COOMatrix> coos;
      for (size_t i = 0; i < component_graphs.size(); ++i) {
        const auto& cg = component_graphs[i];
        aten::COOMatrix coo = cg->GetCOOMatrix(etype);
        coos.push_back(coo);
      }

      aten::COOMatrix res = aten::DisjointUnionCoo(coos);

      rgptr = UnitGraph::CreateFromCOO(
        (src_vtype == dst_vtype) ? 1 : 2, res,
        SparseFormat::kAny);
    } else if (FORMAT_HAS_CSR(format)) {
      std::vector<aten::CSRMatrix> csrs;
      for (size_t i = 0; i < component_graphs.size(); ++i) {
        const auto& cg = component_graphs[i];
        aten::CSRMatrix csr = cg->GetCSRMatrix(etype);
        csrs.push_back(csr);
      }

      aten::CSRMatrix res = aten::DisjointUnionCsr(csrs);

      rgptr = UnitGraph::CreateFromCSR(
        (src_vtype == dst_vtype) ? 1 : 2, res,
        SparseFormat::kAny);
    } else if (FORMAT_HAS_CSC(format)) {
      // CSR and CSC have the same storage format, i.e. CSRMatrix
      std::vector<aten::CSRMatrix> cscs;
      for (size_t i = 0; i < component_graphs.size(); ++i) {
        const auto& cg = component_graphs[i];
        aten::CSRMatrix csc = cg->GetCSCMatrix(etype);
        cscs.push_back(csc);
      }

      aten::CSRMatrix res = aten::DisjointUnionCsr(cscs);
      rgptr = UnitGraph::CreateFromCSC(
        (src_vtype == dst_vtype) ? 1 : 2, res,
        SparseFormat::kAny);
    }
    rel_graphs[etype] = rgptr;
    num_nodes_per_type[src_vtype] = src_offset;
    num_nodes_per_type[dst_vtype] = dst_offset;
  }

  return CreateHeteroGraph(meta_graph, rel_graphs, std::move(num_nodes_per_type));
}

std::vector<HeteroGraphPtr> DisjointPartitionHeteroBySizes2(
    GraphPtr meta_graph, HeteroGraphPtr batched_graph, IdArray vertex_sizes, IdArray edge_sizes) {
  // Sanity check for vertex sizes
  CHECK_EQ(vertex_sizes->dtype.bits, 64) << "dtype of vertex_sizes should be int64";
  CHECK_EQ(edge_sizes->dtype.bits, 64) << "dtype of edge_sizes should be int64";
  const uint64_t len_vertex_sizes = vertex_sizes->shape[0];
  const uint64_t* vertex_sizes_data = static_cast<uint64_t*>(vertex_sizes->data);
  const uint64_t num_vertex_types = meta_graph->NumVertices();
  const uint64_t batch_size = len_vertex_sizes / num_vertex_types;

  // Map vertex type to the corresponding node cum sum
  std::vector<std::vector<uint64_t>> vertex_cumsum;
  vertex_cumsum.resize(num_vertex_types);
  // Loop over all vertex types
  for (uint64_t vtype = 0; vtype < num_vertex_types; ++vtype) {
    vertex_cumsum[vtype].push_back(0);
    for (uint64_t g = 0; g < batch_size; ++g) {
      // We've flattened the number of vertices in the batch for all types
      vertex_cumsum[vtype].push_back(
        vertex_cumsum[vtype][g] + vertex_sizes_data[vtype * batch_size + g]);
    }
    CHECK_EQ(vertex_cumsum[vtype][batch_size], batched_graph->NumVertices(vtype))
      << "Sum of the given sizes must equal to the number of nodes for type " << vtype;
  }

  // Sanity check for edge sizes
  const uint64_t* edge_sizes_data = static_cast<uint64_t*>(edge_sizes->data);
  const uint64_t num_edge_types = meta_graph->NumEdges();
  // Map edge type to the corresponding edge cum sum
  std::vector<std::vector<uint64_t>> edge_cumsum;
  edge_cumsum.resize(num_edge_types);
  // Loop over all edge types
  for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
    edge_cumsum[etype].push_back(0);
    for (uint64_t g = 0; g < batch_size; ++g) {
      // We've flattened the number of edges in the batch for all types
      edge_cumsum[etype].push_back(
        edge_cumsum[etype][g] + edge_sizes_data[etype * batch_size + g]);
    }
    CHECK_EQ(edge_cumsum[etype][batch_size], batched_graph->NumEdges(etype))
      << "Sum of the given sizes must equal to the number of edges for type " << etype;
  }

  // Construct relation graphs for unbatched graphs
  std::vector<std::vector<HeteroGraphPtr>> rel_graphs;
  rel_graphs.resize(batch_size);
  // Loop over all edge types
  auto format = batched_graph->GetRelationGraph(0)->GetFormatInUse();
  auto restrict_format = batched_graph->GetRelationGraph(0)->GetRestrictFormat();

  if (FORMAT_HAS_COO(format)) {
    for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
      auto pair = meta_graph->FindEdge(etype);
      const dgl_type_t src_vtype = pair.first;
      const dgl_type_t dst_vtype = pair.second;
      aten::COOMatrix coo = batched_graph->GetCOOMatrix(etype);
      auto res = aten::DisjointPartitionCooBySizes(coo,
                                                   batch_size,
                                                   edge_cumsum[etype],
                                                   vertex_cumsum[src_vtype],
                                                   vertex_cumsum[dst_vtype]);
      for (uint64_t g = 0; g < batch_size; ++g) {
        HeteroGraphPtr rgptr = UnitGraph::CreateFromCOO(
          (src_vtype == dst_vtype) ? 1 : 2, res[g],
          ParseSparseFormat(restrict_format));

        rel_graphs[g].push_back(rgptr);
      }
    }
  } else if (FORMAT_HAS_CSR(format)) {
    for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
      auto pair = meta_graph->FindEdge(etype);
      const dgl_type_t src_vtype = pair.first;
      const dgl_type_t dst_vtype = pair.second;
      aten::CSRMatrix csr = batched_graph->GetCSRMatrix(etype);
      auto res = aten::DisjointPartitionCsrBySizes(csr,
                                                   batch_size,
                                                   edge_cumsum[etype],
                                                   vertex_cumsum[src_vtype],
                                                   vertex_cumsum[dst_vtype]);
      for (uint64_t g = 0; g < batch_size; ++g) {
        HeteroGraphPtr rgptr = UnitGraph::CreateFromCSR(
          (src_vtype == dst_vtype) ? 1 : 2, res[g],
          ParseSparseFormat(restrict_format));

        rel_graphs[g].push_back(rgptr);
      }
    }
  } else if (FORMAT_HAS_CSC(format)) {
    for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
      auto pair = meta_graph->FindEdge(etype);
      const dgl_type_t src_vtype = pair.first;
      const dgl_type_t dst_vtype = pair.second;
      // CSR and CSC have the same storage format, i.e. CSRMatrix
      aten::CSRMatrix csc = batched_graph->GetCSCMatrix(etype);
      auto res = aten::DisjointPartitionCsrBySizes(csc,
                                                   batch_size,
                                                   edge_cumsum[etype],
                                                   vertex_cumsum[dst_vtype],
                                                   vertex_cumsum[src_vtype]);
      for (uint64_t g = 0; g < batch_size; ++g) {
        HeteroGraphPtr rgptr = UnitGraph::CreateFromCSC(
          (src_vtype == dst_vtype) ? 1 : 2, res[g],
          SparseFormat::kAny);

        rel_graphs[g].push_back(rgptr);
      }
    }
  }

  std::vector<HeteroGraphPtr> rst;
  std::vector<int64_t> num_nodes_per_type(num_vertex_types);
  for (uint64_t g = 0; g < batch_size; ++g) {
    for (uint64_t i = 0; i < num_vertex_types; ++i)
      num_nodes_per_type[i] = vertex_sizes_data[i * batch_size + g];
    rst.push_back(CreateHeteroGraph(meta_graph, rel_graphs[g], num_nodes_per_type));
  }
  return rst;
}

template <class IdType>
HeteroGraphPtr DisjointUnionHeteroGraph(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs) {
  CHECK_GT(component_graphs.size(), 0) << "Input graph list is empty";
  std::vector<HeteroGraphPtr> rel_graphs(meta_graph->NumEdges());
  std::vector<int64_t> num_nodes_per_type(meta_graph->NumVertices(), 0);

  // Loop over all canonical etypes
  for (dgl_type_t etype = 0; etype < meta_graph->NumEdges(); ++etype) {
    auto pair = meta_graph->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    IdType src_offset = 0, dst_offset = 0;
    std::vector<IdType> result_src, result_dst;

    // Loop over all graphs
    for (size_t i = 0; i < component_graphs.size(); ++i) {
      const auto& cg = component_graphs[i];
      EdgeArray edges = cg->Edges(etype);
      size_t num_edges = cg->NumEdges(etype);
      const IdType* edges_src_data = static_cast<const IdType*>(edges.src->data);
      const IdType* edges_dst_data = static_cast<const IdType*>(edges.dst->data);

      // Loop over all edges
      for (size_t j = 0; j < num_edges; ++j) {
        // TODO(mufei): Should use array operations to implement this.
        result_src.push_back(edges_src_data[j] + src_offset);
        result_dst.push_back(edges_dst_data[j] + dst_offset);
      }
      // Update offsets
      src_offset += cg->NumVertices(src_vtype);
      dst_offset += cg->NumVertices(dst_vtype);
    }
    HeteroGraphPtr rgptr = UnitGraph::CreateFromCOO(
        (src_vtype == dst_vtype) ? 1 : 2, src_offset, dst_offset,
        aten::VecToIdArray(result_src, sizeof(IdType) * 8),
        aten::VecToIdArray(result_dst, sizeof(IdType) * 8));
    rel_graphs[etype] = rgptr;
    num_nodes_per_type[src_vtype] = src_offset;
    num_nodes_per_type[dst_vtype] = dst_offset;
  }
  return CreateHeteroGraph(meta_graph, rel_graphs, std::move(num_nodes_per_type));
}

template HeteroGraphPtr DisjointUnionHeteroGraph<int32_t>(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs);

template HeteroGraphPtr DisjointUnionHeteroGraph<int64_t>(
    GraphPtr meta_graph, const std::vector<HeteroGraphPtr>& component_graphs);

template <class IdType>
std::vector<HeteroGraphPtr> DisjointPartitionHeteroBySizes(
    GraphPtr meta_graph, HeteroGraphPtr batched_graph, IdArray vertex_sizes, IdArray edge_sizes) {
  // Sanity check for vertex sizes
  const uint64_t len_vertex_sizes = vertex_sizes->shape[0];
  const uint64_t* vertex_sizes_data = static_cast<uint64_t*>(vertex_sizes->data);
  const uint64_t num_vertex_types = meta_graph->NumVertices();
  const uint64_t batch_size = len_vertex_sizes / num_vertex_types;
  // Map vertex type to the corresponding node cum sum
  std::vector<std::vector<uint64_t>> vertex_cumsum;
  vertex_cumsum.resize(num_vertex_types);
  // Loop over all vertex types
  for (uint64_t vtype = 0; vtype < num_vertex_types; ++vtype) {
    vertex_cumsum[vtype].push_back(0);
    for (uint64_t g = 0; g < batch_size; ++g) {
      // We've flattened the number of vertices in the batch for all types
      vertex_cumsum[vtype].push_back(
        vertex_cumsum[vtype][g] + vertex_sizes_data[vtype * batch_size + g]);
    }
    CHECK_EQ(vertex_cumsum[vtype][batch_size], batched_graph->NumVertices(vtype))
      << "Sum of the given sizes must equal to the number of nodes for type " << vtype;
  }

  // Sanity check for edge sizes
  const uint64_t* edge_sizes_data = static_cast<uint64_t*>(edge_sizes->data);
  const uint64_t num_edge_types = meta_graph->NumEdges();
  // Map edge type to the corresponding edge cum sum
  std::vector<std::vector<uint64_t>> edge_cumsum;
  edge_cumsum.resize(num_edge_types);
  // Loop over all edge types
  for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
    edge_cumsum[etype].push_back(0);
    for (uint64_t g = 0; g < batch_size; ++g) {
      // We've flattened the number of edges in the batch for all types
      edge_cumsum[etype].push_back(
        edge_cumsum[etype][g] + edge_sizes_data[etype * batch_size + g]);
    }
    CHECK_EQ(edge_cumsum[etype][batch_size], batched_graph->NumEdges(etype))
      << "Sum of the given sizes must equal to the number of edges for type " << etype;
  }

  // Construct relation graphs for unbatched graphs
  std::vector<std::vector<HeteroGraphPtr>> rel_graphs;
  rel_graphs.resize(batch_size);
  // Loop over all edge types
  for (uint64_t etype = 0; etype < num_edge_types; ++etype) {
    auto pair = meta_graph->FindEdge(etype);
    const dgl_type_t src_vtype = pair.first;
    const dgl_type_t dst_vtype = pair.second;
    EdgeArray edges = batched_graph->Edges(etype);
    const IdType* edges_src_data = static_cast<const IdType*>(edges.src->data);
    const IdType* edges_dst_data = static_cast<const IdType*>(edges.dst->data);
    // Loop over all graphs to be unbatched
    for (uint64_t g = 0; g < batch_size; ++g) {
      std::vector<IdType> result_src, result_dst;
      // Loop over the chunk of edges for the specified graph and edge type
      for (uint64_t e = edge_cumsum[etype][g]; e < edge_cumsum[etype][g + 1]; ++e) {
        // TODO(mufei): Should use array operations to implement this.
        result_src.push_back(edges_src_data[e] - vertex_cumsum[src_vtype][g]);
        result_dst.push_back(edges_dst_data[e] - vertex_cumsum[dst_vtype][g]);
      }
      HeteroGraphPtr rgptr = UnitGraph::CreateFromCOO(
          (src_vtype == dst_vtype) ? 1 : 2,
          vertex_sizes_data[src_vtype * batch_size + g],
          vertex_sizes_data[dst_vtype * batch_size + g],
          aten::VecToIdArray(result_src, sizeof(IdType) * 8),
          aten::VecToIdArray(result_dst, sizeof(IdType) * 8));
      rel_graphs[g].push_back(rgptr);
    }
  }

  std::vector<HeteroGraphPtr> rst;
  std::vector<int64_t> num_nodes_per_type(num_vertex_types);
  for (uint64_t g = 0; g < batch_size; ++g) {
    for (uint64_t i = 0; i < num_vertex_types; ++i)
      num_nodes_per_type[i] = vertex_sizes_data[i * batch_size + g];
    rst.push_back(CreateHeteroGraph(meta_graph, rel_graphs[g], num_nodes_per_type));
  }
  return rst;
}

template std::vector<HeteroGraphPtr> DisjointPartitionHeteroBySizes<int32_t>(
    GraphPtr meta_graph, HeteroGraphPtr batched_graph, IdArray vertex_sizes, IdArray edge_sizes);

template std::vector<HeteroGraphPtr> DisjointPartitionHeteroBySizes<int64_t>(
    GraphPtr meta_graph, HeteroGraphPtr batched_graph, IdArray vertex_sizes, IdArray edge_sizes);

}  // namespace dgl
