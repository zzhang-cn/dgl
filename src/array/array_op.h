/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array_op.h
 * \brief Array operator templates
 */
#ifndef DGL_ARRAY_ARRAY_OP_H_
#define DGL_ARRAY_ARRAY_OP_H_

#include <dgl/array.h>
#include <dgl/graph_traversal.h>
#include <vector>
#include <tuple>
#include <utility>

namespace dgl {
namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
IdArray Full(IdType val, int64_t length, DLContext ctx);

template <DLDeviceType XPU, typename IdType>
IdArray Range(IdType low, IdType high, DLContext ctx);

template <DLDeviceType XPU, typename IdType>
IdArray AsNumBits(IdArray arr, uint8_t bits);

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdArray rhs);

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdArray lhs, IdType rhs);

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray BinaryElewise(IdType lhs, IdArray rhs);

template <DLDeviceType XPU, typename IdType, typename Op>
IdArray UnaryElewise(IdArray array);

template <DLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index);

template <DLDeviceType XPU, typename DType>
DType IndexSelect(NDArray array, int64_t index);

template <DLDeviceType XPU, typename DType>
IdArray NonZero(BoolArray bool_arr);

template <DLDeviceType XPU, typename DType>
std::pair<IdArray, IdArray> Sort(IdArray array, int num_bits);

template <DLDeviceType XPU, typename DType, typename IdType>
NDArray Scatter(NDArray array, IdArray indices);

template <DLDeviceType XPU, typename DType, typename IdType>
void Scatter_(IdArray index, NDArray value, NDArray out);

template <DLDeviceType XPU, typename DType, typename IdType>
NDArray Repeat(NDArray array, IdArray repeats);

template <DLDeviceType XPU, typename IdType>
IdArray Relabel_(const std::vector<IdArray>& arrays);

template <DLDeviceType XPU, typename IdType>
NDArray Concat(const std::vector<IdArray>& arrays);

template <DLDeviceType XPU, typename DType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, DType pad_value);

template <DLDeviceType XPU, typename DType, typename IdType>
std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths);

template <DLDeviceType XPU, typename IdType>
IdArray CumSum(IdArray array, bool prepend_zero);

template <DLDeviceType XPU, typename IdType>
IdArray NonZero(NDArray array);

// sparse arrays

template <DLDeviceType XPU, typename IdType>
bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray CSRIsNonZero(CSRMatrix csr, runtime::NDArray row, runtime::NDArray col);

template <DLDeviceType XPU, typename IdType>
bool CSRHasDuplicate(CSRMatrix csr);

template <DLDeviceType XPU, typename IdType>
int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowNNZ(CSRMatrix csr, runtime::NDArray row);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray CSRGetRowData(CSRMatrix csr, int64_t row);

template <DLDeviceType XPU, typename IdType>
bool CSRIsSorted(CSRMatrix csr);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols, bool return_eids,
    runtime::NDArray weights, DType filler);

template <DLDeviceType XPU, typename IdType, typename DType>
runtime::NDArray CSRGetData(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols,
    runtime::NDArray weights, DType filler) {
  return CSRGetData<XPU, IdType, DType>(csr, rows, cols, false, weights, filler);
}

template <DLDeviceType XPU, typename IdType>
NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols) {
  return CSRGetData<XPU, IdType, IdType>(csr, rows, cols, true, NullArray(rows->dtype), -1);
}

template <DLDeviceType XPU, typename IdType>
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRTranspose(CSRMatrix csr);

// Convert CSR to COO
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOO(CSRMatrix csr);

// Convert CSR to COO using data array as order
template <DLDeviceType XPU, typename IdType>
COOMatrix CSRToCOODataAsOrder(CSRMatrix csr);

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end);

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceRows(CSRMatrix csr, runtime::NDArray rows);

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRSliceMatrix(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType>
void CSRSort_(CSRMatrix* csr);

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRReorder(CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

template <DLDeviceType XPU, typename IdType>
COOMatrix COOReorder(COOMatrix coo, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

template <DLDeviceType XPU, typename IdType>
CSRMatrix CSRRemove(CSRMatrix csr, IdArray entries);

// FloatType is the type of probability data.
template <DLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob, bool replace);

template <DLDeviceType XPU, typename IdType>
COOMatrix CSRRowWiseSamplingUniform(
    CSRMatrix mat, IdArray rows, int64_t num_samples, bool replace);

// FloatType is the type of weight data.
template <DLDeviceType XPU, typename IdType, typename DType>
COOMatrix CSRRowWiseTopk(
    CSRMatrix mat, IdArray rows, int64_t k, NDArray weight, bool ascending);

// Union CSRMatrixes
template <DLDeviceType XPU, typename IdType>
CSRMatrix UnionCsr(const std::vector<CSRMatrix>& csrs);

template <DLDeviceType XPU, typename IdType>
std::tuple<CSRMatrix, IdArray, IdArray> CSRToSimple(CSRMatrix csr);

///////////////////////////////////////////////////////////////////////////////////////////

template <DLDeviceType XPU, typename IdType>
bool COOIsNonZero(COOMatrix coo, int64_t row, int64_t col);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray COOIsNonZero(COOMatrix coo, runtime::NDArray row, runtime::NDArray col);

template <DLDeviceType XPU, typename IdType>
bool COOHasDuplicate(COOMatrix coo);

template <DLDeviceType XPU, typename IdType>
int64_t COOGetRowNNZ(COOMatrix coo, int64_t row);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray COOGetRowNNZ(COOMatrix coo, runtime::NDArray row);

template <DLDeviceType XPU, typename IdType>
std::pair<runtime::NDArray, runtime::NDArray>
COOGetRowDataAndIndices(COOMatrix coo, int64_t row);

template <DLDeviceType XPU, typename IdType>
std::vector<runtime::NDArray> COOGetDataAndIndices(
    COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType>
runtime::NDArray COOGetData(COOMatrix mat, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType>
COOMatrix COOTranspose(COOMatrix coo);

template <DLDeviceType XPU, typename IdType>
CSRMatrix COOToCSR(COOMatrix coo);

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end);

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSliceRows(COOMatrix coo, runtime::NDArray rows);

template <DLDeviceType XPU, typename IdType>
COOMatrix COOSliceMatrix(COOMatrix coo, runtime::NDArray rows, runtime::NDArray cols);

template <DLDeviceType XPU, typename IdType>
std::pair<COOMatrix, IdArray> COOCoalesce(COOMatrix coo);

template <DLDeviceType XPU, typename IdType>
void COOSort_(COOMatrix* mat, bool sort_column);

template <DLDeviceType XPU, typename IdType>
std::pair<bool, bool> COOIsSorted(COOMatrix coo);

template <DLDeviceType XPU, typename IdType>
COOMatrix COORemove(COOMatrix coo, IdArray entries);

// FloatType is the type of probability data.
template <DLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix COORowWiseSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob, bool replace);

template <DLDeviceType XPU, typename IdType>
COOMatrix COORowWiseSamplingUniform(
    COOMatrix mat, IdArray rows, int64_t num_samples, bool replace);

// FloatType is the type of weight data.
template <DLDeviceType XPU, typename IdType, typename FloatType>
COOMatrix COORowWiseTopk(
    COOMatrix mat, IdArray rows, int64_t k, FloatArray weight, bool ascending);

///////////////////////// Graph Traverse routines //////////////////////////

template <DLDeviceType XPU, typename IdType>
Frontiers BFSNodesFrontiers(const CSRMatrix& csr, IdArray source);

template <DLDeviceType XPU, typename IdType>
Frontiers BFSEdgesFrontiers(const CSRMatrix& csr, IdArray source);

template <DLDeviceType XPU, typename IdType>
Frontiers TopologicalNodesFrontiers(const CSRMatrix& csr);

template <DLDeviceType XPU, typename IdType>
Frontiers DGLDFSEdges(const CSRMatrix& csr, IdArray source);

template <DLDeviceType XPU, typename IdType>
Frontiers DGLDFSLabeledEdges(const CSRMatrix& csr,
                             IdArray source,
                             const bool has_reverse_edge,
                             const bool has_nontree_edge,
                             const bool return_labels);

template <DLDeviceType XPU, typename IdType>
COOMatrix COOLineGraph(const COOMatrix &coo, bool backtracking);

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_ARRAY_OP_H_
