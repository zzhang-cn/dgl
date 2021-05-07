/*!
 *  Copyright (c) 2020 by Contributors
 * \file dgl/aten/csr.h
 * \brief Common CSR operations required by DGL.
 */
#ifndef DGL_ATEN_CSR_H_
#define DGL_ATEN_CSR_H_

#include <dmlc/io.h>
#include <dmlc/serializer.h>
#include <vector>
#include <tuple>
#include <string>
#include "./types.h"
#include "./array_ops.h"
#include "./spmat.h"
#include "./macro.h"

namespace dgl {
namespace aten {

struct COOMatrix;

/*!
 * \brief Plain CSR matrix
 *
 * The column indices are 0-based and are not necessarily sorted. The data array stores
 * integer ids for reading edge features.
 *
 * Note that we do allow duplicate non-zero entries -- multiple non-zero entries
 * that have the same row, col indices. It corresponds to multigraph in
 * graph terminology.
 */

constexpr uint64_t kDGLSerialize_AtenCsrMatrixMagic = 0xDD6cd31205dff127;

struct CSRMatrix {
  /*! \brief the dense shape of the matrix */
  int64_t num_rows = 0, num_cols = 0;
  /*! \brief CSR index arrays */
  IdArray indptr, indices;
  /*! \brief data index array. When is null, assume it is from 0 to NNZ - 1. */
  IdArray data;
  /*! \brief whether the column indices per row are sorted */
  bool sorted = false;
  /*! \brief default constructor */
  CSRMatrix() = default;
  /*! \brief constructor */
  CSRMatrix(int64_t nrows, int64_t ncols, IdArray parr, IdArray iarr,
            IdArray darr = NullArray(), bool sorted_flag = false)
      : num_rows(nrows),
        num_cols(ncols),
        indptr(parr),
        indices(iarr),
        data(darr),
        sorted(sorted_flag) {
    CheckValidity();
  }

  /*! \brief constructor from SparseMatrix object */
  explicit CSRMatrix(const SparseMatrix& spmat)
      : num_rows(spmat.num_rows),
        num_cols(spmat.num_cols),
        indptr(spmat.indices[0]),
        indices(spmat.indices[1]),
        data(spmat.indices[2]),
        sorted(spmat.flags[0]) {
    CheckValidity();
  }

  // Convert to a SparseMatrix object that can return to python.
  SparseMatrix ToSparseMatrix() const {
    return SparseMatrix(static_cast<int32_t>(SparseFormat::kCSR), num_rows,
                        num_cols, {indptr, indices, data}, {sorted});
  }

  bool Load(dmlc::Stream* fs) {
    uint64_t magicNum;
    CHECK(fs->Read(&magicNum)) << "Invalid Magic Number";
    CHECK_EQ(magicNum, kDGLSerialize_AtenCsrMatrixMagic)
        << "Invalid CSRMatrix Data";
    CHECK(fs->Read(&num_cols)) << "Invalid num_cols";
    CHECK(fs->Read(&num_rows)) << "Invalid num_rows";
    CHECK(fs->Read(&indptr)) << "Invalid indptr";
    CHECK(fs->Read(&indices)) << "Invalid indices";
    CHECK(fs->Read(&data)) << "Invalid data";
    CHECK(fs->Read(&sorted)) << "Invalid sorted";
    CheckValidity();
    return true;
  }

  void Save(dmlc::Stream* fs) const {
    fs->Write(kDGLSerialize_AtenCsrMatrixMagic);
    fs->Write(num_cols);
    fs->Write(num_rows);
    fs->Write(indptr);
    fs->Write(indices);
    fs->Write(data);
    fs->Write(sorted);
  }

  inline void CheckValidity() const {
    CHECK_SAME_DTYPE(indptr, indices);
    CHECK_SAME_CONTEXT(indptr, indices);
    if (!aten::IsNullArray(data)) {
      CHECK_SAME_DTYPE(indptr, data);
      CHECK_SAME_CONTEXT(indptr, data);
    }
    CHECK_NO_OVERFLOW(indptr->dtype, num_rows);
    CHECK_NO_OVERFLOW(indptr->dtype, num_cols);
    CHECK_EQ(indptr->shape[0], num_rows + 1);
  }

  /*! \brief Return a copy of this matrix on the give device context. */
  inline CSRMatrix CopyTo(const DLContext& ctx) const {
    if (ctx == indptr->ctx)
      return *this;
    return CSRMatrix(num_rows, num_cols,
                     indptr.CopyTo(ctx), indices.CopyTo(ctx),
                     aten::IsNullArray(data)? data : data.CopyTo(ctx),
                     sorted);
  }
};

///////////////////////// CSR routines //////////////////////////

/*! \brief Return true if the value (row, col) is non-zero */
bool CSRIsNonZero(CSRMatrix , int64_t row, int64_t col);
/*!
 * \brief Batched implementation of CSRIsNonZero.
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 */
runtime::NDArray CSRIsNonZero(CSRMatrix, runtime::NDArray row, runtime::NDArray col);

/*! \brief Return the nnz of the given row */
int64_t CSRGetRowNNZ(CSRMatrix , int64_t row);
runtime::NDArray CSRGetRowNNZ(CSRMatrix , runtime::NDArray row);

/*! \brief Return the column index array of the given row */
runtime::NDArray CSRGetRowColumnIndices(CSRMatrix , int64_t row);

/*! \brief Return the data array of the given row */
runtime::NDArray CSRGetRowData(CSRMatrix , int64_t row);

/*! \brief Whether the CSR matrix contains data */
inline bool CSRHasData(CSRMatrix csr) {
  return !IsNullArray(csr.data);
}

/*! \brief Whether the column indices of each row is sorted. */
bool CSRIsSorted(CSRMatrix csr);

/*!
 * \brief Get the data and the row,col indices for each returned entries.
 *
 * The operator supports matrix with duplicate entries and all the matched entries
 * will be returned. The operator assumes there is NO duplicate (row, col) pair
 * in the given input. Otherwise, the returned result is undefined.
 *
 * If some (row, col) pairs do not contain a valid non-zero elements,
 * they will not be included in the return arrays.
 *
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 * \param mat Sparse matrix
 * \param rows Row index
 * \param cols Column index
 * \return Three arrays {rows, cols, data}
 */
std::vector<runtime::NDArray> CSRGetDataAndIndices(
    CSRMatrix , runtime::NDArray rows, runtime::NDArray cols);

/* \brief Get data. The return type is an ndarray due to possible duplicate entries. */
inline runtime::NDArray CSRGetAllData(CSRMatrix mat, int64_t row, int64_t col) {
  const auto& nbits = mat.indptr->dtype.bits;
  const auto& ctx = mat.indptr->ctx;
  IdArray rows = VecToIdArray<int64_t>({row}, nbits, ctx);
  IdArray cols = VecToIdArray<int64_t>({col}, nbits, ctx);
  const auto& rst = CSRGetDataAndIndices(mat, rows, cols);
  return rst[2];
}

/*!
 * \brief Get the data for each (row, col) pair.
 *
 * The operator supports matrix with duplicate entries but only one matched entry
 * will be returned for each (row, col) pair. Support duplicate input (row, col)
 * pairs.
 *
 * If some (row, col) pairs do not contain a valid non-zero elements,
 * their data values are filled with -1.
 *
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 *
 * \param mat Sparse matrix.
 * \param rows Row index.
 * \param cols Column index.
 * \return Data array. The i^th element is the data of (rows[i], cols[i])
 */
runtime::NDArray CSRGetData(CSRMatrix, runtime::NDArray rows, runtime::NDArray cols);

/*!
 * \brief Get the data for each (row, col) pair, then index into the weights array.
 *
 * The operator supports matrix with duplicate entries but only one matched entry
 * will be returned for each (row, col) pair. Support duplicate input (row, col)
 * pairs.
 *
 * If some (row, col) pairs do not contain a valid non-zero elements to index into the
 * weights array, DGL returns the value \a filler for that pair instead.
 *
 * \note This operator allows broadcasting (i.e, either row or col can be of length 1).
 *
 * \tparam DType the data type of the weights array.
 * \param mat Sparse matrix.
 * \param rows Row index.
 * \param cols Column index.
 * \param weights The weights array.
 * \param filler The value to return for row-column pairs not existent in the matrix.
 * \return Data array. The i^th element is the data of (rows[i], cols[i])
 */
template <typename DType>
runtime::NDArray CSRGetData(
    CSRMatrix, runtime::NDArray rows, runtime::NDArray cols, runtime::NDArray weights,
    DType filler);

/*! \brief Return a transposed CSR matrix */
CSRMatrix CSRTranspose(CSRMatrix csr);

/*!
 * \brief Convert CSR matrix to COO matrix.
 *
 * Complexity: O(nnz)
 *
 * - If data_as_order is false, the column and data arrays of the
 *   result COO are equal to the indices and data arrays of the
 *   input CSR. The result COO is also row sorted.
 * - If the input CSR is further sorted, the result COO is also
 *   column sorted.
 *
 * \param csr Input csr matrix
 * \param data_as_order If true, the data array in the input csr matrix contains the order
 *                      by which the resulting COO tuples are stored. In this case, the
 *                      data array of the resulting COO matrix will be empty because it
 *                      is essentially a consecutive range.
 * \return a coo matrix
 */
COOMatrix CSRToCOO(CSRMatrix csr, bool data_as_order);

/*!
 * \brief Slice rows of the given matrix and return.
 *
 * The sliced row IDs are relabeled to starting from zero.
 *
 * Examples:
 * num_rows = 4
 * num_cols = 4
 * indptr = [0, 2, 3, 3, 5]
 * indices = [1, 0, 2, 3, 1]
 *
 *  After CSRSliceRows(csr, 1, 3)
 *
 * num_rows = 2
 * num_cols = 4
 * indptr = [0, 1, 1]
 * indices = [2]
 *
 * \param csr CSR matrix
 * \param start Start row id (inclusive)
 * \param end End row id (exclusive)
 * \return sliced rows stored in a CSR matrix
 */
CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end);
CSRMatrix CSRSliceRows(CSRMatrix csr, runtime::NDArray rows);

/*!
 * \brief Get the submatrix specified by the row and col ids.
 *
 * In numpy notation, given matrix M, row index array I, col index array J
 * This function returns the submatrix M[I, J]. It assumes that there is no
 * duplicate (row, col) pair in the given indices. M could have duplicate
 * entries.
 *
 * The sliced row and column IDs are relabeled according to the given
 * rows and cols (i.e., row #0 in the new matrix corresponds to rows[0] in
 * the original matrix).
 *
 * \param csr The input csr matrix
 * \param rows The row index to select
 * \param cols The col index to select
 * \return submatrix
 */
CSRMatrix CSRSliceMatrix(CSRMatrix csr, runtime::NDArray rows, runtime::NDArray cols);

/*! \return True if the matrix has duplicate entries */
bool CSRHasDuplicate(CSRMatrix csr);

/*!
 * \brief Sort the column index at each row in ascending order in-place.
 *
 * Only the indices and data arrays (if available) will be mutated. The indptr array
 * stays the same.
 *
 * Examples:
 * num_rows = 4
 * num_cols = 4
 * indptr = [0, 2, 3, 3, 5]
 * indices = [1, 0, 2, 3, 1]
 *
 *  After CSRSort_(&csr)
 *
 * indptr = [0, 2, 3, 3, 5]
 * indices = [0, 1, 1, 2, 3]
 */
void CSRSort_(CSRMatrix* csr);

/*!
 * \brief Sort the column index at each row in ascending order.
 *
 * Return a new CSR matrix with sorted column indices and data arrays.
 */
inline CSRMatrix CSRSort(CSRMatrix csr) {
  if (csr.sorted)
    return csr;
  CSRMatrix ret(csr.num_rows, csr.num_cols,
                csr.indptr, csr.indices.Clone(),
                CSRHasData(csr)? csr.data.Clone() : csr.data,
                csr.sorted);
  CSRSort_(&ret);
  return ret;
}

/*!
 * \brief Reorder the rows and colmns according to the new row and column order.
 * \param csr The input csr matrix.
 * \param new_row_ids the new row Ids (the index is the old row Id)
 * \param new_col_ids the new column Ids (the index is the old col Id).
 */
CSRMatrix CSRReorder(CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids);

/*!
 * \brief Remove entries from CSR matrix by entry indices (data indices)
 * \return A new CSR matrix as well as a mapping from the new CSR entries to the old CSR
 *         entries.
 */
CSRMatrix CSRRemove(CSRMatrix csr, IdArray entries);

/*!
 * \brief Randomly select a fixed number of non-zero entries along each given row independently.
 *
 * The function performs random choices along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than num_samples,
 * all the values are picked.
 *
 * Examples:
 *
 * // csr.num_rows = 4;
 * // csr.num_cols = 4;
 * // csr.indptr = [0, 2, 3, 3, 5]
 * // csr.indices = [0, 1, 1, 2, 3]
 * // csr.data = [2, 3, 0, 1, 4]
 * CSRMatrix csr = ...;
 * IdArray rows = ... ; // [1, 3]
 * COOMatrix sampled = CSRRowWiseSampling(csr, rows, 2, FloatArray(), false);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [1, 3, 3]
 * // sampled.cols = [1, 2, 3]
 * // sampled.data = [3, 0, 4]
 *
 * \param mat Input CSR matrix.
 * \param rows Rows to sample from.
 * \param num_samples Number of samples
 * \param prob Unnormalized probability array. Should be of the same length as the data array.
 *             If an empty array is provided, assume uniform.
 * \param replace True if sample with replacement
 * \return A COOMatrix storing the picked row, col and data indices.
 */
COOMatrix CSRRowWiseSampling(
    CSRMatrix mat,
    IdArray rows,
    int64_t num_samples,
    FloatArray prob = FloatArray(),
    bool replace = true);

/*!
 * \brief Select K non-zero entries with the largest weights along each given row.
 *
 * The function performs top-k selection along each row independently.
 * The picked indices are returned in the form of a COO matrix.
 *
 * If replace is false and a row has fewer non-zero values than k,
 * all the values are picked.
 *
 * Examples:
 *
 * // csr.num_rows = 4;
 * // csr.num_cols = 4;
 * // csr.indptr = [0, 2, 3, 3, 5]
 * // csr.indices = [0, 1, 1, 2, 3]
 * // csr.data = [2, 3, 0, 1, 4]
 * CSRMatrix csr = ...;
 * IdArray rows = ... ;  // [0, 1, 3]
 * FloatArray weight = ... ;  // [1., 0., -1., 10., 20.]
 * COOMatrix sampled = CSRRowWiseTopk(csr, rows, 1, weight);
 * // possible sampled coo matrix:
 * // sampled.num_rows = 4
 * // sampled.num_cols = 4
 * // sampled.rows = [0, 1, 3]
 * // sampled.cols = [1, 1, 2]
 * // sampled.data = [3, 0, 1]
 *
 * \param mat Input CSR matrix.
 * \param rows Rows to sample from.
 * \param k The K value.
 * \param weight Weight associated with each entry. Should be of the same length as the
 *               data array. If an empty array is provided, assume uniform.
 * \param ascending If true, elements are sorted by ascending order, equivalent to find
 *                 the K smallest values. Otherwise, find K largest values.
 * \return A COOMatrix storing the picked row and col indices. Its data field stores the
 *         the index of the picked elements in the value array.
 */
COOMatrix CSRRowWiseTopk(
    CSRMatrix mat,
    IdArray rows,
    int64_t k,
    FloatArray weight,
    bool ascending = false);

/*!
 * \brief Sort the column index according to the tag of each column.
 *
 * Example:
 * indptr  = [0, 5, 8]
 * indices = [0, 1, 2, 3, 4, 0, 1, 2]
 *
 * tag     = [1, 1, 0, 2, 0]
 *
 *  After CSRSortByTar
 *
 * indptr  = [0, 5, 8]
 * indices = [2, 4, 0, 1, 3, 2, 0, 1]
 * (tag)   = [0, 0, 1, 1, 2, 0, 1, 1]
 *           ^    ^     ^  ^
 *                         ^  ^     ^^
 * (the tag array itself is unchanged.)
 *
 * Return:
 * [[0, 2, 4, 5], [0, 1, 3, 3]] (marked with ^)
 *
 * \param csr The csr matrix to be sorted
 * \param tag_array Tag of each column. IdArray with length num_cols
 * \param num_tags Number of tags. It should be equal to max(tag_array)+1.
 * \return the split positions of different tags. NDArray of shape (num_rows, num_tags + 1)
 */
NDArray CSRSortByTag(const CSRMatrix* csr, const IdArray tag_array, int64_t num_tags, CSRMatrix* output);

/*
 * \brief Union two CSRMatrix into one CSRMatrix.
 *
 * Two Matrix must have the same shape.
 *
 * Example:
 *
 * A = [[0, 0, 1, 0],
 *      [1, 0, 1, 1],
 *      [0, 1, 0, 0]]
 *
 * B = [[0, 1, 1, 0],
 *      [0, 0, 0, 1],
 *      [0, 0, 1, 0]]
 *
 * CSRMatrix_A.num_rows : 3
 * CSRMatrix_A.num_cols : 4
 * CSRMatrix_B.num_rows : 3
 * CSRMatrix_B.num_cols : 4
 *
 * C = UnionCsr({A, B});
 *
 * C = [[0, 1, 2, 0],
 *      [1, 0, 1, 2],
 *      [0, 1, 1, 0]]
 *
 * CSRMatrix_C.num_rows : 3
 * CSRMatrix_C.num_cols : 4
 */
CSRMatrix UnionCsr(
  const std::vector<CSRMatrix>& csrs);

/*!
 * \brief Union a list CSRMatrix into one CSRMatrix.
 *
 * Examples:
 *
 * A = [[0, 0, 1],
 *      [1, 0, 1],
 *      [0, 1, 0]]
 *
 * B = [[0, 0],
 *      [1, 0]]
 *
 * CSRMatrix_A.num_rows : 3
 * CSRMatrix_A.num_cols : 3
 * CSRMatrix_B.num_rows : 2
 * CSRMatrix_B.num_cols : 2
 *
 * C = DisjointUnionCsr({A, B});
 *
 * C = [[0, 0, 1, 0, 0],
 *      [1, 0, 1, 0, 0],
 *      [0, 1, 0, 0, 0],
 *      [0, 0, 0, 0, 0],
 *      [0, 0, 0, 1, 0]]
 * CSRMatrix_C.num_rows : 5
 * CSRMatrix_C.num_cols : 5
 *
 * \param csrs The input list of csr matrix.
 * \param src_offset A list of integers recording src vertix id offset of each Matrix in csrs
 * \param src_offset A list of integers recording dst vertix id offset of each Matrix in csrs
 * \return The combined CSRMatrix.
 */
CSRMatrix DisjointUnionCsr(
  const std::vector<CSRMatrix>& csrs);

/*!
 * \brief CSRMatrix toSimple.
 *
 * A = [[0, 0, 0],
 *      [3, 0, 2],
 *      [1, 1, 0],
 *      [0, 0, 4]]
 *
 * B, cnt, edge_map = CSRToSimple(A)
 *
 * B = [[0, 0, 0],
 *      [1, 0, 1],
 *      [1, 1, 0],
 *      [0, 0, 1]]
 * cnt = [3, 2, 1, 1, 4]
 * edge_map = [0, 0, 0, 1, 1, 2, 3, 4, 4, 4, 4]
 *
 * \return The simplified CSRMatrix
 *         The count recording the number of duplicated edges from the original graph.
 *         The edge mapping from the edge IDs of original graph to those of the
 *         returned graph.
 */
std::tuple<CSRMatrix, IdArray, IdArray> CSRToSimple(const CSRMatrix& csr);

/*!
 * \brief Split a CSRMatrix into multiple disjoin components.
 *
 * Examples:
 *
 * C = [[0, 0, 1, 0, 0],
 *      [1, 0, 1, 0, 0],
 *      [0, 1, 0, 0, 0],
 *      [0, 0, 0, 0, 0],
 *      [0, 0, 0, 1, 0],
 *      [0, 0, 0, 0, 1]]
 * CSRMatrix_C.num_rows : 6
 * CSRMatrix_C.num_cols : 5
 *
 * batch_size : 2
 * edge_cumsum : [0, 4, 6]
 * src_vertex_cumsum : [0, 3, 6]
 * dst_vertex_cumsum : [0, 3, 5]
 *
 * ret = DisjointPartitionCsrBySizes(C,
 *                                   batch_size,
 *                                   edge_cumsum,
 *                                   src_vertex_cumsum,
 *                                   dst_vertex_cumsum)
 *
 * A = [[0, 0, 1],
 *      [1, 0, 1],
 *      [0, 1, 0]]
 * CSRMatrix_A.num_rows : 3
 * CSRMatrix_A.num_cols : 3
 *
 * B = [[0, 0],
 *      [1, 0],
 *      [0, 1]]
 * CSRMatrix_B.num_rows : 3
 * CSRMatrix_B.num_cols : 2
 *
 * \param csr CSRMatrix to split.
 * \param batch_size Number of disjoin components (Sub CSRMatrix)
 * \param edge_cumsum Number of edges of each components
 * \param src_vertex_cumsum Number of src vertices of each component.
 * \param dst_vertex_cumsum Number of dst vertices of each component.
 * \return A list of CSRMatrixes representing each disjoint components.
 */
std::vector<CSRMatrix> DisjointPartitionCsrBySizes(
  const CSRMatrix &csrs,
  const uint64_t batch_size,
  const std::vector<uint64_t> &edge_cumsum,
  const std::vector<uint64_t> &src_vertex_cumsum,
  const std::vector<uint64_t> &dst_vertex_cumsum);

}  // namespace aten
}  // namespace dgl

namespace dmlc {
DMLC_DECLARE_TRAITS(has_saveload, dgl::aten::CSRMatrix, true);
}  // namespace dmlc

#endif  // DGL_ATEN_CSR_H_
