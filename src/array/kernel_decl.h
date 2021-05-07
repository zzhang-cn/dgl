/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/kernel_decl.h
 * \brief Sparse matrix format-specific operator declarations.
 */
#ifndef DGL_ARRAY_KERNEL_DECL_H_
#define DGL_ARRAY_KERNEL_DECL_H_

#include <dgl/bcast.h>
#include <dgl/base_heterograph.h>
#include <dgl/runtime/ndarray.h>

#include <string>
#include <vector>
#include <utility>

namespace dgl {
namespace aten {

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Csr format
 with heterograph support.
 */
template <int XPU, typename IdType, int bits>
void SpMMCsrHetero(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const std::vector<CSRMatrix>& csr,
             std::vector<NDArray> ufeat,
             NDArray efeat,
             std::vector<NDArray> out,
             std::vector<NDArray> out_aux,
             const std::vector<dgl_type_t> ufeat_eid,
             const std::vector<dgl_type_t> out_eid);
/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target);
/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Csr 
 format with heterograph support.
  */
template <int XPU, typename IdType, int bits>
void SDDMMCsrHetero(const std::string& op,
              const BcastOff& bcast,
              const std::vector<CSRMatrix>& vec_csr,
              std::vector<NDArray> vec_lhs,
              std::vector<NDArray> vec_rhs,
              std::vector<NDArray> vec_out,
              int lhs_target,
              int rhs_target,
              const std::vector<dgl_type_t> ufeat_eid,
              const std::vector<dgl_type_t> out_eid);

/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, int bits>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target);

/*!
 * \brief Segment reduce.
 */
template <int XPU, typename IdType, int bits>
void SegmentReduce(const std::string& op,
                   NDArray feat,
                   NDArray offsets,
                   NDArray out,
                   NDArray arg);

/*!
 * \brief Scatter Add on first dimension.
 */
template <int XPU, typename IdType, int bits>
void ScatterAdd(NDArray feat,
                NDArray idx,
                NDArray out);

/*!
 * \brief Backward function of segment cmp.
 */
template <int XPU, typename IdType, int bits>
void BackwardSegmentCmp(NDArray feat,
                        NDArray arg,
                        NDArray out);

/*!
 * \brief Sparse-sparse matrix multiplication
 *
 * \param A The left operand.
 * \param A_weights The weights of matrix as a 1D tensor.
 * \param B The right operand.
 * \param B_weights The weights of matrix as a 1D tensor.
 *
 * \note GPU implementation will cast the indices to 32 bit.
 * \note The zero entries in the result are not removed.
 * \note The CSR matrix should not have duplicate entries.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B,
    NDArray B_weights);

/*!
 * \brief Sparse-sparse matrix summation.
 *
 * \param A The sparse matrices with the same size.
 * \param A_weights The weights of each sparse matrix as a 1D tensor.
 *
 * \note GPU implementation will cast the indices to 32 bit.
 * \note The zero entries in the result are not removed.
 * \note The CSR matrix should not have duplicate entries.
 */
template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRSum(
    const std::vector<CSRMatrix>& A,
    const std::vector<NDArray>& A_weights);

}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_KERNEL_DECL_H_
