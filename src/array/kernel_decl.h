/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/kernel_decl.h
 * \brief Sparse matrix format-specific operator declarations.
 */
#ifndef DGL_ARRAY_KERNEL_DECL_H_
#define DGL_ARRAY_KERNEL_DECL_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <dgl/base_heterograph.h>

#include <string>
#include <vector>

namespace dgl {
namespace aten {

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Csr format.
 */
template <int XPU, typename IdType, typename DType>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const aten::CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux);

/*!
 * \brief Generalized Sparse Matrix Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, typename DType>
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
template <int XPU, typename IdType, typename DType>
void SDDMMCsr(const std::string& op,
              const BcastOff& bcast,
              const aten::CSRMatrix& csr,
              NDArray ufeat,
              NDArray efeat,
              NDArray out);

/*!
 * \brief Generalized Sampled Dense-Dense Matrix Multiplication on Coo format.
 */
template <int XPU, typename IdType, typename DType>
void SDDMMCoo(const std::string& op,
              const BcastOff& bcast,
              const aten::COOMatrix& coo,
              NDArray ufeat,
              NDArray efeat,
              NDArray out);

}  // namespace
}  // namespace

#endif  // DGL_ARRAY_KERNEL_DECL_H_
