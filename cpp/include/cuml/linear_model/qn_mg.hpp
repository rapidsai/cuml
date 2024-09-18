/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuml/common/logger.hpp>
#include <cuml/linear_model/qn.h>

#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
#include <raft/core/comms.hpp>

#include <cuda_runtime.h>

#include <vector>
using namespace MLCommon;

namespace ML {
namespace GLM {
namespace opg {

/**
 * @brief Calculate unique class labels across multiple GPUs in a multi-node environment.
 * @param[in] handle: the internal cuml handle object
 * @param[in] input_desc: PartDescriptor object for the input
 * @param[in] labels: labels data
 * @returns host vector that stores the distinct labels
 */
template <typename T>
std::vector<T> getUniquelabelsMG(const raft::handle_t& handle,
                                 Matrix::PartDescriptor& input_desc,
                                 std::vector<Matrix::Data<T>*>& labels);

/**
 * @brief performs MNMG fit operation for the logistic regression using quasi newton methods
 * @param[in] handle: the internal cuml handle object
 * @param[in] input_data: vector holding all partitions for that rank
 * @param[in] input_desc: PartDescriptor object for the input
 * @param[in] labels: labels data
 * @param[out] coef: learned coefficients
 * @param[in] pams: model parameters
 * @param[in] X_col_major: true if X is stored column-major
 * @param[in] standardization: whether to standardize the dataset before training
 * @param[in] n_classes: number of outputs (number of classes or `1` for regression)
 * @param[out] f: host pointer holding the final objective value
 * @param[out] num_iters: host pointer holding the actual number of iterations taken
 */
template <typename T>
void qnFit(raft::handle_t& handle,
           std::vector<Matrix::Data<T>*>& input_data,
           Matrix::PartDescriptor& input_desc,
           std::vector<Matrix::Data<T>*>& labels,
           T* coef,
           const qn_params& pams,
           bool X_col_major,
           bool standardization,
           int n_classes,
           T* f,
           int* num_iters);

/**
 * @brief support sparse vectors (Compressed Sparse Row format) for MNMG logistic regression fit
 * using quasi newton methods
 * @param[in] handle: the internal cuml handle object
 * @param[in] input_values: vector holding non-zero values of all partitions for that rank
 * @param[in] input_cols: vector holding column indices of non-zero values of all partitions for
 * that rank
 * @param[in] input_row_ids: vector holding row pointers of non-zero values of all partitions for
 * that rank
 * @param[in] X_nnz: the number of non-zero values of that rank
 * @param[in] standardization: whether to standardize the dataset before training
 * @param[in] input_desc: PartDescriptor object for the input
 * @param[in] labels: labels data
 * @param[out] coef: learned coefficients
 * @param[in] pams: model parameters
 * @param[in] n_classes: number of outputs (number of classes or `1` for regression)
 * @param[out] f: host pointer holding the final objective value
 * @param[out] num_iters: host pointer holding the actual number of iterations taken
 */
template <typename T, typename I>
void qnFitSparse(raft::handle_t& handle,
                 std::vector<Matrix::Data<T>*>& input_values,
                 I* input_cols,
                 I* input_row_ids,
                 I X_nnz,
                 Matrix::PartDescriptor& input_desc,
                 std::vector<Matrix::Data<T>*>& labels,
                 T* coef,
                 const qn_params& pams,
                 bool standardization,
                 int n_classes,
                 T* f,
                 int* num_iters);

};  // namespace opg
};  // namespace GLM
};  // namespace ML
