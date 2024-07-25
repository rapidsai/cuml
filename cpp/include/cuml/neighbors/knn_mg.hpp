/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#pragma once

#include <cumlprims/opg/matrix/data.hpp>
#include <cumlprims/opg/matrix/part_descriptor.hpp>
#include <raft/core/handle.hpp>

#include <vector>

namespace ML {
namespace KNN {
namespace opg {

using namespace MLCommon;

/**
 * Performs a multi-node multi-GPU KNN.
 * @param[in] handle the raft::handle_t to use for managing resources
 * @param[out] out_I vector of output index partitions. size should match the
 *        number of local input partitions.
 * @param[out] out_D vector of output distance partitions. size should match
 *        the number of local input partitions.
 * @param[in] idx_data vector of local indices to query
 * @param[in] idx_desc describes how the index partitions are distributed
 *        across the ranks.
 * @param[in] query_data vector of local query partitions
 * @param[in] query_desc describes how the query partitions are distributed
 *        across the cluster.
 * @param[in] rowMajorIndex boolean indicating whether the index is row major.
 * @param[in] rowMajorQuery boolean indicating whether the query is row major.
 * @param[in] k the number of neighbors to query
 * @param[in] batch_size the max number of rows to broadcast at a time
 * @param[in] verbose print extra logging info
 */
void knn(raft::handle_t& handle,
         std::vector<Matrix::Data<int64_t>*>* out_I,
         std::vector<Matrix::floatData_t*>* out_D,
         std::vector<Matrix::floatData_t*>& idx_data,
         Matrix::PartDescriptor& idx_desc,
         std::vector<Matrix::floatData_t*>& query_data,
         Matrix::PartDescriptor& query_desc,
         bool rowMajorIndex,
         bool rowMajorQuery,
         int k,
         size_t batch_size,
         bool verbose);

/**
 * Performs a multi-node multi-GPU KNN classify.
 * @param[in] handle the raft::handle_t to use for managing resources
 * @param[out] out vector of output labels partitions. size should match the
 *        number of local input partitions.
 * @param[in] probas (optional) pointer to a vector containing arrays of probabilities
 * @param[in] idx_data vector of local indices to query
 * @param[in] idx_desc describes how the index partitions are distributed
 *        across the ranks.
 * @param[in] query_data vector of local query partitions
 * @param[in] query_desc describes how the query partitions are distributed
 *        across the cluster.
 * @param[in] y vector of vector of label arrays. for multilabel classification, each
 *          element in the vector is a different "output" array of labels corresponding
 *          to the i'th output. size should match the number of local input partitions.
 * @param[in] uniq_labels vector of the sorted unique labels for each array in y
 * @param[in] n_unique vector of sizes for each array in uniq_labels
 * @param[in] rowMajorIndex boolean indicating whether the index is row major.
 * @param[in] rowMajorQuery boolean indicating whether the query is row major.
 * @param[in] probas_only return probas instead of performing complete knn_classify
 * @param[in] k the number of neighbors to query
 * @param[in] batch_size the max number of rows to broadcast at a time
 * @param[in] verbose print extra logging info
 */
void knn_classify(raft::handle_t& handle,
                  std::vector<Matrix::Data<int>*>* out,
                  std::vector<std::vector<float*>>* probas,
                  std::vector<Matrix::floatData_t*>& idx_data,
                  Matrix::PartDescriptor& idx_desc,
                  std::vector<Matrix::floatData_t*>& query_data,
                  Matrix::PartDescriptor& query_desc,
                  std::vector<std::vector<int*>>& y,
                  std::vector<int*>& uniq_labels,
                  std::vector<int>& n_unique,
                  bool rowMajorIndex = false,
                  bool rowMajorQuery = false,
                  bool probas_only   = false,
                  int k              = 10,
                  size_t batch_size  = 1 << 15,
                  bool verbose       = false);

/**
 * Performs a multi-node multi-GPU KNN regress.
 * @param[in] handle the raft::handle_t to use for managing resources
 * @param[out] out vector of output partitions. size should match the
 *        number of local input partitions.
 * @param[in] idx_data vector of local indices to query
 * @param[in] idx_desc describes how the index partitions are distributed
 *        across the ranks.
 * @param[in] query_data vector of local query partitions
 * @param[in] query_desc describes how the query partitions are distributed
 *        across the cluster.
 * @param[in] y vector of vector of output arrays. for multi-output regression, each
 *          element in the vector is a different "output" array corresponding
 *          to the i'th output. size should match the number of local input partitions.
 * @param[in] rowMajorIndex boolean indicating whether the index is row major.
 * @param[in] rowMajorQuery boolean indicating whether the query is row major.
 * @param[in] k the number of neighbors to query
 * @param[in] n_outputs number of outputs
 * @param[in] batch_size the max number of rows to broadcast at a time
 * @param[in] verbose print extra logging info
 */
void knn_regress(raft::handle_t& handle,
                 std::vector<Matrix::Data<float>*>* out,
                 std::vector<Matrix::floatData_t*>& idx_data,
                 Matrix::PartDescriptor& idx_desc,
                 std::vector<Matrix::floatData_t*>& query_data,
                 Matrix::PartDescriptor& query_desc,
                 std::vector<std::vector<float*>>& y,
                 bool rowMajorIndex,
                 bool rowMajorQuery,
                 int k,
                 int n_outputs,
                 size_t batch_size,
                 bool verbose);

};  // END namespace opg
};  // namespace KNN
};  // namespace ML
