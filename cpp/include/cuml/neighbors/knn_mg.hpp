/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <vector>

#include <opg/matrix/data.hpp>
#include <opg/matrix/part_descriptor.hpp>

#include <common/cumlHandle.hpp>

#pragma once

namespace MLCommon {
namespace Selection {
namespace opg {

/**
 * Performs a multi-node multi-GPU brute force nearest neighbors.
 * @param handle the cumlHandle to use for managing resources
 * @param out_I vector of output index partitions. size should match the
 *        number of local input partitions.
 * @param out_D vector of output distance partitions. size should match
 *        the number of local input partitions.
 * @param idx_data vector of local indices to query
 * @param idx_desc describes how the index partitions are distributed
 *        across the ranks.
 * @param query_data vector of local query partitions
 * @param query_desc describes how the query partitions are distributed
 *        across the cluster.
 * @param k the numeber of neighbors to query
 * @param batch_size the max number of rows to broadcast at a time
 * @param verbose print extra logging info
 */
void brute_force_knn(ML::cumlHandle &handle,
                     std::vector<Matrix::Data<int64_t> *> &out_I,
                     std::vector<Matrix::floatData_t *> &out_D,
                     std::vector<Matrix::floatData_t *> &idx_data,
                     Matrix::PartDescriptor &idx_desc,
                     std::vector<Matrix::floatData_t *> &query_data,
                     Matrix::PartDescriptor &query_desc,
                     bool rowMajorIndex = false, bool rowMajorQuery = false,
                     int k = 10, size_t batch_size = 1 << 15,
                     bool verbose = false);

};  // END namespace opg
};  // namespace Selection
};  // namespace MLCommon
