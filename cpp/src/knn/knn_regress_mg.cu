/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "knn_opg_common.cuh"

namespace ML {
namespace KNN {
namespace opg {

using namespace knn_common;

template struct KNN_RE_params<float, int64_t, float, float>;

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
                 bool verbose)
{
  KNN_RE_params<float, int64_t, float, float> params(knn_operation::regression,
                                                     &idx_data,
                                                     &idx_desc,
                                                     &query_data,
                                                     &query_desc,
                                                     rowMajorIndex,
                                                     rowMajorQuery,
                                                     k,
                                                     batch_size,
                                                     verbose,
                                                     n_outputs,
                                                     &y,
                                                     out);

  opg_knn(params, handle);
}
};  // namespace opg
};  // namespace KNN
};  // namespace ML
