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

template struct KNN_CL_params<float, int64_t, float, int>;

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
                  bool rowMajorIndex,
                  bool rowMajorQuery,
                  bool probas_only,
                  int k,
                  size_t batch_size,
                  bool verbose)
{
  knn_operation knn_op = probas_only ? knn_operation::class_proba : knn_operation::classification;
  KNN_CL_params<float, int64_t, float, int> params(knn_op,
                                                   &idx_data,
                                                   &idx_desc,
                                                   &query_data,
                                                   &query_desc,
                                                   rowMajorIndex,
                                                   rowMajorQuery,
                                                   k,
                                                   batch_size,
                                                   verbose,
                                                   n_unique.size(),
                                                   &y,
                                                   &n_unique,
                                                   &uniq_labels,
                                                   out,
                                                   probas);

  opg_knn(params, handle);
}
};  // namespace opg
};  // namespace KNN
};  // namespace ML
