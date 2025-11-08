/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "knn_opg_common.cuh"

namespace ML {
namespace KNN {
namespace opg {

using namespace knn_common;

template struct KNN_params<float, int64_t, float, int>;

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
         bool verbose)
{
  KNN_params<float, int64_t, float, int> params(knn_operation::knn,
                                                &idx_data,
                                                &idx_desc,
                                                &query_data,
                                                &query_desc,
                                                rowMajorIndex,
                                                rowMajorQuery,
                                                k,
                                                batch_size,
                                                verbose,
                                                out_D,
                                                out_I);

  opg_knn(params, handle);
}
};  // namespace opg
};  // namespace KNN
};  // namespace ML
