/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
#define CUSPARSE_CHECK(call)                                               \
  do {                                                                     \
    cusparseStatus_t status = call;                                        \
    ASSERT(status == CUSPARSE_STATUS_SUCCESS, "FAIL: call='%s'\n", #call); \
  } while (0)

template <typename T>
cusparseStatus_t cusparse_gthr(cusparseHandle_t handle, int nnz, float *vals,
                               float *vals_sorted, int *d_P) {
  return cusparseSgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}

template <typename T>
cusparseStatus_t cusparse_gthr(cusparseHandle_t handle, int nnz, double *vals,
                               double *vals_sorted, int *d_P) {
  return cusparseDgthr(handle, nnz, vals, vals_sorted, d_P,
                       CUSPARSE_INDEX_BASE_ZERO);
}
};  // namespace Sparse
};  // namespace MLCommon
