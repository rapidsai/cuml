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

#include <iostream>

#include <raft/linalg/cublas_wrappers.h>
#include <cuml/svm/svc.hpp>
#include <label/classlabels.cuh>
#include <matrix/kernelfactory.cuh>
#include <raft/linalg/unary_op.cuh>
#include "kernelcache.cuh"
#include "smosolver.cuh"
#include "svr_impl.cuh"

namespace ML {
namespace SVM {

// Explicit instantiation for the library
template void svrFit<float>(const raft::handle_t &handle, float *X, int n_rows,
                            int n_cols, float *y, const svmParameter &param,
                            MLCommon::Matrix::KernelParams &kernel_params,
                            svmModel<float> &model, const float *sample_weight);

template void svrFit<double>(const raft::handle_t &handle, double *X,
                             int n_rows, int n_cols, double *y,
                             const svmParameter &param,
                             MLCommon::Matrix::KernelParams &kernel_params,
                             svmModel<double> &model,
                             const double *sample_weight);

};  // namespace SVM
};  // end namespace ML
