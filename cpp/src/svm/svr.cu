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

#include <iostream>

#include "common/device_buffer.hpp"
#include "cuml/svm/svc.hpp"
#include "kernelcache.h"
#include "label/classlabels.h"
#include "linalg/cublas_wrappers.h"
#include "linalg/unary_op.h"
#include "matrix/kernelfactory.h"
#include "smosolver.h"
#include "svr_impl.h"

namespace ML {
namespace SVM {

// Explicit instantiation for the library
template void svrFit<float>(const cumlHandle &handle, float *X, int n_rows,
                            int n_cols, float *y, const svmParameter &param,
                            MLCommon::Matrix::KernelParams &kernel_params,
                            svmModel<float> &model);

template void svrFit<double>(const cumlHandle &handle, double *X, int n_rows,
                             int n_cols, double *y, const svmParameter &param,
                             MLCommon::Matrix::KernelParams &kernel_params,
                             svmModel<double> &model);

};  // namespace SVM
};  // end namespace ML
