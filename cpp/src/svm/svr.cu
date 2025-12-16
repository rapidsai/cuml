/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernelcache.cuh"
#include "smosolver.cuh"
#include "svr_impl.cuh"

#include <cuml/matrix/kernel_params.hpp>
#include <cuml/svm/svc.hpp>

#include <raft/core/handle.hpp>
#include <raft/label/classlabels.cuh>
#include <raft/linalg/unary_op.cuh>

#include <iostream>

namespace ML {
namespace SVM {

// Explicit instantiation for the library
template int svrFit<float>(const raft::handle_t& handle,
                           float* X,
                           int n_rows,
                           int n_cols,
                           float* y,
                           const SvmParameter& param,
                           ML::matrix::KernelParams& kernel_params,
                           SvmModel<float>& model,
                           const float* sample_weight);

template int svrFit<double>(const raft::handle_t& handle,
                            double* X,
                            int n_rows,
                            int n_cols,
                            double* y,
                            const SvmParameter& param,
                            ML::matrix::KernelParams& kernel_params,
                            SvmModel<double>& model,
                            const double* sample_weight);

template int svrFitSparse<float>(const raft::handle_t& handle,
                                 int* indptr,
                                 int* indices,
                                 float* data,
                                 int n_rows,
                                 int n_cols,
                                 int nnz,
                                 float* y,
                                 const SvmParameter& param,
                                 ML::matrix::KernelParams& kernel_params,
                                 SvmModel<float>& model,
                                 const float* sample_weight);

template int svrFitSparse<double>(const raft::handle_t& handle,
                                  int* indptr,
                                  int* indices,
                                  double* data,
                                  int n_rows,
                                  int n_cols,
                                  int nnz,
                                  double* y,
                                  const SvmParameter& param,
                                  ML::matrix::KernelParams& kernel_params,
                                  SvmModel<double>& model,
                                  const double* sample_weight);

};  // namespace SVM
};  // end namespace ML
