/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <common/cudart_utils.h>
#include <common/cumlHandle.hpp>
#include <common/device_buffer.hpp>
#include <cuda_utils.cuh>
#include <cuml/common/cuml_allocator.hpp>
#include <cuml/linear_model/preprocess_mg.hpp>
#include <linalg/gemm.cuh>
#include <linalg/subtract.cuh>
#include <matrix/math.cuh>
#include <opg/linalg/norm.hpp>
#include <opg/matrix/math.hpp>
#include <opg/stats/mean.hpp>
#include <opg/stats/mean_center.hpp>
#include <raft/comms/comms.hpp>

using namespace MLCommon;

namespace ML {
namespace GLM {
namespace opg {

template <typename T>
void preProcessData_impl(raft::handle_t &handle,
                         std::vector<Matrix::Data<T> *> &input_data,
                         Matrix::PartDescriptor &input_desc,
                         std::vector<Matrix::Data<T> *> &labels, T *mu_input,
                         T *mu_labels, T *norm2_input, bool fit_intercept,
                         bool normalize, cudaStream_t *streams, int n_streams,
                         bool verbose) {
  const auto &comm = handle.get_comms();
  cublasHandle_t cublas_handle = handle.get_cublas_handle();
  cusolverDnHandle_t cusolver_handle = handle.get_cusolver_dn_handle();
  const auto allocator = handle.get_device_allocator();

  if (fit_intercept) {
    Matrix::Data<T> mu_input_data{mu_input, size_t(input_desc.N)};
    Stats::opg::mean(mu_input_data, input_data, input_desc, comm, allocator,
                     streams, n_streams, cublas_handle);
    Stats::opg::mean_center(input_data, input_desc, mu_input_data, comm,
                            streams, n_streams);

    Matrix::PartDescriptor labels_desc = input_desc;
    labels_desc.N = size_t(1);

    Matrix::Data<T> mu_labels_data{mu_labels, size_t(1)};
    Stats::opg::mean(mu_labels_data, labels, labels_desc, comm, allocator,
                     streams, n_streams, cublas_handle);
    Stats::opg::mean_center(labels, labels_desc, mu_labels_data, comm, streams,
                            n_streams);

    if (normalize) {
      Matrix::Data<T> norm2_input_data{norm2_input, size_t(input_desc.N)};
      LinAlg::opg::colNorm2(norm2_input_data, input_data, input_desc, comm,
                            allocator, streams, n_streams, cublas_handle);

      Matrix::opg::matrixVectorBinaryDivSkipZero(
        input_data, input_desc, norm2_input_data, false, true, true, comm,
        streams, n_streams);
    }
  }
}

template <typename T>
void postProcessData_impl(raft::handle_t &handle,
                          std::vector<Matrix::Data<T> *> &input_data,
                          Matrix::PartDescriptor &input_desc,
                          std::vector<Matrix::Data<T> *> &labels, T *coef,
                          T *intercept, T *mu_input, T *mu_labels,
                          T *norm2_input, bool fit_intercept, bool normalize,
                          cudaStream_t *streams, int n_streams, bool verbose) {
  const auto &comm = handle.get_comms();
  cublasHandle_t cublas_handle = handle.get_cublas_handle();
  cusolverDnHandle_t cusolver_handle = handle.get_cusolver_dn_handle();
  const auto allocator = handle.get_device_allocator();

  device_buffer<T> d_intercept(allocator, streams[0], 1);

  if (normalize) {
    Matrix::Data<T> norm2_input_data{norm2_input, input_desc.N};
    Matrix::opg::matrixVectorBinaryMult(input_data, input_desc,
                                        norm2_input_data, false, true, comm,
                                        streams, n_streams);
    Matrix::matrixVectorBinaryDivSkipZero(coef, norm2_input, size_t(1),
                                          input_desc.N, false, true, streams[0],
                                          true);
  }

  LinAlg::gemm(mu_input, 1, input_desc.N, coef, d_intercept.data(), 1, 1,
               CUBLAS_OP_N, CUBLAS_OP_N, cublas_handle, streams[0]);

  LinAlg::subtract(d_intercept.data(), mu_labels, d_intercept.data(), 1,
                   streams[0]);
  updateHost(intercept, d_intercept.data(), 1, streams[0]);

  Matrix::Data<T> mu_input_data{mu_input, size_t(input_desc.N)};
  Stats::opg::mean_add(input_data, input_desc, mu_input_data, comm, streams,
                       n_streams);

  Matrix::PartDescriptor label_desc = input_desc;
  label_desc.N = size_t(1);
  Matrix::Data<T> mu_label_data{mu_labels, size_t(1)};
  Stats::opg::mean_add(labels, label_desc, mu_label_data, comm, streams,
                       n_streams);
}

void preProcessData(raft::handle_t &handle,
                    std::vector<Matrix::Data<float> *> &input_data,
                    Matrix::PartDescriptor &input_desc,
                    std::vector<Matrix::Data<float> *> &labels, float *mu_input,
                    float *mu_labels, float *norm2_input, bool fit_intercept,
                    bool normalize, cudaStream_t *streams, int n_streams,
                    bool verbose) {
  preProcessData_impl(handle, input_data, input_desc, labels, mu_input,
                      mu_labels, norm2_input, fit_intercept, normalize, streams,
                      n_streams, verbose);
}

void preProcessData(raft::handle_t &handle,
                    std::vector<Matrix::Data<double> *> &input_data,
                    Matrix::PartDescriptor &input_desc,
                    std::vector<Matrix::Data<double> *> &labels,
                    double *mu_input, double *mu_labels, double *norm2_input,
                    bool fit_intercept, bool normalize, cudaStream_t *streams,
                    int n_streams, bool verbose) {
  preProcessData_impl(handle, input_data, input_desc, labels, mu_input,
                      mu_labels, norm2_input, fit_intercept, normalize, streams,
                      n_streams, verbose);
}

void postProcessData(raft::handle_t &handle,
                     std::vector<Matrix::Data<float> *> &input_data,
                     Matrix::PartDescriptor &input_desc,
                     std::vector<Matrix::Data<float> *> &labels, float *coef,
                     float *intercept, float *mu_input, float *mu_labels,
                     float *norm2_input, bool fit_intercept, bool normalize,
                     cudaStream_t *streams, int n_streams, bool verbose) {
  postProcessData_impl(handle, input_data, input_desc, labels, coef, intercept,
                       mu_input, mu_labels, norm2_input, fit_intercept,
                       normalize, streams, n_streams, verbose);
}

void postProcessData(raft::handle_t &handle,
                     std::vector<Matrix::Data<double> *> &input_data,
                     Matrix::PartDescriptor &input_desc,
                     std::vector<Matrix::Data<double> *> &labels, double *coef,
                     double *intercept, double *mu_input, double *mu_labels,
                     double *norm2_input, bool fit_intercept, bool normalize,
                     cudaStream_t *streams, int n_streams, bool verbose) {
  postProcessData_impl(handle, input_data, input_desc, labels, coef, intercept,
                       mu_input, mu_labels, norm2_input, fit_intercept,
                       normalize, streams, n_streams, verbose);
}

}  // namespace opg
}  // namespace GLM
}  // namespace ML
