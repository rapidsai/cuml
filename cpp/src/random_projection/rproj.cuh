/*
 * Copyright (c) 2018-2021, NVIDIA CORPORATION.
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

#include <random>
#include <unordered_set>
#include <vector>

#include <cuml/random_projection/rproj_c.h>
#include <raft/cudart_utils.h>
#include <raft/linalg/cublas_wrappers.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include "rproj_utils.cuh"

namespace ML {

using namespace MLCommon;

/**
 * @brief generates a gaussian random matrix
 * @param[in] h: cuML handle
 * @param[out] random_matrix: the random matrix to be allocated and generated
 * @param[in] params: data structure that includes all the parameters of the model
 */
template <typename math_t>
void gaussian_random_matrix(const raft::handle_t& h,
                            rand_mat<math_t>* random_matrix,
                            paramsRPROJ& params)
{
  cudaStream_t stream = h.get_stream();
  auto d_alloc        = h.get_device_allocator();
  int len             = params.n_components * params.n_features;
  random_matrix->dense_data.resize(len, stream);
  auto rng     = raft::random::Rng(params.random_state);
  math_t scale = 1.0 / sqrt(double(params.n_components));
  rng.normal(random_matrix->dense_data.data(), len, math_t(0), scale, stream);
}

/**
 * @brief generates a sparse random matrix
 * @param[in] h: cuML handle
 * @param[out] random_matrix: the random matrix to be allocated and generated
 * @param[in] params: data structure that includes all the parameters of the model
 */
template <typename math_t>
void sparse_random_matrix(const raft::handle_t& h,
                          rand_mat<math_t>* random_matrix,
                          paramsRPROJ& params)
{
  cudaStream_t stream = h.get_stream();
  auto d_alloc        = h.get_device_allocator();

  if (params.density == 1.0f) {
    int len = params.n_components * params.n_features;
    random_matrix->dense_data.resize(len, stream);
    auto rng     = raft::random::Rng(params.random_state);
    math_t scale = 1.0 / sqrt(math_t(params.n_components));
    rng.scaled_bernoulli(random_matrix->dense_data.data(), len, math_t(0.5), scale, stream);
  } else {
    auto alloc = h.get_host_allocator();

    double max_total_density = params.density * 1.2;
    size_t indices_alloc =
      (params.n_features * params.n_components * max_total_density) * sizeof(int);
    size_t indptr_alloc = (params.n_components + 1) * sizeof(int);
    int* indices        = (int*)alloc->allocate(indices_alloc, stream);
    int* indptr         = (int*)alloc->allocate(indptr_alloc, stream);

    size_t offset      = 0;
    size_t indices_idx = 0;
    size_t indptr_idx  = 0;

    for (size_t i = 0; i < params.n_components; i++) {
      int n_nonzero = binomial(h, params.n_features, params.density, params.random_state);
      sample_without_replacement(params.n_features, n_nonzero, indices, indices_idx);
      indptr[indptr_idx] = offset;
      indptr_idx++;
      offset += n_nonzero;
    }

    indptr[indptr_idx] = offset;

    size_t len = offset;
    random_matrix->indices.resize(len, stream);
    raft::update_device(random_matrix->indices.data(), indices, len, stream);
    alloc->deallocate(indices, indices_alloc, stream);

    len = indptr_idx + 1;
    random_matrix->indptr.resize(len, stream);
    raft::update_device(random_matrix->indptr.data(), indptr, len, stream);
    alloc->deallocate(indptr, indptr_alloc, stream);

    len = offset;
    random_matrix->sparse_data.resize(len, stream);
    auto rng     = raft::random::Rng(params.random_state);
    math_t scale = sqrt(1.0 / params.density) / sqrt(params.n_components);
    rng.scaled_bernoulli(random_matrix->sparse_data.data(), len, math_t(0.5), scale, stream);
  }
}

/**
 * @brief fits the model by generating appropriate random matrix
 * @param[in] handle: cuML handle
 * @param[out] random_matrix: the random matrix to be allocated and generated
 * @param[in] params: data structure that includes all the parameters of the model
 */
template <typename math_t>
void RPROJfit(const raft::handle_t& handle, rand_mat<math_t>* random_matrix, paramsRPROJ* params)
{
  random_matrix->reset();

  build_parameters(*params);
  check_parameters(*params);

  if (params->gaussian_method) {
    gaussian_random_matrix<math_t>(handle, random_matrix, *params);
    random_matrix->type = dense;
  } else {
    sparse_random_matrix<math_t>(handle, random_matrix, *params);
    random_matrix->type = sparse;
  }
}

/**
 * @brief transforms data according to generated random matrix
 * @param[in] handle: cuML handle
 * @param[in] input: unprojected original dataset
 * @param[in] random_matrix: the random matrix to be allocated and generated
 * @param[out] output: projected dataset
 * @param[in] params: data structure that includes all the parameters of the model
 */
template <typename math_t>
void RPROJtransform(const raft::handle_t& handle,
                    math_t* input,
                    rand_mat<math_t>* random_matrix,
                    math_t* output,
                    paramsRPROJ* params)
{
  cudaStream_t stream = handle.get_stream();

  check_parameters(*params);

  if (random_matrix->type == dense) {
    cublasHandle_t cublas_handle = handle.get_cublas_handle();

    const math_t alfa = 1;
    const math_t beta = 0;

    int& m = params->n_samples;
    int& n = params->n_components;
    int& k = params->n_features;

    int& lda = m;
    int& ldb = k;
    int& ldc = m;

    CUBLAS_CHECK(raft::linalg::cublasgemm(cublas_handle,
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          m,
                                          n,
                                          k,
                                          &alfa,
                                          input,
                                          lda,
                                          random_matrix->dense_data.data(),
                                          ldb,
                                          &beta,
                                          output,
                                          ldc,
                                          stream));

  } else if (random_matrix->type == sparse) {
    cusparseHandle_t cusparse_handle = handle.get_cusparse_handle();

    const math_t alfa = 1;
    const math_t beta = 0;

    int& m     = params->n_samples;
    int& n     = params->n_components;
    int& k     = params->n_features;
    size_t nnz = random_matrix->sparse_data.size();

    int& lda = m;
    int& ldc = m;

    CUSPARSE_CHECK(raft::sparse::cusparsegemmi(cusparse_handle,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               &alfa,
                                               input,
                                               lda,
                                               random_matrix->sparse_data.data(),
                                               random_matrix->indptr.data(),
                                               random_matrix->indices.data(),
                                               &beta,
                                               output,
                                               ldc,
                                               stream));
  } else {
    ASSERT(false,
           "Could not find a random matrix. Please perform a fit operation "
           "before applying transformation");
  }
}

};  // namespace ML
// end namespace ML
