/*
 * Copyright (c) 2018-2022, NVIDIA CORPORATION.
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

#include "rproj_utils.cuh"

#include <cuml/random_projection/rproj_c.h>

#include <raft/core/cudart_utils.hpp>
#include <raft/cuda_utils.cuh>

// TODO: This needs to be removed.
#include <raft/sparse/detail/cusparse_wrappers.h>
// #TODO: Replace with public header when ready
#include <raft/linalg/detail/cublas_wrappers.hpp>

#include <cstddef>
#include <random>
#include <unordered_set>
#include <vector>

namespace ML {

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

  if (params.density == 1.0f) {
    int len = params.n_components * params.n_features;
    random_matrix->dense_data.resize(len, stream);
    auto rng     = raft::random::Rng(params.random_state);
    math_t scale = 1.0 / sqrt(math_t(params.n_components));
    rng.scaled_bernoulli(random_matrix->dense_data.data(), len, math_t(0.5), scale, stream);
  } else {
    std::size_t indices_alloc = params.n_features * params.n_components;
    std::size_t indptr_alloc  = (params.n_components + 1);
    std::vector<int> indices(indices_alloc);
    std::vector<int> indptr(indptr_alloc);

    std::size_t offset      = 0;
    std::size_t indices_idx = 0;
    std::size_t indptr_idx  = 0;

    for (int i = 0; i < params.n_components; i++) {
      int n_nonzero = binomial(h, params.n_features, params.density, params.random_state);
      sample_without_replacement(params.n_features, n_nonzero, indices.data(), indices_idx);
      indptr[indptr_idx] = offset;
      indptr_idx++;
      offset += n_nonzero;
    }

    indptr[indptr_idx] = offset;

    auto len = offset;
    random_matrix->indices.resize(len, stream);
    raft::update_device(random_matrix->indices.data(), indices.data(), len, stream);

    len = indptr_idx + 1;
    random_matrix->indptr.resize(len, stream);
    raft::update_device(random_matrix->indptr.data(), indptr.data(), len, stream);

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

    auto& m = params->n_samples;
    auto& n = params->n_components;
    auto& k = params->n_features;

    auto& lda = m;
    auto& ldb = k;
    auto& ldc = m;

    // #TODO: Call from public API when ready
    RAFT_CUBLAS_TRY(raft::linalg::detail::cublasgemm(cublas_handle,
                                                     CUBLAS_OP_N,
                                                     CUBLAS_OP_N,
                                                     params->n_samples,
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
    auto cusparse_handle = handle.get_cusparse_handle();

    const math_t alfa = 1;
    const math_t beta = 0;

    auto& m         = params->n_samples;
    auto& n         = params->n_components;
    auto& k         = params->n_features;
    std::size_t nnz = random_matrix->sparse_data.size();

    auto& lda = m;
    auto& ldc = m;

    // TODO: Need to wrap this in a RAFT public API.
    RAFT_CUSPARSE_TRY(raft::sparse::detail::cusparsegemmi(cusparse_handle,
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
