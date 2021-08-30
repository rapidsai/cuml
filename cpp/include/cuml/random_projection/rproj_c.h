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

#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

namespace ML {

/**
 * @defgroup paramsRPROJ: structure holding parameters used by random projection model
 * @param n_samples: Number of samples
 * @param n_features: Number of features (original dimension)
 * @param n_components: Number of components (target dimension)
 * @param eps: error tolerance used to decide automatically of n_components
 * @param gaussian_method: boolean describing random matrix generation method
 * @param density: Density of the random matrix
 * @param dense_output: boolean describing sparsity of transformed matrix
 * @param random_state: seed used by random generator
 * @{
 */
struct paramsRPROJ {
  int n_samples;
  int n_features;
  int n_components;
  double eps;
  bool gaussian_method;
  double density;
  bool dense_output;
  int random_state;
};

enum random_matrix_type { unset, dense, sparse };

template <typename math_t>
struct rand_mat {
  rand_mat(cudaStream_t stream)
    : dense_data(0, stream),
      indices(0, stream),
      indptr(0, stream),
      sparse_data(0, stream),
      stream(stream),
      type(unset)
  {
  }

  ~rand_mat() { this->reset(); }

  // For dense matrices
  rmm::device_uvector<math_t> dense_data;

  // For sparse CSC matrices
  rmm::device_uvector<int> indices;
  rmm::device_uvector<int> indptr;
  rmm::device_uvector<math_t> sparse_data;

  cudaStream_t stream;

  random_matrix_type type;

  void reset()
  {
    this->dense_data.release();
    this->indices.release();
    this->indptr.release();
    this->sparse_data.release();
    this->type = unset;
  };
};

template <typename math_t>
void RPROJfit(const raft::handle_t& handle, rand_mat<math_t>* random_matrix, paramsRPROJ* params);

template <typename math_t>
void RPROJtransform(const raft::handle_t& handle,
                    math_t* input,
                    rand_mat<math_t>* random_matrix,
                    math_t* output,
                    paramsRPROJ* params);

size_t johnson_lindenstrauss_min_dim(size_t n_samples, double eps);

}  // namespace ML
