/*
 * Copyright (c) 2018-2024, NVIDIA CORPORATION.
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

#include "tsvd.cuh"

#include <cuml/decomposition/tsvd.hpp>

#include <raft/core/handle.hpp>

namespace ML {

void tsvdFit(raft::handle_t& handle,
             float* input,
             float* components,
             float* singular_vals,
             const paramsTSVD& prms)
{
  tsvdFit(handle, input, components, singular_vals, prms, handle.get_stream());
}

void tsvdFit(raft::handle_t& handle,
             double* input,
             double* components,
             double* singular_vals,
             const paramsTSVD& prms)
{
  tsvdFit(handle, input, components, singular_vals, prms, handle.get_stream());
}

void tsvdFitTransform(raft::handle_t& handle,
                      float* input,
                      float* trans_input,
                      float* components,
                      float* explained_var,
                      float* explained_var_ratio,
                      float* singular_vals,
                      const paramsTSVD& prms)
{
  tsvdFitTransform(handle,
                   input,
                   trans_input,
                   components,
                   explained_var,
                   explained_var_ratio,
                   singular_vals,
                   prms,
                   handle.get_stream());
}

void tsvdFitTransform(raft::handle_t& handle,
                      double* input,
                      double* trans_input,
                      double* components,
                      double* explained_var,
                      double* explained_var_ratio,
                      double* singular_vals,
                      const paramsTSVD& prms)
{
  tsvdFitTransform(handle,
                   input,
                   trans_input,
                   components,
                   explained_var,
                   explained_var_ratio,
                   singular_vals,
                   prms,
                   handle.get_stream());
}

void tsvdTransform(raft::handle_t& handle,
                   float* input,
                   float* components,
                   float* trans_input,
                   const paramsTSVD& prms)
{
  tsvdTransform(handle, input, components, trans_input, prms, handle.get_stream());
}

void tsvdTransform(raft::handle_t& handle,
                   double* input,
                   double* components,
                   double* trans_input,
                   const paramsTSVD& prms)
{
  tsvdTransform(handle, input, components, trans_input, prms, handle.get_stream());
}

void tsvdInverseTransform(raft::handle_t& handle,
                          float* trans_input,
                          float* components,
                          float* input,
                          const paramsTSVD& prms)
{
  tsvdInverseTransform(handle, trans_input, components, input, prms, handle.get_stream());
}

void tsvdInverseTransform(raft::handle_t& handle,
                          double* trans_input,
                          double* components,
                          double* input,
                          const paramsTSVD& prms)
{
  tsvdInverseTransform(handle, trans_input, components, input, prms, handle.get_stream());
}

};  // end namespace ML
