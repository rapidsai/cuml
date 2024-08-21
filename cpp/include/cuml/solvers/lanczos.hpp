/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <raft/core/handle.hpp>

namespace ML {
namespace Solver {

template <typename index_type, typename value_type>
void lanczos_solver(const raft::handle_t& handle,
                    index_type* rows,
                    index_type* cols,
                    value_type* vals,
                    int nnz,
                    int n,
                    int n_components,
                    int max_iterations,
                    int ncv,
                    value_type tolerance,
                    uint64_t seed,
                    value_type* v0,
                    value_type* eigenvalues,
                    value_type* eigenvectors);

void lanczos_solver(const raft::handle_t& handle,
                    int* rows,
                    int* cols,
                    double* vals,
                    int nnz,
                    int n,
                    int n_components,
                    int max_iterations,
                    int ncv,
                    double tolerance,
                    uint64_t seed,
                    double* v0,
                    double* eigenvalues,
                    double* eigenvectors);

void lanczos_solver(const raft::handle_t& handle,
                    int* rows,
                    int* cols,
                    float* vals,
                    int nnz,
                    int n,
                    int n_components,
                    int max_iterations,
                    int ncv,
                    float tolerance,
                    uint64_t seed,
                    float* v0,
                    float* eigenvalues,
                    float* eigenvectors);

template <typename index_type, typename value_type>
void old_lanczos_solver(const raft::handle_t& handle,
                        index_type* rows,
                        index_type* cols,
                        value_type* vals,
                        int nnz,
                        int n,
                        int n_components,
                        int max_iterations,
                        int ncv,
                        value_type tolerance,
                        uint64_t seed,
                        value_type* v0,
                        value_type* eigenvalues,
                        value_type* eigenvectors);

void old_lanczos_solver(const raft::handle_t& handle,
                        int* rows,
                        int* cols,
                        float* vals,
                        int nnz,
                        int n,
                        int n_components,
                        int max_iterations,
                        int ncv,
                        float tolerance,
                        uint64_t seed,
                        float* v0,
                        float* eigenvalues,
                        float* eigenvectors);

void old_lanczos_solver(const raft::handle_t& handle,
                        int* rows,
                        int* cols,
                        double* vals,
                        int nnz,
                        int n,
                        int n_components,
                        int max_iterations,
                        int ncv,
                        double tolerance,
                        uint64_t seed,
                        double* v0,
                        double* eigenvalues,
                        double* eigenvectors);

};  // namespace Solver
};  // end namespace ML
