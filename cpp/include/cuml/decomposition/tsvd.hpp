/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "params.hpp"

#include <cuml/common/export.hpp>

namespace raft {
class handle_t;
}

namespace CUML_EXPORT ML {

void tsvdFit(const raft::handle_t& handle,
             float* input,
             float* components,
             float* singular_vals,
             const paramsTSVD& prms);
void tsvdFit(const raft::handle_t& handle,
             double* input,
             double* components,
             double* singular_vals,
             const paramsTSVD& prms);
void tsvdInverseTransform(const raft::handle_t& handle,
                          float* trans_input,
                          float* components,
                          float* input,
                          const paramsTSVD& prms);
void tsvdInverseTransform(const raft::handle_t& handle,
                          double* trans_input,
                          double* components,
                          double* input,
                          const paramsTSVD& prms);
void tsvdTransform(const raft::handle_t& handle,
                   float* input,
                   float* components,
                   float* trans_input,
                   const paramsTSVD& prms);
void tsvdTransform(const raft::handle_t& handle,
                   double* input,
                   double* components,
                   double* trans_input,
                   const paramsTSVD& prms);
void tsvdFitTransform(const raft::handle_t& handle,
                      float* input,
                      float* trans_input,
                      float* components,
                      float* explained_var,
                      float* explained_var_ratio,
                      float* singular_vals,
                      const paramsTSVD& prms);
void tsvdFitTransform(const raft::handle_t& handle,
                      double* input,
                      double* trans_input,
                      double* components,
                      double* explained_var,
                      double* explained_var_ratio,
                      double* singular_vals,
                      const paramsTSVD& prms);

}  // namespace CUML_EXPORT ML
