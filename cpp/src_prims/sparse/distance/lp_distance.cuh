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

#pragma once

#include <limits.h>
#include <raft/cudart_utils.h>
#include <sparse/distance/common.h>

#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>
#include <raft/device_atomics.cuh>

#include <common/device_buffer.hpp>

#include <sparse/utils.h>
#include <sparse/csr.cuh>

#include <sparse/distance/common.h>
#include <sparse/distance/semiring.cuh>

#include <cuml/common/cuml_allocator.hpp>

#include <nvfunctional>

#include <cusparse_v2.h>

namespace MLCommon {
namespace Sparse {
namespace Distance {

template <typename value_idx = int, typename value_t = float,
          typename reduce_f = auto(value_t, value_t)->value_t,
          typename accum_f = auto(value_t, value_t)->value_t,
          typename write_f = auto(value_t *, value_t)->void>
void unexpanded_lp_distances(
  value_t *out_dists, const distances_config_t<value_idx, value_t> &config_,
  reduce_f reduce_func, accum_f accum_func, write_f write_func,
  const float metric_arg = 2.0) {
  /**
 * @TODO: Main logic here:
 *
 *  - if n_cols < available smem, just use dense conversion for rows of A
 *  - if n_cols > available smem but max nnz < available smem, use hashing
 *    (not yet available)
 *  - if n_cols > available smem & max_nnz > available smem,
 *              use batching + hashing only for those large cols
 */

  if (config_.a_ncols < balanced_coo_spmv_compute_smem<value_idx, value_t>()) {
    raft::mr::device::buffer<value_idx> coo_rows(
      config_.allocator, config_.stream, max(config_.b_nnz, config_.a_nnz));

    MLCommon::Sparse::csr_to_coo(config_.b_indptr, config_.b_nrows,
                                 coo_rows.data(), config_.b_nnz,
                                 config_.stream);

    balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(
      out_dists, config_, coo_rows.data(), reduce_func, accum_func, write_func, metric_arg);

    MLCommon::Sparse::csr_to_coo(config_.a_indptr, config_.a_nrows,
                                 coo_rows.data(), config_.a_nnz,
                                 config_.stream);

    balanced_coo_pairwise_generalized_spmv_rev<value_idx, value_t>(
      out_dists, config_, coo_rows.data(), reduce_func, accum_func, write_func, metric_arg);
  } else {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_, reduce_func, accum_func, metric_arg);
  }
}

/**
 * Computes L1 distances for sparse input. This does not have
 * an equivalent expanded form, so it is only executed in
 * an unexpanded form.
 * @tparam value_idx
 * @tparam value_t
 */
template <typename value_idx = int, typename value_t = float>
class l1_unexpanded_distances_t : public distances_t<value_t> {
 public:
  l1_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Running l1 dists");

    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b, float p) { return fabsf(a - b); },
      [] __device__(value_t a, value_t b) { return a + b; },
      [] __device__(value_t * a, value_t b) { atomicAdd(a, b); });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class l2_unexpanded_distances_t : public distances_t<value_t> {
 public:
  l2_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_,
      [] __host__ __device__(value_t a, value_t b, float p) {
        return (a - b) * (a - b);
      },
      [] __host__ __device__(value_t a, value_t b) { return a + b; },
      [] __host__ __device__(value_t * a, value_t b) { atomicAdd(a, b); });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class linf_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit linf_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_,
      [] __host__ __device__(value_t a, value_t b, float p) { return fabsf(a - b); },
      [] __host__ __device__(value_t a, value_t b) { return fmaxf(a, b); },
      [] __host__ __device__(value_t * a, value_t b) { atomicMax(a, b); });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class canberra_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit canberra_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b, float p) {
        return fabsf(a - b) / (fabsf(a) + fabsf(b));
      },
      [] __host__ __device__(value_t a, value_t b) { return a + b; },
      [] __host__ __device__(value_t * a, value_t b) { atomicAdd(a, b); });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class lp_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit lp_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config, value_t p_)
    : config_(config), p(p_) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b, float p) {
        return powf(a - b, p);
      },
      [] __host__ __device__(value_t a, value_t b) { return a + b; },
      [] __host__ __device__(value_t * a, value_t b) { atomicAdd(a, b); });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
  value_t p;
};

};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
