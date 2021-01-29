/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

#include <raft/mr/device/allocator.hpp>
#include <raft/mr/device/buffer.hpp>

#include <sparse/utils.h>
#include <sparse/csr.cuh>

#include <sparse/distance/common.h>
#include <sparse/convert/coo.cuh>
#include <sparse/distance/csr_spmv.cuh>
#include <sparse/distance/operators.cuh>

#include <nvfunctional>

namespace raft {
namespace sparse {
namespace distance {

template <typename value_idx = int, typename value_t = float,
          typename product_f, typename accum_f, typename write_f>

void unexpanded_lp_distances(
  value_t *out_dists, const distances_config_t<value_idx, value_t> *config_,
  product_f product_func, accum_f accum_func, write_f write_func) {
  /**
 * @TODO: Main logic here:
 *
 *  - if n_cols < available smem, just use dense conversion for rows of A
 *  - if n_cols > available smem but max nnz < available smem, use hashing
 *    (not yet available)
 *  - if n_cols > available smem & max_nnz > available smem,
 *              use batching + hashing only for those large cols
 *  Ref: https://github.com/rapidsai/cuml/issues/3371
 */

  if (config_->a_ncols < max_cols_per_block<value_idx, value_t>()) {
    // TODO: Use n_cols to set shared memory and threads per block
    // for max occupancy.
    // Ref: https://github.com/rapidsai/cuml/issues/3371

    raft::mr::device::buffer<value_idx> coo_rows(
      config_->allocator, config_->stream, max(config_->b_nnz, config_->a_nnz));

    raft::sparse::convert::csr_to_coo(config_->b_indptr, config_->b_nrows,
                                      coo_rows.data(), config_->b_nnz,
                                      config_->stream);

    balanced_coo_pairwise_generalized_spmv<value_idx, value_t>(
      out_dists, *config_, coo_rows.data(), product_func, accum_func,
      write_func);

    raft::sparse::convert::csr_to_coo(config_->a_indptr, config_->a_nrows,
                                      coo_rows.data(), config_->a_nnz,
                                      config_->stream);

    balanced_coo_pairwise_generalized_spmv_rev<value_idx, value_t>(
      out_dists, *config_, coo_rows.data(), product_func, accum_func,
      write_func);

  } else {
    // TODO: Find max nnz and set smem based on this value.
    // Ref: https://github.com/rapidsai/cuml/issues/3371
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, *config_, product_func, accum_func);
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
    : config_(&config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Running l1 dists");

    unexpanded_lp_distances<value_idx, value_t>(out_dists, config_, AbsDiff(),
                                                Sum(), AtomicAdd());
  }

 private:
  const distances_config_t<value_idx, value_t> *config_;
};

template <typename value_idx = int, typename value_t = float>
class l2_unexpanded_distances_t : public distances_t<value_t> {
 public:
  l2_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(&config) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(out_dists, config_, SqDiff(),
                                                Sum(), AtomicAdd());
  }

 private:
  const distances_config_t<value_idx, value_t> *config_;
};

template <typename value_idx = int, typename value_t = float>
class linf_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit linf_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(&config) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(out_dists, config_, AbsDiff(),
                                                Max(), AtomicMax());
  }

 private:
  const distances_config_t<value_idx, value_t> *config_;
};

template <typename value_idx = int, typename value_t = float>
class canberra_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit canberra_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config)
    : config_(&config) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b) {
        value_t d = fabs(a) + fabs(b);

        // deal with potential for 0 in denominator by
        // forcing 1/0 instead
        return ((d != 0) * fabs(a - b)) / (d + (d == 0));
      },
      Sum(), AtomicAdd());
  }

 private:
  const distances_config_t<value_idx, value_t> *config_;
};

template <typename value_idx = int, typename value_t = float>
class lp_unexpanded_distances_t : public distances_t<value_t> {
 public:
  explicit lp_unexpanded_distances_t(
    const distances_config_t<value_idx, value_t> &config, value_t p_)
    : config_(&config), p(p_) {}

  void compute(value_t *out_dists) {
    unexpanded_lp_distances<value_idx, value_t>(out_dists, config_, PDiff(p),
                                                Sum(), AtomicAdd());

    float one_over_p = 1.0f / p;
    raft::linalg::unaryOp<value_t>(
      out_dists, out_dists, config_->a_nrows * config_->b_nrows,
      [=] __device__(value_t input) { return pow(input, one_over_p); },
      config_->stream);
  }

 private:
  const distances_config_t<value_idx, value_t> *config_;
  value_t p;
};

};  // END namespace distance
};  // END namespace sparse
};  // END namespace raft
