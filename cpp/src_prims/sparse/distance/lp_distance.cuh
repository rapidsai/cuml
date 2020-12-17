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
//#include <sparse/semiring.cuh>

#include <raft/cudart_utils.h>
#include <raft/linalg/distance_type.h>
#include <raft/sparse/cusparse_wrappers.h>
#include <raft/cuda_utils.cuh>

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

template <typename value_idx = int, typename value_t = float>
class l1_distances_t : public distances_t<value_t> {
 public:
  l1_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    CUML_LOG_DEBUG("Running l1 dists");
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __host__ __device__(value_t a, value_t b, value_t p) {
        return fabsf(a - b);
      },
      [] __host__ __device__(value_t a, value_t b) { return a + b; });

    CUDA_CHECK(cudaStreamSynchronize(config_.stream));

    std::cout << raft::arr2Str(out_dists, 10, "out_dists", config_.stream)
              << std::endl;
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class l2_unexpanded_distances_t : public distances_t<value_t> {
 public:
  l2_unexpanded_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __host__ __device__(value_t a, value_t b, value_t p) {
        return (a - b) * (a - b);
      },
      [] __host__ __device__(value_t a, value_t b) { return a + b; });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class chebychev_distances_t : public distances_t<value_t> {
 public:
  explicit chebychev_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __host__ __device__(value_t a, value_t b, value_t p) {
        return fabsf(a - b);
      },
      [] __host__ __device__(value_t a, value_t b) { return fmaxf(a, b); });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class canberra_distances_t : public distances_t<value_t> {
 public:
  explicit canberra_distances_t(distances_config_t<value_idx, value_t> config)
    : config_(config) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b, value_t p) {
        return fabsf(a - b) / (fabsf(a) + fabsf(b));
      },
      [] __host__ __device__(value_t a, value_t b) { return a + b; });
  }

 private:
  distances_config_t<value_idx, value_t> config_;
};

template <typename value_idx = int, typename value_t = float>
class minkowski_distances_t : public distances_t<value_t> {
 public:
  explicit minkowski_distances_t(distances_config_t<value_idx, value_t> config,
                                 value_t p_)
    : config_(config), p(p_) {}

  void compute(value_t *out_dists) {
    generalized_csr_pairwise_semiring<value_idx, value_t>(
      out_dists, config_,
      [] __device__(value_t a, value_t b, value_t p) {
        return __powf(a - b, p);
      },
      [] __host__ __device__(value_t a, value_t b) { return a + b; }, p);
  }

 private:
  distances_config_t<value_idx, value_t> config_;
  value_t p;
};

};  // END namespace Distance
};  // END namespace Sparse
};  // END namespace MLCommon
